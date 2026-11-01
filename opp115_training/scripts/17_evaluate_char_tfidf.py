
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    accuracy_score,
    classification_report,
)

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("models")
RESULTS_PATH = OUTPUT_DIR / "char_tfidf_final_results.json"


def parse_labels(value):
    return [x.strip() for x in str(value).split("|") if x.strip()]


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test_no_train_duplicates.csv")

    # Reserve validation policies from the training split only.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )
    fit_idx, val_idx = next(
        splitter.split(train_df, groups=train_df["policy_file"])
    )

    fit_df = train_df.iloc[fit_idx]
    val_df = train_df.iloc[val_idx]

    mlb = MultiLabelBinarizer()
    mlb.fit(train_df["labels"].apply(parse_labels))

    y_fit = mlb.transform(fit_df["labels"].apply(parse_labels))
    y_val = mlb.transform(val_df["labels"].apply(parse_labels))
    y_test = mlb.transform(test_df["labels"].apply(parse_labels))

    # Tune thresholds using validation only.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )

    x_fit = vectorizer.fit_transform(
        fit_df["text"].fillna("").astype(str)
    )
    x_val = vectorizer.transform(
        val_df["text"].fillna("").astype(str)
    )

    validation_model = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
        ),
        n_jobs=1,
    )
    validation_model.fit(x_fit, y_fit)
    val_probs = validation_model.predict_proba(x_val)

    candidates = [
        0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.60, 0.70,
    ]

    thresholds = []
    for i in range(y_val.shape[1]):
        best_f1 = -1
        best_threshold = 0.5

        for threshold in candidates:
            pred = (val_probs[:, i] >= threshold).astype(int)
            score = f1_score(
                y_val[:, i],
                pred,
                zero_division=0,
            )
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        thresholds.append(best_threshold)

    print("Validation thresholds:")
    for label, threshold in zip(mlb.classes_, thresholds):
        print(f"  {label}: {threshold:.2f}")

    # Refit on all training policies; test remains untouched.
    final_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )

    x_train = final_vectorizer.fit_transform(
        train_df["text"].fillna("").astype(str)
    )
    x_test = final_vectorizer.transform(
        test_df["text"].fillna("").astype(str)
    )

    final_model = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
        ),
        n_jobs=1,
    )
    final_model.fit(
        x_train,
        mlb.transform(train_df["labels"].apply(parse_labels)),
    )

    test_probs = final_model.predict_proba(x_test)
    y_pred = (test_probs >= thresholds).astype(int)

    results = {
        "model": "TF-IDF char_wb 3-5 + LogisticRegression balanced",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "micro_f1": float(
            f1_score(y_test, y_pred, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(y_test, y_pred)),
        "exact_match_accuracy": float(accuracy_score(y_test, y_pred)),
        "thresholds": {
            label: float(threshold)
            for label, threshold in zip(mlb.classes_, thresholds)
        },
        "per_class": classification_report(
            y_test,
            y_pred,
            target_names=mlb.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_vectorizer, OUTPUT_DIR / "char_tfidf_vectorizer.joblib")
    joblib.dump(final_model, OUTPUT_DIR / "char_tfidf_model.joblib")
    joblib.dump(mlb, OUTPUT_DIR / "char_tfidf_label_binarizer.joblib")

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nFINAL TEST RESULTS")
    print(f"Micro F1: {results['micro_f1']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Hamming Loss: {results['hamming_loss']:.4f}")
    print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f}")

    print("\nPer-category F1:")
    for label, metrics in results["per_class"].items():
        if label in mlb.classes_:
            print(f"{label}: {metrics['f1-score']:.4f}")

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
