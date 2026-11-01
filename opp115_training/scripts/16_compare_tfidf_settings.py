
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss


DATA_DIR = Path("data/processed")
OUTPUT_PATH = DATA_DIR / "tfidf_experiments.json"


def parse_labels(value):
    return [x.strip() for x in str(value).split("|") if x.strip()]


def tune_thresholds(y_true, probabilities):
    thresholds = []
    predictions = []

    candidates = [
        0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.60, 0.70
    ]

    for i in range(y_true.shape[1]):
        best_f1 = -1
        best_threshold = 0.5

        for threshold in candidates:
            pred = (probabilities[:, i] >= threshold).astype(int)
            score = f1_score(
                y_true[:, i],
                pred,
                zero_division=0
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        thresholds.append(best_threshold)
        predictions.append(
            (probabilities[:, i] >= best_threshold).astype(int)
        )

    return list(zip(*predictions)), thresholds


def main():
    df = pd.read_csv(DATA_DIR / "train.csv")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    fit_idx, val_idx = next(
        splitter.split(df, groups=df["policy_file"])
    )

    fit_df = df.iloc[fit_idx].copy()
    val_df = df.iloc[val_idx].copy()

    mlb = MultiLabelBinarizer()
    mlb.fit(df["labels"].apply(parse_labels))

    y_fit = mlb.transform(fit_df["labels"].apply(parse_labels))
    y_val = mlb.transform(val_df["labels"].apply(parse_labels))

    print(f"Training rows: {len(fit_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(
        "Policy overlap:",
        len(set(fit_df["policy_file"]) & set(val_df["policy_file"]))
    )

    configurations = [
        {
            "name": "word_1_2",
            "analyzer": "word",
            "ngram_range": (1, 2),
        },
        {
            "name": "word_1_3",
            "analyzer": "word",
            "ngram_range": (1, 3),
        },
        {
            "name": "char_3_5",
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
        },
    ]

    results = []

    for config in configurations:
        print(f"\n{'=' * 50}")
        print(f"Configuration: {config['name']}")

        vectorizer = TfidfVectorizer(
            analyzer=config["analyzer"],
            ngram_range=config["ngram_range"],
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

        for weight in [None, "balanced"]:
            print(f"Class weight: {weight}")

            model = OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight=weight,
                ),
                n_jobs=1,
            )

            model.fit(x_fit, y_fit)
            probabilities = model.predict_proba(x_val)

            tuned_columns, thresholds = tune_thresholds(
                y_val, probabilities
            )
            y_pred = pd.DataFrame(
                tuned_columns,
                index=val_df.index
            ).to_numpy()

            result = {
                "configuration": config["name"],
                "class_weight": weight or "none",
                "micro_f1": float(
                    f1_score(y_val, y_pred, average="micro", zero_division=0)
                ),
                "macro_f1": float(
                    f1_score(y_val, y_pred, average="macro", zero_division=0)
                ),
                "hamming_loss": float(hamming_loss(y_val, y_pred)),
                "thresholds": {
                    label: float(threshold)
                    for label, threshold in zip(mlb.classes_, thresholds)
                },
            }

            results.append(result)

            print(f"Micro F1: {result['micro_f1']:.4f}")
            print(f"Macro F1: {result['macro_f1']:.4f}")
            print(f"Hamming Loss: {result['hamming_loss']:.4f}")

    results.sort(key=lambda item: item["macro_f1"], reverse=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("\nRanking by validation Macro F1:")

    for i, result in enumerate(results, start=1):
        print(
            f"{i}. {result['configuration']} | "
            f"weight={result['class_weight']} | "
            f"Macro F1={result['macro_f1']:.4f} | "
            f"Micro F1={result['micro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
