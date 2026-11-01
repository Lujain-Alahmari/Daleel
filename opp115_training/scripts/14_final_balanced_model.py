import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    accuracy_score,
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def parse_labels(value):
    return [x.strip() for x in str(value).split("|") if x.strip()]


# Load data
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test_no_train_duplicates.csv")

# Make a policy-level validation split from training data only
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)
fit_idx, val_idx = next(
    splitter.split(train_df, groups=train_df["policy_file"])
)

fit_df = train_df.iloc[fit_idx].copy()
val_df = train_df.iloc[val_idx].copy()

# Establish the label vocabulary from training data
mlb = MultiLabelBinarizer()
mlb.fit(train_df["labels"].apply(parse_labels))

y_fit = mlb.transform(fit_df["labels"].apply(parse_labels))
y_val = mlb.transform(val_df["labels"].apply(parse_labels))

# Fit vectorizer on fitting subset only for threshold selection
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
)

X_fit = vectorizer.fit_transform(
    fit_df["text"].fillna("").astype(str)
)
X_val = vectorizer.transform(
    val_df["text"].fillna("").astype(str)
)

validation_model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
)
validation_model.fit(X_fit, y_fit)

val_probs = validation_model.predict_proba(X_val)

# Tune thresholds using validation data only
candidates = [
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.60, 0.70
]
thresholds = {}

for i, label in enumerate(mlb.classes_):
    best_f1 = -1
    best_threshold = 0.5

    if y_val[:, i].sum() > 0:
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

    thresholds[label] = float(best_threshold)

print("Validation thresholds:")
for label, threshold in thresholds.items():
    print(f"{label}: {threshold:.2f}")

# Refit vectorizer and classifier on all training data
X_train = vectorizer.fit_transform(
    train_df["text"].fillna("").astype(str)
)
X_test = vectorizer.transform(
    test_df["text"].fillna("").astype(str)
)

y_train = mlb.transform(train_df["labels"].apply(parse_labels))
y_test = mlb.transform(test_df["labels"].apply(parse_labels))

final_model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
)
final_model.fit(X_train, y_train)

# Evaluate once on the held-out test set
test_probs = final_model.predict_proba(X_test)
threshold_array = np.array(
    [thresholds[label] for label in mlb.classes_]
)
y_pred = (test_probs >= threshold_array).astype(int)

report_text = classification_report(
    y_test,
    y_pred,
    target_names=mlb.classes_,
    zero_division=0,
)

micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
h_loss = hamming_loss(y_test, y_pred)
exact_match = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("FINAL BALANCED MODEL — HELD-OUT TEST")
print("=" * 60)
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))
print("Number of categories:", len(mlb.classes_))
print("Micro F1:", round(micro_f1, 4))
print("Macro F1:", round(macro_f1, 4))
print("Hamming Loss:", round(h_loss, 4))
print("Exact Match Accuracy:", round(exact_match, 4))
print("\nPER-CATEGORY REPORT")
print(report_text)

# Save final artifacts separately
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer_final.joblib")
joblib.dump(final_model, MODEL_DIR / "opp115_final_balanced_model.joblib")
joblib.dump(mlb, MODEL_DIR / "label_binarizer_final.joblib")

results = {
    "model": "TF-IDF + OneVsRest LogisticRegression",
    "class_weight": "balanced",
    "thresholds": thresholds,
    "train_samples": len(train_df),
    "validation_samples": len(val_df),
    "test_samples": len(test_df),
    "categories": list(mlb.classes_),
    "micro_f1": float(micro_f1),
    "macro_f1": float(macro_f1),
    "hamming_loss": float(h_loss),
    "exact_match_accuracy": float(exact_match),
    "classification_report": classification_report(
        y_test,
        y_pred,
        target_names=mlb.classes_,
        zero_division=0,
        output_dict=True,
    ),
}

with (DATA_DIR / "final_balanced_results.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nSaved final model to:", MODEL_DIR / "opp115_final_balanced_model.joblib")
print("Saved metrics to:", DATA_DIR / "final_balanced_results.json")
