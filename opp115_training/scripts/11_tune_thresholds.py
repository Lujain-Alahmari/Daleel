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


# Load training and untouched test data
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test_no_train_duplicates.csv")

# Split training policies into fitting and validation sets
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

fit_idx, val_idx = next(
    splitter.split(
        train_df,
        groups=train_df["policy_file"],
    )
)

fit_df = train_df.iloc[fit_idx].copy()
val_df = train_df.iloc[val_idx].copy()

print("Fit samples:", len(fit_df))
print("Validation samples:", len(val_df))
print(
    "Shared policies between fit and validation:",
    len(set(fit_df["policy_file"]) & set(val_df["policy_file"])),
)

# Fixed label vocabulary from training data only
mlb = MultiLabelBinarizer()
mlb.fit(train_df["labels"].apply(parse_labels))

y_fit = mlb.transform(fit_df["labels"].apply(parse_labels))
y_val = mlb.transform(val_df["labels"].apply(parse_labels))

# Fit TF-IDF on the fitting subset only
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
)

X_fit = vectorizer.fit_transform(fit_df["text"].fillna("").astype(str))
X_val = vectorizer.transform(val_df["text"].fillna("").astype(str))

# Train baseline classifier
model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000)
)
model.fit(X_fit, y_fit)

# Tune one probability threshold per category using validation data
val_probs = model.predict_proba(X_val)
thresholds = {}

candidates = np.arange(0.10, 0.81, 0.05)

for i, label in enumerate(mlb.classes_):
    best_threshold = 0.5
    best_f1 = -1.0

    # If this category has no validation positives, keep default threshold
    if y_val[:, i].sum() == 0:
        thresholds[label] = 0.5
        continue

    for threshold in candidates:
        predictions = (val_probs[:, i] >= threshold).astype(int)
        score = f1_score(
            y_val[:, i],
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    thresholds[label] = best_threshold

print("\nValidation thresholds:")
for label, threshold in thresholds.items():
    print(f"{label}: {threshold:.2f}")

# Refit vectorizer and model on all training data
X_train = vectorizer.fit_transform(
    train_df["text"].fillna("").astype(str)
)
X_test = vectorizer.transform(
    test_df["text"].fillna("").astype(str)
)

y_train = mlb.transform(train_df["labels"].apply(parse_labels))
y_test = mlb.transform(test_df["labels"].apply(parse_labels))

model.fit(X_train, y_train)

# Evaluate test set with tuned thresholds
test_probs = model.predict_proba(X_test)
threshold_array = np.array(
    [thresholds[label] for label in mlb.classes_]
)
y_pred = (test_probs >= threshold_array).astype(int)

print("\n" + "=" * 60)
print("OPP-115 THRESHOLD-TUNED BASELINE")
print("=" * 60)
print("Test samples:", len(test_df))
print("Micro F1:", round(f1_score(y_test, y_pred, average="micro", zero_division=0), 4))
print("Macro F1:", round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4))
print("Hamming Loss:", round(hamming_loss(y_test, y_pred), 4))
print("Exact Match Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nPER-CATEGORY REPORT")
print(classification_report(
    y_test,
    y_pred,
    target_names=mlb.classes_,
    zero_division=0,
))

# Save tuned artifacts separately; keep original baseline intact
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer_tuned.joblib")
joblib.dump(model, MODEL_DIR / "opp115_tuned_model.joblib")
joblib.dump(mlb, MODEL_DIR / "label_binarizer_tuned.joblib")

results = {
    "train_samples": len(train_df),
    "validation_samples": len(val_df),
    "test_samples": len(test_df),
    "thresholds": thresholds,
    "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
    "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    "hamming_loss": float(hamming_loss(y_test, y_pred)),
    "exact_match_accuracy": float(accuracy_score(y_test, y_pred)),
    "classification_report": classification_report(
        y_test,
        y_pred,
        target_names=mlb.classes_,
        zero_division=0,
        output_dict=True,
    ),
}

with (DATA_DIR / "tuned_results.json").open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nSaved tuned model to:", MODEL_DIR / "opp115_tuned_model.joblib")
print("Saved results to:", DATA_DIR / "tuned_results.json")
