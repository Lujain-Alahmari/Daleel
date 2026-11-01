import json
import warnings
from pathlib import Path

import pandas as pd
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
from sklearn.exceptions import UndefinedMetricWarning
import joblib

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Paths
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test_no_train_duplicates.csv")

# Prepare multi-label targets
def parse_labels(value):
    return [label.strip() for label in str(value).split("|") if label.strip()]

train_labels = train_df["labels"].apply(parse_labels)
test_labels = test_df["labels"].apply(parse_labels)

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_labels)
y_test = mlb.transform(test_labels)

# Prepare text
X_train_text = train_df["text"].fillna("").astype(str)
X_test_text = test_df["text"].fillna("").astype(str)

# TF-IDF features
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Train baseline classifier
model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
report = classification_report(
    y_test,
    y_pred,
    target_names=mlb.classes_,
    zero_division=0,
    output_dict=True,
)

print("=" * 60)
print("OPP-115 MULTI-LABEL BASELINE")
print("=" * 60)
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))
print("Number of labels:", len(mlb.classes_))
print("TF-IDF features:", X_train.shape[1])
print()
print("Micro F1:", round(f1_score(y_test, y_pred, average="micro", zero_division=0), 4))
print("Macro F1:", round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4))
print("Hamming Loss:", round(hamming_loss(y_test, y_pred), 4))
print("Exact Match Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print()
print("PER-CATEGORY REPORT")
print("-" * 60)
print(classification_report(
    y_test,
    y_pred,
    target_names=mlb.classes_,
    zero_division=0,
))

# Save model artifacts and evaluation results
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
joblib.dump(model, MODEL_DIR / "opp115_baseline_model.joblib")
joblib.dump(mlb, MODEL_DIR / "label_binarizer.joblib")

results = {
    "train_samples": len(train_df),
    "test_samples": len(test_df),
    "categories": list(mlb.classes_),
    "micro_f1": float(f1_score(y_test, y_pred, average="micro", zero_division=0)),
    "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    "hamming_loss": float(hamming_loss(y_test, y_pred)),
    "exact_match_accuracy": float(accuracy_score(y_test, y_pred)),
    "classification_report": report,
}

with (DATA_DIR / "baseline_results.json").open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved model files to:", MODEL_DIR)
print("Saved evaluation results to:", DATA_DIR / "baseline_results.json")
