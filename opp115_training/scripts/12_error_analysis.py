import json
from pathlib import Path

import pandas as pd
import joblib

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

# Load tuned model and test data
test_df = pd.read_csv(DATA_DIR / "test_no_train_duplicates.csv")
model = joblib.load(MODEL_DIR / "opp115_tuned_model.joblib")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer_tuned.joblib")
mlb = joblib.load(MODEL_DIR / "label_binarizer_tuned.joblib")

with (DATA_DIR / "tuned_results.json").open(encoding="utf-8") as f:
    results = json.load(f)

labels = list(mlb.classes_)
thresholds = results["thresholds"]

def parse_labels(value):
    return {x.strip() for x in str(value).split("|") if x.strip()}

# Predict using the tuned thresholds
X_test = vectorizer.transform(test_df["text"].fillna("").astype(str))
probabilities = model.predict_proba(X_test)

threshold_array = [thresholds[label] for label in labels]
predictions = probabilities >= threshold_array

true_labels = [parse_labels(x) for x in test_df["labels"]]
predicted_labels = [
    {labels[j] for j, value in enumerate(row) if value}
    for row in predictions
]

# Build detailed prediction records
records = []

for i, (_, row) in enumerate(test_df.iterrows()):
    actual = true_labels[i]
    predicted = predicted_labels[i]

    missed = actual - predicted
    extra = predicted - actual
    correct = actual & predicted

    records.append({
        "policy_file": row["policy_file"],
        "segment_id": row["segment_id"],
        "text": row["text"],
        "true_labels": " | ".join(sorted(actual)),
        "predicted_labels": " | ".join(sorted(predicted)),
        "correct_labels": " | ".join(sorted(correct)),
        "missed_labels": " | ".join(sorted(missed)),
        "extra_labels": " | ".join(sorted(extra)),
        "num_missed": len(missed),
        "num_extra": len(extra),
        "is_exact_match": actual == predicted,
    })

output_df = pd.DataFrame(records)

output_df.to_csv(
    DATA_DIR / "test_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)

errors_df = output_df[
    (output_df["num_missed"] > 0) |
    (output_df["num_extra"] > 0)
].copy()

errors_df.to_csv(
    DATA_DIR / "error_cases.csv",
    index=False,
    encoding="utf-8-sig",
)

print("=" * 60)
print("OPP-115 ERROR ANALYSIS")
print("=" * 60)
print("Total test samples:", len(output_df))
print("Exact matches:", int(output_df["is_exact_match"].sum()))
print("Samples with at least one error:", len(errors_df))
print(
    "Exact match rate:",
    round(output_df["is_exact_match"].mean(), 4),
)

print("\nMOST FREQUENT MISSED LABELS")
missed_counts = {}
for value in output_df["missed_labels"]:
    for label in parse_labels(value):
        missed_counts[label] = missed_counts.get(label, 0) + 1

for label, count in sorted(
    missed_counts.items(), key=lambda item: item[1], reverse=True
):
    print(f"{label}: {count}")

print("\nMOST FREQUENT EXTRA LABELS")
extra_counts = {}
for value in output_df["extra_labels"]:
    for label in parse_labels(value):
        extra_counts[label] = extra_counts.get(label, 0) + 1

for label, count in sorted(
    extra_counts.items(), key=lambda item: item[1], reverse=True
):
    print(f"{label}: {count}")

print("\nEXAMPLE ERROR CASES")
for _, row in errors_df.head(5).iterrows():
    print("-" * 60)
    print("Text:", str(row["text"])[:400])
    print("Actual:", row["true_labels"])
    print("Predicted:", row["predicted_labels"])
    print("Missed:", row["missed_labels"])
    print("Extra:", row["extra_labels"])

print("\nSaved:")
print(DATA_DIR / "test_predictions.csv")
print(DATA_DIR / "error_cases.csv")
