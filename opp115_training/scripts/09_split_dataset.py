import csv
from pathlib import Path
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit

INPUT = Path("data/processed/opp115_segments_clean.csv")
OUTPUT_DIR = Path("data/processed")
RANDOM_STATE = 42

with INPUT.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError("Dataset is empty.")

groups = [row["policy_file"] for row in rows]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(rows, groups=groups)
)

train_rows = [rows[i] for i in train_idx]
test_rows = [rows[i] for i in test_idx]

train_policies = {r["policy_file"] for r in train_rows}
test_policies = {r["policy_file"] for r in test_rows}
overlap = train_policies & test_policies

if overlap:
    raise RuntimeError("Policy leakage detected!")

fields = [
    "policy_file", "segment_id", "text", "labels", "num_labels"
]

for filename, data in [
    ("train.csv", train_rows),
    ("test.csv", test_rows),
]:
    with (OUTPUT_DIR / filename).open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

def count_labels(data):
    counts = Counter()
    for row in data:
        counts.update(
            label for label in row["labels"].split("|") if label
        )
    return counts

train_labels = count_labels(train_rows)
test_labels = count_labels(test_rows)

print("=" * 50)
print("GROUPED TRAIN/TEST SPLIT")
print("=" * 50)
print(f"Total segments: {len(rows)}")
print(f"Train segments: {len(train_rows)} ({len(train_rows)/len(rows):.1%})")
print(f"Test segments:  {len(test_rows)} ({len(test_rows)/len(rows):.1%})")
print(f"Total policies: {len(set(groups))}")
print(f"Train policies: {len(train_policies)}")
print(f"Test policies:  {len(test_policies)}")
print(f"Policies shared between sets: {len(overlap)}")
print("\nLABEL DISTRIBUTION")
print(f"{'Category':45} {'Train':>7} {'Test':>7}")
for label in sorted(set(train_labels) | set(test_labels)):
    print(f"{label:45} {train_labels[label]:7} {test_labels[label]:7}")

print("\nSaved:")
print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "test.csv")
