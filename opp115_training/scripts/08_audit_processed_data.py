import csv
from pathlib import Path
from collections import Counter, defaultdict
import re

path = Path("data/processed/opp115_segments.csv")

with path.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

print("=" * 55)
print("PROCESSED DATA QUALITY AUDIT")
print("=" * 55)

print("Total segments:", len(rows))
print("Empty texts:", sum(not r["text"].strip() for r in rows))
print("Texts shorter than 20 characters:",
      sum(len(r["text"].strip()) < 20 for r in rows))
print("Texts shorter than 100 characters:",
      sum(len(r["text"].strip()) < 100 for r in rows))
print("Empty labels:", sum(not r["labels"].strip() for r in rows))

# Check repeated segment IDs within the same policy.
segment_keys = [
    (r["policy_file"], r["segment_id"]) for r in rows
]
print("Duplicate policy/segment keys:",
      len(segment_keys) - len(set(segment_keys)))

# Count labels and number of labels per segment.
category_counts = Counter()
label_counts = Counter()
for r in rows:
    labels = [x for x in r["labels"].split("|") if x]
    category_counts.update(labels)
    label_counts[len(labels)] += 1

print("\nLABELS PER SEGMENT")
for n, count in sorted(label_counts.items()):
    print(f"{n} label(s): {count}")

print("\nCATEGORY COUNTS")
for label, count in category_counts.most_common():
    print(f"{label}: {count}")

# Find exact normalized texts used in multiple policies.
text_policies = defaultdict(set)
for r in rows:
    normalized = re.sub(r"\s+", " ", r["text"].strip().lower())
    if normalized:
        text_policies[normalized].add(r["policy_file"])

cross_policy_duplicates = {
    text: policies
    for text, policies in text_policies.items()
    if len(policies) > 1
}

print("\nDUPLICATE TEXT AUDIT")
print("Unique normalized texts:", len(text_policies))
print("Texts appearing in multiple policies:",
      len(cross_policy_duplicates))

for text, policies in list(cross_policy_duplicates.items())[:5]:
    print("\nText:", text[:180])
    print("Number of policies:", len(policies))

print("\nTEXT LENGTHS")
lengths = sorted(len(r["text"]) for r in rows)
if lengths:
    print("Minimum:", lengths[0])
    print("Median:", lengths[len(lengths) // 2])
    print("Maximum:", lengths[-1])

print("\nAudit complete. No data was modified.")
