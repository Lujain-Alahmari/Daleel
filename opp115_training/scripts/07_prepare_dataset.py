import csv
import re
from pathlib import Path
from collections import defaultdict, Counter
from html.parser import HTMLParser

BASE = Path("opp115")
ANNOTATIONS = BASE / "consolidation/threshold-0.75-overlap-similarity"
POLICIES = BASE / "sanitized_policies"
OUTPUT = Path("data/processed/opp115_segments.csv")


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def get_text(self):
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def clean_text(text):
    parser = TextExtractor()
    parser.feed(text)
    return parser.get_text()


def main():
    policy_segments = {}

    for path in POLICIES.glob("*.html"):
        content = path.read_text(encoding="utf-8", errors="replace")
        policy_segments[path.stem] = [
            clean_text(s) for s in content.split("|||")
        ]

    labels_by_segment = defaultdict(set)
    missing_files = set()
    invalid_ids = 0
    rows_read = 0

    for csv_path in ANNOTATIONS.glob("*.csv"):
        key = csv_path.stem

        if key not in policy_segments:
            missing_files.add(csv_path.name)
            continue

        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 9:
                    continue

                rows_read += 1
                category = row[5].strip()

                try:
                    segment_id = int(row[4])
                except (ValueError, TypeError):
                    invalid_ids += 1
                    continue

                segments = policy_segments[key]

                if not 0 <= segment_id < len(segments):
                    invalid_ids += 1
                    continue

                if category and segments[segment_id]:
                    labels_by_segment[(key, segment_id)].add(category)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    category_counts = Counter()
    label_counts = Counter()

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["policy_file", "segment_id", "text", "labels", "num_labels"]
        )
        writer.writeheader()

        for (key, segment_id), labels in sorted(labels_by_segment.items()):
            writer.writerow({
                "policy_file": key,
                "segment_id": segment_id,
                "text": policy_segments[key][segment_id],
                "labels": "|".join(sorted(labels)),
                "num_labels": len(labels),
            })
            category_counts.update(labels)
            label_counts[len(labels)] += 1

    print("=" * 50)
    print("OPP-115 DATA PREPARATION REPORT")
    print("=" * 50)
    print("Annotation rows read:", rows_read)
    print("Unique labeled segments:", len(labels_by_segment))
    print("Categories:", len(category_counts))
    print("Invalid segment IDs:", invalid_ids)
    print("Annotation files without matching policy files:", len(missing_files))
    print("Segments with multiple labels:",
          sum(n for k, n in label_counts.items() if k > 1))
    print("Output:", OUTPUT)

    print("\nCATEGORY COUNTS")
    for category, count in category_counts.most_common():
        print(f"{category}: {count}")

    print("\nSAMPLE")
    with OUTPUT.open(encoding="utf-8", newline="") as f:
        for row in list(csv.DictReader(f))[:3]:
            print("\nPolicy:", row["policy_file"])
            print("Segment:", row["segment_id"])
            print("Labels:", row["labels"])
            print("Text:", row["text"][:200])


if __name__ == "__main__":
    main()
