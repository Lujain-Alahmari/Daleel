
from pathlib import Path
from collections import Counter
import json

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

DATA_PATH = Path("data/processed/test_predictions.csv")
OUTPUT_DIR = Path("data/processed/baseline_error_analysis")


def parse_labels(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    return {
        label.strip()
        for label in str(value).split("|")
        if label.strip()
    }


def main():
    df = pd.read_csv(DATA_PATH)

    required = {
        "text",
        "true_labels",
        "predicted_labels",
        "missed_labels",
        "extra_labels",
        "is_exact_match",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    true_sets = df["true_labels"].apply(parse_labels)
    pred_sets = df["predicted_labels"].apply(parse_labels)

    categories = sorted(set.union(
        set(),
        *true_sets.tolist(),
        *pred_sets.tolist(),
    ))

    rows = []
    for category in categories:
        tp = sum(
            category in true and category in pred
            for true, pred in zip(true_sets, pred_sets)
        )
        fp = sum(
            category not in true and category in pred
            for true, pred in zip(true_sets, pred_sets)
        )
        fn = sum(
            category in true and category not in pred
            for true, pred in zip(true_sets, pred_sets)
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        rows.append({
            "category": category,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    metrics = pd.DataFrame(rows).sort_values("f1")
    metrics.to_csv(
        OUTPUT_DIR / "category_error_metrics.csv",
        index=False,
    )

    missed_counts = Counter()
    extra_counts = Counter()
    confusion_pairs = Counter()

    for true, pred in zip(true_sets, pred_sets):
        missed = true - pred
        extra = pred - true

        missed_counts.update(missed)
        extra_counts.update(extra)

        for missed_category in missed:
            for extra_category in extra:
                confusion_pairs[(missed_category, extra_category)] += 1

    missed_df = pd.DataFrame(
        missed_counts.most_common(),
        columns=["category", "missed_count"],
    )
    extra_df = pd.DataFrame(
        extra_counts.most_common(),
        columns=["category", "extra_prediction_count"],
    )
    confusion_df = pd.DataFrame(
        [
            {
                "missed_category": missed,
                "extra_category": extra,
                "count": count,
            }
            for (missed, extra), count in confusion_pairs.most_common()
        ],
        columns=["missed_category", "extra_category", "count"],
    )

    missed_df.to_csv(OUTPUT_DIR / "most_missed_categories.csv", index=False)
    extra_df.to_csv(OUTPUT_DIR / "most_extra_categories.csv", index=False)
    confusion_df.to_csv(OUTPUT_DIR / "confusion_pairs.csv", index=False)

    # Rank segments with the largest number of prediction errors.
    error_df = df.copy()
    error_df["error_count"] = (
        error_df["num_missed"].fillna(0)
        + error_df["num_extra"].fillna(0)
    )
    error_df = error_df[error_df["error_count"] > 0]
    error_df = error_df.sort_values(
        ["error_count", "num_missed", "num_extra"],
        ascending=False,
    )
    error_df.to_csv(OUTPUT_DIR / "largest_error_cases.csv", index=False)

    summary = {
        "test_rows": int(len(df)),
        "exact_match_count": int(df["is_exact_match"].astype(bool).sum()),
        "exact_match_rate": float(
            df["is_exact_match"].astype(bool).mean()
        ),
        "rows_with_errors": int((~df["is_exact_match"].astype(bool)).sum()),
        "total_missed_labels": int(sum(missed_counts.values())),
        "total_extra_labels": int(sum(extra_counts.values())),
    }

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 48)
    print("BASELINE ERROR ANALYSIS")
    print("=" * 48)
    print(f"Test rows: {summary['test_rows']}")
    print(f"Exact match rate: {summary['exact_match_rate']:.2%}")
    print(f"Rows with errors: {summary['rows_with_errors']}")
    print(f"Missed labels: {summary['total_missed_labels']}")
    print(f"Extra labels: {summary['total_extra_labels']}")

    print("\nLowest-F1 categories:")
    print(metrics[["category", "precision", "recall", "f1"]].head(5).to_string(
        index=False,
        formatters={
            "precision": "{:.3f}".format,
            "recall": "{:.3f}".format,
            "f1": "{:.3f}".format,
        },
    ))

    print("\nMost missed categories:")
    print(missed_df.head(5).to_string(index=False))

    print("\nMost extra-predicted categories:")
    print(extra_df.head(5).to_string(index=False))

    print("\nMost common confusion pairs:")
    print(confusion_df.head(10).to_string(index=False))

    print(f"\nReports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


