"""Compare saved gap reports with frozen policy-text reference judgments; no API calls."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

STATUSES = {'covered', 'partially_covered', 'missing', 'needs_review', 'not_assessed', 'not_applicable'}
TEXT_STATUSES = {'covered', 'partially_covered', 'missing'}

def evaluate(annotations, reports):
    with Path(annotations).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('No reference judgments: no evaluation can be reported.')
    seen = set()
    expected = defaultdict(set)
    for row in rows:
        pid, rid, gold = row['policy_id'], row['requirement_id'], row['gold_status']
        if not pid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in pid):
            raise ValueError('Invalid policy identifier')
        if (pid, rid) in seen:
            raise ValueError('Duplicate reference judgment')
        if gold not in STATUSES or row.get('review_status') != 'reviewed':
            raise ValueError('Reference must have a valid status and completed text review')
        seen.add((pid, rid))
        expected[pid].add(rid)
    predicted, report_hashes = {}, {}
    for pid, ids in expected.items():
        path = Path(reports) / (pid + '.json')
        if not path.is_file():
            raise FileNotFoundError(f'Missing saved report for {pid}; no score produced')
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, list):
            raise ValueError(f'{pid}: expected a completed list report')
        index = {}
        for item in data:
            rid = item.get('requirement_id')
            if rid in index:
                raise ValueError(f'{pid}: duplicate report requirement')
            if rid not in ids:
                raise ValueError(f'{pid}: unknown report requirement')
            index[rid] = item.get('status', 'invalid_result')
        predicted[pid] = index
        report_hashes[pid] = hashlib.sha256(path.read_bytes()).hexdigest()
    confusion, eligible, text_confusion = Counter(), [], Counter()
    counts = Counter()
    for row in rows:
        pid, rid, gold = row['policy_id'], row['requirement_id'], row['gold_status']
        pred = predicted[pid].get(rid, 'no_result')
        if pred not in STATUSES:
            counts['invalid_or_missing_outputs'] += 1
        confusion[gold, pred] += 1
        counts['all_correct'] += gold == pred
        if gold in TEXT_STATUSES:
            eligible.append((gold, pred))
            text_confusion[gold, pred] += 1
            counts['text_correct'] += gold == pred
    per_class = {}
    for label in sorted(TEXT_STATUSES):
        tp = sum(g == label and p == label for g, p in eligible)
        fp = sum(g != label and p == label for g, p in eligible)
        fn = sum(g == label and p != label for g, p in eligible)
        support = sum(g == label for g, p in eligible)
        per_class[label] = {'support': support, 'precision': tp/(tp+fp) if tp+fp else 0.0,
                            'recall': tp/(tp+fn) if tp+fn else 0.0,
                            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}
    supported_f1 = [v['f1'] for v in per_class.values() if v['support']]
    return {'policies': len(expected), 'reference_judgments': len(rows),
            'text_judgments_scored': len(eligible), 'unresolved_or_out_of_scope_reference_judgments': len(rows)-len(eligible),
            'text_status_accuracy_primary': counts['text_correct']/len(eligible) if eligible else None,
            'text_macro_f1_supported_classes': sum(supported_f1)/len(supported_f1) if supported_f1 else None,
            'text_per_class': per_class,
            'all_status_agreement_secondary': counts['all_correct']/len(rows),
            'invalid_or_missing_outputs': counts['invalid_or_missing_outputs'],
            'minimum_ten_policies_met': len(expected) >= 10,
            'all_status_confusion': [{'gold': g, 'predicted': p, 'count': n} for (g,p),n in sorted(confusion.items())],
            'reference_sha256': hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
            'report_sha256': report_hashes, 'API_calls': 0,
            'reference_scope': 'Document-based review of frozen policy texts; independent team confirmation not established.',
            'limitations': ['Comparison checks saved statuses, not semantic correctness of model evidence.',
                           'Saved list reports do not prove the exact policy input or model configuration; retain experiment manifests.',
                           'Unresolved reference judgments are excluded from the primary text score and retained in the secondary stratum.']}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.annotations, args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
