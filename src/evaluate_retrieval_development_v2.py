"""Compare BM25, dense, hybrid and reranked retrieval on the same-language development set.

Place beside search_dense.py in src/ and run from the project environment.
"""
import csv
import hashlib
from datetime import datetime
from uuid import uuid4
import json
import re
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi
from search_hybrid import HybridRetriever
from search_reranked import RerankedRetriever

ROOT = Path(__file__).resolve().parent.parent


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[\u0610-\u061a\u064b-\u065f\u0670ـ]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    return re.findall(r"\w+", text, flags=re.UNICODE)


def metrics(ranked, gold):
    first = next((i for i, key in enumerate(ranked[:5], 1) if key in gold), None)
    return {
        "recall_at5": len(set(ranked[:5]) & gold) / len(gold),
        "recall_at10": len(set(ranked[:10]) & gold) / len(gold),
        "reciprocal_rank_at5": 1 / first if first else 0,
    }


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    articles = json.loads((ROOT / "data/processed/legal_articles.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "data/evaluation/qa_gold.json").read_text(encoding="utf-8"))
    # Keep held-out test questions out of iterative model comparisons.
    development = [q for q in questions if q.get("split") == "development"]
    selected = [q for q in development if q.get("answerable") is True]
    if not selected:
        raise ValueError("No answerable development questions found.")
    if len({q["id"] for q in selected}) != len(selected):
        raise ValueError("Duplicate question IDs")

    indexes = {}
    for language in ("ar", "en"):
        records = [a for a in articles if a["language"] == language]
        indexes[language] = (records, BM25Okapi([tokenize(a["text"]) for a in records]))

    # Validate every gold reference before encoding queries.
    for q in selected:
        records, _ = indexes[q["language"]]
        available = {(a["document_id"], a["article_number"]) for a in records}
        gold = {(a["document_id"], int(a["article_number"])) for a in q["gold_articles"]}
        if not gold or gold - available:
            raise ValueError(f"{q['id']}: empty or unavailable gold references: {gold - available}")

    print("Loading shared dense and hybrid retrievers...", flush=True)
    hybrid = HybridRetriever()
    dense = hybrid.dense
    reranked = RerankedRetriever(hybrid=hybrid)
    details = []
    for position, q in enumerate(selected, 1):
        print(f"Evaluating {position}/{len(selected)}: {q['id']}", flush=True)
        language = q["language"]
        gold = {(a["document_id"], int(a["article_number"])) for a in q["gold_articles"]}
        records, bm25 = indexes[language]
        scores = bm25.get_scores(tokenize(q["question"]))
        order = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
        bm25_keys = [
            (records[i]["document_id"], records[i]["article_number"])
            for i in order if scores[i] > 0
        ][:10]
        dense_keys = [
            (r["article"]["document_id"], r["article"]["article_number"])
            for r in dense.search(q["question"], k=10, language=language)
        ]
        candidates = hybrid.search(
            q["question"], k=20, language=language, candidate_k=20,
        )
        hybrid_keys = [
            (r["article"]["document_id"], r["article"]["article_number"])
            for r in candidates[:10]
        ]
        reranked_results = reranked.rerank(q["question"], candidates, k=10)
        reranked_keys = [
            (r["article"]["document_id"], r["article"]["article_number"])
            for r in reranked_results
        ]
        for method, ranked in (
            ("bm25", bm25_keys), ("dense", dense_keys),
            ("hybrid", hybrid_keys), ("hybrid_reranked", reranked_keys),
        ):
            details.append({
                "question_id": q["id"],
                "language": language,
                "method": method,
                "split": q["split"],
                "review_status": q.get("review_status"),
                "retrieved_top10": ";".join(f"{d}:{n}" for d, n in ranked),
                "gold_articles": ";".join(f"{d}:{n}" for d, n in sorted(gold)),
                **metrics(ranked, gold),
            })

    grouped = defaultdict(list)
    for row in details:
        grouped[(row["language"], row["method"])].append(row)
    summary = []
    for (language, method), rows in sorted(grouped.items()):
        averages = {
            name: sum(r[name] for r in rows) / len(rows)
            for name in ("recall_at5", "recall_at10", "reciprocal_rank_at5")
        }
        summary.append({"language": language, "method": method, "n": len(rows), **averages})
        print(
            f"{language} | {method} | n={len(rows)} | "
            f"Recall@5={averages['recall_at5']:.3f} | "
            f"Recall@10={averages['recall_at10']:.3f} | "
            f"MRR@5={averages['reciprocal_rank_at5']:.3f}"
        )
    output = ROOT / "reports" / ("retrieval_development_60q_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "split": "development", "same_language_only": True,
        "selected_ids": [q["id"] for q in selected],
        "selected_answerable": len(selected),
        "excluded_development": len(development)-len(selected),
        "qa_sha256": hashlib.sha256((ROOT / "data/evaluation/qa_gold.json").read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256((ROOT / "data/processed/legal_articles.json").read_bytes()).hexdigest(),
        "candidate_k": 20, "methods": ["bm25", "dense", "hybrid", "hybrid_reranked"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2),encoding="utf-8")
    save_csv(output / "reranker_comparison_details.csv", details)
    save_csv(output / "reranker_comparison_summary.csv", summary)
    print(f"Saved comparison reports to: {output}")
    print(f"Excluded unanswerable/pending development questions: {len(development) - len(selected)}")
    print("Reranking pool: top 20 hybrid articles; maximum passage score per article.")
    print("Same-language retrieval; scores are article-level, not chunk-level.")
    if any(q.get("review_status") != "reviewed" for q in selected):
        print("Provisional results: reference answers still await final review.")


if __name__ == "__main__":
    main()
