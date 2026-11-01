"""Hybrid retrieval followed by a multilingual cross-encoder.

Place beside search_hybrid.py in the project's src directory.
"""
import math

import numpy as np
from sentence_transformers import CrossEncoder

from search_hybrid import HybridRetriever

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 512


class RerankedRetriever:
    def __init__(self, hybrid=None):
        self.hybrid = hybrid if hybrid is not None else HybridRetriever()
        print(f"Loading {MODEL_NAME} on CPU. First run downloads weights.", flush=True)
        self.reranker = CrossEncoder(MODEL_NAME, device="cpu", max_length=MAX_LENGTH)

    def rerank(self, question, candidates, k=5):
        if k < 1:
            raise ValueError("k must be positive")
        question = question.strip()
        if not question or not candidates:
            return []
        tokenizer = self.reranker.tokenizer
        question_tokens = tokenizer(question, add_special_tokens=False)["input_ids"]
        budget = MAX_LENGTH - len(question_tokens) - tokenizer.num_special_tokens_to_add(pair=True)
        if budget < 96:
            raise ValueError("Question is too long. Shorten it before reranking.")

        pairs, owners, passages = [], [], []
        for owner, result in enumerate(candidates):
            text = result["article"]["text"]
            encoded = tokenizer(
                question, text,
                truncation="only_second",
                max_length=MAX_LENGTH,
                stride=min(64, budget // 4),
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )
            # Obtain every passage window; do not silently discard article tails.
            for window, offsets in enumerate(encoded["offset_mapping"]):
                sequence_ids = encoded.sequence_ids(window)
                spans = [
                    (start, end)
                    for (start, end), sequence in zip(offsets, sequence_ids)
                    if sequence == 1 and end > start
                ]
                if not spans:
                    continue
                passage = text[spans[0][0]:spans[-1][1]]
                # Guard against boundary retokenization changing the pair length.
                pair_length = len(tokenizer(question, passage, truncation=False)["input_ids"])
                if pair_length > MAX_LENGTH:
                    raise ValueError("Passage boundary exceeds token limit; review chunking.")
                pairs.append((question, passage))
                owners.append(owner)
                passages.append(passage)

        if not pairs:
            return []
        print(f"Reranking {len(candidates)} articles / {len(pairs)} passage pairs...", flush=True)
        scores = np.asarray(self.reranker.predict(
            pairs, batch_size=2, show_progress_bar=True, convert_to_numpy=True,
        )).reshape(-1)
        if len(scores) != len(pairs) or not np.isfinite(scores).all():
            raise ValueError("Unexpected reranker scores")
        best = {}
        for owner, passage, score in zip(owners, passages, scores):
            if owner not in best or score > best[owner][0]:
                best[owner] = (float(score), passage)

        results = []
        for owner, (score, passage) in best.items():
            result = candidates[owner]
            results.append({
                "article": result["article"],
                "score": score,
                "hybrid_score": result["score"],
                "hybrid_rank": owner + 1,
                "matched_chunk": passage,
            })
        # Stable secondary order preserves the hybrid order on tied scores.
        return sorted(results, key=lambda r: (-r["score"], r["hybrid_rank"]))[:k]

    def search(self, question, k=5, language=None, candidate_k=20):
        candidate_k = max(candidate_k, k)
        candidates = self.hybrid.search(
            question, k=candidate_k, language=language, candidate_k=candidate_k,
        )
        return self.rerank(question, candidates, k=k)


def main():
    retriever = RerankedRetriever()
    print("Ready. Type exit to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        for rank, result in enumerate(retriever.search(question), 1):
            article = result["article"]
            print("\n" + "=" * 50)
            print(f"{rank}. {article['document_id']}:{article['article_number']} | {article['language']}")
            print(f"Reranker score: {result['score']:.5f} | Previous hybrid rank: {result['hybrid_rank']}")
            print(article["text"])
        print("\nReranker scores are not confidence percentages. This is retrieval, not a generated answer.")


if __name__ == "__main__":
    main()
