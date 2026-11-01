import re

from rank_bm25 import BM25Okapi

from search_dense import DenseRetriever


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[\u0610-\u061a\u064b-\u065f\u0670ـ]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    return re.findall(r"\w+", text, flags=re.UNICODE)


class HybridRetriever:
    def __init__(self):
        self.dense = DenseRetriever()
        self.indexes = {}

        for language in ("ar", "en"):
            records = [
                a for a in self.dense.articles.values()
                if a["language"] == language
            ]
            self.indexes[language] = (
                records,
                BM25Okapi([tokenize(a["text"]) for a in records]),
            )

    def search(self, question, k=5, language=None, candidate_k=20):
        if not question.strip():
            return []

        if language is None:
            language = (
                "ar"
                if re.search(r"[\u0600-\u06ff]", question)
                else "en"
            )

        candidate_k = max(candidate_k, k)
        records, bm25 = self.indexes[language]
        scores = bm25.get_scores(tokenize(question))

        order = sorted(
            range(len(records)),
            key=lambda i: scores[i],
            reverse=True,
        )
        bm25_articles = [
            records[i] for i in order if scores[i] > 0
        ][:candidate_k]

        dense_results = self.dense.search(
            question, k=candidate_k, language=language
        )

        fused = {}

        # Reciprocal Rank Fusion:
        # each result contributes 1 / (60 + its rank).
        ranked_lists = [
            bm25_articles,
            [r["article"] for r in dense_results],
        ]

        for method, ranked in zip(("bm25", "dense"), ranked_lists):
            for rank, article in enumerate(ranked, start=1):
                key = (
                    article["document_id"],
                    article["article_number"],
                )

                if key not in fused:
                    fused[key] = {
                        "article": article,
                        "score": 0.0,
                        "bm25_rank": None,
                        "dense_rank": None,
                    }

                fused[key]["score"] += 1 / (60 + rank)
                fused[key][f"{method}_rank"] = rank

        return sorted(
            fused.values(),
            key=lambda item: item["score"],
            reverse=True,
        )[:k]


def main():
    print("Loading hybrid retriever...", flush=True)
    retriever = HybridRetriever()
    print("Ready. Type exit to stop.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break
        if not question:
            continue

        for rank, result in enumerate(
            retriever.search(question), start=1
        ):
            article = result["article"]

            print("\n" + "=" * 50)
            print(
                f"{rank}. {article['document_id']}:"
                f"{article['article_number']} | "
                f"{article['language']}"
            )
            print(f"RRF score: {result['score']:.5f}")
            print(
                f"BM25 rank: {result['bm25_rank']} | "
                f"Dense rank: {result['dense_rank']}"
            )
            print(article["text"])

        print("\nRRF score is not a confidence percentage.")


if __name__ == "__main__":
    main()