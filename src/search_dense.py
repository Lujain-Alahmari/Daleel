"""Search an existing E5 FAISS index. Put in the project's src directory."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/processed/legal_articles.json"
INDEX_DIR = ROOT / "data/indexes/dense_e5"


class DenseRetriever:
    def __init__(self):
        payload = json.loads((INDEX_DIR / "metadata.json").read_text(encoding="utf-8"))
        self.manifest = payload["manifest"]
        self.chunks = payload["chunks"]
        source = CORPUS.read_bytes()
        if hashlib.sha256(source).hexdigest() != self.manifest["corpus_sha256"]:
            raise ValueError("Corpus changed. Rebuild using src/build_dense_index.py.")
        self.articles = {a["id"]: a for a in json.loads(source.decode("utf-8"))}
        self.index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
        if self.index.ntotal != len(self.chunks):
            raise ValueError("Index and metadata do not match. Rebuild the index.")
        self.model = SentenceTransformer(self.manifest["model"], device="cpu")
        self.model.max_seq_length = self.manifest["max_sequence_length"]

    def search(self, question, k=5, language=None):
        """Maximum chunk score per article, optionally restricted to a language."""
        if not question.strip():
            return []
        vector = self.model.encode(
            [self.manifest["query_prefix"] + question.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores, positions = self.index.search(
            np.ascontiguousarray(vector, dtype=np.float32), self.index.ntotal
        )
        seen, results = set(), []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            chunk = self.chunks[int(position)]
            if language and chunk["language"] != language:
                continue
            # Ignore duplicate chunks; when searching both languages, count
            # translations of the same document/article as one legal article.
            key = (chunk["document_id"], chunk["article_number"])
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "article": self.articles[chunk["article_id"]],
                "score": float(score),
                "matched_chunk": chunk["chunk_text"],
            })
            if len(results) >= k:
                break
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-languages", action="store_true")
    args = parser.parse_args()
    print("Loading the existing dense index...", flush=True)
    retriever = DenseRetriever()
    print("Ready. Default: retrieve in the question's language. Type exit to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        language = None if args.all_languages else (
            "ar" if re.search(r"[\u0600-\u06ff]", question) else "en"
        )
        for rank, result in enumerate(retriever.search(question, language=language), 1):
            article = result["article"]
            print("\n" + "=" * 50)
            print(f"{rank}. {article['document_id']}:{article['article_number']} | {article['language']}")
            print(f"Similarity: {result['score']:.4f}")
            print(article["text"])
        print("\nSimilarity is not a confidence percentage. Results are retrieved text, not an answer.")


if __name__ == "__main__":
    main()
