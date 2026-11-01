"""Build a multilingual FAISS index; put this script in the project's src folder."""
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data/processed/legal_articles.json"
OUTPUT = ROOT / "data/indexes/dense_e5"
MODEL = "intfloat/multilingual-e5-base"
WINDOW = 380
OVERLAP = 64


def main():
    source_bytes = INPUT.read_bytes()
    articles = json.loads(source_bytes.decode("utf-8"))
    if not articles:
        raise ValueError("The corpus is empty.")

    print(f"Loading {MODEL} on CPU. The first run downloads the model.", flush=True)
    model = SentenceTransformer(MODEL, device="cpu")
    model.max_seq_length = 512
    chunks = []

    for article in articles:
        text = article["text"]
        tokens = model.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
            truncation=False,
        )
        offsets = tokens["offset_mapping"]
        if not offsets:
            raise ValueError(f"Empty article: {article['id']}")

        # Subdivide long articles for embedding only; retain their full text
        # and document/article identity in metadata for retrieval and citations.
        for part, start in enumerate(range(0, len(offsets), WINDOW - OVERLAP)):
            end = min(start + WINDOW, len(offsets))
            chunk_text = text[offsets[start][0]:offsets[end - 1][1]]
            length = len(model.tokenizer(
                "passage: " + chunk_text, truncation=False
            )["input_ids"])
            if length > model.max_seq_length:
                raise ValueError(f"Chunk exceeds model limit: {article['id']}/{part}")
            chunks.append({
                "article_id": article["id"],
                "document_id": article["document_id"],
                "article_number": article["article_number"],
                "language": article["language"],
                "chunk_number": part,
                "chunk_text": chunk_text,
            })
            if end == len(offsets):
                break

    print(f"Articles: {len(articles)}; embedding chunks: {len(chunks)}", flush=True)
    embeddings = model.encode(
        ["passage: " + c["chunk_text"] for c in chunks],
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite embeddings")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    assert index.ntotal == len(chunks)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUTPUT / "index.faiss"))
    manifest = {
        "model": MODEL,
        "corpus_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "article_count": len(articles),
        "chunk_count": len(chunks),
        "embedding_dimension": embeddings.shape[1],
        "max_sequence_length": model.max_seq_length,
        "window_tokens": WINDOW,
        "overlap_tokens": OVERLAP,
        "normalized": True,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps({"manifest": manifest, "chunks": chunks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Reopen the saved index to check export consistency.
    saved = faiss.read_index(str(OUTPUT / "index.faiss"))
    assert saved.ntotal == len(chunks) and saved.d == embeddings.shape[1]
    print(f"Saved FAISS index and metadata to: {OUTPUT}")


if __name__ == "__main__":
    main()
