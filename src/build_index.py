"""Build two ChromaDB collections (baseline + engineered) from the corpus.

Every chunk is tagged with `year`, `month`, and `source_file` metadata. The
engineered retriever uses year/month to pre-filter the search space; the
evaluator uses source_file to score retrieval against the answer key.
"""
import sys

import chromadb
from sentence_transformers import SentenceTransformer

import config
from src import chunking
from src.common import parse_year_month

_MODELS: dict[str, SentenceTransformer] = {}


def get_embedder(name: str) -> SentenceTransformer:
    if name not in _MODELS:
        print(f"  loading embedding model: {name}")
        _MODELS[name] = SentenceTransformer(name)
    return _MODELS[name]


def _build(collection_name: str, embed_model: str, chunk_fn) -> int:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    embedder = get_embedder(embed_model)

    ids, docs, metas = [], [], []
    files = sorted(config.CORPUS_DIR.glob("*.txt"))
    for fp in files:
        year, month = parse_year_month(fp.name)
        text = fp.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(chunk_fn(text)):
            ids.append(f"{fp.stem}__{i}")
            docs.append(chunk)
            metas.append({"source_file": fp.name, "year": year, "month": month})

    print(f"  {collection_name}: {len(docs)} chunks from {len(files)} docs -> embedding...")
    embs = embedder.encode(docs, batch_size=64, show_progress_bar=True,
                           normalize_embeddings=True).tolist()
    B = 2000
    for s in range(0, len(docs), B):
        col.add(ids=ids[s:s+B], documents=docs[s:s+B],
                embeddings=embs[s:s+B], metadatas=metas[s:s+B])
    return len(docs)


def main():
    config.CHROMA_DIR.mkdir(exist_ok=True)
    print("[1/2] Baseline index (fixed-width chunks, MiniLM, no metadata filtering)")
    n_base = _build(config.COLLECTIONS["baseline"], config.BASELINE_EMBED_MODEL,
                    chunking.chunk_baseline)
    print("[2/2] Engineered index (structure-aware chunks, bge, year/month tags)")
    n_eng = _build(config.COLLECTIONS["engineered"], config.ENGINEERED_EMBED_MODEL,
                   chunking.chunk_engineered)
    print(f"\nDone. baseline={n_base} chunks, engineered={n_eng} chunks.")


if __name__ == "__main__":
    sys.exit(main())
