"""Retrieval for both systems.

BASELINE  : embed the raw question, cosine top-K over the whole collection.
ENGINEERED: prefix the bge query instruction, and PRE-FILTER by Year (and Month
            when the question names one) using Chroma metadata `where` clauses.
            Metadata filtering shrinks the candidate set to the right bulletins
            before ranking, which is the core retrieval improvement in this lab.
"""
import chromadb

import config
from src.build_index import get_embedder
from src.common import question_quarter, question_years

_client = None


def _col(name: str):
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client.get_collection(name)


def all_chunks_from(collection_name: str, source_file: str):
    """Every stored chunk belonging to one source document (for recall denominators)."""
    res = _col(collection_name).get(where={"source_file": source_file},
                                    include=["documents"])
    return res["documents"]


def _query(col, embedding, k, where=None):
    res = col.query(query_embeddings=[embedding], n_results=k, where=where,
                    include=["documents", "metadatas", "distances"])
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "source_file": meta["source_file"],
                     "year": meta["year"], "month": meta["month"], "distance": dist})
    return hits


def retrieve_baseline(question: str, k: int = config.TOP_K):
    emb = get_embedder(config.BASELINE_EMBED_MODEL).encode(
        question, normalize_embeddings=True).tolist()
    return _query(_col(config.COLLECTIONS["baseline"]), emb, k)


def build_where(question: str):
    """Pure function: question -> Chroma metadata `where` filter (or None).

    Parses years and any month/quarter, and expands years by +1 to cover
    publication lag (year-end data prints in the following bulletin). Kept
    separate from I/O so it is unit-testable without a live index.
    """
    years = question_years(question)
    quarter = question_quarter(question)
    if years:
        expanded = sorted({y for y in years} |
                          {str(int(y) + 1) for y in years if str(int(y) + 1) in config.YEARS})
        year_clause = {"year": {"$in": expanded}}
        # Only keep the month filter when we did NOT expand for lag, otherwise the
        # following-year issue (a different quarter) would be filtered right back out.
        return ({"$and": [year_clause, {"month": quarter}]}
                if quarter and expanded == years else year_clause)
    if quarter:
        return {"month": quarter}
    return None


def retrieve_engineered(question: str, k: int = config.TOP_K):
    emb = get_embedder(config.ENGINEERED_EMBED_MODEL).encode(
        config.ENGINEERED_QUERY_PREFIX + question, normalize_embeddings=True).tolist()
    col = _col(config.COLLECTIONS["engineered"])

    where = build_where(question)
    hits = _query(col, emb, k, where=where)
    # Safety net: if the metadata filter was too aggressive and returned nothing,
    # fall back to an unfiltered search so we never hard-fail a query.
    if not hits:
        hits = _query(col, emb, k, where=None)
    return hits
