"""
Polished version of the local cross-encoder reranker built in 5-rerank.py.

6-evaluate.py imports from here so the eval script can focus on the metric.

This version replaces Cohere with a local BGE reranker:
- no COHERE_API_KEY
- no API quota
- no 429 rate-limit error
"""

import pandas as pd
from sentence_transformers import CrossEncoder

from utils.fusion import hybrid_candidates
from utils.retrievers import BM25Retriever, DenseRetriever

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Load once when this module is imported
_reranker = CrossEncoder(RERANK_MODEL)


def rerank_with_local_model(
    query: str,
    candidate_ids: list[str],
    corpus_by_id: pd.DataFrame,
    k: int = 10,
) -> list[tuple[str, float]]:
    """
    Rerank candidate documents with a local cross-encoder.

    Returns:
        list of (doc_id, rerank_score)
    """

    valid_candidates: list[tuple[str, str]] = []

    for doc_id in candidate_ids:
        if doc_id not in corpus_by_id.index:
            continue

        text = str(corpus_by_id.loc[doc_id, "text"])
        valid_candidates.append((doc_id, text))

    if not valid_candidates:
        return []

    pairs = [(query, text) for _, text in valid_candidates]

    scores = _reranker.predict(pairs)

    ranked = sorted(
        zip(valid_candidates, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (doc_id, float(score))
        for (doc_id, _), score in ranked[:k]
    ]


# Backward-compatible name.
# This lets old imports keep working:
# from utils.reranker import rerank_with_cohere
def rerank_with_cohere(
    query: str,
    candidate_ids: list[str],
    corpus_by_id: pd.DataFrame,
    k: int = 10,
) -> list[tuple[str, float]]:
    return rerank_with_local_model(
        query=query,
        candidate_ids=candidate_ids,
        corpus_by_id=corpus_by_id,
        k=k,
    )


def search_reranked(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    corpus_by_id: pd.DataFrame,
    k: int = 10,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    candidates = hybrid_candidates(
        query,
        bm25,
        dense,
        candidate_k=candidate_k,
    )

    candidate_ids = [doc_id for doc_id, _ in candidates]

    return rerank_with_local_model(
        query=query,
        candidate_ids=candidate_ids,
        corpus_by_id=corpus_by_id,
        k=k,
    )