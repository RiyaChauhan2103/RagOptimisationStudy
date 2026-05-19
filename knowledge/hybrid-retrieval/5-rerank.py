"""
Reranking: take the top-50 candidates from RRF and reorder them with a
local cross-encoder. Return the top-10.

A bi-encoder, such as the dense retriever from 3-embed.py, embeds the query
and document separately, then compares them with cosine similarity.

A cross-encoder feeds the query and document into the same model and returns
a single relevance score. Cross-encoders are slower, but usually more accurate
because they compare the query and document jointly.

This version uses a local BGE reranker instead of Cohere, so it does not need:
- COHERE_API_KEY
- Cohere quota
- paid API calls
"""

from sentence_transformers import CrossEncoder

from utils.fusion import hybrid_candidates
from utils.retrievers import BM25Retriever, DenseRetriever, load_corpus

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


# --------------------------------------------------------------
# Step 1: Load retrievers, corpus, and local reranker
# --------------------------------------------------------------

bm25 = BM25Retriever()
dense = DenseRetriever()
corpus_by_id = load_corpus().set_index("_id")

reranker = CrossEncoder(RERANK_MODEL)


# --------------------------------------------------------------
# Step 2: Rerank with local cross-encoder
# --------------------------------------------------------------


def search_reranked(
    query: str,
    k: int = 10,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    """
    Retrieve candidates using hybrid RRF, then rerank them locally.

    Returns:
        list of (doc_id, rerank_score)
    """

    candidates = hybrid_candidates(
        query=query,
        bm25=bm25,
        dense=dense,
        candidate_k=candidate_k,
    )

    candidate_ids = [doc_id for doc_id, _ in candidates]

    valid_candidates: list[tuple[str, str]] = []

    for doc_id in candidate_ids:
        if doc_id not in corpus_by_id.index:
            continue

        text = str(corpus_by_id.loc[doc_id, "text"])
        valid_candidates.append((doc_id, text))

    if not valid_candidates:
        return []

    pairs = [(query, text) for _, text in valid_candidates]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(valid_candidates, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (doc_id, float(score))
        for (doc_id, _), score in ranked[:k]
    ]


# --------------------------------------------------------------
# Step 3: Compare hybrid vs hybrid + local rerank
# --------------------------------------------------------------


def show(label: str, results: list[tuple[str, float]]) -> None:
    print(f"\n{label}")

    for i, (doc_id, score) in enumerate(results[:5], 1):
        if doc_id not in corpus_by_id.index:
            continue

        text = str(corpus_by_id.loc[doc_id, "text"])
        print(f"  {i}. [{score:.4f}] {doc_id}  {text[:100]}")


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"Query: {query}")

    hybrid_results = hybrid_candidates(
        query=query,
        bm25=bm25,
        dense=dense,
        candidate_k=50,
    )

    reranked_results = search_reranked(
        query=query,
        k=5,
        candidate_k=50,
    )

    show("Hybrid (RRF) only", hybrid_results[:5])
    show("Hybrid + local BGE reranker", reranked_results)