"""
Streamlit UI for RAG optimization study.

Shows a storytelling comparison:
Naive Dense RAG vs Optimized Hybrid RAG.
"""

import math
import time
from collections import defaultdict

import pandas as pd
import streamlit as st

from study_pipelines import NaiveRAGPipeline, OptimizedRAGPipeline
from utils.retrievers import DATA_DIR, load_corpus


K = 10


# --------------------------------------------------------------
# Metrics
# --------------------------------------------------------------


def precision_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    top_k = predicted_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    top_k = predicted_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def mrr_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    for rank, doc_id in enumerate(predicted_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    dcg = sum(
        relevant.get(doc_id, 0) / math.log2(rank + 2)
        for rank, doc_id in enumerate(predicted_ids[:k])
    )
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def first_relevant_rank(predicted_ids: list[str], relevant: dict[str, int], k: int = 10):
    for rank, doc_id in enumerate(predicted_ids[:k], start=1):
        if doc_id in relevant:
            return rank
    return None


def hits_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> int:
    return sum(1 for doc_id in predicted_ids[:k] if doc_id in relevant)


def calculate_metrics(predicted_ids: list[str], relevant: dict[str, int], latency: float) -> dict:
    return {
        "Precision@10": precision_at_k(predicted_ids, relevant, K),
        "Recall@10": recall_at_k(predicted_ids, relevant, K),
        "MRR@10": mrr_at_k(predicted_ids, relevant, K),
        "NDCG@10": ndcg_at_k(predicted_ids, relevant, K),
        "Hits@10": hits_at_k(predicted_ids, relevant, K),
        "First relevant rank": first_relevant_rank(predicted_ids, relevant, K),
        "Latency seconds": latency,
        "Queries per second": 1 / latency if latency > 0 else 0.0,
    }


# --------------------------------------------------------------
# Data loading
# --------------------------------------------------------------


@st.cache_data
def load_eval_data():
    queries = pd.read_parquet(DATA_DIR / "queries.parquet")
    qrels_df = pd.read_parquet(DATA_DIR / "qrels.parquet")
    corpus = load_corpus()

    valid_corpus_ids = set(corpus["_id"].astype(str).tolist())

    qrels = defaultdict(dict)
    for _, row in qrels_df.iterrows():
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])

        if corpus_id in valid_corpus_ids:
            qrels[query_id][corpus_id] = int(row["score"])

    qrels = {
        query_id: relevant_docs
        for query_id, relevant_docs in qrels.items()
        if len(relevant_docs) > 0
    }

    queries_with_qrels = queries[queries["_id"].astype(str).isin(qrels.keys())].copy()
    return queries_with_qrels, qrels, corpus


@st.cache_resource
def load_pipelines():
    naive = NaiveRAGPipeline(top_k=K)

    # Use 10 for UI speed. Use 30 or 50 for final paper experiment.
    optimized = OptimizedRAGPipeline(top_k=K, candidate_k=10)

    return naive, optimized


# --------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------


def render_metric_cards(metrics: dict, title: str):
    st.subheader(title)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision@10", f"{metrics['Precision@10']:.3f}")
    col2.metric("Recall@10", f"{metrics['Recall@10']:.3f}")
    col3.metric("MRR@10", f"{metrics['MRR@10']:.3f}")
    col4.metric("NDCG@10", f"{metrics['NDCG@10']:.3f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Hits@10", metrics["Hits@10"])
    col6.metric("First relevant rank", metrics["First relevant rank"] or "Not found")
    col7.metric("Latency", f"{metrics['Latency seconds']:.3f}s")


def render_documents(result: dict, relevant: dict[str, int]):
    docs = result["retrieved_documents"]

    for rank, doc in enumerate(docs, start=1):
        doc_id = doc["doc_id"]
        is_relevant = doc_id in relevant

        label = "Relevant" if is_relevant else "Not relevant"

        with st.expander(f"Rank {rank}: {label} | {doc_id} | Score: {doc['score']:.4f}"):
            st.write(doc["text"][:1200])


def explain_result(naive_metrics: dict, optimized_metrics: dict):
    st.subheader("Interpretation")

    naive_hits = naive_metrics["Hits@10"]
    optimized_hits = optimized_metrics["Hits@10"]

    naive_rank = naive_metrics["First relevant rank"]
    optimized_rank = optimized_metrics["First relevant rank"]

    if optimized_hits > naive_hits:
        st.success(
            "Optimized RAG retrieved more relevant evidence in the top-10 results. "
            "This explains improvement in Precision@10 and Recall@10."
        )
    elif optimized_hits == naive_hits:
        st.info(
            "Both pipelines retrieved the same number of relevant documents in the top-10. "
            "Therefore Precision@10 and Recall@10 may remain the same."
        )
    else:
        st.warning(
            "Optimized RAG retrieved fewer relevant documents in this query. "
            "This can happen because reranking may move some relevant documents down."
        )

    if naive_rank and optimized_rank and optimized_rank < naive_rank:
        st.success(
            f"Optimized RAG ranks the first relevant document earlier "
            f"(rank {optimized_rank} instead of rank {naive_rank}). "
            "This improves MRR@10 and usually NDCG@10."
        )

    if optimized_metrics["Latency seconds"] > naive_metrics["Latency seconds"]:
        st.warning(
            "Optimized RAG is slower because it performs BM25 retrieval, dense retrieval, "
            "RRF fusion, and cross-encoder reranking."
        )


# --------------------------------------------------------------
# Main UI
# --------------------------------------------------------------


st.set_page_config(
    page_title="Naive RAG vs Optimized RAG",
    layout="wide",
)

st.title("Naive RAG vs Optimized Hybrid RAG")
st.caption("Retrieval-side optimization using BM25, dense retrieval, RRF, and cross-encoder reranking.")

st.markdown(
    """
### Study setup

**Naive RAG** uses dense vector retrieval only.

**Optimized RAG** combines BM25 sparse retrieval and dense retrieval using Reciprocal Rank Fusion,
then applies a local cross-encoder reranker to reorder the final candidates.
"""
)

queries, qrels, corpus = load_eval_data()
naive_pipeline, optimized_pipeline = load_pipelines()

query_options = {
    f"{row['_id']} | {str(row['text'])[:120]}": (str(row["_id"]), str(row["text"]))
    for _, row in queries.head(100).iterrows()
}

selected = st.selectbox("Select a FiQA test query", list(query_options.keys()))
query_id, query_text = query_options[selected]
relevant = qrels[query_id]

st.markdown("### Selected Query")
st.write(query_text)

st.markdown("### Ground-truth relevant documents from qrels")
st.write(list(relevant.keys()))

if st.button("Run RAG Comparison"):
    with st.spinner("Running Naive Dense RAG..."):
        naive_start = time.perf_counter()
        naive_result = naive_pipeline.run(query_text)
        naive_latency = time.perf_counter() - naive_start

    with st.spinner("Running Optimized Hybrid RAG..."):
        optimized_start = time.perf_counter()
        optimized_result = optimized_pipeline.run(query_text)
        optimized_latency = time.perf_counter() - optimized_start

    naive_ids = [doc["doc_id"] for doc in naive_result["retrieved_documents"]]
    optimized_ids = [doc["doc_id"] for doc in optimized_result["retrieved_documents"]]

    naive_metrics = calculate_metrics(naive_ids, relevant, naive_latency)
    optimized_metrics = calculate_metrics(optimized_ids, relevant, optimized_latency)

    st.markdown("---")
    st.header("1. Retrieval Quality")

    left, right = st.columns(2)

    with left:
        render_metric_cards(naive_metrics, "Naive Dense RAG")

    with right:
        render_metric_cards(optimized_metrics, "Optimized Hybrid RAG")

    st.markdown("---")
    st.header("2. Retrieved Evidence Comparison")

    left_docs, right_docs = st.columns(2)

    with left_docs:
        st.subheader("Naive Dense RAG Retrieved Documents")
        render_documents(naive_result, relevant)

    with right_docs:
        st.subheader("Optimized Hybrid RAG Retrieved Documents")
        render_documents(optimized_result, relevant)

    st.markdown("---")
    st.header("3. Storytelling Interpretation")
    explain_result(naive_metrics, optimized_metrics)