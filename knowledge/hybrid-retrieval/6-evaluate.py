"""
Evaluation for RAG optimization study.

Compares:
1. Naive Dense RAG
2. Optimized Hybrid RAG

Retrieval metrics:
- Precision@10
- Recall@10
- MRR@10
- NDCG@10

Performance metrics:
- Average latency
- Median latency
- Min latency
- Max latency
- Throughput/query per second
- Latency overhead
"""

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from study_pipelines import NaiveRAGPipeline, OptimizedRAGPipeline
from utils.retrievers import DATA_DIR, load_corpus

K = 10
K_VALUES=[3,5,10]
SAMPLE_SIZE = 20
SEED = 42

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_LEVEL_CSV = OUTPUT_DIR / "query_level_comparison.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary_metrics_comparison.csv"


# --------------------------------------------------------------
# Retrieval metrics
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
    idcg = sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(ideal_rels)
    )

    return dcg / idcg if idcg > 0 else 0.0


def first_relevant_rank(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> int | None:
    for rank, doc_id in enumerate(predicted_ids[:k], start=1):
        if doc_id in relevant:
            return rank

    return None


def relevant_hits_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> int:
    return sum(1 for doc_id in predicted_ids[:k] if doc_id in relevant)


def evaluate_retrieval(
    predicted_ids: list[str],
    relevant: dict[str, int],
    k: int = 10,
) -> dict[str, Any]:
    return {
        f"precision@{k}": precision_at_k(predicted_ids, relevant, k),
        f"recall@{k}": recall_at_k(predicted_ids, relevant, k),
        f"mrr@{k}": mrr_at_k(predicted_ids, relevant, k),
        f"ndcg@{k}": ndcg_at_k(predicted_ids, relevant, k),
        f"hits@{k}": relevant_hits_at_k(predicted_ids, relevant, k),
        "first_relevant_rank": first_relevant_rank(predicted_ids, relevant, k),
    }


# --------------------------------------------------------------
# Load evaluation data
# --------------------------------------------------------------


print("Loading FiQA evaluation data...")

queries = pd.read_parquet(DATA_DIR / "queries.parquet")
qrels_df = pd.read_parquet(DATA_DIR / "qrels.parquet")

corpus = load_corpus()

valid_corpus_ids = set(corpus["_id"].astype(str).tolist())

qrels: dict[str, dict[str, int]] = defaultdict(dict)

for _, row in qrels_df.iterrows():
    query_id = str(row["query-id"])
    corpus_id = str(row["corpus-id"])

    # Important if you are using only a subset of the FiQA corpus
    if corpus_id in valid_corpus_ids:
        qrels[query_id][corpus_id] = int(row["score"])

qrels = {
    query_id: relevant_docs
    for query_id, relevant_docs in qrels.items()
    if len(relevant_docs) > 0
}

queries_with_qrels = queries[queries["_id"].astype(str).isin(qrels.keys())].copy()

if len(queries_with_qrels) == 0:
    raise ValueError(
        "No valid evaluation queries found. "
        "Increase corpus size in 3-embed.py and utils/retrievers.py, then rebuild embeddings."
    )

sample_size = min(SAMPLE_SIZE, len(queries_with_qrels))
sample = queries_with_qrels.sample(n=sample_size, random_state=SEED)

print(f"Evaluation queries: {len(sample)}")
print(f"Evaluation corpus size: {len(corpus)}")


# --------------------------------------------------------------
# Load pipelines
# --------------------------------------------------------------


print("Loading Naive Dense RAG pipeline...")
naive_pipeline = NaiveRAGPipeline(
    top_k=K,
    generate_answer=False,
)

print("Loading Optimized Hybrid RAG pipeline...")
optimized_pipeline = OptimizedRAGPipeline(
    top_k=K,
    candidate_k=50,
    generate_answer=False,
)


# --------------------------------------------------------------
# Run evaluation
# --------------------------------------------------------------


rows: list[dict[str, Any]] = []

for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Evaluating"):
    query_id = str(row["_id"])
    query_text = str(row["text"])
    relevant = qrels[query_id]

    # --------------------------
    # Naive Dense RAG
    # --------------------------
    naive_start = time.perf_counter()
    naive_result = naive_pipeline.run(query_text)
    naive_latency = time.perf_counter() - naive_start

    naive_doc_ids = [
        doc["doc_id"]
        for doc in naive_result["retrieved_documents"]
    ]

    naive_metrics = evaluate_retrieval(
        predicted_ids=naive_doc_ids,
        relevant=relevant,
        k=K,
    )

    rows.append(
        {
            "query_id": query_id,
            "query": query_text,
            "pipeline": "Naive Dense RAG",
            "latency_seconds": naive_latency,
            **naive_metrics,
        }
    )

    # --------------------------
    # Optimized Hybrid RAG
    # --------------------------
    optimized_start = time.perf_counter()
    optimized_result = optimized_pipeline.run(query_text)
    optimized_latency = time.perf_counter() - optimized_start

    optimized_doc_ids = [
        doc["doc_id"]
        for doc in optimized_result["retrieved_documents"]
    ]

    optimized_metrics = evaluate_retrieval(
        predicted_ids=optimized_doc_ids,
        relevant=relevant,
        k=K,
    )

    rows.append(
        {
            "query_id": query_id,
            "query": query_text,
            "pipeline": "Optimized Hybrid RAG",
            "latency_seconds": optimized_latency,
            **optimized_metrics,
        }
    )


# --------------------------------------------------------------
# Build result tables
# --------------------------------------------------------------


df = pd.DataFrame(rows)
df.to_csv(QUERY_LEVEL_CSV, index=False)

retrieval_summary = (
    df.groupby("pipeline")
    .agg(
        precision_at_10=(f"precision@{K}", "mean"),
        recall_at_10=(f"recall@{K}", "mean"),
        mrr_at_10=(f"mrr@{K}", "mean"),
        ndcg_at_10=(f"ndcg@{K}", "mean"),
        avg_hits_at_10=(f"hits@{K}", "mean"),
    )
    .reset_index()
)

performance_summary = (
    df.groupby("pipeline")
    .agg(
        avg_latency_seconds=("latency_seconds", "mean"),
        median_latency_seconds=("latency_seconds", "median"),
        min_latency_seconds=("latency_seconds", "min"),
        max_latency_seconds=("latency_seconds", "max"),
    )
    .reset_index()
)

performance_summary["queries_per_second"] = (
    1 / performance_summary["avg_latency_seconds"]
)

summary = retrieval_summary.merge(performance_summary, on="pipeline")
summary.to_csv(SUMMARY_CSV, index=False)


# --------------------------------------------------------------
# Print retrieval metrics
# --------------------------------------------------------------


print("\n================ Retrieval Metrics Comparison ================")
print(f"Queries evaluated: {len(sample)}")
print(f"Top-k: {K}")
print("==============================================================\n")

print(
    retrieval_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# --------------------------------------------------------------
# Print performance metrics
# --------------------------------------------------------------


print("\n================ Performance Metrics Comparison ================")
print("===============================================================\n")

print(
    performance_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# --------------------------------------------------------------
# Relative change
# --------------------------------------------------------------


naive = summary[summary["pipeline"] == "Naive Dense RAG"].iloc[0]
optimized = summary[summary["pipeline"] == "Optimized Hybrid RAG"].iloc[0]


def percentage_change(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


relative_change = {
    "precision@10_change_percent": percentage_change(
        optimized["precision_at_10"],
        naive["precision_at_10"],
    ),
    "recall@10_change_percent": percentage_change(
        optimized["recall_at_10"],
        naive["recall_at_10"],
    ),
    "mrr@10_change_percent": percentage_change(
        optimized["mrr_at_10"],
        naive["mrr_at_10"],
    ),
    "ndcg@10_change_percent": percentage_change(
        optimized["ndcg_at_10"],
        naive["ndcg_at_10"],
    ),
    "avg_latency_change_percent": percentage_change(
        optimized["avg_latency_seconds"],
        naive["avg_latency_seconds"],
    ),
    "throughput_change_percent": percentage_change(
        optimized["queries_per_second"],
        naive["queries_per_second"],
    ),
}

print("\n================ Relative Change ================")
for metric, value in relative_change.items():
    print(f"{metric:<35} {value:.2f}%")


# --------------------------------------------------------------
# Query-level comparison
# --------------------------------------------------------------


print("\n================ Query-Level Ranking Difference ================")

for query_id in sample["_id"].astype(str).tolist()[:5]:
    query_rows = df[df["query_id"] == query_id]

    if len(query_rows) != 2:
        continue

    naive_row = query_rows[query_rows["pipeline"] == "Naive Dense RAG"].iloc[0]
    optimized_row = query_rows[query_rows["pipeline"] == "Optimized Hybrid RAG"].iloc[0]

    print(f"\nQuery ID: {query_id}")
    print(f"Naive hits@{K}: {naive_row[f'hits@{K}']}, first relevant rank: {naive_row['first_relevant_rank']}")
    print(f"Optimized hits@{K}: {optimized_row[f'hits@{K}']}, first relevant rank: {optimized_row['first_relevant_rank']}")


print("\nSaved query-level results to:")
print(QUERY_LEVEL_CSV)

print("\nSaved summary metrics to:")
print(SUMMARY_CSV)


# --------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------


print("\n================ Interpretation ================")

if optimized["precision_at_10"] == naive["precision_at_10"]:
    print(f"Precision@{K} is unchanged: both pipelines retrieve the same average number of relevant documents in top-{K}.")
elif optimized["precision_at_10"] > naive["precision_at_10"]:
    print(f"Precision@{K} improved: optimized RAG retrieves more relevant documents in top-{K}.")
else:
    print(f"Precision@{K} decreased: optimized RAG retrieves fewer relevant documents in top-{K}.")

if optimized["recall_at_10"] == naive["recall_at_10"]:
    print(f"Recall@{K} is unchanged: both pipelines recover the same proportion of relevant documents.")
elif optimized["recall_at_10"] > naive["recall_at_10"]:
    print(f"Recall@{K} improved: optimized RAG recovers more relevant documents.")
else:
    print(f"Recall@{K} decreased: optimized RAG recovers fewer relevant documents.")

if optimized["mrr_at_10"] > naive["mrr_at_10"]:
    print(f"MRR@{K} improved: the first relevant document appears earlier in the optimized ranking.")

if optimized["ndcg_at_10"] > naive["ndcg_at_10"]:
    print(f"NDCG@{K} improved: optimized RAG ranks relevant documents higher.")

print(
    "Performance cost: optimized RAG has higher latency because it performs BM25 retrieval, "
    "dense retrieval, RRF fusion, and cross-encoder reranking."
)