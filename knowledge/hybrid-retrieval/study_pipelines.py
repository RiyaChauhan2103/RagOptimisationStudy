"""
Study pipelines for comparing Naive RAG and Optimized RAG.

Naive RAG:
    Query -> Dense retrieval -> Top-k documents

Optimized RAG:
    Query -> BM25 retrieval + Dense retrieval -> RRF fusion -> Local reranking -> Top-k documents

This file is designed for the seminar/research study, not just the original tutorial.
"""

from typing import Any

from utils.fusion import hybrid_candidates
from utils.reranker import rerank_with_local_model
from utils.retrievers import BM25Retriever, DenseRetriever, load_corpus
from utils.llm import build_rag_prompt, generate_with_ollama

class NaiveRAGPipeline:
    """
    Naive dense RAG baseline.

    This pipeline uses only dense semantic retrieval.
    """

    def __init__(self, top_k: int = 10, generate_answer: bool = False) -> None:
        self.top_k = top_k
        self.generate_answer = generate_answer
        self.dense = DenseRetriever()
        self.corpus_by_id = load_corpus().set_index("_id")

    def run(self, query: str) -> dict[str, Any]:
        retrieved = self.dense.search(query, k=self.top_k)

        documents = []
        for doc_id, score in retrieved:
            if doc_id not in self.corpus_by_id.index:
                continue

            text = str(self.corpus_by_id.loc[doc_id, "text"])

            documents.append(
                {
                    "doc_id": doc_id,
                    "score": float(score),
                    "text": text,
                }
            )

        result = {
            "pipeline": "naive_dense_rag",
            "query": query,
            "retrieval_strategy": "dense_vector_search",
            "top_k": self.top_k,
            "retrieved_documents": documents,
        }

        if self.generate_answer:
            contexts = [doc["text"] for doc in documents]
            prompt = build_rag_prompt(query, contexts)
            result["answer"] = generate_with_ollama(prompt)

        return result



class OptimizedRAGPipeline:
    """
    Optimized RAG pipeline.

    This pipeline uses:
    BM25 + Dense Retrieval + Reciprocal Rank Fusion + Local Cross-Encoder Reranking.
    """

    def __init__(
        self,
        top_k: int = 10,
        candidate_k: int = 50,
        generate_answer: bool = False,
    ) -> None:
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.generate_answer = generate_answer

        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.corpus_by_id = load_corpus().set_index("_id")

    def run(self, query: str) -> dict[str, Any]:
        hybrid_results = hybrid_candidates(
            query=query,
            bm25=self.bm25,
            dense=self.dense,
            candidate_k=self.candidate_k,
        )

        candidate_ids = [doc_id for doc_id, _ in hybrid_results]

        reranked = rerank_with_local_model(
            query=query,
            candidate_ids=candidate_ids,
            corpus_by_id=self.corpus_by_id,
            k=self.top_k,
        )

        documents = []
        for doc_id, score in reranked:
            if doc_id not in self.corpus_by_id.index:
                continue

            text = str(self.corpus_by_id.loc[doc_id, "text"])

            documents.append(
                {
                    "doc_id": doc_id,
                    "score": float(score),
                    "text": text,
                }
            )
        result = {
            "pipeline": "naive_dense_rag",
            "query": query,
            "retrieval_strategy": "dense_vector_search",
            "top_k": self.top_k,
            "retrieved_documents": documents,
        }

        if self.generate_answer:
            contexts = [doc["text"] for doc in documents]
            prompt = build_rag_prompt(query, contexts)
            result["answer"] = generate_with_ollama(prompt)

        return result


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"

    naive = NaiveRAGPipeline(top_k=5)
    optimized = OptimizedRAGPipeline(top_k=5, candidate_k=20)

    naive_result = naive.run(query)
    optimized_result = optimized.run(query)

    print("\n==============================")
    print("NAIVE RAG")
    print("==============================")
    for i, doc in enumerate(naive_result["retrieved_documents"], 1):
        print(f"{i}. [{doc['score']:.4f}] {doc['doc_id']} {doc['text'][:120]}")

    print("\n==============================")
    print("OPTIMIZED RAG")
    print("==============================")
    for i, doc in enumerate(optimized_result["retrieved_documents"], 1):
        print(f"{i}. [{doc['score']:.4f}] {doc['doc_id']} {doc['text'][:120]}")