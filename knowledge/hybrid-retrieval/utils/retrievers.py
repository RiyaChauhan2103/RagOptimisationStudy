"""
Polished versions of the BM25 and dense retrievers built in 2-bm25.py and
3-embed.py. The later files import from here so each one can focus on the new
idea it introduces: fusion, reranking, and evaluation.
"""

from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "fiqa"
BM25_DIR = ROOT / "indexes" / "bm25"
DENSE_DIR = ROOT / "indexes" / "dense"

LOCAL_EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
DENSE_EMBEDDINGS_FILE = "multi_qa_mpnet_embeddings.npy"

# IMPORTANT:
# This must match the subset used in 3-embed.py.
# If you used corpus.head(5000) there, use 5000 here too.
DEMO_CORPUS_SIZE =None


def load_corpus() -> pd.DataFrame:
    corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
    if DEMO_CORPUS_SIZE is not None:
        corpus = corpus.head(DEMO_CORPUS_SIZE)

    return corpus


# --------------------------------------------------------------
# BM25
# --------------------------------------------------------------


class BM25Retriever:
    def __init__(self) -> None:
        self._retriever = bm25s.BM25.load(str(BM25_DIR))
        self._doc_ids = (BM25_DIR / "doc_ids.txt").read_text().splitlines()

        if DEMO_CORPUS_SIZE is not None:
            self._doc_ids = self._doc_ids[:DEMO_CORPUS_SIZE]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize([query], stopwords="en")
        indices, scores = self._retriever.retrieve(tokens, k=k)

        results = []
        for j, i in enumerate(indices[0].tolist()):
            if i < len(self._doc_ids):
                results.append((self._doc_ids[i], float(scores[0][j])))

        return results


# --------------------------------------------------------------
# Dense
# --------------------------------------------------------------


class DenseRetriever:
    def __init__(self) -> None:
        corpus = load_corpus()
        self._doc_ids = corpus["_id"].tolist()

        embeddings_path = DENSE_DIR / DENSE_EMBEDDINGS_FILE

        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Dense embeddings not found: {embeddings_path}\n"
                "Run this first:\n"
                "uv run knowledge/hybrid-retrieval/3-embed.py"
            )

        raw = np.load(embeddings_path)

        if len(raw) != len(self._doc_ids):
            raise ValueError(
                f"Embedding/doc_id mismatch: {len(raw)} embeddings but "
                f"{len(self._doc_ids)} document ids.\n"
                "Make sure DEMO_CORPUS_SIZE here matches corpus.head(...) "
                "inside 3-embed.py."
            )

        self._embeddings = self._normalize(raw)
        self._model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )[0].astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Query embedding has zero norm.")

        return vec / norm

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self._embeddings @ self._embed_query(query)
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]