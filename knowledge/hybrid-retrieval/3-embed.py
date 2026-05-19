"""
Dense retrieval with local sentence-transformer embeddings.

No OpenAI key.
No Gemini key.
No API quota.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Local embedding model
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
model = SentenceTransformer(model_name)

DATA_DIR = Path(__file__).parent / "data" / "fiqa"
INDEX_DIR = Path(__file__).parent / "indexes" / "dense"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Separate cache for local embeddings
embeddings_path = INDEX_DIR /  "multi_qa_mpnet_embeddings.npy"


# --------------------------------------------------------------
# Step 1: Embed corpus locally
# --------------------------------------------------------------


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed texts locally and return numpy array.
    """
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    return embeddings.astype(np.float32)


def build_index(doc_texts: list[str], batch_size: int = 512) -> np.ndarray:
    """
    Embed full corpus in batches.
    """
    chunks = []

    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Embedding"):
        batch = doc_texts[i : i + batch_size]
        chunks.append(embed_batch(batch))

    return np.vstack(chunks)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize vectors for cosine similarity.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


# --------------------------------------------------------------
# Step 2: Load data and build/load embeddings
# --------------------------------------------------------------

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")

# For demo, start smaller.
# Remove this line later if you want the full 57k corpus.
corpus = corpus

doc_ids = corpus["_id"].tolist()
doc_texts = [str(t).strip() or "[empty document]" for t in corpus["text"].tolist()]

if embeddings_path.exists():
    print(f"Loading cached local embeddings from {embeddings_path}")
    doc_embeddings = np.load(embeddings_path)
else:
    print(f"Embedding {len(doc_texts)} docs locally with {model_name}")
    doc_embeddings = build_index(doc_texts)
    np.save(embeddings_path, doc_embeddings)
    print(f"Saved local embeddings to {embeddings_path}")

doc_embeddings_normed = normalize_embeddings(doc_embeddings)


# --------------------------------------------------------------
# Step 3: Dense search
# --------------------------------------------------------------


def search_dense(query: str, k: int = 10) -> list[tuple[str, float]]:
    """
    Return top-k dense retrieval results.
    """
    query_vec = embed_batch([query])[0]

    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        raise ValueError("Query embedding has zero norm.")

    query_vec = query_vec / query_norm

    scores = doc_embeddings_normed @ query_vec
    top_k = np.argsort(-scores)[:k]

    return [(doc_ids[i], float(scores[i])) for i in top_k]


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"\nQuery: {query}\n")

    for i, (doc_id, score) in enumerate(search_dense(query, k=5), 1):
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        print(f"{i}. [{score:.3f}] {doc_id} {text}\n")