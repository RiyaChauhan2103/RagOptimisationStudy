# RAG Optimization Study

This repository contains an experimental study on improving Retrieval-Augmented Generation (RAG) through retrieval-side optimization. The project compares a baseline Naive RAG pipeline with an Optimized RAG pipeline that uses query rewriting, sparse-dense hybrid retrieval, candidate fusion, and re-ranking.

The main objective is to evaluate whether improving the retrieval stage can increase the relevance, faithfulness, and reliability of generated answers without fine-tuning the underlying Large Language Model (LLM).

---

## Project Overview

Large Language Models can generate fluent and useful responses, but they may produce unsupported or inaccurate answers when they rely only on internal parametric knowledge. Retrieval-Augmented Generation addresses this limitation by retrieving external context before answer generation.

However, a simple RAG pipeline often relies on only one retrieval strategy, usually dense vector retrieval. This can lead to lexical mismatch, incomplete evidence retrieval, irrelevant chunks, or weakly grounded answers.

This project investigates a hybrid optimization approach based on:

- Query rewriting
- Sparse retrieval using BM25
- Dense vector retrieval
- Candidate fusion
- Re-ranking
- Context-grounded answer generation

---

## Research Objective

The project compares two RAG pipelines:

### 1. Naive RAG Baseline

The baseline pipeline uses the original user query and performs dense vector retrieval.

```text
User Query
→ Dense Retriever
→ Top-k Retrieved Chunks
→ LLM
→ Generated Answer
```

### 2. Optimized RAG Pipeline

The optimized pipeline improves retrieval quality before answer generation.

```text
User Query
→ Query Rewriting
→ BM25 Retrieval + Dense Vector Retrieval
→ Candidate Fusion
→ Re-ranking
→ Final Context Selection
→ LLM
→ Generated Answer
```

The study evaluates whether the optimized pipeline retrieves more relevant evidence and produces more faithful answers than the naive baseline.

---

## Key Features

- Naive RAG baseline implementation
- Optimized RAG pipeline implementation
- Query rewriting before retrieval
- Hybrid retrieval using BM25 and dense embeddings
- Candidate fusion from multiple retrievers
- Re-ranking of retrieved passages
- Structured JSON output from both pipelines
- Retrieval-level and generation-level evaluation
- Streamlit interface for interactive comparison

---

## Project Structure

```text
RagOptimisationStudy/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── experiments/
│   ├── evaluate_pipelines.py
│   ├── analyze_results.py
│   └── run_optimized_rag.py
│
├── src/
│   └── rag_optimization/
│       ├── pipelines/
│       │   ├── naive_rag.py
│       │   └── optimized_rag.py
│       │
│       ├── retrievers/
│       │   ├── dense_chroma.py
│       │   ├── bm25_retriever.py
│       │   └── hybrid_retriever.py
│       │
│       ├── rerankers/
│       │   └── cross_encoder_reranker.py
│       │
│       ├── generation/
│       │   └── llm_generator.py
│       │
│       ├── evaluation/
│       │   ├── retrieval_metrics.py
│       │   └── generation_metrics.py
│       │
│       └── utils/
│           ├── config.py
│           └── helpers.py
│
├── results/
│   ├── query_level_comparison.csv
│   ├── summary_metrics_comparison.csv
│   └── figures/
│
├── docs/
│   └── paper/
│
├── requirements.txt
├── README.md
└── .gitignore
```

> Note: The exact folder structure may vary depending on the final implementation.

---

## Methodology

### Naive RAG

The Naive RAG pipeline is used as the baseline. It takes the original user query, performs dense vector retrieval, retrieves the top-k most similar document chunks, and passes those chunks to the LLM for answer generation.

This pipeline is simple and efficient, but it may fail when exact lexical matches are required or when semantically similar chunks are not sufficiently relevant.

### Optimized RAG

The optimized pipeline introduces retrieval-side improvements.

First, the user query is rewritten to make the retrieval intent clearer. Then, two retrieval methods are applied:

- BM25 retrieval for exact lexical matching
- Dense vector retrieval for semantic similarity

The retrieved candidates are merged, duplicate chunks are removed, and a re-ranking model scores the candidate passages according to their relevance to the original query. The highest-ranked chunks are then passed to the LLM as final context.

---

## Evaluation

The project compares the Naive RAG and Optimized RAG pipelines using retrieval-level and generation-level evaluation metrics.

### Retrieval Metrics

- Precision@k
- Recall@k
- Mean Reciprocal Rank
- nDCG@k
- Context similarity

### Generation Metrics

- Answer relevancy
- Faithfulness
- Correctness
- F1 score, if reference answers are available

Example output files:

```text
results/query_level_comparison.csv
results/summary_metrics_comparison.csv
```

---

## Streamlit Demo

The project includes a Streamlit application for visually comparing the Naive RAG and Optimized RAG pipelines.

The interface allows users to:

- Select or enter a query
- Run both RAG pipelines
- Compare retrieved chunks
- View generated answers
- Inspect retrieval metadata
- Compare evaluation metrics

Run the Streamlit app with:

```bash
streamlit run app/streamlit_app.py
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RiyaChauhan2103/RagOptimisationStudy.git
cd RagOptimisationStudy
```

### 2. Create a Virtual Environment

Using Conda:

```bash
conda create -n rag_env python=3.11
conda activate rag_env
```

Using Python venv:

```bash
python -m venv rag_env
```

Activate on Windows:

```bash
rag_env\Scripts\activate
```

Activate on macOS/Linux:

```bash
source rag_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root if the implementation uses external APIs.

Example:

```env
OPENAI_API_KEY=your_api_key_here
GOOGLE_API_KEY=your_api_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CHROMA_DIR=./chroma_db
TOP_K=5
```

Do not commit `.env` to GitHub.

Add it to `.gitignore`:

```gitignore
.env
```

---

## Running the Project

### Run Evaluation

```bash
python -m experiments.evaluate_pipelines
```

### Analyze Results

```bash
python -m experiments.analyze_results
```

### Run Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

---

## Output Files

The project output is stored as CSV files in the `results/` directory. These files are used to compare the Naive RAG pipeline and the Optimized RAG pipeline at both query level and summary level.


---

## Results

The study reports a comparison between Naive RAG and Optimized RAG.

Example summary table:

| Pipeline | Precision@k | Recall@k | MRR | Faithfulness | Answer Relevancy |
|---|---:|---:|---:|---:|---:|
| Naive RAG | TBD | TBD | TBD | TBD | TBD |
| Optimized RAG | TBD | TBD | TBD | TBD | TBD |

Final results should be updated after running the evaluation scripts.

---

## Research Paper

This implementation supports a seminar research paper on:

**Hybrid Optimization of Retrieval-Augmented Generation Using Query Rewriting, Sparse-Dense Retrieval, and Re-ranking**

The paper investigates whether retrieval-side optimization improves RAG performance compared with a naive dense retrieval baseline.

---

## Limitations

The optimized pipeline may improve retrieval quality but introduces additional computational cost. Query rewriting, hybrid retrieval, and re-ranking can increase latency compared with naive dense retrieval.

Other limitations include:

- Dependency on chunking quality
- Dependency on embedding model quality
- Dependency on re-ranker performance
- Possible noise introduced during query rewriting
- Limited generalization if the evaluation corpus is small

---

## Future Work

Possible future improvements include:

- Domain-specific embedding models
- Adaptive retrieval strategy selection
- More advanced rank fusion methods
- Context compression
- Latency-aware re-ranking
- Larger-scale evaluation across multiple domains
- Human evaluation of faithfulness and answer quality

---

## Technologies Used

- Python
- Streamlit
- ChromaDB
- BM25
- Sentence Transformers
- Cross-Encoder Re-ranker
- Pandas
- NumPy
- Scikit-learn
- Large Language Model API

---

## Repository

```text
https://github.com/RiyaChauhan2103/RagOptimisationStudy
```

---

## Author

**Riya Chauhan**

This project was developed as part of a research study on Retrieval-Augmented Generation optimization, hybrid retrieval, and LLM-based grounded response generation.

---
