# RAG Optimization Study

This repository contains an experimental study on optimizing Retrieval-Augmented Generation (RAG) pipelines using retrieval-side improvements. The project compares a baseline Naive RAG system with an optimized RAG pipeline that combines query rewriting, hybrid retrieval, candidate fusion, and re-ranking.

The goal of this project is to evaluate whether improving the retrieval stage can increase the quality, relevance, and faithfulness of generated answers without fine-tuning the underlying Large Language Model (LLM).

---

## Project Overview

Large Language Models can generate fluent responses, but they may produce unsupported or inaccurate answers when they rely only on their internal parametric knowledge. Retrieval-Augmented Generation reduces this problem by retrieving external context before generation.

However, a simple or naive RAG pipeline often depends on a single dense retriever. This can lead to incomplete retrieval, lexical mismatch, irrelevant chunks, or weakly grounded answers.

This project investigates a hybrid optimization approach for RAG based on:

- Query rewriting
- Sparse retrieval using BM25
- Dense vector retrieval
- Candidate fusion
- Re-ranking
- Context-grounded answer generation

---

## Research Objective

The main objective is to compare two RAG pipelines:

### 1. Naive RAG Baseline

The baseline pipeline uses the original user query and performs dense vector retrieval.

```text
User Query
→ Dense Retriever
→ Top-k Retrieved Chunks
→ LLM
→ Generated Answer