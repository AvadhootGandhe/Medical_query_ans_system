# 🏥 MedQuery — RAG Medical Query Answering System

> **Retrieval-Augmented Generation pipeline for domain-specific medical Q&A using FAISS vector search and Gemini Flash 2.5**

MedQuery is a production-ready medical query answering system built on a Retrieval-Augmented Generation (RAG) architecture. It combines dense vector retrieval via FAISS with Google's Gemini Flash 2.5 LLM to answer medical questions grounded in a curated Kaggle medical knowledge base — evaluated end-to-end with rigorous statistical retrieval benchmarks.

![Tech Stack](https://img.shields.io/badge/Backend-Flask-000000?style=flat-square&logo=flask)
![Tech Stack](https://img.shields.io/badge/Vector%20DB-FAISS-009688?style=flat-square)
![Tech Stack](https://img.shields.io/badge/LLM-Gemini%20Flash%202.5-4285F4?style=flat-square&logo=google)
![Tech Stack](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python)
![Metric](https://img.shields.io/badge/Recall@K-33.94%25-orange?style=flat-square)
![Metric](https://img.shields.io/badge/MRR-27.38%25-blue?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [RAG Pipeline](#-rag-pipeline)
- [Evaluation Metrics](#-evaluation-metrics)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Dataset](#-dataset)
- [Roadmap](#-roadmap)

---

## 🧠 Overview

Medical Q&A is one of the most demanding NLP tasks — answers must be accurate, grounded in verified knowledge, and traceable to source documents. Generic LLMs hallucinate medical facts. MedQuery solves this by grounding every response in a curated medical corpus using RAG architecture:

```
User Query → Dense Retrieval (FAISS) → Context Injection → Gemini Flash 2.5 → Grounded Answer
```

The system is fully evaluated using standard information retrieval metrics — Recall@K, Precision@K, and Mean Reciprocal Rank (MRR) — providing transparent, reproducible benchmarking of retrieval quality.

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     MEDQUERY RAG PIPELINE                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   INDEXING PHASE (offline)                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Kaggle Medical Dataset                                  │   │
│   │        │                                                 │   │
│   │        ▼                                                 │   │
│   │  Text Chunking & Preprocessing                           │   │
│   │        │                                                 │   │
│   │        ▼                                                 │   │
│   │  Embedding Generation                                    │   │
│   │        │                                                 │   │
│   │        ▼                                                 │   │
│   │  FAISS Vector Index (stored on disk)                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   QUERY PHASE (real-time)                                        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  User Medical Query                                      │   │
│   │        │                                                 │   │
│   │        ▼                                                 │   │
│   │  Query Embedding                                         │   │
│   │        │                                                 │   │
│   │        ▼                                                 │   │
│   │  FAISS Similarity Search → Top-K Documents              │   │
│   │        │                                                 │   │
│   │        ▼                                                 │   │
│   │  Context Assembly & Prompt Engineering                   │   │
│   │        │                                                 │   │
│   │        ▼                                                 │   │
│   │  Gemini Flash 2.5 → Grounded Answer Generation          │   │
│   │        │                                                 │   │
│   │        ▼                                                 │   │
│   │  Flask API → JSON Response                               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔄 RAG Pipeline

### Stage 1 — Document Ingestion & Chunking
- Loaded Kaggle medical Q&A dataset into the pipeline
- Applied text chunking strategy to split documents into retrievable segments
- Cleaned and normalised medical text for embedding quality

### Stage 2 — FAISS Vector Indexing
- Generated dense embeddings for all document chunks
- Built a FAISS flat index for exact nearest-neighbour search
- Persisted index to disk for fast reload without re-indexing

### Stage 3 — Query Retrieval
- Embedded incoming user query using the same embedding model
- Ran FAISS similarity search to retrieve Top-K most relevant document chunks
- Ranked results by cosine similarity score

### Stage 4 — Context-Grounded Generation
- Assembled retrieved chunks into a structured prompt context window
- Injected context + user query into Gemini Flash 2.5
- Generated grounded medical answers based only on retrieved evidence

### Stage 5 — API Serving & Evaluation
- Exposed full pipeline as Flask REST API
- Tested end-to-end with Postman across evaluation query set
- Computed Recall@K, Precision@K, and MRR for retrieval quality benchmarking

---

## 📊 Evaluation Metrics

The retrieval pipeline was evaluated using standard information retrieval benchmarks:

| Metric | Score | What It Measures |
|---|---|---|
| **Recall@K** | 33.94% | Fraction of relevant docs successfully retrieved in top-K |
| **Precision@K** | 11.56% | Fraction of retrieved docs that are actually relevant |
| **MRR (Mean Reciprocal Rank)** | 27.38% | Average rank position of the first relevant result |

### Understanding the Scores

Medical domain retrieval is inherently harder than general-domain Q&A:
- Medical terminology is highly specialised — embedding models trained on general text underperform on clinical language
- The Kaggle dataset contains overlapping medical concepts that reduce precision
- These scores reflect honest, untuned baseline performance — a strong foundation for fine-tuning

### Improvement Pathways
- **Fine-tuned medical embeddings** (BioBERT, MedBERT) would significantly improve Recall@K
- **Re-ranking layer** (cross-encoder) after FAISS retrieval would improve Precision@K
- **Hybrid retrieval** (sparse BM25 + dense FAISS) would improve MRR on rare medical terms

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **LLM** | Google Gemini Flash 2.5 |
| **Backend API** | Flask |
| **API Testing** | Postman |
| **NLP / Embeddings** | Google Embedding API |
| **Data Processing** | Python, Pandas, NumPy |
| **Dataset** | Kaggle Medical Q&A Dataset |
| **Evaluation** | Custom Recall@K, Precision@K, MRR implementation |


## 📡 API Reference

### `POST /query`
Submit a medical query and receive a grounded answer.

**Request:**
```json
{
  "query": "What are the symptoms of Type 2 diabetes?",
  "top_k": 5
}
```

**Response:**
```json
{
  "status": "success",
  "query": "What are the symptoms of Type 2 diabetes?",
  "answer": "Based on the retrieved medical sources, Type 2 diabetes symptoms include...",
  "retrieved_contexts": [
    {
      "rank": 1,
      "similarity_score": 0.87,
      "source_chunk": "Type 2 diabetes is characterised by..."
    }
  ],
  "retrieval_metrics": {
    "top_k": 5,
    "retrieval_time_ms": 42
  }
}
```

### `GET /health`

```json
{ "status": "healthy", "index_loaded": true, "model": "gemini-flash-2.5" }
```

---

## 📂 Dataset

- **Source:** Kaggle Medical Q&A Dataset
- **Domain:** General medical knowledge, symptoms, diagnoses, treatments
- **Format:** Question-Answer pairs with medical context
- **Processing:** Chunked into retrievable segments, embedded, and indexed via FAISS

> Built to demonstrate production-grade RAG pipeline design — from vector indexing and retrieval to LLM-grounded generation and statistical evaluation.
