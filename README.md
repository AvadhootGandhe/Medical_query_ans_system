# 🏥 MedQuery — RAG Medical Query Answering System

> **Production-grade Retrieval-Augmented Generation (RAG) pipeline for medical question answering using FAISS vector retrieval and Gemini Flash 2.5**

MedQuery is a domain-specific medical question answering system built using a Retrieval-Augmented Generation (RAG) architecture. The system combines semantic vector retrieval through FAISS with Google's Gemini Flash 2.5 LLM to generate grounded, context-aware medical answers from a large unstructured medical knowledge corpus.

The project focuses on solving one of the biggest challenges in medical NLP systems — reducing hallucinations in LLM-generated responses by grounding every answer in retrieved evidence from relevant medical documents.

---

# 📌 Table of Contents

* [Overview](#-overview)
* [System Architecture](#-system-architecture)
* [RAG Pipeline](#-rag-pipeline)
* [Evaluation Metrics](#-evaluation-metrics)
* [Tech Stack](#-tech-stack)
* [API Reference](#-api-reference)
* [Project Workflow](#-project-workflow)
* [Performance Analysis](#-performance-analysis)
* [Future Improvements](#-future-improvements)
* [Getting Started](#-getting-started)
* [Folder Structure](#-folder-structure)
* [Dataset](#-dataset)
* [Conclusion](#-conclusion)

---

# 🧠 Overview

Medical question answering is one of the most sensitive NLP applications because inaccurate responses can lead to misinformation. Traditional Large Language Models often hallucinate medical facts when operating without verified context.

MedQuery addresses this problem using Retrieval-Augmented Generation (RAG), where relevant medical documents are first retrieved from a vector database and then injected into the LLM prompt before answer generation.

The complete workflow:

```text
User Query
    ↓
Semantic Embedding
    ↓
FAISS Vector Retrieval
    ↓
Top-K Relevant Medical Chunks
    ↓
Context Injection
    ↓
Gemini Flash 2.5
    ↓
Grounded Medical Answer
```

This architecture ensures that generated responses remain context-aware, traceable, and grounded in retrieved medical evidence.

---

# 🏗 System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                    MEDQUERY ARCHITECTURE                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  OFFLINE INDEXING PIPELINE                                   │
│                                                              │
│  Large Unstructured Medical Dataset                          │
│                 │                                            │
│                 ▼                                            │
│      Text Cleaning & Preprocessing                           │
│                 │                                            │
│                 ▼                                            │
│         Document Chunking                                    │
│                 │                                            │
│                 ▼                                            │
│        Embedding Generation                                  │
│                 │                                            │
│                 ▼                                            │
│      FAISS Vector Index Storage                              │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ONLINE QUERY PIPELINE                                       │
│                                                              │
│          User Query                                          │
│                 │                                            │
│                 ▼                                            │
│         Query Embedding                                      │
│                 │                                            │
│                 ▼                                            │
│      FAISS Similarity Search                                 │
│                 │                                            │
│                 ▼                                            │
│      Top-K Relevant Chunks                                   │
│                 │                                            │
│                 ▼                                            │
│   Prompt Engineering + Context Assembly                      │
│                 │                                            │
│                 ▼                                            │
│       Gemini Flash 2.5 LLM                                  │
│                 │                                            │
│                 ▼                                            │
│      Grounded Medical Response                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

# 🔄 RAG Pipeline

## Stage 1 — Data Ingestion & Preprocessing

The system begins by loading a large unstructured medical dataset containing medical questions, answers, and contextual information.

### Processing Steps

* Removed noisy and duplicate entries
* Standardised medical terminology formatting
* Normalised text for embedding consistency
* Split long documents into smaller retrievable chunks

### Why Chunking Matters

Large documents cannot be efficiently retrieved as a single vector. Chunking improves:

* Retrieval granularity
* Semantic similarity matching
* Context precision during generation

---

## Stage 2 — Embedding Generation

Each medical text chunk is converted into dense vector embeddings using Google's embedding API.

These embeddings capture:

* Semantic meaning
* Medical terminology relationships
* Context similarity between documents

The same embedding model is later used for incoming user queries to maintain vector consistency.

---

## Stage 3 — FAISS Vector Indexing

The generated embeddings are stored inside a FAISS vector database.

### Why FAISS?

FAISS enables:

* High-speed similarity search
* Scalable nearest-neighbour retrieval
* Efficient vector indexing for large datasets

### Retrieval Workflow

```text
User Query Vector
        ↓
Cosine Similarity Search
        ↓
Top-K Most Relevant Chunks
```

The retrieved chunks become the grounding context for answer generation.

---

## Stage 4 — Context-Grounded Generation

The retrieved chunks are injected into a carefully engineered prompt alongside the user query.

### Prompt Structure

```text
Retrieved Context:
[Top-K medical chunks]

User Question:
[User Query]

Instruction:
Answer only using the provided medical context.
```

Gemini Flash 2.5 then generates a grounded answer based strictly on the retrieved evidence.

### Benefits

* Reduces hallucination
* Improves factual consistency
* Increases answer traceability
* Keeps responses medically relevant

---

## Stage 5 — API Serving

The complete RAG pipeline is deployed as a Flask REST API.

### Features

* Real-time medical query answering
* JSON-based API responses
* Retrieval metadata support
* Retrieval timing analysis

---

# 📊 Evaluation Metrics

The retrieval system was evaluated using standard Information Retrieval (IR) metrics.

| Metric                     | Score  |
| -------------------------- | ------ |
| Recall@K                   | 33.94% |
| Precision@K                | 11.56% |
| Mean Reciprocal Rank (MRR) | 27.38% |

---

## 📈 Metric Explanation

### Recall@K

Measures how many relevant documents were successfully retrieved within the top-K results.

Higher Recall means:

* Better document coverage
* Lower chance of missing critical context

---

### Precision@K

Measures how many retrieved documents are actually relevant.

Higher Precision means:

* Cleaner retrieval
* Less irrelevant context passed to the LLM

---

### Mean Reciprocal Rank (MRR)

Measures how early the first relevant document appears in retrieval results.

Higher MRR means:

* Faster relevant retrieval
* Better ranking quality

---

# 🔍 Performance Analysis

Medical retrieval is significantly harder than general-domain retrieval because:

* Medical terminology is highly specialised
* Similar diseases often share overlapping symptoms
* Semantic ambiguity exists across clinical terms
* General-purpose embeddings are not fully optimised for medical NLP

The current scores represent an honest baseline implementation without heavy domain fine-tuning.

---

# 🚀 Future Improvements

## 1. Medical-Specific Embeddings

Using:

* BioBERT
* ClinicalBERT
* MedBERT

would significantly improve semantic retrieval quality.

---

## 2. Hybrid Retrieval

Combining:

* BM25 sparse retrieval
* Dense vector retrieval

can improve performance on rare medical terms.

---

## 3. Re-Ranking Layer

Adding a transformer cross-encoder after FAISS retrieval can improve:

* Precision@K
* Ranking quality
* Context relevance

---

## 4. Streaming Responses

Implementing token streaming would improve:

* Response latency
* User experience
* Real-time interaction

---

# 🛠 Tech Stack

| Layer                  | Technology                           |
| ---------------------- | ------------------------------------ |
| Backend Framework      | Flask                                |
| Vector Database        | FAISS                                |
| LLM                    | Gemini Flash 2.5                     |
| Embedding Model        | Google Embedding API                 |
| Programming Language   | Python                               |
| Data Processing        | Pandas, NumPy                        |
| API Testing            | Postman                              |
| Evaluation             | Custom IR Metrics                    |
| Retrieval Architecture | Retrieval-Augmented Generation (RAG) |

---

# 📡 API Reference

## POST `/query`

Submit a medical query.

### Request

```json
{
  "query": "What are the symptoms of Type 2 diabetes?",
  "top_k": 5
}
```

---

### Response

```json
{
  "status": "success",
  "query": "What are the symptoms of Type 2 diabetes?",
  "answer": "Based on the retrieved medical context...",
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

---

## GET `/health`

### Response

```json
{
  "status": "healthy",
  "index_loaded": true,
  "model": "gemini-flash-2.5"
}
```

---

# ⚙ Getting Started

## Clone Repository

```bash
git clone <your-repository-url>
cd MedQuery
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Flask Server

```bash
python app.py
```

---

## API Runs On

```text
http://127.0.0.1:5000
```

---



# 📂 Dataset

### Dataset Characteristics

* Large-scale unstructured medical corpus
* Medical symptoms and diagnosis data
* Disease-treatment relationships
* Clinical question-answer pairs

### Processing Pipeline

* Text cleaning
* Chunking
* Embedding generation
* Vector indexing using FAISS

---

# 🧪 Retrieval Evaluation Workflow

```text
Evaluation Query
        ↓
Embedding Generation
        ↓
FAISS Retrieval
        ↓
Top-K Retrieved Chunks
        ↓
Ground Truth Comparison
        ↓
Recall@K / Precision@K / MRR
```

---

# 🎯 Key Learnings

This project demonstrates:

* End-to-end RAG system development
* Vector database implementation
* Dense retrieval systems
* Prompt engineering
* LLM grounding strategies
* Retrieval evaluation techniques
* Medical-domain NLP challenges

---

# ✅ Conclusion

MedQuery demonstrates how Retrieval-Augmented Generation can significantly improve reliability in medical question answering systems.

By combining:

* FAISS vector retrieval
* Semantic embeddings
* Prompt engineering
* Gemini Flash 2.5

the system produces grounded, context-aware medical responses while maintaining transparency through retrieval evaluation metrics.

The project serves as a strong foundation for building production-scale domain-specific RAG systems in healthcare and other high-accuracy NLP applications.
