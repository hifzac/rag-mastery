# 🤖 ServiceNow RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application built from scratch using Python, FAISS, BM25, Sentence Transformers, Cross-Encoder reranking, Groq Llama, and Streamlit.

The assistant allows users to ask natural language questions over ServiceNow documentation and returns context-aware answers with cited source pages.

---

# Features

- 📄 PDF document ingestion
- ✂️ Custom text chunking
- 🧠 Sentence Transformer embeddings
- 🔍 FAISS semantic vector search
- 🔎 BM25 lexical search
- 🔀 Hybrid Retrieval (FAISS + BM25)
- 🎯 Cross Encoder reranking
- 🧠 Query rewriting using conversation history
- 💬 Conversation memory
- ⚡ Streaming LLM responses
- 📊 Pipeline performance metrics
- 📄 Source page citations
- 🖥 Interactive Streamlit interface

---

# Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| UI | Streamlit |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| Lexical Search | BM25 |
| Retrieval | Hybrid Search |
| Reranking | Cross Encoder |
| LLM | Groq Llama |
| PDF Processing | PyMuPDF |

---

# Project Architecture

```

                         User Question
                               │
                               ▼
                     Query Rewriter (LLM)
                               │
                               ▼
               Hybrid Retrieval (FAISS + BM25)
                               │
                               ▼
                 Cross Encoder Reranker
                               │
                               ▼
                     Prompt Construction
                               │
                               ▼
                        Groq Llama LLM
                               │
                               ▼
                    Streaming Response
                               │
                               ▼
               Answer + Source Page Citations

```

---

# Folder Structure

```

rag-mastery/
│
├── app/
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── hybrid_retriever.py
│   ├── bm25_store.py
│   ├── reranker.py
│   ├── prompt_builder.py
│   ├── llm.py
│   ├── query_rewriter.py
│   ├── memory.py
│   ├── metrics.py
│   ├── streamlit_app.py
│   └── config.py
│
├── data/
│   ├── faiss_index.bin
│   ├── metadata.pkl
│   └── bm25.pkl
│
├── requirements.txt
└── README.md

```

---

# Retrieval Pipeline

1. Load ServiceNow documentation
2. Split documents into overlapping chunks
3. Generate embeddings
4. Store embeddings in FAISS
5. Build BM25 lexical index
6. Rewrite user query
7. Retrieve relevant chunks
8. Rerank retrieved chunks
9. Build optimized prompt
10. Generate answer using Groq Llama
11. Display answer with source pages

---

# Performance Metrics

The application measures execution time for each stage of the pipeline:

- Query Rewrite
- Hybrid Retrieval
- Cross Encoder
- Prompt Builder
- Groq Response
- Total Pipeline Time

These metrics are available in Debug Mode.

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/rag-mastery.git
cd rag-mastery
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

macOS/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file.

```text
GROQ_API_KEY=your_api_key_here
```

---

# Running the Application

```bash
streamlit run app/streamlit_app.py
```

---

# Screenshots

## Main Chat Interface

> Add screenshot here

```

images/chat.png

```

---

## Debug Mode

Shows

- Rewritten Query
- Retrieved Chunks
- Reranked Chunks
- Final Prompt
- Performance Metrics

> Add screenshot here

```

images/debug.png

```

---

## Pipeline Performance

Example

| Stage | Time |
|---------|-------|
| Query Rewrite | 0.32 s |
| Hybrid Retrieval | 0.11 s |
| Cross Encoder | 0.64 s |
| Prompt Builder | 0.01 s |
| Groq Response | 1.48 s |
| Total | 2.56 s |

---

# Example Question

```

What is Service Operations Workspace?

```

Response

```

Service Operations Workspace is a centralized workspace that enables IT agents to manage incidents, problems, changes, and tasks from a unified interface...

Sources:
Page 18
Page 21

```

---

# Future Improvements

- Multiple PDF support
- FastAPI backend
- Docker deployment
- Redis conversation memory
- Hybrid score weighting
- Evaluation framework
- Authentication
- Cloud deployment

---

# Key Learnings

Through this project I gained hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Databases
- Hybrid Retrieval
- Cross Encoder Reranking
- Prompt Engineering
- LLM Integration
- Streamlit Application Development
- Performance Profiling

---

# Author

**Hifza Chhipa**

ServiceNow Developer | AI Engineer

GitHub: https://github.com/hifzac
