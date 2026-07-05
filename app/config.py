# app/config.py

from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

PDF_PATH = DATA_DIR / "servicenow-australia-operational-technology-enus.pdf"

INDEX_PATH = DATA_DIR / "faiss_index.bin"

METADATA_PATH = DATA_DIR / "metadata.pkl"
BM25_PATH = DATA_DIR / "bm25.pkl"
# -----------------------------
# Chunking
# -----------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# -----------------------------
# Retrieval
# -----------------------------

RETRIEVAL_TOP_K = 10

RERANK_TOP_K = 3

# -----------------------------
# Models
# -----------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"