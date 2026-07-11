from fastapi import FastAPI
from vector_store import load_index
from bm25_store import load_bm25
from schemas import ChatRequest
from query_rewriter import rewrite_query
from hybrid_retriever import hybrid_search
from reranker import rerank
from prompt_builder import build_prompt
from llm import generate_response

from config import (
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
)
from config import (
    INDEX_PATH,
    METADATA_PATH,
    BM25_PATH,
)


app = FastAPI(
    title="ServiceNow RAG API",
    version="1.0"
)


# Load once when server starts
index, metadata = load_index(
    INDEX_PATH,
    METADATA_PATH
)

bm25 = load_bm25(BM25_PATH)


@app.get("/")
def root():
    return {
        "message": "ServiceNow RAG API is running."
    }
@app.post("/chat")
def chat(request: ChatRequest):

    rewritten_query = rewrite_query(
        query=request.query,
        memory=[]
    )

    retrieved_chunks = hybrid_search(
        query=rewritten_query,
        index=index,
        bm25=bm25,
        metadata=metadata,
        top_k=RETRIEVAL_TOP_K
    )

    reranked_chunks = rerank(
        query=request.query,
        retrieved_chunks=retrieved_chunks,
        top_k=RERANK_TOP_K
    )

    prompt = build_prompt(
        query=request.query,
        retrieved_chunks=reranked_chunks
    )

    answer = ""

    for token in generate_response(
        prompt=prompt,
        memory=[]
    ):
        answer += token

    pages = sorted(
        {
            chunk["page_number"]
            for chunk in reranked_chunks
        }
    )

    return {
        "answer": answer,
        "sources": pages
    }