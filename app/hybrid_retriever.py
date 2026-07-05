from retriever import retrieve
from bm25_store import bm25_search


def hybrid_search(
    query: str,
    index,
    bm25,
    metadata: list,
    top_k: int = 3,
):
    """
    Hybrid Retrieval using FAISS + BM25.
    """

    # Semantic Search
    faiss_results = retrieve(
        query=query,
        index=index,
        metadata=metadata,
        top_k=top_k,
    )

    # Keyword Search
    bm25_results = bm25_search(
        query=query,
        bm25=bm25,
        metadata=metadata,
        top_k=top_k,
    )

    # Merge Results
    merged = {}

    for chunk in faiss_results:
        merged[chunk["chunk_id"]] = chunk

    for chunk in bm25_results:
        merged[chunk["chunk_id"]] = chunk

    return list(merged.values())