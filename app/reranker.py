from sentence_transformers import CrossEncoder, SentenceTransformer, cross_encoder
import numpy as np
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
def rerank(query: str, retrieved_chunks: list[dict], top_k: int = 3)-> list[dict]:
    """
    Rerank retrieved chunks using a cross-encoder.
    """

    # Prepare pairs for reranking
    pairs=[]
    if not retrieved_chunks:
        return []
    for chunk in retrieved_chunks:
        pairs.append(
            (
                query,
                chunk['text']
            )
        )

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Sort chunks by score
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    reranked_chunks = []

    for idx in sorted_indices[:top_k]:

        reranked_chunks.append(
            retrieved_chunks[idx]
        )

    return reranked_chunks

    

  