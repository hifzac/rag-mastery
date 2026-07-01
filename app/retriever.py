from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_query_embedding(user_query: str) -> np.ndarray:
   

    embedding = model.encode(user_query)

    return np.array([embedding], dtype=np.float32)


def retrieve(
    query: str,
    index,
    metadata: list[dict],
    top_k: int = 3
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks.
    """

    query_embedding = generate_query_embedding(query)

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        results.append(metadata[idx])

    return results