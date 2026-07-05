from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import numpy as np

def build_bm25_index(metadata: list[dict]):
    """
    Build a BM25 index from chunk metadata.
    """

    corpus = []

    for chunk in metadata:

        tokens = chunk["text"].lower().split()

        corpus.append(tokens)

    bm25 = BM25Okapi(corpus)

    return bm25


def save_bm25(bm25, path: Path):
    """
    Save BM25 index.
    """

    with open(path, "wb") as file:
        pickle.dump(bm25, file)


def load_bm25(path: Path):
    """
    Load BM25 index.
    """

    with open(path, "rb") as file:
        bm25 = pickle.load(file)

    return bm25


import numpy as np


def bm25_search(
    query: str,
    bm25,
    metadata: list[dict],
    top_k: int = 3
):
    

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:

        results.append(metadata[idx])

    return results