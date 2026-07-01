import faiss
import numpy as np
import pickle
from pathlib import Path
def build_index(embedded_chunks: list[dict]) -> tuple:
    embeddings=np.array([chunk['embedding'] for chunk in embedded_chunks], dtype=np.float32)
    dimension=embeddings.shape[1]
    index=faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embedded_chunks
def save_index(index,metadata,index_path:Path,metadata_path:Path):
    faiss.write_index(index,str(index_path))
    with open(metadata_path,'wb') as f:
        pickle.dump(metadata,f)

def load_index(index_path:Path,metadata_path:Path) -> tuple:
    index=faiss.read_index(str(index_path))
    with open(metadata_path,'rb') as f:
        metadata=pickle.load(f)
    return index,metadata