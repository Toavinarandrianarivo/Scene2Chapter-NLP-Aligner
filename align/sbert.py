# align/sbert.py --- SBERT encoding with caching
from functools import lru_cache
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def _model():
    """
    Load the SBERT model once and cache it with LRU.
    Default: all-MiniLM-L6-v2 (384 dimensions, fast).
    For higher accuracy, you can swap to another model (e.g. all-mpnet-base-v2) without changing the code elsewhere.
    """
    # return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2") # accurate

def encode(texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts into embeddings using SBERT.
    - Returns a (N, D) numpy array of embeddings.
    - Normalizes embeddings (so cosine similarity = dot product).
    - If texts is empty, returns a (0, D) zero array with the correct dimension.
    """
    if not texts:
        dim = _model().get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype="float32")
    
    return np.asarray(
        _model().encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        ),
        dtype="float32"
    )
