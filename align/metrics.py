# metrics for text similarity and vector similarity
from typing import Iterable
import numpy as np
from rapidfuzz import fuzz, distance

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    - Inputs: numpy arrays a, b (1D or 2D).
    - Returns: similarity in [-1, 1].
    - If embeddings are normalized, cosine = dot product.
    """
    if a.ndim == 1: 
        a = a[None, :]
    if b.ndim == 1: 
        b = b[None, :]
    denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    return float((a @ b.T)[0,0] / denom)

def levenshtein_ratio(a: str, b: str) -> float:
    """
    Normalized Levenshtein similarity in [0,100].
    Uses RapidFuzz's implementation for speed.
    """
    return float(fuzz.ratio(a, b))

def hamming_ratio(a: str, b: str) -> float:
    """
    Normalized Hamming similarity in [0,100].
    Requires strings of equal length. If lengths differ → return 0.0.
    """
    if len(a) != len(b) or not a:
        return 0.0
    return float(distance.Hamming.normalized_similarity(a, b) * 100.0)

def jaccard_tokens(a: str, b: str) -> float:
    """
    Token-level Jaccard similarity in [0,100].
    Intersection / Union of token sets.
    """
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb: 
        return 100.0
    if not sa or not sb:  
        return 0.0
    return 100.0 * len(sa & sb) / len(sa | sb)

def topk_indices(scores: np.ndarray, k: int) -> Iterable[int]:
    """
    Efficiently return indices of top-k highest values from an array.
    Uses numpy.argpartition for O(n) selection instead of full sort.
    """
    k = min(k, scores.size)
    return np.argpartition(-scores, k-1)[:k]
