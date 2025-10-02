# align using aggregated SBERT embeddings and cosine similarity
import numpy as np
from typing import List, Tuple
from .sbert import encode, _model
from .parsing import Scene, Chapter

def scene_embed(scene: Scene) -> np.ndarray:
    """
    Compute embedding for a scene by averaging embeddings of its dialogues.
    Returns a 1D numpy vector of dimension D (SBERT model dim).
    """
    texts = [d.text for d in scene.dialogues]
    if not texts: 
        dim = _model().get_sentence_embedding_dimension()
        return np.zeros((dim,), dtype="float32")
    E = encode(texts)
    return E.mean(axis=0)

def chapter_embed(ch: Chapter) -> np.ndarray:
    """
    Compute embedding for a chapter by averaging embeddings of its dialogues.
    """
    texts = [d.text for d in ch.dialogues]
    if not texts: 
        dim = _model().get_sentence_embedding_dimension()
        return np.zeros((dim,), dtype="float32")
    E = encode(texts)
    return E.mean(axis=0)

def align_aggregated(scenes: List[Scene], chapters: List[Chapter]) -> Tuple[List[Tuple[int,int,float]], np.ndarray]:
    """
    Align scenes to chapters using aggregated embeddings.
    - Compute mean embedding per scene and per chapter.
    - Cosine similarity matrix = dot product since normalized.
    - Each scene is aligned to the chapter with max similarity.
    Returns:
    - mapping: [(scene_id, chapter_id, score), ...]
    - sims: full similarity matrix (S x C)
    """
    S = np.stack([scene_embed(s) for s in scenes], axis=0)
    C = np.stack([chapter_embed(c) for c in chapters], axis=0)
    sims = S @ C.T
    mapping = []
    for si, row in enumerate(sims):
        cj = int(row.argmax())
        mapping.append((scenes[si].scene_id, chapters[cj].chap_id, float(row[cj])))
    return mapping, sims
