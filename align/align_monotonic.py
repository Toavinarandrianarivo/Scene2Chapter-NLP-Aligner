# align with monotonic dynamic programming refinement
import numpy as np
from typing import List, Tuple

def monotonic_dp_refine(score_matrix: np.ndarray) -> List[Tuple[int,int,float]]:
    """
    Refine alignment with monotonicity constraint (order preserved).
    Uses dynamic programming similar to sequence alignment (Needleman-Wunsch).
    - Input: score_matrix (S x C) of similarities
    - Output: list of (scene_id, chapter_id, score)
    """
    S, C = score_matrix.shape
    dp  = -1e9 * np.ones((S+1, C+1), dtype="float32")
    back = np.zeros((S+1, C+1, 2), dtype="int16")
    dp[0,0] = 0.0

    for i in range(1, S+1):
        for j in range(1, C+1):
            v_match = dp[i-1, j-1] + score_matrix[i-1, j-1]  # align scene i with chapter j
            v_skipc = dp[i, j-1]  # skip chapter
            v_skips = dp[i-1, j]  # skip scene
            best = max(v_match, v_skipc, v_skips)
            dp[i, j] = best
            if best == v_match:
                back[i, j] = (i-1, j-1)
            elif best == v_skipc:
                back[i, j] = (i, j-1)
            else:
                back[i, j] = (i-1, j)

    # Traceback from bottom-right
    i, j = S, C
    pairs = []
    while i > 0 and j > 0:
        pi, pj = back[i, j]
        if pi == i-1 and pj == j-1:
            pairs.append((i, j, float(score_matrix[i-1, j-1])))
        i, j = pi, pj
    pairs.reverse()
    return pairs
