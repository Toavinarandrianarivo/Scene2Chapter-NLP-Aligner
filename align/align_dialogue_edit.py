# align dialogues between scenes and chapters using text similarity
import numpy as np
from collections import Counter
from typing import List, Tuple
from .metrics import levenshtein_ratio, hamming_ratio, topk_indices
from .parsing import Scene, Chapter

def align_dialogues(scenes: List[Scene], chapters: List[Chapter], metric="lev") -> Tuple[list, np.ndarray]:
    """
    Dialogue-level alignment between scenes and chapters.

    Steps:
    1. For each scene, compare its dialogues against chapter dialogues.
    2. Use a Jaccard-based coarse filter to pre-select candidate chapters (faster).
    3. For each dialogue, find the best-matching chapter among candidates using Levenshtein or Hamming.
    4. Each dialogue "votes" for its best-matching chapter.
    5. The majority-voted chapter becomes the aligned chapter for that scene.
    6. We also compute:
       - Average similarity score for that scene
       - Vote rate (fraction of dialogues that agreed on the winning chapter)

    Returns:
    - final: list of tuples (scene_id, chapter_id, avg_score, vote_rate)
    - scores: similarity matrix (S x C)
    """
    S, C = len(scenes), len(chapters)
    scores = np.zeros((S, C), dtype="float32")
    final = []

    for si, s in enumerate(scenes):
        votes = Counter()
        if not s.dialogues:
            # If no dialogues, mark as unaligned
            final.append((s.scene_id, -1, 0.0, 0.0))
            continue

        # Precompute token sets for each dialogue once (saves time)
        scene_sets = [set(d.text.split()) for d in s.dialogues]

        # Coarse filtering step: compute Jaccard overlap with chapters
        coarse = np.zeros(C, dtype="float32")
        for dset in scene_sets:
            for cj, ch in enumerate(chapters):
                if not ch.dialogues:
                    continue
                max_j = max(
                    (len(dset & set(x.text.split())) / max(1, len(dset | set(x.text.split()))))
                    for x in ch.dialogues
                )
                coarse[cj] = max(coarse[cj], max_j)

        # Keep only top-k candidate chapters
        cand_js = topk_indices(coarse, k=min(6, C))

        # Each dialogue finds its best-matching candidate chapter
        for d in s.dialogues:
            best_cj, best_score = -1, -1.0
            for cj in cand_js:
                ch = chapters[cj]
                if metric == "ham":
                    cur = max([hamming_ratio(d.text, x.text) for x in ch.dialogues] or [0.0])
                else:
                    cur = max([levenshtein_ratio(d.text, x.text) for x in ch.dialogues] or [0.0])
                if cur > best_score:
                    best_score, best_cj = cur, cj
            if best_cj >= 0:
                # Vote for the chosen chapter
                votes[best_cj + 1] += 1  # +1 since chap_id is 1-based

        if votes:
            # Pick majority-voted chapter
            chap_id, count = votes.most_common(1)[0]
            cj = chap_id - 1

            # Compute average similarity against chosen chapter
            per_d = [
                max([levenshtein_ratio(d.text, x.text) if metric == "lev" else hamming_ratio(d.text, x.text)
                     for x in chapters[cj].dialogues] or [0.0])
                for d in s.dialogues
            ]
            avg_score = float(np.mean(per_d)) if per_d else 0.0

            # Compute vote rate = fraction of dialogues supporting winning chapter
            vote_rate = count / len(s.dialogues)

            scores[si, cj] = avg_score
            final.append((s.scene_id, chap_id, avg_score, vote_rate))
        else:
            final.append((s.scene_id, -1, 0.0, 0.0))

    return final, scores
