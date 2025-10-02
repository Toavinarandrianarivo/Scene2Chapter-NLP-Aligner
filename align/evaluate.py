# compute metrics from alignment results and parsed data
import json
import numpy as np
from typing import List, Tuple
from .parsing import Scene, Chapter

def compute_dialogue_stats(scenes: List[Scene], chapters: List[Chapter], lev_pairs: List[Tuple[int,int,float]]):
    """
    Compute summary statistics for scene-chapter alignment results.

    Inputs:
    - scenes, chapters: parsed objects with dialogues
    - lev_pairs: list of tuples (scene_id, chapter_id, avg_score, [vote_rate])

    Outputs (dict):
    - scene_dialogue_count: total dialogues in scenes
    - book_dialogue_count: total dialogues in chapters
    - scenes_with_dialogue: number of scenes that have at least one dialogue
    - avg_similarity_percent: mean similarity across all aligned scenes
    - avg_vote_rate: mean vote rate across scenes (0-100%)
    - num_scenes_over_80pct: how many scenes had >= 80% average similarity
    - num_scenes_exact: how many scenes had ~100% match
    """
    n_scene_dialogues = sum(len(s.dialogues) for s in scenes)
    n_book_dialogues  = sum(len(c.dialogues) for c in chapters)
    scene_count = sum(1 for s in scenes if s.dialogues)

    # Collect similarity scores
    scores = [p[2] for p in lev_pairs if p[1] != -1]
    # Collect vote rates if provided (align_dialogues now returns them)
    vote_rates = [p[3] for p in lev_pairs if len(p) > 3 and p[1] != -1]

    avg_sim = float(np.mean(scores)) if scores else 0.0
    avg_vote_rate = float(np.mean(vote_rates)) if vote_rates else 0.0
    over_80 = int(sum(1 for s in scores if s >= 80.0))
    exact   = int(sum(1 for s in scores if s >= 99.9))

    metrics = {
        "scene_dialogue_count": n_scene_dialogues,
        "book_dialogue_count": n_book_dialogues,
        "scenes_with_dialogue": scene_count,
        "avg_similarity_percent": round(avg_sim, 2),
        "avg_vote_rate": round(avg_vote_rate * 100, 2),  # expressed as percent
        "num_scenes_over_80pct": over_80,
        "num_scenes_exact": exact
    }
    return metrics

def dump_metrics(path: str, metrics: dict):
    """Write metrics to JSON file with pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
