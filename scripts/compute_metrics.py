# compute metrics from alignment results and parsed data
import argparse
import json
import os
import pandas as pd
from align.parsing import parse_script, parse_book
from align.evaluate import compute_dialogue_stats, dump_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--script", default=None, help="Path to parsed script file")
    ap.add_argument("--book", default=None, help="Path to book text file")
    args = ap.parse_args()

    # Prefer full CSV with vote_rate if available
    lev_full_path = os.path.join(args.outdir, "Alignment_levenshtein_full.csv")
    lev_path = os.path.join(args.outdir, "Alignment_levenshtein.csv")

    if os.path.exists(lev_full_path):
        csv_path = lev_full_path
    elif os.path.exists(lev_path):
        csv_path = lev_path
    else:
        print(f"Missing required alignment file: {lev_path}")
        return

    # Parse script/book if provided (for dialogue counts in metrics)
    if args.script and args.book:
        scenes = parse_script(args.script)
        chapters = parse_book(args.book)
    else:
        print("Provide --script and --book for full metrics. Using CSV only.")
        scenes, chapters = [], []

    # Load alignment CSV
    df = pd.read_csv(csv_path)

    # Convert numeric columns safely
    for col in ["avg_dialogue_score_percent", "vote_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Build pairs: include vote_rate if available
    if "vote_rate" in df.columns:
        pairs = list(
            zip(df["scene_id"], df["chapter_id"],
                df["avg_dialogue_score_percent"], df["vote_rate"])
        )
    else:
        pairs = list(
            zip(df["scene_id"], df["chapter_id"], df["avg_dialogue_score_percent"])
        )

    # Compute metrics
    metrics = compute_dialogue_stats(scenes, chapters, pairs)
    dump_metrics(os.path.join(args.outdir, "metrics.json"), metrics)

    print("Metrics computed:\n", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
