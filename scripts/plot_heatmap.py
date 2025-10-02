# scripts/plot_heatmap.py --- plot heatmap of alignment scores or vote rates
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # --- Parse command-line arguments ---
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--levenshtein", required=True,
        help="Path to Alignment_levenshtein_full.csv (preferred) or Alignment_levenshtein.csv"
    )
    ap.add_argument(
        "--out", required=True,
        help="Output image file (e.g., heatmap.png)"
    )
    ap.add_argument(
        "--metric", choices=["similarity", "vote_rate"], default="similarity",
        help="Which metric to plot: similarity (default) or vote_rate"
    )
    args = ap.parse_args()

    # --- Load alignment CSV ---
    df = pd.read_csv(args.levenshtein)

    # Ensure IDs are integers (important for pivoting and heatmap indexing)
    df["scene_id"] = df["scene_id"].astype(int)
    df["chapter_id"] = df["chapter_id"].astype(int)

    # --- Choose metric to plot ---
    if args.metric == "similarity":
        # Use average dialogue similarity percentage
        df["avg_dialogue_score_percent"] = pd.to_numeric(
            df["avg_dialogue_score_percent"], errors="coerce"
        ).fillna(0.0)

        # Create pivot matrix: rows=scenes, cols=chapters, values=similarity %
        matrix = df.pivot(
            index="scene_id", columns="chapter_id", values="avg_dialogue_score_percent"
        ).fillna(0)

        label = "Similarity %"
        title = "Scene ↔ Chapter Alignment (Levenshtein Similarity)"

    else:  # vote_rate mode
        if "vote_rate" not in df.columns:
            raise ValueError(
                "CSV does not contain vote_rate column. "
                "Use Alignment_levenshtein_full.csv from updated align_dialogues."
            )

        # Vote rate is fraction of dialogues supporting the aligned chapter
        df["vote_rate"] = pd.to_numeric(df["vote_rate"], errors="coerce").fillna(0.0)

        # Create pivot matrix: rows=scenes, cols=chapters, values=vote_rate
        matrix = df.pivot(
            index="scene_id", columns="chapter_id", values="vote_rate"
        ).fillna(0)

        label = "Vote Rate (fraction)"
        title = "Scene ↔ Chapter Alignment (Dialogue Vote Rate)"

    # --- Plot heatmap ---
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap="viridis", cbar_kws={'label': label})

    # Add labels and title
    plt.title(title)
    plt.xlabel("Chapter ID")
    plt.ylabel("Scene ID")

    # Make layout tight so labels don’t overlap
    plt.tight_layout()

    # Save to output file
    plt.savefig(args.out, dpi=300)
    print(f"Heatmap ({args.metric}) saved to {args.out}")

if __name__ == "__main__":
    main()
