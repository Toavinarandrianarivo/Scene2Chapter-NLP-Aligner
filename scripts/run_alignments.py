# run alignments between script scenes and book chapters
import argparse
import time
import os
from align.parsing import parse_script, parse_book
from align.align_agg_cosine import align_aggregated
from align.align_dialogue_edit import align_dialogues
from align.align_monotonic import monotonic_dp_refine
from align.io_formats import write_txt_pairs, write_alignment_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="Path to parsed script file")
    ap.add_argument("--book", required=True, help="Path to book text file")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    scenes = parse_script(args.script)
    chapters = parse_book(args.book)

    if not scenes or not chapters:
        print("No scenes or chapters found. Exiting.")
        return

    # 1) aggregated SBERT cosine
    t0 = time.time()
    agg_pairs, agg_matrix = align_aggregated(scenes, chapters)
    t1 = time.time()
    write_txt_pairs(
        os.path.join(args.outdir, "alignment_scene_chapter_aggregated_embedding.txt"),
        agg_pairs,
    )

    # 2) dialogue-level Hamming
    t2 = time.time()
    ham_pairs, ham_matrix = align_dialogues(scenes, chapters, metric="ham")
    t3 = time.time()
    rows = [["scene_id","chapter_id","avg_dialogue_score_percent"]] + \
       [[s, c, f"{score:.2f}"] for (s, c, score, *_) in ham_pairs]
    write_alignment_csv(os.path.join(args.outdir, "Alignment_hamming.csv"), rows[0], rows[1:])

    # 3) dialogue-level Levenshtein (with vote rate)
    t4 = time.time()
    lev_pairs, lev_matrix = align_dialogues(scenes, chapters, metric="lev")
    t5 = time.time()
    # Save both avg score and vote_rate
    rows = [["scene_id","chapter_id","avg_dialogue_score_percent","vote_rate"]] + \
           [[s, c, f"{score:.2f}", f"{vr:.3f}"] for (s,c,score,vr) in lev_pairs]
    write_alignment_csv(os.path.join(args.outdir, "Alignment_levenshtein_full.csv"), rows[0], rows[1:])

    # For backward compatibility: still write 3-col file
    rows_simple = [["scene_id","chapter_id","avg_dialogue_score_percent"]] + \
                  [[s, c, f"{score:.2f}"] for (s,c,score,vr) in lev_pairs]
    write_alignment_csv(os.path.join(args.outdir, "Alignment_levenshtein.csv"), rows_simple[0], rows_simple[1:])

    # 4) monotonic refinement (using Levenshtein matrix as base)
    mono = monotonic_dp_refine(lev_matrix)
    rows = [["scene_id", "chapter_id", "score"]] + [
        [s, c, f"{score:.2f}"] for (s, c, score) in mono
    ]
    write_alignment_csv(os.path.join(args.outdir, "monotonic_refined.csv"), rows[0], rows[1:])

    # timing log
    with open(os.path.join(args.outdir, "timing.txt"), "w") as f:
        f.write(f"agg_sbert: {t1 - t0:.2f}s\n")
        f.write(f"hamming:   {t3 - t2:.2f}s\n")
        f.write(f"levenshtein: {t5 - t4:.2f}s\n")

    print("Alignments complete. Results saved in", args.outdir)


if __name__ == "__main__":
    main()
