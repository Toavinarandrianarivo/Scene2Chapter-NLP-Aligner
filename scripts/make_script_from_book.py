# make a pseudo-script from a book text by splitting into scenes and extracting dialogue
import argparse
import regex as re
from pathlib import Path

QUOTE_CHARS = r"[\"“”‘’']"

def extract_quoted_spans(text: str):
    """Extract dialogue spans between quotes."""
    spans = re.findall(fr"{QUOTE_CHARS}(.*?){QUOTE_CHARS}", text, flags=re.S)
    spans = [s.strip().replace("\n", " ") for s in spans if s.strip()]
    return spans

def make_script(book_path: str, out_path: str, max_sentences_per_scene: int = 5):
    """
    Convert book text into pseudo-script format:
    - Split into scenes every N sentences
    - Extract quoted dialogue into D: lines
    - Mark scenes with S: header
    """
    text = Path(book_path).read_text(encoding="utf-8")
    # Split text into sentences (simple split on '.')
    sentences = re.split(r'(?<=[.!?])\s+', text)

    scenes, cur = [], []
    for i, sent in enumerate(sentences):
        cur.append(sent)
        if len(cur) >= max_sentences_per_scene:
            scenes.append(" ".join(cur))
            cur = []
    if cur:
        scenes.append(" ".join(cur))

    with open(out_path, "w", encoding="utf-8") as f:
        for i, sc in enumerate(scenes, 1):
            f.write(f"S: AUTO_SCENE_{i}\n")
            for d in extract_quoted_spans(sc):
                f.write(f"D: {d}\n")

    print(f"Script-like file written to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True, help="Path to Zootopia.txt (novelization)")
    ap.add_argument("--out", required=True, help="Path to output pseudo-script file")
    ap.add_argument("--sentences-per-scene", type=int, default=5, help="How many sentences per pseudo-scene")
    args = ap.parse_args()
    make_script(args.book, args.out, args.sentences_per_scene)
