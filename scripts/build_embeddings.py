# build SBERT embeddings cache for script and book dialogues
import argparse
from align.parsing import parse_script, parse_book
from align.sbert import encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="Path to parsed script file")
    ap.add_argument("--book", required=True, help="Path to book text file")
    args = ap.parse_args()

    scenes = parse_script(args.script)
    chapters = parse_book(args.book)

    # Cache embeddings (warm-up SBERT model)
    _ = encode([d.text for s in scenes for d in s.dialogues])
    _ = encode([d.text for c in chapters for d in c.dialogues])

    print("Embeddings cache warmed.")


if __name__ == "__main__":
    main()
