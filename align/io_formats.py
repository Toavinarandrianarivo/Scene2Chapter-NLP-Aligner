# write alignment results to text and CSV files
import csv
from typing import Iterable, Tuple, List

def write_txt_pairs(path: str, pairs: Iterable[Tuple[int,int,float]]):
    """
    Write simple scene→chapter mappings to a tab-separated .txt file.
    Format: scene_id <TAB> chapter_id
    """
    with open(path, "w", encoding="utf-8") as f:
        for s, c, *_ in pairs:
            f.write(f"{s}\t{c}\n")

def write_alignment_csv(path: str, header: List[str], rows: Iterable[List]):
    """
    Write alignment results to CSV.
    - header: list of column names
    - rows: iterable of row lists
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
