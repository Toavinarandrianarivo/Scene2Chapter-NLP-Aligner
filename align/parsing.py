# align/parsing.py --- parsing scripts and books into scenes/chapters and dialogues
from dataclasses import dataclass, field
from typing import List
import regex as re
from .normalize import normalize_text, extract_quoted_spans

# ---- Data structures ----
@dataclass
class Dialogue:
    text: str
    speaker: str | None = None  # reserved for future (BookNLP speaker tags)

@dataclass
class Scene:
    scene_id: int
    header: str
    dialogues: List[Dialogue] = field(default_factory=list)

@dataclass
class Chapter:
    chap_id: int
    title: str
    text: str
    dialogues: List[Dialogue] = field(default_factory=list)

# ---- Regex for parsing ----
# Loosened: accept ANY line that starts with "S:" as a scene header
SCENE_START_RE = re.compile(r"^s:\s*.*", re.I)

# Dialogue lines still require "D:"
DIALOGUE_RE    = re.compile(r"^d:\s*(.*)", re.I)

# Chapter lines: "Chapter 1", "Chapter One", "Ch. 1", etc.
CHAPTER_RE     = re.compile(r"^\s*(chapter\s+\w+|chapter\s+\d+|ch\.\s*\d+)\b", re.I)

def parse_script(path: str) -> List[Scene]:
    """
    Parse a screenplay file with 'S:' (scene) and 'D:' (dialogue) markers.
    - Accepts any 'S:' line as a scene header.
    - Returns list of Scene objects with dialogues.
    """
    scenes, cur = [], None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if SCENE_START_RE.match(line):
                # new scene starts
                if cur:
                    scenes.append(cur)
                cur = Scene(scene_id=len(scenes)+1, header=line.strip(), dialogues=[])
            else:
                m = DIALOGUE_RE.match(line)
                if m and cur:
                    text = normalize_text(m.group(1))
                    if text:
                        cur.dialogues.append(Dialogue(text=text))
        if cur:
            scenes.append(cur)
    return scenes

def parse_book(path: str) -> List[Chapter]:
    """
    Parse a book text file into chapters and extract dialogues (inside quotes).
    - Splits by lines starting with 'Chapter' / 'Ch. N'.
    - Extracts dialogue spans in quotes for each chapter.
    """
    chapters = []
    with open(path, "r", encoding="utf-8") as f:
        buf, title = [], "Front Matter"
        for line in f:
            if CHAPTER_RE.match(line):
                # close previous chapter
                if buf:
                    text = "\n".join(buf)
                    chapters.append(Chapter(chap_id=len(chapters)+1, title=title, text=text))
                    buf = []
                title = line.strip()
            else:
                buf.append(line.rstrip("\n"))
        if buf:
            chapters.append(Chapter(chap_id=len(chapters)+1, title=title, text="\n".join(buf)))

    # Extract dialogues in quotes
    for ch in chapters:
        q = extract_quoted_spans(ch.text)
        ch.dialogues = [Dialogue(text=t) for t in q]
    return chapters
