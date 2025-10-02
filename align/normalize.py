# align/normalize.py --- text normalization and dialogue extraction
import regex as re
from typing import List

# Regex for matching quotation marks (various styles)
QUOTE_CHARS = r"[\"“”‘’']"

def normalize_text(s: str) -> str:
    """
    Normalize raw text:
    - Trim whitespace
    - Collapse multiple spaces
    - Replace em/en dashes with "-"
    - Lowercase everything
    """
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("—", "-").replace("–", "-")
    s = s.lower()
    return s

def extract_quoted_spans(text: str) -> List[str]:
    """
    Extract dialogue-like spans inside quotes from a text.
    - Matches both straight and curly quotes.
    - Returns list of normalized dialogue strings.
    """
    spans = re.findall(fr"{QUOTE_CHARS}(.*?){QUOTE_CHARS}", text, flags=re.S)
    return [normalize_text(x) for x in spans if normalize_text(x)]
