"""Strategy A probe: read existing aligned/*.json (current pipeline output)
and extract (ocr_char, qn_syllable) pairs for tier-1 probe comparison.

The current pipeline (pipeline/step2_align.py) stores `aligned[i]` as one of:
  {type: 'match', char: {ocr_char, bbox, ...}, syllable, column}
  {type: 'deletion', char: {...}, syllable: None, column}
  {type: 'insertion', char: None, syllable: ..., column}

We only count 'match' pairs (those are what current pipeline gives to step 3).
"""

import json
from pathlib import Path


def collect_pairs_a(aligned_path: Path) -> list[tuple[str | None, str | None]]:
    with open(aligned_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for pair in data:
        if pair.get("type") != "match":
            continue
        char_info = pair.get("char") or {}
        ocr_char = char_info.get("ocr_char")
        qn = pair.get("syllable")
        out.append((ocr_char, qn))
    return out
