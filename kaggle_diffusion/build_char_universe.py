"""Build the complete list of chu-Nom characters to generate fd_cache for.

Strategy: take every glyph in NomNaTong-Regular.ttf that falls inside a CJK
block (Unified, Compat, Ext A-F). Skip non-CJK (numbers, ASCII, symbols) and
PUA. This is the **maximum useful universe** — covers any chu-Nom that could
appear in a Vietnamese manuscript.

Output:
    kaggle_diffusion/exports/char_universe.txt   one Unicode char per line
    kaggle_diffusion/exports/char_universe.json  metadata (block distribution)

Run on Mac:
    PATH="$PWD/.venv/bin:$PATH" python kaggle_diffusion/build_char_universe.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fontTools.ttLib import TTFont

REPO = Path(__file__).resolve().parent.parent
FONT = REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"
OUT = REPO / "kaggle_diffusion" / "exports"


def cjk_block(cp: int) -> str | None:
    """Return CJK block name for `cp`, or None if outside CJK."""
    if 0x4E00 <= cp <= 0x9FFF:   return "CJK Unified"
    if 0x3400 <= cp <= 0x4DBF:   return "Ext A"
    if 0x20000 <= cp <= 0x2A6DF: return "Ext B"
    if 0x2A700 <= cp <= 0x2B73F: return "Ext C"
    if 0x2B740 <= cp <= 0x2B81F: return "Ext D"
    if 0x2B820 <= cp <= 0x2CEAF: return "Ext E"
    if 0x2CEB0 <= cp <= 0x2EBEF: return "Ext F"
    if 0xF900 <= cp <= 0xFAFF:   return "Compat"
    return None


def main() -> None:
    if not FONT.exists():
        raise FileNotFoundError(f"NomNaTong font missing: {FONT}")

    font = TTFont(str(FONT))
    cmap = font.getBestCmap()
    codepoints = sorted(cmap.keys())

    chars: list[str] = []
    blocks: Counter = Counter()
    for cp in codepoints:
        block = cjk_block(cp)
        if block is None:
            continue
        chars.append(chr(cp))
        blocks[block] += 1

    OUT.mkdir(parents=True, exist_ok=True)

    # char list — one per line for easy diffing
    with open(OUT / "char_universe.txt", "w", encoding="utf-8") as f:
        for c in chars:
            f.write(c + "\n")

    # metadata
    meta = {
        "total_chars": len(chars),
        "font_total_glyphs": len(codepoints),
        "blocks": dict(blocks.most_common()),
        "font_path": str(FONT.relative_to(REPO)),
    }
    with open(OUT / "char_universe.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Universe: {len(chars):,} chars (CJK only) / {len(codepoints):,} font glyphs total")
    print()
    for k, v in blocks.most_common():
        print(f"  {k:14}: {v:>6,}")
    print()
    print(f"Output: {OUT}/char_universe.txt")
    print(f"        {OUT}/char_universe.json")


if __name__ == "__main__":
    main()
