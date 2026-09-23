"""Build the complete list of CJK characters to generate fd_cache for.

Strategy: UNION of CJK glyphs across a font CHAIN:
  1. NomNaTong  (primary — provides chu-Nom + Han-Viet, ~21,800 glyphs)
  2. HanaMinA   (CJK Unified + Ext A backfill — chars NomNaTong is missing)
  3. HanaMinB   (CJK Ext B/C/D/E/F backfill — rare chars)
  4. NotoSerifCJK SC (final backstop for any CJK Unified gaps)

Each font contributes only the CJK codepoints it has. Earlier fonts win for
provenance (so the per-char `source_font` reports the highest-quality font that
has the glyph).

The Kaggle FontDiffusion notebook reads `char_universe.txt` line-by-line and
generates a stylized image per character. With the font chain, the notebook
must also know which SOURCE font to render each character from (NomNaTong-style
chars are best rendered from NomNaTong; rare Ext-B chars must use HanaMinB,
etc.). That mapping is written to `char_universe.json` under `source_font`.

Output:
    kaggle_diffusion/exports/char_universe.txt
    kaggle_diffusion/exports/char_universe.json   {"chars": [{"cp", "char", "block", "source_font"}], "blocks": {...}}

Run on Mac:
    PATH="$PWD/.venv/bin:$PATH" python kaggle_diffusion/build_char_universe.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fontTools.ttLib import TTFont

REPO = Path(__file__).resolve().parent.parent
FONTS_DIR = REPO / "font_diffusion" / "fonts"
OUT = REPO / "kaggle_diffusion" / "exports"


# Ordered font chain. Earlier entries WIN when a codepoint exists in multiple
# fonts (so a Han-Nom Vietnamese-style glyph beats a generic CJK glyph for the
# same codepoint). Missing files are skipped silently with a warning.
#
# Priority groups:
#   1. Vietnamese chu-Nom fonts (NomNaTong, HanNom A/B, Han-Nom Kai/Minh) —
#      strokes and proportions match the historical Catholic books we OCR.
#   2. Generic CJK backfill (HanaMin A/B/C) for chars no Vietnamese font has.
#   3. Noto Serif CJK SC (optional) as final backstop.
FONT_CHAIN: list[tuple[str, str]] = [
    # Vietnamese Han-Nom fonts (preferred — match historical manuscript style)
    ("NomNaTong",      "NomNaTong-Regular.ttf"),
    ("NomNaTongLight", "NomNaTongLight.ttf"),
    ("HanNomA",        "HAN NOM A.ttf"),
    ("HanNomB",        "HAN NOM B.ttf"),
    ("HanNomKai",      "Han-Nom-Khai-Regular-300623.ttf"),
    ("HanNomMinh",     "Han-nom Minh 1.42.otf"),
    # Generic CJK backfill (large coverage, less stylistic match)
    ("HanaMinA",       "HanaMinA.ttf"),
    ("HanaMinB",       "HanaMinB.ttf"),
    ("HanaMinC",       "HanaMinC.otf"),
    # Optional — final backstop, install if you want full Unicode CJK coverage
    ("NotoSerifCJKsc", "NotoSerifCJKsc-Regular.otf"),
]


CJK_BLOCKS: list[tuple[int, int, str]] = [
    (0x4E00, 0x9FFF,   "CJK Unified"),
    (0x3400, 0x4DBF,   "Ext A"),
    (0xF900, 0xFAFF,   "Compat"),
    (0x20000, 0x2A6DF, "Ext B"),
    (0x2A700, 0x2B73F, "Ext C"),
    (0x2B740, 0x2B81F, "Ext D"),
    (0x2B820, 0x2CEAF, "Ext E"),
    (0x2CEB0, 0x2EBEF, "Ext F"),
    (0x2F800, 0x2FA1F, "Compat Supp"),
    (0x30000, 0x3134F, "Ext G"),
]


def cjk_block(cp: int) -> str | None:
    for lo, hi, name in CJK_BLOCKS:
        if lo <= cp <= hi:
            return name
    return None


def load_font_cmap(path: Path) -> set[int]:
    """Return the codepoints supported by a font, or empty set if missing."""
    if not path.exists():
        print(f"  ! font missing, skipping: {path.relative_to(REPO)}")
        return set()
    font = TTFont(str(path))
    return set(font.getBestCmap().keys())


def main() -> None:
    # cp -> (block, source_font) for the FIRST font in the chain that has it
    cp_owner: dict[int, tuple[str, str]] = {}
    per_font_counts: Counter = Counter()
    per_block_counts: Counter = Counter()

    for source_name, fname in FONT_CHAIN:
        cmap = load_font_cmap(FONTS_DIR / fname)
        if not cmap:
            continue
        added = 0
        for cp in cmap:
            block = cjk_block(cp)
            if block is None:
                continue
            if cp in cp_owner:
                continue  # earlier font in chain already owns this codepoint
            cp_owner[cp] = (block, source_name)
            per_block_counts[block] += 1
            added += 1
        per_font_counts[source_name] = added
        print(f"  {source_name:18} contributed {added:>6,} new chars "
              f"(font cmap = {len(cmap):,})")

    if not cp_owner:
        raise SystemExit("No fonts produced any CJK glyphs — check FONTS_DIR.")

    OUT.mkdir(parents=True, exist_ok=True)

    # Sorted char list (one per line)
    sorted_cps = sorted(cp_owner)
    with open(OUT / "char_universe.txt", "w", encoding="utf-8") as f:
        for cp in sorted_cps:
            f.write(chr(cp) + "\n")

    # JSON with source-font mapping (needed by the Kaggle notebook to pick the
    # right source font when rendering each character)
    chars_meta = [
        {
            "cp": cp,
            "hex": f"U+{cp:04X}",
            "char": chr(cp),
            "block": cp_owner[cp][0],
            "source_font": cp_owner[cp][1],
        }
        for cp in sorted_cps
    ]
    meta = {
        "total_chars": len(sorted_cps),
        "font_chain": [name for name, _ in FONT_CHAIN],
        "contributions_per_font": dict(per_font_counts.most_common()),
        "blocks": dict(per_block_counts.most_common()),
        "chars": chars_meta,
    }
    with open(OUT / "char_universe.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print()
    print(f"Universe: {len(sorted_cps):,} chars across {len(per_font_counts)} fonts")
    print()
    print("By Unicode block:")
    for k, v in per_block_counts.most_common():
        print(f"  {k:14}: {v:>6,}")
    print()
    print(f"Output: {OUT}/char_universe.txt   ({len(sorted_cps):,} lines)")
    print(f"        {OUT}/char_universe.json  (per-char source_font mapping)")


if __name__ == "__main__":
    main()
