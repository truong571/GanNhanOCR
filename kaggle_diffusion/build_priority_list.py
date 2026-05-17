"""Build priority list of chars used by the current corpus.

Reads dataset/all/labels.csv (the final pipeline output) to extract every
unique Nom char actually labelled. These are the chars whose fd_cache image
matters MOST — generate them first on Kaggle so the user can resume the
local pipeline after just ~3-4h instead of waiting 50h for full universe.

Output:
    kaggle_diffusion/exports/priority_chars.txt   one char per line
    kaggle_diffusion/exports/priority_chars.json  metadata
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LABELS = REPO / "dataset" / "all" / "labels.csv"
UNIVERSE = REPO / "kaggle_diffusion" / "exports" / "char_universe.json"
OUT_TXT = REPO / "kaggle_diffusion" / "exports" / "priority_chars.txt"
OUT_JSON = REPO / "kaggle_diffusion" / "exports" / "priority_chars.json"


def main() -> None:
    if not LABELS.exists():
        raise SystemExit(f"{LABELS} not found — run pipeline first")
    if not UNIVERSE.exists():
        raise SystemExit(f"{UNIVERSE} not found — run build_char_universe.py first")

    df = pd.read_csv(LABELS)
    print(f"Read {len(df):,} labels from {LABELS.name}")

    # Unique chars in current dataset
    corpus_chars: set[str] = set()
    for ch in df["nom_char"].dropna().astype(str):
        if ch and len(ch) == 1:
            corpus_chars.add(ch)
    print(f"Unique chars in current corpus: {len(corpus_chars):,}")

    # Load universe
    meta = json.loads(UNIVERSE.read_text())
    universe_chars = {m["char"] for m in meta["chars"]}
    print(f"Universe chars: {len(universe_chars):,}")

    # Priority = corpus chars that are ALSO in universe (skip any outside)
    in_universe = corpus_chars & universe_chars
    outside_universe = corpus_chars - universe_chars
    print(f"  in universe: {len(in_universe):,}  → priority queue")
    print(f"  outside universe: {len(outside_universe):,}  → cannot generate (no font)")

    # Sort by codepoint for stability
    priority_sorted = sorted(in_universe, key=ord)

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(priority_sorted), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "total_priority": len(priority_sorted),
        "total_corpus_chars": len(corpus_chars),
        "outside_universe": sorted(outside_universe),
        "chars": [
            {"cp": ord(c), "hex": f"U+{ord(c):04X}", "char": c}
            for c in priority_sorted
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"Wrote:")
    print(f"  {OUT_TXT.relative_to(REPO)}   ({len(priority_sorted):,} chars)")
    print(f"  {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
