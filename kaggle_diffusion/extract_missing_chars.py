"""Find chars needed by Tier-3 in our current corpus but missing from fd_cache.

Output:
    kaggle_diffusion/exports/missing_chars.txt   one char per line
    kaggle_diffusion/exports/missing_chars.json  per-source-font breakdown

Run this on the local machine after step 1+2 finish. The output file is what
you upload + generate on Kaggle (diffusion_run.ipynb already loops over the
font chain so the chars get rendered with the right source font).

Usage:
    PATH="$PWD/.venv/bin:$PATH" python kaggle_diffusion/extract_missing_chars.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml

from pipeline.step3_label import _collect_tier3_candidates
from core.text.dictionary import (
    build_nom_to_qn, load_qn_to_nom, load_similarity_dict,
)

FD_CACHE = REPO / "prepared" / "_universal_fd_cache"
UNIVERSE_JSON = REPO / "kaggle_diffusion" / "exports" / "char_universe.json"
OUT_TXT = REPO / "kaggle_diffusion" / "exports" / "missing_chars.txt"
OUT_JSON = REPO / "kaggle_diffusion" / "exports" / "missing_chars.json"


def main() -> None:
    cfg = yaml.safe_load(open(REPO / "config" / "pipeline.yaml"))
    paths = cfg["paths"]
    data_dir = Path(paths["data_dir"])

    print("Loading dictionaries...")
    qn_to_nom = load_qn_to_nom(paths["qn_to_nom_dict"])
    nom_to_qn = build_nom_to_qn(qn_to_nom)
    similar_dict = load_similarity_dict(paths["similar_dict"])

    # Existing fd_cache codepoints
    cached_cps: set[int] = set()
    if FD_CACHE.exists():
        for png in FD_CACHE.iterdir():
            if png.name.startswith("U+") and png.name.endswith(".png"):
                try:
                    cached_cps.add(int(png.name[2:-4], 16))
                except ValueError:
                    pass
    print(f"  fd_cache currently has {len(cached_cps):,} codepoints")

    # Universe metadata (per-char source font)
    universe_meta = json.loads(UNIVERSE_JSON.read_text()) if UNIVERSE_JSON.exists() else None
    cp_to_font = (
        {m["cp"]: m["source_font"] for m in universe_meta["chars"]}
        if universe_meta else {}
    )

    # Walk all books in config, collect tier-3 candidates
    all_missing: set[str] = set()
    per_book: dict[str, int] = {}
    for book in cfg["books"]:
        name = book["name"]
        book_dir = data_dir / name / "aligned"
        if not book_dir.exists():
            continue
        aligned_files = sorted(book_dir.glob("page_*_aligned.json"))
        if not aligned_files:
            continue
        tier3_chars = _collect_tier3_candidates(
            aligned_files, data_dir / name, qn_to_nom, nom_to_qn, similar_dict,
        )
        missing = {c for c in tier3_chars if ord(c) not in cached_cps}
        all_missing |= missing
        per_book[name] = len(missing)
        print(f"  {name:25s} tier3={len(tier3_chars):>5}  "
              f"missing-from-cache={len(missing):>5}")

    print()
    print(f"Total unique missing chars: {len(all_missing):,}")

    # Group by source font in chain
    by_font: Counter = Counter()
    for ch in all_missing:
        by_font[cp_to_font.get(ord(ch), "UNKNOWN")] += 1
    print()
    print("Breakdown by source font (priority order in font chain):")
    for fname, cnt in by_font.most_common():
        print(f"  {fname:18s} {cnt:>5,}")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(sorted(all_missing)), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "total": len(all_missing),
        "by_book": per_book,
        "by_source_font": dict(by_font),
        "chars": sorted([
            {"char": c, "cp": ord(c), "hex": f"U+{ord(c):04X}",
             "source_font": cp_to_font.get(ord(c), "UNKNOWN")}
            for c in all_missing
        ], key=lambda x: x["cp"]),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"Output:")
    print(f"  {OUT_TXT.relative_to(REPO)}   ({len(all_missing):,} lines)")
    print(f"  {OUT_JSON.relative_to(REPO)}  (with source_font mapping)")
    print()
    print("Next:")
    print("  1. Commit + push to repo so Kaggle notebook can fetch via git pull.")
    print("  2. On Kaggle: in diffusion_run.ipynb cell 13, point work-list to")
    print("     missing_chars.txt (or modify cell to read missing_chars.json")
    print("     and group by source font as the notebook already supports).")
    print("  3. After Kaggle generates them, pull cache back:")
    print("     huggingface-cli download <repo> --repo-type=dataset \\")
    print("       --local-dir prepared/_universal_fd_cache/")
    print("  4. Re-run only step 3 + step 4 — Tier-3 will now resolve.")


if __name__ == "__main__":
    main()
