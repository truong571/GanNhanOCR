"""Extract tier-3 char lists per book — for offloading FontDiffusion to Kaggle.

Reads each book's aligned/page_*_aligned.json, runs tier-1 + tier-2 dictionary
matches, collects unique chars that fall through to tier 3 (need FontDiffusion).

Output (under kaggle_diffusion/exports/):
    chars_<book>.txt          one char per line
    chars_sach_union.txt      union across the 3 Sach books (dedup)
    book_chars_summary.json   counts and book→char mapping

Run:
    PATH="$PWD/.venv/bin:$PATH" python kaggle_diffusion/extract_book_chars.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from pipeline.step3_label import _collect_tier3_candidates
from core.text.dictionary import (
    build_nom_to_qn,
    load_qn_to_nom,
    load_similarity_dict,
)

BOOKS = ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]
OUT_DIR = PROJECT_ROOT / "kaggle_diffusion" / "exports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = yaml.safe_load(open(PROJECT_ROOT / "config" / "pipeline.yaml"))
    paths = cfg["paths"]
    data_dir = Path(paths["data_dir"])

    print("Loading dictionaries...")
    qn_to_nom = load_qn_to_nom(paths["qn_to_nom_dict"])
    nom_to_qn = build_nom_to_qn(qn_to_nom)
    similar_dict = load_similarity_dict(paths["similar_dict"])
    print(f"  QN→Nom: {len(qn_to_nom)} entries  |  Similar: {len(similar_dict)} entries\n")

    summary = {}
    union: set[str] = set()

    for book in BOOKS:
        aligned_dir = data_dir / book / "aligned"
        files = sorted(aligned_dir.glob("page_*_aligned.json"))
        if not files:
            print(f"[skip] {book}: no aligned files at {aligned_dir}")
            continue

        chars = _collect_tier3_candidates(
            files, data_dir / book, qn_to_nom, nom_to_qn, similar_dict,
        )
        chars_sorted = sorted(chars)

        out = OUT_DIR / f"chars_{book}.txt"
        out.write_text("\n".join(chars_sorted) + "\n", encoding="utf-8")
        print(f"  {book}: {len(chars_sorted):>5d} unique chars → {out.name}")
        summary[book] = {"count": len(chars_sorted), "file": out.name}
        union.update(chars_sorted)

    union_path = OUT_DIR / "chars_sach_union.txt"
    union_path.write_text("\n".join(sorted(union)) + "\n", encoding="utf-8")
    print(f"\n  UNION (dedup across 3 books): {len(union):>5d} chars → {union_path.name}")
    summary["__union__"] = {"count": len(union), "file": union_path.name}

    (OUT_DIR / "book_chars_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"\n✓ Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
