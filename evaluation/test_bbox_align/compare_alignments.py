"""Compare OLD Levenshtein vs NEW bbox-aware on existing pipeline data.

Metric: Tier-1 hit rate (= % of match pairs where ocr_char ∈ dict_candidates[syllable]).
Higher is better — means alignment put the right char next to the right syllable.

Reads raw Kimhannom OCR cache + post-LLM-fix transcriptions, runs both aligners
on the same input, measures Tier-1 success rate. NO writes to pipeline data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.text.alignment import levenshtein_align  # OLD
from evaluation.test_bbox_align.bbox_aligner import bbox_aware_align  # NEW

DICT_PATH = REPO / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


def load_qn_to_nom() -> dict[str, set[str]]:
    df = pd.read_csv(DICT_PATH)
    qn_col = next(c for c in df.columns if "quoc" in c.lower())
    nom_col = next(c for c in df.columns if "nom" in c.lower())
    out: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        qn = str(r[qn_col]).strip().lower()
        nom = str(r[nom_col]).strip()
        if not qn or nom == "nan":
            continue
        out.setdefault(qn, set()).update(ch for ch in nom if ch and not ch.isspace())
    return out


def char_dict_from_ocr(ocr_chars: list[dict]) -> list[dict]:
    """Convert Kimhannom OCR per-col list to the same dict shape as the
    aligners expect (char, bbox, height, width, ocr_char)."""
    out = []
    for i, c in enumerate(ocr_chars):
        bb = c["bbox"]
        out.append({
            "char_idx": i,
            "bbox": [int(b) for b in bb],
            "height": int(bb[3] - bb[1]),
            "width": int(bb[2] - bb[0]),
            "ocr_char": c.get("char"),
        })
    return out


def tier1_score(pairs: list[dict], qn_to_nom: dict[str, set[str]]) -> tuple[int, int]:
    """Return (hits, total_matches). Hit = ocr_char in dict candidates of syllable."""
    hits, total = 0, 0
    for p in pairs:
        if p["type"] != "match":
            continue
        ch = p.get("char")
        syl = p.get("syllable")
        if not ch or not syl:
            continue
        ocr_ch = ch.get("ocr_char", "")
        if not ocr_ch:
            continue
        total += 1
        candidates = qn_to_nom.get(syl.lower(), set())
        if ocr_ch in candidates:
            hits += 1
    return hits, total


def process_book(book: str, qn_to_nom: dict[str, set[str]], max_pages: int = 0
                 ) -> dict:
    """Run both aligners on every page of `book`, return aggregate stats."""
    data_dir = REPO / "prepared" / book
    trans_dir = data_dir / "transcriptions"
    det_dir = data_dir / "detected"
    if not trans_dir.exists():
        return {"book": book, "skipped": True}

    pages = sorted(trans_dir.glob("page_*.json"))
    if max_pages:
        pages = pages[:max_pages]

    old_hits, old_tot = 0, 0
    new_hits, new_tot = 0, 0
    col_count = 0
    detail_examples = []

    for pf in pages:
        trans = json.loads(pf.read_text())
        page_name = pf.stem
        cache_path = det_dir / f"{page_name}_ocr_cache.json"
        if not cache_path.exists():
            continue
        cache = json.loads(cache_path.read_text())
        ocr_cols = cache.get("columns", [])
        trans_cols = trans.get("columns", [])

        for col_i in range(min(len(ocr_cols), len(trans_cols))):
            ocr_col = ocr_cols[col_i]
            syls = trans_cols[col_i].get("syllables", [])
            if not ocr_col or not syls:
                continue
            chars = char_dict_from_ocr(ocr_col)
            col_count += 1

            # OLD
            old_pairs = levenshtein_align(chars, syls, qn_to_nom=qn_to_nom)
            h, t = tier1_score(old_pairs, qn_to_nom)
            old_hits += h; old_tot += t

            # NEW
            new_pairs = bbox_aware_align(chars, syls)
            h2, t2 = tier1_score(new_pairs, qn_to_nom)
            new_hits += h2; new_tot += t2

            # Capture a few interesting cases (NEW wins big)
            if len(detail_examples) < 5 and h2 - h >= 3:
                detail_examples.append({
                    "page": page_name, "col": col_i,
                    "n_chars": len(chars), "n_syllables": len(syls),
                    "old_hits": h, "old_total": t,
                    "new_hits": h2, "new_total": t2,
                    "syllables": syls[:10],
                })

    return {
        "book": book,
        "columns_evaluated": col_count,
        "old_tier1_hits": old_hits, "old_tier1_total": old_tot,
        "old_tier1_rate": round(old_hits / max(old_tot, 1) * 100, 2),
        "new_tier1_hits": new_hits, "new_tier1_total": new_tot,
        "new_tier1_rate": round(new_hits / max(new_tot, 1) * 100, 2),
        "delta_pp": round((new_hits / max(new_tot, 1) - old_hits / max(old_tot, 1)) * 100, 2),
        "examples": detail_examples,
    }


def main() -> None:
    print("Loading dict...")
    qn_to_nom = load_qn_to_nom()
    print(f"  {len(qn_to_nom):,} QN syllables in dict")
    print()

    results = []
    for book in ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]:
        print(f"Processing {book}...")
        r = process_book(book, qn_to_nom)
        if r.get("skipped"):
            continue
        results.append(r)
        print(f"  cols={r['columns_evaluated']}")
        print(f"  OLD tier-1: {r['old_tier1_hits']:>5d}/{r['old_tier1_total']:<5d} = "
              f"{r['old_tier1_rate']:>5.2f}%")
        print(f"  NEW tier-1: {r['new_tier1_hits']:>5d}/{r['new_tier1_total']:<5d} = "
              f"{r['new_tier1_rate']:>5.2f}%")
        print(f"  Δ = {'+' if r['delta_pp'] >= 0 else ''}{r['delta_pp']:.2f}pp")
        print()

    # Aggregate
    o_h = sum(r["old_tier1_hits"] for r in results)
    o_t = sum(r["old_tier1_total"] for r in results)
    n_h = sum(r["new_tier1_hits"] for r in results)
    n_t = sum(r["new_tier1_total"] for r in results)
    print("=== OVERALL (3 Sach books) ===")
    print(f"  OLD: {o_h:>6d}/{o_t:<6d} = {o_h/max(o_t,1)*100:.2f}%")
    print(f"  NEW: {n_h:>6d}/{n_t:<6d} = {n_h/max(n_t,1)*100:.2f}%")
    print(f"  Δ   = {(n_h/max(n_t,1) - o_h/max(o_t,1))*100:+.2f}pp")

    # Write report
    md = ["# Bbox-aware vs Levenshtein alignment comparison", "",
          "Tier-1 hit rate = % of match pairs where ocr_char ∈ dict candidates",
          "for the aligned syllable. Higher = alignment kept right pairs together.",
          "",
          "## Per-book", "",
          "| Book | Cols | OLD hits/total | OLD % | NEW hits/total | NEW % | Δ pp |",
          "|------|-----:|---------------:|------:|---------------:|------:|-----:|"]
    for r in results:
        md.append(f"| {r['book']} | {r['columns_evaluated']:,} | "
                  f"{r['old_tier1_hits']:,}/{r['old_tier1_total']:,} | "
                  f"{r['old_tier1_rate']}% | "
                  f"{r['new_tier1_hits']:,}/{r['new_tier1_total']:,} | "
                  f"{r['new_tier1_rate']}% | "
                  f"{'+' if r['delta_pp'] >= 0 else ''}{r['delta_pp']:.2f} |")
    md += ["", "## Overall", "",
           f"- OLD: {o_h:,}/{o_t:,} = **{o_h/max(o_t,1)*100:.2f}%**",
           f"- NEW: {n_h:,}/{n_t:,} = **{n_h/max(n_t,1)*100:.2f}%**",
           f"- Δ: **{(n_h/max(n_t,1) - o_h/max(o_t,1))*100:+.2f}pp**", ""]

    md.append("## Sample wins (NEW gained ≥3 hits on a column)\n")
    for r in results:
        for ex in r.get("examples", []):
            md.append(f"- {r['book']} {ex['page']} col{ex['col']}: "
                      f"{ex['n_chars']} chars / {ex['n_syllables']} syllables  "
                      f"OLD {ex['old_hits']}/{ex['old_total']} → "
                      f"NEW {ex['new_hits']}/{ex['new_total']}")
    out_path = OUT / "comparison.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
