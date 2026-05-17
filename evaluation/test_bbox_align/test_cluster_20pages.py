"""Test cluster-based alignment on 20 random pages — confirm ROI.

For each page:
  1. Read raw Kimhannom OCR cache + (post-LLM-fix) transcription
  2. Run BOTH aligners on same input:
     - OLD: levenshtein_align (pipeline default)
     - NEW: cluster_align (cluster-by-Y-gap)
  3. Compute tier-1 hit rate per page + total
  4. Report breakdown
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.text.alignment import levenshtein_align
from evaluation.test_bbox_align.cluster_aligner import cluster_align

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 20
SEED = 7


def load_qn_to_nom() -> dict[str, set[str]]:
    df = pd.read_csv(REPO / "Dict" / "QuocNgu_SinoNom_TongHop3.csv")
    out: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        qn = str(r["QuocNgu"]).strip().lower()
        nom = str(r["SinoNom"]).strip()
        if qn and nom != "nan":
            out.setdefault(qn, set()).update(c for c in nom if not c.isspace())
    return out


def char_from_ocr(ocr_chars: list[dict]) -> list[dict]:
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


def tier1_hits(pairs: list[dict], qn_to_nom: dict[str, set[str]]
               ) -> tuple[int, int]:
    hits, total = 0, 0
    for p in pairs:
        if p.get("type") != "match":
            continue
        ch = p.get("char") or {}
        syl = (p.get("syllable") or "").strip().lower()
        ocr_ch = ch.get("ocr_char", "")
        if not syl or not ocr_ch:
            continue
        total += 1
        if ocr_ch in qn_to_nom.get(syl, set()):
            hits += 1
    return hits, total


def collect_random_pages(n: int) -> list[tuple[str, Path]]:
    """Collect (book, transcription_json_path) for n random pages."""
    pages = []
    for book_dir in sorted((REPO / "prepared").iterdir()):
        if not book_dir.is_dir() or book_dir.name.startswith("_"):
            continue
        trans_dir = book_dir / "transcriptions"
        if not trans_dir.exists():
            continue
        for f in sorted(trans_dir.glob("page_*.json")):
            pages.append((book_dir.name, f))
    random.seed(SEED)
    return random.sample(pages, min(n, len(pages)))


def main() -> None:
    qn_to_nom = load_qn_to_nom()
    sampled = collect_random_pages(SAMPLE_SIZE)
    print(f"Testing {len(sampled)} random pages\n")

    rows = []
    sum_old_h = sum_old_t = sum_new_h = sum_new_t = 0

    for book, trans_path in sampled:
        page = trans_path.stem
        cache_path = REPO / "prepared" / book / "detected" / f"{page}_ocr_cache.json"
        if not cache_path.exists():
            continue
        trans = json.loads(trans_path.read_text())
        cache = json.loads(cache_path.read_text())

        kim_cols = cache.get("columns", [])
        qn_cols = trans.get("columns", [])

        old_h = old_t = new_h = new_t = 0
        for col_i in range(min(len(kim_cols), len(qn_cols))):
            chars = char_from_ocr(kim_cols[col_i])
            syls = qn_cols[col_i].get("syllables", [])
            if not chars or not syls:
                continue
            # OLD: pipeline's Levenshtein
            old_pairs = levenshtein_align(chars, syls, qn_to_nom=qn_to_nom)
            h, t = tier1_hits(old_pairs, qn_to_nom)
            old_h += h; old_t += t
            # NEW: cluster-based
            new_pairs = cluster_align(chars, syls)
            h, t = tier1_hits(new_pairs, qn_to_nom)
            new_h += h; new_t += t

        rows.append({
            "book": book, "page": page,
            "old_h": old_h, "old_t": old_t,
            "old_pct": round(old_h / max(old_t, 1) * 100, 1),
            "new_h": new_h, "new_t": new_t,
            "new_pct": round(new_h / max(new_t, 1) * 100, 1),
            "delta": round(new_h / max(new_t, 1) * 100 -
                           old_h / max(old_t, 1) * 100, 1),
        })
        sum_old_h += old_h; sum_old_t += old_t
        sum_new_h += new_h; sum_new_t += new_t

    print(f"{'BOOK':22s} {'PAGE':12s} {'OLD':>10s} {'NEW':>10s} {'Δ':>7s}")
    print("-" * 70)
    for r in rows:
        print(f"  {r['book']:20s} {r['page']:12s} "
              f"{r['old_h']:>3}/{r['old_t']:<3} {r['old_pct']:>4.1f}%  "
              f"{r['new_h']:>3}/{r['new_t']:<3} {r['new_pct']:>4.1f}%  "
              f"{r['delta']:+5.1f}pp")

    print("-" * 70)
    old_pct = sum_old_h / max(sum_old_t, 1) * 100
    new_pct = sum_new_h / max(sum_new_t, 1) * 100
    print(f"  TOTAL                            "
          f"{sum_old_h:>3}/{sum_old_t:<3} {old_pct:>4.1f}%  "
          f"{sum_new_h:>3}/{sum_new_t:<3} {new_pct:>4.1f}%  "
          f"{new_pct-old_pct:+5.1f}pp")
    print()
    print(f"=== SUMMARY ===")
    print(f"OLD Levenshtein:   {sum_old_h:>5}/{sum_old_t:<5} = {old_pct:.2f}%")
    print(f"NEW cluster-Y-gap: {sum_new_h:>5}/{sum_new_t:<5} = {new_pct:.2f}%")
    print(f"Δ = {new_pct - old_pct:+.2f}pp  ({(new_pct/max(old_pct,0.01)):.1f}× change)")

    # Write report
    md = [
        "# Cluster alignment — 20-page ROI test",
        "",
        f"Sample: {len(rows)} pages random across 3 Sach books (seed={SEED}).",
        "Tier-1 hit = align_pair.ocr_char ∈ dict_candidates[align_pair.syllable]",
        "",
        f"## TOTAL",
        f"- OLD (Levenshtein): **{sum_old_h:,}/{sum_old_t:,} = {old_pct:.2f}%**",
        f"- NEW (Cluster Y-gap): **{sum_new_h:,}/{sum_new_t:,} = {new_pct:.2f}%**",
        f"- Δ = **{new_pct - old_pct:+.2f}pp**",
        "",
        "## Per-page",
        "| Book | Page | OLD | NEW | Δ |",
        "|------|------|----:|----:|--:|",
    ]
    for r in rows:
        md.append(f"| {r['book']} | {r['page']} | "
                  f"{r['old_pct']}% | {r['new_pct']}% | "
                  f"{'+' if r['delta'] >= 0 else ''}{r['delta']:.1f}pp |")

    (OUT / "cluster_20pages.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT / 'cluster_20pages.md'}")


if __name__ == "__main__":
    main()
