"""Stress-test: re-OCR every QN page in SachThanhTruyen11 (smallest Sach book).

Mirrors pipeline/step1_extract.py logic: iterate Nom pages (is_image_page),
take next page as QN, re-OCR with VietOCR. Measures:
  - total elapsed time
  - per-page lines & timing
  - global dict-hit rate vs current PDF-text dict-hit rate
  - any pages that fail line detection

Writes evaluation/test_qn_ocr/out/stress_<book>.{md,json}.

Usage:
    PYTHONPATH=. .venv/bin/python evaluation/test_qn_ocr/stress_test_full_book.py
    PYTHONPATH=. .venv/bin/python evaluation/test_qn_ocr/stress_test_full_book.py --book SachThanhTruyen2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import fitz
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from core.pdf.pdf_parser import is_image_page, parse_numbered_lines  # noqa: E402

# reuse helpers from POC
sys.path.insert(0, str(Path(__file__).parent))
from run_poc import (  # noqa: E402
    detect_text_lines, init_predictor, load_qn_dict, render_page,
    syllable_hit_rate,
)

OUT = Path(__file__).resolve().parent / "out"


def stress(book: str, max_pages: int | None = None) -> dict:
    pdf_path = REPO / "Data" / f"{book}.pdf"
    doc = fitz.open(str(pdf_path))
    total = doc.page_count

    # Find Nom -> QN pairs (mirror step1_extract.py logic)
    pairs: list[tuple[int, int]] = []
    for i in range(total):
        if not is_image_page(doc[i]):
            continue
        if i + 1 >= total:
            break
        nxt = doc[i + 1]
        if is_image_page(nxt):
            continue
        pairs.append((i, i + 1))

    if max_pages:
        pairs = pairs[:max_pages]

    print(f"Book: {book}")
    print(f"PDF pages: {total}, Nom-QN pairs: {len(pairs)}")
    print(f"Loading VietOCR (this caches across pages)...")
    predictor = init_predictor()
    qn_dict = load_qn_dict()

    rows = []
    t_total = time.time()
    pdf_hits_g, pdf_total_g = 0, 0
    ocr_hits_g, ocr_total_g = 0, 0
    fails = []

    for k, (nom_idx, qn_idx) in enumerate(pairs):
        t0 = time.time()
        try:
            img = render_page(pdf_path, qn_idx, dpi=300)
            boxes = detect_text_lines(img)
            if not boxes:
                fails.append({"qn_idx": qn_idx, "reason": "no lines detected"})
                continue
            from PIL import Image as PImg
            pil = PImg.fromarray(img)
            lines = []
            for (x1, y1, x2, y2) in boxes:
                crop = pil.crop((x1, y1, x2, y2))
                if crop.size[0] < 10 or crop.size[1] < 10:
                    continue
                try:
                    lines.append(predictor.predict(crop).strip())
                except Exception as e:
                    lines.append(f"<ERR:{e}>")
            ocr_txt = "\n".join(lines)

            # Compare with PyMuPDF text
            pdf_txt = doc[qn_idx].get_text("text")
            pdf_h, pdf_t = syllable_hit_rate(pdf_txt, qn_dict)
            ocr_h, ocr_t = syllable_hit_rate(ocr_txt, qn_dict)
            pdf_hits_g += pdf_h; pdf_total_g += pdf_t
            ocr_hits_g += ocr_h; ocr_total_g += ocr_t

            # also: parse_numbered_lines coverage on OCR output
            parsed = parse_numbered_lines(ocr_txt)
            elapsed = time.time() - t0

            rows.append({
                "k": k + 1,
                "qn_idx": qn_idx,
                "lines": len(boxes),
                "parsed_lines": len(parsed),
                "pdf_hit_pct": round(pdf_h / pdf_t * 100, 1) if pdf_t else 0,
                "ocr_hit_pct": round(ocr_h / ocr_t * 100, 1) if ocr_t else 0,
                "s": round(elapsed, 2),
            })
            print(f"  [{k+1:3d}/{len(pairs)}] qn={qn_idx:3d} "
                  f"lines={len(boxes):2d}  parsed={len(parsed):2d}  "
                  f"PDF {rows[-1]['pdf_hit_pct']:5.1f}% -> OCR {rows[-1]['ocr_hit_pct']:5.1f}%  "
                  f"({elapsed:.1f}s)")
        except Exception as e:
            fails.append({"qn_idx": qn_idx, "reason": str(e)})
            print(f"  [{k+1:3d}/{len(pairs)}] qn={qn_idx} FAIL: {e}")

    elapsed_total = time.time() - t_total
    doc.close()

    summary = {
        "book": book,
        "pairs_total": len(pairs),
        "pages_processed": len(rows),
        "pages_failed": len(fails),
        "elapsed_total_s": round(elapsed_total, 1),
        "elapsed_avg_per_page_s": round(elapsed_total / max(len(rows), 1), 2),
        "pdf_dict_hit_global": round(pdf_hits_g / max(pdf_total_g, 1) * 100, 2),
        "ocr_dict_hit_global": round(ocr_hits_g / max(ocr_total_g, 1) * 100, 2),
        "pdf_tokens_total": pdf_total_g,
        "ocr_tokens_total": ocr_total_g,
        "failures": fails,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stress_{book}.json").write_text(
        json.dumps({"summary": summary, "pages": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        f"# Stress test — {book}",
        "",
        f"- Pairs (Nom + QN): **{summary['pairs_total']}**",
        f"- Processed: {summary['pages_processed']}, failed: {summary['pages_failed']}",
        f"- Total elapsed: **{summary['elapsed_total_s']}s** "
        f"({summary['elapsed_total_s']/60:.1f} min)",
        f"- Avg per page: {summary['elapsed_avg_per_page_s']}s",
        "",
        "## Global dictionary-hit rate",
        "",
        f"- **PyMuPDF text** (current): {summary['pdf_dict_hit_global']}% "
        f"({pdf_hits_g:,}/{pdf_total_g:,} tokens)",
        f"- **VietOCR re-OCR** (proposed): **{summary['ocr_dict_hit_global']}%** "
        f"({ocr_hits_g:,}/{ocr_total_g:,} tokens)",
        f"- Delta: **+{summary['ocr_dict_hit_global'] - summary['pdf_dict_hit_global']:.1f}pp**",
        "",
        "## Failures",
        "",
    ]
    if fails:
        for f in fails:
            md.append(f"- qn_idx={f['qn_idx']}: {f['reason']}")
    else:
        md.append("None.")

    md += ["", "## Sample per-page (first 20)", "",
           "| k | qn_idx | lines | parsed | PDF % | OCR % | s |",
           "|--:|------:|------:|------:|-----:|-----:|--:|"]
    for r in rows[:20]:
        md.append(f"| {r['k']} | {r['qn_idx']} | {r['lines']} | "
                  f"{r['parsed_lines']} | {r['pdf_hit_pct']} | "
                  f"{r['ocr_hit_pct']} | {r['s']} |")
    (OUT / f"stress_{book}.md").write_text("\n".join(md), encoding="utf-8")

    print()
    print(f"== DONE ==")
    print(f"  PDF dict hit (global): {summary['pdf_dict_hit_global']}%")
    print(f"  OCR dict hit (global): {summary['ocr_dict_hit_global']}%")
    print(f"  Time: {summary['elapsed_total_s']}s ({summary['elapsed_total_s']/60:.1f} min)")
    print(f"  Wrote {OUT / f'stress_{book}.md'}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--book", default="SachThanhTruyen11")
    p.add_argument("--max-pages", type=int, default=None)
    args = p.parse_args()
    stress(args.book, args.max_pages)


if __name__ == "__main__":
    main()
