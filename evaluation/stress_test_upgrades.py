"""Stress test: prove the three pipeline upgrades recover labeling coverage
UNDER the adversarial conditions they were built for, using the real 3 books.

The current corpus is clean (digital-text QN pages, 0° skew) and its OCR already
agrees with the old-style dictionary, so the upgrades are 0-delta there. This
script injects the conditions that DO occur on scanned / modern-OCR books and
measures the coverage they rescue.

Part 1  — Line detector (real end-to-end OCR). Take real QN pages, inject skew,
          run VietOCR through the OLD projection finder vs the NEW deskew finder,
          and count dict-covered syllables (= labelable pairs). Reports coverage
          RETENTION vs the upright reference.
Part 1b — Detection-only line-recovery curve over more pages (cheap, no OCR).
Part 2  — Tone canonicalization (dict level). Simulate a modern-tone OCR over the
          words the dictionary stores only in the old style, and measure recall
          WITHOUT vs WITH the fix.

Writes evaluation/results/stress_test.json (+ a chart) for the PDF report.
Run:  .venv/bin/python evaluation/stress_test_upgrades.py [--pages-per-book N] [--angle D]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.text.text_utils import normalize_syllables, normalize_tone_marks as NT
from core.text.dictionary import load_qn_to_nom
from core.ocr.qn_ocr import ocr_qn_page
from core.ocr import line_detector as LD
from core.pdf.pdf_parser import parse_numbered_lines, build_transcription_columns

BOOKS = ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]
DICT_CSV = REPO / "dict" / "QuocNgu_SinoNom_TongHop3.csv"
RES = REPO / "evaluation" / "results"
RES.mkdir(parents=True, exist_ok=True)
SCRATCH = Path("/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/"
               "cebd7cad-62ee-4640-b5f3-1b4e1632eae2/scratchpad/stress")
SCRATCH.mkdir(parents=True, exist_ok=True)

_QN_TO_NOM = load_qn_to_nom(str(DICT_CSV))
_QN_SET = set(_QN_TO_NOM.keys())


def _rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderValue=(255, 255, 255))


def _covered_syllables(ocr_text: str) -> tuple[int, int]:
    """Replicate the step1 flow -> (#syllables, #dict-covered syllables)."""
    ocr_text = re.sub(r"\n\d+\s*$", "", (ocr_text or "").strip())
    raw_lines = parse_numbered_lines(ocr_text)
    columns = build_transcription_columns(raw_lines)
    syls: list[str] = []
    for col in columns:
        syls += normalize_syllables(col["syllables"], _QN_SET)
    covered = sum(1 for s in syls if NT(s).lower() in _QN_TO_NOM)
    return len(syls), covered


def _ocr_image(arr_rgb, backend: str, tag: str) -> str:
    p = SCRATCH / f"{tag}.png"
    cv2.imwrite(str(p), cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
    text, _ = ocr_qn_page(str(p), verbose=False, cache_path=None, backend=backend)
    return text


def sample_pages(per_book: int) -> list[Path]:
    out = []
    for b in BOOKS:
        ps = sorted(glob.glob(f"{REPO}/prepared/{b}/transcriptions/*_qn_tmp.png"))
        # spread across the book, skip the first few (covers/blanks)
        if not ps:
            continue
        step = max(1, len(ps) // (per_book + 1))
        out += [Path(p) for p in ps[step::step][:per_book]]
    return out


def part1_line_detector(pages: list[Path], angle: float) -> dict:
    """Real end-to-end OCR coverage: upright ref vs skewed(old) vs skewed(new)."""
    agg = {"ref": [0, 0], "skew_proj": [0, 0], "skew_deskew": [0, 0]}
    per_page = []
    for i, p in enumerate(pages):
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        sk = _rotate(rgb, angle)
        t0 = time.time()
        # reference: upright, original projection finder
        _, c_ref = _covered_syllables(_ocr_image(rgb, "projection", f"ref_{i}"))
        # skewed + OLD projection finder
        _, c_old = _covered_syllables(_ocr_image(sk, "projection", f"old_{i}"))
        # skewed + NEW deskew finder
        _, c_new = _covered_syllables(_ocr_image(sk, "projection_deskew", f"new_{i}"))
        agg["ref"][1] += c_ref
        agg["skew_proj"][1] += c_old
        agg["skew_deskew"][1] += c_new
        per_page.append({"page": p.name, "ref": c_ref, "old": c_old, "new": c_new})
        print(f"  [part1] {i+1}/{len(pages)} {p.name}: ref={c_ref} "
              f"skew{angle}°(old)={c_old} skew{angle}°(new)={c_new}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    ref = agg["ref"][1] or 1
    return {
        "angle": angle, "n_pages": len(pages), "per_page": per_page,
        "covered_ref": agg["ref"][1],
        "covered_skew_old": agg["skew_proj"][1],
        "covered_skew_new": agg["skew_deskew"][1],
        "retention_old_pct": round(agg["skew_proj"][1] / ref * 100, 1),
        "retention_new_pct": round(agg["skew_deskew"][1] / ref * 100, 1),
    }


def part1b_curve(pages: list[Path], degs=(0, 1, 2, 3, 4, 5, 6, 8)) -> dict:
    """Detection-only (no OCR): mean lines recovered vs skew, old vs new."""
    old = {d: 0 for d in degs}
    new = {d: 0 for d in degs}
    for p in pages:
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        for d in degs:
            sk = _rotate(rgb, d)
            old[d] += len(LD.detect_line_crops(sk, backend="projection"))
            new[d] += len(LD.detect_line_crops(sk, backend="projection_deskew"))
    n = len(pages)
    return {"degs": list(degs), "n_pages": n,
            "old_mean": [round(old[d] / n, 1) for d in degs],
            "new_mean": [round(new[d] / n, 1) for d in degs]}


def part2_tone() -> dict:
    """Modern-OCR-vs-old-dict recall, WITHOUT vs WITH canonicalization."""
    # raw dict (pre-change): lowercased keys, no canonicalization
    raw_keys = set()
    with open(DICT_CSV, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                raw_keys.add(row[0].strip().lower())

    # words the dictionary stores ONLY in the old style (no modern twin present)
    old_only = sorted({k for k in raw_keys
                       if NT(k) != k and NT(k) not in raw_keys})
    # a modern OCR would emit the modern spelling of these words
    modern = [NT(k) for k in old_only]
    hit_without = sum(1 for w in modern if w in raw_keys)          # old dict
    hit_with = sum(1 for w in modern if NT(w) in _QN_TO_NOM)       # canon dict
    n = len(modern) or 1
    return {
        "n_words_old_only": len(old_only),
        "recall_without_pct": round(hit_without / n * 100, 1),
        "recall_with_pct": round(hit_with / n * 100, 1),
        "examples": [f"{k}→{NT(k)}" for k in old_only[:12]],
    }


def make_chart(curve: dict, p1: dict):
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                         "font.family": "DejaVu Sans"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.2))
    ax1.plot(curve["degs"], curve["old_mean"], "o-", color="#b22222",
             label="Chiếu ngang (cũ)", lw=2)
    ax1.plot(curve["degs"], curve["new_mean"], "s-", color="#1f6f3f",
             label="Deskew (mới)", lw=2)
    ax1.set_xlabel("Góc nghiêng (độ)")
    ax1.set_ylabel("Dòng phát hiện TB / trang")
    ax1.set_title(f"Phục hồi dòng vs nghiêng (n={curve['n_pages']} trang)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    labels = ["Gốc\n(0°)", f"Nghiêng {p1['angle']:.0f}°\nCŨ", f"Nghiêng {p1['angle']:.0f}°\nMỚI"]
    vals = [p1["covered_ref"], p1["covered_skew_old"], p1["covered_skew_new"]]
    bars = ax2.bar(labels, vals, color=["#888", "#b22222", "#1f6f3f"])
    ax2.set_ylabel("Âm tiết có nhãn (dict-covered)")
    ax2.set_title(f"Coverage end-to-end (OCR thật, n={p1['n_pages']} trang)")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v, str(v),
                 ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = RES / "stress_test_chart.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-per-book", type=int, default=2)
    ap.add_argument("--angle", type=float, default=4.0)
    args = ap.parse_args()

    pages = sample_pages(args.pages_per_book)
    print(f"[stress] sampled {len(pages)} pages, angle={args.angle}°", flush=True)

    print("[stress] Part 2: tone (dict level) ...", flush=True)
    p2 = part2_tone()
    print("[stress] Part 1b: detection curve ...", flush=True)
    curve = part1b_curve(pages)
    print("[stress] Part 1: end-to-end OCR coverage under skew "
          "(this loads VietOCR) ...", flush=True)
    p1 = part1_line_detector(pages, args.angle)

    chart = make_chart(curve, p1)
    result = {"angle": args.angle, "pages": [p.name for p in pages],
              "part1_line_detector": p1, "part1b_curve": curve, "part2_tone": p2,
              "chart": str(chart.relative_to(REPO))}
    (RES / "stress_test.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n==== STRESS TEST SUMMARY ====")
    print(f"Line detector @ {args.angle}°: coverage retention "
          f"OLD={p1['retention_old_pct']}%  NEW={p1['retention_new_pct']}%  "
          f"(ref={p1['covered_ref']} âm tiết có nhãn)")
    print(f"Tone fix: {p2['n_words_old_only']} từ điển chỉ-có-kiểu-cũ; "
          f"recall modern-OCR WITHOUT={p2['recall_without_pct']}%  "
          f"WITH={p2['recall_with_pct']}%")
    print(f"wrote {RES/'stress_test.json'}")


if __name__ == "__main__":
    main()
