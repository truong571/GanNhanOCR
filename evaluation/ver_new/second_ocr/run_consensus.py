"""Measure a SECOND OCR as the visual signal — compare vs DINOv2 / classifier.

Two checks (forced 4-choice: the answer + 3 distractors, shuffled):
  A (calibrate, default): on GOLD crops where the TRUE char is known -> 2nd-OCR
     top-1 accuracy = "can it read woodblock Nôm at all?" (DINOv2 = 0%; the
     trained classifier targets >= 80%).
  B (--target review): on REVIEW 'unconfirmed' crops (cropped from the page via
     bbox) -> agreement with the OCR char (S1) = consensus-confirm rate (how many
     unconfirmed crops a 2nd-OCR consensus would promote to char-level).

Uses GEMINI_API_KEY from .env (backend=gemini). Costs ~1 API call/crop, so --n
is small by default (this is a comparison on a sample, not a full run).

Run:
  cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
  .venv/bin/python evaluation/ver_new/second_ocr/run_consensus.py --n 50
  .venv/bin/python evaluation/ver_new/second_ocr/run_consensus.py --target review --n 50
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
import sys
sys.path.insert(0, str(HERE))
from ocr_backends import get_backend          # noqa: E402

DATASET = HERE.parent / "dataset_out"


def png_bytes_from_file(p: Path) -> bytes:
    return p.read_bytes()


def png_bytes_from_page(book: str, page: str, bbox, pad: float = 0.12) -> bytes | None:
    img = cv2.imread(str(REPO / "prepared" / f"SachThanhTruyen{book.replace('yen','')}"
                     / "pages" / f"{page}.png"))
    if img is None or not bbox:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ok, buf = cv2.imencode(".png", crop)
    return buf.tobytes() if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gemini")
    ap.add_argument("--model", default="gemini-2.0-flash")
    ap.add_argument("--target", choices=["gold", "review"], default="gold")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=4.0,
                    help="giây nghỉ giữa các call (free-tier Gemini ~15 RPM -> dùng ~4s)")
    args = ap.parse_args()
    random.seed(args.seed)

    be = get_backend(args.backend, model=args.model) if args.backend == "gemini" \
        else get_backend(args.backend)
    rows = list(csv.DictReader(open(DATASET / "labels.csv", encoding="utf-8")))
    gold_chars = sorted({r["label"] for r in rows if r["tier"] == "GOLD" and r["label"]})

    if args.target == "gold":
        pool = [r for r in rows if r["tier"] == "GOLD" and r["image"] and r["label"]]
        random.shuffle(pool)
        n_ok = n = 0
        ex = []
        for r in pool:
            if n >= args.n:
                break
            b = png_bytes_from_file(DATASET / r["image"])
            cands = [r["label"]] + random.sample([c for c in gold_chars if c != r["label"]], 3)
            random.shuffle(cands)
            try:
                pick = be.pick(b, cands)
            except Exception as e:
                print("  [warn]", type(e).__name__, e); continue
            n += 1; n_ok += (pick == r["label"])
            if len(ex) < 8:
                ex.append((r["label"], pick, "OK" if pick == r["label"] else "x"))
            time.sleep(args.sleep)
        print(f"\n=== 2nd-OCR [{args.backend}/{args.model}] — GOLD (ground truth) ===")
        print(f"forced 4-choice top-1: {n_ok}/{n} = {n_ok/max(n,1)*100:.0f}%  "
              f"(DINOv2 = 0%; classifier target >= 80%)")
        print("ví dụ (true -> pick):", ex)
    else:
        pool = [r for r in rows if r["rule"] == "unconfirmed_no_s3" and r["ocr_char"] and r["bbox"]]
        random.shuffle(pool)
        n_agree = n = 0
        for r in pool:
            if n >= args.n:
                break
            try:
                bbox = json.loads(r["bbox"])
            except Exception:
                continue
            b = png_bytes_from_page(r["book"], r["page"], bbox)
            if b is None:
                continue
            cands = [r["ocr_char"]] + random.sample([c for c in gold_chars if c != r["ocr_char"]], 3)
            random.shuffle(cands)
            try:
                pick = be.pick(b, cands)
            except Exception as e:
                print("  [warn]", type(e).__name__, e); continue
            n += 1; n_agree += (pick == r["ocr_char"])
            time.sleep(args.sleep)
        print(f"\n=== 2nd-OCR [{args.backend}] — REVIEW unconfirmed (agreement vs S1 ocr_char) ===")
        print(f"consensus-confirm rate: {n_agree}/{n} = {n_agree/max(n,1)*100:.0f}%  "
              f"(ước lượng % unconfirmed có thể nâng lên char-level bằng đồng thuận 2-OCR)")


if __name__ == "__main__":
    main()
