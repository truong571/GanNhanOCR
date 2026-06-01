"""Build the FINAL labeled dataset end-to-end (the NEW pipeline, no A/B).

One command: production-faithful column detection (detect_nom_columns_v3) +
QN parse (parse_v5) -> banded anchored-DP alignment -> 3-signal consensus
(S1 OCR + S2 dictionary + optional S3 DINOv2/FontDiffusion) -> materialize the
GOLD (and SILVER with --use-s3) character crops + a labels manifest.

Output (default evaluation/ver_new/dataset_out/):
  gold/<book>_<page>_c<col>_<idx>.png      cropped Nôm glyph, dict-confirmed
  silver/<book>_<page>_c<col>_<idx>.png    visual-recovered (only with --use-s3)
  labels.csv                               every pair: image, book, page, column,
                                           ocr_char, syllable, label, tier, rule, s3_cosine, bbox
  summary.json

Run:
  cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
  # GOLD dataset (fast, dictionary only, no model):
  .venv/bin/python evaluation/ver_new/build_dataset.py
  # GOLD + SILVER (adds DINOv2/FD visual recovery via gannhanocr-fd, ~30-45 min):
  .venv/bin/python evaluation/ver_new/build_dataset.py --use-s3
  # quick smoke:
  .venv/bin/python evaluation/ver_new/build_dataset.py --use-s3 --limit 10
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom, load_similarity_dict  # noqa: E402
from evaluation.ver_new.align_production import align_page          # noqa: E402
from evaluation.ver_new.consensus import decide_label              # noqa: E402
from evaluation.ver_new.bbox_fix import tighten_box                # noqa: E402


def maybe_s3(p, page_png, qn_to_nom, vs3):
    """S3 only where it can change the tier: anchored, non-dict-confirmed pairs."""
    if vs3 is None or not p.get("ocr_char"):
        return None
    if not (p.get("matched") or p.get("anchored")):
        return None
    cands = qn_to_nom.get((p["syllable"] or "").lower(), [])
    if p["ocr_char"] in cands:
        return None
    return vs3.compute(page_png, p.get("bbox"), p["ocr_char"], cands)


def save_crop(img, bbox, pad, path: Path, tighten: bool = True) -> bool:
    if img is None or not bbox:
        return False
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(W, x2 + pw), min(H, y2 + ph)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    if tighten:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        tb = tighten_box(gray)
        if tb is not None:
            a, c, b, d = tb
            crop = crop[c:d, a:b]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "dataset_out"))
    ap.add_argument("--use-s3", action="store_true",
                    help="enable visual S3 (DINOv2+FontDiffusion) -> adds SILVER tier")
    ap.add_argument("--limit", type=int, default=0, help="max pages per book (0=all)")
    ap.add_argument("--no-crops", action="store_true", help="write labels.csv only, no PNGs")
    ap.add_argument("--no-tighten", action="store_true",
                    help="skip binary-projection tightening (keep loose OCR bbox)")
    ap.add_argument("--pad", type=float, default=0.12, help="bbox pad fraction per side")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    paths = config["paths"]
    qn_to_nom = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    qn_dict_set = set(qn_to_nom.keys())
    similar = load_similarity_dict(str(REPO / paths["similar_dict"]))
    data_root = REPO / paths["data_dir"]

    vs3 = None
    if args.use_s3:
        from evaluation.ver_new.visual_signal import VisualS3
        print("Loading S3 (DINOv2 + FontDiffusion cache gannhanocr-fd)...", flush=True)
        vs3 = VisualS3(REPO, font_path=str(REPO / paths["font_path"]),
                       fd_dir=str(REPO / paths["fd_cache_universal"]),
                       cache_dir=str(out / "emb_cache"))
        print(f"  FD glyph index: {len(vs3.fd_index)} chars", flush=True)

    labels = []
    counts = Counter()
    pages_done = 0
    n_s3 = 0

    for b in config["books"]:
        book = b["name"]
        data_dir = data_root / book
        trans = sorted(glob.glob(str(data_dir / "transcriptions" / "page_*.json")))
        trans = [t for t in trans if not t.endswith("_qn_ocr_cache.json")]
        if args.limit:
            trans = trans[:args.limit]
        print(f"[{book}] {len(trans)} pages ...", flush=True)

        for pi, tf in enumerate(trans):
            page = Path(tf).stem
            try:
                rec = align_page(page, data_dir, qn_dict_set, qn_to_nom, similar, "new")
            except Exception as e:
                print(f"   [warn] {book}/{page}: {type(e).__name__}: {e}", flush=True)
                continue
            if rec is None:
                continue
            pages_done += 1
            if (pi + 1) % 20 == 0:
                print(f"    {pi + 1}/{len(trans)} trang | GOLD {counts['GOLD']} "
                      f"SILVER {counts['SILVER']} (S3 {n_s3})", flush=True)
            page_png = str(data_dir / "pages" / f"{page}.png")
            img = None if args.no_crops else cv2.imread(page_png, cv2.IMREAD_COLOR)

            for idx, p in enumerate(rec["pairs"]):
                s3 = maybe_s3(p, page_png, qn_to_nom, vs3) if vs3 else None
                if s3 is not None:
                    n_s3 += 1
                dec = decide_label(p.get("ocr_char"), p["syllable"], p.get("matched", False),
                                   qn_to_nom, similar, s3=s3, anchored=p.get("anchored", False))
                counts[dec.tier] += 1
                img_rel = ""
                if dec.tier in ("GOLD", "SILVER") and not args.no_crops:
                    fname = f"{book[12:]}_{page}_c{p['column']:02d}_{idx:03d}.png"
                    if save_crop(img, p.get("bbox"), args.pad, out / dec.tier.lower() / fname,
                                 tighten=not args.no_tighten):
                        img_rel = f"{dec.tier.lower()}/{fname}"
                labels.append({
                    "image": img_rel, "book": book[12:], "page": page,
                    "column": p["column"], "ocr_char": p.get("ocr_char") or "",
                    "syllable": p["syllable"], "label": dec.label or "",
                    "tier": dec.tier, "rule": dec.rule_id,
                    "s3_cosine": round(s3.cosine, 3) if s3 else "",
                    "bbox": json.dumps(p.get("bbox")),
                })

    # ---- write manifest ---------------------------------------------------
    with open(out / "labels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "book", "page", "column", "ocr_char",
                                          "syllable", "label", "tier", "rule",
                                          "s3_cosine", "bbox"])
        w.writeheader(); w.writerows(labels)
    summary = {"pages": pages_done, "use_s3": bool(vs3), "s3_computed": n_s3,
               "tiers": dict(counts), "total_labels": len(labels),
               "usable_gold_silver": counts["GOLD"] + counts["SILVER"]}
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f" DATASET BUILT -> {out}")
    print("=" * 64)
    print(f" pages {pages_done} | total labels {len(labels)}")
    for t in ("GOLD", "SILVER", "REVIEW"):
        crop_note = "" if args.no_crops or t == "REVIEW" else f" -> {t.lower()}/*.png"
        print(f"   {t:7s}: {counts.get(t,0):6d}{crop_note}")
    print(f" USABLE (gold+silver): {counts['GOLD']+counts['SILVER']}")
    if vs3 is not None:
        print(f" S3 computed on {n_s3} crops (FD refs {vs3.n_fd}, font refs {vs3.n_font})")
    print(f" manifest: {out}/labels.csv   summary: {out}/summary.json")


if __name__ == "__main__":
    main()
