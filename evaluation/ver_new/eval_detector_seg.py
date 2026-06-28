"""Deployment test of the trained detector: on DIVERGED columns, does
count-constrained detector segmentation beat the midpoint baseline on the
merged-crop proxies (two_blob, MLS) — the metric that actually matters (VAL F1
is NOT, because count_constrained forces N at inference)?

Compares, per diverged column (OCR count != N = QN syllables):
  A (midpoint, current production)  = align_production._reseg_column
  D (detector, count-constrained)   = DetectorInfer.column_boxes(page_boxes, x_range, N)
on tall%, two_blob% (2 stacked masses = merged), and MLS (single-glyph-ness;
clean GOLD glyph ~0.68, midpoint diverged ~0.57, valley ~0.44).

Run:
  .venv/bin/python evaluation/ver_new/eval_detector_seg.py --limit 0
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from evaluation.ver_new.align_production import _detect, _reseg_column  # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3              # noqa: E402
from evaluation.ver_new.seg_valley_n_ab import metrics            # noqa: E402
from evaluation.ver_new.char_detector.detector_infer import DetectorInfer  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt", default=str(HERE / "char_detector" / "detector.pt"))
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        sys.exit(f"no detector at {args.ckpt} — download detector.best.pt first.")
    cfg = load_config(args.config); paths = cfg["paths"]
    qn = set(load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"])).keys())
    data_root = REPO / paths["data_dir"]
    enc = VisualS3(REPO, fd_dir="").enc
    det = DetectorInfer(ckpt=args.ckpt)
    print(f"  detector trained={det.trained} img={det.img} | encoder {enc.device}", flush=True)

    agg = {"A": defaultdict(float), "D": defaultdict(float)}
    nb = {"A": 0, "D": 0}
    n_div = n_det_ok = 0

    for b in cfg["books"]:
        book = b["name"]; data_dir = data_root / book
        trans = [t for t in sorted(glob.glob(str(data_dir / "transcriptions" / "page_*.json")))
                 if not t.endswith("_qn_ocr_cache.json")]
        if args.limit:
            trans = trans[: args.limit]
        for tf in trans:
            page = Path(tf).stem
            try:
                d = _detect(page, data_dir, qn)
            except Exception:
                d = None
            if not d or d[3] is None:
                continue
            cols, qn_lines, iter_pairs, binary, _ = d
            page_bgr = cv2.imread(str(data_dir / "pages" / f"{page}.png"), cv2.IMREAD_COLOR)
            if page_bgr is None:
                continue
            page_boxes = None
            for nom_idx, line_id in iter_pairs:
                cluster = cols[nom_idx]; syl = qn_lines[line_id]
                if not syl or not cluster.get("chars"):
                    continue
                N = len(syl)
                if len(cluster["chars"]) == N:
                    continue                      # only diverged columns
                n_div += 1
                if page_boxes is None:
                    page_boxes = det.boxes_for_page(page_bgr)   # detector once per page
                boxes_A = _reseg_column(cluster) or []
                xr = cluster.get("x_range")
                boxes_D = det.column_boxes(page_boxes, xr, N) if xr else []
                if len(boxes_D) == N:
                    n_det_ok += 1
                for tag, boxes in (("A", boxes_A), ("D", boxes_D)):
                    for bx in boxes:
                        m = metrics(page_bgr, bx, enc)
                        if m is None:
                            continue
                        nb[tag] += 1
                        for k in ("tall", "two_blob", "mls"):
                            agg[tag][k] += m[k]
        print(f"  [{book}] diverged {n_div}", flush=True)

    print("\n" + "=" * 64)
    print(" DETECTOR vs MIDPOINT on diverged columns (deployment metric)")
    print("=" * 64)
    print(f"  diverged columns {n_div} | detector produced exactly N in {n_det_ok}")
    print(f"  {'metric':10s} {'A midpoint':>12s} {'D detector':>12s}   verdict (ref: clean glyph MLS ~0.68)")
    res = {}
    for k in ("tall", "two_blob", "mls"):
        a = agg["A"][k] / max(nb["A"], 1); dd = agg["D"][k] / max(nb["D"], 1)
        if k == "mls":
            va, vb = f"{a:.3f}", f"{dd:.3f}"
            verdict = "D better" if dd > a + 0.005 else ("A better" if a > dd + 0.005 else "≈")
        else:
            va, vb = f"{a:.1%}", f"{dd:.1%}"
            verdict = "D better" if dd < a - 0.005 else ("A better" if a < dd - 0.005 else "≈")
        print(f"  {k:10s} {va:>12s} {vb:>12s}   {verdict}")
        res[k] = {"A": round(a, 4), "D": round(dd, 4), "verdict": verdict}
    helps = res["two_blob"]["verdict"] == "D better" and res["mls"]["verdict"] != "A better"
    print(f"\n  boxes A={nb['A']} D={nb['D']}")
    print(f"  >>> Detector {'HELPS — wire --reseg detector (cleaner diverged crops)' if helps else 'does NOT clearly beat midpoint yet — retrain (early-stop ~ep10 / --img 1024 / TKH pretrain)'}")
    out = {"diverged": n_div, "detector_exact_N": n_det_ok, "boxes": nb, "metrics": res, "detector_helps": helps,
           "ckpt_val": json.load(open(Path(args.ckpt).with_suffix(".json"))) if Path(args.ckpt).with_suffix(".json").exists() else None}
    json.dump(out, open(HERE / "results" / "eval_detector_seg.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  -> {HERE/'results'/'eval_detector_seg.json'}")


if __name__ == "__main__":
    main()
