"""A/B test on DIVERGED columns: does forcing N via char_segmenter valley
segmentation reduce merged-2-char crops vs the current OCR-center midpoints —
WITHOUT training anything?

A "diverged" column is one where the SinoNom OCR's character count != N (= the QN
syllable count we know from alignment). There the current production path
(align_production._reseg_column) builds one box per OCR char from the OCR
y-centers; if the OCR merged two glyphs into one, that box spans two characters
-> a merged crop (the failure visible in crop_audit.pdf groups 2 & 7).

Method B forces exactly N boxes by valley segmentation of the column image
(core.image.char_segmenter.segment_characters_in_column, expected_count=N), which
splits at horizontal-projection valleys. This harness cuts the crops for BOTH and
measures three no-label merged-crop proxies per box, on the SAME real diverged
columns:

  tall      box aspect h/w > 1.8        (the seg_flag='tall' merge signature)
  two_blob  the box's ink splits into >=2 vertically-separated masses (2 stacked glyphs)
  MLS       max cosine to any of the 1,591 trained classes (single-glyph-ness:
            a clean glyph scores high, a merged/garbage box scores low)

If B lowers tall%/two_blob% and raises MLS, valley-N helps on diverged columns and
is worth wiring into production (align_production, opt-in). If not (valley mis-packs
when the column window catches neighbour ink — the reason midpoints were chosen,
FLOW.md §B3), we keep midpoints. Either way it is a real, data-grounded answer.

Run:
  .venv/bin/python evaluation/ver_new/seg_valley_n_ab.py --limit 0   # 0 = all pages
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom, load_similarity_dict  # noqa: E402
from core.image.char_segmenter import segment_characters_in_column  # noqa: E402
from evaluation.ver_new.align_production import _detect, _reseg_column  # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3              # noqa: E402
from evaluation.ver_new.bbox_fix import tighten_box                # noqa: E402

HERE = Path(__file__).resolve().parent


def two_blob(gray: np.ndarray) -> bool:
    """True if the box ink forms >=2 vertically-separated masses (stacked glyphs)."""
    bw = gray < 128
    h, w = bw.shape
    if h < 16 or w < 8 or bw.sum() < 20:
        return False
    rows = bw.sum(axis=1).astype(float)
    k = max(3, h // 30) | 1
    rows = np.convolve(rows, np.ones(k) / k, mode="same")
    active = rows > max(1.0, 0.06 * w)
    min_gap = max(4, int(0.12 * h))            # a real inter-glyph gap
    runs, in_run, gap = 0, False, 0
    for a in active:
        if a:
            if not in_run:
                runs += 1; in_run = True
            gap = 0
        elif in_run:
            gap += 1
            if gap >= min_gap:
                in_run = False
    return runs >= 2


def metrics(page_bgr, box, enc):
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    H, W = page_bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    crop = page_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape
    tall = h > 1.8 * max(w, 1)
    tb = two_blob(gray)
    g = gray
    t = tighten_box(gray)
    if t is not None:
        a, c, b, d = t
        if b - a >= 8 and d - c >= 8:
            g = gray[c:d, a:b]
    mls = enc.mls(enc.embed_gray(g))
    return {"tall": int(tall), "two_blob": int(tb), "mls": float(mls if mls is not None else 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--limit", type=int, default=0, help="pages per book (0 = all)")
    args = ap.parse_args()

    cfg = load_config(args.config); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    qn_set = set(qn.keys())
    sim = load_similarity_dict(str(REPO / paths["similar_dict"]))
    data_root = REPO / paths["data_dir"]
    vs3 = VisualS3(REPO, fd_dir="")
    enc = vs3.enc

    agg = {"A": defaultdict(float), "B": defaultdict(float)}
    n_boxes = {"A": 0, "B": 0}
    n_div_cols = n_cols = 0
    n_B_count_ok = 0

    for b in cfg["books"]:
        book = b["name"]; data_dir = data_root / book
        trans = sorted(glob.glob(str(data_dir / "transcriptions" / "page_*.json")))
        trans = [t for t in trans if not t.endswith("_qn_ocr_cache.json")]
        if args.limit:
            trans = trans[: args.limit]
        for tf in trans:
            page = Path(tf).stem
            try:
                det = _detect(page, data_dir, qn_set)
            except Exception:
                det = None
            if det is None:
                continue
            cols, qn_lines, iter_pairs, binary, _ = det
            if binary is None:
                continue
            page_bgr = cv2.imread(str(data_dir / "pages" / f"{page}.png"), cv2.IMREAD_COLOR)
            if page_bgr is None:
                continue
            for nom_idx, line_id in iter_pairs:
                cluster = cols[nom_idx]; syl = qn_lines[line_id]
                if not syl or not cluster.get("chars"):
                    continue
                n_cols += 1
                N = len(syl)
                if len(cluster["chars"]) == N:
                    continue                      # not diverged
                n_div_cols += 1
                # A: current production reseg (OCR centers -> len(OCR) boxes)
                boxes_A = _reseg_column(cluster) or []
                # B: valley segmentation forced to N. Clusters carry x_range (column
                # x-span) + per-char bboxes (no 'bbox' key) -> build the column box =
                # x_range × full y-extent of the detected chars.
                chars = cluster["chars"]
                if "x_range" in cluster and cluster["x_range"]:
                    cx1, cx2 = int(cluster["x_range"][0]), int(cluster["x_range"][1])
                else:
                    cx1 = min(int(c["bbox"][0]) for c in chars)
                    cx2 = max(int(c["bbox"][2]) for c in chars)
                cy1 = min(int(c["bbox"][1]) for c in chars)
                cy2 = max(int(c["bbox"][3]) for c in chars)
                col_bbox = (cx1, cy1, cx2, cy2)
                try:
                    boxes_B = segment_characters_in_column(binary, col_bbox, expected_count=N)
                except Exception:
                    boxes_B = []
                if len(boxes_B) == N:
                    n_B_count_ok += 1
                for tag, boxes in (("A", boxes_A), ("B", boxes_B)):
                    for bx in boxes:
                        m = metrics(page_bgr, bx, enc)
                        if m is None:
                            continue
                        n_boxes[tag] += 1
                        for k in ("tall", "two_blob", "mls"):
                            agg[tag][k] += m[k]
        print(f"  [{book}] diverged cols so far {n_div_cols}", flush=True)

    print("\n" + "=" * 66)
    print(" A/B on DIVERGED columns — force-N valley vs OCR-center midpoints")
    print("=" * 66)
    print(f"  columns total {n_cols} | diverged {n_div_cols} "
          f"({n_div_cols/max(n_cols,1):.1%}) | B produced exactly N in {n_B_count_ok}/{n_div_cols} cols")
    print(f"\n  {'metric':10s} {'A (OCR-center, now)':>20s} {'B (valley force-N)':>20s}   verdict")
    res = {}
    for k, lo_is_better in (("tall", True), ("two_blob", True), ("mls", False)):
        a = agg["A"][k] / max(n_boxes["A"], 1)
        bb = agg["B"][k] / max(n_boxes["B"], 1)
        if k == "mls":
            va, vb = f"{a:.3f}", f"{bb:.3f}"
            better = "B better" if bb > a + 0.005 else ("A better" if a > bb + 0.005 else "≈")
        else:
            va, vb = f"{a:.1%}", f"{bb:.1%}"
            better = "B better" if bb < a - 0.005 else ("A better" if a < bb - 0.005 else "≈")
        print(f"  {k:10s} {va:>20s} {vb:>20s}   {better}")
        res[k] = {"A": round(a, 4), "B": round(bb, 4), "verdict": better}
    print(f"\n  boxes scored: A={n_boxes['A']}  B={n_boxes['B']}")
    helps = (res["two_blob"]["verdict"] == "B better" or res["tall"]["verdict"] == "B better") \
        and res["mls"]["verdict"] != "A better"
    print(f"\n  >>> VERDICT: valley force-N {'REDUCES merged crops on diverged cols -> worth wiring (opt-in)' if helps else 'does NOT clearly help (keep midpoints; FLOW.md §B3 mis-pack risk confirmed)'}")

    import json
    out = {"columns": n_cols, "diverged": n_div_cols, "B_count_exact_N": n_B_count_ok,
           "boxes": n_boxes, "metrics": res, "valley_helps": helps}
    p = HERE / "results" / "seg_valley_n_ab.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
