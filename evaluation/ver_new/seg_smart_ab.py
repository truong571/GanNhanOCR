"""#2 — SMART guarded valley-N: only split UNDER-counted columns, and only accept
the split when it does NOT drop glyph quality (MLS guard). Measured on real data.

The naive force-N (seg_valley_n_ab.py) traded merged crops (two_blob 60→21%) for
fragments (MLS 0.57→0.44). Two fixes, both no-train:

  (1) ONLY under-counted columns. A diverged column is either OCR UNDER-counted
      (merged 2 glyphs -> the box that needs splitting) or OVER-counted (split 1
      glyph -> the boxes that need MERGING). Valley force-N only makes sense for the
      under-counted case; on over-counted columns it should MERGE (force-N < OCR
      count). We measure both groups separately.
  (2) MLS GUARD. Per column, compute both box sets' mean MLS (single-glyph-ness)
      and ACCEPT the valley split only if it does not lower MLS (>= mid - eps). So
      by construction the guarded output never makes glyphs worse, and we report how
      much merging (two_blob) it removes for free.

Method C (guarded) = valley where the guard accepts, midpoint elsewhere. We report
A (all-midpoint) vs B (all-valley) vs C (guarded) on under- and over-counted
columns, plus the guard's accept rate.

Run:
  .venv/bin/python evaluation/ver_new/seg_smart_ab.py --limit 0
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
from core.image.char_segmenter import segment_characters_in_column  # noqa: E402
from evaluation.ver_new.align_production import _detect, _reseg_column  # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3              # noqa: E402
from evaluation.ver_new.seg_valley_n_ab import metrics            # noqa: E402

HERE = Path(__file__).resolve().parent


def col_bbox_of(cluster):
    chars = cluster["chars"]
    if cluster.get("x_range"):
        cx1, cx2 = int(cluster["x_range"][0]), int(cluster["x_range"][1])
    else:
        cx1 = min(int(c["bbox"][0]) for c in chars); cx2 = max(int(c["bbox"][2]) for c in chars)
    cy1 = min(int(c["bbox"][1]) for c in chars); cy2 = max(int(c["bbox"][3]) for c in chars)
    return (cx1, cy1, cx2, cy2)


def box_stats(page_bgr, boxes, enc):
    ms = [metrics(page_bgr, b, enc) for b in boxes]
    ms = [m for m in ms if m]
    if not ms:
        return None
    return {"two_blob": np.mean([m["two_blob"] for m in ms]),
            "mls": np.mean([m["mls"] for m in ms]), "n": len(ms)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.0, help="MLS guard slack (accept valley if mls_B >= mls_A - eps)")
    args = ap.parse_args()

    cfg = load_config(args.config); paths = cfg["paths"]
    qn = set(load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"])).keys())
    data_root = REPO / paths["data_dir"]
    enc = VisualS3(REPO, fd_dir="").enc

    # per-group accumulators of per-column means
    grp = {"under": defaultdict(list), "over": defaultdict(list)}
    guard_accept = defaultdict(int); guard_total = defaultdict(int)

    for b in cfg["books"]:
        book = b["name"]; data_dir = data_root / book
        trans = [t for t in sorted(glob.glob(str(data_dir / "transcriptions" / "page_*.json")))
                 if not t.endswith("_qn_ocr_cache.json")]
        if args.limit:
            trans = trans[: args.limit]
        for tf in trans:
            page = Path(tf).stem
            try:
                det = _detect(page, data_dir, qn)
            except Exception:
                det = None
            if not det or det[3] is None:
                continue
            cols, qn_lines, iter_pairs, binary, _ = det
            page_bgr = cv2.imread(str(data_dir / "pages" / f"{page}.png"), cv2.IMREAD_COLOR)
            if page_bgr is None:
                continue
            for nom_idx, line_id in iter_pairs:
                cluster = cols[nom_idx]; syl = qn_lines[line_id]
                if not syl or not cluster.get("chars"):
                    continue
                N = len(syl); oc = len(cluster["chars"])
                if oc == N:
                    continue
                g = "under" if oc < N else "over"
                boxes_A = _reseg_column(cluster) or []
                try:
                    boxes_B = segment_characters_in_column(binary, col_bbox_of(cluster), expected_count=N)
                except Exception:
                    boxes_B = []
                sA = box_stats(page_bgr, boxes_A, enc)
                sB = box_stats(page_bgr, boxes_B, enc)
                if not sA or not sB:
                    continue
                grp[g]["A_tb"].append(sA["two_blob"]); grp[g]["A_mls"].append(sA["mls"])
                grp[g]["B_tb"].append(sB["two_blob"]); grp[g]["B_mls"].append(sB["mls"])
                # guard: accept valley for this column iff it doesn't lower MLS
                guard_total[g] += 1
                accept = sB["mls"] >= sA["mls"] - args.eps
                if accept:
                    guard_accept[g] += 1
                grp[g]["C_tb"].append(sB["two_blob"] if accept else sA["two_blob"])
                grp[g]["C_mls"].append(sB["mls"] if accept else sA["mls"])
        print(f"  [{book}] under {guard_total['under']} over {guard_total['over']}", flush=True)

    print("\n" + "=" * 70)
    print(" #2 SMART guarded valley-N  (per-column means; A=midpoint B=valley C=guarded)")
    print("=" * 70)
    out = {}
    for g in ("under", "over"):
        d = grp[g]
        if not d["A_tb"]:
            print(f"\n  [{g}-counted] no columns."); continue
        row = {m: round(float(np.mean(d[m])), 4) for m in d}
        acc = guard_accept[g] / max(guard_total[g], 1)
        print(f"\n  [{g}-counted columns: {guard_total[g]}]   guard accepted valley in {acc:.1%}")
        print(f"    {'':8s} {'two_blob':>10s} {'MLS':>8s}")
        print(f"    {'A mid':8s} {row['A_tb']:>10.1%} {row['A_mls']:>8.3f}")
        print(f"    {'B valley':8s} {row['B_tb']:>10.1%} {row['B_mls']:>8.3f}")
        print(f"    {'C guard':8s} {row['C_tb']:>10.1%} {row['C_mls']:>8.3f}   <- valley only when it doesn't hurt MLS")
        row["guard_accept_rate"] = round(acc, 4); row["columns"] = guard_total[g]
        out[g] = row
        win = (row["C_tb"] < row["A_tb"] - 0.01) and (row["C_mls"] >= row["A_mls"] - 0.005)
        print(f"    -> guarded C {'REDUCES merging at no MLS cost (worth it)' if win else 'no free win here'}")

    json.dump(out, open(HERE / "results" / "seg_smart_ab.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n  -> {HERE/'results'/'seg_smart_ab.json'}")


if __name__ == "__main__":
    main()
