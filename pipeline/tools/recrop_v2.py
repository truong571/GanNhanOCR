"""D-1 · Cắt lại crops_v2 (F3g) TỪ labels_final.csv CÓ SẴN — không chạy lại căn chỉnh, không đụng crop giao nộp.

Đọc cột bbox (+ prev/next theo cột, pitch = trung vị khoảng cách tâm cùng cột, rơi về trang) của mọi ô có
crop (image != ''), gọi pipeline.align_engine.recenter_f3g.save_crop_v2 và ghi:
    <out>/<tier>/<cùng tên tệp>.png          crop v2 ('f3g' tái định tâm | 'fallback' = byte-trùng crop v1)
    <out>/labels_crops_v2.csv                sidecar: image, bbox_v2, crop_mode, recenter_shift, image_md5_v2,
                                             ink_pct_v2, crop_quality_flag_v2, stray_ink_v2, border_ink_v2, …
    <out>/recrop_v2_summary.json             số liệu tái lập (tham số, thời gian, tỉ lệ f3g/fallback, md5 fallback == v1)
Ô QĐ-01 khoá (qd01_locked=1) dùng prev/next_bbox_cu của config/qd01_cells.csv như PASS 2 (N7).

    .venv/bin/python -m pipeline.tools.recrop_v2 --labels dataset_out/labels_final.csv --out dataset_out/khoi_d/crops_v2
    .venv/bin/python -m pipeline.tools.recrop_v2 --limit 300 --seed 0 --out /tmp/x      # thử nhanh
Tiền đề đã kiểm (2026-09-16): bbox + prev/next cột tái lập image_md5 giao nộp 1907/1907 ô trên 12 trang.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipeline.align_engine import recenter_f3g as rf   # noqa: E402

BOOKDIR = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "stt11": "SachThanhTruyen11"}
FIELDS = ["image", "image_v2", "book", "page", "column", "nom_idx", "syl_idx", "tier", "tier_v3", "qd01_locked",
          "bbox", "bbox_v2", "bbox_v2_win", "crop_mode", "guard_reason", "prior_used", "recenter_shift", "pitch",
          "image_md5", "image_md5_v2", "ink_pct_v2", "crop_w_v2", "crop_h_v2",
          "crop_quality_flag_v2", "stray_ink_v2", "border_ink_v2"]


def _parse_bbox(s):
    if not s:
        return None
    try:
        b = json.loads(s)
        return [int(v) for v in b] if b and len(b) == 4 else None
    except Exception:
        return None


def column_context(rows: list[dict], cells: dict) -> list[tuple[dict, list | None, list | None, float | None]]:
    """rows = mọi dòng của MỘT TRANG (mọi tier, có bbox). Trả [(row, prev_bbox, next_bbox, pitch)] cho
    các dòng có crop (image != ''). prev/next = bản ghi kề cùng cột theo y (như build_dataset PASS 2);
    ô QĐ-01 khoá -> prev/next_bbox_cu. pitch cột rơi về trung vị trang."""
    by_col: dict[str, list] = defaultdict(list)
    for r in rows:
        b = _parse_bbox(r.get("bbox", ""))
        if b:
            r["_bb"] = b
            by_col[r["column"]].append(r)
    col_cys = {}
    for col, rs in by_col.items():
        rs.sort(key=lambda r: (r["_bb"][1] + r["_bb"][3]) / 2.0)
        col_cys[col] = [(r["_bb"][1] + r["_bb"][3]) / 2.0 for r in rs]
    pp = rf.page_pitch(col_cys)
    out = []
    for col, rs in by_col.items():
        pitch = rf.column_pitch(col_cys[col], fallback=pp)
        for i, r in enumerate(rs):
            if not r.get("image"):
                continue
            pv = rs[i - 1]["_bb"] if i > 0 else None
            nx = rs[i + 1]["_bb"] if i < len(rs) - 1 else None
            if r.get("qd01_locked") == "1":
                c = cells.get((r["book"], r["page"], r["column"], r["nom_idx"]))
                if c is not None:
                    pv, nx = _parse_bbox(c.get("prev_bbox_cu")), _parse_bbox(c.get("next_bbox_cu"))
            out.append((r, pv, nx, pitch))
    return out


def process_page(args) -> tuple[list[dict], float, int]:
    """Một trang: cắt v2 mọi ô có crop. Trả (rows sidecar, giây, số ô)."""
    book, page, rows, cells, out_dir, pad, write, pages_dir, guard = args
    import cv2
    png = Path(pages_dir) / BOOKDIR[book] / "pages" / f"{page}.png"
    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    res = []
    if img is None:
        for r in rows:
            if r.get("image"):
                res.append({"image": r["image"], "crop_mode": "", "guard_reason": "no_page_image"})
        return res, 0.0, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.time()
    n = 0
    for r, pv, nx, pitch in column_context(rows, cells):
        path = (Path(out_dir) / r["image"]) if write else None
        q = rf.save_crop_v2(img, gray, r["_bb"], pv, nx, pitch, pad, path, guard=guard)
        n += 1
        row = {k: r.get(k, "") for k in ("image", "book", "page", "column", "nom_idx", "syl_idx", "tier",
                                          "tier_v3", "qd01_locked", "bbox", "image_md5")}
        if q is None:
            row.update(crop_mode="", guard_reason="cut_fail")
            res.append(row)
            continue
        row.update(image_v2=r["image"] if write else "", bbox_v2=json.dumps(q["bbox_v2"]),
                   bbox_v2_win=json.dumps(q["bbox_v2_win"]), crop_mode=q["crop_mode"],
                   guard_reason=q["meta"].get("reason", ""), prior_used=int(bool(q["meta"].get("prior"))),
                   recenter_shift=q["recenter_shift"], pitch=q["pitch"], image_md5_v2=q["md5"],
                   ink_pct_v2=q["ink"], crop_w_v2=q["w"], crop_h_v2=q["h"],
                   crop_quality_flag_v2=q["crop_quality_flag"], stray_ink_v2=q["stray_ink"],
                   border_ink_v2=q["border_ink"])
        res.append(row)
    return res, time.time() - t0, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(REPO / "dataset_out/labels_final.csv"))
    ap.add_argument("--out", default=str(REPO / "dataset_out/khoi_d/crops_v2"))
    ap.add_argument("--qd01", default=str(REPO / "config/qd01_cells.csv"))
    ap.add_argument("--pages-dir", default=str(REPO / "prepared"))
    ap.add_argument("--pad", type=float, default=rf.X_PAD, help="pad ngang theo w (= step2.crop_pad_frac 0,12)")
    ap.add_argument("--guard", type=float, default=rf.GUARD_SHIFT,
                    help="bảo hiểm |cy − tâm| > guard·pitch -> fallback (mặc định 0,40; hướng C đo 0,35 chặn 86/99 ca nhầm)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ N trang (rút ngẫu nhiên theo --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--no-write", action="store_true", help="chỉ đo, không ghi PNG (vẫn ghi sidecar)")
    args = ap.parse_args(argv)

    t_all = time.time()
    lab = Path(args.labels)
    raw = lab.read_bytes()
    labels_md5 = hashlib.md5(raw).hexdigest()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    need = {"image", "book", "page", "column", "nom_idx", "bbox", "tier", "image_md5"}
    miss = need - set(rows[0].keys())
    assert not miss, f"labels thiếu cột {miss}"
    cells = {}
    if Path(args.qd01).exists():
        for c in csv.DictReader(open(args.qd01, encoding="utf-8")):
            cells[(c["book"], c["page"], c["column"], c["nom_idx"])] = c
    by_page: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by_page[(r["book"], r["page"])].append(r)
    pages = sorted(by_page)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        pages = [pages[i] for i in sorted(rng.choice(len(pages), min(args.limit, len(pages)), replace=False))]
    n_cells = sum(1 for p in pages for r in by_page[p] if r.get("image"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"labels {lab} md5={labels_md5} · {len(rows):,} dòng · {len(pages)} trang · {n_cells:,} ô có crop "
          f"· workers {args.workers} · {'KHÔNG ghi PNG' if args.no_write else 'ghi ' + str(out)}", flush=True)

    jobs = [(b, p, by_page[(b, p)], cells, str(out), args.pad, not args.no_write, args.pages_dir, args.guard)
            for b, p in pages]
    results, secs, ncut = [], 0.0, 0
    if args.workers > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            for k, (res, dt, n) in enumerate(ex.map(process_page, jobs, chunksize=4)):
                results.extend(res); secs += dt; ncut += n
                if (k + 1) % 50 == 0:
                    print(f"  {k+1}/{len(jobs)} trang · {ncut:,} ô · {secs/max(1,ncut)*1000:.1f} ms/ô (CPU 1 lõi)", flush=True)
    else:
        for k, j in enumerate(jobs):
            res, dt, n = process_page(j)
            results.extend(res); secs += dt; ncut += n
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(jobs)} trang · {ncut:,} ô · {secs/max(1,ncut)*1000:.1f} ms/ô", flush=True)

    results.sort(key=lambda r: r["image"])
    with open(out / "labels_crops_v2.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    modes = defaultdict(int)
    for r in results:
        modes[r.get("crop_mode") or "fail"] += 1
    fb = [r for r in results if r.get("crop_mode") == "fallback"]
    fb_eq = sum(1 for r in fb if r.get("image_md5_v2") == r.get("image_md5"))
    f3 = [r for r in results if r.get("crop_mode") == "f3g"]
    f3_eq = sum(1 for r in f3 if r.get("image_md5_v2") == r.get("image_md5"))
    sh = np.array([abs(float(r["recenter_shift"])) for r in results if r.get("recenter_shift") not in ("", None)])
    reasons = defaultdict(int)
    for r in fb:
        reasons[r.get("guard_reason", "")] += 1
    summary = {
        "labels": str(lab), "labels_md5": labels_md5, "n_rows": len(rows), "n_pages": len(pages),
        "n_cells": n_cells, "n_cut": ncut, "out": str(out), "write_png": not args.no_write,
        "params": {k: getattr(rf, k) for k in ("X_PAD", "H_RANGE", "H_TARGET", "TOP_SEARCH", "LAMBDA_H", "LAMBDA_C",
                                                "GUARD_SHIFT", "EXT", "SEAM_BAND", "NEIGHBOR_OK", "SMOOTH_PX",
                                                "MIN_PX", "PROF_WIN")},
        "seed": args.seed, "limit": args.limit, "guard": args.guard, "pad": args.pad,
        "crop_mode": dict(modes), "fallback_reason": dict(reasons),
        "pct_f3g": round(100.0 * modes.get("f3g", 0) / max(1, ncut), 2),
        "fallback_md5_eq_v1": [fb_eq, len(fb)], "f3g_md5_eq_v1": [f3_eq, len(f3)],
        "abs_shift_pitch": {"p50": float(np.percentile(sh, 50)) if len(sh) else None,
                            "p90": float(np.percentile(sh, 90)) if len(sh) else None,
                            "gt_0.15": float((sh > 0.15).mean()) if len(sh) else None,
                            "gt_0.25": float((sh > 0.25).mean()) if len(sh) else None,
                            "gt_0.40": float((sh > 0.40).mean()) if len(sh) else None},
        "ms_per_crop_1core": round(secs / max(1, ncut) * 1000, 2), "wall_s": round(time.time() - t_all, 1),
    }
    (out / "recrop_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"xong: {ncut:,} ô · f3g {modes.get('f3g',0):,} ({summary['pct_f3g']}%) · fallback {len(fb):,} "
          f"(md5 == v1: {fb_eq}/{len(fb)}) · |shift| p50 {summary['abs_shift_pitch']['p50']:.3f}p "
          f"p90 {summary['abs_shift_pitch']['p90']:.3f}p · {summary['ms_per_crop_1core']} ms/ô · "
          f"tường {summary['wall_s']}s -> {out / 'labels_crops_v2.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
