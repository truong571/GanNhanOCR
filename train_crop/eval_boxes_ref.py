"""Đánh giá detector (ckpt CenterNet) trên Ô THAM CHIẾU held-out của manifest nhãn yếu.

Đầu vào: manifest JSON của build_lithograph_manifest.py (val/test), ckpt (v1 hoặc v2).
Với mỗi trang: chạy detector 1 lần (decode thr 0,05, top-k 1024), rồi ở mỗi thr:
  • ghép hộp ↔ ô tham chiếu 1-1 theo IoU ≥ 0,3 (tham lam) → miss / extra / ok (IoU ≥ 0,5) /
    shifted; sai số tâm |dy| (px, % bước tầng); IoU trung vị;
  • thạch bản: mỗi TẦNG (manifest `tiers` [x1,y0,x2,y1,N,start]) đếm hộp có tâm trong
    tầng ± 0,35 bước → % tầng n == N (proxy I5 theo tầng, không ép N nên không tautology);
  • "cắt vào thân chữ" = mực ở 2 hàng mép trên/dưới của hộp thô > 0,20 (crop_quality
    BORDER_INK_MAX) — chỉ số KHÔNG phụ thuộc ô tham chiếu;
  • STT: cùng ghép hộp ↔ hộp kim/pipeline (P/R/F1 @IoU 0,5 như train_centernet.validate).
Vùng `ignore_boxes` (tầng không verified / REVIEW) không tính: hộp detector rơi vào đó bị bỏ.

Chạy:
  .venv/bin/python train_crop/eval_boxes_ref.py --ckpt train_crop/detector_r34.best.pt \\
      --manifest train_crop/data_lithograph/manifest_val.json --thr 0.15,0.2 [--limit 10]
So v1 ↔ v2: chạy 2 lần với 2 ckpt, cùng manifest; đọc JSON stdout / --out.
Không có GT người: ô tham chiếu = kim + chiếu mực → mọi số là proxy (docs/HUONG_DAN_HUAN_LUYEN_I5).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

IOU_MATCH, IOU_OK, BORDER_MAX, Y_MARGIN = 0.3, 0.5, 0.20, 0.35


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(cells, boxes):
    cand = sorted(((iou(c, b), ci, bi) for ci, c in enumerate(cells) for bi, b in enumerate(boxes)
                   if iou(c, b) >= IOU_MATCH), reverse=True)
    uc, ub, pairs = set(), set(), []
    for v, ci, bi in cand:
        if ci in uc or bi in ub:
            continue
        uc.add(ci); ub.add(bi); pairs.append((ci, bi, v))
    return pairs, [ci for ci in range(len(cells)) if ci not in uc], [bi for bi in range(len(boxes)) if bi not in ub]


def _in_ignore(b, ignore):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return any(g[0] <= cx <= g[2] and g[1] <= cy <= g[3] for g in ignore)


def border_ink(binimg, b):
    x1, y1, x2, y2 = [int(v) for v in b[:4]]
    H, W = binimg.shape
    x1, x2 = max(0, x1), min(W, x2); y1, y2 = max(0, y1), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 4:
        return 0.0
    sub = binimg[y1:y2, x1:x2]
    return float(max(sub[:2].mean(), sub[-2:].mean()))


def evaluate(det, items, thrs, limit=0, verbose=False):
    """det: infer_centernet.CenterNetDetector (thr 0,05). → {domain: {thr: stats}}."""
    import cv2
    acc = {}
    t0 = time.time()
    for it in (items[:limit] if limit else items):
        img = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binimg = (bw > 0).astype(np.uint8)
        allb = det.boxes_for_image(img)
        ignore = it.get("ignore_boxes") or []
        allb = [b for b in allb if not _in_ignore(b, ignore)]
        cells = it["boxes"]
        dom = it.get("domain", "litho")
        for thr in thrs:
            boxes = [b for b in allb if b[4] >= thr]
            pairs, miss, extra = match(cells, boxes)
            a = acc.setdefault(dom, {}).setdefault(thr, {"n_cells": 0, "miss": 0, "extra": 0, "ok": 0, "shifted": 0,
                                                          "dy": [], "iou": [], "cut": 0, "n_boxes": 0,
                                                          "tiers": 0, "tiers_eq": 0, "pages": 0})
            a["pages"] += 1; a["n_cells"] += len(cells); a["miss"] += len(miss); a["extra"] += len(extra)
            a["n_boxes"] += len(boxes)
            for ci, bi, v in pairs:
                c, b = cells[ci], boxes[bi]
                a["ok"] += int(v >= IOU_OK); a["shifted"] += int(v < IOU_OK); a["iou"].append(v)
                h = max(1.0, c[3] - c[1])
                a["dy"].append(abs((b[1] + b[3]) / 2 - (c[1] + c[3]) / 2) / h)
            a["cut"] += sum(int(border_ink(binimg, b) > BORDER_MAX) for b in boxes)
            for tr in it.get("tiers") or []:
                x1, y0, x2, y1, n = tr[:5]
                p = (y1 - y0) / max(1, n)
                m = 0.05 * (x2 - x1)
                k = sum(1 for b in boxes if x1 - m <= (b[0] + b[2]) / 2 <= x2 + m
                        and y0 - Y_MARGIN * p <= (b[1] + b[3]) / 2 <= y1 + Y_MARGIN * p)
                a["tiers"] += 1; a["tiers_eq"] += int(k == n)
        if verbose:
            print(f"  {it.get('book')}/{it.get('page')}: {len(allb)} hộp", flush=True)
    out = {}
    for dom, per in acc.items():
        out[dom] = {}
        for thr, a in per.items():
            n = a["n_cells"]; nb = a["n_boxes"]
            tp = n - a["miss"]
            P = tp / nb if nb else 0.0; R = tp / n if n else 0.0
            out[dom][str(thr)] = {
                "pages": a["pages"], "n_cells": n, "n_boxes": nb,
                "miss_pct": round(100 * a["miss"] / n, 2) if n else None,
                "extra_per_100": round(100 * a["extra"] / n, 2) if n else None,
                "ok_iou50_pct": round(100 * a["ok"] / n, 2) if n else None,
                "shifted_pct": round(100 * a["shifted"] / n, 2) if n else None,
                "iou_med": round(float(np.median(a["iou"])), 3) if a["iou"] else None,
                "abs_dy_pct_pitch_med": round(100 * float(np.median(a["dy"])), 1) if a["dy"] else None,
                "abs_dy_pct_pitch_p90": round(100 * float(np.percentile(a["dy"], 90)), 1) if a["dy"] else None,
                "cut_glyph_pct": round(100 * a["cut"] / nb, 2) if nb else None,
                "tiers_n_eq_N_pct": round(100 * a["tiers_eq"] / a["tiers"], 1) if a["tiers"] else None,
                "P": round(P, 4), "R": round(R, 4), "F1": round(2 * P * R / (P + R), 4) if P + R else 0.0}
    out["_runtime_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "detector_r34.best.pt"))
    ap.add_argument("--manifest", default=str(HERE / "data_lithograph" / "manifest_val.json"))
    ap.add_argument("--thr", default="0.15,0.2")
    ap.add_argument("--img", type=int, default=0, help="0 = theo ckpt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--domain", default="all", choices=["all", "litho", "stt"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.workers)
    from infer_centernet import CenterNetDetector
    det = CenterNetDetector(a.ckpt, img=a.img or 1024, thr=0.05, device="cpu")
    if a.img:
        det.img = a.img
    assert det.trained, f"không nạp được {a.ckpt}"
    items = json.load(open(a.manifest, encoding="utf-8"))
    if a.domain != "all":
        items = [it for it in items if it.get("domain", "litho") == a.domain]
    thrs = [float(x) for x in a.thr.split(",") if x]
    res = evaluate(det, items, thrs, limit=a.limit)
    res["_ckpt"] = a.ckpt; res["_manifest"] = a.manifest; res["_img"] = det.img
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    for dom in ("litho", "stt"):
        for thr, s in (res.get(dom) or {}).items():
            print(f"{dom}@{thr}: pages {s['pages']} cells {s['n_cells']} | miss {s['miss_pct']}% extra/100 {s['extra_per_100']} "
                  f"ok50 {s['ok_iou50_pct']}% IoU {s['iou_med']} |dy| {s['abs_dy_pct_pitch_med']}%p cut {s['cut_glyph_pct']}% "
                  f"| tiers n==N {s['tiers_n_eq_N_pct']}% | F1 {s['F1']}")
    print(f"runtime {res['_runtime_s']}s" + (f" -> {a.out}" if a.out else ""))


if __name__ == "__main__":
    main()
