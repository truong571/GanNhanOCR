"""Đánh giá TRUNG THỰC detector trên manifest held-out (bản sao logic train_crop/eval_boxes_ref.py).

Mỗi trang: detector chạy 1 lần (thr 0,05, top-k 1024); hộp rơi vào ignore_boxes bị bỏ; rồi ở mỗi ngưỡng:
  • ghép hộp ↔ ô tham chiếu 1-1 theo IoU ≥ 0,3 (tham lam theo IoU giảm dần) → ok (IoU ≥ 0,5) / shifted /
    miss / extra; |dy| tâm theo % bước tầng; IoU trung vị;
  • % TẦNG n == N: đếm hộp THÔ có tâm trong tầng ± 0,35 bước (không ép N → không tautology) — proxy I5;
  • "cắt vào thân chữ": mực ở 2 hàng mép trên/dưới của hộp thô > 0,20 (crop_quality.BORDER_INK_MAX) —
    chỉ số KHÔNG phụ thuộc ô tham chiếu (ở độ phân giải bundle: 2 hàng ≈ 4 hàng ảnh gốc);
  • STT: P/R/F1 @IoU 0,5 (hồi quy so v1).
Trả {domain: {thr: {...}}}. Không có GT người → mọi số là proxy (docs/HUONG_DAN_HUAN_LUYEN_I5 §4).
"""
from __future__ import annotations

import time

import numpy as np

IOU_MATCH, IOU_OK, BORDER_MAX, Y_MARGIN = 0.3, 0.5, 0.20, 0.35


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (N,4) b (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ua = area_a[:, None] + area_b[None, :] - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0).astype(np.float32)


def match(cells: np.ndarray, boxes: np.ndarray):
    """Ghép 1-1 tham lam theo IoU giảm dần (IoU ≥ 0,3). -> pairs [(ci, bi, iou)], miss idx, extra idx."""
    M = iou_matrix(cells, boxes)
    cand = np.argwhere(M >= IOU_MATCH)
    order = np.argsort(-M[cand[:, 0], cand[:, 1]], kind="stable") if len(cand) else []
    uc, ub, pairs = set(), set(), []
    for k in order:
        ci, bi = int(cand[k, 0]), int(cand[k, 1])
        if ci in uc or bi in ub:
            continue
        uc.add(ci); ub.add(bi); pairs.append((ci, bi, float(M[ci, bi])))
    miss = [ci for ci in range(len(cells)) if ci not in uc]
    extra = [bi for bi in range(len(boxes)) if bi not in ub]
    return pairs, miss, extra


def in_ignore(boxes, ignore) -> np.ndarray:
    """mask True = tâm hộp nằm trong 1 vùng ignore."""
    if len(boxes) == 0 or not ignore:
        return np.zeros(len(boxes), bool)
    g = np.array(ignore, np.float32).reshape(-1, 4)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2; cy = (boxes[:, 1] + boxes[:, 3]) / 2
    inside = ((cx[:, None] >= g[None, :, 0]) & (cx[:, None] <= g[None, :, 2])
              & (cy[:, None] >= g[None, :, 1]) & (cy[:, None] <= g[None, :, 3]))
    return inside.any(axis=1)


def border_ink(binimg: np.ndarray, b) -> float:
    x1, y1, x2, y2 = [int(v) for v in b[:4]]
    H, W = binimg.shape
    x1, x2 = max(0, x1), min(W, x2); y1, y2 = max(0, y1), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 4:
        return 0.0
    sub = binimg[y1:y2, x1:x2]
    return float(max(sub[:2].mean(), sub[-2:].mean()))


def binarize(img_bgr) -> np.ndarray:
    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (bw > 0).astype(np.uint8)


def _new_acc():
    return {"n_cells": 0, "miss": 0, "extra": 0, "ok": 0, "shifted": 0, "dy": [], "iou": [], "cut": 0,
            "n_boxes": 0, "tiers": 0, "tiers_eq": 0, "tiers_minus": 0, "tiers_plus": 0, "pages": 0}


def evaluate(det, items, thrs=(0.15, 0.2), limit: int = 0, verbose: bool = False, boxes_cache: dict | None = None):
    """det: có boxes_for_image(bgr) -> [(x1,y1,x2,y2,score)] px ảnh manifest. -> {domain: {str(thr): stats}}."""
    import cv2
    acc: dict = {}
    t0 = time.time()
    for it in (items[:limit] if limit else items):
        img = cv2.imread(it["image_abs"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        key = it["image_abs"]
        if boxes_cache is not None and key in boxes_cache:
            allb = boxes_cache[key]
        else:
            allb = np.array(det.boxes_for_image(img), np.float32).reshape(-1, 5)
            if boxes_cache is not None:
                boxes_cache[key] = allb
        binimg = binarize(img)
        allb = allb[~in_ignore(allb, it.get("ignore_boxes") or [])]
        cells = np.array(it.get("boxes") or [], np.float32).reshape(-1, 4)
        dom = it.get("domain", "litho")
        for thr in thrs:
            boxes = allb[allb[:, 4] >= thr]
            a = acc.setdefault(dom, {}).setdefault(float(thr), _new_acc())
            a["pages"] += 1; a["n_cells"] += len(cells); a["n_boxes"] += len(boxes)
            if len(cells):
                pairs, miss, extra = match(cells, boxes[:, :4])
                a["miss"] += len(miss); a["extra"] += len(extra)
                for ci, bi, v in pairs:
                    c, b = cells[ci], boxes[bi]
                    a["ok"] += int(v >= IOU_OK); a["shifted"] += int(v < IOU_OK); a["iou"].append(v)
                    h = max(1.0, float(c[3] - c[1]))
                    a["dy"].append(abs((b[1] + b[3]) / 2 - (c[1] + c[3]) / 2) / h)
            a["cut"] += sum(int(border_ink(binimg, b) > BORDER_MAX) for b in boxes)
            if len(boxes):
                bcx = (boxes[:, 0] + boxes[:, 2]) / 2; bcy = (boxes[:, 1] + boxes[:, 3]) / 2
            for tr in it.get("tiers") or []:
                x1, y0, x2, y1, n = tr[:5]
                p = (y1 - y0) / max(1, n); m = 0.05 * (x2 - x1)
                k = 0 if not len(boxes) else int(((bcx >= x1 - m) & (bcx <= x2 + m)
                                                   & (bcy >= y0 - Y_MARGIN * p) & (bcy <= y1 + Y_MARGIN * p)).sum())
                a["tiers"] += 1; a["tiers_eq"] += int(k == n)
                a["tiers_minus"] += int(k < n); a["tiers_plus"] += int(k > n)
        if verbose:
            print(f"  {it.get('book')}/{it.get('page')}: {len(allb)} hộp", flush=True)
    out: dict = {}
    for dom, per in acc.items():
        out[dom] = {}
        for thr, a in per.items():
            n = a["n_cells"]; nb = a["n_boxes"]; tp = n - a["miss"]
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
                "tiers": a["tiers"],
                "tiers_n_eq_N_pct": round(100 * a["tiers_eq"] / a["tiers"], 1) if a["tiers"] else None,
                "tiers_minus_pct": round(100 * a["tiers_minus"] / a["tiers"], 1) if a["tiers"] else None,
                "tiers_plus_pct": round(100 * a["tiers_plus"] / a["tiers"], 1) if a["tiers"] else None,
                "P": round(P, 4), "R": round(R, 4), "F1": round(2 * P * R / (P + R), 4) if P + R else 0.0}
    out["_runtime_s"] = round(time.time() - t0, 1)
    return out


def flatten(res: dict, thr_main: float = 0.15, thr_alt: float = 0.2) -> dict:
    """Bảng 1 hàng cho metrics.csv / chọn best."""
    L = (res.get("litho") or {}).get(str(thr_main), {}); L2 = (res.get("litho") or {}).get(str(thr_alt), {})
    S = (res.get("stt") or {}).get(str(thr_alt), {}); S1 = (res.get("stt") or {}).get(str(thr_main), {})
    C = (res.get("chresto") or {}).get(str(thr_main), {}); C2 = (res.get("chresto") or {}).get(str(thr_alt), {})
    return {"litho_ok50": L.get("ok_iou50_pct"), "litho_miss": L.get("miss_pct"), "litho_extra": L.get("extra_per_100"),
            "litho_dy_med": L.get("abs_dy_pct_pitch_med"), "litho_dy_p90": L.get("abs_dy_pct_pitch_p90"),
            "litho_iou_med": L.get("iou_med"),
            "litho_tiers_eq_015": L.get("tiers_n_eq_N_pct"), "litho_tiers_eq_020": L2.get("tiers_n_eq_N_pct"),
            "litho_tiers_minus_015": L.get("tiers_minus_pct"), "litho_tiers_plus_015": L.get("tiers_plus_pct"),
            "litho_cut": L.get("cut_glyph_pct"), "litho_ok50_020": L2.get("ok_iou50_pct"),
            "stt_F1_020": S.get("F1"), "stt_P_020": S.get("P"), "stt_R_020": S.get("R"),
            "stt_F1_015": S1.get("F1"), "stt_cut_020": S.get("cut_glyph_pct"),
            "chresto_tiers_eq_015": C.get("tiers_n_eq_N_pct"), "chresto_tiers_eq_020": C2.get("tiers_n_eq_N_pct"),
            "chresto_ok50": C.get("ok_iou50_pct"), "chresto_cut": C.get("cut_glyph_pct"),
            "eval_s": res.get("_runtime_s")}


def selftest():
    cells = np.array([[0, 0, 10, 10], [0, 12, 10, 22], [0, 24, 10, 34]], np.float32)
    boxes = np.array([[0, 0, 10, 10], [0, 13, 10, 23], [0, 50, 10, 60]], np.float32)
    pairs, miss, extra = match(cells, boxes)
    assert len(pairs) == 2 and miss == [2] and extra == [2], (pairs, miss, extra)
    assert in_ignore(boxes, [[0, 45, 10, 65]]).tolist() == [False, False, True]

    class _Det:
        def boxes_for_image(self, img):
            return [(0, 0, 10, 10, 0.9), (0, 13, 10, 23, 0.18), (0, 50, 10, 60, 0.5)]
    import cv2, tempfile, os
    tmp = tempfile.mkdtemp()
    img = np.full((80, 40, 3), 255, np.uint8); cv2.rectangle(img, (2, 2), (8, 8), (0, 0, 0), -1)
    p = os.path.join(tmp, "a.png"); cv2.imwrite(p, img)
    it = {"image_abs": p, "domain": "litho", "boxes": cells.tolist(), "ignore_boxes": [[0, 45, 10, 65]],
          "tiers": [[0, 0, 10, 34, 3, 0]]}
    r = evaluate(_Det(), [it], thrs=(0.15, 0.2))
    L = r["litho"]["0.15"]; L2 = r["litho"]["0.2"]
    assert L["n_boxes"] == 2 and L["miss_pct"] == round(100 / 3, 2) and L["tiers_n_eq_N_pct"] == 0.0
    assert L2["n_boxes"] == 1 and L["ok_iou50_pct"] == round(200 / 3, 2)
    f = flatten(r); assert f["litho_ok50"] == L["ok_iou50_pct"] and f["litho_tiers_eq_020"] == 0.0
    print("i5v2.evalref selftest OK")


if __name__ == "__main__":
    selftest()
