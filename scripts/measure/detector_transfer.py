#!/usr/bin/env python
"""Phép đo: CenterNet detector (train_crop/detector_r34.best.pt) chuyển sang thạch bản.

Gộp scripts/measure_wf_2026-09-21/detector/{layout_bands,measure_detector,variants,
best_variant}.py + detector_review/verify*.py thành 1 lệnh, không đường dẫn cứng.

Chạy detector 1 lần/trang/biến thể ở thr=0.05 (decode top-k 1024) rồi lọc lại ở
thr ∈ {0.2,0.3,0.4} — đúng cách _legacy_page_boxes của align_production làm.
Biến thể tiền xử lý: raw (ảnh gốc, nền xám ~128), stretch (p2→0, p90→255), otsu.
Gán hộp vào cột theo 2 cách:
  pipeline : DetectorInfer.raw_column_boxes(x_range ± 0.25·w, NMS dọc 0.45) — đúng
             phép lọc production (DETECTOR_THR=0.2, DETECTOR_XMARGIN=0.25); với thạch
             bản lọc thêm theo dải y của tầng (2 tầng chung x).
  excl     : gán độc quyền tâm cột gần nhất, |dx| ≤ 0.6·pitch.
Kỳ vọng N: thạch bản (LVT1883/KVK1884) tầng trên = 6 (câu lục), tầng dưới = 8 (câu
bát); STT = số chữ kim-OCR trong cột (detected/*_ocr_cache.json), x_range = bao bbox kim
(đúng cluster["x_range"] của pipeline).
Phương pháp độc lập: (i) số cột: chiếu mực (layout) vs gom cụm tâm-x hộp detector;
(ii) số chữ/cột: chiếu ngang (proj_count) vs detector; (iii) STT: recall hộp kim IoU≥0.3.

Chạy:  .venv/bin/python scripts/measure/detector_transfer.py --book all --pages 27   (runner mặc định 27; --page-ids để cố định trang)
Thử:   .venv/bin/python scripts/measure/detector_transfer.py --limit 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LITHO_BOOKS = {
    "LucVanTien1883": REPO / "data/LucVanTien1883/pages",
    "KimVanKieu1884": REPO / "data/KimVanKieu1884/pages",
}
STT_BOOKS = ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]
VARIANTS = ["raw", "stretch", "otsu"]
THRS = [0.2, 0.3, 0.4]
PIPE_VARIANT, PIPE_THR = "raw", 0.2          # cấu hình production hiện hành


# ----------------------------------------------------------------------------- ảnh
def binarize(gray):
    """Nhị phân mực=1 (Otsu trên ảnh làm mờ nhẹ)."""
    import cv2
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (b > 0).astype(np.uint8)


def stretch(gray):
    """Kéo tương phản tuyến tính: mực (p2) → 0, nền (p90) → 255."""
    lo, hi = np.percentile(gray, 2), np.percentile(gray, 90)
    if hi - lo < 10:
        return gray
    return np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def otsu(gray):
    import cv2
    _, b = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def preprocess(gray, variant):
    return {"raw": lambda g: g, "stretch": stretch, "otsu": otsu}[variant](gray)


def to_bgr(g):
    import cv2
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------------- layout (litho)
def _runs(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def find_blocks(binimg, x_lo_frac=0.02, x_hi_frac=0.97):
    """2 tầng văn bản theo chiếu ngang; bỏ lề gáy & lề phải; loại nhãn số câu/số trang.
    Lề trái chỉ cắt 2 % (KVK canvas lẻ x_min ≈ 33–78 px; cắt 6 % làm mất cột 10 — đặc tả §1.4)."""
    H, W = binimg.shape
    xa, xb = int(W * x_lo_frac), int(W * x_hi_frac)
    row = binimg[:, xa:xb].sum(axis=1).astype(np.float32)
    row = np.convolve(row, np.ones(15, np.float32) / 15, mode="same")
    runs = _runs(row > 0.12 * np.percentile(row, 98))
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 45:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    merged = [r for r in merged if r[1] - r[0] > 0.08 * H]
    merged.sort(key=lambda r: r[1] - r[0], reverse=True)
    return sorted(merged[:2]), (xa, xb)


def find_columns(binimg, yr, xa, xb, n_expected=10):
    """10 dải cột trong tầng theo chiếu dọc (mượt 41px, ngưỡng 0,25·p98); pitch = tự tương quan.
    Dự phòng: n_expected đỉnh mạnh nhất cách nhau ≥ 0,7·pitch. Trả ([(x1,x2)] RTL, info)."""
    y1, y2 = yr
    col = binimg[y1:y2, xa:xb].sum(axis=0).astype(np.float32)
    sm = np.convolve(col, np.ones(41, np.float32) / 41, mode="same")
    c = sm - sm.mean()
    ac = np.correlate(c, c, mode="full")[len(c) - 1:]
    ac = ac / (ac[0] if ac[0] else 1.0)
    lo, hi = int(0.07 * len(sm)), int(0.14 * len(sm))
    pitch = int(lo + np.argmax(ac[lo:hi]))
    p98 = float(np.percentile(sm, 98))
    runs = _runs(sm > 0.25 * p98)
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 0.1 * pitch:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    cands = [r for r in merged if r[1] - r[0] > 0.3 * pitch]
    info = {"pitch": pitch, "n_runs": len(merged), "method": "runs"}
    if len(cands) != n_expected:
        peaks = sorted(((sm[i], i) for i in range(1, len(sm) - 1)
                        if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > 0.25 * p98), reverse=True)
        chosen = []
        for _, i in peaks:
            if all(abs(i - j) >= 0.7 * pitch for j in chosen):
                chosen.append(i)
            if len(chosen) == n_expected:
                break
        chosen.sort()
        cands = []
        for k, cx in enumerate(chosen):
            left = (chosen[k - 1] + cx) // 2 if k > 0 else max(0, cx - pitch // 2)
            right = (chosen[k + 1] + cx) // 2 if k + 1 < len(chosen) else min(len(sm) - 1, cx + pitch // 2)
            on = np.where(sm[left:right] > 0.25 * p98)[0]
            if len(on):
                cands.append((left + int(on[0]), left + int(on[-1]) + 1))
        info["method"] = "peaks"
    bands = sorted([(xa + r[0], xa + r[1]) for r in cands], key=lambda b: -b[0])   # RTL
    return bands, info


def columns_litho(gray):
    binimg = binarize(gray)
    blocks, (xa, xb) = find_blocks(binimg)
    cols = []
    for bi, yr in enumerate(blocks):
        bands, info = find_columns(binimg, yr, xa, xb)
        for ci, (x1, x2) in enumerate(bands):
            cols.append({"block": bi, "col": ci, "x1": int(x1), "x2": int(x2), "y1": int(yr[0]), "y2": int(yr[1]),
                         "n_exp": 6 if bi == 0 else 8, "pitch_x": info["pitch"], "layout_method": info["method"]})
    return cols, binimg, blocks


def columns_stt(cache_path):
    d = json.load(open(cache_path))
    cols = []
    for ci, chars in enumerate(d["columns"]):
        if not chars:
            continue
        bb = np.array([c["bbox"] for c in chars])
        cols.append({"block": 0, "col": ci, "x1": int(bb[:, 0].min()), "x2": int(bb[:, 2].max()),
                     "y1": int(bb[:, 1].min()), "y2": int(bb[:, 3].max()), "n_exp": len(chars), "pitch_x": None,
                     "layout_method": "kim_cache", "kim_boxes": bb.tolist()})
    return cols


def proj_count(binimg, c):
    """Đếm chữ bằng chiếu ngang trong cột (độc lập detector): run mực trên profile hàng
    (mượt 0,15·pitch_y, ngưỡng 0,15·p95), gộp khe < 0,1·pitch_y, bỏ run < 0,25·pitch_y,
    run dài (chữ dính) tính round(len/pitch_y). pitch_y = (y2−y1)/N chỉ dùng làm thang đo."""
    sub = binimg[c["y1"]:c["y2"], c["x1"]:c["x2"]]
    if sub.size == 0:
        return -1
    pitch_y = (c["y2"] - c["y1"]) / max(1, c["n_exp"])
    k = max(3, int(0.15 * pitch_y) | 1)
    row = np.convolve(sub.sum(axis=1).astype(np.float32), np.ones(k, np.float32) / k, mode="same")
    merged = []
    for r in _runs(row > 0.15 * np.percentile(row, 95)):
        if merged and r[0] - merged[-1][1] < 0.1 * pitch_y:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    return int(sum(max(1, round((r[1] - r[0]) / pitch_y)) for r in merged if r[1] - r[0] >= 0.25 * pitch_y))


def detector_column_clusters(boxes, blocks, pitch):
    """Số cột theo detector: gom cụm tâm-x hộp trong dải y của tầng, khe > 0,5·pitch = cột mới."""
    out = []
    for (y1, y2) in blocks:
        cxs = sorted((b[0] + b[2]) / 2 for b in boxes if y1 <= (b[1] + b[3]) / 2 <= y2)
        n = 0
        prev = None
        for cx in cxs:
            if prev is None or cx - prev > 0.5 * pitch:
                n += 1
            prev = cx
        out.append(n)
    return out


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ------------------------------------------------------------------------ chọn trang
def pick_even(items, n, interior=True):
    """Chọn n phần tử cách đều (np.linspace) theo thứ tự đã sắp; interior=True bỏ 2 đầu
    (trang đầu/cuối sách thường thiếu cột)."""
    items = sorted(items)
    if n >= len(items):
        return items
    if interior and len(items) > n + 2:
        idx = sorted(set(int(round(v)) for v in np.linspace(0, len(items) - 1, n + 2)[1:-1]))
    else:
        idx = sorted(set(int(round(v)) for v in np.linspace(0, len(items) - 1, n)))
    return [items[i] for i in idx]


def litho_text_pages(book):
    files = sorted(LITHO_BOOKS[book].glob("*.jpg"))
    if book == "KimVanKieu1884":                       # canvas chữ 4–166 (bìa/tựa/trang trống ngoài dải)
        files = [f for f in files if 4 <= int(f.stem.split("_")[-1]) <= 166]
    return files


def parse_page_ids(spec):
    """'LucVanTien1883:010,030;KimVanKieu1884:0020,0050' → {book: [suffix, ...]} (tái lập mẫu trang cũ)."""
    out = {}
    for part in (spec or "").split(";"):
        if ":" in part:
            b, ids = part.split(":", 1)
            out[b.strip()] = [x.strip() for x in ids.split(",") if x.strip()]
    return out


def build_page_list(book, n_pages, stt_pages, limit, page_ids=None):
    pages = []
    books = list(LITHO_BOOKS) if book == "all" else [book]
    n_lit = min(n_pages, limit) if limit else n_pages
    page_ids = page_ids or {}
    for b in books:
        if b in page_ids:                                  # --page-ids: danh sách trang cố định
            files = [f for f in litho_text_pages(b) if f.stem.split("_")[-1] in page_ids[b]]
            missing = set(page_ids[b]) - {f.stem.split("_")[-1] for f in files}
            assert not missing, f"{b}: không thấy trang {sorted(missing)}"
            pages += [(b, "litho", f, None) for f in sorted(files)]
            continue
        pages += [(b, "litho", f, None) for f in pick_even(litho_text_pages(b), n_lit)]
    n_stt = min(stt_pages, limit) if limit else stt_pages
    if n_stt > 0:
        for b in STT_BOOKS:
            d = REPO / "prepared" / b
            if not d.exists():
                continue
            imgs = sorted(p for p in (d / "pages").glob("page_*.png")
                          if (d / "detected" / f"{p.stem}_ocr_cache.json").exists())
            for f in pick_even(imgs, n_stt):
                pages.append((b, "stt", f, d / "detected" / f"{f.stem}_ocr_cache.json"))
    return pages


# ----------------------------------------------------------------------------- đo
def measure_page(det, book, kind, img_path, cache, variants):
    import cv2
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    H, W = gray.shape
    if kind == "litho":
        cols, binimg, blocks = columns_litho(gray)
    else:
        cols, binimg = columns_stt(cache), binarize(gray)
        blocks = [(min(c["y1"] for c in cols), max(c["y2"] for c in cols))] if cols else []
    for c in cols:
        c["proj_n"] = proj_count(binimg, c)
    page = {"book": book, "kind": kind, "page": img_path.name, "W": W, "H": H, "n_blocks": len(blocks),
            "n_cols": len(cols), "n_exp_total": sum(c["n_exp"] for c in cols),
            "ncols_per_block": ";".join(str(sum(1 for c in cols if c["block"] == bi)) for bi in range(len(blocks))),
            "layout_method": ";".join(sorted({c["layout_method"] for c in cols})),
            "pitch_x": cols[0]["pitch_x"] if cols and cols[0]["pitch_x"] else None,
            "cols_proj_eq": sum(1 for c in cols if c["proj_n"] == c["n_exp"])}
    col_rows, boxes_by_variant = [], {}
    for v in variants:
        g = preprocess(gray, v)
        t = time.time()
        allb = det.boxes_for_page(to_bgr(g))              # score >= 0.05
        page[f"infer_s_{v}"] = round(time.time() - t, 2)
        boxes_by_variant[v] = allb
        page[f"score_med_{v}"] = round(float(np.median([b[4] for b in allb])), 3) if allb else 0.0
        if kind == "litho" and page["pitch_x"]:
            page[f"det_cols_per_block_{v}"] = ";".join(
                str(n) for n in detector_column_clusters([b for b in allb if b[4] >= PIPE_THR], blocks, page["pitch_x"]))
        for thr in THRS:
            pb = [b for b in allb if b[4] >= thr]
            page[f"boxes_{v}_thr{thr}"] = len(pb)
            centers = [((c["x1"] + c["x2"]) / 2, c) for c in cols]
            excl = {id(c): 0 for c in cols}
            for b in pb:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                best, bd = None, 1e9
                for xc, c in centers:
                    m = 0.15 * (c["y2"] - c["y1"])
                    if kind == "litho" and not (c["y1"] - m <= cy <= c["y2"] + m):
                        continue
                    if abs(cx - xc) < bd:
                        best, bd = c, abs(cx - xc)
                if best is not None and bd <= 0.6 * (best["pitch_x"] or (best["x2"] - best["x1"]) * 1.3):
                    excl[id(best)] += 1
            eq = over = under = 0
            for c in cols:
                m = 0.15 * (c["y2"] - c["y1"])
                pbb = [b for b in pb if c["y1"] - m <= (b[1] + b[3]) / 2 <= c["y2"] + m] if kind == "litho" else pb
                raw = det.raw_column_boxes(pbb, (c["x1"], c["x2"]), 0.25)
                M = len(raw)
                eq += M == c["n_exp"]; over += M > c["n_exp"]; under += M < c["n_exp"]
                row = {"book": book, "page": img_path.name, "variant": v, "thr": thr, "block": c["block"], "col": c["col"],
                       "x1": c["x1"], "x2": c["x2"], "y1": c["y1"], "y2": c["y2"], "pitch_x": c["pitch_x"],
                       "n_exp": c["n_exp"], "proj_n": c["proj_n"], "M_pipe": M, "M_excl": excl[id(c)],
                       "w_med": None, "h_med": None, "dy_med": None, "dy_cv": None, "ink_recall": None,
                       "score_med": None, "kim_recall": None}
                if raw:
                    ws = [b[2] - b[0] for b in raw]; hs = [b[3] - b[1] for b in raw]
                    row["w_med"] = round(float(np.median(ws)), 1); row["h_med"] = round(float(np.median(hs)), 1)
                    row["score_med"] = round(float(np.median([b[4] for b in raw])), 3)
                    cys = sorted((b[1] + b[3]) / 2 for b in raw)
                    if len(cys) > 1:
                        dy = np.diff(cys)
                        row["dy_med"] = round(float(np.median(dy)), 1)
                        row["dy_cv"] = round(float(dy.std() / dy.mean()), 3) if dy.mean() > 0 else None
                    sub = binimg[c["y1"]:c["y2"], c["x1"]:c["x2"]]
                    tot = int(sub.sum())
                    if tot:
                        mask = np.zeros_like(sub)
                        for b in raw:
                            yy1 = max(0, int(b[1]) - c["y1"]); yy2 = min(sub.shape[0], int(b[3]) - c["y1"])
                            xx1 = max(0, int(b[0]) - c["x1"]); xx2 = min(sub.shape[1], int(b[2]) - c["x1"])
                            if yy2 > yy1 and xx2 > xx1:
                                mask[yy1:yy2, xx1:xx2] = 1
                        row["ink_recall"] = round(float((sub * mask).sum() / tot), 3)
                if kind == "stt":
                    kb = c["kim_boxes"]
                    row["kim_recall"] = round(sum(1 for k in kb if any(iou(k, b) >= 0.3 for b in raw)) / len(kb), 3)
                col_rows.append(row)
            page[f"eq_{v}_thr{thr}"] = eq; page[f"over_{v}_thr{thr}"] = over; page[f"under_{v}_thr{thr}"] = under
    return page, col_rows, cols, boxes_by_variant


def draw_overlay(gray, cols, boxes, thr, rows_for_page, out_path):
    import cv2
    im = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mm = {(r["block"], r["col"]): r["M_pipe"] for r in rows_for_page}
    for c in cols:
        cv2.rectangle(im, (c["x1"], c["y1"]), (c["x2"], c["y2"]), (255, 160, 0), 2)
        m = mm.get((c["block"], c["col"]), "?")
        cv2.putText(im, f"{m}/{c['n_exp']}", (c["x1"], max(20, c["y1"] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 160, 0) if m == c["n_exp"] else (0, 0, 255), 2)
    for b in boxes:
        if b[4] < thr:
            continue
        color = (0, 200, 0) if b[4] >= 0.4 else ((0, 200, 200) if b[4] >= 0.3 else (0, 120, 255))
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
    cv2.putText(im, f"thr={thr} green>=0.4 yellow>=0.3 orange>=0.2", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    h, w = im.shape[:2]
    im = cv2.resize(im, (int(w * 1400 / h), 1400))
    cv2.imwrite(str(out_path), im, [cv2.IMWRITE_JPEG_QUALITY, 80])


# ------------------------------------------------------------------------- tổng hợp
def pct(a, n):
    return round(100.0 * a / n, 1) if n else None


GEOM_KEYS = ("w_med", "h_med", "dy_med", "dy_cv", "ink_recall", "score_med", "kim_recall")


def config_stats(crs, prs, v, thr, n_cols):
    """Thống kê 1 (sách, biến thể, thr) từ dòng cột crs + dòng trang prs."""
    d = np.array([r["M_pipe"] - r["n_exp"] for r in crs]) if crs else np.array([0])
    ver = [r for r in crs if r["proj_n"] == r["n_exp"]]           # N được chiếu mực xác nhận
    cfg = {"variant": v, "thr": thr, "n_cols": n_cols,
           "pct_cols_M_eq_N": pct(int((d == 0).sum()), n_cols), "pct_cols_M_gt_N": pct(int((d > 0).sum()), n_cols),
           "pct_cols_M_lt_N": pct(int((d < 0).sum()), n_cols),
           "pct_cols_M_eq_N_excl": pct(sum(1 for r in crs if r["M_excl"] == r["n_exp"]), n_cols),
           "n_cols_N_verified": len(ver),
           "pct_cols_M_eq_N_verified": pct(sum(1 for r in ver if r["M_pipe"] == r["n_exp"]), len(ver)),
           "boxes_per_page": round(float(np.mean([p[f"boxes_{v}_thr{thr}"] for p in prs])), 1),
           "M_minus_N_hist": " ".join(f"{k}:{int((d == k).sum())}" for k in sorted(set(d.tolist())))}
    for key in GEOM_KEYS:
        vals = [r[key] for r in crs if r[key] is not None]
        cfg[key + "_med"] = round(float(np.median(vals)), 3) if vals else None
    hp = [r["h_med"] / ((r["y2"] - r["y1"]) / r["n_exp"]) for r in crs if r["h_med"] and r["n_exp"]]
    wc = [r["w_med"] / (r["x2"] - r["x1"]) for r in crs if r["w_med"] and r["x2"] > r["x1"]]
    cfg["h_over_pitch_y_med"] = round(float(np.median(hp)), 3) if hp else None
    cfg["w_over_colw_med"] = round(float(np.median(wc)), 3) if wc else None
    return cfg


def summarize(page_rows, col_rows, variants):
    """per_book (gọn, cho summary.json) + config_rows (đầy đủ, cho detector_configs.csv)."""
    books = []
    for b in page_rows:
        if b["book"] not in books:
            books.append(b["book"])
    per_book, config_rows = {}, []
    for book in books:
        prs = [p for p in page_rows if p["book"] == book]
        n_cols = sum(p["n_cols"] for p in prs)
        entry = {"kind": prs[0]["kind"], "n_pages": len(prs), "n_cols": n_cols,
                 "n_exp_chars": sum(p["n_exp_total"] for p in prs), "table": {}}
        for v in variants:
            for thr in THRS:
                crs = [r for r in col_rows if r["book"] == book and r["variant"] == v and r["thr"] == thr]
                cfg = config_stats(crs, prs, v, thr, n_cols)
                config_rows.append({"book": book, **cfg})
                entry["table"].setdefault(v, []).append(
                    f"{cfg['pct_cols_M_eq_N']}/{cfg['pct_cols_M_gt_N']}/{cfg['pct_cols_M_lt_N']}")
                entry.setdefault("_cfg", {})[f"{v}_thr{thr}"] = cfg
        c0 = entry["_cfg"][f"{PIPE_VARIANT}_thr{PIPE_THR}"]
        entry["pipeline_cfg"] = {"eq_excl": c0["pct_cols_M_eq_N_excl"], "boxes_per_page": c0["boxes_per_page"],
                                 "eq_N_verified": f"{c0['pct_cols_M_eq_N_verified']} (n={c0['n_cols_N_verified']})"}
        entry["proj_count"] = {"method": "chiếu ngang cột (run mực, không detector)",
                               "pct_cols_eq_N": pct(sum(p["cols_proj_eq"] for p in prs), n_cols)}
        if entry["kind"] == "litho":
            entry["layout"] = {"method": "chiếu mực 2 tầng × 10 cột vs gom cụm tâm-x hộp detector (thr 0.2)",
                               "pages_2_blocks": sum(1 for p in prs if p["n_blocks"] == 2),
                               "pages_10x2_cols": sum(1 for p in prs if p["ncols_per_block"] == "10;10"),
                               "pitch_x_range": [min(p["pitch_x"] for p in prs), max(p["pitch_x"] for p in prs)]}
            for v in variants:
                entry["layout"][f"pages_det_agree_{v}"] = sum(
                    1 for p in prs if p.get(f"det_cols_per_block_{v}") == p["ncols_per_block"])
        per_book[book] = entry
    return per_book, config_rows


def best_config(per_book, kind):
    best = None
    for book, e in per_book.items():
        if e["kind"] != kind:
            continue
        for k, cfg in e["_cfg"].items():
            if cfg["pct_cols_M_eq_N"] is None:
                continue
            if best is None or cfg["pct_cols_M_eq_N"] > best[1]:
                best = (k, cfg["pct_cols_M_eq_N"], book)
    return best


def build_invariants(per_book, page_rows, variants):
    inv = []

    def add(name, expected, observed, ok, method=""):
        inv.append({"name": name, "expected": expected, "observed": observed, "pass": bool(ok), "method": method})

    stt = {b: e for b, e in per_book.items() if e["kind"] == "stt"}
    lit = {b: e for b, e in per_book.items() if e["kind"] == "litho"}
    key = f"{PIPE_VARIANT}_thr{PIPE_THR}"
    if stt:
        cfgs = [e["_cfg"][key] for e in stt.values()]
        n = sum(c["n_cols"] for c in cfgs)
        eq = sum(c["pct_cols_M_eq_N"] * c["n_cols"] / 100 for c in cfgs)
        add("stt_control_pct_cols_eq_pipeline_cfg", ">= 90 (mọi cột)", round(100 * eq / n, 1),
            100 * eq / n >= 90, "thr0.2 raw ±0.25w vs N kim")
        nv = sum(c["n_cols_N_verified"] for c in cfgs)
        eqv = sum((c["pct_cols_M_eq_N_verified"] or 0) * c["n_cols_N_verified"] / 100 for c in cfgs)
        add("stt_control_pct_cols_eq_N_verified", ">= 90", f"{round(100 * eqv / nv, 1) if nv else None} (n={nv})",
            nv > 0 and 100 * eqv / nv >= 90, "cột có N kim == chiếu mực")
        kr = [c["kim_recall_med"] for c in cfgs if c["kim_recall_med"] is not None]
        if kr:
            add("stt_control_kim_box_recall_iou0.3", ">= 0.95", round(float(np.median(kr)), 3), np.median(kr) >= 0.95,
                "IoU>=0.3 kim vs detector")
        for v in variants:
            if v == "raw":
                continue
            d = sum(abs(e["_cfg"][f"{v}_thr{PIPE_THR}"]["pct_cols_M_eq_N"] - e["_cfg"][key]["pct_cols_M_eq_N"])
                    * e["n_cols"] / 100 for e in stt.values())
            lim = max(1, 0.05 * n)
            add(f"stt_control_{v}_vs_raw_delta_cols", f"<= {lim:.1f} cột (max(1, 5%))", round(d, 1), d <= lim,
                "STT nền trắng sẵn")
    for book, e in lit.items():
        prs = [p for p in page_rows if p["book"] == book]
        add(f"{book}_layout_2_blocks_x_10_cols", f"{len(prs)}/{len(prs)} trang", f"{e['layout']['pages_10x2_cols']}/{len(prs)}",
            e["layout"]["pages_10x2_cols"] == len(prs), "chiếu mực")
        cand = [v for v in ("stretch", "otsu") if v in variants] or variants
        bestv = max(cand, key=lambda v: e["_cfg"][f"{v}_thr{PIPE_THR}"]["pct_cols_M_eq_N"])
        pe = e["_cfg"][f"{bestv}_thr{PIPE_THR}"]["pct_cols_M_eq_N"]
        add(f"{book}_best_preproc_pct_cols_eq_thr0.2", ">= 70", f"{pe} ({bestv})", pe >= 70,
            "M == 6/8 theo tầng")
        pr = e["_cfg"][key]["pct_cols_M_eq_N"]
        add(f"{book}_preproc_gain_over_raw_pp", ">= 5", round(pe - pr, 1), pe - pr >= 5, "nền xám 128")
        hp = e["_cfg"][f"{bestv}_thr{PIPE_THR}"]["h_over_pitch_y_med"]
        if hp is not None:
            add(f"{book}_box_h_over_pitch_y", "0.5..1.3", hp, 0.5 <= hp <= 1.3, "hộp ~1 chữ")
        agree = e["layout"][f"pages_det_agree_{bestv}"]
        need = max(1, int(round(0.8 * len(prs))))
        add(f"{book}_ncols_agreement_layout_vs_detector", f">= {need}/{len(prs)} trang", f"{agree}/{len(prs)}",
            agree >= need, f"chiếu mực vs gom cụm tâm-x hộp ({bestv} thr0.2)")
        pj = e["proj_count"]["pct_cols_eq_N"]
        add(f"{book}_proj_count_eq_N", ">= 60 %cột", pj, pj >= 60, "chiếu ngang vs 6/8")
    mono = all(p[f"boxes_{v}_thr0.2"] >= p[f"boxes_{v}_thr0.3"] >= p[f"boxes_{v}_thr0.4"] for p in page_rows for v in variants)
    add("boxes_monotone_in_thr", "boxes@0.2>=@0.3>=@0.4", mono, mono)
    return inv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default="all", choices=["all", *LITHO_BOOKS])
    ap.add_argument("--out", default=None, help="mặc định measure_out/<book>/ (all → measure_out/detector_transfer/)")
    ap.add_argument("--pages", type=int, default=9, help="số trang thạch bản mỗi sách (chọn đều)")
    ap.add_argument("--stt-pages", type=int, default=1, help="số trang đối chứng mỗi sách STT2/4/11 (0 = tắt)")
    ap.add_argument("--limit", type=int, default=0, help="chạy thử: tối đa N trang mỗi sách")
    ap.add_argument("--workers", type=int, default=4, help="số luồng torch CPU")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--debug-pages", type=int, default=4, help="số ảnh overlay (≤5)")
    ap.add_argument("--page-ids", default="", help="trang cố định thay --pages: 'LucVanTien1883:010,030;KimVanKieu1884:0020'")
    a = ap.parse_args()
    variants = [v for v in a.variants.split(",") if v in VARIANTS]
    if PIPE_VARIANT not in variants:
        variants.insert(0, PIPE_VARIANT)
    out = Path(a.out) if a.out else REPO / "measure_out" / ("detector_transfer" if a.book == "all" else a.book)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", str(a.workers))
    import torch
    torch.manual_seed(0); torch.set_num_threads(a.workers)
    np.random.seed(0)
    from pipeline.align_engine.char_detector.detector_infer import DetectorInfer
    det = DetectorInfer(thr=0.05, device="cpu")
    assert det.trained, "không nạp được train_crop/detector_r34.best.pt"

    pages = build_page_list(a.book, a.pages, a.stt_pages, a.limit, parse_page_ids(a.page_ids))
    print(f"detector img={det.img} ckpt=train_crop/detector_r34.best.pt | {len(pages)} trang × {variants} × thr {THRS}", flush=True)
    page_rows, col_rows, overlays = [], [], 0
    t0 = time.time()
    for book, kind, img_path, cache in pages:
        page, rows, cols, boxes = measure_page(det, book, kind, img_path, cache, variants)
        page_rows.append(page); col_rows += rows
        if overlays < min(5, a.debug_pages) and not any(p["book"] == book for p in page_rows[:-1]):
            import cv2
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            bv = "stretch" if (kind == "litho" and "stretch" in variants) else "raw"
            draw_overlay(preprocess(gray, bv), cols, boxes[bv], PIPE_THR,
                         [r for r in rows if r["variant"] == bv and r["thr"] == PIPE_THR],
                         out / f"debug_{book}_{img_path.stem}_{bv}_thr{PIPE_THR}.jpg")
            overlays += 1
    per_book, config_rows = summarize(page_rows, col_rows, variants)
    inv = build_invariants(per_book, page_rows, variants)
    with open(out / "detector_configs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(config_rows[0].keys())); w.writeheader(); w.writerows(config_rows)
    keys = list(col_rows[0].keys())
    with open(out / "detector_columns.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(col_rows)
    keys = sorted({k for r in page_rows for k in r}, key=lambda k: (k not in ("book", "kind", "page"), k))
    with open(out / "detector_pages.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(page_rows)
    bl, bs = best_config(per_book, "litho"), best_config(per_book, "stt")
    geom = {}
    for b, e in per_book.items():
        ks = [f"{PIPE_VARIANT}_thr{PIPE_THR}"] + ([bl[0]] if bl and e["kind"] == "litho" and bl[0] in e["_cfg"] else [])
        geom[b] = {k: {kk: e["_cfg"][k][kk] for kk in ("w_med_med", "h_med_med", "dy_med_med", "dy_cv_med", "ink_recall_med",
                                                     "score_med_med", "h_over_pitch_y_med", "w_over_colw_med", "kim_recall_med")
                       if e["_cfg"][k][kk] is not None} for k in dict.fromkeys(ks)}
        e.pop("_cfg")
    summary = {
        "measure": "detector_transfer", "date": time.strftime("%Y-%m-%d"),
        "cli": " ".join(["scripts/measure/detector_transfer.py"] + [x for x in sys.argv[1:] if not x.startswith("/")]),
        "detector": {"ckpt": "train_crop/detector_r34.best.pt", "img": det.img, "decode_thr": 0.05, "refilter_thr": THRS,
                     "assign": "raw_column_boxes ±0.25w NMS 0.45 (pipeline) | excl nearest-centre 0.6·pitch"},
        "pages": {"n": len(pages), "per_book": {b: e["n_pages"] for b, e in per_book.items()},
                  "selection": "chọn đều theo thứ tự tên tệp (np.linspace), STT: trang có ocr_cache"},
        "variants": variants, "runtime_s": round(time.time() - t0, 1),
        "table_cols": f"per_book.table[variant] = pct_M_eq_N/gt/lt tại thr {THRS} (gán pipeline); chi tiết: detector_configs.csv",
        "per_book": per_book,
        "box_geometry": geom,
        "best": {"litho": {"config": bl[0], "pct_cols_M_eq_N": bl[1], "book": bl[2]} if bl else None,
                 "stt": {"config": bs[0], "pct_cols_M_eq_N": bs[1], "book": bs[2]} if bs else None},
        "agreement": {"ncols_layout_vs_detector": "per_book.*.layout.pages_det_agree_*",
                      "nchars_proj_vs_N": {b: e["proj_count"]["pct_cols_eq_N"] for b, e in per_book.items()},
                      "stt_kim_vs_detector": {b: geom[b].get(f"{PIPE_VARIANT}_thr{PIPE_THR}", {}).get("kim_recall_med")
                                              for b, e in per_book.items() if e["kind"] == "stt"}},
        "invariants": inv,
        "files": ["detector_columns.csv", "detector_pages.csv", "detector_configs.csv", f"debug_*.jpg ×{overlays}"],
    }
    txt = json.dumps(summary, ensure_ascii=False, indent=1)
    if len(txt.encode()) > 8000:            # quá 8 KB: viết gọn; vẫn quá thì bỏ hình học hộp (có trong detector_configs.csv)
        txt = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    if len(txt.encode()) > 8000:
        summary.pop("box_geometry"); summary["note"] = "box_geometry: xem detector_configs.csv"
        txt = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    (out / "summary.json").write_text(txt, encoding="utf-8")
    for b, e in per_book.items():
        line = f"{b:16s} n_cols={e['n_cols']:3d} | "
        for v in variants:
            line += f"{v}@0.2 {e['table'][v][0]} | "
        print(line + "(eq/gt/lt %)")
    for i in inv:
        print(f"  [{'PASS' if i['pass'] else 'FAIL'}] {i['name']}: expected {i['expected']} | observed {i['observed']}")
    print(f"saved {out}/summary.json ({len(txt.encode())} B), {len(col_rows)} dòng cột, {overlays} overlay, {summary['runtime_s']}s")


if __name__ == "__main__":
    main()
