#!/usr/bin/env python
"""PHÉP ĐO — Bố cục trang thạch bản (LucVanTien1883, KimVanKieu1884).

Gộp thuật toán của scripts/measure_wf_2026-09-21/{lvt1883_layout,kvk1884_layout}:
  * tách 2 tầng (chiếu ngang + pitch tự tương quan), 10 cột/tầng (đỉnh chiếu dọc),
    ghép cột trên+dưới thành cặp lục bát, đếm cụm mực/cột (kỳ vọng 6 trên / 8 dưới);
  * số câu in nhỏ trên đỉnh tầng: tesseract (ensemble 8 lần) + k-NN mẫu chữ số học từ
    chính sách (nhãn = số tesseract đọc "hợp lệ": bội 5, đúng chẵn/lẻ tầng, đủ chữ số)
    → agreement hai phương pháp; số cột: 2 phương pháp chiếu độc lập (đỉnh vs chuỗi run);
  * chuỗi số câu toàn sách; KVK: canvas→page = 167 − canvas, phân loại trang trắng/bìa,
    số trang góc (kỳ vọng 169 − canvas), liệt kê canvas lệch giả thuyết 20·(166−k)+…;
  * tuỳ chọn --detector: CenterNet (train_crop/detector_r34.best.pt, ảnh kéo nền) đếm
    chữ/cột làm phương pháp thứ 2 cho số chữ.
Chỉ đọc data/; đầu ra: summary.json, layout_pages.csv, layout_columns.csv, numbers.csv,
verse_chain.csv, debug/*.jpg (≤ 5 trang). Idempotent (seed 0, thứ tự tệp cố định).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2  # noqa: E402
from scipy.ndimage import gaussian_filter1d, uniform_filter1d  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SZ = 20  # kích thước ô chữ số khi DIGIT_FEATURE=square (mặc định: 14×20 làm mờ)

# ----------------------------------------------------------------------------- cấu hình sách
BOOKS = {
    "LucVanTien1883": dict(
        pages=REPO / "data/LucVanTien1883/pages", pattern="lucvantien_manuscript_*.jpg",
        unit="page", total_verses=2088, expected_pairs=1044, per_page=20,
        text_range=None,  # mọi trang đều là trang chữ
        title_cols_first_page=1,  # trang 1: cột 1 (phải nhất) là tựa
        first_page_verses=18,
        number_read_min=0.9, number_read_note="in rõ; wf 21/09: union tess/template 397/417",
    ),
    "KimVanKieu1884": dict(
        pages=REPO / "data/KimVanKieu1884/pages", pattern="canvas_*.jpg",
        unit="canvas", total_verses=3256, expected_pairs=1628, per_page=20,
        text_range=(4, 166),  # canvas 1-3, 167-171 = bìa/trắng/đề từ
        page_of_canvas=lambda k: 167 - k,
        page_number_hyp=lambda k: 169 - k,  # số trang in góc
        title_cols_first_page=0, first_page_verses=20,
        number_read_min=0.6, number_read_note="chữ số in mảnh, vỡ nét, 3↔8 lẫn; thứ tự trang kiểm bằng giải mã theo trang + Viterbi",
    ),
}


def unit_id(name: str) -> int:
    stem = Path(name).stem
    return int("".join(ch for ch in stem.split("_")[-1] if ch.isdigit()))


def expected_numbers(book: dict, page: int):
    """{số câu bội 5: (tier, cột từ phải)} + (first, last) kỳ vọng cho trang đọc thứ `page`."""
    tv, pp = book["total_verses"], book["per_page"]
    if page == 1:
        first, last, offset = 1, book["first_page_verses"], book["title_cols_first_page"]
    else:
        first = book["first_page_verses"] + 1 + pp * (page - 2)
        last, offset = min(first + pp - 1, tv), 0
    out = {}
    first_couplet = (first + 1) // 2
    for n in range(first, last + 1):
        if n % 5 == 0:
            out[n] = (0 if n % 2 == 1 else 1, (n + 1) // 2 - first_couplet + 1 + offset)
    return out, first, last, offset


# ----------------------------------------------------------------------------- tiện ích 1-D
def runs_above(v, thr):
    m = v > thr
    if not m.any():
        return []
    d = np.diff(m.astype(int), prepend=0, append=0)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def merge_runs(runs, max_gap):
    out = []
    for s, e in runs:
        if out and s - out[-1][1] <= max_gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(r) for r in out]


def autocorr_pitch(v, lo, hi):
    v = v - v.mean()
    if np.abs(v).sum() == 0:
        return None, 0.0
    n = len(v)
    f = np.fft.rfft(v, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (ac[0] + 1e-9)
    seg = ac[lo:hi]
    if len(seg) == 0:
        return None, 0.0
    k = int(np.argmax(seg)) + lo
    # ưu tiên chu kỳ cơ bản: đỉnh cục bộ nhỏ nhất có ac ≥ 0.75·đỉnh cao nhất (tránh hài 2×)
    pk, _ = find_peaks(seg)
    good = [int(q) + lo for q in pk if seg[q] >= 0.75 * ac[k]]
    if good and good[0] < k:
        k = good[0]
    return k, float(ac[k])


def refine_runs(runs, proj, min_frac=0.45, max_frac=1.6):
    """Phương pháp B (kvk_layout.py): tách run quá rộng tại cực tiểu, bỏ run quá hẹp."""
    if not runs:
        return runs
    out = list(runs)
    changed = True
    while changed:
        changed = False
        med = float(np.median([e - s + 1 for s, e in out]))
        new = []
        for s, e in out:
            w = e - s + 1
            if w > max_frac * med and w > 60:
                lo, hi = s + int(.3 * w), s + int(.7 * w)
                cut = lo + int(np.argmin(proj[lo:hi + 1]))
                new += [(s, cut), (cut + 1, e)]
                changed = True
            else:
                new.append((s, e))
        out = new
    med = float(np.median([e - s + 1 for s, e in out]))
    return [(s, e) for s, e in out if e - s + 1 >= min_frac * med]


# ----------------------------------------------------------------------------- tesseract
def tesseract_digits(img, psm=7):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        name = f.name
    try:
        r = subprocess.run(["tesseract", name, "stdout", "--psm", str(psm),
                            "-c", "tessedit_char_whitelist=0123456789", "-l", "eng"],
                           capture_output=True, text=True, timeout=60)
        return "".join(ch for ch in r.stdout if ch.isdigit())
    except Exception:
        return ""
    finally:
        os.unlink(name)


def read_number(gray_crop, total):
    """Ensemble 8 lần đọc (scale 4/6 × bin/gray/blur/psm13), bỏ phiếu; hoà → ưu tiên bội 5."""
    reads = []
    for scale in (4, 6):
        up = cv2.resize(gray_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        up = cv2.copyMakeBorder(up, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=int(np.percentile(up, 95)))
        _, ub = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(up, (0, 0), scale * 0.25)
        reads += [tesseract_digits(ub, 8), tesseract_digits(up, 8), tesseract_digits(blur, 8), tesseract_digits(ub, 13)]
    cands = [r for r in reads if r and len(r) <= 4]
    val = None
    if cands:
        cnt = collections.Counter(cands).most_common()
        top = [c for c, k in cnt if k == cnt[0][1]]
        if len(top) == 1:
            val = int(top[0])
        else:
            m5 = [c for c in top if int(c) % 5 == 0 and 1 <= int(c) <= total]
            val = int(m5[0]) if m5 else int(top[0])
    return val, reads


# ----------------------------------------------------------------------------- chữ số → vector
def clean_small(binimg, min_area=6):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binimg, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab].astype(np.uint8)


def digit_feature(tight):
    """Đặc trưng ô chữ số: kéo về 14×20 (giữ như kvk digits_decode) → phóng 3×, làm mờ σ=1.5, thu về 14×20.
    DIGIT_FEATURE=square: đệm vuông rồi 20×20 (lvt digits_template) — để so sánh."""
    if os.environ.get("DIGIT_FEATURE") == "square":
        h, w = tight.shape
        side = max(h, w) + 4
        canvas = np.zeros((side, side), np.uint8)
        oy, ox = (side - h) // 2, (side - w) // 2
        canvas[oy:oy + h, ox:ox + w] = tight
        img = cv2.resize(canvas * 255, (SZ, SZ), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return img.ravel().tolist()
    g = cv2.resize(tight.astype(np.float32), (14, 20), interpolation=cv2.INTER_AREA)
    up = cv2.resize(g, (42, 60), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 1.5)
    return cv2.resize(up, (14, 20), interpolation=cv2.INTER_AREA).ravel().tolist()


def segment_digits(b, x0, y0, x1, y1):
    """Tách ô chữ số theo khe trắng chiếu cột (digits_template.py); trả [{x,w,h,vec}]."""
    m = 4
    crop = clean_small((b[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m] > 0).astype(np.uint8))
    vp = crop.sum(0)
    xs = np.where(vp > 0)[0]
    if len(xs) == 0:
        return []
    runs, s, prev = [], xs[0], xs[0]
    for x in xs[1:]:
        if x - prev > 1:
            runs.append((s, prev + 1))
            s = x
        prev = x
    runs.append((s, prev + 1))
    merged = [list(runs[0])]
    for a, bb in runs[1:]:
        if a - merged[-1][1] <= 3 and bb - merged[-1][0] <= 22:
            merged[-1][1] = bb
        else:
            merged.append([a, bb])
    # ô quá rộng so với chiều cao (2 chữ số dính nhau, vd "61", "10") → chia đều theo w/(0.65·h)
    split = []
    for a, bb in merged:
        sub = crop[:, a:bb]
        ys = np.where(sub.sum(1) > 0)[0]
        if len(ys) == 0:
            continue
        h, w = ys[-1] - ys[0] + 1, bb - a
        k = int(round(w / max(6.0, 0.65 * h)))
        if k >= 2 and w >= 1.2 * h:
            split += [(a + int(round(i * w / k)), a + int(round((i + 1) * w / k))) for i in range(k)]
        else:
            split.append((a, bb))
    cells = []
    for a, bb in split:
        sub = crop[:, a:bb]
        ys = np.where(sub.sum(1) > 0)[0]
        if len(ys) == 0:
            continue
        h, w = ys[-1] - ys[0] + 1, bb - a
        if w < 3 or h < 12:
            continue
        tight = sub[ys[0]:ys[-1] + 1]
        cells.append({"x": int(x0 - m + a), "w": int(w), "h": int(h), "vec": digit_feature(tight)})
    return cells


def detect_number_groups(binimg, y_top, row_pitch, xmin, xmax):
    """Nhóm chữ số trong dải phía trên đỉnh tầng: closing 5×5 nối nét vỡ (digits_template.py)."""
    sa, sb = max(0, int(y_top - 0.9 * row_pitch)), max(0, int(y_top - 6))
    if sb - sa < 10 or xmax - xmin < 10:
        return []
    strip = clean_small((binimg[sa:sb, xmin:xmax] > 0).astype(np.uint8))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(strip, cv2.MORPH_CLOSE, ker)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if h < 10 or h > 0.35 * row_pitch or a < 20 or w > 0.6 * row_pitch:
            continue
        boxes.append([x, y, x + w, y + h])
    boxes.sort()
    groups = []
    for bx in boxes:
        if groups and bx[0] - groups[-1][2] < 30 and abs(bx[1] - groups[-1][1]) < 25:
            g = groups[-1]
            g[2] = max(g[2], bx[2]); g[1] = min(g[1], bx[1]); g[3] = max(g[3], bx[3])
        else:
            groups.append(list(bx))
    return [(gx0 + xmin, gy0 + sa, gx1 + xmin, gy1 + sa) for gx0, gy0, gx1, gy1 in groups if gx1 - gx0 >= 6]


def corner_number_groups(binimg, H, W):
    """Số trang góc trên trái/phải (page_numbers.py): trả [(x0,y0,x1,y1,side)]."""
    y1 = int(.09 * H)
    out = []
    for xa, xb, side in ((8, int(.30 * W), "L"), (int(.70 * W), W - 8, "R")):
        reg = clean_small((binimg[0:y1, xa:xb] > 0).astype(np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(reg, connectivity=8)
        comps = [[x, y, x + w - 1, y + h - 1] for i, (x, y, w, h, a) in enumerate(st)
                 if i > 0 and a >= 4 and 6 <= h <= 32 and w <= 40 and x > 5 and y > 5]
        comps.sort()
        clusters = []
        for c in comps:
            for cl in clusters:
                if c[0] - cl[2] <= 16 and c[1] <= cl[3] + 6 and c[3] >= cl[1] - 6:
                    cl[2] = max(cl[2], c[2]); cl[0] = min(cl[0], c[0]); cl[1] = min(cl[1], c[1]); cl[3] = max(cl[3], c[3])
                    break
            else:
                clusters.append(list(c))
        for cl in clusters:
            w, h = cl[2] - cl[0] + 1, cl[3] - cl[1] + 1
            if 10 <= h <= 32 and 6 <= w <= 70 and reg[cl[1]:cl[3] + 1, cl[0]:cl[2] + 1].sum() >= 40:
                out.append((cl[0] + xa, cl[1], cl[2] + xa + 1, cl[3] + 1, side))
    return out


# ----------------------------------------------------------------------------- phân tích 1 trang
def analyse_page(path: str, uid: int, book_name: str, want_debug: bool):
    book = BOOKS[book_name]
    im = cv2.imread(path, 0)
    H, W = im.shape
    th, b = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bg = float(np.median(im))
    ink = b > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < 6
    small[0] = False
    ink[small[lab]] = False
    ink_u8 = ink.astype(np.uint8)
    # mặt nạ "mềm" cho chữ số in mảnh/mờ (kvk_layout.py: bg − 20), chỉ dùng trong dải số câu / số trang
    soft = clean_small((im < max(th, bg - 20)).astype(np.uint8))
    rec = {"uid": uid, "file": os.path.basename(path), "W": W, "H": H, "otsu": float(th), "bg_median": bg,
           "ink_density": float(ink.mean()),
           "ink_density_center": float(ink[int(.05 * H):int(.95 * H), int(.05 * W):int(.95 * W)].mean())}

    # --- 1. TẦNG (chiếu ngang) — bỏ 2% mép trái (gáy scan)
    x_lo, x_hi = int(.02 * W), int(.98 * W)
    hp = ink[:, x_lo:x_hi].sum(1).astype(float)
    hps = uniform_filter1d(hp, 9)
    row_pitch, row_ac = autocorr_pitch(hps, 100, 260)
    if row_pitch is None:
        row_pitch = 160
    thr_low = max(3.0, 0.01 * hps.max()) if hps.max() > 0 else 1e9
    raw = merge_runs(runs_above(hps, thr_low), 8)
    full = [(a, c) for a, c in raw if (c - a) >= 0.4 * row_pitch and hps[a:c].mean() >= 0.12 * hps.max()]
    groups = []
    for a, c in full:
        if groups and a - groups[-1][1] <= 0.75 * row_pitch:
            groups[-1][1] = c; groups[-1][2] += 1
        else:
            groups.append([a, c, 1])
    big = [(a, c, k) for a, c, k in groups if (c - a) >= 2.5 * row_pitch]
    rec["row_pitch"] = int(row_pitch); rec["row_ac"] = round(row_ac, 3)
    rec["n_tiers_raw"] = len(big)
    if len(big) > 2:
        big = sorted(sorted(big, key=lambda r: r[1] - r[0], reverse=True)[:2])
    thr_high = 0.12 * hps.max()
    tiers = []
    for a, c, k in big:
        idx = np.where(hps[a:c] >= thr_high)[0]
        a2, b2 = (max(a, a + idx[0] - 15), min(c, a + idx[-1] + 16)) if len(idx) else (a, c)
        # hàng SỐ CÂU in mảnh dính vào đỉnh tầng (KVK: khe chỉ ~1 px mực): cắt tại thung lũng
        # trong 0.6 pitch đầu nếu phần trên thung lũng mỏng (< 0.2 max)
        top = hps[a2:int(a2 + 0.6 * row_pitch)]
        if len(top) > 10:
            valleys = np.where(top < 0.05 * hps.max())[0]
            valleys = valleys[valleys > 5]
            if len(valleys) and top[:valleys[0]].max() < 0.2 * hps.max():
                a2 = a2 + int(valleys[0])
        tiers.append((int(a2), int(b2)))
    rec["tiers"] = tiers
    rec["n_tiers"] = len(tiers)
    if len(tiers) == 2:
        rec["tier_gap"] = int(tiers[1][0] - tiers[0][1])
        rec["tier0_rows_est"] = round((tiers[0][1] - tiers[0][0]) / row_pitch, 2)
        rec["tier1_rows_est"] = round((tiers[1][1] - tiers[1][0]) / row_pitch, 2)

    # --- 2. CỘT mỗi tầng: phương pháp A (đỉnh chiếu, lvt1883) + B (run ngưỡng, kvk1884)
    cols_all, tier_info = [], []
    for ti, (y0, y1) in enumerate(tiers):
        band = ink[y0:y1]
        vp = band.sum(0).astype(float)
        vp[:x_lo] = 0; vp[x_hi:] = 0
        col_pitch, col_ac = autocorr_pitch(uniform_filter1d(vp, 5), 100, 300)
        if col_pitch is None:
            col_pitch = 160
        vps = gaussian_filter1d(vp, 0.18 * col_pitch)
        pad = int(col_pitch)  # đệm 0 hai đầu để cột sát mép ảnh vẫn thành đỉnh
        peaks, _ = find_peaks(np.pad(vps, pad), distance=int(0.6 * col_pitch), prominence=0.12 * vps.max())
        peaks = sorted(int(q) - pad for q in peaks)
        peaks = [min(max(q, 0), W - 1) for q in peaks]
        vpl = uniform_filter1d(vp, 7)
        bounds = []
        for i, p in enumerate(peaks):
            x0 = max(0, int(p - 0.55 * col_pitch)) if i == 0 else int(peaks[i - 1] + np.argmin(vpl[peaks[i - 1]:p]))
            x1 = min(W, int(p + 0.55 * col_pitch)) if i == len(peaks) - 1 else int(p + np.argmin(vpl[p:peaks[i + 1]]))
            seg = vpl[x0:x1]
            nz = np.where(seg > max(1.0, 0.03 * vpl.max()))[0]
            if len(nz):
                x0, x1 = x0 + nz[0], x0 + nz[-1] + 1
            bounds.append((int(x0), int(x1), int(p)))
        bounds = [bq for bq in bounds if (bq[1] - bq[0]) >= 0.3 * col_pitch]
        # k cột dính (1 đỉnh, bề rộng ≈ k·pitch) → chia k phần đều, tinh chỉnh mỗi vết cắt về cực tiểu
        # của chiếu trong ±0.15 pitch (bố cục thạch bản có pitch rất đều)
        runs_a = []
        for x0, x1, _ in bounds:
            w = x1 - x0
            k = int(round(w / col_pitch))
            if k >= 2 and w > 1.5 * col_pitch:
                cuts = [x0]
                for i in range(1, k):
                    c0 = x0 + int(round(i * w / k))
                    lo, hi = max(x0 + 1, int(c0 - 0.15 * col_pitch)), min(x1 - 1, int(c0 + 0.15 * col_pitch))
                    cuts.append(lo + int(np.argmin(vpl[lo:hi])) if hi > lo else c0)
                cuts.append(x1)
                runs_a += [(cuts[i], cuts[i + 1]) for i in range(k)]
            else:
                runs_a.append((x0, x1))
        bounds = [(int(x0), int(x1), int((x0 + x1) // 2)) for x0, x1 in runs_a if x1 - x0 >= 0.3 * col_pitch]
        bounds = sorted(bounds, key=lambda t: -t[2])  # PHẢI → TRÁI
        merged_b = []  # 2 đỉnh của cùng 1 chữ rộng (tâm cách < 0.6 pitch, gộp lại vẫn ≤ 1.25 pitch) → gộp
        for bq in bounds:
            if merged_b and abs(merged_b[-1][2] - bq[2]) < 0.6 * col_pitch \
                    and max(merged_b[-1][1], bq[1]) - min(merged_b[-1][0], bq[0]) <= 1.25 * col_pitch:
                p0 = merged_b[-1]
                x0m, x1m = min(p0[0], bq[0]), max(p0[1], bq[1])
                merged_b[-1] = (x0m, x1m, (x0m + x1m) // 2)
            else:
                merged_b.append(bq)
        bounds = merged_b
        # phương pháp B
        cthr = max(20.0, 0.3 * np.percentile(vp[x_lo:x_hi], 75))
        rb = [(s, e - 1) for s, e in merge_runs(runs_above(vp, cthr), 12) if e - s >= 40]
        rb = refine_runs(rb, vp)
        tier_info.append({"tier": ti, "y0": int(y0), "y1": int(y1), "col_pitch": int(col_pitch),
                          "col_ac": round(col_ac, 3), "n_cols": len(bounds), "n_cols_B": len(rb)})
        for ci, (x0, x1, xc) in enumerate(bounds):
            cols_all.append((ti, ci + 1, x0, x1, xc))
    rec["tier_info"] = tier_info
    pitch_ref = int(np.median([t["col_pitch"] for t in tier_info])) if tier_info else 160
    n_by_tier = [t["n_cols"] for t in tier_info]
    if len(n_by_tier) == 2 and n_by_tier[0] == n_by_tier[1]:
        # 2 tầng cùng số cột → ghép theo THỨ HẠNG từ phải (pitch đều, tránh lệch tâm do cột sáng/tối)
        slot_of = {c[4]: c[1] for c in cols_all}
        n_slots = n_by_tier[0]
    else:
        centres = sorted({c[4] for c in cols_all}, reverse=True)
        slots = []
        for xc in centres:
            if slots and abs(slots[-1][-1] - xc) <= 0.5 * pitch_ref:
                slots[-1].append(xc)
            else:
                slots.append([xc])
        slot_of = {xc: si + 1 for si, grp in enumerate(slots) for xc in grp}
        n_slots = len(slots)
    rec["n_slots"] = n_slots
    rec["col_pitch_ref"] = pitch_ref

    # --- 3. ĐẾM CHỮ mỗi cột (cụm mực theo chiếu ngang cột)
    col_recs = []
    for (ti, ci, x0, x1, xc) in cols_all:
        y0, y1 = tiers[ti]
        ya, yb = max(0, int(y0 - 0.05 * row_pitch)), min(H, int(y1 + 0.08 * row_pitch))
        crop = ink[ya:yb, x0:x1]
        hpc = uniform_filter1d(crop.sum(1).astype(float), 3)
        thr_c = max(3.0, 0.05 * hpc.max()) if hpc.max() > 0 else 1e9
        rr = merge_runs(runs_above(hpc, thr_c), 10)
        rr = [(a, c) for a, c in rr if (c - a) >= 8 and crop[a:c].sum() >= 25]
        changed = True
        while changed and len(rr) > 1:
            changed, best = False, None
            for i in range(len(rr) - 1):
                gap, tot = rr[i + 1][0] - rr[i][1], rr[i + 1][1] - rr[i][0]
                if gap <= 0.3 * row_pitch and tot <= 0.8 * row_pitch and (best is None or gap < best[1]):
                    best = (i, gap)
            if best is not None:
                i = best[0]
                rr[i:i + 2] = [(rr[i][0], rr[i + 1][1])]
                changed = True
        n_runs_raw = len(rr)
        out_rr, stack = [], list(rr)
        while stack:
            a, c = stack.pop(0)
            if (c - a) > 1.3 * row_pitch:
                lo, hi = int(a + 0.3 * row_pitch), int(c - 0.3 * row_pitch)
                if hi - lo > 5:
                    k = lo + int(np.argmin(hpc[lo:hi]))
                    stack = [(a, k), (k, c)] + stack
                    continue
            out_rr.append((a, c))
        rr = sorted(out_rr)
        heights = [c - a for a, c in rr]
        expect = 6 if ti == 0 else 8
        col_recs.append({"uid": uid, "tier": ti, "col": ci, "slot": slot_of.get(xc, -1), "x0": int(x0), "x1": int(x1),
                         "xc": int(xc), "y0": int(ya), "y1": int(yb), "n_chars": len(rr), "n_runs_raw": n_runs_raw,
                         "expect": expect, "match": int(len(rr) == expect),
                         "char_h_med": int(np.median(heights)) if heights else 0,
                         "first_top": int(ya + rr[0][0]) if rr else -1, "ink": int(crop.sum()),
                         "char_boxes": [(int(ya + a), int(ya + c)) for a, c in rr]})

    # --- 4. SỐ CÂU trên đỉnh tầng: nhóm chữ số → tesseract + ô chữ số (k-NN sau)
    num_recs = []
    for ti, (y0, y1) in enumerate(tiers):
        tier_cols = [c for c in col_recs if c["tier"] == ti]
        if not tier_cols:
            continue
        xmin = max(0, min(c["x0"] for c in tier_cols) - 60)
        xmax = min(W, max(c["x1"] for c in tier_cols) + 60)
        for (gx0, gy0, gx1, gy1) in detect_number_groups(soft, y0, row_pitch, xmin, xmax):
            m = 8
            gcrop = im[max(0, gy0 - m):gy1 + m, max(0, gx0 - m):gx1 + m]
            val, reads = read_number(gcrop, book["total_verses"])
            xc = (gx0 + gx1) / 2
            nearest = min(tier_cols, key=lambda c: abs(c["xc"] - xc))
            slot_idx = nearest["slot"] if abs(nearest["xc"] - xc) < 0.75 * pitch_ref else -1
            num_recs.append({"uid": uid, "tier": ti, "x0": int(gx0), "y0": int(gy0), "x1": int(gx1), "y1": int(gy1),
                             "tess": val, "reads": "|".join(reads), "slot": slot_idx,
                             "cells": segment_digits(soft, gx0, gy0, gx1, gy1)})
    rec["numbers"] = num_recs

    # --- 4b. SỐ TRANG in góc (KVK)
    corner = []
    if "page_number_hyp" in book:
        for (cx0, cy0, cx1, cy1, side) in corner_number_groups(soft, H, W):
            m = 6
            gcrop = im[max(0, cy0 - m):cy1 + m, max(0, cx0 - m):cx1 + m]
            val, reads = read_number(gcrop, 999)
            corner.append({"side": side, "x0": int(cx0), "y0": int(cy0), "x1": int(cx1), "y1": int(cy1), "tess": val,
                           "reads": "|".join(reads),
                           "cells": segment_digits(soft, cx0, cy0, cx1, cy1)})
    rec["corner_numbers"] = corner
    rec["cols"] = col_recs

    # --- 5. Skew từ đỉnh chữ đầu mỗi cột
    for ti in range(len(tiers)):
        tc = [c for c in col_recs if c["tier"] == ti and c["first_top"] > 0]
        if len(tc) >= 3:
            xs = np.array([c["xc"] for c in tc]); ys = np.array([c["first_top"] for c in tc])
            rec[f"tier{ti}_skew_deg"] = round(math.degrees(math.atan(np.polyfit(xs, ys, 1)[0])), 2)
    return rec


# ----------------------------------------------------------------------------- k-NN chữ số
def _norm(A):
    A = A - A.mean(1, keepdims=True)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-6)


def fit_templates(Xtr, ytr):
    return np.stack([Xtr[ytr == k].mean(0) if (ytr == k).any() else np.zeros(Xtr.shape[1]) for k in range(10)])


def predict_digits(tpl, Xte, Xtr, ytr, k=3):
    sim = _norm(Xte) @ _norm(tpl).T
    p_tpl = sim.argmax(1)
    s2 = _norm(Xte) @ _norm(Xtr).T
    kk = min(k, len(Xtr))
    idx = np.argsort(-s2, axis=1)[:, :kk]
    p_knn = np.array([np.bincount(ytr[i], minlength=10).argmax() for i in idx])
    return p_tpl, p_knn, sim.max(1)


def plausible_value(v, tier, total):
    """Số câu hợp lệ: bội 5, tầng trên ≡ 5 (mod 10) (câu lẻ), tầng dưới ≡ 0 (mod 10)."""
    return v is not None and 1 <= v <= total and v % 5 == 0 and ((v % 10 == 5) == (tier == 0))


def _collect_labels(items, accept):
    X, y, pg = [], [], []
    for uid, nr in items:
        v = accept(nr)
        if v is not None and len(nr["cells"]) == len(str(v)):
            for c, d in zip(nr["cells"], str(v)):
                X.append(c["vec"]); y.append(int(d)); pg.append(uid)
    return np.array(X, dtype=np.float32), np.array(y), np.array(pg)


def _cv_by_page(X, y, pg):
    pages_u = np.unique(pg)
    rng = np.random.default_rng(0)
    rng.shuffle(pages_u)
    nf = min(5, len(pages_u))
    acc_t = acc_k = ntot = 0
    for f in np.array_split(pages_u, nf):
        te = np.isin(pg, f); tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        pt, pk, _ = predict_digits(fit_templates(X[tr], y[tr]), X[te], X[tr], y[tr])
        acc_t += int((pt == y[te]).sum()); acc_k += int((pk == y[te]).sum()); ntot += int(te.sum())
    return nf, ntot, (round(acc_k / ntot, 4) if ntot else None), (round(acc_t / ntot, 4) if ntot else None)


def train_digit_knn(items, plausible, min_unanimous=6):
    """k-NN chữ số tự huấn luyện 2 vòng, KHÔNG dùng giả thuyết trang:
    vòng 1: nhãn = số tesseract hợp lệ (plausible) và ≥ 6/8 lần đọc trùng;
    vòng 2: nhãn = số mà k-NN vòng 1 đọc TRÙNG tesseract (và hợp lệ) → nhãn sạch hơn, nhiều hơn.
    items: [(uid, {cells, tess, reads, ...})]."""
    info = {"label_source": f"round1: tesseract plausible & ≥{min_unanimous}/8 reads unanimous; "
                            "round2: knn(round1) == tesseract & plausible; no page hypothesis used"}

    def acc1(nr):
        v = nr["tess"]
        if plausible(v, nr) and sum(x == str(v) for x in nr["reads"].split("|")) >= min_unanimous:
            return v
        return None
    X, y, pg = _collect_labels(items, acc1)
    info["round1_labelled_digits"] = int(len(y))
    if len(y) < 20 or len(np.unique(y)) < 5:
        info["status"] = "insufficient labels (need >=20 digits, >=5 classes); k-NN skipped"
        info["labelled_digits"] = int(len(y))
        return None, info
    m1 = {"tpl": fit_templates(X, y), "X": X, "y": y}

    def acc2(nr):
        v = nr["tess"]
        if not plausible(v, nr) or not nr["cells"]:
            return None
        kv = knn_read(m1, nr["cells"])[0]
        return v if kv == v else None
    X2, y2, pg2 = _collect_labels(items, acc2)
    if len(y2) >= len(y):
        X, y, pg = X2, y2, pg2
        info["round2_used"] = True
    else:
        info["round2_used"] = False
    info["labelled_digits"] = int(len(y))
    info["class_counts"] = np.bincount(y, minlength=10).tolist()
    nf, ntot, acc_k, acc_t = _cv_by_page(X, y, pg)
    info.update(cv_folds_by_page=nf, cv_n=ntot, knn3_acc=acc_k, mean_template_acc=acc_t, status="ok")
    return {"tpl": fit_templates(X, y), "X": X, "y": y}, info


def knn_proba(model, cells, k=10):
    """Xác suất 10 lớp / ô chữ số: k láng giềng cosine gần nhất, trọng số exp((sim−1)/0.05), sàn 3%."""
    if model is None or not cells:
        return None
    V = _norm(np.array([c["vec"] for c in cells], dtype=np.float32))
    s2 = V @ _norm(model["X"]).T
    kk = min(k, s2.shape[1])
    idx = np.argsort(-s2, axis=1)[:, :kk]
    P = np.zeros((len(cells), 10))
    for i in range(len(cells)):
        w = np.exp((s2[i, idx[i]] - 1.0) / 0.05)
        for j, wj in zip(idx[i], w):
            P[i, model["y"][j]] += wj
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
    return (0.97 * P + 0.003).tolist()


def score_value(probs, val):
    """log P(ô chữ số | val); thiếu ô → căn lề phải (mất nét mảnh đầu số), thừa ô → giữ các ô đầu;
    phạt log 0.15 mỗi chữ số lệch (digits_decode.py, đã kiểm 163/163 canvas)."""
    ds = [int(c) for c in str(val)]
    n, m = len(probs), len(ds)
    pen = 0.0
    if n == m:
        pairs = zip(probs, ds)
    elif n < m:
        pen = math.log(0.15) * (m - n); pairs = zip(probs, ds[m - n:])
    else:
        pen = math.log(0.15) * (n - m); pairs = zip(probs[:m], ds)
    return pen + sum(math.log(p[d]) for p, d in pairs)


def viterbi_path(E):
    """Viterbi trên chuỗi trang: chuyển ±1 (0.485 mỗi chiều, KHÔNG cố định chiều), đứng yên 0.01,
    nhảy khác chia đều 0.02 (sách đóng, quét liên tiếp). E: (n, J) log-emission (0 = không có bằng chứng)."""
    n, J = E.shape
    T = np.full((J, J), math.log(0.02 / max(1, J - 3)))
    for j in range(J):
        if j > 0:
            T[j, j - 1] = math.log(0.485)
        if j + 1 < J:
            T[j, j + 1] = math.log(0.485)
        T[j, j] = math.log(0.01)
    V = np.zeros((n, J)); B = np.zeros((n, J), int)
    V[0] = E[0]
    for t in range(1, n):
        cand = V[t - 1][:, None] + T
        B[t] = cand.argmax(0); V[t] = cand.max(0) + E[t]
    path = np.zeros(n, int); path[-1] = int(V[-1].argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = B[t, path[t]]
    return path


def decode_page_numbers(text_recs, n_pages):
    """Số trang in góc: đọc tự do (argmax) + Viterbi theo thứ tự tệp; ứng viên 1..n_pages+8."""
    J = n_pages + 8
    E = np.zeros((len(text_recs), J))
    n_has = 0
    for i, r in enumerate(text_recs):
        pr = r.get("corner_proba")
        r["page_no_free"] = None
        if not pr:
            continue
        n_has += 1
        S = np.array([score_value(pr, j + 1) for j in range(J)])
        E[i] = S
        r["page_no_free"] = int(S.argmax()) + 1
    if n_has == 0:
        for r in text_recs:
            r["page_no_seq"] = None
        return
    for r, j in zip(text_recs, viterbi_path(E)):
        r["page_no_seq"] = int(j) + 1


def decode_sequence(text_recs, book, total):
    """Giải mã số câu đầu trang THEO TRANG (điểm = Σ log P(ô | giá trị kỳ vọng tại (tầng, cột)) trên mọi
    trang giả định j) rồi Viterbi theo thứ tự tệp với chuyển trạng thái ±1 đối xứng (không cố định chiều),
    KHÔNG dùng số hiệu canvas/trang (digits_decode.py + finalize.py)."""
    firsts, p = [], 1
    while True:
        _, f, l, off = expected_numbers(book, p)
        firsts.append((f, l, off))
        if l >= total or p > 10000:
            break
        p += 1
    J = len(firsts)
    E = np.zeros((len(text_recs), J))
    has = np.zeros(len(text_recs), bool)
    for i, r in enumerate(text_recs):
        # nhóm 1 ô = mảnh vỡ của số in mờ (số câu < 10 chỉ có 1 lần/sách) → không dùng làm bằng chứng
        groups = [nr for nr in r["numbers"] if nr["slot"] > 0 and nr.get("proba") and len(nr["proba"]) >= 2]
        if not groups:
            r["first_free"], r["free_margin"] = None, None
            continue
        has[i] = True
        S = np.zeros(J)
        for j, (f, l, off) in enumerate(firsts):
            for nr in groups:
                ci = nr["slot"] - 1 - off
                v = f + 2 * ci + nr["tier"]
                if ci < 0 or v % 5 != 0 or v > l:
                    S[j] += math.log(0.15) * max(1, len(nr["proba"]))
                else:
                    S[j] += score_value(nr["proba"], v)
        E[i] = S
        order = np.argsort(-S)
        r["first_free"] = firsts[int(order[0])][0]
        r["free_margin"] = round(float(S[order[0]] - S[order[1]]), 2) if J > 1 else None
    if not has.any():
        for r in text_recs:
            r["first_seq"] = None
        return {"status": "no numbers decoded"}
    path = viterbi_path(E)
    for r, j in zip(text_recs, path):
        r["first_seq"] = firsts[int(j)][0]
    return {"status": "ok", "n_candidate_pages": J, "n_pages_with_numbers": int(has.sum()),
            "transition": "±1 page: 0.485 each, stay 0.01, other jumps share 0.02"}


def knn_read(model, cells):
    if model is None or not cells:
        return None, None, 0.0
    V = np.array([c["vec"] for c in cells], dtype=np.float32)
    pt, pk, sm = predict_digits(model["tpl"], V, model["X"], model["y"])
    return int("".join(map(str, pk))), int("".join(map(str, pt))), float(sm.min())


# ----------------------------------------------------------------------------- detector (tuỳ chọn)
def run_detector(recs, book_name):
    try:
        sys.path.insert(0, str(REPO))
        from pipeline.align_engine.char_detector.detector_infer import DetectorInfer
        det = DetectorInfer(ckpt=str(REPO / "train_crop/detector_r34.best.pt"), thr=0.05, device="cpu")
        if not det.trained:
            return {"status": "skipped: checkpoint not loaded"}
    except Exception as e:  # noqa: BLE001
        return {"status": f"skipped: {type(e).__name__}: {str(e)[:80]}"}
    pages_dir = BOOKS[book_name]["pages"]
    eq = tot = 0
    by_tier = collections.Counter()
    for r in recs:
        if not r["cols"]:
            continue
        gray = cv2.imread(str(pages_dir / r["file"]), 0)
        lo, hi = np.percentile(gray, 2), np.percentile(gray, 90)
        g = np.clip((gray.astype(np.float32) - lo) / max(10, hi - lo) * 255.0, 0, 255).astype(np.uint8)
        boxes = [bx for bx in det.boxes_for_page(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)) if bx[4] >= 0.2]
        for c in r["cols"]:
            m = 0.15 * (c["y1"] - c["y0"])
            pbb = [bx for bx in boxes if c["y0"] - m <= (bx[1] + bx[3]) / 2 <= c["y1"] + m]
            raw = det.raw_column_boxes(pbb, (c["x0"], c["x1"]), 0.25)
            c["n_chars_det"] = len(raw)
            tot += 1
            eq += int(len(raw) == c["n_chars"])
            by_tier[(c["tier"], len(raw) == c["expect"])] += 1
    return {"status": "ok", "method": "CenterNet r34, stretch p2→0/p90→255, thr 0.2, raw_column_boxes ±0.25w, tier y-filter",
            "cols": tot, "agree_with_projection_count": eq, "agreement": round(eq / tot, 4) if tot else None,
            "tier0_expect6_ok": by_tier[(0, True)], "tier0_n": by_tier[(0, True)] + by_tier[(0, False)],
            "tier1_expect8_ok": by_tier[(1, True)], "tier1_n": by_tier[(1, True)] + by_tier[(1, False)]}


# ----------------------------------------------------------------------------- debug
def draw_debug(rec, page_path, out_path, title):
    im = cv2.imread(str(page_path), 0)
    dbg = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    W = rec["W"]
    for (y0, y1) in rec["tiers"]:
        cv2.rectangle(dbg, (5, y0), (W - 5, y1), (255, 0, 0), 4)
    for c in rec["cols"]:
        colr = (0, 160, 0) if c["match"] else (0, 0, 255)
        cv2.rectangle(dbg, (c["x0"], c["y0"]), (c["x1"], c["y1"]), colr, 3)
        lab = f"S{c['slot']}:{c['n_chars']}/{c['expect']}" + (f" d{c['n_chars_det']}" if "n_chars_det" in c else "")
        cv2.putText(dbg, lab, (c["x0"], c["y1"] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colr, 2)
        for (s, e) in c["char_boxes"]:
            cv2.rectangle(dbg, (c["x0"] + 3, s), (c["x1"] - 3, e), (200, 120, 0), 1)
    for nr in rec["numbers"]:
        colr = (0, 160, 0) if nr.get("status") == "ok" else (0, 0, 255)
        cv2.rectangle(dbg, (nr["x0"] - 4, nr["y0"] - 4), (nr["x1"] + 4, nr["y1"] + 4), colr, 2)
        cv2.putText(dbg, f"tess={nr['tess'] or '?'} knn={nr.get('knn') or '?'} exp={nr.get('expected') or '-'} {nr.get('status')}",
                    (max(0, nr["x0"] - 80), nr["y0"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colr, 2)
    for cn in rec.get("corner_numbers", []):
        cv2.rectangle(dbg, (cn["x0"] - 4, cn["y0"] - 4), (cn["x1"] + 4, cn["y1"] + 4), (255, 0, 255), 2)
        cv2.putText(dbg, f"pg tess={cn['tess']} knn={cn.get('knn')}", (cn["x0"] - 20, cn["y1"] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(dbg, title, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2)
    cv2.imwrite(str(out_path), dbg, [cv2.IMWRITE_JPEG_QUALITY, 75])


# ----------------------------------------------------------------------------- chính
def _dist(vals):
    return {str(k): v for k, v in sorted(collections.Counter(vals).items())}


def _stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "median": float(np.median(vals)), "min": float(min(vals)), "max": float(max(vals))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", required=True, choices=sorted(BOOKS))
    ap.add_argument("--out", default=None, help="mặc định <repo>/measure_out/<book>/layout/ (thư mục riêng, không đè summary.json của phép đo khác)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N trang đầu (thử)")
    ap.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    ap.add_argument("--detector", action="store_true", help="đếm chữ/cột bằng CenterNet (phương pháp 2, chậm)")
    args = ap.parse_args()
    book_name = args.book
    book = BOOKS[book_name]
    out = Path(args.out) if args.out else REPO / "measure_out" / book_name / "layout"
    (out / "debug").mkdir(parents=True, exist_ok=True)
    for old in (out / "debug").glob("*_debug.jpg"):  # idempotent: xoá ảnh debug của lần chạy trước
        old.unlink()

    files = sorted(book["pages"].glob(book["pattern"]), key=lambda p: unit_id(p.name))
    n_files_total = len(files)
    if args.limit:
        files = files[:args.limit]
    uids = [unit_id(p.name) for p in files]
    n_dbg = min(5, len(files))
    dbg_idx = sorted({int(round(i)) for i in np.linspace(0, len(files) - 1, n_dbg)}) if files else []
    dbg_set = {uids[i] for i in dbg_idx}
    jobs = [(str(p), u, book_name, u in dbg_set) for p, u in zip(files, uids)]
    if args.workers > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            recs = pool.starmap(analyse_page, jobs, chunksize=1)
    else:
        recs = [analyse_page(*j) for j in jobs]

    total = book["total_verses"]
    # ---- phân loại trang (KVK: trắng/bìa/đề từ) & ánh xạ canvas→page
    tr = book.get("text_range")
    for r in recs:
        k = r["uid"]
        cols = [t["n_cols"] for t in r["tier_info"]]
        # trang chữ lục bát: 2 tầng, ≥ 4 cột/tầng, ≥ 50% cột đếm đúng 6 (trên) / 8 (dưới)
        # (đề từ Hán 7 chữ/cột trên canvas 167 KVK không đạt)
        m_rate = float(np.mean([c["match"] for c in r["cols"]])) if r["cols"] else 0.0
        r["parity_match_rate"] = round(m_rate, 3)
        r["is_text_layout"] = int(r["n_tiers"] == 2 and len(cols) == 2 and min(cols) >= 4 and m_rate >= 0.5)
        if r["ink_density_center"] < 0.003:
            r["category"] = "blank"
        elif r["is_text_layout"]:
            r["category"] = "text"
        else:
            r["category"] = "other"
        r["in_text_range"] = int(tr is None or tr[0] <= k <= tr[1])
        r["page"] = book["page_of_canvas"](k) if "page_of_canvas" in book else k
        if not r["in_text_range"]:
            r["page"] = None

    # ---- k-NN chữ số (nhãn tự lấy từ tesseract "hợp lệ"), đọc lại toàn bộ số
    verse_items = [(r["uid"], nr) for r in recs for nr in r["numbers"]]
    model, knn_info = train_digit_knn(verse_items, lambda v, nr: plausible_value(v, nr["tier"], total))
    corner_items = [(r["uid"], cn) for r in recs for cn in r.get("corner_numbers", [])]
    corner_model, corner_info = (train_digit_knn(corner_items, lambda v, nr: v is not None and 1 <= v <= 999, min_unanimous=5)
                                 if corner_items else (None, {"status": "no corner numbers"}))
    prefer_knn = bool(knn_info.get("knn3_acc") and knn_info["knn3_acc"] >= 0.95 and knn_info["labelled_digits"] >= 100)
    knn_info["prefer_knn_as_final"] = prefer_knn
    st_cnt = collections.Counter()
    agree = both = 0
    for r in recs:
        exp_map, first, last, offset = ({}, None, None, 0)
        if r["page"] is not None:
            exp_map, first, last, offset = expected_numbers(book, r["page"])
        r["first_hyp"], r["last_hyp"], r["offset"] = first, last, offset
        exp_by_pos = {v: k for k, v in exp_map.items()}
        implied = []
        for nr in r["numbers"]:
            kv, tv, ms = knn_read(model, nr["cells"])
            nr["knn"], nr["knn_tpl"], nr["min_sim"] = kv, tv, round(ms, 3)
            nr["proba"] = knn_proba(model, nr["cells"])
            if kv is not None and nr["tess"] is not None:
                both += 1; agree += int(kv == nr["tess"])
            cands = [kv, nr["tess"]] if prefer_knn else [nr["tess"], kv]
            # ưu tiên giá trị hợp lệ (bội 5, đúng chẵn/lẻ tầng) của bộ đọc được ưu tiên
            good = [v for v in cands if plausible_value(v, nr["tier"], total)]
            final = good[0] if good else next((v for v in cands if v is not None), None)
            nr["final"] = final
            nr["expected"] = exp_by_pos.get((nr["tier"], nr["slot"]))
            e = exp_map.get(final) if final is not None else None
            if final is None:
                st = "no_read"
            elif r["page"] is None:
                st = "outside_text_range"
            elif e == (nr["tier"], nr["slot"]):
                st = "ok"
            elif e is not None and e[0] == nr["tier"]:
                st = "val_ok_col_wrong"
            elif e is not None:
                st = "val_ok_tier_wrong"
            else:
                st = "mismatch"
            nr["status"] = st
            st_cnt[st] += 1
            # số câu đầu trang suy từ số đọc được (không dùng giả thuyết trang)
            if nr["slot"] > 0:  # mỗi bộ đọc 1 phiếu cho số câu đầu trang (không dùng giả thuyết)
                for v in (kv, nr["tess"]):  # 2 phiếu nếu 2 bộ đọc trùng nhau
                    if plausible_value(v, nr["tier"], total):
                        implied.append(2 * ((v + 1) // 2 - nr["slot"] + 1 + offset) - 1)
            nr.pop("cells", None)
        # số câu đầu trang ước từ số đọc: cần ≥ 2 phiếu và không hoà (không thì None = chưa xác định)
        r["first_est"], r["first_est_votes"] = None, ""
        if implied:
            mc = collections.Counter(implied).most_common()
            if mc[0][1] >= 2 and (len(mc) == 1 or mc[1][1] < mc[0][1]):
                r["first_est"] = mc[0][0]
            r["first_est_votes"] = f"{mc[0][1]}/{len(implied)}"
        best_cn = None
        for cn in r.get("corner_numbers", []):
            cn["knn"] = knn_read(corner_model, cn["cells"])[0]
            pr = knn_proba(corner_model, cn["cells"])
            if pr and (best_cn is None or len(pr) > len(best_cn)):
                best_cn = pr
            cn.pop("cells", None)
        r["corner_proba"] = best_cn
        if "page_number_hyp" in book and r["page"] is not None:
            hyp = book["page_number_hyp"](r["uid"])
            vals = [v for cn in r["corner_numbers"] for v in (cn["knn"], cn["tess"]) if v is not None]
            r["page_no_hyp"], r["page_no_read"] = hyp, (vals[0] if vals else None)
            r["page_no_match"] = int(hyp in vals) if vals else None
        # cặp lục bát: slot có cột ở cả 2 tầng
        s0 = {c["slot"] for c in r["cols"] if c["tier"] == 0}
        s1 = {c["slot"] for c in r["cols"] if c["tier"] == 1}
        r["n_pairs"] = len(s0 & s1) if r["category"] == "text" else 0
        r["n_verses_layout"] = 2 * r["n_pairs"]
        r["n_numbers_ok"] = sum(1 for nr in r["numbers"] if nr["status"] == "ok")
        r["n_numbers_expected"] = len(exp_map)

    text_uid_order = [r for r in sorted(recs, key=lambda r: r["uid"]) if r["page"] is not None]
    seq_info = decode_sequence(text_uid_order, book, total)
    if "page_number_hyp" in book:
        decode_page_numbers(text_uid_order, math.ceil(total / book["per_page"]))
        for r in text_uid_order:
            r["page_no_seq_match"] = int(r["page_no_seq"] == r["page_no_hyp"]) if r.get("page_no_seq") is not None else None
    for r in recs:
        r.pop("corner_proba", None)
        for nr in r["numbers"]:
            nr.pop("proba", None)
    det_info = run_detector(recs, book_name) if args.detector else {"status": "skipped (--detector not set)"}

    # ---- CSV
    with open(out / "layout_pages.csv", "w", newline="") as f:
        keys = ["uid", "file", "page", "category", "in_text_range", "W", "H", "ink_density", "ink_density_center", "n_tiers",
                "tier0_y0", "tier0_y1", "tier1_y0", "tier1_y1", "tier_gap", "row_pitch", "tier0_rows_est", "tier1_rows_est",
                "cols_t0_A", "cols_t1_A", "cols_t0_B", "cols_t1_B", "col_pitch", "n_slots", "n_pairs", "n_verses_layout",
                "n_cols_match", "n_cols_total", "first_hyp", "last_hyp", "first_est", "first_est_votes",
                "n_numbers_expected", "n_numbers_detected", "n_numbers_ok", "numbers_status",
                "page_no_hyp", "page_no_read", "page_no_match", "page_no_free", "page_no_seq", "page_no_seq_match",
                "first_free", "free_margin", "first_seq", "tier0_skew_deg", "tier1_skew_deg"]
        w = csv.writer(f); w.writerow(keys)
        for r in recs:
            t = r["tiers"] + [(-1, -1)] * (2 - len(r["tiers"]))
            ti = r["tier_info"] + [{"n_cols": -1, "n_cols_B": -1}] * (2 - len(r["tier_info"]))
            row = {**{k: r.get(k, "") for k in keys},
                   "tier0_y0": t[0][0], "tier0_y1": t[0][1], "tier1_y0": t[1][0], "tier1_y1": t[1][1],
                   "cols_t0_A": ti[0]["n_cols"], "cols_t1_A": ti[1]["n_cols"], "cols_t0_B": ti[0]["n_cols_B"], "cols_t1_B": ti[1]["n_cols_B"],
                   "col_pitch": r["col_pitch_ref"], "n_cols_match": sum(c["match"] for c in r["cols"]), "n_cols_total": len(r["cols"]),
                   "n_numbers_detected": len(r["numbers"]),
                   "numbers_status": ";".join(f"{x['final'] if x['final'] is not None else '?'}:{x['status']}" for x in r["numbers"]),
                   "ink_density": round(r["ink_density"], 5), "ink_density_center": round(r["ink_density_center"], 5)}
            w.writerow([row.get(k, "") if row.get(k) is not None else "" for k in keys])
    with open(out / "layout_columns.csv", "w", newline="") as f:
        keys = ["uid", "tier", "col", "slot", "x0", "x1", "xc", "y0", "y1", "n_chars", "n_runs_raw", "expect", "match", "char_h_med", "first_top", "ink", "n_chars_det"]
        w = csv.writer(f); w.writerow(keys)
        for r in recs:
            for c in r["cols"]:
                w.writerow([c.get(k, "") for k in keys])
    with open(out / "numbers.csv", "w", newline="") as f:
        keys = ["uid", "tier", "slot", "x0", "y0", "x1", "y1", "tess", "knn", "knn_tpl", "min_sim", "final", "expected", "status", "reads"]
        w = csv.writer(f); w.writerow(keys)
        for r in recs:
            for nr in r["numbers"]:
                w.writerow([nr.get(k, "") if nr.get(k) is not None else "" for k in keys])

    # ---- chuỗi số câu toàn sách (theo thứ tự ĐỌC)
    text = sorted([r for r in recs if r["page"] is not None], key=lambda r: r["page"])
    cum = 0
    chain_rows, chain_mismatch = [], []
    for r in text:
        first_layout = cum + 1
        cum += r["n_verses_layout"]
        row = {"page": r["page"], "uid": r["uid"], "file": r["file"], "category": r["category"], "n_pairs": r["n_pairs"],
               "first_layout": first_layout, "last_layout": cum, "first_hyp": r["first_hyp"], "last_hyp": r["last_hyp"],
               "first_est": r["first_est"], "first_est_votes": r["first_est_votes"],
               "match_est_hyp": int(r["first_est"] == r["first_hyp"]) if r["first_est"] is not None else "",
               "first_free": r.get("first_free"), "free_margin": r.get("free_margin"), "first_seq": r.get("first_seq"),
               "match_seq_hyp": int(r["first_seq"] == r["first_hyp"]) if r.get("first_seq") is not None else "",
               "numbers_ok": r["n_numbers_ok"], "numbers_expected": r["n_numbers_expected"],
               "numbers_status": ";".join(f"{x['final'] if x['final'] is not None else '?'}:{x['status']}" for x in r["numbers"])}
        chain_rows.append(row)
        if r.get("first_seq") is not None and r["first_seq"] != r["first_hyp"]:
            chain_mismatch.append((r["uid"], r["page"], r["first_hyp"], r["first_seq"], f"free={r.get('first_free')} margin={r.get('free_margin')}"))
    with open(out / "verse_chain.csv", "w", newline="") as f:
        if chain_rows:
            w = csv.DictWriter(f, fieldnames=list(chain_rows[0].keys())); w.writeheader(); w.writerows(chain_rows)
    est_seq = [r["first_seq"] for r in text if r.get("first_seq") is not None]
    monotonic = all(a < b for a, b in zip(est_seq, est_seq[1:]))
    free_ok = sum(1 for r in text if r.get("first_free") == r["first_hyp"])
    free_n = sum(1 for r in text if r.get("first_free") is not None)
    low_margin = sorted([(r["free_margin"], r["uid"], r["page"], r["first_free"]) for r in text if r.get("free_margin") is not None])[:8]
    # trang có số cặp ≠ kỳ vọng (định vị lệch tổng câu)
    pair_dev = []
    for r in text:
        if r["category"] != "text":
            continue
        exp_pairs = (r["last_hyp"] - r["first_hyp"] + 1) // 2
        if r["n_pairs"] != exp_pairs:
            pair_dev.append({"uid": r["uid"], "page": r["page"], "n_pairs": r["n_pairs"], "expected_pairs": exp_pairs,
                             "cols_A": [t["n_cols"] for t in r["tier_info"]], "cols_B": [t["n_cols_B"] for t in r["tier_info"]]})

    # ---- debug (≤ 5 trang)
    by_uid = {r["uid"]: r for r in recs}
    if len(files) >= 5:
        base = [uids[0], uids[len(uids) // 2], uids[-1]]
        extra = [u for _, u, _, _ in low_margin if u not in base][:2]
        dbg_set = set(base + extra)
        while len(dbg_set) < 5 and len(uids) > len(dbg_set):
            dbg_set.add(next(u for u in uids if u not in dbg_set))
    for u in sorted(dbg_set):
        r = by_uid[u]
        draw_debug(r, book["pages"] / r["file"], out / "debug" / f"{book['unit']}{u:04d}_debug.jpg",
                   f"{book_name} {book['unit']} {u} page={r['page']} {r['category']} verses_hyp={r['first_hyp']}-{r['last_hyp']} "
                   f"first_free={r.get('first_free')} margin={r.get('free_margin')} first_seq={r.get('first_seq')}")

    # ---- tổng hợp + bất biến
    text_recs = [r for r in text if r["category"] == "text"]
    n_pages = len(recs)
    cols_t0 = [t["n_cols"] for r in text_recs for t in r["tier_info"] if t["tier"] == 0]
    cols_t1 = [t["n_cols"] for r in text_recs for t in r["tier_info"] if t["tier"] == 1]
    col_agree = sum(1 for r in text_recs for t in r["tier_info"] if t["n_cols"] == t["n_cols_B"])
    col_tot = sum(len(r["tier_info"]) for r in text_recs)
    all_cols = [c for r in text_recs for c in r["cols"]]
    n_pairs_total = sum(r["n_pairs"] for r in text_recs)
    n_num = sum(len(r["numbers"]) for r in text)
    n_exp_num = sum(r["n_numbers_expected"] for r in text)
    n_ok = sum(r["n_numbers_ok"] for r in text)
    n_tess_ok = sum(1 for r in text for x in r["numbers"] if x["tess"] is not None and expected_numbers(book, r["page"])[0].get(x["tess"]) == (x["tier"], x["slot"]))
    pages_all_ok = sum(1 for r in text if r["n_numbers_expected"] and r["n_numbers_ok"] >= r["n_numbers_expected"])
    full_run = not args.limit or args.limit >= n_files_total
    interior = [r for r in text_recs if r["page"] != 1 and r["last_hyp"] < total]
    exp_pairs_total = book["expected_pairs"]

    def inv(name, expected, observed, ok, note=""):
        d = {"name": name, "expected": expected, "observed": observed, "pass": (None if observed is None else bool(ok))}
        if note:
            d["note"] = note
        if not full_run and ok is not True and observed is not None:
            d["note"] = (d.get("note", "") + " | partial run (--limit): k-NN/Viterbi under-trained, judge on full run").strip(" |")
        return d

    invariants = [
        inv("pages_2_tiers == text_pages", len(text_recs), sum(1 for r in text_recs if r["n_tiers"] == 2), all(r["n_tiers"] == 2 for r in text_recs)),
        inv("cols_per_tier == 10 (trừ trang đầu/cuối)", "10", _dist([t["n_cols"] for r in interior for t in r["tier_info"]]),
            all(t["n_cols"] == 10 for r in interior for t in r["tier_info"]), f"n_interior_pages={len(interior)}"),
        inv("sum(pairs) == expected", exp_pairs_total if full_run else f"partial run ({len(files)}/{n_files_total})", n_pairs_total,
            (n_pairs_total == exp_pairs_total) if full_run else True, "" if full_run else "chỉ kiểm khi chạy đủ trang"),
        inv("verse numbers monotonic in reading order", True, monotonic, monotonic, f"n_est={len(est_seq)}"),
        inv("parity 6 trên / 8 dưới (cột khớp)", "≥95%", round(sum(c["match"] for c in all_cols) / len(all_cols), 4) if all_cols else None,
            bool(all_cols) and sum(c["match"] for c in all_cols) / len(all_cols) >= 0.95),
        inv("numbers read at expected position (final = tess/knn)", f"≥{book['number_read_min']:.0%}",
            round(n_ok / n_exp_num, 4) if n_exp_num else None,
            bool(n_exp_num) and n_ok / n_exp_num >= book["number_read_min"], book.get("number_read_note", "")),
        inv("first_seq (joint decode + Viterbi) == first_hyp on every page", 0, len(chain_mismatch), len(chain_mismatch) == 0,
            "mismatch list in summary.verse_chain.mismatch"),
        inv("col-count agreement method A (peaks) vs B (runs)", "≥90%", round(col_agree / col_tot, 4) if col_tot else None,
            bool(col_tot) and col_agree / col_tot >= 0.9),
    ]
    if tr is not None:
        non_text = sorted(r["uid"] for r in recs if r["category"] != "text")
        exp_non_text = sorted(u for u in uids if not (tr[0] <= u <= tr[1]))
        invariants.append(inv("non-text canvases == outside text_range", exp_non_text, non_text, non_text == exp_non_text))
        pn = [r for r in text if r.get("page_no_seq_match") is not None]
        invariants.append(inv("corner page number (Viterbi) == 169 − canvas", f"≥95% of {len(pn)}",
                              sum(r["page_no_seq_match"] for r in pn) if pn else None,
                              bool(pn) and sum(r["page_no_seq_match"] for r in pn) >= 0.95 * len(pn),
                              f"free reads match: {sum(1 for r in text if r.get('page_no_free') == r.get('page_no_hyp'))}/{sum(1 for r in text if r.get('page_no_free') is not None)}; non-match listed in summary.page_numbers"))

    summary = {
        "book": book_name, "unit": book["unit"], "pages_total": n_files_total, "pages_processed": n_pages,
        "limit": args.limit or None, "workers": args.workers,
        "method": {"tiers": "Otsu ink, horizontal projection, autocorr row pitch, runs ≥2.5 pitch",
                   "cols_A": "vertical projection per tier, gaussian(0.18 pitch) peaks, valley bounds, RTL",
                   "cols_B": "projection runs > max(20, 0.3·p75), merge gap 12, refine split/drop (kvk_layout.py)",
                   "chars": "ink clusters per column: runs ≥8px merged (gap ≤0.3 pitch, tot ≤0.8 pitch), split >1.3 pitch",
                   "numbers": "digit groups above tier top; tesseract 8-read vote (psm 8/13, ×4/×6) + self-trained kNN3 on 14×20 blurred cells; page-level joint decode + Viterbi (±1 page) for first verse"},
        "category_counts": _dist([r["category"] for r in recs]),
        "canvas_to_page": ("page = 167 − canvas (canvas 4..166)" if "page_of_canvas" in book else "page = file index"),
        "non_text_units": sorted(r["uid"] for r in recs if r["category"] != "text"),
        "tier_gap_px": _stat([r.get("tier_gap") for r in text_recs]),
        "row_pitch_px": _stat([r["row_pitch"] for r in text_recs]),
        "col_pitch_px": _stat([r["col_pitch_ref"] for r in text_recs]),
        "cols_per_tier": {"tier0": _dist(cols_t0), "tier1": _dist(cols_t1), "n_pages": len(text_recs)},
        "cols_method_agreement": {"n_tiers": col_tot, "A_eq_B": col_agree, "rate": round(col_agree / col_tot, 4) if col_tot else None},
        "pairs": {"total": n_pairs_total, "expected_full_book": exp_pairs_total, "verses_from_layout": 2 * n_pairs_total,
                  "pages_with_pairs_deviation": pair_dev[:20], "n_pages_deviation": len(pair_dev)},
        "chars_per_col": {"n_cols": len(all_cols), "match_6_8": sum(c["match"] for c in all_cols),
                          "rate": round(sum(c["match"] for c in all_cols) / len(all_cols), 4) if all_cols else None,
                          "tier0_dist": _dist([c["n_chars"] for c in all_cols if c["tier"] == 0]),
                          "tier1_dist": _dist([c["n_chars"] for c in all_cols if c["tier"] == 1]),
                          "detector_2nd_method": det_info},
        "numbers": {"detected": n_num, "expected": n_exp_num, "tesseract_ok": n_tess_ok, "final_ok": n_ok,
                    "status_counts": dict(st_cnt), "pages_all_numbers_ok": pages_all_ok, "n_text_pages": len(text),
                    "tess_knn_both_read": both, "tess_knn_agree": agree, "agreement": round(agree / both, 4) if both else None,
                    "digit_knn": knn_info},
        "verse_chain": {"n_pages_with_seq": len(est_seq), "monotonic": monotonic,
                        "first_seq_range": [est_seq[0], est_seq[-1]] if est_seq else None,
                        "decode": seq_info, "free_decode_match_hyp": f"{free_ok}/{free_n}",
                        "vote_est_match_hyp": f"{sum(1 for r in text if r['first_est'] == r['first_hyp'])}/{sum(1 for r in text if r['first_est'] is not None)}",
                        "lowest_free_margin": [{"margin": m, "uid": u, "page": p, "first_free": f} for m, u, p, f in low_margin],
                        "mismatch": [{"uid": u, "page": p, "first_hyp": h, "first_seq": e, "note": v} for u, p, h, e, v in chain_mismatch[:20]],
                        "n_mismatch": len(chain_mismatch)},
        "skew_deg_abs": {"tier0_p50": round(float(np.median([abs(r["tier0_skew_deg"]) for r in text_recs if "tier0_skew_deg" in r])), 2) if any("tier0_skew_deg" in r for r in text_recs) else None,
                         "max": round(max([abs(r.get(f"tier{i}_skew_deg", 0)) for r in text_recs for i in (0, 1)] or [0]), 2)},
        "invariants": invariants,
        "outputs": ["summary.json", "layout_pages.csv", "layout_columns.csv", "numbers.csv", "verse_chain.csv", "debug/"],
    }
    if "page_number_hyp" in book:
        pn = [r for r in text if r.get("page_no_free") is not None]
        summary["page_numbers"] = {"hyp": "169 − canvas", "n_read": len(pn), "corner_font_knn": corner_info,
                                   "free_match": sum(1 for r in pn if r["page_no_free"] == r["page_no_hyp"]),
                                   "viterbi_match": sum(1 for r in text if r.get("page_no_seq_match")),
                                   "viterbi_non_match": [(r["uid"], r["page_no_hyp"], r["page_no_seq"]) for r in text if r.get("page_no_seq_match") == 0][:15],
                                   "free_non_match": [(r["uid"], r["page_no_hyp"], r["page_no_free"]) for r in pn if r["page_no_free"] != r["page_no_hyp"]][:10]}
        # định vị lệch 3.256 (Nôm) vs 3.253 (QN): trang có số in không khớp 20·(166−k)+…
        cands = sorted({m["uid"] for m in summary["verse_chain"]["mismatch"]} | {d["uid"] for d in pair_dev})
        qn_jump = 1581  # QN in 1884 nhảy 1581→1585 (SOURCE.md); câu 1581 nằm ở trang p → canvas 167−p
        p_jump = (qn_jump - 1) // 20 + 1
        summary["discrepancy_candidates"] = {
            "rule": "canvas whose decoded first ≠ 20·(166−k)+1 (Viterbi), or n_pairs ≠ expected",
            "canvases": cands,
            "nom_verse_slots_from_layout": 2 * n_pairs_total,
            "qn_jump_1581_to_1585_location": {"page": p_jump, "canvas": 167 - p_jump,
                                              "canvas_n_pairs": next((r["n_pairs"] for r in text if r["uid"] == 167 - p_jump), None)},
            "conclusion": ("no Nôm canvas deviates from the 20-verse grid → the 3.256 vs 3.253 gap is not in the Nôm layout; "
                           "check QN numbering around 1581–1585 (canvas above)" if not cands and full_run else
                           "see canvases listed" if cands else "partial run: not conclusive")}
    js = json.dumps(summary, ensure_ascii=False, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    if len(js.encode()) > 8000:  # cắt bớt danh sách dài để giữ ≤ 8 KB
        summary["pairs"]["pages_with_pairs_deviation"] = summary["pairs"]["pages_with_pairs_deviation"][:5]
        summary["verse_chain"]["mismatch"] = summary["verse_chain"]["mismatch"][:5]
        summary["non_text_units"] = summary["non_text_units"][:20]
        js = json.dumps(summary, ensure_ascii=False, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    (out / "summary.json").write_text(js)
    with open(out / "pages_full.json", "w") as f:
        json.dump(recs, f, ensure_ascii=False, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))

    # ---- stdout ngắn
    print(f"[{book_name}] {n_pages}/{n_files_total} {book['unit']}s -> {out}")
    print(f" categories={summary['category_counts']} tiers2={sum(r['n_tiers']==2 for r in text_recs)}/{len(text_recs)}")
    print(f" cols t0={summary['cols_per_tier']['tier0']} t1={summary['cols_per_tier']['tier1']} A==B {col_agree}/{col_tot}")
    print(f" pairs={n_pairs_total} (expected full book {exp_pairs_total}); chars 6/8 match {summary['chars_per_col']['match_6_8']}/{len(all_cols)}")
    print(f" numbers: detected {n_num} expected {n_exp_num} tess_ok {n_tess_ok} final_ok {n_ok}; tess/knn agree {agree}/{both}; knn {knn_info.get('status')} acc={knn_info.get('knn3_acc')}")
    print(f" chain: free decode {free_ok}/{free_n} == hyp; Viterbi seq mismatch={len(chain_mismatch)} monotonic={monotonic}; pair_dev={len(pair_dev)}; detector: {det_info.get('status')} {det_info.get('agreement', '')}")
    for iv in invariants:
        tag = "SKIP" if iv["pass"] is None else ("PASS" if iv["pass"] else "FAIL")
        print(f" [{tag}] {iv['name']}: expected={iv['expected']} observed={iv['observed']}")


if __name__ == "__main__":
    main()
