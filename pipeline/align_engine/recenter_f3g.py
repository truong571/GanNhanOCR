"""D-1 · crops_v2 — TÁI ĐỊNH TÂM HỘP THEO HÌNH CHIẾU MỰC (F3g, hướng C).

Nguồn thiết kế: docs/nghien_cuu_dang_do/hoan_tat_4_huong.json[0] (chi_tiet[3] + ket_qua + so_o);
DANH_GIA_KE_HOACH_NANG_CAP_2026-09-15 §2.3; DANH_MUC_SUA_DOI_CUOI_2026-09-16 Khối D. Mã F3g gốc
(/tmp/huong_c/save_crop_v2.py) đã mất — mô-đun này DỰNG LẠI theo mô tả, tham số y nguyên bản gốc.

VÌ SAO: tâm hộp bản ghi (= hộp CenterNet thô ở 90% ô) lệch tâm mực của chính chữ >0,25 pitch ở
34–38% ô (heatmap stride 4 trên letterbox 512 ≈ 0,22–0,33 pitch/ô lưới). Cửa sổ sản xuất
(pad 0,12 + carve + tighten, cao 1,44×pitch) đang CHE lỗi đó bằng cách ôm cả mực hàng xóm
(44,6% ô có >25% mực hàng xóm). Mọi cửa sổ hẹp đặt quanh tâm hộp chỉ đổi mực hàng xóm lấy cắt cụt.

F3g (mỗi ô, chỉ cần hộp của chính nó + hộp 2 bản ghi kề để làm TIÊN NGHIỆM tâm):
  1. x = hộp ± X_PAD·w (như save_crop).
  2. hình chiếu mực theo hàng (tỷ lệ điểm mực/hàng, gray<128) trong cửa sổ cp ± ~1,1p; làm trơn
     hộp 3 px; lấy min ±2 px (ranh giới bắt được thung lũng cách 2 px vẫn tính là "sạch").
  3. tiên nghiệm tâm cp = (cy + (cy_prev+cy_next)/2)/2 nếu CẢ HAI hàng xóm cách 0,7–1,4 pitch,
     không thì cp = cy (hàng xóm ảo / lỗ cột không kéo tâm đi xa — chi_tiet[5]).
  4. chọn (top, h): top ∈ cp − 0,5p ± 0,45p, h ∈ [0,95; 1,10]p (bước 1 px), chi phí
        prof(top) + prof(top+h) + 3·((h − 1,02p)/p)² + 0,5·((tâm − cp)/p)²
  5. BẢO HIỂM: |cy − tâm| > 0,40p → 'fallback' = cửa sổ cũ (byte-trùng crop giao nộp). Đo ở hướng C:
     ngưỡng 0,35/0,40/0,45/0,50 chặn 86/72/43/3 trên 99 ca nhầm chữ mô phỏng, rơi về 23/15/4,7/0,2%;
     thực tế rơi về 4,6–5,2%.
  6. 'f3g': nới EXT=0,10p hai đầu → xoá mực ngoài SEAM CONG (bbox_fix._seam_boundary) tìm trong dải
     ±SEAM_BAND=0,10p quanh mỗi ranh giới → tighten_box → PNG.

ĐƠN VỊ: recenter_shift = (tâm_v2 − cy)/pitch (có dấu; + = xuống). pitch = trung vị khoảng cách tâm
các bản ghi cùng cột (như align_production._reseg_column / BOX_OVERLAP_FRAC), rơi về trung vị trang
khi cột <2 bản ghi hoặc lệch >[0,6; 1,6]× trang.

KHÔNG đổi crop giao nộp, KHÔNG đổi labels.csv: build_dataset --crops-v2 (mặc định TẮT) ghi song song
$out/crops_v2/<tier>/<cùng tên>.png + $out/labels_crops_v2.csv; pipeline/tools/recrop_v2.py cắt lại
từ labels_final.csv có sẵn (bbox tái lập md5 giao nộp 1907/1907 ô, 12 trang — kiểm 2026-09-16).

    .venv/bin/python -m pipeline.align_engine.recenter_f3g --selftest
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

from pipeline.align_engine.bbox_fix import _seam_boundary, carve_neighbor_ink, tighten_box
from pipeline.align_engine import crop_quality as cq

# --- tham số F3g (hoan_tat_4_huong.json[0].chi_tiet[3]) ---------------------------------
X_PAD = 0.12          # pad ngang theo w (= step2.crop_pad_frac của save_crop)
H_RANGE = (0.95, 1.10)  # chiều cao hộp v2 theo pitch
H_TARGET = 1.02       # chiều cao ưa thích
TOP_SEARCH = 0.45     # top ∈ cp − 0,5p ± TOP_SEARCH·p
LAMBDA_H = 3.0        # phạt lệch chiều cao
LAMBDA_C = 0.5        # phạt lệch tâm khỏi tiên nghiệm (μ = 0,5)
GUARD_SHIFT = 0.40    # bảo hiểm: |cy − tâm| > 0,40p -> fallback
EXT = 0.10            # nới cửa sổ mỗi đầu (pitch)
SEAM_BAND = 0.10      # dải tìm seam quanh ranh giới (pitch)
NEIGHBOR_OK = (0.7, 1.4)  # hàng xóm hợp lệ để dùng tiên nghiệm
SMOOTH_PX = 3
MIN_PX = 2
INK_THR = 128
PROF_WIN = 1.10       # cửa sổ hình chiếu cp ± PROF_WIN·p
PITCH_RATIO_OK = (0.6, 1.6)   # pitch cột / pitch trang hợp lệ


# --------------------------------------------------------------------------- pitch
def column_pitch(cys, fallback: float | None = None) -> float | None:
    """Trung vị khoảng cách tâm các bản ghi cùng cột (y hệt _reseg_column); rơi về `fallback`
    (trung vị trang) khi <2 bản ghi hoặc lệch quá PITCH_RATIO_OK× fallback. None nếu không có gì."""
    cys = np.sort(np.asarray([float(c) for c in cys], float))
    p = float(np.median(np.diff(cys))) if len(cys) >= 2 else None
    if p is not None and not (p > 0):
        p = None
    if fallback is not None and fallback > 0:
        if p is None or not (PITCH_RATIO_OK[0] * fallback <= p <= PITCH_RATIO_OK[1] * fallback):
            return float(fallback)
    return p


def page_pitch(col_cys: dict) -> float | None:
    """Trung vị của các pitch cột (chỉ cột ≥2 bản ghi) — mốc rơi về cho column_pitch."""
    ps = []
    for cys in col_cys.values():
        c = np.sort(np.asarray(cys, float))
        if len(c) >= 2:
            d = float(np.median(np.diff(c)))
            if d > 0:
                ps.append(d)
    return float(np.median(ps)) if ps else None


# --------------------------------------------------------------------------- hình chiếu mực
def ink_profile(gray_full: np.ndarray, xa: int, xb: int, ya: int, yb: int) -> np.ndarray:
    """Tỷ lệ điểm mực (gray<INK_THR) từng hàng y ∈ [ya, yb) trong dải x ∈ [xa, xb); hàng ngoài ảnh = 0.
    Đã làm trơn hộp SMOOTH_PX và lấy min ±MIN_PX."""
    H, W = gray_full.shape[:2]
    n = max(0, yb - ya)
    prof = np.zeros(n, np.float32)
    if n == 0:
        return prof
    x0, x1 = max(0, xa), min(W, xb)
    y0, y1 = max(0, ya), min(H, yb)
    if x1 > x0 and y1 > y0:
        sub = gray_full[y0:y1, x0:x1] < INK_THR
        prof[y0 - ya:y1 - ya] = sub.mean(axis=1)
    if SMOOTH_PX > 1:
        k = np.ones(SMOOTH_PX, np.float32) / SMOOTH_PX
        prof = np.convolve(prof, k, mode="same").astype(np.float32)
    if MIN_PX > 0:
        pad = np.pad(prof, MIN_PX, mode="edge")
        prof = np.min(np.stack([pad[i:i + n] for i in range(2 * MIN_PX + 1)]), axis=0)
    return prof


def prior_center(cy: float, cy_prev, cy_next, pitch: float) -> tuple[float, bool]:
    """cp = (cy + (cy_prev+cy_next)/2)/2 khi CẢ HAI hàng xóm cách 0,7–1,4p; không thì cy."""
    if cy_prev is None or cy_next is None or not (pitch > 0):
        return float(cy), False
    lo, hi = NEIGHBOR_OK[0] * pitch, NEIGHBOR_OK[1] * pitch
    if lo <= (cy - cy_prev) <= hi and lo <= (cy_next - cy) <= hi:
        return float((cy + (cy_prev + cy_next) / 2.0) / 2.0), True
    return float(cy), False


def _cy(b) -> float | None:
    return None if b is None else (float(b[1]) + float(b[3])) / 2.0


# --------------------------------------------------------------------------- F3g
def recenter_box(gray_full: np.ndarray, bbox, prev_bbox, next_bbox, pitch: float,
                 guard: float = GUARD_SHIFT):
    """-> (bbox_v2, crop_mode, recenter_shift_pitch, meta).

    bbox_v2 = [x1, top, x2, top+h] hộp tái định tâm (CHƯA nới EXT; x giữ nguyên) khi 'f3g';
    = bbox gốc khi 'fallback' (bảo hiểm) hoặc 'fallback' vì thiếu dữ liệu (pitch/ảnh).
    recenter_shift = (tâm_v2 − cy)/pitch, ghi cả khi fallback (để đo phân bố)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    meta = {"pitch": pitch, "cy": (y1 + y2) / 2.0, "cp": None, "prior": False, "top": None, "h": None,
            "center": None, "cost": None, "reason": ""}
    if gray_full is None or not (pitch and pitch > 0) or x2 <= x1 or y2 <= y1:
        meta["reason"] = "no_data"
        return [x1, y1, x2, y2], "fallback", 0.0, meta
    p = float(pitch)
    w = x2 - x1
    cy = (y1 + y2) / 2.0
    cp, used = prior_center(cy, _cy(prev_bbox), _cy(next_bbox), p)
    meta.update(cp=cp, prior=used)

    pw = int(w * X_PAD)
    xa, xb = x1 - pw, x2 + pw
    ya = int(np.floor(cp - PROF_WIN * p)) - 3
    yb = int(np.ceil(cp + PROF_WIN * p)) + 4
    prof = ink_profile(gray_full, xa, xb, ya, yb)

    t_lo = int(np.round(cp - 0.5 * p - TOP_SEARCH * p))
    t_hi = int(np.round(cp - 0.5 * p + TOP_SEARCH * p))
    h_lo = max(2, int(np.round(H_RANGE[0] * p)))
    h_hi = max(h_lo, int(np.round(H_RANGE[1] * p)))
    tops = np.arange(t_lo, t_hi + 1)
    hs = np.arange(h_lo, h_hi + 1)
    T, Hh = np.meshgrid(tops, hs, indexing="ij")
    B = T + Hh
    n = len(prof)
    ti = np.clip(T - ya, 0, n - 1)
    bi = np.clip(B - ya, 0, n - 1)
    # ranh giới rơi ngoài cửa sổ hình chiếu (chỉ khi ảnh cụt) -> coi là 0 mực
    pt = np.where((T - ya >= 0) & (T - ya < n), prof[ti], 0.0)
    pb = np.where((B - ya >= 0) & (B - ya < n), prof[bi], 0.0)
    cost = (pt + pb + LAMBDA_H * ((Hh - H_TARGET * p) / p) ** 2
            + LAMBDA_C * (((T + Hh / 2.0) - cp) / p) ** 2)
    k = int(np.argmin(cost))
    i, j = divmod(k, cost.shape[1])
    top, h = int(tops[i]), int(hs[j])
    center = top + h / 2.0
    shift = float((center - cy) / p)
    meta.update(top=top, h=h, center=center, cost=float(cost[i, j]),
                prof_top=float(pt[i, j]), prof_bot=float(pb[i, j]))
    if abs(center - cy) > guard * p:
        meta["reason"] = "guard"
        return [x1, y1, x2, y2], "fallback", shift, meta
    # bảo hiểm phụ (ngoài bản gốc, chỉ SIẾT thêm): tâm v2 phải gần tâm bản ghi này hơn tâm bản ghi kề —
    # hộp lệch > TOP_SEARCH thì chữ thật ngoài tầm tìm, cặp ranh giới có thể khoá vào chữ hàng xóm
    # gần (<0,8p) mà |tâm − cy| vẫn ≤ 0,40p (ca "nhầm chữ" lọt bảo hiểm 27/99 trong mô phỏng hướng C).
    for nb in (_cy(prev_bbox), _cy(next_bbox)):
        if nb is not None and abs(center - nb) < abs(center - cy):
            meta["reason"] = "guard_neighbor"
            return [x1, y1, x2, y2], "fallback", shift, meta
    return [x1, top, x2, top + h], "f3g", shift, meta


def window_v2(bbox_v2, pitch: float) -> list:
    """Cửa sổ cắt của hộp v2 (nới EXT·p hai đầu; x giữ)."""
    x1, y1, x2, y2 = (int(v) for v in bbox_v2)
    e = int(np.round(EXT * pitch))
    return [x1, y1 - e, x2, y2 + e]


def crop_f3g(img: np.ndarray, gray_full: np.ndarray, bbox_v2, pitch: float,
             pad: float = X_PAD, tighten: bool = True, bg: int = 255):
    """Cắt crop 'f3g': cửa sổ x ± pad·w, y nới EXT·p; xoá mực ngoài seam cong tìm trong dải
    ±SEAM_BAND·p quanh mỗi ranh giới; tighten_box. Trả (crop BGR/gray, gray) hoặc (None, None)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox_v2)
    e = int(np.round(EXT * pitch))
    band = int(np.round(SEAM_BAND * pitch))
    pw = int((x2 - x1) * pad)
    cx1, cx2 = max(0, x1 - pw), min(W, x2 + pw)
    cy1, cy2 = max(0, y1 - e), min(H, y2 + e)
    if cx2 <= cx1 or cy2 <= cy1:
        return None, None
    crop = img[cy1:cy2, cx1:cx2].copy()
    ch, cw = crop.shape[:2]
    # seam trên: dải [y1 − band, y1 + band] ∩ cửa sổ
    ta, tb = max(cy1, y1 - band), min(cy2, y1 + band)
    if tb - ta >= 3:
        seam = _seam_boundary(gray_full, cx1, cx2, ta, tb)
        if seam is not None and len(seam) == cw:
            for j in range(cw):
                yy = min(max(int(seam[j]) - cy1, 0), ch)
                crop[:yy, j] = bg
    # seam dưới: dải [y2 − band, y2 + band] ∩ cửa sổ
    ba, bb = max(cy1, y2 - band), min(cy2, y2 + band)
    if bb - ba >= 3:
        seam = _seam_boundary(gray_full, cx1, cx2, ba, bb)
        if seam is not None and len(seam) == cw:
            for j in range(cw):
                yy = max(min(int(seam[j]) - cy1, ch), 0)
                crop[yy:, j] = bg
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if tighten:
        tb_ = tighten_box(gray)
        if tb_ is not None:
            a, c, b, d = tb_
            crop, gray = crop[c:d, a:b], gray[c:d, a:b]
    if crop.size == 0:
        return None, None
    return crop, gray


def crop_legacy(img: np.ndarray, gray_full, bbox, pad: float, prev_bbox, next_bbox,
                tighten: bool = True):
    """Cửa sổ CŨ — chép đúng build_dataset.save_crop (pad ± carve hàng xóm ± tighten) để
    'fallback' trùng byte crop giao nộp (selftest đối chiếu md5 với save_crop trên trang thật)."""
    H, W = img.shape[:2]
    ox1, oy1, ox2, oy2 = (int(v) for v in bbox)
    pw, ph = int((ox2 - ox1) * pad), int((oy2 - oy1) * pad)
    x1, y1 = max(0, ox1 - pw), max(0, oy1 - ph)
    x2, y2 = min(W, ox2 + pw), min(H, oy2 + ph)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    crop = crop.copy()
    if gray_full is not None and (prev_bbox is not None or next_bbox is not None):
        crop = carve_neighbor_ink(crop, gray_full, x1, y1, x2, y2, (oy1, oy2), prev_bbox, next_bbox)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if tighten:
        tb = tighten_box(gray)
        if tb is not None:
            a, c, b, d = tb
            crop, gray = crop[c:d, a:b], gray[c:d, a:b]
    if crop.size == 0:
        return None, None
    return crop, gray


def save_crop_v2(img, gray_full, bbox, prev_bbox, next_bbox, pitch, pad: float, path: Path | None,
                 tighten: bool = True, guard: float = GUARD_SHIFT) -> dict | None:
    """Một ô: recenter_box → cắt ('f3g' hoặc 'fallback' = cửa sổ cũ) → (ghi PNG) → thống kê.
    path=None: không ghi (chỉ đo). Trả dict: bbox_v2, bbox_v2_win, crop_mode, recenter_shift,
    pitch, md5, ink, w, h, stray_ink, border_ink, crop_quality_flag, meta; None nếu không cắt được."""
    if img is None or not bbox:
        return None
    bbox_v2, mode, shift, meta = recenter_box(gray_full, bbox, prev_bbox, next_bbox, pitch, guard=guard)
    if mode == "f3g":
        crop, gray = crop_f3g(img, gray_full, bbox_v2, pitch, pad=pad, tighten=tighten)
        if crop is None:                      # cửa sổ rỗng (không xảy ra với hộp hợp lệ) -> cũ
            mode, bbox_v2 = "fallback", [int(v) for v in bbox]
            meta["reason"] = "empty_f3g"
    if mode != "f3g":
        crop, gray = crop_legacy(img, gray_full, bbox, pad, prev_bbox, next_bbox, tighten=tighten)
    if crop is None:
        return None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), crop)
        md5 = hashlib.md5(path.read_bytes()).hexdigest()[:12]
    else:
        ok, buf = cv2.imencode(".png", crop)
        md5 = hashlib.md5(buf.tobytes()).hexdigest()[:12] if ok else ""
    q = cq.measure(gray)
    ch, cw = crop.shape[:2]
    return {"bbox_v2": [int(v) for v in bbox_v2],
            "bbox_v2_win": window_v2(bbox_v2, pitch) if (mode == "f3g" and pitch) else [int(v) for v in bbox],
            "crop_mode": mode, "recenter_shift": round(shift, 4),
            "pitch": round(float(pitch), 2) if pitch else "",
            "md5": md5, "ink": round(float((gray < INK_THR).mean()), 4), "w": cw, "h": ch,
            "stray_ink": q["stray_ink"], "border_ink": q["border_ink"],
            "crop_quality_flag": q["crop_quality_flag"], "meta": meta, "crop": crop}


# --------------------------------------------------------------------------- selftest
def _synth_column(pitch=100, w=90, n=5, glyph_h=0.80, offsets=None, touch=False, seed=0):
    """Cột tổng hợp: n 'chữ' = khối mực chữ nhật (có khe ngang nội tại) tâm tại y_k = 150 + k·pitch
    (+ offset). Trả (img BGR, gray, boxes hộp 'detector' lệch theo offsets, true_boxes)."""
    H, W = 150 * 2 + n * pitch, w + 200
    gray = np.full((H, W), 255, np.uint8)
    x1, x2 = 100, 100 + w
    true_boxes, det_boxes = [], []
    offsets = offsets or [0] * n
    gh = int(glyph_h * pitch)
    for k in range(n):
        yc = 150 + k * pitch
        gt, gb = yc - gh // 2, yc + gh // 2
        gray[gt:gb, x1 + 8:x2 - 8] = 30
        # khe ngang NỘI TẠI (kiểu 哭/意) — chỉ 60% bề rộng, hai nét dọc hai bên còn nguyên
        gray[gt + gh // 2 - 3:gt + gh // 2 + 3, x1 + 8 + int(0.2 * w):x2 - 8 - int(0.2 * w)] = 255
        if touch:                                                           # nét tràn xuống dưới
            gray[gb:gb + int(0.12 * pitch), x1 + w // 2 - 5:x1 + w // 2 + 5] = 30
        true_boxes.append([x1, gt, x2, gb])
        bh = int(1.2 * pitch)
        dy = int(offsets[k] * pitch)
        det_boxes.append([x1, yc + dy - bh // 2, x2, yc + dy + bh // 2])
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img, gray, det_boxes, true_boxes


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def selftest() -> int:
    fails = passed = 0

    def check(cond, msg):
        nonlocal fails, passed
        print(("  ok   " if cond else "  FAIL ") + msg)
        fails += (not cond)
        passed += bool(cond)

    p = 100
    # 1) chữ lệch 0,30p: hộp detector lệch xuống 0,3p -> v2 về tâm mực (|tâm_v2 − tâm thật| < 0,05p)
    # (khe trắng giữa 2 chữ 0,10p như chữ Kai thật 0,86–0,92p; khe rộng hơn thì cặp ranh giới có
    # thể trượt trong vùng prof=0 tới ±(khe−h)/2 và tiên nghiệm μ=0,5 kéo về cp = cy − 0,15p)
    img, gray, det, true = _synth_column(pitch=p, glyph_h=0.90, offsets=[0, 0, 0.30, 0, 0])
    bb, mode, shift, meta = recenter_box(gray, det[2], det[1], det[3], p)
    tc = (true[2][1] + true[2][3]) / 2
    check(mode == "f3g", f"lệch 0,30p -> f3g (mode={mode}, shift={shift:+.3f}p)")
    check(abs((bb[1] + bb[3]) / 2 - tc) < 0.07 * p, f"tâm v2 về tâm mực: |Δ|={abs((bb[1]+bb[3])/2-tc)/p:.3f}p (det lệch 0,30p)")
    check(-0.31 < shift < -0.22, f"recenter_shift ≈ −0,30 (đo {shift:+.3f})")
    check(H_RANGE[0] * p <= bb[3] - bb[1] <= H_RANGE[1] * p, f"h v2 = {(bb[3]-bb[1])/p:.2f}p ∈ [0,95;1,10]")
    check(_iou(bb, true[2]) > 0.75 and _iou(bb, true[2]) > _iou(det[2], true[2]),
          f"IoU với hộp thật: det {_iou(det[2], true[2]):.3f} -> v2 {_iou(bb, true[2]):.3f}")
    # 1b) tiên nghiệm tâm: hàng xóm cách đúng pitch -> prior dùng; hàng xóm xa (lỗ) -> không dùng
    _, _, _, meta_ok = recenter_box(gray, det[2], det[1], det[3], p)
    _, _, _, meta_far = recenter_box(gray, det[2], [det[1][0], det[1][1] - 2 * p, det[1][2], det[1][3] - 2 * p], det[3], p)
    check(meta_ok["prior"] is True and meta_far["prior"] is False, "tiên nghiệm tâm: dùng khi hàng xóm 0,7–1,4p, bỏ khi lỗ cột")
    # 2) hộp không lệch: v2 ≈ hộp thật, crop f3g giữ đủ mực chữ (không cắt cụt)
    img, gray, det, true = _synth_column(pitch=p)
    bb, mode, shift, _ = recenter_box(gray, det[2], det[1], det[3], p)
    check(mode == "f3g" and abs(shift) < 0.05, f"không lệch -> f3g, shift {shift:+.3f}p")
    crop, g = crop_f3g(img, gray, bb, p)
    ink_true = int((gray[true[2][1]:true[2][3], true[2][0]:true[2][2]] < 128).sum())
    check(abs(int((g < 128).sum()) - ink_true) <= 0.02 * ink_true,
          f"crop f3g giữ mực chữ: {int((g<128).sum())} vs thật {ink_true}")
    # 3) hàng xóm dính (nét tràn xuống): seam cong xoá phần tràn của chữ TRÊN khỏi crop chữ dưới
    img, gray, det, true = _synth_column(pitch=p, glyph_h=0.86, touch=True)
    bb, mode, shift, _ = recenter_box(gray, det[2], det[1], det[3], p)
    crop, g = crop_f3g(img, gray, bb, p)
    # phần tràn của chữ 1 nằm ở hàng gb_1 .. gb_1+0,12p; sau seam, crop chữ 2 không được chứa mực ở dải đó
    win = window_v2(bb, p)
    y_spill = true[1][3] + int(0.12 * p)          # hàng cuối của vệt tràn
    check(mode == "f3g", f"hàng xóm dính -> f3g (shift {shift:+.3f}p)")
    # crop f3g đã tighten nên đo trên bản chưa tighten
    crop_nt, g_nt = crop_f3g(img, gray, bb, p, tighten=False)
    top_rows = g_nt[: max(0, y_spill - max(0, win[1])), :]
    check(top_rows.size == 0 or (top_rows < 128).mean() < 0.005,
          f"seam xoá vệt tràn của chữ trên: mực dải trên = {(top_rows<128).mean() if top_rows.size else 0:.4f}")
    # đối chứng: cửa sổ cũ (legacy, có carve) vẫn còn vệt tràn? -> chỉ báo (không bắt buộc)
    # 4) lệch 0,50p: thung lũng thật còn trong tầm tìm (±0,45p quanh cp), |cy − tâm| > 0,40p -> bảo hiểm
    #    -> fallback, bbox_v2 = bbox cũ, crop trùng byte cửa sổ cũ. (Lệch 0,40–0,45p là vùng mờ: cặp
    #    ranh giới trượt được ±0,06p trong khe trắng 0,10p nên tiên nghiệm kéo |shift| xuống dưới 0,40.)
    img, gray, det, true = _synth_column(pitch=p, glyph_h=0.90, offsets=[0, 0, 0.50, 0, 0])
    bb, mode, shift, meta = recenter_box(gray, det[2], det[1], det[3], p)
    check(mode == "fallback" and bb == [int(v) for v in det[2]] and meta["reason"] in ("guard", "guard_neighbor"),
          f"lệch 0,50p -> fallback (shift đo {shift:+.3f}p, lý do {meta['reason']})")
    # 4b) lệch 0,55p (NGOÀI tầm tìm): chữ thật không với tới, cặp ranh giới khoá vào chữ kề cách 0,45p
    #     -> bảo hiểm phụ (tâm v2 gần tâm hàng xóm hơn) bắt -> fallback
    img2, gray2, det2, true2 = _synth_column(pitch=p, glyph_h=0.90, offsets=[0, 0, 0.55, 0, 0])
    bb2, mode2, shift2, meta2 = recenter_box(gray2, det2[2], det2[1], det2[3], p)
    check(mode2 == "fallback" and meta2["reason"] in ("guard", "guard_neighbor"),
          f"lệch 0,55p (ngoài tầm) -> fallback (shift đo {shift2:+.3f}p, lý do {meta2['reason']})")
    r = save_crop_v2(img, gray, det[2], det[1], det[3], p, X_PAD, None)
    cl, _ = crop_legacy(img, gray, det[2], X_PAD, det[1], det[3])
    check(r["crop_mode"] == "fallback" and np.array_equal(r["crop"], cl), "fallback = crop cửa sổ cũ (byte)")
    # 5) fallback trùng byte build_dataset.save_crop trên TRANG THẬT (nếu có ảnh)
    try:
        import json as _json
        import tempfile
        from pipeline.align_engine.build_dataset import save_crop
        repo = Path(__file__).resolve().parents[2]
        lab = repo / "dataset_out/labels_final.csv"
        png = None
        if lab.exists():
            import csv
            rows = []
            for r in csv.DictReader(open(lab, encoding="utf-8")):
                if r["book"] != "stt2" or not r["bbox"] or not r["image"]:
                    continue
                if png is None:
                    png = repo / "prepared/SachThanhTruyen2/pages" / f"{r['page']}.png"
                    page = r["page"]
                if r["page"] != page:
                    break
                rows.append(r)
            rows = rows[:40]
        if png is not None and png.exists():
            im = cv2.imread(str(png), cv2.IMREAD_COLOR)
            gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            n_eq = 0
            with tempfile.TemporaryDirectory() as td:
                for r in rows:
                    b = _json.loads(r["bbox"])
                    q = save_crop(im, gr, b, X_PAD, Path(td) / "a.png", prev_bbox=None, next_bbox=None)
                    c, _ = crop_legacy(im, gr, b, X_PAD, None, None)
                    cv2.imwrite(str(Path(td) / "b.png"), c)
                    n_eq += (Path(td) / "b.png").read_bytes() == (Path(td) / "a.png").read_bytes()
            check(n_eq == len(rows), f"crop_legacy trùng byte save_crop trên trang thật ({png.name}): {n_eq}/{len(rows)}")
            # thời gian: 5 ms/crop?
            import time
            t0 = time.time()
            for r in rows:
                save_crop_v2(im, gr, _json.loads(r["bbox"]), None, None, 95.0, X_PAD, None)
            dt = (time.time() - t0) / max(1, len(rows)) * 1000
            check(dt < 30, f"tốc độ {dt:.1f} ms/crop (mốc hướng C 5 ms)")
    except Exception as e:                                # pragma: no cover
        check(False, f"đối chiếu save_crop lỗi: {type(e).__name__}: {e}")
    # 6) pitch: cột 1 bản ghi -> trung vị trang; cột lệch quá -> trung vị trang
    check(column_pitch([100.0], fallback=95.0) == 95.0, "pitch cột <2 bản ghi -> trang")
    check(column_pitch([100.0, 195.0, 291.0], fallback=95.0) == 95.5, "pitch cột = trung vị diff")
    check(column_pitch([100.0, 400.0], fallback=95.0) == 95.0, "pitch cột lệch >1,6× trang -> trang")
    print(f"[recenter_f3g selftest]\nRESULT: {passed} passed, {fails} failed")
    return fails


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if selftest() else 0)
    print(__doc__)
