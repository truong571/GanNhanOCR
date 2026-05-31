"""Phát hiện KHUNG chữ trang Hán-Nôm (bao đúng 9 cột, loại số 1-9 + số trang).

Module chuẩn dùng cho pipeline: crop ảnh sát thân chữ TRƯỚC khi gửi kinhhannom
để OCR không nhận dạng dư cột / kí tự rác.

Hàm chính:
    detect_frame_hybrid(img_bgr) -> (x0, y0, x1, y1)
    crop_to_frame(img_bgr, pad)  -> ảnh đã crop (np.ndarray)
    crop_path_to_frame(path, pad, out_path) -> out_path

Cơ chế (đã kiểm trên 445 trang/3 cuốn — QA 100%):
  contour (viền in) ∪ textbbox  -> bao trùm, không sót cột/dòng
   -> refine_to_text:
        • Y: bỏ run mỏng ở rìa (số 1-9 / số trang) theo ngưỡng TƯƠNG ĐỐI 0.20*thân
        • _trim_marker_band: loại số DÍNH SÁT chữ bằng connected-components
          (số = đốm nhỏ, chữ = đốm lớn) — xử lý cả trang mật độ tăng đều
        • X: bỏ run dọc mỏng ở rìa (viền in / mực thấm) -> không "dư cột"
  Trang khổ lớn có vệt đen scan -> _remove_scan_border rồi detect lại.
"""
from __future__ import annotations

import cv2
import numpy as np


def _otsu(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def detect_frame_contour(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Khung viền in: nét dọc/ngang DÀI nhất (morphology + contour lớn nhất)."""
    H, W = img_bgr.shape[:2]
    bw = _otsu(img_bgr)
    horiz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, W // 20), 1)))
    vert = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(50, H // 20))))
    frame_mask = cv2.bitwise_or(horiz, vert)
    frame_mask = cv2.dilate(frame_mask, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        best, best_area = None, 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            a = w * h
            if 0.20 * H * W < a < 0.97 * H * W and a > best_area:
                best, best_area = (x, y, x + w, y + h), a
        if best:
            return best
    ys, xs = np.where(bw > 0)
    if len(xs) == 0:
        return (0, 0, W, H)
    return (max(0, int(np.percentile(xs, 1))), max(0, int(np.percentile(ys, 1))),
            min(W, int(np.percentile(xs, 99))), min(H, int(np.percentile(ys, 99))))


def detect_frame_textbbox(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Bao quanh vùng có chữ (dilate gom thành 1 blob, lấy contour lớn nhất)."""
    H, W = img_bgr.shape[:2]
    bw = _otsu(img_bgr)
    margin = max(int(0.02 * min(H, W)), 8)
    bw[:margin, :] = 0; bw[-margin:, :] = 0; bw[:, :margin] = 0; bw[:, -margin:] = 0
    dil = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)), iterations=2)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, W, H
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, x + w, y + h


def _split_runs(active: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Gom chỉ số True thành các 'run'; tách khi khoảng trống >= min_gap."""
    runs = []
    in_run = False
    s = gap = 0
    for i, a in enumerate(active):
        if a:
            if not in_run:
                s = i; in_run = True
            gap = 0
        else:
            if in_run:
                gap += 1
                if gap >= min_gap:
                    runs.append((s, i - gap + 1))
                    in_run = False; gap = 0
    if in_run:
        runs.append((s, len(active)))
    return runs


def _trim_marker_band(bw: np.ndarray, ys: int, ye: int) -> tuple[int, int]:
    """Cắt dải số 1-9 (trên) / số trang (dưới) DÍNH SÁT thân chữ bằng
    connected-components: số + mũi tên = đốm NHỎ, chữ Nôm = đốm LỚN.
    Mép khung = đỉnh/đáy chữ-lớn. Giới hạn 15% mỗi đầu -> không sập khung."""
    span = ye - ys
    if span < 150:
        return ys, ye
    H, W = bw.shape
    cap = int(0.15 * span)
    reg = bw[ys:ye]
    n, lab, st, _ = cv2.connectedComponentsWithStats(reg, 8)
    heights = [st[i][3] for i in range(1, n)
               if st[i][4] > 0.00008 * W * span and st[i][3] > 5]
    if len(heights) < 5:
        return ys, ye
    char_h = float(np.median(heights))
    h_min = 0.5 * char_h
    a_min = 0.00015 * W * span
    tops = [st[i][1] for i in range(1, n) if st[i][3] >= h_min and st[i][4] >= a_min]
    bots = [st[i][1] + st[i][3] for i in range(1, n)
            if st[i][3] >= h_min and st[i][4] >= a_min]
    if not tops:
        return ys, ye
    top, bot = min(tops), max(bots)
    if 0 < top <= cap:
        ys = ys + top
    if 0 < (span - bot) <= cap:
        ye = ye - (span - bot)
    return ys, ye


def refine_to_text(img_bgr: np.ndarray, bbox: tuple[int, int, int, int],
                   ink_ratio_th: float = 0.020,
                   min_gap_ratio: float = 0.004) -> tuple[int, int, int, int]:
    """Co bbox về đúng khối 9 cột chữ, loại số 1-9 (trên), số trang (dưới),
    viền in dọc + mực thấm (2 mép) -> không dư cột / không cắt chữ."""
    x0, y0, x1, y1 = bbox
    crop = img_bgr[y0:y1, x0:x1]
    H, W = crop.shape[:2]
    if H < 50 or W < 50:
        return bbox
    bw = _otsu(crop)

    # ----- TRỤC Y: span run-sống đầu->cuối, bỏ run nhỏ ở rìa (số 1-9/số trang)
    h_proj = (bw > 0).sum(axis=1).astype(np.float32)
    max_h = h_proj.max() or 1
    active_y = h_proj > (ink_ratio_th * max_h)
    runs_y = _split_runs(active_y, max(3, int(min_gap_ratio * H)))
    if not runs_y:
        ys, ye = 0, H
    else:
        largest = max(e - s for s, e in runs_y)
        min_keep = max(40, int(0.20 * largest))
        while len(runs_y) > 1 and (runs_y[0][1] - runs_y[0][0]) < min_keep:
            runs_y.pop(0)
        while len(runs_y) > 1 and (runs_y[-1][1] - runs_y[-1][0]) < min_keep:
            runs_y.pop()
        ys, ye = runs_y[0][0], runs_y[-1][1]

    # Loại số DÍNH SÁT chữ (khe < min_gap) bằng connected-components
    ys, ye = _trim_marker_band(bw, ys, ye)

    # ----- TRỤC X: bỏ run dọc RẤT hẹp ở rìa = viền in / mực thấm (giữ cột mờ)
    v_proj = (bw > 0).sum(axis=0).astype(np.float32)
    max_v = v_proj.max() or 1
    x_runs = _split_runs(v_proj > (0.02 * max_v), max(3, int(0.004 * W)))
    if not x_runs:
        xs, xe = 0, W
    else:
        med_w = np.median([e - s for s, e in x_runs])
        keep_w = max(15, 0.10 * med_w)
        while len(x_runs) > 1 and (x_runs[0][1] - x_runs[0][0]) < keep_w:
            x_runs.pop(0)
        while len(x_runs) > 1 and (x_runs[-1][1] - x_runs[-1][0]) < keep_w:
            x_runs.pop()
        xs, xe = x_runs[0][0], x_runs[-1][1]

    return x0 + xs, y0 + ys, x0 + xe, y0 + ye


def _remove_scan_border(img_bgr: np.ndarray) -> np.ndarray:
    """Tẩy mảng ĐEN lớn/dài chạm biên ảnh (vệt scan) -> trắng (trang khổ lớn)."""
    H, W = img_bgr.shape[:2]
    bw = _otsu(img_bgr)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    mask = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        touch = (x <= 3 or y <= 3 or x + w >= W - 3 or y + h >= H - 3)
        if touch and (area > 0.01 * H * W or w > 0.6 * W or h > 0.6 * H):
            mask[lab == i] = 255
    out = img_bgr.copy()
    out[mask > 0] = 255
    return out


def detect_frame_hybrid(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Khung cuối: contour ∪ textbbox -> refine_to_text. Trả (x0, y0, x1, y1)."""
    H, W = img_bgr.shape[:2]
    cb = detect_frame_contour(img_bgr)
    tb = detect_frame_textbbox(img_bgr)
    c_ratio = (cb[2] - cb[0]) * (cb[3] - cb[1]) / (W * H)
    t_ratio = (tb[2] - tb[0]) * (tb[3] - tb[1]) / (W * H)

    if t_ratio > 0.92:  # textbbox ô nhiễm
        if 0.30 <= c_ratio <= 0.92:
            return refine_to_text(img_bgr, cb)
        cleaned = _remove_scan_border(img_bgr)   # trang khổ lớn vệt đen scan
        tb2 = detect_frame_textbbox(cleaned)
        if (tb2[2] - tb2[0]) * (tb2[3] - tb2[1]) / (W * H) <= 0.92:
            return refine_to_text(cleaned, tb2)
        return refine_to_text(img_bgr, (0, 0, W, H))

    bb = (min(cb[0], tb[0]), min(cb[1], tb[1]),
          max(cb[2], tb[2]), max(cb[3], tb[3]))
    return refine_to_text(img_bgr, bb)


def crop_to_frame(img_bgr: np.ndarray, pad: int = 12) -> np.ndarray:
    """Crop ảnh về khung chữ (đã nới pad px, kẹp trong ảnh)."""
    H, W = img_bgr.shape[:2]
    x0, y0, x1, y1 = detect_frame_hybrid(img_bgr)
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    return img_bgr[y0:y1, x0:x1].copy()


def crop_path_to_frame(image_path: str, pad: int = 12,
                       out_path: str | None = None) -> str | None:
    """Đọc ảnh, crop về khung, lưu ra out_path (mặc định <stem>_framecrop.png).
    Trả đường dẫn ảnh đã crop, hoặc None nếu đọc lỗi."""
    from pathlib import Path
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None
    crop = crop_to_frame(bgr, pad=pad)
    if out_path is None:
        p = Path(image_path)
        out_path = str(p.with_name(p.stem + "_framecrop.png"))
    cv2.imwrite(out_path, crop)
    return out_path
