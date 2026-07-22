"""Fix the frame-crop coordinate offset in cached SinoNom OCR bboxes.

THE BUG (pre-existing, affects production too): core/ocr/ocr_api.py runs the
HCMUS OCR on a FRAME-CROPPED image (framed=True, crop_to_frame with frame_pad),
so the cached character bboxes are in CROPPED-image coordinates. Everything
downstream (step2_align.py, align_engine align_production) crops the FULL page at
those bboxes -> the whole column grid is shifted left by the frame's left margin
(~252px median, ~1.7 columns). Result: leftmost columns fall into the margin
(blank crops, ~30%), interior crops capture the neighbouring glyph.

THE FIX (this module): crop_to_frame uses origin
    (x0, y0) = (max(0, frame_x0 - pad), max(0, frame_y0 - pad))
where frame_* = detect_frame_hybrid(img). So mapping a cropped-coord bbox back
to full-page coords is just `bbox + (x0, y0)`. We recompute (x0, y0) per page
with the SAME detector + the frame_pad stored in the cache.

Verified end-to-end: blank-crop rate 31% -> 0%, median crop ink 9.8% -> 18.1%
(pure translation, no scale). Written entirely in pipeline/align_engine — production
core/ is untouched.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.image.frame_detector import detect_frame_hybrid

# Per-page memo so detect_frame_hybrid runs once per page, not per call.
_offset_cache: dict[str, tuple[int, int]] = {}


def frame_offset(page_png: str, framed: bool, frame_pad: int) -> tuple[int, int]:
    """Return (ox, oy) to add to cached OCR bboxes -> full-page coords."""
    if not framed:
        return (0, 0)
    key = str(page_png)
    if key in _offset_cache:
        return _offset_cache[key]
    bgr = cv2.imread(key)
    if bgr is None:
        return (0, 0)
    try:
        fx0, fy0, _, _ = detect_frame_hybrid(bgr)
    except Exception:
        return (0, 0)
    off = (max(0, int(fx0) - int(frame_pad)), max(0, int(fy0) - int(frame_pad)))
    _offset_cache[key] = off
    return off


def correct_columns(columns: list, ox: int, oy: int) -> list:
    """In-place add (ox, oy) to every char bbox in the OCR columns."""
    if ox == 0 and oy == 0:
        return columns
    for col in columns:
        for c in col:
            b = c.get("bbox")
            if b and len(b) == 4:
                c["bbox"] = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
    return columns


def tighten_box(gray: "np.ndarray", thr: int = 128, edge_line: float = 0.65,
                margin: int = 4) -> tuple[int, int, int, int] | None:
    """Tighten a grayscale glyph crop to its ink via binary projection.

    Two cleanups, both projection-based:
      1) Strip COLUMN-RULE / border lines: an edge row/col that is mostly ink
         (> edge_line of its length) is the woodblock ruling line, not the
         glyph — trim it inward.
      2) Crop to the ink bounding box (rows/cols that contain any ink), + a
         small margin.

    Returns (x0, y0, x1, y1) within the input crop, or None if (near-)empty.
    The OCR per-char box is loose (captures ruling lines, neighbour slivers,
    whitespace); this re-centres on the actual glyph.
    """
    bw = gray < thr
    h, w = bw.shape
    if bw.sum() < 8:
        return None
    cs = bw.sum(axis=0)
    rs = bw.sum(axis=1)
    # 1) trim long thin border lines
    x0 = 0
    while x0 < int(w * 0.40) and cs[x0] > edge_line * h:
        x0 += 1
    x1 = w
    while x1 > int(w * 0.60) and cs[x1 - 1] > edge_line * h:
        x1 -= 1
    y0 = 0
    while y0 < int(h * 0.40) and rs[y0] > edge_line * w:
        y0 += 1
    y1 = h
    while y1 > int(h * 0.60) and rs[y1 - 1] > edge_line * w:
        y1 -= 1
    sub = bw[y0:y1, x0:x1]
    if sub.sum() < 8:
        return None
    # 2) ink bounding box inside the trimmed region
    ys = np.where(sub.sum(axis=1) > 0)[0]
    xs = np.where(sub.sum(axis=0) > 0)[0]
    a = max(0, x0 + int(xs.min()) - margin)
    b = min(w, x0 + int(xs.max()) + 1 + margin)
    c = max(0, y0 + int(ys.min()) - margin)
    d = min(h, y0 + int(ys.max()) + 1 + margin)
    if b - a < 6 or d - c < 6:
        return None
    return (a, c, b, d)


def _compute_seam(gray_box: "np.ndarray", y_lo: int, y_hi: int) -> "np.ndarray | None":
    """Đường đi NGANG năng-lượng-tối-thiểu (mực) qua dải [y_lo,y_hi] của gray_box.

    Mỗi cột x chọn 1 hàng y, |y(x)-y(x-1)| <= 1 (liền mạch); năng lượng = mực
    (255-gray) — seam luồn theo kẽ hở ít mực nhất giữa 2 ký tự chồng nhau theo
    chiều dọc. Trả seam[W] (toạ độ y TRONG gray_box), hoặc None nếu dải rỗng.

    Bản port thuần numpy của train_crop/infer_centernet.compute_seam — port
    thay vì import để module này không phải kéo theo torch (train_crop/infer_centernet.py
    nạp model_centernet/train_centernet ở top-level) chỉ để tính 1 đường seam.
    """
    H, W = gray_box.shape[:2]
    y_lo = max(0, min(y_lo, H - 1))
    y_hi = max(y_lo, min(y_hi, H - 1))
    if W < 2 or y_hi <= y_lo:
        return None
    ink = (255.0 - gray_box.astype(np.float32))[y_lo:y_hi + 1, :]
    R = ink.shape[0]
    M = np.empty((R, W), np.float32)
    back = np.zeros((R, W), np.int32)
    M[:, 0] = ink[:, 0]
    rows = np.arange(R)
    for x in range(1, W):
        prev = M[:, x - 1]
        up = np.concatenate(([np.inf], prev[:-1]))
        down = np.concatenate((prev[1:], [np.inf]))
        cand = np.stack([up, prev, down])
        arg = np.argmin(cand, axis=0)
        M[:, x] = ink[:, x] + cand[arg, rows]
        back[:, x] = np.clip(rows + (arg - 1), 0, R - 1)
    r = int(np.argmin(M[:, -1]))
    seam = np.empty(W, np.int32)
    for x in range(W - 1, -1, -1):
        seam[x] = y_lo + r
        r = int(back[r, x])
    return seam


def _seam_boundary(gray_full: "np.ndarray", x1: int, x2: int, ya: int, yb: int):
    """Seam (toạ độ y TUYỆT ĐỐI, hệ full-page) giữa 2 ký tự trong dải y∈[ya,yb], x∈[x1,x2]."""
    sub = gray_full[max(0, ya):yb, max(0, x1):x2]
    if sub.size == 0 or sub.shape[0] < 3 or sub.shape[1] < 2:
        return None
    seam = _compute_seam(sub, 0, sub.shape[0] - 1)
    return None if seam is None else (seam + max(0, ya))


def carve_neighbor_ink(crop: "np.ndarray", gray_full: "np.ndarray", cx1: int, cy1: int,
                       cx2: int, cy2: int, own_y: tuple[int, int],
                       prev_bbox=None, next_bbox=None, bg: int = 255) -> "np.ndarray":
    """Xoá (tô nền) mực của ký tự HÀNG XÓM (cùng cột, trên/dưới) tràn vào `crop`,
    theo seam cong tại ranh giới — bịt đúng lỗ hổng mà `tighten_box` không xử lý
    được: nó chỉ co về hộp bao mực, không tách được 2 khối mực DÍNH NHAU.

    crop bị SỬA TRỰC TIẾP (in-place, cũng trả về để tiện dùng). (cx1,cy1,cx2,cy2)
    = cửa sổ đã pad của `crop` trong hệ toạ độ full-page. own_y=(y1,y2) = bbox GỐC
    (chưa pad) của CHÍNH ký tự này — mốc chia đôi trên/dưới để biết seam nào
    giới hạn phía nào. prev_bbox/next_bbox = bbox GỐC của ký tự liền trước/sau
    trong CÙNG CỘT (None nếu không có, hoặc không hạn chế thêm).

    Logic giống hệt train_crop/infer_centernet.carve_crops() — port lại (áp lên
    cửa sổ pad theo TỶ LỆ của build_dataset thay vì pad cố định 2px của bản gốc).

    CHẶN CỨNG (khác bản gốc carve_crops): erasure KHÔNG BAO GIỜ được lấn vào
    [oy1,oy2] — bbox GỐC của chính ký tự — dù seam tìm được nằm sâu hơn. Lý do:
    khi 2 box hàng xóm CHỒNG LẤN nhau (lỗi định vị box có sẵn từ align, không
    hiếm), dải tìm seam có thể trùm cả lên nét của chính ký tự; nếu không chặn,
    đường seam "rẻ nhất" (ít mực nhất) có thể cắt xuyên qua giữa ký tự, xoá
    trắng gần hết hoặc toàn bộ crop (đo được: ink_pct=0.0 hàng loạt trước khi
    thêm chặn này). Bleed hàng xóm CHỈ có thể nằm trong phần PAD (ngoài
    [oy1,oy2]), nên giới hạn cả vùng tìm-seam lẫn vùng-được-xoá vào đúng phần
    pad là đủ để loại bỏ rủi ro này mà không mất tác dụng carve.
    """
    oy1, oy2 = own_y
    ch, cw = crop.shape[:2]
    if prev_bbox is not None and oy1 > cy1:
        ya = max(int((prev_bbox[1] + prev_bbox[3]) / 2), cy1)
        if ya < oy1:
            seam = _seam_boundary(gray_full, cx1, cx2, ya, oy1)
            if seam is not None and len(seam) == cw:
                top_limit = oy1 - cy1
                for j in range(cw):
                    yy = min(max(int(seam[j]) - cy1, 0), top_limit)
                    crop[:yy, j] = bg
    if next_bbox is not None and oy2 < cy2:
        yb = min(int((next_bbox[1] + next_bbox[3]) / 2) + 1, cy2)
        if yb > oy2:
            seam = _seam_boundary(gray_full, cx1, cx2, oy2, yb)
            if seam is not None and len(seam) == cw:
                bot_limit = oy2 - cy1
                for j in range(cw):
                    yy = max(min(int(seam[j]) - cy1, ch), bot_limit)
                    crop[yy:, j] = bg
    return crop
