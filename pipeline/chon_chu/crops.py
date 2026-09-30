"""crops.py — crop MỌI ô (mọi tầng) đúng hình học `build_dataset.save_crop` (pad 0,12, carve mực chữ kề cùng cột, tighten,
điểm ảnh GỐC khi sách khai `crop_source: original`) — bản chép lab/thu_nghiem_kim/TN8_chon_chu/t02_embed.cut. Không ghi tệp
trừ khi gọi `write_crops` (chỉ cho ô được nâng GOLD mà chưa có tệp ảnh).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from pipeline.align_engine.bbox_fix import carve_neighbor_ink, tighten_box
from pipeline.align_engine.build_dataset import _paper_bg, load_original_page

PAD = 0.12


def cut(img, gray_full, bbox, prev_bbox, next_bbox, img_orig):
    """== save_crop (không ghi) -> dict(out = mảng ảnh sẽ giao (gốc nếu có), proc = crop xám đã xử lý) hoặc None."""
    H, W = img.shape[:2]
    ox1, oy1, ox2, oy2 = (int(v) for v in bbox)
    pw, ph = int((ox2 - ox1) * PAD), int((oy2 - oy1) * PAD)
    x1, y1 = max(0, ox1 - pw), max(0, oy1 - ph)
    x2, y2 = min(W, ox2 + pw), min(H, oy2 + ph)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = crop.copy()
    crop_o = img_orig[y1:y2, x1:x2].copy() if img_orig is not None else None
    if gray_full is not None and (prev_bbox is not None or next_bbox is not None):
        crop = carve_neighbor_ink(crop, gray_full, x1, y1, x2, y2, (oy1, oy2), prev_bbox, next_bbox)
        if crop_o is not None:
            crop_o = carve_neighbor_ink(crop_o, gray_full, x1, y1, x2, y2, (oy1, oy2), prev_bbox, next_bbox,
                                        bg=_paper_bg(crop_o))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    tb = tighten_box(gray)
    if tb is not None:
        a, c, b, d = tb
        crop = crop[c:d, a:b]; gray = gray[c:d, a:b]
        if crop_o is not None:
            crop_o = crop_o[c:d, a:b]
    if crop.size == 0:
        return None
    return dict(out=crop_o if crop_o is not None else crop, proc=gray)


def encode(arr) -> tuple[bytes, np.ndarray]:
    """(byte PNG của ảnh giao, ảnh xám đọc lại như scorer đọc tệp: IMREAD_GRAYSCALE)."""
    ok, buf = cv2.imencode(".png", arr)
    b = buf.tobytes()
    return b, cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _neighbours(recs):
    by_col = defaultdict(list)
    for i, col, bb in recs:
        if bb is not None:
            by_col[col].append((i, bb))
    nb = {}
    for col, L in by_col.items():
        L.sort(key=lambda t: (t[1][1] + t[1][3]) / 2.0)
        for k, (i, bb) in enumerate(L):
            nb[i] = (L[k - 1][1] if k > 0 else None, L[k + 1][1] if k < len(L) - 1 else None)
    return nb


def page_job(args):
    """args = (png, prep_dir, orig: bool, recs [(i, column, bbox)], want: set | None).
    Trả [(i, png_bytes, gray, (w, h, ink))] cho ô trong want (None = mọi ô). Hàng xóm luôn tính trên MỌI ô của trang."""
    png, prep, orig, recs, want = args
    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if img is None:
        return [(i, None, None, None) for i, *_ in recs if want is None or i in want]
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_orig = None
    if orig:
        img_orig = load_original_page(prep, Path(png).stem, shape_like=img.shape[:2])
        if img_orig is None:
            raise FileNotFoundError(f"crop_source original: thiếu ảnh gốc cùng kích thước cho {png} (prepared {prep})")
    nb = _neighbours(recs)
    out = []
    for i, col, bb in recs:
        if want is not None and i not in want:
            continue
        if bb is None:
            out.append((i, None, None, None)); continue
        pv, nx = nb.get(i, (None, None))
        r = cut(img, gray_full, bb, pv, nx, img_orig)
        if r is None:
            out.append((i, None, None, None)); continue
        b, g = encode(r["out"])
        p = r["proc"]
        out.append((i, b, g, (p.shape[1], p.shape[0], float((p < 128).mean()))))
    return out


def iter_pages(D, prep: Path, orig: bool, want=None, workers: int = 6):
    """Duyệt trang (song song) -> từng kết quả page_job. D cần cột page, column (int), bbox4 (list|None)."""
    jobs = []
    for pg, g in D.groupby("page", sort=True):
        recs = [(i, c, bb) for i, c, bb in zip(g.index, g.column_i, g.bbox4)]
        w = None if want is None else {i for i in g.index if i in want}
        if w is not None and not w:
            continue
        jobs.append((prep / "pages" / f"{pg}.png", prep, orig, recs, w))
    with ThreadPoolExecutor(workers) as ex:
        yield from ex.map(page_job, jobs)
