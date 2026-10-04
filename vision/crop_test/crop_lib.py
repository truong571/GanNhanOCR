#!/usr/bin/env python3
"""Thư viện thử nghiệm crop bằng SO ẢNH (0 yêu cầu Vision, chạy cục bộ).

Sao ĐÚNG đường cắt của pipeline (build_dataset.save_crop): cửa sổ bbox ± pad → xoá mực hàng xóm (carve_neighbor_ink) → tighten_box → ảnh crop
(điểm ảnh GỐC nếu sách khai crop_source=original). Chỉ thay tham số cửa sổ/bbox ⇒ so ĐƠN GIẢN được "crop cũ" với "crop mới" trên cùng ô.
Đo "độ giống glyph nhãn" bằng đúng bộ nhúng của pipeline (MultiEnc v1+v2, norm=True cho crop, norm=False cho glyph phông) và đúng công thức M-OCR:
  s(c) = e_crop · e_font(c);  m_hom = s(nhãn) − max_{c ∈ R(âm)∖{nhãn}, glyph khác} s(c);  top1 ⇔ m_hom > 0.
KHÔNG sửa pipeline/, core/, config/, data/, dataset/.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                 # .../vision/crop_test
VISION = HERE.parent
REPO = VISION.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(VISION))

from pipeline.align_engine.bbox_fix import carve_neighbor_ink, tighten_box        # noqa: E402
from pipeline.align_engine.build_dataset import _paper_bg, load_original_page      # noqa: E402

# khoá sách → (thư mục prepared, bảng ô, nguồn pixel crop, mã sách trong bảng)
BOOKS = {
    "L16": dict(prep="LucVanTien1916", table="prepared/LucVanTien1916/dataset_out/labels_gated.csv", src="original", ihr=True),
    "TK": dict(prep="TruyenKieu1872", table="prepared/TruyenKieu1872/dataset_out/labels_gated.csv", src="original", ihr=True),
    "KVK": dict(prep="KimVanKieu1884", table="prepared/KimVanKieu1884/dataset_out/labels_gated.csv", src="original", ihr=False),
    "L83": dict(prep="LucVanTien1883", table="prepared/LucVanTien1883/dataset_out/labels_gated.csv", src="original", ihr=False),
    "Chr": dict(prep="Chrestomathie1872", table="prepared/Chrestomathie1872/dataset_out/labels_gated.csv", src="original", ihr=False),
    "stt2": dict(prep="SachThanhTruyen2", table="dataset_out/labels.csv", src="processed", ihr=False, book="stt2"),
    "stt4": dict(prep="SachThanhTruyen4", table="dataset_out/labels.csv", src="processed", ihr=False, book="stt4"),
    "stt11": dict(prep="SachThanhTruyen11", table="dataset_out/labels.csv", src="processed", ihr=False, book="stt11"),
}
PAD = 0.12


# ───────────────────────── bảng ô ─────────────────────────
def load_cells(key: str) -> pd.DataFrame:
    """Mọi ô (mọi tầng) của sách: page, column, x0..y1, tier, label, ocr_char, syllable, syl_idx, image (+ gt người cho L16/TK) + prev/next bbox cùng cột theo thứ tự y."""
    b = BOOKS[key]
    d = pd.read_csv(REPO / b["table"], dtype=str, keep_default_na=False)
    if "book" in b:
        d = d[d["book"] == b["book"]]
    d = d[d["bbox"] != ""].copy()
    bb = d["bbox"].map(json.loads)
    for i, nm in enumerate(["x0", "y0", "x1", "y1"]):
        d[nm] = bb.map(lambda v, i=i: int(round(float(v[i]))))
    d["column"] = d["column"].astype(int)
    d["syl_idx"] = pd.to_numeric(d.get("syl_idx", ""), errors="coerce")
    d["book_key"] = key
    if b.get("ihr"):
        ih = pd.read_csv(REPO / "measure_out" / b["prep"] / "ihr_endtoend" / "cells.csv", usecols=["page", "column", "syl_idx", "gt"], dtype={"page": str})
        ih["column"] = ih["column"].astype(int)
        d = d.merge(ih, on=["page", "column", "syl_idx"], how="left")
    else:
        d["gt"] = np.nan
    d = d.reset_index(drop=True)
    prev, nxt = {}, {}
    for (pg, col), g in d.groupby(["page", "column"]):
        g = g.assign(cy=(g.y0 + g.y1) / 2.0).sort_values("cy")
        idx = list(g.index)
        boxes = [(int(d.at[i, "x0"]), int(d.at[i, "y0"]), int(d.at[i, "x1"]), int(d.at[i, "y1"])) for i in idx]
        for k, i in enumerate(idx):
            prev[i] = boxes[k - 1] if k > 0 else None
            nxt[i] = boxes[k + 1] if k < len(idx) - 1 else None
    d["prev_bbox"] = pd.Series(prev)
    d["next_bbox"] = pd.Series(nxt)
    return d


class PageCache:
    """Một trang: ảnh đã xử lý (hình học) + xám + ảnh quét gốc (nếu sách khai crop_source=original)."""

    def __init__(self, key: str):
        self.key, self.cfg = key, BOOKS[key]
        self.pdir = REPO / "prepared" / self.cfg["prep"]
        self.cur = None

    def get(self, page: str):
        if self.cur and self.cur[0] == page:
            return self.cur[1]
        img = cv2.imread(str(self.pdir / "pages" / f"{page}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"thiếu trang {page}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orig = None
        if self.cfg["src"] == "original":
            orig = load_original_page(self.pdir, page, shape_like=img.shape[:2])
            if orig is None:
                raise FileNotFoundError(f"không tra được ảnh quét gốc {page}")
        self.cur = (page, dict(img=img, gray=gray, orig=orig))
        return self.cur[1]


# ───────────────────────── cắt crop (sao save_crop) ─────────────────────────
def _window(shape, bbox, pad_x, pad_y):
    H, W = shape[:2]
    ox1, oy1, ox2, oy2 = (int(v) for v in bbox)
    pw, ph = int((ox2 - ox1) * pad_x), int((oy2 - oy1) * pad_y)
    return max(0, ox1 - pw), max(0, oy1 - ph), min(W, ox2 + pw), min(H, oy2 + ph)


def cut_window(pg, bbox, window, prev_bbox, next_bbox, carve=True):
    """Cắt cửa sổ + xoá mực hàng xóm (CHƯA tighten). → (crop BGR đã carve, crop_o gốc đã carve | None)."""
    x1, y1, x2, y2 = window
    img, gray_full, orig = pg["img"], pg["gray"], pg["orig"]
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    crop = crop.copy()
    crop_o = orig[y1:y2, x1:x2].copy() if orig is not None else None
    oy1, oy2 = int(bbox[1]), int(bbox[3])
    if carve and gray_full is not None and (prev_bbox is not None or next_bbox is not None):
        crop = carve_neighbor_ink(crop, gray_full, x1, y1, x2, y2, (oy1, oy2), prev_bbox, next_bbox)
        if crop_o is not None:
            crop_o = carve_neighbor_ink(crop_o, gray_full, x1, y1, x2, y2, (oy1, oy2), prev_bbox, next_bbox, bg=_paper_bg(crop_o))
    return crop, crop_o


def finish(crop, crop_o):
    """tighten_box + chọn pixel giao nộp → ảnh xám uint8 (đúng thứ pipeline nhúng: tệp PNG đọc IMREAD_GRAYSCALE)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    tb = tighten_box(gray)
    if tb is not None:
        a, c, b, d = tb
        crop = crop[c:d, a:b]
        crop_o = crop_o[c:d, a:b] if crop_o is not None else None
    if crop.size == 0:
        return None
    out = crop_o if crop_o is not None else crop
    if out.ndim != 3:
        return out
    ok, buf = cv2.imencode(".png", out)                         # đúng đường pipeline: tệp PNG giao nộp rồi decode_gray (IMREAD_GRAYSCALE) — tránh lệch làm tròn ±1
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def make_crop(pg, bbox, prev_bbox, next_bbox, pad_x=PAD, pad_y=PAD, window=None, carve=True):
    w = window if window is not None else _window(pg["img"].shape, bbox, pad_x, pad_y)
    crop, crop_o = cut_window(pg, bbox, w, prev_bbox, next_bbox, carve)
    return None if crop is None else finish(crop, crop_o)


def edge_touch(crop, window, shape, rows=2, min_px=3):
    """Mực chạm mép TRÊN/DƯỚI cửa sổ (sau carve): dấu hiệu chữ bị cắt. Mép trùng biên trang thì không tính."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    bw = gray < 128
    top = bool(window[1] > 0 and bw[:rows].sum() >= min_px)
    bot = bool(window[3] < shape[0] and bw[-rows:].sum() >= min_px)
    return top, bot


def make_crop_adaptive(pg, bbox, prev_bbox, next_bbox, pad_x=PAD, start=PAD, step=0.06, cap=0.45, carve=True):
    """Luật KHÔNG cần Vision: nới pad dọc từng bước 0,06 (tối đa 0,45) chừng nào mực còn chạm mép trên/dưới cửa sổ."""
    py = start
    while True:
        w = _window(pg["img"].shape, bbox, pad_x, py)
        crop, crop_o = cut_window(pg, bbox, w, prev_bbox, next_bbox, carve)
        if crop is None:
            return None, py
        top, bot = edge_touch(crop, w, pg["img"].shape)
        if (not (top or bot)) or py + 1e-9 >= cap:
            return finish(crop, crop_o), py
        py = round(py + step, 4)


def vision_union_window(pg, bbox, vbox, margin=0.04, pad=PAD):
    """(ORACLE dùng Vision) cửa sổ = hợp của cửa sổ chuẩn và hộp glyph Vision nở margin."""
    x1, y1, x2, y2 = _window(pg["img"].shape, bbox, pad, pad)
    H, W = pg["img"].shape[:2]
    vx0, vy0, vx1, vy1 = vbox
    mw, mh = (vx1 - vx0) * margin, (vy1 - vy0) * margin
    return (max(0, min(x1, int(vx0 - mw))), max(0, min(y1, int(vy0 - mh))), min(W, max(x2, int(np.ceil(vx1 + mw)))), min(H, max(y2, int(np.ceil(vy1 + mh)))))


def shifted_bbox(bbox, dx_px, dy_px):
    x0, y0, x1, y1 = bbox
    return (int(round(x0 + dx_px)), int(round(y0 + dy_px)), int(round(x1 + dx_px)), int(round(y1 + dy_px)))


# ───────────────────────── nhúng + điểm M-OCR ─────────────────────────
class Scorer:
    def __init__(self, device=None):
        import torch
        from PIL import ImageFont
        from fontTools.ttLib import TTFont
        from pipeline.gold_exact.signals_img import ENC_FILES, FONT_FILES, MultiEnc
        from pipeline.gold_exact.common import R_of
        self.R_of = R_of
        dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.enc = MultiEnc([str(REPO / ENC_FILES["v1"]), str(REPO / ENC_FILES["v2"])], dev, True)
        self.fonts = []
        for f in FONT_FILES:
            p = REPO / f
            if p.exists():
                self.fonts.append((ImageFont.truetype(str(p), 84), set(TTFont(str(p)).getBestCmap())))
        self.FE, self.gmd5 = {}, {}

    def glyph_img(self, ch):
        from PIL import Image, ImageDraw
        for f, cmap in self.fonts:
            if ord(ch) in cmap:
                im = Image.new("L", (112, 112), 255)
                ImageDraw.Draw(im).text((56, 56), ch, font=f, fill=0, anchor="mm")
                return np.array(im)
        return None

    def ensure_glyphs(self, chars):
        new, ims = [], []
        for ch in chars:
            if ch in self.FE or ch in self.gmd5:
                continue
            im = self.glyph_img(ch)
            if im is None:
                self.gmd5[ch] = None
                continue
            self.gmd5[ch] = hashlib.md5(im.tobytes()).hexdigest()
            new.append(ch)
            ims.append(im)
        for s in range(0, len(ims), 512):
            E = self.enc.embed(ims[s:s + 512], norm=False)
            for ch, e in zip(new[s:s + 512], E):
                self.FE[ch] = e

    def embed(self, grays, bs=512):
        out = []
        for s in range(0, len(grays), bs):
            out.append(self.enc.embed(grays[s:s + bs]))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def metrics(self, E, labels, syllables):
        """→ (s_label, m_hom, top1) mảng float; NaN khi nhãn không có glyph phông hoặc không có đối thủ."""
        n = len(labels)
        s_lab, m = np.full(n, np.nan), np.full(n, np.nan)
        for i in range(n):
            L = labels[i]
            if not isinstance(L, str) or len(L) != 1 or self.gmd5.get(L) is None or L not in self.FE:
                continue
            s_lab[i] = float(E[i] @ self.FE[L])
            C0 = set(self.R_of(syllables[i])) - {L}
            same = {c for c in C0 if self.gmd5.get(c) == self.gmd5.get(L)}
            cf = sorted(c for c in C0 - same if c in self.FE)
            if cf:
                sc = np.stack([self.FE[c] for c in cf]) @ E[i]
                m[i] = s_lab[i] - float(sc.max())
        top1 = np.where(np.isnan(m), np.nan, (m > 0).astype(float))
        return s_lab, m, top1

    def prepare(self, labels, syllables):
        need = set()
        for L, s in zip(labels, syllables):
            if isinstance(L, str) and len(L) == 1:
                need.add(L)
                need |= set(self.R_of(s))
        self.ensure_glyphs(sorted(need))


# ───────────────────────── thống kê ─────────────────────────
def cluster_boot(vals, clusters, B=5000, seed=17):
    """Trung bình của vals với KTC95 % bootstrap theo CỤM (trang). NaN bị bỏ."""
    v = np.asarray(vals, float)
    c = np.asarray(clusters)
    ok = ~np.isnan(v)
    v, c = v[ok], c[ok]
    if len(v) == 0:
        return (float("nan"),) * 3 + (0,)
    u, inv = np.unique(c, return_inverse=True)
    S = np.bincount(inv, weights=v)
    N = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    means = S[idx].sum(1) / N[idx].sum(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), int(len(v))
