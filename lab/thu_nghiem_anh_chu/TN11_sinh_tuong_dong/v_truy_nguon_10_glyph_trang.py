"""v_truy_nguon_10 (04/10) — chữ KHÔNG có trong NomNaTong của p05 được render ra ẢNH TRẮNG? 0 API, CPU, chỉ ĐỌC.

p05 (p05_full_corpus_he_quy_chieu.py:58-70) render_font_raw dùng font_diffusion/fonts/NomNaTong-Regular.ttf (v5.12) bằng PIL, không phông dự phòng.
Script sao nguyên hàm đó, lấy mọi chữ ứng viên của 4 bộ có nhãn người (cand/<bộ>.pkl), tách chữ CÓ / KHÔNG có trong cmap của font đó, rồi đếm số điểm mực
(< 128) của ảnh render: nếu mọi chữ thiếu glyph cho 0 điểm mực thì mọi ứng viên thiếu glyph có CÙNG một ảnh trắng (hòa điểm, thua mọi glyph thật).
Ra: measure_out/_tn11/verify/truy_nguon/glyph_trang.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import numpy as np
import pandas as pd
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "truy_nguon"
FONT = str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf")


def render_font_raw(char: str, size: int = 128) -> np.ndarray:        # sao nguyên văn p05:58-70
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT, int(size * 0.75))
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - w) // 2 - bbox[0]
        y = (size - h) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
    except Exception:
        draw.rectangle([(10, 10), (size - 10, size - 10)], outline=128, width=2)
    return np.array(img)


def main():
    cm = set(TTFont(FONT).getBestCmap())
    cm_pipe = set(TTFont(str(REPO / "fonts/NomNaTong-Regular.ttf")).getBestCmap())
    chars = set()
    for b in ("B18", "B34", "L16", "TK"):
        chars |= set(pd.read_pickle(T.OUT / "cand" / f"{b}.pkl", ).c.unique())
    chars = sorted(c for c in chars if len(c) == 1)
    miss = [c for c in chars if ord(c) not in cm]
    pres = [c for c in chars if ord(c) in cm]
    rng = np.random.default_rng(20261004)
    ms = list(rng.choice(miss, size=min(3000, len(miss)), replace=False))
    ps = list(rng.choice(pres, size=min(1000, len(pres)), replace=False))
    ink_m = np.array([int((render_font_raw(c) < 128).sum()) for c in ms])
    ink_p = np.array([int((render_font_raw(c) < 128).sum()) for c in ps])
    blank_imgs = {render_font_raw(c).tobytes() for c in ms}
    res = dict(so_chu_ung_vien_4_bo=len(chars), thieu_glyph_p05=len(miss), thieu_glyph_p05_pct=round(100 * len(miss) / len(chars), 2),
               co_trong_font_pipeline_fonts_NomNaTong_v518_nhung_thieu_o_p05=int(sum(1 for c in miss if ord(c) in cm_pipe)),
               mau_thieu=len(ms), mau_thieu_so_anh_khac_nhau=len(blank_imgs), mau_thieu_diem_muc_max=int(ink_m.max()),
               mau_thieu_ty_le_0_diem_muc_pct=round(100 * float((ink_m == 0).mean()), 2),
               mau_co=len(ps), mau_co_diem_muc_min=int(ink_p.min()), mau_co_ty_le_0_diem_muc_pct=round(100 * float((ink_p == 0).mean()), 2))
    (OUT / "glyph_trang.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
