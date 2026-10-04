"""Rà soát GÓI: vẽ THẬT (pygame/ttf2im của font_diffusion, CPU, không GPU) MỌI cặp (phông được gán, chữ) trong kế hoạch và kiểm
  - không rỗng (mực > 0), không trùng pixel với glyph .notdef của chính phông đó (cmap có mã nhưng vẽ ra ô trống/hộp thiếu chữ);
  - thống kê mực theo phông; lỗi vẽ (ttf2im trả None / ngoại lệ).
Chạy từ cwd = thư mục dưới review/ (font_diffusion ghi .log vào cwd).
Ra: measure_out/_tn11/full/review/chay_thu_dau_cuoi/00b_ve_het_chu.json (+ danh sách chữ nghi vấn)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import r_chay_thu_dau_cuoi_lib as L

os.chdir(L.R / "mps_cwd")                                   # .log của font_diffusion nằm ở đây
CODE = L.PACK / "code"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "font_diffusion"))
import numpy as np  # noqa: E402

K = L.load_runner()
rep = L.Rep("00b_ve_het_chu")
plan = K.load_plan(L.PACK)
fmap = json.loads((L.PACK / "fonts.json").read_text(encoding="utf-8"))
from src.tools.utils import load_ttf, ttf2im  # noqa: E402

fonts = {n: load_ttf(str(CODE / "font_diffusion" / "fonts" / fn)) for n, fn in fmap.items()}
pairs = defaultdict(set)                                    # phông -> tập chữ
for b, v in plan["books"].items():
    for c, i in zip(v["chars"], v["font_idx"]):
        pairs[plan["font_names"][i]].add(c)

# .notdef của từng phông: vẽ một mã chắc chắn không có (U+0378 chưa gán; nếu phông có thì thử U+FFFF)
notdef = {}
for n, f in fonts.items():
    for cp in (0x0378, 0xFFFF, 0x10FFFD):
        im = ttf2im(font=f, char=chr(cp))
        if im is not None:
            notdef[n] = (cp, np.array(im.convert("L")))
            break
t0 = time.time()
res = {}
susp = defaultdict(list)
for n, S in pairs.items():
    ink, blank, same_nd, none_ = [], 0, 0, 0
    for c in sorted(S, key=ord):
        try:
            im = ttf2im(font=fonts[n], char=c)
        except Exception as e:  # noqa: BLE001
            im = None
            susp[n].append((f"U+{ord(c):04X}", f"exc:{type(e).__name__}"))
        if im is None:
            none_ += 1; susp[n].append((f"U+{ord(c):04X}", "None")); continue
        a = np.array(im.convert("L"))
        ink_pct = float((a < 128).mean() * 100)
        ink.append(ink_pct)
        if ink_pct == 0:
            blank += 1; susp[n].append((f"U+{ord(c):04X}", "rong"))
        elif n in notdef and a.shape == notdef[n][1].shape and np.array_equal(a, notdef[n][1]):
            same_nd += 1; susp[n].append((f"U+{ord(c):04X}", "trung_notdef"))
    ink = np.array(ink) if ink else np.array([0.0])
    res[n] = dict(so_chu=len(S), ve_none=none_, rong=blank, trung_notdef=same_nd, ink_pct_p1=round(float(np.percentile(ink, 1)), 2),
                  ink_pct_trung_vi=round(float(np.median(ink)), 2), ink_pct_p99=round(float(np.percentile(ink, 99)), 2))
rep.data("theo_phong", res)
rep.data("notdef_dung_ma", {n: f"U+{v[0]:04X}" for n, v in notdef.items()})
rep.data("nghi_van", {n: v[:12] for n, v in susp.items()})
tot = sum(r["so_chu"] for r in res.values())
bad = sum(r["ve_none"] + r["rong"] + r["trung_notdef"] for r in res.values())
rep.inv("V1.moi_cap_phong_chu_ve_duoc_khong_rong_khong_notdef", bad == 0, dict(cap_kiem=tot, nghi_van=bad, giay=round(time.time() - t0, 1),
                                                                               theo_phong={n: (r["so_chu"], r["ve_none"] + r["rong"] + r["trung_notdef"]) for n, r in res.items()}))
rep.save()
