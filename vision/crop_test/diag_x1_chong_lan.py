#!/usr/bin/env python3
"""CHẨN ĐOÁN X1c — NGOÀI giao thức, làm SAU KHI thấy kết quả X1 (mô tả; đọc cột đã lưu của X1, 0 tính toán mới).
Hộp Vision (V) của ô 'bị cắt' ở L16 có chặt vào chữ không, hay lỏng và chồng lấn ký hiệu láng giềng?
  own_overlap = diện tích giao lớn nhất của hộp một ký hiệu CJK/PUA khác với V / diện tích V;  vh_/bh = cao V / cao hộp ô.
  .venv/bin/python vision/crop_test/diag_x1_chong_lan.py     → vision/ket_qua/crop_test/X1c_BAO_CAO.md"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "ket_qua" / "crop_test"
d = pd.read_csv(OUT / "X1_L16_vwindow.csv.gz", low_memory=False)
c, r = d[d.clipped.astype(bool)], d[~d.clipped.astype(bool)]
q = lambda x: " / ".join(f"{v:.2f}" for v in np.nanpercentile(x.dropna(), [10, 25, 50, 75, 90]))
L = [f"# X1c — hộp Vision của ô 'bị cắt' ở L16 chặt hay lỏng? (NGOÀI giao thức; mô tả) — {len(c)} ô bị cắt, {len(r)} ô Vision khớp còn lại\n",
     "Phân vị 10/25/50/75/90:\n",
     f"- chồng lấn lớn nhất của hộp ký hiệu khác với V (own_overlap): bị cắt {q(c.own_overlap)} | còn lại {q(r.own_overlap)}",
     f"- tỉ lệ ô có chồng lấn ≥ 0,10: bị cắt {100 * (c.own_overlap >= .10).mean():.1f} % | còn lại {100 * (r.own_overlap >= .10).mean():.1f} %; ≥ 0,30: bị cắt {100 * (c.own_overlap >= .30).mean():.1f} % | còn lại {100 * (r.own_overlap >= .30).mean():.1f} %",
     f"- cao V / cao hộp ô: bị cắt {q(c.vh_ / c.bh)} | còn lại {q(r.vh_ / r.bh)}",
     f"- rộng V / rộng hộp ô: bị cắt {q(c.vw_ / c.bw)} | còn lại {q(r.vw_ / r.bw)}\n",
     "Đọc (mô tả): nếu V cao hơn hộp ô nhiều và chồng lấn mạnh láng giềng thì 'V vượt cửa sổ' không có nghĩa là crop giao nộp cắt mất nét; cov (mực trong V) cũng gồm mực láng giềng nằm trong V."]
(OUT / "X1c_BAO_CAO.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
