"""v_kythuat (03/10): độ PHỦ của tín hiệu f_fd (ảnh FontDiffuser sinh sẵn) trong bảng ứng viên TN8, và so f_fd với f_font CÙNG tập ứng viên.

Phát hiện: gannhanocr-fd/ có 89.898 tệp tên U+XXXX.png nhưng gần hết là con trỏ git-LFS (< 1 KB; mã Glyphs/signals_img bỏ qua),
nên f_fd chỉ có cho ~1,5 nghìn chữ (ArcFace/data/glyphs 1.564 + 85 tệp thật). Trong top1() của p01/p05 giá trị thiếu được điền -9
=> ứng viên không có ảnh FD luôn thua => Top-1 f_fd bị chặn bởi độ phủ, không phản ánh chất lượng ảnh FD.
Script (CPU, chỉ đọc pickle TN8 + thống kê tệp) tính:
  a) số tệp kho FD: tổng / ảnh thật (>= 1 KB) / con trỏ LFS; số chữ ứng viên có FD ở mỗi bộ; phủ theo dòng và theo chữ ĐÚNG;
  b) Top-1 CÙNG TẬP: chỉ xét ô mà chữ đúng có FD và >= 3 ứng viên có FD; xếp hạng CHỈ trong ứng viên có FD, so f_fd với f_font
     (cùng ô, cùng tập ứng viên => không còn thiên lệch phủ); kèm Top-1 ngẫu nhiên kỳ vọng (1/số ứng viên);
  c) Top-1 f_fd theo cách p01 (điền -9) trên toàn bộ ô có sự thật, để thấy trần do độ phủ: tỉ lệ ô mà chữ đúng có FD.
Ra: measure_out/_tn11/verify/kythuat/fd_phu.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CAND = REPO / "measure_out" / "_tn8" / "cand"
OUT = REPO / "measure_out" / "_tn11" / "verify" / "kythuat"
OUT.mkdir(parents=True, exist_ok=True)

res = {}
n_all = n_real = 0
for r, _, fs in os.walk(REPO / "gannhanocr-fd"):
    for f in fs:
        if f.startswith("U+") and f.endswith(".png"):
            n_all += 1
            n_real += os.path.getsize(os.path.join(r, f)) >= 1024
glyph_dir = len(list((REPO / "ArcFace/data/glyphs").glob("**/U+*.png"))) if (REPO / "ArcFace/data/glyphs").exists() else None
glyph_dir_real = sum(p.stat().st_size >= 1024 for p in (REPO / "ArcFace/data/glyphs").glob("**/U+*.png")) if glyph_dir else None
res["kho_fd"] = dict(tep_gannhanocr_fd=n_all, anh_that_ge_1KB=n_real, con_tro_LFS=n_all - n_real,
                     ArcFace_data_glyphs=glyph_dir, ArcFace_data_glyphs_that=glyph_dir_real)

res["bo"] = {}
for b in ("B18", "B34", "L16", "TK"):
    F = pd.read_pickle(CAND / f"{b}.pkl")[["i", "c", "y", "f_font", "f_fd"]]
    hy = F.groupby("i").y.max()
    F = F[F.i.isin(hy[hy == 1].index)].copy()                       # ô có sự thật trong ứng viên
    n_cells = F.i.nunique()
    true_has = F[(F.y == 1) & F.f_fd.notna()].i.unique()
    d = dict(o_co_su_that=int(n_cells), phu_dong=float(F.f_fd.notna().mean()), phu_chu_dung=float(len(true_has) / n_cells),
             chu_UV_khac_nhau=int(F.c.nunique()), chu_UV_co_FD=int(F[F.f_fd.notna()].c.nunique()))
    # (c) Top-1 kiểu p01 (điền -9) trên mọi ô
    x = F.copy(); x["f_fd9"] = x.f_fd.fillna(-9); x["f_font9"] = x.f_font.fillna(-9)
    for col in ("f_font9", "f_fd9"):
        t = x.loc[x.groupby("i")[col].idxmax()].set_index("i").y
        d[f"top1_{col}_kieu_p01"] = float(t.mean())
    # (b) cùng tập: ô mà chữ đúng có FD và >= 3 ứng viên có FD; xếp hạng chỉ trong ứng viên có FD & có f_font
    G = F[F.f_fd.notna() & F.f_font.notna()]
    cnt = G.groupby("i").size()
    ok_cells = set(cnt[cnt >= 3].index) & set(G[G.y == 1].i.unique())
    H = G[G.i.isin(ok_cells)]
    d["o_cung_tap"] = int(len(ok_cells))
    d["ung_vien_tb_cung_tap"] = float(H.groupby("i").size().mean())
    d["top1_ngau_nhien_ky_vong"] = float((1.0 / H.groupby("i").size()).mean())
    for col in ("f_font", "f_fd"):
        t = H.loc[H.groupby("i")[col].idxmax()].set_index("i").y
        d[f"top1_{col}_cung_tap"] = float(t.mean())
    # cùng tập, kết hợp z-score hai tín hiệu (xem phần bổ trợ)
    def z(col):
        g = H.groupby("i")[col]
        return (H[col] - g.transform("mean")) / g.transform("std").replace(0, 1)
    H = H.assign(zsum=z("f_font").fillna(0) + z("f_fd").fillna(0))
    t = H.loc[H.groupby("i")["zsum"].idxmax()].set_index("i").y
    d["top1_font_cong_fd_cung_tap"] = float(t.mean())
    # CI 95 % bootstrap ô cho chênh FD - font (cùng tập, cùng ô)
    pc = lambda col: H.loc[H.groupby("i")[col].idxmax()].set_index("i").y
    dd = (pc("f_fd") - pc("f_font")).to_numpy()
    rng = np.random.default_rng(0)
    bs = [dd[rng.integers(0, len(dd), len(dd))].mean() * 100 for _ in range(2000)]
    d["chenh_FD_tru_font_cung_tap_CI95"] = [round(float(dd.mean() * 100), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
    res["bo"][b] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
    print(b, json.dumps(res["bo"][b], ensure_ascii=False), flush=True)
(OUT / "fd_phu.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res["kho_fd"], ensure_ascii=False))
