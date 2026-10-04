"""v_pb_nguon_goc_12 (04/10) — PHẢN BIỆN verdict K06 ("FD thích ứng sách TỐT HƠN phông in và FD chung?"): kiểm ý nghĩa thống kê trên p02 (n=189, B34, top-8).

Dựng lại độc lập phần chấm của p02 (p02:130-176) từ plan.pkl + ảnh sinh đã lưu (p02_B34/gen/s0), CPU, KHÔNG ghi đè ket_qua.json của p02:
  Top-1 f_font / f_fd / f_gen_t0 / f_gen_t2 (như p02), rồi McNemar chính xác theo ô: gen vs phông, gen vs FD kho, và chia theo
  "ô mà MỌI ứng viên đều có ảnh sinh" (cùng đối xử) vs "ô có ứng viên thiếu ảnh sinh" (chấm NaN->-9 như p02).
Ra: measure_out/_tn11/verify/pb_nguon_goc/p02_font_vs_sinh.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

torch.set_num_threads(2)
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
b = "B34"
d = REPO / "measure_out/_tn11/p02_B34"
P = pd.read_pickle(d / "plan.pkl")
S = P["S"].copy()
E = T.OUT / "emb" / f"{b}_enc.npy"
E = np.load(E).astype(np.float32)
E = E / np.maximum(np.linalg.norm(E, axis=-1, keepdims=True), 1e-9)
from pipeline.gold_exact import signals_img as SI
from pipeline.gold_exact.common import Assets
enc = SI.Scorers(Assets(), "cpu", T.OUT / "emb", False, lambda m: None).enc()
k3 = np.ones((3, 3), np.uint8)


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


G = {}
for thin in (0, 2):
    ims, cs = [], []
    for c in P["chars"]:
        f = d / "gen" / "s0" / f"U+{ord(c):04X}.png"
        if f.exists():
            im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if thin:
                im = cv2.dilate(im, k3, iterations=thin)
            ims.append(im); cs.append(c)
    G[thin] = dict(zip(cs, nrm(np.asarray(enc.embed(ims, norm=False), np.float32))))
ci = S.i.to_numpy(); cc = S.c.to_numpy()
for thin in (0, 2):
    S[f"f_gen_t{thin}"] = [float(E[i] @ G[thin][c]) if c in G[thin] else np.nan for i, c in zip(ci, cc)]
cov = S.groupby("i").y.max()
Sv = S[S.i.isin(cov[cov == 1].index)]
has_gen = Sv.groupby("i").f_gen_t2.apply(lambda s: bool(s.notna().all()))


def per_cell(col):
    x = Sv[["i", col, "y"]].copy()
    x[col] = x[col].fillna(-9)
    return x.loc[x.groupby("i")[col].idxmax()].set_index("i").y


R = {c: per_cell(c) for c in ("f_font", "f_fd", "f_gen_t0", "f_gen_t2", "f_vW")}


def mc(a, bb, mask=None):
    x, y = R[a], R[bb]
    if mask is not None:
        x, y = x[mask], y[mask]
    A = x.to_numpy() == 1; B = y.to_numpy() == 1
    bcnt = int((A & ~B).sum()); ccnt = int((~A & B).sum())
    p = float(binomtest(min(bcnt, ccnt), bcnt + ccnt, 0.5).pvalue) if bcnt + ccnt else 1.0
    return {"n": int(len(x)), "top1_a": round(float(A.mean()) * 100, 1), "top1_b": round(float(B.mean()) * 100, 1), "a_dung_b_sai": bcnt, "a_sai_b_dung": ccnt, "p": round(p, 4)}


idx_all = R["f_font"].index
m_full = has_gen.reindex(idx_all).fillna(False).to_numpy()
res = {"n_o": int(len(idx_all)), "n_o_moi_ung_vien_co_anh_sinh": int(m_full.sum()),
       "top1": {k: round(float((v == 1).mean()) * 100, 1) for k, v in R.items()},
       "tat_ca_o": {"gen_t2_vs_font": mc("f_gen_t2", "f_font"), "gen_t0_vs_font": mc("f_gen_t0", "f_font"), "gen_t2_vs_fd": mc("f_gen_t2", "f_fd"), "fd_vs_font": mc("f_fd", "f_font")},
       "chi_o_moi_ung_vien_co_anh_sinh": {"gen_t2_vs_font": mc("f_gen_t2", "f_font", m_full), "gen_t0_vs_font": mc("f_gen_t0", "f_font", m_full), "gen_t2_vs_fd": mc("f_gen_t2", "f_fd", m_full)}}
print(json.dumps(res, ensure_ascii=False, indent=1))
(OUT / "p02_font_vs_sinh.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
