"""v_kythuat (04/10): p02 "FD một-ảnh-phong-cách theo sách (34-38 %) kém kho FD (54 %)" — điều chỉnh độ PHỦ của f_fd (CPU, 0 API).

Trong p02 (eval_, dòng top1) giá trị thiếu được điền -9. Ảnh FD thật chỉ có cho ~1,6 nghìn chữ => f_fd chỉ "thắng" được khi chữ đúng CÓ ảnh FD.
Script tái lập p02 (cùng plan.pkl, cùng ảnh sinh gen/s0, cùng mã nhúng; invariant: f_font 29,6 / f_fd 54,0 / f_gen_t0 33,9 / f_gen_t2 37,6 của
p02_B34/ket_qua.json) rồi so CÙNG TẬP: chỉ ô mà chữ đúng có ảnh FD và >= 2 ứng viên có ảnh FD, xếp hạng CHỈ trong ứng viên có ảnh FD
(khi đó f_font, f_fd, f_gen cùng xếp một danh sách, không còn thiên lệch phủ). Ra: measure_out/_tn11/verify/kythuat/p02_cungtap.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 CUDA_VISIBLE_DEVICES= .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_kythuat_p02_cungtap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

torch.set_num_threads(2)
OUT = REPO / "measure_out" / "_tn11" / "verify" / "kythuat"
EMB = T.OUT / "emb"
P02 = REPO / "measure_out" / "_tn11" / "p02_B34"


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def top1_in(H, col, mask=None):
    x = H if mask is None else H[mask]
    x = x[["i", col, "y"]].copy(); x[col] = x[col].fillna(-9)
    return float(x.loc[x.groupby("i")[col].idxmax()].y.mean()) * 100


def main():
    b = "B34"
    P = pd.read_pickle(P02 / "plan.pkl"); S = P["S"].copy()
    E = nrm(np.load(EMB / f"{b}_enc.npy").astype(np.float32))
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    Sc = SI.Scorers(Assets(), "cpu", OUT / "emb_cache", False, lambda m: None)
    enc = Sc.enc()
    k3 = np.ones((3, 3), np.uint8)
    G = {}
    for thin in (0, 2):
        ims, cs = [], []
        for c in P["chars"]:
            f = P02 / "gen" / "s0" / f"U+{ord(c):04X}.png"
            if f.exists():
                im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if thin:
                    im = cv2.dilate(im, k3, iterations=thin)
                ims.append(im); cs.append(c)
        G[thin] = dict(zip(cs, nrm(np.asarray(enc.embed(ims, norm=False), np.float32))))
    ci = S.i.to_numpy(); cc = S.c.to_numpy()
    for thin in (0, 2):
        S[f"f_gen_t{thin}"] = [float(E[i] @ G[thin][c]) if c in G[thin] else np.nan for i, c in zip(ci, cc)]
    cov = S.groupby("i").y.max(); Sv = S[S.i.isin(cov[cov == 1].index)].copy()
    res = dict(o=int(Sv.i.nunique()), k=8)
    # invariant: tái lập p02
    ref = json.load(open(P02 / "ket_qua.json"))["top1_trong_top_k"]
    res["tai_lap"] = {c: dict(p02=ref[c], ban_sao=round(top1_in(Sv, c), 1)) for c in ("f_font", "f_fd", "f_gen_t0", "f_gen_t2") if c in Sv}
    res["tai_lap_pass"] = bool(all(abs(v["p02"] - v["ban_sao"]) < 0.8 for v in res["tai_lap"].values()))
    # độ phủ
    t = Sv[Sv.y == 1].drop_duplicates("i").set_index("i")
    res["phu"] = dict(dong_co_FD=round(float(Sv.f_fd.notna().mean()), 3), o_chu_dung_co_FD=round(float(t.f_fd.notna().mean()), 3),
                      o_chu_dung_co_FD_n=int(t.f_fd.notna().sum()))
    # cùng tập
    G2 = Sv[Sv.f_fd.notna()]
    cnt = G2.groupby("i").size()
    ok = set(cnt[cnt >= 2].index) & set(G2[G2.y == 1].i)
    H = G2[G2.i.isin(ok)].copy()
    res["cung_tap"] = dict(o=len(ok), ung_vien_tb=round(float(H.groupby("i").size().mean()), 2),
                           ngau_nhien=round(float((1.0 / H.groupby("i").size()).mean()) * 100, 1))
    for c in ("f_font", "f_fd", "f_gen_t0", "f_gen_t2", "f_vW"):
        if c in H:
            res["cung_tap"][c] = round(top1_in(H, c), 1)

    def z(col):
        g = H.groupby("i")[col]
        return (H[col] - g.transform("mean")) / g.transform("std").replace(0, 1)
    for a_, b_ in (("f_vW", "f_fd"), ("f_vW", "f_gen_t2")):
        H[f"{a_}+{b_}"] = z(a_).fillna(0) + z(b_).fillna(0)
        res["cung_tap"][f"{a_}+{b_}"] = round(top1_in(H, f"{a_}+{b_}"), 1)
    # CI bootstrap ô cho f_gen_t2 - f_fd trên tập chung
    rng = np.random.default_rng(0)
    pc = lambda col: H[["i", col, "y"]].fillna({col: -9}).loc[lambda d: d.groupby("i")[col].idxmax()].set_index("i").y
    d = (pc("f_gen_t2") - pc("f_fd")).to_numpy()
    bs = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(2000)]
    res["cung_tap"]["f_gen_t2_tru_f_fd_CI95"] = [round(float(d.mean() * 100), 1), round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]
    (OUT / "p02_cungtap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
