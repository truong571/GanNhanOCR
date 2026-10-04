"""v_pb_kythuat (04/10) — PHẢN BIỆN độc lập p02: so ảnh SINH theo sách với PHÔNG trên đúng nhóm ô mà kho FD KHÔNG có ảnh (CPU, 0 API, không sửa tệp).

Người kiểm chứng (v_kythuat_p02_cungtap.py) chỉ so sinh-vs-FD trên tập CÙNG (ô mà chữ đúng CÓ ảnh FD, n=111): phông 61,3 / FD 62,2 / sinh 64,9.
Nhưng ý đồ của p02 là CHỮ HIẾM = chữ KHÔNG có ảnh FD (FD chỉ phủ ~1,6 nghìn chữ thường gặp) — nơi ảnh sinh theo sách là nguồn mẫu DUY NHẤT ngoài phông.
Phép so đúng cho nhóm đó là sinh vs PHÔNG (không dính độ phủ FD). Script tái lập điểm p02 (invariant: f_font 29,6 / f_fd 54,0 / f_gen_t0 33,9 / f_gen_t2 37,6)
rồi báo cáo, trên (a) cả 189 ô, (b) ô mà chữ đúng KHÔNG có ảnh FD, (c) ô mà chữ đúng CÓ ảnh FD:  Top-1 phông, sinh t0, sinh t2 trong top-8; chênh sinh−phông ghép cặp
theo ô + CI bootstrap 95 %; kèm Top-1 ngẫu nhiên kỳ vọng.
Ra: measure_out/_tn11/verify/pb_kythuat/p02_font_vs_sinh.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 nice -n 10 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_pb_kythuat_p02_font_vs_sinh.py
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

torch.set_num_threads(3)
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_kythuat"
EMB = T.OUT / "emb"
P02 = REPO / "measure_out" / "_tn11" / "p02_B34"


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def top1(H, col):
    x = H[["i", col, "y"]].copy()
    x[col] = x[col].fillna(-9)
    return x.loc[x.groupby("i")[col].idxmax()].set_index("i").y


def ci(d, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    bs = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(n)]
    return [round(float(d.mean() * 100), 1), round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]


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
    ci_, cc_ = S.i.to_numpy(), S.c.to_numpy()
    for thin in (0, 2):
        S[f"f_gen_t{thin}"] = [float(E[i] @ G[thin][c]) if c in G[thin] else np.nan for i, c in zip(ci_, cc_)]
    cov = S.groupby("i").y.max(); Sv = S[S.i.isin(cov[cov == 1].index)].copy()
    ref = json.load(open(P02 / "ket_qua.json"))["top1_trong_top_k"]
    res = dict(o=int(Sv.i.nunique()), invariant={c: dict(p02=ref[c], ban_sao=round(float(top1(Sv, c).mean()) * 100, 1)) for c in ("f_font", "f_fd", "f_gen_t0", "f_gen_t2")})
    res["invariant_PASS"] = bool(all(abs(v["p02"] - v["ban_sao"]) < 0.8 for v in res["invariant"].values()))
    t = Sv[Sv.y == 1].drop_duplicates("i").set_index("i")
    has_fd = set(t.index[t.f_fd.notna()])
    groups = {"tat_ca": set(Sv.i.unique()), "chu_dung_KHONG_co_FD": set(Sv.i.unique()) - has_fd, "chu_dung_CO_FD": has_fd}
    for name, ids in groups.items():
        H = Sv[Sv.i.isin(ids)]
        kk = H.groupby("i").size()
        rnd = float((1.0 / kk).mean()) * 100
        r = dict(o=len(ids), ngau_nhien=round(rnd, 1))
        tf = top1(H, "f_font")
        for col in ("f_font", "f_gen_t0", "f_gen_t2", "f_fd"):
            r[col] = round(float(top1(H, col).mean()) * 100, 1)
        for col in ("f_gen_t0", "f_gen_t2"):
            d = (top1(H, col) - tf).to_numpy(float)
            r[f"{col}_tru_font_CI95"] = ci(d)
        # tác dụng của "giãn nở ×2" (thin2) ghép cặp theo ô: gen_t2 − gen_t0
        r["f_gen_t2_tru_f_gen_t0_CI95"] = ci((top1(H, "f_gen_t2") - top1(H, "f_gen_t0")).to_numpy(float))
        res[name] = r
    (OUT / "p02_font_vs_sinh.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
