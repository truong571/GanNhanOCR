"""TN11 p01 (03/10) — thử nhanh "SINH MẪU THEO NÉT CỦA TỪNG CUỐN" trong không gian đặc trưng (0 API, không sửa pipeline).

Câu hỏi: crop so với glyph sinh sẵn (font/FD một phong cách STT2) chỉ đúng Top-1 60–65 % trên Borg, so với nguyên mẫu THẬT
cùng sách 68–78 %. Nếu học được phép biến đổi glyph -> "chữ của cuốn này" từ các chữ ĐÃ có ô neo tự động (kim ∈ R, nhãn = kim,
không nhãn người), rồi áp cho chữ CHƯA có ô neo, có lấp được khoảng cách không? Đây là cận rẻ cho hướng sinh ảnh theo phong cách
từng cuốn (fine-tune FontDiffuser): nếu ánh xạ tuyến tính đã giúp, sinh ảnh thật sự có cơ sở.

Với mỗi khối trang k (5 khối như TN8): học trên ô neo của 4 khối KHÁC
  shift  h_c = norm(g_c + mean(p_c - g_c))
  ridge  h_c = norm(W·[font_c ; fd_c ; 1]),  W ridge, λ chọn bằng CV theo CHỮ trên ô neo (không nhìn nhãn người)
và chấm ô của khối k: f_map = cos(crop, h_c). Đo Top-1 một tín hiệu trên ô có sự thật (Borg: chữ người; IHR: GT).
Ra: measure_out/_tn11/p01_<bộ>.json

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p01_anh_xa_phong_cach.py B18 B34 L16 TK
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11"
EMB = T.OUT / "emb"
CAND = T.OUT / "cand"


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def ridge(X, Y, lam, w=None):
    if w is not None:
        X = X * np.sqrt(w)[:, None]; Y = Y * np.sqrt(w)[:, None]
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float64)
    return np.linalg.solve(A, X.T @ Y)


def pick_lam(X, Y, w, seed=0):
    """CV 5 phần theo CHỮ trên nguyên mẫu ô neo: λ cho cos(dự đoán, nguyên mẫu) cao nhất."""
    rng = np.random.default_rng(seed); f = rng.integers(0, 5, len(X)); best = (-1, None)
    for lam in (0.3, 1, 3, 10, 30, 100):
        s = []
        for k in range(5):
            tr, te = f != k, f == k
            W = ridge(X[tr], Y[tr], lam, w[tr])
            s.append(float(np.mean(np.sum(nrm(X[te] @ W) * Y[te], 1))))
        if np.mean(s) > best[0]:
            best = (float(np.mean(s)), lam)
    return best[1], best[0]


def top1(F, col):
    x = F[["i", col, "y"]].copy(); x[col] = x[col].fillna(-9)
    return x.loc[x.groupby("i")[col].idxmax(), ["i", "y"]].set_index("i").y


def run(b):
    C = T.lex()
    D = T.load_base(b)
    E = nrm(np.load(EMB / f"{b}_enc.npy").astype(np.float32))
    meta = np.load(EMB / f"{b}_crop_meta.npy")
    blk = T.page_blocks(D.page.values, 5)
    Rm = {s: C.R_of(s) for s in set(D.syllable)}
    anc = np.array((D.rule.str.startswith("s1_inter_s2_direct") & (D.label != "") & (D.label == D.ocr_char)).values, bool)
    anc &= np.array([l in Rm[s] for l, s in zip(D.label, D.syllable)]) & (meta[:, 3] > 0)
    F = pd.read_pickle(CAND / f"{b}.pkl")[["i", "c", "y", "n_self", "f_font", "f_fd", "f_self", "f_hum"]]
    chars = sorted(set(F.c))
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    S = SI.Scorers(Assets(), "mps" if torch.backends.mps.is_available() else "cpu", EMB, False, lambda m: None)
    FE, _, DE, _ = S.glyph_emb(chars, with_aug=False, with_fd=True)
    cid = {c: j for j, c in enumerate(chars)}
    G = np.zeros((len(chars), 1025), np.float64); has = np.zeros(len(chars), bool)
    for c, j in cid.items():
        if c in FE:
            G[j, :512] = FE[c]; has[j] = True
            G[j, 512:1024] = DE[c] if c in DE else FE[c]
            G[j, 1024] = 1.0
    gfont = nrm(G[:, :512])
    ai = np.nonzero(anc)[0]
    F["f_shift"] = np.nan; F["f_ridge"] = np.nan
    fi = F.i.to_numpy(); fj = np.array([cid[c] for c in F.c]); fblk = blk[fi]
    lam_log = []
    for k in range(5):
        tot = defaultdict(lambda: np.zeros(512)); n = defaultdict(int)
        for i in ai[blk[ai] != k]:
            tot[D.label.values[i]] += E[i]; n[D.label.values[i]] += 1
        tr = [c for c in tot if n[c] >= 3 and c in cid and has[cid[c]]]
        J = np.array([cid[c] for c in tr]); Y = nrm(np.stack([tot[c] for c in tr]))
        w = np.log1p(np.array([n[c] for c in tr], float))
        mu = np.average(Y - gfont[J], axis=0, weights=w)
        Hs = nrm(gfont + mu)
        lam, cv = pick_lam(G[J], Y, w)
        W = ridge(G[J], Y, lam, w)
        Hr = nrm(G @ W)
        lam_log.append(dict(khoi=k, chu_hoc=len(tr), lam=lam, cos_cv=round(cv, 4)))
        m = (fblk == k) & has[fj]
        F.loc[m, "f_shift"] = np.einsum("ij,ij->i", E[fi[m]], Hs[fj[m]])
        F.loc[m, "f_ridge"] = np.einsum("ij,ij->i", E[fi[m]], Hr[fj[m]])
    OUT.mkdir(parents=True, exist_ok=True)
    F[["i", "c", "f_shift", "f_ridge"]].astype({"f_shift": np.float32, "f_ridge": np.float32}).to_pickle(OUT / f"p01_{b}_scores.pkl")
    hy = F.groupby("i").y.max(); cells = hy[hy == 1].index
    F = F[F.i.isin(cells)].copy()
    F["f_self_or_fd"] = F.f_self.where(F.f_self.notna(), F.f_fd)
    F["f_self_or_ridge"] = F.f_self.where(F.f_self.notna(), F.f_ridge)
    tru = F[F.y == 1].drop_duplicates("i").set_index("i")
    rare = tru.n_self.fillna(0) < 2
    res = dict(bo=b, o_co_su_that=int(len(cells)), o_chu_dung_khong_co_mau_cung_sach=int(rare.sum()), lam=lam_log, top1={})
    for col in ("f_font", "f_fd", "f_shift", "f_ridge", "f_self", "f_self_or_fd", "f_self_or_ridge", "f_hum"):
        t = top1(F, col)
        res["top1"][col] = dict(tat_ca=round(float(t.mean()) * 100, 1),
                                chu_hiem=round(float(t[rare[rare].index].mean()) * 100, 1),
                                chu_co_mau=round(float(t[rare[~rare].index].mean()) * 100, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"p01_{b}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"== {b}: {res['o_co_su_that']} ô có sự thật; chữ đúng chưa có mẫu cùng sách: {res['o_chu_dung_khong_co_mau_cung_sach']}")
    print(f"   λ: {[d['lam'] for d in lam_log]}  cos CV {[d['cos_cv'] for d in lam_log]}")
    for col, v in res["top1"].items():
        print(f"   {col:16s} tất cả {v['tat_ca']:5.1f}  chữ hiếm {v['chu_hiem']:5.1f}  chữ có mẫu {v['chu_co_mau']:5.1f}")


if __name__ == "__main__":
    for b in sys.argv[1:] or ["B18", "B34", "L16", "TK"]:
        run(b)
