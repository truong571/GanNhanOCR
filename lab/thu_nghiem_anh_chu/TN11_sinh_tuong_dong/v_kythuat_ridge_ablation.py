"""v_kythuat (04/10): đặc trưng X = [font | fd | 1] của Ridge (p01) lệch giữa lúc HỌC và lúc ÁP — thử bỏ/che khối fd (CPU, 0 API).

Quan sát (v_kythuat_fd_phu): ảnh FD thật chỉ có cho ~1,6 nghìn chữ. Trong p01 (dòng 83-88) khối fd = DE[c] nếu có, NGƯỢC LẠI = FE[c]
(chép lại khối font). Chữ DÙNG ĐỂ HỌC (>= 3 ô neo) có ảnh FD thật ở 45-86 % trường hợp, còn chữ HIẾM (đích cần suy ra) chỉ 3-6 %
=> lúc áp, khối fd gần như luôn là bản sao khối font, khác phân phối lúc học.
Script chép ĐÚNG logic p01 (cùng ô neo, cùng khối trang, cùng pick_lam seed 0, cùng dữ liệu nhúng đã cache) và so ba biến thể:
  p01        X = [font | fd-hoặc-font | 1]            (đúng p01; phải TÁI LẬP f_ridge của p01_<bộ>.json = invariant)
  chi_font   X = [font | 1]                            (nhất quán học/áp)
  fd_che     X = [font | fd-hoặc-0 | co_fd | 1]        (khối fd chỉ có nghĩa khi có ảnh thật)
Đo Top-1 một tín hiệu (như p01) trên mọi ô có sự thật và trên ô chữ hiếm. Ra: measure_out/_tn11/verify/kythuat/ridge_ablation.json
Không ghi vào _tn8/_tn11 gốc: bộ nhớ đệm nhúng glyph được SAO sang verify/kythuat/emb_cache/ rồi mới dùng.
    PYTORCH_ENABLE_MPS_FALLBACK=0 CUDA_VISIBLE_DEVICES= .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_kythuat_ridge_ablation.py B34 B18 L16 TK
"""
from __future__ import annotations

import json
import shutil
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

torch.set_num_threads(2)
OUT = REPO / "measure_out" / "_tn11" / "verify" / "kythuat"
OUT.mkdir(parents=True, exist_ok=True)
EMB = T.OUT / "emb"
CAND = T.OUT / "cand"
MYCACHE = OUT / "emb_cache"
MYCACHE.mkdir(exist_ok=True)
for f in EMB.glob("glyph_font_*.npz"):
    if not (MYCACHE / f.name).exists():
        shutil.copy2(f, MYCACHE / f.name)


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def ridge(X, Y, lam, w=None):          # == p01.ridge
    if w is not None:
        X = X * np.sqrt(w)[:, None]; Y = Y * np.sqrt(w)[:, None]
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float64)
    return np.linalg.solve(A, X.T @ Y)


def pick_lam(X, Y, w, seed=0):         # == p01.pick_lam
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


def top1(F, col):                      # == p01.top1
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
    F = pd.read_pickle(CAND / f"{b}.pkl")[["i", "c", "y", "n_self", "f_font", "f_fd"]]
    chars = sorted(set(F.c))
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    S = SI.Scorers(Assets(), "cpu", MYCACHE, False, lambda m: print("   ", m, flush=True))
    FE, _, DE, _ = S.glyph_emb(chars, with_aug=False, with_fd=True)
    cid = {c: j for j, c in enumerate(chars)}
    n = len(chars)
    has = np.zeros(n, bool); hasfd = np.zeros(n, bool)
    Gfont = np.zeros((n, 512)); Gfd_or_font = np.zeros((n, 512)); Gfd_or_0 = np.zeros((n, 512))
    for c, j in cid.items():
        if c in FE:
            has[j] = True
            Gfont[j] = FE[c]
            if c in DE:
                hasfd[j] = True; Gfd_or_font[j] = DE[c]; Gfd_or_0[j] = DE[c]
            else:
                Gfd_or_font[j] = FE[c]
    one = np.ones((n, 1))
    VAR = {
        "p01": np.hstack([Gfont, Gfd_or_font, one]),
        "chi_font": np.hstack([Gfont, one]),
        "fd_che": np.hstack([Gfont, Gfd_or_0, hasfd[:, None].astype(float), one]),
    }
    gfont = nrm(Gfont)
    ai = np.nonzero(anc)[0]
    for v in VAR:
        F[f"f_{v}"] = np.nan
    fi = F.i.to_numpy(); fj = np.array([cid[c] for c in F.c]); fblk = blk[fi]
    info = []
    for k in range(5):
        tot = defaultdict(lambda: np.zeros(512)); cnt = defaultdict(int)
        for i in ai[blk[ai] != k]:
            tot[D.label.values[i]] += E[i]; cnt[D.label.values[i]] += 1
        tr = [c for c in tot if cnt[c] >= 3 and c in cid and has[cid[c]]]
        J = np.array([cid[c] for c in tr]); Y = nrm(np.stack([tot[c] for c in tr]))
        w = np.log1p(np.array([cnt[c] for c in tr], float))
        m = (fblk == k) & has[fj]
        rec = dict(khoi=k, chu_hoc=len(tr), ti_le_chu_hoc_co_FD_that=float(hasfd[J].mean()))
        for v, G in VAR.items():
            lam, cv = pick_lam(G[J], Y, w)
            W = ridge(G[J], Y, lam, w)
            H = nrm(G @ W)
            F.loc[m, f"f_{v}"] = np.einsum("ij,ij->i", E[fi[m]], H[fj[m]])
            rec[f"lam_{v}"] = lam; rec[f"cv_{v}"] = round(cv, 4)
        info.append(rec)
        print(f"  [{b}] khối {k}: {rec}", flush=True)
    hy = F.groupby("i").y.max(); cells = hy[hy == 1].index
    F = F[F.i.isin(cells)].copy()
    tru = F[F.y == 1].drop_duplicates("i").set_index("i")
    rare = tru.n_self.fillna(0) < 2
    res = dict(bo=b, o_co_su_that=int(len(cells)), o_hiem=int(rare.sum()), khoi=info,
               ti_le_chu_hiem_co_FD_that=float(hasfd[[cid[c] for c in tru[rare].c if c in cid]].mean()),
               top1={})
    p01 = json.load(open(REPO / "measure_out/_tn11" / f"p01_{b}.json"))["top1"]
    for col in ("f_font", "f_fd", "f_p01", "f_chi_font", "f_fd_che"):
        t = top1(F, col)
        res["top1"][col] = dict(tat_ca=round(float(t.mean()) * 100, 1), chu_hiem=round(float(t[rare[rare].index].mean()) * 100, 1),
                                chu_co_mau=round(float(t[rare[~rare].index].mean()) * 100, 1))
    ref = p01["f_ridge"]
    res["invariant_tai_lap_p01"] = dict(p01_json=ref, ban_sao=res["top1"]["f_p01"],
                                         pass_=bool(abs(ref["tat_ca"] - res["top1"]["f_p01"]["tat_ca"]) < 0.15 and
                                                    abs(ref["chu_hiem"] - res["top1"]["f_p01"]["chu_hiem"]) < 0.3))
    print(f"== {b}: {json.dumps(res['top1'], ensure_ascii=False)}\n   invariant: {res['invariant_tai_lap_p01']}", flush=True)
    return res


if __name__ == "__main__":
    allres = {}
    p = OUT / "ridge_ablation.json"
    if p.exists():
        allres = json.loads(p.read_text(encoding="utf-8"))
    for b in sys.argv[1:] or ["B34", "B18", "L16", "TK"]:
        allres[b] = run(b)
        p.write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")
