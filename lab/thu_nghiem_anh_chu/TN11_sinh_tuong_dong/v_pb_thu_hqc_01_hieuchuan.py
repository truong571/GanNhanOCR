"""v_pb_thu_hqc_01_hieuchuan.py — PHẢN BIỆN độc lập (04/10): chạy LẠI bước hiệu chuẩn của hướng "Thử HQC quy mô lớn" trên 1 bộ, mẫu ô KHÁC.
  (A) f_font / f_fd: glyph sản xuất (SI.Glyphs -> MultiEnc.embed(norm=False), CPU) · E[i] vs cột cand/<bộ>.pkl (max|Δ|, Top-1 từng ô).
  (B) E[i] sản xuất vs crop CẮT LẠI (t02_embed.page_crops) nhúng CPU (norm mặc định True).
  (C) NHẤT QUÁN C1: nhúng lại 120 chữ bằng cấu hình tune.json['cfg_font'] + mảng giấy #ids_a[0] của họ, so với nhúng C1 họ đã lưu (primary_p0_<bộ>_font.npz).
  .venv/bin/python lab/.../v_pb_thu_hqc_01_hieuchuan.py --book L16 [--seed 99]
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import tn8lib as T  # noqa: E402

SRC = REPO / "measure_out/_tn11/verify/thu_hqc"
OUT = REPO / "measure_out/_tn11/verify/pb_thu_hqc"
K3 = np.ones((3, 3), np.uint8)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def render(g, patch, ink, P, d, sigma):
    if d > 0:
        g = cv2.erode(g, K3, iterations=d)
    elif d < 0:
        g = cv2.dilate(g, K3, iterations=-d)
    if sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), float(sigma))
    S = g.shape[0]
    if P and P != S:
        g = cv2.resize(g, (int(P), int(P)), interpolation=cv2.INTER_AREA)
    else:
        P = S
    a = (255.0 - g.astype(np.float32)) / 255.0
    h = patch.shape[0]; o = (h - P) // 2
    paper = patch[o:o + P, o:o + P].astype(np.float32)
    return np.clip(np.rint(paper * (1.0 - a) + float(ink) * a), 0, 255).astype(np.uint8)


def emb_list(enc, imgs, norm):
    return np.concatenate([enc.embed(imgs[s:s + 32], norm=norm) for s in range(0, len(imgs), 32)])


def top1_cells(i, s, y):
    s9 = np.where(np.isnan(s), np.float32(-9), s).astype(np.float32)
    starts = np.r_[0, np.nonzero(np.diff(i))[0] + 1]
    mx = np.maximum.reduceat(s9, starts)
    cell_of = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(i)]))
    pos = np.nonzero(s9 == mx[cell_of])[0]
    _, ui = np.unique(cell_of[pos], return_index=True)
    return y[pos[ui]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="L16")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--n-cells", type=int, default=300)
    ap.add_argument("--parts", default="ABC", help="phần chạy: A (glyph), B (crop cắt lại), C (nhất quán C1)")
    a = ap.parse_args()
    b = a.book
    import torch
    torch.set_num_threads(2)
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    A = Assets()                                   # verify=True: kiểm sha256 như sản xuất
    enc = SI.MultiEnc([A.ext(SI.ENC_FILES["v1"]), A.ext(SI.ENC_FILES["v2"])], "cpu", True)
    gl = SI.Glyphs()
    D = T.load_base(b)
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "f_font", "f_fd"]]
    hy = F.groupby("i").y.max(); truth = hy[hy == 1].index.values
    mine = json.loads((SRC / "tune.json").read_text(encoding="utf-8"))[b]
    theirs_cal = set(np.load(SRC / f"mau_{b}.npz")["cal"].tolist())
    rng = np.random.default_rng(a.seed)
    pool_cells = np.array(sorted(set(int(x) for x in truth) - theirs_cal))
    cells = np.sort(rng.choice(pool_cells, size=a.n_cells, replace=False))
    Fc = F[F.i.isin(set(int(x) for x in cells))].reset_index(drop=True)
    E = np.load(T.OUT / "emb" / f"{b}_enc.npy").astype(np.float32)
    res = dict(book=b, seed=a.seed, n_cells=int(len(cells)), disjoint_from_their_cal=True, rows=int(len(Fc)))
    # ---- (A) glyph font / FD
    chars = sorted(set(Fc.c.astype(str)))
    t0 = time.time()
    for kind, col, getter in ((("font", "f_font", gl.font), ("fd", "f_fd", gl.fdimg)) if "A" in a.parts else ()):
        gimg = {c: getter(c) for c in chars}
        have = [c for c in chars if gimg[c] is not None]
        V = emb_list(enc, [gimg[c] for c in have], False)
        G = dict(zip(have, V))
        # sản xuất chuẩn hoá glyph về L2 (signals_img.glyph_emb không chuẩn hoá FA; FE/DE là embed thô) -> so cả hai
        s = np.full(len(Fc), np.nan, np.float32)
        for j, (i, c) in enumerate(zip(Fc.i.values, Fc.c.astype(str).values)):
            if c in G:
                s[j] = float(E[i] @ G[c])
        sp = Fc[col].values
        both = ~np.isnan(s) & ~np.isnan(sp)
        dd = np.abs(s[both] - sp[both])
        t_my = top1_cells(Fc.i.values, s, Fc.y.values); t_pr = top1_cells(Fc.i.values, sp, Fc.y.values)
        res[col] = dict(n_rows=int(len(Fc)), n_both=int(both.sum()), nan_mismatch=int((np.isnan(s) != np.isnan(sp)).sum()), max_abs=float(dd.max()),
                        mean_abs=float(dd.mean()), pearson=float(np.corrcoef(s[both], sp[both])[0, 1]), top1_mine=float(t_my.mean()), top1_prod=float(t_pr.mean()),
                        cells_top1_equal=int((t_my == t_pr).sum()), n_cells=int(len(t_my)), norm_glyph_emb_mean=float(np.linalg.norm(V, axis=1).mean()))
        log(f"{b} {col}: max|Δ|={res[col]['max_abs']:.2e}, r={res[col]['pearson']:.8f}, top1 {res[col]['top1_mine']:.4f} vs {res[col]['top1_prod']:.4f}, "
            f"ô trùng {res[col]['cells_top1_equal']}/{res[col]['n_cells']} ({time.time() - t0:.0f}s)")
    # ---- (B) crop cắt lại + nhúng CPU == E[i]
    from t02_embed import page_crops
    cfg = T.BOOKS[b]; prep = T.REPO / cfg["prep"]
    want = set(int(x) for x in cells[:60])
    jobs = []
    for pg, g in D.groupby("page", sort=True):
        if not (set(g.index) & want):
            continue
        recs = [(i, c, T.bbox_of(bb)) for i, c, bb in zip(g.index, g.column, g.bbox)]
        jobs.append((b, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", recs))
    crops = {}
    for j in (jobs if "B" in a.parts else []):
        for i, g, m in page_crops(j):
            if int(i) in want and g is not None:
                crops[int(i)] = g
    ids = sorted(crops)
    if ids:
        E2 = emb_list(enc, [crops[i] for i in ids], None)
        E1 = E[ids]
        cos = np.einsum("ij,ij->i", E1, E2) / (np.linalg.norm(E1, axis=1) * np.linalg.norm(E2, axis=1))
        res["crop_recut"] = dict(n=len(ids), cos_min=float(cos.min()), cos_mean=float(cos.mean()), max_abs=float(np.abs(E1 - E2).max()))
        log(f"{b} crop cắt lại {len(ids)} ô: cos min {cos.min():.7f}, max|Δ| {res['crop_recut']['max_abs']:.2e}")
    # ---- (C) nhất quán C1: nhúng lại bằng cấu hình + mảng giấy của họ, so nhúng đã lưu
    if "C" not in a.parts:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"hieuchuan_{b}_s{a.seed}_{a.parts}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    pz = np.load(SRC / "giay" / f"{b}.npz"); pj = json.loads((SRC / "giay" / f"{b}.json").read_text(encoding="utf-8"))
    patch = pz["patches"][pj["patch_ids_a"][0]]; ink = pj["ink"]
    c = mine["cfg_font"]
    z = np.load(SRC / "emb" / f"primary_p0_{b}_font.npz", allow_pickle=False)
    stored = {str(k): v for k, v in zip(z["keys"], z["E"])}
    sample_chars = [c_ for c_ in rng.permutation(sorted(stored))[:120] if gl.font(c_) is not None]
    imgs = [render(gl.font(c_), patch, ink, c["P"] if c["P"] else None, c["d"], c["sigma"]) for c_ in sample_chars]
    V = emb_list(enc, imgs, bool(c["norm"]))
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = np.stack([stored[c_] for c_ in sample_chars])
    cc = np.einsum("ij,ij->i", V, S)
    res["c1_consistency"] = dict(n=len(sample_chars), cfg=c, patch_page=pj["patch_pages_a"][0], cos_min=float(cc.min()), cos_mean=float(cc.mean()),
                                 max_abs=float(np.abs(V - S).max()))
    log(f"{b} C1 nhất quán (cfg {c['P']},{c['d']},{c['sigma']},norm={c['norm']}): cos min {cc.min():.6f}, max|Δ| {res['c1_consistency']['max_abs']:.2e}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (f"hieuchuan_{b}_s{a.seed}.json" if a.parts == "ABC" else f"hieuchuan_{b}_s{a.seed}_{a.parts}.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
