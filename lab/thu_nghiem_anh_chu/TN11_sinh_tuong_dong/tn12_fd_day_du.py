"""TN12 (04/10) — KHO ẢNH FD ĐẦY ĐỦ (git lfs pull) có giúp so ảnh và bộ chọn chữ không? 0 API, không sửa repo.

Hiện trạng (thẩm định 04/10): gannhanocr-fd là submodule HuggingFace có 89.813/89.898 con trỏ Git-LFS; pipeline chỉ nạp được 1.646 ảnh FD thật
(pipeline/gold_exact/signals_img.Glyphs bỏ tệp < 1 KB). f_fd = NaN ở ~80 % ứng viên và hoạt động như tiên nghiệm tần suất chữ.
Thí nghiệm: nạp thêm ảnh từ bản clone đã `git lfs pull` (--fd-dir; đặt NGOÀI repo), giữ ưu tiên ArcFace/data/glyphs như cũ, nhúng MultiEnc,
dựng f_fd_full cho 4 bộ có nhãn người, rồi
  embed   nhúng ảnh FD của mọi chữ ứng viên; hiệu chuẩn: f_fd tái tạo == cột sản xuất (sai khác < 1e-3) trên chữ đã có ảnh
  single  Top-1 một tín hiệu: f_font, f_fd (hiện tại), f_fd_full — naive (NaN = −9) và cùng tập (như p08), phủ
  chooser bộ chọn chữ TN8 LOBO (cùng t06_eval, 4 cặp) với f_fd := f_fd_full: Top-1 mọi ô và chữ hiếm, trước/sau
Ra: measure_out/_tn11/tn12_fd_day_du/ (ket_qua.json, cand/, pred/, emb_cache/). Không ghi vào measure_out/_tn8.

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/tn12_fd_day_du.py --fd-dir <clone đã lfs pull> --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

SRC = T.OUT
W = REPO / "measure_out" / "_tn11" / "tn12_fd_day_du"
BOOKS = ["B18", "B34", "L16", "TK"]
PAIRS = [("B34", "B18"), ("B18", "B34"), ("TK", "L16"), ("L16", "TK")]


def rowdot(E, idx_e, M, idx_m, chunk=500000):
    out = np.full(len(idx_e), np.nan, np.float32)
    ok = np.nonzero(idx_m >= 0)[0]
    for s in range(0, len(ok), chunk):
        k = ok[s:s + chunk]
        out[k] = np.einsum("ij,ij->i", E[idx_e[k]], M[idx_m[k]])
    return out


def stage_embed(a):
    import torch
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    (W / "emb_cache").mkdir(parents=True, exist_ok=True)
    chars = sorted({c for b in BOOKS for c in pd.read_pickle(SRC / "cand" / f"{b}.pkl")["c"]})
    S = SI.Scorers(Assets(), a.device, W / "emb_cache", False, lambda m: None)
    gl = S.glyphs()
    n0 = len(gl.fd)
    n_new = 0
    for p in sorted(Path(a.fd_dir).rglob("U+*.png")):
        if p.stat().st_size < 200:                     # con trỏ LFS (129 B) — ảnh thật ≥ ~1 KB
            continue
        ch = chr(int(p.name[2:-4], 16))
        if ch not in gl.fd:                            # giữ ưu tiên ArcFace/data/glyphs như pipeline
            gl.fd[ch] = str(p); n_new += 1
    print(f"[tn12] ảnh FD: {n0} (pipeline) + {n_new} (clone lfs pull) = {len(gl.fd)}; chữ ứng viên 4 bộ: {len(chars)}", flush=True)
    enc = S.enc()
    have = [c for c in chars if c in gl.fd]
    t0 = time.time()
    ims, keep = [], []
    for c in have:
        im = gl.fdimg(c)
        if im is not None:
            ims.append(im); keep.append(c)
    V = []
    for s in range(0, len(ims), 2048):
        V.append(np.asarray(enc.embed(ims[s:s + 2048], norm=False), np.float32))
        print(f"[tn12] nhúng {min(s + 2048, len(ims))}/{len(ims)} [{time.time() - t0:.0f}s]", flush=True)
    V = np.concatenate(V) if V else np.zeros((0, 512), np.float32)
    np.savez_compressed(W / "fd_full_emb.npz", chars=np.array(keep, dtype=object), V=V, n_pipeline=n0, n_new=n_new)
    print(f"[tn12] nhúng xong {len(keep)} chữ có ảnh FD (phủ {len(keep) / len(chars) * 100:.1f} % chữ ứng viên)", flush=True)


def load_full():
    z = np.load(W / "fd_full_emb.npz", allow_pickle=True)
    chars = list(z["chars"]); V = z["V"]
    return {c: j for j, c in enumerate(chars)}, V


def build_cand():
    """cand/<bộ>.pkl với f_fd := f_fd_full (giữ f_fd_old); hiệu chuẩn so cột sản xuất."""
    cid, V = load_full()
    (W / "cand").mkdir(parents=True, exist_ok=True)
    cal = {}
    for b in BOOKS:
        F = pd.read_pickle(SRC / "cand" / f"{b}.pkl")
        E = np.load(SRC / "emb" / f"{b}_enc.npy").astype(np.float32)
        ci = F.i.to_numpy(np.int64); cc = F.c.to_numpy(object)
        jd = np.array([cid.get(c, -1) for c in cc], np.int64)
        full = rowdot(E, ci, V, jd)
        old = F["f_fd"].to_numpy(np.float32)
        both = np.isfinite(old) & np.isfinite(full)
        cal[b] = dict(n_old=int(np.isfinite(old).sum()), n_full=int(np.isfinite(full).sum()),
                      max_abs_diff_chung=float(np.abs(old[both] - full[both]).max()) if both.any() else None,
                      old_co_full_khong=int((np.isfinite(old) & ~np.isfinite(full)).sum()))
        G = F.copy(); G["f_fd_old"] = old; G["f_fd"] = full
        G.to_pickle(W / "cand" / f"{b}.pkl")
        c = W / "cand" / f"{b}_cells.pkl"
        if not c.exists():
            os.symlink(SRC / "cand" / f"{b}_cells.pkl", c)
        print(f"[tn12] {b}: hàng có f_fd {cal[b]['n_old']} -> {cal[b]['n_full']}; hiệu chuẩn max|Δ| trên phần chung = {cal[b]['max_abs_diff_chung']}", flush=True)
    for d in ("base", "emb"):
        if not (W / d).exists():
            os.symlink(SRC / d, W / d)
    return cal


def top1_per_cell(F, col):
    x = F[["i", col, "y"]]
    return x.loc[x.groupby("i")[col].idxmax()].set_index("i").y.astype(float)


def stage_single(a, cal):
    out = {}
    print(f"{'bộ':4s} | phủ FD (chữ đúng / mọi ƯV): cũ -> đầy đủ | naive Top-1: font | FD cũ | FD đầy đủ | cùng tập (FD đầy đủ − font) [CI95]")
    for b in BOOKS:
        F = pd.read_pickle(W / "cand" / f"{b}.pkl")[["i", "c", "y", "f_font", "f_fd", "f_fd_old"]]
        D = T.load_base(b); pages = D.page.to_numpy()
        hy = F.groupby("i").y.max(); cells = hy[hy == 1].index
        Fc = F[F.i.isin(cells)].copy()
        r = {}
        for nm in ("f_fd_old", "f_fd"):
            r[f"phu_{nm}"] = (round(float(Fc[Fc.y == 1][nm].notna().mean()) * 100, 1), round(float(Fc[nm].notna().mean()) * 100, 1))
        for nm in ("f_font", "f_fd_old", "f_fd"):
            Fc[nm + "9"] = Fc[nm].fillna(-9)
            r[f"naive_{nm}"] = round(float(top1_per_cell(Fc, nm + "9").mean()) * 100, 1)
        for nm in ("f_fd_old", "f_fd"):                  # cùng tập: chỉ ứng viên có cả font và ảnh FD
            M = Fc[Fc[nm].notna() & Fc.f_font.notna()]
            k = M.groupby("i").size(); ht = M.groupby("i").y.max()
            keep = k[(k >= 2) & (ht.reindex(k.index) == 1)].index
            M = M[M.i.isin(keep)]
            t_fo = top1_per_cell(M, "f_font"); t_fd = top1_per_cell(M, nm)
            d = (t_fd - t_fo).to_numpy(); pg = pages[t_fd.index.to_numpy()]
            _, lo, hi = T.boot_ci_pages(d, pg)
            r[f"cungtap_{nm}"] = dict(n=int(len(keep)), uv_o=round(float(k.loc[keep].mean()), 1), font=round(float(t_fo.mean()) * 100, 1),
                                      fd=round(float(t_fd.mean()) * 100, 1), delta=round(float(d.mean()) * 100, 1),
                                      ci=[round(lo * 100, 1), round(hi * 100, 1)])
        out[b] = r
        f = r["cungtap_f_fd"]
        print(f"{b:4s} | {r['phu_f_fd_old'][0]:5.1f}/{r['phu_f_fd_old'][1]:4.1f} -> {r['phu_f_fd'][0]:5.1f}/{r['phu_f_fd'][1]:4.1f} | "
              f"{r['naive_f_font']:5.1f} | {r['naive_f_fd_old']:5.1f} | {r['naive_f_fd']:5.1f} | n={f['n']:6d} ({f['uv_o']:.1f} ƯV/ô) "
              f"{f['delta']:+5.1f} [{f['ci'][0]:+.1f}, {f['ci'][1]:+.1f}]", flush=True)
    return out


def stage_chooser():
    import tn8model as M  # noqa: F401
    import t06_eval as E
    res = {}
    for run, work in (("goc", SRC), ("fd_day_du", W)):
        T.OUT = work
        E.PRED = W / f"pred_{run}"
        for tr, te in PAIRS:
            E.run_pair([tr], te, tag=f"{te}__from_{tr}")
    T.OUT = SRC
    for tr, te in PAIRS:
        F = pd.read_pickle(SRC / "cand" / f"{te}.pkl")
        tru = F[F.y == 1].drop_duplicates("i").set_index("i")
        rare = set(tru.index[tru.n_self.fillna(0) < 2])
        r = {}
        for run in ("goc", "fd_day_du"):
            P = pd.read_pickle(W / f"pred_{run}" / f"{te}__from_{tr}.pkl")
            P = P[P.any_y == 1]
            m = P.i.isin(rare)
            r[run] = dict(tat_ca=round(float(P.y1.mean()) * 100, 2), chu_hiem=round(float(P[m].y1.mean()) * 100, 2), n=int(len(P)),
                          n_hiem=int(m.sum()))
        res[f"{te}<-{tr}"] = r
        print(f"{te} <- {tr}: gốc {r['goc']['tat_ca']} % (chữ hiếm {r['goc']['chu_hiem']} %) -> FD đầy đủ {r['fd_day_du']['tat_ca']} % "
              f"(chữ hiếm {r['fd_day_du']['chu_hiem']} %), n {r['goc']['n']}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd-dir", required=True, help="clone gannhanocr-fd đã git lfs pull (ngoài repo)")
    ap.add_argument("--stage", choices=["embed", "single", "chooser", "all"], default="all")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    W.mkdir(parents=True, exist_ok=True)
    res = {}
    if a.stage in ("embed", "all"):
        stage_embed(a)
    if a.stage in ("single", "chooser", "all"):
        res["hieu_chuan"] = build_cand()
    if a.stage in ("single", "all"):
        res["single"] = stage_single(a, res["hieu_chuan"])
    if a.stage in ("chooser", "all"):
        res["chooser"] = stage_chooser()
    prev = json.loads((W / "ket_qua.json").read_text()) if (W / "ket_qua.json").exists() else {}
    prev.update(res)
    (W / "ket_qua.json").write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
