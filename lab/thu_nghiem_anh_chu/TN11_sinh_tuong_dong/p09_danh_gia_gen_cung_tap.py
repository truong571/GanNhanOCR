"""TN11 p09 (04/10) — chấm ảnh sinh theo phong cách sách ĐÚNG CHUẨN: cùng tập ứng viên, kiểm định cặp. 0 API, CPU.

Sửa hai lỗi của p02 --stage eval phát hiện khi thẩm định:
  1. f_fd có độ phủ rất thấp (1.646 ảnh thật) và xếp NaN = −9 ⇒ so "ảnh sinh vs FD" lệch; ở đây chỉ so trên ứng viên CÓ ảnh FD (cùng tập).
  2. p02 không có kiểm định cặp; ở đây bootstrap theo ô + McNemar chính xác cho từng cặp tín hiệu.
Tín hiệu: f_font (render phông), f_fd (ảnh FD sinh sẵn có trong pipeline), f_gen (ảnh sinh theo phong cách sách, thư mục --gen-sub; --thin = số lần giãn 3x3),
f_vW (CNN kiểm, dùng để chọn top-8 ở p02). Nhãn người chỉ dùng để chấm.

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p09_danh_gia_gen_cung_tap.py --book B34 --gen-sub gen_fix --style 0
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

T11 = REPO / "measure_out" / "_tn11"


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def top1(S, col):
    x = S[["i", col, "y"]]
    return x.loc[x.groupby("i")[col].idxmax()].set_index("i").y.astype(int)


def z(S, col):
    g = S.groupby("i")[col]
    return (S[col] - g.transform("mean")) / g.transform("std").replace(0, 1)


def paired(a, b, rng, B=4000):
    """a, b: Series 0/1 cùng chỉ số ô. Δ (điểm %), CI95 bootstrap theo ô, McNemar chính xác."""
    d = (a - b).to_numpy()
    bs = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(B)]
    n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return dict(delta=round(float(d.mean()) * 100, 1), ci=[round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)],
                a_hon=n10, b_hon=n01, p=round(float(p), 4))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="B34")
    ap.add_argument("--gen-sub", default="gen_fix")
    ap.add_argument("--style", type=int, default=0)
    a = ap.parse_args(argv)
    out = T11 / f"p02_{a.book}"
    P = pd.read_pickle(out / "plan.pkl"); S = P["S"].copy()
    E = nrm(np.load(T.OUT / "emb" / f"{a.book}_enc.npy").astype(np.float32))
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    Sc = SI.Scorers(Assets(), "cpu", T11 / "verify" / "p09_emb_cache", False, lambda m: None)
    (T11 / "verify" / "p09_emb_cache").mkdir(parents=True, exist_ok=True)
    enc = Sc.enc()
    k3 = np.ones((3, 3), np.uint8)
    gdir = out / a.gen_sub / f"s{a.style}"
    chars = [c for c in P["chars"] if (gdir / f"U+{ord(c):04X}.png").exists()]
    print(f"[p09] {a.book} {a.gen_sub}/s{a.style}: {len(chars)}/{len(P['chars'])} chữ đã sinh", flush=True)
    for thin in (0, 2):
        ims = []
        for c in chars:
            im = cv2.imread(str(gdir / f"U+{ord(c):04X}.png"), cv2.IMREAD_GRAYSCALE)
            ims.append(cv2.dilate(im, k3, iterations=thin) if thin else im)
        V = nrm(np.asarray(enc.embed(ims, norm=False), np.float32)); M = dict(zip(chars, V))
        S[f"f_gen_t{thin}"] = [float(E[i] @ M[c]) if c in M else np.nan for i, c in zip(S.i.to_numpy(), S.c.to_numpy())]
    # ô có chữ đúng trong top-8 và MỌI ứng viên đều có ảnh sinh (để so f_gen công bằng); ứng viên thiếu (không có trong phông NomNaTong) bị loại ở cả ba tín hiệu
    S = S[S.f_gen_t0.notna() & S.f_font.notna()].copy()
    rows = {}
    rng = np.random.default_rng(20261004)
    for tag, extra in (("tat_ca_top8", None), ("cung_tap_FD", "f_fd")):
        X = S if extra is None else S[S[extra].notna()]
        k = X.groupby("i").size(); ht = X.groupby("i").y.max()
        keep = k[(k >= 2) & (ht.reindex(k.index) == 1)].index
        X = X[X.i.isin(keep)].copy()
        X["f_fd9"] = X["f_fd"].fillna(-9)
        for c in ("f_vW", "f_font", "f_gen_t0", "f_gen_t2"):
            X[c + "_z"] = z(X, c)
        X["vW+font"] = X.f_vW_z + X.f_font_z
        X["vW+gen_t0"] = X.f_vW_z + X.f_gen_t0_z
        X["vW+gen_t2"] = X.f_vW_z + X.f_gen_t2_z
        cols = ["f_vW", "f_font", "f_fd9", "f_gen_t0", "f_gen_t2", "vW+font", "vW+gen_t0", "vW+gen_t2"]
        t = {c: top1(X, c) for c in cols}
        res = dict(n_o=int(len(keep)), uv_o=round(float(k.loc[keep].mean()), 2),
                   top1={c: round(float(t[c].mean()) * 100, 1) for c in cols}, so_cap={})
        pairs = [("f_gen_t0", "f_font"), ("f_gen_t2", "f_font"), ("f_gen_t0", "f_fd9"), ("f_gen_t2", "f_fd9"), ("f_fd9", "f_font"),
                 ("vW+gen_t0", "f_vW"), ("vW+gen_t2", "f_vW"), ("vW+gen_t0", "vW+font"), ("vW+gen_t2", "vW+font")]
        for x, y in pairs:
            res["so_cap"][f"{x} − {y}"] = paired(t[x], t[y], rng)
        rows[tag] = res
        print(f"\n== {tag}: {res['n_o']} ô, {res['uv_o']} ƯV/ô ==\n  Top-1: " + " · ".join(f"{c} {v}" for c, v in res["top1"].items()))
        for kx, v in res["so_cap"].items():
            print(f"  {kx:28s} Δ {v['delta']:+5.1f} [{v['ci'][0]:+5.1f}, {v['ci'][1]:+5.1f}]  hơn/kém {v['a_hon']}/{v['b_hon']}  McNemar p={v['p']}")
    (out / f"p09_{a.gen_sub}_s{a.style}.json").write_text(json.dumps(dict(book=a.book, gen_sub=a.gen_sub, n_chu_sinh=len(chars), **rows), ensure_ascii=False, indent=1),
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()
