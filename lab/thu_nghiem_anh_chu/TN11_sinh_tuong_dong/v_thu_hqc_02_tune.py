"""v_thu_hqc_02_tune.py — chọn tham số quang học của C1 (HQC2 "bản tốt nhất") CHO TỪNG SÁCH, KHÔNG dùng nhãn người:
mục tiêu = Top-1 NHẬN DẠNG giữa các chữ neo (|L| ~ 170 chữ; ô neo tự động: kim ∈ R(âm), nhãn = kim) của 250 ô neo (hạt giống cố định).
Lưới (font): độ phân giải {gốc-crop P_font, khung glyph 112} × độ đậm nét d {-1,0,1} × nhoè σ_c {0,1.5}; norm=True (chuẩn hoá giãn p2/p98 GIỐNG crop sản xuất),
K=1 mảng giấy (mảng #1 của bộ ba A). Lưới (FD): {P_fd, 96} × d {-1,0} × σ_c {0,1.5}.
Luật chọn (đăng ký TRƯỚC khi có kết quả): mặc định = giấy-chỉ (khung glyph, d=0, σ=0); đổi sang cấu hình tốt nhất của lưới CHỈ NẾU hơn mặc định >= 1,0 điểm Top-1 neo.
Đối chiếu thêm: 'lit' = công thức NGUYÊN VĂN của báo cáo (khung glyph, 1 mảng, norm=False) và đối chứng nền trắng cùng quang học.
Ra: measure_out/_tn11/verify/thu_hqc/tune.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v_thu_hqc_lib as H  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T = H.T
OUT = H.OUT
N_TUNE = 250
MIN_GAIN = 1.0       # điểm %


def ident(E, rows, labels, emb):
    """Top-1 nhận dạng giữa các chữ trong emb (dict) cho các ô rows có nhãn trong emb; trả (top1 %, cos_true TB, n)."""
    chars = [c for c in sorted(set(labels)) if c in emb and not np.isnan(emb[c][0])]
    if not chars:
        return float("nan"), float("nan"), 0
    G = np.stack([emb[c] for c in chars]).astype(np.float32)
    idx = {c: j for j, c in enumerate(chars)}
    keep = np.array([l in idx for l in labels])
    S = E[rows[keep]] @ G.T
    y = np.array([idx[l] for l in np.asarray(labels)[keep]])
    top = S.argmax(1)
    return float((top == y).mean() * 100), float(S[np.arange(len(y)), y].mean()), int(keep.sum())


def cfgd(P, d, sigma, norm, ids, K=1, white=False):
    return dict(P=P, d=d, sigma=sigma, norm=norm, K=K, patch_ids=list(ids[:K]), white_bg=white)


def tune_book(pool, b, wd):
    pm = H.paper_model(b)
    D = T.load_base(b)
    E = np.load(T.OUT / "emb" / f"{b}_enc.npy").astype(np.float32)
    cells = pd.read_pickle(T.OUT / "cand" / f"{b}_cells.pkl")
    anc = np.nonzero(cells.anchor.values.astype(bool) & (cells.crop_ok.values > 0))[0]
    rng = np.random.default_rng(H.SEED + 41 + H.BI[b])
    Tb = np.sort(rng.choice(anc, size=min(N_TUNE, len(anc)), replace=False))
    lab = D.label.values[Tb]
    L = sorted(set(lab))
    fdset = set(wd)
    anc_fd = np.array([i for i in anc if D.label.values[i] in fdset])
    Tfd = np.sort(rng.choice(anc_fd, size=min(200, len(anc_fd)), replace=False)) if len(anc_fd) else np.array([], int)
    labfd = D.label.values[Tfd]
    Lfd = sorted(set(labfd))
    H.log(f"{b}: neo {len(anc)}, ô tuning {len(Tb)} ({len(L)} chữ); ô neo có ảnh FD {len(anc_fd)}, tuning FD {len(Tfd)} ({len(Lfd)} chữ)")
    tdir = OUT / "tune"
    ids_a, ids_b = pm["patch_ids_a"], pm["patch_ids_b"]
    Pf, Pd = pm["P_font"], pm["P_fd"]
    res = dict(book=b, n_anchor_all=int(len(anc)), n_tune=int(len(Tb)), n_chars=len(L), n_tune_fd=int(len(Tfd)), n_chars_fd=len(Lfd),
               P_font=Pf, P_fd=Pd, font={}, fd={}, diag={}, tune_cells=[int(x) for x in Tb], tune_cells_fd=[int(x) for x in Tfd])
    # ---- font
    emb_w = H.embed_chars(pool, "white", "font", L, tdir / f"{b}_white_font.npz", tag=f"{b}.white")
    t1, c1, n1 = ident(E, Tb, lab, emb_w)
    res["font"]["white"] = dict(top1=t1, cos_true=c1, n=n1)
    grid = []
    for P in (None, Pf):
        for d in (-1, 0, 1):
            for s in (0.0, 1.5):
                grid.append((f"{'canvas' if P is None else 'native'}_d{d}_s{s:g}", cfgd(P, d, s, True, ids_a)))
    embs = {}
    for name, cfg in grid + [("lit", cfgd(None, 0, 0.0, False, ids_a))]:
        emb = H.embed_chars(pool, "paper", "font", L, tdir / f"{b}_{name}_font.npz", book=b, cfg=cfg, tag=f"{b}.{name}")
        embs[name] = emb
        t, c, n = ident(E, Tb, lab, emb)
        res["font"][name] = dict(top1=t, cos_true=c, n=n, cfg=cfg)
    default = "canvas_d0_s0"
    best = max([g[0] for g in grid], key=lambda k: (res["font"][k]["top1"], -[g[0] for g in grid].index(k)))
    pick = best if res["font"][best]["top1"] - res["font"][default]["top1"] >= MIN_GAIN else default
    res["font"]["_default"] = default; res["font"]["_best_grid"] = best; res["font"]["_picked"] = pick
    cfg_pick = dict(res["font"][pick]["cfg"]); cfg_pick["K"] = 3; cfg_pick["patch_ids"] = list(ids_a[:3])
    # đối chứng / độ nhạy ở cấu hình đã chọn
    ctrl = dict(cfg_pick); ctrl.update(K=1, patch_ids=list(ids_a[:1]), white_bg=True)
    e = H.embed_chars(pool, "paper", "font", L, tdir / f"{b}_{pick}_whiteoptics_font.npz", book=b, cfg=ctrl, tag=f"{b}.whiteoptics")
    res["font"]["picked_white_bg_same_optics_K1"] = dict(zip(("top1", "cos_true", "n"), ident(E, Tb, lab, e)))
    res["cfg_font"] = cfg_pick
    # độ lệch nền: cos(trắng, giấy) cho các cấu hình đại diện
    def cosd(a, bb):
        ch = [c for c in L if c in a and c in bb and not np.isnan(a[c][0]) and not np.isnan(bb[c][0])]
        x = np.array([float(a[c] @ bb[c] / (np.linalg.norm(a[c]) * np.linalg.norm(bb[c]))) for c in ch])
        return dict(n=len(x), mean=float(x.mean()), p05=float(np.quantile(x, .05)), p50=float(np.median(x)), p95=float(np.quantile(x, .95)))
    res["diag"]["font_cos_white_vs_lit_norm_false"] = cosd(emb_w, embs["lit"])
    res["diag"]["font_cos_white_vs_canvas_d0_s0_norm_true"] = cosd(emb_w, embs["canvas_d0_s0"])
    # ---- FD
    if len(Tfd):
        wfd = {c: wd[c] for c in Lfd}
        t, c, n = ident(E, Tfd, labfd, wfd)
        res["fd"]["white"] = dict(top1=t, cos_true=c, n=n)
        gridf = []
        for P in (None, Pd):
            for d in (-1, 0):
                for s in (0.0, 1.5):
                    gridf.append((f"{'canvas' if P is None else 'native'}_d{d}_s{s:g}", cfgd(P, d, s, True, ids_a)))
        embf = {}
        for name, cfg in gridf + [("lit", cfgd(None, 0, 0.0, False, ids_a))]:
            emb = H.embed_chars(pool, "paper", "fd", Lfd, tdir / f"{b}_{name}_fd.npz", book=b, cfg=cfg, tag=f"{b}.fd.{name}")
            embf[name] = emb
            t, c, n = ident(E, Tfd, labfd, emb)
            res["fd"][name] = dict(top1=t, cos_true=c, n=n, cfg=cfg)
        bestf = max([g[0] for g in gridf], key=lambda k: (res["fd"][k]["top1"], -[g[0] for g in gridf].index(k)))
        pickf = bestf if res["fd"][bestf]["top1"] - res["fd"]["canvas_d0_s0"]["top1"] >= MIN_GAIN else "canvas_d0_s0"
        res["fd"]["_default"] = "canvas_d0_s0"; res["fd"]["_best_grid"] = bestf; res["fd"]["_picked"] = pickf
        cf = dict(res["fd"][pickf]["cfg"]); cf["K"] = 3; cf["patch_ids"] = list(ids_a[:3])
        res["cfg_fd"] = cf
        a = {c: wd[c] for c in Lfd}
        def cosd2(a, bb):
            ch = [c for c in Lfd if c in a and c in bb and not np.isnan(a[c][0]) and not np.isnan(bb[c][0])]
            x = np.array([float(a[c] @ bb[c] / (np.linalg.norm(a[c]) * np.linalg.norm(bb[c]))) for c in ch])
            return dict(n=len(x), mean=float(x.mean()), p05=float(np.quantile(x, .05)), p50=float(np.median(x)), p95=float(np.quantile(x, .95)))
        res["diag"]["fd_cos_white_vs_lit_norm_false"] = cosd2(a, embf["lit"])
        res["diag"]["fd_cos_white_vs_canvas_d0_s0_norm_true"] = cosd2(a, embf["canvas_d0_s0"])
    else:
        res["cfg_fd"] = cfgd(None, 0, 0.0, True, ids_a, K=3)
    H.log(f"{b}: font nhận dạng neo: trắng {res['font']['white']['top1']:.1f} | giấy-chỉ {res['font']['canvas_d0_s0']['top1']:.1f} | tốt nhất lưới {best} {res['font'][best]['top1']:.1f} "
          f"| chọn {pick} | nguyên văn(norm=False) {res['font']['lit']['top1']:.1f} "
          f"| nền trắng cùng quang học {res['font']['picked_white_bg_same_optics_K1']['top1']:.1f}")
    if "fd" in res and res["fd"]:
        H.log(f"{b}: FD nhận dạng neo: trắng {res['fd']['white']['top1']:.1f} | giấy-chỉ {res['fd']['canvas_d0_s0']['top1']:.1f} | chọn {res['fd']['_picked']} {res['fd'][res['fd']['_picked']]['top1']:.1f} | nguyên văn {res['fd']['lit']['top1']:.1f}")
    return res


def main():
    (OUT / "tune").mkdir(parents=True, exist_ok=True)
    z = np.load(OUT / "emb" / "white_fd.npz", allow_pickle=False)
    wd = {str(k): v for k, v in zip(z["keys"], z["E"])}
    pool = H.make_pool()
    allres = {}
    for b in H.BOOKS:
        allres[b] = tune_book(pool, b, wd)
        H.jdump(allres, OUT / "tune.json")
    pool.close(); pool.join()


if __name__ == "__main__":
    main()
