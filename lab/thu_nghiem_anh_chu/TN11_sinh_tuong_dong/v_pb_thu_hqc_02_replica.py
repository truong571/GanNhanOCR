"""v_pb_thu_hqc_02_replica.py — PHẢN BIỆN độc lập hướng "Thử HQC quy mô lớn" (04/10): TÁI LẬP phép so C1 - C0 (f_font) với
  * hạt giống KHÁC (--seed), trang giấy KHÁC (loại 16 trang của người kiểm chứng), mảng giấy KHÁC (300 cửa sổ/trang rút trên TOÀN trang, giữ 2 mảng ít mực nhất, mực <= 35 %, điền điểm mực bằng điểm giấy sạch),
  * bộ ô tuning KHÁC (250 ô neo tự động, loại ô tuning của họ), tuning lại cùng lưới + cùng luật (mặc định trừ khi hơn >= 1,0 điểm),
  * mẫu ô đánh giá KHÁC (n ô ngẫu nhiên có chữ đúng, loại mọi ô tuning của hai bên).
Mã viết lại từ đầu (không import v_thu_hqc_lib); chỉ dùng mã sản xuất ở chế độ ĐỌC (SI.Glyphs, SI.MultiEnc, t02_embed.page_crops... ) trên CPU.
Biến thể đo (cùng tập ô, ghép cặp theo ô): C0 (cột sản xuất), THEIR (nhúng C1 đã lưu của họ, mảng của họ), MY_tc (cấu hình của HỌ, mảng của TÔI),
MY_my (cấu hình tuning lại của TÔI, mảng của TÔI; bỏ nếu trùng), WHITE_tc (cùng quang học, nền trắng), tuỳ chọn --p1 (mảng thứ hai).
  .venv/bin/python lab/.../v_pb_thu_hqc_02_replica.py --book B34 --seed 777001 --n-eval 3000 --workers 3
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

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
WIN = 112


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------------------------------------ worker
_W = {}


def _winit():
    import torch
    torch.set_num_threads(1)
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    A = Assets(verify=False)
    _W["enc"] = SI.MultiEnc([A.ext(SI.ENC_FILES["v1"]), A.ext(SI.ENC_FILES["v2"])], "cpu", True)
    _W["gl"] = SI.Glyphs()


def render(g, patch, ink, P, d, sigma, white):
    """glyph (uint8 112x112, nền 255, nét 0) -> độ đậm d -> nhoè sigma -> thu về PxP (INTER_AREA) -> alpha-blend lên (mảng giấy | trắng)."""
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
    if white:
        paper = np.full((P, P), 255.0, np.float32); iv = 0.0
    else:
        h = patch.shape[0]; o = (h - P) // 2
        paper = patch[o:o + P, o:o + P].astype(np.float32); iv = float(ink)
    return np.clip(np.rint(paper * (1.0 - a) + iv * a), 0, 255).astype(np.uint8)


def _task(t):
    """t: dict(chars, cfg, patches[list of uint8 112x112], ink) -> (id, E[n,512]; NaN nếu chữ không có glyph font)."""
    enc, gl = _W["enc"], _W["gl"]
    chars, cfg = t["chars"], t["cfg"]
    out = np.full((len(chars), 512), np.nan, np.float32)
    imgs, rows = [], []
    for j, ch in enumerate(chars):
        g = gl.font(ch)
        if g is None:
            continue
        for p in t["patches"]:
            imgs.append(render(g, p, t["ink"], cfg["P"], cfg["d"], cfg["sigma"], cfg["white"])); rows.append(j)
    if imgs:
        E = np.concatenate([enc.embed(imgs[s:s + 32], norm=bool(cfg["norm"])) for s in range(0, len(imgs), 32)])
        K = len(t["patches"])
        E = E.reshape(-1, K, 512).mean(1)
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        out[np.array(rows[::K])] = E
    return t["id"], out


def embed(pool, chars, cfg, patches, ink, chunk=64, cache=None):
    """Nhúng danh sách chữ; nếu `cache` (đường dẫn .npz) cho trước thì đọc/ghi tiếp (chạy lại không mất việc)."""
    chars = list(dict.fromkeys(chars))
    res = {}
    if cache is not None and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        res = {str(k): v for k, v in zip(z["keys"], z["E"])}
    todo = [c for c in chars if c not in res]
    tasks = [dict(id=s, chars=todo[s:s + chunk], cfg=cfg, patches=patches, ink=ink) for s in range(0, len(todo), chunk)]
    t0 = time.time()

    def save():
        if cache is None:
            return
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        ks = list(res)
        tmp = Path(str(cache) + ".tmp.npz")
        np.savez(tmp, keys=np.array(ks), E=np.stack([res[k] for k in ks]).astype(np.float32))
        tmp.replace(cache)
    for n, (tid, E) in enumerate(pool.imap(_task, tasks, chunksize=1)):
        for ch, v in zip(todo[tid:tid + chunk], E):
            res[ch] = v
        if (n + 1) % 40 == 0:
            log(f"    {min((n + 1) * chunk, len(todo))}/{len(todo)} chữ cần nhúng, {time.time() - t0:.0f}s")
            save()
    save()
    return {c: res[c] for c in chars}


# ------------------------------------------------------------------------------------------------ mô hình giấy (viết lại)
def build_paper(b, D, rng, n_pages=16, n_win=300, max_ink=0.35):
    from pipeline.align_engine.build_dataset import load_original_page
    cfg = T.BOOKS[b]; prep = T.REPO / cfg["prep"]
    theirs = set(json.loads((SRC / "giay" / f"{b}.json").read_text(encoding="utf-8"))["pages_used"])
    allp = sorted(D.page.unique())
    cand = [p for p in allp if p not in theirs] or allp
    sel = sorted(rng.choice(cand, size=min(n_pages, len(cand)), replace=False).tolist())
    ink_core, paper90, pool, pool_pg, pool_st = [], [], [], [], []
    for pg in sel:
        P0 = cv2.imread(str(prep / "pages" / f"{pg}.png"), cv2.IMREAD_COLOR)
        O = load_original_page(prep, pg, shape_like=P0.shape[:2])
        g = cv2.cvtColor(O, cv2.COLOR_BGR2GRAY); H, W = g.shape
        bbs = np.array([bb for bb in (T.bbox_of(x) for x in D[D.page == pg].bbox) if bb])
        pi, pp = [], []
        for bb in bbs[rng.permutation(len(bbs))[:80]]:
            a, c, e, f = [int(v) for v in bb]
            reg = g[max(0, c):max(0, f), max(0, a):max(0, e)]
            if reg.size >= 64:
                pi.append(np.percentile(reg, 3)); pp.append(np.percentile(reg, 90))
        if not pi:
            continue
        ink_core += pi; paper90 += pp
        Ip, Pp = float(np.median(pi)), float(np.median(pp))
        cands = []
        m = int(0.03 * min(H, W))
        for _ in range(n_win):
            x = int(rng.integers(m, max(m + 1, W - WIN - m))); y = int(rng.integers(m, max(m + 1, H - WIN - m)))
            win = g[y:y + WIN, x:x + WIN]
            if win.shape != (WIN, WIN):
                continue
            if abs(float(np.percentile(win, 90)) - Pp) > 15:       # chỉ lấy mảng có mức giấy gần giấy trong khối chữ
                continue
            thr = Pp - 0.35 * max(Pp - Ip, 20.0)
            mask = cv2.dilate((win < thr).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            cands.append((float(mask.mean()), win, mask))
        cands.sort(key=lambda t: t[0])
        for fr, win, mask in cands[:2]:
            if fr > max_ink:
                continue
            w2 = win.copy()
            if mask.any():
                clean = win[~mask]
                w2[mask] = rng.choice(clean, size=int(mask.sum()))
            pool.append(w2); pool_pg.append(pg); pool_st.append(dict(page=pg, ink_frac=fr, mean=float(w2.mean()), std=float(w2.std())))
    assert len(pool) >= 3, f"{b}: chỉ có {len(pool)} mảng giấy sạch"
    pool = np.stack(pool).astype(np.uint8)
    pgs = np.array(pool_pg)
    up = sorted(set(pool_pg))
    pick = []
    for j in rng.permutation(len(up))[:3]:
        cand_i = np.nonzero(pgs == up[j])[0]
        pick.append(int(rng.choice(cand_i)))
    while len(pick) < 3:
        pick.append(int(rng.integers(len(pool))))
    ink = float(np.median(ink_core))
    st = dict(pages=sel, n_pool=int(len(pool)), n_pool_pages=len(up), ink=ink, paper_p90=float(np.median(paper90)),
              pool_ink_frac_median=float(np.median([s["ink_frac"] for s in pool_st])),
              pool_mean_median=float(np.median([s["mean"] for s in pool_st])), pool_std_median=float(np.median([s["std"] for s in pool_st])),
              pick_ids=pick, pick_pages=[pool_pg[i] for i in pick], overlap_pages_with_theirs=len(set(sel) & theirs))
    return pool, pick, ink, st


def p_font_of(b, gl):
    meta = np.load(T.OUT / "emb" / f"{b}_crop_meta.npy")
    ok = meta[:, 3] > 0
    L = float(np.median(np.maximum(meta[ok, 0], meta[ok, 1])))
    ls = []
    for x in range(0x4E00, 0x4E00 + 3000):
        g = gl.font(chr(x))
        if g is None:
            continue
        ys, xs = np.nonzero(g < 128)
        if len(ys):
            ls.append(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    Lf = float(np.median(ls))
    return int(round(L * 112.0 / Lf)), L, Lf


# ------------------------------------------------------------------------------------------------ chỉ số
def per_cell_ok(i, s, y):
    s9 = np.where(np.isnan(s), np.float32(-9), s).astype(np.float32)
    starts = np.r_[0, np.nonzero(np.diff(i))[0] + 1]
    mx = np.maximum.reduceat(s9, starts)
    cell_of = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(i)]))
    pos = np.nonzero(s9 == mx[cell_of])[0]
    _, ui = np.unique(cell_of[pos], return_index=True)
    return y[pos[ui]].astype(float), i[starts]


def cluster_ci(delta, cl, B=2000, seed=0):
    up, inv = np.unique(cl, return_inverse=True)
    k = np.bincount(inv, weights=delta, minlength=len(up)); n = np.bincount(inv, minlength=len(up)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(up), size=(B, len(up)))
    r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(delta.mean()), float(np.quantile(r, .025)), float(np.quantile(r, .975)), len(up)


def scores(F, E, emb):
    cats = pd.Categorical(F.c)
    G = np.zeros((len(cats.categories), 512), np.float32); has = np.zeros(len(cats.categories), bool)
    for j, ch in enumerate(cats.categories):
        v = emb.get(str(ch))
        if v is not None and not np.isnan(v[0]):
            G[j] = v; has[j] = True
    codes = cats.codes.astype(np.int64)
    ii = F.i.values.astype(np.int64)
    s = np.full(len(F), np.nan, np.float32)
    k = np.nonzero(has[codes])[0]
    for a in range(0, len(k), 300000):
        kk = k[a:a + 300000]
        s[kk] = np.einsum("ij,ij->i", E[ii[kk]], G[codes[kk]])
    return s


def ident(E, rows, labels, emb):
    chars = [c for c in sorted(set(labels)) if c in emb and not np.isnan(emb[c][0])]
    G = np.stack([emb[c] for c in chars]).astype(np.float32)
    idx = {c: j for j, c in enumerate(chars)}
    keep = np.array([l in idx for l in labels])
    S = E[rows[keep]] @ G.T
    y = np.array([idx[l] for l in np.asarray(labels)[keep]])
    return float((S.argmax(1) == y).mean() * 100), int(keep.sum())


# ------------------------------------------------------------------------------------------------ chính
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--seed", type=int, default=777001)
    ap.add_argument("--n-eval", type=int, default=2500)
    ap.add_argument("--n-tune", type=int, default=250)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--p1", action="store_true", help="thêm biến thể: cấu hình của họ trên mảng thứ hai của tôi")
    ap.add_argument("--skip-my", action="store_true", help="không tuning lại; chỉ dùng cấu hình của họ")
    ap.add_argument("--no-white", action="store_true", help="bỏ biến thể nền trắng (tiết kiệm CPU)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    b = a.book
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed + 17 * len(b) + ord(b[0]))
    res = dict(book=b, seed=a.seed, n_eval_req=a.n_eval, device="cpu")
    D = T.load_base(b)
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "n_self", "f_font"]]
    hy = F.groupby("i").y.max(); truth = hy[hy == 1].index.values
    cells_df = pd.read_pickle(T.OUT / "cand" / f"{b}_cells.pkl")
    E = np.load(T.OUT / "emb" / f"{b}_enc.npy").astype(np.float32)
    tune_t = json.loads((SRC / "tune.json").read_text(encoding="utf-8"))[b]
    their_cfg = tune_t["cfg_font"]
    excl = set(tune_t["tune_cells"]) | set(tune_t["tune_cells_fd"])
    from pipeline.gold_exact import signals_img as SI
    gl = SI.Glyphs()
    P_font, L_med, Lf = p_font_of(b, gl)
    res["P_font_mine"] = P_font; res["P_font_theirs"] = tune_t["P_font"]; res["their_cfg"] = their_cfg
    log(f"{b}: P_font mình {P_font} (L_med {L_med}, Lf {Lf}) vs họ {tune_t['P_font']}")
    # ---- giấy
    pool_p, pick, ink, st = build_paper(b, D, rng)
    res["paper_mine"] = st
    patches = [pool_p[j] for j in pick]
    log(f"{b}: giấy mới: {st['n_pool']} mảng/{st['n_pool_pages']} trang, mực {ink:.0f}, giấy p90 {st['paper_p90']:.0f}, ink_frac TV {st['pool_ink_frac_median']:.3f}, "
        f"TB {st['pool_mean_median']:.0f} σ {st['pool_std_median']:.1f}; trùng trang với họ: {st['overlap_pages_with_theirs']}")
    pool = __import__("multiprocessing").get_context("spawn").Pool(a.workers, initializer=_winit)

    def cfgd(P, d, s, norm=True, white=False):
        return dict(P=P, d=d, sigma=s, norm=norm, white=white)

    # ---- tuning lại (ô neo tự động, loại ô tuning của họ)
    my_cfg = None
    if not a.skip_my:
        anc = np.nonzero(cells_df.anchor.values.astype(bool) & (cells_df.crop_ok.values > 0))[0]
        anc = np.array([i for i in anc if i not in excl])
        Tb = np.sort(rng.choice(anc, size=min(a.n_tune, len(anc)), replace=False))
        lab = D.label.values[Tb]; L = sorted(set(lab))
        res["tune_cells"] = [int(x) for x in Tb]
        grid = []
        for P in (None, P_font):
            for d in (-1, 0, 1):
                for s in (0.0, 1.5):
                    grid.append((f"{'canvas' if P is None else 'native'}_d{d}_s{s:g}", cfgd(P, d, s)))
        tr = {}
        wh = embed(pool, L, cfgd(None, 0, 0.0, norm=False, white=True), patches[:1], ink)
        tr["white_prod"] = ident(E, Tb, lab, wh)[0]
        for nm, c in grid:
            em = embed(pool, L, c, patches[:1], ink)
            tr[nm] = ident(E, Tb, lab, em)[0]
        default = "canvas_d0_s0"
        best = max([g[0] for g in grid], key=lambda k: (tr[k], -[g[0] for g in grid].index(k)))
        pick_nm = best if tr[best] - tr[default] >= 1.0 else default
        my_cfg = dict(dict(grid)[pick_nm])
        res["tune"] = dict(n=int(len(Tb)), n_chars=len(L), top1=tr, best=best, picked=pick_nm, cfg=my_cfg)
        same = (my_cfg["P"] == (their_cfg["P"] if their_cfg["P"] else None)) and my_cfg["d"] == their_cfg["d"] and my_cfg["sigma"] == their_cfg["sigma"]
        res["tune"]["same_as_theirs"] = bool(same)
        log(f"{b}: tuning lại: chọn {pick_nm} (họ: P={their_cfg['P']} d={their_cfg['d']} σ={their_cfg['sigma']}) ; trắng {tr['white_prod']:.1f}, mặc định {tr[default]:.1f}, tốt nhất {best} {tr[best]:.1f}")
        excl |= set(int(x) for x in Tb)
    # ---- mẫu đánh giá
    tc = np.array(sorted(set(int(x) for x in truth) - excl))
    ev = np.sort(rng.choice(tc, size=min(a.n_eval, len(tc)), replace=False))
    res["n_eval"] = int(len(ev))
    Fe = F[F.i.isin(set(int(x) for x in ev))].reset_index(drop=True)
    chars = sorted(set(Fe.c.astype(str)[Fe.f_font.notna().values]))
    log(f"{b}: mẫu đánh giá {len(ev)} ô, {len(Fe)} hàng, {len(chars)} chữ có glyph font")
    y = Fe.y.values.astype(np.int8)
    tru = Fe[Fe.y == 1].drop_duplicates("i").set_index("i")
    page = D.page.values
    cdir = OUT / "cache"
    ctag = f"replica{a.tag}_{b}_s{a.seed}"
    results = {"C0": Fe.f_font.values}
    z = np.load(SRC / "emb" / f"primary_p0_{b}_font.npz", allow_pickle=False)
    their = {str(k): v for k, v in zip(z["keys"], z["E"])}
    results["THEIR"] = scores(Fe, E, their)
    tcfg = cfgd(their_cfg["P"] if their_cfg["P"] else None, their_cfg["d"], their_cfg["sigma"], norm=bool(their_cfg["norm"]))
    wcfg = dict(tcfg); wcfg["white"] = True
    plan = [("MY_tc", tcfg, patches[:1], "cfg họ, mảng #0 của tôi")]
    if my_cfg is not None and not res["tune"]["same_as_theirs"]:
        plan.append(("MY_my", my_cfg, patches[:1], "cfg tuning lại của tôi, mảng #0"))
    if not a.no_white:
        plan.append(("WHITE_tc", wcfg, patches[:1], "cfg họ, nền trắng"))
    if a.p1:
        plan.append(("MY_tc_p1", tcfg, patches[1:2], "cfg họ, mảng #1 của tôi"))

    def evaluate():
        o0, cells = per_cell_ok(Fe.i.values, results["C0"], y)
        pg = page[cells]
        tch = tru.c.reindex(cells).astype(str).values
        rare = (tru.n_self.fillna(0) < 2).reindex(cells).values.astype(bool)
        res["top1"] = {"C0": round(float(o0.mean() * 100), 2)}; res["delta_vs_C0"] = {}
        for nm, s_ in results.items():
            if nm == "C0":
                continue
            o, c2 = per_cell_ok(Fe.i.values, s_, y)
            assert (c2 == cells).all()
            miss = int((~np.isnan(results["C0"]) & np.isnan(s_)).sum())
            res["top1"][nm] = round(float(o.mean() * 100), 2)
            d = (o - o0) * 100
            m, lo, hi, npg = cluster_ci(d, pg, seed=a.seed + 1)
            m2, lo2, hi2, nch = cluster_ci(d, tch, seed=a.seed + 2)
            mr, lr, hr, _ = cluster_ci(d[rare], pg[rare], seed=a.seed + 3)
            res["delta_vs_C0"][nm] = dict(delta=round(m, 2), ci_trang=[round(lo, 2), round(hi, 2)], ci_chu_dung=[round(lo2, 2), round(hi2, 2)], n=int(len(d)),
                                          n_trang=npg, n_chu_dung=nch, thieu_glyph_moi=miss,
                                          chu_hiem=dict(n=int(rare.sum()), delta=round(mr, 2), ci_trang=[round(lr, 2), round(hr, 2)]))

        def pair(x, y_):
            ox, _ = per_cell_ok(Fe.i.values, results[x], y); oy, _ = per_cell_ok(Fe.i.values, results[y_], y)
            d = (ox - oy) * 100
            m, lo, hi, _ = cluster_ci(d, pg, seed=a.seed + 9)
            return dict(delta=round(m, 2), ci_trang=[round(lo, 2), round(hi, 2)])
        res["pairs"] = {"MY_tc-THEIR (đổi mảng giấy)": pair("MY_tc", "THEIR")} if "MY_tc" in results else {}
        if "WHITE_tc" in results and "MY_tc" in results:
            res["pairs"]["MY_tc-WHITE_tc (tác dụng của GIẤY, cùng quang học)"] = pair("MY_tc", "WHITE_tc")
            res["pairs"]["WHITE_tc-C0 (tác dụng của QUANG HỌC thuần)"] = pair("WHITE_tc", "C0")
        if "MY_my" in results and "MY_tc" in results:
            res["pairs"]["MY_my-MY_tc (cấu hình tuning lại vs cấu hình của họ, cùng mảng)"] = pair("MY_my", "MY_tc")
        if "MY_tc_p1" in results and "MY_tc" in results:
            res["pairs"]["MY_tc_p1-MY_tc (mảng #1 vs #0 của tôi)"] = pair("MY_tc_p1", "MY_tc")
        (OUT / f"replica{a.tag}_{b}_s{a.seed}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"{b}: " + json.dumps({k: v for k, v in res.items() if k in ("top1", "delta_vs_C0", "pairs")}, ensure_ascii=False))

    evaluate()                      # chỉ C0 + THEIR (nhúng có sẵn) — số tham chiếu trên mẫu của tôi
    for nm, cfg_, pat_, desc in plan:
        t0 = time.time()
        log(f"{b}: nhúng {nm} ({desc}) {len(chars)} chữ")
        results[nm] = scores(Fe, E, embed(pool, chars, cfg_, pat_, ink, cache=cdir / f"{ctag}_{nm}.npz"))
        log(f"  xong {time.time() - t0:.0f}s")
        evaluate()
    pool.close(); pool.join()


if __name__ == "__main__":
    main()
