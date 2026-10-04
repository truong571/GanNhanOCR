"""v_pb_thu_hqc_00_tai_tinh.py — PHẢN BIỆN độc lập hướng "Thử HQC quy mô lớn" (04/10).
Tính LẠI từ đầu (không import v_thu_hqc_lib) C0 và C1 (f_font, K=1) cho MỌI ô có chữ đúng, từ
  (a) cột sản xuất f_font trong measure_out/_tn8/cand/<bộ>.pkl  (C0)
  (b) nhúng glyph C1 mà người kiểm chứng đã lưu: measure_out/_tn11/verify/thu_hqc/emb/primary_p0_<bộ>_font.npz
rồi kiểm: số Top-1 / Δ có trùng không; CI cụm TRANG (hạt giống KHÁC), CI cụm CHỮ-ĐÚNG, CI ngây thơ theo ô; luật hoà; ô tuning;
và khẳng định K09 (n, hash, ...) khi cần. Chỉ ĐỌC; ghi vào measure_out/_tn11/verify/pb_thu_hqc/.
  .venv/bin/python lab/.../v_pb_thu_hqc_00_tai_tinh.py [B34 L16 TK B18]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out/_tn11/verify/pb_thu_hqc"
SRC = REPO / "measure_out/_tn11/verify/thu_hqc"
OUT.mkdir(parents=True, exist_ok=True)
B_BOOT = 2000
SEED_PB = 424242                       # KHÁC hạt giống của người kiểm chứng (20260930)


def cluster_ci(delta, cl, B=B_BOOT, seed=SEED_PB):
    """delta: vector theo ô (đã nhân 100 nếu muốn %), cl: nhãn cụm. Trả (mean, lo, hi) của mean(delta) bootstrap CỤM (ratio estimator)."""
    up, inv = np.unique(cl, return_inverse=True)
    k = np.bincount(inv, weights=delta, minlength=len(up)); n = np.bincount(inv, minlength=len(up)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(up), size=(B, len(up)))
    r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(delta.mean()), float(np.quantile(r, .025)), float(np.quantile(r, .975)), int(len(up))


def per_cell(i, s, y, tie="first"):
    """ok theo ô. i tăng dần theo hàng (cand/<bộ>.pkl đã nhóm theo ô). NaN -> -9. tie='first' (như TN8) | 'frac' (chia đều khi hoà) | 'worst' (hoà = sai)."""
    s9 = np.where(np.isnan(s), np.float32(-9), s).astype(np.float32)
    starts = np.r_[0, np.nonzero(np.diff(i))[0] + 1]
    mx = np.maximum.reduceat(s9, starts)
    cell_of = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(i)]))
    is_max = s9 == mx[cell_of]
    if tie == "first":
        # hàng đầu tiên đạt max trong ô
        first = np.full(len(starts), -1, np.int64)
        pos = np.nonzero(is_max)[0]
        # pos tăng dần; lấy phần tử đầu tiên của mỗi ô
        _, ui = np.unique(cell_of[pos], return_index=True)
        first = pos[ui]
        return y[first].astype(float), i[starts]
    cnt = np.bincount(cell_of, weights=is_max.astype(float), minlength=len(starts))
    hit = np.bincount(cell_of, weights=(is_max & (y == 1)).astype(float), minlength=len(starts))
    if tie == "frac":
        return hit / cnt, i[starts]
    return np.where(cnt == 1, hit, 0.0), i[starts]


def load_c1(b):
    z = np.load(SRC / "emb" / f"primary_p0_{b}_font.npz", allow_pickle=False)
    return {str(k): v for k, v in zip(z["keys"], z["E"])}


def run(b):
    D = T.load_base(b)
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "n_self", "f_font"]]
    assert (np.diff(F.i.values) >= 0).all(), "bảng ứng viên không nhóm theo ô"
    hy = F.groupby("i").y.max()
    cells = hy[hy == 1].index.values
    F = F[F.i.isin(cells)].reset_index(drop=True)
    E = np.load(T.OUT / "emb" / f"{b}_enc.npy").astype(np.float32)
    emb = load_c1(b)
    cats = pd.Categorical(F.c)
    G = np.zeros((len(cats.categories), 512), np.float32); has = np.zeros(len(cats.categories), bool)
    for j, ch in enumerate(cats.categories):
        v = emb.get(str(ch))
        if v is not None and not np.isnan(v[0]):
            G[j] = v; has[j] = True
    codes = cats.codes.astype(np.int64)
    ok_rows = has[codes]
    s1 = np.full(len(F), np.nan, np.float32)
    ii = F.i.values.astype(np.int64)
    k = np.nonzero(ok_rows)[0]
    for a in range(0, len(k), 300000):
        kk = k[a:a + 300000]
        s1[kk] = np.einsum("ij,ij->i", E[ii[kk]], G[codes[kk]])
    s0 = F.f_font.values.astype(np.float32)
    # phủ: hàng có glyph sản xuất mà thiếu glyph C1
    miss = int((~np.isnan(s0) & np.isnan(s1)).sum()); extra = int((np.isnan(s0) & ~np.isnan(s1)).sum())
    y = F.y.values.astype(np.int8)
    page = D.page.values
    # chữ đúng theo ô
    tru = F[F.y == 1].drop_duplicates("i").set_index("i")
    rare = (tru.n_self.fillna(0) < 2)
    res = dict(book=b, n_cells=int(len(cells)), rows=int(len(F)), miss_c1_where_prod=miss, c1_where_prod_nan=extra,
               n_pages_truth=int(len(set(page[cells]))), n_true_chars=int(tru.c.nunique()))
    for tie in ("first", "frac", "worst"):
        o0, c0 = per_cell(F.i.values, s0, y, tie)
        o1, c1 = per_cell(F.i.values, s1, y, tie)
        assert (c0 == c1).all()
        d = (o1 - o0) * 100
        pg = page[c0]
        m, lo, hi, npg = cluster_ci(d, pg)
        res[f"tie_{tie}"] = dict(top1_C0=round(float(o0.mean() * 100), 3), top1_C1=round(float(o1.mean() * 100), 3), delta=round(m, 3),
                                 ci_trang=[round(lo, 3), round(hi, 3)], n_trang=npg)
        if tie == "first":
            # CI cụm chữ-đúng, CI ngây thơ theo ô, CI cụm trang với hạt giống khác (đã) + jackknife trang
            tc = tru.c.reindex(c0).astype(str).values
            m2, lo2, hi2, nch = cluster_ci(d, tc, seed=SEED_PB + 1)
            res["tie_first"]["ci_chu_dung"] = [round(lo2, 3), round(hi2, 3)]; res["tie_first"]["n_chu_dung"] = nch
            m3, lo3, hi3, nce = cluster_ci(d, np.arange(len(d)), seed=SEED_PB + 2)
            res["tie_first"]["ci_o_ngay_tho"] = [round(lo3, 3), round(hi3, 3)]
            # độ lệch tiêu chuẩn jackknife theo trang (rời từng trang)
            up, inv = np.unique(pg, return_inverse=True)
            ks = np.bincount(inv, weights=d, minlength=len(up)); ns = np.bincount(inv, minlength=len(up)).astype(float)
            loo = (ks.sum() - ks) / (ns.sum() - ns)
            res["tie_first"]["jackknife_trang"] = dict(min=round(float(loo.min()), 3), max=round(float(loo.max()), 3),
                                                       n_trang_dao_dau=int((loo <= 0).sum()))
            # theo chữ hiếm / thường
            rr = rare.reindex(c0).values.astype(bool)
            for nm, mk in (("chu_hiem", rr), ("chu_thuong", ~rr)):
                mm, l_, h_, _ = cluster_ci(d[mk], pg[mk], seed=SEED_PB + 3)
                res["tie_first"][nm] = dict(n=int(mk.sum()), delta=round(mm, 3), ci=[round(l_, 3), round(h_, 3)])
            # tập trung theo chữ-đúng: phần Δ đến từ top-k chữ phổ biến
            vc = pd.Series(tc).value_counts()
            top = set(vc.index[:20])
            mk = np.array([t in top for t in tc])
            res["tie_first"]["top20_chu_dung"] = dict(share_cells=round(float(mk.mean()), 3), delta_in=round(float(d[mk].mean()), 3),
                                                       delta_out=round(float(d[~mk].mean()), 3), delta_tong=round(float(d.mean()), 3))
            # đánh dấu ô tuning của người kiểm chứng
            tune = json.loads((SRC / "tune.json").read_text(encoding="utf-8"))[b]
            tcl = np.isin(c0, np.array(tune["tune_cells"] + tune["tune_cells_fd"]))
            mt, lt, ht, _ = cluster_ci(d[~tcl], pg[~tcl], seed=SEED_PB + 4)
            res["tie_first"]["ngoai_o_tuning"] = dict(n=int((~tcl).sum()), delta=round(mt, 3), ci=[round(lt, 3), round(ht, 3)])
    print(json.dumps(res, ensure_ascii=False))
    (OUT / f"tai_tinh_{b}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    books = sys.argv[1:] or ["B34", "L16", "TK", "B18"]
    for b in books:
        run(b)
