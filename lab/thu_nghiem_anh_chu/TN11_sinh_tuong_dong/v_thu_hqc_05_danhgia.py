"""v_thu_hqc_05_danhgia.py — ĐO: Top-1 MỘT tín hiệu (f_font, f_fd) trên TOÀN BỘ ứng viên của ô, mọi ô có chữ đúng trong ứng viên
(C0 sản xuất = cột có sẵn đã hiệu chuẩn; C1 = HQC2 nền giấy sách; C2 = HQC3 nhị phân chuẩn trên mẫu con), tách chữ hiếm (n_self < 2) / chữ thường,
cos TB chữ đúng + margin, CI95 % bootstrap cụm trang, chênh lệch cặp (C1-C0, C2-C0, C2-C1, nguyên văn-C0); tiêu chí đăng ký trước.
Ra: measure_out/_tn11/verify/thu_hqc/danhgia.json (+ ket_qua.json do v_thu_hqc_07_tonghop.py gộp).
  .venv/bin/python .../v_thu_hqc_05_danhgia.py [--books B34 ...] [--variants primary literal c2 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v_thu_hqc_lib as H  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = H.OUT


def load_npz(f):
    f = Path(f)
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    return {str(k): v for k, v in zip(z["keys"], z["E"])}


def load_primary(b, kind, ks=(0, 1, 2)):
    """C1 = trung bình (rồi chuẩn hoá L2) embedding của K mảng giấy (lần chạy pk=0..K-1); None nếu thiếu lần nào."""
    parts = [load_npz(OUT / "emb" / f"primary_p{k}_{b}_{kind}.npz") for k in ks]
    if any(p is None for p in parts):
        return None
    keys = set(parts[0]).intersection(*[set(p) for p in parts[1:]])
    out = {}
    for c in keys:
        v = np.mean([p[c] for p in parts], axis=0)
        n = np.linalg.norm(v)
        out[c] = v / n if n > 0 else v
    return out


def available_ks(b, kind):
    """tiền tố liên tục các mảng giấy đã nhúng xong (0..K-1)."""
    if KS_FORCE.get(kind) is not None:
        return tuple(KS_FORCE[kind])
    ks = []
    for k in (0, 1, 2):
        if (OUT / "emb" / f"primary_p{k}_{b}_{kind}.npz").exists():
            ks.append(k)
        else:
            break
    return tuple(ks)


def rows_scores(Fs, Emat, row_of_cell, emb):
    """điểm theo hàng Fs: Emat[row_of_cell[i]] · emb[c] (NaN nếu chữ không có)."""
    cats = list(Fs.c.cat.categories)
    G = np.zeros((len(cats), 512), np.float32); has = np.zeros(len(cats), bool)
    for j, ch in enumerate(cats):
        v = emb.get(ch)
        if v is not None and not np.isnan(v[0]):
            G[j] = v; has[j] = True
    codes = Fs.c.cat.codes.values.astype(np.int64)
    gi = np.where(has[codes], codes, -1)
    ei = row_of_cell[Fs.i.values.astype(np.int64)]
    return H.rowdot(Emat, ei, G, gi)


def covered_cells(Fs, s_prod, s_new):
    """ô mà MỌI hàng có glyph sản xuất (điểm không NaN) đều có điểm mới (không NaN)."""
    bad = ~np.isnan(s_prod) & np.isnan(s_new)
    badc = pd.Series(bad).groupby(Fs.i.values).any()
    return badc.index.values[~badc.values]


def summarize(m, rare, pages):
    """m: DataFrame theo ô (ok, cos_true, margin); rare: bool theo ô; pages: trang theo ô. -> dict theo nhóm."""
    out = {}
    for g, mask in (("tat_ca", np.ones(len(m), bool)), ("chu_hiem", rare), ("chu_thuong", ~rare)):
        if mask.sum() == 0:
            continue
        ok = m.ok.values[mask] * 100.0
        t, lo, hi = H.boot_mean(ok, pages[mask])
        ct = H.boot_mean(m.cos_true.values[mask], pages[mask])
        mg = H.boot_mean(m.margin.values[mask], pages[mask])
        out[g] = dict(n=int(mask.sum()), top1=round(t, 2), ci_thap=round(lo, 2), ci_cao=round(hi, 2),
                      cos_true=None if np.isnan(ct[0]) else round(ct[0], 4), cos_true_n=int((~np.isnan(m.cos_true.values[mask])).sum()),
                      margin=None if np.isnan(mg[0]) else round(mg[0], 4))
    return out


def delta(ma, mb, rare, pages):
    out = {}
    for g, mask in (("tat_ca", np.ones(len(ma), bool)), ("chu_hiem", rare), ("chu_thuong", ~rare)):
        if mask.sum() == 0:
            continue
        d, lo, hi, n = H.boot_diff(ma.ok.values[mask].astype(float), mb.ok.values[mask].astype(float), pages[mask])
        dc = H.boot_diff(ma.cos_true.values[mask], mb.cos_true.values[mask], pages[mask])
        out[g] = dict(n=int(n), delta=round(d * 100, 2), ci_thap=round(lo * 100, 2), ci_cao=round(hi * 100, 2),
                      delta_cos_true=None if np.isnan(dc[0]) else round(dc[0], 4))
    return out


KS_FORCE = {"font": None, "fd": None}
SETS = None


def eval_book(b, variants, exclude_tune):
    Bk = H.load_eval_book(b)
    F, E, rare_s, page_all = Bk["F"], Bk["E"], Bk["rare"], Bk["page"]
    N = len(page_all)
    S = np.load(OUT / f"mau_{b}.npz")
    res = dict(book=b, n_truth_cells=int(len(Bk["cells"])), sets={})
    ident = np.arange(N)
    tune = json.loads((OUT / "tune.json").read_text(encoding="utf-8")) if (OUT / "tune.json").exists() else {}
    tcells = set(tune.get(b, {}).get("tune_cells", [])) | set(tune.get(b, {}).get("tune_cells_fd", []))
    for kind, col in (("font", "f_font"), ("fd", "f_fd")):
        s0_all = F[col].values
        ks = available_ks(b, kind)       # K = số mảng giấy đã có (trung bình embedding)
        emb1 = load_primary(b, kind, ks) if ("primary" in variants and ks) else None
        emb1k1 = load_primary(b, kind, (0,)) if ("primary" in variants and ks) else None
        res.setdefault("K_dung", {})[kind] = len(ks)
        embL = load_npz(OUT / "emb" / f"literal_{b}_{kind}.npz") if "literal" in variants else None
        embP = load_npz(OUT / "emb" / f"paperonly_{b}_{kind}.npz") if "paperonly" in variants else None
        embW = load_npz(OUT / "emb" / f"ctrlwhite_{b}_{kind}.npz") if "ctrlwhite" in variants else None
        sc = {"C0": s0_all}
        if emb1 is not None:
            sc["C1"] = rows_scores(F, E, ident, emb1)
        # ---------------- tập ALL (mọi ô được phủ)
        if "C1" in sc and (SETS is None or "all" in SETS):
            cov = covered_cells(F, s0_all, sc["C1"])
            mk = F.i.isin(set(int(x) for x in cov)).values
            Fs = F[mk]
            m0 = H.cell_metrics(Fs.i.values, s0_all[mk], Fs.y.values); m1 = H.cell_metrics(Fs.i.values, sc["C1"][mk], Fs.y.values)
            cells = m0.index.values
            rare = rare_s.reindex(cells).values.astype(bool); pg = page_all[cells]
            r = dict(n_cells=int(len(cells)), full_coverage=bool(len(cells) == len(Bk["cells"])), C0=summarize(m0, rare, pg), C1=summarize(m1, rare, pg),
                     d_C1_C0=delta(m1, m0, rare, pg))
            if tcells:
                keep = ~np.isin(cells, list(tcells))
                r["d_C1_C0_ngoai_o_tuning"] = delta(m1[keep], m0[keep], rare[keep], pg[keep])
            res["sets"].setdefault("all", {})[kind] = r
        # ---------------- mẫu con (3000 / 1000): C0, C1, C2, nguyên văn, đối chứng
        for sname in ("s3000", "s1000", "s500"):
            if SETS is not None and sname not in SETS:
                continue
            cells_s = S["s1000"][:500] if sname == "s500" else S[sname]
            conds = {}
            Fm = F.i.isin(set(int(x) for x in cells_s)).values
            Fs = F[Fm]
            base = {"C0": s0_all[Fm]}
            if "C1" in sc:
                base["C1"] = sc["C1"][Fm]
            for nm, emb in (("C1K1", emb1k1), ("LIT", embL), ("PAPERONLY", embP), ("CTRLWHITE", embW)):
                if emb is not None:
                    base[nm] = rows_scores(Fs, E, ident, emb)
            # C2 / C2g (cần crop nhị phân của mẫu con)
            for pre, nm in (("c2", "C2"), ("c2g", "C2G")):
                f = OUT / "emb" / f"{pre}_crops_{b}.npy"
                g = load_npz(OUT / "emb" / f"{pre}_canon_{kind}.npz")
                if f.exists() and g is not None:
                    ids = np.load(OUT / "emb" / f"{pre}_crops_{b}_ids.npy"); E2 = np.load(f)
                    row = np.full(N, -1, np.int64); row[ids] = np.arange(len(ids))
                    ok_rows = row[Fs.i.values] >= 0
                    s = np.full(len(Fs), np.nan, np.float32)
                    sub = Fs[ok_rows]
                    s[ok_rows] = rows_scores(sub, E2, row, g)
                    base[nm] = s
            # mỗi điều kiện tính trên các ô nó PHỦ ĐỦ; so cặp trên giao của hai điều kiện (tránh tập con lệch do phủ dở)
            mm = {}
            for nm, sv in base.items():
                if nm == "C0":
                    mm[nm] = H.cell_metrics(Fs.i.values, sv, Fs.y.values)
                    continue
                cov = covered_cells(Fs, base["C0"], sv)
                if len(cov) == 0:
                    continue
                mk2 = Fs.i.isin(set(int(x) for x in cov)).values
                mm[nm] = H.cell_metrics(Fs.i.values[mk2], sv[mk2], Fs.y.values[mk2])
            ref = mm.get("C1", mm["C0"])                       # tập chính = ô được C1 phủ (nếu có)
            def at(m, cells):
                return m.loc[cells]
            def aux(cells):
                return rare_s.reindex(cells).values.astype(bool), page_all[cells]
            cells0 = ref.index.values
            rare, pg = aux(cells0)
            r = dict(n_cells=int(len(cells0)), n_yeu_cau=int(len(cells_s)), phu_day_du=bool(len(cells0) == len(cells_s)), K_dung=len(ks),
                     cond={}, delta={}, n_theo_dieu_kien={nm: int(len(m)) for nm, m in mm.items()})
            for nm, m in mm.items():
                cc = cells0 if nm == "C0" else m.index.values
                r["cond"][nm] = summarize(at(m, cc), *aux(cc))
            for a_, c_ in (("C1", "C0"), ("C2", "C0"), ("C2", "C1"), ("C1K1", "C0"), ("C1", "C1K1"), ("LIT", "C0"), ("LIT", "C1K1"),
                           ("PAPERONLY", "C0"), ("PAPERONLY", "C1K1"), ("CTRLWHITE", "C0"), ("CTRLWHITE", "C1K1"), ("C2G", "C0"), ("C2", "C2G")):
                if a_ in mm and c_ in mm:
                    common = mm[a_].index.intersection(mm[c_].index).values
                    rr, pp = aux(common)
                    r["delta"][f"{a_}-{c_}"] = delta(at(mm[a_], common), at(mm[c_], common), rr, pp)
            res["sets"].setdefault(sname, {})[kind] = r
        H.log(f"{b} {kind}: xong")
    # ---------------- lệch nền: cos(embedding glyph nền trắng, glyph nền giấy)
    res["diag_nen"] = diag_nen(b, F)
    return res


def cos_stats(x):
    x = np.asarray(x)
    return dict(n=int(len(x)), mean=round(float(x.mean()), 4), p05=round(float(np.quantile(x, .05)), 4), p50=round(float(np.median(x)), 4),
                p95=round(float(np.quantile(x, .95)), 4), min=round(float(x.min()), 4))


def diag_nen(b, F):
    out = {}
    wf = load_npz(OUT / "emb" / "white_font.npz"); wd = load_npz(OUT / "emb" / "white_fd.npz")
    S = np.load(OUT / f"mau_{b}.npz")
    cal = set(int(x) for x in S["cal"])
    sub = F[F.i.isin(cal)]
    chars_cal = sorted(set(sub.c.astype(str)[sub.f_font.notna().values]))
    for kind, white, chars in (("font", wf, chars_cal), ("fd", wd, None)):
        for var in ("primary", "literal", "paperonly", "ctrlwhite"):
            e = (load_primary(b, kind, available_ks(b, kind)) if available_ks(b, kind) else None) if var == "primary" else load_npz(OUT / "emb" / f"{var}_{b}_{kind}.npz")
            if e is None:
                continue
            ch = [c for c in (chars if chars is not None else e.keys()) if c in white and c in e and not np.isnan(white[c][0]) and not np.isnan(e[c][0])]
            if not ch:
                continue
            W = np.stack([white[c] for c in ch]); P = np.stack([e[c] for c in ch])
            cs = np.einsum("ij,ij->i", W, P) / (np.linalg.norm(W, axis=1) * np.linalg.norm(P, axis=1))
            out[f"{kind}_{var}"] = cos_stats(cs)
        if kind == "font" and chars_cal:
            # thước so: cos giữa glyph NỀN TRẮNG của hai chữ khác nhau cùng ô (cặp ứng viên) trong ô hiệu chuẩn
            Fa = sub[sub.f_font.notna()]
            rng = np.random.default_rng(H.SEED + 51)
            cs = []
            for i, g in Fa.groupby("i"):
                cc = [c for c in g.c.astype(str).tolist() if c in wf and not np.isnan(wf[c][0])]
                if len(cc) >= 2:
                    a, c2 = rng.choice(len(cc), 2, replace=False)
                    cs.append(float(wf[cc[a]] @ wf[cc[c2]] / (np.linalg.norm(wf[cc[a]]) * np.linalg.norm(wf[cc[c2]]))))
            out["font_white_vs_white_khac_chu_cung_o"] = cos_stats(cs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="*", default=H.BOOKS)
    ap.add_argument("--variants", nargs="*", default=["primary", "literal", "paperonly", "ctrlwhite"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--one", default=None)
    ap.add_argument("--ks-font", nargs="*", type=int, default=None, help="ép các mảng giấy dùng cho C1 font (vd 0 = K=1; 0 1 2 = K=3)")
    ap.add_argument("--ks-fd", nargs="*", type=int, default=None, help="như trên cho C1 FD")
    ap.add_argument("--sets", nargs="*", default=None, help="chỉ đo các tập này (all s3000 s1000 s500)")
    a = ap.parse_args()
    global KS_FORCE, SETS
    KS_FORCE = {"font": a.ks_font, "fd": a.ks_fd}
    SETS = a.sets
    if a.one:                                    # một sách trong tiến trình riêng (RAM)
        r = eval_book(a.one, a.variants, True)
        H.jdump(r, OUT / f"danhgia{a.tag}_{a.one}.json")
        return
    import subprocess
    res = {}
    for b in a.books:
        cmd = [sys.executable, __file__, "--one", b, "--variants", *a.variants, "--tag", a.tag]
        if a.ks_font is not None:
            cmd += ["--ks-font", *map(str, a.ks_font)]
        if a.ks_fd is not None:
            cmd += ["--ks-fd", *map(str, a.ks_fd)]
        if a.sets is not None:
            cmd += ["--sets", *a.sets]
        subprocess.run(cmd, check=True)
        res[b] = json.loads((OUT / f"danhgia{a.tag}_{b}.json").read_text(encoding="utf-8"))
    H.jdump(res, OUT / f"danhgia{a.tag}.json")
    H.log("XONG")


if __name__ == "__main__":
    main()
