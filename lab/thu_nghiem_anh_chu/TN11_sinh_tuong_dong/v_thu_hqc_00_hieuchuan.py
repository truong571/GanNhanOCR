"""v_thu_hqc_00_hieuchuan.py — BƯỚC 0 (hiệu chuẩn): tái tạo cột f_font / f_fd của measure_out/_tn8/cand/<bộ>.pkl từ glyph HQC1 (nền trắng)
bằng đúng cách sản xuất (SI.Glyphs.font 112×112 / SI.Glyphs.fdimg 96×96 -> MultiEnc.embed(norm=False) -> E[i]·G[c]) trên CPU,
rồi so với cột có sẵn; kiểm thêm crop cắt lại (t02_embed.page_crops) + nhúng CPU == E[i] sản xuất (cần cho C2).
Ra: measure_out/_tn11/verify/thu_hqc/hieuchuan.json, emb/white_font.npz, emb/white_fd.npz, mau_<bộ>.npz.
  .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_thu_hqc_00_hieuchuan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v_thu_hqc_lib as H  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = H.OUT


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "emb").mkdir(exist_ok=True)
    res = dict(seed=H.SEED, device="cpu", workers=H.NWORK, books={})
    # ---- 0a. toàn vẹn tài sản (sha256 như sản xuất)
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    A = Assets()
    for f in (SI.ENC_FILES["v1"], SI.ENC_FILES["v2"], *SI.FONT_FILES, *SI.FD_DIRS):
        A.ext(f)
    res["assets_sha256_ok"] = True
    H.log("tài sản (encoder v1+v2, 3 font, 2 thư mục FD) khớp sha256/digest MANIFEST")
    pool = H.make_pool()
    # ---- 0b. dữ liệu + mẫu con
    BK, chars_cal, S = {}, {}, {}
    all_font = set()
    for b in H.BOOKS:
        Bk = H.load_eval_book(b)
        BK[b] = Bk
        S[b] = H.sample_cells(b, Bk["cells"], 3000)
        np.savez(OUT / f"mau_{b}.npz", s3000=S[b], s1000=S[b][:1000], cal=S[b][:250])
        F = Bk["F"]
        cal = set(S[b][:250])
        sub = F[F.i.isin(cal)]
        ch = sorted(set(sub.c.astype(str)[sub.f_font.notna().values]))
        chars_cal[b] = ch
        all_font |= set(ch)
        H.log(f"{b}: ô có sự thật {len(Bk['cells'])}, mẫu 3000/1000/250, ký tự font của 250 ô hiệu chuẩn {len(ch)}")
    # ---- 0c. nhúng NỀN TRẮNG (HQC1) = cách sản xuất, CPU
    fonts = sorted(all_font)
    wf = H.embed_chars(pool, "white", "font", fonts, OUT / "emb" / "white_font.npz", tag="white_font")
    fdc = sorted(SI.Glyphs().fd)
    wd = H.embed_chars(pool, "white", "fd", fdc, OUT / "emb" / "white_fd.npz", tag="white_fd")
    nf = np.array([np.linalg.norm(v) for v in list(wf.values())[:2000] if not np.isnan(v[0])])
    res["norm_glyph_emb"] = dict(min=float(nf.min()), max=float(nf.max()), mean=float(nf.mean()))
    # ---- 0d. so với cột có sẵn
    for b in H.BOOKS:
        Bk = BK[b]; F = Bk["F"]
        r = {}
        # f_font: toàn bộ hàng của 250 ô hiệu chuẩn
        cal = S[b][:250]
        m = F.i.isin(set(cal)).values
        Fc = F[m]
        E = Bk["E"]
        emb = {c: wf[c] for c in chars_cal[b]}
        s_my = H.score_rows(dict(F=Fc.reset_index(drop=True), E=E), emb)
        s_pr = Fc.f_font.values
        both = ~np.isnan(s_my) & ~np.isnan(s_pr)
        d = np.abs(s_my[both] - s_pr[both])
        r["f_font"] = dict(n_rows=int(len(Fc)), n_both=int(both.sum()), nan_mismatch=int((np.isnan(s_my) != np.isnan(s_pr)).sum()),
                           max_abs=float(d.max()), mean_abs=float(d.mean()),
                           pearson=float(np.corrcoef(s_my[both], s_pr[both])[0, 1]))
        m1 = H.cell_metrics(Fc.i.values, s_my, Fc.y.values).ok.values
        m0 = H.cell_metrics(Fc.i.values, s_pr, Fc.y.values).ok.values
        r["f_font"]["top1_my"] = float(m1.mean()); r["f_font"]["top1_prod"] = float(m0.mean())
        r["f_font"]["cells_top1_equal"] = int((m1 == m0).sum()); r["f_font"]["n_cells"] = int(len(m0))
        # f_fd: TOÀN BỘ hàng (đã nhúng mọi FD)
        s_my = H.score_rows(Bk, wd)
        s_pr = F.f_fd.values
        both = ~np.isnan(s_my) & ~np.isnan(s_pr)
        d = np.abs(s_my[both] - s_pr[both])
        r["f_fd"] = dict(n_rows=int(len(F)), n_both=int(both.sum()), nan_mismatch=int((np.isnan(s_my) != np.isnan(s_pr)).sum()),
                         max_abs=float(d.max()), mean_abs=float(d.mean()), pearson=float(np.corrcoef(s_my[both], s_pr[both])[0, 1]))
        mm = H.cell_metrics(F.i.values, s_my, F.y.values)
        mp = H.cell_metrics(F.i.values, s_pr, F.y.values)
        r["f_fd"]["top1_my_all_cells"] = float(mm.ok.mean()); r["f_fd"]["top1_prod_all_cells"] = float(mp.ok.mean())
        r["f_fd"]["cells_top1_equal"] = int((mm.ok.values == mp.ok.values).sum()); r["f_fd"]["n_cells"] = int(len(mm))
        # so với p01_<b>.json (số đã công bố của dự án)
        p01 = REPO_P01(b)
        r["p01_ref"] = p01
        res["books"][b] = r
        H.log(f"{b}: f_font max|Δ| {r['f_font']['max_abs']:.2e} (r={r['f_font']['pearson']:.6f}, top1 {r['f_font']['top1_my']:.4f} vs {r['f_font']['top1_prod']:.4f}) | "
              f"f_fd max|Δ| {r['f_fd']['max_abs']:.2e} (r={r['f_fd']['pearson']:.6f}, top1 {r['f_fd']['top1_my_all_cells']:.4f} vs {r['f_fd']['top1_prod_all_cells']:.4f}; p01 {p01})")
    # ---- 0e. crop cắt lại + nhúng CPU == E sản xuất (150 ô/bộ), + mức nền thô vs sau giãn
    for b in H.BOOKS:
        Bk = BK[b]
        want = S[b][:150]
        crops = H.recut_crops(b, Bk["D"], want)
        ids = [int(i) for i in want if int(i) in crops]
        imgs = [crops[i] for i in ids]
        E2 = H.embed_images(pool, "crops_std", imgs)
        E1 = Bk["E"][ids]
        cos = np.einsum("ij,ij->i", E1, E2) / (np.linalg.norm(E1, axis=1) * np.linalg.norm(E2, axis=1))
        dmax = np.abs(E1 - E2).max(1)
        raw98 = np.array([np.percentile(g, 98) for g in imgs]); raw2 = np.array([np.percentile(g, 2) for g in imgs])
        def st(g):
            lo, hi = np.percentile(g, 2), np.percentile(g, 98)
            if hi - lo > 10:
                g = np.clip((g.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
            return np.percentile(g, 2), np.percentile(g, 98), np.median(g)
        S2 = np.array([st(g) for g in imgs])
        r = res["books"][b]
        r["crop_recut"] = dict(n=len(ids), cos_min=float(cos.min()), cos_mean=float(cos.mean()), max_abs_min_over_cells=float(dmax.max()),
                               raw_p98_median=float(np.median(raw98)), raw_p2_median=float(np.median(raw2)),
                               after_stretch_p98_median=float(np.median(S2[:, 1])), after_stretch_p2_median=float(np.median(S2[:, 0])),
                               after_stretch_median_gray=float(np.median(S2[:, 2])))
        H.log(f"{b}: crop cắt lại {len(ids)} ô: cos(mine,prod) min {cos.min():.6f}; max|Δ| {dmax.max():.2e}; "
              f"nền thô p98 {np.median(raw98):.0f} -> sau giãn {np.median(S2[:, 1]):.0f}; mực p2 {np.median(raw2):.0f} -> {np.median(S2[:, 0]):.0f}")
    worst = max(max(r["f_font"]["max_abs"], r["f_fd"]["max_abs"]) for r in res["books"].values())
    res["max_abs_all"] = float(worst)
    res["pass_1e3"] = bool(worst <= 1e-3)
    H.jdump(res, OUT / "hieuchuan.json")
    H.log(f"HIỆU CHUẨN: max|Δ| mọi bộ/tín hiệu = {worst:.2e} -> {'ĐẠT' if worst <= 1e-3 else 'KHÔNG ĐẠT'} (ngưỡng 1e-3)")
    pool.close(); pool.join()


def REPO_P01(b):
    import json
    p = H.REPO / "measure_out" / "_tn11" / f"p01_{b}.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    return dict(f_font=j["top1"]["f_font"]["tat_ca"], f_fd=j["top1"]["f_fd"]["tat_ca"])


if __name__ == "__main__":
    main()
