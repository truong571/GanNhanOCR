"""v_thu_hqc_08_kiem_nhanh.py — kiểm nhanh (0 mô hình): ảnh glyph HQC2 dựng có đúng thống kê giấy/mực của sách không, và chuẩn hoá giãn p2/p98 của
encoder (Enc.prep, norm=True) làm gì với crop thô và với glyph nền giấy; mật độ mực glyph (font/FD, khung 112/96 và sau khi thu về độ phân giải crop)
so với crop thật (crop_meta.ink). Ra: measure_out/_tn11/verify/thu_hqc/kiem_nhanh.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v_thu_hqc_lib as H  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

T = H.T


def stretch(g):
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    if hi - lo > 10:
        g = np.clip((g.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
    return g


def main():
    from pipeline.gold_exact import signals_img as SI
    gl = SI.Glyphs()
    rng = np.random.default_rng(H.SEED + 61)
    chars = [c for c in (chr(x) for x in range(0x4E00, 0x4E00 + 6000)) if gl.font(c) is not None]
    chars = [chars[i] for i in rng.choice(len(chars), 300, replace=False)]
    fdch = sorted(gl.fd)
    out = {}
    for b in H.BOOKS:
        pm = H.paper_model(b)
        Pf = pm["P_font"]
        meta = np.load(T.OUT / "emb" / f"{b}_crop_meta.npy")
        ok = meta[:, 3] > 0
        r = dict(crop_ink_frac_median=float(np.median(meta[ok, 2])), paper=pm["paper_level_p90_median"], ink=pm["ink"],
                 patch_mean=float(pm["patches"][pm["patch_ids_a"]].mean()), patch_std=float(pm["patches"][pm["patch_ids_a"]].std()))
        for name, P, d, s in (("canvas_d0_s0", None, 0, 0.0), ("native_d0_s0", Pf, 0, 0.0), ("native_d1_s0", Pf, 1, 0.0), ("native_d1_s1.5", Pf, 1, 1.5)):
            inkf, bgm, inkm, p2, p98, p2s, p98s, inkf_st = [], [], [], [], [], [], [], []
            for ch in chars:
                g = gl.font(ch)
                im = H.hqc2_img(g, pm["patches"][pm["patch_ids_a"][0]], pm["ink"], P, d, s)
                a = (255.0 - H.thick(g, d)).astype(np.float32) / 255.0
                p2.append(np.percentile(im, 2)); p98.append(np.percentile(im, 98))
                st = stretch(im)
                inkf.append(float((im < (pm["ink"] + pm["paper_level_p90_median"]) / 2).mean()))
                inkf_st.append(float((st < 128).mean()))
            r[name] = dict(raw_p2=float(np.median(p2)), raw_p98=float(np.median(p98)), ink_frac_mid_threshold=float(np.median(inkf)),
                           ink_frac_after_stretch_lt128=float(np.median(inkf_st)))
        # glyph trắng (sản xuất): mật độ mực
        wf = [float((gl.font(c) < 128).mean()) for c in chars]
        wd = [float((gl.fdimg(c) < 128).mean()) for c in fdch]
        r["glyph_white_font_ink_frac_canvas112"] = float(np.median(wf)); r["glyph_white_fd_ink_frac_canvas96"] = float(np.median(wd))
        out[b] = r
        H.log(f"{b}: crop ink {r['crop_ink_frac_median']:.3f} | glyph trắng font {r['glyph_white_font_ink_frac_canvas112']:.3f} fd {r['glyph_white_fd_ink_frac_canvas96']:.3f} | "
              + " | ".join(f"{k}: p2 {v['raw_p2']:.0f} p98 {v['raw_p98']:.0f} inkfrac {v['ink_frac_after_stretch_lt128']:.3f}" for k, v in r.items() if isinstance(v, dict)))
    H.jdump(out, H.OUT / "kiem_nhanh.json")


if __name__ == "__main__":
    main()
