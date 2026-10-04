"""TN11 p03 (03/10) — DIFFUSION LÀM BỘ PHÂN LOẠI (phân tích bằng tổng hợp) cho chọn chữ, 0 API, không sửa pipeline.

Ý tưởng (Li et al., ICCV 2023 "Your Diffusion Model is Secretly a Zero-Shot Classifier"; Clark & Jaini, NeurIPS 2023): với ảnh
crop x0 và mỗi ứng viên c, thêm nhiễu ε ở bước t rồi để FontDiffuser (điều kiện: glyph nội dung của c + ảnh phong cách của
CHÍNH cuốn) dự đoán lại ε. Ứng viên đúng giải thích ảnh tốt hơn -> sai số ||ε − ε̂|| nhỏ hơn. Cùng (t, ε) cho mọi ứng viên của
một ô (giảm phương sai). Không sinh ảnh nào; điểm f_dc = −sai số trung bình.

Dùng đúng 200 ô chữ hiếm + danh sách top-k của p02 (plan.pkl) để so trực tiếp với f_fd / f_gen.
Glyph nội dung: phông NomNaTong (như FontDiffuser); chữ không có trong NomNaTong -> render Plangothic (Glyphs.font) — đánh dấu.
Ra: measure_out/_tn11/p02_<bộ>/dc_scores.pkl + ket_qua_dc.json

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p03_diffusion_phan_loai.py --book B34 [--max-cells 200]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "font_diffusion"))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
from p02_fd_phong_cach_sach import style_image  # noqa: E402

T_SET = (100, 250, 400, 550, 700, 850)
N_NOISE = 2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="B34")
    ap.add_argument("--max-cells", type=int, default=200)
    ap.add_argument("--style", type=int, default=0)
    a = ap.parse_args(argv)
    out = REPO / "measure_out" / "_tn11" / f"p02_{a.book}"
    P = pd.read_pickle(out / "plan.pkl"); S = P["S"]
    cells = sorted(S.i.unique())[: a.max_cells]
    S = S[S.i.isin(cells)]
    D = T.load_base(a.book)
    from t02_embed import page_crops
    cfg = T.BOOKS[a.book]; prep = T.REPO / cfg["prep"]
    crops = {}
    for pg in sorted(set(D.page.values[cells])):
        allp = D[D.page == pg]
        recs = [(ii, c, T.bbox_of(bb)) for ii, c, bb in zip(allp.index, allp.column, allp.bbox)]
        want = set(cells) & set(allp.index)
        for ii, gg, _ in page_crops((a.book, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", recs)):
            if ii in want and gg is not None:
                crops[ii] = gg
    from core.ranking.fontdiffusion_gen import FontDiffusionGenerator
    from inference.sample_optimized import get_content_transform, get_style_transform
    from src.tools.utils import ttf2im
    from pipeline.gold_exact.signals_img import Glyphs
    ck = str(REPO / "font_diffusion/ckpt/PROD")
    g = FontDiffusionGenerator(ckpt_dir=ck, phase1_ckpt_dir=ck, font_path=str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf"),
                               cache_dir=str(out / "_dc_tmp"), batch_size=8)
    g._load_pipeline()
    model = g.pipe.model.eval(); dev = g.device; args = g.args
    sched = g.pipe.ddpm_train_scheduler if hasattr(g.pipe, "ddpm_train_scheduler") else None
    if sched is None:
        from src.builders.build import build_ddpm_scheduler
        sched = build_ddpm_scheduler(args)
    acp = sched.alphas_cumprod.to(torch.float32)
    ct = get_content_transform(args.content_image_size); stt = get_style_transform(args.style_image_size)
    font = g.font_manager.get_font(g.font_manager.get_font_names()[0])
    GL = Glyphs()
    style = stt(Image.open(out / f"style_{a.style}.png").convert("RGB"))
    cache = {}

    def content(c):
        if c not in cache:
            im, src = None, "nomna"
            try:
                im = ttf2im(font=font, char=c)
            except Exception:
                im = None
            if im is None or np.asarray(im).min() > 200:
                r = GL.font(c); src = "plan"
                im = Image.fromarray(cv2.resize(r, (128, 128))).convert("RGB") if r is not None else None
            cache[c] = (ct(im) if im is not None else None, src)
        return cache[c]
    rows = []
    t0 = time.time()
    gen = torch.Generator().manual_seed(0)
    for n, i in enumerate(cells):
        if i not in crops:
            continue
        x0 = stt(Image.fromarray(style_image(crops[i])).convert("RGB"))       # 96×96, [-1, 1]
        cs = S[S.i == i].c.tolist()
        cont = [(c, *content(c)) for c in cs]
        cont = [(c, t, s) for c, t, s in cont if t is not None]
        ts = torch.tensor([t for t in T_SET for _ in range(N_NOISE)])
        eps = torch.randn((len(ts), 3, 96, 96), generator=gen)
        ab = acp[ts].view(-1, 1, 1, 1)
        xt = ab.sqrt() * x0[None] + (1 - ab).sqrt() * eps
        err = {}
        with torch.no_grad():
            for c, tc, src in cont:
                B = len(ts)
                pred = model(xt.to(dev), ts.to(dev), [tc[None].repeat(B, 1, 1, 1).to(dev), style[None].repeat(B, 1, 1, 1).to(dev)],
                             content_encoder_downsample_size=args.content_encoder_downsample_size, version="V3")
                e = ((pred.float().cpu() - eps) ** 2).mean(dim=(1, 2, 3)).numpy()
                err[c] = e
        for c, tc, src in cont:
            e = err[c]
            rows.append(dict(i=i, c=c, src=src, f_dc=-float(e.mean()), f_dc_lo=-float(e[: len(e) // 2].mean()),
                             f_dc_hi=-float(e[len(e) // 2:].mean())))
        if n % 20 == 0:
            print(f"[p03] {n}/{len(cells)} ô [{time.time() - t0:.0f}s]", flush=True)
    R = pd.DataFrame(rows)
    R.to_pickle(out / "dc_scores.pkl")
    S = S.merge(R, on=["i", "c"], how="left")
    cov = S.groupby("i").y.max(); Sv = S[S.i.isin(cov[cov == 1].index) & S.i.isin(R.i.unique())]

    def z(col):
        x = Sv[col]; gg = Sv.groupby("i")[col]
        return (x - gg.transform("mean")) / gg.transform("std").replace(0, 1)
    Sv = Sv.assign(**{"vW+dc": z("f_vW") + z("f_dc").fillna(0), "vW+fd": z("f_vW") + z("f_fd").fillna(0)})

    def per_cell(col):
        x = Sv[["i", col, "y"]].copy(); x[col] = x[col].fillna(-9e9)
        return x.loc[x.groupby("i")[col].idxmax()].set_index("i").y
    cols = ["f_vW", "f_font", "f_fd", "f_dc", "f_dc_lo", "f_dc_hi", "vW+fd", "vW+dc", "f_hum"]
    res = dict(bo=a.book, o=int(Sv.i.nunique()), t=list(T_SET), n_noise=N_NOISE, phong_cach=P["style_chars"][a.style],
               glyph_plangothic=int((R.src == "plan").sum()),
               top1_trong_top_k={c: round(float(per_cell(c).mean()) * 100, 1) for c in cols})
    rng = np.random.default_rng(0)
    res["chenh_lech_CI95"] = {}
    for new, old in (("f_dc", "f_fd"), ("vW+dc", "vW+fd"), ("vW+dc", "f_vW")):
        d = (per_cell(new) - per_cell(old)).to_numpy()
        bs = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(2000)]
        res["chenh_lech_CI95"][f"{new} − {old}"] = [round(d.mean() * 100, 1), round(float(np.percentile(bs, 2.5)), 1),
                                                     round(float(np.percentile(bs, 97.5)), 1)]
    (out / "ket_qua_dc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
