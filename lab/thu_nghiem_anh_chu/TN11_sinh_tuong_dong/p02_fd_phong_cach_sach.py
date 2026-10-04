"""TN11 p02 (03/10) — SINH ẢNH THEO NÉT CỦA CHÍNH CUỐN rồi so ảnh, trên ô CHỮ HIẾM của Borg (0 API, không sửa pipeline).

Giả thuyết: glyph FD hiện tại sinh MỘT lần với ảnh phong cách STT (dùng chung cho mọi bộ) nên lệch nét so với từng cuốn; sinh lại
glyph ứng viên với ảnh phong cách lấy từ ô neo TỰ ĐỘNG của chính cuốn (kim ∈ R, nhãn = kim; không nhãn người) sẽ giúp so ảnh
đúng hơn ở chữ chưa có mẫu thật trong sách (nơi dồn phần lớn lỗi của bộ chọn, TN11 p01).

Thiết kế (B34 mặc định; sự thật = chữ người, CHỈ để đo):
  ô mẫu  = ô có chữ đúng trong ứng viên, chữ đúng có < 2 ô neo cùng sách (khối khác) — "chữ hiếm"; lấy ngẫu nhiên --n ô (seed 0)
  rút gọn = top --k ứng viên theo f_vW (CNN kiểm ảnh↔chữ học sách Borg KIA + IHR — sạch với sách này) -> so lại trong danh sách
  phong cách = --styles ô neo điển hình (cos cao nhất tới nguyên mẫu chữ của nó, cỡ gần trung vị), nhị phân Otsu, nền trắng
  sinh   = FontDiffuser checkpoint dự án (font_diffusion/ckpt/PROD), cùng tham số kho FD (20 bước, guidance 2, nét mảnh dilate 2)
  đo     = Top-1 trong danh sách rút gọn theo: f_font, f_fd (kho FD hiện có), f_gen_s<j> (mỗi phong cách), f_gen (trung bình),
           kết hợp z(f_vW)+z(f_fd) vs z(f_vW)+z(f_gen); tham chiếu trần: f_hum (ảnh NGƯỜI sách kia)
Ra: measure_out/_tn11/p02_<bộ>/ (style_*.png, gen/<style>/U+XXXX.png, plan.pkl, ket_qua.json)

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p02_fd_phong_cach_sach.py --book B34 --stage plan|gen|eval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "font_diffusion"))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

EMB = T.OUT / "emb"
CAND = T.OUT / "cand"


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def anchors(b, D, meta):
    C = T.lex()
    Rm = {s: C.R_of(s) for s in set(D.syllable)}
    a = np.array((D.rule.str.startswith("s1_inter_s2_direct") & (D.label != "") & (D.label == D.ocr_char)).values, bool)
    return a & np.array([l in Rm[s] for l, s in zip(D.label, D.syllable)]) & (meta[:, 3] > 0)


def style_image(g):
    """crop xám -> ảnh phong cách: Otsu (mực đen / nền trắng), đặt vào ô vuông lề 10 %."""
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = bw.shape
    s = int(max(h, w) * 1.2)
    out = np.full((s, s), 255, np.uint8)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = bw
    return cv2.resize(out, (128, 128), interpolation=cv2.INTER_AREA)


def plan(a, out):
    b = a.book
    D = T.load_base(b)
    E = nrm(np.load(EMB / f"{b}_enc.npy").astype(np.float32))
    meta = np.load(EMB / f"{b}_crop_meta.npy")
    F = pd.read_pickle(CAND / f"{b}.pkl")
    hy = F.groupby("i").y.max()
    tru = F[F.y == 1].drop_duplicates("i").set_index("i")
    rare = tru.index[tru.n_self.fillna(0) < 2]
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(rare, size=min(a.n, len(rare)), replace=False))
    S = F[F.i.isin(pick)].copy()
    S["rk"] = S.f_vW.fillna(-9)
    S = S.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(a.k)
    cov = S.groupby("i").y.max()
    # phong cách: ô neo điển hình
    an = np.nonzero(anchors(b, D, meta))[0]
    lab = D.label.values[an]
    proto = {}
    for c in set(lab):
        m = an[lab == c]
        if len(m) >= 5:
            proto[c] = nrm(E[m].mean(0))
    hmed = np.median(meta[an, 1])
    cand = [(float(E[i] @ proto[D.label.values[i]]), i) for i in an if D.label.values[i] in proto
            and 0.8 * hmed <= meta[i, 1] <= 1.25 * hmed and 0.08 <= meta[i, 2] <= 0.35]
    cand.sort(reverse=True)
    seen, st = set(), []
    for _, i in cand:
        if D.label.values[i] in seen:
            continue
        seen.add(D.label.values[i]); st.append(int(i))
        if len(st) == a.styles:
            break
    from t02_embed import page_crops
    cfg = T.BOOKS[b]; prep = T.REPO / cfg["prep"]
    for j, i in enumerate(st):
        pg = D.page.values[i]
        allp = D[D.page == pg]
        recs = [(ii, c, T.bbox_of(bb)) for ii, c, bb in zip(allp.index, allp.column, allp.bbox)]
        res = {ii: gg for ii, gg, _ in page_crops((b, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", recs))}
        cv2.imwrite(str(out / f"style_{j}.png"), style_image(res[i]))
    chars = sorted(set(S.c))
    pd.to_pickle(dict(S=S, styles=st, style_chars=[D.label.values[i] for i in st], chars=chars), out / "plan.pkl")
    print(f"[p02] {b}: {len(rare)} ô chữ hiếm; lấy {len(pick)}; top-{a.k} theo f_vW chứa chữ đúng {cov.mean() * 100:.1f} %; "
          f"{len(chars)} chữ cần sinh × {len(st)} phong cách (chữ phong cách: {''.join(D.label.values[i] for i in st)})")


def gen(a, out):
    if a.fixed:        # đường chuẩn use_fst=False (fd_wrapper_fix.py); mặc định cũ = FST ngẫu nhiên (xem docstring fd_wrapper_fix)
        sys.path.insert(0, str(Path(__file__).parent))
        from fd_wrapper_fix import FixedFontDiffusionGenerator as FontDiffusionGenerator
    else:
        from core.ranking.fontdiffusion_gen import FontDiffusionGenerator
    sub = "gen_fix" if a.fixed else "gen"
    P = pd.read_pickle(out / "plan.pkl")
    ck = str(REPO / "font_diffusion/ckpt/PROD")
    torch.manual_seed(0)
    g = FontDiffusionGenerator(ckpt_dir=ck, phase1_ckpt_dir=ck, font_path=str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf"),
                               cache_dir=str(out / sub), batch_size=8)
    g._load_pipeline(); g.pipe.guidance_scale = 2.0
    for j in ([a.only_style] if a.only_style is not None else range(len(P["styles"]))):
        todo = [c for c in P["chars"] if not (out / sub / f"s{j}" / f"U+{ord(c):04X}.png").exists()]
        if a.max_chars:
            todo = todo[: a.max_chars]
        t0 = time.time()
        if todo:
            g.generate(todo, str(out / f"style_{j}.png"), style_name=f"s{j}")
        print(f"[p02] phong cách {j}: sinh {len(todo)} chữ [{time.time() - t0:.0f}s]", flush=True)


def eval_(a, out):
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    b = a.book
    P = pd.read_pickle(out / "plan.pkl"); S = P["S"].copy()
    E = nrm(np.load(EMB / f"{b}_enc.npy").astype(np.float32))
    Sc = SI.Scorers(Assets(), "mps" if torch.backends.mps.is_available() else "cpu", EMB, False, lambda m: None)
    enc = Sc.enc()
    k3 = np.ones((3, 3), np.uint8)
    G = {}
    done = [j for j in range(len(P["styles"]))
            if sum((out / a.gen_sub / f"s{j}" / f"U+{ord(c):04X}.png").exists() for c in P["chars"]) >= 0.7 * len(P["chars"])]
    for j in done:
        for thin in (0, 2):
            ims, cs = [], []
            for c in P["chars"]:
                f = out / a.gen_sub / f"s{j}" / f"U+{ord(c):04X}.png"
                if f.exists():
                    im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                    if thin:
                        im = cv2.dilate(im, k3, iterations=thin)
                    ims.append(im); cs.append(c)
            V = nrm(np.asarray(enc.embed(ims, norm=False), np.float32))
            G[(j, thin)] = dict(zip(cs, V))
    ci = S.i.to_numpy(); cc = S.c.to_numpy()
    for (j, thin), M in G.items():
        S[f"f_gen_s{j}_t{thin}"] = [float(E[i] @ M[c]) if c in M else np.nan for i, c in zip(ci, cc)]
    for thin in (0, 2):
        cols = [f"f_gen_s{j}_t{thin}" for j in done]
        S[f"f_gen_t{thin}"] = S[cols].mean(1)

    def z(col):
        x = S[col]; g = S.groupby("i")[col]
        return (x - g.transform("mean")) / g.transform("std").replace(0, 1)
    S["vW+fd"] = z("f_vW") + z("f_fd").fillna(0)
    for thin in (0, 2):
        S[f"vW+gen_t{thin}"] = z("f_vW") + z(f"f_gen_t{thin}").fillna(0)
    cov = S.groupby("i").y.max(); Sv = S[S.i.isin(cov[cov == 1].index)]

    def top1(col):
        x = Sv[["i", col, "y"]].copy(); x[col] = x[col].fillna(-9)
        return float(x.loc[x.groupby("i")[col].idxmax(), "y"].mean()) * 100
    cols = ["f_vW", "f_font", "f_fd"] + [c for c in S.columns if c.startswith("f_gen")] + ["vW+fd"] + \
           [c for c in S.columns if c.startswith("vW+gen")] + ["f_hum"]
    R = {c: round(top1(c), 1) for c in cols}
    # khoảng tin cậy bootstrap ô cho chênh f_gen_t2 − f_fd
    rng = np.random.default_rng(0)
    def per_cell(col):
        x = Sv[["i", col, "y"]].copy(); x[col] = x[col].fillna(-9)
        return x.loc[x.groupby("i")[col].idxmax()].set_index("i").y
    dd = {}
    for new, old in (("f_gen_t2", "f_fd"), ("f_gen_t0", "f_fd"), ("vW+gen_t2", "vW+fd"), ("vW+gen_t0", "vW+fd")):
        d = (per_cell(new) - per_cell(old)).to_numpy()
        bs = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(2000)]
        dd[f"{new} − {old}"] = [round(d.mean() * 100, 1), round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]
    res = dict(bo=b, o=int(cov.size), o_co_chu_dung_trong_top_k=int((cov == 1).sum()), k=a.k, phong_cach=[P["style_chars"][j] for j in done],
               top1_trong_top_k=R, chenh_lech_CI95=dd)
    (out / ("ket_qua.json" if a.gen_sub == "gen" else f"ket_qua_{a.gen_sub}.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="B34")
    ap.add_argument("--stage", choices=["plan", "gen", "eval"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--styles", type=int, default=2)
    ap.add_argument("--fixed", action="store_true", help="stage gen: dùng đường FontDiffuser chuẩn (use_fst=False), ghi vào gen_fix/")
    ap.add_argument("--max-chars", type=int, default=0, help="stage gen: chỉ sinh N chữ đầu (thử)")
    ap.add_argument("--only-style", type=int, default=None, help="stage gen: chỉ sinh phong cách j (mặc định: tất cả)")
    ap.add_argument("--gen-sub", default="gen", help="thư mục ảnh sinh trong p02_<bộ>/ (gen = one-shot; gen_ft = p04 tinh chỉnh)")
    a = ap.parse_args(argv)
    out = REPO / "measure_out" / "_tn11" / f"p02_{a.book}"
    out.mkdir(parents=True, exist_ok=True)
    {"plan": plan, "gen": gen, "eval": eval_}[a.stage](a, out)


if __name__ == "__main__":
    main()
