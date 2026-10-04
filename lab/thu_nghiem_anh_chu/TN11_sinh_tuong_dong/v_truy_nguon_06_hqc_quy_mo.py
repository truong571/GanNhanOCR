"""v_truy_nguon_06 (04/10) — HQC1/2/3 của p05 đo ở QUY MÔ LỚN HƠN n ~ 20 (CPU, 0 API, chỉ ĐỌC, không sửa p0x).

p05 (phiên khác) báo HQC1 "nền trắng" / HQC2 "mẫu giấy sách" / HQC3 "chuẩn hoá nhị phân" trên n = 16-20 ô/bộ, Top-1 trong top-3 ứng viên theo f_vW.
Script này làm LẠI đúng các phép biến đổi đó (sao nguyên văn render_font_raw / extract_book_paper_model / blend_on_paper / binarize_canonical của
p05_full_corpus_he_quy_chieu.py:58-163) nhưng trên ~16 trang x <= 50 ô = <= 800 ô/bộ có chữ người (B18, B34, L16, TK), cả hai giao thức:
  top3   top-3 ứng viên theo f_vW, chỉ giữ ô có chữ đúng trong top-3 (như p05; hòa -> ứng viên đứng trước theo f_vW)
  tatca  mọi ứng viên R(âm) ∪ {kim} (~26/ô), ô có chữ đúng trong ứng viên (như p01/TN8)
Hai cách dựng glyph:
  p05    NomNaTong của font_diffusion/ (v5.12, 30.273 chữ), KHÔNG phông dự phòng -> chữ thiếu glyph = ẢNH TRẮNG (đúng như p05)
  pipe   glyph của pipeline (gold_exact Glyphs: fonts/NomNaTong v5.18 + Plangothic P1/P2, 112 px) = đúng f_font; HQC1-pipe == f_font (kiểm)
Tín hiệu: f_font (cand pkl), HQC1p/2p/3p, HQC2f/3f (biến đổi HQC2/HQC3 trên glyph pipeline, công bằng về phông). Phía crop: HQC1/2 dùng nhúng đã lưu
(measure_out/_tn8/emb/<bộ>_enc.npy); HQC3 cắt lại crop (page_crops với MỌI ô của trang để carve hàng xóm như t02) -> binarize_canonical -> nhúng.
CI 95 % bootstrap cụm trang; chênh ghép cặp bootstrap cụm trang. CPU (torch.set_num_threads=3), PYTORCH_ENABLE_MPS_FALLBACK=0.
Ra: measure_out/_tn11/verify/truy_nguon/hqc_quy_mo.json

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_truy_nguon_06_hqc_quy_mo.py [bộ ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2
import numpy as np
import pandas as pd
import torch
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

torch.set_num_threads(int(os.environ.get("VT_THREADS", "3")))

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "truy_nguon"
OUT.mkdir(parents=True, exist_ok=True)
EMB = T.OUT / "emb"
CAND = T.OUT / "cand"
FONT_P05 = str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf")
N_PAGES, PER_PAGE, SEED, K3 = int(os.environ.get("VT_PAGES", "16")), int(os.environ.get("VT_PER_PAGE", "50")), 20261004, 3
BATCH = 256


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


# ---- sao nguyên văn từ p05_full_corpus_he_quy_chieu.py (dòng 58-163), chỉ đổi FONT_PATH -> FONT_P05 -------------------------------
def render_font_raw(char: str, size: int = 128) -> np.ndarray:
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_P05, int(size * 0.75))
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - w) // 2 - bbox[0]
        y = (size - h) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
    except Exception:
        draw.rectangle([(10, 10), (size - 10, size - 10)], outline=128, width=2)
    return np.array(img)


def extract_book_paper_model(book: str) -> dict:
    cfg = T.BOOKS[book]
    prep = T.REPO / cfg["prep"]
    pages_dir = prep / "pages"
    sample_pages = sorted(pages_dir.glob("*.png"))
    sample_page = sample_pages[min(10, len(sample_pages) - 1)]
    page_img = cv2.imread(str(sample_page), cv2.IMREAD_GRAYSCALE)
    T_val, _ = cv2.threshold(page_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper_mask = page_img >= T_val
    paper_pixels = page_img[paper_mask]
    ink_pixels = page_img[~paper_mask]
    bg_median = float(np.median(paper_pixels)) if len(paper_pixels) else 240.0
    ink_median = float(np.median(ink_pixels)) if len(ink_pixels) else 60.0
    bg_std = float(np.std(paper_pixels)) if len(paper_pixels) else 10.0
    h, w = page_img.shape
    best_patch = None
    min_ink = 999999
    step = 32
    for y in range(20, h - 148, step):
        for x in range(20, w - 148, step):
            sub = page_img[y:y + 128, x:x + 128]
            n_ink = np.sum(sub < T_val)
            if n_ink < min_ink:
                min_ink = n_ink
                best_patch = sub.copy()
            if min_ink == 0:
                break
        if min_ink == 0:
            break
    if best_patch is None or min_ink > 200:
        rng = np.random.default_rng(42)
        best_patch = np.clip(rng.normal(bg_median, max(bg_std, 4.0), (128, 128)), 0, 255).astype(np.uint8)
    return {"book": book, "sample_page": sample_page.name, "bg_median": round(bg_median, 1), "ink_median": round(ink_median, 1),
            "bg_std": round(bg_std, 1), "contrast": round(bg_median - ink_median, 1), "paper_patch": best_patch}


def blend_on_paper(raw_glyph: np.ndarray, paper_model: dict) -> np.ndarray:
    patch = paper_model["paper_patch"].astype(np.float32)
    ink_target = paper_model["ink_median"]
    alpha = (255.0 - raw_glyph.astype(np.float32)) / 255.0
    blended = (1.0 - alpha) * patch + alpha * ink_target
    return np.clip(blended, 0, 255).astype(np.uint8)


def binarize_canonical(img: np.ndarray) -> np.ndarray:
    _, bin_inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)
    clean_inv = np.zeros_like(bin_inv)
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] >= 6:
            clean_inv[lab == j] = 255
    ys, xs = np.nonzero(clean_inv)
    if len(ys) == 0:
        return np.full((128, 128), 255, dtype=np.uint8)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped_char = clean_inv[y0:y1, x0:x1]
    h, w = cropped_char.shape
    max_dim = max(h, w)
    target_dim = int(128 * 0.8)
    scale = target_dim / max(max_dim, 1)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    scaled = cv2.resize(cropped_char, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((128, 128), dtype=np.uint8)
    sy = (128 - new_h) // 2
    sx = (128 - new_w) // 2
    canvas[sy:sy + new_h, sx:sx + new_w] = scaled
    return 255 - canvas
# ---------------------------------------------------------------------------------------------------------------------------------


def embed_many(enc, imgs):
    out = []
    for s in range(0, len(imgs), BATCH):
        out.append(np.asarray(enc.embed(imgs[s:s + BATCH], norm=False), np.float32))
    return nrm(np.concatenate(out)) if out else np.zeros((0, 512), np.float32)


def boot_diff(a, b, pages, B=T.BOOT_B, seed=T.BOOT_SEED):
    d = np.asarray(a, float) - np.asarray(b, float)
    up, inv = np.unique(pages, return_inverse=True)
    k = np.bincount(inv, weights=d, minlength=len(up)); n = np.bincount(inv, minlength=len(up)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(up), size=(B, len(up)))
    r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return [round(100 * float(d.mean()), 2), round(100 * float(np.quantile(r, 0.025)), 2), round(100 * float(np.quantile(r, 0.975)), 2)]


def rate(ok, pages):
    m, lo, hi = T.boot_ci_pages(np.asarray(ok, float), pages)
    return dict(p=round(100 * m, 2), lo=round(100 * lo, 2), hi=round(100 * hi, 2), n=int(len(ok)))


def top1_vec(S, col, cells):
    """S đã sắp (i, rk giảm dần); Top-1 = ứng viên đầu tiên đạt max (hòa -> đứng trước theo f_vW) — y của nó, theo thứ tự `cells`."""
    x = S[["i", col, "y"]].copy(); x[col] = x[col].fillna(-9)
    return x.loc[x.groupby("i", sort=False)[col].idxmax()].set_index("i").y.reindex(cells).values.astype(float)


def run(b, enc, gl, nom_cmap):
    t0 = time.time()
    cfg = T.BOOKS[b]; prep = T.REPO / cfg["prep"]
    D = T.load_base(b)
    E = nrm(np.load(EMB / f"{b}_enc.npy").astype(np.float32))
    F = pd.read_pickle(CAND / f"{b}.pkl")[["i", "c", "y", "f_font", "f_vW"]]
    hy = F.groupby("i").y.max(); truth_cells = hy[hy == 1].index.values
    pg_all = D.page.values
    rng = np.random.default_rng(SEED)
    pages = sorted(set(pg_all[truth_cells]))
    pick = rng.choice(pages, size=min(N_PAGES, len(pages)), replace=False)
    cells = []
    for p in pick:
        c = truth_cells[pg_all[truth_cells] == p]
        cells += list(rng.choice(c, size=min(PER_PAGE, len(c)), replace=False))
    cells = np.sort(np.array(cells))
    S = F[F.i.isin(cells)].copy()
    S["rk"] = S.f_vW.fillna(-9)
    S = S.sort_values(["i", "rk"], ascending=[True, False], kind="stable").reset_index(drop=True)
    chars = sorted(set(S.c))
    paper = extract_book_paper_model(b)
    # ---- glyph p05 (NomNaTong v5.12 only; thiếu -> ảnh trắng) và glyph pipeline (có phông dự phòng) ----
    present = [c for c in chars if len(c) == 1 and ord(c) in nom_cmap]
    blank = np.full((128, 128), 255, np.uint8)
    raw_p = {c: render_font_raw(c, 128) for c in present}
    raw_pipe = {}
    for c in chars:
        g = gl.font(c) if len(c) == 1 else None
        if g is not None:
            raw_pipe[c] = cv2.resize(g, (128, 128), interpolation=cv2.INTER_LINEAR)
    V = {}
    for name, fn in (("1", lambda x: x), ("2", lambda x: blend_on_paper(x, paper)), ("3", binarize_canonical)):
        ims = [fn(raw_p[c]) for c in present] + [fn(blank)]
        em = embed_many(enc, ims)
        d = dict(zip(present, em[:-1])); bl = em[-1]
        V["p" + name] = (d, bl)
    for name, fn in (("2", lambda x: blend_on_paper(x, paper)), ("3", binarize_canonical)):
        cs = sorted(raw_pipe)
        em = embed_many(enc, [fn(raw_pipe[c]) for c in cs])
        V["f" + name] = (dict(zip(cs, em)), None)
    # kiểm: glyph pipeline (HQC1-pipe) == f_font của cand pkl
    chk = S.drop_duplicates("c").head(300)
    gl_im = [(c, gl.font(c)) for c in chk.c if len(c) == 1 and gl.font(c) is not None]
    em = embed_many(enc, [g for _, g in gl_im]); dd = dict(zip([c for c, _ in gl_im], em))
    sub = S[S.c.isin(dd)].head(3000)
    diff = np.abs(np.array([float(E[i] @ dd[c]) for i, c in zip(sub.i, sub.c)]) - sub.f_font.values.astype(np.float64))
    kiem = dict(n=int(len(sub)), max_abs_diff=round(float(np.nanmax(diff)), 4), mean_abs_diff=round(float(np.nanmean(diff)), 5))
    # ---- crop-side HQC3 ----
    crop_raw = {}
    cellset = set(int(x) for x in cells)
    for p in pick:
        allp = D[D.page == p]
        recs = [(ii, c, T.bbox_of(bb)) for ii, c, bb in zip(allp.index, allp.column, allp.bbox)]
        from t02_embed import page_crops
        for ii, g, _ in page_crops((b, prep / "pages" / f"{p}.png", prep, cfg["crop"] == "original", recs)):
            if ii in cellset and g is not None:
                crop_raw[ii] = g
    ci = [int(i) for i in cells if int(i) in crop_raw]
    Ecanon = dict(zip(ci, embed_many(enc, [binarize_canonical(cv2.resize(crop_raw[i], (128, 128))) for i in ci])))
    # ---- điểm ----
    ii = S.i.values; cc = S.c.values
    def sc(vname, crop_side):
        d, bl = V[vname]
        out = np.full(len(S), np.nan)
        for k, (i, c) in enumerate(zip(ii, cc)):
            v = d.get(c)
            if v is None:
                v = bl                               # p05: chữ thiếu glyph -> embedding của ảnh trắng (hoặc None -> NaN)
            if v is None:
                continue
            e = crop_side(i)
            if e is None:
                continue
            out[k] = float(e @ v)
        return out
    S["HQC1p"] = sc("p1", lambda i: E[i]); S["HQC2p"] = sc("p2", lambda i: E[i]); S["HQC3p"] = sc("p3", lambda i: Ecanon.get(i))
    S["HQC2f"] = sc("f2", lambda i: E[i]); S["HQC3f"] = sc("f3", lambda i: Ecanon.get(i))
    sig = ["f_font", "HQC1p", "HQC2p", "HQC3p", "HQC2f", "HQC3f"]
    # chỉ giữ ô có đủ crop HQC3 để mọi tín hiệu cùng tập ô
    S = S[S.i.isin(ci)].copy()
    res = dict(bo=b, so_trang=int(len(pick)), so_o=int(len(ci)), paper=dict((k, v) for k, v in paper.items() if k != "paper_patch"),
               kiem_HQC1_pipe_bang_f_font=kiem, so_chu=int(len(chars)), chu_thieu_glyph_p05=int(len(chars) - len(present)),
               chu_khong_co_glyph_pipe=int(len(chars) - len(raw_pipe)))
    for proto in ("tatca", "top3"):
        if proto == "top3":
            S3 = S.groupby("i", sort=False).head(K3)
            keep = S3.groupby("i").y.max(); keep = keep[keep == 1].index.values
            Sx = S3[S3.i.isin(keep)]
            cells_x = np.array(sorted(keep))
        else:
            Sx = S; cells_x = np.array(sorted(S.i.unique()))
        pages_x = pg_all[cells_x]
        okm = {m: top1_vec(Sx, m, cells_x) for m in sig}
        r = dict(so_o=int(len(cells_x)), top1={m: rate(v, pages_x) for m, v in okm.items()})
        r["chenh_ghep_cap_diem"] = {k: boot_diff(okm[a], okm[c], pages_x) for k, (a, c) in {
            "HQC2p_tru_HQC1p": ("HQC2p", "HQC1p"), "HQC3p_tru_HQC1p": ("HQC3p", "HQC1p"),
            "HQC1p_tru_f_font": ("HQC1p", "f_font"), "HQC2f_tru_f_font": ("HQC2f", "f_font"), "HQC3f_tru_f_font": ("HQC3f", "f_font"),
            "HQC3f_tru_HQC2f": ("HQC3f", "HQC2f")}.items()}
        res[proto] = r
    res["giay"] = round(time.time() - t0, 1)
    print(f"\n== {b}: {res['so_o']} ô / {res['so_trang']} trang; chữ {res['so_chu']} (thiếu glyph p05 {res['chu_thieu_glyph_p05']}, không glyph pipeline {res['chu_khong_co_glyph_pipe']}); "
          f"paper {res['paper']['bg_median']}/{res['paper']['ink_median']}; kiểm HQC1-pipe vs f_font {kiem}; {res['giay']}s", flush=True)
    for proto in ("top3", "tatca"):
        r = res[proto]
        print(f"  [{proto}] n={r['so_o']}: " + " | ".join(f"{m} {v['p']:.1f} [{v['lo']:.1f},{v['hi']:.1f}]" for m, v in r["top1"].items()), flush=True)
        print(f"     chênh: " + " | ".join(f"{k} {v[0]:+.1f} [{v[1]:+.1f},{v[2]:+.1f}]" for k, v in r["chenh_ghep_cap_diem"].items()), flush=True)
    return res


def main():
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    S_ = SI.Scorers(Assets(), "cpu", EMB, False, lambda m: None)
    enc = S_.enc()
    gl = S_.glyphs()
    nom_cmap = set(TTFont(FONT_P05).getBestCmap())
    books = sys.argv[1:] or ["L16", "TK", "B18", "B34"]
    path = OUT / "hqc_quy_mo.json"
    allres = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for b in books:
        allres[b] = run(b, enc, gl, nom_cmap)
        path.write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
