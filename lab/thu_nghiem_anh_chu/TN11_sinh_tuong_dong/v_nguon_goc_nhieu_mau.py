"""TN11 v_nguon_goc_nhieu_mau (03/10) — KIỂM ĐỊNH ĐỘC LẬP: nhiễu lấy mẫu của thí nghiệm HQC1/2/3 (p05).

Mục đích: p05_full_corpus_he_quy_chieu.py chọn 20 ô/bộ bằng `np.random.default_rng(hash(book) % 100000)`; `hash()` của chuỗi
Python bị ngẫu nhiên hoá theo từng tiến trình nên bảng KHÔNG tái lập. Script này SAO CHÉP logic `evaluate_single_book` của p05
(KHÔNG import để chạy, KHÔNG sửa tệp gốc) nhưng thay hạt giống bằng `default_rng(seed)`, seed cố định 0..9 (mặc định), và
đo Top-1 HQC1/HQC2/HQC3 theo từng hạt giống để so dao động do lấy mẫu với mức chênh giữa HQC mà báo cáo dùng để kết luận.

Giữ nguyên so với p05 (sao chép nguyên văn hàm trợ giúp): n_sample=20, k_cand=3 (top-3 theo f_vW), sự thật = nhãn người (y==1)
với B18/B34/L16/TK, ảnh ứng viên = render phông NomNaTong (HQC1 thô / HQC2 blend lên mẫu giấy / HQC3 nhị phân hoá),
vector crop HQC1/HQC2 = nhúng sản xuất E_full (measure_out/_tn8/emb/<bộ>_enc.npy), vector crop HQC3 = cắt lại + nhị phân hoá + nhúng lại.
Khác p05 (có chủ đích): (1) hạt giống cố định; (2) thiết bị CPU (không MPS); (3) cache nhúng glyph theo chữ giữa các hạt giống
(glyph chỉ phụ thuộc chữ + hồ sơ giấy của sách, không phụ thuộc mẫu) ; (4) thêm chẩn đoán (ô đa sự thật, hoà điểm, chữ không có
trong phông, crop cắt lại có khác crop sản xuất không, f_vW@1 trong cùng top-3...). 0 API.

Ra: measure_out/_tn11/verify/nguon_goc/nhieu_mau_<tag>.json (+ nhieu_mau_<tag>_tom_tat.json)

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_nguon_goc_nhieu_mau.py \
        --books B34 B18 L16 TK --seeds 0-9 --tag s0_9
    ... --selfcheck   (đối chiếu hàm sao chép với hàm gốc của p05 qua monkeypatch hash + OUT_DIR tạm; không ghi vào thư mục p05)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT_DIR = REPO / "measure_out" / "_tn11" / "verify" / "nguon_goc"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR = T.OUT / "emb"
CAND_DIR = T.OUT / "cand"
FONT_PATH = str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf")
P05_PATH = Path(__file__).with_name("p05_full_corpus_he_quy_chieu.py")


# ------------------------------------------------------------------------------------------------------------------
# Hàm trợ giúp: SAO CHÉP NGUYÊN VĂN từ p05_full_corpus_he_quy_chieu.py (dòng 54-163)
# ------------------------------------------------------------------------------------------------------------------
def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def render_font_raw(char: str, size: int = 128) -> np.ndarray:
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_PATH, int(size * 0.75))
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
    if not sample_pages:
        raise FileNotFoundError(f"Không tìm thấy ảnh trang trong {pages_dir}")

    sample_page = sample_pages[min(10, len(sample_pages) - 1)]
    page_img = cv2.imread(str(sample_page), cv2.IMREAD_GRAYSCALE)
    if page_img is None:
        raise ValueError(f"Không đọc được ảnh {sample_page}")

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

    return {
        "book": book,
        "sample_page": sample_page.name,
        "bg_median": round(bg_median, 1),
        "ink_median": round(ink_median, 1),
        "bg_std": round(bg_std, 1),
        "contrast": round(bg_median - ink_median, 1),
        "paper_patch": best_patch,
    }


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


# ------------------------------------------------------------------------------------------------------------------
# Thống kê
# ------------------------------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, ctr - hw), min(1.0, ctr + hw))


def exact_mcnemar(b: int, c: int) -> float:
    """p hai phía chính xác (nhị thức) cho cặp bất đồng b = (A đúng, B sai), c = (A sai, B đúng)."""
    from scipy.stats import binomtest
    n = b + c
    return 1.0 if n == 0 else float(binomtest(min(b, c), n, 0.5).pvalue)


def sd(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


# ------------------------------------------------------------------------------------------------------------------
# Một sách: nạp một lần, chạy nhiều hạt giống
# ------------------------------------------------------------------------------------------------------------------
class BookRun:
    def __init__(self, book: str, encoder):
        from fontTools.ttLib import TTFont
        self.book, self.enc = book, encoder
        self.cfg = T.BOOKS[book]
        self.prep = T.REPO / self.cfg["prep"]
        self.paper = extract_book_paper_model(book)
        self.D = T.load_base(book)
        self.E = nrm(np.load(EMB_DIR / f"{book}_enc.npy").astype(np.float32))
        F = pd.read_pickle(CAND_DIR / f"{book}.pkl")
        # --- p05 dòng 176-188: xác định sự thật / ô đích
        if "y" in F.columns and F.y.sum() > 0:
            tru = F[F.y == 1].drop_duplicates("i").set_index("i")
            self.target_cells = list(tru.index)
            self.is_human = True
        else:
            lbl = self.D.label.values
            tgt = lbl[F.i.values]
            valid_mask = (F.c.values == tgt) & (tgt != "")
            self.target_cells = sorted(set(F.i.values[valid_mask]))
            F["y"] = np.where(valid_mask, 1, 0)
            self.is_human = False
        self.F = F
        self.cmap = set(TTFont(FONT_PATH).getBestCmap())
        self.glyph_emb = {"1": {}, "2": {}, "3": {}, "2n": {}}  # cache theo chữ (độc lập mẫu); 2n = bổ sung: HQC2 nhưng nhúng norm=True như crop sản xuất
        self.raw_font = {}

    def _glyph_embs(self, chars):
        need = [c for c in chars if c not in self.glyph_emb["1"]]
        if need:
            raw = {c: render_font_raw(c, 128) for c in need}
            self.raw_font.update(raw)
            imgs1 = [raw[c] for c in need]
            imgs2 = [blend_on_paper(raw[c], self.paper) for c in need]
            imgs3 = [binarize_canonical(raw[c]) for c in need]
            for h, imgs, nm in (("1", imgs1, False), ("2", imgs2, False), ("3", imgs3, False), ("2n", imgs2, True)):
                V = nrm(np.asarray(self.enc.embed(imgs, norm=nm), np.float32))
                self.glyph_emb[h].update(dict(zip(need, V)))

    def run_seed(self, seed: int, n_sample: int = 20, k_cand: int = 3, diag_crop: bool = True) -> dict:
        from t02_embed import page_crops
        t0 = time.time()
        D, E, F = self.D, self.E, self.F
        rng = np.random.default_rng(seed)
        selected_cells = np.sort(rng.choice(self.target_cells, size=min(n_sample, len(self.target_cells)), replace=False))

        S = F[F.i.isin(selected_cells)].copy()
        S["rk"] = S.f_vW.fillna(-9) if "f_vW" in S.columns else S.f_font.fillna(-9)
        S = S.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(k_cand)

        has_truth = S.groupby("i").y.max()
        valid_cells = sorted(has_truth[has_truth == 1].index)
        S = S[S.i.isin(valid_cells)].copy()

        # --- crop thật: như p05 (recs CHỈ gồm ô hợp lệ của mẫu -> ngữ cảnh láng giềng khác sản xuất)
        cfg, prep = self.cfg, self.prep
        vset = set(valid_cells)
        crop_raw_map, crop_full_ctx = {}, {}
        pages_needed = set(D.loc[valid_cells, "page"])
        for pg in pages_needed:
            allp = D[D.page == pg]
            recs = [(ii, c, T.bbox_of(bb)) for ii, c, bb in zip(allp.index, allp.column, allp.bbox) if ii in vset]
            if recs:
                res = {ii: gg for ii, gg, _ in page_crops((self.book, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", recs))}
                crop_raw_map.update(res)
            if diag_crop:  # đối chứng: recs = MỌI ô cùng trang (đúng cách t02_embed dựng E_full)
                recs_all = [(ii, c, T.bbox_of(bb)) for ii, c, bb in zip(allp.index, allp.column, allp.bbox)]
                res2 = {ii: gg for ii, gg, _ in page_crops((self.book, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", recs_all)) if ii in vset}
                crop_full_ctx.update(res2)

        chars_needed = sorted(set(S.c))
        self._glyph_embs(chars_needed)
        emb1, emb2, emb3, emb2n = self.glyph_emb["1"], self.glyph_emb["2"], self.glyph_emb["3"], self.glyph_emb["2n"]

        crop_hqc3_imgs, cell_order = [], []
        for cell_id in valid_cells:
            if cell_id in crop_raw_map and crop_raw_map[cell_id] is not None:
                cr = crop_raw_map[cell_id]
                cr_128 = cv2.resize(cr, (128, 128))
                crop_hqc3_imgs.append(binarize_canonical(cr_128))
                cell_order.append(cell_id)
        emb_crop_hqc3 = {}
        if crop_hqc3_imgs:
            V = nrm(np.asarray(self.enc.embed(crop_hqc3_imgs, norm=False), np.float32))
            emb_crop_hqc3 = dict(zip(cell_order, V))

        # chẩn đoán vector crop: E_full (sản xuất) so với nhúng lại crop theo cách p05 (recs chỉ ô mẫu) và theo cách sản xuất (recs mọi ô trang)
        cos_p05, cos_full = {}, {}
        if diag_crop:
            ids = [c for c in valid_cells if crop_raw_map.get(c) is not None and crop_full_ctx.get(c) is not None]
            if ids:
                Vp = nrm(np.asarray(self.enc.embed([crop_raw_map[c] for c in ids]), np.float32))     # norm mặc định = True (như t02_embed)
                Vf = nrm(np.asarray(self.enc.embed([crop_full_ctx[c] for c in ids]), np.float32))
                for j, c in enumerate(ids):
                    cos_p05[c] = float(E[c] @ Vp[j]); cos_full[c] = float(E[c] @ Vf[j])

        recs_out = []
        for cell_id in valid_cells:
            sub = S[S.i == cell_id]
            gt_row = sub[sub.y == 1]
            if gt_row.empty:
                continue
            gt_char = gt_row.iloc[0].c
            crop_vec_orig = E[cell_id]
            crop_vec_canon = emb_crop_hqc3.get(cell_id, crop_vec_orig)

            s1 = {c: float(crop_vec_orig @ emb1[c]) for c in sub.c if c in emb1}
            s2 = {c: float(crop_vec_orig @ emb2[c]) for c in sub.c if c in emb2}
            s3 = {c: float(crop_vec_canon @ emb3[c]) for c in sub.c if c in emb3}
            s2n = {c: float(crop_vec_orig @ emb2n[c]) for c in sub.c if c in emb2n}
            if gt_char not in s1:
                continue
            t1, t2, t3 = max(s1, key=s1.get), max(s2, key=s2.get), max(s3, key=s3.get)
            t2n = max(s2n, key=s2n.get)
            wrong = [c for c in sub.c if c != gt_char]
            m1 = s1[gt_char] - max(s1[w] for w in wrong) if wrong else 0.0
            m2 = s2[gt_char] - max(s2[w] for w in wrong) if wrong else 0.0
            m3 = s3[gt_char] - max(s3[w] for w in wrong) if wrong else 0.0
            m2n = s2n[gt_char] - max(s2n[w] for w in wrong) if wrong else 0.0

            def tie(s):
                mx = max(s.values())
                return int(sum(1 for v in s.values() if v == mx) > 1)

            same_ctx = None
            if diag_crop:
                a, b = crop_raw_map.get(cell_id), crop_full_ctx.get(cell_id)
                same_ctx = int(a is not None and b is not None and a.shape == b.shape and int(np.abs(a.astype(int) - b.astype(int)).max()) == 0)
            fF = sub.f_font.fillna(-9).to_numpy()
            recs_out.append({
                "cell_id": int(cell_id), "gt": gt_char,
                "acc1": int(t1 == gt_char), "acc2": int(t2 == gt_char), "acc3": int(t3 == gt_char), "acc2n": int(t2n == gt_char),
                "cos1": s1[gt_char], "cos2": s2[gt_char], "cos3": s3[gt_char], "cos2n": s2n[gt_char],
                "mar1": m1, "mar2": m2, "mar3": m3, "mar2n": m2n,
                "acc_vW_top1": int(sub.iloc[0].y == 1),                 # f_vW tự chọn hạng 1 trong top-3
                "acc_fontF_top1": int(sub.iloc[int(np.argmax(fF))].y == 1),  # f_font sản xuất trong cùng top-3
                "n_truth_in_top3": int((sub.y == 1).sum()),
                "tie1": tie(s1), "tie2": tie(s2), "tie3": tie(s3),
                "crop_missing_hqc3_fallback": int(cell_id not in emb_crop_hqc3),
                "gt_in_font": int(ord(gt_char[0]) in self.cmap) if gt_char else 0,
                "n_cand_in_font": int(sum(1 for c in sub.c if c and ord(c[0]) in self.cmap)),
                "crop_same_as_full_ctx": same_ctx,
                "cos_E_vs_p05crop": cos_p05.get(cell_id), "cos_E_vs_fullctxcrop": cos_full.get(cell_id),
            })
        df = pd.DataFrame(recs_out)
        n = len(df)
        res = {
            "book": self.book, "seed": seed, "n_selected": int(len(selected_cells)), "n_after_filter": n,
            "cells": [int(x) for x in df.cell_id] if n else [],
            "top1": {h: (round(float(df[f"acc{h}"].mean() * 100), 1) if n else None) for h in ("1", "2", "3", "2n")},
            "mean_cos": {h: (round(float(df[f"cos{h}"].mean()), 4) if n else None) for h in ("1", "2", "3", "2n")},
            "mean_margin": {h: (round(float(df[f"mar{h}"].mean()), 4) if n else None) for h in ("1", "2", "3", "2n")},
            "ref_vW_top1": round(float(df.acc_vW_top1.mean() * 100), 1) if n else None,
            "ref_fontF_top1": round(float(df.acc_fontF_top1.mean() * 100), 1) if n else None,
            "n_multi_truth": int((df.n_truth_in_top3 >= 2).sum()) if n else 0,
            "n_tie": {h: int(df[f"tie{h}"].sum()) for h in ("1", "2", "3")} if n else {},
            "n_crop_missing": int(df.crop_missing_hqc3_fallback.sum()) if n else 0,
            "n_gt_not_in_font": int((df.gt_in_font == 0).sum()) if n else 0,
            "n_crop_ctx_differs": (int((df.crop_same_as_full_ctx == 0).sum()) if (n and diag_crop) else None),
            "mean_cos_E_vs_p05crop": (round(float(df.cos_E_vs_p05crop.mean()), 4) if (n and diag_crop) else None),
            "mean_cos_E_vs_fullctxcrop": (round(float(df.cos_E_vs_fullctxcrop.mean()), 4) if (n and diag_crop) else None),
            "elapsed_s": round(time.time() - t0, 1),
            "records": recs_out,
        }
        return res


# ------------------------------------------------------------------------------------------------------------------
# Tổng hợp qua hạt giống
# ------------------------------------------------------------------------------------------------------------------
def summarize(book: str, runs: list[dict], p05_row: dict | None, boots: int = 10000) -> dict:
    HS = ("1", "2", "3", "2n")
    ns = [r["n_after_filter"] for r in runs]
    out = {"book": book, "n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
           "n_after_filter": {"mean": round(float(np.mean(ns)), 2), "min": int(min(ns)), "max": int(max(ns)), "list": ns}}
    per = {}
    for h in HS:
        x = [r["top1"][h] for r in runs]
        per[f"HQC{h}"] = {"mean": round(float(np.mean(x)), 2), "sd": round(sd(x), 2), "min": min(x), "max": max(x), "per_seed": x}
    out["top1_by_seed"] = per
    for a, b in (("2", "1"), ("3", "1"), ("3", "2"), ("2n", "1"), ("2n", "2")):
        d = [r["top1"][a] - r["top1"][b] for r in runs]
        out[f"diff_HQC{a}_minus_HQC{b}"] = {"mean": round(float(np.mean(d)), 2), "sd": round(sd(d), 2), "min": round(min(d), 1), "max": round(max(d), 1),
                                            "n_seeds_positive": int(sum(v > 0 for v in d)), "n_seeds_negative": int(sum(v < 0 for v in d)),
                                            "per_seed": [round(v, 1) for v in d]}
    # độ lệch chuẩn nhị thức kỳ vọng ở n trung bình và p = trung bình Top-1 của HQC tương ứng
    nbar = float(np.mean(ns))
    out["sd_nhi_thuc_ky_vong"] = {f"HQC{h}": round(100 * math.sqrt(max(per[f'HQC{h}']['mean'] / 100 * (1 - per[f'HQC{h}']['mean'] / 100), 0) / nbar), 2) for h in HS}
    out["ref_vW_top1_mean"] = round(float(np.mean([r["ref_vW_top1"] for r in runs])), 2)
    out["ref_fontF_top1_mean"] = round(float(np.mean([r["ref_fontF_top1"] for r in runs])), 2)
    cE = [r["mean_cos_E_vs_p05crop"] for r in runs if r["mean_cos_E_vs_p05crop"] is not None]
    cF = [r["mean_cos_E_vs_fullctxcrop"] for r in runs if r["mean_cos_E_vs_fullctxcrop"] is not None]
    out["chan_doan"] = {
        "tong_o_sau_loc": int(sum(ns)),
        "o_da_su_that_trong_top3": int(sum(r["n_multi_truth"] for r in runs)),
        "o_hoa_diem_top1": {h: int(sum(r["n_tie"].get(h, 0) for r in runs)) for h in ("1", "2", "3")},
        "o_crop_thieu_fallback_hqc3": int(sum(r["n_crop_missing"] for r in runs)),
        "o_chu_dung_khong_co_trong_phong": int(sum(r["n_gt_not_in_font"] for r in runs)),
        "o_crop_cat_lai_khac_crop_san_xuat": (int(sum(r["n_crop_ctx_differs"] for r in runs)) if runs[0]["n_crop_ctx_differs"] is not None else None),
        "cos_E_vs_crop_cat_kieu_p05_tb": (round(float(np.mean(cE)), 4) if cE else None),
        "cos_E_vs_crop_cat_kieu_san_xuat_tb": (round(float(np.mean(cF)), 4) if cF else None),
    }
    # gộp ô DUY NHẤT (mỗi ô đếm một lần; lấy lần xuất hiện đầu) -> mẫu lớn hơn, so cặp
    seen, rows = set(), []
    for r in runs:
        for rec in r["records"]:
            if rec["cell_id"] not in seen:
                seen.add(rec["cell_id"]); rows.append(rec)
    n = len(rows)
    pooled = {"n_o_duy_nhat": n}
    if n:
        acc = {h: np.array([r[f"acc{h}"] for r in rows]) for h in HS}
        for h in HS:
            lo, hi = wilson(int(acc[h].sum()), n)
            pooled[f"HQC{h}"] = {"top1": round(float(acc[h].mean() * 100), 1), "wilson95": [round(lo * 100, 1), round(hi * 100, 1)]}
        vW = np.array([r["acc_vW_top1"] for r in rows]); ff = np.array([r["acc_fontF_top1"] for r in rows])
        pooled["f_vW_hang1_trong_top3"] = round(float(vW.mean() * 100), 1)
        pooled["f_font_san_xuat_trong_top3"] = round(float(ff.mean() * 100), 1)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, size=(boots, n))
        for a, b in (("2", "1"), ("3", "1"), ("3", "2"), ("2n", "1"), ("2n", "2")):
            d = acc[a] - acc[b]
            bs = d[idx].mean(1) * 100
            bb = int(((acc[a] == 1) & (acc[b] == 0)).sum()); cc = int(((acc[a] == 0) & (acc[b] == 1)).sum())
            pooled[f"HQC{a}_tru_HQC{b}"] = {"mean_diff": round(float(d.mean() * 100), 1), "ci95_bootstrap_o": [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)],
                                            "o_a_dung_b_sai": bb, "o_a_sai_b_dung": cc, "mcnemar_exact_p": round(exact_mcnemar(bb, cc), 4)}
    out["gop_o_duy_nhat"] = pooled
    if p05_row:
        t = {"n_cells": p05_row["n_cells"], "HQC1": p05_row["hqc1_white_canvas"]["top1"], "HQC2": p05_row["hqc2_book_paper_canvas"]["top1"],
             "HQC3": p05_row["hqc3_canonical_normalized"]["top1"]}
        out["p05_bang_goc"] = t
        out["p05_khoang_chenh_max_min"] = round(max(t["HQC1"], t["HQC2"], t["HQC3"]) - min(t["HQC1"], t["HQC2"], t["HQC3"]), 1)
        out["p05_vi_tri_trong_phan_bo_hat_giong"] = {f"HQC{h}": {"p05": t[f"HQC{h}"], "pct_seed_nho_hon_hoac_bang": round(float(np.mean([v <= t[f"HQC{h}"] for v in per[f"HQC{h}"]["per_seed"]]) * 100), 0)} for h in ("1", "2", "3")}
    return out


def parse_seeds(s: str):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def load_p05_module():
    spec = importlib.util.spec_from_file_location("p05_orig", str(P05_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # chỉ định nghĩa hàm; main() không chạy (guard __main__)
    return mod


def selfcheck(books, encoder, seeds):
    """Chạy HÀM GỐC evaluate_single_book của p05 (hash -> seed qua monkeypatch, OUT_DIR -> thư mục tạm) và so với bản sao chép."""
    p05 = load_p05_module()
    tmp = Path(tempfile.mkdtemp(prefix="p05_selfcheck_"))
    p05.OUT_DIR = tmp
    res = []
    for b in books:
        br = BookRun(b, encoder)
        for s in seeds:
            p05.hash = (lambda seed: (lambda _x: seed))(s)
            orig = p05.evaluate_single_book(b, encoder, n_sample=20, k_cand=3)
            mine = br.run_seed(s, diag_crop=False)
            ok = (orig["n_cells"] == mine["n_after_filter"] and
                  orig["hqc1_white_canvas"]["top1"] == mine["top1"]["1"] and orig["hqc2_book_paper_canvas"]["top1"] == mine["top1"]["2"] and
                  orig["hqc3_canonical_normalized"]["top1"] == mine["top1"]["3"] and
                  abs(orig["hqc1_white_canvas"]["mean_cos"] - mine["mean_cos"]["1"]) <= 2e-4 and abs(orig["hqc2_book_paper_canvas"]["mean_cos"] - mine["mean_cos"]["2"]) <= 2e-4 and
                  abs(orig["hqc3_canonical_normalized"]["mean_cos"] - mine["mean_cos"]["3"]) <= 2e-4)   # dung sai 2e-4: khác kích thước lô nhúng -> sai số float ~1e-7
            res.append({"book": b, "seed": s, "orig": {"n": orig["n_cells"], "top1": [orig["hqc1_white_canvas"]["top1"], orig["hqc2_book_paper_canvas"]["top1"], orig["hqc3_canonical_normalized"]["top1"]],
                                                       "cos": [orig["hqc1_white_canvas"]["mean_cos"], orig["hqc2_book_paper_canvas"]["mean_cos"], orig["hqc3_canonical_normalized"]["mean_cos"]]},
                        "copy": {"n": mine["n_after_filter"], "top1": [mine["top1"]["1"], mine["top1"]["2"], mine["top1"]["3"]], "cos": [mine["mean_cos"]["1"], mine["mean_cos"]["2"], mine["mean_cos"]["3"]]},
                        "giong_het": bool(ok)})
            print(f"[selfcheck] {b} seed {s}: gốc n={orig['n_cells']} top1={res[-1]['orig']['top1']} cos={res[-1]['orig']['cos']} | sao chép n={mine['n_after_filter']} "
                  f"top1={res[-1]['copy']['top1']} cos={res[-1]['copy']['cos']} -> {'GIỐNG HỆT' if ok else 'KHÁC'}", flush=True)
    (OUT_DIR / "nhieu_mau_selfcheck.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"selfcheck: {sum(r['giong_het'] for r in res)}/{len(res)} giống hệt (thư mục tạm của hàm gốc: {tmp})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", default=["B34", "B18", "L16", "TK"])
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tag", default="s0_9")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-diag-crop", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(2)
    seeds = parse_seeds(a.seeds)

    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    scorers = SI.Scorers(Assets(), "cpu", EMB_DIR, False, lambda m: None)   # CPU, không MPS
    encoder = scorers.enc()
    print(f"Thiết bị: cpu | torch threads {torch.get_num_threads()} | sách {a.books} | hạt giống {seeds} | n={a.n} k={a.k}", flush=True)

    if a.selfcheck:
        selfcheck(a.books, encoder, seeds)
        return

    p05_json = {r["book"]: r for r in json.loads((REPO / "measure_out/_tn11/p05_full_corpus/ket_qua_full_10_bo.json").read_text(encoding="utf-8"))}
    allruns, summaries = {}, {}
    for b in a.books:
        t0 = time.time()
        br = BookRun(b, encoder)
        print(f"== {b}: {len(br.target_cells)} ô đích ({'nhãn người' if br.is_human else 'nhãn pipeline'}); hồ sơ giấy {br.paper['bg_median']}/{br.paper['ink_median']}/{br.paper['contrast']} "
              f"(trang {br.paper['sample_page']})", flush=True)
        runs = []
        for s in seeds:
            r = br.run_seed(s, n_sample=a.n, k_cand=a.k, diag_crop=not a.no_diag_crop)
            runs.append(r)
            print(f"   seed {s:3d}: n={r['n_after_filter']:2d} | HQC1 {r['top1']['1']:5.1f} HQC2 {r['top1']['2']:5.1f} HQC3 {r['top1']['3']:5.1f} HQC2n {r['top1']['2n']:5.1f} | f_vW@1 {r['ref_vW_top1']:5.1f} "
                  f"| đa-sự-thật {r['n_multi_truth']} hoà {r['n_tie']} crop≠sx {r['n_crop_ctx_differs']} [{r['elapsed_s']}s]", flush=True)
        allruns[b] = runs
        summaries[b] = summarize(b, runs, p05_json.get(b))
        s = summaries[b]
        print(f"   -> {b} Top-1 theo hạt giống: " + " | ".join(f"{k} {v['mean']:.1f}±{v['sd']:.1f} [{v['min']}–{v['max']}]" for k, v in s["top1_by_seed"].items()) +
              f" | n sau lọc {s['n_after_filter']['mean']} [{s['n_after_filter']['min']}–{s['n_after_filter']['max']}] [{time.time() - t0:.0f}s]", flush=True)

    (OUT_DIR / f"nhieu_mau_{a.tag}.json").write_text(json.dumps(allruns, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (OUT_DIR / f"nhieu_mau_{a.tag}_tom_tat.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"Đã ghi {OUT_DIR / f'nhieu_mau_{a.tag}_tom_tat.json'}")


if __name__ == "__main__":
    main()
