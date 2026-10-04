"""TN11 p05 (03/10) — THỰC NGHIỆM TOÀN BỘ 10 BỘ DỮ LIỆU TRONG DỰ ÁN
Đưa về cùng Hệ Quy Chiếu Mẫu Giấy & Chuẩn Hoá Thị Giác trên toàn bộ kho ngữ liệu:
  1. stt2   — Sách Các Thánh Truyện Q2 (Chép tay Công giáo)
  2. stt4   — Sách Các Thánh Truyện Q4 (Chép tay Công giáo)
  3. stt11  — Sách Các Thánh Truyện Q11 (Chép tay Công giáo)
  4. Chr    — Chrestomathie 1872 (Sách in văn xuôi)
  5. L83    — Lục Vân Tiên 1883 (Thạch bản)
  6. KVK    — Kim Vân Kiều 1884 (Thạch bản)
  7. L16    — Lục Vân Tiên 1916 (Mộc bản IHR)
  8. TK     — Truyện Kiều 1872 (Mộc bản IHR)
  9. B18    — Sách Kinh Thầy Cả Bình (Chép tay Vatican)
 10. B34    — Sách Đúng Lý Hộ Thân (Chép tay Vatican)

Đầu ra ghi vào: measure_out/_tn11/p05_full_corpus/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT_DIR = REPO / "measure_out" / "_tn11" / "p05_full_corpus"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR = T.OUT / "emb"
CAND_DIR = T.OUT / "cand"
FONT_PATH = str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf")

BOOK_GENRES = {
    "stt2": ("Chép tay Công giáo", "hand"),
    "stt4": ("Chép tay Công giáo", "hand"),
    "stt11": ("Chép tay Công giáo", "hand"),
    "Chr": ("In văn xuôi", "print"),
    "L83": ("Thạch bản", "litho"),
    "KVK": ("Thạch bản", "litho"),
    "L16": ("Mộc bản IHR", "wood"),
    "TK": ("Mộc bản IHR", "wood"),
    "B18": ("Chép tay Vatican", "hand_borg"),
    "B34": ("Chép tay Vatican", "hand_borg"),
}


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


def evaluate_single_book(book: str, encoder, n_sample: int = 20, k_cand: int = 3) -> dict:
    t0 = time.time()
    genre_name, genre_key = BOOK_GENRES[book]
    paper_model = extract_book_paper_model(book)

    # Nạp base và cand
    D = T.load_base(book)
    E_full = nrm(np.load(EMB_DIR / f"{book}_enc.npy").astype(np.float32))
    F = pd.read_pickle(CAND_DIR / f"{book}.pkl")

    # Xác định Ground Truth / Target Label
    if "y" in F.columns and F.y.sum() > 0:
        tru = F[F.y == 1].drop_duplicates("i").set_index("i")
        target_cells = list(tru.index)
        is_human = True
    else:
        # Lấy từ D.label (đã qua kiểm định của pipeline)
        lbl = D.label.values
        tgt = lbl[F.i.values]
        valid_mask = (F.c.values == tgt) & (tgt != "")
        target_cells = sorted(set(F.i.values[valid_mask]))
        F["y"] = np.where(valid_mask, 1, 0)
        is_human = False

    # Chọn mẫu n_sample ô
    rng = np.random.default_rng(hash(book) % 100000)
    selected_cells = np.sort(rng.choice(target_cells, size=min(n_sample, len(target_cells)), replace=False))

    # Lấy top k ứng viên
    S = F[F.i.isin(selected_cells)].copy()
    S["rk"] = S.f_vW.fillna(-9) if "f_vW" in S.columns else S.f_font.fillna(-9)
    S = S.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(k_cand)

    has_truth = S.groupby("i").y.max()
    valid_cells = sorted(has_truth[has_truth == 1].index)
    S = S[S.i.isin(valid_cells)].copy()

    # Đọc ảnh crop thật
    from t02_embed import page_crops
    cfg = T.BOOKS[book]
    prep = T.REPO / cfg["prep"]

    crop_raw_map = {}
    pages_needed = set(D.loc[valid_cells, "page"])
    for pg in pages_needed:
        allp = D[D.page == pg]
        recs = [(ii, c, T.bbox_of(bb)) for ii, c, bb in zip(allp.index, allp.column, allp.bbox) if ii in valid_cells]
        if recs:
            res = {ii: gg for ii, gg, _ in page_crops((book, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", recs))}
            crop_raw_map.update(res)

    chars_needed = sorted(set(S.c))
    raw_font_dict = {c: render_font_raw(c, 128) for c in chars_needed}

    imgs_hqc1 = [raw_font_dict[c] for c in chars_needed]
    imgs_hqc2 = [blend_on_paper(raw_font_dict[c], paper_model) for c in chars_needed]
    imgs_hqc3 = [binarize_canonical(raw_font_dict[c]) for c in chars_needed]

    V_hqc1 = nrm(np.asarray(encoder.embed(imgs_hqc1, norm=False), np.float32))
    V_hqc2 = nrm(np.asarray(encoder.embed(imgs_hqc2, norm=False), np.float32))
    V_hqc3 = nrm(np.asarray(encoder.embed(imgs_hqc3, norm=False), np.float32))

    emb_hqc1 = dict(zip(chars_needed, V_hqc1))
    emb_hqc2 = dict(zip(chars_needed, V_hqc2))
    emb_hqc3 = dict(zip(chars_needed, V_hqc3))

    crop_hqc3_imgs = []
    cell_order = []
    for cell_id in valid_cells:
        if cell_id in crop_raw_map:
            cr = crop_raw_map[cell_id]
            cr_128 = cv2.resize(cr, (128, 128))
            crop_hqc3_imgs.append(binarize_canonical(cr_128))
            cell_order.append(cell_id)

    if crop_hqc3_imgs:
        V_crop_hqc3 = nrm(np.asarray(encoder.embed(crop_hqc3_imgs, norm=False), np.float32))
        emb_crop_hqc3 = dict(zip(cell_order, V_crop_hqc3))
    else:
        emb_crop_hqc3 = {}

    records = []
    montage_rows = []

    for cell_id in valid_cells:
        sub = S[S.i == cell_id]
        gt_row = sub[sub.y == 1]
        if gt_row.empty:
            continue
        gt_char = gt_row.iloc[0].c
        crop_vec_orig = E_full[cell_id]
        crop_vec_canon = emb_crop_hqc3.get(cell_id, crop_vec_orig)

        s_hqc1 = {c: float(crop_vec_orig @ emb_hqc1[c]) for c in sub.c if c in emb_hqc1}
        s_hqc2 = {c: float(crop_vec_orig @ emb_hqc2[c]) for c in sub.c if c in emb_hqc2}
        s_hqc3 = {c: float(crop_vec_canon @ emb_hqc3[c]) for c in sub.c if c in emb_hqc3}

        if gt_char not in s_hqc1:
            continue

        top1_1 = max(s_hqc1, key=s_hqc1.get)
        top1_2 = max(s_hqc2, key=s_hqc2.get)
        top1_3 = max(s_hqc3, key=s_hqc3.get)

        wrong = [c for c in sub.c if c != gt_char]
        m1 = s_hqc1[gt_char] - max(s_hqc1[w] for w in wrong) if wrong else 0.0
        m2 = s_hqc2[gt_char] - max(s_hqc2[w] for w in wrong) if wrong else 0.0
        m3 = s_hqc3[gt_char] - max(s_hqc3[w] for w in wrong) if wrong else 0.0

        records.append({
            "cell_id": int(cell_id),
            "ground_truth": gt_char,
            "acc_hqc1": int(top1_1 == gt_char),
            "cos_hqc1": s_hqc1[gt_char],
            "margin_hqc1": m1,
            "acc_hqc2": int(top1_2 == gt_char),
            "cos_hqc2": s_hqc2[gt_char],
            "margin_hqc2": m2,
            "acc_hqc3": int(top1_3 == gt_char),
            "cos_hqc3": s_hqc3[gt_char],
            "margin_hqc3": m3,
        })

        if cell_id in crop_raw_map and len(montage_rows) < 6:
            cr = cv2.resize(crop_raw_map[cell_id], (96, 96))
            im_w = cv2.resize(raw_font_dict[gt_char], (96, 96))
            im_paper = cv2.resize(blend_on_paper(raw_font_dict[gt_char], paper_model), (96, 96))
            cr_canon = cv2.resize(binarize_canonical(cr), (96, 96))
            im_canon = cv2.resize(binarize_canonical(raw_font_dict[gt_char]), (96, 96))
            sep = np.full((96, 3), 160, dtype=np.uint8)
            row = np.hstack([cr, sep, im_w, sep, im_paper, sep, cr_canon, sep, im_canon])
            montage_rows.append(row)

    if montage_rows:
        cv2.imwrite(str(OUT_DIR / f"montage_{book}.png"), np.vstack(montage_rows))

    df_b = pd.DataFrame(records)
    res = {
        "book": book,
        "genre": genre_name,
        "genre_key": genre_key,
        "is_human_label": is_human,
        "n_cells": len(records),
        "paper_stats": {
            "bg": paper_model["bg_median"],
            "ink": paper_model["ink_median"],
            "contrast": paper_model["contrast"],
        },
        "hqc1_white_canvas": {
            "top1": round(float(df_b.acc_hqc1.mean() * 100), 1),
            "mean_cos": round(float(df_b.cos_hqc1.mean()), 4),
            "mean_margin": round(float(df_b.margin_hqc1.mean()), 4),
        },
        "hqc2_book_paper_canvas": {
            "top1": round(float(df_b.acc_hqc2.mean() * 100), 1),
            "mean_cos": round(float(df_b.cos_hqc2.mean()), 4),
            "mean_margin": round(float(df_b.margin_hqc2.mean()), 4),
        },
        "hqc3_canonical_normalized": {
            "top1": round(float(df_b.acc_hqc3.mean() * 100), 1),
            "mean_cos": round(float(df_b.cos_hqc3.mean()), 4),
            "mean_margin": round(float(df_b.margin_hqc3.mean()), 4),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }

    print(f"[{book:6s} | {genre_name:20s}] n={len(records):2d} | "
          f"HQC1: {res['hqc1_white_canvas']['top1']:5.1f}% (cos {res['hqc1_white_canvas']['mean_cos']:.3f}) | "
          f"HQC2: {res['hqc2_book_paper_canvas']['top1']:5.1f}% (cos {res['hqc2_book_paper_canvas']['mean_cos']:.3f}) | "
          f"HQC3: {res['hqc3_canonical_normalized']['top1']:5.1f}% (cos {res['hqc3_canonical_normalized']['mean_cos']:.3f}) [{res['elapsed_s']}s]")
    return res


def main():
    print("=" * 95)
    print("THỰC NGHIỆM TN11-p05: ĐO ĐẠC HỆ QUY CHIẾU THỊ GIÁC TRÊN TOÀN BỘ 10 BỘ DỮ LIỆU CỦA DỰ ÁN")
    print("=" * 95)

    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Thiết bị phần cứng: {device}\n")
    scorers = SI.Scorers(Assets(), device, EMB_DIR, False, lambda m: None)
    encoder = scorers.enc()

    all_results = []
    t_start = time.time()

    for book in T.ORDER:
        res = evaluate_single_book(book, encoder, n_sample=20, k_cand=3)
        all_results.append(res)

    # Lưu kết quả JSON
    json_path = OUT_DIR / "ket_qua_full_10_bo.json"
    json_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Tạo bảng Markdown tổng hợp
    md_lines = [
        "# BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ĐỒNG BỘ HỆ QUY CHIẾU TRÊN TOÀN BỘ 10 BỘ DỮ LIỆU",
        "",
        "| STT | Mã bộ | Tên sách / Thể loại | Nhãn kiểm định | Mẫu giấy (Nền/Mực/Tương phản) | HQC 1: Nền trắng | HQC 2: MẪU GIẤY SÁCH | HQC 3: CHUẨN HOÁ NHỊ PHÂN |",
        "|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for idx, r in enumerate(all_results, 1):
        lbl_type = "Người (GT)" if r["is_human_label"] else "Pipeline GOLD"
        p = r["paper_stats"]
        paper_str = f"{p['bg']}/{p['ink']}/{p['contrast']}"
        h1 = f"{r['hqc1_white_canvas']['top1']:.1f}% (cos {r['hqc1_white_canvas']['mean_cos']:.3f})"
        h2 = f"{r['hqc2_book_paper_canvas']['top1']:.1f}% (cos {r['hqc2_book_paper_canvas']['mean_cos']:.3f})"
        h3 = f"{r['hqc3_canonical_normalized']['top1']:.1f}% (cos {r['hqc3_canonical_normalized']['mean_cos']:.3f})"
        md_lines.append(f"| {idx} | **{r['book']}** | {r['genre']} | {lbl_type} | {paper_str} | {h1} | {h2} | {h3} |")

    md_path = OUT_DIR / "BANG_TONG_HOP_10_BO.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n" + "=" * 95)
    print(f"HOÀN TẤT THỰC NGHIỆM TRÊN TẤT CẢ 10 BỘ DỮ LIỆU TRONG {time.time() - t_start:.1f}s!")
    print(f"  -> File JSON: {json_path}")
    print(f"  -> File Báo cáo Markdown: {md_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()
