"""TN11 p04 (03/10) — THỰC NGHIỆM ĐA BỘ DỮ LIỆU: ĐƯA VỀ CÙNG HỆ QUY CHIẾU MẪU GIẤY & CHUẨN HOÁ

Thực nghiệm theo yêu cầu:
"thử nghiệm trên từng bộ data sinh ảnh font tương đồng cùng mẫu giấy đưa hết về một hệ quy chiếu so sánh mới chính xác"

Nội dung:
1. Chạy trên cả 4 bộ dữ liệu có nhãn người:
   - Borg.34 (SachDungLyHoThan, chép tay Vatican)
   - Borg.18 (SachKinhThayCaBinh, chép tay Vatican)
   - L16 (LucVanTien1916, mộc bản IHR)
   - TK (TruyenKieu1872, mộc bản IHR)
2. Thiết lập 3 Hệ Quy Chiếu (HQC) thị giác để so khớp với Crop thật:
   - HQC 1 (Baseline cũ - Lệch miền): Font/Glyph trên nền trắng tinh 255.
   - HQC 2 (Cùng Mẫu Giấy Sách): Nét chữ sinh được nhúng lên chính texture nền giấy thật
     của cuốn sách đó (cùng độ sáng nền, cùng độ tương phản mực-giấy).
   - HQC 3 (Cùng Chuẩn Hoá Nhị Phân): Cả Crop thật và Glyph đều được bóc tách nền giấy,
     nhị phân hoá Otsu, căn giữa khung 128x128 với lề chuẩn 10%.
3. Đánh giá qua Sub-Center ArcFace ResNet Encoder:
   - Top-1 Accuracy (%)
   - Mean Cosine Similarity với nhãn đúng
   - Margin Gap Δ = cos(đúng) - max(sai)
4. Xuất ảnh trực quan đối chiếu và file kết quả JSON.

Đầu ra ghi vào: measure_out/_tn11/p04_he_quy_chieu/
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

OUT_DIR = REPO / "measure_out" / "_tn11" / "p04_he_quy_chieu"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR = T.OUT / "emb"
CAND_DIR = T.OUT / "cand"
FONT_PATH = str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf")


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def render_font_raw(char: str, size: int = 128) -> np.ndarray:
    """Render Nom character as clean black on white (0 on 255)."""
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
    """Trích xuất đặc trưng nền giấy và texture patch của từng cuốn sách."""
    cfg = T.BOOKS[book]
    prep = T.REPO / cfg["prep"]
    pages_dir = prep / "pages"
    sample_pages = sorted(pages_dir.glob("*.png"))
    if not sample_pages:
        raise FileNotFoundError(f"Không có ảnh trang trong {pages_dir}")

    # Lấy 1 trang tiêu biểu (trang 10 hoặc trang giữa)
    sample_page = sample_pages[min(10, len(sample_pages) - 1)]
    page_img = cv2.imread(str(sample_page), cv2.IMREAD_GRAYSCALE)
    if page_img is None:
        raise ValueError(f"Không đọc được ảnh {sample_page}")

    # Tìm ngưỡng Otsu phân tách mực và giấy
    T_val, _ = cv2.threshold(page_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper_mask = page_img >= T_val

    # Thống kê màu giấy và màu mực
    paper_pixels = page_img[paper_mask]
    ink_pixels = page_img[~paper_mask]

    bg_median = float(np.median(paper_pixels)) if len(paper_pixels) else 240.0
    ink_median = float(np.median(ink_pixels)) if len(ink_pixels) else 60.0
    bg_std = float(np.std(paper_pixels)) if len(paper_pixels) else 10.0

    # Tìm một patch giấy sạch (128x128) không chứa mực
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
        # Nếu trang quá đặc chữ, tạo synthetic texture từ thống kê thật của sách
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
    """HQC 2: Nhúng nét chữ lên chính texture mẫu giấy của cuốn sách."""
    patch = paper_model["paper_patch"].astype(np.float32)
    ink_target = paper_model["ink_median"]

    # raw_glyph có nền 255, nét chữ 0
    alpha = (255.0 - raw_glyph.astype(np.float32)) / 255.0  # 1 ở nét mực, 0 ở nền

    # Blend nét mực lên nền giấy
    blended = (1.0 - alpha) * patch + alpha * ink_target
    return np.clip(blended, 0, 255).astype(np.uint8)


def binarize_canonical(img: np.ndarray) -> np.ndarray:
    """HQC 3: Chuẩn hoá nhị phân, bóc sạch nền về trắng 255, căn giữa khung 128x128."""
    # Nhị phân hoá Otsu
    _, bin_inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Loại bỏ đốm nhiễu li ti
    n, lab, st, _ = cv2.connectedComponentsWithStats(bin_inv, connectivity=8)
    clean_inv = np.zeros_like(bin_inv)
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] >= 6:  # bỏ đốm < 6 pixel
            clean_inv[lab == j] = 255

    # Cắt sát bounding box mực
    ys, xs = np.nonzero(clean_inv)
    if len(ys) == 0:
        return np.full((128, 128), 255, dtype=np.uint8)

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped_char = clean_inv[y0:y1, x0:x1]

    # Căn giữa vào khung 128x128 với lề 10%
    h, w = cropped_char.shape
    max_dim = max(h, w)
    target_dim = int(128 * 0.8)  # 80% của 128 (lề 10% mỗi phía)
    scale = target_dim / max(max_dim, 1)

    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    scaled = cv2.resize(cropped_char, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((128, 128), dtype=np.uint8)
    sy = (128 - new_h) // 2
    sx = (128 - new_w) // 2
    canvas[sy:sy + new_h, sx:sx + new_w] = scaled

    # Đảo về nền trắng mực đen
    return 255 - canvas


def evaluate_book(book: str, encoder, n_sample: int = 15, k_cand: int = 3) -> dict:
    print(f"\n[{book}] Đang xử lý bộ dữ liệu: {book}...")
    paper_model = extract_book_paper_model(book)
    print(f"  -> Nền giấy trung vị = {paper_model['bg_median']}, Mực = {paper_model['ink_median']}, Tương phản = {paper_model['contrast']}")

    # Nạp dữ liệu
    D = T.load_base(book)
    E_full = nrm(np.load(EMB_DIR / f"{book}_enc.npy").astype(np.float32))
    F = pd.read_pickle(CAND_DIR / f"{book}.pkl")

    # Lấy các ô có sự thật người (y == 1)
    tru = F[F.y == 1].drop_duplicates("i").set_index("i")
    # Ưu tiên các ô chữ hiếm
    rare_idx = tru.index[tru.n_self.fillna(0) < 3]
    if len(rare_idx) < n_sample:
        rare_idx = tru.index

    rng = np.random.default_rng(hash(book) % 100000)
    selected_cells = np.sort(rng.choice(rare_idx, size=min(n_sample, len(rare_idx)), replace=False))

    # Lấy top k ứng viên
    S = F[F.i.isin(selected_cells)].copy()
    S["rk"] = S.f_vW.fillna(-9)
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

    # Tạo các phiên bản ảnh ứng viên theo 3 HQC
    raw_font_dict = {c: render_font_raw(c, 128) for c in chars_needed}

    # HQC 1: Nền trắng tinh
    imgs_hqc1 = [raw_font_dict[c] for c in chars_needed]

    # HQC 2: Nền giấy của chính cuốn sách
    imgs_hqc2 = [blend_on_paper(raw_font_dict[c], paper_model) for c in chars_needed]

    # HQC 3: Chuẩn hoá nhị phân canonical
    imgs_hqc3 = [binarize_canonical(raw_font_dict[c]) for c in chars_needed]

    # Trích xuất embedding cho các glyph
    V_hqc1 = nrm(np.asarray(encoder.embed(imgs_hqc1, norm=False), np.float32))
    V_hqc2 = nrm(np.asarray(encoder.embed(imgs_hqc2, norm=False), np.float32))
    V_hqc3 = nrm(np.asarray(encoder.embed(imgs_hqc3, norm=False), np.float32))

    emb_hqc1 = dict(zip(chars_needed, V_hqc1))
    emb_hqc2 = dict(zip(chars_needed, V_hqc2))
    emb_hqc3 = dict(zip(chars_needed, V_hqc3))

    # Đưa crop thật vào HQC tương ứng:
    # Với HQC 1 & HQC 2: crop thật dùng nguyên bản (đã trích xuất trong E_full)
    # Với HQC 3: crop thật được đưa qua binarize_canonical
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

    # Đánh giá so sánh trên từng ô
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

        s_hqc1 = {}
        s_hqc2 = {}
        s_hqc3 = {}

        for _, row in sub.iterrows():
            c = row.c
            s_hqc1[c] = float(crop_vec_orig @ emb_hqc1[c])
            s_hqc2[c] = float(crop_vec_orig @ emb_hqc2[c])
            s_hqc3[c] = float(crop_vec_canon @ emb_hqc3[c])

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
            # HQC 1
            "acc_hqc1": int(top1_1 == gt_char),
            "cos_hqc1": s_hqc1[gt_char],
            "margin_hqc1": m1,
            # HQC 2 (Cùng mẫu giấy)
            "acc_hqc2": int(top1_2 == gt_char),
            "cos_hqc2": s_hqc2[gt_char],
            "margin_hqc2": m2,
            # HQC 3 (Cùng chuẩn hoá nhị phân)
            "acc_hqc3": int(top1_3 == gt_char),
            "cos_hqc3": s_hqc3[gt_char],
            "margin_hqc3": m3,
        })

        # Ghép ảnh trực quan cho hàng mẫu
        if cell_id in crop_raw_map and len(montage_rows) < 6:
            cr = cv2.resize(crop_raw_map[cell_id], (96, 96))
            im_w = cv2.resize(raw_font_dict[gt_char], (96, 96))
            im_paper = cv2.resize(blend_on_paper(raw_font_dict[gt_char], paper_model), (96, 96))
            cr_canon = cv2.resize(binarize_canonical(cr), (96, 96))
            im_canon = cv2.resize(binarize_canonical(raw_font_dict[gt_char]), (96, 96))

            sep = np.full((96, 3), 160, dtype=np.uint8)
            # Hàng: [Crop gốc | Glyph nền trắng | Glyph mẫu giấy sách | Crop chuẩn hoá | Glyph chuẩn hoá]
            row = np.hstack([cr, sep, im_w, sep, im_paper, sep, cr_canon, sep, im_canon])
            montage_rows.append(row)

    if montage_rows:
        montage_img = np.vstack(montage_rows)
        cv2.imwrite(str(OUT_DIR / f"montage_{book}.png"), montage_img)

    df_b = pd.DataFrame(records)
    res = {
        "book": book,
        "n_cells": len(records),
        "paper_stats": {k: v for k, v in paper_model.items() if k != "paper_patch"},
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
    }
    print(f"  -> [HQC 1 - Nền trắng]: Top-1 = {res['hqc1_white_canvas']['top1']}% | Cos = {res['hqc1_white_canvas']['mean_cos']} | Margin = {res['hqc1_white_canvas']['mean_margin']:+.4f}")
    print(f"  -> [HQC 2 - MẪU GIẤY]: Top-1 = {res['hqc2_book_paper_canvas']['top1']}% | Cos = {res['hqc2_book_paper_canvas']['mean_cos']} | Margin = {res['hqc2_book_paper_canvas']['mean_margin']:+.4f}")
    print(f"  -> [HQC 3 - CHUẨN HOÁ]: Top-1 = {res['hqc3_canonical_normalized']['top1']}% | Cos = {res['hqc3_canonical_normalized']['mean_cos']} | Margin = {res['hqc3_canonical_normalized']['mean_margin']:+.4f}")
    return res


def main():
    print("=" * 75)
    print("THỰC NGHIỆM TN11-p04: SO SÁNH TRÊN CÙNG HỆ QUY CHIẾU MẪU GIẤY & CHUẨN HOÁ")
    print("=" * 75)

    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Thiết bị tính toán: {device}")
    scorers = SI.Scorers(Assets(), device, EMB_DIR, False, lambda m: None)
    encoder = scorers.enc()

    all_books = ["B34", "B18", "L16", "TK"]
    results = {}

    for b in all_books:
        results[b] = evaluate_book(b, encoder, n_sample=15, k_cand=3)

    # Tổng hợp toàn bộ
    out_file = OUT_DIR / "ket_qua_cac_bo.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 75)
    print(f"HOÀN TẤT THỰC NGHIỆM TRÊN TẤT CẢ 4 BỘ SÁCH. KẾT QUẢ ĐÃ LƯU VÀO:\n{out_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
