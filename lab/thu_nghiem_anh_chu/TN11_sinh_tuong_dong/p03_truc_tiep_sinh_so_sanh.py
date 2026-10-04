"""TN11 p03 (03/10) — THỰC NGHIỆM TRỰC TIẾP: Sinh ảnh tương đồng theo phong cách cuốn sách rồi so sánh ảnh.

Mục tiêu phục vụ Đồ án ThS & Bản góp ý MI3:
1. Lấy mẫu ô chữ viết tay có nhãn người thực tế (Ground Truth) từ sách Borg.34.
2. Sinh ảnh tương đồng trực tiếp bằng FontDiffuser với phong cách trích xuất từ chính cuốn sách.
3. So sánh ảnh thị giác qua mạng Sub-center ArcFace / ResNet encoder:
   - Baseline 1: Phông in chuẩn NomNaTong (f_font)
   - Baseline 2: FontDiffuser phong cách chung STT2 (f_fd)
   - Đề xuất: FontDiffuser thích ứng phong cách cuốn sách (f_gen_book)
4. Đo đạc định lượng: Top-1 accuracy, Cosine Similarity, Margin Gap Δ = cos(đúng) - max(sai).
5. Xuất bảng số liệu và ảnh ghép trực quan so sánh.

Toàn bộ đầu ra ghi vào: measure_out/_tn11/p03_direct/
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
sys.path.insert(0, str(REPO / "font_diffusion"))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT_DIR = REPO / "measure_out" / "_tn11" / "p03_direct"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR = T.OUT / "emb"
CAND_DIR = T.OUT / "cand"


def nrm(M):
    """L2 normalize matrix or vector."""
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def render_font_image(char: str, font_path: str, size: int = 128) -> np.ndarray:
    """Render Nom character as grayscale image with NomNaTong font."""
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, int(size * 0.75))
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - w) // 2 - bbox[0]
        y = (size - h) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
    except Exception:
        draw.rectangle([(10, 10), (size - 10, size - 10)], outline=128, width=2)
    return np.array(img)


def main():
    print("=" * 70)
    print("THỰC NGHIỆM TN11-p03: SINH ẢNH TƯƠNG ĐỒNG & SO SÁNH ẢNH THỊ GIÁC")
    print("=" * 70)

    book = "B34"
    print(f"\n[1] Nạp dữ liệu sách {book} (SachDungLyHoThan, chép tay Vatican Borg)...")
    D = T.load_base(book)
    E = nrm(np.load(EMB_DIR / f"{book}_enc.npy").astype(np.float32))
    F = pd.read_pickle(CAND_DIR / f"{book}.pkl")

    # Chọn các ô có sự thật người (y == 1) và là chữ hiếm (ít hơn 2 mẫu cùng sách)
    tru = F[F.y == 1].drop_duplicates("i").set_index("i")
    rare_idx = tru.index[tru.n_self.fillna(0) < 2]

    # Lấy mẫu hạt giống tất định 12 ô tiêu biểu để sinh và so sánh
    rng = np.random.default_rng(42)
    selected_cells = np.sort(rng.choice(rare_idx, size=12, replace=False))

    # Lấy top k=3 ứng viên cho mỗi ô
    k = 3
    S = F[F.i.isin(selected_cells)].copy()
    S["rk"] = S.f_vW.fillna(-9)
    S = S.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(k)

    # Đảm bảo mỗi ô có chữ đúng trong tập ứng viên
    has_truth = S.groupby("i").y.max()
    valid_cells = has_truth[has_truth == 1].index
    S = S[S.i.isin(valid_cells)].copy()
    cells_list = sorted(set(S.i))
    print(f"  -> Đã chọn {len(cells_list)} ô chữ hiếm có nhãn người để thử nghiệm.")

    # Thu thập danh sách các ký tự duy nhất cần sinh ảnh
    chars_to_gen = sorted(set(S.c))
    print(f"  -> Tập ứng viên gồm {len(chars_to_gen)} ký tự Nôm: {' '.join(chars_to_gen)}")

    # [2] Chuẩn bị Style Reference từ chính sách B34
    style_path = REPO / "measure_out" / "_tn11" / f"p02_{book}" / "style_0.png"
    if not style_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh phong cách {style_path}")
    print(f"\n[2] Ảnh phong cách mẫu của sách: {style_path.name}")

    # [3] Sinh ảnh bằng FontDiffuser với phong cách sách B34
    print(f"\n[3] Tiến hành sinh ảnh bằng FontDiffuser (Style-Adaptive)...")
    from core.ranking.fontdiffusion_gen import FontDiffusionGenerator

    gen_dir = OUT_DIR / "gen_book_style"
    gen_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(REPO / "font_diffusion/ckpt/PROD")
    font_path = str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf")

    # Khởi tạo generator
    gen_engine = FontDiffusionGenerator(
        ckpt_dir=ckpt_path,
        phase1_ckpt_dir=ckpt_path,
        font_path=font_path,
        cache_dir=str(gen_dir),
        batch_size=4,
    )

    t0 = time.time()
    generated_map = gen_engine.generate(
        characters=chars_to_gen,
        style_image_path=str(style_path),
        style_name="b34_style",
    )
    print(f"  -> Hoàn thành sinh {len(generated_map)}/{len(chars_to_gen)} ảnh trong {time.time() - t0:.1f}s.")

    # [4] Nạp mô hình ArcFace Encoder để so sánh đặc trưng ảnh
    print(f"\n[4] Nạp mô hình Feature Encoder (Sub-Center ArcFace ResNet)...")
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    scorers = SI.Scorers(Assets(), device, EMB_DIR, False, lambda m: None)
    encoder = scorers.enc()

    # Lấy ảnh crop thực tế từ trang sách
    from t02_embed import page_crops

    cfg = T.BOOKS[book]
    prep = T.REPO / cfg["prep"]

    # Đọc ảnh crop gốc
    crop_images = {}
    pages_needed = set(D.loc[cells_list, "page"])
    for pg in pages_needed:
        allp = D[D.page == pg]
        recs = [(ii, c, T.bbox_of(bb)) for ii, c, bb in zip(allp.index, allp.column, allp.bbox) if ii in cells_list]
        if recs:
            res = {ii: gg for ii, gg, _ in page_crops((book, prep / "pages" / f"{pg}.png", prep, cfg["crop"] == "original", recs))}
            crop_images.update(res)

    # [5] Trích xuất embedding cho các nguồn ảnh:
    print(f"\n[5] Trích xuất embedding & So sánh độ tương đồng góc siêu cầu cos(θ)...")

    font_imgs, fd_imgs, gen_imgs = [], [], []
    valid_chars = []
    k3 = np.ones((3, 3), np.uint8)

    for c in chars_to_gen:
        # Font in
        f_im = render_font_image(c, font_path, size=128)
        font_imgs.append(f_im)

        # FD chung (từ gannhanocr-fd hoặc cache)
        fd_path = REPO / "gannhanocr-fd" / f"{ord(c):02X}"[:2] / f"U+{ord(c):04X}.png"
        if not fd_path.exists():
            fd_im = f_im
        else:
            fd_im = cv2.imread(str(fd_path), cv2.IMREAD_GRAYSCALE)
            if fd_im is None:
                fd_im = f_im
        fd_imgs.append(fd_im)

        # Gen book style
        gen_path = gen_dir / "b34_style" / f"U+{ord(c):04X}.png"
        if gen_path.exists():
            g_im = cv2.imread(str(gen_path), cv2.IMREAD_GRAYSCALE)
            g_im = cv2.dilate(g_im, k3, iterations=2)
        else:
            g_im = fd_im
        gen_imgs.append(g_im)

        valid_chars.append(c)

    V_font = nrm(np.asarray(encoder.embed(font_imgs, norm=False), np.float32))
    V_fd = nrm(np.asarray(encoder.embed(fd_imgs, norm=False), np.float32))
    V_gen = nrm(np.asarray(encoder.embed(gen_imgs, norm=False), np.float32))

    emb_font = dict(zip(valid_chars, V_font))
    emb_fd = dict(zip(valid_chars, V_fd))
    emb_gen = dict(zip(valid_chars, V_gen))

    # [6] Tính toán điểm và Margin cho từng ô
    records = []
    montage_rows = []

    for cell_id in cells_list:
        sub = S[S.i == cell_id]
        gt_row = sub[sub.y == 1]
        if gt_row.empty:
            continue
        gt_char = gt_row.iloc[0].c
        crop_vec = E[cell_id]
        crop_raw = crop_images.get(cell_id, None)

        scores_font = {}
        scores_fd = {}
        scores_gen = {}

        for _, row in sub.iterrows():
            c = row.c
            scores_font[c] = float(crop_vec @ emb_font[c]) if c in emb_font else 0.0
            scores_fd[c] = float(crop_vec @ emb_fd[c]) if c in emb_fd else 0.0
            scores_gen[c] = float(crop_vec @ emb_gen[c]) if c in emb_gen else 0.0

        top1_font = max(scores_font, key=scores_font.get)
        top1_fd = max(scores_fd, key=scores_fd.get)
        top1_gen = max(scores_gen, key=scores_gen.get)

        wrong_chars = [c for c in sub.c if c != gt_char]
        if wrong_chars:
            max_wrong_font = max(scores_font[w] for w in wrong_chars)
            max_wrong_fd = max(scores_fd[w] for w in wrong_chars)
            max_wrong_gen = max(scores_gen[w] for w in wrong_chars)

            margin_font = scores_font[gt_char] - max_wrong_font
            margin_fd = scores_fd[gt_char] - max_wrong_fd
            margin_gen = scores_gen[gt_char] - max_wrong_gen
        else:
            margin_font = margin_fd = margin_gen = 0.0

        records.append({
            "cell_id": int(cell_id),
            "page": str(D.loc[cell_id, "page"]),
            "syllable": str(D.loc[cell_id, "syllable"]),
            "ground_truth": gt_char,
            "candidates": [str(c) for c in sub.c],
            # Font
            "top1_font": top1_font,
            "acc_font": int(top1_font == gt_char),
            "cos_font": round(scores_font[gt_char], 4),
            "margin_font": round(margin_font, 4),
            # FD chung
            "top1_fd": top1_fd,
            "acc_fd": int(top1_fd == gt_char),
            "cos_fd": round(scores_fd[gt_char], 4),
            "margin_fd": round(margin_fd, 4),
            # Gen phong cách sách
            "top1_gen": top1_gen,
            "acc_gen": int(top1_gen == gt_char),
            "cos_gen": round(scores_gen[gt_char], 4),
            "margin_gen": round(margin_gen, 4),
        })

        if crop_raw is not None:
            c_gt = gt_char
            c_h = 100
            cr_resized = cv2.resize(crop_raw, (c_h, c_h))
            f_resized = cv2.resize(render_font_image(c_gt, font_path, 128), (c_h, c_h))
            fd_raw = cv2.imread(str(REPO / "gannhanocr-fd" / f"{ord(c_gt):02X}"[:2] / f"U+{ord(c_gt):04X}.png"), cv2.IMREAD_GRAYSCALE)
            fd_resized = cv2.resize(fd_raw if fd_raw is not None else f_resized, (c_h, c_h))
            g_raw = cv2.imread(str(gen_dir / "b34_style" / f"U+{ord(c_gt):04X}.png"), cv2.IMREAD_GRAYSCALE)
            g_resized = cv2.resize(g_raw if g_raw is not None else fd_resized, (c_h, c_h))

            sep = np.full((c_h, 4), 180, dtype=np.uint8)
            row_img = np.hstack([cr_resized, sep, f_resized, sep, fd_resized, sep, g_resized])
            montage_rows.append(row_img)

    if montage_rows:
        montage = np.vstack(montage_rows[:8])
        cv2.imwrite(str(OUT_DIR / "so_sanh_truc_quan.png"), montage)
        print(f"  -> Đã lưu ảnh đối chiếu trực quan tại: {OUT_DIR / 'so_sanh_truc_quan.png'}")

    df_res = pd.DataFrame(records)
    acc_font = df_res.acc_font.mean() * 100
    acc_fd = df_res.acc_fd.mean() * 100
    acc_gen = df_res.acc_gen.mean() * 100

    margin_font_mean = df_res.margin_font.mean()
    margin_fd_mean = df_res.margin_fd.mean()
    margin_gen_mean = df_res.margin_gen.mean()

    cos_font_mean = df_res.cos_font.mean()
    cos_fd_mean = df_res.cos_fd.mean()
    cos_gen_mean = df_res.cos_gen.mean()

    summary = {
        "book": book,
        "n_cells": len(records),
        "metrics": {
            "top1_accuracy": {
                "font_nomnatong": round(acc_font, 1),
                "fontdiffuser_global_stt": round(acc_fd, 1),
                "fontdiffuser_book_adaptive": round(acc_gen, 1),
            },
            "mean_cosine_similarity": {
                "font_nomnatong": round(cos_font_mean, 4),
                "fontdiffuser_global_stt": round(cos_fd_mean, 4),
                "fontdiffuser_book_adaptive": round(cos_gen_mean, 4),
            },
            "mean_margin_gap": {
                "font_nomnatong": round(margin_font_mean, 4),
                "fontdiffuser_global_stt": round(margin_fd_mean, 4),
                "fontdiffuser_book_adaptive": round(margin_gen_mean, 4),
            },
        },
        "details": records,
    }

    report_json = OUT_DIR / "ket_qua_p03.json"
    report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG (TN11-p03):")
    print(f"  Số ô kiểm tra (chữ hiếm, có nhãn người Borg.34): {len(records)} ô")
    print("-" * 70)
    print(f"  1. Phông in NomNaTong      : Top-1 = {acc_font:5.1f}% | Cosine = {cos_font_mean:.4f} | Margin = {margin_font_mean:+.4f}")
    print(f"  2. FontDiffuser phong cách chung: Top-1 = {acc_fd:5.1f}% | Cosine = {cos_fd_mean:.4f} | Margin = {margin_fd_mean:+.4f}")
    print(f"  3. FontDiffuser PHONG CÁCH SÁCH: Top-1 = {acc_gen:5.1f}% | Cosine = {cos_gen_mean:.4f} | Margin = {margin_gen_mean:+.4f}")
    print("=" * 70)
    print(f"Kết quả chi tiết đã ghi vào: {report_json}")


if __name__ == "__main__":
    main()
