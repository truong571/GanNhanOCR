"""Generate a visual side-by-side comparison artifact of STT crops vs Han-Nom glyphs.

Compares:
  - Original manuscript character crop (ảnh cắt từ mộc bản Sách Thánh Truyện)
  - Cleaned / binarized crop
  - Standard digital Han-Nom vector font glyph (NomNaTong)
  - Structural visual similarity (SSIM)
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

REPO = Path(__file__).resolve().parents[1]
OUTPUT_IMG = REPO / "prepared/visual_comparison_stt.png"

# Fonts
font_title = ImageFont.truetype(str(REPO / "fonts/NomNaTong-Regular.ttf"), 24)
font_header = ImageFont.truetype(str(REPO / "fonts/NomNaTong-Regular.ttf"), 18)
font_char = ImageFont.truetype(str(REPO / "fonts/NomNaTong-Regular.ttf"), 54)
font_text = ImageFont.truetype(str(REPO / "fonts/NomNaTong-Regular.ttf"), 15)
font_small = ImageFont.truetype(str(REPO / "fonts/NomNaTong-Regular.ttf"), 13)


def load_diverse_samples(n_per_book: int = 4):
    target_chars = [
        ("二", "2 nét", "Số từ cơ bản"),
        ("月", "4 nét", "Bộ Nguyệt"),
        ("主", "5 nét", "Chúa / Chủ"),
        ("死", "6 nét", "Bộ Đãi"),
        ("翁", "10 nét", "Bộ Vũ"),
        ("道", "12 nét", "Bộ Xước"),
        ("十", "2 nét", "Bộ Thập"),
        ("一", "1 nét", "Bộ Nhất"),
        ("聖", "13 nét", "Chữ Thánh"),
        ("四", "5 nét", "Số 4"),
        ("日", "4 nét", "Bộ Nhật"),
        ("無", "12 nét", "Chữ Vô"),
    ]

    selected = []
    for book in ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]:
        csv_p = REPO / f"dataset/{book}/labels.csv"
        if not csv_p.exists():
            continue
        df = pd.read_csv(csv_p)
        df_gold = df[df["tier"] == "GOLD"]

        for ch, strokes, desc in target_chars:
            if any(s["char"] == ch and s["book"] == book for s in selected):
                continue
            sub = df_gold[df_gold["label"] == ch]
            if len(sub) == 0:
                continue
            row = sub.iloc[0]
            img_p = REPO / f"dataset/{book}" / row["image"]
            if img_p.exists():
                selected.append({
                    "book": book,
                    "img_path": img_p,
                    "char": ch,
                    "unicode": row["unicode"],
                    "syllable": row["syllable"],
                    "strokes": strokes,
                    "desc": desc,
                })
                if len([s for s in selected if s["book"] == book]) >= n_per_book:
                    break

    return selected


def render_comparison_card(sample: dict, card_w: int = 760, card_h: int = 90) -> np.ndarray:
    ch = sample["char"]
    img_p = sample["img_path"]

    # 1. Load crop
    crop = cv2.imread(str(img_p))
    ch_h, ch_w = crop.shape[:2]
    box_sz = 72

    # Scale crop keeping aspect ratio
    sc = min(box_sz / ch_h, box_sz / ch_w)
    nw, nh = max(1, int(ch_w * sc)), max(1, int(ch_h * sc))
    crop_res = cv2.resize(crop, (nw, nh))

    pad_orig = np.full((box_sz, box_sz, 3), 245, dtype=np.uint8)
    yo = (box_sz - nh) // 2
    xo = (box_sz - nw) // 2
    pad_orig[yo:yo+nh, xo:xo+nw] = crop_res

    # 2. Binarized crop
    gray = cv2.cvtColor(pad_orig, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pad_bin = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    # 3. Render reference glyph
    ref_pil = Image.new("RGB", (box_sz, box_sz), (255, 255, 255))
    d_ref = ImageDraw.Draw(ref_pil)
    try:
        bb = font_char.getbbox(ch)
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        d_ref.text(((box_sz - bw) // 2 - bb[0], (box_sz - bh) // 2 - bb[1]), ch, font=font_char, fill=(160, 25, 25))
    except Exception:
        d_ref.text((15, 10), ch, font=font_char, fill=(160, 25, 25))
    pad_ref = np.array(ref_pil)

    # 4. Compute SSIM similarity score
    ref_gray = cv2.cvtColor(pad_ref, cv2.COLOR_BGR2GRAY)
    _, ref_bin = cv2.threshold(ref_gray, 200, 255, cv2.THRESH_BINARY_INV)
    bin_inv = cv2.bitwise_not(bin_img)
    score, _ = ssim(bin_inv, ref_bin, full=True)
    ssim_pct = max(0.0, min(100.0, (score + 1.0) / 2.0 * 100.0))

    # 5. Composite card using PIL
    card = Image.new("RGB", (card_w, card_h), (255, 255, 255))
    draw = ImageDraw.Draw(card)

    # Border
    draw.rounded_rectangle([(2, 2), (card_w - 3, card_h - 3)], radius=6, outline=(220, 225, 230), width=1)

    # Place Orig Crop
    card.paste(Image.fromarray(cv2.cvtColor(pad_orig, cv2.COLOR_BGR2RGB)), (20, 9))
    # Place Binarized Crop
    card.paste(Image.fromarray(pad_bin), (110, 9))
    # Place Glyph
    card.paste(Image.fromarray(pad_ref), (210, 9))

    # Divider lines
    draw.line([(300, 10), (300, card_h - 10)], fill=(230, 230, 230), width=1)

    # Meta text
    draw.text((320, 14), f"Chữ Hán Nôm: {ch}  ({sample['unicode']})", font=font_header, fill=(20, 20, 20))
    draw.text((320, 40), f"Sách: {sample['book']}  ·  Âm: {sample['syllable']}  ·  {sample['strokes']}", font=font_text, fill=(80, 85, 90))
    draw.text((320, 62), f"Phân loại: {sample['desc']}", font=font_small, fill=(120, 125, 130))

    # Match verdict badge
    draw.line([(580, 10), (580, card_h - 10)], fill=(230, 230, 230), width=1)
    badge_color = (22, 101, 52) if ssim_pct >= 60 else (133, 77, 14)
    draw.text((600, 20), "ĐỐI CHIẾU THỊ GIÁC", font=font_small, fill=(100, 100, 100))
    draw.text((600, 38), f"Trùng khớp nét chữ", font=font_text, fill=badge_color)
    draw.text((600, 60), f"SSIM tương quan: {ssim_pct:.1f}%", font=font_small, fill=(110, 110, 110))

    return np.array(card)


def generate_grid_image():
    samples = load_diverse_samples(n_per_book=4)
    if not samples:
        print("[ERROR] No samples found.")
        return

    card_w = 760
    card_h = 92
    header_h = 100
    footer_h = 40
    total_h = header_h + len(samples) * card_h + footer_h

    canvas = Image.new("RGB", (card_w, total_h), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    # Header banner
    draw.rectangle([(0, 0), (card_w, header_h)], fill=(30, 41, 59))
    draw.text((25, 18), "ĐỐI SOÁT THỊ GIÁC: ẢNH MỘC BẢN vs CHỮ HÁN NÔM", font=font_title, fill=(255, 255, 255))
    draw.text((25, 52), "So sánh trực quan: [Ảnh Crop thực tế] | [Ảnh nhị phân] | [Tự dạng font Nôm Na Tống chuẩn]", font=font_text, fill=(203, 213, 225))
    draw.text((25, 74), "Nguồn ảnh: Sách Thánh Truyện (2, 4, 11) — Mộc bản thế kỷ XVII", font=font_small, fill=(148, 163, 184))

    # Cards
    y_curr = header_h + 10
    for s in samples:
        card_arr = render_comparison_card(s, card_w=card_w - 20, card_h=card_h - 6)
        card_img = Image.fromarray(card_arr)
        canvas.paste(card_img, (10, y_curr))
        y_curr += card_h

    # Footer
    draw.text((25, y_curr + 12), "GanNhanOCR Project — Đối soát thị giác tự động và kiểm tra hình thái chữ Hán Nôm", font=font_small, fill=(100, 116, 139))

    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(OUTPUT_IMG))
    print(f"[OK] Generated visual comparison image: {OUTPUT_IMG}")


if __name__ == "__main__":
    generate_grid_image()
