"""Xuất PDF báo cáo tiến độ (kèm hình chứng minh) để gửi thầy.

Trang 1: tóm tắt pipeline + biểu đồ phân bố tầng nhãn (đọc thật từ labels.csv).
Trang 2: chứng minh trực quan S3 — nhúng 2 ảnh montage [crop | glyph + cosine]
         do debug_s3.py sinh (chạy debug_s3.py --from-page trước nếu chưa có).

Chạy:
  .venv/bin/python evaluation/ver_new/make_report.py
  -> evaluation/ver_new/BAOCAO_GanNhanOCR.pdf
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATASET = HERE / "dataset_out"
DEBUG = HERE / "debug_out"
OUT = HERE / "BAOCAO_GanNhanOCR.pdf"

VI = "/System/Library/Fonts/Supplemental/Arial.ttf"
VIB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
NOM = str(REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf")

W, H = 1240, 1754           # A4 @ ~150 dpi
MARGIN = 90
INK = (20, 20, 20)
MUTE = (90, 90, 90)
TIER_COLOR = {"GOLD": (200, 160, 30), "SILVER": (130, 130, 140),
              "SYLLABLE": (45, 110, 190), "REVIEW": (210, 120, 40)}


def f(path, size):
    return ImageFont.truetype(path, size)


def draw_rich(d: ImageDraw.ImageDraw, xy, s: str, vi, nom, fill=INK):
    """Vẽ chuỗi trộn Việt + Nôm: mỗi ký tự CJK dùng font Nôm, còn lại font Việt."""
    x, y = xy
    for ch in s:
        fnt = nom if ord(ch) > 0x2E80 else vi
        d.text((x, y), ch, font=fnt, fill=fill)
        x += fnt.getlength(ch)
    return x


def wrap(d, s, vi, max_w):
    out, line = [], ""
    for word in s.split(" "):
        t = (line + " " + word).strip()
        if d.textlength(t, font=vi) <= max_w:
            line = t
        else:
            if line:
                out.append(line)
            line = word
    if line:
        out.append(line)
    return out


def bullets(d, x, y, items, vi, nom, lh=40, gap=14, max_w=None):
    max_w = max_w or (W - MARGIN - x - 20)
    for it in items:
        d.ellipse([x, y + 13, x + 9, y + 22], fill=(60, 60, 60))
        for i, ln in enumerate(wrap(d, it, vi, max_w)):
            draw_rich(d, (x + 26, y), ln, vi, nom)
            y += lh
        y += gap
    return y


def tier_counts():
    rows = list(csv.DictReader(open(DATASET / "labels.csv", encoding="utf-8")))
    c = Counter(r["tier"] for r in rows)
    return c, sum(c.values())


def bar_chart(d, x, y, counts, total, vi, vib, w=820, bar_h=46, gap=26):
    order = ["GOLD", "SILVER", "SYLLABLE", "REVIEW"]
    mx = max(counts.get(t, 0) for t in order) or 1
    label_w = 230
    for t in order:
        n = counts.get(t, 0)
        bw = int((w - label_w) * n / mx)
        cy = y
        d.text((x, cy + 10), t, font=vib, fill=TIER_COLOR[t])
        bx = x + label_w
        d.rectangle([bx, cy, bx + bw, cy + bar_h], fill=TIER_COLOR[t])
        pct = 100 * n / total if total else 0
        d.text((bx + bw + 14, cy + 10), f"{n:,}  ({pct:.0f}%)", font=vi, fill=INK)
        y += bar_h + gap
    d.text((x, y + 4), f"Tổng: {total:,} chữ Nôm có nhãn", font=vib, fill=INK)
    return y + 50


def page1():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    vi, vib, nom = f(VI, 27), f(VIB, 27), f(NOM, 27)
    h1, h2 = f(VIB, 40), f(VIB, 30)
    counts, total = tier_counts()

    y = MARGIN
    d.text((MARGIN, y), "BÁO CÁO TIẾN ĐỘ", font=h1, fill=INK); y += 52
    draw_rich(d, (MARGIN, y), "Gán nhãn dữ liệu OCR chữ Nôm — bộ song ngữ Nôm ↔ Quốc ngữ",
              f(VI, 26), f(NOM, 26), fill=MUTE); y += 50
    d.line([MARGIN, y, W - MARGIN, y], fill=(200, 200, 200), width=2); y += 30

    d.text((MARGIN, y), "1. Pipeline gán nhãn (3 bước, 1 lệnh ./run_pipeline.sh)",
           font=h2, fill=INK); y += 46
    y = bullets(d, MARGIN, y, [
        "Bước 0–1: tách PDF → crop khung → OCR SinoNom (9 cột/trang).",
        "Bước 2: căn chỉnh banded-DP neo từ điển (thay ghép theo vị trí) + đồng "
        "thuận 3 tín hiệu → labels.csv + 3 chuẩn quốc tế (HuggingFace / Frictionless / Croissant).",
    ], vi, nom)

    y += 6
    d.text((MARGIN, y), "2. Ba tín hiệu độc lập (1 chữ Nôm = 1 âm tiết QN)",
           font=h2, fill=INK); y += 46
    y = bullets(d, MARGIN, y, [
        "S1 = ký tự OCR SinoNom.  S2 = từ điển QN↔Nôm + chữ tương tự.  "
        "S3 = so khớp ảnh bằng encoder Nôm TỰ TRAIN (ResNet + ArcFace).",
    ], vi, nom)

    y += 6
    d.text((MARGIN, y), "3. Kết quả dataset — phân bố theo độ tin cậy", font=h2, fill=INK); y += 52
    y = bar_chart(d, MARGIN, y, counts, total, vi, vib)
    y = bullets(d, MARGIN, y + 6, [
        "GOLD: từ điển xác nhận (mức ký tự).   SILVER: S3 sửa thị giác.",
        "SYLLABLE: vay mượn nhất quán giữa các trang (mức âm tiết).   REVIEW: cần soát tay.",
        "Chia train/val/test theo nhóm (sách·trang·cột) → chống rò rỉ dữ liệu.",
    ], vi, nom)

    y += 6
    d.text((MARGIN, y), "4. Encoder Nôm thay DINOv2", font=h2, fill=INK); y += 46
    y = bullets(d, MARGIN, y, [
        "Chứng minh DINOv2 KHÔNG phân biệt được chữ Nôm (retrieval 0%).",
        "Tự train encoder Nôm (Kaggle P100) → retrieval 76,5% → kích hoạt tầng SILVER. "
        "DINOv2 đã tắt, giữ code để đối chiếu trong luận văn.",
    ], vi, nom)

    d.text((MARGIN, H - 70), "Trang 1/3", font=f(VI, 22), fill=MUTE)
    return img


def fit(im: Image.Image, max_w):
    if im.width <= max_w:
        return im
    r = max_w / im.width
    return im.resize((max_w, int(im.height * r)))


def page2():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    vi, vib, nom = f(VI, 27), f(VIB, 27), f(NOM, 27)
    h1, h2, cap = f(VIB, 40), f(VIB, 30), f(VI, 24)

    y = MARGIN
    d.text((MARGIN, y), "CHỨNG MINH TRỰC QUAN: S3 so ảnh thế nào", font=h1, fill=INK); y += 56
    y = bullets(d, MARGIN, y, [
        "Cơ chế: crop → embed thành vector 256 chiều → cosine với glyph "
        "FontDiffusion của từng ký tự ứng viên → xếp hạng, top-1 = nhãn S3.",
    ], vi, nom)
    y += 4

    cards = [
        (DEBUG / "s3_gold_yen2_page_0012_c01_000.png",
         "Ca GOLD: chữ ‘二’ — model xếp đúng nhãn thật ở hạng 1 (cosine 0,77), "
         "tách rõ khỏi chữ nhìn giống ‘三’ (0,70)."),
        (DEBUG / "s3_silver_yen2_page_0012_c03_025.png",
         "Ca SILVER: OCR đọc SAI ‘幹’ → S3 sửa lại ‘螉’ (cosine 0,712 — đúng bằng "
         "số ghi trong dataset), vượt cả OCR lẫn loạt chữ nhìn giống. Đây là giá "
         "trị của S3: sửa lỗi OCR bằng thị giác."),
    ]
    for path, capt in cards:
        d.text((MARGIN, y), "Ô trái = ảnh mộc bản; các ô sau = glyph ứng viên kèm cosine "
               "(viền lá = nhãn thật, cam = OCR).", font=cap, fill=MUTE); y += 38
        if path.exists():
            m = fit(Image.open(path).convert("RGB"), W - 2 * MARGIN)
            d.rectangle([MARGIN - 2, y - 2, MARGIN + m.width + 2, y + m.height + 2],
                        outline=(210, 210, 210), width=2)
            img.paste(m, (MARGIN, y)); y += m.height + 16
        else:
            d.text((MARGIN, y), f"[thiếu {path.name} — chạy debug_s3.py --from-page]",
                   font=cap, fill=(200, 60, 60)); y += 40
        for ln in wrap(d, capt, vi, W - 2 * MARGIN - 26):
            draw_rich(d, (MARGIN, y), ln, vi, nom); y += 38
        y += 26

    d.text((MARGIN, y + 4), "Kết luận: dataset ~82k chữ Nôm có nhãn nhiều mức tin cậy, "
           "có encoder Nôm riêng,", font=vib, fill=INK); y += 38
    draw_rich(d, (MARGIN, y), "pipeline 1 lệnh, đạt 3 chuẩn dataset quốc tế.", vib, f(NOM, 27))

    d.text((MARGIN, H - 70), "Trang 2/3", font=f(VI, 22), fill=MUTE)
    return img


def tier_card(d, img, x, y, w, h, tier, count, pct, title, meaning, ex, usable, uc):
    vi, vib = f(VI, 24), f(VIB, 25)
    color = TIER_COLOR[tier]
    band = 250
    d.rectangle([x, y, x + band, y + h], fill=color)
    d.rectangle([x, y, x + w, y + h], outline=(205, 205, 205), width=2)
    d.text((x + 22, y + 28), tier, font=f(VIB, 30), fill="white")
    d.text((x + 22, y + 78), f"{count:,}", font=f(VIB, 30), fill="white")
    d.text((x + 22, y + 120), f"({pct:.0f}%)", font=f(VI, 24), fill="white")
    bx, by = x + band + 26, y + 24
    d.text((bx, by), title, font=vib, fill=INK); by += 40
    for ln in wrap(d, meaning, vi, w - band - 52):
        draw_rich(d, (bx, by), ln, vi, f(NOM, 24)); by += 34
    by += 4
    ex_x = draw_rich(d, (bx, by), "Ví dụ: ", vi, f(NOM, 24), fill=MUTE)
    draw_rich(d, (ex_x, by), ex, vi, f(NOM, 24)); by += 40
    draw_rich(d, (bx, by), usable, f(VIB, 24), f(NOM, 24), fill=uc)


def page3():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    GREEN, RED = (30, 140, 50), (200, 70, 50)
    counts, total = tier_counts()

    y = MARGIN
    d.text((MARGIN, y), "4 TẦNG NHÃN — xếp theo độ tin cậy", font=f(VIB, 40), fill=INK); y += 54
    for ln in wrap(d, "Pipeline CHỈ gán nhãn khi đủ tin; phần còn lại gắn cờ để soát "
                   "tay — không đoán bừa. Đó là lý do có 4 mức.", f(VI, 25), W - 2 * MARGIN):
        d.text((MARGIN, y), ln, font=f(VI, 25), fill=MUTE); y += 36
    y += 16

    cw, ch, gap = W - 2 * MARGIN, 232, 22
    cards = [
        ("GOLD", "Từ điển xác nhận chữ",
         "Ký tự OCR trùng đúng cách đọc của âm trong từ điển → chốt luôn.",
         "OCR ‘二’, từ điển đọc “Nhị” → nhãn ‘二’.", "→ Có nhãn chữ, train được.", GREEN),
        ("SILVER", "Encoder Nôm (nhìn ảnh) sửa & xác nhận",
         "OCR/từ điển chưa chắc, nhưng S3 so ảnh chọn được chữ đủ tin (cosine ≥ 0,62).",
         "OCR đọc nhầm ‘幹’ → S3 nhìn ảnh sửa thành ‘螉’ (âm “ong”, cosine 0,71).",
         "→ Có nhãn chữ, train được.", GREEN),
        ("SYLLABLE", "Chưa chốt chữ, nhưng chắc ÂM ĐỌC",
         "Cùng ký tự OCR luôn cho cùng một âm Quốc ngữ qua ≥ 3 trang → tin âm, chưa pin mã chữ.",
         "OCR ‘要’ → chắc đọc “là”, chưa gắn mã chữ.",
         "→ Chỉ nhãn mức âm tiết (chưa train nhận dạng chữ).", (180, 120, 20)),
        ("REVIEW", "Không đủ tin → soát tay",
         "Ảnh xấu / cắt lệch cột / OCR sai mà S3 không cứu được → gắn cờ thay vì gán đại.",
         "Cột lệch: OCR ‘六’ (sáu) nhưng âm “bát” (tám) → mâu thuẫn.",
         "→ Chưa có nhãn.", RED),
    ]
    for tier, title, meaning, ex, usable, uc in cards:
        tier_card(d, img, MARGIN, y, cw, ch, tier, counts.get(tier, 0),
                  100 * counts.get(tier, 0) / total, title, meaning, ex, usable, uc)
        y += ch + gap

    y += 6
    box = [MARGIN, y, W - MARGIN, y + 92]
    d.rectangle(box, fill=(245, 246, 248), outline=(210, 210, 210), width=2)
    d.text((MARGIN + 20, y + 16), "Lưu ý: 23% REVIEW KHÔNG phải lỗi — là thiết kế ưu tiên",
           font=f(VIB, 24), fill=INK)
    d.text((MARGIN + 20, y + 50), "ĐỘ CHÍNH XÁC hơn độ phủ; cũng là nguyên liệu cho vòng "
           "cải tiến (self-training).", font=f(VI, 24), fill=INK)

    d.text((MARGIN, H - 70), "Trang 3/3", font=f(VI, 22), fill=MUTE)
    return img


def main():
    p1, p2, p3 = page1(), page2(), page3()
    # P-mode (FlateDecode) để tránh phụ thuộc libjpeg khi lưu PDF
    pages = [im.convert("RGB").quantize(colors=256) for im in (p1, p2, p3)]
    pages[0].save(OUT, save_all=True, append_images=pages[1:], resolution=150.0)
    print(f"Đã xuất: {OUT.relative_to(REPO)}  ({OUT.stat().st_size // 1024} KB, 3 trang)")


if __name__ == "__main__":
    main()
