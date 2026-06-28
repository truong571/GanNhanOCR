"""PDF báo cáo nhanh, DỄ ĐỌC: HUẤN LUYỆN mô hình nhận dạng chữ Nôm từ kho nhãn tự tạo.
(Đã bỏ cách nói 'học trò vượt thầy'. Tập trung: mô hình là gì, dữ liệu gì — có dùng
gannhanocr-fd, train như nào, kết quả bước đầu.)

Sinh:
  results/fig_train_flow.png   — sơ đồ cách huấn luyện
  results/montage_nom.png      — ví dụ chữ Nôm đặc thù mô hình nhận đúng
  BAO_CAO_NHANH_HUAN_LUYEN_NHANDANG.pdf

Run:
  .venv/bin/python evaluation/ver_new/make_recognizer_report.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle, ListFlowable, ListItem)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RES = HERE / "results"
FD = REPO / "gannhanocr-fd"
OUT = REPO / "evaluation" / "BAO_CAO_NHANH_HUAN_LUYEN_NHANDANG.pdf"
OLD = REPO / "evaluation" / "BAO_CAO_NHANH_HOC_TRO_VUOT_THAY.pdf"
_DEJAVU = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
plt.rcParams["font.family"] = "DejaVu Sans"


def fd_path(ch):
    if not ch or len(ch) != 1:
        return None
    hx = f"{ord(ch):X}"
    p = FD / hx[:2] / f"U+{hx}.png"
    if p.exists():
        return p
    h = list(FD.rglob(f"U+{hx}.png"))
    return h[0] if h else None


def _afont(sz):
    for c in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"]:
        if Path(c).exists():
            return ImageFont.truetype(c, sz)
    return ImageFont.load_default()


# ---------------------------------------------------- 1) sơ đồ cách train
def fig_train_flow():
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, t, c, fs=9.2, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.08",
                     fc=c, ec="#444", lw=1.1))
        ax.text(x, y, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#555"))

    ax.text(5, 5.95, "Cách huấn luyện mô hình nhận dạng chữ Nôm", ha="center", fontsize=12.5,
            fontweight="bold", color="#7a1f1f")
    box(2.4, 4.9, 4.2, 0.95, "Ảnh chữ THẬT đã gán nhãn\n(GOLD + SILVER ≈ 62.000 ô)", "#e3f2fd")
    box(2.4, 3.55, 4.2, 0.95, "Glyph chữ MẪU sinh từ\ngannhanocr-fd (cho chữ ít/hiếm)", "#e8f5e9")
    arr(4.5, 4.9, 6.2, 4.3); arr(4.5, 3.55, 6.2, 4.1)
    box(8.0, 4.2, 3.4, 1.15, "Mạng ResNet34\n(khởi đầu từ 'bộ mắt'\nencoder đã học sẵn)", "#fff3e0", bold=True)
    arr(8.0, 3.6, 8.0, 2.75)
    box(8.0, 2.3, 3.7, 0.95, "Train trên Kaggle GPU\n~30 vòng (~1–2 giờ)", "#f3e5f5")
    arr(6.15, 2.3, 4.0, 2.3)
    box(2.2, 2.3, 3.7, 0.95, "Mô hình nhận chữ\n+ lưu lên HuggingFace", "#eceff1", bold=True)
    arr(2.2, 1.8, 2.2, 1.15)
    box(4.9, 0.7, 8.4, 0.8, "Kiểm độ chính xác trên phần dữ liệu ĐỂ RIÊNG (test tách-sách) → con số trung thực",
        "#fff8e1", fs=9)
    fig.tight_layout(); fig.savefig(RES / "fig_train_flow.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------- 2) montage chữ Nôm
def montage_nom(n=6):
    sys.path.insert(0, str(REPO))
    from evaluation.ver_new.visual_signal import VisualS3, _is_cjk
    D = HERE / "dataset_out"
    rows = list(csv.DictReader(open(D / "labels.csv", encoding="utf-8")))
    emittable = {r["ocr_char"] for r in rows if r["ocr_char"] and _is_cjk(r["ocr_char"])}
    rare = [r for r in rows if r["tier"] == "GOLD" and r["split"] == "test" and r["image"]
            and r["label"] and _is_cjk(r["label"]) and r["label"] not in emittable]
    enc = VisualS3(REPO, fd_dir="").enc
    picks, seen = [], set()
    for r in rare:
        if r["label"] in seen:
            continue
        emb = enc.embed_path(str(D / r["image"]))
        if emb is None:
            continue
        tk = enc.predict_topk(emb, 1)
        if tk and tk[0][0] == r["label"]:
            picks.append(r); seen.add(r["label"])
        if len(picks) >= n:
            break

    g = 120; cw, ch, cols = 300, 172, 3
    rn = (len(picks) + cols - 1) // cols
    W, Hh = 20 + cols * cw, 64 + rn * ch
    im = Image.new("RGB", (W, Hh), "white"); dd = ImageDraw.Draw(im)
    dd.text((20, 12), "Ví dụ: chữ Nôm đặc thù (chữ riêng của Nôm) — mô hình nhận ĐÚNG", fill=(120, 0, 0), font=_afont(20))
    dd.text((20, 38), "ẢNH THẬT (đỏ)   →   chữ mô hình đoán, ĐÚNG (xanh)", fill=(90, 90, 90), font=_afont(14))

    def fit(p, boost=False):
        x = Image.open(p).convert("L")
        if boost:
            x = ImageOps.autocontrast(x, 1)
        w, h = x.size; s = g / max(w, h)
        x = x.resize((max(1, int(w * s)), max(1, int(h * s)))).convert("RGB")
        c = Image.new("RGB", (g, g), "white"); c.paste(x, ((g - x.width) // 2, (g - x.height) // 2)); return c

    for k, r in enumerate(picks):
        cx = 20 + (k % cols) * cw; cy = 64 + (k // cols) * ch
        cp = D / r["image"]
        if cp.exists():
            im.paste(fit(cp, True), (cx, cy)); dd.rectangle([cx, cy, cx + g, cy + g], outline=(200, 0, 0), width=3)
        dd.text((cx + g + 2, cy + g // 2 - 12), "→", fill=(120, 120, 120), font=_afont(24))
        rg = fd_path(r["label"]); x2 = cx + g + 30
        if rg:
            im.paste(fit(rg), (x2, cy)); dd.rectangle([x2, cy, x2 + g, cy + g], outline=(0, 150, 0), width=3)
        dd.text((cx, cy + g + 6), f"âm '{r['syllable']}'", fill=(0, 0, 0), font=_afont(16))
    im.convert("P", palette=Image.ADAPTIVE, colors=64).save(RES / "montage_nom.png")
    return picks


# ---------------------------------------------------- 3) PDF
def build_pdf(d):
    pdfmetrics.registerFont(TTFont("DV", str(_DEJAVU)))
    pdfmetrics.registerFont(TTFont("DV-B", str(_DEJAVU.parent / "DejaVuSans-Bold.ttf")))
    def st(n, **k):
        b = dict(fontName="DV", fontSize=11, leading=15.5); b.update(k); return ParagraphStyle(n, **b)
    H1 = st("h1", fontName="DV-B", fontSize=14, textColor=colors.HexColor("#7a1f1f"), spaceBefore=9, spaceAfter=5)
    BODY = st("b", alignment=TA_JUSTIFY, spaceAfter=5)
    def P(s, sty=BODY): return Paragraph(s, sty)
    def img(p, w, cap=None):
        o = []
        if Path(p).exists():
            iw, ih = Image.open(p).size
            o.append(RLImage(str(p), width=w * cm, height=w * cm * ih / iw))
            if cap: o.append(Paragraph(cap, st("c", fontSize=8.5, alignment=TA_CENTER, textColor=colors.HexColor("#666"))))
            o.append(Spacer(1, 6))
        return o

    acc = d.get("student_acc", 0) * 100
    story = [P("Huấn luyện mô hình nhận dạng chữ Nôm (báo cáo nhanh)",
               st("t", fontName="DV-B", fontSize=17, alignment=TA_CENTER, textColor=colors.HexColor("#7a1f1f"))),
             Spacer(1, 8)]

    story += [P("1. Mô hình này làm gì", H1)]
    story += [P("Cho mô hình xem <b>một ô ảnh chứa một chữ Nôm</b>, nó <b>đoán đó là chữ nào</b> trong "
                "bộ ~1.590 chữ. Đây là bước biến kho nhãn tự tạo thành một <b>công cụ đọc chữ</b> dùng được.")]

    story += [P("2. Dữ liệu để huấn luyện", H1)]
    story += [P("Mô hình học từ <b>hai nguồn</b>:")]
    story += [ListFlowable([ListItem(P(x, st("bl", spaceAfter=3))) for x in [
        "<b>Ảnh chữ thật đã gán nhãn</b> — các ô crop GOLD + SILVER (~62.000 ô) cắt từ trang Nôm.",
        "<b>Glyph chữ mẫu sinh từ <font color='#1a6b1a'>gannhanocr-fd</font></b> — bộ chữ mẫu của mình. "
        "Thêm vào để <b>chữ nào cũng có mẫu học</b>, kể cả chữ hiếm/ít ảnh thật (đợt này thêm ~4.300 mẫu, "
        "chữ hiếm được nhân thêm). <i>Đúng vậy — cách này vẫn dùng bộ gannhanocr-fd.</i>",
    ]], bulletType="bullet")]

    story += [P("3. Cách huấn luyện", H1)]
    story += img(RES / "fig_train_flow.png", 15, "Hình 1 — Quy trình: gộp ảnh thật + glyph mẫu → ResNet34 (khởi đầu từ encoder) → train Kaggle → mô hình + đẩy HuggingFace.")
    story += [ListFlowable([ListItem(P(x, st("bl", spaceAfter=3))) for x in [
        "<b>Mô hình:</b> mạng nơ-ron <b>ResNet34</b> (chuyên nhìn ảnh).",
        "<b>Khởi đầu nhanh:</b> dùng lại 'bộ mắt' của encoder đã học sẵn đặc trưng chữ Nôm (warm-start) → đỡ phải học lại từ đầu.",
        "<b>Chạy ở đâu:</b> Kaggle GPU (T4), ~30 vòng, khoảng 1–2 giờ; xong tự lưu lên HuggingFace.",
        "<b>Chấm điểm trung thực:</b> để riêng một phần dữ liệu (theo sách khác) làm bài kiểm tra → không học vẹt.",
    ]], bulletType="bullet")]

    story += [P("4. Kết quả bước đầu", H1)]
    story += [P(f"Mới chạy thử 1–2 vòng trên máy, mô hình đã nhận đúng <b>~{acc:.0f}%</b> số chữ trên bài kiểm tra "
                f"(sẽ <b>cao hơn</b> khi train đủ 30 vòng trên Kaggle). Đáng chú ý: mô hình <b>đọc được cả những "
                f"chữ Nôm đặc thù</b> — chữ riêng của Nôm, không có trong bộ chữ Hán chuẩn — như ví dụ dưới.")]
    story += img(RES / "montage_nom.png", 15.5, "Ảnh thật (đỏ) → chữ mô hình đoán, đúng (xanh). Đây là các chữ Nôm riêng, khó.")

    story += [P("5. Việc tiếp theo", H1)]
    story += [ListFlowable([ListItem(P(x, st("bl", spaceAfter=3))) for x in [
        "Train đủ 30 vòng trên Kaggle → cập nhật con số chính xác.",
        "Dùng mô hình <b>đọc lại các chữ còn 'để xem lại' (REVIEW)</b> để cứu thêm nhãn.",
    ]], bulletType="bullet")]

    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm).build(story)
    if OLD.exists():
        OLD.unlink()
    print(f"PDF -> {OUT}  ({OUT.stat().st_size//1024} KB) | đã xoá bản cũ {OLD.name}")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    d = json.load(open(RES / "eval_teacher_vs_student.json", encoding="utf-8"))
    fig_train_flow()
    montage_nom()
    build_pdf(d)


if __name__ == "__main__":
    main()
