"""Báo cáo NHANH (PDF, có hình): sơ đồ luồng S3 + ảnh ví dụ + số liệu đã làm.

Sinh:
  results/s3_flow.png        — sơ đồ luồng so-khớp-ảnh S3 (matplotlib)
  results/montage_head.png   — vài ví dụ head∩bank (crop ↔ glyph đề xuất)
  BAO_CAO_NHANH_S3.pdf       — 3-4 trang tóm tắt công việc + hình

Run:
  .venv/bin/python evaluation/ver_new/make_quick_report.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle, ListFlowable, ListItem)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RES = HERE / "results"
OUT = REPO / "evaluation" / "BAO_CAO_NHANH_S3.pdf"

_DEJAVU = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
_NOM = REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"
plt.rcParams["font.family"] = "DejaVu Sans"


# --------------------------------------------------------- 1) sơ đồ luồng
def flow_png():
    fig, ax = plt.subplots(figsize=(8.6, 10.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14.2); ax.axis("off")
    C = {"in": "#e3f2fd", "enc": "#fff3e0", "bank": "#e8f5e9", "head": "#f3e5f5",
         "dec": "#eceff1", "gold": "#fff8e1", "silver": "#e0f2f1", "rev": "#ffebee"}

    def box(x, y, w, h, t, c, bold=False, fs=8.6):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.08", fc=c, ec="#444", lw=1.1))
        ax.text(x, y, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2, t=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#555"))
        if t:
            ax.text((x1 + x2) / 2 + 0.25, (y1 + y2) / 2, t, fontsize=6.8, color="#777", style="italic")

    ax.text(5, 13.9, "Luồng so-khớp-ảnh S3 (tín hiệu thị giác)", ha="center", fontsize=12,
            fontweight="bold", color="#7a1f1f")
    box(5, 13.1, 7.8, 0.7, "Crop chữ Nôm CHƯA xác nhận  (cột matched/anchored, ocr_char ∉ từ điển)", C["in"])
    arr(5, 12.75, 5, 12.25)
    box(5, 11.9, 7.4, 0.8, "tighten_box → NomEncoder (ResNet34 + ArcFace, 160px)\n→ embedding 256-D (L2-norm)", C["enc"])
    arr(5, 11.5, 5, 11.0)
    box(5, 10.65, 7.4, 0.7, "Tập ứng viên  R = {ocr_char} ∪ {cách đọc từ điển của âm tiết}", C["in"])
    # branch
    arr(5, 10.3, 3.0, 9.55); arr(5, 10.3, 7.0, 9.55)
    box(3.0, 9.1, 4.0, 0.95, "Bank tham chiếu / ứng viên:\ncrop thật → simfont → glyph FD\ncosine → isotonic P(match)", C["bank"])
    box(7.0, 9.1, 4.0, 0.95, "ArcFace HEAD (1591-lớp)\nlogit trên ứng viên\n→ head_top, head_margin", C["head"])
    arr(3.0, 8.6, 3.0, 8.05); arr(7.0, 8.6, 7.0, 8.05)
    box(3.0, 7.65, 4.0, 0.7, "top_char (bank) · p_match\nglyph-guard · reject?", C["bank"])
    box(7.0, 7.65, 4.0, 0.7, "head_agree = head_top\n== bank top_char ?", C["head"])
    arr(3.0, 7.3, 5, 6.75); arr(7.0, 7.3, 5, 6.75)
    box(5, 6.35, 6.4, 0.7, "ĐỒNG THUẬN  decide_label  (S1 ∩ S2 ∩ S3)", C["dec"], bold=True)
    arr(5, 6.0, 2.0, 5.05); arr(5, 6.0, 5.0, 5.05); arr(5, 6.0, 8.2, 5.05)
    box(2.0, 4.45, 3.4, 1.0, "GOLD\nocr_char ∈ từ điển\n(dict-direct / similar)", C["gold"], bold=True, fs=8)
    box(5.0, 4.0, 3.5, 1.9, "SILVER (thị giác)\n• s2∩s3: bank sửa OCR\n• s1∩s3: ngoài từ điển\n• head∩bank [#4]:\n  bank reject NHƯNG head\n  đồng thuận + margin≥0.3", C["silver"], bold=True, fs=7.6)
    box(8.2, 4.45, 3.0, 1.0, "REVIEW\nreject /\nkhông đồng thuận", C["rev"], bold=True, fs=8)
    ax.text(5, 2.6, "3 tín hiệu độc lập — S1 (OCR) ∩ S2 (từ điển) ∩ S3 (bank cosine + head). Chỉ gán khi đồng thuận.",
            ha="center", fontsize=8.2, color="#444", style="italic")
    ax.text(5, 2.15, "DINOv2 zero-shot đã loại (retrieval 0%). Ngưỡng: s3_calibration.json / conformal (LTT).",
            ha="center", fontsize=8.2, color="#444", style="italic")
    fig.tight_layout(); fig.savefig(RES / "s3_flow.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------- 2) montage ví dụ
def _afont(sz):
    for c in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
              "/Library/Fonts/Arial Unicode.ttf"]:
        if Path(c).exists():
            return ImageFont.truetype(c, sz)
    return ImageFont.load_default()


def montage_png(n=8):
    pkt = HERE / "eval_sample_head"
    rows = list(csv.DictReader(open(pkt / "verify.csv", encoding="utf-8")))[:n]
    g, pad = 110, 16
    cell_w, cell_h, cols = 300, 165, 2
    rn = (len(rows) + cols - 1) // cols
    W, Hh = pad + cols * cell_w, 40 + rn * cell_h
    im = Image.new("RGB", (W, Hh), "white"); d = ImageDraw.Draw(im)
    d.text((pad, 10), "Ví dụ tier head∩bank: crop (đỏ) → chữ đề xuất (xanh)", fill=(120, 0, 0), font=_afont(20))
    def fit(p, sz, boost=False):
        x = Image.open(p).convert("L")
        if boost:
            from PIL import ImageOps; x = ImageOps.autocontrast(x, 1)
        w, h = x.size; s = sz / max(w, h)
        x = x.resize((max(1, int(w * s)), max(1, int(h * s)))).convert("RGB")
        c = Image.new("RGB", (sz, sz), "white"); c.paste(x, ((sz - x.width) // 2, (sz - x.height) // 2)); return c
    fd_dir = REPO / "gannhanocr-fd"
    for k, r in enumerate(rows):
        cx = pad + (k % cols) * cell_w; cy = 40 + (k // cols) * cell_h
        cp = pkt / r["image"] if r["image"] else None
        if cp and cp.exists():
            im.paste(fit(cp, g, True), (cx, cy + 6))
        d.rectangle([cx, cy + 6, cx + g, cy + 6 + g], outline=(200, 0, 0), width=3)
        hx = f"{ord(r['label']):X}" if r["label"] else ""
        ref = fd_dir / hx[:2] / f"U+{hx}.png" if hx else None
        fx = cx + g + 20
        if ref and ref.exists():
            im.paste(fit(ref, g), (fx, cy + 6))
            d.rectangle([fx, cy + 6, fx + g, cy + 6 + g], outline=(0, 150, 0), width=2)
        d.text((cx, cy + g + 12), f"{r['label']} ({r['unicode']}) · âm {r['syllable']}", fill=(0, 0, 0), font=_afont(17))
    im.convert("P", palette=Image.ADAPTIVE, colors=64).save(RES / "montage_head.png")


# --------------------------------------------------------- 3) PDF
def build_pdf():
    pdfmetrics.registerFont(TTFont("DejaVu", str(_DEJAVU)))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_DEJAVU.parent / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Nom", str(_NOM)))
    def st(n, **kw):
        b = dict(fontName="DejaVu", fontSize=10, leading=14); b.update(kw); return ParagraphStyle(n, **b)
    H1 = st("h1", fontName="DejaVu-Bold", fontSize=15, textColor=colors.HexColor("#7a1f1f"), spaceBefore=10, spaceAfter=6)
    BODY = st("b", alignment=TA_JUSTIFY, spaceAfter=4)
    SMALL = st("s", fontSize=8.4, leading=11, textColor=colors.HexColor("#555"))
    def P(s, sty=BODY): return Paragraph(s, sty)

    def img(p, w_cm, cap=None):
        out = []
        if Path(p).exists():
            iw, ih = Image.open(p).size
            out.append(RLImage(str(p), width=w_cm * cm, height=w_cm * cm * ih / iw))
            if cap: out.append(Paragraph(cap, st("c", fontSize=8.2, alignment=TA_CENTER, textColor=colors.HexColor("#666"))))
            out.append(Spacer(1, 6))
        return out

    def tbl(header, rows, w):
        data = [[Paragraph(f"<b>{h}</b>", st("th", fontName="DejaVu-Bold", fontSize=8.6, textColor=colors.white)) for h in header]]
        data += [[Paragraph(str(c), st("td", fontSize=8.6, leading=11)) for c in r] for r in rows]
        t = Table(data, colWidths=w, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7a1f1f")),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbb")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f1f1")]),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 4)]))
        return t

    g1 = json.load(open(RES / "group1_rescue.json")) if (RES / "group1_rescue.json").exists() else {}
    cr = json.load(open(RES / "conformal_reject.json")) if (RES / "conformal_reject.json").exists() else {}
    vh = json.load(open(RES / "validate_head_consensus.json")) if (RES / "validate_head_consensus.json").exists() else {}

    story = [P("BÁO CÁO NHANH — Phần so khớp ảnh (S3) gán nhãn Hán-Nôm",
               st("t", fontName="DejaVu-Bold", fontSize=17, alignment=TA_CENTER, textColor=colors.HexColor("#7a1f1f"))),
             Spacer(1, 4),
             P("Tín hiệu thị giác S3: với mỗi crop chữ Nôm chưa được từ điển xác nhận, so khớp ảnh "
               "crop với glyph tham chiếu của các chữ ứng viên để gán nhãn (SILVER) hoặc loại (REVIEW).", SMALL),
             Spacer(1, 8)]
    story += [P("1. Sơ đồ luồng S3", H1)]
    story += img(RES / "s3_flow.png", 15, "Hình 1 — Luồng so-khớp-ảnh S3 (encoder + bank cosine + ArcFace head → đồng thuận).")
    story += [P("2. Số liệu chính (đo trên dữ liệu thật — 445 trang, 1591 lớp)", H1)]
    story += [tbl(["Hạng mục", "Kết quả"], [
        ["Encoder S3", "ResNet34 + ArcFace, 256-D; retrieval@1 ≈ 89%; rò rỉ xuyên-sách (LOBO) ≈ 1%"],
        ["DINOv2 zero-shot (đã loại)", "retrieval@1 = 0% (không phân biệt chữ Nôm)"],
        ["Reject có bảo đảm (conformal)",
         (f"precision ≥ {(1-cr.get('alpha',0.1))*100:.0f}% @ coverage {cr.get('coverage',0)*100:.0f}% "
          f"(LTT, tin cậy {(1-cr.get('delta',0.05))*100:.0f}%)") if cr.get("guaranteed") else "LTT — conformal_reject.py"],
        ["#4 head∩bank (gate ≥0.3)", f"+1.505 nhãn SILVER, precision GOLD-proxy {vh.get('precision',0)*100:.0f}% (cận trên)" if vh else "tier mới s3_head_bank_consensus"],
        ["Phần crop ảnh", "midpoint là tốt nhất; valley/detector CenterNet đều KHÔNG vượt (có số đo)"],
    ], [4.5 * cm, 11.5 * cm])]
    story += [Spacer(1, 6), P("3. Ví dụ trực quan (tier head∩bank — đã gate ≥0.3)", H1)]
    story += img(RES / "montage_head.png", 13, "Hình 2 — Crop thật (đỏ) ↔ glyph chữ đề xuất (xanh). Đa số khớp; soát tay xác nhận precision.")
    story += [P("4. Công việc đã làm", H1)]
    story += [ListFlowable([ListItem(P(x, st("bl", spaceAfter=2))) for x in [
        "<b>Thay DINOv2 → encoder Nôm tự-train</b> (ResNet34+ArcFace): retrieval 0% → ~89%.",
        "<b>Bank tham chiếu 3-tier</b> (crop thật / simfont / FD glyph) + calibration isotonic P(match).",
        "<b>#4 — thêm ArcFace head làm tín hiệu thị giác thứ 2</b>; tier `s3_head_bank_consensus` (gate head_margin≥0.3 → 97.6% proxy).",
        "<b>Reject có BẢO ĐẢM</b> (Learn-Then-Test): thay grid-search → 'precision ≥ 1−α với tin cậy 1−δ' (conformal_reject.py).",
        "<b>Đánh giá trung thực:</b> risk-coverage/AURC, leave-one-book-out (gap ~1%), packet soát người (eval_sample_head).",
        "<b>Phần crop:</b> chốt midpoint là tốt nhất; đã thử + loại valley, valley-guarded, detector CenterNet (đều có số đo).",
    ]], bulletType="bullet")]
    story += [Spacer(1, 6), P("Còn lại: soát tay ~200 mẫu (eval_sample_head/verify.csv) → precision SILVER thật + "
                              "ngưỡng bảo đảm; rồi adopt #4 vào dataset chính.", SMALL)]

    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.6 * cm, bottomMargin=1.6 * cm).build(story)
    print(f"PDF -> {OUT}  ({OUT.stat().st_size//1024} KB)")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    flow_png(); montage_png(); build_pdf()


if __name__ == "__main__":
    main()
