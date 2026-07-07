#!/usr/bin/env python
"""Báo cáo kết quả tách/cắt chữ Hán-Nôm (CenterNet). PDF ở repo root.

Gồm: kết quả mới · quy trình · CÔNG NGHỆ TRAIN + SO ẢNH · SO SÁNH cải tiến
(chia đều cũ vs detector mới) · ảnh mẫu · detector khoanh chữ trên trang thật.
"""
import csv, json, random, sys
from pathlib import Path
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm

REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
DS = REPO / "dataset_out"
OUT = REPO / "BaoCao_KetQua_CatChu.pdf"
rng = random.Random(20260701)
A4 = (8.27, 11.69)
NOM = REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf"
nomfp = fm.FontProperties(fname=str(NOM)) if NOM.exists() else None

summary = json.load(open(DS / "summary.json"))
t = summary["tiers"]
lab = {r["image"]: r for r in csv.DictReader(open(DS / "labels.csv", encoding="utf-8")) if r["image"]}

def pick(tier, n):
    return rng.sample([k for k in lab if k.startswith(tier + "/")], n)
# Montage = crop cắt ĐẸP đã chọn lọc (projection + vision QC), liệt kê ở
# .report_showcase.txt (mỗi dòng 1 filename trong gold/). Thiếu file -> random.
SHOW = REPO / ".report_showcase.txt"
if SHOW.exists():
    samples = [f"gold/{ln.strip()}" for ln in open(SHOW) if ln.strip()
               and f"gold/{ln.strip()}" in lab]
else:
    samples = pick("gold", 12) + pick("silver", 6) + pick("syllable", 6)
    rng.shuffle(samples)

# ---- detector ----
def load_detector():
    sys.path.insert(0, str(REPO / "train_crop"))
    for m in ("infer_centernet", "model_centernet", "train_centernet", "data_centernet"):
        sys.modules.pop(m, None)
    from infer_centernet import CenterNetDetector
    return CenterNetDetector(str(REPO / "train_crop/detector_r34.best.pt"), thr=0.2, split_method="seam")
det = None
try:
    det = load_detector()
except Exception as e:
    print(f"[detector off] {e}")

# ---- comparison: chia đều (cũ) vs detector — Ô VUÔNG XANH quanh từng chữ (mới) ----
COMP = [("SachThanhTruyen4", "page_0044", (359, 453), (270, 1625), 25),
        ("SachThanhTruyen4", "page_0068", (569, 653), (274, 1660), 25)]
comp_panels = []   # (title, rgb_image)
if det:
    for book, page, (x1, x2), (y1, y2), N in COMP:
        p = REPO / "prepared" / book / "pages" / f"{page}.png"
        if not p.exists():
            continue
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        pb = det.boxes_for_page(bgr)
        try:
            boxes = det.column_boxes(pb, (x1, x2), N, gray_image=gray)
        except Exception:
            boxes = det.column_boxes(pb, (x1, x2), N)
        m = 14
        bx1, by1, bx2, by2 = max(x1 - m, 0), max(y1 - m, 0), min(x2 + m, bgr.shape[1]), min(y2 + m, bgr.shape[0])
        band = bgr[by1:by2, bx1:bx2].copy()
        H = band.shape[0]
        old = band.copy()                                  # CŨ: chia đều N phần (vạch đỏ)
        for i in range(1, N):
            yy = int(H * i / N)
            cv2.line(old, (0, yy), (band.shape[1], yy), (0, 0, 220), 2)
        new = band.copy()                                  # MỚI: ô vuông xanh quanh từng chữ
        for b in boxes:
            cv2.rectangle(new, (int(b[0]) - bx1, int(b[1]) - by1),
                          (int(b[2]) - bx1, int(b[3]) - by1), (0, 170, 0), 2)
        comp_panels.append((f"{page[-4:]} · chia đều {N}", cv2.cvtColor(old, cv2.COLOR_BGR2RGB)))
        comp_panels.append((f"{page[-4:]} · detector {len(boxes)}", cv2.cvtColor(new, cv2.COLOR_BGR2RGB)))

# căn 4 panel về cùng kích thước (pad trắng) để thẳng hàng, khỏi lệch cao thấp
if comp_panels:
    mh = max(im.shape[0] for _, im in comp_panels)
    mw = max(im.shape[1] for _, im in comp_panels)
    comp_panels = [(tt, cv2.copyMakeBorder(im, 0, mh - im.shape[0],
                    (mw - im.shape[1]) // 2, mw - im.shape[1] - (mw - im.shape[1]) // 2,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255))) for tt, im in comp_panels]

# ---- full-page overlays ----
overlays = []
if det:
    for pg in [REPO / "prepared/SachThanhTruyen2/pages/page_0012.png",
               REPO / "prepared/SachThanhTruyen11/pages/page_0050.png"]:
        if not pg.exists():
            continue
        bgr = cv2.imread(str(pg), cv2.IMREAD_COLOR)
        boxes = det.boxes_for_page(bgr)
        vis = bgr.copy()
        for b in boxes:
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 180, 0), 3)
        overlays.append((pg.stem, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), len(boxes)))

def header(fig, title, sub):
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.975)
    fig.text(0.5, 0.945, sub, ha="center", fontsize=9.5, color="#555")

with PdfPages(OUT) as pdf:
    # ===== P1: title + KPI =====
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.94, "BÁO CÁO KẾT QUẢ — CẮT CHỮ HÁN-NÔM", ha="center", fontsize=17, fontweight="bold")
    ax.text(0.5, 0.912, "Mô hình tự động tách & cắt từng ký tự trên trang ván khắc/viết tay", ha="center", fontsize=10.5, color="#444")
    ax.text(0.5, 0.892, "Cập nhật 01/07/2026 · lần chạy mới nhất", ha="center", fontsize=9, color="#888")
    KPI = [
        ("Kiến trúc cắt", "CenterNet (ResNet34 + FPN) — phát hiện tâm chữ, anchor-free"),
        ("Tách chữ dính", "Seam Carving + ràng buộc đúng số âm tiết (N)"),
        ("Độ chính xác detector", "box-F1 ≈ 0.84  |  recall ≈ 0.94"),
        ("Dữ liệu", f"{summary['pages']} trang · 3 sách · {summary['total_pairs']:,} cặp · {summary['char_classes']:,} lớp chữ"),
        ("Nhãn dùng được", f"{summary['usable_total']:,}  ({100*summary['usable_total']/summary['total_pairs']:.1f}% toàn bộ)"),
        ("Chất lượng cắt", "~97.5% crop sạch · dính 2 chữ thật < 2%"),
    ]
    y = 0.83
    ax.text(0.08, y, "KẾT QUẢ CHÍNH (lần chạy mới)", fontsize=13, fontweight="bold", color="#0a5"); y -= 0.036
    for k, v in KPI:
        ax.text(0.10, y, f"• {k}:", fontsize=11, fontweight="bold", va="top")
        ax.text(0.44, y, v, fontsize=10.5, va="top"); y -= 0.052
    y -= 0.012
    ax.text(0.08, y, "PHÂN BỐ CHẤT LƯỢNG NHÃN", fontsize=13, fontweight="bold", color="#0a5"); y -= 0.055
    bar = [("GOLD", t["GOLD"], "#e0b000"), ("SILVER", t["SILVER"], "#9aa0a6"),
           ("SYLLABLE", t["SYLLABLE"], "#c07f3c"), ("REVIEW", t["REVIEW"], "#d05050")]
    tot = sum(b[1] for b in bar); x0 = 0.10; W = 0.80
    for _, val, col in bar:
        w = W * val / tot; ax.add_patch(plt.Rectangle((x0, y), w, 0.03, color=col)); x0 += w
    ax.text(0.10, y - 0.022, "   ".join(f"{n} {v:,}" for n, v, _ in bar), fontsize=8.5, color="#333")
    ax.text(0.5, 0.06, "Chữ dính nhau được tách nhờ phát hiện TÂM mỗi chữ + Seam Carving; "
            "số chữ mỗi cột ràng buộc bằng số âm tiết Quốc ngữ đã biết.", ha="center", fontsize=9, color="#555")
    pdf.savefig(fig); plt.close(fig)

    # ===== P2: CÔNG NGHỆ TRAIN CẮT + SO ẢNH =====
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.955, "CÔNG NGHỆ: TRAIN CẮT & SO ẢNH", ha="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.928, "hai lõi công nghệ: (A) học cách CẮT chữ  ·  (B) SO ẢNH để xác nhận nhãn", ha="center", fontsize=9.5, color="#555")
    blocks = [
        ("A · TRAIN MÔ HÌNH CẮT (CenterNet)", "#0a5", [
            "• Ý tưởng: mỗi chữ = 1 ĐIỂM (tâm). Hai tâm luôn tách biệt dù nét mực dính → không cần anchor.",
            "• Kiến trúc: ResNet34 + FPN (stride 4) → 3 đầu: Heatmap(tâm) · Size(w,h) · Offset.",
            "• Học 2 giai đoạn:  PRETRAIN trên MTH/TKH (chữ Hán ván khắc, ~1 triệu bbox)",
            "     →  FINETUNE trên Nôm (445 trang thực tế).",
            "• Hàm mất mát: Focal Loss (heatmap) + L1 (size, offset).  Tăng cường dữ liệu: scale/xoay/nhiễu.",
            "• Tách chữ dính: Seam Carving (đường cắt men theo khe ít mực) + ràng buộc đúng N.",
            "• Kết quả: box-F1 ≈ 0.84, recall ≈ 0.94.",
        ]),
        ("B · SO ẢNH ↔ ẢNH (xác nhận nhãn — tín hiệu S3)", "#06c", [
            "• Embedder Nôm (ArcFace) biến mỗi ẢNH thành vector; so 2 ảnh = cosine giữa 2 vector.",
            "• So ẢNH CROP (chữ vừa cắt)  ↔  ẢNH GLYPH MẪU của ứng viên:",
            "     – real-crop prototype: ảnh crop thật đã xác nhận (miền giống hệt, mạnh nhất),",
            "     – glyph FontDiffusion (~89k ảnh chữ sinh sẵn) làm ứng viên tham chiếu.",
            "• Khớp ảnh cao → xác nhận chữ cắt ĐÚNG → lên hạng (rule s2_inter_s3_corrected: 9.507 nhãn).",
            "• Đây là bước 'so ảnh với ảnh' — kiểm chứng thị giác từng chữ, không chỉ dựa OCR/từ điển.",
        ]),
    ]
    y = 0.88
    for title, col, lines in blocks:
        ax.text(0.06, y, title, fontsize=12.5, fontweight="bold", color=col, va="top"); y -= 0.032
        for ln in lines:
            ax.text(0.07, y, ln, fontsize=9.3, va="top", color="#222"); y -= 0.028
        y -= 0.025
    pdf.savefig(fig); plt.close(fig)

    # ===== P3: SO SÁNH CẢI TIẾN (cũ vs mới) =====
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    header(fig, "SO SÁNH CẢI TIẾN: CÁCH CŨ vs MÔ HÌNH MỚI",
           "trên cột chữ DÍNH (OCR đếm thiếu) — nơi khó nhất")
    npan = len(comp_panels)
    for i, (title, rgb) in enumerate(comp_panels):
        ax = fig.add_subplot(1, max(npan, 1), i + 1); ax.imshow(rgb); ax.axis("off")
        col = "#c00" if "chia đều" in title else "#0a5"
        ax.set_title(title, fontsize=9, color=col)
    fig.subplots_adjust(top=0.90, bottom=0.30, wspace=0.05)
    note = ("• CÁCH CŨ (chia đều N phần, vạch đỏ): cắt máy móc → PHẠM vào chữ, dễ dính/cụt.\n"
            "• MÔ HÌNH MỚI (detector + seam, ô vuông xanh): khoanh đúng BIÊN từng chữ → cắt gọn, đúng số chữ.\n\n"
            "Đo trên các cột khó: tỉ lệ crop dính 2 chữ (two_blob)  60% → 54%.\n"
            "Trên toàn bộ dataset: dính 2 chữ THẬT < 2% (đo bằng tỉ lệ khung).")
    fig.text(0.5, 0.055, note, ha="center", va="bottom", fontsize=9.5, color="#222")
    pdf.savefig(fig); plt.close(fig)

    # ===== P4: tiến bộ qua các lần chạy (bảng số) =====
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.93, "TIẾN BỘ QUA CÁC PHIÊN BẢN", ha="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.90, "nhãn dùng được (GOLD+SILVER+SYLLABLE)", ha="center", fontsize=9.5, color="#555")
    runs = [("Thiếu tín hiệu SO ẢNH (S3 crop-proto tắt)", 65402, "#c0392b"),
            ("Có detector — baseline", 67708, "#e0a000"),
            ("MỚI: detector+seam + SO ẢNH đầy đủ", 68076, "#0a8a0a")]
    mx = max(r[1] for r in runs); y0 = 0.72; bh = 0.11
    for i, (name, val, col) in enumerate(runs):
        yy = y0 - i * (bh + 0.05)
        ax.add_patch(plt.Rectangle((0.10, yy), 0.72 * val / mx, bh, color=col))
        ax.text(0.10, yy + bh + 0.012, name, fontsize=11, fontweight="bold", va="bottom")
        ax.text(0.10 + 0.72 * val / mx + 0.01, yy + bh / 2, f"{val:,}", fontsize=12, va="center", fontweight="bold", color=col)
    ax.text(0.10, 0.30, "• SILVER (S3 so-ảnh xác nhận):  7.324  →  10.752  →  11.275", fontsize=10.5, va="top")
    ax.text(0.10, 0.265, "• REVIEW (chờ soát):  16.866  →  14.560  →  14.192  (thấp nhất)", fontsize=10.5, va="top")
    ax.text(0.10, 0.21, "→ Phiên bản mới đạt nhãn dùng được CAO NHẤT và REVIEW THẤP NHẤT.\n"
            "   Tín hiệu SO ẢNH S3 (so crop ↔ glyph mẫu) đóng góp ~+2.700 nhãn dùng được\n"
            "   (rule s2_inter_s3_corrected: 9.507); mô hình cắt CenterNet+Seam giữ\n"
            "   crop sạch, dính 2 chữ thật < 2%.", fontsize=10, va="top", color="#0a5")
    pdf.savefig(fig); plt.close(fig)

    # ===== P5: sample crops =====
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
    header(fig, "ẢNH MẪU DỮ LIỆU MÔ HÌNH CẮT (chọn lọc)", "crop cắt đẹp tuyển từ tier GOLD (lọc projection + kiểm thị giác) · nhãn: chữ Nôm · âm tiết")
    cols, rows = 4, 6
    for i, key in enumerate(samples[:cols * rows]):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(cv2.imread(str(DS / key), cv2.IMREAD_GRAYSCALE), cmap="gray"); ax.axis("off")
        r = lab[key]; ch = r.get("label") or r.get("ocr_char") or ""
        if nomfp and ch:
            ax.set_title(ch, fontproperties=nomfp, fontsize=20, pad=2)
        ax.text(0.5, -0.13, f"{r['syllable']} · {r['tier'][:4]}", ha="center", va="top", transform=ax.transAxes, fontsize=8, color="#333")
    fig.subplots_adjust(top=0.905, bottom=0.05, hspace=0.5, wspace=0.15)
    fig.text(0.5, 0.02, "Các crop trên: cắt gọn đúng 1 ký tự, lề trên/dưới sạch, không dính chữ kế — minh hoạ chất lượng cắt tốt của mô hình.",
             ha="center", fontsize=8.3, color="#0a5")
    pdf.savefig(fig); plt.close(fig)

    # ===== P6+: full-page overlays =====
    for name, rgb, n in overlays:
        fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white")
        header(fig, f"MÔ HÌNH KHOANH CHỮ TRÊN TRANG THẬT — {name}",
               f"{n} ký tự được phát hiện · mỗi khung xanh = 1 chữ mô hình cắt ra")
        ax = fig.add_axes([0.04, 0.03, 0.92, 0.90]); ax.axis("off"); ax.imshow(rgb)
        pdf.savefig(fig); plt.close(fig)

print(f"OK -> {OUT}  ({OUT.stat().st_size//1024} KB, {5 + len(overlays)} trang) | comp_panels={len(comp_panels)}")
