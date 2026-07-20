"""Tạo PDF báo cáo kết quả test PP-OCRv5_server_det trên 1 trang."""
import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

plt.rcParams["font.family"] = "DejaVu Sans"


def show(ax, path):
    """imshow xử lý cả ảnh xám (2D) lẫn ảnh màu."""
    im = plt.imread(path)
    if im.ndim == 2:
        ax.imshow(im, cmap="gray")
    else:
        ax.imshow(im)

ORIG = "prepared/SachThanhTruyen11/pages/page_0010.png"
VIS = "scratch_ppocrv5_out/page_0010_det_vis.png"
OUT = "scratch_ppocrv5_out/PP-OCRv5_ket_qua.pdf"

res = json.load(open(glob.glob("scratch_ppocrv5_out/*_res.json")[0]))
polys = res["dt_polys"]
scores = res.get("dt_scores", [])
tall = sum(1 for p in polys if (np.array(p)[:, 1].max() - np.array(p)[:, 1].min()) > 500)
small = len(polys) - tall
avg = sum(scores) / len(scores) if scores else float("nan")

with PdfPages(OUT) as pdf:
    # ---- Trang 1: tổng quan + ảnh detect ----
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.suptitle("Kết quả detect văn bản — PP-OCRv5_server_det",
                 fontsize=16, fontweight="bold", y=0.975)

    ax_t = fig.add_axes([0.07, 0.80, 0.86, 0.13]); ax_t.axis("off")
    info = [
        ("Model", "PaddlePaddle/PP-OCRv5_server_det (inference 3.0)"),
        ("Ảnh test", os.path.basename(ORIG) + "  (2701×1691, chữ Nôm viết tay)"),
        ("Số box phát hiện", f"{len(polys)}   →   {tall} cột chữ (cao > 500px)  +  {small} box nhỏ"),
        ("Confidence TB", f"{avg:.3f}"),
        ("Tốc độ", "≈ 1.98 s / trang (CPU, Mac ARM)"),
        ("Ghi chú box nhỏ", "1 tiêu đề + 9 số chú thích cột (1–9) in sẵn trên trang gốc"),
    ]
    y = 1.0
    for k, v in info:
        ax_t.text(0.0, y, f"{k}:", fontsize=10, fontweight="bold", va="top")
        ax_t.text(0.30, y, v, fontsize=10, va="top")
        y -= 0.17

    ax_i = fig.add_axes([0.10, 0.05, 0.80, 0.72]); ax_i.axis("off")
    show(ax_i, VIS)
    ax_i.set_title("Ảnh trang + box detect (đỏ)", fontsize=11)
    pdf.savefig(fig); plt.close(fig)

    # ---- Trang 2: so sánh gốc vs detect + nhận xét ----
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("So sánh & nhận xét", fontsize=16, fontweight="bold", y=0.975)

    a1 = fig.add_axes([0.05, 0.40, 0.44, 0.52]); a1.axis("off")
    show(a1, ORIG); a1.set_title("Ảnh gốc", fontsize=11)
    a2 = fig.add_axes([0.51, 0.40, 0.44, 0.52]); a2.axis("off")
    show(a2, VIS); a2.set_title("Sau khi detect", fontsize=11)

    ax_n = fig.add_axes([0.07, 0.05, 0.86, 0.31]); ax_n.axis("off")
    notes = (
        "NHẬN XÉT\n\n"
        "• Tách gọn từng cột chữ Nôm viết tay dọc — dù model train chủ yếu trên chữ in ngang.\n"
        "  Biên box ôm sát, 9 cột thân bài không dính nhau.\n\n"
        "• Đây là detect ở MỨC CỘT / DÒNG, không phải mức KÝ TỰ như detector CenterNet (train_crop).\n"
        "  → Không thay thế trực tiếp detector ký tự; phù hợp làm bước tách cột trước khi segment chữ.\n\n"
        "• Bắt nhầm 9 số chú thích cột (1–9) in trên trang — dễ lọc theo vị trí / kích thước box.\n\n"
        "• Tốc độ ~2s/trang trên CPU; confidence trung bình 0.83."
    )
    ax_n.text(0.0, 1.0, notes, fontsize=10, va="top", linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)

print("OK ->", OUT)
