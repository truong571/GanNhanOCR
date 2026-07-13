"""Nôm COLUMN(s) side-by-side: strip ảnh cột (trái) + bảng [crop] kim|qwen|nna→chốt
từng dòng (phải), tô ĐỎ chữ nào ≠ chốt. Mặc định xuất TẤT CẢ 9 cột (mỗi cột 1 trang PDF);
--col N để chỉ 1 cột (0-based).

Run:
  .venv/bin/python evaluation/tri_consensus/make_column_example.py \
      --book SachThanhTruyen2 --page page_0012 --qwen_dir qwen_cache_235b
Output: evaluation/tri_columns_<book>_<page>.pdf
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from run_tri_consensus import (is_cjk, align_map, align_to, load_qn_dict, load_similar,
                               load_syllables, make_decider)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NNA_CACHE = HERE / "nomnaocr_cache"
NOM = FontProperties(fname=str(REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"))
TIER_COL = {"CONSENSUS3": "#16a34a", "MAJORITY": "#d97706", "DICT": "#0891b2", "REVIEW": "#dc2626"}
RED, BLACK = "#dc2626", "#111827"


def build(book, page, qwen_dir):
    qn, sim, sylpg = load_qn_dict(), load_similar(), load_syllables(book)
    decide = make_decider(qn, sim, True, True)
    cdir = REPO / "prepared" / book / "detected"
    pdir = REPO / "prepared" / book / "pages_denoised"
    cache = json.loads((cdir / f"{page}_ocr_cache.json").read_text(encoding="utf-8"))
    kcols = [c for c in cache.get("columns", []) if c]
    kflat = [c["char"] for col in kcols for c in col]
    pg = Image.open(pdir / f"{page}.png").convert("L")
    smap = [""] * len(kflat)
    if page in sylpg:
        oseq, sseq = sylpg[page]
        smap = [s or "" for s in align_to(kflat, sseq, oseq)]
    qf = HERE / qwen_dir / f"{book}_{page}.json"
    qmap = align_map(kflat, [c for c in json.loads(qf.read_text(encoding="utf-8")).get("text", "") if is_cjk(c)]) if qf.exists() else [None] * len(kflat)
    nmap = []
    nf = NNA_CACHE / f"{book}_{page}.json"
    if nf.exists():
        nc = json.loads(nf.read_text(encoding="utf-8")).get("columns", [])
        if len(nc) == len(kcols):
            for col, ns in zip(kcols, nc):
                nmap += align_map([c["char"] for c in col], list(ns))
    if len(nmap) != len(kflat):
        nmap = [None] * len(kflat)
    per_col, base = [], 0
    for ci, col in enumerate(kcols):
        rows, err = [], [0, 0, 0]
        for j in range(len(col)):
            i = base + j
            kim, qw, nn, syl = col[j]["char"], qmap[i], nmap[i], smap[i]
            lbl, tier = decide(kim, qw, nn, syl)
            err[0] += kim != lbl; err[1] += bool(qw) and qw != lbl; err[2] += bool(nn) and nn != lbl
            rows.append(dict(kim=kim, qw=qw, nn=nn, syl=syl, lbl=lbl, tier=tier, bbox=col[j]["bbox"]))
        per_col.append((ci, rows, err)); base += len(col)
    return per_col, pg


def strip_crops(pg, rows):
    xs1 = [r["bbox"][0] for r in rows]; ys1 = [r["bbox"][1] for r in rows]
    xs2 = [r["bbox"][2] for r in rows]; ys2 = [r["bbox"][3] for r in rows]
    strip = np.asarray(pg.crop((min(xs1) - 4, min(ys1) - 4, max(xs2) + 4, max(ys2) + 4)))
    crops = []
    for r in rows:
        x1, y1, x2, y2 = (int(v) for v in r["bbox"])
        g = pg.crop((x1 - 3, y1 - 3, x2 + 3, y2 + 3)); g.thumbnail((48, 48))
        crops.append(np.asarray(g))
    return strip, crops


def p_intro(pdf, book, page):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.955, "Công nghệ: Đồng thuận 3 Mô hình OCR Hán-Nôm", ha="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.928, f"({book} · {page})", ha="center", fontsize=9, color="#6b7280")
    ax.text(0.06, 0.885, "Ý tưởng", fontsize=12, fontweight="bold", color="#111827")
    ax.text(0.08, 0.855, "Không tin 1 model đơn lẻ: cho 3 model ĐỘC LẬP cùng đọc → BỎ PHIẾU theo từng vị trí,\n"
            "dùng TỪ ĐIỂN Quốc-Ngữ (ngữ âm, do người dịch) làm trọng tài. Nhãn nào được nhiều tín\n"
            "hiệu độc lập xác nhận mới tin.", fontsize=10.5, va="top")
    ax.text(0.06, 0.775, "3 mô hình (3 tầm nhìn)", fontsize=12, fontweight="bold")
    rows = [("Kinhhannom", "#2563eb", "đọc CẢ TRANG → chữ + bbox 9 cột", "gốc TỌA ĐỘ + phiếu 1"),
            ("Qwen3-VL-235b", "#16a34a", "đọc CẢ TRANG → text", "phiếu ngữ cảnh cả trang (khác modality)"),
            ("NomNaOCR (fine-tuned)", "#d97706", "đọc TỪNG CỘT (theo bbox kim) → chữ", "phiếu thị giác cột")]
    yy = 0.745
    for nm, cl, what, role in rows:
        ax.text(0.075, yy, "●", color=cl, fontsize=13, va="center")
        ax.text(0.105, yy + 0.008, nm, fontsize=10.5, fontweight="bold", va="center")
        ax.text(0.105, yy - 0.016, f"{what}  —  {role}", fontsize=9, color="#374151", va="center")
        yy -= 0.055
    ax.text(0.06, 0.555, "Cách làm (4 bước)", fontsize=12, fontweight="bold")
    ax.text(0.08, 0.525, "1. Kim cắt khung 9 cột + phát hiện vị trí từng chữ (xương sống tọa độ).\n"
            "2. Căn chỉnh (Levenshtein) đưa Qwen + NomNaOCR VỀ ĐÚNG vị trí kim; gắn ÂM Quốc-Ngữ mỗi vị trí.\n"
            "3. VOTE {kim, qwen, nna} + 2 đòn bẩy: chuẩn hoá BIẾN THỂ (cùng chữ khác mã Unicode) và TỪ ĐIỂN\n"
            "    phân xử\n"
            "    (chữ ∈ dict(âm) thắng khi bất đồng).\n"
            "4. Xếp tầng tin cậy.", fontsize=10.5, va="top")
    ax.text(0.06, 0.375, "4 tầng nhãn", fontsize=12, fontweight="bold")
    tiers = [("CONSENSUS3", "#16a34a", "cả 3 model khớp → tin cậy CAO NHẤT"),
             ("MAJORITY", "#d97706", "2/3 khớp → chữ đa số"),
             ("DICT", "#0891b2", "kim đọc SAI, model độc lập + từ điển cứu chữ đúng (\"context gold\")"),
             ("REVIEW", "#dc2626", "cả 3 khác → chuyên gia soi thủ công")]
    yy = 0.345
    for nm, cl, ds in tiers:
        ax.add_artist(plt.matplotlib.patches.FancyBboxPatch((0.08, yy - 0.012), 0.16, 0.026,
                      boxstyle="round,pad=0.005", lw=0, facecolor=cl, transform=ax.transAxes))
        ax.text(0.16, yy + 0.001, nm, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        ax.text(0.27, yy + 0.001, ds, va="center", fontsize=9.5)
        yy -= 0.045
    ax.text(0.06, 0.135, "Nguyên tắc lõi", fontsize=12, fontweight="bold")
    ax.text(0.08, 0.108, "Tín hiệu độc lập THẬT SỰ là kênh Quốc-Ngữ (ngữ ÂM) — khác hẳn 3 model đọc-HÌNH —\n"
            "nên nó là trọng tài mạnh nhất chặn LỖI TƯƠNG QUAN (3 model cùng đọc mực có thể cùng sai).",
            fontsize=10, va="top", color="#374151")
    pdf.savefig(fig); plt.close(fig)


def render(pdf, book, page, ci, rows, err, strip, crops):
    N = len(rows)
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.965, f"{book} · {page} · CỘT {ci+1}/9 ({N} chữ)", ha="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.94, f"Số chữ SAI (≠ chốt):  kim {err[0]}/{N}   ·   qwen {err[1]}/{N}   ·   nna {err[2]}/{N}"
            "     (đỏ = sai)", ha="center", fontsize=10, color="#374151")
    for lab, xx in [("hình", 0.30), ("kim", 0.395), ("qwen", 0.495), ("nna", 0.60), ("→ chốt", 0.70), ("âm", 0.82)]:
        ax.text(xx, 0.905, lab, fontsize=8, color="#6b7280")
    top, bot = 0.89, 0.05
    sax = fig.add_axes([0.06, bot, 0.15, top - bot]); sax.imshow(strip, cmap="gray", aspect="auto"); sax.axis("off")
    rh = (top - bot) / N
    for k, r in enumerate(rows):
        y = top - (k + 0.5) * rh
        tc = TIER_COL[r["tier"]]
        ax.add_artist(AnnotationBbox(OffsetImage(crops[k], zoom=0.42, cmap="gray"), (0.315, y),
                                     xycoords=ax.transAxes, frameon=True, bboxprops=dict(edgecolor=tc, lw=1)))
        for m, (ch, xx) in enumerate([(r["kim"], 0.40), (r["qw"] or "·", 0.505), (r["nn"] or "·", 0.61), (r["lbl"], 0.72)]):
            color = tc if m == 3 else (RED if (ch != "·" and ch != r["lbl"]) else BLACK)
            ax.text(xx, y, ch, fontproperties=NOM, fontsize=13, va="center", ha="center", transform=ax.transAxes, color=color)
        ax.text(0.70, y, "→", fontsize=9, va="center", ha="right", transform=ax.transAxes, color="#9ca3af")
        ax.text(0.83, y, r["syl"] or "?", fontsize=8, va="center", color="#6b7280", transform=ax.transAxes)
    ax.text(0.5, 0.02, "Cột đọc TRÊN→DƯỚI. Strip trái = cột gốc scan; mỗi dòng: hình crop + 3 model → chốt "
            "(viền/chốt màu tầng; chữ ĐỎ = model đọc SAI so chốt).", ha="center", fontsize=7.5, style="italic", color="#6b7280")
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen2")
    ap.add_argument("--page", default="page_0012")
    ap.add_argument("--qwen_dir", default="qwen_cache_235b")
    ap.add_argument("--col", type=int, default=None, help="0-based; mặc định = TẤT CẢ cột")
    args = ap.parse_args()
    per_col, pg = build(args.book, args.page, args.qwen_dir)
    targets = [args.col] if args.col is not None else list(range(len(per_col)))
    outp = REPO / "evaluation" / f"tri_columns_{args.book}_{args.page}.pdf"
    tot_err = [0, 0, 0]; tot_n = 0
    with PdfPages(outp) as pdf:
        p_intro(pdf, args.book, args.page)
        for ci in targets:
            _, rows, err = per_col[ci]
            strip, crops = strip_crops(pg, rows)
            render(pdf, args.book, args.page, ci, rows, err, strip, crops)
            for i in range(3):
                tot_err[i] += err[i]
            tot_n += len(rows)
            print(f"  cột {ci+1}: {len(rows)} chữ | sai kim={err[0]} qwen={err[1]} nna={err[2]}")
    print(f"TỔNG {tot_n} chữ | sai kim={tot_err[0]} ({100*tot_err[0]/tot_n:.0f}%) "
          f"qwen={tot_err[1]} ({100*tot_err[1]/tot_n:.0f}%) nna={tot_err[2]} ({100*tot_err[2]/tot_n:.0f}%)")
    print(f"PDF -> {outp}")


if __name__ == "__main__":
    main()
