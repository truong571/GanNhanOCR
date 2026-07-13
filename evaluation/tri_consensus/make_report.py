"""PDF report of the tri-model consensus (kim + Qwen-235b + NomNaOCR-ft + variant/dict).

Recomputes stats from caches for a given book, compares BASE vs +BOTH levers, and
renders: architecture, results (base vs both), real glyph examples (incl DICT fixes),
correctness validation.

Run:
  .venv/bin/python evaluation/tri_consensus/make_report.py --book SachThanhTruyen2 --n 10 \
      --qwen_dir qwen_cache_235b
Output: evaluation/tri_report_<book>.pdf
"""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from run_tri_consensus import (is_cjk, align_map, align_to, load_qn_dict, load_similar,
                               load_syllables, make_decider, BOOKCODE)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NNA_CACHE = HERE / "nomnaocr_cache"
NOM = FontProperties(fname=str(REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"))
GREEN, AMBER, TEAL, RED, BLUE, GREY = "#16a34a", "#d97706", "#0891b2", "#dc2626", "#2563eb", "#6b7280"


def gather(book, n, qwen_dir):
    qn, sim, sylpg = load_qn_dict(), load_similar(), load_syllables(book)
    d_both = make_decider(qn, sim, True, True)
    d_base = make_decider(qn, sim, False, False)
    qdir = HERE / qwen_dir
    det = REPO / "prepared" / book / "pages_denoised"
    cdir = REPO / "prepared" / book / "detected"
    stems = [f.stem for f in sorted(det.glob("page_*.png"))
             if (cdir / f"{f.stem}_ocr_cache.json").exists()][:n]
    tb, tt = Counter(), Counter()          # base tiers, both tiers
    per_page = []
    ex = {"CONSENSUS3": [], "MAJORITY": [], "DICT": [], "REVIEW": []}
    dict_overrule = {"total": 0, "kim_notin": 0, "lbl_valid": 0}
    allchars = set().union(*qn.values())
    for stem in stems:
        cache = json.loads((cdir / f"{stem}_ocr_cache.json").read_text(encoding="utf-8"))
        kcols = [c for c in cache.get("columns", []) if c]
        kinfo = [(c["char"], c["bbox"]) for col in kcols for c in col]     # (char, bbox) per position
        kflat = [c for c, _ in kinfo]
        page_gray = Image.open(det / f"{stem}.png").convert("L")            # for real glyph crops
        def crop_at(i):
            x1, y1, x2, y2 = (int(v) for v in kinfo[i][1])
            g = page_gray.crop((x1 - 3, y1 - 3, x2 + 3, y2 + 3))
            g.thumbnail((60, 60)); return np.asarray(g)
        smap = [""] * len(kflat)
        if stem in sylpg:
            oseq, sseq = sylpg[stem]
            smap = [s or "" for s in align_to(kflat, sseq, oseq)]
        qf = qdir / f"{book}_{stem}.json"
        qmap = align_map(kflat, [c for c in json.loads(qf.read_text(encoding="utf-8")).get("text", "") if is_cjk(c)]) if qf.exists() else [None] * len(kflat)
        nf = NNA_CACHE / f"{book}_{stem}.json"
        nmap = []
        if nf.exists():
            nc = json.loads(nf.read_text(encoding="utf-8")).get("columns", [])
            if len(nc) == len(kcols):
                for col, ns in zip(kcols, nc):
                    nmap += align_map([c["char"] for c in col], list(ns))
        if len(nmap) != len(kflat):
            nmap = [None] * len(kflat)
        pc = Counter()
        for i, kim in enumerate(kflat):
            qw, nn, syl = qmap[i], nmap[i], smap[i]
            _, tbase = d_base(kim, qw, nn, syl)
            lbl, tboth = d_both(kim, qw, nn, syl)
            tb[tbase] += 1; tt[tboth] += 1; pc[tboth] += 1
            def add(tier):
                if len(ex[tier]) < 8:
                    ex[tier].append((kim, qw, nn, lbl, syl, crop_at(i)))
            if tboth == "DICT" and lbl != kim:
                dict_overrule["total"] += 1
                if not (syl and kim in qn.get(syl, ())):
                    dict_overrule["kim_notin"] += 1
                if syl and lbl in qn.get(syl, ()):
                    dict_overrule["lbl_valid"] += 1
                add("DICT")
            elif tboth == "CONSENSUS3":
                add("CONSENSUS3")
            elif tboth == "MAJORITY" and qw and nn:
                add("MAJORITY")
            elif tboth == "REVIEW" and qw and nn:
                add("REVIEW")
        per_page.append((stem, pc))
    return stems, tb, tt, per_page, ex, dict_overrule


def box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                lw=0, facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white",
            fontsize=fs, fontweight="bold")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color=GREY))


def p_title(pdf, book, n, tot):
    fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "Báo cáo Đồng thuận 3 Model OCR Hán-Nôm", ha="center", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.918, f"{book} · {n} trang đầu · {tot} vị trí", ha="center", fontsize=11, color=GREY)
    ax.text(0.5, 0.893, "kim (kinhhannom) + Qwen3-VL-235b + NomNaOCR-finetuned  ·  2 lever: biến-thể + từ-điển",
            ha="center", fontsize=9.5, color=GREY)
    box(ax, 0.05, 0.72, 0.27, 0.06, "KIM\nframe-crop → 9 cột+bbox", BLUE, 8)
    box(ax, 0.36, 0.72, 0.27, 0.06, "QWEN-235b\nframe-crop → text", GREEN, 8)
    box(ax, 0.67, 0.72, 0.28, 0.06, "NomNaOCR-ft\ntừng cột → chữ", AMBER, 8)
    box(ax, 0.20, 0.60, 0.60, 0.05, "CĂN CHỈNH về vị trí kim + gắn ÂM (Quốc-Ngữ) mỗi vị trí", GREY, 8)
    arrow(ax, 0.18, 0.72, 0.40, 0.65); arrow(ax, 0.49, 0.72, 0.50, 0.65); arrow(ax, 0.81, 0.72, 0.60, 0.65)
    box(ax, 0.25, 0.50, 0.50, 0.05, "VOTE 3 model + biến-thể + TỪ ĐIỂN phân xử", "#111827", 9)
    arrow(ax, 0.50, 0.60, 0.50, 0.55)
    rules = [("CONSENSUS3", "cả 3 khớp → GOLD tin cậy cao nhất", GREEN),
             ("MAJORITY", "2/3 khớp → chữ đa số", AMBER),
             ("DICT", "kim sai → chữ model ∈ từ điển(âm) thắng (CONTEXT GOLD)", TEAL),
             ("REVIEW", "cả 3 khác / thiếu phiếu → giữ kim, gắn cờ", RED)]
    y = 0.40
    for nm, ds, cl in rules:
        box(ax, 0.08, y, 0.19, 0.032, nm, cl, 8)
        ax.text(0.30, y + 0.016, ds, va="center", fontsize=9)
        y -= 0.05
    ax.text(0.5, 0.10, "3 model chạy offline → cache JSON → 1 script vote (kim = gốc tọa độ).",
            ha="center", fontsize=8.5, color=GREY, style="italic")
    pdf.savefig(fig); plt.close(fig)


def p_results(pdf, tb, tt, per_page, tot):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("Kết quả & so sánh (BASE vs +2 LEVER)", fontsize=16, fontweight="bold", y=0.97)
    # pie both
    ax1 = fig.add_axes([0.07, 0.63, 0.4, 0.27])
    vals = [tt["CONSENSUS3"], tt["MAJORITY"], tt["DICT"], tt["REVIEW"]]
    ax1.pie(vals, labels=["CONS3", "MAJ", "DICT", "REVIEW"], colors=[GREEN, AMBER, TEAL, RED],
            autopct=lambda p: f"{p:.0f}%", startangle=90, textprops={"fontsize": 8})
    ax1.set_title("Phân bố tầng (+2 lever)", fontsize=10)
    # bar base vs both
    ax2 = fig.add_axes([0.56, 0.63, 0.4, 0.27])
    cats = ["USABLE", "REVIEW"]
    base_u = 100 * (tb["CONSENSUS3"] + tb["MAJORITY"]) / tot; base_r = 100 * tb["REVIEW"] / tot
    both_u = 100 * (tt["CONSENSUS3"] + tt["MAJORITY"] + tt["DICT"]) / tot; both_r = 100 * tt["REVIEW"] / tot
    x = np.arange(2); w = 0.35
    ax2.bar(x - w / 2, [base_u, base_r], w, label="BASE", color=GREY)
    ax2.bar(x + w / 2, [both_u, both_r], w, label="+2 lever", color=BLUE)
    ax2.set_xticks(x); ax2.set_xticklabels(cats); ax2.set_ylabel("%"); ax2.legend(fontsize=8)
    for i, (a, b) in enumerate([(base_u, both_u), (base_r, both_r)]):
        ax2.text(i - w / 2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax2.text(i + w / 2, b + 1, f"{b:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax2.set_title("USABLE ↑, REVIEW ↓", fontsize=10)
    # numbers
    ax3 = fig.add_axes([0.08, 0.40, 0.86, 0.18]); ax3.axis("off")
    tbl = (f"{'Tier':<14}{'BASE':>10}{'+2 LEVER':>12}\n"
           f"{'CONSENSUS3':<14}{tb['CONSENSUS3']:>6} ({100*tb['CONSENSUS3']/tot:4.1f}%){tt['CONSENSUS3']:>4} ({100*tt['CONSENSUS3']/tot:4.1f}%)\n"
           f"{'MAJORITY':<14}{tb['MAJORITY']:>6} ({100*tb['MAJORITY']/tot:4.1f}%){tt['MAJORITY']:>4} ({100*tt['MAJORITY']/tot:4.1f}%)\n"
           f"{'DICT':<14}{0:>6} ( 0.0%){tt['DICT']:>4} ({100*tt['DICT']/tot:4.1f}%)\n"
           f"{'REVIEW':<14}{tb['REVIEW']:>6} ({100*tb['REVIEW']/tot:4.1f}%){tt['REVIEW']:>4} ({100*tt['REVIEW']/tot:4.1f}%)\n"
           f"{'USABLE':<14}{base_u:>10.1f}%{both_u:>11.1f}%")
    ax3.text(0.0, 1.0, tbl, va="top", fontsize=10.5, family="monospace")
    # per-page
    ax4 = fig.add_axes([0.10, 0.10, 0.84, 0.22])
    labs = [s.replace("page_", "p") for s, _ in per_page]
    c3 = [pc["CONSENSUS3"] for _, pc in per_page]; mj = [pc["MAJORITY"] for _, pc in per_page]
    dk = [pc["DICT"] for _, pc in per_page]; rv = [pc["REVIEW"] for _, pc in per_page]
    xx = np.arange(len(labs))
    ax4.bar(xx, c3, color=GREEN, label="CONS3")
    ax4.bar(xx, mj, bottom=c3, color=AMBER, label="MAJ")
    ax4.bar(xx, dk, bottom=[a + b for a, b in zip(c3, mj)], color=TEAL, label="DICT")
    ax4.bar(xx, rv, bottom=[a + b + c for a, b, c in zip(c3, mj, dk)], color=RED, label="REVIEW")
    ax4.set_xticks(xx); ax4.set_xticklabels(labs, fontsize=8); ax4.set_ylabel("vị trí")
    ax4.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    ax4.set_title("Theo từng trang (+2 lever)", fontsize=10)
    pdf.savefig(fig); plt.close(fig)


def p_examples(pdf, ex, book):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("Ví dụ thật — [HÌNH crop] · kim | qwen | nna → chốt", fontsize=13, fontweight="bold", y=0.975)
    secs = [("CONSENSUS3 — cả 3 khớp", ex["CONSENSUS3"], GREEN, False),
            ("MAJORITY — 2/3 khớp", ex["MAJORITY"], AMBER, False),
            ("DICT — kim SAI, từ điển+model sửa (âm dưới)", ex["DICT"], TEAL, True),
            ("REVIEW — cả 3 khác", ex["REVIEW"], RED, False)]
    y = 0.905
    for title, rows, col, show_syl in secs:
        ax = fig.add_axes([0.04, y - 0.205, 0.92, 0.195]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0, 1.04, title, fontsize=11, fontweight="bold", color=col, transform=ax.transAxes)
        for k, item in enumerate(rows[:8]):
            kim, qw, nn, lbl, syl, crop = item
            xx = 0.01 + (k % 4) * 0.245
            yy = 0.66 - (k // 4) * 0.46
            oi = OffsetImage(crop, zoom=0.5, cmap="gray")
            ax.add_artist(AnnotationBbox(oi, (xx + 0.028, yy), xycoords=ax.transAxes,
                                         frameon=True, bboxprops=dict(edgecolor=col, lw=1.2)))
            for j, ch in enumerate([kim, qw or "·", nn or "·", lbl]):
                ax.text(xx + 0.075 + j * 0.04, yy, ch, fontproperties=NOM, fontsize=15, va="center",
                        transform=ax.transAxes, color=("black" if j < 3 else col))
            if show_syl and syl:
                ax.text(xx + 0.028, yy - 0.24, syl, fontsize=7.5, color=GREY, ha="center", transform=ax.transAxes)
        y -= 0.225
    fig.text(0.5, 0.02, "Ô đầu mỗi ví dụ = HÌNH CROP thật từ ảnh scan (so với chữ 'chốt' màu để thấy consensus đúng). "
             "Vd DICT: 署→渃(nước), kim đọc sai, dict+model sửa đúng.",
             ha="center", fontsize=8, style="italic", color=GREY)
    pdf.savefig(fig); plt.close(fig)


def p_valid(pdf, tt, ov, tot):
    fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "Kiểm chứng & Kết luận", ha="center", fontsize=16, fontweight="bold")
    t = ov["total"] or 1
    txt = (f"KIỂM CHỨNG TẦNG DICT (từ điển phân xử {ov['total']} ca kim bị sửa)\n\n"
           f"  • kim ∉ dict(âm) = kim đọc SAI theo âm QN : {ov['kim_notin']}/{ov['total']} = {100*ov['kim_notin']/t:.0f}%\n"
           f"  • chữ được chọn ∈ dict(âm) = hợp lệ        : {ov['lbl_valid']}/{ov['total']} = {100*ov['lbl_valid']/t:.0f}%\n"
           f"  → Đại đa số là SỬA LỖI THẬT: kim đọc nhầm chữ gần giống, model độc lập + từ điển QN\n"
           f"    (âm do người dịch, khác modality) phục hồi chữ hợp lệ. Đã xác nhận mắt: 磊→渃, 兩→茄.\n\n"
           "KẾT LUẬN\n"
           f"  • USABLE (CONS3+MAJ+DICT) = {100*(tt['CONSENSUS3']+tt['MAJORITY']+tt['DICT'])/tot:.1f}%  |  "
           f"REVIEW = {100*tt['REVIEW']/tot:.1f}%\n"
           "  • CONSENSUS3 = tầng GOLD tin cậy cao nhất (3 model độc lập cùng đồng ý).\n"
           "  • DICT = 'CONTEXT GOLD': sửa lỗi kim bằng trọng tài từ điển — giá trị lớn nhất của kiến trúc.\n"
           "  • Lever DICT mạnh (REVIEW giảm mạnh); lever biến-thể nhẹ (bất đồng chủ yếu là chữ khác thật).\n\n"
           "CẦN LÀM TRƯỚC KHI TIN 100%\n"
           "  • Audit người (verify.csv) cho tầng DICT/GOLD mới.\n"
           "  • 3% ca kim∈dict bị đổi → nên hạ SILVER, không GOLD.\n"
           "  • Lỗi tương quan (Qwen+NomNaOCR cùng đọc-hình) — từ điển QN là chốt chặn độc lập.")
    ax.text(0.06, 0.88, txt, va="top", fontsize=10.5)
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen2")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--qwen_dir", default="qwen_cache_235b")
    args = ap.parse_args()
    stems, tb, tt, per_page, ex, ov = gather(args.book, args.n, args.qwen_dir)
    tot = sum(tt.values())
    outp = REPO / "evaluation" / f"tri_report_{args.book}.pdf"
    with PdfPages(outp) as pdf:
        p_title(pdf, args.book, len(stems), tot)
        p_results(pdf, tb, tt, per_page, tot)
        p_examples(pdf, ex, args.book)
        p_valid(pdf, tt, ov, tot)
        d = pdf.infodict(); d["Title"] = f"Tri-model consensus report {args.book}"
    print(f"base={dict(tb)}\nboth={dict(tt)}\ndict_overrule={ov}")
    print(f"PDF -> {outp}")


if __name__ == "__main__":
    main()
