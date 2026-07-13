"""Full-page consensus sheet: EVERY position of ONE page as [crop] · kim|qwen|nna→chốt,
tier-coloured. Reading order (cột phải→trái, trên→dưới). Paginated grid.

Run:
  .venv/bin/python evaluation/tri_consensus/make_page_examples.py \
      --book SachThanhTruyen2 --page page_0012 --qwen_dir qwen_cache_235b
Output: evaluation/tri_page_<book>_<page>.pdf
"""
from __future__ import annotations
import argparse, json
from collections import Counter
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
COLS, ROWS = 4, 15                      # cells per PDF page
PERPAGE = COLS * ROWS


def build(book, page, qwen_dir):
    qn, sim, sylpg = load_qn_dict(), load_similar(), load_syllables(book)
    decide = make_decider(qn, sim, True, True)
    cdir = REPO / "prepared" / book / "detected"
    pdir = REPO / "prepared" / book / "pages_denoised"
    cache = json.loads((cdir / f"{page}_ocr_cache.json").read_text(encoding="utf-8"))
    kcols = [c for c in cache.get("columns", []) if c]
    kinfo = [(ci + 1, c["char"], c["bbox"]) for ci, col in enumerate(kcols) for c in col]
    kflat = [ch for _, ch, _ in kinfo]
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
    cells, tally = [], Counter()
    for i, (colno, kim, bbox) in enumerate(kinfo):
        qw, nn, syl = qmap[i], nmap[i], smap[i]
        lbl, tier = decide(kim, qw, nn, syl)
        tally[tier] += 1
        x1, y1, x2, y2 = (int(v) for v in bbox)
        g = pg.crop((x1 - 3, y1 - 3, x2 + 3, y2 + 3)); g.thumbnail((54, 54))
        cells.append(dict(colno=colno, kim=kim, qw=qw, nn=nn, syl=syl, lbl=lbl,
                          tier=tier, crop=np.asarray(g)))
    return cells, tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen2")
    ap.add_argument("--page", default="page_0012")
    ap.add_argument("--qwen_dir", default="qwen_cache_235b")
    args = ap.parse_args()
    cells, tally = build(args.book, args.page, args.qwen_dir)
    tot = sum(tally.values())
    outp = REPO / "evaluation" / f"tri_page_{args.book}_{args.page}.pdf"
    with PdfPages(outp) as pdf:
        for p0 in range(0, len(cells), PERPAGE):
            chunk = cells[p0:p0 + PERPAGE]
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if p0 == 0:
                ax.text(0.5, 0.975, f"{args.book} · {args.page} — TOÀN TRANG", ha="center",
                        fontsize=14, fontweight="bold")
                leg = "  ".join(f"{k} {tally[k]} ({100*tally[k]/tot:.0f}%)"
                                for k in ("CONSENSUS3", "MAJORITY", "DICT", "REVIEW"))
                ax.text(0.5, 0.952, f"{tot} vị trí · " + leg, ha="center", fontsize=8.5, color="#374151")
                ax.text(0.5, 0.935, "mỗi ô:  [HÌNH crop] · kim  qwen  nna → chốt  (viền/chốt = màu tầng · âm dưới)",
                        ha="center", fontsize=8, style="italic", color="#6b7280")
            else:
                ax.text(0.5, 0.975, f"{args.book} · {args.page} (tiếp)", ha="center", fontsize=11, fontweight="bold")
            top = 0.90 if p0 == 0 else 0.94
            for k, c in enumerate(chunk):
                col, row = k % COLS, k // COLS
                x = 0.02 + col * 0.245
                y = top - row * 0.060
                tc = TIER_COL[c["tier"]]
                oi = OffsetImage(c["crop"], zoom=0.42, cmap="gray")
                ax.add_artist(AnnotationBbox(oi, (x + 0.022, y), xycoords=ax.transAxes,
                                             frameon=True, bboxprops=dict(edgecolor=tc, lw=1.1)))
                for j, ch in enumerate([c["kim"], c["qw"] or "·", c["nn"] or "·", c["lbl"]]):
                    ax.text(x + 0.058 + j * 0.032, y, ch, fontproperties=NOM, fontsize=12, va="center",
                            transform=ax.transAxes, color=("black" if j < 3 else tc))
                ax.text(x + 0.022, y - 0.026, f"c{c['colno']}·{c['syl'] or '?'}", fontsize=5.6,
                        ha="center", color="#6b7280", transform=ax.transAxes)
            pdf.savefig(fig); plt.close(fig)
    print(f"tally={dict(tally)}  tot={tot}  pages={(len(cells)+PERPAGE-1)//PERPAGE}")
    print(f"PDF -> {outp}")


if __name__ == "__main__":
    main()
