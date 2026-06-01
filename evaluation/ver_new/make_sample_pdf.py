"""Render a 100-sample GOLD review sheet as a multi-page PDF for MANUAL audit.

Each cell shows, side by side:
  - LEFT  (red)   : the real woodblock CROP that was labelled
  - RIGHT (green) : the FontDiffusion reference glyph of the assigned label
plus a caption (index, label char = syllable, book/page/column) and a tick box.
A reviewer compares the two glyphs: same character => label correct, and checks
the crop is a clean single glyph (no merge / clip).

Run:
  cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
  .venv/bin/python evaluation/ver_new/make_sample_pdf.py            # 100 samples
  .venv/bin/python evaluation/ver_new/make_sample_pdf.py --n 100 --seed 123
Output: evaluation/ver_new/results/sample_100.pdf
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
D = ROOT / "dataset_out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=str(ROOT / "results" / "sample_100.pdf"))
    ap.add_argument("--min-ink", type=float, default=0.06)
    args = ap.parse_args()
    random.seed(args.seed)

    fd: dict[str, str] = {}
    for p in (REPO / "gannhanocr-fd").rglob("U+*.png"):
        try:
            fd[chr(int(p.stem.replace("U+", ""), 16))] = str(p)
        except ValueError:
            pass

    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["image"] and r["tier"] == "GOLD" and r["label"] in fd]
    random.shuffle(rows)

    def ink(img):
        a = np.asarray(Image.open(D / img).convert("L"))
        return (a < 128).mean()

    sample = []
    for r in rows:
        try:
            if ink(r["image"]) >= args.min_ink:
                sample.append(r)
        except Exception:
            continue
        if len(sample) >= args.n:
            break

    def load(path, sz, boost=False):
        im = Image.open(path).convert("L")
        if boost:
            im = ImageOps.autocontrast(im, 1)
        im = im.convert("RGB")
        im.thumbnail((sz, sz))
        c = Image.new("RGB", (sz, sz), "white")
        c.paste(im, ((sz - im.width) // 2, (sz - im.height) // 2))
        return c

    COLS, ROWS = 4, 6
    PER = COLS * ROWS
    gw = 120
    cellW, cellH = 2 * gw + 34, gw + 60
    margin, gap = 34, 16
    PW = margin * 2 + COLS * cellW + (COLS - 1) * gap
    PH = margin * 2 + ROWS * cellH + (ROWS - 1) * gap + 36
    npages = (len(sample) + PER - 1) // PER

    pages = []
    for pi in range(0, len(sample), PER):
        cv = Image.new("RGB", (PW, PH), "white")
        d = ImageDraw.Draw(cv)
        d.text((margin, 10), f"GOLD review — trang {pi // PER + 1}/{npages}   "
               f"(TRAI = crop that [do], PHAI = glyph FD tham chieu [xanh]; "
               f"khop = nhan dung)", fill="black")
        for k, r in enumerate(sample[pi:pi + PER]):
            cc, rr = k % COLS, k // COLS
            x = margin + cc * (cellW + gap)
            y = margin + 32 + rr * (cellH + gap)
            idx = pi + k
            d.text((x, y), f"#{idx + 1}  {r['label']}={r['syllable']}",
                   fill=(0, 0, 0))
            d.text((x, y + 14), f"{r['book']} {r['page'][5:]} c{r['column']}",
                   fill=(110, 110, 110))
            cv.paste(load(D / r["image"], gw, True), (x, y + 30))
            d.rectangle([x, y + 30, x + gw, y + 30 + gw], outline=(200, 0, 0), width=2)
            cv.paste(load(fd[r["label"]], gw), (x + gw + 24, y + 30))
            d.rectangle([x + gw + 24, y + 30, x + 2 * gw + 24, y + 30 + gw],
                        outline=(0, 150, 0), width=2)
            d.text((x, y + 32 + gw), "khop? [  ]   loi: [  ]", fill=(0, 0, 0))
        pages.append(cv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Palette mode -> PDF uses flate (no JPEG codec needed in this Pillow build).
    pp = [p.convert("P", palette=Image.ADAPTIVE, colors=128) for p in pages]
    pp[0].save(str(out), save_all=True, append_images=pp[1:], resolution=150.0)
    print(f"PDF saved: {out}  | pages {len(pages)}  | samples {len(sample)}")


if __name__ == "__main__":
    main()
