"""Export a 100-sample PDF for MANUAL inspection of the built dataset.

Each sample shows, side by side:
  [ crop thật (woodblock, viền đỏ) ]   →   [ glyph tham chiếu FontDiffusion (viền xanh) ]
with caption = nhãn (chữ SinoNom = âm tiết Quốc-Ngữ) + (book/page/col).

You eyeball whether the LEFT crop is really the SAME character as the RIGHT
reference glyph -> a quick human accuracy check on the dataset.

Run:
  cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
  .venv/bin/python evaluation/ver_new/sample_pdf.py
  # options: --n 100 --out <path.pdf> --dataset <dir> --seed 123 --min-ink 0.08
Output: evaluation/ver_new/results/sample_100.pdf
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _font(size: int):
    for cand in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _fd_index() -> dict[str, str]:
    idx = {}
    for p in (REPO / "gannhanocr-fd").rglob("U+*.png"):
        try:
            idx[chr(int(p.stem.replace("U+", ""), 16))] = str(p)
        except ValueError:
            pass
    return idx


def _load(path: Path, sz: int, boost: bool = False) -> Image.Image:
    im = Image.open(path).convert("L")
    if boost:
        im = ImageOps.autocontrast(im, 1)
    im = im.convert("RGB")
    im.thumbnail((sz, sz))
    c = Image.new("RGB", (sz, sz), "white")
    c.paste(im, ((sz - im.width) // 2, (sz - im.height) // 2))
    return c


def _ink(path: Path) -> float:
    return (np.asarray(Image.open(path).convert("L")) < 128).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--out", default=str(HERE / "results" / "sample_100.pdf"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-ink", type=float, default=0.08)
    args = ap.parse_args()
    random.seed(args.seed)

    D = Path(args.dataset)
    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["image"] and r["tier"] == "GOLD"]
    fd = _fd_index()
    # keep samples whose crop has ink and whose label has an FD reference
    pool = [r for r in rows if r["label"] in fd and (D / r["image"]).exists()
            and _ink(D / r["image"]) >= args.min_ink]
    random.shuffle(pool)
    sample = pool[:args.n]

    # layout
    glyph = 120
    cell_w, cell_h = 300, 190
    cols, per_page = 4, 20
    margin, top = 30, 70
    W = margin * 2 + cols * cell_w
    rows_per = per_page // cols
    H = top + rows_per * cell_h + 30
    fnt = _font(20)
    fnt_s = _font(17)
    fnt_h = _font(24)

    pages = []
    for pi in range(0, len(sample), per_page):
        chunk = sample[pi:pi + per_page]
        pg = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(pg)
        d.text((margin, 16), f"Dataset Nom — mau {pi + 1}-{pi + len(chunk)}/"
               f"{len(sample)}   |   TRAI = crop that (do)   PHAI = glyph tham chieu FD (xanh)",
               fill="black", font=fnt_h)
        d.line([margin, 58, W - margin, 58], fill=(200, 200, 200))
        for k, r in enumerate(chunk):
            cx = margin + (k % cols) * cell_w
            cy = top + (k // cols) * cell_h
            # crop (boosted) + FD reference
            pg.paste(_load(D / r["image"], glyph, boost=True), (cx, cy + 8))
            d.rectangle([cx, cy + 8, cx + glyph, cy + 8 + glyph], outline=(200, 0, 0), width=3)
            fx = cx + glyph + 20
            pg.paste(_load(Path(fd[r["label"]]), glyph), (fx, cy + 8))
            d.rectangle([fx, cy + 8, fx + glyph, cy + 8 + glyph], outline=(0, 150, 0), width=2)
            d.text((cx + glyph + 4, cy - 4), "→", fill=(120, 120, 120), font=fnt_h)
            # caption: syllable + codepoint + provenance
            cp = f"U+{ord(r['label']):X}"
            d.text((cx, cy + glyph + 14), f"= {r['syllable']}", fill=(0, 0, 0), font=fnt)
            d.text((cx, cy + glyph + 40),
                   f"{cp}  {r['book']} {r['page']} c{r['column']}",
                   fill=(110, 110, 110), font=fnt_s)
        pages.append(pg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # This Pillow build has no JPEG codec; PDF save of RGB tries JPEG -> fails.
    # Palette ("P") mode uses Flate, no JPEG. Colours here are few (b/w/red/green/gray).
    pp = [p.convert("P", palette=Image.ADAPTIVE, colors=128) for p in pages]
    pp[0].save(str(out), save_all=True, append_images=pp[1:], resolution=150)
    print(f"PDF: {out}  ({len(sample)} mau, {len(pages)} trang)")


if __name__ == "__main__":
    main()
