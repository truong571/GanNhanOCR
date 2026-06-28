"""PDF to EYEBALL the new `s3_head_bank_consensus` rescues (gated #4).

Renders the audit-packet sample (eval_sample_head/) as a grid: each woodblock crop
(red border) next to the reference glyph of its PROPOSED label (green border) +
caption. You look: is the crop really that character? -> a quick visual read on the
tier's precision before adopting it.

Run (after validate_head_consensus.py --packet built eval_sample_head/):
  .venv/bin/python evaluation/ver_new/head_audit_pdf.py
Output: evaluation/ver_new/results/head_audit.pdf
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from evaluation.ver_new.crop_audit_pdf import _font, _fit, _fd_index   # noqa: E402

HERE = Path(__file__).resolve().parent
PKT = HERE / "eval_sample_head"
OUT = HERE / "results" / "head_audit.pdf"


def main():
    rows = list(csv.DictReader(open(PKT / "verify.csv", encoding="utf-8")))
    fd = _fd_index()
    glyph, cell_w, cell_h, cols, per = 120, 360, 200, 3, 15
    margin, top = 30, 70
    W = margin * 2 + cols * cell_w
    H = top + (per // cols) * cell_h + 24
    f_h, f_c, f_s = _font(24), _font(19), _font(15)
    pages = []
    for pi in range(0, len(rows), per):
        chunk = rows[pi:pi + per]
        pg = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(pg)
        d.text((margin, 14), f"Soát tier head∩bank (gated ≥0.3) — mẫu {pi+1}-{pi+len(chunk)}/{len(rows)}",
               fill=(120, 0, 0), font=f_h)
        d.text((margin, 46), "TRÁI = crop thật (đỏ)   PHẢI = glyph chữ ĐỀ XUẤT (xanh).  Crop có đúng chữ đó không?",
               fill=(90, 90, 90), font=f_s)
        d.line([margin, 66, W - margin, 66], fill=(200, 200, 200))
        for k, r in enumerate(chunk):
            cx = margin + (k % cols) * cell_w; cy = top + (k // cols) * cell_h
            cp = PKT / r["image"] if r["image"] else None
            if cp and cp.exists():
                pg.paste(_fit(cp, glyph, boost=True), (cx, cy + 6))
            d.rectangle([cx, cy + 6, cx + glyph, cy + 6 + glyph], outline=(200, 0, 0), width=3)
            fx = cx + glyph + 26
            ref = fd.get(r["label"])
            if ref:
                pg.paste(_fit(Path(ref), glyph), (fx, cy + 6))
                d.rectangle([fx, cy + 6, fx + glyph, cy + 6 + glyph], outline=(0, 150, 0), width=2)
            d.text((cx + glyph + 2, cy + 40), "→", fill=(140, 140, 140), font=f_h)
            d.text((cx, cy + glyph + 12), f"{r['label']}  {r['unicode']}", fill=(0, 0, 0), font=f_c)
            d.text((cx, cy + glyph + 38), f"âm {r['syllable']} · ocr {r['ocr_char']} · {r['book']} {r['page']}",
                   fill=(110, 110, 110), font=f_s)
        pages.append(pg)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pp = [p.convert("P", palette=Image.ADAPTIVE, colors=128) for p in pages]
    pp[0].save(str(OUT), save_all=True, append_images=pp[1:], resolution=150)
    print(f"PDF -> {OUT}  ({len(pages)} trang, {len(rows)} mẫu head-consensus)")


if __name__ == "__main__":
    main()
