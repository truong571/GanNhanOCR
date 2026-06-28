"""PDF chẩn đoán tín hiệu IDS: vài crop đuôi hiếm -> tách vùng theo IDS -> match bộ thủ.
Cho thấy TRỰC QUAN vì sao compositional no-train khó (vùng tách trên crop ván khắc không
sạch). Mỗi ví dụ: CROP | vùng-A ↔ glyph-A (cos) | vùng-B ↔ glyph-B (cos) + kết luận.

Run: .venv/bin/python evaluation/ver_new/make_ids_test_pdf.py --n 8
Output: results/ids_test.pdf
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from pipeline.step0_setup import load_config                       # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402
from evaluation.ver_new.eval_ids_signal import load_ids, fd_path, tight, cos  # noqa: E402

HERE = Path(__file__).resolve().parent
D = HERE / "dataset_out"
OUT = HERE / "results" / "ids_test.pdf"


def afont(sz):
    for c in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"]:
        if Path(c).exists():
            return ImageFont.truetype(c, sz)
    return ImageFont.load_default()


def cell(gray_or_path, sz=96, boost=False):
    if isinstance(gray_or_path, (str, Path)):
        im = Image.open(gray_or_path).convert("L")
    else:
        im = Image.fromarray(gray_or_path)
    if boost:
        from PIL import ImageOps; im = ImageOps.autocontrast(im, 1)
    w, h = im.size; s = sz / max(w, h, 1)
    im = im.resize((max(4, int(w * s)), max(4, int(h * s)))).convert("RGB")
    c = Image.new("RGB", (sz, sz), "white"); c.paste(im, ((sz - im.width) // 2, (sz - im.height) // 2))
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); args = ap.parse_args()
    cfg = load_config(str(REPO / "config" / "pipeline.yaml"))
    qn = load_qn_to_nom(str(REPO / cfg["paths"]["qn_to_nom_dict"]))
    enc = VisualS3(REPO, fd_dir="").enc
    ids = load_ids(); gemb = {}
    def gE(ch):
        if ch not in gemb:
            p = fd_path(ch); gemb[ch] = enc.embed_path(str(p)) if p else None
        return gemb[ch]

    rows = list(csv.DictReader(open(D / "labels.csv", encoding="utf-8")))
    gold = Counter(r["label"] for r in rows if r["tier"] == "GOLD" and r["label"])
    test = [r for r in rows if r["tier"] == "GOLD" and r["split"] == "test" and r["image"]
            and r["label"] and _is_cjk(r["label"]) and gold.get(r["label"], 0) < 5 and r["label"] in ids]

    ex = []
    for r in test:
        if len(ex) >= args.n:
            break
        true = r["label"]; idc, A, B = ids[true]
        if gE(A) is None or gE(B) is None:
            continue
        g = cv2.imread(str(D / r["image"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        gt = tight(g)
        if gt is None:
            continue
        h, w = gt.shape; best = None
        for ratio in (0.4, 0.5, 0.6):
            if idc == "⿰":
                cut = int(w * ratio); rA, rB = gt[:, :cut], gt[:, cut:]
            else:
                cut = int(h * ratio); rA, rB = gt[:cut, :], gt[cut:, :]
            tA, tB = tight(rA), tight(rB)
            if tA is None or tB is None:
                continue
            sA = cos(enc.embed_gray(tA), gE(A)); sB = cos(enc.embed_gray(tB), gE(B))
            if best is None or min(sA, sB) > min(best[2], best[3]):
                best = (tA, tB, sA, sB, ratio)
        if best is None:
            continue
        # encoder thuần: so toàn-chữ với glyph true
        enc_full = cos(enc.embed_gray(gt), gE(true))
        ex.append((r, true, idc, A, B, gt, best, enc_full))

    if not ex:
        print("không gom được ví dụ"); return
    ft, fs, fc = afont(19), afont(15), afont(30)
    rowh = 150; W = 1180
    pages = []
    per = 6
    for pi in range(0, len(ex), per):
        chunk = ex[pi:pi + per]
        pg = Image.new("RGB", (W, 56 + len(chunk) * rowh), "white"); d = ImageDraw.Draw(pg)
        d.text((20, 16), "Chẩn đoán IDS: tách crop theo bộ thủ — vùng tách (xanh dương) có khớp glyph bộ thủ (xanh lá)?",
               fill=(120, 0, 0), font=ft)
        for k, (r, true, idc, A, B, gt, best, enc_full) in enumerate(chunk):
            tA, tB, sA, sB, ratio = best
            y = 56 + k * rowh
            d.text((14, y + 4), f"{true} (U+{ord(true):X}) · âm '{r['syllable']}' · IDS {idc}{A}{B} · {gold.get(true,0)} crop",
                   fill=(0, 0, 0), font=fs)
            x = 14; yy = y + 28
            def put(img, bx, lbl, lblc=(90, 90, 90)):
                nonlocal x
                pg.paste(img, (x, yy)); d.rectangle([x, yy, x + 96, yy + 96], outline=bx, width=3)
                d.text((x, yy + 98), lbl, fill=lblc, font=afont(13)); x += 104
            put(cell(gt, boost=True), (200, 0, 0), "CROP thật")
            d.text((x, yy + 36), "→", fill=(120, 120, 120), font=fc); x += 40
            put(cell(tA, boost=True), (0, 80, 220), "vùng-A")
            put(cell(fd_path(A)), (0, 150, 0), f"{A}  cos {sA:.2f}", (0, 110, 0) if sA > 0.4 else (180, 0, 0))
            x += 16
            put(cell(tB, boost=True), (0, 80, 220), "vùng-B")
            put(cell(fd_path(B)), (0, 150, 0), f"{B}  cos {sB:.2f}", (0, 110, 0) if sB > 0.4 else (180, 0, 0))
            verdict = "KHỚP cả 2" if min(sA, sB) > 0.4 else ("1 vùng hỏng" if max(sA, sB) > 0.4 else "cả 2 hỏng")
            d.text((x + 16, yy + 30), f"toàn-chữ cos {enc_full:.2f}\nbộ thủ min {min(sA,sB):.2f}\n→ {verdict}",
                   fill=(120, 0, 0), font=fs)
        pages.append(pg)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pp = [p.convert("P", palette=Image.ADAPTIVE, colors=128) for p in pages]
    pp[0].save(str(OUT), save_all=True, append_images=pp[1:], resolution=150)
    print(f"PDF -> {OUT}  ({len(pages)} trang, {len(ex)} ví dụ)")


if __name__ == "__main__":
    main()
