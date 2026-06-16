"""Crop-QUALITY audit PDF — eyeball whether the cropping is good, across the whole
quality spectrum (NOT cherry-picked good ones).

Re-evaluation of "the crop part": shows the ACTUAL cropped PNGs grouped by stratum,
each next to its reference glyph (gannhanocr-fd, woodblock-style) so you can judge
both (a) crop QUALITY — is it one clean glyph, not merged / clipped / blank — and
(b) crop IDENTITY — is it the labelled character. Groups, in order:

  1. GOLD ngẫu nhiên            — baseline (kỳ vọng tốt)
  2. seg_flag = 'tall'          — NGHI dính 2 chữ (cắt lỗi hay gặp nhất)
  3. ink% thấp nhất             — NGHI trắng / cụt nét
  4. ink% cao nhất              — NGHI dày / dính nền / 2 chữ
  5. SILVER (S3 sửa OCR)        — crop ở tầng thị giác
  6. SYLLABLE                   — crop ở tầng âm tiết
  7. Head FLAG look-alike       — crop bị encoder ngờ là chữ khác (từ label_error_candidates.csv):
                                   caption ghi "gán X | head nghĩ Y" để soi cắt-đúng-chữ-chưa

Header trang đầu in lại số đánh giá segmentation (count-accuracy, crop recognizability).

Run:
  .venv/bin/python evaluation/ver_new/crop_audit_pdf.py            # --per-group 24
Output: evaluation/ver_new/results/crop_audit.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _font(size: int):
    for c in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
              "/Library/Fonts/Arial Unicode.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
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


def _fit(path: Path, sz: int, boost: bool = False) -> Image.Image:
    """Scale (up or down) to fit sz×sz, centred on white. Crisp upscale for tiny crops."""
    im = Image.open(path).convert("L")
    if boost:
        im = ImageOps.autocontrast(im, 1)
    w, h = im.size
    s = sz / max(w, h)
    im = im.resize((max(1, int(w * s)), max(1, int(h * s))),
                   Image.LANCZOS if s < 1 else Image.NEAREST).convert("RGB")
    c = Image.new("RGB", (sz, sz), "white")
    c.paste(im, ((sz - im.width) // 2, (sz - im.height) // 2))
    return c


def render_group(title, desc, items, fd, D, per_page=15):
    """items: list of dict(crop, ref_char, cap1, cap2). -> list of PIL pages."""
    glyph = 120
    cell_w, cell_h = 360, 196
    cols = 3
    margin, top = 30, 78
    W = margin * 2 + cols * cell_w
    rows_per = per_page // cols
    H = top + rows_per * cell_h + 24
    f_h = _font(26); f_d = _font(16); f_c = _font(19); f_s = _font(15); f_arrow = _font(26)
    pages = []
    for pi in range(0, max(len(items), 1), per_page):
        chunk = items[pi:pi + per_page]
        pg = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(pg)
        d.text((margin, 14), title, fill=(120, 0, 0), font=f_h)
        d.text((margin, 48), desc, fill=(90, 90, 90), font=f_d)
        d.line([margin, 72, W - margin, 72], fill=(200, 200, 200))
        for k, it in enumerate(chunk):
            cx = margin + (k % cols) * cell_w
            cy = top + (k // cols) * cell_h
            try:
                pg.paste(_fit(Path(it["crop"]), glyph, boost=True), (cx, cy + 6))
            except Exception:
                d.rectangle([cx, cy + 6, cx + glyph, cy + 6 + glyph], outline=(150, 150, 150))
                d.text((cx + 20, cy + 50), "no crop", fill=(150, 150, 150), font=f_s)
            d.rectangle([cx, cy + 6, cx + glyph, cy + 6 + glyph], outline=(200, 0, 0), width=3)
            fx = cx + glyph + 28
            ref = fd.get(it["ref_char"])
            if ref:
                pg.paste(_fit(Path(ref), glyph), (fx, cy + 6))
                d.rectangle([fx, cy + 6, fx + glyph, cy + 6 + glyph], outline=(0, 150, 0), width=2)
            else:
                d.rectangle([fx, cy + 6, fx + glyph, cy + 6 + glyph], outline=(210, 210, 210))
                d.text((fx + 16, cy + 50), "no ref", fill=(180, 180, 180), font=f_s)
            d.text((cx + glyph + 4, cy + 40), "→", fill=(140, 140, 140), font=f_arrow)
            d.text((cx, cy + glyph + 12), it["cap1"], fill=(0, 0, 0), font=f_c)
            d.text((cx, cy + glyph + 38), it["cap2"], fill=(110, 110, 110), font=f_s)
        pages.append(pg)
    return pages


def cover(seg, n_by_group, W=1100):
    f_h = _font(40); f = _font(22); f_s = _font(18)
    H = 720
    pg = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(pg)
    d.text((40, 40), "AUDIT CHẤT LƯỢNG CROP — Hán-Nôm", fill=(120, 0, 0), font=f_h)
    d.text((40, 100), "Soi tay: crop (viền ĐỎ) có phải MỘT chữ sạch & đúng glyph tham chiếu (viền XANH)?",
           fill=(80, 80, 80), font=f_s)
    d.line([40, 134, W - 40, 134], fill=(200, 200, 200))
    y = 168
    rec = (seg or {}).get("recognizability", {})
    lines = [
        "ĐÁNH GIÁ LẠI SEGMENTATION (eval_segmentation.py):",
        f"   • Count-accuracy (cột có đúng N box): {(seg or {}).get('count_accuracy',0)*100:.1f}%  "
        f"→ {(1-(seg or {}).get('count_accuracy',0))*100:.0f}% cột 'diverged' (đích của detector)",
        f"   • Crop recognizability (head top-1 == nhãn): {rec.get('top1_acc',0)*100:.1f}%  "
        f"→ crop ở cột matched đã sạch",
        "",
        "PDF này gồm các NHÓM (có cả ca NGHI VẤN, không lọc ca đẹp):",
    ]
    for ln in lines:
        d.text((48, y), ln, fill=(0, 0, 0), font=f); y += 34
    for name, c in n_by_group:
        d.text((64, y), f"   – {name}: {c} ảnh", fill=(60, 60, 60), font=f_s); y += 28
    d.text((48, y + 16), "Viền đỏ = crop thật (woodblock).  Viền xanh = glyph tham chiếu (gannhanocr-fd).",
           fill=(120, 120, 120), font=f_s)
    return pg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--out", default=str(HERE / "results" / "crop_audit.pdf"))
    ap.add_argument("--per-group", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    D = Path(args.dataset)
    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8")) if r["image"]]
    fd = _fd_index()
    seg = {}
    if (HERE / "results" / "eval_segmentation.json").exists():
        seg = json.load(open(HERE / "results" / "eval_segmentation.json", encoding="utf-8"))

    def ink(r):
        try:
            return float(r["ink_pct"] or 0)
        except ValueError:
            return 0.0

    def cap(r, lab=None):
        lab = lab or r["label"] or r["ocr_char"]
        cp = f"U+{ord(lab):X}" if lab and len(lab) == 1 else ""
        c1 = f"{lab}  {cp}   âm: {r['syllable']}"
        c2 = f"{r['tier']} · ink {ink(r)*100:.0f}% · {r.get('seg_flag','')} · {r['book']} {r['page']} c{r['column']}"
        return c1, c2

    def items_from(rs, lab_key=None):
        out = []
        for r in rs:
            lab = r["label"] or r["ocr_char"]
            c1, c2 = cap(r, lab)
            out.append({"crop": str(D / r["image"]), "ref_char": lab, "cap1": c1, "cap2": c2})
        return out

    N = args.per_group
    gold = [r for r in rows if r["tier"] == "GOLD" and r["label"]]
    tall = [r for r in gold if (r.get("seg_flag") == "tall")]
    silver = [r for r in rows if r["tier"] == "SILVER" and r["label"]]
    syll = [r for r in rows if r["tier"] == "SYLLABLE"]
    random.shuffle(gold); random.shuffle(tall); random.shuffle(silver); random.shuffle(syll)
    low_ink = sorted([r for r in gold], key=ink)[:N]
    high_ink = sorted([r for r in gold], key=ink, reverse=True)[:N]

    groups = [
        ("1. GOLD ngẫu nhiên (baseline)", "kỳ vọng: crop sạch 1 chữ, khớp glyph phải",
         items_from(gold[:N])),
        ("2. seg_flag = 'tall' — NGHI dính 2 chữ", "crop cao bất thường: kiểm có lẫn chữ hàng xóm không",
         items_from(tall[:N])),
        ("3. ink% THẤP nhất — NGHI trắng/cụt nét", "mực ít: kiểm có bị cắt cụt / gần trắng không",
         items_from(low_ink)),
        ("4. ink% CAO nhất — NGHI dày/dính nền", "mực nhiều: kiểm có dính nét cột / 2 chữ không",
         items_from(high_ink)),
        ("5. SILVER (S3 sửa OCR đọc nhầm)", "crop tầng thị giác: chữ phải khớp glyph phải",
         items_from(silver[:N])),
        ("6. SYLLABLE (đúng âm, chữ chưa chắc)", "soi crop + âm tiết (glyph phải = ocr_char)",
         items_from(syll[:N])),
    ]

    # 7. head-flagged look-alikes (crop maybe mis-cut OR mislabelled)
    lec = HERE / "results" / "label_error_candidates.csv"
    if lec.exists():
        flags = list(csv.DictReader(open(lec, encoding="utf-8")))
        random.shuffle(flags)
        it7 = []
        for r in flags[:N]:
            c1 = f"gán: {r['assigned']}  |  head nghĩ: {r['head_alt']} (p={r['p_alt']})"
            c2 = f"{r['tier']} · âm {r['syllable']} · {r['book']} {r['page']}"
            it7.append({"crop": str(D / r["image"]), "ref_char": r["assigned"], "cap1": c1, "cap2": c2})
        groups.append(("7. Head FLAG look-alike — crop sai cắt HAY sai nhãn?",
                       "encoder ngờ chữ khác: so crop với glyph 'gán' (phải) — crop có đúng chữ gán không?",
                       it7))

    pages = [cover(seg, [(g[0].split(' — ')[0], len(g[2])) for g in groups])]
    for title, desc, items in groups:
        if items:
            pages += render_group(title, desc, items, fd, D, per_page=args.per_group)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pp = [p.convert("P", palette=Image.ADAPTIVE, colors=128) for p in pages]   # no-JPEG-codec safe
    pp[0].save(str(out), save_all=True, append_images=pp[1:], resolution=150)
    print(f"crop-audit PDF: {out}  ({len(pages)} trang, {sum(len(g[2]) for g in groups)} crop)")
    for title, _, items in groups:
        print(f"  - {title}: {len(items)}")


if __name__ == "__main__":
    main()
