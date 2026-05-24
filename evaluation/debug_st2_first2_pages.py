"""Debug dump for SachThanhTruyen2, first 2 pages (page_0012, page_0014).

Outputs to evaluation/debug_st2_first2/:
  - report.md            : Human-readable summary, per-column tables
  - pairs.json           : Full record per (col, char_idx)
  - page_0012_viz.png    : Original page with bbox + tier color overlay
  - page_0014_viz.png    : Same
  - contact_sheet_pXX.png: Grid of all crops with label under each
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
BOOK_DIR = ROOT / "prepared" / "SachThanhTruyen2"
OUT_DIR = ROOT / "evaluation" / "debug_st2_first2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAGES = ["page_0012", "page_0014"]

TIER_COLOR = {  # BGR for cv2
    1: (0, 180, 0),     # green   - dict
    2: (0, 165, 255),   # orange  - similar
    3: (255, 80, 0),    # blue    - DINOv2
    0: (60, 60, 200),   # red     - none
}
TIER_NAME = {1: "T1 dict", 2: "T2 sim", 3: "T3 vis", 0: "T0 -"}

FONT_PATH = ROOT / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"


def load_records():
    ds = json.load(open(BOOK_DIR / "labeled" / "dataset.json"))
    aligned = {p: json.load(open(BOOK_DIR / "aligned" / f"{p}_aligned.json"))
               for p in PAGES}
    return ds, aligned


def per_page_records(ds, page):
    return [r for r in ds if r["page"] == page]


def write_pairs_json(ds, aligned):
    out = {}
    for p in PAGES:
        lbl = per_page_records(ds, p)
        ali = aligned[p]
        out[p] = {
            "n_pairs_aligned": len(ali),
            "n_records_labeled": len(lbl),
            "records": lbl,
        }
    (OUT_DIR / "pairs.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2)
    )


def write_report(ds, aligned):
    lines = ["# Debug — SachThanhTruyen2, 2 trang đầu", ""]

    for p in PAGES:
        recs = per_page_records(ds, p)
        ali = aligned[p]
        n_match = sum(1 for r in recs if r["matched"])
        n_unmatched = sum(1 for r in recs if r["type"] == "match" and not r["matched"])
        n_gap = sum(1 for r in recs if r["type"] != "match")
        tiers = {1: 0, 2: 0, 3: 0, 0: 0}
        for r in recs:
            tiers[r.get("tier", 0)] += 1

        lines += [
            f"## {p}", "",
            f"- Số cặp align: **{len(ali)}**, số record labeled: **{len(recs)}**",
            f"- Matched: **{n_match}**  |  Unmatched: **{n_unmatched}**  |  Gap: **{n_gap}**",
            f"- Tier 1 (dict): {tiers[1]}  |  Tier 2 (sim): {tiers[2]}  "
            f"|  Tier 3 (DINOv2): {tiers[3]}  |  Tier 0 (none): {tiers[0]}",
            f"- Match rate: **{100*n_match/max(1,len(recs)):.1f}%**",
            "",
        ]

        cols = sorted({r["column"] for r in recs if r.get("column") is not None})
        for col in cols:
            col_recs = [r for r in recs if r.get("column") == col]
            qn = " ".join(r.get("syllable") or "·" for r in col_recs)
            nom = "".join(r.get("nom_char") or "·" for r in col_recs)
            ocr = "".join(r.get("ocr_char") or "·" for r in col_recs)
            lines += [
                f"### Cột {col}  ({len(col_recs)} ký tự)",
                "",
                f"- **QN  :** {qn}",
                f"- **OCR :** {ocr}   (gợi ý từ Kimhannom)",
                f"- **Gán :** {nom}",
                "",
                "| # | syl | ocr | gán | U+ | tier | matched | vis | crop |",
                "|--:|-----|-----|-----|-----|------|---------|----:|------|",
            ]
            for i, r in enumerate(col_recs):
                vis = r.get("visual_score")
                vis_s = f"{vis:.3f}" if isinstance(vis, (int, float)) else ""
                cf = r.get("crop_file") or ""
                lines.append(
                    f"| {i} | {r.get('syllable') or ''} "
                    f"| {r.get('ocr_char') or ''} "
                    f"| {r.get('nom_char') or ''} "
                    f"| {r.get('unicode') or ''} "
                    f"| {TIER_NAME.get(r.get('tier',0))} "
                    f"| {'✓' if r.get('matched') else '✗'} "
                    f"| {vis_s} "
                    f"| `{cf}` |"
                )
            lines.append("")

        lines += ["![viz](" + f"{p}_viz.png)",
                  "![sheet](" + f"contact_{p}.png)", ""]

    (OUT_DIR / "report.md").write_text("\n".join(lines))


def draw_viz(ds):
    """Draw per-page image with bbox + tier color + label."""
    for p in PAGES:
        recs = per_page_records(ds, p)
        img_path = BOOK_DIR / "pages" / f"{p}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] {img_path}")
            continue

        H, W = img.shape[:2]
        # Use PIL for unicode Nom rendering
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype(str(FONT_PATH), 28)
        except Exception:
            font = ImageFont.load_default()

        for r in recs:
            bb = r.get("bbox")
            if not bb:
                continue
            x1, y1, x2, y2 = bb
            color = TIER_COLOR.get(r.get("tier", 0), (200, 0, 0))
            color_rgb = (color[2], color[1], color[0])
            draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)
            label = r.get("nom_char") or "?"
            draw.text((x2 + 2, y1), label, fill=color_rgb, font=font)

        # Legend
        legend = Image.new("RGB", (W, 90), (255, 255, 255))
        ld = ImageDraw.Draw(legend)
        try:
            lfont = ImageFont.truetype(str(FONT_PATH), 24)
        except Exception:
            lfont = ImageFont.load_default()
        x = 20
        for tier, name in [(1, "Tier1 dict"), (2, "Tier2 similar"),
                           (3, "Tier3 DINOv2"), (0, "Tier0/unmatched")]:
            c = TIER_COLOR[tier]
            ld.rectangle([x, 20, x + 60, 70],
                         outline=(c[2], c[1], c[0]), width=4)
            ld.text((x + 70, 30), name, fill=(0, 0, 0), font=lfont)
            x += 320

        out = Image.new("RGB", (W, H + 90), (255, 255, 255))
        out.paste(legend, (0, 0))
        out.paste(pil, (0, 90))
        out.save(OUT_DIR / f"{p}_viz.png")
        print(f"[viz] {p}_viz.png  ({W}x{H+90})")


def draw_contact_sheet(ds):
    """Grid of all crops, with QN syl / Nôm gán under each."""
    for p in PAGES:
        recs = per_page_records(ds, p)
        recs = [r for r in recs if r.get("crop_file")]
        if not recs:
            continue

        cell_w, cell_h = 96, 140
        cols = 16
        rows = (len(recs) + cols - 1) // cols
        sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), (255, 255, 255))
        try:
            font = ImageFont.truetype(str(FONT_PATH), 16)
            small = ImageFont.truetype(str(FONT_PATH), 12)
        except Exception:
            font = ImageFont.load_default()
            small = font
        draw = ImageDraw.Draw(sheet)

        for idx, r in enumerate(recs):
            row, col = divmod(idx, cols)
            x0, y0 = col * cell_w, row * cell_h
            cp = BOOK_DIR / "detected" / r["crop_file"]
            if cp.exists():
                im = Image.open(cp).convert("RGB").resize((80, 80))
                sheet.paste(im, (x0 + 8, y0 + 4))
            color = TIER_COLOR.get(r.get("tier", 0), (200, 0, 0))
            color_rgb = (color[2], color[1], color[0])
            draw.rectangle([x0 + 7, y0 + 3, x0 + 89, y0 + 85],
                           outline=color_rgb, width=2)
            syl = r.get("syllable") or ""
            nom = r.get("nom_char") or "?"
            tag = TIER_NAME.get(r.get("tier", 0))
            draw.text((x0 + 8, y0 + 86), f"{syl}", fill=(0, 0, 0), font=small)
            draw.text((x0 + 8, y0 + 100), f"→ {nom}",
                      fill=color_rgb, font=font)
            draw.text((x0 + 8, y0 + 120), tag, fill=(80, 80, 80), font=small)

        sheet.save(OUT_DIR / f"contact_{p}.png")
        print(f"[sheet] contact_{p}.png  ({len(recs)} crops)")


def main():
    ds, aligned = load_records()
    write_pairs_json(ds, aligned)
    write_report(ds, aligned)
    draw_viz(ds)
    draw_contact_sheet(ds)
    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
