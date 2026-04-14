#!/usr/bin/env python3
"""
verify_labels.py - Visual Review Tool cho labels

Tạo HTML report hiển thị crop image cạnh chữ Nôm render,
cho phép người dùng kiểm tra bằng mắt.

Với các từ QN có nhiều candidates, hiển thị tất cả candidates
để người dùng chọn đúng.

Usage:
  python verify_labels.py Data/prepared/CacThanhTruyen2 --limit 50
  python verify_labels.py Data/prepared/CacThanhTruyen2 --page 12
  python verify_labels.py Data/prepared/CacThanhTruyen2 --ambiguous-only
"""

import argparse
import base64
import csv
import io
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("[ERROR] PIL required: pip install Pillow")
    sys.exit(1)


def load_font(font_path: str, size: int = 80):
    """Load NomNaTong font."""
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def render_char_to_base64(char: str, font, size: int = 96) -> str:
    """Render 1 chữ Nôm thành base64 PNG."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    bbox = font.getbbox(char)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (size - tw) // 2 - bbox[0]
    y = (size - th) // 2 - bbox[1]
    draw.text((x, y), char, fill="black", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def crop_to_base64(crop_path: str) -> str:
    """Load crop image và convert sang base64."""
    img = cv2.imread(crop_path)
    if img is None:
        return ""
    # Resize to ~96px height
    h, w = img.shape[:2]
    if h > 0:
        scale = 96 / h
        img = cv2.resize(img, (max(1, int(w * scale)), 96))
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def generate_review_html(
    rows: list[dict],
    trans_dict: dict,
    font,
    output_path: Path,
    book_dir: Path,
):
    """Tạo HTML visual review report."""

    html = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Visual Review - Label Verification</title>
<style>
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }
h1 { color: #2c3e50; margin-bottom: 5px; }
.subtitle { color: #7f8c8d; margin-bottom: 20px; }

.summary { background: white; padding: 15px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.summary .stat { display: inline-block; margin-right: 25px; }
.summary .num { font-size: 24px; font-weight: bold; }
.summary .num.green { color: #27ae60; }
.summary .num.orange { color: #f39c12; }
.summary .num.red { color: #e74c3c; }

.filters { margin-bottom: 15px; }
.filters button { padding: 6px 14px; margin-right: 8px; border: 1px solid #bdc3c7; border-radius: 4px; cursor: pointer; background: white; }
.filters button.active { background: #3498db; color: white; border-color: #3498db; }

.card { background: white; border-radius: 8px; padding: 15px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); display: flex; align-items: flex-start; gap: 15px; }
.card.ambiguous { border-left: 4px solid #f39c12; }
.card.unambiguous { border-left: 4px solid #27ae60; }
.card.hidden { display: none; }

.card-num { color: #95a5a6; font-size: 13px; min-width: 30px; padding-top: 20px; }
.crop-box { text-align: center; min-width: 100px; }
.crop-box img { border: 2px solid #ecf0f1; border-radius: 4px; max-height: 80px; }
.crop-box .label { font-size: 11px; color: #95a5a6; margin-top: 4px; }

.arrow { font-size: 28px; color: #bdc3c7; padding-top: 15px; }

.assigned-box { text-align: center; min-width: 110px; }
.assigned-box .char { font-size: 56px; line-height: 1.1; }
.assigned-box .render img { border: 1px solid #ddd; border-radius: 4px; }
.assigned-box .info { font-size: 12px; color: #7f8c8d; }

.candidates-box { flex: 1; }
.candidates-box .title { font-size: 12px; color: #7f8c8d; margin-bottom: 6px; }
.cand-grid { display: flex; flex-wrap: wrap; gap: 6px; }
.cand-item { text-align: center; padding: 4px 6px; border: 1px solid #ecf0f1; border-radius: 4px; cursor: pointer; min-width: 60px; }
.cand-item:hover { background: #ebf5fb; border-color: #3498db; }
.cand-item .c { font-size: 28px; }
.cand-item img { border: 1px solid #eee; border-radius: 3px; }
.cand-item .reading { font-size: 10px; color: #95a5a6; }
.cand-item.current { background: #eafaf1; border-color: #27ae60; }

.meta { font-size: 11px; color: #95a5a6; min-width: 120px; padding-top: 10px; }
.confidence { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; }
.confidence.high { background: #d5f5e3; color: #1e8449; }
.confidence.medium { background: #fef9e7; color: #9a7d0a; }
.confidence.low { background: #fadbd8; color: #922b21; }
</style>
</head>
<body>
<h1>Label Verification Report</h1>
<p class="subtitle">BOOK_NAME - So sanh crop viet tay vs chu Nom gan nhan</p>
"""

    # Build reverse reading map
    reading_map = {}
    for qn, chars in trans_dict.items():
        for c in chars:
            reading_map.setdefault(c, [])
            if qn not in reading_map[c]:
                reading_map[c].append(qn)

    # Process each row
    cards_data = []
    n_ambiguous = 0
    n_unambiguous = 0

    for i, row in enumerate(rows):
        crop_rel = row.get("image", "")
        nom_char = row.get("nom_char", "")
        reading = row.get("reading", "")
        confidence = row.get("confidence", "")

        if not nom_char:
            continue

        # Find crop
        crop_path = book_dir / crop_rel
        if not crop_path.exists():
            crop_path = book_dir / "detected" / crop_rel
        if not crop_path.exists():
            crop_path = book_dir / "detected" / crop_rel.replace("crops_cleaned/", "crops/")

        # Get candidates
        candidates = trans_dict.get(reading.lower(), [])
        is_ambiguous = len(candidates) > 1

        if is_ambiguous:
            n_ambiguous += 1
        else:
            n_unambiguous += 1

        # Build card data
        crop_b64 = crop_to_base64(str(crop_path)) if crop_path.exists() else ""
        assigned_b64 = render_char_to_base64(nom_char, font)

        cand_renders = []
        for c in candidates[:15]:  # Limit to 15 candidates
            cb64 = render_char_to_base64(c, font, size=64)
            c_readings = reading_map.get(c, [])
            cand_renders.append({
                "char": c,
                "b64": cb64,
                "readings": ", ".join(c_readings[:3]),
                "is_current": c == nom_char,
                "unicode": f"U+{ord(c):04X}" if len(c) == 1 else "",
            })

        cards_data.append({
            "idx": i + 1,
            "crop_b64": crop_b64,
            "crop_file": Path(crop_rel).name,
            "nom_char": nom_char,
            "assigned_b64": assigned_b64,
            "reading": reading,
            "confidence": confidence,
            "n_candidates": len(candidates),
            "is_ambiguous": is_ambiguous,
            "candidates": cand_renders,
            "page": row.get("page", ""),
            "unicode": row.get("label", ""),
        })

    # Summary
    html = html.replace("BOOK_NAME", book_dir.name)
    html += f"""
<div class="summary">
    <div class="stat"><span class="num">{len(cards_data)}</span><br>Total</div>
    <div class="stat"><span class="num orange">{n_ambiguous}</span><br>Ambiguous (>1 cand.)</div>
    <div class="stat"><span class="num green">{n_unambiguous}</span><br>Unambiguous (1 cand.)</div>
</div>

<div class="filters">
    <button class="active" onclick="filterCards('all')">All ({len(cards_data)})</button>
    <button onclick="filterCards('ambiguous')">Ambiguous ({n_ambiguous})</button>
    <button onclick="filterCards('unambiguous')">Unambiguous ({n_unambiguous})</button>
</div>
"""

    # Cards
    for card in cards_data:
        amb_cls = "ambiguous" if card["is_ambiguous"] else "unambiguous"

        html += f"""
<div class="card {amb_cls}" data-type="{amb_cls}">
    <div class="card-num">#{card['idx']}</div>

    <div class="crop-box">
        <img src="data:image/png;base64,{card['crop_b64']}" alt="crop">
        <div class="label">{card['crop_file']}</div>
    </div>

    <div class="arrow">&rarr;</div>

    <div class="assigned-box">
        <div class="char">{card['nom_char']}</div>
        <div class="render"><img src="data:image/png;base64,{card['assigned_b64']}" width="64" height="64"></div>
        <div class="info">{card['unicode']} / {card['reading']}</div>
        <span class="confidence {card['confidence']}">{card['confidence']}</span>
    </div>
"""

        if card["is_ambiguous"]:
            html += f"""
    <div class="candidates-box">
        <div class="title">Candidates ({card['n_candidates']}):</div>
        <div class="cand-grid">
"""
            for cand in card["candidates"]:
                cur_cls = "current" if cand["is_current"] else ""
                html += f"""
            <div class="cand-item {cur_cls}" title="{cand['unicode']}  readings: {cand['readings']}">
                <img src="data:image/png;base64,{cand['b64']}" width="48" height="48"><br>
                <span class="c">{cand['char']}</span><br>
                <span class="reading">{cand['readings'][:15]}</span>
            </div>
"""
            html += """
        </div>
    </div>
"""

        html += f"""
    <div class="meta">
        Page {card['page']}<br>
        {card['n_candidates']} candidates
    </div>
</div>
"""

    # JavaScript for filtering
    html += """
<script>
function filterCards(type) {
    document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');

    document.querySelectorAll('.card').forEach(card => {
        if (type === 'all') {
            card.classList.remove('hidden');
        } else {
            card.classList.toggle('hidden', card.dataset.type !== type);
        }
    });
}
</script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    print(f"  Report: {output_path}")
    return n_ambiguous, n_unambiguous


def main():
    parser = argparse.ArgumentParser(
        description="Visual review tool for label verification")
    parser.add_argument("book_dir",
                        help="Thu muc prepared data (vd: Data/prepared/CacThanhTruyen2)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Gioi han so dong")
    parser.add_argument("--page", type=int, default=0,
                        help="Chi kiem tra 1 trang")
    parser.add_argument("--ambiguous-only", action="store_true",
                        help="Chi hien thi cac label co nhieu candidates")
    args = parser.parse_args()

    book_dir = Path(args.book_dir)
    csv_path = book_dir / "labeled" / "labels.csv"

    if not csv_path.exists():
        print(f"[ERROR] Khong tim thay: {csv_path}")
        sys.exit(1)

    # Font
    base = Path(__file__).parent
    font_path = str(base / "FontDiffusion" / "fonts" / "NomNaTong-Regular.ttf")
    font = load_font(font_path, size=80)

    # Dictionary
    try:
        from lib.dictionary import load_qn_to_nom
        dict_path = base / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"
        trans_dict = load_qn_to_nom(str(dict_path))
        print(f"  Dictionary: {len(trans_dict)} entries")
    except Exception as e:
        print(f"  [WARN] Dictionary not loaded: {e}")
        trans_dict = {}

    # Load labels
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.page:
                page_prefix = f"page_{args.page:04d}"
                if page_prefix not in row.get("image", ""):
                    continue

            if args.ambiguous_only:
                reading = row.get("reading", "").lower()
                candidates = trans_dict.get(reading, [])
                if len(candidates) <= 1:
                    continue

            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break

    print(f"  Labels to review: {len(rows)}")

    # Generate report
    output_path = book_dir / "labeled" / "visual_review.html"
    n_amb, n_unamb = generate_review_html(
        rows, trans_dict, font, output_path, book_dir,
    )

    print(f"\n  Ambiguous: {n_amb} (need review)")
    print(f"  Unambiguous: {n_unamb}")
    print(f"\n  Open: {output_path}")


if __name__ == "__main__":
    main()
