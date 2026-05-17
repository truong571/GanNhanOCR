"""Quantify stroke loss in fd_cache vs source font.

3 tests:
  (1) Random-sample 30 chars: pixel-level diff vs font-rendered + ink ratio loss
  (2) Confusable pairs: does fd_cache(X) look more like font(X) or font(Y)?
      (e.g., 大/天 differ by ONE horizontal stroke at top)
  (3) Visual grid: write side-by-side comparison images to evaluation/reports/

Output:
    evaluation/reports/fd_cache_stroke_eval.md
    evaluation/reports/fd_cache_grid.png        (30 chars × 2 columns)
    evaluation/reports/fd_cache_confusable.png  (8 pairs × 4 columns)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
FD_CACHE = REPO / "prepared" / "_universal_fd_cache"
FD_DILATED = REPO / "prepared" / "_universal_fd_cache_dilated"
NOMNATONG = REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"
OUT = REPO / "evaluation" / "reports"
OUT.mkdir(parents=True, exist_ok=True)


# Confusable pairs — each differs by 1-2 strokes
# Format: (char_a, char_b, why)
CONFUSABLE_PAIRS = [
    ("大", "天", "天 has 1 extra horizontal stroke at top"),
    ("大", "夫", "夫 has 1 extra horizontal across"),
    ("千", "干", "千 has slanted top, 干 horizontal"),
    ("土", "士", "士 top horizontal longer than bottom"),
    ("人", "入", "入 has left stroke crossing"),
    ("日", "目", "目 has 1 extra horizontal"),
    ("石", "右", "right side differs"),
    ("未", "末", "末 top longer than middle"),
]


def render_from_font(ch: str, size: int = 96) -> np.ndarray:
    """Return grayscale 96×96 of `ch` rendered from NomNaTong."""
    font = ImageFont.truetype(str(NOMNATONG), int(size * 0.75))
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), ch, font=font)
        x = (size - (bbox[2] - bbox[0])) / 2 - bbox[0]
        y = (size - (bbox[3] - bbox[1])) / 2 - bbox[1]
        draw.text((x, y), ch, fill=0, font=font)
    except Exception:
        pass
    return np.array(img)


def ink_ratio(img: np.ndarray) -> float:
    _, bw = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return float(bw.sum()) / (img.shape[0] * img.shape[1])


def stroke_width(img: np.ndarray) -> float:
    _, bw = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if bw.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    return float(dist[bw == 1].mean()) * 2


def iou_ink(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of ink pixels — measures structural agreement."""
    _, ba = cv2.threshold(a, 128, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, bb = cv2.threshold(b, 128, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    inter = (ba & bb).sum()
    union = (ba | bb).sum()
    return float(inter) / float(union) if union else 0.0


def pixel_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a.astype(int) - b.astype(int)) ** 2).mean())


def load_cache(ch: str, cache_dir: Path) -> np.ndarray | None:
    p = cache_dir / f"U+{ord(ch):04X}.png"
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return img


def test1_random_sample(n: int = 30) -> tuple[list[dict], list]:
    """Compare ink/stroke/IoU for 30 random fd_cache chars vs font."""
    files = sorted(FD_CACHE.glob("U+*.png"))
    random.seed(42)
    sample = random.sample(files, n)
    rows = []
    grid_samples = []
    for f in sample:
        cp = int(f.stem.replace("U+", ""), 16)
        ch = chr(cp)
        cache_img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        font_img = render_from_font(ch)
        dilated_img = load_cache(ch, FD_DILATED)

        rows.append({
            "char": ch,
            "cp": f.stem,
            "ink_cache": round(ink_ratio(cache_img) * 100, 1),
            "ink_dilated": round(ink_ratio(dilated_img) * 100, 1) if dilated_img is not None else None,
            "ink_font": round(ink_ratio(font_img) * 100, 1),
            "stroke_cache": round(stroke_width(cache_img), 2),
            "stroke_font": round(stroke_width(font_img), 2),
            "iou_cache_vs_font": round(iou_ink(cache_img, font_img), 3),
            "iou_dilated_vs_font": round(iou_ink(dilated_img, font_img), 3) if dilated_img is not None else None,
        })
        grid_samples.append((ch, font_img, cache_img, dilated_img))
    return rows, grid_samples


def test2_confusable() -> list[dict]:
    """For each (X, Y) pair: is fd_cache(X) closer to font(X) or font(Y)?"""
    results = []
    for a, b, why in CONFUSABLE_PAIRS:
        fa, fb = render_from_font(a), render_from_font(b)
        ca = load_cache(a, FD_CACHE)
        cb = load_cache(b, FD_CACHE)
        if ca is None or cb is None:
            results.append({"a": a, "b": b, "why": why, "status": "missing in cache"})
            continue
        # cache vs each font option
        results.append({
            "a": a, "b": b, "why": why,
            "iou_ca_vs_fa": round(iou_ink(ca, fa), 3),  # cache(a) vs font(a) — should be HIGH
            "iou_ca_vs_fb": round(iou_ink(ca, fb), 3),  # cache(a) vs font(b) — should be LOW
            "iou_cb_vs_fb": round(iou_ink(cb, fb), 3),
            "iou_cb_vs_fa": round(iou_ink(cb, fa), 3),
            "a_correct": iou_ink(ca, fa) > iou_ink(ca, fb),
            "b_correct": iou_ink(cb, fb) > iou_ink(cb, fa),
        })
    return results


def make_grid(samples: list, path: Path, cols: int = 3) -> None:
    """Save side-by-side grid: each row = (font, cache, dilated)."""
    n = len(samples)
    cell = 100
    grid_w = cell * cols + (cols + 1) * 5
    grid_h = cell * n + (n + 1) * 5
    canvas = np.full((grid_h, grid_w), 255, dtype=np.uint8)
    for i, (ch, font_img, cache_img, dilated_img) in enumerate(samples):
        y = 5 + i * (cell + 5)
        for j, im in enumerate([font_img, cache_img, dilated_img]):
            if im is None:
                continue
            x = 5 + j * (cell + 5)
            tile = cv2.resize(im, (cell, cell))
            canvas[y:y + cell, x:x + cell] = tile
    cv2.imwrite(str(path), canvas)


def make_confusable_grid(pairs: list[dict], path: Path) -> None:
    """Per pair: font(a), cache(a), font(b), cache(b)."""
    cell = 80
    rows = [p for p in pairs if "iou_ca_vs_fa" in p]
    grid_w = cell * 4 + 25
    grid_h = cell * len(rows) + (len(rows) + 1) * 5
    canvas = np.full((grid_h, grid_w), 255, dtype=np.uint8)
    for i, p in enumerate(rows):
        y = 5 + i * (cell + 5)
        ims = [
            render_from_font(p["a"]), load_cache(p["a"], FD_CACHE),
            render_from_font(p["b"]), load_cache(p["b"], FD_CACHE),
        ]
        for j, im in enumerate(ims):
            if im is None: continue
            x = 5 + j * (cell + 5)
            canvas[y:y + cell, x:x + cell] = cv2.resize(im, (cell, cell))
    cv2.imwrite(str(path), canvas)


def main() -> None:
    print("Test 1: random sample (30 chars)")
    rows, grid_samples = test1_random_sample(30)

    avg_cache_ink   = np.mean([r["ink_cache"] for r in rows])
    avg_dilated_ink = np.mean([r["ink_dilated"] for r in rows if r["ink_dilated"]])
    avg_font_ink    = np.mean([r["ink_font"] for r in rows])
    avg_iou_cache   = np.mean([r["iou_cache_vs_font"] for r in rows])
    avg_iou_dilated = np.mean([r["iou_dilated_vs_font"] for r in rows if r["iou_dilated_vs_font"]])

    # IoU bands
    iou_vals = [r["iou_cache_vs_font"] for r in rows]
    bad = sum(1 for v in iou_vals if v < 0.40)
    medium = sum(1 for v in iou_vals if 0.40 <= v < 0.55)
    good = sum(1 for v in iou_vals if v >= 0.55)

    print(f"  ink ratio  cache={avg_cache_ink:.1f}%  dilated={avg_dilated_ink:.1f}%  font={avg_font_ink:.1f}%")
    print(f"  IoU vs font  cache={avg_iou_cache:.3f}  dilated={avg_iou_dilated:.3f}")
    print(f"  IoU buckets: bad<0.40: {bad}  medium 0.40-0.55: {medium}  good>=0.55: {good}")

    print()
    print("Test 2: confusable pairs")
    pairs = test2_confusable()
    correct_count = sum(1 for p in pairs if p.get("a_correct") and p.get("b_correct"))
    print(f"  {correct_count}/{len(pairs)} pairs unambiguously matched correct identity")
    for p in pairs:
        if "iou_ca_vs_fa" in p:
            same = p["iou_ca_vs_fa"]
            other = p["iou_ca_vs_fb"]
            flag = "OK" if same > other else "⚠️  WRONG"
            print(f"  {p['a']} ({p['cp_a'] if 'cp_a' in p else hex(ord(p['a']))}) "
                  f"vs {p['b']}: cache({p['a']}) vs font({p['a']})={same:.3f}, "
                  f"vs font({p['b']})={other:.3f}  [{flag}]")

    print()
    print("Writing grids...")
    make_grid(grid_samples, OUT / "fd_cache_grid.png")
    make_confusable_grid(pairs, OUT / "fd_cache_confusable.png")

    md = [
        "# fd_cache stroke loss evaluation",
        "",
        "Sample: 30 random chars + 8 confusable pairs.",
        "Columns: font (NomNaTong baseline) | fd_cache | fd_cache_dilated.",
        "",
        "## Aggregate (Test 1)",
        "",
        f"- Ink ratio:  cache **{avg_cache_ink:.1f}%**, dilated **{avg_dilated_ink:.1f}%**, font **{avg_font_ink:.1f}%**",
        f"- IoU vs font:  cache **{avg_iou_cache:.3f}**, dilated **{avg_iou_dilated:.3f}**",
        f"- IoU buckets (cache vs font):  bad <0.40: **{bad}**  medium 0.40-0.55: **{medium}**  good ≥0.55: **{good}**",
        "",
        "**Interpretation:**",
        "- IoU < 0.40 = severe stroke loss, char likely unrecognizable",
        "- IoU 0.40-0.55 = noticeable degradation, may confuse DINOv2",
        "- IoU ≥ 0.55 = acceptable for visual matching",
        "",
        "## Per-char sample (Test 1)",
        "",
        "| char | ink cache % | ink dilated % | ink font % | IoU cache | IoU dilated |",
        "|------|------------:|--------------:|-----------:|----------:|------------:|",
    ]
    for r in rows:
        md.append(f"| {r['char']} ({r['cp']}) | {r['ink_cache']} | "
                  f"{r['ink_dilated'] or '-'} | {r['ink_font']} | "
                  f"{r['iou_cache_vs_font']} | {r['iou_dilated_vs_font'] or '-'} |")

    md += ["", "## Confusable pairs (Test 2)", "",
           f"**{correct_count}/{len(pairs)} pairs** unambiguously matched correct identity.",
           "",
           "| pair | why | IoU cache(a)·font(a) | IoU cache(a)·font(b) | correct? |",
           "|------|-----|---------------------:|---------------------:|---------|"]
    for p in pairs:
        if "iou_ca_vs_fa" not in p:
            md.append(f"| {p['a']}/{p['b']} | {p['why']} | - | - | {p['status']} |")
            continue
        ok = "✓" if p["iou_ca_vs_fa"] > p["iou_ca_vs_fb"] else "✗ WRONG"
        md.append(f"| {p['a']}/{p['b']} | {p['why']} | {p['iou_ca_vs_fa']} | "
                  f"{p['iou_ca_vs_fb']} | {ok} |")

    md += ["", "## Visual",
           "- `fd_cache_grid.png`        — 30 chars × 3 columns (font / cache / dilated)",
           "- `fd_cache_confusable.png` — 8 confusable pairs × 4 columns"]

    (OUT / "fd_cache_stroke_eval.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT / 'fd_cache_stroke_eval.md'}")
    print(f"      {OUT / 'fd_cache_grid.png'}")
    print(f"      {OUT / 'fd_cache_confusable.png'}")


if __name__ == "__main__":
    main()
