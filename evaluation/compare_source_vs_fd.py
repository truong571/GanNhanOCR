"""Compare font SOURCE char (input to FontDiffusion) vs fd_cache OUTPUT.

For each char:
  - Render the char from NomNaTong font at same size as fd_cache (96×96)
  - Load fd_cache[char] — what FontDiffusion produced from that source
  - Quantify stroke preservation:
      * Ink ratio (does output have less ink = strokes dropped?)
      * IoU of ink pixels
      * Stroke width statistics
      * DINOv2 cosine between the two
  - Save side-by-side visualization: source | fd_cache | dilated

Output:
    evaluation/reports/source_vs_fd_eval.md
    evaluation/reports/source_vs_fd_grid.png    (30 chars × 3 columns)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "evaluation" / "reports"
OUT.mkdir(parents=True, exist_ok=True)

CACHE_DIR = REPO / "prepared" / "_universal_fd_cache"
DILATED_DIR = REPO / "prepared" / "_universal_fd_cache_dilated"
FONT_PATH = REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"

CANVAS = 96


def render_source(ch: str) -> np.ndarray:
    """Render char from NomNaTong at CANVAS×CANVAS, centered."""
    font = ImageFont.truetype(str(FONT_PATH), int(CANVAS * 0.78))
    img = Image.new("L", (CANVAS, CANVAS), 255)
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), ch, font=font)
    x = (CANVAS - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (CANVAS - (bbox[3] - bbox[1])) / 2 - bbox[1]
    d.text((x, y), ch, fill=0, font=font)
    return np.array(img)


def ink_ratio(img: np.ndarray) -> float:
    _, bw = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return float(bw.sum()) / (img.shape[0] * img.shape[1])


def stroke_width_mean(img: np.ndarray) -> float:
    _, bw = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if bw.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    return float(dist[bw == 1].mean()) * 2


def iou_ink(a: np.ndarray, b: np.ndarray) -> float:
    _, ba = cv2.threshold(a, 128, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, bb = cv2.threshold(b, 128, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    inter = (ba & bb).sum()
    union = (ba | bb).sum()
    return float(inter) / float(union) if union else 0.0


def main() -> None:
    # Load DINOv2 for cosine
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading DINOv2 on {device}...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg",
                           verbose=False).to(device).eval()
    tx = T.Compose([
        T.ToTensor(), T.Resize(244, antialias=True), T.CenterCrop(224),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    @torch.no_grad()
    def embed(np_gray):
        pil = Image.fromarray(np_gray).convert("RGB")
        t = tx(pil).unsqueeze(0).to(device)
        e = model(t)
        if isinstance(e, tuple):
            e = e[0]
        return F.normalize(e.squeeze(0), dim=0)

    def cos(a, b):
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

    # Sample 30 random chars that exist in BOTH source font + fd_cache
    files = sorted(CACHE_DIR.glob("U+*.png"))
    random.seed(42)
    candidates = random.sample(files, min(150, len(files)))
    chosen = []
    for f in candidates:
        if len(chosen) >= 30:
            break
        try:
            cp = int(f.stem.replace("U+", ""), 16)
            ch = chr(cp)
        except (ValueError, OverflowError):
            continue
        src = render_source(ch)
        # Skip if font doesn't have this glyph (renders as box)
        if ink_ratio(src) < 0.005 or ink_ratio(src) > 0.5:
            continue
        chosen.append((ch, cp, f, src))

    print(f"Comparing {len(chosen)} chars (source font vs fd_cache)...\n")

    rows = []
    grid_samples = []
    for ch, cp, fd_path, src in chosen:
        cache = cv2.imread(str(fd_path), cv2.IMREAD_GRAYSCALE)
        dilated_p = DILATED_DIR / fd_path.name
        dilated = cv2.imread(str(dilated_p), cv2.IMREAD_GRAYSCALE) if dilated_p.exists() else None

        src_ink = ink_ratio(src)
        cache_ink = ink_ratio(cache)
        dilated_ink = ink_ratio(dilated) if dilated is not None else 0
        src_stroke = stroke_width_mean(src)
        cache_stroke = stroke_width_mean(cache)
        iou = iou_ink(src, cache)

        e_src = embed(src)
        e_cache = embed(cache)
        cos_sc = cos(e_src, e_cache)
        cos_sd = cos(e_src, embed(dilated)) if dilated is not None else None

        rows.append({
            "char": ch, "cp": f"U+{cp:04X}",
            "ink_src": round(src_ink * 100, 1),
            "ink_cache": round(cache_ink * 100, 1),
            "ink_dilated": round(dilated_ink * 100, 1) if dilated is not None else None,
            "stroke_src": round(src_stroke, 2),
            "stroke_cache": round(cache_stroke, 2),
            "ink_change_pct": round((cache_ink - src_ink) / src_ink * 100, 1) if src_ink else 0,
            "iou_src_vs_cache": round(iou, 3),
            "cos_src_vs_cache": round(cos_sc, 3),
            "cos_src_vs_dilated": round(cos_sd, 3) if cos_sd is not None else None,
        })
        grid_samples.append((ch, src, cache, dilated))

    # Aggregate
    avg_ink_src = np.mean([r["ink_src"] for r in rows])
    avg_ink_cache = np.mean([r["ink_cache"] for r in rows])
    avg_ink_change = np.mean([r["ink_change_pct"] for r in rows])
    avg_iou = np.mean([r["iou_src_vs_cache"] for r in rows])
    avg_cos = np.mean([r["cos_src_vs_cache"] for r in rows])
    n_under = sum(1 for r in rows if r["ink_change_pct"] < -20)
    n_over = sum(1 for r in rows if r["ink_change_pct"] > 20)

    print(f"=== Stroke preservation: source font vs fd_cache ({len(rows)} chars) ===\n")
    print(f"  Ink ratio:   source {avg_ink_src:.1f}% → fd_cache {avg_ink_cache:.1f}%  "
          f"(avg change {avg_ink_change:+.1f}%)")
    print(f"  IoU(src, cache):  {avg_iou:.3f}  (1.0 = identical pixels)")
    print(f"  DINOv2 cosine:    {avg_cos:.3f}  (1.0 = identical embedding)")
    print(f"  Chars with ink LOSS > 20%: {n_under}/{len(rows)} ({n_under/len(rows)*100:.0f}%)")
    print(f"  Chars with ink GAIN > 20%: {n_over}/{len(rows)} ({n_over/len(rows)*100:.0f}%)")

    print(f"\n=== Per-char details ===")
    print(f"  {'char':6s} {'cp':8s}  {'src_ink':>7s}  {'fd_ink':>6s}  {'change':>7s}  {'iou':>5s}  {'cos':>5s}")
    for r in sorted(rows, key=lambda x: x["ink_change_pct"]):
        print(f"  {r['char']:5s}  {r['cp']:8s}  {r['ink_src']:>6.1f}%  "
              f"{r['ink_cache']:>5.1f}%  {r['ink_change_pct']:+6.1f}%  "
              f"{r['iou_src_vs_cache']:.3f}  {r['cos_src_vs_cache']:.3f}")

    # Save grid: src | cache | dilated
    cell = 100
    n = len(grid_samples)
    grid_w = cell * 3 + 25
    grid_h = cell * n + (n + 1) * 5
    canvas = np.full((grid_h, grid_w), 255, dtype=np.uint8)
    for i, (ch, src, cache, dilated) in enumerate(grid_samples):
        y = 5 + i * (cell + 5)
        for j, im in enumerate([src, cache, dilated]):
            if im is None:
                continue
            x = 5 + j * (cell + 5)
            canvas[y:y + cell, x:x + cell] = cv2.resize(im, (cell, cell))
    cv2.imwrite(str(OUT / "source_vs_fd_grid.png"), canvas)

    # MD report
    md = [
        "# Source font vs fd_cache — stroke preservation check",
        "",
        "Compares the **input** to FontDiffusion (NomNaTong source render) vs the",
        "**output** (fd_cache PNG). Tells us if FontDiffusion is dropping strokes",
        "between input and output.",
        "",
        f"Sample: {len(rows)} random chars present in BOTH source font + fd_cache.",
        "",
        "## Aggregate",
        "",
        f"| Metric | source | fd_cache | change |",
        f"|--------|-------:|---------:|-------:|",
        f"| Ink ratio (mean %) | {avg_ink_src:.1f}% | {avg_ink_cache:.1f}% | {avg_ink_change:+.1f}% |",
        f"| IoU(source, fd_cache pixels) | — | — | **{avg_iou:.3f}** |",
        f"| DINOv2 cosine(source, fd_cache) | — | — | **{avg_cos:.3f}** |",
        f"| Chars with ink LOSS > 20% | — | — | {n_under}/{len(rows)} ({n_under/len(rows)*100:.0f}%) |",
        f"| Chars with ink GAIN > 20% | — | — | {n_over}/{len(rows)} ({n_over/len(rows)*100:.0f}%) |",
        "",
        "## Per-char details (sorted by ink change)",
        "",
        "| char | cp | source ink% | fd_cache ink% | change | IoU | cosine |",
        "|------|----|------------:|--------------:|-------:|----:|-------:|",
    ]
    for r in sorted(rows, key=lambda x: x["ink_change_pct"]):
        md.append(f"| {r['char']} | {r['cp']} | {r['ink_src']}% | "
                  f"{r['ink_cache']}% | **{r['ink_change_pct']:+.1f}%** | "
                  f"{r['iou_src_vs_cache']} | {r['cos_src_vs_cache']} |")

    md += [
        "",
        "## Interpretation",
        "",
        ("⚠️  fd_cache has MUCH less ink than source — significant stroke loss."
         if avg_ink_change < -25 else
         "⚠️  fd_cache loses some ink vs source (10-25% reduction) — minor strokes dropped."
         if avg_ink_change < -10 else
         "✓ fd_cache ink ratio close to source (within ±10%) — minimal stroke loss."
         if abs(avg_ink_change) < 10 else
         "⚠️  fd_cache has MORE ink than source — FontDiffusion adds artifacts."),
        "",
        ("- IoU < 0.30 = pixels barely overlap (style change drastic)" if avg_iou < 0.30 else
         "- IoU 0.30-0.50 = moderate overlap (expected for style transfer)" if avg_iou < 0.50 else
         "- IoU > 0.50 = high overlap (output close to source)"),
        "",
        ("- DINOv2 cosine < 0.6 = embedding very different (chars may be confusable)"
         if avg_cos < 0.6 else
         "- DINOv2 cosine 0.6-0.8 = embedding similar (style change but identity preserved)"
         if avg_cos < 0.8 else
         "- DINOv2 cosine > 0.8 = embedding nearly identical"),
        "",
        "## Visual",
        "",
        "`source_vs_fd_grid.png` — left=NomNaTong source, mid=fd_cache, right=dilated.",
        "Inspect: do the strokes in fd_cache MATCH the source? Are any minor",
        "strokes (radicals, dots) missing?",
    ]

    (OUT / "source_vs_fd_eval.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT / 'source_vs_fd_eval.md'}")
    print(f"      {OUT / 'source_vs_fd_grid.png'}")


if __name__ == "__main__":
    main()
