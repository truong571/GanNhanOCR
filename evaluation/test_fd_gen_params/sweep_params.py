"""Sweep FontDiffusion generation params to find the best stroke-preservation
combo before pushing a long Kaggle run.

For each (content_image_size, num_inference_steps, guidance_scale) combo:
  - Generate 5 test chars (1 simple, 2 medium, 2 complex)
  - Measure:
      * Ink ratio change vs source font (negative = stroke loss)
      * IoU(source, generated) pixels
      * DINOv2 cosine(source, generated)
      * Per-image generation time
  - Save side-by-side grid for visual inspection

Output:
    evaluation/test_fd_gen_params/out/sweep_metrics.json
    evaluation/test_fd_gen_params/out/sweep_grid.png   (5 chars × N variants)
    evaluation/test_fd_gen_params/out/sweep_report.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
FONT_PATH = REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"
STYLE_IMAGE = REPO / "kaggle_diffusion" / "style_references" / "medoid.png"
CKPT_DIR = REPO / "font_diffusion" / "ckpt" / "PROD"

# Test chars: 1 simple, 2 medium, 2 complex (different stroke densities)
TEST_CHARS = ["二", "徳", "經", "鬱", "𡗶"]   # 2 strokes / 14 / 13 / 29 / Nôm "trời"

# Variants: (size, steps, guidance, label)
VARIANTS = [
    (96,  20, 2.0, "baseline"),       # CURRENT pipeline default
    (96,  40, 2.5, "more_steps_guid"),
    (128, 40, 2.5, "bigger_size"),
    (160, 40, 2.5, "biggest_size"),
    (128, 60, 3.0, "aggressive"),
]


def render_source(ch: str, canvas: int = 128) -> np.ndarray:
    """Render char from NomNaTong at fixed canvas size."""
    font = ImageFont.truetype(str(FONT_PATH), int(canvas * 0.78))
    img = Image.new("L", (canvas, canvas), 255)
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), ch, font=font)
    x = (canvas - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (canvas - (bbox[3] - bbox[1])) / 2 - bbox[1]
    d.text((x, y), ch, fill=0, font=font)
    return np.array(img)


def ink_ratio(img: np.ndarray) -> float:
    _, bw = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return float(bw.sum()) / (img.shape[0] * img.shape[1])


def iou_ink(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    _, ba = cv2.threshold(a, 128, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, bb = cv2.threshold(b, 128, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    inter = (ba & bb).sum()
    union = (ba | bb).sum()
    return float(inter) / float(union) if union else 0.0


def main() -> None:
    # DINOv2 once
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading DINOv2 on {device}...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg",
                          verbose=False).to(device).eval()
    tx = T.Compose([
        T.ToTensor(), T.Resize(244, antialias=True), T.CenterCrop(224),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    @torch.no_grad()
    def embed_gray(arr):
        pil = Image.fromarray(arr).convert("RGB")
        t = tx(pil).unsqueeze(0).to(device)
        e = dino(t)
        if isinstance(e, tuple):
            e = e[0]
        return F.normalize(e.squeeze(0), dim=0)

    def cos(a, b):
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

    # Pre-render source font images (use 128 canvas as common reference)
    print(f"\nPre-rendering {len(TEST_CHARS)} source font images (canvas=128)...")
    source_imgs = {ch: render_source(ch, canvas=128) for ch in TEST_CHARS}

    # Load FontDiffusion ONCE per size (reload only when size changes)
    from core.ranking.fontdiffusion_gen import FontDiffusionGenerator

    results = []
    cache_by_size: dict[int, FontDiffusionGenerator] = {}

    for size, steps, guidance, label in VARIANTS:
        print(f"\n{'='*70}")
        print(f"Variant: size={size}  steps={steps}  guidance={guidance}  ({label})")
        print(f"{'='*70}")

        if size not in cache_by_size:
            print(f"  Loading FontDiffusion at size={size}... (~30s)")
            t0 = time.time()
            gen = FontDiffusionGenerator(
                ckpt_dir=str(CKPT_DIR),
                phase1_ckpt_dir=str(CKPT_DIR),
                font_path=str(FONT_PATH),
                cache_dir=str(OUT / f"_size{size}"),
            )
            # Override args before pipeline load
            gen._build_args = lambda gen=gen, size=size: _override_args(gen, size)
            gen._load_pipeline()
            print(f"  ✓ Loaded in {time.time()-t0:.1f}s")
            cache_by_size[size] = gen
        gen = cache_by_size[size]

        # Override per-call params
        gen.args.num_inference_steps = steps
        try:
            gen.pipe.guidance_scale = guidance
        except Exception:
            pass

        # Generate test chars
        style_path = str(STYLE_IMAGE)
        style_name = f"{label}_{size}_{steps}_{guidance}"
        # Clear cache for fresh run
        sub_cache = OUT / f"_size{size}" / style_name
        if sub_cache.exists():
            for f in sub_cache.glob("*.png"):
                f.unlink()

        t_gen0 = time.time()
        try:
            gen.generate(TEST_CHARS, style_path, style_name=style_name)
        except Exception as e:
            print(f"  ✗ Generation failed: {type(e).__name__}: {e}")
            results.append({"variant": label, "error": str(e)[:200]})
            continue
        t_gen = time.time() - t_gen0
        per_img = t_gen / len(TEST_CHARS)
        print(f"  Generated {len(TEST_CHARS)} chars in {t_gen:.1f}s ({per_img:.1f}s/img)")

        # Measure each generated char
        for ch in TEST_CHARS:
            cp = ord(ch)
            gen_path = sub_cache / f"U+{cp:04X}.png"
            if not gen_path.exists():
                continue
            gen_img = cv2.imread(str(gen_path), cv2.IMREAD_GRAYSCALE)
            src_img = source_imgs[ch]

            # Resize gen to match src for comparison
            if gen_img.shape != src_img.shape:
                gen_cmp = cv2.resize(gen_img, src_img.shape[::-1])
            else:
                gen_cmp = gen_img
            src_ink = ink_ratio(src_img)
            gen_ink = ink_ratio(gen_cmp)
            iou = iou_ink(src_img, gen_cmp)
            cos_val = cos(embed_gray(src_img), embed_gray(gen_cmp))

            results.append({
                "variant": label,
                "size": size, "steps": steps, "guidance": guidance,
                "char": ch, "cp": f"U+{cp:04X}",
                "time_per_img_s": round(per_img, 2),
                "ink_src_pct": round(src_ink * 100, 1),
                "ink_gen_pct": round(gen_ink * 100, 1),
                "ink_change_pct": round((gen_ink - src_ink) / max(src_ink, 1e-6) * 100, 1),
                "iou": round(iou, 3),
                "cos_dinov2": round(cos_val, 3),
            })
            print(f"  {ch}: src_ink={src_ink*100:.1f}% gen_ink={gen_ink*100:.1f}% "
                  f"({(gen_ink-src_ink)/max(src_ink,1e-6)*100:+.1f}%) "
                  f"IoU={iou:.3f} cos={cos_val:.3f}")

    # Save results
    (OUT / "sweep_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT / 'sweep_metrics.json'}")

    # Build comparison grid: rows = chars, cols = source + each variant
    cell = 100
    n_cols = 1 + len(VARIANTS)  # source + variants
    n_rows = len(TEST_CHARS)
    canvas = np.full((cell * n_rows + (n_rows + 1) * 5,
                      cell * n_cols + (n_cols + 1) * 5), 255, dtype=np.uint8)
    for i, ch in enumerate(TEST_CHARS):
        y = 5 + i * (cell + 5)
        canvas[y:y + cell, 5:5 + cell] = cv2.resize(source_imgs[ch], (cell, cell))
        for j, (size, steps, guidance, label) in enumerate(VARIANTS):
            sub_cache = OUT / f"_size{size}" / f"{label}_{size}_{steps}_{guidance}"
            gen_path = sub_cache / f"U+{ord(ch):04X}.png"
            x = 5 + (j + 1) * (cell + 5)
            if gen_path.exists():
                im = cv2.imread(str(gen_path), cv2.IMREAD_GRAYSCALE)
                canvas[y:y + cell, x:x + cell] = cv2.resize(im, (cell, cell))
    cv2.imwrite(str(OUT / "sweep_grid.png"), canvas)
    print(f"      {OUT / 'sweep_grid.png'}")

    # Aggregate per variant
    by_var: dict[str, list] = {}
    for r in results:
        if "error" in r:
            continue
        by_var.setdefault(r["variant"], []).append(r)

    md = [
        "# FontDiffusion param sweep — find best stroke preservation",
        "",
        f"Test chars: {', '.join(TEST_CHARS)}  (mixed: simple → complex)",
        f"Style ref: medoid.png (same as production)",
        "",
        "## Aggregate per variant",
        "",
        "| Variant | size | steps | guid | time/img | ink change | IoU | cosine |",
        "|---------|-----:|------:|-----:|---------:|-----------:|----:|-------:|",
    ]
    for label, runs in by_var.items():
        if not runs:
            continue
        avg_time = np.mean([r["time_per_img_s"] for r in runs])
        avg_ink = np.mean([r["ink_change_pct"] for r in runs])
        avg_iou = np.mean([r["iou"] for r in runs])
        avg_cos = np.mean([r["cos_dinov2"] for r in runs])
        size = runs[0]["size"]; steps = runs[0]["steps"]; g = runs[0]["guidance"]
        md.append(f"| {label} | {size} | {steps} | {g} | "
                  f"{avg_time:.1f}s | {avg_ink:+.1f}% | "
                  f"{avg_iou:.3f} | {avg_cos:.3f} |")

    md += ["",
           "## Per-char detail",
           "",
           "| variant | char | ink src | ink gen | change | IoU | cosine |",
           "|---------|------|--------:|--------:|-------:|----:|-------:|"]
    for r in results:
        if "error" in r:
            continue
        md.append(f"| {r['variant']} | {r['char']} | {r['ink_src_pct']}% | "
                  f"{r['ink_gen_pct']}% | {r['ink_change_pct']:+.1f}% | "
                  f"{r['iou']} | {r['cos_dinov2']} |")
    md += ["",
           "## Visual",
           "",
           "`sweep_grid.png` — rows = chars, cols = source | each variant.",
           "",
           "## Decision criteria",
           "",
           "- **ink change closest to 0%** = best stroke preservation",
           "- **IoU highest** = generated overlaps source pixels well",
           "- **cosine highest** = embedding identity preserved",
           "- **time/img reasonable** = full Kaggle run feasible (<60h on P100)"]

    (OUT / "sweep_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"      {OUT / 'sweep_report.md'}")


def _override_args(gen, size: int):
    """Patch _build_args to use custom image size."""
    from argparse import Namespace
    from src.configs.fontdiffuser import get_parser
    parser = get_parser()
    args = parser.parse_args([])
    args.ckpt_dir = gen.ckpt_dir
    args.phase_1_ckpt_dir = gen.phase1_ckpt_dir
    args.fst_ckpt_path = gen.ckpt_dir
    args.ttf_path = gen.font_path
    args.device = gen.device
    args.use_fst = False
    args.batch_size = gen.batch_size
    args.character_input = True
    args.save_image = False
    args.style_image_size = (size, size)
    args.content_image_size = (size, size)
    return args


if __name__ == "__main__":
    main()
