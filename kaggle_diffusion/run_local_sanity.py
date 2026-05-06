"""Local sanity test — mirror of notebook cell 7.5.

Generates 5 reference chu-Nom characters using the production FontDiffusion
weights and the same hyperparameters as the Kaggle universal cache run.

Use this before kicking off the full 10+ hour Kaggle job to verify the pipeline
end-to-end on your Mac.

First run:  ~3 minutes (downloads ~400 MB of weights)
Subsequent runs: ~70 seconds (5 chars × ~12 s/char on CPU/MPS)

Usage:
    PATH="$PWD/.venv/bin:$PATH" python kaggle_diffusion/run_local_sanity.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.ranking.fontdiffusion_gen import FontDiffusionGenerator

# ─── Tuned defaults (match diffusion_run.ipynb cell 7.5) ─────────────────────
HF_REPO = "dzungpham/font-diffusion-weights"
STYLE_BOOK = "SachThanhTruyen2"
GUIDANCE_SCALE = 2.0
ERODE_ITERS = 2
TEST_CHARS = ["一", "二", "三", "人", "月"]

CKPT_DIR = PROJECT_ROOT / "font_diffusion" / "ckpt" / "PROD"
FONT_PATH = PROJECT_ROOT / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"
STYLE_PATH = PROJECT_ROOT / "kaggle_diffusion" / "style_references" / f"{STYLE_BOOK}.png"
OUT_DIR = PROJECT_ROOT / "kaggle_diffusion" / "sanity_test_output"


def fetch_weights() -> None:
    """Download production weights from HF root (not /FST/checkpoint_step_1500)."""
    from huggingface_hub import hf_hub_download

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    files = ["unet.safetensors", "style_encoder.safetensors", "content_encoder.safetensors"]
    for fn in files:
        dst = CKPT_DIR / fn
        if dst.exists():
            continue
        print(f"  fetching {fn}...", flush=True)
        cached = hf_hub_download(repo_id=HF_REPO, filename=fn)
        shutil.copy2(cached, dst)
    print(f"  ✓ weights ready at {CKPT_DIR}")


def erode(path: Path, iters: int) -> None:
    if iters <= 0:
        return
    arr = np.array(Image.open(path).convert("L"))
    arr = cv2.dilate(arr, np.ones((3, 3), np.uint8), iterations=iters)
    Image.fromarray(arr).save(path)


def main() -> None:
    print(f"Style:           {STYLE_PATH.name}")
    print(f"guidance_scale:  {GUIDANCE_SCALE}")
    print(f"erode_iters:     {ERODE_ITERS}")
    print(f"chars:           {TEST_CHARS}")
    print()

    print("─── checking production weights ───")
    fetch_weights()

    print("\n─── loading FontDiffusion ───")
    torch.manual_seed(42)
    gen = FontDiffusionGenerator(
        ckpt_dir=str(CKPT_DIR),
        phase1_ckpt_dir=str(CKPT_DIR),
        font_path=str(FONT_PATH),
        cache_dir=str(OUT_DIR.parent / "_local_cache"),
        batch_size=2,
    )
    gen._load_pipeline()
    gen.pipe.guidance_scale = GUIDANCE_SCALE
    print(f"  device = {gen.device}")

    print("\n─── generating ───")
    t0 = time.time()
    gen.generate(TEST_CHARS, str(STYLE_PATH), style_name="_sanity")
    print(f"  done in {time.time() - t0:.1f}s")

    print("\n─── post-processing (erosion) + copying to sanity_test_output/ ───")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_dir = OUT_DIR.parent / "_local_cache" / "_sanity"
    for c in TEST_CHARS:
        src = src_dir / f"U+{ord(c):04X}.png"
        dst = OUT_DIR / f"U+{ord(c):04X}.png"
        if src.exists():
            shutil.copy2(src, dst)
            erode(dst, ERODE_ITERS)
            arr = np.array(Image.open(dst).convert("L"))
            print(f"  {c}  ink={(arr < 128).sum():4d}  std={arr.std():.0f}  → {dst}")
        else:
            print(f"  {c}  MISSING")

    shutil.rmtree(src_dir.parent, ignore_errors=True)
    print(f"\n✓ Sanity outputs in {OUT_DIR}")
    print("  Open them to verify chu-Nom-style characters before running Kaggle.")


if __name__ == "__main__":
    main()
