"""Pick 1 medoid style image from 5 crops × 6 books = 30 candidates.

Workflow:
  1. For each book in config/pipeline.yaml, sample N crops from
     prepared/<book>/detected/crops_cleaned/ (deterministic seed; biased
     toward larger crops to avoid noise fragments).
  2. Encode all 30 with FontDiffusion's StyleEncoder (PROD weights).
  3. Find the crop whose pooled style vector is closest to the centroid
     (cosine distance) — that is the medoid.
  4. Copy it as style_references/medoid.png and write a debug grid.

Usage:
    PATH="$PWD/.venv/bin:$PATH" python kaggle_diffusion/build_style_medoid.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "font_diffusion"))

from safetensors.torch import load_file as safe_load
from src.modules.style_encoder import StyleEncoder


def is_single_char(path: Path) -> bool:
    # Reject crops that are unlikely to be a clean single chu-Nom char:
    #   - too small (segmentation fragment)
    #   - aspect ratio far from square (two chars merged, or strip noise)
    #   - too little ink (mostly paper) or too much (mostly black smudge)
    try:
        img = np.array(Image.open(path).convert("L"))
    except Exception:
        return False
    h, w = img.shape
    if min(h, w) < 40:
        return False
    ar = w / h
    if ar < 0.75 or ar > 1.35:
        return False
    ink = (img < 128).mean()
    if ink < 0.10 or ink > 0.50:
        return False
    return True


def find_book_crops(book: str, n: int, rng: random.Random) -> list[Path]:
    # Use detected/crops/ (raw color crop from pages/<p>.png) — NOT crops_cleaned/,
    # which has been Sauvola-binarized and resized to 64×64. The medoid must come
    # from the original render so style_encoder sees real ink texture / paper noise.
    crops_dir = PROJECT_ROOT / "prepared" / book / "detected" / "crops"
    if not crops_dir.exists():
        raise FileNotFoundError(
            f"{crops_dir} missing — run `./run_pipeline.sh --step 1 --book {book}` "
            f"then `--step 2 --book {book}` first."
        )
    pngs = sorted(crops_dir.rglob("*.png"))
    if not pngs:
        raise RuntimeError(f"{book}: no crops")

    # Two-stage filter:
    #   1. Drop crops that aren't a clean single char (size / ratio / ink).
    #   2. Among survivors, take top by file size (proxy for sharpness/ink), then random.
    valid = [p for p in pngs if is_single_char(p)]
    if len(valid) < n:
        raise RuntimeError(
            f"{book}: only {len(valid)}/{len(pngs)} crops pass single-char filter, need ≥{n}"
        )
    by_size = sorted(valid, key=lambda p: p.stat().st_size, reverse=True)
    pool = by_size[: max(n * 4, len(by_size) // 3)]
    return rng.sample(pool, n)


def encode(
    encoder: StyleEncoder,
    paths: list[Path],
    device: str,
    size: int,
) -> torch.Tensor:
    tx = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    vecs = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = tx(img).unsqueeze(0).to(device)
            _, h, _ = encoder(x)
            vecs.append(h.squeeze(0).cpu())
    return torch.stack(vecs, dim=0)  # [N, C]


def find_medoid(vecs: torch.Tensor) -> int:
    v = F.normalize(vecs, dim=1)
    centroid = F.normalize(v.mean(dim=0, keepdim=True), dim=1)
    sims = (v @ centroid.T).squeeze(1)  # cosine similarity to centroid
    return int(sims.argmax().item())


def save_debug_grid(paths: list[Path], medoid_idx: int, out: Path, cell: int = 128) -> None:
    cols = 10 if len(paths) > 30 else 5
    rows = (len(paths) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * cell), "white")
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB").resize((cell, cell))
        if i == medoid_idx:
            # Red border = medoid
            from PIL import ImageDraw

            d = ImageDraw.Draw(img)
            for k in range(4):
                d.rectangle([k, k, cell - 1 - k, cell - 1 - k], outline="red")
        grid.paste(img, ((i % cols) * cell, (i // cols) * cell))
    grid.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--per-book", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="font_diffusion/ckpt/PROD/style_encoder.safetensors")
    ap.add_argument("--size", type=int, default=96, help="StyleEncoder resolution (training default = 96)")
    ap.add_argument("--out", default="kaggle_diffusion/style_references/medoid.png")
    args = ap.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    books = [b["name"] for b in cfg["books"]]
    rng = random.Random(args.seed)

    print(f"Books: {books}  |  per-book: {args.per_book}  |  total: {args.per_book * len(books)}")
    paths: list[Path] = []
    book_of: list[str] = []
    for b in books:
        chosen = find_book_crops(b, args.per_book, rng)
        paths.extend(chosen)
        book_of.extend([b] * len(chosen))
        names = [p.name for p in chosen]
        preview = names[:3] + ["…"] + names[-2:] if len(names) > 6 else names
        print(f"  {b}: {len(names)} crops ({', '.join(preview)})")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nLoading StyleEncoder (G_ch=64, resolution={args.size}) on {device}…")
    enc = StyleEncoder(G_ch=64, resolution=args.size).to(device).eval()
    state = safe_load(str(PROJECT_ROOT / args.ckpt))
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected (ok if minor)")

    print(f"Encoding {len(paths)} crops…")
    vecs = encode(enc, paths, device, args.size)

    medoid_idx = find_medoid(vecs)
    medoid_path = paths[medoid_idx]
    medoid_book = book_of[medoid_idx]

    print(f"\n✓ Medoid: {medoid_book}/{medoid_path.name}")
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(medoid_path, out)
    print(f"  → {out.relative_to(PROJECT_ROOT)}")

    debug = out.with_name("_medoid_grid.png")
    save_debug_grid(paths, medoid_idx, debug)
    print(f"  → {debug.relative_to(PROJECT_ROOT)} (red border = medoid)")

    manifest = out.with_suffix(".json")
    manifest.write_text(
        json.dumps(
            {
                "medoid": {"book": medoid_book, "path": str(medoid_path.relative_to(PROJECT_ROOT))},
                "candidates": [
                    {"book": book_of[i], "path": str(paths[i].relative_to(PROJECT_ROOT))}
                    for i in range(len(paths))
                ],
                "config": {"per_book": args.per_book, "seed": args.seed, "size": args.size},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"  → {manifest.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
