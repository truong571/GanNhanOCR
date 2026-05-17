"""Deep test: fd_cache image vs ACTUAL book crops (not font baseline).

This is more directly relevant than the font-baseline test — pipeline's Tier 3
compares fd_cache[char] vs real crop_image[char] using DINOv2 cosine.

Tests:
  A. Average cosine(fd_cache[ch], real_crop[ch]) over 100 random labeled crops
     → how well does fd_cache RECOGNIZE the right char in the book?
  B. Compare fd_cache vs fd_cache_dilated — which one matches real crops better?
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

REPO = Path(__file__).resolve().parent.parent

OUT = REPO / "evaluation" / "reports"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg",
                           verbose=False).to(device).eval()
    transform = T.Compose([
        T.ToTensor(), T.Resize(244, antialias=True), T.CenterCrop(224),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    @torch.no_grad()
    def embed(pil):
        t = transform(pil).unsqueeze(0).to(device)
        e = model(t)
        if isinstance(e, tuple):
            e = e[0]
        return F.normalize(e.squeeze(0), dim=0)

    def cos(a, b):
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

    # Pick 100 labeled crops (matched=True for cleaner signal)
    df = pd.read_csv(REPO / "dataset" / "all" / "labels.csv")
    df = df[df["matched"] == True].reset_index(drop=True)
    random.seed(42)
    sample = df.sample(n=100, random_state=42).to_dict("records")

    cache_dir = REPO / "prepared" / "_universal_fd_cache"
    dilated_dir = REPO / "prepared" / "_universal_fd_cache_dilated"

    sims_cache, sims_dilated = [], []
    n_cache_missing = 0
    print("Embedding 100 real crops + their fd_cache references...")
    for i, r in enumerate(sample):
        ch = r["nom_char"]
        if not ch or len(ch) != 1:
            continue
        cp = ord(ch)
        crop_path = REPO / "dataset" / r["source"] / r["crop_file"]
        cache_path = cache_dir / f"U+{cp:04X}.png"
        dilated_path = dilated_dir / f"U+{cp:04X}.png"
        if not crop_path.exists() or not cache_path.exists():
            n_cache_missing += 1
            continue
        crop = Image.open(crop_path).convert("RGB")
        e_crop = embed(crop)
        e_cache = embed(Image.open(cache_path).convert("RGB"))
        sims_cache.append(cos(e_crop, e_cache))
        if dilated_path.exists():
            sims_dilated.append(cos(e_crop, embed(Image.open(dilated_path).convert("RGB"))))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/100", flush=True)

    import numpy as np
    m_cache = float(np.mean(sims_cache)) if sims_cache else 0
    m_dilated = float(np.mean(sims_dilated)) if sims_dilated else 0
    pass_cache = sum(1 for s in sims_cache if s >= 0.75)
    pass_dilated = sum(1 for s in sims_dilated if s >= 0.75)

    print()
    print(f"=== Cosine similarity: fd_cache vs REAL book crops (N={len(sims_cache)}) ===")
    print(f"  cache:   mean {m_cache:.3f}  min {min(sims_cache):.3f}  max {max(sims_cache):.3f}")
    print(f"  dilated: mean {m_dilated:.3f}  min {min(sims_dilated):.3f}  max {max(sims_dilated):.3f}")
    print(f"  Pass threshold 0.75:  cache {pass_cache}/{len(sims_cache)}  dilated {pass_dilated}/{len(sims_dilated)}")
    print(f"  Missing in cache: {n_cache_missing}")
    print()
    print("Interpretation:")
    print(f"  Mean cosine {m_cache:.3f} = fd_cache matches REAL crops at this level.")
    print(f"  Threshold 0.75 = pipeline's Tier-3 'matched=True' bar.")
    if pass_cache / max(len(sims_cache), 1) < 0.5:
        print(f"  ⚠️  Less than 50% pass threshold — fd_cache is degrading Tier-3 quality.")
    else:
        print(f"  ✓ Most fd_cache passes threshold — quality is OK for visual matching.")

    md = [
        "# fd_cache vs REAL book crops — DINOv2 cosine test",
        "",
        "Tests if fd_cache (FontDiffusion-generated) actually matches the",
        "handwritten crops from the books, since that's what pipeline Tier-3 uses.",
        "",
        f"## Result (N={len(sims_cache)} labeled crops)",
        "",
        f"|              | mean cosine | min  | max  | ≥0.75 |",
        f"|--------------|------------:|-----:|-----:|------:|",
        f"| **cache**    | **{m_cache:.3f}** | {min(sims_cache):.3f} | {max(sims_cache):.3f} | {pass_cache}/{len(sims_cache)} |",
        f"| **dilated**  | **{m_dilated:.3f}** | {min(sims_dilated):.3f} | {max(sims_dilated):.3f} | {pass_dilated}/{len(sims_dilated)} |",
        "",
        "## Interpretation",
        "",
        f"- Mean cosine {m_cache:.3f} = average similarity between fd_cache and real",
        "  book crops. Higher = fd_cache better represents real handwriting.",
        f"- Threshold 0.75 = pipeline Tier-3 'matched=True' cutoff.",
        f"- {pass_cache}/{len(sims_cache)} ({pass_cache/max(len(sims_cache),1)*100:.0f}%) crops",
        f"  have fd_cache match ≥0.75 → would 'pass' Tier-3.",
        "",
        "## Verdict",
        "",
        ("⚠️  fd_cache cosine to real crops is LOW. Tier-3 matching is unreliable."
         if m_cache < 0.6 else
         "✓ fd_cache cosine acceptable. Tier-3 matching reasonably valid."),
    ]
    (OUT / "fd_cache_vs_real_crops.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT / 'fd_cache_vs_real_crops.md'}")


if __name__ == "__main__":
    main()
