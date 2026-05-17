"""DINOv2-based stroke-loss check.

IoU is too strict for handwritten-vs-print comparison (always low even when
same char). Pipeline's actual metric is DINOv2 cosine similarity, so test that.

Two sub-tests:
  A. Random 30 chars: cosine(cache, font) and cosine(dilated, font)
  B. Confusable pairs: cache(X) closer to font(X) or font(Y)?
     If wrong direction, character is genuinely "losing meaning" to DINOv2.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

REPO = Path(__file__).resolve().parent.parent
FD_CACHE = REPO / "prepared" / "_universal_fd_cache"
FD_DILATED = REPO / "prepared" / "_universal_fd_cache_dilated"
NOMNATONG = REPO / "font_diffusion" / "fonts" / "NomNaTong-Regular.ttf"
OUT = REPO / "evaluation" / "reports"

CONFUSABLE = [
    ("大", "天"), ("大", "夫"), ("千", "干"), ("土", "士"),
    ("人", "入"), ("日", "目"), ("石", "右"), ("未", "末"),
    ("木", "本"), ("田", "由"), ("白", "百"),
]


def render_font(ch: str, size: int = 96) -> Image.Image:
    font = ImageFont.truetype(str(NOMNATONG), int(size * 0.75))
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), ch, font=font)
    x = (size - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (size - (bbox[3] - bbox[1])) / 2 - bbox[1]
    d.text((x, y), ch, fill="black", font=font)
    return img


def load_pil(ch: str, cache_dir: Path) -> Image.Image | None:
    p = cache_dir / f"U+{ord(ch):04X}.png"
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def main() -> None:
    print("Loading DINOv2 (this is what Tier-3 actually uses)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg", verbose=False)
    model.to(device).eval()
    transform = T.Compose([
        T.ToTensor(), T.Resize(244, antialias=True), T.CenterCrop(224),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    @torch.no_grad()
    def embed(pil_img: Image.Image) -> torch.Tensor:
        t = transform(pil_img).unsqueeze(0).to(device)
        e = model(t)
        if isinstance(e, tuple):
            e = e[0]
        return F.normalize(e.squeeze(0), dim=0)

    def cos(a, b):
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))

    # ----- Test A: 30 random chars -----
    print("\nTest A: 30 random chars — DINOv2 cosine similarity vs font")
    files = sorted(FD_CACHE.glob("U+*.png"))
    random.seed(42)
    sample = random.sample(files, 30)
    sims_cache, sims_dilated = [], []
    for f in sample:
        cp = int(f.stem.replace("U+", ""), 16)
        ch = chr(cp)
        e_font = embed(render_font(ch))
        cache_pil = load_pil(ch, FD_CACHE)
        dilated_pil = load_pil(ch, FD_DILATED)
        if cache_pil is not None:
            sims_cache.append(cos(e_font, embed(cache_pil)))
        if dilated_pil is not None:
            sims_dilated.append(cos(e_font, embed(dilated_pil)))

    print(f"  cache:   mean={np.mean(sims_cache):.3f}  min={min(sims_cache):.3f}  max={max(sims_cache):.3f}")
    print(f"  dilated: mean={np.mean(sims_dilated):.3f}  min={min(sims_dilated):.3f}  max={max(sims_dilated):.3f}")
    # Pipeline threshold = 0.75
    n_pass_cache = sum(1 for s in sims_cache if s >= 0.75)
    n_pass_dilated = sum(1 for s in sims_dilated if s >= 0.75)
    print(f"  ≥ 0.75 threshold:  cache {n_pass_cache}/30  dilated {n_pass_dilated}/30")

    # ----- Test B: confusable pairs -----
    print("\nTest B: confusable pairs — does cache(X) lean to font(X) or font(Y)?")
    n_correct = 0
    rows = []
    for a, b in CONFUSABLE:
        e_fa, e_fb = embed(render_font(a)), embed(render_font(b))
        ca, cb = load_pil(a, FD_CACHE), load_pil(b, FD_CACHE)
        if ca is None or cb is None:
            continue
        ea, eb = embed(ca), embed(cb)
        cos_aa = cos(ea, e_fa)  # cache(a) vs font(a) — should be > cos(a,b)
        cos_ab = cos(ea, e_fb)
        cos_bb = cos(eb, e_fb)
        cos_ba = cos(eb, e_fa)
        a_ok = cos_aa > cos_ab
        b_ok = cos_bb > cos_ba
        if a_ok and b_ok:
            n_correct += 1
        flag_a = "OK" if a_ok else "⚠️"
        flag_b = "OK" if b_ok else "⚠️"
        print(f"  {a} vs {b}:  cache({a})·font({a})={cos_aa:.3f}  vs font({b})={cos_ab:.3f}  [{flag_a}]")
        print(f"           cache({b})·font({b})={cos_bb:.3f}  vs font({a})={cos_ba:.3f}  [{flag_b}]")
        rows.append({
            "pair": f"{a}/{b}",
            "cos_aa": round(cos_aa, 3), "cos_ab": round(cos_ab, 3),
            "cos_bb": round(cos_bb, 3), "cos_ba": round(cos_ba, 3),
            "a_correct": a_ok, "b_correct": b_ok,
        })

    print(f"\n  {n_correct}/{len(rows)} pairs unambiguously distinguish correct identity")

    # Markdown report
    OUT.mkdir(parents=True, exist_ok=True)
    md = [
        "# fd_cache stroke loss — DINOv2 evaluation",
        "",
        "Using **DINOv2 cosine similarity** (the actual Tier-3 metric, not pixel IoU).",
        "",
        "## Test A: 30 random chars — similarity to font baseline",
        "",
        f"|              | mean | min | max | ≥0.75 (threshold) |",
        f"|--------------|-----:|----:|----:|------------------:|",
        f"| **cache**    | {np.mean(sims_cache):.3f} | {min(sims_cache):.3f} | {max(sims_cache):.3f} | {n_pass_cache}/30 |",
        f"| **dilated**  | {np.mean(sims_dilated):.3f} | {min(sims_dilated):.3f} | {max(sims_dilated):.3f} | {n_pass_dilated}/30 |",
        "",
        "Pipeline threshold = 0.75 (set in config). Higher mean = more confident matching.",
        "",
        "## Test B: confusable pairs (each differs by 1-2 strokes)",
        "",
        f"**{n_correct}/{len(rows)}** pairs unambiguously distinguished correct identity.",
        "",
        "| pair | cache(a)·font(a) | cache(a)·font(b) | a→a? | cache(b)·font(b) | cache(b)·font(a) | b→b? |",
        "|------|-----------------:|-----------------:|-----|-----------------:|-----------------:|-----|",
    ]
    for r in rows:
        a_mark = "OK" if r["a_correct"] else "**WRONG**"
        b_mark = "OK" if r["b_correct"] else "**WRONG**"
        md.append(f"| {r['pair']} | {r['cos_aa']} | {r['cos_ab']} | {a_mark} "
                  f"| {r['cos_bb']} | {r['cos_ba']} | {b_mark} |")

    (OUT / "fd_cache_dinov2_eval.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT / 'fd_cache_dinov2_eval.md'}")


if __name__ == "__main__":
    main()
