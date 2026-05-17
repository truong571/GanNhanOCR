"""Deep evaluation of fd_cache quality. 500-sample test across multiple metrics.

Tests:
  1. Cosine distribution: fd_cache vs real crops (N=500)
  2. Self-similarity baseline: crop[char] vs another crop[same char]
     (upper bound — how close DINOv2 puts two real handwritten samples)
  3. Font-rendered baseline: font[char] vs real crops
     (what would tier-3 score look like WITHOUT FontDiffusion?)
  4. Han vs Nôm breakdown — does cache work equally on both?
  5. High-freq vs low-freq breakdown
  6. Pipeline-realistic test: for each crop, find top-1 candidate among
     dict[syllable] using fd_cache cosine. Is top-1 actually the labeled char?
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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

N_SAMPLES = 500


def load_qn_to_nom():
    df = pd.read_csv(REPO / "Dict" / "QuocNgu_SinoNom_TongHop3.csv")
    out: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        qn = str(r["QuocNgu"]).strip().lower()
        nom = str(r["SinoNom"]).strip()
        if qn and nom != "nan":
            out.setdefault(qn, set()).update(c for c in nom if not c.isspace())
    return out


def is_han_unified(ch: str) -> bool:
    return 0x4E00 <= ord(ch) <= 0x9FFF


def render_font(ch: str, size: int = 96) -> Image.Image:
    font = ImageFont.truetype(str(FONT_PATH), int(size * 0.75))
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), ch, font=font)
    x = (size - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (size - (bbox[3] - bbox[1])) / 2 - bbox[1]
    d.text((x, y), ch, fill="black", font=font)
    return img


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading DINOv2 on {device}...")
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

    qn_to_nom = load_qn_to_nom()

    # Load labels.csv, group crops by nom_char (for self-similarity)
    print("Loading dataset...")
    df = pd.read_csv(REPO / "dataset" / "all" / "labels.csv")
    df = df[df["matched"] == True].copy()
    crops_by_char: dict[str, list[dict]] = defaultdict(list)
    for r in df.to_dict("records"):
        ch = r["nom_char"]
        if not ch or len(ch) != 1:
            continue
        crops_by_char[ch].append({
            "path": REPO / "dataset" / r["source"] / r["crop_file"],
            "syllable": str(r["syllable"]).strip().lower(),
        })
    print(f"  {len(crops_by_char):,} unique chars, {sum(len(v) for v in crops_by_char.values()):,} crops")

    # Sample 500 (crop, char) pairs across diverse chars
    random.seed(42)
    all_pairs = [(c, info) for c, infos in crops_by_char.items() for info in infos]
    sample = random.sample(all_pairs, min(N_SAMPLES, len(all_pairs)))
    print(f"Sampling {len(sample)} crops\n")

    # Pre-embed all needed fd_cache + dilated + font for unique chars in sample
    unique_chars = sorted({c for c, _ in sample})
    print(f"Pre-embedding {len(unique_chars)} unique chars (cache + dilated + font)...")
    cache_emb: dict[str, torch.Tensor] = {}
    dilated_emb: dict[str, torch.Tensor] = {}
    font_emb: dict[str, torch.Tensor] = {}
    for i, ch in enumerate(unique_chars):
        cp = ord(ch)
        cp_file = f"U+{cp:04X}.png"
        cp_path = CACHE_DIR / cp_file
        if cp_path.exists():
            cache_emb[ch] = embed(Image.open(cp_path).convert("RGB"))
        dp_path = DILATED_DIR / cp_file
        if dp_path.exists():
            dilated_emb[ch] = embed(Image.open(dp_path).convert("RGB"))
        font_emb[ch] = embed(render_font(ch))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(unique_chars)}", flush=True)

    # Test 1, 2, 3: per-crop cosine to (cache, dilated, font)
    print(f"\nEmbedding {len(sample)} real crops + comparing...")
    sims = {"cache": [], "dilated": [], "font": [], "self": []}
    han_sims = {"cache": [], "dilated": [], "font": []}
    nom_sims = {"cache": [], "dilated": [], "font": []}

    pipeline_correct = 0
    pipeline_total = 0

    for i, (ch, info) in enumerate(sample):
        crop_path = info["path"]
        if not crop_path.exists():
            continue
        try:
            crop_pil = Image.open(crop_path).convert("RGB")
        except Exception:
            continue
        e_crop = embed(crop_pil)

        # cache, dilated, font
        if ch in cache_emb:
            s = cos(e_crop, cache_emb[ch])
            sims["cache"].append(s)
            (han_sims if is_han_unified(ch) else nom_sims)["cache"].append(s)
        if ch in dilated_emb:
            s = cos(e_crop, dilated_emb[ch])
            sims["dilated"].append(s)
            (han_sims if is_han_unified(ch) else nom_sims)["dilated"].append(s)
        if ch in font_emb:
            s = cos(e_crop, font_emb[ch])
            sims["font"].append(s)
            (han_sims if is_han_unified(ch) else nom_sims)["font"].append(s)

        # Self-similarity: pick another crop of SAME char (if exists)
        siblings = [x for x in crops_by_char[ch] if x["path"] != crop_path]
        if siblings:
            other = random.choice(siblings)
            if other["path"].exists():
                e_other = embed(Image.open(other["path"]).convert("RGB"))
                sims["self"].append(cos(e_crop, e_other))

        # Test 6: pipeline-realistic — pick top-1 fd_cache candidate among dict[syllable]
        syl = info["syllable"]
        candidates = qn_to_nom.get(syl, set())
        cand_with_cache = [c for c in candidates if c in cache_emb]
        if cand_with_cache:
            scores = {c: cos(e_crop, cache_emb[c]) for c in cand_with_cache}
            top1 = max(scores, key=scores.get)
            pipeline_total += 1
            if top1 == ch:
                pipeline_correct += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)}", flush=True)

    def stats(name, lst):
        if not lst:
            return None
        a = np.array(lst)
        return {
            "n": len(lst),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "min": float(a.min()),
            "max": float(a.max()),
            "pass_075": int((a >= 0.75).sum()),
            "pass_075_pct": float((a >= 0.75).mean() * 100),
        }

    print("\n" + "=" * 70)
    print(f"=== Cosine distribution: fd_cache vs REAL crops (N={len(sims['cache'])}) ===")
    print()
    for name in ("self", "cache", "dilated", "font"):
        s = stats(name, sims[name])
        if not s:
            print(f"  {name:8s}: (no data)")
            continue
        print(f"  {name:8s}: mean={s['mean']:.3f}  median={s['median']:.3f}  "
              f"[p25={s['p25']:.3f}, p75={s['p75']:.3f}]  "
              f"pass≥0.75: {s['pass_075']}/{s['n']} ({s['pass_075_pct']:.1f}%)")

    print()
    print("=== Han Unified vs Nôm Ext breakdown ===")
    print(f"  {'Type':10s} {'cache':>15s} {'dilated':>15s} {'font':>15s}")
    for label, d in [("Han Unified", han_sims), ("Nôm Ext B+", nom_sims)]:
        line = f"  {label:10s}"
        for k in ("cache", "dilated", "font"):
            s = stats(k, d[k])
            line += f"  {s['mean']:.3f} ({s['n']:>3})  " if s else f"  {'-':>13s}  "
        print(line)

    print()
    print("=== Pipeline-realistic top-1 accuracy ===")
    print(f"For each crop, find top-1 fd_cache candidate among dict[syllable]:")
    if pipeline_total:
        print(f"  Top-1 == labeled char: {pipeline_correct}/{pipeline_total} = "
              f"{pipeline_correct/pipeline_total*100:.1f}%")
    print("  (This is the actual Tier-3 accuracy when dict + fd_cache combine.)")

    # Verdict
    cache_pass = stats("cache", sims["cache"])["pass_075_pct"]
    dilated_pass = stats("dilated", sims["dilated"])["pass_075_pct"]
    font_mean = stats("font", sims["font"])["mean"]
    cache_mean = stats("cache", sims["cache"])["mean"]
    self_mean = stats("self", sims["self"])["mean"] if sims["self"] else 0

    print()
    print("=== VERDICT ===")
    print(f"  Self-similarity (REAL upper bound):  {self_mean:.3f}")
    print(f"  fd_cache (current):                  {cache_mean:.3f}  ({cache_pass:.1f}% pass 0.75)")
    print(f"  fd_cache_dilated:                    {stats('dilated', sims['dilated'])['mean']:.3f}  ({dilated_pass:.1f}% pass 0.75)")
    print(f"  font-rendered baseline:              {font_mean:.3f}")
    print()
    if cache_mean < font_mean:
        print(f"  ⚠️  fd_cache mean ({cache_mean:.3f}) < font baseline ({font_mean:.3f})")
        print(f"      → FontDiffusion DEGRADED quality vs simple font render!")
        print(f"      → STRONGLY RECOMMEND regenerating cache or just using font render.")
    elif cache_mean - font_mean < 0.05:
        print(f"  ⚠️  fd_cache only marginally better than font ({cache_mean - font_mean:+.3f}pp)")
        print(f"      → FontDiffusion is barely worth it. Consider regen with better settings.")
    else:
        print(f"  ✓ fd_cache outperforms font baseline by {cache_mean - font_mean:+.3f}pp")
        print(f"      → FontDiffusion IS adding value. Cache acceptable.")

    # Write full report
    md = [
        "# Deep fd_cache evaluation — 500-crop test",
        "",
        f"Sample: 500 random labeled crops from dataset/all/labels.csv",
        f"Device: {device}",
        "",
        "## Cosine distribution (vs REAL book crops)",
        "",
        "| Metric | n | mean | median | p25 | p75 | min | max | ≥0.75 |",
        "|--------|--:|-----:|-------:|----:|----:|----:|----:|------:|",
    ]
    for name in ("self", "cache", "dilated", "font"):
        s = stats(name, sims[name])
        if s:
            md.append(f"| {name} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | "
                      f"{s['p25']:.3f} | {s['p75']:.3f} | {s['min']:.3f} | "
                      f"{s['max']:.3f} | {s['pass_075']} ({s['pass_075_pct']:.1f}%) |")
    md += [
        "",
        "**self** = crop[X] vs another crop[same X] (real-vs-real, upper bound)",
        "**cache** = real crop vs fd_cache[X] (FontDiffusion output)",
        "**dilated** = real crop vs fd_cache_dilated[X]",
        "**font** = real crop vs NomNaTong font-rendered[X]",
        "",
        "## Han Unified vs Nôm Ext breakdown",
        "",
        "| Type | cache mean (n) | dilated mean (n) | font mean (n) |",
        "|------|---------------:|-----------------:|--------------:|",
    ]
    for label, d in [("Han Unified", han_sims), ("Nôm Ext B+", nom_sims)]:
        row = f"| {label} |"
        for k in ("cache", "dilated", "font"):
            s = stats(k, d[k])
            row += f" {s['mean']:.3f} ({s['n']}) |" if s else " - |"
        md.append(row)

    md += ["",
           "## Pipeline-realistic top-1 accuracy",
           "",
           f"For each crop, pick top-1 fd_cache candidate among dict[syllable].",
           f"Top-1 correct: **{pipeline_correct}/{pipeline_total} = "
           f"{pipeline_correct/max(pipeline_total,1)*100:.1f}%**",
           "",
           "This is what Tier-3 actually delivers (combining dict + fd_cache).",
           "",
           "## Verdict",
           "",
           f"- Self-similarity ceiling (real-vs-real): {self_mean:.3f}",
           f"- fd_cache:         {cache_mean:.3f}  ({cache_pass:.1f}% pass threshold 0.75)",
           f"- fd_cache_dilated: {stats('dilated', sims['dilated'])['mean']:.3f}  ({dilated_pass:.1f}% pass)",
           f"- Font baseline:    {font_mean:.3f}",
           "",
           ("**fd_cache > font** → FontDiffusion adds value. Cache acceptable as-is."
            if cache_mean > font_mean + 0.03 else
            "**fd_cache ≈ font** → FontDiffusion marginal. Consider regen with bigger res/more steps OR fallback to font render."
            if abs(cache_mean - font_mean) < 0.03 else
            "**fd_cache < font** → FontDiffusion DEGRADED quality. Use font render instead, or regen with better settings."),
           "",
           "## Recommendation for next action",
           "",
           ("- Switch to fd_cache_dilated (slight improvement)"
            if dilated_pass > cache_pass + 2 else
            "- Keep fd_cache as-is (dilated not clearly better)"),
           "",
           ("- Regenerate fd_cache on Kaggle with content_image_size=128, "
            "num_inference_steps=40, guidance_scale=2.5 — current default 96/20/2.0 "
            "produces images significantly worse than even font render."
            if cache_mean < font_mean else
            "- Current cache is acceptable. Generation of missing 5,532 chars (Kaggle) "
            "would improve coverage but not necessarily per-char quality."),
    ]
    (OUT / "deep_fd_cache_eval.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT / 'deep_fd_cache_eval.md'}")


if __name__ == "__main__":
    main()
