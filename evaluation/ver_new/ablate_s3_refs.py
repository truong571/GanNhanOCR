"""Ablation: how much does each S3 reference type contribute to the visual match?

Measures retrieval@1 (does the TRUE char win its real candidate set R = {ocr_char}
∪ dict-readings) on held-out VAL GOLD crops, under three reference configs:
  (A) GLYPH-ONLY  — rank by cosine to the similar-font glyph (gannhanocr-fd). This
                    is what the generated similar images achieve ON THEIR OWN.
  (B) CROP-ONLY   — rank by cosine to the real-crop prototype (0 if the candidate
                    has no crops).
  (C) COMBINED    — the live system: per-tier calibrated P(match), best tier wins.
Also prints the per-tier same/diff cosine separation (the domain-gap probe).

This is the ablation table for the thesis: it isolates the marginal value of the
similar-font references vs the real crops, non-circularly (VAL crops never seen by
the prototypes, which are TRAIN-split only).

Run:
  .venv/bin/python evaluation/ver_new/ablate_s3_refs.py            # --limit N
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(Path(__file__).resolve().parent / "dataset_out"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]),
                   simfont_dir=str(REPO / paths["fd_cache_similar"]) if paths.get("fd_cache_similar") else "")

    D = Path(args.dataset)
    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["tier"] == "GOLD" and r["split"] == "val" and r["image"] and r["syllable"]]
    if args.limit:
        rows = rows[:args.limit]
    print(f"VAL GOLD crops: {len(rows)}\n", flush=True)

    # retrieval@1 counters per config; plus separation accumulators per tier
    hit = defaultdict(int)
    n = 0
    sep = defaultdict(lambda: ([], []))   # tier -> (same_cos, diff_cos)
    crop_avail = 0

    for i, r in enumerate(rows):
        ce = vs3.enc.embed_path(str(D / r["image"]))
        if ce is None:
            continue
        true = r["label"]
        syl = (r["syllable"] or "").lower()
        cands = []
        for c in ([r["ocr_char"]] if r["ocr_char"] else []) + qn.get(syl, []):
            if _is_cjk(c) and c not in cands:
                cands.append(c)
        if true not in cands:
            cands.append(true)
        if len(cands) < 2:
            continue
        n += 1
        tc = {c: vs3.tier_cosines(ce, c) for c in cands}     # {c: {tier: rawcos}}
        if "crop" in tc.get(true, {}):
            crop_avail += 1
        # accumulate separation per tier (same vs diff) for the domain-gap probe
        for c, d in tc.items():
            for t, v in d.items():
                sep[t][0 if c == true else 1].append(v)

        def argmax_correct(score_fn):
            best_c, best_s = None, -1e9
            for c in cands:
                s = score_fn(c, tc[c])
                if s > best_s:
                    best_s, best_c = s, c
            return best_c == true

        # (A) glyph-only (similar-font / fd tier)
        hit["A_glyph_only"] += argmax_correct(lambda c, d: d.get("fd", -1.0))
        # (B) crop-only (real-crop prototype)
        hit["B_crop_only"] += argmax_correct(lambda c, d: d.get("crop", -1.0))
        # (C) combined: per-tier calibrated P(match), best tier
        def combined(c, d):
            ps = [vs3._p_tier(t, v) for t, v in d.items()]
            ps = [p for p in ps if p is not None]
            return max(ps) if ps else 0.0
        hit["C_combined"] += argmax_correct(combined)
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{len(rows)}", flush=True)

    print("\n=== retrieval@1 on real candidate sets (VAL GOLD, held out) ===")
    print(f"  decisions: {n}  |  true char has a real-crop prototype: {crop_avail} ({crop_avail/max(n,1):.1%})")
    for k in ("A_glyph_only", "B_crop_only", "C_combined"):
        print(f"  {k:14s}: {hit[k]/max(n,1):.1%}  ({hit[k]}/{n})")

    print("\n=== domain-gap probe: per-tier mean cosine (same-char vs other candidates) ===")
    for t, (same, diff) in sep.items():
        s = float(np.mean(same)) if same else float("nan")
        df = float(np.mean(diff)) if diff else float("nan")
        print(f"  {t:8s}: same={s:+.3f}  diff={df:+.3f}  separation={s - df:+.3f}  (n_same={len(same)})")

    print("\n  Reading it: A = what your similar-font glyphs achieve ALONE; B = real crops alone;")
    print("  C = the deployed combination. The A vs C gap is the marginal value of the crops;")
    print("  the per-tier separation is the domain-gap measure for the similar-font references.")


if __name__ == "__main__":
    main()
