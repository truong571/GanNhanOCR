"""Validate the trained Nôm embedder with the SAME tests DINOv2 failed.

Re-runs the discrimination tests from REPORT_dinov2_unsuitable.md, but with the
trained NomEncoder instead of DINOv2. Acceptance criteria:
  T2 (real crops): same-char cosine − different-char cosine  >= ~0.20
  T3 (retrieval) : crop -> nearest FD glyph top-1 accuracy   >= ~0.80
(DINOv2 was 0.012 and 0.0% respectively.)

Run:
  .venv/bin/python evaluation/ver_new/nom_classifier/eval_discrim.py --ckpt checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from infer import NomEncoder            # noqa: E402

REPO = HERE.parent.parent.parent


def fd_index(fd_dir: Path):
    idx = {}
    for p in fd_dir.rglob("U+*.png"):
        try:
            idx[chr(int(p.stem.replace("U+", ""), 16))] = str(p)
        except ValueError:
            pass
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints" / "best.pt"))
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--dataset", default=str(HERE.parent / "dataset_out"))
    ap.add_argument("--fd", default=str(REPO / "gannhanocr-fd"))
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    random.seed(0)
    enc = NomEncoder(args.ckpt)
    fd = fd_index(Path(args.fd))

    # use TEST split crops only (never seen in training)
    rows = [r for r in csv.DictReader(open(Path(args.dataset) / "labels.csv", encoding="utf-8"))
            if r["tier"] == "GOLD" and r["label_level"] == "char" and r["image"]
            and r["split"] == "test" and r["label"] in fd]
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(str(Path(args.dataset) / r["image"]))
    ecache = {}
    def ce(p):
        if p not in ecache:
            ecache[p] = enc.embed_path(p)
        return ecache[p]

    # T2: same vs different (real crops)
    multich = [c for c, v in by.items() if len(v) >= 2]
    random.shuffle(multich); multich = multich[:args.n]
    same = [enc.cosine(ce(random.choice(by[c])), ce(random.choice(by[c]))) for c in multich]
    diff = [enc.cosine(ce(random.choice(by[multich[i]])), ce(random.choice(by[multich[(i+1) % len(multich)]])))
            for i in range(len(multich))]
    same = [s for s in same if s is not None]; diff = [d for d in diff if d is not None]

    # T3: retrieval crop -> nearest FD glyph (gallery 500)
    gal = random.sample(list(fd), min(500, len(fd)))
    for c in multich:
        if c not in gal:
            gal.append(c)
    galE = np.stack([enc.embed_path(fd[c]) for c in gal])
    galN = galE / np.linalg.norm(galE, axis=1, keepdims=True)
    hit = tot = 0
    for c in multich:
        e = ce(random.choice(by[c]))
        if e is None:
            continue
        pred = gal[int(np.argmax(galN @ (e / np.linalg.norm(e))))]
        tot += 1; hit += (pred == c)

    print("=== Nôm embedder discrimination (TEST split) ===")
    print(f"T2 same-char cosine : {statistics.mean(same):.3f}")
    print(f"T2 diff-char cosine : {statistics.mean(diff):.3f}")
    print(f"T2 separation       : {statistics.mean(same)-statistics.mean(diff):+.3f}  (cần >= ~0.20)")
    print(f"T3 retrieval top-1  : {hit/max(tot,1):.1%}  (cần >= ~80%; DINOv2 = 0%)")


if __name__ == "__main__":
    main()
