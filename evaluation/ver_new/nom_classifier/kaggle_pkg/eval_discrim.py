"""Validate the trained Nôm embedder with the SAME tests DINOv2 failed.

Reads index.csv (--index) + --root (so it runs both locally and inside the
Kaggle package). Uses TEST-split real crops (source=crop, split=test, never seen
in training) and the FD reference glyphs (source=fd).

Acceptance (DINOv2 was 0.012 and 0.0%):
  T2 real crops : same-char cosine − diff-char cosine  >= ~0.20
  T3 retrieval  : crop -> nearest FD glyph top-1        >= ~0.80

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints" / "best.pt"))
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--index", default=str(HERE / "index.csv"))
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    random.seed(0)
    enc = NomEncoder(args.ckpt)
    root = Path(args.root)
    rows = list(csv.DictReader(open(args.index, encoding="utf-8")))

    test_by = defaultdict(list)
    fd = {}
    for r in rows:
        if r["source"] == "crop" and r["split"] == "test":
            test_by[r["label"]].append(str(root / r["path"]))
        elif r["source"] == "fd":
            fd[r["label"]] = str(root / r["path"])

    ecache = {}
    def ce(p):
        if p not in ecache:
            ecache[p] = enc.embed_path(p)
        return ecache[p]

    # T2: same vs different (real test crops)
    multi = [c for c, v in test_by.items() if len(v) >= 2 and c in fd]
    random.shuffle(multi); multi = multi[:args.n]
    same, diff = [], []
    for i, c in enumerate(multi):
        a, b = ce(random.choice(test_by[c])), ce(random.choice(test_by[c]))
        if a is not None and b is not None:
            same.append(enc.cosine(a, b))
        d = multi[(i + 1) % len(multi)]
        ea, eb = ce(random.choice(test_by[c])), ce(random.choice(test_by[d]))
        if ea is not None and eb is not None:
            diff.append(enc.cosine(ea, eb))

    # T3: retrieval crop -> nearest FD glyph (gallery up to 500)
    gal = random.sample(list(fd), min(500, len(fd)))
    for c in multi:
        if c not in gal:
            gal.append(c)
    galE = np.stack([enc.embed_path(fd[c]) for c in gal])
    galN = galE / np.linalg.norm(galE, axis=1, keepdims=True)
    hit = tot = 0
    for c in multi:
        e = ce(random.choice(test_by[c]))
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
