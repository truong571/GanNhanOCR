"""Build the training index for the Nôm glyph embedding model.

Sources (both labeled by the SinoNom character = the class):
  - GOLD char crops (real woodblock):  dataset_out/labels.csv where
    tier==GOLD and label_level==char and image != '' (uses the leakage-safe
    `split` column already in labels.csv).
  - FontDiffusion reference glyphs:     gannhanocr-fd/U+*.png for every class
    char -> always 'train' (these BRIDGE the woodblock<->clean domain gap and
    guarantee >=1 sample for every class, incl. the long tail / singletons).

Outputs (into this folder):
  index.csv     path,label,unicode,split,source
  classes.json  {char: class_id}  (sorted by char)
  stats.json

`path` is relative to --root (default = repo root) so the same index works
locally and after uploading {dataset_out/gold, gannhanocr-fd, index.csv} to Kaggle.

Run:
  cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
  .venv/bin/python evaluation/ver_new/nom_classifier/prepare_data.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent


def fd_index(fd_dir: Path) -> dict[str, str]:
    idx = {}
    for p in fd_dir.rglob("U+*.png"):
        try:
            idx[chr(int(p.stem.replace("U+", ""), 16))] = p
        except ValueError:
            pass
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO), help="base dir paths are relative to")
    ap.add_argument("--dataset", default=str(HERE.parent / "dataset_out"))
    ap.add_argument("--fd", default=str(REPO / "gannhanocr-fd"))
    ap.add_argument("--min-per-class", type=int, default=0,
                    help="drop classes with fewer than this many TOTAL samples (0=keep all)")
    args = ap.parse_args()

    root = Path(args.root)
    rows = list(csv.DictReader(open(Path(args.dataset) / "labels.csv", encoding="utf-8")))
    gold = [r for r in rows if r["tier"] == "GOLD" and r["label_level"] == "char"
            and r["image"] and r["label"]]
    classes = sorted({r["label"] for r in gold})
    fd = fd_index(Path(args.fd))

    out = []
    # real woodblock crops (carry their split)
    for r in gold:
        p = (Path(args.dataset) / r["image"]).resolve()
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = p
        out.append({"path": str(rel), "label": r["label"], "unicode": r["unicode"],
                    "split": r["split"], "source": "crop"})
    # one FD reference glyph per class (train only) -> domain bridge + tail coverage
    n_fd = 0
    for ch in classes:
        if ch in fd:
            try:
                rel = fd[ch].resolve().relative_to(root)
            except ValueError:
                rel = fd[ch]
            out.append({"path": str(rel), "label": ch,
                        "unicode": f"U+{ord(ch):04X}", "split": "train", "source": "fd"})
            n_fd += 1

    # optional min-per-class filter
    if args.min_per_class > 0:
        cnt = Counter(r["label"] for r in out)
        keep = {c for c, n in cnt.items() if n >= args.min_per_class}
        out = [r for r in out if r["label"] in keep]
        classes = sorted({r["label"] for r in out})

    cls_map = {c: i for i, c in enumerate(classes)}
    with open(HERE / "index.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "unicode", "split", "source"])
        w.writeheader(); w.writerows(out)
    json.dump(cls_map, open(HERE / "classes.json", "w", encoding="utf-8"), ensure_ascii=False)
    sp = Counter(r["split"] for r in out)
    src = Counter(r["source"] for r in out)
    stats = {"rows": len(out), "classes": len(classes), "fd_glyphs": n_fd,
             "split": dict(sp), "source": dict(src),
             "root": str(root)}
    json.dump(stats, open(HERE / "stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"index.csv: {len(out)} rows | {len(classes)} classes | "
          f"crops {src['crop']} + fd {src['fd']} | split {dict(sp)}")
    print(f"-> {HERE}/index.csv, classes.json, stats.json")


if __name__ == "__main__":
    main()
