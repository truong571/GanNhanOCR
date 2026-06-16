"""Pack a SELF-CONTAINED Kaggle dataset folder for training.

Reads the local index.csv (run prepare_data.py first) and copies every
referenced image + the code into `kaggle_pkg/`, rewriting index.csv with paths
relative to the package root. Upload `kaggle_pkg/` as ONE Kaggle Dataset and
point --root at its mount — no path/read-only headaches.

  kaggle_pkg/
    images/crop/<...>.png    real GOLD crops
    images/fd/U+*.png         FontDiffusion reference glyphs
    index.csv  classes.json   (paths relative to kaggle_pkg/)
    model.py dataset.py train.py eval_discrim.py infer.py

Run (local):
  cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
  .venv/bin/python evaluation/ver_new/nom_classifier/prepare_data.py
  .venv/bin/python evaluation/ver_new/nom_classifier/pack_for_kaggle.py
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
CODE = ["model.py", "dataset.py", "train.py", "kaggle_train.py", "eval_discrim.py",
        "infer.py", "prepare_data.py", "classes.json", "README.md", "KAGGLE_TRAIN.md",
        "train_kaggle.ipynb", "KAGGLE.md"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(HERE / "index.csv"))
    ap.add_argument("--out", default=str(HERE / "kaggle_pkg"))
    args = ap.parse_args()
    out = Path(args.out)
    for s in ("crop", "fd", "font"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.index, encoding="utf-8")))
    new = []
    copied = skipped = 0
    for r in rows:
        src = (REPO / r["path"]).resolve()
        sub = r["source"] if r["source"] in ("fd", "font") else "crop"
        rel = f"images/{sub}/{Path(r['path']).name}"
        dst = out / rel
        if not dst.exists():
            if src.exists():
                shutil.copy(src, dst); copied += 1
            else:
                skipped += 1; continue
        new.append({**r, "path": rel})
        if (copied + skipped) % 10000 == 0 and (copied + skipped):
            print(f"  ...{copied} copied", flush=True)

    with open(out / "index.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "unicode", "split", "source"])
        w.writeheader(); w.writerows(new)
    for c in CODE:
        s = HERE / c
        if s.exists():
            shutil.copy(s, out / c)

    n_img = len(list((out / "images").rglob("*.png")))
    print(f"\nkaggle_pkg -> {out}")
    print(f"  images: {n_img} (copied {copied}, missing {skipped})")
    print(f"  index.csv rows: {len(new)} | code: {len([c for c in CODE if (out/c).exists()])} files")
    print("Upload this folder as a Kaggle Dataset; see KAGGLE.md.")


if __name__ == "__main__":
    main()
