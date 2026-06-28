"""Pack the recognizer training set for Kaggle: GOLD (+SILVER) crops + a slim
labels.csv + the trainer + the encoder checkpoint (for warm-start).

Output: nom_recognizer/kaggle_rec_pkg/
  gold/ silver/             the crop PNGs (already small/tightened, no downscale)
  labels.csv                slim (image,tier,split,label,ocr_char,syllable)
  train_recognizer.py
  encoder_best.pt           nom-embed/best.pt (warm-start the ResNet34 backbone)

Run:
  .venv/bin/python evaluation/ver_new/nom_recognizer/pack_for_kaggle.py
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
DATASET = HERE.parent / "dataset_out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--out", default=str(HERE / "kaggle_rec_pkg"))
    ap.add_argument("--with-silver", action="store_true", default=True,
                    help="include SILVER crops (default on; trainer chooses via --use-silver)")
    args = ap.parse_args()
    D = Path(args.dataset); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(D / "labels.csv", encoding="utf-8")))
    keep_tiers = {"GOLD", "SILVER"}
    slim, copied, missing = [], 0, 0
    for r in rows:
        if r["tier"] not in keep_tiers or not r["image"]:
            continue
        src = D / r["image"]
        if not src.exists():
            missing += 1; continue
        dst = out / r["image"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(src, dst); copied += 1
        slim.append({k: r[k] for k in ("image", "tier", "split", "label", "ocr_char", "syllable")})

    with open(out / "labels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "tier", "split", "label", "ocr_char", "syllable"])
        w.writeheader(); w.writerows(slim)
    shutil.copy(HERE / "train_recognizer.py", out / "train_recognizer.py")
    enc = REPO / "nom-embed" / "best.pt"
    if enc.exists():
        shutil.copy(enc, out / "encoder_best.pt")

    # bundle the gannhanocr-fd GENERATED glyph for each GOLD class (1/class, flat fd/)
    # so the Kaggle trainer can add them to train (rare / 0-crop class coverage).
    fd_src = REPO / "gannhanocr-fd"
    (out / "fd").mkdir(exist_ok=True)
    classes = sorted({r["label"] for r in slim if r["tier"] == "GOLD" and r["label"]})
    fd_n = 0
    for c in classes:
        if len(c) != 1:
            continue
        hx = f"{ord(c):X}"
        p = next((q for q in (fd_src / f"U+{hx}.png", fd_src / hx[:2] / f"U+{hx}.png") if q.exists()), None)
        if p:
            d = out / "fd" / f"U+{hx}.png"
            if not d.exists():
                shutil.copy(p, d); fd_n += 1
    print(f"  + bundled {fd_n} gannhanocr-fd glyphs (1/class) -> {out}/fd/")

    mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    nt = {}
    for r in slim:
        nt[r["tier"]] = nt.get(r["tier"], 0) + 1
    print(f"packed -> {out}")
    print(f"  crops copied {copied} (missing {missing}) | rows {len(slim)} {nt} | size {mb:.0f} MB")
    print(f"  upload {out} as a Kaggle Dataset, then in a GPU notebook:")
    print(f"    !python train_recognizer.py --root . --target consensus --epochs 30 --img 160 \\")
    print(f"            --init encoder_best.pt --out /kaggle/working/recognizer.pt --hf-repo <user>/nom-recognizer")
    print(f"  (see nom_recognizer/KAGGLE.md)")


if __name__ == "__main__":
    main()
