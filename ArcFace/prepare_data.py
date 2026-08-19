"""Bundle an ALL-IN, self-contained training set for Kaggle.

Reads the repo's frozen release (dataset_out/labels_final.csv + crops), the
SinoNom similarity dict, and the FontDiffusion glyph bank, and writes a single
portable folder you can zip → upload as one Kaggle dataset:

    ArcFace/data/
      ├── crops/       real woodblock crops (GOLD + SILVER, char-labeled)
      ├── glyphs/      one FontDiffusion glyph per class (domain-alignment anchor)
      ├── manifest.csv path,label,unicode,book,page,column,tier,source
      └── similar_map.json   {char: [visually-similar chars present in this set]}

Design choices baked into the manifest (roadmap P2):
  * only char-labeled GOLD+SILVER rows (REVIEW/QUARANTINE excluded — no label).
  * tier kept per row so train.py can DOWN-WEIGHT SILVER (weaker, AI-audited)
    relative to GOLD, instead of trusting them equally.
  * book/page kept so dataset.assign_splits can do page-disjoint / LOBO (kills
    the old 86% page leak). NO split is baked here — the trainer decides.

Usage:
    python ArcFace/prepare_data.py                      # copy everything (portable)
    python ArcFace/prepare_data.py --link               # hardlink (fast, same disk)
    python ArcFace/prepare_data.py --tiers GOLD         # GOLD-only (max de-circular)
    python ArcFace/prepare_data.py --limit 2000         # smoke subset
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def find_fd_glyph(fd_dir: Path, unicode_str: str) -> Path | None:
    """gannhanocr-fd/<HEX2>/U+XXXX.png — try the hex-prefix folder, then rglob."""
    if not unicode_str.startswith("U+"):
        return None
    cp = unicode_str[2:].upper()
    cand = fd_dir / cp[:2] / f"U+{cp}.png"
    if cand.exists():
        return cand
    hits = list(fd_dir.rglob(f"U+{cp}.png"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(REPO / "dataset_out" / "labels_final.csv"))
    ap.add_argument("--crops-root", default=str(REPO / "dataset_out"))
    ap.add_argument("--fd-dir", default=str(REPO / "gannhanocr-fd"))
    ap.add_argument("--similar", default=str(REPO / "Dict" / "SinoNom_Similar.csv"))
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--tiers", default="GOLD,SILVER", help="comma tiers to include")
    ap.add_argument("--link", action="store_true", help="hardlink instead of copy (same FS)")
    ap.add_argument("--manifest-only", action="store_true", help="don't copy files")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    crops_out, glyph_out = out / "crops", out / "glyphs"
    for d in (crops_out, glyph_out):
        d.mkdir(parents=True, exist_ok=True)
    keep_tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}
    crops_root = Path(args.crops_root)

    rows = list(csv.DictReader(open(args.labels, encoding="utf-8")))
    kept, classes, place = [], set(), (os.link if args.link else shutil.copy)
    n_missing = 0
    for r in rows:
        if r.get("label_level") != "char" or not r["label"].strip():
            continue
        if r["tier"] not in keep_tiers:
            continue
        src = crops_root / r["image"]
        if not src.exists():
            n_missing += 1
            continue
        base = Path(r["image"]).name
        if not args.manifest_only:
            dst = crops_out / base
            if not dst.exists():
                place(src, dst)
        kept.append({
            "path": f"crops/{base}", "label": r["label"], "unicode": r["unicode"],
            "book": r["book"], "page": r["page"], "column": r["column"],
            "tier": r["tier"], "source": "crop",
        })
        classes.add((r["label"], r["unicode"]))
        if args.limit and len(kept) >= args.limit:
            break

    # one FD glyph per class (domain-alignment anchor; ALWAYS train split)
    fd_dir = Path(args.fd_dir)
    n_fd = 0
    for label, uni in sorted(classes):
        g = find_fd_glyph(fd_dir, uni)
        if not g:
            continue
        base = g.name
        if not args.manifest_only:
            dst = glyph_out / base
            if not dst.exists():
                place(g, dst)
        kept.append({"path": f"glyphs/{base}", "label": label, "unicode": uni,
                     "book": "FD", "page": "FD", "column": "0", "tier": "FD", "source": "fd"})
        n_fd += 1

    # similarity map restricted to classes present here (hard-negative mining)
    present = {label for label, _ in classes}
    sim = {}
    if Path(args.similar).exists():
        for row in csv.reader(open(args.similar, encoding="utf-8-sig")):
            if len(row) < 2 or row[0] not in present:
                continue
            try:
                cand = ast.literal_eval(row[1])
            except (ValueError, SyntaxError):
                cand = [row[1]]
            near = [c for c in cand if c in present and c != row[0]]
            if near:
                sim[row[0]] = near

    fields = ["path", "label", "unicode", "book", "page", "column", "tier", "source"]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(kept)
    json.dump(sim, open(out / "similar_map.json", "w", encoding="utf-8"), ensure_ascii=False)

    print(f"[prepare] crops={sum(1 for k in kept if k['source']=='crop')} "
          f"fd={n_fd} classes={len(classes)} missing-crop={n_missing}")
    print(f"[prepare] similarity groups={len(sim)}  -> {out}")
    print(f"[prepare] mode={'manifest-only' if args.manifest_only else ('hardlink' if args.link else 'copy')}")


if __name__ == "__main__":
    main()
