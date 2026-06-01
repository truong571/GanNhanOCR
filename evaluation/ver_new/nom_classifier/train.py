"""Train the Nôm glyph embedding (ResNet-18 + ArcFace) on Kaggle P100.

Fits comfortably on a P100 (16 GB): ResNet-18 @128px, batch 256, AMP ->
~4-6 GB, ~3-5 min/epoch, ~30-40 epochs ~= 1.5-3 h (within the 12h session /
30h-week quota).

Run (local):
  cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
  .venv/bin/python evaluation/ver_new/nom_classifier/train.py --epochs 30
Run (Kaggle, P100 GPU): see README.md (set --root to the uploaded dataset dir).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from model import NomEmbedder, ArcMargin          # noqa: E402
from dataset import NomDataset                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parent.parent.parent))
    ap.add_argument("--index", default=str(HERE / "index.csv"))
    ap.add_argument("--classes", default=str(HERE / "classes.json"))
    ap.add_argument("--out", default=str(HERE / "checkpoints"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={dev}")
    classes = json.load(open(args.classes, encoding="utf-8"))
    n_cls = len(classes)

    tr = NomDataset(args.index, args.root, classes, "train", args.img, train=True)
    va = NomDataset(args.index, args.root, classes, "val", args.img, train=False)
    print(f"train {len(tr)} | val {len(va)} | classes {n_cls}")
    pin = (dev == "cuda")
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=pin, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin)

    net = NomEmbedder(args.embed, pretrained=True).to(dev)
    head = ArcMargin(args.embed, n_cls).to(dev)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(args.epochs):
        net.train(); head.train(); t0 = time.time(); tot = 0.0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                logits = head(net(x), y)
                loss = crit(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0)
        sched.step()
        # val top-1 (closed-set, via ArcFace logits)
        net.eval(); head.eval(); correct = n = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                pred = head(net(x)).argmax(1)
                correct += (pred == y).sum().item(); n += y.size(0)
        acc = correct / max(n, 1)
        print(f"epoch {ep+1}/{args.epochs} loss={tot/len(tr):.3f} "
              f"val_top1={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save({"backbone": net.state_dict(), "embed_dim": args.embed,
                    "img": args.img, "classes": classes},
                   Path(args.out) / "last.pt")
        if acc > best:
            best = acc
            torch.save({"backbone": net.state_dict(), "embed_dim": args.embed,
                        "img": args.img, "classes": classes},
                       Path(args.out) / "best.pt")
    print(f"done. best val_top1={best:.3f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
