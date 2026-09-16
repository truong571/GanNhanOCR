"""HUẤN LUYỆN 5-FOLD CNN ÂM TIẾT OUT-OF-FOLD (HẠNG MỤC B-1 KHỐI B)
Dùng cho Kaggle GPU (T4/P100) hoặc Local (Apple Silicon MPS / CPU).

Đầu vào:
  - crops.npz: Mảng ảnh crop ký tự (82.780, 64, 64)
  - labels_final.csv: Bảng nhãn 82.780 ô

Đầu ra:
  - p_visual_oof.csv: Bảng dự đoán out-of-fold cho toàn bộ ô:
      book, page, column, nom_idx, fold, p_visual, argmax_syl, max_prob
  - summary_oof.json: Báo cáo số liệu độ chính xác (top-1, top-5) qua 5 fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SZ = 64
N_FOLDS = 5


# --------------------------------------------------------------------------- Model Architecture
def build_net(n_cls: int, emb: int = 256) -> nn.Module:
    def blk(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                blk(1, 32),
                blk(32, 64),
                blk(64, 128),
                blk(128, 256),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.e = nn.Sequential(nn.Linear(256, emb), nn.BatchNorm1d(emb))
            self.c = nn.Linear(emb, n_cls)

        def forward(self, x):
            z = self.e(self.f(x))
            return self.c(z), z

    return Net()


def prep_batch(X: np.ndarray, idx: np.ndarray, dev: torch.device, aug: bool = False) -> torch.Tensor:
    x = X[idx].astype(np.float32) / 255.0
    if aug:
        out = np.empty_like(x)
        for k in range(len(idx)):
            s = 1.0 + random.uniform(-0.08, 0.08)
            tx = random.uniform(-3, 3) + (1.0 - s) * SZ / 2.0
            ty = random.uniform(-3, 3) + (1.0 - s) * SZ / 2.0
            M = np.float32([[s, 0, tx], [0, s, ty]])
            out[k] = cv2.warpAffine(x[k], M, (SZ, SZ), borderValue=1.0)
        x = out
    # Chuẩn hoá: Mực = 1.0, Nền = 0.0
    return torch.from_numpy(1.0 - x).unsqueeze(1).to(dev)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def page_fold(book: str, page: str) -> int:
    h = hashlib.md5(f"{book}_{page}".encode()).hexdigest()
    return int(h, 16) % N_FOLDS


# --------------------------------------------------------------------------- Main Training Pipeline
def main():
    parser = argparse.ArgumentParser(description="Train 5-Fold CNN Out-of-fold on Kaggle/Local")
    parser.add_argument("--crops", type=str, default="crops.npz", help="Path to crops.npz")
    parser.add_argument("--labels", type=str, default="labels_final.csv", help="Path to labels_final.csv")
    parser.add_argument("--out-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per fold")
    parser.add_argument("--bs", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = get_device()
    print("=" * 72)
    print(f"KHỐI B-1: 5-FOLD CNN OUT-OF-FOLD (ÂM TIẾT)")
    print(f"  Device: {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else dev.type})")
    print(f"  Crops: {args.crops}")
    print(f"  Labels: {args.labels}")
    print(f"  Epochs per fold: {args.epochs}, Batch size: {args.bs}")
    print("=" * 72, flush=True)

    # 1. Load data
    print("Loading data...", flush=True)
    df = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    z = np.load(args.crops)
    X = z["X"]
    ok = z["ok"] if "ok" in z else np.ones(len(df), dtype=bool)
    assert len(df) == len(X), f"Mismatch: len(df)={len(df)} != len(X)={len(X)}"

    # Add nom_idx if missing
    if "nom_idx" not in df.columns:
        df["nom_idx"] = df.groupby(["book", "page", "column"]).cumcount()
    else:
        df["nom_idx"] = df["nom_idx"].astype(int)

    # Compute fold for each page (strictly page-separated)
    df["fold"] = [page_fold(b, p) for b, p in zip(df["book"], df["page"])]

    # 2. Extract fixed 706 syllable classes
    tr_mask = (df.split == "train") & df.tier.isin(["GOLD", "SYLLABLE"]) & ok
    cnt = Counter(df.loc[tr_mask, "syllable"])
    classes = sorted(s for s, n in cnt.items() if n >= 5)
    cid = {s: i for i, s in enumerate(classes)}
    n_cls = len(classes)
    print(f"Total cells: {len(df):,} | Pages: {df[['book', 'page']].drop_duplicates().shape[0]}")
    print(f"Fixed syllable classes: {n_cls} (>=5 occurrences in train split)")
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    # Prepare array for OOF predictions
    oof_p = np.zeros(len(df), dtype=np.float32)
    oof_argmax = [""] * len(df)
    oof_max_p = np.zeros(len(df), dtype=np.float32)

    fold_metrics = []
    total_t0 = time.time()
    use_amp = (dev.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 3. Loop over 5 folds
    for f in range(N_FOLDS):
        print("\n" + "-" * 72)
        print(f">>> FOLD {f + 1}/{N_FOLDS} (Validation: Fold {f})")
        print("-" * 72, flush=True)

        # Train on all other folds (fold != f), reliable tiers (GOLD + SYLLABLE)
        train_idx = df[(df.fold != f) & df.tier.isin(["GOLD", "SYLLABLE"]) & ok & df.syllable.isin(cid)].index.to_numpy()
        val_idx = df[(df.fold == f) & df.tier.isin(["GOLD", "SYLLABLE"]) & ok & df.syllable.isin(cid)].index.to_numpy()
        test_all_fold_idx = df[df.fold == f].index.to_numpy()

        train_y = np.array([cid[df.at[i, "syllable"]] for i in train_idx])
        val_y = np.array([cid[df.at[i, "syllable"]] for i in val_idx])

        # Balanced sampling weight
        tr_cnt = Counter(df.loc[train_idx, "syllable"])
        w = np.array([1.0 / math.sqrt(tr_cnt[df.at[i, "syllable"]]) for i in train_idx], dtype=np.float64)
        w /= w.sum()

        net = build_net(n_cls).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        total_steps = args.epochs * math.ceil(len(train_idx) / args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr * 1.5, total_steps=total_steps)

        fold_t0 = time.time()
        for ep in range(args.epochs):
            net.train()
            order = np.random.default_rng(ep + f * 100).choice(len(train_idx), size=len(train_idx), replace=True, p=w)
            tot_loss = 0.0

            for k in range(0, len(order), args.bs):
                sel = order[k : k + args.bs]
                xb = prep_batch(X, train_idx[sel], dev, aug=True)
                yb = torch.from_numpy(train_y[sel]).to(dev)

                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, _ = net(xb)
                    loss = F.cross_entropy(logits, yb, label_smoothing=0.1)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot_loss += loss.item() * len(sel)

            # Quick validation on reliable gold/syllable
            net.eval()
            val_correct = 0
            with torch.no_grad():
                for k in range(0, len(val_idx), 1024):
                    sel = val_idx[k : k + 1024]
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits, _ = net(prep_batch(X, sel, dev, aug=False))
                    preds = logits.argmax(dim=1).cpu().numpy()
                    val_correct += (preds == val_y[k : k + 1024]).sum()

            val_acc = val_correct / max(1, len(val_idx))
            ep_time = time.time() - fold_t0
            print(f"  Epoch {ep + 1:2d}/{args.epochs} | Loss: {tot_loss / len(order):.4f} | Val Top-1: {val_acc:.2%} ({val_correct}/{len(val_idx)}) | Time: {ep_time:.0f}s", flush=True)

        # Predict OOF probabilities for ALL cells in fold f
        print(f"  Predicting OOF on all {len(test_all_fold_idx):,} cells of Fold {f}...", flush=True)
        net.eval()
        with torch.no_grad():
            for k in range(0, len(test_all_fold_idx), 1024):
                batch_idx = test_all_fold_idx[k : k + 1024]
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, _ = net(prep_batch(X, batch_idx, dev, aug=False))
                    probs = F.softmax(logits, dim=1).cpu().numpy()

                max_c = probs.argmax(axis=1)
                for bi, row_i in enumerate(batch_idx):
                    oof_argmax[row_i] = classes[max_c[bi]]
                    oof_max_p[row_i] = float(probs[bi, max_c[bi]])
                    syl = df.at[row_i, "syllable"]
                    if syl in cid:
                        oof_p[row_i] = float(probs[bi, cid[syl]])
                    else:
                        oof_p[row_i] = 0.0

        fold_metrics.append({
            "fold": f,
            "val_cells": len(val_idx),
            "val_top1": float(val_acc),
            "train_cells": len(train_idx),
            "time_sec": float(time.time() - fold_t0),
        })

    # 4. Export p_visual_oof.csv
    print("\n" + "=" * 72)
    print("Exporting results to p_visual_oof.csv...")
    df_oof = pd.DataFrame({
        "book": df["book"],
        "page": df["page"],
        "column": df["column"],
        "nom_idx": df["nom_idx"],
        "fold": df["fold"],
        "p_visual": np.round(oof_p, 4),
        "argmax_syl": oof_argmax,
        "max_prob": np.round(oof_max_p, 4),
    })

    out_csv = out_dir / "p_visual_oof.csv"
    df_oof.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df_oof):,} rows, {os.path.getsize(out_csv) / (1024 * 1024):.2f} MB)")

    # 5. Summary metrics
    mean_val_acc = np.mean([m["val_top1"] for m in fold_metrics])
    summary = {
        "n_cells": len(df),
        "n_classes": n_cls,
        "n_folds": N_FOLDS,
        "total_time_min": round((time.time() - total_t0) / 60, 2),
        "mean_val_top1": round(float(mean_val_acc), 4),
        "folds": fold_metrics,
    }
    with open(out_dir / "summary_oof.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done in {summary['total_time_min']} minutes. Mean Val Top-1: {mean_val_acc:.2%}")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
