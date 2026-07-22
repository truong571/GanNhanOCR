"""Train the Nôm embedder + Sub-center ArcFace head (roadmap P2).

Levers wired in:
  * page-disjoint / LOBO split          (dataset.assign_splits)   — kills 86% leak
  * sub-center ArcFace, K sub-centers   (model.SubCenterArcMargin)— noisy labels
  * class-balanced or confusion batches (dataset samplers)        — tail + hard-neg
  * tier-weighted loss (GOLD>SILVER>FD) (--w-silver/--w-fd)       — de-circular
  * SAM flat-minima + optional SWA      (sam.SAM / swa_utils)     — raises error-AUC
  * label smoothing                     (--smooth)                — calibration

Saves the raw training checkpoint (sub-center head kept whole). Run
export_checkpoint.py afterwards to collapse K→1 into an infer.py-compatible best.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (NomCropDataset, assign_splits, class_balanced_weights,
                     ConfusionBatchSampler)
from model import NomEmbedder, SubCenterArcMargin
from sam import SAM

HERE = Path(__file__).resolve().parent


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_manifest(data_dir: Path):
    rows = list(csv.DictReader(open(data_dir / "manifest.csv", encoding="utf-8")))
    labels = sorted({r["label"] for r in rows})
    lab2idx = {c: i for i, c in enumerate(labels)}
    sim = {}
    sp = data_dir / "similar_map.json"
    if sp.exists():
        raw = json.load(open(sp, encoding="utf-8"))
        sim = {lab2idx[k]: [lab2idx[v] for v in vs if v in lab2idx]
               for k, vs in raw.items() if k in lab2idx}
    return rows, labels, lab2idx, sim


def build_loaders(args, data_dir: Path):
    rows, labels, lab2idx, sim = load_manifest(data_dir)
    import pandas as pd
    df = pd.DataFrame(rows)
    df["split"] = assign_splits(df, mode=args.split, holdout_book=args.holdout,
                                val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    tier_w = {"GOLD": 1.0, "SILVER": args.w_silver, "FD": args.w_fd}

    def subset(split, sources):
        d = df[(df["split"] == split) & (df["source"].isin(sources))]
        paths = [str(data_dir / p) for p in d["path"]]
        y = [lab2idx[c] for c in d["label"]]
        w = [tier_w.get(t, 1.0) for t in d["tier"]]
        return paths, y, w

    tr_p, tr_y, tr_w = subset("train", {"crop", "fd"})
    va_p, va_y, va_w = subset("val", {"crop"})
    tr = NomCropDataset(tr_p, tr_y, img=args.img, train=True, weights=tr_w)
    va = NomCropDataset(va_p, va_y, img=args.img, train=False, weights=va_w)

    if args.sampler == "confusion" and sim:
        bs = ConfusionBatchSampler(tr_y, sim, batch_size=args.batch, per_class=args.per_class,
                                   seed=args.seed, length=max(1, len(tr_y) // args.batch))
        tl = DataLoader(tr, batch_sampler=bs, num_workers=args.workers, pin_memory=True)
    else:
        samp = WeightedRandomSampler(class_balanced_weights(tr_y, len(labels)),
                                     num_samples=len(tr_y), replacement=True)
        tl = DataLoader(tr, batch_size=args.batch, sampler=samp,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    print(f"[data] classes={len(labels)} train={len(tr_y)} val={len(va_y)} "
          f"sampler={args.sampler} sim-groups={len(sim)}")
    return tl, vl, labels, lab2idx


@torch.no_grad()
def evaluate(emb, head, vl, device):
    emb.eval(); head.eval()
    correct = total = 0
    for x, y, _ in vl:
        x, y = x.to(device), y.to(device)
        logits = head(emb(x))                       # no margin -> plain sub-center cosine*s
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "checkpoints"))
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--k", type=int, default=3, help="ArcFace sub-centers per class")
    ap.add_argument("--s", type=float, default=30.0)
    ap.add_argument("--m", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--smooth", type=float, default=0.1, help="label smoothing")
    ap.add_argument("--sam", action="store_true", help="SAM flat-minima (recommended)")
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--swa", action="store_true", help="SWA over the last swa-epochs")
    ap.add_argument("--swa-epochs", type=int, default=5)
    ap.add_argument("--sampler", default="balanced", choices=["balanced", "confusion"])
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--split", default="page_disjoint", choices=["page_disjoint", "lobo"])
    ap.add_argument("--holdout", default="", help="book code held out for --split lobo (e.g. stt4)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--w-silver", type=float, default=0.5)
    ap.add_argument("--w-fd", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    # ---- resume / Hugging Face sync (survive Kaggle session resets) ----
    ap.add_argument("--hf-repo", default="", help="HF model repo, e.g. mdnt571/nom-embed-arcface")
    ap.add_argument("--hf-token", default="", help="HF token (else env HF_TOKEN / Kaggle Secret)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="resume from local/HF last.pt if present (default ON)")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--push-every", type=int, default=1, help="push last.pt to HF every N epochs")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = _device()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tl, vl, labels, lab2idx = build_loaders(args, Path(args.data))
    n_cls = len(labels)

    emb = NomEmbedder(args.embed_dim, pretrained=True, arch=args.arch).to(device)
    head = SubCenterArcMargin(args.embed_dim, n_cls, k=args.k, s=args.s, m=args.m).to(device)
    params = list(emb.parameters()) + list(head.parameters())

    if args.sam:
        opt = SAM(params, torch.optim.AdamW, rho=args.rho, lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt.base_optimizer, T_max=args.epochs)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    swa_emb = torch.optim.swa_utils.AveragedModel(emb) if args.swa else None

    def wloss(x, y, w):
        logits = head(emb(x), y)                    # margin on target
        per = F.cross_entropy(logits, y, label_smoothing=args.smooth, reduction="none")
        return (per * w).mean()

    # ---- resume + Hugging Face sync (survive Kaggle resets) ---------------
    from hub import resolve_token, ensure_repo, push as hf_push, pull as hf_pull
    token = resolve_token(args.hf_token)
    if args.hf_repo and token:
        ensure_repo(args.hf_repo, token)
    elif args.hf_repo:
        print("[hf] --hf-repo set but no token (HF_TOKEN / Kaggle Secret) -> local-only")
    last_path = Path(args.out) / "last.pt"
    base_opt = (lambda: opt.base_optimizer) if args.sam else (lambda: opt)

    def save_full(ep, best_):
        torch.save({"backbone": emb.state_dict(), "head": head.state_dict(),
                    "opt": base_opt().state_dict(), "sched": sched.state_dict(),
                    "swa": swa_emb.state_dict() if swa_emb else None,
                    "epoch": ep, "best": best_, "classes": lab2idx, "k": args.k,
                    "arch": args.arch, "embed_dim": args.embed_dim, "img": args.img},
                   last_path)

    start_epoch, best = 1, -1.0
    if args.resume:
        rp = str(last_path) if last_path.exists() else (
            hf_pull(args.hf_repo, "last.pt", token, args.out) if args.hf_repo else None)
        if rp and Path(rp).exists():
            ck = torch.load(rp, map_location=device)
            if len(ck.get("classes", {})) == n_cls:
                emb.load_state_dict(ck["backbone"]); head.load_state_dict(ck["head"])
                base_opt().load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
                if swa_emb and ck.get("swa"):
                    swa_emb.load_state_dict(ck["swa"])
                start_epoch, best = int(ck["epoch"]) + 1, float(ck.get("best", -1.0))
                print(f"[resume] {rp} -> continue at epoch {start_epoch} (best {best:.4f})")
            else:
                print(f"[resume] class mismatch ({len(ck.get('classes',{}))} vs {n_cls}) -> fresh")
    if start_epoch > args.epochs:
        print(f"[resume] already at {args.epochs} epochs — nothing to do."); return

    for ep in range(start_epoch, args.epochs + 1):
        emb.train(); head.train()
        run = 0.0; nb = 0
        for x, y, w in tl:
            x, y, w = x.to(device), y.to(device), w.float().to(device)
            if args.sam:
                loss = wloss(x, y, w); loss.backward(); opt.first_step(zero_grad=True)
                wloss(x, y, w).backward(); opt.second_step(zero_grad=True)
            else:
                opt.zero_grad(); loss = wloss(x, y, w); loss.backward(); opt.step()
            run += float(loss.detach()); nb += 1
        sched.step()
        print(f"[ep {ep:02d}] train loss={run / max(1, nb):.4f}", end="  ")
        if args.swa and ep > args.epochs - args.swa_epochs:
            swa_emb.update_parameters(emb)
        acc = evaluate(emb, head, vl, device)
        print(f"[ep {ep:02d}/{args.epochs}] val head-top1={acc:.4f} lr={sched.get_last_lr()[0]:.2e}")
        if acc >= best:
            best = acc
            torch.save({"backbone": emb.state_dict(),
                        "head_W": head.W.detach().cpu(), "k": args.k,
                        "classes": lab2idx, "arch": args.arch,
                        "embed_dim": args.embed_dim, "img": args.img,
                        "val_head_top1": acc},
                       Path(args.out) / "train_best.pt")
            print(f"          -> saved train_best.pt (val head-top1 {acc:.4f})")
        # full-state resume checkpoint every epoch + push to HF (Kaggle-reset safe)
        save_full(ep, best)
        if args.hf_repo and token and (ep % args.push_every == 0 or ep == args.epochs):
            hf_push(last_path, args.hf_repo, "last.pt", token)
            hf_push(Path(args.out) / "train_best.pt", args.hf_repo, "train_best.pt", token)

    if args.swa:
        torch.optim.swa_utils.update_bn(
            ((x.to(device),) for x, _, _ in tl), swa_emb, device=device)
        acc = evaluate(swa_emb.module, head, vl, device)
        print(f"[swa] val head-top1={acc:.4f}")
        if acc >= best:
            torch.save({"backbone": swa_emb.module.state_dict(),
                        "head_W": head.W.detach().cpu(), "k": args.k,
                        "classes": lab2idx, "arch": args.arch,
                        "embed_dim": args.embed_dim, "img": args.img,
                        "val_head_top1": acc}, Path(args.out) / "train_best.pt")
            print(f"[swa] -> saved train_best.pt (val head-top1 {acc:.4f})")
    print(f"[done] best val head-top1={best:.4f}")


if __name__ == "__main__":
    main()
