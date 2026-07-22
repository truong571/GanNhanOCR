"""Honest evaluation of the exported best.pt on the PAGE-DISJOINT test split.

Reports the numbers the old encoder failed on:
  * retrieval@1 / @5 via the head  — ranking quality (was 78.1% offline).
  * proxy error-gate AUC — can MLS / Energy tell when the head is WRONG?
    Splits test predictions into head-correct vs head-wrong (auto-label as
    reference) and measures AUC of the open-set score at separating them. This is
    the P1 metric; with HUMAN verdicts (GĐ0) rerun the same code with a verdict
    column for the true error-AUC (auto-label proxy over-states it, so treat as
    an upper bound).
  * MLS = Max-Logit-Score (Vaze ICLR'22); Energy = logsumexp over classes
    (Liu NeurIPS'20) — the two candidate open-set gates.
Same split seed as train.py, so test pages were never trained on.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from model import NomEmbedder                       # noqa: E402
from dataset import NomCropDataset, assign_splits   # noqa: E402


def _auc(scores, is_pos):
    """AUC that a HIGH score marks a positive (here: head-CORRECT). Mann-Whitney."""
    s = np.asarray(scores); y = np.asarray(is_pos).astype(bool)
    p, n = s[y], s[~y]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints" / "best.pt"))
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--split", default="page_disjoint", choices=["page_disjoint", "lobo"])
    ap.add_argument("--holdout", default="")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available()
                       else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ck = torch.load(args.ckpt, map_location=dev)
    lab2idx = ck["classes"]; n_cls = len(lab2idx)
    emb = NomEmbedder(ck["embed_dim"], pretrained=False, arch=ck["arch"]).to(dev)
    emb.load_state_dict(ck["backbone"]); emb.eval()
    Wn = F.normalize(ck["head"]["W"].to(dev).float(), dim=1)     # (C, E)

    import pandas as pd
    df = pd.DataFrame(list(csv.DictReader(open(Path(args.data) / "manifest.csv", encoding="utf-8"))))
    df["split"] = assign_splits(df, mode=args.split, holdout_book=args.holdout,
                                val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    te = df[(df["split"] == "test") & (df["source"] == "crop") & (df["label"].isin(lab2idx))]
    paths = [str(Path(args.data) / p) for p in te["path"]]
    y = [lab2idx[c] for c in te["label"]]
    if not paths:
        print("[eval] test split empty (try --split lobo --holdout stt4)"); return
    ds = NomCropDataset(paths, y, img=ck["img"], train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=4)

    top1 = top5 = tot = 0
    mls_all, energy_all, correct_all = [], [], []
    for x, yy, _ in dl:
        x, yy = x.to(dev), yy.to(dev)
        logits = emb(x) @ Wn.t()                    # (B, C) cosine logits in [-1,1]
        t1 = logits.argmax(1)
        top1 += (t1 == yy).sum().item()
        top5 += (logits.topk(5, 1).indices == yy[:, None]).any(1).sum().item()
        tot += yy.numel()
        mls_all += logits.max(1).values.cpu().tolist()               # Max-Logit-Score
        energy_all += torch.logsumexp(logits * 30.0, 1).cpu().tolist()  # Energy (s=30)
        correct_all += (t1 == yy).cpu().tolist()

    auc_mls = _auc(mls_all, correct_all)
    auc_en = _auc(energy_all, correct_all)
    print(f"[eval] split={args.split}{'/' + args.holdout if args.holdout else ''}  "
          f"test crops={tot}  classes touched={len(set(y))}/{n_cls}")
    print(f"[eval] retrieval@1={top1/tot:.4f}  @5={top5/tot:.4f}")
    print(f"[eval] proxy error-gate AUC (high score => head correct):  "
          f"MLS={auc_mls:.3f}  Energy={auc_en:.3f}")
    print("[eval] NOTE: proxy uses auto-labels as truth (upper bound). Rerun with "
          "human verdicts (GĐ0) for the real error-AUC.")


if __name__ == "__main__":
    main()
