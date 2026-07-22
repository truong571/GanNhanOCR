"""Collapse the sub-center head K→1 and write an infer.py-compatible best.pt.

infer.NomEncoder expects ck = {backbone, head:{W:(n_classes, embed)}, classes,
arch, embed_dim, img} and multiplies head["W"] @ e for the head-logit gate (P0).
Sub-center training produced W of shape (n_classes*K, embed). For each class we
keep the DOMINANT sub-center — the one its real GOLD crops align to most — so the
exported per-class vector is the CLEAN centroid, with noisy/mis-cut variants left
behind on the discarded sub-centers (that is the whole point of sub-centers).

    python ArcFace/export_checkpoint.py            # dominant (embeds GOLD crops)
    python ArcFace/export_checkpoint.py --collapse mean   # fast, no embedding
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from model import NomEmbedder                       # noqa: E402
from dataset import NomCropDataset                  # noqa: E402


def _device():
    return torch.device("cuda" if torch.cuda.is_available()
                        else ("mps" if torch.backends.mps.is_available() else "cpu"))


@torch.no_grad()
def dominant_subcenters(ck, data_dir, device, cap=40):
    """Per class, pick argmax_k mean cosine(GOLD crops_c, subcenter_{c,k})."""
    lab2idx = ck["classes"]; n_cls = len(lab2idx); k = ck["k"]
    emb = NomEmbedder(ck["embed_dim"], pretrained=False, arch=ck["arch"]).to(device)
    emb.load_state_dict(ck["backbone"]); emb.eval()
    W = F.normalize(ck["head_W"].to(device).float(), dim=1).view(n_cls, k, -1)  # (C,K,E)

    by_cls = defaultdict(list)
    for r in csv.DictReader(open(Path(data_dir) / "manifest.csv", encoding="utf-8")):
        if r["source"] == "crop" and r["tier"] == "GOLD" and r["label"] in lab2idx:
            by_cls[lab2idx[r["label"]]].append(str(Path(data_dir) / r["path"]))

    chosen = torch.zeros(n_cls, dtype=torch.long)
    for c in range(n_cls):
        paths = by_cls.get(c, [])[:cap]
        if not paths:                                # crop-less class -> sub-center 0
            continue
        ds = NomCropDataset(paths, [c] * len(paths), img=ck["img"], train=False)
        E = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
        e = emb(E)                                    # (n, E) L2-normed
        # mean cosine of this class's crops to each of its K sub-centers
        score = (e @ W[c].t()).mean(0)                # (K,)
        chosen[c] = int(score.argmax())
    Wc = torch.stack([W[c, chosen[c]] for c in range(n_cls)])   # (C, E)
    return Wc.cpu(), chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints" / "train_best.pt"))
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "checkpoints" / "best.pt"))
    ap.add_argument("--collapse", default="dominant", choices=["dominant", "mean"])
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--hf-token", default="")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    n_cls = len(ck["classes"]); k = ck["k"]
    if args.collapse == "mean" or k == 1:
        W = F.normalize(ck["head_W"].float().view(n_cls, k, -1).mean(1), dim=1)
    else:
        W, chosen = dominant_subcenters(ck, args.data, _device())
        print(f"[export] dominant sub-center picked for {n_cls} classes "
              f"(non-zero picks: {int((chosen > 0).sum())})")

    out = {"backbone": ck["backbone"], "head": {"W": W}, "classes": ck["classes"],
           "arch": ck["arch"], "embed_dim": ck["embed_dim"], "img": ck["img"]}
    torch.save(out, args.out)
    print(f"[export] wrote {args.out}  head.W={tuple(W.shape)}  "
          f"val_head_top1={ck.get('val_head_top1')}")
    if args.hf_repo:
        from hub import resolve_token, ensure_repo, push as hf_push
        tok = resolve_token(args.hf_token)
        if tok:
            ensure_repo(args.hf_repo, tok)
            hf_push(args.out, args.hf_repo, "best.pt", tok)   # deploy artifact on HF too
    print("[export] drop-in: copy to pipeline/align_engine/nom-embed/best.pt "
          "(or point NomEncoder ckpt at it).")


if __name__ == "__main__":
    main()
