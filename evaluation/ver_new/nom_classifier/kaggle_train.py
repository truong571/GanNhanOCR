"""SELF-CONTAINED Kaggle trainer — Nôm glyph embedder (ResNet + ArcFace, 256-D).

ONE file: add it as a Kaggle "Utility Script" or paste into a notebook cell.
It trains the encoder used by S3 (visual_signal.NomEncoder) on:
  • real GOLD woodblock crops   (source=crop)  — the target domain
  • the SIMILAR-FONT glyphs      (source=fd)    — gannhanocr-fd; the domain BRIDGE
  • (optional) multi-font glyphs (source=font)  — extra long-tail coverage
trained TOGETHER so ArcFace pulls a char's crop and its similar-font glyph close
(this is what makes the glyph reference usable at inference — see KAGGLE_TRAIN.md).

Checkpoints (best.pt / last.pt) are pushed to a HuggingFace model repo every epoch
→ survive Kaggle's 12h session reset (resume with --resume) and are downloadable.
The checkpoint is drop-in for infer.py / visual_signal.py (keys: backbone,
embed_dim, img, arch, classes).

DATA: upload the folder made by pack_for_kaggle.py (kaggle_pkg/ with images/ +
index.csv + classes.json) as a Kaggle Dataset. --root auto-detects its mount.

Example (Kaggle notebook, GPU P100, Internet ON, Secret HF_TOKEN set):
  !python kaggle_train.py --epochs 40 --arch resnet34 --img 160 \
      --hf-repo <user>/nom-embed --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
_BACKBONES = {"resnet18": torchvision.models.resnet18,
              "resnet34": torchvision.models.resnet34,
              "resnet50": torchvision.models.resnet50}


# --------------------------- augmentation (woodblock) ---------------------------
def _elastic(g, alpha=9.0, sigma=4.0):
    h, w = g.shape
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(g, (x + dx).astype(np.float32), (y + dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderValue=255)


def _augment(g):
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-9, 9), random.uniform(0.85, 1.15))
    M[0, 2] += random.uniform(-0.07, 0.07) * w
    M[1, 2] += random.uniform(-0.07, 0.07) * h
    g = cv2.warpAffine(g, M, (w, h), borderValue=255)
    if random.random() < 0.45:
        g = _elastic(g)
    if random.random() < 0.55:                              # nét đậm/mảnh
        k = np.ones((random.choice([2, 3]),) * 2, np.uint8)
        g = (cv2.erode if random.random() < 0.5 else cv2.dilate)(g, k)
    if random.random() < 0.6:                               # nhiễu mực
        g = np.clip(g.astype(np.float32) + np.random.normal(0, random.uniform(5, 16), g.shape),
                    0, 255).astype(np.uint8)
    if random.random() < 0.3:                               # nhị phân ngẫu nhiên
        g = ((g > random.randint(105, 155)) * 255).astype(np.uint8)
    if random.random() < 0.25:                              # cutout (đứt nét)
        s = int(min(h, w) * random.uniform(0.12, 0.30))
        cy, cx = random.randint(0, h), random.randint(0, w)
        g[max(0, cy - s // 2):cy + s // 2, max(0, cx - s // 2):cx + s // 2] = 255
    return g


def _prep(g, size):
    h, w = g.shape
    s = max(h, w)
    canvas = np.full((s, s), 255, np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = g
    g = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    x = np.repeat(g[None].astype(np.float32) / 255.0, 3, axis=0)
    for c in range(3):
        x[c] = (x[c] - MEAN[c]) / STD[c]
    return torch.from_numpy(x)


class NomDataset(Dataset):
    def __init__(self, index_csv, root, classes, split, img=160, train=True):
        self.root, self.classes, self.size, self.train = Path(root), classes, img, train
        rows = list(csv.DictReader(open(index_csv, encoding="utf-8")))
        self.rows = [r for r in rows if r["split"] == split and r["label"] in classes]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        g = cv2.imread(str(self.root / r["path"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.full((self.size, self.size), 255, np.uint8)
        if self.train:
            g = _augment(g)
        return _prep(g, self.size), self.classes[r["label"]]


# --------------------------- model: backbone + ArcFace ---------------------------
class NomEmbedder(nn.Module):
    def __init__(self, embed=256, pretrained=True, arch="resnet34"):
        super().__init__()
        w = getattr(torchvision.models, {"resnet18": "ResNet18_Weights", "resnet34":
                    "ResNet34_Weights", "resnet50": "ResNet50_Weights"}[arch]).IMAGENET1K_V1 \
            if pretrained else None
        bb = _BACKBONES[arch](weights=w)
        self.arch, n = arch, bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone, self.proj = bb, nn.Linear(n, embed)

    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), dim=1)


class ArcMargin(nn.Module):
    def __init__(self, embed, n_cls, s=30.0, m=0.30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, embed)); nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m

    def forward(self, emb, labels=None):
        cos = emb @ F.normalize(self.W, dim=1).t()
        if labels is None:
            return cos * self.s
        cos = cos.clamp(-1 + 1e-6, 1 - 1e-6)
        target = torch.cos(torch.acos(cos) + self.m)
        oh = F.one_hot(labels, cos.size(1)).to(cos.dtype)
        return (oh * target + (1 - oh) * cos) * self.s


# --------------------------- HuggingFace push / pull ---------------------------
def hf_push(repo, files, token):
    from huggingface_hub import HfApi, create_repo
    create_repo(repo, repo_type="model", exist_ok=True, token=token)
    api = HfApi()
    for f in files:
        if Path(f).exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=Path(f).name,
                            repo_id=repo, repo_type="model", token=token)


def hf_pull(repo, name, dst, token):
    from huggingface_hub import hf_hub_download
    import shutil
    try:
        shutil.copy(hf_hub_download(repo_id=repo, filename=name, repo_type="model", token=token), dst)
        return True
    except Exception:
        return False


def save_ck(path, net, head, opt, sched, scaler, ep, best, a, classes):
    torch.save({"backbone": net.state_dict(), "head": head.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": ep, "best": best,
                "embed_dim": a.embed, "img": a.img, "arch": a.arch, "classes": classes}, path)


# --------------------------- acceptance eval (T2 / T3) ---------------------------
@torch.no_grad()
def quick_eval(net, root, index, classes, img, dev, n=200):
    net.eval()
    rows = list(csv.DictReader(open(index, encoding="utf-8")))
    test, fd = defaultdict(list), {}
    for r in rows:
        if r["source"] == "crop" and r["split"] == "test":
            test[r["label"]].append(r["path"])
        elif r["source"] == "fd":
            fd[r["label"]] = r["path"]

    def emb(p):
        g = cv2.imread(str(Path(root) / p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            return None
        return net(_prep(g, img).unsqueeze(0).to(dev)).squeeze(0).cpu().numpy()

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    multi = [c for c, v in test.items() if len(v) >= 2 and c in fd]
    random.seed(0); random.shuffle(multi); multi = multi[:n]
    if not multi:
        return None
    same, diff = [], []
    for i, c in enumerate(multi):
        a, b = emb(random.choice(test[c])), emb(random.choice(test[c]))
        if a is not None and b is not None:
            same.append(cos(a, b))
        d = multi[(i + 1) % len(multi)]
        ea, eb = emb(random.choice(test[c])), emb(random.choice(test[d]))
        if ea is not None and eb is not None:
            diff.append(cos(ea, eb))
    gal = random.sample(list(fd), min(500, len(fd)))
    for c in multi:
        if c not in gal:
            gal.append(c)
    galE = np.stack([emb(fd[c]) for c in gal]); galN = galE / np.linalg.norm(galE, axis=1, keepdims=True)
    hit = tot = 0
    for c in multi:
        e = emb(random.choice(test[c]))
        if e is None:
            continue
        tot += 1; hit += (gal[int(np.argmax(galN @ (e / np.linalg.norm(e))))] == c)
    return dict(t2_same=statistics.mean(same), t2_diff=statistics.mean(diff),
                t2_sep=statistics.mean(same) - statistics.mean(diff), t3=hit / max(tot, 1), n=len(multi))


# --------------------------- root auto-detect (Kaggle) ---------------------------
def auto_root(given):
    if given:
        return given
    for base in ("/kaggle/input", "."):
        for p in Path(base).rglob("index.csv"):
            return str(p.parent)
    return "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="", help="dir containing index.csv (auto-detected on Kaggle)")
    ap.add_argument("--out", default="/kaggle/working" if Path("/kaggle/working").exists() else ".")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--img", type=int, default=160)
    ap.add_argument("--arch", default="resnet34", choices=list(_BACKBONES))
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hf-repo", default="", help="HF model repo user/name (Secret HF_TOKEN)")
    ap.add_argument("--exclude-glyphs", action="store_true",
                    help="ABLATION: train on real crops ONLY (drops the similar-font/FD glyphs). "
                         "Expect the glyph reference tier to collapse — see KAGGLE_TRAIN.md.")
    a = ap.parse_args()

    root = auto_root(a.root)
    index = str(Path(root) / "index.csv")
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"root={root} | device={dev}")

    # classes.json (build from index if absent)
    cls_path = Path(root) / "classes.json"
    if cls_path.exists():
        classes = json.load(open(cls_path, encoding="utf-8"))
    else:
        labels = sorted({r["label"] for r in csv.DictReader(open(index, encoding="utf-8")) if r["label"]})
        classes = {c: i for i, c in enumerate(labels)}
    n_cls = len(classes)

    if a.exclude_glyphs:                       # ablation: drop synthetic refs from TRAIN
        rows = [r for r in csv.DictReader(open(index, encoding="utf-8"))
                if not (r["split"] == "train" and r["source"] in ("fd", "font"))]
        index = str(Path(a.out) / "index_cropsonly.csv")
        with open(index, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "label", "unicode", "split", "source"])
            w.writeheader(); w.writerows(rows)
        print("  [ablation] TRAIN uses real crops only (glyphs excluded).")

    tr = NomDataset(index, root, classes, "train", a.img, train=True)
    va = NomDataset(index, root, classes, "val", a.img, train=False)
    src = defaultdict(int)
    for r in tr.rows:
        src[r["source"]] += 1
    print(f"train {len(tr)} (by source {dict(src)}) | val {len(va)} | classes {n_cls}")

    pin = dev == "cuda"
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=pin, drop_last=True)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=pin)

    net = NomEmbedder(a.embed, pretrained=not a.no_pretrained, arch=a.arch).to(dev)
    head = ArcMargin(a.embed, n_cls).to(dev)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=a.lr, weight_decay=1e-4)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=max(1, a.warmup))
    cosA = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs - a.warmup))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cosA], milestones=[a.warmup])
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    best, start = 0.0, 0
    last, bestp = out / "last.pt", out / "best.pt"
    tok = os.environ.get("HF_TOKEN", "")
    if a.hf_repo and "/" not in a.hf_repo:
        print(f"  [hf] '{a.hf_repo}' không phải repo_id (user/name) -> bỏ push"); a.hf_repo = ""
    if a.hf_repo and not tok:
        print("  [hf] thiếu HF_TOKEN (Add-ons -> Secrets) -> bỏ push")

    if a.resume:
        if not last.exists() and a.hf_repo and tok:
            hf_pull(a.hf_repo, "last.pt", last, tok)
        if last.exists():
            ck = torch.load(last, map_location=dev)
            net.load_state_dict(ck["backbone"])
            for o, k in [(head, "head"), (opt, "opt"), (sched, "sched"), (scaler, "scaler")]:
                if k in ck:
                    o.load_state_dict(ck[k])
            start, best = int(ck.get("epoch", 0)), float(ck.get("best", 0.0))
            print(f"resume from epoch {start} (best={best:.3f})")

    for ep in range(start, a.epochs):
        net.train(); head.train(); t0 = time.time(); tot = 0.0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                loss = crit(head(net(x), y), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0)
        sched.step()
        net.eval(); head.eval(); correct = ntot = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                correct += (head(net(x)).argmax(1) == y).sum().item(); ntot += y.size(0)
        acc = correct / max(ntot, 1)
        print(f"epoch {ep+1}/{a.epochs} loss={tot/len(tr):.3f} val_top1={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        save_ck(last, net, head, opt, sched, scaler, ep + 1, best, a, classes)
        if acc > best:
            best = acc
            save_ck(bestp, net, head, opt, sched, scaler, ep + 1, best, a, classes)
        if a.hf_repo and tok:
            try:
                hf_push(a.hf_repo, [last] + ([bestp] if acc >= best else []), tok)
            except Exception as e:
                print(f"  [hf-push warn] {type(e).__name__}: {e}")

    print(f"\ndone. best val_top1={best:.3f} -> {bestp}")
    # acceptance: the metrics S3 actually needs (vs DINOv2 +0.01 / 0%)
    if bestp.exists():
        net.load_state_dict(torch.load(bestp, map_location=dev)["backbone"])
    m = quick_eval(net, root, index, classes, a.img, dev)
    if m:
        print(f"ACCEPTANCE (test): T2 separation {m['t2_sep']:+.3f} "
              f"(same {m['t2_same']:.3f}/diff {m['t2_diff']:.3f}) · "
              f"T3 crop->FD retrieval {m['t3']:.1%}  on {m['n']} chars  "
              f"(cần T2>=~0.20, T3>=~0.80; DINOv2 = +0.01 / 0%)")


if __name__ == "__main__":
    main()
