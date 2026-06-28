"""ĐỘT PHÁ #1 — train a per-character Nom RECOGNIZER on the auto-labels and show it
beats the SinoNom OCR teacher on the teacher-OOV / error strata. Self-contained
Kaggle trainer + HF push + local --smoke.

Student = ResNet34 -> softmax over the K char classes, trained on the consensus
auto-labels (GOLD [+ SILVER]). Warm-start the backbone from the trained encoder
(nom-embed/best.pt) for fast convergence. Each epoch evaluates on the held-out
(book-disjoint) test split:
   STUDENT acc (argmax==true)  vs  TEACHER acc (ocr_char==true)
   + TEACHER-OOV student acc (true ∉ {emittable ocr_char}; teacher=0 there)
   + teacher-WRONG student recovery (ocr_char!=true)

--target consensus  -> the real student (A).   --target ocr -> the control (B,
trained on the teacher's raw labels) for the 3-way A/B/C comparison.

USAGE
  Local smoke (CPU, tiny, verifies build+train-step+eval):
    .venv/bin/python evaluation/ver_new/nom_recognizer/train_recognizer.py --smoke
  Kaggle (GPU T4/P100):
    python train_recognizer.py --root . --target consensus --epochs 30 --img 160 \
           --init encoder_best.pt --out recognizer.pt --hf-repo <user>/nom-recognizer
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
except Exception:
    torch = None

HERE = Path(__file__).resolve().parent
MEAN = STD = 0.5
_IDC = None


def _is_cjk(ch):
    if not ch or len(ch) != 1:
        return False
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2A6DF
            or 0x2A700 <= o <= 0x2EBEF or 0xF900 <= o <= 0xFAFF)


# ----------------------------------------------------------------- model
def _backbone(arch, pretrained):
    ctor = {"resnet18": torchvision.models.resnet18, "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50}[arch]
    w = None
    if pretrained:
        w = getattr(torchvision.models, {"resnet18": "ResNet18_Weights", "resnet34": "ResNet34_Weights",
                                         "resnet50": "ResNet50_Weights"}[arch]).IMAGENET1K_V1
    bb = ctor(weights=w)
    nf = bb.fc.in_features
    bb.fc = nn.Identity()
    return bb, nf


class Recognizer(nn.Module):
    def __init__(self, n_cls, arch="resnet34", pretrained=True):
        super().__init__()
        self.backbone, nf = _backbone(arch, pretrained)
        self.fc = nn.Linear(nf, n_cls)
        self.arch = arch

    def forward(self, x):
        return self.fc(self.backbone(x))


def _prep(gray, img):
    h, w = gray.shape
    s = max(h, w)
    cv = np.full((s, s), 255, np.uint8)
    cv[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = gray
    g = cv2.resize(cv, (img, img), interpolation=cv2.INTER_AREA)
    x = np.repeat(g[None].astype(np.float32) / 255.0, 3, axis=0)
    return (x - MEAN) / STD


def _augment(g):
    import random
    if random.random() < 0.5:
        k = np.ones((2, 2), np.uint8)
        g = cv2.erode(g, k) if random.random() < 0.5 else cv2.dilate(g, k)
    if random.random() < 0.5:
        a = random.uniform(-5, 5); h, w = g.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, random.uniform(0.92, 1.06))
        g = cv2.warpAffine(g, M, (w, h), borderValue=255)
    if random.random() < 0.3:
        g = np.clip(g.astype(np.int16) + np.random.randint(-18, 18, g.shape), 0, 255).astype(np.uint8)
    return g


# ----------------------------------------------------------------- HF push
def hf_push(repo, files, token):
    from huggingface_hub import HfApi, create_repo
    create_repo(repo, repo_type="model", exist_ok=True, token=token)
    api = HfApi()
    for f in files:
        if Path(f).exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=Path(f).name,
                            repo_id=repo, repo_type="model", token=token)
    print(f"  [HF] pushed {[Path(f).name for f in files if Path(f).exists()]} -> {repo}", flush=True)


def _hf_token(arg):
    import os
    if arg:
        return arg
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


# ----------------------------------------------------------------- data
def load_rows(root, labels_csv):
    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8")))
    classes = sorted({r["label"] for r in rows
                      if r["tier"] == "GOLD" and r["label"] and _is_cjk(r["label"])})
    cls2idx = {c: i for i, c in enumerate(classes)}
    emittable = {r["ocr_char"] for r in rows if r["ocr_char"] and _is_cjk(r["ocr_char"])}
    return rows, classes, cls2idx, emittable


class CropDS:
    """Module-level so DataLoader workers can pickle it (num_workers>0 on Kaggle)."""
    def __init__(self, items, root, img, aug):
        self.items, self.root, self.img, self.aug = items, root, img, aug
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        img, y, _, _ = self.items[i]
        path = img if Path(img).is_absolute() else str(self.root / img)
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.full((self.img, self.img), 255, np.uint8)
        if self.aug:
            g = _augment(g)
        return torch.from_numpy(_prep(g, self.img)).float(), y


# ----------------------------------------------------------------- smoke
def smoke():
    assert torch is not None
    net = Recognizer(50, "resnet18", pretrained=False)
    x = torch.randn(4, 3, 96, 96)
    out = net(x)
    assert out.shape == (4, 50), out.shape
    loss = F.cross_entropy(out, torch.randint(0, 50, (4,)))
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
    # prep + augment on a fake glyph
    fake = np.full((70, 60), 255, np.uint8); fake[20:50, 15:45] = 0
    p = _prep(_augment(fake), 96)
    assert p.shape == (3, 96, 96) and g > 0
    print(f"smoke OK | logits {tuple(out.shape)} | loss {loss.item():.3f} | grad {g:.1f} | prep {p.shape}")
    print("  build + forward + CE backward + prep/augment work.")


# ----------------------------------------------------------------- train
def train(a):
    import os, random
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    root = Path(a.root)
    labels_csv = Path(a.labels) if a.labels else root / "labels.csv"
    rows, classes, cls2idx, emittable = load_rows(root, labels_csv)
    K = len(classes)
    print(f"classes {K} | emittable(teacher) {len(emittable)} | device {dev} | target={a.target}", flush=True)

    tiers = {"GOLD"} | ({"SILVER"} if a.use_silver else set())
    def target_of(r):
        if a.target == "ocr":
            return cls2idx.get(r["ocr_char"], -1)        # control B: teacher's raw label
        return cls2idx.get(r["label"], -1)               # A: consensus label

    def pick(split):
        out = []
        for r in rows:
            if r["tier"] in tiers and r["split"] == split and r["image"] and (root / r["image"]).exists():
                y = target_of(r)
                if y >= 0:
                    out.append((r["image"], y, r["ocr_char"], r["label"]))
        return out
    tr, te = pick("train"), pick("test")
    print(f"train(crops) {len(tr)} | test {len(te)}", flush=True)

    # add gannhanocr-fd GENERATED glyphs to TRAIN (the user's glyph set). Gives the softmax
    # head a sample for EVERY class incl. rare / 0-crop ones; rare classes oversampled.
    # NEVER added to test (test = real crops only -> honest eval).
    n_fd = 0
    if not a.no_fd:
        fd_dir = Path(a.fd_dir) if a.fd_dir else (root / "fd")
        if not fd_dir.exists():
            cand = Path(__file__).resolve().parents[3] / "gannhanocr-fd"
            fd_dir = cand if cand.exists() else fd_dir
        cropcnt = Counter(y for _, y, _, _ in tr)
        if fd_dir.exists():
            for c, idx in cls2idx.items():
                hx = f"{ord(c):X}"
                p = next((q for q in (fd_dir / f"U+{hx}.png", fd_dir / hx[:2] / f"U+{hx}.png") if q.exists()), None)
                if p is None:
                    continue
                reps = a.fd_reps if cropcnt.get(idx, 0) < a.fd_rare else 1
                for _ in range(reps):
                    tr.append((str(p.resolve()), idx, "", c)); n_fd += 1
        print(f"  + {n_fd} gannhanocr-fd glyphs -> train (fd_dir={fd_dir.name}; rare<{a.fd_rare} crops x{a.fd_reps})", flush=True)
    print(f"train(total) {len(tr)}", flush=True)

    dl = torch.utils.data.DataLoader(CropDS(tr, root, a.img, True), batch_size=a.batch, shuffle=True,
                                     num_workers=a.workers, drop_last=True)

    net = Recognizer(K, a.arch, pretrained=not a.no_pretrained)
    if a.init and Path(a.init).exists():                 # warm-start backbone from the encoder
        ck = torch.load(a.init, map_location="cpu")
        sd = ck.get("backbone", ck)
        bsd = {k.replace("backbone.", "", 1): v for k, v in sd.items() if k.startswith("backbone.")}
        miss = net.backbone.load_state_dict(bsd, strict=False)
        print(f"  warm-start backbone from {a.init} (missing {len(miss.missing_keys)})", flush=True)
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda")) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    hf_tok = _hf_token(a.hf_token) if a.hf_repo else None

    @torch.no_grad()
    def evaluate():
        net.eval()
        n = s_hit = t_hit = oov_n = oov_s = tw_n = tw_s = 0
        for img, y, ocr, true in te:
            g = cv2.imread(str(root / img), cv2.IMREAD_GRAYSCALE)
            if g is None: continue
            x = torch.from_numpy(_prep(g, a.img)).float().unsqueeze(0).to(dev)
            pred = classes[int(net(x).argmax(1))]
            n += 1; s_ok = int(pred == true); s_hit += s_ok; t_hit += int(ocr == true)
            if true not in emittable: oov_n += 1; oov_s += s_ok
            if ocr != true: tw_n += 1; tw_s += s_ok
        return dict(n=n, student=s_hit/max(n,1), teacher=t_hit/max(n,1),
                    oov_n=oov_n, oov_student=oov_s/max(oov_n,1),
                    tw_n=tw_n, tw_recovery=tw_s/max(tw_n,1))

    best = -1.0
    for ep in range(a.epochs):
        net.train(); tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with (torch.amp.autocast("cuda", enabled=(dev=="cuda")) if hasattr(torch,"amp") else torch.cuda.amp.autocast(enabled=(dev=="cuda"))):
                loss = F.cross_entropy(net(x), y, label_smoothing=0.05)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        sched.step()
        ev = evaluate()
        print(f"epoch {ep+1}/{a.epochs} loss {tot/len(dl):.3f} | TEST student {ev['student']:.3f} "
              f"teacher {ev['teacher']:.3f} | OOV student {ev['oov_student']:.3f} (n={ev['oov_n']}) "
              f"| teacher-wrong recovery {ev['tw_recovery']:.3f} (n={ev['tw_n']})", flush=True)
        torch.save({"backbone": net.backbone.state_dict(), "fc": net.fc.state_dict(),
                    "classes": classes, "arch": a.arch, "img": a.img, "target": a.target, "test": ev}, a.out)
        if ev["student"] > best:
            best = ev["student"]
            bp = str(Path(a.out).with_suffix(".best.pt"))
            torch.save({"backbone": net.backbone.state_dict(), "fc": net.fc.state_dict(),
                        "classes": classes, "arch": a.arch, "img": a.img, "target": a.target, "test": ev}, bp)
            if a.hf_repo and hf_tok:
                try: hf_push(a.hf_repo, [bp], hf_tok)
                except Exception as e: print(f"  [HF] {e}", flush=True)
    if a.hf_repo and hf_tok:
        try: hf_push(a.hf_repo, [str(Path(a.out).with_suffix(".best.pt")), a.out], hf_tok)
        except Exception as e: print(f"  [HF] {e}", flush=True)
    print(f"\nsaved {a.out} | best TEST student acc {best:.4f}"
          + (f" | HF {a.hf_repo}" if (a.hf_repo and hf_tok) else ""))
    print("  HEADLINE (target=consensus): OOV-student >0 while teacher=0, + teacher-wrong recovery "
          "= dataset enabled a model that fixes/handles what the OCR teacher cannot.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent / "dataset_out"))
    ap.add_argument("--labels", default="")
    ap.add_argument("--target", default="consensus", choices=["consensus", "ocr"])
    ap.add_argument("--use-silver", action="store_true", help="also train on SILVER (noisier)")
    ap.add_argument("--fd-dir", default="", help="gannhanocr-fd glyph dir (default: root/fd or repo gannhanocr-fd)")
    ap.add_argument("--no-fd", action="store_true", help="do NOT add gannhanocr-fd glyphs to train")
    ap.add_argument("--fd-reps", type=int, default=4, help="copies of the fd glyph for rare classes")
    ap.add_argument("--fd-rare", type=int, default=5, help="a class with < this many crops is 'rare'")
    ap.add_argument("--arch", default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--img", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--init", default="", help="warm-start backbone from encoder ckpt (nom-embed/best.pt)")
    ap.add_argument("--hf-repo", default=""); ap.add_argument("--hf-token", default="")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "recognizer.pt"))
    a = ap.parse_args()
    if a.smoke:
        smoke()
    else:
        train(a)


if __name__ == "__main__":
    main()
