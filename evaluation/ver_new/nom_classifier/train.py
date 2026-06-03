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
import os
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


def _save_ck(path, net, head, opt, sched, scaler, ep, best, args, classes):
    """Full training state -> resumable. Keeps 'backbone' key for infer.py."""
    torch.save({"backbone": net.state_dict(), "head": head.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": ep, "best": best,
                "embed_dim": args.embed, "img": args.img, "arch": args.arch,
                "classes": classes}, path)


def _hf_push(repo, files, token):
    """Upload checkpoint files to a HF model repo (survives Kaggle restart)."""
    from huggingface_hub import HfApi, create_repo
    create_repo(repo, repo_type="model", exist_ok=True, token=token)
    api = HfApi()
    for f in files:
        if Path(f).exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=Path(f).name,
                            repo_id=repo, repo_type="model", token=token)


def _hf_pull(repo, name, dst, token) -> bool:
    from huggingface_hub import hf_hub_download
    import shutil
    try:
        p = hf_hub_download(repo_id=repo, filename=name, repo_type="model", token=token)
        shutil.copy(p, dst); return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parent.parent.parent))
    ap.add_argument("--index", default=str(HERE / "index.csv"))
    ap.add_argument("--classes", default=str(HERE / "classes.json"))
    ap.add_argument("--out", default=str(HERE / "checkpoints"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--img", type=int, default=160)
    ap.add_argument("--arch", default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2, help="số epoch warmup tuyến tính")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="ResNet-18 từ scratch (khi Kaggle tắt Internet, không tải được weights)")
    ap.add_argument("--resume", action="store_true",
                    help="tiếp tục từ {out}/last.pt (hoặc kéo từ --hf-repo nếu có)")
    ap.add_argument("--hf-repo", default="",
                    help="HF model repo (vd user/nom-embed) để push/pull checkpoint mỗi "
                         "epoch -> KHÔNG mất khi Kaggle restart. Token: env HF_TOKEN.")
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

    net = NomEmbedder(args.embed, pretrained=not args.no_pretrained, arch=args.arch).to(dev)
    head = ArcMargin(args.embed, n_cls).to(dev)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=max(1, args.warmup))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - args.warmup))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[args.warmup])
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    best = 0.0
    start_ep = 0
    last = Path(args.out) / "last.pt"
    hf_token = os.environ.get("HF_TOKEN", "")
    # Footgun guard: --hf-repo must be a repo_id (user/name), NOT a token.
    if args.hf_repo and ("/" not in args.hf_repo or args.hf_repo.startswith("hf_")):
        print(f"  [hf] BỎ QUA push: '--hf-repo {args.hf_repo}' trông như TOKEN, không "
              f"phải repo_id. Đặt token vào Secret HF_TOKEN; HF_REPO='' (tự thành "
              f"<user>/nom-embed) hoặc 'user/ten'. (Train vẫn chạy + lưu local.)", flush=True)
        args.hf_repo = ""
    if args.hf_repo and not hf_token:
        print("  [hf] BỎ QUA push: có --hf-repo nhưng thiếu HF_TOKEN (Add-ons→Secrets).", flush=True)
    # RESUME: continue from last.pt (local; or pull from HF if missing) ----------
    if args.resume:
        if not last.exists() and args.hf_repo and hf_token:
            if _hf_pull(args.hf_repo, "last.pt", last, hf_token):
                print(f"resume: pulled last.pt from HF {args.hf_repo}", flush=True)
        if last.exists():
            ck = torch.load(last, map_location=dev)
            net.load_state_dict(ck["backbone"])
            for obj, k in [(head, "head"), (opt, "opt"), (sched, "sched"), (scaler, "scaler")]:
                if k in ck:
                    obj.load_state_dict(ck[k])
            start_ep = int(ck.get("epoch", 0)); best = float(ck.get("best", 0.0))
            print(f"resume from epoch {start_ep} (best={best:.3f})", flush=True)
        else:
            print("resume: no last.pt found -> training from scratch", flush=True)

    for ep in range(start_ep, args.epochs):
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
        _save_ck(last, net, head, opt, sched, scaler, ep + 1, best, args, classes)
        if acc > best:
            best = acc
            _save_ck(Path(args.out) / "best.pt", net, head, opt, sched, scaler,
                     ep + 1, best, args, classes)
        # Persist OFF Kaggle each epoch -> survives session restart, downloadable.
        if args.hf_repo and hf_token:
            try:
                files = [last] + ([Path(args.out) / "best.pt"] if acc >= best else [])
                _hf_push(args.hf_repo, files, hf_token)
            except Exception as e:
                print(f"  [hf-push warn] {type(e).__name__}: {e}", flush=True)
    print(f"done. best val_top1={best:.3f} -> {args.out}/best.pt"
          + (f" (+HF {args.hf_repo})" if args.hf_repo else ""))


if __name__ == "__main__":
    main()
