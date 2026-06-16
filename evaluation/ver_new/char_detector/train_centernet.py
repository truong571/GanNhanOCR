"""#3 — CenterNet-style anchorless character detector for woodblock Nôm (roadmap
#5 / pain point A). Self-contained Kaggle trainer + local smoke test.

Trains on char_detector/detect_manifest.json (66,630 boxes / 445 pages bootstrapped
from confirmed crops by bootstrap_boxes.py). Single class ("character"); the count
constraint (N = QN syllables) is applied at INFERENCE via count_constrained.py, so
the network only has to localise — it does not need to count.

Architecture: ResNet-18 backbone + 3 deconv up-sampling -> output stride 4, three
heads (heatmap[1] sigmoid, wh[2], offset[2]) — the standard CenterNet (Zhou et al.,
"Objects as Points", 2019; the HRCenterNet recipe, Tang et al. IEEE BigData 2020).
Loss: penalty-reduced focal on the heatmap + L1 on wh/offset at GT centers.

USAGE
  Local smoke test (CPU, no data, verifies build+forward+backward+decode+count):
    .venv/bin/python evaluation/ver_new/char_detector/train_centernet.py --smoke
  Kaggle (GPU P100/T4):
    pretrain on TKH/MTHv2 then:
    python train_centernet.py --manifest detect_manifest.json --img 768 --epochs 40 \
           --batch 8 --out detector.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
except Exception as e:                       # torch optional for import; required to run
    torch = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from count_constrained import constrain_to_count          # noqa: E402

STRIDE = 4


# --------------------------------------------------------------------------- model
def _make_model():
    class CenterNet(nn.Module):
        def __init__(self, pretrained=False):
            super().__init__()
            bb = torchvision.models.resnet18(weights=(torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                                                      if pretrained else None))
            self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                                      bb.layer1, bb.layer2, bb.layer3, bb.layer4)  # /32, 512ch
            ch = 512
            ups = []
            for oc in (256, 128, 64):
                ups += [nn.ConvTranspose2d(ch, oc, 4, stride=2, padding=1),
                        nn.BatchNorm2d(oc), nn.ReLU(inplace=True)]
                ch = oc
            self.up = nn.Sequential(*ups)                 # /32 -> /4
            def head(out, bias=0.0):
                m = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, out, 1))
                nn.init.constant_(m[-1].bias, bias)
                return m
            self.hm = head(1, bias=-2.19)                 # focal-friendly prior
            self.wh = head(2)
            self.off = head(2)

        def forward(self, x):
            f = self.up(self.stem(x))
            return torch.sigmoid(self.hm(f)).clamp(1e-4, 1 - 1e-4), self.wh(f), self.off(f)
    return CenterNet


# --------------------------------------------------------------------------- targets
def gaussian2D(radius, sigma):
    m = radius
    y, x = np.ogrid[-m:m + 1, -m:m + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(hm, cx, cy, radius):
    d = gaussian2D(radius, max(radius / 3, 1))
    H, W = hm.shape
    l, r = min(cx, radius), min(W - cx, radius + 1)
    t, b = min(cy, radius), min(H - cy, radius + 1)
    if r <= -l or b <= -t:
        return
    masked = hm[cy - t:cy + b, cx - l:cx + r]
    g = d[radius - t:radius + b, radius - l:radius + r]
    np.maximum(masked, g, out=masked)


def build_targets(boxes_xyxy, out_h, out_w, max_obj=256):
    hm = np.zeros((1, out_h, out_w), np.float32)
    wh = np.zeros((max_obj, 2), np.float32)
    off = np.zeros((max_obj, 2), np.float32)
    ind = np.zeros((max_obj,), np.int64)
    mask = np.zeros((max_obj,), np.float32)
    for k, (x1, y1, x2, y2) in enumerate(boxes_xyxy[:max_obj]):
        w, h = (x2 - x1) / STRIDE, (y2 - y1) / STRIDE
        if w <= 0 or h <= 0:
            continue
        cxf, cyf = ((x1 + x2) / 2) / STRIDE, ((y1 + y2) / 2) / STRIDE
        cx, cy = int(cxf), int(cyf)
        if not (0 <= cx < out_w and 0 <= cy < out_h):
            continue
        radius = max(1, int(0.3 * min(w, h)))
        draw_gaussian(hm[0], cx, cy, radius)
        wh[k] = [w, h]; off[k] = [cxf - cx, cyf - cy]
        ind[k] = cy * out_w + cx; mask[k] = 1
    return hm, wh, off, ind, mask


# --------------------------------------------------------------------------- losses
def focal_loss(pred, gt):
    pos = gt.eq(1).float(); neg = (1 - pos)
    neg_w = torch.pow(1 - gt, 4)
    pl = torch.log(pred) * torch.pow(1 - pred, 2) * pos
    nl = torch.log(1 - pred) * torch.pow(pred, 2) * neg_w * neg
    npos = pos.sum()
    pl, nl = pl.sum(), nl.sum()
    return -(nl if npos == 0 else (pl + nl) / npos)


def _gather(feat, ind):
    # feat (B,C,H,W) -> (B,K,C) at flattened indices ind (B,K)
    B, C, H, W = feat.shape
    feat = feat.view(B, C, H * W).permute(0, 2, 1)
    ind = ind.unsqueeze(2).expand(B, ind.size(1), C)
    return feat.gather(1, ind)


def reg_l1(pred, ind, target, mask):
    p = _gather(pred, ind)
    m = mask.unsqueeze(2).expand_as(p)
    return (F.l1_loss(p * m, target * m, reduction="sum")) / (mask.sum() + 1e-4)


# --------------------------------------------------------------------------- decode
def decode(hm, wh, off, k=128, thr=0.2):
    """heatmap peaks -> [(x1,y1,x2,y2,score)] in INPUT pixels (single image)."""
    hm, wh, off = hm.detach(), wh.detach(), off.detach()
    hmax = F.max_pool2d(hm, 3, stride=1, padding=1)
    keep = (hmax == hm).float() * hm
    H, W = hm.shape[-2:]
    scores, idx = torch.topk(keep.view(-1), min(k, H * W))
    ys = (idx // W).float(); xs = (idx % W).float()
    whf = wh.view(2, -1)[:, idx]; offf = off.view(2, -1)[:, idx]
    xs = xs + offf[0]; ys = ys + offf[1]
    w = whf[0]; h = whf[1]
    out = []
    for i in range(len(scores)):
        if scores[i] < thr:
            continue
        cx, cy = float(xs[i]) * STRIDE, float(ys[i]) * STRIDE
        ww, hh = float(w[i]) * STRIDE, float(h[i]) * STRIDE
        out.append((cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2, float(scores[i])))
    return out


# --------------------------------------------------------------------------- eval
def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(net, items, img, dev, thr=0.3, iou_th=0.5):
    """Box-level precision/recall/F1 @IoU + median per-page count error on val pages.
    The number to judge 'đạt hay chưa': F1 should clear ~0.85 and count-error ~0."""
    import cv2
    net.eval()
    tp = fp = fn = 0
    cnt_err = []
    for it in items:
        im = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]
        s = img / max(H, W); nh, nw = int(H * s), int(W * s)
        canvas = np.zeros((img, img, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(im, (nw, nh))
        x = torch.from_numpy((canvas.astype(np.float32) / 255 - 0.5) / 0.5).permute(2, 0, 1)
        with torch.no_grad():
            hm, wh, off = net(x.unsqueeze(0).to(dev))
        dets = [d[:4] for d in decode(hm[0:1], wh[0], off[0], k=256, thr=thr)]
        gt = [(x1 * s, y1 * s, x2 * s, y2 * s) for x1, y1, x2, y2 in it["boxes"]]
        cnt_err.append(abs(len(dets) - len(gt)))
        used = set()
        for d in dets:
            best, bj = iou_th, -1
            for j, g in enumerate(gt):
                if j in used:
                    continue
                v = _iou(d, g)
                if v >= best:
                    best, bj = v, j
            if bj >= 0:
                used.add(bj); tp += 1
            else:
                fp += 1
        fn += len(gt) - len(used)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return {"F1": round(F1, 4), "P": round(P, 4), "R": round(R, 4),
            "median_count_err": float(np.median(cnt_err)) if cnt_err else 0.0, "pages": len(cnt_err)}


# --------------------------------------------------------------------------- HF push
def hf_push(repo, files, token):
    """Upload files to a HuggingFace model repo (mirrors kaggle_train.hf_push)."""
    from huggingface_hub import HfApi, create_repo
    create_repo(repo, repo_type="model", exist_ok=True, token=token)
    api = HfApi()
    for f in files:
        if Path(f).exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=Path(f).name,
                            repo_id=repo, repo_type="model", token=token)
    print(f"  [HF] pushed {[Path(f).name for f in files if Path(f).exists()]} -> {repo}", flush=True)


def _hf_token(arg_token):
    """Resolve HF token: --hf-token arg, env HF_TOKEN, or Kaggle Secret HF_TOKEN."""
    import os
    if arg_token:
        return arg_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


# --------------------------------------------------------------------------- smoke
def smoke():
    assert torch is not None, "torch required"
    dev = "cpu"
    Model = _make_model()
    net = Model(pretrained=False).to(dev)
    img = 256
    x = torch.randn(2, 3, img, img)
    hm, wh, off = net(x)
    oh, ow = img // STRIDE, img // STRIDE
    assert hm.shape == (2, 1, oh, ow), hm.shape
    # fake one column of 9 vertically-stacked boxes
    boxes = [(40, 10 + i * 26, 80, 32 + i * 26) for i in range(9)]
    thm, twh, toff, tind, tmask = build_targets(boxes, oh, ow)
    thm = torch.from_numpy(thm).unsqueeze(0); twh = torch.from_numpy(twh).unsqueeze(0)
    toff = torch.from_numpy(toff).unsqueeze(0); tind = torch.from_numpy(tind).unsqueeze(0)
    tmask = torch.from_numpy(tmask).unsqueeze(0)
    loss = focal_loss(hm[:1], thm) + 0.1 * reg_l1(wh[:1], tind, twh, tmask) \
        + reg_l1(off[:1], tind, toff, tmask)
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
    # decode + count-constrain to N=9
    dets = decode(hm[0:1], wh[0], off[0], k=64, thr=0.0)
    fixed = constrain_to_count([d[:4] for d in dets[:20]], 9)
    print(f"smoke OK | hm {tuple(hm.shape)} | loss {loss.item():.3f} | grad-sum {g:.1f} "
          f"| decoded {len(dets)} -> count_constrained {len(fixed)} (target 9)")
    assert len(fixed) == 9
    assert g > 0
    print("  build + forward + backward + target-gen + decode + count-constraint all work.")


# --------------------------------------------------------------------------- train
def train(args):
    assert torch is not None
    import cv2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = json.load(open(args.manifest, encoding="utf-8"))
    nval = max(1, int(len(man) * args.val_frac))
    val_items, man = man[:nval], man[nval:]       # held-out pages for the đạt/chưa-đạt number
    print(f"manifest pages: train {len(man)} | val {len(val_items)} | device {dev}")

    class DS(torch.utils.data.Dataset):
        def __init__(self, items, img):
            self.items, self.img = items, img
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            it = self.items[i]
            im = cv2.imread(it["image"], cv2.IMREAD_COLOR)
            if im is None:
                im = np.zeros((self.img, self.img, 3), np.uint8)
            H, W = im.shape[:2]
            s = self.img / max(H, W)
            nh, nw = int(H * s), int(W * s)
            im = cv2.resize(im, (nw, nh))
            canvas = np.zeros((self.img, self.img, 3), np.uint8)
            canvas[:nh, :nw] = im
            boxes = [(x1 * s, y1 * s, x2 * s, y2 * s) for x1, y1, x2, y2 in it["boxes"]]
            oh = ow = self.img // STRIDE
            hm, wh, off, ind, mask = build_targets(boxes, oh, ow)
            x = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5
            return (torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(hm),
                    torch.from_numpy(wh), torch.from_numpy(off),
                    torch.from_numpy(ind), torch.from_numpy(mask))

    dl = torch.utils.data.DataLoader(DS(man, args.img), batch_size=args.batch,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    hf_token = _hf_token(args.hf_token) if args.hf_repo else None
    if args.hf_repo and not hf_token:
        print("  [HF] --hf-repo set but no token (env HF_TOKEN / Kaggle Secret HF_TOKEN / --hf-token) "
              "-> skipping HF push; saving locally only.", flush=True)
    net = _make_model()(pretrained=not args.no_pretrained)
    if args.init and Path(args.init).exists():     # e.g. TKH/MTHv2-pretrained weights
        sd = torch.load(args.init, map_location="cpu")
        net.load_state_dict(sd.get("model", sd), strict=False)
        print(f"  init from {args.init}")
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    _HAS_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")   # torch>=2.3 API
    def autocast():
        return (torch.amp.autocast("cuda", enabled=(dev == "cuda")) if _HAS_AMP
                else torch.cuda.amp.autocast(enabled=(dev == "cuda")))
    scaler = (torch.amp.GradScaler("cuda", enabled=(dev == "cuda")) if _HAS_AMP
              else torch.cuda.amp.GradScaler(enabled=(dev == "cuda")))
    best_f1 = -1.0
    for ep in range(args.epochs):
        net.train(); tot = 0.0
        for x, thm, twh, toff, tind, tmask in dl:
            x, thm = x.to(dev), thm.to(dev)
            twh, toff, tind, tmask = twh.to(dev), toff.to(dev), tind.to(dev), tmask.to(dev)
            opt.zero_grad()
            with autocast():
                hm, wh, off = net(x)
                loss = focal_loss(hm, thm) + 0.1 * reg_l1(wh, tind, twh, tmask) \
                    + reg_l1(off, tind, toff, tmask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        ev = evaluate(net, val_items, args.img, dev)
        print(f"epoch {ep+1}/{args.epochs}  loss {tot/len(dl):.4f}  | VAL "
              f"F1 {ev['F1']} P {ev['P']} R {ev['R']} count-err {ev['median_count_err']}", flush=True)
        torch.save({"model": net.state_dict(), "stride": STRIDE, "img": args.img, "val": ev}, args.out)
        best_path = str(Path(args.out).with_suffix(".best.pt"))
        if ev["F1"] > best_f1:
            best_f1 = ev["F1"]
            torch.save({"model": net.state_dict(), "stride": STRIDE, "img": args.img, "val": ev}, best_path)
            if args.hf_repo and hf_token:        # push on each new best (survives Kaggle resets)
                try:
                    hf_push(args.hf_repo, [best_path], hf_token)
                except Exception as e:
                    print(f"  [HF] push failed ({type(e).__name__}: {e})", flush=True)
    if args.hf_repo and hf_token:                # final push: best + last
        try:
            hf_push(args.hf_repo, [str(Path(args.out).with_suffix(".best.pt")), args.out], hf_token)
        except Exception as e:
            print(f"  [HF] final push failed ({type(e).__name__}: {e})", flush=True)
    print(f"\nsaved {args.out} | best VAL F1 {best_f1:.4f}"
          + (f" | HF: {args.hf_repo}" if (args.hf_repo and hf_token) else ""))
    print("  ĐẠT (worth wiring) if F1 >= ~0.85 AND median count-err ~0; else iterate: "
          "add TKH/MTHv2 pretrain via --init, more epochs, or larger --img.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--manifest", default=str(HERE / "detect_manifest.json"))
    ap.add_argument("--img", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out fraction for the đạt/chưa eval")
    ap.add_argument("--init", default="", help="warm-start weights (e.g. TKH/MTHv2-pretrained detector.pt)")
    ap.add_argument("--hf-repo", default="", help="push detector to this HuggingFace model repo (e.g. user/nom-char-det)")
    ap.add_argument("--hf-token", default="", help="HF token (else env HF_TOKEN / Kaggle Secret HF_TOKEN)")
    ap.add_argument("--out", default=str(HERE / "detector.pt"))
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        train(args)


if __name__ == "__main__":
    main()
