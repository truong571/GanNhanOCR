"""PROMPT 3 — Hàm mất mát CenterNet & vòng huấn luyện (AMP, AdamW, Cosine LR).

CenterNetLoss = FocalLoss(heatmap) + λ_size·L1(size) + λ_off·L1(offset)
  • Focal sửa đổi (penalty-reduced) xử lý mất cân bằng nền/tâm cực lớn.
  • L1 chỉ tính tại các ô có mask=1 (tâm chữ thật) qua phép gather theo ind.
  • λ_size = 0.1, λ_off = 1.0 (theo Zhou et al. "Objects as Points").

train_one_epoch / validate dùng AMP (chỉ bật trên CUDA – Kaggle T4/P100), AdamW,
CosineAnnealingLR; lưu checkpoint TỐT NHẤT theo Val F1 (box-level @IoU=0.5) và in
log từng thành phần loss + F1/P/R + sai số đếm trung vị (median count-error).

USAGE
  Smoke (CPU/MPS, không cần dữ liệu — build+forward+backward+decode):
    .venv/bin/python test/train_centernet.py --smoke
  Train trên manifest nội bộ (MPS):
    .venv/bin/python test/train_centernet.py --manifest <detect_manifest.json> \
        --img 512 --epochs 12 --batch 4 --out test/detector_r34.pt
  Pretrain MTHv2 (VOC) rồi fine-tune:
    .venv/bin/python test/train_centernet.py --voc-img <dir> --voc-xml <dir> ... \
        --out mthv2_pretrain.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_centernet import CenterNetDataset, STRIDE, read_mth_items   # noqa: E402
from model_centernet import build_model                      # noqa: E402


# ===========================================================================
#  LOSS
# ===========================================================================
def focal_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Penalty-reduced focal loss cho heatmap (CornerNet/CenterNet).

    pred, gt : (B,1,H,W). Điểm gt==1 là tâm thật; điểm còn lại bị phạt giảm dần
    theo (1-gt)^4 (gần tâm bị phạt nhẹ nhờ Gaussian)."""
    pos = gt.eq(1).float()
    neg = 1.0 - pos
    neg_weights = torch.pow(1 - gt, 4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg
    n_pos = pos.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    return -(neg_loss if n_pos == 0 else (pos_loss + neg_loss) / n_pos)


def _gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B,K,C) tại các chỉ số phẳng ind (B,K)."""
    B, C, H, W = feat.shape
    feat = feat.view(B, C, H * W).permute(0, 2, 1).contiguous()
    ind = ind.unsqueeze(2).expand(B, ind.size(1), C)
    return feat.gather(1, ind)


def reg_l1(pred: torch.Tensor, ind: torch.Tensor, target: torch.Tensor,
           mask: torch.Tensor) -> torch.Tensor:
    """L1 loss cho size/offset, chỉ tại tâm (mask=1)."""
    p = _gather(pred, ind)                       # (B,K,2)
    m = mask.unsqueeze(2).expand_as(p)
    return F.l1_loss(p * m, target * m, reduction="sum") / (mask.sum() + 1e-4)


class CenterNetLoss(nn.Module):
    def __init__(self, w_hm=1.0, w_size=0.1, w_off=1.0):
        super().__init__()
        self.w_hm, self.w_size, self.w_off = w_hm, w_size, w_off

    def forward(self, outputs, targets):
        hm, wh, off = outputs
        l_hm = focal_loss(hm, targets["hm"])
        l_wh = reg_l1(wh, targets["ind"], targets["wh"], targets["mask"])
        l_off = reg_l1(off, targets["ind"], targets["off"], targets["mask"])
        total = self.w_hm * l_hm + self.w_size * l_wh + self.w_off * l_off
        # detach trước khi -> float: bỏ cảnh báo mỗi batch + tránh đồng bộ host (MPS)
        return total, {"hm": float(l_hm.detach()), "wh": float(l_wh.detach()),
                       "off": float(l_off.detach())}


# ===========================================================================
#  DECODE  (heatmap peaks -> box ở pixel ẢNH ĐẦU VÀO)
# ===========================================================================
def decode(hm, wh, off, k: int = 256, thr: float = 0.2):
    """1 ảnh: hm(1,H,W) wh(2,H,W) off(2,H,W) -> [(x1,y1,x2,y2,score), ...] px input."""
    hm, wh, off = hm.detach(), wh.detach(), off.detach()
    hmax = F.max_pool2d(hm, 3, stride=1, padding=1)      # NMS bằng max-pool 3×3
    keep = (hmax == hm).float() * hm
    H, W = hm.shape[-2:]
    k = min(k, H * W)
    scores, idx = torch.topk(keep.view(-1), k)
    ys = (idx // W).float()
    xs = (idx % W).float()
    whf = wh.reshape(2, -1)[:, idx]
    offf = off.reshape(2, -1)[:, idx]
    xs = xs + offf[0]
    ys = ys + offf[1]
    w = whf[0].clamp(min=0)
    h = whf[1].clamp(min=0)
    out = []
    for i in range(len(scores)):
        sc = float(scores[i])
        if sc < thr:
            continue
        cx, cy = float(xs[i]) * STRIDE, float(ys[i]) * STRIDE
        ww, hh = float(w[i]) * STRIDE, float(h[i]) * STRIDE
        out.append((cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2, sc))
    return out


# ===========================================================================
#  EVAL  (box-level P/R/F1 + sai số đếm)
# ===========================================================================
def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@torch.no_grad()
def validate(net, items, img, device, thr=0.3, iou_th=0.5, max_pages=60):
    import cv2
    net.eval()
    tp = fp = fn = 0
    cnt_err = []
    for it in items[:max_pages]:
        im = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]
        s = img / max(H, W)
        nh, nw = int(H * s), int(W * s)
        canvas = np.zeros((img, img, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(im, (nw, nh))
        x = torch.from_numpy((canvas.astype(np.float32) / 255 - 0.5) / 0.5).permute(2, 0, 1)
        hm, wh, off = net(x.unsqueeze(0).to(device))
        dets = [d[:4] for d in decode(hm[0], wh[0], off[0], k=512, thr=thr)]
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
            "median_count_err": float(np.median(cnt_err)) if cnt_err else 0.0,
            "pages": len(cnt_err)}


# ===========================================================================
#  TRAIN
# ===========================================================================
def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _move_targets(batch, device):
    return {k: batch[k].to(device) for k in ("hm", "wh", "off", "ind", "mask")}


def train_one_epoch(net, loader, crit, opt, scaler, device, use_amp, log_every=20):
    net.train()
    agg = {"total": 0.0, "hm": 0.0, "wh": 0.0, "off": 0.0}
    n = 0
    for bi, batch in enumerate(loader):
        x = batch["image"].to(device)
        tg = _move_targets(batch, device)
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                out = net(x)
                loss, parts = crit(out, tg)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out = net(x)
            loss, parts = crit(out, tg)
            loss.backward()
            opt.step()
        agg["total"] += float(loss.detach()); agg["hm"] += parts["hm"]
        agg["wh"] += parts["wh"]; agg["off"] += parts["off"]; n += 1
        if log_every and bi % log_every == 0:
            print(f"    batch {bi:4d}/{len(loader)}  loss {float(loss.detach()):.4f} "
                  f"(hm {parts['hm']:.4f}  wh {parts['wh']:.4f}  off {parts['off']:.4f})",
                  flush=True)
    return {k: v / max(n, 1) for k, v in agg.items()}


def _build_items(args):
    """Trả về (items) list {image, boxes} từ manifest JSON hoặc thư mục VOC."""
    if args.manifest:
        import json
        man = json.load(open(args.manifest, encoding="utf-8"))
        return [{"image": it["image"], "boxes": it["boxes"]} for it in man]
    if args.voc_img and args.voc_xml:
        ds = CenterNetDataset.from_voc_dir(args.voc_img, args.voc_xml, img=args.img)
        return ds.items
    if args.mth_root:                      # MTHv2/TKH (.txt hoặc .xml) -> tự quét
        items = read_mth_items(args.mth_root, coord=args.mth_coord)
        if not items:
            raise SystemExit("--mth-root không ghép được ảnh-nhãn nào (xem log [MTH]).")
        return items
    raise SystemExit("Cần --manifest, HOẶC (--voc-img & --voc-xml), HOẶC --mth-root.")


def hf_push(repo, files, token):
    """Đẩy file lên HuggingFace model repo (giúp ckpt sống sót khi Kaggle reset)."""
    from huggingface_hub import HfApi, create_repo
    create_repo(repo, repo_type="model", exist_ok=True, token=token)
    api = HfApi()
    pushed = []
    for f in files:
        if Path(f).exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=Path(f).name,
                            repo_id=repo, repo_type="model", token=token)
            pushed.append(Path(f).name)
    print(f"  [HF] đẩy {pushed} -> {repo}", flush=True)


def _hf_token(arg_token):
    """Lấy HF token: --hf-token > env HF_TOKEN > Kaggle Secret HF_TOKEN."""
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


def train(args):
    device = _pick_device()
    items = _build_items(args)
    if args.max_items:
        items = items[:args.max_items]
    nval = max(1, int(len(items) * args.val_frac))
    val_items, train_items = items[:nval], items[nval:]
    print(f"[train] device={device} | pages: train {len(train_items)} val {len(val_items)} "
          f"| img {args.img} | DCN {args.dcn}", flush=True)

    train_ds = CenterNetDataset(train_items, img=args.img, train=True)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, drop_last=True,
        pin_memory=(device == "cuda"), persistent_workers=(args.workers > 0))

    net = build_model(arch="resnet34_fpn", pretrained=not args.no_pretrained,
                      use_dcn=args.dcn).to(device)
    if args.init and Path(args.init).exists():       # warm-start (vd. ckpt MTHv2)
        sd = torch.load(args.init, map_location="cpu")
        net.load_state_dict(sd.get("model", sd), strict=False)
        print(f"  init from {args.init}", flush=True)

    crit = CenterNetLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hf_token = _hf_token(args.hf_token) if args.hf_repo else None
    if args.hf_repo and not hf_token:
        print("  [HF] --hf-repo có nhưng thiếu token (env HF_TOKEN / Kaggle Secret / --hf-token) "
              "-> chỉ lưu local.", flush=True)

    best_f1 = -1.0
    best_path = str(Path(args.out).with_suffix(".best.pt"))
    for ep in range(args.epochs):
        t0 = time.time()
        tr = train_one_epoch(net, loader, crit, opt, scaler, device, use_amp,
                             log_every=args.log_every)
        sched.step()
        ev = validate(net, val_items, args.img, device, thr=args.val_thr,
                      max_pages=args.val_pages)
        dt = time.time() - t0
        print(f"epoch {ep+1:2d}/{args.epochs}  {dt:5.1f}s  lr {opt.param_groups[0]['lr']:.2e}"
              f"  loss {tr['total']:.4f} (hm {tr['hm']:.4f} wh {tr['wh']:.4f} off {tr['off']:.4f})"
              f"  | VAL F1 {ev['F1']} P {ev['P']} R {ev['R']} cnt-err {ev['median_count_err']}",
              flush=True)
        ckpt = {"model": net.state_dict(), "arch": "resnet34_fpn", "stride": STRIDE,
                "img": args.img, "use_dcn": bool(args.dcn), "val": ev, "epoch": ep + 1,
                "val_images": [it["image"] for it in val_items]}   # split chuẩn cho PDF
        torch.save(ckpt, args.out)
        if ev["F1"] > best_f1:
            best_f1 = ev["F1"]
            torch.save(ckpt, best_path)
            print(f"    ** best F1 {best_f1:.4f} -> {best_path}", flush=True)
            if args.hf_repo and hf_token:        # đẩy mỗi lần cải thiện (chống reset)
                try:
                    hf_push(args.hf_repo, [best_path], hf_token)
                except Exception as e:
                    print(f"  [HF] push lỗi ({type(e).__name__}: {e})", flush=True)
    if args.hf_repo and hf_token:                # đẩy cuối: best + last
        try:
            hf_push(args.hf_repo, [best_path, args.out], hf_token)
        except Exception as e:
            print(f"  [HF] final push lỗi ({type(e).__name__}: {e})", flush=True)
    print(f"\n[done] saved {args.out} | best VAL F1 {best_f1:.4f} | best ckpt {best_path}")
    print("  ĐẠT (đáng nối vào pipeline) nếu F1 >= ~0.85 và cnt-err nhỏ; nếu chưa: "
          "pretrain MTHv2 (--init), thêm epoch, hoặc tăng --img.")


# ===========================================================================
#  SMOKE
# ===========================================================================
def smoke():
    device = _pick_device()
    print(f"[smoke] device={device}")
    net = build_model(pretrained=False, use_dcn=False).to(device)
    img = 256
    x = torch.randn(2, 3, img, img, device=device)
    hm, wh, off = net(x)
    oh = img // STRIDE
    assert hm.shape == (2, 1, oh, oh)
    # 1 cột 9 ký tự dọc giả
    from data_centernet import build_targets
    boxes = [(40, 10 + i * 26, 80, 32 + i * 26) for i in range(9)]
    thm, twh, toff, tind, tmask = build_targets(boxes, oh, oh)
    tg = {"hm": torch.from_numpy(thm).unsqueeze(0).to(device),
          "wh": torch.from_numpy(twh).unsqueeze(0).to(device),
          "off": torch.from_numpy(toff).unsqueeze(0).to(device),
          "ind": torch.from_numpy(tind).unsqueeze(0).to(device),
          "mask": torch.from_numpy(tmask).unsqueeze(0).to(device)}
    crit = CenterNetLoss()
    loss, parts = crit((hm[:1], wh[:1], off[:1]), tg)
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
    dets = decode(hm[0].float().cpu(), wh[0].float().cpu(), off[0].float().cpu(), k=64, thr=0.0)
    assert g > 0 and len(dets) > 0
    print(f"  smoke OK | loss {float(loss.detach()):.3f} {parts} | grad-sum {g:.1f} | decoded {len(dets)}")
    print("  build + forward + backward + target + loss + decode đều chạy.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--voc-img", default="")
    ap.add_argument("--voc-xml", default="")
    ap.add_argument("--mth-root", default="", help="thư mục gốc MTH/TKH (tự quét ảnh+nhãn .txt/.xml)")
    ap.add_argument("--mth-coord", default="auto", choices=["auto", "xyxy", "xywh"],
                    help="cách hiểu nhãn 4-số khi không phải đa giác 8-số")
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--dcn", action="store_true", help="bật DCNv2 ở conv làm mượt P2")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-thr", type=float, default=0.3)
    ap.add_argument("--val-pages", type=int, default=60)
    ap.add_argument("--max-items", type=int, default=0, help="giới hạn số trang (debug)")
    ap.add_argument("--init", default="", help="warm-start (vd. ckpt MTHv2 đã pretrain)")
    ap.add_argument("--hf-repo", default="", help="đẩy ckpt lên HF model repo (vd. user/nom-char-det)")
    ap.add_argument("--hf-token", default="", help="HF token (hoặc env HF_TOKEN / Kaggle Secret)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--out", default=str(HERE / "detector_r34.pt"))
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        train(args)


if __name__ == "__main__":
    main()
