"""Giải mã heatmap -> hộp và lớp Detector (suy luận 1 trang) — cùng phép tiền xử lý/decode với
train_crop/infer_centernet.CenterNetDetector để số đo ở Kaggle và ở pipeline là một.

  decode(hm, wh, off, k, thr)   : max-pool 3×3 NMS + top-k -> [(x1,y1,x2,y2,score)] px ảnh letterbox
  Detector(ckpt, img, device)   : nạp ckpt {"model", "arch", "use_dcn", "img"}; boxes_for_image(bgr)
                                  -> hộp px ẢNH GỐC (score ≥ 0,05, top-k 1024)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import build_model

STRIDE = 4


def decode(hm, wh, off, k: int = 1024, thr: float = 0.05):
    """1 ảnh: hm(1,H,W) wh(2,H,W) off(2,H,W) -> list (x1,y1,x2,y2,score) px input (letterbox)."""
    hm, wh, off = hm.detach().float(), wh.detach().float(), off.detach().float()
    hmax = F.max_pool2d(hm, 3, stride=1, padding=1)
    keep = (hmax == hm).float() * hm
    H, W = hm.shape[-2:]
    k = min(k, H * W)
    scores, idx = torch.topk(keep.view(-1), k)
    sel = scores >= thr
    scores, idx = scores[sel], idx[sel]
    ys = (idx // W).float() + off.reshape(2, -1)[1, idx]
    xs = (idx % W).float() + off.reshape(2, -1)[0, idx]
    w = wh.reshape(2, -1)[0, idx].clamp(min=0) * STRIDE
    h = wh.reshape(2, -1)[1, idx].clamp(min=0) * STRIDE
    cx, cy = xs * STRIDE, ys * STRIDE
    out = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, scores], dim=1).cpu().numpy()
    return [tuple(float(v) for v in row) for row in out]


RESIZE_INTERP = {"linear": 1, "area": 3}     # cv2.INTER_LINEAR = 1, cv2.INTER_AREA = 3


def letterbox(img_bgr: np.ndarray, size: int, resize: str = "area"):
    """Như CenterNetDetector._preprocess (s = size/max(H,W), int cắt, pad đen góc trên-trái) nhưng phép
    resize chọn được: 'linear' = cv2 mặc định của pipeline v1 (răng cưa khi thu ~3×: nét mảnh rụng —
    đo 22/09: v1 tầng n==N 77,0 % linear → 91,9 % area trên cùng 27 trang gốc); 'area' = INTER_AREA
    (khử răng cưa) = mặc định của i5v2 và của books[].detector_resize: area trong pipeline."""
    import cv2
    H, W = img_bgr.shape[:2]
    s = size / max(H, W)
    nh, nw = max(1, int(H * s)), max(1, int(W * s))
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh), interpolation=RESIZE_INTERP[resize])
    return canvas, s


def to_tensor(canvas: np.ndarray) -> torch.Tensor:
    x = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def load_ckpt(path):
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    if "model" not in d:                      # state_dict trần
        d = {"model": d}
    return d


def build_from_ckpt(ckpt_path, pretrained=False):
    """-> (net, meta) ; kiến trúc/DCN đọc từ ckpt (v1: resnet34_fpn + DCN)."""
    d = load_ckpt(ckpt_path)
    arch = d.get("arch", "resnet34_fpn")
    use_dcn = bool(d.get("use_dcn", False))
    net = build_model(arch=arch, pretrained=pretrained, use_dcn=use_dcn)
    missing, unexpected = net.load_state_dict(d["model"], strict=False)
    meta = {"arch": arch, "use_dcn": use_dcn, "img": int(d.get("img", 1024)), "epoch": d.get("epoch"),
            "missing": list(missing), "unexpected": list(unexpected), "path": str(ckpt_path)}
    return net, meta


class Detector:
    """Suy luận 1 trang; thr mặc định 0,05 (ứng viên), lọc lại theo ngưỡng khi đo."""

    def __init__(self, ckpt=None, net=None, img: int = 1024, thr: float = 0.05, device=None, k: int = 1024,
                 resize: str = "area"):
        assert resize in RESIZE_INTERP, resize
        self.resize = resize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if net is None:
            assert ckpt and Path(ckpt).exists(), f"không thấy ckpt {ckpt}"
            net, meta = build_from_ckpt(ckpt)
            self.meta = meta
        else:
            self.meta = {}
        self.net = net.to(self.device).eval()
        self.img, self.thr, self.k = int(img), float(thr), int(k)

    @torch.no_grad()
    def boxes_for_image(self, img_bgr: np.ndarray):
        canvas, s = letterbox(img_bgr, self.img, self.resize)
        x = to_tensor(canvas).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                hm, wh, off = self.net(x)
        else:
            hm, wh, off = self.net(x)
        dets = decode(hm[0], wh[0], off[0], k=self.k, thr=self.thr)
        return [(d[0] / s, d[1] / s, d[2] / s, d[3] / s, d[4]) for d in dets]


def nms_vertical(bx, iou_thr=0.45):
    """Khử 1-chữ-bắt-2-lần theo trục y (bản sao infer_centernet._nms_vertical)."""
    keep = []
    for b in sorted(bx, key=lambda b: b[4], reverse=True):
        y1, y2 = b[1], b[3]
        dup = False
        for kk in keep:
            inter = max(0.0, min(y2, kk[3]) - max(y1, kk[1]))
            union = (y2 - y1) + (kk[3] - kk[1]) - inter
            if union > 0 and inter / union > iou_thr:
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep
