"""Dataset nhãn yếu (thạch bản + STT) cho CenterNet v2 + augment theo chẩn đoán I5.

Manifest (bundle/manifest_{train,val,test}.json, toạ độ ở ẢNH BUNDLE): mỗi item
  {image (đường dẫn tương đối bundle), book, page, domain litho|stt|chresto, boxes [[x1,y1,x2,y2]],
   ignore_boxes [[x1,y1,x2,y2]], tiers [[x1,y0,x2,y1,N,start]], split}

Với mỗi ảnh khi train:
  0. tô TRẮNG ignore_boxes (tầng không verified / ô xấu / REVIEW STT) — glyph không nhãn không dạy bỏ sót;
  1. random crop theo cột (p=0,35): cửa sổ rộng 35–75 % W, cao 50–100 % H — phóng to chữ, đa tỉ lệ;
  2. kéo dọc ±10 % (jitter bước cột);
  3. photometric MIỀN: nền xám ngẫu nhiên 100–200 + mực 20–80 (mô phỏng JPG gốc mà v1 "mù"), hoặc
     Otsu (nhị phân), hoặc stretch phân vị, hoặc giữ nguyên;
  4. jitter độ dày nét (erode/dilate 2–3 px, p=0,3), blur nhẹ (p=0,3);
  5. affine của v1 (scale 0,7–1,25 × letterbox, xoay ±5°, dịch ±10 %), tương phản/độ sáng, nhiễu Gauss.
Nhãn: heatmap Gaussian bán kính thích ứng + size + offset (stride 4), như train_crop/data_centernet.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

STRIDE = 4
MAX_OBJ = 384


# ----------------------------------------------------------------------------- manifest
def load_split(data_dir, split: str) -> list[dict]:
    p = Path(data_dir) / f"manifest_{split}.json"
    items = json.load(open(p, encoding="utf-8"))
    for it in items:
        it["image_abs"] = str(Path(data_dir) / it["image"])
    return items


def limit_per_domain(items, n: int) -> list[dict]:
    out = []
    for d in ("litho", "stt", "chresto"):
        out += [it for it in items if it.get("domain") == d][:n]
    return out


def balanced_sampler(items, n_samples: int | None = None, domains=("litho", "stt")):
    """Mỗi miền được rút 50 % số mẫu mỗi epoch (WeightedRandomSampler, có lặp)."""
    cnt = {d: sum(1 for it in items if it.get("domain") == d) for d in domains}
    w = [1.0 / max(1, cnt.get(it.get("domain"), 0)) if it.get("domain") in cnt else 0.0 for it in items]
    n = n_samples or 2 * min(v for v in cnt.values() if v > 0)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=n, replacement=True), cnt


# ----------------------------------------------------------------------------- nhãn CenterNet
def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    a1 = 1.0; b1 = (h + w); c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0))) / 2
    a2 = 4.0; b2 = 2 * (h + w); c2 = (1 - min_overlap) * w * h
    r2 = (b2 - math.sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0))) / 2
    a3 = 4 * min_overlap; b3 = -2 * min_overlap * (h + w); c3 = (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0))) / 2
    return max(0.0, min(r1, r2, r3))


def _gaussian2d(radius: int, sigma: float) -> np.ndarray:
    m = radius
    y, x = np.ogrid[-m:m + 1, -m:m + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(hm: np.ndarray, cx: int, cy: int, radius: int) -> None:
    radius = max(1, int(radius))
    d = _gaussian2d(radius, sigma=max(radius / 3.0, 1.0))
    H, W = hm.shape
    left, right = min(cx, radius), min(W - cx, radius + 1)
    top, bottom = min(cy, radius), min(H - cy, radius + 1)
    if right <= -left or bottom <= -top:
        return
    masked = hm[cy - top:cy + bottom, cx - left:cx + right]
    g = d[radius - top:radius + bottom, radius - left:radius + right]
    if masked.shape == g.shape and masked.size:
        np.maximum(masked, g, out=masked)


def build_targets(boxes_xyxy, out_h: int, out_w: int, max_obj: int = MAX_OBJ):
    hm = np.zeros((1, out_h, out_w), np.float32)
    wh = np.zeros((max_obj, 2), np.float32)
    off = np.zeros((max_obj, 2), np.float32)
    ind = np.zeros((max_obj,), np.int64)
    mask = np.zeros((max_obj,), np.float32)
    k = 0
    for (x1, y1, x2, y2) in boxes_xyxy:
        w = (x2 - x1) / STRIDE
        h = (y2 - y1) / STRIDE
        if w <= 0 or h <= 0:
            continue
        cxf = ((x1 + x2) / 2) / STRIDE
        cyf = ((y1 + y2) / 2) / STRIDE
        cx, cy = int(cxf), int(cyf)
        if not (0 <= cx < out_w and 0 <= cy < out_h):
            continue
        radius = max(1, int(gaussian_radius(math.ceil(h), math.ceil(w))))
        draw_gaussian(hm[0], cx, cy, radius)
        if k < max_obj:
            wh[k] = [w, h]; off[k] = [cxf - cx, cyf - cy]; ind[k] = cy * out_w + cx; mask[k] = 1.0
            k += 1
    return hm, wh, off, ind, mask


# ----------------------------------------------------------------------------- augment
def letterbox(img: np.ndarray, size: int):
    import cv2
    H, W = img.shape[:2]
    s = size / max(H, W)
    nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, s


def _affine(img, boxes, size, rng):
    """Affine của v1: scale 0,7–1,25 × letterbox, xoay ±5°, dịch ±10 %; tương phản/độ sáng; blur/nhiễu."""
    import cv2
    H, W = img.shape[:2]
    s0 = size / max(H, W)
    scale = s0 * float(rng.uniform(0.70, 1.25))
    angle = float(rng.uniform(-5.0, 5.0))
    cx, cy = W / 2.0, H / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += size / 2.0 + float(rng.uniform(-0.10, 0.10)) * size - cx
    M[1, 2] += size / 2.0 + float(rng.uniform(-0.10, 0.10)) * size - cy
    canvas = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    out = []
    for (x1, y1, x2, y2) in boxes:
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
        warped = (M @ np.concatenate([corners, np.ones((4, 1), np.float32)], axis=1).T).T
        nx1, ny1 = warped[:, 0].min(), warped[:, 1].min()
        nx2, ny2 = warped[:, 0].max(), warped[:, 1].max()
        nx1, nx2 = np.clip([nx1, nx2], 0, size - 1)
        ny1, ny2 = np.clip([ny1, ny2], 0, size - 1)
        if nx2 - nx1 >= 2 and ny2 - ny1 >= 2:
            out.append([nx1, ny1, nx2, ny2])
    alpha = float(rng.uniform(0.8, 1.2)); beta = float(rng.uniform(-20, 20))
    canvas = np.clip(canvas.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        kk = int(rng.choice([3, 5]))
        canvas = cv2.GaussianBlur(canvas, (kk, kk), 0)
    if rng.random() < 0.3:
        canvas = np.clip(canvas.astype(np.float32) + rng.normal(0, float(rng.uniform(3, 10)), canvas.shape),
                         0, 255).astype(np.uint8)
    return canvas, np.array(out, np.float32).reshape(-1, 4)


def random_column_crop(img, boxes, rng, w_frac=(0.35, 0.75), h_frac=(0.5, 1.0)):
    """Cắt cửa sổ ngẫu nhiên (dải cột); giữ hộp có tâm trong cửa sổ, kẹp vào biên."""
    H, W = img.shape[:2]
    cw = int(W * rng.uniform(*w_frac)); ch = int(H * rng.uniform(*h_frac))
    x0 = int(rng.integers(0, max(1, W - cw + 1))); y0 = int(rng.integers(0, max(1, H - ch + 1)))
    sub = img[y0:y0 + ch, x0:x0 + cw]
    out = []
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) / 2 - x0, (y1 + y2) / 2 - y0
        if 0 <= cx < cw and 0 <= cy < ch:
            nx1, ny1 = max(0, x1 - x0), max(0, y1 - y0)
            nx2, ny2 = min(cw, x2 - x0), min(ch, y2 - y0)
            if nx2 - nx1 >= 2 and ny2 - ny1 >= 2:
                out.append([nx1, ny1, nx2, ny2])
    return np.ascontiguousarray(sub), np.array(out, np.float32).reshape(-1, 4)


def photometric_domain(img, rng, p_gray=0.4, p_otsu=0.15, p_stretch=0.15):
    """Đổi MIỀN ảnh: nền xám 100–200/mực 20–80 | Otsu nhị phân | stretch phân vị | giữ nguyên."""
    import cv2
    r = rng.random()
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if r < p_gray:
        lo, hi = float(rng.uniform(20, 80)), float(rng.uniform(100, 200))
        g = np.clip(lo + g.astype(np.float32) * (hi - lo) / 255.0, 0, 255).astype(np.uint8)
    elif r < p_gray + p_otsu:
        _, g = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if rng.random() < 0.5:                                  # otsu-giữ-xám (KVK): nền xám nhẹ
            hi = float(rng.uniform(150, 230)); g = np.where(g > 0, hi, g).astype(np.uint8)
    elif r < p_gray + p_otsu + p_stretch:
        lo_p, hi_p = np.percentile(g, 2), np.percentile(g, 98)
        if hi_p > lo_p + 10:
            g = np.clip((g.astype(np.float32) - lo_p) * 255.0 / (hi_p - lo_p), 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def stroke_jitter(img, rng, p=0.3):
    """Độ dày nét: erode (mực dày hơn) hoặc dilate (mực mảnh hơn) 2–3 px."""
    import cv2
    if rng.random() >= p:
        return img
    kk = int(rng.choice([2, 3]))
    ker = np.ones((kk, kk), np.uint8)
    return cv2.erode(img, ker) if rng.random() < 0.5 else cv2.dilate(img, ker)


class I5Dataset(Dataset):
    def __init__(self, items, img: int = 1024, train: bool = True, seed: int = 0,
                 p_crop: float = 0.35, ystretch: float = 0.10, p_gray: float = 0.4,
                 p_otsu: float = 0.15, p_stretch: float = 0.15, p_stroke: float = 0.3,
                 max_obj: int = MAX_OBJ):
        assert img % 32 == 0, "img phải chia hết 32"
        self.items, self.img, self.train, self.seed, self.max_obj = items, img, train, seed, max_obj
        self.p_crop, self.ystretch = p_crop, ystretch
        self.p_gray, self.p_otsu, self.p_stretch, self.p_stroke = p_gray, p_otsu, p_stretch, p_stroke

    def __len__(self):
        return len(self.items)

    def load(self, it):
        import cv2
        im = cv2.imread(it["image_abs"], cv2.IMREAD_COLOR)
        if im is None:
            return None, np.zeros((0, 4), np.float32)
        boxes = np.array(it.get("boxes") or [], np.float32).reshape(-1, 4)
        for g in it.get("ignore_boxes") or []:
            x1, y1, x2, y2 = [int(v) for v in g[:4]]
            im[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 255
        return im, boxes

    def __getitem__(self, i):
        import cv2
        it = self.items[i]
        im, boxes = self.load(it)
        rng = np.random.default_rng(None if self.train else self.seed + i)
        if im is None:
            im = np.full((self.img, self.img, 3), 255, np.uint8)
        if self.train and im.shape[0] > 1:
            if len(boxes) and rng.random() < self.p_crop:
                im, boxes = random_column_crop(im, boxes, rng)
            if self.ystretch > 0:
                sy = float(rng.uniform(1 - self.ystretch, 1 + self.ystretch))
                H, W = im.shape[:2]
                im = cv2.resize(im, (W, max(1, int(H * sy))), interpolation=cv2.INTER_LINEAR)
                if len(boxes):
                    boxes[:, [1, 3]] *= sy
            im = photometric_domain(im, rng, self.p_gray, self.p_otsu, self.p_stretch)
            im = stroke_jitter(im, rng, self.p_stroke)
            canvas, boxes = _affine(im, boxes, self.img, rng)
        else:
            canvas, s = letterbox(im, self.img)
            boxes = boxes * s if len(boxes) else boxes
        oh = ow = self.img // STRIDE
        hm, wh, off, ind, mask = build_targets(boxes, oh, ow, self.max_obj)
        x = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5
        return {"image": torch.from_numpy(x).permute(2, 0, 1).contiguous(), "hm": torch.from_numpy(hm),
                "wh": torch.from_numpy(wh), "off": torch.from_numpy(off), "ind": torch.from_numpy(ind),
                "mask": torch.from_numpy(mask), "n_boxes": int(mask.sum())}


def selftest():
    import cv2
    H, W = 900, 300
    img = np.full((H, W, 3), 255, np.uint8)
    boxes = []
    for k in range(8):
        y1 = 20 + k * 100
        cv2.rectangle(img, (60, y1), (220, y1 + 90), (0, 0, 0), 3)
        boxes.append([60, y1, 220, y1 + 90])
    it = {"image_abs": "<mem>", "boxes": boxes, "ignore_boxes": [[0, 0, 40, 40]], "domain": "litho"}
    for train in (False, True):
        ds = I5Dataset([it], img=256, train=train)
        ds.load = lambda it_, _b=np.array(boxes, np.float32): (img.copy(), _b.copy())
        s = ds[0]
        assert s["image"].shape == (3, 256, 256) and s["hm"].shape == (1, 64, 64)
        if not train:
            assert s["n_boxes"] == 8 and float(s["hm"].max()) == 1.0
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = photometric_domain(img, rng); b = stroke_jitter(img, rng, 1.0)
        c, bb = random_column_crop(img, np.array(boxes, np.float32), rng)
        assert a.shape == img.shape and b.shape == img.shape and c.ndim == 3 and bb.shape[1] == 4
    sampler, cnt = balanced_sampler([{"domain": "litho"}] * 3 + [{"domain": "stt"}] * 9)
    assert cnt == {"litho": 3, "stt": 9} and len(sampler) == 6
    print("i5v2.data selftest OK")


if __name__ == "__main__":
    selftest()
