"""PROMPT 1 — Lớp Dataset & tiền xử lý dữ liệu cho CenterNet (chữ Hán–Nôm ván khắc).

Module này phục vụ bài toán *định vị tâm chữ* (center-point detection) cho tài liệu
Hán–Nôm mật độ cao, dính nét. Nó hỗ trợ HAI nguồn nhãn:

  • Pascal VOC XML  – đúng định dạng bộ MTHv2 (pretrain).  -> ``read_voc_xml`` /
    ``CenterNetDataset.from_voc_dir``.
  • JSON manifest   – {"image": ..., "boxes": [[x1,y1,x2,y2], ...]} (tập GOLD fine-tune
    nội bộ).  -> ``CenterNetDataset.from_manifest``.

Với mỗi ảnh, lớp Dataset:
  1. Đọc ảnh + box mức ký tự [xmin, ymin, xmax, ymax].
  2. Augmentation cho tài liệu cổ: random scale, dịch/crop, xoay nhẹ, nhiễu
     tương phản/độ sáng (xem ``_augment``).
  3. Letterbox ảnh về ``img × img`` rồi sinh 3 nhãn CenterNet với stride = 4:
       - Heatmap  (1, H/4, W/4)     : Gaussian quanh tâm chữ, bán kính thích ứng.
       - Size     (max_obj, 2)      : [w, h] tại tâm.
       - Offset   (max_obj, 2)      : sai số lượng tử hoá [dx, dy] tại tâm.
       - ind/mask : chỉ số phẳng & cờ hợp lệ để gom (gather) khi tính L1 loss.
  4. Trả về dict: image, hm, wh, off, ind, mask, meta.

Chạy thử (không cần dữ liệu, tự tạo ảnh giả + kiểm tra hình dạng tensor):
    .venv/bin/python test/data_centernet.py --selftest
"""
from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import cv2
except Exception:                                  # cho phép import khi chưa có cv2
    cv2 = None

import torch
from torch.utils.data import Dataset

STRIDE = 4               # tỉ lệ downsampling đầu ra (output stride) — cố định toàn dự án
MAX_OBJ = 384            # số ký tự tối đa / ảnh (trang ván khắc dày ~150–300 box)


# ===========================================================================
#  ĐỌC NHÃN
# ===========================================================================
def read_voc_xml(xml_path: str | Path) -> list[list[float]]:
    """Đọc 1 file Pascal VOC XML (MTHv2) -> list box [xmin, ymin, xmax, ymax].

    MTHv2 gắn nhãn nhiều lớp (text-line, char...). Ta chỉ lấy box MỨC KÝ TỰ:
    object nào có ``<bndbox>`` hợp lệ đều được nhận; có thể lọc theo ``<name>``
    nếu bộ dữ liệu của bạn dùng tên lớp riêng cho ký tự.
    """
    boxes: list[list[float]] = []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception as e:
        print(f"[VOC] lỗi đọc {xml_path}: {e}")
        return boxes
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes


# --------------------------------------------------------------------------- #
#  ĐỌC NHÃN MTH/TKH (HCIILAB) — .txt toạ độ, tự nhận diện định dạng
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def _to_num(tok: str):
    return float(tok) if _NUM_RE.match(tok) else None


def read_label_file(path: str | Path, coord: str = "auto") -> list[list[float]]:
    """Đọc 1 file nhãn -> list box [xmin, ymin, xmax, ymax] mức ký tự.

    Hỗ trợ:
      • .xml  -> Pascal VOC (gọi read_voc_xml).
      • .txt  -> mỗi DÒNG = 1 ký tự. Tự bỏ token nhãn (ký tự CJK/unicode ở đầu),
                 rồi đọc các SỐ còn lại:
                   - >= 8 số  : đa giác 4 góc (x1,y1,...,x4,y4) -> AABB (định dạng
                                phổ biến nhất của MTH/TKH).
                   - == 4 số  : `coord='xyxy'` -> [x1,y1,x2,y2]; `coord='xywh'` ->
                                [x,y,x+w,y+h]; `coord='auto'` đoán theo dấu hiệu.
      Bỏ qua dòng không đủ số / box suy biến.
    """
    p = Path(path)
    if p.suffix.lower() == ".xml":
        return read_voc_xml(p)
    boxes: list[list[float]] = []
    try:
        lines = open(p, encoding="utf-8", errors="ignore").read().splitlines()
    except Exception as e:
        print(f"[MTH] lỗi đọc {p}: {e}")
        return boxes
    for line in lines:
        line = line.strip()
        if not line:
            continue
        toks = re.split(r"[\s,;]+", line)
        # bỏ token đầu nếu KHÔNG phải số (thường là ký tự nhãn)
        if toks and _to_num(toks[0]) is None:
            toks = toks[1:]
        nums = [v for v in (_to_num(t) for t in toks) if v is not None]
        if len(nums) >= 8:                       # đa giác 4 góc
            xs = nums[0:8:2]; ys = nums[1:8:2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        elif len(nums) >= 4:
            a, b, c, d = nums[:4]
            if coord == "xywh" or (coord == "auto" and not (c > a and d > b)):
                x1, y1, x2, y2 = a, b, a + abs(c), b + abs(d)   # x,y,w,h
            else:
                x1, y1, x2, y2 = a, b, c, d                     # x1,y1,x2,y2
        else:
            continue
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
    return boxes


def read_mth_items(root: str | Path, coord: str = "auto",
                   img_exts=(".jpg", ".jpeg", ".png", ".JPG", ".PNG"),
                   lab_exts=(".txt", ".xml"), verbose: bool = True) -> list[dict]:
    """Quét thư mục MTH/TKH (đệ quy) -> list {image, boxes}.

    Ghép ảnh với file nhãn CÙNG TÊN (khác đuôi), dù nằm ở thư mục img/ và label/
    riêng. In thống kê + 1 ví dụ để bạn kiểm tra parser đọc đúng."""
    root = Path(root)
    # bản đồ stem -> file nhãn (đệ quy). MTH/TKH: nhãn KÝ TỰ ở label_char/, nhãn
    # DÒNG (đa giác 8 số) ở label_textline/ -> ƯU TIÊN label_char khi trùng tên.
    lab_files: list[Path] = []
    for ext in lab_exts:
        lab_files += list(root.rglob(f"*{ext}"))

    def _pri(p):
        s = str(p).lower()
        return 0 if "char" in s else (2 if ("textline" in s or "label_line" in s) else 1)

    lab_files.sort(key=_pri)
    lab_map: dict[str, Path] = {}
    for f in lab_files:
        lab_map.setdefault(f.stem, f)
    items, n_pair, n_box = [], 0, 0
    sample = None
    for ext in img_exts:
        for img in root.rglob(f"*{ext}"):
            lab = lab_map.get(img.stem)
            if lab is None:
                continue
            boxes = read_label_file(lab, coord=coord)
            if not boxes:
                continue
            items.append({"image": str(img), "boxes": boxes})
            n_pair += 1; n_box += len(boxes)
            if sample is None:
                sample = (img.name, lab.name, len(boxes), boxes[0])
    if verbose:
        print(f"[MTH] {root}: ghép {n_pair} ảnh-nhãn | {n_box} box "
              f"| nhãn tìm thấy {len(lab_map)}")
        if sample:
            print(f"[MTH] ví dụ: ảnh={sample[0]} nhãn={sample[1]} "
                  f"n_box={sample[2]} box[0]={[round(v,1) for v in sample[3]]}")
        else:
            print("[MTH] ⚠️ KHÔNG ghép được ảnh-nhãn nào — kiểm tra cấu trúc thư mục / "
                  "--mth-coord, hoặc gửi tôi 1 dòng nhãn mẫu.")
    return items


# ===========================================================================
#  SINH NHÃN CENTERNET (heatmap Gaussian / size / offset)
# ===========================================================================
def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    """Bán kính Gaussian thích ứng theo kích thước hộp (CornerNet/CenterNet).

    Chọn bán kính sao cho một hộp dự đoán lệch trong bán kính đó vẫn còn IoU
    >= ``min_overlap`` với hộp thật -> heatmap không phạt nặng các điểm gần tâm.
    """
    a1 = 1.0
    b1 = (h + w)
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0))
    r1 = (b1 - sq1) / 2

    a2 = 4.0
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = math.sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0))
    r2 = (b2 - sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = math.sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / 2
    return max(0.0, min(r1, r2, r3))


def _gaussian2d(radius: int, sigma: float) -> np.ndarray:
    m = radius
    y, x = np.ogrid[-m:m + 1, -m:m + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(hm: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Vẽ (lấy max) một kernel Gaussian vào heatmap tại (cx, cy)."""
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


def build_targets(boxes_xyxy: Sequence[Sequence[float]], out_h: int, out_w: int,
                  max_obj: int = MAX_OBJ):
    """Box trong toạ độ ẢNH ĐÃ LETTERBOX -> (hm, wh, off, ind, mask).

    hm  : (1, out_h, out_w)   heatmap Gaussian.
    wh  : (max_obj, 2)        [w, h] (đơn vị: pixel trên feature map = pixel ảnh / 4).
    off : (max_obj, 2)        [dx, dy] sai số lượng tử hoá.
    ind : (max_obj,)          chỉ số phẳng cy*out_w+cx của tâm.
    mask: (max_obj,)          1 nếu ô đó là tâm hợp lệ.
    """
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
            wh[k] = [w, h]
            off[k] = [cxf - cx, cyf - cy]
            ind[k] = cy * out_w + cx
            mask[k] = 1.0
            k += 1
    return hm, wh, off, ind, mask


# ===========================================================================
#  AUGMENTATION
# ===========================================================================
def _letterbox(img: np.ndarray, size: int):
    """Resize giữ tỉ lệ + pad về size×size. Trả về (canvas, scale, pad_x, pad_y)."""
    H, W = img.shape[:2]
    s = size / max(H, W)
    nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 0, np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, s, 0, 0


def _augment(img: np.ndarray, boxes: np.ndarray, size: int, rng: np.random.Generator):
    """Affine (scale + xoay nhẹ + dịch) trên ảnh gốc, biến đổi box tương ứng,
    rồi nhiễu tương phản/độ sáng. Trả về (ảnh size×size, box đã biến đổi).

    Box xoay -> lấy hộp bao trục (AABB) của 4 góc đã xoay (xấp xỉ chuẩn của
    CenterNet cho góc nhỏ).
    """
    H, W = img.shape[:2]
    s0 = size / max(H, W)                       # scale letterbox cơ sở
    scale = s0 * float(rng.uniform(0.70, 1.25))  # phóng/thu ngẫu nhiên (RỘNG hơn -> đa tỉ lệ)
    angle = float(rng.uniform(-5.0, 5.0))        # xoay nhẹ ±5°

    cx, cy = W / 2.0, H / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    # đưa tâm ảnh về tâm canvas + dịch ngẫu nhiên (jitter)
    new_cx = size / 2.0 + float(rng.uniform(-0.10, 0.10)) * size
    new_cy = size / 2.0 + float(rng.uniform(-0.10, 0.10)) * size
    M[0, 2] += new_cx - cx
    M[1, 2] += new_cy - cy

    canvas = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                            borderValue=(0, 0, 0))

    out = []
    if len(boxes):
        for (x1, y1, x2, y2) in boxes:
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
            ones = np.ones((4, 1), np.float32)
            warped = (M @ np.concatenate([corners, ones], axis=1).T).T  # (4,2)
            nx1, ny1 = warped[:, 0].min(), warped[:, 1].min()
            nx2, ny2 = warped[:, 0].max(), warped[:, 1].max()
            nx1, nx2 = np.clip([nx1, nx2], 0, size - 1)
            ny1, ny2 = np.clip([ny1, ny2], 0, size - 1)
            if nx2 - nx1 >= 2 and ny2 - ny1 >= 2:
                out.append([nx1, ny1, nx2, ny2])

    # nhiễu tương phản (alpha) + độ sáng (beta)
    alpha = float(rng.uniform(0.8, 1.2))
    beta = float(rng.uniform(-20, 20))
    canvas = np.clip(canvas.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    # mô phỏng chất lượng scan (chống overfit -> tăng precision): blur nhẹ + Gaussian noise
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)
    if rng.random() < 0.3:
        canvas = np.clip(canvas.astype(np.float32)
                         + rng.normal(0, float(rng.uniform(3, 10)), canvas.shape),
                         0, 255).astype(np.uint8)
    return canvas, np.array(out, np.float32).reshape(-1, 4)


# ===========================================================================
#  DATASET
# ===========================================================================
class CenterNetDataset(Dataset):
    """Dataset CenterNet đọc danh sách item ``{"image": path, "boxes": [...]}``.

    Dùng ``from_voc_dir`` (MTHv2) hoặc ``from_manifest`` (JSON nội bộ) để khởi tạo.
    """

    def __init__(self, items: list[dict], img: int = 512, train: bool = True,
                 max_obj: int = MAX_OBJ, seed: int = 0):
        assert cv2 is not None, "cần OpenCV (cv2) để chạy Dataset"
        assert img % STRIDE == 0, "img phải chia hết cho STRIDE(=4)"
        self.items = items
        self.img = img
        self.train = train
        self.max_obj = max_obj
        self._seed = seed

    # ---- factory ----------------------------------------------------------
    @classmethod
    def from_manifest(cls, manifest_path: str | Path, **kw) -> "CenterNetDataset":
        items = json.load(open(manifest_path, encoding="utf-8"))
        items = [{"image": it["image"], "boxes": it["boxes"]} for it in items]
        return cls(items, **kw)

    @classmethod
    def from_voc_dir(cls, img_dir: str | Path, xml_dir: str | Path,
                     exts=(".jpg", ".png", ".jpeg", ".JPG"), **kw) -> "CenterNetDataset":
        """Ghép ảnh trong ``img_dir`` với XML cùng tên trong ``xml_dir`` (MTHv2)."""
        img_dir, xml_dir = Path(img_dir), Path(xml_dir)
        items = []
        for xml in sorted(xml_dir.glob("*.xml")):
            stem = xml.stem
            img_path = next((img_dir / f"{stem}{e}" for e in exts
                             if (img_dir / f"{stem}{e}").exists()), None)
            if img_path is None:
                continue
            boxes = read_voc_xml(xml)
            if boxes:
                items.append({"image": str(img_path), "boxes": boxes})
        if not items:
            raise RuntimeError(f"Không ghép được ảnh/XML từ {img_dir} & {xml_dir}")
        return cls(items, **kw)

    # ---- core -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path: str):
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        return im

    def __getitem__(self, i: int):
        it = self.items[i]
        # xử lý ngoại lệ: ảnh lỗi/không có nhãn -> trả mẫu rỗng hợp lệ (mask=0)
        im = self._load(it["image"])
        if im is None:
            im = np.zeros((self.img, self.img, 3), np.uint8)
            boxes = np.zeros((0, 4), np.float32)
        else:
            boxes = np.array(it["boxes"], np.float32).reshape(-1, 4)

        rng = np.random.default_rng(self._seed + i if not self.train else None)
        if self.train and im.shape[0] > 1:
            canvas, boxes = _augment(im, boxes, self.img, rng)
        else:
            canvas, s, _, _ = _letterbox(im, self.img)
            boxes = boxes * s if len(boxes) else boxes.reshape(-1, 4)

        oh = ow = self.img // STRIDE
        hm, wh, off, ind, mask = build_targets(boxes, oh, ow, self.max_obj)

        x = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5     # chuẩn hoá [-1, 1]
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        return {
            "image": x,
            "hm": torch.from_numpy(hm),
            "wh": torch.from_numpy(wh),
            "off": torch.from_numpy(off),
            "ind": torch.from_numpy(ind),
            "mask": torch.from_numpy(mask),
            "n_boxes": int(mask.sum()),
        }


# ===========================================================================
#  SELF-TEST
# ===========================================================================
def _selftest():
    assert cv2 is not None, "cần cv2"
    # tạo 1 ảnh giả 1 cột dọc 8 ký tự dính nhau
    H, W = 800, 200
    img = np.full((H, W, 3), 255, np.uint8)
    boxes = []
    for k in range(8):
        y1 = 20 + k * 95
        cv2.rectangle(img, (40, y1), (160, y1 + 90), (0, 0, 0), 2)
        boxes.append([40, y1, 160, y1 + 90])
    items = [{"image": "<mem>", "boxes": boxes}]

    ds = CenterNetDataset(items, img=256, train=False)
    # ép _load trả ảnh trong bộ nhớ
    ds._load = lambda p: img
    s = ds[0]
    oh = 256 // STRIDE
    assert s["image"].shape == (3, 256, 256), s["image"].shape
    assert s["hm"].shape == (1, oh, oh), s["hm"].shape
    assert s["wh"].shape == (MAX_OBJ, 2)
    assert s["n_boxes"] == 8, s["n_boxes"]
    assert float(s["hm"].max()) == 1.0, "đỉnh heatmap phải = 1 tại tâm"
    # augmentation không làm vỡ
    ds_tr = CenterNetDataset(items, img=256, train=True)
    ds_tr._load = lambda p: img
    s2 = ds_tr[0]
    assert s2["image"].shape == (3, 256, 256)
    # bán kính Gaussian thích ứng > 0
    assert gaussian_radius(90, 120) > 0
    print(f"data_centernet self-test OK | n_boxes={s['n_boxes']} | hm{tuple(s['hm'].shape)} "
          f"| hm.max={float(s['hm'].max()):.2f} | aug n_boxes={s2['n_boxes']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        print("dùng --selftest để kiểm tra nhanh.")
