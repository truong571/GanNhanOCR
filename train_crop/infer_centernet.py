"""PROMPT 4 (+ Bước 6) — Suy luận CenterNet + RÀNG BUỘC N + SEAM CARVING + adapter pipeline.

Đòn bẩy riêng của dự án: sau căn chỉnh Nôm↔Quốc-ngữ ta BIẾT trước số âm tiết N của
mỗi cột. Detector chỉ cần *định vị*; module này hoà giải số tâm tìm được M với N:

  • Trích đỉnh cục bộ trên heatmap bằng Max-Pool 3×3 (không cần NMS theo IoU).
  • M == N : nhận.
  • M  > N : cắt thừa nét -> GIỮ ĐÚNG N tâm có điểm tin cậy cao nhất.
  • M  < N : dính chữ nặng -> TÁCH hộp cao vượt trội. Điểm cắt chọn theo:
        - 'seam'     (mặc định, Bước 6): ĐƯỜNG ĐI NĂNG LƯỢNG TỐI THIỂU (seam carving)
                     len dọc theo kẽ hở ít mực nhất giữa hai chữ — bền hơn cắt thẳng.
        - 'valley'   : thung lũng hình chiếu ngang (1 hàng ít mực nhất).
        - 'midpoint' : cắt giữa hộp.

TÍCH HỢP PIPELINE CHÍNH (align_production --reseg detector):
  • boxes_for_page(page_bgr)            -> [(x1,y1,x2,y2,score)] px gốc (chạy detector 1 lần/trang)
  • column_boxes(page_boxes, x_range, n)-> đúng n hộp của 1 cột (giống detector_infer.py)
  • make_valley_split(gray, 'seam')     -> callback cho count_constrained.constrain_to_count

USAGE
  Smoke: .venv/bin/python test/infer_centernet.py --smoke
  Thật:  .venv/bin/python test/infer_centernet.py --ckpt test/detector_r34.best.pt \
             --image <cot.png> --n 9 --out test/crops_out --carve
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model_centernet import build_model, STRIDE              # noqa: E402
from train_centernet import decode                           # noqa: E402


# ===========================================================================
#  ĐIỂM CẮT: projection valley & SEAM CARVING (đường đi năng lượng tối thiểu)
# ===========================================================================
def projection_valley(gray_box: np.ndarray) -> int:
    """y (cục bộ) của thung lũng mực sâu nhất trong dải giữa hộp (cắt THẲNG)."""
    H = gray_box.shape[0]
    if H < 4:
        return H // 2
    ink = (255.0 - gray_box.astype(np.float32)).sum(axis=1)
    if H >= 5:
        ink = np.convolve(ink, np.ones(3, np.float32) / 3.0, mode="same")
    lo, hi = max(1, int(0.2 * H)), min(H - 1, int(0.8 * H))
    if hi <= lo:
        return H // 2
    return lo + int(np.argmin(ink[lo:hi]))


def compute_seam(gray_box: np.ndarray, y_lo: int, y_hi: int):
    """SEAM CARVING: đường đi NGANG năng-lượng-tối-thiểu (ink) qua dải [y_lo,y_hi].

    Mỗi cột x chọn 1 hàng y, ràng buộc |y(x)-y(x-1)| <= 1 (liền mạch). Năng lượng =
    ink = 255-gray (mực càng đậm càng "đắt"); seam luồn theo KẼ HỞ giữa 2 chữ.
    Trả về (seam[W] toạ-độ-y-tuyệt-đối-trong-box, cut = trung vị seam)."""
    H, W = gray_box.shape[:2]
    y_lo = max(0, min(y_lo, H - 1))
    y_hi = max(y_lo, min(y_hi, H - 1))
    if W < 2 or y_hi <= y_lo:
        return None, (y_lo + y_hi) // 2
    ink = (255.0 - gray_box.astype(np.float32))[y_lo:y_hi + 1, :]   # (R,W) năng lượng
    R = ink.shape[0]
    M = np.empty((R, W), np.float32)
    back = np.zeros((R, W), np.int32)
    M[:, 0] = ink[:, 0]
    rows = np.arange(R)
    for x in range(1, W):
        prev = M[:, x - 1]
        up = np.concatenate(([np.inf], prev[:-1]))     # prev[r-1]
        down = np.concatenate((prev[1:], [np.inf]))    # prev[r+1]
        cand = np.stack([up, prev, down])              # (3,R): r-1, r, r+1
        arg = np.argmin(cand, axis=0)
        M[:, x] = ink[:, x] + cand[arg, rows]
        back[:, x] = np.clip(rows + (arg - 1), 0, R - 1)
    r = int(np.argmin(M[:, -1]))
    seam = np.empty(W, np.int32)
    for x in range(W - 1, -1, -1):
        seam[x] = y_lo + r
        r = int(back[r, x])
    return seam, int(np.median(seam))


def _split_cut(sub_gray: np.ndarray, method: str) -> int:
    """Điểm cắt cục bộ (y) trong sub_gray theo method ('seam'|'valley'|'midpoint')."""
    H = sub_gray.shape[0]
    if method == "midpoint" or H < 6 or sub_gray.size == 0:
        return H // 2
    lo, hi = int(0.2 * H), int(0.8 * H)
    if method == "seam":
        _, cut = compute_seam(sub_gray, lo, hi)
        return cut
    return projection_valley(sub_gray)


def make_valley_split(gray_image: np.ndarray, method: str = "seam"):
    """Trả callback `f(box)->y_tuyệt_đối` cho count_constrained.constrain_to_count.

    Đây là CẦU NỐI vào pipeline chính: align_production gọi
    constrain_to_count(col_boxes, n, valley_split=make_valley_split(page_gray, 'seam'))."""
    def f(box) -> int:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        xs, ys = max(0, x1), max(0, y1)
        sub = gray_image[ys:y2, xs:x2]
        if sub.size == 0:
            return (y1 + y2) // 2
        return ys + _split_cut(sub, method)
    return f


# ===========================================================================
#  RÀNG BUỘC SỐ LƯỢNG N
# ===========================================================================
def _nms_vertical(bx, iou_thr=0.45):
    """Khử trùng lặp theo trục y: bỏ box chồng MẠNH (y-IoU>thr) với box điểm cao hơn.

    Chữ Nôm xếp DỌC nên 2 chữ kề nhau gần như không chồng y (chỉ chạm biên); chồng
    y mạnh = 1 chữ bị detector bắt 2 lần -> giữ box điểm cao hơn, bỏ box kia."""
    keep = []
    for b in sorted(bx, key=lambda b: b[4], reverse=True):
        y1, y2 = b[1], b[3]
        dup = False
        for k in keep:
            inter = max(0.0, min(y2, k[3]) - max(y1, k[1]))
            union = (y2 - y1) + (k[3] - k[1]) - inter
            if union > 0 and inter / union > iou_thr:
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep


def enforce_count(boxes, n, gray_image=None, split_method: str = "seam"):
    """Hoà giải list box (x1,y1,x2,y2[,score]) về ĐÚNG n hộp, trên→dưới.

    M>N: giữ N hộp điểm cao nhất. M<N: tách hộp cao nhất tại điểm cắt theo
    `split_method` ('seam' mặc định). gray_image (toạ độ tuyệt đối) để tìm seam/valley."""
    if n <= 0:
        return []
    bx = [list(b) for b in boxes]
    for b in bx:                                 # đảm bảo có cột score
        if len(b) == 4:
            b.append(1.0)
    if not bx:
        if gray_image is not None:               # gieo 1 hộp bao cả cột -> tách thành n
            gh, gw = gray_image.shape[:2]
            bx = [[0, 0, int(gw), int(gh), 0.0]]
        else:
            return []

    if len(bx) > 1:                              # khử 1-chữ-bắt-2-lần trước khi hoà giải
        bx = _nms_vertical(bx, iou_thr=0.45)

    if len(bx) > n:                              # M > N
        bx = sorted(bx, key=lambda b: b[4], reverse=True)[:n]
    bx.sort(key=lambda b: (b[1] + b[3]) / 2.0)

    guard = 0
    while len(bx) < n and guard < 1000:          # M < N
        guard += 1
        i = max(range(len(bx)), key=lambda j: bx[j][3] - bx[j][1])
        a = bx[i]
        x1, y1, x2, y2 = int(a[0]), int(a[1]), int(a[2]), int(a[3])
        if gray_image is not None and y2 - y1 >= 6:
            xs, ys = max(0, x1), max(0, y1)
            sub = gray_image[ys:y2, xs:x2]
            cut = (ys + _split_cut(sub, split_method)) if sub.size else (y1 + y2) // 2
        else:
            cut = (y1 + y2) // 2
        cut = min(max(cut, y1 + 2), y2 - 2)
        sc = a[4]
        bx[i:i + 1] = [[x1, y1, x2, cut, sc], [x1, cut, x2, y2, sc]]
        bx.sort(key=lambda b: (b[1] + b[3]) / 2.0)
    return bx[:n]


# ===========================================================================
#  DETECTOR
# ===========================================================================
class CenterNetDetector:
    def __init__(self, ckpt: str | None = None, img: int = 512, thr: float = 0.2,
                 split_method: str = "seam", device=None):
        import torch
        self.torch = torch
        self.split_method = split_method
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else ("mps" if torch.backends.mps.is_available() else "cpu"))
        arch, use_dcn = "resnet34_fpn", False
        if ckpt and Path(ckpt).exists():
            d = torch.load(ckpt, map_location="cpu")
            arch = d.get("arch", "resnet34_fpn")
            use_dcn = bool(d.get("use_dcn", False))
            self.img = d.get("img", img)
            self.net = build_model(arch=arch, pretrained=False, use_dcn=use_dcn)
            self.net.load_state_dict(d.get("model", d))
            self.trained = True
            self.val = d.get("val", {})
            self.val_images = d.get("val_images", [])
        else:
            self.img = img
            self.net = build_model(arch=arch, pretrained=False, use_dcn=use_dcn)
            self.trained = False
            self.val = {}
            self.val_images = []
        self.thr = thr
        self.net.eval().to(self.device)

    def _preprocess(self, img_bgr):
        import cv2
        H, W = img_bgr.shape[:2]
        s = self.img / max(H, W)
        nh, nw = max(1, int(H * s)), max(1, int(W * s))
        canvas = np.zeros((self.img, self.img, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh))
        x = self.torch.from_numpy((canvas.astype(np.float32) / 255 - 0.5) / 0.5).permute(2, 0, 1)
        return x.unsqueeze(0), s

    def forward_maps(self, img_bgr):
        """-> (heatmap_np[H/4,W/4], scale s) — heatmap đã letterbox về self.img."""
        x, s = self._preprocess(img_bgr)
        with self.torch.no_grad():
            hm, wh, off = self.net(x.to(self.device))
        return hm[0, 0].float().cpu().numpy(), s

    def boxes_for_image(self, img_bgr, k=1024):
        """Detector chạy 1 lần -> [(x1,y1,x2,y2,score)] px ẢNH GỐC."""
        x, s = self._preprocess(img_bgr)
        with self.torch.no_grad():
            hm, wh, off = self.net(x.to(self.device))
        dets = decode(hm[0], wh[0], off[0], k=k, thr=self.thr)
        return [(d[0] / s, d[1] / s, d[2] / s, d[3] / s, d[4]) for d in dets]

    # ---- ADAPTER PIPELINE (tương thích detector_infer.py) -----------------
    def boxes_for_page(self, page_bgr, k=1024):
        """Bí danh boxes_for_image — tên dùng trong align_production."""
        return self.boxes_for_image(page_bgr, k=k)

    def column_boxes(self, page_boxes, x_range, n, gray_image=None, x_margin=0.5):
        """Box 1 cột (center-x trong x_range±margin·width) -> ĐÚNG n hộp.

        gray_image = ảnh xám TOÀN TRANG (để seam/valley theo toạ độ tuyệt đối)."""
        x1, x2 = x_range
        m = (x2 - x1) * x_margin
        col = [b for b in page_boxes if x1 - m <= (b[0] + b[2]) / 2 <= x2 + m]
        col.sort(key=lambda b: (b[1] + b[3]) / 2)
        out = enforce_count([b[:5] if len(b) >= 5 else (tuple(b[:4]) + (1.0,)) for b in col],
                            n, gray_image=gray_image, split_method=self.split_method)
        # trả 4-int [x1,y1,x2,y2] ĐÚNG hợp đồng detector_infer.py / constrain_to_count
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in out]

    @staticmethod
    def group_columns(boxes, gap_factor=0.6, reading="rtl"):
        """Gom box thành cột dọc (cụm theo center-x). reading='rtl' xếp phải→trái."""
        if not boxes:
            return []
        bs = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2.0)
        wmed = float(np.median([b[2] - b[0] for b in bs])) or 1.0
        cols, cur = [], [bs[0]]
        for b in bs[1:]:
            prev_cx = (cur[-1][0] + cur[-1][2]) / 2.0
            cx = (b[0] + b[2]) / 2.0
            if cx - prev_cx > gap_factor * wmed:
                cols.append(cur); cur = [b]
            else:
                cur.append(b)
        cols.append(cur)
        out = []
        for c in cols:
            cxs = [(b[0] + b[2]) / 2.0 for b in c]
            out.append(((min(cxs), max(cxs)), sorted(c, key=lambda b: (b[1] + b[3]) / 2.0)))
        out.sort(key=lambda t: (-t[0][0] if reading == "rtl" else t[0][0]))
        return out


# ===========================================================================
#  CROP  (+ carving theo seam cong)
# ===========================================================================
def _seam_boundary(gray, x1, x2, ya, yb):
    """Seam (toạ-độ-y-tuyệt-đối/cột) giữa hai chữ trong dải y∈[ya,yb], x∈[x1,x2]."""
    sub = gray[max(0, ya):yb, max(0, x1):x2]
    if sub.size == 0 or sub.shape[0] < 3 or sub.shape[1] < 2:
        return None
    seam, _ = compute_seam(sub, 0, sub.shape[0] - 1)
    return None if seam is None else (seam + max(0, ya))   # -> y tuyệt đối toàn ảnh


def carve_crops(img_bgr, gray, boxes, pad=2, bg=255):
    """Cắt crop với BIÊN CONG theo seam: tại mỗi ranh giới giữa 2 hộp kề, tính seam
    và XOÁ (tô nền) phần mực thuộc chữ hàng xóm vắt qua. Trả về list ảnh crop."""
    n = len(boxes)
    H, W = img_bgr.shape[:2]
    crops = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        cx1, cx2 = max(0, x1 - pad), min(W, x2 + pad)
        cy1, cy2 = max(0, y1 - pad), min(H, y2 + pad)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crop = img_bgr[cy1:cy2, cx1:cx2].copy()
        ch, cw = crop.shape[:2]
        midy = (y1 + y2) // 2
        # seam TRÊN: ranh giới với hộp trước -> xoá mực phía trên seam.
        # Dải seam KẸP vào [cy1, midy] để seam chắc chắn cắt qua crop (nếu không,
        # seam có thể nằm ngoài crop -> không xoá được gì).
        if i > 0:
            prev = boxes[i - 1]
            ya = max(int((prev[1] + prev[3]) / 2), cy1)
            seam = _seam_boundary(gray, cx1, cx1 + cw, ya, midy + 1)
            if seam is not None and len(seam) == cw:
                for j in range(cw):
                    yy = min(max(int(seam[j]) - cy1, 0), ch)
                    crop[:yy, j] = bg
        # seam DƯỚI: ranh giới với hộp sau -> xoá mực phía dưới seam (dải kẹp [midy, cy2])
        if i < n - 1:
            nxt = boxes[i + 1]
            yb = min(int((nxt[1] + nxt[3]) / 2) + 1, cy2)
            seam = _seam_boundary(gray, cx1, cx1 + cw, midy, yb)
            if seam is not None and len(seam) == cw:
                for j in range(cw):
                    yy = min(max(int(seam[j]) - cy1, 0), ch)
                    crop[yy:, j] = bg
        crops.append(crop)
    return crops


def crop_and_save(img_bgr, boxes, out_dir, prefix="char", pad=2, carve=False, gray=None):
    """Cắt & lưu crop theo thứ tự. carve=True -> biên cong theo seam. Trả list path."""
    import cv2
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if carve:
        if gray is None:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        imgs = carve_crops(img_bgr, gray, boxes, pad=pad)
    else:
        H, W = img_bgr.shape[:2]
        imgs = []
        for b in boxes:
            x1, y1 = max(0, int(b[0]) - pad), max(0, int(b[1]) - pad)
            x2, y2 = min(W, int(b[2]) + pad), min(H, int(b[3]) + pad)
            imgs.append(img_bgr[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None)
    paths = []
    for i, im in enumerate(imgs):
        if im is None or im.size == 0:
            continue
        p = out / f"{prefix}_{i:03d}.png"
        cv2.imwrite(str(p), im)
        paths.append(str(p))
    return paths


def segment_column(detector: "CenterNetDetector", img_bgr, n, out_dir=None, carve=False):
    """Cắt 1 ảnh (cột/trang) thành ĐÚNG n crop theo thứ tự trên→dưới."""
    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    boxes = detector.boxes_for_image(img_bgr)
    fixed = enforce_count(boxes, n, gray_image=gray, split_method=detector.split_method)
    paths = crop_and_save(img_bgr, fixed, out_dir, carve=carve, gray=gray) if out_dir else []
    return fixed, boxes, paths


# ===========================================================================
#  SMOKE
# ===========================================================================
def _smoke():
    import cv2
    H, W = 700, 120
    img = np.full((H, W, 3), 245, np.uint8)
    for k in range(7):
        y = 15 + k * 95
        cv2.rectangle(img, (20, y), (100, y + 80), (10, 10, 10), -1)
        cv2.rectangle(img, (35, y + 15), (85, y + 65), (245, 245, 245), -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = CenterNetDetector(ckpt=None, img=256, thr=0.0)
    boxes = det.boxes_for_image(img)
    fixed = enforce_count(boxes, 7, gray_image=gray)
    assert len(fixed) == 7, len(fixed)
    # SEAM: 1 hộp cao -> tách 5 bằng seam carving
    split5 = enforce_count([(20, 0, 100, 700, 0.9)], 5, gray_image=gray, split_method="seam")
    assert len(split5) == 5 and all(split5[i][3] <= split5[i + 1][3] for i in range(4))
    # so seam vs valley vs midpoint đều cho 5 hộp
    for mth in ("seam", "valley", "midpoint"):
        assert len(enforce_count([(20, 0, 100, 700, 0.9)], 5, gray_image=gray, split_method=mth)) == 5
    # compute_seam: seam phải luồn qua kẽ hở (toạ độ hợp lệ trong box)
    seam, cut = compute_seam(gray, int(0.2 * H), int(0.8 * H))
    assert seam is not None and seam.min() >= 0 and seam.max() < H and 0 < cut < H
    # M>N prune
    assert len(enforce_count([(20, i*60, 100, i*60+50, 0.1*i) for i in range(10)], 3)) == 3
    # carving crops -> 5 ảnh
    carved = carve_crops(img, gray, split5)
    assert len(carved) == 5 and all(c.size > 0 for c in carved)
    # adapter pipeline
    page_boxes = boxes
    cols = CenterNetDetector.group_columns(page_boxes)
    cb = det.column_boxes(page_boxes, (0, W), 7, gray_image=gray)
    assert len(cb) == 7
    # valley_split callback (cho count_constrained pipeline chính)
    f = make_valley_split(gray, "seam")
    yc = f((20, 0, 100, 200)); assert 0 <= yc <= 200
    print(f"infer_centernet smoke OK | device {det.device} | raw {len(boxes)} -> enforce(7)={len(fixed)} "
          f"| seam-split5={len(split5)} | carve={len(carved)} | column_boxes={len(cb)} | cols={len(cols)}")
    print("  decode -> peaks -> N-constraint(seam/valley/midpoint) -> carve crops -> adapter pipeline: chạy.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", default=str(HERE / "detector_r34.best.pt"))
    ap.add_argument("--image", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--split", default="seam", choices=["seam", "valley", "midpoint"])
    ap.add_argument("--carve", action="store_true", help="cắt crop biên cong theo seam")
    ap.add_argument("--out", default=str(HERE / "crops_out"))
    a = ap.parse_args()
    if a.smoke:
        _smoke()
    elif a.image and a.n > 0:
        import cv2
        det = CenterNetDetector(ckpt=a.ckpt, img=a.img, thr=a.thr, split_method=a.split)
        img = cv2.imread(a.image, cv2.IMREAD_COLOR)
        fixed, raw, paths = segment_column(det, img, a.n, out_dir=a.out, carve=a.carve)
        print(f"ckpt trained={det.trained} | split={a.split} carve={a.carve} | "
              f"raw {len(raw)} -> {len(fixed)} crops (N={a.n}) -> {a.out}")
    else:
        print("dùng --smoke, hoặc --ckpt --image --n [--split seam|valley|midpoint] [--carve].")
