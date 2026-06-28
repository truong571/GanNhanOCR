"""Sinh PDF kết quả: minh hoạ toàn bộ pipeline CenterNet anchorless trên dữ liệu thật.

Trang PDF gồm:
  1. Bìa + tóm tắt phương pháp/kiến trúc/loss + chỉ số VAL (F1/P/R/count-err).
  2. Mỗi trang demo (held-out): ảnh gốc + box GROUND-TRUTH | + box DỰ ĐOÁN | HEATMAP.
  3. Mỗi cột demo: detect thô (M hộp) -> sau RÀNG BUỘC N (đúng N hộp) + lưới crop.
  4. Minh hoạ cơ chế tách "projection valley" khi M < N.

USAGE
  .venv/bin/python test/make_report_pdf.py --ckpt test/detector_r34.best.pt \
      --manifest <detect_manifest.json> --out test/ket_qua_centernet.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from infer_centernet import (CenterNetDetector, enforce_count, projection_valley,  # noqa: E402
                             compute_seam, carve_crops)

plt.rcParams["font.family"] = "DejaVu Sans"   # hỗ trợ dấu tiếng Việt


# --------------------------------------------------------------------------- helpers
def _bgr2rgb(img):
    return img[:, :, ::-1]


def _downscale(img, max_side=1100):
    import cv2
    H, W = img.shape[:2]
    if max(H, W) <= max_side:
        return img, 1.0
    s = max_side / max(H, W)
    return cv2.resize(img, (int(W * s), int(H * s))), s


def _draw_boxes(ax, boxes, scale=1.0, color="red", lw=1.0, alpha=1.0):
    for b in boxes:
        x1, y1, x2, y2 = b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor=color, linewidth=lw, alpha=alpha))


def _heatmap_on_page(det, page_bgr):
    """Heatmap dự đoán, upsample về kích thước trang gốc (cắt phần letterbox-pad)."""
    import cv2
    hm, s = det.forward_maps(page_bgr)            # hm: (img/4, img/4)
    H, W = page_bgr.shape[:2]
    nh, nw = int(H * s), int(W * s)
    valid = hm[0:max(1, nh // 4), 0:max(1, nw // 4)]   # bỏ phần letterbox-pad
    return cv2.resize(valid, (W, H))


# --------------------------------------------------------------------------- pages
def cover_page(pdf, det, ckpt_path, n_pages, n_cols):
    fig = plt.figure(figsize=(8.27, 11.69))    # A4 dọc
    fig.text(0.5, 0.95, "Định vị & tách chữ Hán–Nôm dính nhau",
             ha="center", fontsize=18, weight="bold")
    fig.text(0.5, 0.915, "CenterNet anchorless (ResNet34 + FPN) + ràng buộc số ký tự N",
             ha="center", fontsize=12)
    v = det.val or {}
    trained = "ĐÃ HUẤN LUYỆN" if det.trained else "TRỌNG SỐ NGẪU NHIÊN (chưa có ckpt)"
    body = (
        f"Checkpoint: {Path(ckpt_path).name}  ({trained})\n"
        f"Thiết bị suy luận: {det.device}   |   Ảnh đầu vào: {det.img}px   |   Output stride: 4\n\n"
        "━━━━━━━━━━━━━━━━━━━━  KIẾN TRÚC  ━━━━━━━━━━━━━━━━━━━━\n"
        "  Ảnh (3,H,W)\n"
        "    └ ResNet34 (ImageNet)  →  C2(/4) C3(/8) C4(/16) C5(/32)\n"
        "        └ FPN top-down (lateral 1×1 + cộng dồn + upsample)  →  P2 ở /4\n"
        "            ├ Head Heatmap (1ch, sigmoid, bias −2.19)  — xác suất tâm chữ\n"
        "            ├ Head Size    (2ch)  — [w, h]\n"
        "            └ Head Offset  (2ch)  — bù sai số lượng tử hoá\n\n"
        "━━━━━━━━━━━━━━━━━━━━  VÌ SAO TÁCH ĐƯỢC CHỮ DÍNH  ━━━━━━━━━━━━━━━━━━━━\n"
        "  • Mỗi ký tự = MỘT ĐIỂM (tâm). Hai tâm kề nhau luôn tách biệt về toạ độ,\n"
        "    bất kể nét mực dính tới đâu → không cần anchor/NMS-IoU.\n"
        "  • Tách hộp = Max-Pool 3×3 tìm cực trị địa phương trên heatmap.\n\n"
        "━━━━━━━━━━━━━━━━━━━━  HÀM MẤT MÁT  ━━━━━━━━━━━━━━━━━━━━\n"
        "  L = FocalLoss(heatmap) + 0.1·L1(size) + 1.0·L1(offset)\n\n"
        "━━━━━━━━━━━━━━━━━━━━  RÀNG BUỘC N (= số âm tiết Quốc ngữ)  ━━━━━━━━━━━━━━━━━━━━\n"
        "  M = N : nhận.\n"
        "  M > N : giữ N tâm điểm-tin-cậy cao nhất (bỏ nhiễu cắt thừa).\n"
        "  M < N : tách hộp cao vượt trội tại 'projection valley' (dòng ít mực nhất).\n\n"
        "━━━━━━━━━━━━━━━━━━━━  CHỈ SỐ VALIDATION  ━━━━━━━━━━━━━━━━━━━━\n"
        f"  Box-level @IoU 0.5:  F1 = {v.get('F1','—')}   P = {v.get('P','—')}   "
        f"R = {v.get('R','—')}\n"
        f"  Sai số đếm trung vị / trang (median count-err) = {v.get('median_count_err','—')}   "
        f"trên {v.get('pages','—')} trang\n"
    )
    fig.text(0.07, 0.86, body, ha="left", va="top", fontsize=9.2, family="monospace")
    fig.text(0.5, 0.04, f"Demo: {n_pages} trang held-out · {n_cols} cột minh hoạ ràng buộc N",
             ha="center", fontsize=9, style="italic", color="#555")
    pdf.savefig(fig); plt.close(fig)


def page_detection(pdf, det, item):
    import cv2
    page = cv2.imread(item["image"], cv2.IMREAD_COLOR)
    if page is None:
        return None
    gt = item["boxes"]
    pred = det.boxes_for_image(page)
    hm = _heatmap_on_page(det, page)
    disp, s = _downscale(page)

    fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    name = Path(item["image"]).parent.parent.name + "/" + Path(item["image"]).name
    fig.suptitle(f"Trang held-out: {name}   |   GT {len(gt)} ký tự  ·  dự đoán {len(pred)} box",
                 fontsize=11)
    for ax in axes:
        ax.imshow(_bgr2rgb(disp)); ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title(f"Ground-truth ({len(gt)})", fontsize=9)
    _draw_boxes(axes[0], gt, scale=s, color="#00cc44", lw=0.6)
    axes[1].set_title(f"Dự đoán CenterNet ({len(pred)})", fontsize=9)
    _draw_boxes(axes[1], pred, scale=s, color="#ff2222", lw=0.6)
    axes[2].set_title("Heatmap tâm chữ", fontsize=9)
    hm_disp = cv2.resize(hm, (disp.shape[1], disp.shape[0]))
    axes[2].imshow(hm_disp, cmap="jet", alpha=0.65)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig); plt.close(fig)
    return {"page": page, "gt": gt, "pred": pred}


def page_columns(pdf, det, page, gt, pred, max_cols=3):
    """Minh hoạ ràng buộc N trên vài cột: detect thô -> đúng N -> crop."""
    import cv2
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    # cột & N "thật" suy từ GT (cụm GT theo center-x)
    gt_cols = CenterNetDetector.group_columns([tuple(b) + (1.0,) for b in gt], reading="rtl")
    gt_cols = [c for c in gt_cols if len(c[1]) >= 4][:max_cols]
    for ci, ((cx1, cx2), gboxes) in enumerate(gt_cols):
        N = len(gboxes)
        ys = [b[1] for b in gboxes] + [b[3] for b in gboxes]
        xs = [b[0] for b in gboxes] + [b[2] for b in gboxes]
        x1, x2 = int(min(xs)) - 8, int(max(xs)) + 8
        y1, y2 = int(min(ys)) - 8, int(max(ys)) + 8
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(page.shape[1], x2), min(page.shape[0], y2)
        col_img = page[y1:y2, x1:x2]
        if col_img.size == 0:
            continue
        # box dự đoán rơi vào cột này
        m = (cx2 - cx1) * 0.6 + 1
        raw = [b for b in pred if cx1 - m <= (b[0] + b[2]) / 2 <= cx2 + m]
        raw_local = [(b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1, b[4]) for b in raw]
        col_gray = gray[y1:y2, x1:x2]
        fixed = enforce_count(raw_local, N, gray_image=col_gray, split_method="seam")

        # crop biên cong theo SEAM CARVING (xoá mực chữ hàng xóm vắt qua)
        crops = [c for c in carve_crops(col_img, col_gray, fixed) if c is not None and c.size]

        n_crop = len(crops)
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = fig.add_gridspec(1, 3 + min(n_crop, 12), width_ratios=[2, 2, 2] + [1] * min(n_crop, 12))
        title = (f"Cột {ci+1}: N (GT) = {N}  ·  detect thô M = {len(raw_local)}  "
                 f"→ sau ràng buộc = {len(fixed)}")
        fig.suptitle(title, fontsize=11)

        ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(_bgr2rgb(col_img))
        ax0.set_title(f"Ảnh cột", fontsize=8); ax0.set_xticks([]); ax0.set_yticks([])
        ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(_bgr2rgb(col_img))
        _draw_boxes(ax1, raw_local, color="#ff2222", lw=0.8)
        ax1.set_title(f"Detect thô (M={len(raw_local)})", fontsize=8)
        ax1.set_xticks([]); ax1.set_yticks([])
        ax2 = fig.add_subplot(gs[0, 2]); ax2.imshow(_bgr2rgb(col_img))
        _draw_boxes(ax2, fixed, color="#1166ff", lw=1.0)
        for i, b in enumerate(fixed):
            ax2.text(b[0], (b[1] + b[3]) / 2, str(i + 1), color="yellow",
                     fontsize=6, va="center")
        ax2.set_title(f"Ràng buộc N={N}", fontsize=8); ax2.set_xticks([]); ax2.set_yticks([])
        for j, cr in enumerate(crops[:12]):
            axc = fig.add_subplot(gs[0, 3 + j]); axc.imshow(_bgr2rgb(cr))
            axc.set_title(str(j + 1), fontsize=7); axc.set_xticks([]); axc.set_yticks([])
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig); plt.close(fig)


def valley_demo_page(pdf):
    """Bước 6 — minh hoạ SEAM CARVING khi M<N: đường cắt CONG theo kẽ hở vs cắt THẲNG."""
    import cv2
    H, W = 600, 130
    img = np.full((H, W), 250, np.uint8)
    # 5 chữ giả DÍNH nhau, ranh giới LỆCH (để seam cong khác hẳn cắt thẳng)
    centers = [70, 195, 320, 445, 545]
    for k, cy in enumerate(centers):
        cv2.circle(img, (65, cy), 36, 30, -1)
        cv2.circle(img, (65, cy), 19, 250, 2)
    for k in range(len(centers) - 1):                     # cầu nối mực lệch giữa 2 chữ
        midy = (centers[k] + centers[k + 1]) // 2
        cv2.line(img, (40 + 18 * (k % 2), midy - 6), (90 - 18 * (k % 2), midy + 6), 40, 5)
    one_tall = [(10, 25, 120, 585, 0.9)]
    fixed = enforce_count(one_tall, 5, gray_image=img, split_method="seam")
    valley_fixed = enforce_count(one_tall, 5, gray_image=img, split_method="valley")

    fig, axes = plt.subplots(1, 3, figsize=(11.69, 6.5))
    fig.suptitle("Bước 6 — Seam Carving: tách chữ dính bằng ĐƯỜNG ĐI NĂNG LƯỢNG TỐI THIỂU",
                 fontsize=11)
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Cột dính (M=1 hộp)", fontsize=9)
    axes[0].add_patch(Rectangle((10, 25), 110, 560, fill=False, edgecolor="red", lw=1.5))
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # giữa: cắt THẲNG (valley) vs seam median
    axes[1].imshow(img, cmap="gray")
    for b in valley_fixed[:-1]:
        axes[1].axhline(b[3], color="green", ls="--", lw=1.2)
    axes[1].set_title("Cắt THẲNG (projection valley)\n— xanh lá = đường ngang", fontsize=9)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # phải: SEAM cong theo kẽ hở + hộp kết quả
    axes[2].imshow(img, cmap="gray")
    xs = np.arange(0, W)
    for k in range(len(fixed) - 1):
        a, c = fixed[k], fixed[k + 1]
        ya, yb = int((a[1] + a[3]) / 2), int((c[1] + c[3]) / 2)
        seam, _ = compute_seam(img[ya:yb, :], 0, yb - ya - 1)
        if seam is not None:
            axes[2].plot(xs, seam + ya, color="orange", lw=1.6)
    for i, b in enumerate(fixed):
        axes[2].text(b[0] + 4, (b[1] + b[3]) / 2, str(i + 1), color="#1166ff", fontsize=9)
    axes[2].set_title("Seam CONG (cam) — luồn theo kẽ hở,\nné nét mực dính", fontsize=9)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig); plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "detector_r34.best.pt"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=str(HERE / "ket_qua_centernet.pdf"))
    ap.add_argument("--pages", type=int, default=3, help="số trang demo")
    ap.add_argument("--cols", type=int, default=3, help="số cột minh hoạ N / trang")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--thr", type=float, default=0.3)
    a = ap.parse_args()

    man = json.load(open(a.manifest, encoding="utf-8"))
    det = CenterNetDetector(ckpt=a.ckpt, img=a.img, thr=a.thr)
    # Ưu tiên danh sách trang val LƯU TRONG CKPT (đảm bảo "held-out" đúng với lúc train,
    # bất kể --max-items/--val-frac); chỉ suy lại khi ckpt cũ không có.
    if det.val_images:
        vset = set(det.val_images)
        val_items = [it for it in man if it["image"] in vset]
        split_src = "ckpt.val_images"
    else:
        nval = max(1, int(len(man) * a.val_frac))
        val_items = man[:nval]
        split_src = "fallback man[:nval]"
    print(f"[pdf] ckpt trained={det.trained} device={det.device} | "
          f"val pages {len(val_items)} ({split_src})")

    with PdfPages(a.out) as pdf:
        cover_page(pdf, det, a.ckpt, a.pages, a.cols)
        # chọn các trang có nhiều ký tự để demo đẹp
        cand = sorted(val_items, key=lambda it: -len(it["boxes"]))[:a.pages]
        for k, it in enumerate(cand):
            print(f"  [pdf] trang {k+1}/{len(cand)}: {Path(it['image']).name}", flush=True)
            res = page_detection(pdf, det, it)
            if res:
                page_columns(pdf, det, res["page"], res["gt"], res["pred"], max_cols=a.cols)
        valley_demo_page(pdf)
    print(f"[pdf] đã ghi: {a.out}")


if __name__ == "__main__":
    main()
