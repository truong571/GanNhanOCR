"""T4-B — chọn `pad` bằng ĐÁP ÁN THẬT trên ngữ liệu tổng hợp (họ A).

VÌ SAO PHẢI CÓ TỆP NÀY
----------------------
T4-A chứng minh `crop_quality_flag` KHÔNG có thẩm quyền chọn `pad`: nó tăng đơn điệu
tới pad 0,60 và không có cực đại nội, vì cả hai thành phần đều đơn điệu-tốt-lên theo
cấu trúc (`border_ink` = mực chạm mép nên pad lớn thì giảm; `stray_ink` chỉ bắt MẢNH
VỤN láng giềng, không bắt láng giềng LỌT TRỌN). Muốn chọn pad thì phải có ĐÁP ÁN.

Ngữ liệu tổng hợp (`lab/synth.py`) cho hộp đáp án chính xác, nên đo được HAI đại lượng
mà IoU trộn lẫn:

    ĐỘ CHUẨN (precision)  = mực trong crop THUỘC chữ đích / toàn bộ mực trong crop
    ĐỘ ĐỦ    (recall)     = mực chữ đích nằm trong crop / toàn bộ mực chữ đích

Hai hỏng hóc đối nghịch nhau, nên chỉ số F1 có CỰC ĐẠI NỘI theo pad **do cấu trúc**:
  * pad quá nhỏ  -> cắt mất nét  -> recall tụt
  * pad quá lớn  -> ăn láng giềng -> precision tụt
Đây chính là tính "bấu vào trục" mà họ D thiếu.

HIỆU CHUẨN SAI SỐ HỘP ĐẦU VÀO — TỪ DỮ LIỆU THẬT
-----------------------------------------------
Không được lấy hộp đáp án làm đầu vào: pipeline thật nhận hộp OCR, vốn lệch. Đo trên
3.191 hộp/20 trang thật (phần hộp OCR thừa ra so với hộp bao mực, theo tỷ lệ cạnh):

    trái  trung vị +0,0392  tb +0,0645  p90 +0,1757
    phải  trung vị +0,0000  tb +0,0348  p90 +0,1188
    trên  trung vị +0,0000  tb +0,0083  p90 +0,0247
    dưới  trung vị +0,0000  tb +0,0066  p90 +0,0130

BẤT ĐỐI XỨNG MẠNH: hộp SÁT MỰC theo chiều dọc, LỎNG theo chiều ngang. Mô phỏng đúng
bằng số này, chứ không dùng jitter đối xứng cho tiện.

    python -m pipeline.lab.crop_iou --pages 12
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "lab" / "t4b_iou.csv"

# hiệu chuẩn từ 3.191 hộp thật (trung bình phần thừa theo tỷ lệ cạnh)
SLACK = {"left": 0.0645, "right": 0.0348, "top": 0.0083, "bottom": 0.0066}
SLACK_SD = 0.0589                      # lệch chuẩn gộp, dùng làm nhiễu quanh trung bình

# Lưới pad: đẳng hướng (như production) + bất đẳng hướng (ứng viên mới).
PAD_GRID = [0.06, 0.09, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35]
ANISO_GRID = [(0.06, 0.15), (0.06, 0.22), (0.09, 0.18), (0.09, 0.25),
              (0.12, 0.20), (0.12, 0.28), (0.15, 0.30)]


def _detector_box(true_bbox, rng: random.Random):
    """Hộp OCR mô phỏng: nới hộp đáp án theo phần thừa ĐÃ HIỆU CHUẨN + nhiễu."""
    x1, y1, x2, y2 = true_bbox
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)

    def s(mu):
        return max(0.0, rng.gauss(mu, SLACK_SD * 0.5))
    return (int(x1 - w * s(SLACK["left"])), int(y1 - h * s(SLACK["top"])),
            int(x2 + w * s(SLACK["right"])), int(y2 + h * s(SLACK["bottom"])))


def score_crop(page_bw, crop_gray, rect, true_bbox) -> tuple[float, float]:
    """(độ chuẩn, độ đủ) tính trên MỰC, không trên diện tích hộp.

    Dùng ẢNH CROP THẬT (đã carve) chứ không dùng hình chữ nhật: `carve_neighbor_ink`
    xoá mực láng giềng ĐI, nên đo trên hình chữ nhật sẽ phạt oan nó.
    """
    import numpy as np
    if crop_gray is None or rect is None:
        return 0.0, 0.0
    rx1, ry1, rx2, ry2 = rect
    H, W = page_bw.shape
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(W, rx2), min(H, ry2)
    if rx2 <= rx1 or ry2 <= ry1:
        return 0.0, 0.0
    crop_ink = crop_gray < 128
    ch, cw = crop_ink.shape
    crop_ink = crop_ink[:ry2 - ry1, :rx2 - rx1]
    if crop_ink.size == 0:
        return 0.0, 0.0
    # mặt nạ "thuộc chữ đích" = trong hộp đáp án, quy về toạ độ crop
    tx1, ty1, tx2, ty2 = (int(v) for v in true_bbox)
    tgt = np.zeros_like(crop_ink)
    ax1, ay1 = max(rx1, tx1), max(ry1, ty1)
    ax2, ay2 = min(rx2, tx2), min(ry2, ty2)
    if ax2 > ax1 and ay2 > ay1:
        tgt[ay1 - ry1:ay2 - ry1, ax1 - rx1:ax2 - rx1] = True
    inter = int((crop_ink & tgt).sum())
    n_crop = int(crop_ink.sum())
    n_tgt = int(page_bw[max(0, ty1):ty2, max(0, tx1):tx2].sum())
    if n_crop == 0 or n_tgt == 0:
        return 0.0, 0.0
    return inter / n_crop, min(1.0, inter / n_tgt)


def run(n_pages: int = 12, seed: int = 20260824) -> list[dict]:
    import numpy as np
    from core.text.dictionary import dict_dir, load_qn_to_nom
    from pipeline.lab import synth
    from pipeline.lab.crop_grid import cut

    qn = load_qn_to_nom(str(dict_dir() / "QuocNgu_SinoNom.csv"))
    glyphs = synth.glyph_index()
    print(f"[T4-B] kho glyph: {len(glyphs):,} ký tự")
    corpus = synth.make_corpus(n_pages, qn, glyphs, seed=seed)
    print(f"[T4-B] {len(corpus)} trang tổng hợp có đáp án hộp")

    cfgs = [{"pad": p, "aniso": False} for p in PAD_GRID] + \
           [{"pad": p, "aniso": True} for p in ANISO_GRID]
    rows = []
    for cfg in cfgs:
        pad = cfg["pad"]
        pr_sum = rc_sum = 0.0
        n = 0
        for page in corpus:
            img, boxes = page["image"], page["boxes"]
            page_bw = img < 128
            bgr = np.stack([img] * 3, axis=-1)
            rng = random.Random(f"{seed}|{page['page']}")
            by_col: dict[int, list] = {}
            for b in boxes:
                by_col.setdefault(b["col"], []).append(b)
            for col, items in by_col.items():
                items.sort(key=lambda b: b["idx"])
                det = [_detector_box(b["bbox"], rng) for b in items]
                for k, b in enumerate(items):
                    g, rect = cut(bgr, img, det[k], pad, "fixed128", True,
                                  det[k - 1] if k else None,
                                  det[k + 1] if k + 1 < len(det) else None,
                                  return_rect=True)
                    pr, rc = score_crop(page_bw, g, rect, b["bbox"])
                    pr_sum += pr; rc_sum += rc; n += 1
        d = max(n, 1)
        pr, rc = pr_sum / d, rc_sum / d
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        rows.append({"pad": str(pad), "aniso": cfg["aniso"], "n": n,
                     "precision": round(pr, 5), "recall": round(rc, 5),
                     "f1": round(f1, 5)})
        print(f"  pad={str(pad):<14} {'bất đẳng hướng' if cfg['aniso'] else 'đẳng hướng   '} "
              f"-> chuẩn={pr:.4f} đủ={rc:.4f} F1={f1:.4f}", flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.lab.crop_iou")
    ap.add_argument("--pages", type=int, default=12)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    rows = run(args.pages)
    iso = [r for r in rows if not r["aniso"]]
    best_iso = max(iso, key=lambda r: r["f1"])
    best_all = max(rows, key=lambda r: r["f1"])
    base = next((r for r in iso if r["pad"] == "0.12"), None)
    print("\n=== KẾT LUẬN T4-B ===")
    if base:
        print(f"  MỐC (pad 0,12)        F1 = {base['f1']:.4f}")
    print(f"  tốt nhất đẳng hướng    pad {best_iso['pad']}  F1 = {best_iso['f1']:.4f}")
    print(f"  tốt nhất toàn lưới     pad {best_all['pad']}"
          f"{' (bất đẳng hướng)' if best_all['aniso'] else ''}  F1 = {best_all['f1']:.4f}")
    interior = iso[0]["f1"] < best_iso["f1"] > iso[-1]["f1"]
    print(f"  có CỰC ĐẠI NỘI theo pad: {interior}"
          f"  ({'thước đo bấu vào trục -> có thẩm quyền chọn pad' if interior else 'KHÔNG -> vẫn không quyết được'})")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (REPO / "lab" / "t4b_decision.json").write_text(json.dumps(
        {"baseline": base, "best_isotropic": best_iso, "best_overall": best_all,
         "interior_optimum": interior}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[T4-B] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
