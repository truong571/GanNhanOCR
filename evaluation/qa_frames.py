"""QA khung từng trang — kiểm mỗi trang có "chuẩn" như page_0012 không.

Khung CHUẨN (đối chiếu page_0012) phải:
  • Bao trọn 9 cột chữ (kể cả tiêu đề ở ĐẦU CỘT 1, vd "二月二十八日" -> GIỮ).
  • LOẠI dãy số cột 1-9 ở trên.
  • LOẠI số trang ở dưới.
  • KHÔNG cắt mất cột / dòng chữ.
  • KHÔNG là fallback nguyên ảnh (area_ratio ~0.58-0.66; page_0012 = 0.656).

Cách kiểm (tín hiệu ĐÁNG TIN, không dùng tỉ lệ mực thô vì dễ hiểu ngược —
"mực ngoài khung nhiều" = đã loại số/viền = TỐT, không phải lỗi):
  • kẹp số 1-9 / số trang: dò DẢI MỎNG CÔ LẬP (tách bởi khe trắng) ngay bên
    TRONG đỉnh/đáy khung. Header ở đầu cột 1 dính liền thân chữ -> KHÔNG bị nhầm.
  • fallback: area_ratio quá lớn.
  • cắt cột: mực cột-dạng (cao) ngay sát ngoài mép trái/phải.

Chấm PASS / WARN / FAIL kèm lý do. Xuất:
  <out>/<book>_qa.csv        — chỉ số + trạng thái từng trang
  <out>/<book>_flagged.png   — contact sheet CHỈ trang WARN/FAIL (review nhanh)

Usage:
    python3 evaluation/qa_frames.py
    python3 evaluation/qa_frames.py --book SachThanhTruyen4
    python3 evaluation/qa_frames.py --book SachThanhTruyen2 --pages page_0012,page_0026
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.image.frame_detector import detect_frame_hybrid, _remove_scan_border  # noqa: E402

PAD = 12
AR_FALLBACK = 0.88     # > -> khung nguyên ảnh (fallback)
MIN_DIM = 0.20         # khung < 20% bề rộng/cao ảnh -> quá nhỏ / hỏng
SIDE_CLIP = 0.025      # mực trái/phải ngoài khung > -> nghi cắt cột


def _otsu(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _isolated_band(sub_bw, reverse=False):
    """Còn KẸP số 1-9 (đỉnh) / số trang (đáy) trong khung không?

    Dùng CONNECTED COMPONENTS (đáng tin cả khi mật độ tăng đều, không có 'hố'
    — vd page_0144): số/mũi tên = đốm NHỎ; chữ Nôm = đốm LỚN. Kẹp số khi có
    ≥4 đốm nhỏ NẰM TRÊN chữ-lớn cao nhất, và chữ-lớn đó cách mép > 3% chiều cao.
    """
    region = sub_bw[::-1] if reverse else sub_bw
    h, w = region.shape
    if h < 120:
        return False
    n, lab, st, _ = cv2.connectedComponentsWithStats(region, 8)
    heights = [st[i][3] for i in range(1, n)
               if st[i][4] > 0.00008 * w * h and st[i][3] > 5]
    if len(heights) < 5:
        return False
    char_h = float(np.median(heights))
    h_min = 0.5 * char_h
    large_tops = [st[i][1] for i in range(1, n)
                  if st[i][3] >= h_min and st[i][4] >= 0.00015 * w * h]
    if not large_tops:
        return False
    first_char = min(large_tops)
    small_above = sum(1 for i in range(1, n)
                      if st[i][1] + st[i][3] <= first_char
                      and st[i][3] < h_min and st[i][4] > 10)
    return first_char > 0.03 * h and small_above >= 4


def _flipped_signal(bgr, fr) -> bool:
    """Nghi trang LẬT NGƯỢC 180°: số 1-9 (trải rộng cả trang) nằm ở ĐÁY thay vì
    đỉnh, số trang (hẹp 1 bên) nằm ở ĐỈNH. Đo spread đốm nhỏ trong dải lề trắng
    trên vs dưới khung (cách mép 20px để né nét chữ thò, bỏ đường kẻ ngang).
    Tự tính binary sạch (không mask viền giấy) để không che mất số trang ở mép.
    Đây là cờ NGHI NGỜ (có thể sai khi số 1-9 mờ) -> để người kiểm & xoay TAY,
    KHÔNG tự xoay."""
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = fr
    fw = x1 - x0
    if fw < 50 or (y1 - y0) < 100:
        return False
    bw = _otsu(_remove_scan_border(bgr))
    off, band = 20, int(0.08 * (y1 - y0))

    def spread(strip):
        if strip.shape[0] < 6:
            return 0.0
        n, lab, st, _ = cv2.connectedComponentsWithStats(strip, 8)
        cx = [st[i][0] + st[i][2] / 2 for i in range(1, n)
              if st[i][2] <= 0.4 * fw and st[i][4] >= 8]
        if len(cx) < 2:
            return 0.0
        cx.sort()
        return (cx[-1] - cx[0]) / fw

    top = bw[max(0, y0 - off - band):max(0, y0 - off), x0:x1]
    bot = bw[min(H, y1 + off):min(H, y1 + off + band), x0:x1]
    return spread(bot) > 0.5 and spread(top) < 0.35


def analyze_page(path: Path) -> dict:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return {"page": path.stem, "status": "ERR", "reasons": "không đọc được ảnh"}
    H, W = bgr.shape[:2]
    fr = detect_frame_hybrid(bgr)
    x0 = max(0, fr[0] - PAD); y0 = max(0, fr[1] - PAD)
    x1 = min(W, fr[2] + PAD); y1 = min(H, fr[3] + PAD)
    ar = (x1 - x0) * (y1 - y0) / (W * H)

    # Đo trên ảnh đã TẨY VỆT ĐEN SCAN (nhất quán với detector) -> vệt đen ngoài
    # khung không bị tính nhầm là "cắt cột".
    bw = _otsu(_remove_scan_border(bgr))
    m = max(int(0.02 * min(H, W)), 8)
    bw[:m, :] = 0; bw[-m:, :] = 0; bw[:, :m] = 0; bw[:, -m:] = 0
    tot = int((bw > 0).sum()) or 1
    # Chỉ đo dải SÁT NGAY ngoài mép khung (rộng ~1 cột) — nơi cột bị cắt sẽ nằm.
    # Tránh tính nhầm vệt đen/nhiễu ở xa tận biên ảnh là "cắt cột".
    sw = max(20, int(0.12 * (x1 - x0)))
    left = (bw[y0:y1, max(0, x0 - sw):x0] > 0).sum() / tot
    right = (bw[y0:y1, x1:x1 + sw] > 0).sum() / tot

    sub = bw[y0:y1, x0:x1]
    marker_in = _isolated_band(sub, reverse=False)
    pagenum_in = _isolated_band(sub, reverse=True)
    flipped = _flipped_signal(bgr, fr)

    reasons = []
    status = "PASS"

    def fail(msg):
        nonlocal status
        status = "FAIL"; reasons.append(msg)

    def warn(msg):
        nonlocal status
        if status != "FAIL":
            status = "WARN"
        reasons.append(msg)

    fw, fh = x1 - x0, y1 - y0
    if ar > AR_FALLBACK:
        fail(f"fallback / khung nguyên ảnh (ar={ar:.2f})")
    elif fw < MIN_DIM * W or fh < MIN_DIM * H:
        fail(f"khung quá nhỏ ({fw}x{fh})")
    if marker_in:
        fail("KẸP số 1-9 trong khung")
    if pagenum_in:
        fail("KẸP số trang trong khung")
    if max(left, right) > SIDE_CLIP:
        fail(f"nghi CẮT CỘT (L={left:.1%} R={right:.1%})")
    if flipped:
        warn("NGHI LẬT NGƯỢC 180° (kiểm & xoay tay)")

    return {
        "page": path.stem, "W": W, "H": H, "bbox": (x0, y0, x1, y1),
        "ar": round(ar, 3),
        "marker_in": int(marker_in), "pagenum_in": int(pagenum_in),
        "flipped": int(flipped),
        "left": round(left, 4), "right": round(right, 4),
        "status": status, "reasons": "; ".join(reasons) or "-",
    }


def _flagged_contact(rows, pages_dir, out_path):
    flagged = [r for r in rows if r["status"] in ("FAIL", "WARN", "ERR")]
    if not flagged:
        return 0
    TH_H, COLS = 320, 6
    tiles = []
    for r in flagged:
        bgr = cv2.imread(str(pages_dir / f"{r['page']}.png"))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]; sc = TH_H / H
        th = cv2.resize(bgr, (int(W * sc), TH_H))
        if "bbox" in r:
            x0, y0, x1, y1 = [int(v * sc) for v in r["bbox"]]
            color = (0, 0, 255) if r["status"] == "FAIL" else (0, 165, 255)
            cv2.rectangle(th, (x0, y0), (x1, y1), color, 2)
        col = (0, 0, 255) if r["status"] == "FAIL" else (0, 140, 220)
        cv2.putText(th, f"{r['page'].replace('page_','')} {r['status']}",
                    (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(th, r["reasons"][:36], (3, TH_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 255), 1)
        tiles.append(th)
    if not tiles:
        return 0
    wmax = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 2, 2, 2, wmax - t.shape[1] + 2,
                                cv2.BORDER_CONSTANT, value=(190, 190, 190)) for t in tiles]
    while len(tiles) % COLS:
        tiles.append(np.full_like(tiles[0], 240))
    grid = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
    cv2.imwrite(str(out_path), grid)
    return len(flagged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="all")
    ap.add_argument("--pages", help="comma-separated page stems")
    ap.add_argument("--out", default="evaluation/_kinhhannom_debug/_qa")
    args = ap.parse_args()

    books = (["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]
             if args.book == "all" else [args.book])
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    grand = {"PASS": 0, "WARN": 0, "FAIL": 0, "ERR": 0}
    for book in books:
        pages_dir = ROOT / "prepared" / book / "pages"
        pages = sorted(pages_dir.glob("*.png"))
        if args.pages:
            want = set(args.pages.split(","))
            pages = [p for p in pages if p.stem in want]
        if not pages:
            print(f"[skip] {book}: không có trang")
            continue

        rows = [analyze_page(p) for p in pages]
        cnt = {"PASS": 0, "WARN": 0, "FAIL": 0, "ERR": 0}
        for r in rows:
            cnt[r["status"]] = cnt.get(r["status"], 0) + 1
            grand[r["status"]] = grand.get(r["status"], 0) + 1

        fields = ["page", "status", "reasons", "ar",
                  "marker_in", "pagenum_in", "flipped", "left", "right", "W", "H"]
        with (out_dir / f"{book}_qa.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

        nflag = _flagged_contact(rows, pages_dir, out_dir / f"{book}_flagged.png")

        print(f"\n===== {book}: {len(rows)} trang | "
              f"PASS {cnt['PASS']}  WARN {cnt['WARN']}  FAIL {cnt['FAIL']}"
              f"{'  ERR ' + str(cnt['ERR']) if cnt['ERR'] else ''} =====")
        for r in rows:
            if r["status"] in ("FAIL", "WARN", "ERR"):
                print(f"  [{r['status']:4}] {r['page']}: {r['reasons']}")
        if cnt["FAIL"] == 0 and cnt["WARN"] == 0:
            print("  ✓ tất cả PASS")
        print(f"  → CSV: {out_dir / f'{book}_qa.csv'}")
        if nflag:
            print(f"  → Ảnh trang lỗi: {out_dir / f'{book}_flagged.png'}")

    tot = sum(grand.values())
    print(f"\n##### TỔNG: {tot} trang | PASS {grand['PASS']} "
          f"({100*grand['PASS']/max(1,tot):.0f}%)  WARN {grand['WARN']}  FAIL {grand['FAIL']}"
          f"{'  ERR ' + str(grand['ERR']) if grand['ERR'] else ''} #####")


if __name__ == "__main__":
    main()
