"""TEST các luật sửa số cột (CHƯA nhập code chính).

Thêm 2 luật so với boxes_to_columns hiện tại:
  (A) BỎ CỘT RỖNG: cột không có ký tự nào ([]) -> xoá (ô trống không thành cột).
  (B) BỎ "CỘT MA" GIỮA: cột chen giữa 2 cột chính, CẢ HAI khe x < 0.6 nhịp
      (chú thích nhỏ giữa cột) -> xoá khỏi 9 cột chính.

Chạy: python3 evaluation/test_col_fix.py            # test toàn bộ 445 trang (từ cache)
      python3 evaluation/test_col_fix.py --show 0120 # in chi tiết 1 trang
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------- gom cột thô (giống boxes_to_columns) ----------
def group_raw(boxes):
    sb = sorted(boxes, key=lambda b: b["points"][0][0], reverse=True)
    cols = []
    for b in sb:
        if cols and abs(cols[-1][-1]["points"][0][0] - b["points"][0][0]) < 15:
            cols[-1].append(b)
        else:
            cols.append([b])
    # mỗi box -> các ký tự; bỏ box rỗng
    result = []
    for col in cols:
        chars = []
        for box in sorted(col, key=lambda b: b["points"][0][1]):
            text = (box.get("transcription") or "").strip()
            vc = [c for c in text if c.strip()]
            if not vc:
                continue
            yt, yb = box["points"][0][1], box["points"][2][1]
            xl, xr = box["points"][0][0], box["points"][1][0]
            ch_h = (yb - yt) / len(vc)
            for k, c in enumerate(vc):
                chars.append({"char": c, "y_center": yt + ch_h * (k + 0.5),
                              "bbox": [xl, int(yt + ch_h * k), xr, int(yt + ch_h * (k + 1))]})
        result.append(chars)
    return result


def cx(c):
    return statistics.mean([b["bbox"][0] for b in c]) if c else 0


def col_h(c):
    ys = [b["bbox"][1] for b in c] + [b["bbox"][3] for b in c]
    return (max(ys) - min(ys)) if ys else 0


def col_xr(c):
    return (min(b["bbox"][0] for b in c), max(b["bbox"][2] for b in c)) if c else (0, 0)


def fix_columns(result, verbose=False):
    # (A) BỎ CỘT RỖNG ngay từ đầu
    cols = [list(c) for c in result if len(c) > 0]
    if len(cols) <= 1:
        return cols, []

    dropped = []
    sizes = [len(c) for c in cols]
    med_size = statistics.median(sizes)
    centers = [cx(c) for c in cols]
    spacings = sorted(abs(centers[i] - centers[i + 1]) for i in range(len(centers) - 1))
    med_space = statistics.median(spacings) if spacings else 100
    frag_max = max(2, int(0.25 * med_size))

    changed = True
    while changed and len(cols) > 1:
        changed = False
        # (1) mảnh vụn 1-2 ký tự -> gộp hàng xóm gần
        for i, c in enumerate(cols):
            if 0 < len(c) <= frag_max:
                nb = [k for k in (i - 1, i + 1) if 0 <= k < len(cols)]
                j = min(nb, key=lambda k: abs(cx(cols[k]) - cx(c)))
                if abs(cx(cols[j]) - cx(c)) < 0.6 * med_space:
                    cols[j] = sorted(cols[j] + c, key=lambda b: b["y_center"]); cols.pop(i)
                    changed = True; break
        if changed:
            continue
        # (2) cặp sát: gap < 0.35 nhịp, tổng <= 1.3 median -> tách-đôi-giữa-thân
        for i in range(len(cols) - 1):
            if (abs(cx(cols[i]) - cx(cols[i + 1])) < 0.35 * med_space
                    and len(cols[i]) + len(cols[i + 1]) <= 1.3 * med_size):
                cols[i] = sorted(cols[i] + cols[i + 1], key=lambda b: b["y_center"]); cols.pop(i + 1)
                changed = True; break
        if changed:
            continue
        # (3) box NGẮN + CHỒNG x -> gộp hàng xóm
        med_h = statistics.median([col_h(c) for c in cols])
        for i, c in enumerate(cols):
            if col_h(c) < 0.80 * med_h:
                xl, xr = col_xr(c); ov = None
                for k in (i - 1, i + 1):
                    if 0 <= k < len(cols):
                        xl2, xr2 = col_xr(cols[k])
                        if (min(xr, xr2) - max(xl, xl2)) > 0.25 * (xr - xl):
                            ov = k
                if ov is not None:
                    cols[ov] = sorted(cols[ov] + c, key=lambda b: b["y_center"]); cols.pop(i)
                    changed = True; break
        if changed:
            continue
        # (4) NEW — "CỘT MA" GIỮA: chen giữa 2 cột, CẢ HAI khe < 0.6 nhịp -> XOÁ
        for i in range(1, len(cols) - 1):
            gl = abs(cx(cols[i]) - cx(cols[i - 1]))
            gr = abs(cx(cols[i]) - cx(cols[i + 1]))
            if gl < 0.7 * med_space and gr < 0.7 * med_space:
                txt = "".join(b["char"] for b in cols[i])
                dropped.append((round(cx(cols[i])), len(cols[i]), txt[:18]))
                cols.pop(i)
                changed = True; break
    return cols, dropped


def run_all():
    tot = ok = 0
    bad = []
    for book in ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]:
        for cf in sorted((ROOT / "prepared" / book / "detected").glob("*_ocr_cache.json")):
            d = json.load(open(cf))
            if d.get("boxes_raw") is None:
                continue
            cols, _ = fix_columns(group_raw(d["boxes_raw"]))
            n = len(cols); tot += 1
            if n == 9:
                ok += 1
            else:
                bad.append((book, cf.stem.replace("_ocr_cache", ""), n))
    print(f"TEST luật mới (bỏ cột rỗng + bỏ cột-ma giữa): {tot} trang | "
          f"OK(=9) {ok} ({100*ok/tot:.0f}%) | ≠9: {len(bad)}")
    for b in bad:
        print(f"   {b[0]}/{b[1]}: {b[2]} cột")


def show(stem_frag):
    for book in ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]:
        for cf in (ROOT / "prepared" / book / "detected").glob(f"*{stem_frag}*_ocr_cache.json"):
            d = json.load(open(cf))
            raw = group_raw(d["boxes_raw"])
            cols, dropped = fix_columns(raw)
            print(f"\n{book}/{cf.stem.replace('_ocr_cache','')}: "
                  f"thô {len(raw)} cột -> sau fix {len(cols)} cột")
            if dropped:
                for x, n, t in dropped:
                    print(f"   ĐÃ BỎ cột-ma x={x} ({n} ký tự): {t}")
            for i, c in enumerate(cols):
                print(f"   cột{i+1} x={cx(c):.0f} n={len(c)} | {''.join(b['char'] for b in c)[:30]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="in chi tiết 1 trang (vd 0120)")
    args = ap.parse_args()
    if args.show:
        show(args.show)
    else:
        run_all()
