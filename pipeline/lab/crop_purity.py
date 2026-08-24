"""T4-B' — thước đo chất lượng crop KHÔNG CẦN NHÃN mà thật sự BẤU VÀO trục `pad`.

VÌ SAO CẦN THƯỚC ĐO MỚI
-----------------------
T4-A chứng minh `crop_quality_flag` không có thẩm quyền chọn `pad`: nó tăng đơn điệu
tới pad 0,60. Nguyên nhân ở định nghĩa `stray_ink`: nó chỉ gọi một dải là "lạ" khi dải
đó giữ < 35% tổng mực VÀ nằm hẳn trong 30% trên/dưới. Chữ láng giềng LỌT TRỌN giữ ~33%
mực và trải quá dải 30% -> KHÔNG bị tính. Nó bắt mảnh vụn, không bắt láng giềng nguyên chữ.

T4-B (ngữ liệu tổng hợp) cũng không giải được: `synth` vẽ glyph vừa khít ô nên `recall`
luôn = 1,0000, tức mất hẳn cánh "pad quá nhỏ thì cắt mất nét".

THƯỚC ĐO ĐÚNG — DỰA TRÊN LIÊN THÔNG, ĐO ĐƯỢC TRÊN DỮ LIỆU THẬT
--------------------------------------------------------------
Bước lặp (pitch) cho một BIÊN VẬT LÝ: chữ thuộc ô cao đúng `pitch` quanh tâm nó. Mực
nằm ngoài ô đó là (a) đuôi nét của CHÍNH chữ ấy, hoặc (b) chữ láng giềng. Hai thứ này
phân biệt được bằng LIÊN THÔNG, không cần nhãn:

    mực CỦA MÌNH  = các thành phần liên thông có giao với ô trung tâm
    mực LẠ        = phần còn lại

    độ tinh khiết (purity)  = mực của mình / tổng mực trong crop
    bị cắt (clipped)        = thành phần của mình CHẠM mép trên/dưới crop

Hai đại lượng đối nghịch THẬT: pad nhỏ -> clipped tăng; pad lớn -> purity giảm (và lần
này láng giềng lọt trọn BỊ BẮT, vì nó là thành phần liên thông rời). Nên có cực đại nội.

    python -m pipeline.lab.crop_purity --pages 12
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "lab" / "t4bp_purity.csv"
PADS = (0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35, 0.45, 0.60)


def score(crop_gray, rect, own_center_y: float, pitch: float):
    """(purity, clipped, n_ink) cho một crop, dựa trên liên thông.

    `own_center_y`/`pitch` ở TOẠ ĐỘ TRANG; ô trung tâm = tâm ± pitch/2 quy về crop.
    """
    import cv2
    import numpy as np
    if crop_gray is None or rect is None or crop_gray.size == 0:
        return None
    bw = (crop_gray < 128).astype("uint8")
    n_ink = int(bw.sum())
    if n_ink == 0:
        return None
    rx1, ry1, rx2, ry2 = rect
    h = bw.shape[0]
    c0 = int(round(own_center_y - pitch / 2.0 - ry1))
    c1 = int(round(own_center_y + pitch / 2.0 - ry1))
    c0, c1 = max(0, min(h, c0)), max(0, min(h, c1))
    if c1 <= c0:
        return None
    n_lab, lab = cv2.connectedComponents(bw, connectivity=8)
    if n_lab <= 1:
        return None
    # nhãn nào có giao với ô trung tâm -> thuộc CHÍNH chữ này
    own = set(int(v) for v in np.unique(lab[c0:c1, :]) if v != 0)
    if not own:
        return None
    own_mask = np.isin(lab, list(own))
    n_own = int(own_mask.sum())
    # bỏ hạt bụi: thành phần < 0,3% tổng mực không tính vào "lạ"
    foreign = bw.astype(bool) & ~own_mask
    if foreign.any():
        n_lab2, lab2 = cv2.connectedComponents(foreign.astype("uint8"), connectivity=8)
        keep = 0
        for v in range(1, n_lab2):
            s = int((lab2 == v).sum())
            if s >= max(4, 0.003 * n_ink):
                keep += s
        n_foreign = keep
    else:
        n_foreign = 0
    denom = n_own + n_foreign
    purity = n_own / denom if denom else 0.0
    clipped = bool(own_mask[:2, :].any() or own_mask[-2:, :].any())
    return purity, clipped, n_ink


def run(n_pages: int = 12) -> list[dict]:
    import numpy as np
    from pipeline.lab.crop_grid import cut, load_boxes, page_cache
    cols, npg = load_boxes(n_pages=n_pages)
    cache = page_cache()
    # chuẩn bị: pitch + tâm y cho từng cột (tính MỘT LẦN, dùng cho mọi pad)
    prepped = []
    for col in cols:
        bs = sorted(col["bboxes"], key=lambda b: b[1])
        if len(bs) < 4:
            continue
        cys = [(b[1] + b[3]) / 2.0 for b in bs]
        d = np.diff(cys)
        d = d[(d > 0) & (d < 3 * np.median([b[3] - b[1] for b in bs]))]
        if len(d) < 3:
            continue
        prepped.append({**col, "bboxes": bs, "cys": cys, "pitch": float(np.median(d))})
    n_box = sum(len(c["bboxes"]) for c in prepped)
    print(f"[T4-B'] {npg} trang · {len(prepped):,} cột · {n_box:,} hộp")

    rows = []
    for pad in PADS:

        p_sum, n, clip, skip = 0.0, 0, 0, 0
        for col in prepped:
            img, gray_full = cache(col["book"], col["page"])
            if img is None:
                continue
            bs = col["bboxes"]
            for k, bb in enumerate(bs):
                g, rect = cut(img, gray_full, bb, pad, "fixed128", True,
                              bs[k - 1] if k else None,
                              bs[k + 1] if k + 1 < len(bs) else None,
                              return_rect=True)
                r = score(g, rect, col["cys"][k], col["pitch"])
                if r is None:
                    skip += 1
                    continue
                pur, cl, _ = r
                p_sum += pur; clip += int(cl); n += 1
        d = max(n, 1)
        pur, clr = p_sum / d, clip / d
        # F1 giữa "tinh khiết" và "không bị cắt" — hai hỏng hóc đối nghịch
        f1 = 2 * pur * (1 - clr) / max(pur + (1 - clr), 1e-9)
        rows.append({"pad": pad, "n": n, "skipped": skip, "purity": round(pur, 5),
                     "clipped": round(clr, 5), "f1": round(f1, 5)})
        print(f"  pad={pad:.2f}  tinh khiết={pur:.4f}  bị cắt={clr:.4f}  F1={f1:.4f}",
              flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.lab.crop_purity")
    ap.add_argument("--pages", type=int, default=12)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    rows = run(args.pages)
    best = max(rows, key=lambda r: r["f1"])
    base = next((r for r in rows if abs(r["pad"] - 0.12) < 1e-9), None)
    i = rows.index(best)
    interior = 0 < i < len(rows) - 1
    print("\n=== KẾT LUẬN T4-B' ===")
    if base:
        print(f"  MỐC pad 0,12       tinh khiết {base['purity']:.4f}  bị cắt {base['clipped']:.4f}"
              f"  F1 {base['f1']:.4f}")
    print(f"  tốt nhất pad {best['pad']:.2f}   tinh khiết {best['purity']:.4f}"
          f"  bị cắt {best['clipped']:.4f}  F1 {best['f1']:.4f}")
    print(f"  CỰC ĐẠI NỘI: {interior}"
          f"  ({'thước đo BẤU VÀO trục -> có thẩm quyền chọn pad' if interior else 'KHÔNG -> vẫn chưa quyết được'})")
    if base:
        print(f"  chênh F1 so với mốc: {best['f1'] - base['f1']:+.5f}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (REPO / "lab" / "t4bp_decision.json").write_text(json.dumps(
        {"baseline": base, "best": best, "interior_optimum": interior},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[T4-B'] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# T4.e — quét HẰNG SỐ NỚI DỌC `m = pitch * M`, núm vặn THẬT của chiều dọc
# =============================================================================
# `align_production.py:158` đặt `m = pitch * 0.10`, nên cao hộp = pitch*(1+2M) =
# pitch*1,20. Đặc tả T4.2 KHÔNG có trục này — nó chỉ quét `pad`, vốn nhân LÊN TRÊN
# hộp đã nới. Đo 537 cột: 99,44% hộp cao hơn bước lặp, 84,46% cặp liền kề chồng nhau.
#
# Ở đây dựng lại hộp từ ĐÚNG công thức của production (trung điểm ± m) với M thay
# đổi, rồi cắt bằng cấu hình đã giao (pad 0,12 + carve) và đo bằng thước đo liên
# thông. Đây là phép đo trực tiếp cho câu "nới 1,2× có đáng không".
M_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)


def boxes_from_pitch(bs, cys, pitch, M):
    """Dựng lại hộp theo ĐÚNG công thức production với hằng số nới `M`."""
    m = pitch * M
    n = len(cys)
    out = []
    for i, cy in enumerate(cys):
        top = (cys[i - 1] + cy) / 2.0 - m if i > 0 else cy - pitch / 2.0
        bot = (cys[i + 1] + cy) / 2.0 + m if i < n - 1 else cy + pitch / 2.0
        out.append((bs[i][0], int(round(top)), bs[i][2], int(round(bot))))
    return out


def sweep_m(n_pages: int = 20, pad: float = 0.12) -> list[dict]:
    import numpy as np
    from pipeline.lab.crop_grid import cut, load_boxes, page_cache
    cols, npg = load_boxes(n_pages=n_pages)
    cache = page_cache()
    prepped = []
    for col in cols:
        bs = sorted(col["bboxes"], key=lambda b: b[1])
        if len(bs) < 4:
            continue
        cys = [(b[1] + b[3]) / 2.0 for b in bs]
        d = np.diff(cys)
        d = d[(d > 0) & (d < 3 * np.median([b[3] - b[1] for b in bs]))]
        if len(d) < 3:
            continue
        prepped.append({**col, "bboxes": bs, "cys": cys, "pitch": float(np.median(d))})
    print(f"[T4.e] {npg} trang · {len(prepped):,} cột · "
          f"{sum(len(c['bboxes']) for c in prepped):,} hộp · pad={pad}")
    rows = []
    for M in M_GRID:
        p_sum = 0.0
        n = clip = 0
        for col in prepped:
            img, gray_full = cache(col["book"], col["page"])
            if img is None:
                continue
            nb = boxes_from_pitch(col["bboxes"], col["cys"], col["pitch"], M)
            for k, bb in enumerate(nb):
                g, rect = cut(img, gray_full, bb, pad, "fixed128", True,
                              nb[k - 1] if k else None,
                              nb[k + 1] if k + 1 < len(nb) else None,
                              return_rect=True)
                r = score(g, rect, col["cys"][k], col["pitch"])
                if r is None:
                    continue
                # bị cắt: đo theo ĐÚNG ngưỡng của border_ink (>0,20 hàng biên),
                # KHÔNG phải 'có pixel nào chạm mép' (lỗi tôi đã mắc lần đầu)
                bw = (g < 128)
                p_sum += r[0]
                clip += int(max(bw[:2, :].mean(), bw[-2:, :].mean()) > 0.20)
                n += 1
        d0 = max(n, 1)
        pur, clr = p_sum / d0, clip / d0
        f1 = 2 * pur * (1 - clr) / max(pur + (1 - clr), 1e-9)
        rows.append({"M": M, "box_h_over_pitch": round(1 + 2 * M, 3), "n": n,
                     "purity": round(pur, 5), "clipped": round(clr, 5),
                     "f1": round(f1, 5)})
        print(f"  M={M:.2f} (cao hộp = {1 + 2 * M:.2f}×pitch)  tinh khiết={pur:.4f}"
              f"  bị cắt={clr:.4f}  F1={f1:.4f}"
              f"{'   <-- PRODUCTION' if abs(M - 0.10) < 1e-9 else ''}", flush=True)
    return rows
