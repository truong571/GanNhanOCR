#!/usr/bin/env python
"""Phép đo: hộp chữ detector (legacy) và bộ giải mã theo bước (pitch_decode) so với Ô THAM CHIẾU tự động.

Ô tham chiếu (0 người, 0 API): hộp TẦNG của kim (adapter ingest: 1 hộp / tầng-cột, 6 hoặc 8 chữ) cắt
tại N−1 khe mực yếu nhất theo chiếu mực ngang (pitch_decode.ink_cut_cells, quy hoạch động có phạt lệch
bước — KHÔNG chia đều), N = số âm QN của câu (transcriptions/<page>.json verse_odd/verse_even.n_syll;
dự phòng 6/8). Ô "verified" = số chữ kim của tầng == N (kim và QN đồng ý về N).

So sánh trên cùng 27 trang/sách của measure_out/detector_transfer (chọn đều np.linspace):
  legacy@thr : DetectorInfer.raw_column_boxes(x_range ± 0,05w, NMS dọc 0,45) ở thr 0,15 / 0,2 —
               đúng phép lọc production (config LVT det_thr 0,15, det_xmargin 0,05); tách thêm theo tầng
               (cửa sổ y = tầng ± 0,35 bước) để đếm hộp NGOÀI tầng (số câu in, rác lề) lọt vào dải x.
  legacy_final: n_det == N → G; khác → enforce_count(G → N) (đường ép đếm cũ; KHÔNG mô phỏng
               _monotone_assign/midpoint — số thật lấy từ build --limit).
  pitch      : pitch_decode.decode_column (ứng viên ≥ 0,05 + ô ảo chiếu mực, DP chọn đúng N).
Chỉ số từng ô tham chiếu: ghép 1-1 theo IoU ≥ 0,3 → ok (IoU ≥ 0,5) | shifted (ghép nhưng IoU < 0,5) |
miss (không hộp); extra = hộp không ghép; sai số tâm dy (px, % bước); IoU; "cắt vào thân chữ" =
mực ở 2 hàng mép trên/dưới của hộp > BORDER_INK_MAX (crop_quality, 0,20) — chỉ số duy nhất KHÔNG
phụ thuộc ô tham chiếu.

TRUNG THỰC: (i) pitch luôn trả đúng N hộp nên "n == N" là hằng đúng — không dùng làm bằng chứng;
(ii) ô nguồn 'ink_cut' của pitch trùng ô tham chiếu THEO CẤU TẠO (cùng hàm cắt) → IoU của chúng
không phải bằng chứng; bảng tách theo nguồn và bảng "honest" chỉ tính ô nguồn detector/detector_low;
(iii) không có GT người → mọi số là proxy.

Chạy:  .venv/bin/python scripts/measure/box_ref_eval.py --book all --pages 27
Thử:   .venv/bin/python scripts/measure/box_ref_eval.py --limit 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "measure"))

from detector_transfer import (LITHO_BOOKS, litho_text_pages, pick_even, parse_page_ids,   # noqa: E402
                               otsu, stretch, iou)
from pipeline.align_engine.char_detector import pitch_decode as PD                        # noqa: E402
from pipeline.align_engine.crop_quality import BORDER_INK_MAX                              # noqa: E402

PAGE_OF_UID = {"LucVanTien1883": lambda u: u, "KimVanKieu1884": lambda u: 167 - u}
THRS = [0.15, 0.2]
X_MARGIN = 0.05            # books[].det_xmargin thạch bản
Y_MARGIN = 0.35            # cửa sổ y tầng (× bước)
IOU_MATCH, IOU_OK = 0.3, 0.5
VARIANTS = ["prepared", "otsu"]   # prepared = ảnh pipeline thấy (LVT stretch, KVK otsu); otsu = otsu trên JPG gốc


# ----------------------------------------------------------------------------- dữ liệu
def load_page(book, jpg: Path):
    """→ (page_name, gray_prepared, gray_raw, cache, trans, col_pitch)."""
    uid = int(jpg.stem.split("_")[-1])
    page = f"page_{PAGE_OF_UID[book](uid):04d}"
    pdir = REPO / "prepared" / book
    import cv2
    gp = cv2.imread(str(pdir / "pages" / f"{page}.png"), cv2.IMREAD_GRAYSCALE)
    gr = cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE)
    cache = json.load(open(pdir / "detected" / f"{page}_ocr_cache.json", encoding="utf-8"))
    tp = pdir / "transcriptions" / f"{page}.json"
    trans = json.load(open(tp, encoding="utf-8")) if tp.exists() else {}
    pitch = None
    lp = REPO / "measure_out" / book / "layout" / "layout_pages.csv"
    if lp.exists():
        for r in csv.DictReader(open(lp, encoding="utf-8")):
            if int(r["uid"]) == uid and r.get("col_pitch"):
                pitch = float(r["col_pitch"]); break
    return page, gp, gr, cache, trans, pitch


def ref_cells_for_page(gray, cache, trans, col_pitch):
    """→ list cột: {col, x_range, chars, n_qn, tiers: [{t, box, n_kim, n_ref, verified, cells}]}."""
    out = []
    tcols = {c.get("column", i + 1): c for i, c in enumerate(trans.get("columns") or [])}
    for j, chars in enumerate(cache.get("columns") or []):
        if not chars:
            continue
        split = (cache.get("tier_split") or [[None, None]] * (j + 1))[j] if j < len(cache.get("tier_split") or []) else None
        tc = tcols.get(j + 1, {})
        n_odd = (tc.get("verse_odd") or {}).get("n_syll") or tc.get("len_odd") or 6
        n_even = (tc.get("verse_even") or {}).get("n_syll") or (tc.get("num_syllables", 14) - n_odd if tc else 8)
        if split and split[0] is not None:
            groups = [chars[:split[0]], chars[split[0]:]]
        else:
            idx = PD.group_tiers(chars)
            groups = [[chars[i] for i in t] for t in idx]
        n_refs = [int(n_odd), int(n_even)][:len(groups)]
        xs = [c["bbox"][0] for c in chars] + [c["bbox"][2] for c in chars]
        col = {"col": j + 1, "x_range": (min(xs), max(xs)), "chars": chars,
               "n_qn": int(tc.get("num_syllables") or (n_odd + n_even)), "tiers": []}
        for t, (g, n_ref) in enumerate(zip(groups, n_refs)):
            if not g or n_ref <= 0:
                continue
            x1, y0, x2, y1 = PD.tier_box(g, list(range(len(g))))
            if col_pitch:
                xc = (x1 + x2) / 2.0
                x1, x2 = int(max(x1, xc - 0.5 * col_pitch)), int(min(x2, xc + 0.5 * col_pitch))
            cells = PD.ink_cut_cells(gray, x1, x2, y0, y1, n_ref)
            col["tiers"].append({"t": t, "box": [x1, y0, x2, y1], "n_kim": len(g), "n_ref": n_ref,
                                 "verified": len(g) == n_ref, "cells": cells,
                                 "pitch": (y1 - y0) / n_ref})
        out.append(col)
    return out


# ----------------------------------------------------------------------------- chỉ số
def border_ink(binimg, b):
    """Mực ở 2 hàng mép trên/dưới của hộp (crop_quality.border_ink trên hộp thô)."""
    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
    H, W = binimg.shape
    x1, x2 = max(0, x1), min(W, x2); y1, y2 = max(0, y1), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 4:
        return 0.0
    sub = binimg[y1:y2, x1:x2]
    return float(max(sub[:2].mean(), sub[-2:].mean()))


def match_cells(cells, boxes):
    """Ghép 1-1 tham lam theo IoU ≥ IOU_MATCH. → (pairs[(ci, bi, iou)], miss_idx, extra_idx)."""
    pairs = []
    cand = sorted(((iou(c, b), ci, bi) for ci, c in enumerate(cells) for bi, b in enumerate(boxes)
                   if iou(c, b) >= IOU_MATCH), reverse=True)
    uc, ub = set(), set()
    for v, ci, bi in cand:
        if ci in uc or bi in ub:
            continue
        uc.add(ci); ub.add(bi); pairs.append((ci, bi, v))
    miss = [ci for ci in range(len(cells)) if ci not in uc]
    extra = [bi for bi in range(len(boxes)) if bi not in ub]
    return pairs, miss, extra


def cell_rows(book, page, col, tier, method, boxes, srcs, binimg, low_boxes=None):
    """Một dòng / ô tham chiếu (+ dòng extra) cho CSV; boxes = hộp của phương pháp trong tầng.
    kind: miss → 'below_thr' (có hộp ≥ 0,05 trùng ô, chỉ thiếu điểm) | 'absent' (detector không thấy);
          extra → 'dup' (chồng ≥ 0,3 với hộp đã ghép: 1 chữ bắt 2 lần) | 'stray' (rơi vào khe/ngoài ô)."""
    cells, p = tier["cells"], tier["pitch"]
    pairs, miss, extra = match_cells(cells, boxes)
    by_c = {ci: (bi, v) for ci, bi, v in pairs}
    matched_boxes = [boxes[bi] for _, bi, _ in pairs]
    rows = []
    for ci, c in enumerate(cells):
        r = {"book": book, "page": page, "col": col["col"], "tier": tier["t"], "cell": ci, "method": method,
             "verified": int(tier["verified"]), "n_ref": tier["n_ref"], "n_kim": tier["n_kim"], "pitch": round(p, 1),
             "cls": "miss", "kind": "", "iou": None, "dy_px": None, "dy_pct": None, "src": "", "score": None,
             "cut_glyph": None, "h_over_p": None, "bx1": None, "by1": None, "bx2": None, "by2": None}
        if ci in by_c:
            bi, v = by_c[ci]; b = boxes[bi]
            dy = (b[1] + b[3]) / 2.0 - (c[1] + c[3]) / 2.0
            r.update(cls="ok" if v >= IOU_OK else "shifted", iou=round(v, 3), dy_px=round(dy, 1),
                     dy_pct=round(100 * dy / p, 1), src=srcs[bi] if srcs else "", score=round(float(b[4]), 3) if len(b) > 4 else None,
                     cut_glyph=int(border_ink(binimg, b) > BORDER_INK_MAX), h_over_p=round((b[3] - b[1]) / p, 2),
                     bx1=int(b[0]), by1=int(b[1]), bx2=int(b[2]), by2=int(b[3]))
        elif low_boxes is not None:
            r["kind"] = "below_thr" if any(iou(c, lb) >= IOU_MATCH for lb in low_boxes) else "absent"
        rows.append(r)
    for bi in extra:
        b = boxes[bi]
        kind = "dup" if any(iou(b, mb) >= IOU_MATCH for mb in matched_boxes) else "stray"
        rows.append({"book": book, "page": page, "col": col["col"], "tier": tier["t"], "cell": -1, "method": method,
                     "verified": int(tier["verified"]), "n_ref": tier["n_ref"], "n_kim": tier["n_kim"], "pitch": round(p, 1),
                     "cls": "extra", "kind": kind, "iou": None, "dy_px": None, "dy_pct": None, "src": srcs[bi] if srcs else "",
                     "score": round(float(b[4]), 3) if len(b) > 4 else None,
                     "cut_glyph": int(border_ink(binimg, b) > BORDER_INK_MAX), "h_over_p": round((b[3] - b[1]) / p, 2),
                     "bx1": int(b[0]), "by1": int(b[1]), "bx2": int(b[2]), "by2": int(b[3])})
    return rows


def in_tier(boxes, tier):
    x1, y0, x2, y1 = tier["box"]; p = tier["pitch"]
    return [b for b in boxes if y0 - Y_MARGIN * p <= (b[1] + b[3]) / 2.0 <= y1 + Y_MARGIN * p]


# ----------------------------------------------------------------------------- đo 1 trang
def measure_page(det, book, jpg, variants, out_dbg=None):
    import cv2
    page, gp, gr, cache, trans, col_pitch = load_page(book, jpg)
    if gp is None or gr is None:
        return None, [], []
    _, binp = cv2.threshold(cv2.GaussianBlur(gp, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binp = (binp > 0).astype(np.uint8)
    cols = ref_cells_for_page(gp, cache, trans, col_pitch)
    prow = {"book": book, "page": page, "jpg": jpg.name, "n_cols": len(cols),
            "n_tiers": sum(len(c["tiers"]) for c in cols),
            "n_tiers_verified": sum(t["verified"] for c in cols for t in c["tiers"]),
            "n_cells": sum(t["n_ref"] for c in cols for t in c["tiers"])}
    crow, cell_out = [], []
    for v in variants:
        g = gp if v == "prepared" else otsu(gr)
        allb = det.boxes_for_page(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))    # ≥ 0,05
        prow[f"boxes_{v}"] = len(allb)
        gray_for_decode = g
        for col in cols:
            N = sum(t["n_ref"] for t in col["tiers"])
            for thr in THRS:
                pb = [b for b in allb if b[4] >= thr]
                G = det.raw_column_boxes(pb, col["x_range"], X_MARGIN)
                n_tier_boxes = [in_tier(G, t) for t in col["tiers"]]
                n_in = sum(len(x) for x in n_tier_boxes)
                cr = {"book": book, "page": page, "variant": v, "thr": thr, "col": col["col"], "N": N,
                      "n_qn": col["n_qn"], "n_det": len(G), "n_det_in_tiers": n_in, "n_outside": len(G) - n_in,
                      "eq": int(len(G) == N), "eq_in_tiers": int(n_in == N),
                      "tiers_eq": sum(1 for t, x in zip(col["tiers"], n_tier_boxes) if len(x) == t["n_ref"]),
                      "n_tiers": len(col["tiers"]),
                      "verified": int(all(t["verified"] for t in col["tiers"]))}
                crow.append(cr)
                # legacy: hộp thô trong tầng (low_boxes = ứng viên ≥ 0,05 trong tầng, để phân loại miss)
                for t, x in zip(col["tiers"], n_tier_boxes):
                    low = in_tier(det.raw_column_boxes(allb, col["x_range"], X_MARGIN), t)
                    cell_out += cell_rows(book, page, col, t, f"legacy@{thr}_{v}", x, None, binp, low_boxes=low)
                # legacy_final: ép đếm cả cột về N (đường cũ khi n_det ≠ N)
                fin = G if len(G) == N else [list(b) + [0.0] for b in det.enforce_count(G, N)]
                for t in col["tiers"]:
                    cell_out += cell_rows(book, page, col, t, f"legacy_final@{thr}_{v}", in_tier(fin, t), None, binp)
            # pitch (không phụ thuộc thr trừ nhãn nguồn; nhãn theo 0,15 = config LVT)
            boxes, srcs, info = PD.decode_column({"x_range": col["x_range"], "chars": col["chars"]},
                                                 allb, gray_for_decode, N, det_thr=0.15, x_margin=X_MARGIN,
                                                 tier_n=[t["n_ref"] for t in col["tiers"]])
            # gán hộp về tầng theo thứ tự (decode trả tầng trên → dưới, đúng N mỗi tầng)
            k = 0
            for t in col["tiers"]:
                n = t["n_ref"]
                cell_out += cell_rows(book, page, col, t, f"pitch_{v}", boxes[k:k + n], srcs[k:k + n], binp)
                k += n
        # ô tham chiếu: chính nó (cắt vào thân chữ của tham chiếu)
        if v == "prepared":
            for col in cols:
                for t in col["tiers"]:
                    cell_out += cell_rows(book, page, col, t, "ref", [c + [1.0] for c in t["cells"]], None, binp)
    if out_dbg is not None:
        im = cv2.cvtColor(gp, cv2.COLOR_GRAY2BGR)
        for col in cols:
            for t in col["tiers"]:
                for c in t["cells"]:
                    cv2.rectangle(im, (c[0], c[1]), (c[2], c[3]), (255, 160, 0), 2)
            boxes, srcs, _ = PD.decode_column({"x_range": col["x_range"], "chars": col["chars"]}, allb, gp,
                                              sum(t["n_ref"] for t in col["tiers"]), det_thr=0.15, x_margin=X_MARGIN,
                                              tier_n=[t["n_ref"] for t in col["tiers"]])
            for b, s in zip(boxes, srcs):
                color = {"detector": (0, 180, 0), "detector_low": (0, 200, 200), "ink_cut": (0, 0, 255)}[s]
                cv2.rectangle(im, (int(b[0]) + 3, int(b[1]) + 3), (int(b[2]) - 3, int(b[3]) - 3), color, 2)
        cv2.putText(im, "orange=ref cell | green=detector | yellow=detector_low | red=ink_cut", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        h, w = im.shape[:2]
        cv2.imwrite(str(out_dbg), cv2.resize(im, (int(w * 1400 / h), 1400)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return prow, crow, cell_out


# ----------------------------------------------------------------------------- tổng hợp
def pct(a, n):
    return round(100.0 * a / n, 1) if n else None


def med(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 2) if vals else None


def method_stats(rows):
    """rows = dòng ô của 1 (book, method) — chỉ ô verified. → dict chỉ số."""
    cells = [r for r in rows if r["cell"] >= 0]
    extra = [r for r in rows if r["cell"] < 0]
    n = len(cells)
    matched = [r for r in cells if r["cls"] != "miss"]
    st = {"n_cells": n, "n_extra": len(extra), "extra_per_100_cells": round(100.0 * len(extra) / n, 1) if n else None,
          "miss_pct": pct(sum(r["cls"] == "miss" for r in cells), n),
          "ok_iou50_pct": pct(sum(r["cls"] == "ok" for r in cells), n),
          "shifted_pct": pct(sum(r["cls"] == "shifted" for r in cells), n),
          "iou_med": med([r["iou"] for r in matched]),
          "abs_dy_px_med": med([abs(r["dy_px"]) for r in matched]),
          "abs_dy_pct_pitch_med": med([abs(r["dy_pct"]) for r in matched]),
          "abs_dy_pct_pitch_p90": (round(float(np.percentile([abs(r["dy_pct"]) for r in matched], 90)), 1) if matched else None),
          "cut_glyph_pct": pct(sum(r["cut_glyph"] or 0 for r in matched + extra), len(matched) + len(extra)),
          "h_over_p_med": med([r["h_over_p"] for r in matched])}
    srcs = {}
    for r in cells:
        srcs[r["src"] or "-"] = srcs.get(r["src"] or "-", 0) + 1
    st["by_src"] = srcs
    kinds = {}
    for r in cells + extra:
        if r["cls"] in ("miss", "extra") and r.get("kind"):
            k = f"{r['cls']}_{r['kind']}"; kinds[k] = kinds.get(k, 0) + 1
    st["miss_extra_kinds"] = kinds
    return st


def summarize(books, prow, crow, cell_out, variants):
    per_book = {}
    for book in books:
        pr = [p for p in prow if p["book"] == book]
        cr = [c for c in crow if c["book"] == book]
        ce = [c for c in cell_out if c["book"] == book]
        e = {"n_pages": len(pr), "n_cols": sum(p["n_cols"] for p in pr),
             "n_tiers": sum(p["n_tiers"] for p in pr), "n_tiers_verified": sum(p["n_tiers_verified"] for p in pr),
             "n_cells": sum(p["n_cells"] for p in pr), "columns": {}, "cells_verified": {}, "cells_all": {}}
        e["tiers_verified_pct"] = pct(e["n_tiers_verified"], e["n_tiers"])
        for v in variants:
            for thr in THRS:
                c = [x for x in cr if x["variant"] == v and x["thr"] == thr]
                n = len(c)
                dh = {}
                for x in c:
                    d = x["n_det"] - x["N"]; dh[d] = dh.get(d, 0) + 1
                e["columns"][f"{v}@{thr}"] = {
                    "n_cols": n, "I5_n_det_eq_N_pct": pct(sum(x["eq"] for x in c), n),
                    "n_det_in_tiers_eq_N_pct": pct(sum(x["eq_in_tiers"] for x in c), n),
                    "tiers_eq_pct": pct(sum(x["tiers_eq"] for x in c), sum(x["n_tiers"] for x in c)),
                    "cols_with_outside_box_pct": pct(sum(x["n_outside"] > 0 for x in c), n),
                    "outside_boxes_total": sum(x["n_outside"] for x in c),
                    "n_det_minus_N_hist": " ".join(f"{k}:{dh[k]}" for k in sorted(dh))}
        methods = sorted({r["method"] for r in ce})
        for mth in methods:
            rows = [r for r in ce if r["method"] == mth]
            e["cells_all"][mth] = method_stats(rows)
            e["cells_verified"][mth] = method_stats([r for r in rows if r["verified"]])
            if mth.startswith("pitch"):
                hon = [r for r in rows if r["verified"] and (r["cell"] < 0 or r["src"] in ("detector", "detector_low"))]
                e["cells_verified"][mth + "_honest(det_src_only)"] = method_stats(hon)
        per_book[book] = e
    return per_book


def build_invariants(per_book, variants, limited):
    inv = []

    def add(name, expected, observed, ok, method=""):
        inv.append({"name": name, "expected": expected, "observed": observed,
                    "pass": (None if limited else bool(ok)), "method": method})

    for book, e in per_book.items():
        add(f"{book}_tiers_verified_pct", ">= 85", e["tiers_verified_pct"], (e["tiers_verified_pct"] or 0) >= 85,
            "số chữ kim tầng == n_syll QN")
        ref = e["cells_verified"].get("ref", {})
        add(f"{book}_ref_cut_glyph_pct", "<= 10 (khe cắt trên hàng ít mực)", ref.get("cut_glyph_pct"),
            (ref.get("cut_glyph_pct") or 0) <= 10, "border_ink > 0,20 tại mép ô tham chiếu")
        for v in variants:
            leg = e["cells_verified"].get(f"legacy@0.15_{v}", {})
            pit = e["cells_verified"].get(f"pitch_{v}", {})
            hon = e["cells_verified"].get(f"pitch_{v}_honest(det_src_only)", {})
            add(f"{book}_{v}_pitch_n_boxes_eq_N_tautology", "== 100 (hằng đúng khi ép N — KHÔNG phải bằng chứng)",
                100.0 if (pit.get("n_cells") or 0) == (pit.get("n_cells") or 0) else None, True,
                "decode_column trả đúng N hộp/tầng theo cấu tạo; vị trí vẫn có thể sai → xem pitch miss_pct (IoU < 0,3)")
            add(f"{book}_{v}_pitch_miss_iou_lt_0.3_pct", "<= legacy@0.15 (lỗi VỊ TRÍ, không phải số đếm)",
                f"{pit.get('miss_pct')} vs {leg.get('miss_pct')}",
                (pit.get("miss_pct") or 0) <= (leg.get("miss_pct") or 0) + 0.5, "ô tham chiếu không hộp nào IoU ≥ 0,3")
            add(f"{book}_{v}_pitch_cut_glyph_le_legacy", "pitch <= legacy@0.15 (chỉ số không phụ thuộc tham chiếu)",
                f"{pit.get('cut_glyph_pct')} vs {leg.get('cut_glyph_pct')}",
                (pit.get("cut_glyph_pct") or 0) <= (leg.get("cut_glyph_pct") or 0) + 0.5, "border_ink > 0,20")
            add(f"{book}_{v}_pitch_honest_ok50_ge_legacy", "ô nguồn detector của pitch: IoU≥0,5 % >= legacy@0.15",
                f"{hon.get('ok_iou50_pct')} vs {leg.get('ok_iou50_pct')}",
                (hon.get("ok_iou50_pct") or 0) >= (leg.get("ok_iou50_pct") or 0) - 0.5, "chỉ ô detector/detector_low")
            col = e["columns"].get(f"{v}@0.15", {})
            add(f"{book}_{v}_outside_tier_boxes_explain_plus1", "cột có hộp ngoài tầng > 0 (chẩn đoán, không ngưỡng)",
                col.get("cols_with_outside_box_pct"), True, "hộp trong dải x nhưng ngoài 2 tầng")
    return inv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default="all", choices=["all", *LITHO_BOOKS])
    ap.add_argument("--out", default=None, help="mặc định measure_out/<book>/box_ref/ (all → measure_out/box_ref/)")
    ap.add_argument("--pages", type=int, default=27)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--page-ids", default="")
    ap.add_argument("--debug-pages", type=int, default=2)
    a = ap.parse_args()
    variants = [v for v in a.variants.split(",") if v in VARIANTS]
    books = list(LITHO_BOOKS) if a.book == "all" else [a.book]
    out = Path(a.out) if a.out else REPO / "measure_out" / ("box_ref" if a.book == "all" else f"{a.book}/box_ref")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", str(a.workers))
    import torch
    torch.manual_seed(0); torch.set_num_threads(a.workers); np.random.seed(0)
    from pipeline.align_engine.char_detector.detector_infer import DetectorInfer
    det = DetectorInfer(thr=PD.PITCH_CAND_THR, device="cpu")
    assert det.trained, "không nạp được train_crop/detector_r34.best.pt"
    page_ids = parse_page_ids(a.page_ids)
    pages = []
    for b in books:
        files = litho_text_pages(b)
        if b in page_ids:
            files = [f for f in files if f.stem.split("_")[-1] in page_ids[b]]
        else:
            files = pick_even(files, min(a.pages, a.limit) if a.limit else a.pages)
        pages += [(b, f) for f in files]
    print(f"box_ref_eval | {len(pages)} trang × {variants} × thr {THRS} | ckpt train_crop/detector_r34.best.pt", flush=True)
    t0 = time.time()
    prow, crow, cell_out, dbg = [], [], [], 0
    for b, f in pages:
        d = None
        if dbg < a.debug_pages and not any(p["book"] == b for p in prow):
            d = out / f"debug_{b}_{f.stem}_pitch.jpg"; dbg += 1
        p, c, ce = measure_page(det, b, f, variants, out_dbg=d)
        if p is None:
            continue
        prow.append(p); crow += c; cell_out += ce
    per_book = summarize(books, prow, crow, cell_out, variants)
    inv = build_invariants(per_book, variants, limited=bool(a.limit))
    for name, rows in (("box_ref_pages.csv", prow), ("box_ref_columns.csv", crow), ("box_ref_cells.csv", cell_out)):
        if rows:
            keys = list(rows[0].keys())
            with open(out / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summary = {"measure": "box_ref_eval", "date": time.strftime("%Y-%m-%d"),
               "cli": " ".join(["scripts/measure/box_ref_eval.py"] + sys.argv[1:]),
               "reference": "hộp tầng kim cắt N−1 khe mực yếu nhất (pitch_decode.ink_cut_cells, DP phạt lệch bước), "
                            "N = n_syll QN; x kẹp ±0,5 col_pitch quanh tâm hộp kim; verified = n_kim == N",
               "methods": {"legacy@thr": "raw_column_boxes ±0,05w NMS 0,45, tách theo tầng ±0,35p",
                           "legacy_final@thr": "n_det==N → G, khác → enforce_count (không mô phỏng midpoint)",
                           "pitch": "pitch_decode.decode_column (ứng viên ≥ 0,05 + ô ảo, DP), nhãn nguồn theo det_thr 0,15"},
               "caveats": ["pitch: n == N là hằng đúng", "ô ink_cut trùng tham chiếu theo cấu tạo → xem *_honest",
                           "không GT người → proxy"],
               "pages": {"n": len(prow), "per_book": {b: e["n_pages"] for b, e in per_book.items()}},
               "variants": variants, "runtime_s": round(time.time() - t0, 1),
               "per_book": per_book, "invariants": inv,
               "files": ["box_ref_pages.csv", "box_ref_columns.csv", "box_ref_cells.csv", f"debug_*.jpg ×{dbg}"]}
    txt = json.dumps(summary, ensure_ascii=False, indent=1)
    if len(txt.encode()) > 8000:
        for e in per_book.values():
            e.pop("cells_all", None)
        summary["note"] = "cells_all bỏ (xem box_ref_cells.csv)"
        txt = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    (out / "summary.json").write_text(txt, encoding="utf-8")
    for b, e in per_book.items():
        c15 = e["columns"].get("prepared@0.15", {})
        print(f"{b}: cols {e['n_cols']} | tiers verified {e['tiers_verified_pct']}% | I5@0.15 {c15.get('I5_n_det_eq_N_pct')}% "
              f"| in-tier eq {c15.get('n_det_in_tiers_eq_N_pct')}% | cols w/ outside box {c15.get('cols_with_outside_box_pct')}%")
        for m in ("legacy@0.15_prepared", "legacy@0.2_prepared", "legacy_final@0.15_prepared", "pitch_prepared",
                  "pitch_prepared_honest(det_src_only)", "ref"):
            s = e["cells_verified"].get(m)
            if s:
                print(f"   {m:38s} n={s['n_cells']:4d} miss {s['miss_pct']}% extra/100 {s['extra_per_100_cells']} "
                      f"ok50 {s['ok_iou50_pct']}% IoU {s['iou_med']} |dy| {s['abs_dy_px_med']}px {s['abs_dy_pct_pitch_med']}%p "
                      f"cut {s['cut_glyph_pct']}% src {s['by_src']}")
    for i in inv:
        print(f"  [{'PASS' if i['pass'] else ('SKIP' if i['pass'] is None else 'FAIL')}] {i['name']}: {i['observed']}")
    print(f"saved {out}/summary.json ({len(txt.encode())} B), {len(cell_out)} dòng ô, {summary['runtime_s']}s")


if __name__ == "__main__":
    main()
