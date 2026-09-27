"""geom.py — hình học trang: gom cột phải->trái, chuỗi chữ NGƯỜI, đơn vị (hộp thật + ô ảo), crop nhúng.

Bản chép NGUYÊN hành vi r4/borg_align/scripts/blib.py (nms_vertical, columns, human_seq) và a04_units.py
(units_for_page, crop_gray). Tham số ở params.COLUMNS / params.UNITS.
"""
from __future__ import annotations

import json
import unicodedata

import numpy as np

from .params import COLUMNS as C, REPO, UNITS as UP


def stretch(g, lo=1.0, hi=99.0):
    """a01_detect.stretch — kiểm prepared/pages == stretch(gray(jpg gốc))."""
    a, b = np.percentile(g, lo), np.percentile(g, hi)
    return np.clip((g.astype(np.float32) - a) * (255.0 / max(b - a, 1)), 0, 255).astype(np.uint8)


def nms_vertical(bx, iou_thr=C["nms_iou"]):
    keep = []
    for b in sorted(bx, key=lambda b: b[4], reverse=True):
        dup = False
        for k in keep:
            inter = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
            union = (b[3] - b[1]) + (k[3] - k[1]) - inter
            xo = max(0.0, min(b[2], k[2]) - max(b[0], k[0])) / max(1e-6, min(b[2] - b[0], k[2] - k[0]))
            if union > 0 and inter / union > iou_thr and xo > C["nms_xo"]:
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep


def columns(boxes, thr_core=C["thr_core"], thr_min=C["thr_min"], gap_factor=C["gap_factor"]):
    """Cột = cụm tâm-x của hộp tin cậy (>= thr_core); hộp yếu (>= thr_min) gán về cột gần nhất nếu |dx| <= 0.5·rộng trung vị.
    Trả (list cột phải->trái, mỗi cột dict(boxes sắp theo y, cx, fit), wmed)."""
    core = [b for b in boxes if b[4] >= thr_core]
    if not core:
        return [], 0.0
    wmed = float(np.median([b[2] - b[0] for b in core]))
    bs = sorted(core, key=lambda b: (b[0] + b[2]) / 2)
    cl, cur = [], [bs[0]]
    for b in bs[1:]:
        if (b[0] + b[2]) / 2 - (cur[-1][0] + cur[-1][2]) / 2 > gap_factor * wmed:
            cl.append(cur)
            cur = [b]
        else:
            cur.append(b)
    cl.append(cur)
    cols = [dict(boxes=list(c)) for c in cl]
    for c in cols:   # đường tâm cột: hồi quy x theo y (chữ viết tay có thể nghiêng)
        ys = np.array([(b[1] + b[3]) / 2 for b in c["boxes"]])
        xs = np.array([(b[0] + b[2]) / 2 for b in c["boxes"]])
        if len(c["boxes"]) >= C["fit_min"]:
            A = np.vstack([ys, np.ones_like(ys)]).T
            c["fit"] = np.linalg.lstsq(A, xs, rcond=None)[0]
        else:
            c["fit"] = np.array([0.0, float(np.median(xs))])
    for b in [b for b in boxes if thr_min <= b[4] < thr_core]:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        d = [abs(cx - (c["fit"][0] * cy + c["fit"][1])) for c in cols]
        j = int(np.argmin(d))
        if d[j] <= C["weak_dx"] * wmed:
            cols[j]["boxes"].append(b)
    out = []
    for c in cols:
        bb = nms_vertical(c["boxes"])
        bb.sort(key=lambda b: (b[1] + b[3]) / 2)
        out.append(dict(boxes=bb, cx=float(np.median([(b[0] + b[2]) / 2 for b in bb])), fit=c["fit"].tolist()))
    out.sort(key=lambda c: -c["cx"])
    return out, wmed


def human_seq(book, pn):
    """Chuỗi chữ Hán/Nôm (Lo) của trang theo thứ tự câu (nom_clean); âm QN chỉ khi câu có #chữ == #âm.
    Trả (chars, syls, sent_idx, after_dot)."""
    d = json.load(open(REPO / "prepared" / book / "transcriptions" / f"{pn}.json", encoding="utf-8"))
    chars, syls, sid, after_dot = [], [], [], []
    for si, s in enumerate(d["sentences"]):
        cs = [c for c in s["nom_clean"] if unicodedata.category(c) == "Lo"]
        q = s.get("qn_syllables") or []
        aligned = len(q) == len(cs)
        dots, k = set(), 0
        for c in s["nom_raw"]:
            if unicodedata.category(c) == "Lo":
                k += 1
            elif c == "。":
                dots.add(k - 1)
        for i, c in enumerate(cs):
            chars.append(c)
            syls.append(q[i] if aligned else "")
            sid.append(si)
            after_dot.append(i in dots)
    return chars, syls, sid, after_dot


def units_for_page(bx):
    """a04_units.units_for_page: đơn vị theo cột (phải->trái) rồi hàng; ô ảo ở đầu/giữa/cuối cột theo bước chữ trang."""
    cols, wmed = columns(bx)
    ct = UP["core_thr"]
    core_h = [b[3] - b[1] for c in cols for b in c["boxes"] if b[4] >= ct]
    dys = []
    for c in cols:
        cy = [(b[1] + b[3]) / 2 for b in c["boxes"] if b[4] >= ct]
        dys += list(np.diff(cy))
    pitch = float(np.median(dys)) if dys else float(np.median(core_h))
    tops = [c["boxes"][0][1] for c in cols if len(c["boxes"]) >= UP["line_min_boxes"]]
    bots = [c["boxes"][-1][3] for c in cols if len(c["boxes"]) >= UP["line_min_boxes"]]
    top_line = float(np.median(tops)) if tops else 0
    bot_line = float(np.median(bots)) if bots else 1e9
    U = []
    for ci, c in enumerate(cols):
        bb = c["boxes"]
        a, b0 = c["fit"]
        hh = pitch * UP["virt_h"]

        def virt(cy, a=a, b0=b0, hh=hh):
            cx = a * cy + b0
            return [cx - wmed / 2, cy - hh / 2, cx + wmed / 2, cy + hh / 2, 0.0]
        seq = []
        if bb:   # đầu cột
            g = (bb[0][1] + bb[0][3]) / 2 - (top_line + pitch / 2)
            k = int(round(g / pitch)) if g > UP["edge_open"] * pitch else 0
            for t in range(k):
                seq.append((virt(top_line + pitch / 2 + t * pitch), 1))
        for i, b in enumerate(bb):
            if i > 0:
                g = (b[1] + b[3]) / 2 - (bb[i - 1][1] + bb[i - 1][3]) / 2
                k = int(round(g / pitch)) - 1
                if g > UP["gap_split"] * pitch and k >= 1:
                    y0 = (bb[i - 1][1] + bb[i - 1][3]) / 2
                    for t in range(1, k + 1):
                        seq.append((virt(y0 + t * g / (k + 1)), 1))
            seq.append((list(b), 0))
        if bb:   # cuối cột
            g = (bot_line - pitch / 2) - (bb[-1][1] + bb[-1][3]) / 2
            k = int(round(g / pitch)) if g > UP["edge_open"] * pitch else 0
            for t in range(1, k + 1):
                seq.append((virt((bb[-1][1] + bb[-1][3]) / 2 + t * pitch), 1))
        for r, (b, v) in enumerate(seq):
            U.append(dict(col=ci, row=r, x1=b[0], y1=b[1], x2=b[2], y2=b[3], score=b[4], virtual=v))
    return U, dict(pitch=pitch, wmed=wmed, ncol=len(cols))


def crop_gray(g, u, pad=UP["pad"]):
    """a04_units.crop_gray: crop nhúng (pad + tighten_box) trên ảnh xám prepared."""
    from pipeline.align_engine.bbox_fix import tighten_box   # chỉ import
    H, Wd = g.shape
    x1, y1, x2, y2 = u["x1"], u["y1"], u["x2"], u["y2"]
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    a, b, c, d = max(0, int(x1 - pw)), max(0, int(y1 - ph)), min(Wd, int(x2 + pw)), min(H, int(y2 + ph))
    cr = g[b:d, a:c]
    if cr.size == 0:
        return np.full((8, 8), 255, np.uint8)
    tb = tighten_box(cr)
    if tb is not None:
        p, q, r, s = tb
        if s > q and r > p:
            cr = cr[q:s, p:r]
    return cr
