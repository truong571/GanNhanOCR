#!/usr/bin/env python
"""cclib_v2.py — CROP CHUẨN bản 2 (CC v2). VIẾT SAU khi thấy kết quả hình học của CC v1 (s02_geom, 26/09/2026) — KHÔNG
phải thiết kế đăng ký trước; đóng băng (md5 vào t00_freeze_v2.json) TRƯỚC khi đo v2. Không dò tham số: mọi giá trị là giá
trị tự nhiên hoặc giữ nguyên của v1; không chọn trên IHR, không dùng bộ kiểm ảnh.

Ba lỗi của v1 đã chẩn đoán (chỉ bằng số, không mở ảnh) và cách v2 sửa — đưa thuật toán về ĐÚNG đặc tả của người giao việc
("ranh giới = trung điểm tâm với ô kề; gán thành phần theo vị trí/chồng lấn; thành phần vắt qua ranh giới thì cắt theo seam"):
  (1) v1 lấy CHÍNH đường seam làm ranh giới, dải seam ±0,25·d quanh trung điểm, và khi nhiều hàng cùng năng lượng thì chọn hàng
      TRÊN CÙNG -> ranh giới dưới bám sát chữ (cắt mất nửa dưới chữ ⿱; L16 179 ô mất > 25 % mực khe người) và ranh giới trên bám
      sát chữ kề (lọt mảnh chữ kề; KVK 640 ô stray_ink > 0,08).
      v2: ranh giới = ĐƯỜNG THẲNG qua trung điểm hai tâm. Thành phần nằm trọn một phía được gán trọn theo phía đó. CHỈ thành phần
      vắt qua trung điểm mới bị cắt theo seam; seam tìm trong trung điểm ± 0,25·d, năng lượng thêm phạt khoảng cách tới trung
      điểm (tối đa 20/255) để khi hoà thì seam nằm ở trung điểm.
  (2) v1 để nét kẻ cột/khung dính chữ thành MỘT thành phần khổng lồ (lưới) -> không nhận ra là nét kẻ -> mực ô chạm mép cửa sổ
      (L16 837/850 cờ cut là "chạm mép trái/phải").
      v2: bóc nét kẻ TRƯỚC khi tách thành phần: mở hình thái với nhân dọc dài 1,2·p và nhân ngang dài 1,2·w (dài hơn một chữ);
      điểm ảnh nét kẻ = ngoại lai (tô nền).
  (3) v1 khi thiếu ô kề (chữ đầu/cuối cột) đặt ranh giới ở 0,8·max(p,h) -> lấn vào đầu khung/chữ tầng trên.
      v2: láng giềng ảo ở khoảng 2·0,55·max(p,h) -> trung điểm ảo ở 0,55·max(p,h) (hơi hơn nửa bước để không cắt chữ cao).
Giữ nguyên v1: láng giềng cột (bỏ hộp trùng khe), p, dải ngang (xc ± 0,75·w, cắt ở trung điểm với cột kề), nhị phân Otsu cục
bộ kẹp [64, 200], đốm, luật đầu nét lạ (20 % / 10 %), cờ blank/cut/two/tall/ink, xuất khung vuông lề 10 % nền giấy gốc + 128×128.
Trung điểm thật xa hơn 0,8·max(p, h) (khoảng trống lớn) bị kẹp về 0,8·max(p, h) như v1.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from cclib import REPO, _compute_seam, col_pitch, distinct_col_boxes, old_crop, otsu  # noqa: F401

PARAMS = dict(
    dup_frac=0.25, cap=0.8, virt=0.55, seam_band=0.25, seam_pen=20.0,
    x_half=0.75, win_ext=0.35, otsu_lo=64, otsu_hi=200,
    line_v=1.2, line_h=1.2,
    speck_frac=0.0015, speck_min=3,
    tip_frac=0.20, tip_mass=0.10, cut_cols=0.30,
    two_ext=1.35, two_gap=0.10, two_mass=0.20, blank_frac=0.02,
    ink_lo=0.4, ink_hi=2.5, ink_lo_w=0.25, ink_hi_w=4.0, ink_nmin=5,
    margin=0.10, size=128, tight_margin=4,
)


def _seam(gray_win, ink_win, xa, xb, ylo, yhi, mid, half, pen):
    sub_g = gray_win[ylo:yhi + 1, xa:xb].astype(np.float32)
    sub_i = ink_win[ylo:yhi + 1, xa:xb].astype(np.float32)
    rows = (np.arange(ylo, yhi + 1, dtype=np.float32) - mid)[:, None]
    e = 255.0 * sub_i + 0.1 * (255.0 - sub_g) + pen * np.abs(rows) / max(half, 1.0)
    pseudo = (255.0 - np.clip(e, 0, 255)).astype(np.uint8)
    s = _compute_seam(pseudo, 0, pseudo.shape[0] - 1)
    if s is None:
        return np.full(xb - xa, int(round(mid)), np.int32)
    return s.astype(np.int32) + ylo


def canon(G, SRC, bbox, col_boxes, other_col_cx, page_pitch=None, P=PARAMS):
    Hp, Wp = G.shape
    x1, y1, x2, y2 = (float(v) for v in bbox)
    h, w = max(1.0, y2 - y1), max(1.0, x2 - x1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    boxes = distinct_col_boxes(col_boxes) if col_boxes else []
    if not any(tuple(int(v) for v in b) == tuple(int(v) for v in bbox) for b in boxes):
        boxes = distinct_col_boxes(boxes + [list(bbox)])
    own_k = [i for i, b in enumerate(boxes) if tuple(int(v) for v in b) == tuple(int(v) for v in bbox)][0]
    others = [b for i, b in enumerate(boxes) if i != own_k and abs((b[1] + b[3]) / 2.0 - cy) >= P['dup_frac'] * h]
    prev = [b for b in others if (b[1] + b[3]) / 2.0 < cy]
    nxt = [b for b in others if (b[1] + b[3]) / 2.0 > cy]
    prev = prev[-1] if prev else None
    nxt = nxt[0] if nxt else None
    p = col_pitch(boxes)
    p_src = 'col'
    if p is None:
        p, p_src = (page_pitch, 'page') if page_pitch else (h, 'h')
    loc = boxes[max(0, own_k - 2): own_k + 3]
    xc = float(np.median([(b[0] + b[2]) / 2.0 for b in loc]))
    wc = max(float(np.median([b[2] - b[0] for b in loc])), 4.0)
    mh = max(p, h)
    cap = P['cap'] * mh
    xa, xb = xc - P['x_half'] * wc, xc + P['x_half'] * wc
    lefts = [c for c in other_col_cx if c < xc - 0.3 * wc]
    rights = [c for c in other_col_cx if c > xc + 0.3 * wc]
    if lefts:
        xa = max(xa, (xc + max(lefts)) / 2.0)
    if rights:
        xb = min(xb, (xc + min(rights)) / 2.0)
    if xb - xa < 0.5 * wc:
        xa, xb = xc - 0.5 * wc, xc + 0.5 * wc

    def side(nb, sign):     # sign +1 = trên, -1 = dưới ; trả (mid, d, kind)
        if nb is None:
            return cy - sign * P['virt'] * mh, 2 * P['virt'] * mh, 'virt'
        cyn = (nb[1] + nb[3]) / 2.0
        d = abs(cy - cyn)
        if d / 2.0 > cap:
            return cy - sign * cap, 2 * cap, 'capped'
        return (cy + cyn) / 2.0, d, 'mid'
    tmid, td, tk = side(prev, +1)
    bmid, bd_, bk = side(nxt, -1)
    ext_y, ext_x = P['win_ext'] * p, P['win_ext'] * wc
    wy0 = int(max(0, math.floor(tmid - P['seam_band'] * td - ext_y)))
    wy1 = int(min(Hp, math.ceil(bmid + P['seam_band'] * bd_ + ext_y) + 1))
    wx0 = int(max(0, math.floor(xa - ext_x)))
    wx1 = int(min(Wp, math.ceil(xb + ext_x) + 1))
    out = dict(p=round(p, 2), p_src=p_src, wc=round(wc, 2), xc=round(xc, 2), band_x=(round(xa, 1), round(xb, 1)),
               top_kind=tk, bot_kind=bk, win=(wx0, wy0, wx1, wy1), mid=(round(tmid, 1), round(bmid, 1)))
    if wy1 - wy0 < 4 or wx1 - wx0 < 4:
        out.update(status='bad_window', flags='blank')
        return out
    g = G[wy0:wy1, wx0:wx1]
    T = min(max(otsu(g), P['otsu_lo']), P['otsu_hi'])
    ink = g < T
    Hw, Ww = g.shape
    # ---- bóc nét kẻ
    lines = np.zeros_like(ink)
    vk, hk = int(round(P['line_v'] * p)), int(round(P['line_h'] * wc))
    u8 = ink.astype(np.uint8)
    if 3 <= vk < Hw:
        lines |= cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((vk, 1), np.uint8)).astype(bool)
    if 3 <= hk < Ww:
        lines |= cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((1, hk), np.uint8)).astype(bool)
    ink2 = ink & ~lines
    bxa, bxb = int(max(0, round(xa - wx0))), int(min(Ww, round(xb - wx0)))
    if bxb - bxa < 2:
        bxa, bxb = 0, Ww
    tm, bm = tmid - wy0, bmid - wy0
    # ---- seam quanh trung điểm (chỉ dùng cho thành phần vắt qua)
    def seam_full(mid, d):
        half = P['seam_band'] * d
        lo, hi = int(max(0, math.floor(mid - half))), int(min(Hw - 1, math.ceil(mid + half)))
        if hi <= lo or not (0 <= mid < Hw):
            full = np.full(Ww, int(round(min(max(mid, 0), Hw - 1))), np.int32)
            return full, full[bxa:bxb].copy()
        s = _seam(g, ink2, bxa, bxb, lo, hi, mid, half, P['seam_pen'])
        full = np.empty(Ww, np.int32); full[bxa:bxb] = s; full[:bxa] = s[0]; full[bxb:] = s[-1]
        return full, s
    top_s, top_band = seam_full(tm, td)
    bot_s, bot_band = seam_full(bm, bd_)
    yy = np.arange(Hw)[:, None]
    in_mid = (yy > tm) & (yy < bm)                         # (Hw, 1) — mọi cột
    in_seam = (yy > top_s[None, :]) & (yy < bot_s[None, :])
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink2.astype(np.uint8), connectivity=8)
    area = st[:, cv2.CC_STAT_AREA].astype(np.int64)
    V = np.bincount(lab[np.broadcast_to(in_mid, lab.shape)], minlength=n)
    Ks = np.bincount(lab[in_seam], minlength=n)
    speck_thr = max(P['speck_min'], P['speck_frac'] * p * wc)
    kind = np.zeros(n, np.int8)   # 0 ngoại lai, 1 trọn, 2 cắt seam, 3 đốm
    kept = np.zeros(n, np.int64)
    for j in range(1, n):
        a = area[j]
        if a < speck_thr:
            kind[j] = 3; continue
        if not (bxa <= cen[j][0] < bxb):
            continue
        if V[j] == a:
            kind[j] = 1; kept[j] = a; continue
        if V[j] == 0:
            continue
        kind[j] = 2; kept[j] = Ks[j]
    tot = kept.sum()
    n_tip = 0
    for j in np.nonzero(kind == 2)[0]:
        if kept[j] == 0 or (kept[j] < P['tip_frac'] * area[j] and kept[j] < P['tip_mass'] * tot):
            kind[j] = 0; kept[j] = 0; n_tip += 1
    own_ids = np.nonzero((kind == 1) | (kind == 2))[0]
    cut_ids = np.nonzero(kind == 2)[0]
    M = np.isin(lab, own_ids) & ink2
    M &= np.where(np.isin(lab, cut_ids), in_seam, True)
    mass = int(M.sum())
    pw_ = p * wc
    flags = []
    out.update(T=round(T, 1), n_comp=int(n - 1), n_own=int(len(own_ids)), n_cut_comp=int(len(cut_ids)), n_tip=n_tip,
               n_line=int(lines.sum()), mass=mass, ink_norm=round(mass / pw_, 4))
    if mass < P['blank_frac'] * pw_ or mass == 0:
        flags.append('blank')
    cutr = 0.0
    for sarr in (top_band, bot_band):
        if sarr is None or not len(cut_ids):
            continue
        cols = np.arange(bxa, bxa + len(sarr))
        ys = np.clip(sarr, 0, Hw - 1)
        on = ink2[ys, cols] & np.isin(lab[ys, cols], cut_ids)
        cutr = max(cutr, float(on.sum()) / max(1, len(sarr)))
    out['cut_ratio'] = round(cutr, 3)
    edge = bool(mass and (M[0, :].any() or M[-1, :].any() or M[:, 0].any() or M[:, -1].any()))
    out['edge'] = edge
    if cutr > P['cut_cols'] or edge:
        flags.append('cut')
    if mass:
        ys_, xs_ = np.nonzero(M)
        ay0, ay1, ax0, ax1 = int(ys_.min()), int(ys_.max()) + 1, int(xs_.min()), int(xs_.max()) + 1
        E, Wd = ay1 - ay0, ax1 - ax0
        two = False
        for proj, ext, unit in ((M[ay0:ay1].sum(1), E, p), (M[:, ax0:ax1].sum(0), Wd, wc)):
            if ext <= P['two_ext'] * unit:
                continue
            z = proj == 0
            cs = np.cumsum(proj)
            i = 0
            while i < len(z):
                if z[i]:
                    j = i
                    while j < len(z) and z[j]:
                        j += 1
                    if j - i >= P['two_gap'] * unit:
                        a_ = cs[i - 1] if i > 0 else 0
                        if min(a_, mass - a_) >= P['two_mass'] * mass:
                            two = True
                    i = j
                else:
                    i += 1
        if two:
            flags.append('two')
        tmg = P['tight_margin']
        ty0, ty1 = max(0, ay0 - tmg), min(Hw, ay1 + tmg)
        tx0, tx1 = max(0, ax0 - tmg), min(Ww, ax1 + tmg)
        foreign = ink & ~M
        fd = cv2.dilate(foreign.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~M
        spk = np.isin(lab, np.nonzero(kind == 3)[0]) & np.broadcast_to(in_mid, lab.shape)
        fd &= ~spk
        tg = g.copy()
        tg[fd] = 255
        tight = tg[ty0:ty1, tx0:tx1]
        out['tall'] = bool(tight.shape[0] > 1.8 * max(tight.shape[1], 1))
        L = max(E, Wd)
        m = int(math.ceil(P['margin'] * L))
        S = L + 2 * m
        icx, icy = wx0 + (ax0 + ax1) / 2.0, wy0 + (ay0 + ay1) / 2.0
        sx0, sy0 = int(round(icx - S / 2.0)), int(round(icy - S / 2.0))
        out.update(ink_box=(wx0 + ax0, wy0 + ay0, wx0 + ax1, wy0 + ay1), ink_cx=round(icx, 2), ink_cy=round(icy, 2),
                   sq=(sx0, sy0, S), E=E, Wd=Wd)
        px0, py0, px1, py1 = max(0, sx0), max(0, sy0), min(Wp, sx0 + S), min(Hp, sy0 + S)
        src = SRC[py0:py1, px0:px1]
        gs = G[py0:py1, px0:px1]
        inkp = gs < T
        Mp = np.zeros_like(inkp); Fp = np.zeros_like(inkp); inwin = np.zeros_like(inkp)
        ox0, oy0 = max(px0, wx0), max(py0, wy0)
        ox1, oy1 = min(px1, wx1), min(py1, wy1)
        if ox1 > ox0 and oy1 > oy0:
            sl_p = (slice(oy0 - py0, oy1 - py0), slice(ox0 - px0, ox1 - px0))
            sl_w = (slice(oy0 - wy0, oy1 - wy0), slice(ox0 - wx0, ox1 - wx0))
            Mp[sl_p] = M[sl_w]; Fp[sl_p] = fd[sl_w]; inwin[sl_p] = True
        out_ink = inkp & ~inwin
        if out_ink.any():
            Fp |= cv2.dilate(out_ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        Fp &= ~Mp
        nonink = ~inkp
        if src.ndim == 3:
            bg = (np.median(src[nonink], axis=0) if nonink.any() else np.array([255, 255, 255])).astype(np.uint8)
            canvas = np.empty((S, S, 3), np.uint8); canvas[:] = bg
        else:
            bg = np.uint8(np.median(src[nonink]) if nonink.any() else 255)
            canvas = np.full((S, S), bg, np.uint8)
        piece = src.copy()
        piece[Fp] = bg
        canvas[py0 - sy0:py1 - sy0, px0 - sx0:px1 - sx0] = piece
        out['sq_img'] = canvas
        out['sq128'] = cv2.resize(canvas, (P['size'], P['size']),
                                  interpolation=cv2.INTER_AREA if S >= P['size'] else cv2.INTER_CUBIC)
        out['tight_img'] = tight
        out['foreign_erased_px'] = int(Fp.sum())
        out['M_page'] = (px0, py0, Mp)
    else:
        out['tall'] = False
    out['flags'] = '|'.join(flags)
    out['status'] = 'ok' if mass else 'empty'
    out['_T'] = T
    return out
