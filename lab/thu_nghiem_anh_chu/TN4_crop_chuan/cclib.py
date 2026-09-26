#!/usr/bin/env python
"""cclib.py — THỬ NGHIỆM 4: thuật toán "CROP CHUẨN" (CC v1) cho MỘT ô, tính từ ảnh TRANG (không từ crop cũ).

THIẾT KẾ ĐÓNG BĂNG — viết 26/09/2026 TRƯỚC khi đo bất cứ gì của TN4. Tham số PARAMS dưới đây đặt theo lý lẽ hình học
(bước chữ p, bề ngang cột w), KHÔNG chọn trên IHR, KHÔNG chọn bằng bộ kiểm ảnh. md5 của tệp này ghi vào
t00_freeze.json trước lượt đo đầu; mọi lượt đo sau kiểm lại md5 (đổi tệp = hỏng invariant `design_frozen`).
Chỉ ĐỌC repo; dùng (import, không sửa) pipeline.align_engine.{bbox_fix,build_dataset,crop_quality}.

ĐẦU VÀO của một ô: G = ảnh xám của trang ĐÃ XỬ LÝ (prepared/<Book>/pages — hệ toạ độ của bbox), SRC = ảnh trang GỐC
(crop_source: original, nạp bằng build_dataset.load_original_page; không có thì dùng trang đã xử lý), bbox của ô,
các hộp khác CÙNG CỘT (bản ghi build mọi tầng, như PASS 2), tâm x các cột khác của trang.

B1  DẢI KHE (slot band) của ô:
    - hộp cùng cột, bỏ hộp trùng khe (|cy_b − cy| < 0,25·h), xếp theo tâm y; prev/next = hộp kề trên/dưới.
    - p = trung vị khoảng cách tâm liên tiếp trong cột (≥ 2 khoảng; thiếu -> trung vị trang; thiếu nữa -> h).
    - cap = 0,8·max(p, h): ranh giới trên/dưới không xa tâm ô quá cap.
    - ranh giới TRÊN: có prev và (cy − cy_prev)/2 ≤ cap -> đường SEAM ít mực nhất (thuật toán _compute_seam của
      bbox_fix, dịch chuyển dọc ≤ 1 px/cột) tìm trong dải [mid − 0,25·d, mid + 0,25·d] ∩ [cy − cap, cy − 0,1·h],
      mid = trung điểm hai tâm, d = khoảng cách hai tâm; năng lượng = 255·mực + 0,1·(255 − xám).
      Ngược lại -> đường thẳng y = cy − cap.  Ranh giới DƯỚI đối xứng.
    - chiều ngang: xc, w = trung vị tâm x / bề ngang của các hộp vị trí k−2..k+2 trong cột (gồm ô); dải
      [xc − 0,75·w, xc + 0,75·w], cắt thêm bởi trung điểm với tâm cột kề trái/phải (nếu có).
B2  NHỊ PHÂN CỤC BỘ: cửa sổ phân tích = hộp bao dải + 0,30·p (dọc) / 0,30·w (ngang); ngưỡng Otsu trên cửa sổ,
    kẹp vào [64, 200]; mực = xám < ngưỡng.
B3  THÀNH PHẦN LIÊN THÔNG (8-lân cận) trên mực của cửa sổ; phân loại:
    - đốm: diện tích < max(3, 0,0015·p·w)  -> không tính vào hộp/khối mực; đốm TRONG khe giữ nguyên điểm ảnh,
      đốm NGOÀI khe tô nền.
    - nét kẻ: (cao ≥ 1,2·p và rộng ≤ 0,2·w) hoặc (rộng ≥ 1,2·w và cao ≤ 0,2·p) -> ngoại lai (tô nền).
    - trọn trong khe -> của ô; trọn ngoài khe -> ngoại lai.
    - vắt qua ranh giới: tâm x ngoài dải ngang -> ngoại lai (thuộc cột khác); ngược lại giữ PHẦN nằm giữa hai seam
      (cắt theo seam như carve; không cắt ngang). Phần giữ là "đầu nét lạ" và bị bỏ nếu < 20 % thành phần đó
      VÀ < 10 % tổng mực của ô.
B4  HỘP CHẶT quanh mực của ô (mặt nạ M).
B5  KIỂM "MỘT CHỮ" (chỉ gắn cờ, KHÔNG ép):
    - blank : mực của ô < 0,02·p·w.
    - cut   : (a) ranh giới trên hoặc dưới (seam hoặc đường thẳng) đi qua mực của một thành phần được giữ ở > 30 %
              số cột của dải (không tìm
              được khe giữa hai chữ -> nét bị cắt), hoặc (b) mực của ô chạm mép cửa sổ phân tích (nét còn tiếp ra ngoài).
    - two   : chiều cao mực > 1,35·p và có khe trắng ≥ 0,10·p chia khối mực thành hai phần mỗi phần ≥ 20 %;
              hoặc bề ngang > 1,35·w với khe dọc ≥ 0,10·w chia 20/20 (hai chữ / chữ cột kề).
    - tall  : luật `_seg_flag` của pipeline (h > 1,8·w) trên crop chặt đã xử lý (hộp mực + lề 4 px) — giữ để so.
    - ink   : (lượt 2, cần cả bộ) r = (mực/(p·w)) / trung vị cùng (bộ, nhãn) nếu lớp có ≥ 5 ô; ngoài [0,4; 2,5] -> cờ.
              Lớp < 5 ô: so với trung vị cả bộ, khoảng [0,25; 4,0].
    one_char_ok = không cờ nào trong {blank, cut, two, ink}.  ('tall' báo riêng, được dùng trong cổng hình học H1'.)
B6  XUẤT: khung VUÔNG, cạnh S = L + 2·ceil(0,10·L), L = cạnh dài hộp mực; tâm = tâm hộp mực; điểm ảnh lấy từ SRC
    (giữ nền giấy gốc); điểm ảnh mực ngoại lai (mực ∧ ¬M, nở 1 px, không đè M) và phần ngoài trang tô màu GIẤY
    = trung vị từng kênh của điểm không-mực trong khung. Bản cố định 128×128 (INTER_AREA khi thu, INTER_CUBIC khi phóng).
    Kèm crop "chặt đã xử lý" (G, hộp mực + 4 px, ngoại lai = 255) — dùng để đo cờ pipeline (crop_quality) trước/sau.
CROP CŨ được dựng lại đúng luật save_crop (pad 0,12, carve láng giềng theo thứ tự y trong cột, tighten_box) để đo
trên CÙNG thang (đã kiểm md5 byte với tệp giao ở bước s01 trên mẫu).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path('/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from pipeline.align_engine.bbox_fix import _compute_seam, carve_neighbor_ink, tighten_box  # noqa: E402

PARAMS = dict(
    dup_frac=0.25,          # hộp cùng cột có |Δcy| < 0,25·h = trùng khe, không làm láng giềng
    cap=0.8,                # ranh giới ≤ 0,8·max(p, h) tính từ tâm ô
    seam_band=0.25,         # seam tìm trong mid ± 0,25·d
    own_guard=0.1,          # seam không lấn quá cy ∓ 0,1·h
    x_half=0.75,            # nửa bề ngang dải = 0,75·w
    win_ext=0.30,           # cửa sổ phân tích nới 0,30·p / 0,30·w
    otsu_lo=64, otsu_hi=200,
    speck_frac=0.0015, speck_min=3,
    line_len=1.2, line_thin=0.2,
    tip_frac=0.20, tip_mass=0.10,
    cut_cols=0.30,
    two_ext=1.35, two_gap=0.10, two_mass=0.20,
    blank_frac=0.02,
    ink_lo=0.4, ink_hi=2.5, ink_lo_w=0.25, ink_hi_w=4.0, ink_nmin=5,
    margin=0.10, size=128, tight_margin=4,
)


# ----------------------------------------------------------------------------- ngữ cảnh cột
def distinct_col_boxes(boxes):
    """list bbox -> list bbox khác nhau, xếp theo tâm y."""
    seen, out = set(), []
    for b in boxes:
        t = tuple(int(v) for v in b)
        if t not in seen:
            seen.add(t); out.append(list(t))
    out.sort(key=lambda b: (b[1] + b[3]) / 2.0)
    return out


def col_pitch(boxes):
    cs = sorted((b[1] + b[3]) / 2.0 for b in boxes)
    d = np.diff(cs)
    d = d[d > 0]
    return float(np.median(d)) if len(d) >= 2 else None


def otsu(g):
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def _seam_line(gray_win, ink_win, xa, xb, ylo, yhi):
    """seam trong cửa sổ (toạ độ cửa sổ) qua cột [xa, xb), hàng [ylo, yhi]; trả mảng y theo cột xa..xb-1."""
    sub_g = gray_win[ylo:yhi + 1, xa:xb].astype(np.float32)
    sub_i = ink_win[ylo:yhi + 1, xa:xb].astype(np.float32)
    e = np.clip(255.0 * sub_i + 0.1 * (255.0 - sub_g), 0, 255)
    pseudo = (255.0 - e).astype(np.uint8)
    s = _compute_seam(pseudo, 0, pseudo.shape[0] - 1)
    if s is None:
        return np.full(xb - xa, (ylo + yhi) // 2, np.int32)
    return s.astype(np.int32) + ylo


def canon(G, SRC, bbox, col_boxes, other_col_cx, page_pitch=None, P=PARAMS):
    """Crop chuẩn cho một ô. Trả dict: ảnh (sq, sq128, tight), hình học (toạ độ trang), cờ, số đo."""
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
    p = col_pitch([b for b in boxes])
    p_src = 'col'
    if p is None:
        p, p_src = (page_pitch, 'page') if page_pitch else (h, 'h')
    loc = boxes[max(0, own_k - 2): own_k + 3]
    xc = float(np.median([(b[0] + b[2]) / 2.0 for b in loc]))
    wc = float(np.median([b[2] - b[0] for b in loc]))
    wc = max(wc, 4.0)
    cap = P['cap'] * max(p, h)
    # --- chiều ngang
    xa, xb = xc - P['x_half'] * wc, xc + P['x_half'] * wc
    lefts = [c for c in other_col_cx if c < xc - 0.3 * wc]
    rights = [c for c in other_col_cx if c > xc + 0.3 * wc]
    if lefts:
        xa = max(xa, (xc + max(lefts)) / 2.0)
    if rights:
        xb = min(xb, (xc + min(rights)) / 2.0)
    if xb - xa < 0.5 * wc:
        xa, xb = xc - 0.5 * wc, xc + 0.5 * wc
    # --- chiều dọc: dải tìm seam hoặc đường thẳng
    def side(nb, sign):
        if nb is None:
            return ('line', cy - sign * cap)
        cyn = (nb[1] + nb[3]) / 2.0
        d = abs(cy - cyn)
        if d / 2.0 > cap:
            return ('line', cy - sign * cap)
        mid = (cy + cyn) / 2.0
        if sign > 0:   # trên
            lo, hi = max(mid - P['seam_band'] * d, cy - cap), min(mid + P['seam_band'] * d, cy - P['own_guard'] * h)
        else:
            lo, hi = max(mid - P['seam_band'] * d, cy + P['own_guard'] * h), min(mid + P['seam_band'] * d, cy + cap)
        if hi - lo < 1:
            return ('line', mid)
        return ('seam', lo, hi)
    top, bot = side(prev, +1), side(nxt, -1)
    ytop_min = top[1]
    ybot_max = bot[2] if bot[0] == 'seam' else bot[1]
    ext_y, ext_x = P['win_ext'] * p, P['win_ext'] * wc
    wy0 = int(max(0, math.floor(ytop_min - ext_y)))
    wy1 = int(min(Hp, math.ceil(ybot_max + ext_y) + 1))
    wx0 = int(max(0, math.floor(xa - ext_x)))
    wx1 = int(min(Wp, math.ceil(xb + ext_x) + 1))
    out = dict(p=round(p, 2), p_src=p_src, wc=round(wc, 2), xc=round(xc, 2), band_x=(round(xa, 1), round(xb, 1)),
               top_kind=top[0], bot_kind=bot[0], win=(wx0, wy0, wx1, wy1))
    if wy1 - wy0 < 4 or wx1 - wx0 < 4:
        out.update(status='bad_window', flags='blank')
        return out
    g = G[wy0:wy1, wx0:wx1]
    T = min(max(otsu(g), P['otsu_lo']), P['otsu_hi'])
    ink = g < T
    Hw, Ww = g.shape
    bxa, bxb = int(max(0, round(xa - wx0))), int(min(Ww, round(xb - wx0)))
    if bxb - bxa < 2:
        bxa, bxb = 0, Ww
    # seams (toạ độ cửa sổ) cho MỌI cột của cửa sổ (ngoài dải: kéo dài giá trị mép)
    def seam_arr(sd):
        if sd[0] == 'line':
            yy = int(round(sd[1] - wy0))
            full = np.full(Ww, yy, np.int32)
            return full, full[bxa:bxb].copy()
        lo, hi = int(max(0, math.floor(sd[1] - wy0))), int(min(Hw - 1, math.ceil(sd[2] - wy0)))
        if hi <= lo:
            full = np.full(Ww, lo, np.int32)
            return full, full[bxa:bxb].copy()
        s = _seam_line(g, ink, bxa, bxb, lo, hi)
        full = np.empty(Ww, np.int32)
        full[bxa:bxb] = s
        full[:bxa] = s[0]
        full[bxb:] = s[-1]
        return full, s
    top_s, top_band = seam_arr(top)
    bot_s, bot_band = seam_arr(bot)
    yy = np.arange(Hw)[:, None]
    inside_v = (yy > top_s[None, :]) & (yy < bot_s[None, :])
    xband = np.zeros(Ww, bool)
    xband[bxa:bxb] = True
    inside = inside_v & xband[None, :]
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    area = st[:, cv2.CC_STAT_AREA].astype(np.int64)
    I = np.bincount(lab[inside], minlength=n)
    Iv = np.bincount(lab[inside_v], minlength=n)
    speck_thr = max(P['speck_min'], P['speck_frac'] * p * wc)
    kind = np.zeros(n, np.int8)     # 0 ngoại lai, 1 của ô (trọn), 2 của ô (cắt seam), 3 đốm, 4 nét kẻ
    kept = np.zeros(n, np.int64)
    for j in range(1, n):
        a = area[j]
        cw_, ch_ = st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]
        if a < speck_thr:
            kind[j] = 3; continue
        if (ch_ >= P['line_len'] * p and cw_ <= P['line_thin'] * wc) or (cw_ >= P['line_len'] * wc and ch_ <= P['line_thin'] * p):
            kind[j] = 4; continue
        if I[j] == 0:
            continue
        if I[j] == a:
            kind[j] = 1; kept[j] = a; continue
        if not (bxa <= cen[j][0] < bxb):
            continue
        kind[j] = 2; kept[j] = Iv[j]
    tot = kept.sum()
    n_tip = 0
    for j in np.nonzero(kind == 2)[0]:
        if kept[j] < P['tip_frac'] * area[j] and kept[j] < P['tip_mass'] * tot:
            kind[j] = 0; kept[j] = 0; n_tip += 1
    own_ids = np.nonzero((kind == 1) | (kind == 2))[0]
    M = np.isin(lab, own_ids) & ink
    M &= np.where(np.isin(lab, np.nonzero(kind == 2)[0]), inside_v, True)
    mass = int(M.sum())
    pw_ = p * wc
    flags = []
    out.update(T=round(T, 1), n_comp=int(n - 1), n_own=int(len(own_ids)), n_cut_comp=int((kind == 2).sum()),
               n_tip=n_tip, n_line=int((kind == 4).sum()), mass=mass, ink_norm=round(mass / pw_, 4))
    if mass < P['blank_frac'] * pw_ or mass == 0:
        flags.append('blank')
    # cut (a): seam đi qua mực của thành phần được giữ
    cutr = 0.0
    for sarr in (top_band, bot_band):
        if sarr is None:
            continue
        cols = np.arange(bxa, bxa + len(sarr))
        ys = np.clip(sarr, 0, Hw - 1)
        on = ink[ys, cols] & np.isin(lab[ys, cols], own_ids)
        # seam trên ranh giới: mực ngay hai phía đều thuộc thành phần giữ
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
        # crop chặt đã xử lý (để đo cờ pipeline)
        tm = P['tight_margin']
        ty0, ty1 = max(0, ay0 - tm), min(Hw, ay1 + tm)
        tx0, tx1 = max(0, ax0 - tm), min(Ww, ax1 + tm)
        foreign = ink & ~M
        fd = cv2.dilate(foreign.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~M
        tg = g.copy()
        tg[fd] = 255
        tight = tg[ty0:ty1, tx0:tx1]
        tall = tight.shape[0] > 1.8 * max(tight.shape[1], 1)
        out['tall'] = bool(tall)
        # ---- khung vuông trên trang
        L = max(E, Wd)
        m = int(math.ceil(P['margin'] * L))
        S = L + 2 * m
        icx, icy = wx0 + (ax0 + ax1) / 2.0, wy0 + (ay0 + ay1) / 2.0
        sx0, sy0 = int(round(icx - S / 2.0)), int(round(icy - S / 2.0))
        out.update(ink_box=(wx0 + ax0, wy0 + ay0, wx0 + ax1, wy0 + ay1), ink_cx=round(icx, 2), ink_cy=round(icy, 2),
                   sq=(sx0, sy0, S), E=E, Wd=Wd)
        # mặt nạ ở toạ độ trang cho khung
        px0, py0, px1, py1 = max(0, sx0), max(0, sy0), min(Wp, sx0 + S), min(Hp, sy0 + S)
        src = SRC[py0:py1, px0:px1]
        gs = G[py0:py1, px0:px1]
        inkp = gs < T
        Mp = np.zeros_like(inkp)
        Fp = np.zeros_like(inkp)
        # chép M/fd của cửa sổ vào khung
        ox0, oy0 = max(px0, wx0), max(py0, wy0)
        ox1, oy1 = min(px1, wx1), min(py1, wy1)
        if ox1 > ox0 and oy1 > oy0:
            Mp[oy0 - py0:oy1 - py0, ox0 - px0:ox1 - px0] = M[oy0 - wy0:oy1 - wy0, ox0 - wx0:ox1 - wx0]
            Fp[oy0 - py0:oy1 - py0, ox0 - px0:ox1 - px0] = fd[oy0 - wy0:oy1 - wy0, ox0 - wx0:ox1 - wx0]
        # mực nằm ngoài cửa sổ phân tích = ngoại lai; đốm TRONG khe giữ nguyên
        inwin = np.zeros_like(inkp)
        if ox1 > ox0 and oy1 > oy0:
            inwin[oy0 - py0:oy1 - py0, ox0 - px0:ox1 - px0] = True
            spk_in = np.zeros_like(inkp)
            spk_in[oy0 - py0:oy1 - py0, ox0 - px0:ox1 - px0] = (np.isin(lab, np.nonzero(kind == 3)[0]) & inside)[oy0 - wy0:oy1 - wy0, ox0 - wx0:ox1 - wx0]
        else:
            spk_in = np.zeros_like(inkp)
        out_ink = inkp & ~inwin
        if out_ink.any():
            Fp |= cv2.dilate(out_ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        Fp &= ~Mp
        Fp &= ~spk_in
        nonink = ~inkp
        if src.ndim == 3:
            bg = np.median(src[nonink], axis=0) if nonink.any() else np.array([255, 255, 255])
            bg = bg.astype(np.uint8)
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
        out['M_page'] = (px0, py0, Mp)   # cho phép đo khe người (không ghi ra đĩa)
    else:
        out['tall'] = False
    out['flags'] = '|'.join(flags)
    out['status'] = 'ok' if mass else 'empty'
    out['_ink_win'] = (wx0, wy0, ink)
    out['_T'] = T
    return out


# ----------------------------------------------------------------------------- crop cũ (luật save_crop)
def old_crop(img, G, bbox, prev_b, next_b, pad=0.12):
    """Dựng lại crop cũ ĐÃ XỬ LÝ + khung trên trang + mặt nạ vùng carve, theo đúng luật save_crop."""
    H, W = G.shape
    ox1, oy1, ox2, oy2 = (int(v) for v in bbox)
    pw, ph = int((ox2 - ox1) * pad), int((oy2 - oy1) * pad)
    x1, y1 = max(0, ox1 - pw), max(0, oy1 - ph)
    x2, y2 = min(W, ox2 + pw), min(H, oy2 + ph)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = crop.copy()
    carved = np.zeros(crop.shape[:2], bool)
    if prev_b is not None or next_b is not None:
        canvas = np.zeros(crop.shape[:2], np.uint8)
        carve_neighbor_ink(canvas, G, x1, y1, x2, y2, (oy1, oy2), prev_b, next_b, bg=1)
        carved = canvas == 1
        crop = carve_neighbor_ink(crop, G, x1, y1, x2, y2, (oy1, oy2), prev_b, next_b)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    tb = tighten_box(gray)
    a, c, b, d = (0, 0, gray.shape[1], gray.shape[0]) if tb is None else tb
    return dict(gray=gray[c:d, a:b], rect=(x1 + a, y1 + c, x1 + b, y1 + d), carved=carved[c:d, a:b],
                win=(x1, y1, x2, y2))
