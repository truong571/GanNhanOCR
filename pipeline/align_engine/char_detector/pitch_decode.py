"""pitch_decode — giải mã hộp chữ theo BƯỚC CỘT với RÀNG BUỘC SỐ LƯỢNG (tuỳ chọn, 2026-09-22).

Bật qua khoá config `books[].box_decoder: pitch` (mặc định "legacy" = hành vi cũ: hộp thô
detector ở det_thr + assign_boxes 3 nhánh). Không sách STT nào khai khoá này nên STT
không đổi byte.

Vì sao (đo 27 trang/sách LVT1883 + KVK1884, scripts/measure/box_ref_eval.py): CenterNet
học STT trên thạch bản có điểm tin cậy thấp (trung vị 0,35) và lệch ±1 hộp/cột ở 35 %
cột (I5 65,2 / 59,2 %): −1 = bỏ sót chữ nhạt, +1 = hộp thừa (số câu in trên đỉnh tầng lọt
vào dải x của cột vì raw_column_boxes không lọc theo y; chữ bắt 2 lần). Luật cũ hoà giải
bằng enforce_count (giữ top-N điểm / bổ đôi hộp cao nhất) rồi _monotone_assign → hộp
midpoint/split, không dùng thông tin BƯỚC LẶP đều của thạch bản (dy_cv ≈ 0,07).

Cách làm — trong MỖI TẦNG-CỘT (thạch bản: tầng trên 6 chữ, tầng dưới 8 chữ; STT: 1 tầng):
  1. Tầng = nhóm chữ kim liên tiếp (hộp kim chia đều liền nhau; khe > 0,5·h_chữ = tầng mới).
     N mỗi tầng = số chữ kim của tầng nếu Σ == n_qn, ngược lại chia n_qn theo chiều cao tầng.
  2. Ứng viên = hộp detector ở ngưỡng THẤP (PITCH_CAND_THR 0,05) có tâm trong cửa sổ
     x = x_range ± x_margin·w, y = [y0 − m·p, y1 + m·p] (p = bước = H_tầng/N)
     + N "ô ảo" cắt theo CHIẾU MỰC NGANG (ink_cut_cells: N−1 khe mực yếu nhất, quy hoạch
     động có phạt lệch bước — KHÔNG chia đều) với điểm âm nhẹ (chỉ được chọn khi thiếu).
  3. Quy hoạch động chọn ĐÚNG N ứng viên theo thứ tự y, tối đa
        Σ điểm − λ_pitch·Σ((Δy − p)/p)² − λ_overlap·Σ chồng_y/p − λ_size·Σ max(0, h/p − 1,5)²
        − λ_border·(lệch ô đầu/cuối so với mép tầng)².
  4. Nguồn hộp từng ô: 'detector' (điểm ≥ det_thr), 'detector_low' (0,05 ≤ điểm < det_thr),
     'ink_cut' (ô ảo). Engine ghi vào labels.csv cột box_source; n_det VẪN là số hộp thô ở
     det_thr trong dải x (như cũ) để I5 không trở thành hằng đúng khi ép N.

Không có hộp GT người: mọi số đo so với Ô THAM CHIẾU tự động (kim tầng + chiếu mực) là proxy;
ô 'ink_cut' trùng tham chiếu theo cấu tạo nên KHÔNG tính là bằng chứng (xem box_ref_eval.py).

Selftest: .venv/bin/python -m pipeline.align_engine.char_detector.pitch_decode --selftest
"""
from __future__ import annotations

import numpy as np

PITCH_CAND_THR = 0.05          # ngưỡng ứng viên (decode top-k của CenterNet vốn ≥ 0,05)
BOX_DECODERS = ("legacy", "pitch")
SRC_DET, SRC_LOW, SRC_INK = "detector", "detector_low", "ink_cut"

DEFAULT_PARAMS = dict(
    lam_pitch=0.6,      # phạt lệch bước giữa 2 ô kề: ((Δy − p)/p)²
    lam_overlap=1.0,    # phạt chồng y giữa 2 ô kề (đơn vị p)
    lam_size=1.0,       # phạt hộp cao bất thường: max(0, h/p − 1,5)²
    lam_border=0.3,     # phạt ô đầu/cuối lệch mép tầng
    virtual_score=-0.05,  # điểm ô ảo (chỉ thắng khi không có ứng viên thật hợp lý)
    y_margin=0.35,      # cửa sổ y của tầng: ± y_margin·p
    max_cands=4,        # giữ tối đa max_cands·N ứng viên thật (điểm cao nhất)
    cut_lam=1.5,        # ink_cut_cells: phạt lệch bước khi chọn khe
    cut_lo=0.55, cut_hi=1.6,   # chiều cao ô cắt ∈ [lo·p, hi·p]
    tier_gap_frac=0.5,  # khe giữa 2 chữ kim > frac·h_chữ → tầng mới
)


# ---------------------------------------------------------------------------
# 1. Tầng
# ---------------------------------------------------------------------------
def group_tiers(chars: list, gap_frac: float = 0.5) -> list[list[int]]:
    """Chỉ số chữ (theo thứ tự y) gom thành tầng: khe y giữa 2 hộp kim kề > gap_frac·h_med
    → tầng mới. Hộp kim chia đều liền nhau nên trong tầng khe ≈ 0."""
    if not chars:
        return []
    order = sorted(range(len(chars)), key=lambda i: (chars[i]["bbox"][1] + chars[i]["bbox"][3]) / 2.0)
    hs = [max(1, chars[i]["bbox"][3] - chars[i]["bbox"][1]) for i in order]
    h_med = float(np.median(hs))
    tiers, cur = [], [order[0]]
    for a, b in zip(order, order[1:]):
        gap = chars[b]["bbox"][1] - chars[a]["bbox"][3]
        if gap > gap_frac * h_med:
            tiers.append(cur); cur = [b]
        else:
            cur.append(b)
    tiers.append(cur)
    return tiers


def tier_counts(n_qn: int, tiers: list[list[int]], chars: list) -> list[int]:
    """N mỗi tầng: số chữ kim của tầng nếu Σ == n_qn; ngược lại chia n_qn theo chiều cao
    tầng (số dư lớn nhất), mỗi tầng ≥ 1."""
    if n_qn <= 0 or not tiers:
        return []
    if sum(len(t) for t in tiers) == n_qn:
        return [len(t) for t in tiers]
    hs = [max(1.0, max(chars[i]["bbox"][3] for i in t) - min(chars[i]["bbox"][1] for i in t)) for t in tiers]
    tot = sum(hs)
    raw = [n_qn * h / tot for h in hs]
    base = [max(1, int(r)) for r in raw]
    while sum(base) > n_qn:                          # quá tay vì ép ≥ 1
        j = max(range(len(base)), key=lambda k: base[k])
        base[j] -= 1
    rem = n_qn - sum(base)
    frac = sorted(range(len(raw)), key=lambda k: raw[k] - int(raw[k]), reverse=True)
    for k in frac[:rem]:
        base[k] += 1
    return base


def tier_box(chars: list, idx: list[int]) -> tuple[int, int, int, int]:
    xs1 = [chars[i]["bbox"][0] for i in idx]; ys1 = [chars[i]["bbox"][1] for i in idx]
    xs2 = [chars[i]["bbox"][2] for i in idx]; ys2 = [chars[i]["bbox"][3] for i in idx]
    return int(min(xs1)), int(min(ys1)), int(max(xs2)), int(max(ys2))


# ---------------------------------------------------------------------------
# 2. Cắt theo chiếu mực ngang (ô ảo / ô tham chiếu)
# ---------------------------------------------------------------------------
def ink_profile(gray: np.ndarray, x1: int, x2: int, y1: int, y2: int, shrink: float = 0.08) -> np.ndarray:
    """Chiếu mực theo hàng trong hộp (x co shrink mỗi bên để bớt mực cột kề), nhị phân Otsu
    (dự phòng 128), mượt cửa sổ ≈ 5 % chiều cao ô, chuẩn hoá về [0, 1]."""
    H, W = gray.shape[:2]
    x1, x2 = max(0, int(x1)), min(W, int(x2)); y1, y2 = max(0, int(y1)), min(H, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.zeros(max(0, y2 - y1), np.float32)
    dx = int((x2 - x1) * shrink)
    sub = gray[y1:y2, x1 + dx:x2 - dx] if x2 - x1 - 2 * dx >= 2 else gray[y1:y2, x1:x2]
    thr = 128.0
    try:
        import cv2
        t, _ = cv2.threshold(np.ascontiguousarray(sub, dtype=np.uint8), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if 20 < t < 235:
            thr = float(t)
    except Exception:
        pass
    ink = (sub.astype(np.float32) < thr).sum(axis=1).astype(np.float32)
    k = max(3, int(0.05 * (y2 - y1)) | 1)
    ink = np.convolve(ink, np.ones(k, np.float32) / k, mode="same")
    m = float(ink.max())
    return ink / m if m > 0 else ink


def ink_cut_cells(gray: np.ndarray, x1: int, x2: int, y1: int, y2: int, n: int,
                  lam: float = 1.5, lo: float = 0.55, hi: float = 1.6) -> list[list[int]]:
    """Cắt hộp [y1, y2) thành ĐÚNG n ô tại n−1 khe mực yếu nhất (quy hoạch động):
        min Σ_k ink[c_k] + lam·Σ_k ((c_k − c_{k−1} − p)/p)²,  c_0 = 0, c_n = H, p = H/n,
        c_k − c_{k−1} ∈ [lo·p, hi·p].
    Không chia đều: vị trí khe theo mực, bước chỉ là ràng buộc mềm. Trả n hộp [x1, ya, x2, yb]."""
    y1, y2 = int(y1), int(y2)
    H = y2 - y1
    if n <= 0 or H < 2:
        return []
    if n == 1:
        return [[int(x1), y1, int(x2), y2]]
    prof = ink_profile(gray, x1, x2, y1, y2)
    if len(prof) != H:
        prof = np.zeros(H, np.float32)
    p = H / n
    dmin, dmax = max(1, int(round(lo * p))), max(2, int(round(hi * p)))
    if dmin * n > H:                                  # hộp quá thấp so với n: chia đều
        cuts = [int(round(k * p)) for k in range(n + 1)]
        return [[int(x1), y1 + cuts[k], int(x2), y1 + cuts[k + 1]] for k in range(n)]
    ds = np.arange(dmin, min(dmax, H) + 1)
    pen = lam * ((ds - p) / p) ** 2
    INF = np.float32(1e9)
    dp = np.full((n + 1, H + 1), INF, np.float32)
    back = np.zeros((n + 1, H + 1), np.int32)
    dp[0, 0] = 0.0
    cut_cost = np.concatenate([prof, [0.0]]).astype(np.float32)   # chi phí cắt tại y (y = H: mép, 0)
    for k in range(1, n + 1):
        prev = dp[k - 1]
        best = np.full(H + 1, INF, np.float32)
        arg = np.zeros(H + 1, np.int32)
        for d, pe in zip(ds, pen):
            cand = np.full(H + 1, INF, np.float32)
            cand[d:] = prev[:H + 1 - d] + pe
            better = cand < best
            best[better] = cand[better]; arg[better] = d
        cost = best + (cut_cost if k < n else 0.0)
        dp[k] = cost; back[k] = arg
    if dp[n, H] >= INF / 2:                               # không có đường hợp lệ: chia đều
        cuts = [int(round(k * p)) for k in range(n + 1)]
    else:
        cuts = [H]
        y = H
        for k in range(n, 0, -1):
            y = y - int(back[k, y]); cuts.append(y)
        cuts = cuts[::-1]
        cuts[0] = 0
    return [[int(x1), y1 + cuts[k], int(x2), y1 + cuts[k + 1]] for k in range(n)]


# ---------------------------------------------------------------------------
# 3. Quy hoạch động chọn đúng N ứng viên
# ---------------------------------------------------------------------------
def _select_n(cands: list[list[float]], n: int, p: float, y0: float, y1: float, P: dict):
    """cands: [x1,y1,x2,y2,score] đã sắp theo tâm y. Trả chỉ số n ứng viên được chọn."""
    M = len(cands)
    if n <= 0 or M < n:
        return None
    cy = np.array([(c[1] + c[3]) / 2.0 for c in cands])
    hh = np.array([max(1.0, c[3] - c[1]) for c in cands])
    sc = np.array([c[4] for c in cands], np.float64)
    node = sc - P["lam_size"] * np.maximum(0.0, hh / p - 1.5) ** 2
    first_pen = P["lam_border"] * ((cy - (y0 + p / 2)) / p) ** 2
    last_pen = P["lam_border"] * ((cy - (y1 - p / 2)) / p) ** 2
    NEG = -1e18
    dp = np.full((n, M), NEG); back = np.full((n, M), -1, np.int64)
    dp[0] = node - first_pen
    for k in range(1, n):
        for j in range(k, M):
            best, bi = NEG, -1
            for i in range(k - 1, j):
                if dp[k - 1, i] <= NEG / 2:
                    continue
                dy = cy[j] - cy[i]
                if dy <= 0.15 * p:                    # gần trùng: không phải 2 chữ
                    continue
                ov = max(0.0, min(cands[i][3], cands[j][3]) - max(cands[i][1], cands[j][1]))
                cost = P["lam_pitch"] * ((dy - p) / p) ** 2 + P["lam_overlap"] * ov / p
                v = dp[k - 1, i] - cost
                if v > best:
                    best, bi = v, i
            if bi >= 0:
                dp[k, j] = best + node[j]; back[k, j] = bi
    fin = dp[n - 1] - last_pen
    j = int(np.argmax(fin))
    if fin[j] <= NEG / 2:
        return None
    sel = [j]
    for k in range(n - 1, 0, -1):
        j = int(back[k, j]); sel.append(j)
    return sel[::-1]


def decode_tier(real: list, x1: int, x2: int, y0: int, y1: int, n: int, gray: np.ndarray,
                det_thr: float, P: dict) -> tuple[list, list, dict]:
    """Một tầng-cột: real = hộp detector [x1,y1,x2,y2,score] (đã lọc theo cửa sổ). Trả
    (n hộp [x1,y1,x2,y2,score], n nguồn, info)."""
    p = max(1.0, (y1 - y0) / max(1, n))
    real = sorted(real, key=lambda b: -b[4])[: max(n, P["max_cands"] * n)]
    virt = [[*c, P["virtual_score"]] for c in ink_cut_cells(gray, x1, x2, y0, y1, n,
                                                           lam=P["cut_lam"], lo=P["cut_lo"], hi=P["cut_hi"])]
    cands = [[float(v) for v in b] for b in real] + [[float(v) for v in b] for b in virt]
    tags = [SRC_DET if b[4] >= det_thr else SRC_LOW for b in real] + [SRC_INK] * len(virt)
    order = sorted(range(len(cands)), key=lambda i: (cands[i][1] + cands[i][3]) / 2.0)
    cands = [cands[i] for i in order]; tags = [tags[i] for i in order]
    sel = _select_n(cands, n, p, y0, y1, P)
    if sel is None:                                   # không thể (n = 0 / không ứng viên)
        sel = list(range(min(n, len(cands))))
    boxes = [cands[i] for i in sel]; srcs = [tags[i] for i in sel]
    # ô ảo lấy bề rộng theo hộp thật đã chọn (crop cùng cỡ)
    reals = [b for b, s in zip(boxes, srcs) if s != SRC_INK]
    if len(reals) >= 2:
        rx1 = float(np.median([b[0] for b in reals])); rx2 = float(np.median([b[2] for b in reals]))
        boxes = [[rx1, b[1], rx2, b[3], b[4]] if s == SRC_INK else b for b, s in zip(boxes, srcs)]
    cys = [(b[1] + b[3]) / 2.0 for b in boxes]
    dev = [abs((cys[k + 1] - cys[k] - p) / p) for k in range(len(cys) - 1)]
    info = {"n": n, "pitch": round(p, 1), "n_real_in_window": len(real),
            "n_real_ge_thr": sum(1 for b in real if b[4] >= det_thr),
            "n_used_det": srcs.count(SRC_DET), "n_used_low": srcs.count(SRC_LOW),
            "n_used_ink": srcs.count(SRC_INK),
            "pitch_dev_max": round(max(dev), 3) if dev else 0.0}
    return boxes, srcs, info


def decode_column(cluster: dict, page_boxes_low: list, gray: np.ndarray, n_qn: int,
                  det_thr: float, x_margin: float = 0.05, params: dict | None = None,
                  tier_n: list[int] | None = None) -> tuple[list, list, dict]:
    """Cột (cluster của engine: x_range + chars kim) → ĐÚNG n_qn hộp theo thứ tự đọc.

    page_boxes_low: hộp cả trang ở ngưỡng PITCH_CAND_THR [(x1,y1,x2,y2,score)].
    gray: ảnh xám cả trang (det._gray). tier_n: số âm QN mỗi tầng (transcriptions len_odd;
    book_layout.expected_tier_counts) — ưu tiên hơn số chữ kim khi số tầng khớp và Σ == n_qn.
    Trả (boxes [x1,y1,x2,y2,score] int-able, sources, info{tiers:[...], n_det_tier}).
    Thứ tự = tầng trên→dưới, trong tầng trên→dưới."""
    P = dict(DEFAULT_PARAMS); P.update(params or {})
    chars = cluster.get("chars") or []
    xr = cluster.get("x_range")
    if n_qn <= 0 or not chars or not xr:
        return [], [], {"tiers": [], "n_det_tier": 0}
    x1c, x2c = int(xr[0]), int(xr[1])
    m = (x2c - x1c) * x_margin
    tiers = group_tiers(chars, P["tier_gap_frac"])
    if tier_n and len(tier_n) == len(tiers) and sum(tier_n) == n_qn and min(tier_n) > 0:
        counts = [int(v) for v in tier_n]
    else:
        counts = tier_counts(n_qn, tiers, chars)
    boxes, srcs, tinfo, n_det_tier = [], [], [], 0
    for idx, n in zip(tiers, counts):
        tx1, ty0, tx2, ty1 = tier_box(chars, idx)
        p = max(1.0, (ty1 - ty0) / max(1, n))
        ylo, yhi = ty0 - P["y_margin"] * p, ty1 + P["y_margin"] * p
        real = [list(b[:5]) for b in page_boxes_low
                if x1c - m <= (b[0] + b[2]) / 2.0 <= x2c + m and ylo <= (b[1] + b[3]) / 2.0 <= yhi
                and b[4] >= PITCH_CAND_THR]
        n_det_tier += sum(1 for b in real if b[4] >= det_thr)
        b, s, inf = decode_tier(real, tx1, tx2, ty0, ty1, n, gray, det_thr, P)
        inf["tier_box"] = [tx1, ty0, tx2, ty1]
        boxes += b; srcs += s; tinfo.append(inf)
    return boxes, srcs, {"tiers": tinfo, "n_det_tier": n_det_tier}


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def _synthetic_column(n: int = 8, W: int = 140, p: int = 150, top: int = 40, gap_ratio: float = 0.22):
    """Ảnh xám cột dọc n chữ (khối đen có lỗ) cách đều p, khe trắng gap_ratio·p."""
    H = top * 2 + n * p
    img = np.full((H, W), 255, np.uint8)
    boxes = []
    for k in range(n):
        y = top + k * p + int(gap_ratio * p / 2)
        h = p - int(gap_ratio * p)
        img[y:y + h, 20:W - 20] = 0
        img[y + h // 3:y + h // 3 + 8, 40:W - 40] = 255      # lỗ ngang trong chữ (⿱ giả)
        boxes.append([20, y, W - 20, y + h])
    return img, boxes


def _chars_from_boxes(boxes, x1=15, x2=125):
    """Hộp kim chia đều: 1 hộp tầng bao n chữ, chia đều theo chiều cao."""
    y0, y1 = boxes[0][1] - 8, boxes[-1][3] + 8
    n = len(boxes)
    h = (y1 - y0) / n
    return [{"char": "x", "y_center": y0 + h * (i + 0.5),
             "bbox": [x1, int(y0 + h * i), x2, int(y0 + h * (i + 1))]} for i in range(n)]


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def selftest() -> int:
    n_pass = n_fail = 0

    def check(name, ok, extra=""):
        nonlocal n_pass, n_fail
        n_pass += int(bool(ok)); n_fail += int(not ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + str(extra)) if extra and not ok else ''}")

    img, gt = _synthetic_column(8)
    chars = _chars_from_boxes(gt)
    # 1. group_tiers: 1 tầng khi hộp kim liền nhau
    check("group_tiers: 8 chữ liền → 1 tầng", group_tiers(chars) == [list(range(8))])
    # 2. group_tiers: 2 tầng khi có khe
    img2, gt2 = _synthetic_column(6)
    chars2 = _chars_from_boxes(gt2)
    off = img.shape[0] + 120
    chars_2t = chars + [{"char": "y", "y_center": c["y_center"] + off,
                         "bbox": [c["bbox"][0], c["bbox"][1] + off, c["bbox"][2], c["bbox"][3] + off]} for c in chars2]
    tiers = group_tiers(chars_2t)
    check("group_tiers: khe lớn → 2 tầng (8, 6)", [len(t) for t in tiers] == [8, 6], tiers)
    # 3. tier_counts: Σ == n_qn giữ nguyên; khác → theo chiều cao
    check("tier_counts: Σ == n_qn → (8, 6)", tier_counts(14, tiers, chars_2t) == [8, 6])
    tc = tier_counts(13, tiers, chars_2t)
    check("tier_counts: n_qn 13 chia theo chiều cao, Σ = 13", sum(tc) == 13 and len(tc) == 2 and min(tc) >= 1, tc)
    # 4. ink_cut_cells: n ô, khe rơi vào vùng trắng giữa chữ, không chia đều
    x1, y0, x2, y1 = 15, gt[0][1] - 8, 125, gt[-1][3] + 8
    cells = ink_cut_cells(img, x1, x2, y0, y1, 8)
    ok = len(cells) == 8 and all(cells[k][3] == cells[k + 1][1] for k in range(7))
    check("ink_cut_cells: 8 ô liền nhau phủ kín hộp", ok and cells[0][1] == y0 and cells[-1][3] == y1, cells)
    cuts = [c[3] for c in cells[:-1]]
    in_gap = all(img[c, 20:120].min() == 255 for c in cuts)
    check("ink_cut_cells: 7 khe đều nằm trên hàng trắng (không cắt vào thân chữ)", in_gap, cuts)
    ious = [_iou(c, g) for c, g in zip(cells, gt)]
    check("ink_cut_cells: IoU với chữ thật ≥ 0,7 mọi ô", min(ious) >= 0.7, [round(v, 2) for v in ious])
    # 5. ink_cut_cells: hộp lệch (kim bao rộng hơn) vẫn cắt vào khe, KHÔNG chia đều
    cells_b = ink_cut_cells(img, x1, x2, y0 - 25, y1 + 40, 8)
    even = [(y0 - 25) + int(round(k * (y1 + 40 - (y0 - 25)) / 8)) for k in range(1, 8)]
    cuts_b = [c[3] for c in cells_b[:-1]]
    check("ink_cut_cells: hộp bao lệch → khe vẫn trên hàng trắng, khác chia đều",
          all(img[c, 20:120].min() == 255 for c in cuts_b) and cuts_b != even, (cuts_b, even))
    # 6. ink_cut_cells: n = 1 và ảnh trống
    check("ink_cut_cells: n=1 → 1 ô = hộp", ink_cut_cells(img, x1, x2, y0, y1, 1) == [[x1, y0, x2, y1]])
    blank = np.full_like(img, 255)
    cb = ink_cut_cells(blank, x1, x2, y0, y1, 8)
    check("ink_cut_cells: ảnh trống → 8 ô gần đều (bước ràng buộc)",
          len(cb) == 8 and max(c[3] - c[1] for c in cb) - min(c[3] - c[1] for c in cb) <= 2, cb)
    # 7. decode_column: detector đủ N hộp đúng → giữ nguyên hộp detector, nguồn 'detector'
    det_boxes = [[*g, 0.4] for g in gt]
    cluster = {"x_range": (15, 125), "chars": chars}
    b, s, info = decode_column(cluster, det_boxes, img, 8, det_thr=0.2, x_margin=0.05)
    check("decode: M == N hộp tốt → 8 hộp 'detector', IoU 1", len(b) == 8 and s == [SRC_DET] * 8
          and all(_iou(x, g) > 0.99 for x, g in zip(b, gt)), s)
    # 8. decode: thiếu 1 hộp (bỏ sót chữ 4) → ô 4 = ink_cut, 7 ô kia vẫn detector, thứ tự đúng
    miss = [x for k, x in enumerate(det_boxes) if k != 3]
    b, s, info = decode_column(cluster, miss, img, 8, det_thr=0.2)
    check("decode: thiếu 1 → 8 hộp, ô 4 'ink_cut', còn lại 'detector', IoU ≥ 0,7",
          len(b) == 8 and s[3] == SRC_INK and s.count(SRC_DET) == 7 and all(_iou(x, g) >= 0.7 for x, g in zip(b, gt)),
          (s, [round(_iou(x, g), 2) for x, g in zip(b, gt)]))
    # 9. decode: thừa 1 hộp trùng (chữ bắt 2 lần, điểm cao) → loại hộp trùng, giữ 8
    dup = det_boxes + [[gt[2][0], gt[2][1] + 6, gt[2][2], gt[2][3] + 6, 0.55]]
    b, s, info = decode_column(cluster, dup, img, 8, det_thr=0.2)
    check("decode: +1 hộp trùng → 8 hộp, không hộp nào chồng nhau > 0,3 IoU",
          len(b) == 8 and all(_iou(b[k], b[k + 1]) < 0.3 for k in range(7)) and s.count(SRC_INK) == 0, s)
    # 10. decode: hộp thừa NGOÀI tầng (số câu in trên đỉnh) → không được chọn
    num = [[40, gt[0][1] - 90, 100, gt[0][1] - 50, 0.5]]
    b, s, info = decode_column(cluster, det_boxes + num, img, 8, det_thr=0.2)
    check("decode: hộp số câu trên đỉnh tầng bị bỏ", len(b) == 8 and all(x[1] >= gt[0][1] - 30 for x in b))
    # 11. decode: hộp cao 2 chữ (dính) điểm cao + thiếu 2 hộp → thay bằng 2 ô ink_cut
    tall = [x for k, x in enumerate(det_boxes) if k not in (5, 6)] + [[20, gt[5][1], 120, gt[6][3], 0.6]]
    b, s, info = decode_column(cluster, tall, img, 8, det_thr=0.2)
    check("decode: hộp dính 2 chữ → không chọn, 2 ô ink_cut, IoU ≥ 0,7",
          len(b) == 8 and s.count(SRC_INK) == 2 and all(_iou(x, g) >= 0.7 for x, g in zip(b, gt)),
          (s, [round(_iou(x, g), 2) for x, g in zip(b, gt)]))
    # 12. decode: hộp điểm thấp (0,08 < det_thr) đúng chỗ → được dùng, nguồn 'detector_low'
    low = [x[:4] + [0.08] if k == 1 else x for k, x in enumerate(det_boxes)]
    b, s, info = decode_column(cluster, low, img, 8, det_thr=0.2)
    check("decode: hộp 0,08 đúng chỗ → 'detector_low', không rơi về ink_cut", s[1] == SRC_LOW and s.count(SRC_INK) == 0, s)
    # 13. decode: không có ứng viên thật → 8 ô ink_cut
    b, s, info = decode_column(cluster, [], img, 8, det_thr=0.2)
    check("decode: không hộp detector → 8 ô ink_cut", len(b) == 8 and s == [SRC_INK] * 8)
    # 14. decode 2 tầng: n_qn 14, hộp đủ ở tầng 1, thiếu 1 ở tầng 2 → 14 hộp, 1 ink_cut
    H2 = off + img2.shape[0]
    page = np.full((H2, 140), 255, np.uint8); page[:img.shape[0]] = img; page[off:off + img2.shape[0]] = img2
    det2 = det_boxes + [[g[0], g[1] + off, g[2], g[3] + off, 0.35] for k, g in enumerate(gt2) if k != 2]
    b, s, info = decode_column({"x_range": (15, 125), "chars": chars_2t}, det2, page, 14, det_thr=0.2)
    check("decode 2 tầng: 14 hộp, đúng 1 ink_cut ở tầng dưới, y tăng dần",
          len(b) == 14 and s.count(SRC_INK) == 1 and s[8 + 2] == SRC_INK
          and all(b[k][1] < b[k + 1][1] for k in range(13)), s)
    check("decode 2 tầng: info n_det_tier đếm hộp ≥ det_thr trong cửa sổ tầng = 13", info["n_det_tier"] == 13, info)
    # 15. decode: n_qn 0 / cluster rỗng → rỗng, không ném
    check("decode: n_qn 0 → rỗng", decode_column(cluster, det_boxes, img, 0, 0.2)[0] == [])
    # 16. decode 2 tầng: tier_n (QN) ưu tiên hơn số chữ kim khi kim đếm sai (7+7 nhưng QN 8+6)
    chars_bad = [dict(c) for c in chars_2t]
    b, s, info = decode_column({"x_range": (15, 125), "chars": chars_bad}, det2, page, 14, det_thr=0.2, tier_n=[8, 6])
    check("decode: tier_n=[8,6] → tầng trên 8 ô, dưới 6 ô", [t["n"] for t in info["tiers"]] == [8, 6], info["tiers"])
    b, s, info = decode_column({"x_range": (15, 125), "chars": chars_bad}, det2, page, 14, det_thr=0.2, tier_n=[9, 6])
    check("decode: tier_n Σ ≠ n_qn → bỏ qua, dùng số chữ kim (8, 6)", [t["n"] for t in info["tiers"]] == [8, 6])
    print(f"pitch_decode selftest: {n_pass} PASS / {n_fail} FAIL")
    return n_fail


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(1 if selftest() else 0)
    print("dùng --selftest")
