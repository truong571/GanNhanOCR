#!/usr/bin/env python3
"""THĂM DÒ X1 (NGOÀI prereg_crop_test), bản 2 sau phản biện độc lập: crop theo hộp Vision với vùng xoá nét láng giềng KHỚP, cho ô L16 mà Vision xác nhận chữ nhãn.

Giao thức đăng ký trước khi chạy: vision/prereg/thamdo_X1_cua_so_vision_L16.json (+ .sha256). Chạy cục bộ, 0 yêu cầu Vision/kim.
  .venv/bin/python vision/crop_test/run_explore_vwindow.py            # chạy thật (ô 'bị cắt' đủ biến thể; ô Vision khớp còn lại 3 biến thể)
  .venv/bin/python vision/crop_test/run_explore_vwindow.py --limit 30 # thử khói (KHÔNG diễn giải số)
Đầu ra: vision/ket_qua/crop_test/X1_L16_vwindow[_khoi].csv.gz, X1_invariants[_khoi].json. Phân tích: analyze_explore_vwindow.py.

Dùng lại ĐÚNG bộ ô và thuộc tính Vision của E1 (rct.load_eval_cells("L16")) và đúng đường cắt của pipeline (crop_lib.cut_window/finish).
Biến thể CHÍNH `vpipe` = chạy NGUYÊN đường cắt của pipeline (pad 0,12; carve với own_y = hộp; tighten) trên HỘP HỢP của ô và hộp Vision ⇒ vùng bảo vệ
của carve và dải xoá nét cùng khớp với chữ thật (làm đúng ý "nới vùng xoá nét láng giềng cho khớp"). Cách đặt own_y = hộp hợp trên cửa sổ hẹp (vw nở 4 %) KHÔNG làm
được việc đó: dải xoá còn 1–2 hàng (< 3 hàng ⇒ seam bị bỏ qua) — phản biện G1; vì vậy vw_off/vw là biến thể THAM CHIẾU, không phải biến thể chính.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crop_lib as cl                                         # noqa: E402
import run_crop_test as rct                                   # noqa: E402  (tái dùng load_eval_cells: cùng bộ ô + thuộc tính Vision như E1)
from pipeline.align_engine.bbox_fix import carve_neighbor_ink   # noqa: E402
from pipeline.align_engine.build_dataset import _paper_bg       # noqa: E402

OUT = cl.VISION / "ket_qua" / "crop_test"
CFG = rct.CFG
PAD = cl.PAD
MARGIN = 0.04                       # như cl.vision_union_window (chỉ dùng cho vw/vw_off/vbox)
TOL = 2                             # px: dung sai quanh hộp Vision khi tính mực "của chính ô"
INK = 128                           # ngưỡng mực, như tighten_box
SEED = 17
ALL_VARIANTS = ["base", "py20", "adapt", "vw", "vw_off", "vpipe", "vpipe_flip", "vpipe_shuf", "vpipe_own", "vbox"]
GUARD_VARIANTS = ["base", "vw", "vpipe"]
METRICS = ["cov", "fo", "nb", "nbA", "foA", "ink", "foO", "er_own", "er_out", "seam_top", "seam_bot", "fh", "fw", "mm"]
NAN = float("nan")


# ───────────────────────── hình học ─────────────────────────
def clamp_box(V, shape):
    H, W = shape[:2]
    return (min(max(V[0], 0.0), W), min(max(V[1], 0.0), H), min(max(V[2], 0.0), W), min(max(V[3], 0.0), H))


def union_box(bbox, V):
    """Hộp hợp (số nguyên) của hộp ô và hộp glyph Vision."""
    return (min(int(bbox[0]), int(np.floor(V[0]))), min(int(bbox[1]), int(np.floor(V[1]))),
            max(int(bbox[2]), int(np.ceil(V[2]))), max(int(bbox[3]), int(np.ceil(V[3]))))


def mirror_box(bbox, V, shape):
    """Đối chứng: lật hộp Vision theo chiều dọc qua tâm bbox ô (cùng kích thước/độ tràn, SAI phía), kẹp trong trang."""
    cy = (bbox[1] + bbox[3]) / 2.0
    return clamp_box((V[0], 2.0 * cy - V[3], V[2], 2.0 * cy - V[1]), shape)


def box_window(shape, V, margin=MARGIN):
    H, W = shape[:2]
    mw, mh = (V[2] - V[0]) * margin, (V[3] - V[1]) * margin
    return (max(0, int(V[0] - mw)), max(0, int(V[1] - mh)), min(W, int(np.ceil(V[2] + mw))), min(H, int(np.ceil(V[3] + mh))))


def _region(b, x1, y1, w, h, tol=0):
    """Hộp b (hệ trang) → (a, b, c, d) trong toạ độ crop, kẹp [0, w]×[0, h]."""
    return (max(int(np.floor(b[0])) - tol - x1, 0), max(int(np.floor(b[1])) - tol - y1, 0),
            min(int(np.ceil(b[2])) + tol - x1, w), min(int(np.ceil(b[3])) + tol - y1, h))


def seam_flags(window, own_y, prev_bbox, next_bbox, page_h):
    """Seam có chạy ở phía trên/dưới không — theo ĐÚNG các điều kiện của carve_neighbor_ink + _seam_boundary (dải ≥ 3 hàng, rộng ≥ 2)."""
    cx1, cy1, cx2, cy2 = window
    oy1, oy2 = own_y
    wide = (cx2 - max(0, cx1)) >= 2
    top = bot = 0
    if prev_bbox is not None and oy1 > cy1:
        ya = max(int((prev_bbox[1] + prev_bbox[3]) / 2), cy1)
        if ya < oy1:
            top = int((min(oy1, page_h) - max(0, ya)) >= 3 and wide)
    if next_bbox is not None and oy2 < cy2:
        yb = min(int((next_bbox[1] + next_bbox[3]) / 2) + 1, cy2)
        if yb > oy2:
            bot = int((min(yb, page_h) - max(0, oy2)) >= 3 and wide)
    return top, bot


# ───────────────────────── cắt crop tham số hoá ─────────────────────────
def cut_window2(pg, window, own_y, prev_bbox, next_bbox, carve=True, erase_boxes=None, keep_box=None, V=None):
    """Như cl.cut_window nhưng own_y (khoảng DỌC được BẢO VỆ khỏi carve) truyền vào thay vì lấy từ bbox. → (crop, crop_o, info).
    erase_boxes (Nx4, hệ trang): tô nền mực nằm trong hộp của KÝ HIỆU KHÁC, trừ vùng keep_box (đúng như truyền vào, không nở thêm).
    V (tuỳ chọn): đo phần mực của chính ô (trong V) và ngoài V mà carve đã xoá (er_own, er_out) + cờ seam."""
    x1, y1, x2, y2 = window
    img, gray_full, orig = pg["img"], pg["gray"], pg["orig"]
    crop = img[y1:y2, x1:x2]
    info = dict(er_own=NAN, er_out=NAN, seam_top=0, seam_bot=0)
    if crop.size == 0:
        return None, None, info
    crop = crop.copy()
    crop_o = orig[y1:y2, x1:x2].copy() if orig is not None else None
    bg_o = _paper_bg(crop_o) if crop_o is not None else None
    oy1, oy2 = own_y
    pre = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < INK) if V is not None else None
    if carve and gray_full is not None and (prev_bbox is not None or next_bbox is not None):
        info["seam_top"], info["seam_bot"] = seam_flags(window, own_y, prev_bbox, next_bbox, gray_full.shape[0])
        crop = carve_neighbor_ink(crop, gray_full, x1, y1, x2, y2, (oy1, oy2), prev_bbox, next_bbox)
        if crop_o is not None:
            crop_o = carve_neighbor_ink(crop_o, gray_full, x1, y1, x2, y2, (oy1, oy2), prev_bbox, next_bbox, bg=bg_o)
    if pre is not None:
        post = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < INK
        h, w = pre.shape
        a, b, c, d = _region(V, x1, y1, w, h)
        in_v = np.zeros((h, w), bool)
        if c > a and d > b:
            in_v[b:d, a:c] = True
        erased = pre & ~post
        info["er_own"] = int((erased & in_v).sum()) / max(int((pre & in_v).sum()), 1)
        info["er_out"] = int((erased & ~in_v).sum()) / max(int((pre & ~in_v).sum()), 1)
    if erase_boxes is not None and len(erase_boxes):
        h, w = crop.shape[:2]
        m = np.zeros((h, w), bool)
        for bx in erase_boxes:
            a, b, c, d = _region(bx, x1, y1, w, h)
            if c > a and d > b:
                m[b:d, a:c] = True
        if keep_box is not None:
            a, b, c, d = _region(keep_box, x1, y1, w, h)
            if c > a and d > b:
                m[b:d, a:c] = False
        crop[m] = 255
        if crop_o is not None:
            crop_o[m] = bg_o
    return crop, crop_o, info


def diagnostics(crop, crop_o, window, V, others, gray_full):
    """Trên crop ĐÃ carve/xoá (trước tighten), mực = xám < 128 trên TRANG đã xử lý.
    cov = mực của chính ô còn giữ / mực của chính ô trên trang (mực trong V); fo = tỉ lệ mực crop NGOÀI V (dung sai TOL); nb = tỉ lệ mực crop trong hộp ký hiệu CJK khác
    và ngoài V; nbA/foA = cùng tử số nhưng chia cho mực của chính ô (số tuyệt đối, không bị pha loãng khi cửa sổ nới); foO = tỉ lệ điểm tối (Otsu theo cửa sổ) NGOÀI V±TOL trên crop điểm ảnh GỐC TRƯỚC tighten."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw = gray < INK
    x1, y1 = window[0], window[1]
    h, w = bw.shape
    tot = int(bw.sum())
    H, W = gray_full.shape
    px0, py0, px1, py1 = max(0, int(np.floor(V[0]))), max(0, int(np.floor(V[1]))), min(W, int(np.ceil(V[2]))), min(H, int(np.ceil(V[3])))
    tot_own = int((gray_full[py0:py1, px0:px1] < INK).sum())
    a, b, c, d = _region(V, x1, y1, w, h)
    kept_own = int(bw[b:d, a:c].sum()) if (c > a and d > b) else 0
    a, b, c, d = _region(V, x1, y1, w, h, TOL)
    vtol = np.zeros((h, w), bool)
    if c > a and d > b:
        vtol[b:d, a:c] = True
    in_vtol = int((bw & vtol).sum())
    om = np.zeros((h, w), bool)
    for bx in others:
        aa, bb, cc, dd = _region(bx, x1, y1, w, h)
        if cc > aa and dd > bb:
            om[bb:dd, aa:cc] = True
    nb_ink = int((bw & om & ~vtol).sum())
    out = dict(cov=kept_own / tot_own if tot_own >= 8 else NAN,
               fo=(tot - in_vtol) / tot if tot > 0 else NAN,
               nb=nb_ink / tot if tot > 0 else NAN,
               nbA=nb_ink / tot_own if tot_own >= 8 else NAN,
               foA=(tot - in_vtol) / tot_own if tot_own >= 8 else NAN,
               ink=float(tot), foO=NAN)
    if crop_o is not None:
        g = cv2.cvtColor(crop_o, cv2.COLOR_BGR2GRAY) if crop_o.ndim == 3 else crop_o
        thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dk = g < thr
        n = int(dk.sum())
        out["foO"] = float((dk & ~vtol).sum()) / n if n > 0 else NAN
    return out


def framing(g):
    """Khung crop giao nộp (sau tighten): cao, rộng, lề nhỏ nhất từ hộp mực tới mép."""
    if g is None:
        return dict(fh=NAN, fw=NAN, mm=NAN)
    bw = g < INK
    ys, xs = np.where(bw.any(1))[0], np.where(bw.any(0))[0]
    if len(ys) == 0 or len(xs) == 0:
        return dict(fh=float(g.shape[0]), fw=float(g.shape[1]), mm=NAN)
    return dict(fh=float(g.shape[0]), fw=float(g.shape[1]), mm=float(min(ys.min(), g.shape[0] - 1 - ys.max(), xs.min(), g.shape[1] - 1 - xs.max())))


def _empty():
    return {k: NAN for k in METRICS}


def build(pg, bbox, V, v, pv, nx, others, alt):
    """→ (ảnh xám giao nộp | None, dict số đo). alt = {'vpipe_flip': hộp, 'vpipe_shuf': hộp} (hộp thay cho V ở hai đối chứng)."""
    shape = pg["img"].shape
    own0 = (int(bbox[1]), int(bbox[3]))
    carve, erase, keep = True, None, None
    if v == "adapt":
        py = PAD
        while True:
            w = cl._window(shape, bbox, PAD, py)
            crop, crop_o, info = cut_window2(pg, w, own0, pv, nx, V=V)
            if crop is None:
                return None, _empty()
            top, bot = cl.edge_touch(crop, w, shape)
            if (not (top or bot)) or py + 1e-9 >= 0.45:
                break
            py = round(py + 0.06, 4)
        own = own0
    else:
        if v == "base":
            w, own = cl._window(shape, bbox, PAD, PAD), own0
        elif v == "py20":
            w, own = cl._window(shape, bbox, PAD, 0.20), own0
        elif v in ("vw", "vw_off"):
            w, own, carve = cl.vision_union_window(pg, bbox, V, margin=MARGIN), own0, v == "vw"
        elif v in ("vpipe", "vpipe_own", "vpipe_flip", "vpipe_shuf"):
            Vx = alt[v] if v in ("vpipe_flip", "vpipe_shuf") else V
            ub = union_box(bbox, Vx)
            w, own = cl._window(shape, ub, PAD, PAD), (ub[1], ub[3])
            if v == "vpipe_own":
                erase, keep = others, (V[0] - TOL, V[1] - TOL, V[2] + TOL, V[3] + TOL)
        elif v == "vbox":
            w, own, carve = box_window(shape, V), own0, False
        else:
            raise ValueError(v)
        crop, crop_o, info = cut_window2(pg, w, own, pv, nx, carve=carve, erase_boxes=erase, keep_box=keep, V=V)
        if crop is None:
            return None, _empty()
    g = cl.finish(crop, crop_o)
    m = diagnostics(crop, crop_o, w, V, others, pg["gray"])
    m.update(info)
    m.update(framing(g))
    m["_win"] = w
    return g, m


# ───────────────────────── Vision ─────────────────────────
def page_boxes(page):
    """Hộp của MỌI ký hiệu CJK/PUA Vision trên trang (cùng bộ lọc với analyse_page khi ghép) — dùng cho 'ký hiệu khác'."""
    f = cl.VISION / "cache" / CFG / "pages" / "L16" / f"{page}.json"
    pj = json.loads(f.read_text(encoding="utf-8"))
    assert pj.get("cols", ["ch", "conf", "x0", "y0", "x1", "y1", "ang"])[:6] == ["ch", "conf", "x0", "y0", "x1", "y1"], "thứ tự cột ký hiệu Vision khác dự kiến"
    syms = [s for s in pj["sym"] if rct.vc.is_cjk(s[0])]
    return np.array([[s[2], s[3], s[4], s[5]] for s in syms], float).reshape(-1, 4)


def own_index(boxes, V):
    if len(boxes) == 0:
        return -1
    d = np.abs(boxes - np.array(V, float)[None, :]).max(1)
    j = int(d.argmin())
    return j if d[j] < 1e-6 else -1


def own_overlap(boxes, j, V):
    """max trên ký hiệu khác của (diện tích giao với V) / diện tích V."""
    if j < 0 or len(boxes) <= 1:
        return 0.0 if j >= 0 else NAN
    o = np.delete(boxes, j, axis=0)
    ix = np.clip(np.minimum(o[:, 2], V[2]) - np.maximum(o[:, 0], V[0]), 0, None)
    iy = np.clip(np.minimum(o[:, 3], V[3]) - np.maximum(o[:, 1], V[1]), 0, None)
    return float((ix * iy).max() / max((V[2] - V[0]) * (V[3] - V[1]), 1e-6))


def overflow(shape, bbox, V):
    """Phần hộp Vision tràn ra ngoài cửa sổ chuẩn (bbox ± 0,12), theo tỉ lệ cạnh hộp Vision: (dọc, ngang)."""
    w0 = cl._window(shape, bbox, PAD, PAD)
    top, bot = max(0.0, w0[1] - V[1]), max(0.0, V[3] - w0[3])
    lef, rig = max(0.0, w0[0] - V[0]), max(0.0, V[2] - w0[2])
    vh, vw = max(V[3] - V[1], 1.0), max(V[2] - V[0], 1.0)
    return (top + bot) / vh, (lef + rig) / vw


def make_shuffle(run, ov_tot, seed=SEED, return_src=False):
    """Đối chứng 'xáo': hình học V so với bbox (độ lệch/kích thước, chuẩn hoá theo bbox) lấy từ ô BỊ CẮT KHÁC cùng tam phân vị tổng độ tràn; hoán vị vòng quanh sau khi xáo
    ⇒ không ô nào nhận lại hình học của chính nó. → dict {chỉ số hàng trong run: hộp V' (hệ trang, chưa kẹp)}."""
    idx = [i for i in range(len(run)) if run.at[i, "clipped"]]
    if len(idx) < 2:
        return ({}, {}) if return_src else {}
    rel = {}
    for i in idx:
        r = run.loc[i]
        bw, bh = max(float(r.x1 - r.x0), 1.0), max(float(r.y1 - r.y0), 1.0)
        rel[i] = ((r.vx0 - r.x0) / bw, (r.vy0 - r.y0) / bh, (r.vx1 - r.x0) / bw, (r.vy1 - r.y0) / bh)
    nbin = 3 if len(idx) >= 6 else 1
    q = pd.qcut(pd.Series([ov_tot[i] for i in idx], index=idx).rank(method="first"), nbin, labels=False)
    rng = np.random.default_rng(seed)
    src = {}
    pool = []
    for b in sorted(set(q)):
        mem = [i for i in idx if q[i] == b]
        if len(mem) < 2:
            pool += mem
            continue
        p = list(rng.permutation(mem))
        for k, i in enumerate(p):
            src[i] = p[(k + 1) % len(p)]
    for i in pool:                                              # tầng chỉ có 1 ô: lấy ô bất kỳ khác
        src[i] = idx[(idx.index(i) + 1) % len(idx)]
    out = {}
    for i in idx:
        r, s = run.loc[i], rel[src[i]]
        bw, bh = max(float(r.x1 - r.x0), 1.0), max(float(r.y1 - r.y0), 1.0)
        out[i] = (r.x0 + s[0] * bw, r.y0 + s[1] * bh, r.x0 + s[2] * bw, r.y0 + s[3] * bh)
    return (out, src) if return_src else out


def compare_e1(rs, rt, rm, cs, ct, cm, att):
    """So điểm một biến thể với E1 trên các ô biến thể ĐƯỢC DỰNG ở lần chạy này (att). Lệch top1 chỉ đếm khi CẢ HAI t có giá trị (cặp NaN không phải lệch);
    lệch top1 'giải thích được' nếu |m_E1| < 2·max|Δs| (gần hoà)."""
    ok = att & ~np.isnan(rs) & ~np.isnan(cs)
    ds = float(np.abs(rs[ok] - cs[ok]).max()) if ok.any() else None
    both = ok & ~np.isnan(rt) & ~np.isnan(ct)
    mism = both & (rt != ct)
    expl = mism & (np.abs(rm) < 2 * (ds or 0.0) + 1e-9)
    dm_ok = both & ~np.isnan(rm) & ~np.isnan(cm)
    return dict(so_o=int(ok.sum()), max_abs_ds=ds, max_abs_dm=float(np.abs(rm[dm_ok] - cm[dm_ok]).max()) if dm_ok.any() else None,
                top1_lech=int(mism.sum()), top1_lech_giai_thich_duoc=int(expl.sum()), mau_nan_t_giong=bool((np.isnan(rt[att]) == np.isnan(ct[att])).all()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="thử khói: tối đa N ô mỗi tầng (bị cắt / còn lại); KHÔNG diễn giải số")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_khoi" if a.limit else ""
    t00 = time.time()

    def log(*m):
        print(*m, flush=True)

    cells_all = rct.load_eval_cells("L16").reset_index(drop=True)
    cells_all["e1_row"] = np.arange(len(cells_all))                      # vị trí hàng trong E1_L16.csv.gz (E1 không lấy mẫu)
    ag = cells_all[(cells_all["status"] == "agree") & cells_all["vx0"].notna()].copy()
    ag["clipped"] = (ag["harm"] == 1)
    clipped, other = ag[ag["clipped"]], ag[~ag["clipped"]]
    if a.limit:
        clipped, other = clipped.sample(min(a.limit, len(clipped)), random_state=1), other.sample(min(a.limit, len(other)), random_state=1)
    log(f"L16: {len(cells_all)} ô GOLD có nhãn người; Vision khớp {len(ag)}; bị cắt (agree & hại=1) {int(ag['clipped'].sum())}; chạy: bị cắt {len(clipped)}, còn lại {len(other)}")
    run = pd.concat([clipped, other]).sort_values(["page", "column", "y0"]).reset_index(drop=True)
    n = len(run)
    inv = dict(n_bi_cat=int(len(clipped)), n_con_lai=int(len(other)), own_khong_tim_thay=0, cua_so_kiem_tra=0, cua_so_chua_hop_V=0, trang_lech_kich_thuoc=0, trang_kiem_tra=0,
               pixel_kiem_tra=0, pixel_base_khop=0, pixel_vpipe_kiem_tra=0, pixel_vpipe_khop=0, seam_khong_nhat_quan=0, ring_trung_vi=None)
    # ── lượt 1 (không dựng crop): kích thước trang, hộp Vision của chính ô, độ tràn, chồng lấn
    pshape, boxes_of = {}, {}
    for page in sorted(run["page"].unique()):
        W_, H_ = Image.open(cl.REPO / "prepared" / "LucVanTien1916" / "pages" / f"{page}.png").size
        Wd, Hd = Image.open(cl.REPO / "prepared" / "LucVanTien1916" / "pages_denoised" / f"{page}.png").size
        inv["trang_kiem_tra"] += 1
        inv["trang_lech_kich_thuoc"] += int((W_, H_) != (Wd, Hd))
        pshape[page] = (H_, W_)
        boxes_of[page] = page_boxes(page)
    ov = np.full((n, 2), NAN)
    ovl = np.full(n, NAN)
    own_j = np.full(n, -1)
    for i, r in run.iterrows():
        V = (float(r.vx0), float(r.vy0), float(r.vx1), float(r.vy1))
        own_j[i] = own_index(boxes_of[r.page], V)
        ovl[i] = own_overlap(boxes_of[r.page], own_j[i], V)
        ov[i] = overflow(pshape[r.page], (int(r.x0), int(r.y0), int(r.x1), int(r.y1)), V)
    inv["own_khong_tim_thay"] = int((own_j < 0).sum())
    inv["n_eval"] = int(len(cells_all))
    shuf = make_shuffle(run, ov[:, 0] + ov[:, 1])
    # ── lượt 2: dựng crop từng ô × biến thể
    rng = np.random.default_rng(SEED)
    chk = set(int(x) for x in rng.choice(n, min(150, n), replace=False))                          # base == make_crop: 150 ô bất kỳ
    clip_idx = [i for i in range(n) if run.at[i, "clipped"]]
    chk_v = set(int(x) for x in rng.choice(clip_idx, min(150, len(clip_idx)), replace=False)) if clip_idx else set()   # vpipe == make_crop(U): 150 ô bị cắt
    pc = cl.PageCache("L16")
    grays = {v: [None] * n for v in ALL_VARIANTS}
    M = {v: {k: np.full(n, NAN) for k in METRICS} for v in ALL_VARIANTS}
    chg = {v: np.full(n, NAN) for v in ALL_VARIANTS}
    ring = np.full(n, NAN)
    t0 = time.time()
    for n_pg, (page, g) in enumerate(run.groupby("page", sort=True)):
        pg = pc.get(page)
        shape = pg["img"].shape
        assert tuple(shape[:2]) == pshape[page], f"hình dạng trang {page} lệch với tiêu đề tệp"
        boxes = boxes_of[page]
        for i, r in g.iterrows():
            bbox = (int(r.x0), int(r.y0), int(r.x1), int(r.y1))
            V = clamp_box((float(r.vx0), float(r.vy0), float(r.vx1), float(r.vy1)), shape)
            j = own_j[i]
            others = np.delete(boxes, j, axis=0) if j >= 0 else boxes
            alt = {"vpipe_flip": mirror_box(bbox, V, shape)}
            if i in shuf:
                alt["vpipe_shuf"] = clamp_box(shuf[i], shape)
            gf = pg["gray"]
            x0_, y0_, x1_, y1_ = int(np.floor(V[0])), int(np.floor(V[1])), int(np.ceil(V[2])), int(np.ceil(V[3]))
            in_v = int((gf[y0_:y1_, x0_:x1_] < INK).sum())
            in_v3 = int((gf[max(0, y0_ - 3):y1_ + 3, max(0, x0_ - 3):x1_ + 3] < INK).sum())
            ring[i] = (in_v3 - in_v) / max(in_v, 1)
            variants = ALL_VARIANTS if r["clipped"] else GUARD_VARIANTS
            base_key = None
            for v in variants:
                if v in ("vpipe_flip", "vpipe_shuf") and v not in alt:
                    continue
                g_, m = build(pg, bbox, V, v, r.prev_bbox, r.next_bbox, others, alt)
                grays[v][i] = g_
                if v in ("vw", "vpipe", "vbox") and "_win" in m:                       # bất biến: cửa sổ THẬT dùng cho biến thể chứa V (số nguyên, kẹp biên trang)
                    wv_, lo = m["_win"], (int(np.floor(V[0])), int(np.floor(V[1])), int(np.ceil(V[2])), int(np.ceil(V[3])))
                    inv["cua_so_kiem_tra"] += 1
                    inv["cua_so_chua_hop_V"] += int(not (wv_[0] <= lo[0] and wv_[1] <= lo[1] and wv_[2] >= lo[2] and wv_[3] >= lo[3]))
                for k in METRICS:
                    M[v][k][i] = m.get(k, NAN)
                key = None if g_ is None else (g_.shape, g_.tobytes())
                if v == "base":
                    base_key = key
                if key is not None and base_key is not None:
                    chg[v][i] = float(key != base_key)
            # bất biến: đường mặc định tái lập đúng crop_lib.make_crop; vpipe == make_crop trên hộp hợp
            if i in chk:
                inv["pixel_kiem_tra"] += 1
                ref = cl.make_crop(pg, bbox, r.prev_bbox, r.next_bbox)
                inv["pixel_base_khop"] += int(ref is not None and grays["base"][i] is not None and ref.shape == grays["base"][i].shape and bool((ref == grays["base"][i]).all()))
            if i in chk_v:
                inv["pixel_vpipe_kiem_tra"] += 1
                ref2 = cl.make_crop(pg, union_box(bbox, V), r.prev_bbox, r.next_bbox)
                inv["pixel_vpipe_khop"] += int(ref2 is not None and grays["vpipe"][i] is not None and ref2.shape == grays["vpipe"][i].shape and bool((ref2 == grays["vpipe"][i]).all()))
            # bất biến: mực bị carve xoá ⇒ seam đã chạy
            for v in variants:
                eo, ou, st, sb = M[v]["er_own"][i], M[v]["er_out"][i], M[v]["seam_top"][i], M[v]["seam_bot"][i]
                if (np.nan_to_num(eo) > 0 or np.nan_to_num(ou) > 0) and not (st or sb):
                    inv["seam_khong_nhat_quan"] += 1
        if (n_pg + 1) % 25 == 0:
            log(f"  {n_pg + 1} trang, {time.time() - t0:.0f}s")
    inv["ring_trung_vi"] = float(np.nanmedian(ring))
    log(f"dựng crop xong {time.time() - t0:.0f}s; nhúng + chấm điểm…")
    S = cl.Scorer()
    labels, syls = run["Leval"].tolist(), run["syllable"].tolist()
    S.prepare(labels, syls)
    res = run[["page", "column", "syl_idx", "e1_row", "tier", "label", "Leval", "syllable", "status", "harm", "dy_pitch", "vconf", "clipped"]].copy()
    res["label_eq"] = (res["label"].astype(str) == res["Leval"].astype(str)).astype(int)
    res["bw"], res["bh"] = (run["x1"] - run["x0"]).values, (run["y1"] - run["y0"]).values
    res["vw_"], res["vh_"] = (run["vx1"] - run["vx0"]).values, (run["vy1"] - run["vy0"]).values
    res["ov_v"], res["ov_h"], res["own_overlap"], res["ring"] = ov[:, 0], ov[:, 1], ovl, ring
    extra = {}
    for v in ALL_VARIANTS:
        idx = [i for i, g_ in enumerate(grays[v]) if g_ is not None]
        E = S.embed([grays[v][i] for i in idx]) if idx else np.zeros((0, 512), np.float32)
        full = np.zeros((n, E.shape[1] if len(idx) else 512), np.float32)
        have = np.zeros(n, bool)
        for k, i in enumerate(idx):
            full[i] = E[k]
            have[i] = True
        s, m, t = S.metrics(full, labels, syls)
        if v == "base":
            res["rival_ok"] = (~np.isnan(m)).astype(int)                 # chỉ phụ thuộc nhãn/phông/đối thủ, KHÔNG phụ thuộc crop (vector 0 cho m=0 khi có đối thủ)
        for arr in (s, m, t):
            arr[~have] = np.nan
        extra[f"have_{v}"] = have.astype(int)
        extra[f"s_{v}"], extra[f"m_{v}"], extra[f"t_{v}"] = s, m, t
        extra[f"chg_{v}"] = chg[v]
        for k in METRICS:
            extra[f"{k}_{v}"] = M[v][k]
        tv = t[res["clipped"].values & have]
        log(f"  {v:11s} {int(have.sum())}/{n} crop; top1 (ô bị cắt)={np.nanmean(tv) * 100:.2f} %" if len(tv) else f"  {v:11s} {int(have.sum())}/{n} crop")
    res = pd.concat([res, pd.DataFrame(extra, index=res.index)], axis=1)
    # ghi điểm TRƯỚC khối bất biến E1 (một lỗi ở đó không được làm mất kết quả)
    res.to_csv(OUT / f"X1_L16_vwindow{suffix}.csv.gz", index=False)
    f1 = OUT / "E1_L16.csv.gz"
    try:
        if not f1.exists():
            inv["E1_thieu_tep"] = True
        else:
            e1 = pd.read_csv(f1, low_memory=False)
            rows = res["e1_row"].values
            inv["E1_cung_o"] = bool((e1.loc[rows, "page"].to_numpy() == res["page"].to_numpy()).all() and (e1.loc[rows, "column"].to_numpy() == res["column"].to_numpy()).all()
                                    and (e1.loc[rows, "Leval"].astype(str).to_numpy() == res["Leval"].astype(str).to_numpy()).all())
            for v in ("base", "py20", "adapt"):
                rs, rt, rm = (e1.loc[rows, f"{c}_{v}"].to_numpy(float) for c in ("s", "t", "m"))
                cs, ct, cm = (res[f"{c}_{v}"].to_numpy(float) for c in ("s", "t", "m"))
                inv[f"E1_{v}"] = compare_e1(rs, rt, rm, cs, ct, cm, res[f"have_{v}"].to_numpy() == 1)
    except Exception as e:                                                  # noqa: BLE001
        inv["E1_loi"] = f"{type(e).__name__}: {e}"
    inv["thoi_gian_s"] = round(time.time() - t00)
    (OUT / f"X1_invariants{suffix}.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(json.dumps(inv, ensure_ascii=False, default=float))
    log("xong")
    return 0


if __name__ == "__main__":
    sys.exit(main())
