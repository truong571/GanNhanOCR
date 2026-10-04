#!/usr/bin/env python3
"""Kiểm thử OFFLINE mã thăm dò X1 (vision/crop_test/run_explore_vwindow.py) bằng trang tổng hợp có đáp án biết trước (0 yêu cầu, không nạp bộ mã hoá).
  .venv/bin/python vision/tests/test_explore_vwindow.py"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CT = HERE.parent / "crop_test"
sys.path.insert(0, str(CT))
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


def load():
    spec = importlib.util.spec_from_file_location("rx", CT / "run_explore_vwindow.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["rx"] = m
    spec.loader.exec_module(m)
    return m


def page(w=240, h=400, rects=(), noise_seed=None):
    """Trang trắng + hình chữ nhật đen (giả chữ). rects: (x0,y0,x1,y1)."""
    img = np.full((h, w, 3), 255, np.uint8)
    for x0, y0, x1, y1 in rects:
        img[y0:y1, x0:x1] = 0
    orig = img.copy()
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        orig = np.clip(orig.astype(int) + rng.integers(-6, 7, orig.shape), 0, 255).astype(np.uint8)
    return dict(img=img, gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), orig=orig)


def main():
    rx = load()
    cl = rx.cl
    from pipeline.align_engine import bbox_fix as bf

    # ── T1: cut_window2 mặc định == crop_lib.cut_window (cả crop xử lý lẫn crop gốc)
    rects = [(60, 40, 120, 100), (60, 112, 120, 170), (60, 182, 120, 240)]
    pg = page(rects=rects, noise_seed=1)
    bbox, prev, nxt = (60, 112, 120, 170), (60, 40, 120, 100), (60, 182, 120, 240)
    w = cl._window(pg["img"].shape, bbox, cl.PAD, cl.PAD)
    a_c, a_o = cl.cut_window(pg, bbox, w, prev, nxt)
    b_c, b_o, _ = rx.cut_window2(pg, w, (bbox[1], bbox[3]), prev, nxt)
    check("T1.cut_window2_mac_dinh_giong_cut_window", np.array_equal(a_c, b_c) and np.array_equal(a_o, b_o), f"crop {a_c.shape}")

    # ── T2: vpipe == make_crop trên hộp hợp
    V = (58.0, 100.0, 122.0, 172.0)
    ub = rx.union_box(bbox, V)
    ref = cl.make_crop(pg, ub, prev, nxt)
    wv = cl._window(pg["img"].shape, ub, cl.PAD, cl.PAD)
    crop, crop_o, _ = rx.cut_window2(pg, wv, (ub[1], ub[3]), prev, nxt, V=V)
    mine = cl.finish(crop, crop_o)
    check("T2.vpipe_giong_make_crop_tren_hop_hop", ref is not None and ref.shape == mine.shape and bool((ref == mine).all()), f"union {ub}")

    # ── T3: seam_flags khớp việc _seam_boundary thật sự trả seam (200 hình học ngẫu nhiên, từng phía riêng)
    calls = []
    orig_sb = bf._seam_boundary

    def spy(gray_full, x1, x2, ya, yb):
        r = orig_sb(gray_full, x1, x2, ya, yb)
        calls.append(r is not None)
        return r
    bf._seam_boundary = spy
    rng = np.random.default_rng(5)
    H, W = 400, 240
    bad_top = bad_bot = 0
    try:
        for _ in range(200):
            gray_full = np.where(rng.random((H, W)) < 0.15, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
            cx1, cx2 = int(rng.integers(0, 60)), int(rng.integers(120, W))
            cy1 = int(rng.integers(0, 150))
            cy2 = int(rng.integers(cy1 + 60, H))
            oy1 = int(rng.integers(cy1 - 5, cy1 + 60))
            oy2 = int(rng.integers(max(oy1 + 5, cy1 + 10), cy2 + 8))
            pbox = (0, int(rng.integers(0, 120)), 10, int(rng.integers(120, 160)))
            nbox = (0, int(rng.integers(200, 300)), 10, int(rng.integers(300, H)))
            for side in ("top", "bot"):
                calls.clear()
                crop = img[cy1:cy2, cx1:cx2].copy()
                if crop.size == 0:
                    continue
                pv, nx = (pbox, None) if side == "top" else (None, nbox)
                bf.carve_neighbor_ink(crop, gray_full, cx1, cy1, cx2, cy2, (oy1, oy2), pv, nx)
                actual = int(any(calls))
                top, bot = rx.seam_flags((cx1, cy1, cx2, cy2), (oy1, oy2), pv, nx, H)
                if side == "top" and top != actual:
                    bad_top += 1
                if side == "bot" and bot != actual:
                    bad_bot += 1
    finally:
        bf._seam_boundary = orig_sb
    check("T3.seam_flags_khop_carve_that", bad_top == 0 and bad_bot == 0, f"lệch trên={bad_top} dưới={bad_bot}")

    # ── T4: hình học hộp
    sh = (400, 240, 3)
    Vc = rx.clamp_box((-5.0, -2.0, 300.0, 410.0), sh)
    ubx = rx.union_box((10, 20, 50, 80), (8.4, 30.0, 52.2, 90.9))
    mb = rx.mirror_box((10, 100, 50, 160), (8.0, 90.0, 52.0, 170.0), sh)
    mb2 = rx.mirror_box((10, 100, 50, 160), mb, sh)
    check("T4a.clamp_va_union", Vc == (0.0, 0.0, 240.0, 400.0) and ubx == (8, 20, 53, 91), f"{Vc} {ubx}")
    check("T4b.lat_giu_cao_va_khu_thanh_chinh_no", abs((mb[3] - mb[1]) - 80.0) < 1e-9 and mb2 == (8.0, 90.0, 52.0, 170.0), f"{mb} {mb2}")

    # ── T5: diagnostics theo số đếm tay: chữ chính (40x50) trong V, chữ láng giềng (40x30) nằm ngoài V nhưng trong cửa sổ
    pg5 = page(rects=[(80, 100, 120, 150), (80, 160, 120, 190)])
    V5 = (80.0, 100.0, 120.0, 150.0)
    win = (70, 90, 130, 200)
    crop5, crop_o5, _ = rx.cut_window2(pg5, win, (100, 150), None, None, V=V5)
    dg = rx.diagnostics(crop5, crop_o5, win, V5, np.array([[80.0, 160.0, 120.0, 190.0]]), pg5["gray"])
    own, nbr = 40 * 50, 40 * 30
    check("T5.so_do_dung_so_dem_tay", dg["cov"] == 1.0 and abs(dg["fo"] - nbr / (own + nbr)) < 1e-9 and abs(dg["nb"] - nbr / (own + nbr)) < 1e-9
          and abs(dg["nbA"] - nbr / own) < 1e-9 and abs(dg["foA"] - nbr / own) < 1e-9 and dg["ink"] == own + nbr, f"{dg}")

    # ── T6: carve cắt vào chính chữ khi hộp quá thấp (own_y=bbox) và KHÔNG cắt khi bảo vệ cả V; dải xoá thu hẹp ở cửa sổ hẹp
    # chữ trước 40..80; chữ chính 100..160 gồm nét ngang đầu (100..106, rộng), nét sổ (107..159, hẹp) và nét ngang giữa — hàng 107–109 ít mực nên seam đi qua đó và xoá nét đầu
    pg6 = page(w=200, h=300, rects=[(60, 40, 120, 80), (60, 100, 120, 107), (85, 107, 95, 160), (60, 125, 120, 129)])
    bbox6, prev6 = (60, 110, 120, 160), (60, 40, 120, 80)                         # bbox quá thấp: bỏ 10 hàng trên của chữ
    V6 = (60.0, 100.0, 120.0, 160.0)
    w6 = cl._window(pg6["img"].shape, bbox6, cl.PAD, cl.PAD)
    _, _, i_bbox = rx.cut_window2(pg6, w6, (bbox6[1], bbox6[3]), prev6, None, V=V6)
    _, _, i_prot = rx.cut_window2(pg6, w6, (100, 160), prev6, None, V=V6)
    wide = rx.union_box(bbox6, V6)
    w7 = cl._window(pg6["img"].shape, wide, cl.PAD, cl.PAD)
    _, _, i_pipe = rx.cut_window2(pg6, w7, (wide[1], wide[3]), prev6, None, V=V6)
    check("T6a.carve_cat_vao_chu_khi_own_y_la_bbox_thap", i_bbox["er_own"] > 0.15 and i_bbox["seam_top"] == 1, f"er_own={i_bbox['er_own']:.3f} seam_top={i_bbox['seam_top']}")
    check("T6b.bao_ve_ca_V_trong_cua_so_cu_thi_khong_con_dai_xoa", i_prot["er_own"] == 0 and i_prot["seam_top"] == 0, f"{i_prot}")
    check("T6c.vpipe_giu_dai_xoa_that_va_khong_cat_chu", i_pipe["seam_top"] == 1 and i_pipe["er_own"] == 0, f"{i_pipe}")

    # ── T7: xoá hộp ký hiệu khác nhưng giữ keep_box
    pg7 = page(w=200, h=300, rects=[(60, 100, 120, 160), (50, 150, 130, 200)])    # chữ chính + láng giềng chồng lấn mép dưới V
    V7 = (60.0, 100.0, 120.0, 160.0)
    others = np.array([[50.0, 150.0, 130.0, 200.0]])
    win7 = (40, 90, 140, 210)
    c7, _, _ = rx.cut_window2(pg7, win7, (100, 160), None, None, erase_boxes=others, keep_box=(V7[0] - 2, V7[1] - 2, V7[2] + 2, V7[3] + 2))
    bw = cv2.cvtColor(c7, cv2.COLOR_BGR2GRAY) < 128
    inside_keep = bw[(100 - 90):(160 - 90), (60 - 40):(120 - 40)].all()            # toàn bộ chữ chính còn nguyên (kể cả phần chồng lấn)
    below = bw[(163 - 90):(200 - 90), (50 - 40):(130 - 40)].any()                  # ngoài keep, trong hộp láng giềng → đã xoá
    check("T7.xoa_hop_khac_nhung_giu_vung_chu_chinh", bool(inside_keep) and not bool(below), f"giữ={bool(inside_keep)} còn_mực_dưới={bool(below)}")

    # ── T8: xáo trộn: tất định, không ô nào nhận lại chính nó, trong tầng cùng tổng độ tràn
    n = 18
    rng = np.random.default_rng(9)
    run = pd.DataFrame(dict(x0=rng.integers(10, 50, n), y0=rng.integers(10, 90, n), clipped=True))
    run["x1"], run["y1"] = run.x0 + rng.integers(30, 50, n), run.y0 + rng.integers(40, 60, n)
    run["vx0"], run["vy0"], run["vx1"], run["vy1"] = run.x0 - rng.integers(0, 4, n), run.y0 - rng.integers(0, 20, n), run.x1 + rng.integers(0, 4, n), run.y1 + rng.integers(0, 20, n)
    ovt = np.linspace(0.1, 1.0, n)
    s1, s2, s3 = rx.make_shuffle(run, ovt, seed=17), rx.make_shuffle(run, ovt, seed=17), rx.make_shuffle(run, ovt, seed=18)
    own_rel = {i: ((run.at[i, "vx0"] - run.at[i, "x0"]) / (run.at[i, "x1"] - run.at[i, "x0"]), (run.at[i, "vy0"] - run.at[i, "y0"]) / (run.at[i, "y1"] - run.at[i, "y0"])) for i in range(n)}
    new_rel = {i: ((s1[i][0] - run.at[i, "x0"]) / (run.at[i, "x1"] - run.at[i, "x0"]), (s1[i][1] - run.at[i, "y0"]) / (run.at[i, "y1"] - run.at[i, "y0"])) for i in range(n)}
    no_fixed = sum(1 for i in range(n) if np.allclose(own_rel[i], new_rel[i], atol=1e-9))
    check("T8.xao_tat_dinh_khong_nhan_lai_chinh_no", s1 == s2 and s1 != s3 and len(s1) == n and no_fixed == 0, f"n={len(s1)} trùng chính nó={no_fixed}")
    _, src = rx.make_shuffle(run, ovt, seed=17, return_src=True)
    terc = pd.qcut(pd.Series(ovt).rank(method="first"), 3, labels=False)
    same_tier = all(terc[src[i]] == terc[i] for i in src)
    check("T8b.nguon_cung_tam_phan_vi_va_la_o_bi_cat", same_tier and all(src[i] != i and bool(run.at[src[i], "clipped"]) for i in src), f"{len(src)} ô, cùng tầng={same_tier}")

    # ── T9: so E1 — cặp NaN KHÔNG phải lệch top1; lệch gần hoà thì giải thích được, lệch xa thì không
    rs = np.array([0.5, 0.4, 0.3, 0.2, 0.6])
    rt = np.array([1, np.nan, 1, 0, 1], float)
    rm = np.array([0.1, np.nan, 1e-5, -0.2, 0.3])
    cs = rs + np.array([0, 0, 1e-3, 0, 0])
    ct = np.array([1, np.nan, 0, 0, 0], float)
    cm = rm.copy()
    out = rx.compare_e1(rs, rt, rm, cs, ct, cm, np.ones(5, bool))
    check("T9.so_E1_cap_NaN_khong_la_lech", out["top1_lech"] == 2 and out["top1_lech_giai_thich_duoc"] == 1 and out["mau_nan_t_giong"] and abs(out["max_abs_ds"] - 1e-3) < 1e-12 and out["max_abs_dm"] == 0.0, f"{out}")

    # ── T10: build() — đúng tham số từng biến thể (cửa sổ thật, cờ seam, xoá hộp khác)
    pg10 = page(w=200, h=300, rects=[(60, 100, 120, 160), (60, 163, 120, 200)])      # chữ chính 100..160 + chữ dưới sát (163..200)
    bb, V10 = (60, 110, 120, 160), (60.0, 100.0, 120.0, 160.0)
    oth = np.array([[60.0, 163.0, 120.0, 200.0]])
    alt = {"vpipe_flip": rx.mirror_box(bb, V10, pg10["img"].shape), "vpipe_shuf": (60.0, 120.0, 120.0, 190.0)}
    res10 = {v: rx.build(pg10, bb, V10, v, None, None, oth, alt) for v in ("base", "vw", "vw_off", "vpipe", "vpipe_own", "vpipe_flip", "vpipe_shuf", "vbox")}
    m = {v: r[1] for v, r in res10.items()}
    ub10 = rx.union_box(bb, V10)
    check("T10a.cua_so_that_dung_dinh_nghia", m["base"]["_win"] == cl._window(pg10["img"].shape, bb, cl.PAD, cl.PAD) and m["vw"]["_win"] == cl.vision_union_window(pg10, bb, V10, margin=0.04)
          and m["vpipe"]["_win"] == cl._window(pg10["img"].shape, ub10, cl.PAD, cl.PAD) and m["vbox"]["_win"] == rx.box_window(pg10["img"].shape, V10), "base/vw/vpipe/vbox")
    check("T10b.vw_off_va_vbox_khong_carve", m["vw_off"]["seam_top"] == 0 and m["vw_off"]["seam_bot"] == 0 and m["vbox"]["seam_top"] == 0 and m["vbox"]["seam_bot"] == 0, f"{m['vw_off']['seam_top']},{m['vw_off']['seam_bot']}")
    # không prev/next ⇒ không carve ở mọi biến thể; chữ dưới nằm trong cửa sổ vpipe ⇒ nbA>0, và vpipe_own xoá sạch (nbA=0)
    check("T10c.vpipe_own_xoa_muc_hop_khac_con_vpipe_thi_khong", m["vpipe"]["nbA"] > 0 and m["vpipe_own"]["nbA"] == 0 and m["vpipe_own"]["cov"] == m["vpipe"]["cov"], f"nbA vpipe={m['vpipe']['nbA']:.3f} own={m['vpipe_own']['nbA']:.3f}")
    check("T10d.flip_va_shuf_dung_hop_thay_the", m["vpipe_flip"]["_win"] != m["vpipe"]["_win"] and m["vpipe_shuf"]["_win"] == cl._window(pg10["img"].shape, rx.union_box(bb, alt["vpipe_shuf"]), cl.PAD, cl.PAD),
          f"flip win {m['vpipe_flip']['_win']} vpipe win {m['vpipe']['_win']}")

    nf = [n_ for n_, ok in RES if not ok]
    print(f"TỔNG: pass={len(RES) - len(nf)} fail={len(nf)} n={len(RES)}")
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
