#!/usr/bin/env python3
"""CHẨN ĐOÁN X1b — NGOÀI giao thức, làm SAU KHI đã thấy kết quả X1 (nên chỉ mô tả, không phải kiểm định).

Câu hỏi: crop chứa trọn chữ (vbox/vpipe) bị bộ mã hoá chấm thấp hơn crop giao nộp (bị cắt) là do NỘI DUNG (thiếu/thừa mực) hay do KHUNG (tỉ lệ cao×rộng, độ lớn chữ
trong khung sau khi bộ mã hoá đệm vuông + co về 128)? Thiết kế 2×2 trên 601 ô bị cắt, KHÔNG thêm mực mới:
  base         nội dung bị cắt  × khung gốc
  base_pad     nội dung bị cắt  × khung của vbox  (đệm giấy trống quanh crop giao nộp cho bằng khung vbox, căn giữa)
  vbox_squash  nội dung trọn chữ × khung gốc       (co dãn không đều crop vbox về đúng cao×rộng của crop giao nộp)
  vbox         nội dung trọn chữ × khung của vbox
  vpipe_squash nội dung vpipe (kèm mực lạ) × khung gốc
Nếu điểm theo KHUNG (base_pad thấp, vbox_squash cao) thì độ giống glyph của bộ mã hoá không đo được 'crop trọn chữ hơn'.
  .venv/bin/python vision/crop_test/diag_x1_khung.py
Ghi: vision/ket_qua/crop_test/X1b_khung.csv.gz và X1b_BAO_CAO.md."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_crop_test as A                       # noqa: E402
import crop_lib as cl                               # noqa: E402
import run_crop_test as rct                         # noqa: E402
import run_explore_vwindow as rx                    # noqa: E402

OUT = HERE.parent / "ket_qua" / "crop_test"


def paper(g):
    return int(np.percentile(g, 90))


def pad_to(g, H, W):
    h, w = g.shape[:2]
    t, l = (H - h) // 2, (W - w) // 2
    return cv2.copyMakeBorder(g, t, H - h - t, l, W - w - l, cv2.BORDER_CONSTANT, value=paper(g))


def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)
    cells = rct.load_eval_cells("L16").reset_index(drop=True)
    run = cells[(cells["status"] == "agree") & cells["vx0"].notna() & (cells["harm"] == 1)].copy().sort_values(["page", "column", "y0"]).reset_index(drop=True)
    log(f"# X1b — chẩn đoán khung × nội dung (NGOÀI giao thức, làm sau khi thấy X1; mô tả) — n={len(run)} ô bị cắt\n")
    pc = cl.PageCache("L16")
    VAR = ["base", "base_pad", "vbox_squash", "vbox", "vpipe", "vpipe_squash"]
    grays = {v: [None] * len(run) for v in VAR}
    empty = np.zeros((0, 4))
    for page, g in run.groupby("page", sort=True):
        pg = pc.get(page)
        for i, r in g.iterrows():
            bbox = (int(r.x0), int(r.y0), int(r.x1), int(r.y1))
            V = rx.clamp_box((float(r.vx0), float(r.vy0), float(r.vx1), float(r.vy1)), pg["img"].shape)
            gb, _ = rx.build(pg, bbox, V, "base", r.prev_bbox, r.next_bbox, empty, {})
            gv, _ = rx.build(pg, bbox, V, "vbox", r.prev_bbox, r.next_bbox, empty, {})
            gp, _ = rx.build(pg, bbox, V, "vpipe", r.prev_bbox, r.next_bbox, empty, {})
            if gb is None or gv is None or gp is None:
                continue
            hb, wb = gb.shape[:2]
            hv, wv = gv.shape[:2]
            grays["base"][i], grays["vbox"][i], grays["vpipe"][i] = gb, gv, gp
            grays["base_pad"][i] = pad_to(gb, max(hb, hv), max(wb, wv))
            grays["vbox_squash"][i] = cv2.resize(gv, (wb, hb), interpolation=cv2.INTER_AREA)
            grays["vpipe_squash"][i] = cv2.resize(gp, (wb, hb), interpolation=cv2.INTER_AREA)
    S = cl.Scorer()
    labels, syls = run["Leval"].tolist(), run["syllable"].tolist()
    S.prepare(labels, syls)
    res = run[["page", "column", "Leval"]].copy()
    res["cl"] = res["page"]
    frame = {}
    for v in VAR:
        idx = [i for i, g_ in enumerate(grays[v]) if g_ is not None]
        E = S.embed([grays[v][i] for i in idx])
        full = np.zeros((len(run), E.shape[1]), np.float32)
        have = np.zeros(len(run), bool)
        for k, i in enumerate(idx):
            full[i] = E[k]
            have[i] = True
        s, m, t = S.metrics(full, labels, syls)
        for arr in (s, m, t):
            arr[~have] = np.nan
        res[f"s_{v}"], res[f"m_{v}"], res[f"t_{v}"] = s, m, t
        res[f"chg_{v}"] = np.nan
        hh = [grays[v][i].shape[0] for i in idx]
        ww = [grays[v][i].shape[1] for i in idx]
        frame[v] = (float(np.median(hh)), float(np.median(ww)), float(np.median(np.array(hh) / np.array(ww))))
    log("| biến thể | nội dung | khung (cao×rộng trung vị; tỉ lệ cao/rộng) | top1 | Δ so với base [KTC95 % theo trang] |")
    log("|---|---|---|---|---|")
    desc = {"base": "bị cắt", "base_pad": "bị cắt (đệm giấy trống)", "vbox_squash": "trọn chữ (co dãn về khung gốc)", "vbox": "trọn chữ", "vpipe": "trọn chữ + mực lạ", "vpipe_squash": "trọn chữ + mực lạ (co dãn về khung gốc)"}
    out = {}
    for v in VAR:
        h, w, ar = frame[v]
        if v == "base":
            log(f"| base | {desc[v]} | {h:.0f}×{w:.0f}; {ar:.2f} | {100 * res['t_base'].mean():.2f} % | — |")
            continue
        dd = res[res[f"t_{v}"].notna() & res["t_base"].notna()]
        r = A.delta(dd, v)
        t = r["top1_pp"]
        out[v] = dict(top1=r["top1_var"], d=t["mean"], lo=t["lo"], hi=t["hi"], n=r["n"])
        log(f"| {v} | {desc[v]} | {h:.0f}×{w:.0f}; {ar:.2f} | {r['top1_var']:.2f} % | {t['mean']:+.2f} [{t['lo']:+.2f}; {t['hi']:+.2f}] |")
    log("\nĐọc (mô tả): nếu base_pad ≈ vbox (thấp) và vbox_squash ≈ base (cao) thì điểm của bộ mã hoá phụ thuộc KHUNG (tỉ lệ/độ lớn chữ sau đệm vuông + co về 128), không phụ thuộc việc crop có trọn chữ hay không; "
        "khi đó 'top1' không phải thước đo hợp lệ cho 'crop trọn chữ hơn' ở L16 và mọi kết luận 'nới/dời làm tệ hơn' bằng thước này (E1, X1) chỉ nói về độ giống glyph phông, không nói về chất lượng crop.")
    res.drop(columns=["cl"]).to_csv(OUT / "X1b_khung.csv.gz", index=False)
    (OUT / "X1b_BAO_CAO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT / 'X1b_BAO_CAO.md'}")


if __name__ == "__main__":
    main()
