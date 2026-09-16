"""Cắt crop 64×64 cho MỌI dòng của bộ nhãn v3 (dataset_out/labels_final.csv, bản đã thăng cấp 2026-09-16)
bằng ĐÚNG hàm cut() của lab/gan_nhan_2026-09-13/thi_giac_am_tiet.py (pad 0,08 + tighten_box + vuông hoá + 64×64).

    .venv/bin/python KhoiB/v3/make_crops_v3.py            # ≈15–30 s trên Mac
    .venv/bin/python KhoiB/v3/make_crops_v3.py --labels <csv> --out <npz>

Đầu ra: KhoiB/v3/crops_v3.npz
    X        uint8  (N, 64, 64)   crop (255 = nền, mực tối) — thứ tự = thứ tự dòng labels_final.csv
    ok       bool   (N,)          cắt được (bbox hợp lệ, có ảnh trang)
    book, page, column, nom_idx, syl_idx, bbox   khoá từng dòng (str) — để train_oof_cnn_v3.py đối chiếu với csv
    labels_md5   md5 của tệp labels_final.csv đã dùng (bảo chứng "đúng thế hệ")
    labels_name  tên tệp nhãn
Khác bản cũ (lab crops.npz): 83.239 dòng (không phải 82.780), bbox v3 (10.311 ô usable đổi hộp), có khoá nom_idx.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LAB = REPO / "lab/gan_nhan_2026-09-13/thi_giac_am_tiet.py"
sys.path.insert(0, str(REPO))


def load_lab():
    """Nạp đúng module thí nghiệm để dùng cut()/BOOKDIR/SZ của nó (không chép lại mã)."""
    spec = importlib.util.spec_from_file_location("thi_giac_am_tiet", LAB)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(REPO / "dataset_out/labels_final.csv"))
    ap.add_argument("--out", default=str(HERE / "crops_v3.npz"))
    ap.add_argument("--pad", type=float, default=0.08, help="giống lab cut(pad=0.08)")
    args = ap.parse_args()

    T = load_lab()
    SZ = T.SZ
    t0 = time.time()
    raw = Path(args.labels).read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    df = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    need = ["book", "page", "column", "nom_idx", "syl_idx", "bbox"]
    miss = [c for c in need if c not in df.columns]
    assert not miss, f"labels thiếu cột {miss} — phải là bộ v3 (có nom_idx/bbox)"
    key = df.book + "|" + df.page + "|" + df.column + "|" + df.nom_idx
    assert key.is_unique, "khoá (book,page,column,nom_idx) không duy nhất"
    N = len(df)
    print(f"labels: {args.labels}  md5={md5}  N={N:,}", flush=True)

    X = np.zeros((N, SZ, SZ), np.uint8)
    ok = np.zeros(N, bool)
    no_img = []
    bad_bbox = 0
    for (book, page), g in df.groupby(["book", "page"], sort=False):
        p = REPO / "prepared" / T.BOOKDIR[book] / "pages" / f"{page}.png"
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            no_img.append(str(p))
            continue
        for idx, bb in zip(g.index, g.bbox):
            try:
                bbox = json.loads(bb)
                c = T.cut(img, bbox, pad=args.pad)
            except Exception:
                bad_bbox += 1
                continue
            if c is not None:
                X[idx] = c
                ok[idx] = True
    np.savez_compressed(
        args.out, X=X, ok=ok,
        book=df.book.to_numpy(str), page=df.page.to_numpy(str), column=df.column.to_numpy(str),
        nom_idx=df.nom_idx.to_numpy(str), syl_idx=df.syl_idx.to_numpy(str), bbox=df.bbox.to_numpy(str),
        labels_md5=np.array(md5), labels_name=np.array(Path(args.labels).name), sz=np.array(SZ), pad=np.array(args.pad),
    )
    print(f"crops: ok {int(ok.sum()):,}/{N:,} · bbox lỗi {bad_bbox} · trang thiếu ảnh {len(no_img)} · {time.time()-t0:.0f}s -> {args.out}")
    if no_img:
        print("  trang thiếu ảnh:", no_img[:5])


if __name__ == "__main__":
    main()
