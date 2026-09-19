"""D-1 · Dựng crops npz (khuôn crops_v3.npz) cho MỌI dòng labels_final.csv từ TỆP CROP (v1 giao nộp hoặc crops_v2 F3g)
để huấn luyện lại CNN OOF 5-fold (train_oof_cnn_v3.py) trên crop v2 và so OOF top-1 đối xứng với bản v1.

Ô có crop (image != '') -> cut(tệp crop, hộp toàn ảnh, pad 0,08) = visual_emission.cut_box (tighten + vuông + 64×64);
ô không có crop (REVIEW/QUARANTINE không cắt) -> cut(trang, bbox, pad 0,08) như make_crops_v3.py (không dự huấn luyện,
chỉ để OOF phủ mọi dòng như bản gốc).

    .venv/bin/python KhoiB/v3/make_crops_v2_npz.py --labels dataset_out_v3/labels_final.csv \
        --crop-root <thư mục crop> --out <npz>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from pipeline.align_engine.visual_emission import cut_box, SZ, CUT_PAD  # noqa: E402

BOOKDIR = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "stt11": "SachThanhTruyen11"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(REPO / "dataset_out/labels_final.csv"))
    ap.add_argument("--crop-root", required=True, help="thư mục chứa <tier>/<tên>.png (dataset_out_v3 | crops_v2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=CUT_PAD)
    args = ap.parse_args()
    t0 = time.time()
    raw = Path(args.labels).read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    df = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    N = len(df)
    X = np.zeros((N, SZ, SZ), np.uint8)
    ok = np.zeros(N, bool)
    src = np.zeros(N, np.int8)      # 1 = từ tệp crop, 2 = từ trang (bbox)
    root = Path(args.crop_root)
    n_file = 0
    for (book, page), g in df.groupby(["book", "page"], sort=False):
        img = None
        for idx, im, bb in zip(g.index, g.image, g.bbox):
            c = None
            if im:
                gcrop = cv2.imread(str(root / im), cv2.IMREAD_GRAYSCALE)
                if gcrop is not None and gcrop.size:
                    c = cut_box(gcrop, [0, 0, gcrop.shape[1], gcrop.shape[0]], pad=args.pad)
                    if c is not None:
                        src[idx] = 1; n_file += 1
            if c is None and bb:
                if img is None:
                    img = cv2.imread(str(REPO / "prepared" / BOOKDIR[book] / "pages" / f"{page}.png"), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    try:
                        c = cut_box(img, json.loads(bb), pad=args.pad)
                        if c is not None:
                            src[idx] = 2
                    except Exception:
                        c = None
            if c is not None:
                X[idx] = c; ok[idx] = True
    np.savez_compressed(
        args.out, X=X, ok=ok, src=src,
        book=df.book.to_numpy(str), page=df.page.to_numpy(str), column=df.column.to_numpy(str),
        nom_idx=df.nom_idx.to_numpy(str), syl_idx=df.syl_idx.to_numpy(str), bbox=df.bbox.to_numpy(str),
        labels_md5=np.array(md5), labels_name=np.array(Path(args.labels).name), sz=np.array(SZ), pad=np.array(args.pad),
        crop_root=np.array(str(root)),
    )
    print(f"crops: ok {int(ok.sum()):,}/{N:,} · từ tệp crop {n_file:,} · từ trang {int((src == 2).sum()):,} · "
          f"{time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
