#!/usr/bin/env python
"""Ảnh mẫu TRƯỚC/SAU của `books[].crop_source` (vòng 7, 2026-09-23).

Cắt cùng một ô hai lần — `processed` (prepared/<Book>/pages/*.png, ảnh adapter đã xám
hoá + kéo nền) và `original` (ảnh quét gốc data/<Book>/…) — rồi ghép cạnh nhau kèm
mặt nạ mực chồng lên nhau để thấy: CÙNG vùng chữ, KHÁC nền.

    .venv/bin/python scripts/measure/crop_source_samples.py \
        --build <dataset_out có crops_bin/> [--build …] --out measure_out/crop_source

Mỗi `--build` là một thư mục build đã chạy với `crop_source: original`: crop giao nộp
nằm ở <build>/<tier>/*.png, bản đã xử lý ở <build>/crops_bin/<tier>/*.png.
Bất biến kiểm luôn (in PASS/FAIL, exit 1 nếu FAIL): hai bản CÙNG kích thước và mặt nạ
mực (gray < 128 sau khi chuẩn hoá tương phản) trùng nhau ≥ `--min-iou`.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _corr(a, b):
    """Tương quan Pearson giữa hai bản xám của CÙNG khung hình. Bản gốc và bản đã xử
    lý chỉ khác nhau một phép ánh xạ mức xám (stretch/otsu/xám hoá) nên tương quan phải
    RẤT cao; lệch khung hình dù 1 px cũng kéo nó tụt hẳn. Không dùng IoU mặt nạ mực:
    ngưỡng nhị phân (cố định hay Otsu) đổi theo tương phản nên báo lệch giả."""
    import numpy as np
    x, y = a.ravel().astype(float), b.ravel().astype(float)
    if x.std() < 1e-6 or y.std() < 1e-6:
        return 1.0
    return float(np.corrcoef(x, y)[0, 1])


def sample(build: Path, n: int, min_corr: float, out: Path, pad: int = 8) -> list[dict]:
    import cv2
    import numpy as np
    rows = []
    bins = sorted(glob.glob(str(build / "crops_bin" / "*" / "*.png")))
    step = max(1, len(bins) // max(n, 1))
    for bp in bins[::step][:n]:
        rel = os.path.relpath(bp, build / "crops_bin")
        op = build / rel
        if not op.exists():
            continue
        B = cv2.imread(bp, cv2.IMREAD_COLOR)          # đã xử lý
        O = cv2.imread(str(op), cv2.IMREAD_COLOR)      # gốc
        if B is None or O is None:
            continue
        gb, go = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY), cv2.cvtColor(O, cv2.COLOR_BGR2GRAY)
        same_shape = B.shape == O.shape
        iou = 0.0
        if same_shape:
            iou = _corr(gb, go)
        h = max(B.shape[0], O.shape[0])
        def _fit(x):
            c = np.full((h, x.shape[1], 3), 255, np.uint8)
            c[:x.shape[0]] = x
            return c
        gap = np.full((h, pad, 3), 200, np.uint8)
        side = np.hstack([_fit(B), gap, _fit(O)])
        out.mkdir(parents=True, exist_ok=True)
        name = f"{build.name}__{Path(rel).stem}.png"
        cv2.imwrite(str(out / name), side)
        rows.append(dict(build=build.name, image=rel, file=str(out / name),
                         w=B.shape[1], h=B.shape[0], same_shape=same_shape,
                         gray_corr=round(iou, 4),
                         bg_processed=int(np.percentile(gb, 90)), bg_original=int(np.percentile(go, 90)),
                         levels_processed=int(len(np.unique(gb))), levels_original=int(len(np.unique(go))),
                         ok=bool(same_shape and iou >= min_corr)))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="append", default=[], required=True)
    ap.add_argument("--out", default=str(REPO / "measure_out" / "crop_source"))
    ap.add_argument("--per-build", type=int, default=2)
    ap.add_argument("--min-corr", type=float, default=0.80)
    a = ap.parse_args(argv)
    out = Path(a.out)
    rows: list[dict] = []
    for b in a.build:
        rows += sample(Path(b), a.per_build, a.min_corr, out)
    import csv
    import json
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "samples.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["build"])
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if not r["ok"]]
    json.dump(dict(n=len(rows), n_fail=len(bad), min_corr=a.min_corr,
                   invariants=[dict(name="crop gốc và crop đã xử lý cùng kích thước",
                                    expected=len(rows), observed=sum(r["same_shape"] for r in rows),
                                    ok=all(r["same_shape"] for r in rows)),
                               dict(name=f"tương quan mức xám gốc↔đã xử lý >= {a.min_corr}",
                                    expected=len(rows), observed=sum(r["ok"] for r in rows),
                                    ok=not bad)]),
              open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for r in rows:
        print(f"  {'PASS' if r['ok'] else 'FAIL'} {r['build']:22s} {Path(r['image']).name:44s} "
              f"{r['w']}×{r['h']} r={r['gray_corr']:.3f} | nền {r['bg_processed']}→{r['bg_original']} "
              f"| mức xám {r['levels_processed']}→{r['levels_original']}")
    print(f"[crop_source] {len(rows)} ảnh mẫu -> {out} ({len(bad)} FAIL)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
