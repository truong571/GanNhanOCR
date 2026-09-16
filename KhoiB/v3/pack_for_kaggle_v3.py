"""Đóng gói Khối B-1 thế hệ v3 cho Kaggle (KHÔNG huấn luyện trên máy).
    .venv/bin/python KhoiB/v3/pack_for_kaggle_v3.py
Tạo: KhoiB/v3/KhoiB_v3_kaggle_dataset.zip gồm
  - crops_v3.npz        (từ KhoiB/v3/make_crops_v3.py — 83.239 crop 64×64, bbox v3, kèm khoá nom_idx)
  - labels_final.csv    (dataset_out/labels_final.csv, bộ v3 đã thăng cấp 2026-09-16)
  - train_oof_cnn_v3.py
  - MANIFEST.json       (md5 từng tệp, git HEAD, số dòng) — để đối chiếu khi nhận kết quả về
Kiểm trước khi nén: labels_md5 trong npz == md5 labels_final.csv; N npz == N csv; khoá khớp từng dòng.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CROPS = HERE / "crops_v3.npz"
LABELS = REPO / "dataset_out/labels_final.csv"
SCRIPT = HERE / "train_oof_cnn_v3.py"
ZIP_OUT = HERE / "KhoiB_v3_kaggle_dataset.zip"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    for p in (CROPS, LABELS, SCRIPT):
        assert p.exists(), f"Thiếu {p} (crops: chạy KhoiB/v3/make_crops_v3.py trước)"
    m_labels = md5(LABELS)
    z = np.load(CROPS)
    df = pd.read_csv(LABELS, dtype=str, keep_default_na=False, usecols=["book", "page", "column", "nom_idx"])
    assert str(z["labels_md5"]) == m_labels, f"crops_v3.npz cắt từ labels md5 {str(z['labels_md5'])[:8]} ≠ csv hiện tại {m_labels[:8]} — chạy lại make_crops_v3.py"
    assert len(z["ok"]) == len(df), "N lệch"
    for c in ("book", "page", "column", "nom_idx"):
        assert (z[c].astype(str) == df[c].to_numpy(str)).all(), f"khoá {c} lệch"
    try:
        head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        head = ""
    manifest = {
        "generation": "v3", "made_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": head,
        "files": {
            "crops_v3.npz": {"md5": md5(CROPS), "bytes": os.path.getsize(CROPS), "n": int(len(z["ok"])), "ok": int(z["ok"].sum()), "labels_md5": str(z["labels_md5"])},
            "labels_final.csv": {"md5": m_labels, "bytes": os.path.getsize(LABELS), "rows": int(len(df)), "source": str(LABELS.relative_to(REPO))},
            "train_oof_cnn_v3.py": {"md5": md5(SCRIPT), "bytes": os.path.getsize(SCRIPT)},
        },
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    with zipfile.ZipFile(ZIP_OUT, "w") as zf:
        zf.write(CROPS, arcname="crops_v3.npz", compress_type=zipfile.ZIP_STORED)      # npz đã nén sẵn
        zf.write(LABELS, arcname="labels_final.csv", compress_type=zipfile.ZIP_DEFLATED)
        zf.write(SCRIPT, arcname="train_oof_cnn_v3.py", compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nHOÀN TẤT: {ZIP_OUT} ({os.path.getsize(ZIP_OUT) / 1e6:.1f} MB)")
    print("Kaggle: Datasets → New Dataset → tải zip này (tên gợi ý: khoib-v3-data) → Notebook GPU T4/P100 → Add Input → chạy kaggle_run_khoib_v3.ipynb")


if __name__ == "__main__":
    main()
