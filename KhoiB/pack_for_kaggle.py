"""Đóng gói dữ liệu và mã nguồn cho Kaggle (Khối B-1).
Chạy trên máy Mac:
    .venv/bin/python KhoiB/pack_for_kaggle.py

Tạo ra file zip: KhoiB/KhoiB_kaggle_dataset.zip (~89 MB)
Gồm:
  - crops.npz (75 MB)
  - labels_final.csv (14 MB)
  - train_oof_cnn.py
Sẵn sàng upload lên Kaggle Dataset để chạy song song.
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CROPS_SRC = REPO / "lab/gan_nhan_2026-09-13/crops.npz"
LABELS_SRC = REPO / "dataset_out/labels_final.csv"
SCRIPT_SRC = HERE / "train_oof_cnn.py"

ZIP_OUT = HERE / "KhoiB_kaggle_dataset.zip"


def main():
    print("=" * 72)
    print("ĐÓNG GÓI DỮ LIỆU KHỐI B CHO KAGGLE")
    print("=" * 72)

    assert CROPS_SRC.exists(), f"Không tìm thấy: {CROPS_SRC}"
    assert LABELS_SRC.exists(), f"Không tìm thấy: {LABELS_SRC}"
    assert SCRIPT_SRC.exists(), f"Không tìm thấy: {SCRIPT_SRC}"

    print(f"1. crops.npz:       {os.path.getsize(CROPS_SRC) / (1024*1024):.2f} MB")
    print(f"2. labels_final.csv: {os.path.getsize(LABELS_SRC) / (1024*1024):.2f} MB")
    print(f"3. train_oof_cnn.py: {os.path.getsize(SCRIPT_SRC) / 1024:.2f} KB")

    print(f"\nĐang nén vào {ZIP_OUT}...")
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CROPS_SRC, arcname="crops.npz")
        z.write(LABELS_SRC, arcname="labels_final.csv")
        z.write(SCRIPT_SRC, arcname="train_oof_cnn.py")

    size_mb = os.path.getsize(ZIP_OUT) / (1024 * 1024)
    print(f"✅ HOÀN TẤT: {ZIP_OUT} ({size_mb:.2f} MB)")
    print("\nCÁCH DÙNG TRÊN KAGGLE:")
    print("  1. Vào kaggle.com -> Datasets -> New Dataset -> Tải file 'KhoiB_kaggle_dataset.zip' lên.")
    print("  2. Đặt tên Dataset: 'khoib-data'.")
    print("  3. Tạo New Notebook, đính kèm dataset này và chọn Accelerator: GPU T4 x2 (hoặc P100).")
    print("  4. Chạy file 'kaggle_run_khoib.ipynb' (khoảng 12–15 phút).")
    print("=" * 72)


if __name__ == "__main__":
    main()
