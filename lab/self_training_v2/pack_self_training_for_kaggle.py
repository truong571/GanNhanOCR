"""Đóng gói dữ liệu Self-Training Vòng 2 cho Kaggle GPU.

Chạy trên máy Mac:
    .venv/bin/python lab/self_training_v2/pack_self_training_for_kaggle.py

Tạo ra file:
    lab/self_training_v2/self_training_kaggle_dataset.zip (~85 MB)
Gồm:
  - crops.npz (72 MB, 82.780 ảnh 64x64)
  - labels_final.csv (nhãn và phân tầng GOLD/REVIEW)
  - QuocNgu_SinoNom.csv (từ điển 104k cặp để lọc ứng viên)
  - train_self_training_ocr.py (mã nguồn huấn luyện & suy diễn)
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

CROPS_SRC = REPO / "KhoiB/v3/crops_v3.npz"
LABELS_SRC = REPO / "dataset_out/labels_final.csv"
DICT_SRC = REPO / "Dict/QuocNgu_SinoNom.csv"
SCRIPT_SRC = HERE / "train_self_training_ocr.py"

ZIP_OUT = HERE / "self_training_kaggle_dataset.zip"


def pack():
    print("=" * 72)
    print("ĐÓNG GÓI DỮ LIỆU SELF-TRAINING VÒNG 2 CHO KAGGLE GPU")
    print("=" * 72)

    assert CROPS_SRC.exists(), f"Thiếu: {CROPS_SRC}"
    assert LABELS_SRC.exists(), f"Thiếu: {LABELS_SRC}"
    assert DICT_SRC.exists(), f"Thiếu: {DICT_SRC}"
    assert SCRIPT_SRC.exists(), f"Thiếu: {SCRIPT_SRC}"

    print(f"1. crops.npz:          {os.path.getsize(CROPS_SRC) / (1024*1024):.2f} MB")
    print(f"2. labels_final.csv:    {os.path.getsize(LABELS_SRC) / (1024*1024):.2f} MB")
    print(f"3. QuocNgu_SinoNom.csv: {os.path.getsize(DICT_SRC) / (1024*1024):.2f} MB")
    print(f"4. train script:        {os.path.getsize(SCRIPT_SRC) / 1024:.2f} KB")

    print(f"\nĐang nén vào {ZIP_OUT}...")
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(CROPS_SRC, arcname="crops.npz")
        z.write(LABELS_SRC, arcname="labels_final.csv")
        z.write(DICT_SRC, arcname="QuocNgu_SinoNom.csv")
        z.write(SCRIPT_SRC, arcname="train_self_training_ocr.py")

    size_mb = os.path.getsize(ZIP_OUT) / (1024 * 1024)
    print(f"✅ HOÀN TẤT: {ZIP_OUT} ({size_mb:.2f} MB)")
    print("\nCÁCH SỬ DỤNG TRÊN KAGGLE GPU:")
    print("  1. Vào kaggle.com -> Datasets -> New Dataset -> Kéo thả file 'self_training_kaggle_dataset.zip' lên.")
    print("  2. Đặt tên Dataset: 'nom-self-training'.")
    print("  3. Tạo New Notebook, chọn Accelerator: GPU T4 x2 hoặc GPU P100.")
    print("  4. Upload file 'kaggle_run_self_training.ipynb' và bấm Run All (chỉ mất ~8-10 phút).")
    print("=" * 72)


if __name__ == "__main__":
    pack()
