"""Đóng gói dữ liệu Enhanced Self-Training v3 (Phương án 1 + 2) cho Kaggle GPU.

Tạo tệp:
- lab/enhanced_self_training_v3/enhanced_ocr_kaggle.zip (~80 MB)
Bao gồm:
1. crops.npz: 83.239 ảnh crop xám (64x64)
2. labels_expanded.csv: 83.239 dòng (54.094 nhãn sạch is_train_v2=True)
3. QuocNgu_SinoNom.csv: Từ điển đối chiếu
4. train_enhanced_ocr.py: Script huấn luyện và suy diễn
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

CROPS_SRC = REPO / "KhoiB/v3/crops_v3.npz"
LABELS_SRC = HERE / "labels_expanded.csv"
DICT_SRC = REPO / "Dict/QuocNgu_SinoNom.csv"
SCRIPT_SRC = HERE / "train_enhanced_ocr.py"

ZIP_OUT = HERE / "enhanced_ocr_kaggle.zip"


def pack():
    print("=" * 72)
    print("ĐÓNG GÓI DỮ LIỆU ENHANCED SELF-TRAINING V3 CHO KAGGLE GPU")
    print("=" * 72)

    assert CROPS_SRC.exists(), f"Thiếu: {CROPS_SRC}"
    assert LABELS_SRC.exists(), f"Thiếu: {LABELS_SRC}"
    assert DICT_SRC.exists(), f"Thiếu: {DICT_SRC}"
    assert SCRIPT_SRC.exists(), f"Thiếu: {SCRIPT_SRC}"

    print(f"1. crops.npz:          {os.path.getsize(CROPS_SRC) / (1024*1024):.2f} MB")
    print(f"2. labels_expanded.csv: {os.path.getsize(LABELS_SRC) / (1024*1024):.2f} MB")
    print(f"3. QuocNgu_SinoNom.csv: {os.path.getsize(DICT_SRC) / (1024*1024):.2f} MB")
    print(f"4. train script:        {os.path.getsize(SCRIPT_SRC) / 1024:.2f} KB")

    print(f"\nĐang nén vào: {ZIP_OUT} ...")
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(CROPS_SRC, arcname="crops.npz")
        zf.write(LABELS_SRC, arcname="labels_expanded.csv")
        zf.write(DICT_SRC, arcname="QuocNgu_SinoNom.csv")
        zf.write(SCRIPT_SRC, arcname="train_enhanced_ocr.py")

    zip_size = os.path.getsize(ZIP_OUT) / (1024 * 1024)
    print(f"✓ Hoàn tất! Kích thước file zip: {zip_size:.2f} MB")
    print("\n👉 BẠN CHỈ CẦN:")
    print(f"   1. Kéo thả file '{ZIP_OUT.name}' vào Kaggle Dataset.")
    print("   2. Mở file 'kaggle_run_enhanced_pipeline.ipynb' và bấm 'Run All'.")


if __name__ == "__main__":
    pack()
