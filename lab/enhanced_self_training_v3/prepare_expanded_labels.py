"""Chuẩn bị tập nhãn mở rộng (53.543 nhãn sạch) cho Enhanced Self-Training v3.

Kết hợp:
- 51.371 nhãn GOLD hiện hành (100% tự động).
- 2.172 nhãn vừa được giải cứu an toàn từ tập REVIEW (RESCUE_GOLD_HIGH + RESCUE_GOLD_MED).

Đầu ra:
- lab/enhanced_self_training_v3/labels_expanded.csv (83.239 dòng, khớp 1:1 với crops.npz)
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

BASE_LABELS_PATH = REPO / "dataset_out/labels_final.csv"
RV_LABELS_PATH = REPO / "lab/self_training_v2/review_pseudo_labels.csv"
OUT_CSV_PATH = HERE / "labels_expanded.csv"
SUMMARY_PATH = HERE / "expanded_summary.json"


def prepare():
    print("=" * 72)
    print("CHUẨN BỊ DỮ LIỆU NHÃN MỞ RỘNG (ENHANCED SELF-TRAINING V3)")
    print("=" * 72)

    assert BASE_LABELS_PATH.exists(), f"Thiếu: {BASE_LABELS_PATH}"
    assert RV_LABELS_PATH.exists(), f"Thiếu: {RV_LABELS_PATH}"

    print(f"1. Nạp nhãn gốc: {BASE_LABELS_PATH}")
    df_base = pd.read_csv(BASE_LABELS_PATH)
    print(f"   Tổng số dòng: {len(df_base):,}")

    print(f"2. Nạp kết quả giải cứu REVIEW: {RV_LABELS_PATH}")
    df_rv = pd.read_csv(RV_LABELS_PATH)
    print(f"   Tổng số dòng REVIEW: {len(df_rv):,}")

    # Lọc các ô được giải cứu
    rescued_mask = df_rv["decision"].isin(["RESCUE_GOLD_HIGH", "RESCUE_GOLD_MED"])
    df_rescued = df_rv[rescued_mask].copy()
    print(f"   Số ô được giải cứu: {len(df_rescued):,} (HIGH: {(df_rv['decision'] == 'RESCUE_GOLD_HIGH').sum():,}, MED: {(df_rv['decision'] == 'RESCUE_GOLD_MED').sum():,})")

    # Tạo key để join
    df_base["_key"] = (
        df_base["book"].astype(str)
        + "_"
        + df_base["page"].astype(str)
        + "_"
        + df_base["column"].astype(str)
        + "_"
        + df_base["nom_idx"].astype(str)
    )
    df_rescued["_key"] = (
        df_rescued["book"].astype(str)
        + "_"
        + df_rescued["page"].astype(str)
        + "_"
        + df_rescued["column"].astype(str)
        + "_"
        + df_rescued["nom_idx"].astype(str)
    )

    rescue_dict = df_rescued.set_index("_key")[["predicted_nom", "unicode", "prob", "decision"]].to_dict("index")

    # Tạo các cột mới
    df_out = df_base.copy()
    df_out["tier_v2"] = df_out["tier"]
    df_out["label_v2"] = df_out["label"]
    df_out["unicode_v2"] = df_out["unicode"]
    df_out["is_train_v2"] = False
    df_out["rescue_prob"] = 0.0

    # 1. Gán cho GOLD ban đầu
    gold_mask = (df_out["tier"] == "GOLD") & (df_out["label"].notna()) & (df_out["label"].str.strip() != "")
    df_out.loc[gold_mask, "is_train_v2"] = True
    print(f"   Số nhãn GOLD ban đầu: {gold_mask.sum():,}")

    # 2. Cập nhật các ô RESCUED
    updated_cnt = 0
    for idx, row in df_out.iterrows():
        k = row["_key"]
        if k in rescue_dict:
            res = rescue_dict[k]
            df_out.at[idx, "label_v2"] = res["predicted_nom"]
            df_out.at[idx, "unicode_v2"] = res["unicode"]
            df_out.at[idx, "tier_v2"] = "GOLD_RESCUED"
            df_out.at[idx, "is_train_v2"] = True
            df_out.at[idx, "rescue_prob"] = res["prob"]
            updated_cnt += 1

    print(f"   Đã cập nhật nhãn cho {updated_cnt:,} ô giải cứu.")

    total_train = df_out["is_train_v2"].sum()
    print(f"\n=> TỔNG CỘNG TẬP HUẤN LUYỆN MỚI (is_train_v2 = True): {total_train:,} ô!")

    # Thống kê số lớp ký tự
    train_classes = df_out[df_out["is_train_v2"]]["label_v2"].nunique()
    print(f"=> Tổng số lớp ký tự Nôm độc nhất trong tập huấn luyện: {train_classes:,} lớp.")

    # Xoá key tạm
    df_out.drop(columns=["_key"], inplace=True)

    # Lưu file
    df_out.to_csv(OUT_CSV_PATH, index=False)
    print(f"\n✓ Đã lưu bảng nhãn mở rộng: {OUT_CSV_PATH}")

    summary = {
        "tong_so_dong": len(df_out),
        "so_o_gold_ban_dau": int(gold_mask.sum()),
        "so_o_giai_cuu": updated_cnt,
        "tong_tap_huan_luyen_v2": int(total_train),
        "so_lop_ky_tu": int(train_classes),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    prepare()
