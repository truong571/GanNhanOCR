"""Tự động giải cứu các ô REVIEW bằng mô hình Self-Training / Enhanced OCR.

Tích hợp vào pipeline (sau bước confusion_fix, trước export):
- Nạp checkpoint mô hình OCR Nôm nội bộ (đã học nét bút lông) hoặc bảng nhãn suy diễn.
- Hỗ trợ kết hợp đa nguồn (v2 Self-Training + v3 Enhanced SE-ResNet).
- Quét các ô REVIEW (hoặc tuỳ chọn SYLLABLE) có âm Quốc ngữ hợp lệ trong từ điển.
- Nếu mô hình dự đoán ký tự c với xác suất P >= tau VÀ c thuộc Dict(âm):
  1. Thăng cấp ô thành tier GOLD với rule 'self_training_rescue'.
  2. Xuất ảnh crop từ crops.npz ra dataset_out/gold/ để phục vụ bước export.
  3. Cập nhật nhãn và mã Unicode.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import List, Set, Union

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_dict(dict_path: Path) -> dict[str, Set[str]]:
    """Nạp từ điển âm Quốc ngữ -> tập chữ Nôm."""
    qn_to_nom: dict[str, Set[str]] = {}
    if not dict_path.exists():
        return qn_to_nom
    df = pd.read_csv(dict_path, keep_default_na=False, na_values=[""])
    syl_col = df.columns[0]
    nom_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    for _, row in df.iterrows():
        s = str(row[syl_col]).strip().lower()
        c = str(row[nom_col]).strip()
        if s and c:
            qn_to_nom.setdefault(s, set()).add(c)
    return qn_to_nom


def run_rescue(
    in_csv: Path,
    out_csv: Path,
    crops_path: Path,
    dict_path: Path,
    model_path: Path | None = None,
    pseudo_csv: Union[List[Union[str, Path]], str, Path, None] = None,
    tau: float = 0.70,
    rescue_syllable: bool = False,
    report_path: Path | None = None,
    verbose: bool = True,
) -> int:
    if not in_csv.exists():
        print(f"[rescue] Lỗi: Không thấy tệp đầu vào {in_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(in_csv)
    total_rows = len(df)
    n_gold_before = (df["tier"] == "GOLD").sum()
    n_syl_before = (df["tier"] == "SYLLABLE").sum()
    n_rv_before = (df["tier"] == "REVIEW").sum()

    if verbose:
        print("=" * 64)
        print("BƯỚC GIẢI CỨU TỰ ĐỘNG: SELF-TRAINING IN-DOMAIN RESCUE")
        print("=" * 64)
        print(f"Tổng số dòng nhãn: {total_rows:,}")
        print(f"Số ô GOLD trước giải cứu:     {n_gold_before:,}")
        print(f"Số ô SYLLABLE trước giải cứu: {n_syl_before:,}")
        print(f"Số ô REVIEW trước giải cứu:   {n_rv_before:,}")

    # Lập chỉ mục ảnh crops từ crops.npz
    crop_lookup = {}
    X_crops = None
    if crops_path.exists():
        try:
            npz = np.load(crops_path)
            X_crops = npz["X"]
            books = npz["book"]
            pages = npz["page"]
            cols = npz["column"]
            nom_idxs = npz["nom_idx"]
            for i in range(len(X_crops)):
                k = (str(books[i]), str(pages[i]), int(cols[i]), int(nom_idxs[i]))
                crop_lookup[k] = i
            if verbose:
                print(f"[rescue] Đã lập chỉ mục {len(crop_lookup):,} ảnh crop từ {crops_path.name}")
        except Exception as e:
            print(f"[rescue] Cảnh báo khi nạp crops: {e}")

    # Nạp từ điển
    qn_to_nom = load_dict(dict_path)
    if verbose:
        print(f"[rescue] Từ điển nạp được: {len(qn_to_nom):,} âm Quốc ngữ")

    # Kiểm tra nguồn dự đoán (Model Checkpoint hoặc Pseudo CSV)
    predictions = {}  # key -> {"pred_char": ..., "unicode": ..., "prob": ..., "source": ...}

    # Xác định danh sách pseudo CSVs
    pseudo_files: list[Path] = []
    if pseudo_csv:
        if isinstance(pseudo_csv, (str, Path)):
            for p in str(pseudo_csv).split(","):
                p_clean = p.strip()
                if p_clean:
                    pseudo_files.append(Path(p_clean))
        elif isinstance(pseudo_csv, list):
            for p in pseudo_csv:
                if str(p).strip():
                    pseudo_files.append(Path(p))
    else:
        # Tự động tìm kiếm nguồn dự đoán mặc định
        default_candidates = [
            REPO / "lab/self_training_v2/review_pseudo_labels.csv",
            REPO / "lab/enhanced_self_training_v3/enhanced_pseudo_labels.csv",
        ]
        for p in default_candidates:
            if p.exists():
                pseudo_files.append(p)

    # Ưu tiên 1: Nạp từ các file pseudo CSV có sẵn
    if pseudo_files:
        for pf in pseudo_files:
            if not pf.exists():
                continue
            if verbose:
                print(f"[rescue] Nạp dự đoán từ: {pf}")
            try:
                df_pseudo = pd.read_csv(pf)
                for _, r in df_pseudo.iterrows():
                    k = (str(r["book"]), str(r["page"]), int(r["column"]), int(r["nom_idx"]))
                    pred_c = str(r.get("predicted_nom", "")).strip()
                    prob = float(r.get("prob", 0.0))
                    u = str(r.get("unicode", "")).strip()
                    if not u and pred_c:
                        u = f"U+{ord(pred_c):04X}"
                    
                    if not pred_c:
                        continue

                    # Cập nhật nếu chưa có hoặc có xác suất cao hơn
                    if k not in predictions or prob > predictions[k]["prob"]:
                        predictions[k] = {
                            "pred_char": pred_c,
                            "unicode": u,
                            "prob": prob,
                            "source": pf.name,
                        }
            except Exception as e:
                print(f"[rescue] Lỗi khi đọc {pf}: {e}")

    # Ưu tiên 2: Chạy suy diễn bằng PyTorch model nếu có model và còn ô chưa dự đoán
    elif model_path and Path(model_path).exists() and X_crops is not None:
        if verbose:
            print(f"[rescue] Chạy suy diễn bằng mô hình PyTorch: {model_path}")
        try:
            import torch
            import torch.nn.functional as F

            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            ckpt = torch.load(model_path, map_location=device)
            id_to_char = ckpt.get("id_to_char", {})
            num_classes = len(id_to_char)

            # Lấy các ô REVIEW để suy diễn
            rv_indices = df[df["tier"] == "REVIEW"].index.tolist()
            rv_crops = []
            valid_keys = []

            for idx in rv_indices:
                row = df.iloc[idx]
                k = (str(row["book"]), str(row["page"]), int(row["column"]), int(row["nom_idx"]))
                if k in crop_lookup:
                    arr = X_crops[crop_lookup[k]].astype(np.float32) / 255.0
                    t = (arr - 0.5) / 0.5
                    rv_crops.append(t)
                    valid_keys.append(k)

            if rv_crops:
                batch_tensor = torch.from_numpy(np.array(rv_crops)).unsqueeze(1).to(device)
                state_dict = ckpt["model_state"]

                # Nhận diện kiến trúc mô hình (EnhancedNomOCRNet vs NomOCRNet)
                if any("se" in k for k in state_dict.keys()):
                    from lab.enhanced_self_training_v3.train_enhanced_ocr import EnhancedNomOCRNet
                    model = EnhancedNomOCRNet(num_classes=num_classes).to(device)
                else:
                    from lab.self_training_v2.train_self_training_ocr import NomOCRNet
                    model = NomOCRNet(n_classes=num_classes).to(device)

                model.load_state_dict(state_dict)
                model.eval()

                with torch.no_grad():
                    outputs = model(batch_tensor)
                    probs = F.softmax(outputs, dim=1)
                    top_probs, top_idxs = probs.max(dim=1)

                for i, k in enumerate(valid_keys):
                    c = id_to_char.get(top_idxs[i].item(), "")
                    p = top_probs[i].item()
                    u = f"U+{ord(c):04X}" if c else ""
                    predictions[k] = {"pred_char": c, "unicode": u, "prob": p, "source": Path(model_path).name}
        except Exception as e:
            print(f"[rescue] Cảnh báo: Lỗi khi chạy suy diễn trực tiếp ({e})")

    if not predictions:
        if verbose:
            print(f"[rescue] Không tìm thấy dữ liệu dự đoán -> bỏ qua giải cứu, giữ nguyên nhãn.")
        return 0

    if verbose:
        print(f"[rescue] Tổng số dự đoán đã nạp: {len(predictions):,} ô")

    # Thực hiện giải cứu
    gold_dir = in_csv.parent / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)

    rescued_rv_count = 0
    rescued_syl_count = 0
    rescued_details = []

    target_tiers = {"REVIEW"}
    if rescue_syllable:
        target_tiers.add("SYLLABLE")

    for idx, row in df.iterrows():
        orig_tier = row["tier"]
        if orig_tier not in target_tiers:
            continue

        k = (str(row["book"]), str(row["page"]), int(row["column"]), int(row["nom_idx"]))
        if k not in predictions:
            continue

        pred = predictions[k]
        pred_c = pred["pred_char"]
        prob = pred["prob"]
        syl = str(row.get("syllable", "")).strip().lower()

        # Điều kiện giải cứu an toàn:
        # 1. Ký tự dự đoán phải nằm trong tập ứng viên từ điển của âm Quốc ngữ
        # 2. Xác suất độ tin cậy >= tau
        cand_dict = qn_to_nom.get(syl, set())
        if pred_c and (pred_c in cand_dict) and (prob >= tau):
            crop_filename = f"{row['book']}_{row['page']}_c{int(row['column']):02d}_{int(row['nom_idx']):03d}.png"
            crop_rel_path = f"gold/{crop_filename}"
            crop_abs_path = gold_dir / crop_filename

            md5_hex = ""
            if not crop_abs_path.exists() and (k in crop_lookup) and (X_crops is not None):
                arr = X_crops[crop_lookup[k]]
                im = Image.fromarray(arr)
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                md5_hex = hashlib.md5(png_bytes).hexdigest()
                with open(crop_abs_path, "wb") as f:
                    f.write(png_bytes)
            elif crop_abs_path.exists():
                with open(crop_abs_path, "rb") as f:
                    md5_hex = hashlib.md5(f.read()).hexdigest()

            rule_name = "self_training_rescue" if orig_tier == "REVIEW" else "self_training_syllable_upgrade"

            # Cập nhật thông tin hàng
            df.at[idx, "tier"] = "GOLD"
            df.at[idx, "rule"] = rule_name
            df.at[idx, "label"] = pred_c
            df.at[idx, "unicode"] = pred["unicode"]
            df.at[idx, "image"] = crop_rel_path
            df.at[idx, "label_level"] = "char"
            if md5_hex:
                df.at[idx, "image_md5"] = md5_hex

            if orig_tier == "REVIEW":
                rescued_rv_count += 1
            else:
                rescued_syl_count += 1

            rescued_details.append({
                "book": row["book"],
                "page": row["page"],
                "column": row["column"],
                "nom_idx": row["nom_idx"],
                "orig_tier": orig_tier,
                "syllable": syl,
                "ocr_char_old": row.get("ocr_char", ""),
                "rescued_char": pred_c,
                "unicode": pred["unicode"],
                "prob": round(prob, 4),
                "source": pred.get("source", ""),
            })

    total_rescued = rescued_rv_count + rescued_syl_count

    # Lưu lại file CSV kết quả
    df.to_csv(out_csv, index=False)
    n_gold_after = (df["tier"] == "GOLD").sum()
    n_syl_after = (df["tier"] == "SYLLABLE").sum()
    n_rv_after = (df["tier"] == "REVIEW").sum()

    if verbose:
        print(f"\n✓ ĐÃ GIẢI CỨU THÀNH CÔNG TỔNG CỘNG: {total_rescued:,} Ô LÊN GOLD!")
        print(f"  - Giải cứu từ REVIEW:   +{rescued_rv_count:,} ô")
        if rescue_syllable:
            print(f"  - Nâng cấp từ SYLLABLE: +{rescued_syl_count:,} ô")
        print(f"  - Tập GOLD sau giải cứu:     {n_gold_after:,} (tăng +{total_rescued:,})")
        print(f"  - Tập SYLLABLE sau giải cứu: {n_syl_after:,}")
        print(f"  - Tập REVIEW sau giải cứu:   {n_rv_after:,} (giảm -{rescued_rv_count:,})")
        print(f"✓ Đã cập nhật tệp nhãn: {out_csv}")

    # Báo cáo JSON
    if report_path:
        report = {
            "tau": tau,
            "rescue_syllable": rescue_syllable,
            "so_o_giai_cuu_tong": total_rescued,
            "giai_cuu_tu_review": rescued_rv_count,
            "giai_cuu_tu_syllable": rescued_syl_count,
            "gold_truoc": int(n_gold_before),
            "gold_sau": int(n_gold_after),
            "syllable_truoc": int(n_syl_before),
            "syllable_sau": int(n_syl_after),
            "review_truoc": int(n_rv_before),
            "review_sau": int(n_rv_after),
            "mau_giai_cuu_dau_tien": rescued_details[:30],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"✓ Đã lưu báo cáo giải cứu: {report_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Tự động giải cứu REVIEW bằng Self-Training")
    parser.add_argument("--in", dest="in_csv", default="dataset_out/labels_final.csv", help="Nhãn đầu vào")
    parser.add_argument("--out", dest="out_csv", default="dataset_out/labels_final.csv", help="Nhãn đầu ra")
    parser.add_argument("--crops", default="KhoiB/v3/crops_v3.npz", help="crops.npz")
    parser.add_argument("--dict", default="Dict/QuocNgu_SinoNom.csv", help="Từ điển âm-chữ")
    parser.add_argument("--model", default=None, help="Trọng số PyTorch (.pt)")
    parser.add_argument("--pseudo-csv", nargs="*", default=None, help="Bảng nhãn suy diễn CSV (hỗ trợ nhiều file)")
    parser.add_argument("--tau", type=float, default=0.70, help="Ngưỡng xác suất tin cậy (mặc định 0.70)")
    parser.add_argument("--rescue-syllable", action="store_true", help="Nâng cấp cả các ô SYLLABLE có độ tin cậy cao lên GOLD")
    parser.add_argument("--report", default="dataset_out/self_training_rescue_report.json", help="Báo cáo JSON")
    args = parser.parse_args()

    ret = run_rescue(
        in_csv=Path(args.in_csv),
        out_csv=Path(args.out_csv),
        crops_path=Path(args.crops),
        dict_path=Path(args.dict),
        model_path=Path(args.model) if args.model else None,
        pseudo_csv=args.pseudo_csv,
        tau=args.tau,
        rescue_syllable=args.rescue_syllable,
        report_path=Path(args.report) if args.report else None,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
