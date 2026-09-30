"""Tự động giải cứu các ô REVIEW bằng mô hình Self-Training / Enhanced OCR.

Tích hợp vào pipeline (sau bước confusion_fix, trước export):
- Nạp checkpoint mô hình OCR Nôm nội bộ (đã học nét bút lông) hoặc bảng nhãn suy diễn.
- Hỗ trợ kết hợp đa nguồn (v2 Self-Training + v3 Enhanced SE-ResNet).
- Quét các ô REVIEW (hoặc tuỳ chọn SYLLABLE) có âm Quốc ngữ hợp lệ trong từ điển.
- Nếu mô hình dự đoán ký tự c với xác suất P >= tau VÀ c thuộc Dict(âm):
  1. Thăng cấp ô thành tier GOLD với rule 'self_training_rescue'.
  2. Xuất ảnh crop từ crops.npz ra dataset_out/gold/ để phục vụ bước export.
  3. Cập nhật nhãn và mã Unicode.

SỬA GỐC VA TÊN TỆP (28/09): trước đây crop giải cứu đặt tên `<book>_<page>_c<cột>_<nom_idx>.png` — TRÙNG khuôn tên
`<book>_<page>_c<cột>_<chỉ số>.png` mà build_dataset dùng cho crop gold/ của ô KHÁC, và mã TÁI DÙNG tệp có sẵn
(`elif crop_abs_path.exists()`) -> 22 ô rescue STT mang ảnh của ô khác (dataset/_ALL/TRUNG_ANH.csv 42 dòng, 21 đường dẫn
dùng chung giữa hai nhãn khác nhau). Nay:
  * tên theo KHOÁ RIÊNG của ô, không thể va với build: `rescue_<book>_<page>_c<cột>_n<nom_idx>_s<syl_idx>.png` (rescue_name);
  * KHÔNG BAO GIỜ tái dùng/ghi đè tệp có sẵn: tệp đã có mà khác byte -> RescueFileConflict (dừng); trùng byte (chạy lại
    trên cùng DS_OUT) -> dùng, ghi nhận `tep_da_co_trung_byte`;
  * ô không có crop trong crops.npz -> KHÔNG giải cứu (GOLD phải có ảnh của CHÍNH ô đó), đếm `bo_qua_khong_co_anh`.
Nhãn/dự đoán/ngưỡng giữ nguyên; chỉ đổi đường ảnh + image_md5 của dòng rescue.
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


RESCUE_PREFIX = "rescue_"


class RescueFileConflict(RuntimeError):
    """Tệp crop giải cứu đã tồn tại với nội dung KHÁC — không bao giờ ghi đè/tái dùng."""


def rescue_name(book, page, column, nom_idx, syl_idx=None) -> str:
    """Tên crop giải cứu theo khoá riêng của ô (không va khuôn `<book>_<page>_c<cột>_<idx>.png` của build_dataset)."""
    try:
        s = f"s{int(syl_idx):03d}"
    except (TypeError, ValueError):
        s = "sx"
    return f"{RESCUE_PREFIX}{book}_{page}_c{int(column):02d}_n{int(nom_idx):03d}_{s}.png"


def write_rescue_png(path: Path, png_bytes: bytes) -> tuple[str, bool]:
    """Ghi crop giải cứu; tệp có sẵn: trùng byte -> (md5, True), khác byte -> RescueFileConflict. Trả (md5, đã_có)."""
    md5_hex = hashlib.md5(png_bytes).hexdigest()
    if path.exists():
        old = path.read_bytes()
        if old != png_bytes:
            raise RescueFileConflict(f"{path} đã tồn tại với nội dung KHÁC (md5 {hashlib.md5(old).hexdigest()} ≠ {md5_hex}) "
                                     "— từ chối ghi đè/tái dùng ảnh (gốc lỗi 22 ô mang ảnh của ô khác). Dựng lại DS_OUT (build --force).")
        return md5_hex, True
    tmp = path.with_suffix(".png.tmp")
    tmp.write_bytes(png_bytes)
    os.replace(tmp, path)
    return md5_hex, False


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
    n_skip_no_img = 0          # đủ điều kiện nhãn nhưng không có crop của CHÍNH ô trong crops.npz -> không giải cứu
    n_same_bytes = 0           # tệp đã có, trùng byte (chạy lại trên cùng DS_OUT)

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
            # 28/09: tên theo khoá riêng của ô + không bao giờ tái dùng tệp có sẵn của ô khác (docstring đầu tệp)
            if X_crops is None or k not in crop_lookup:
                n_skip_no_img += 1
                continue
            crop_filename = rescue_name(row["book"], row["page"], row["column"], row["nom_idx"], row.get("syl_idx"))
            crop_rel_path = f"gold/{crop_filename}"
            crop_abs_path = gold_dir / crop_filename
            im = Image.fromarray(X_crops[crop_lookup[k]])
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            md5_hex, existed = write_rescue_png(crop_abs_path, buf.getvalue())
            n_same_bytes += int(existed)

            rule_name = "self_training_rescue" if orig_tier == "REVIEW" else "self_training_syllable_upgrade"

            # Cập nhật thông tin hàng
            df.at[idx, "tier"] = "GOLD"
            df.at[idx, "rule"] = rule_name
            df.at[idx, "label"] = pred_c
            df.at[idx, "unicode"] = pred["unicode"]
            df.at[idx, "image"] = crop_rel_path
            df.at[idx, "label_level"] = "char"
            if md5_hex:
                df.at[idx, "image_md5"] = md5_hex[:12]   # cùng quy ước 12 hex với build_dataset (01/10: trước ghi đủ 32)

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
        print(f"  - Bỏ qua (không có crop của chính ô trong crops.npz): {n_skip_no_img:,} ô · tệp đã có trùng byte: {n_same_bytes:,}")
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
            "ten_tep": "gold/rescue_<book>_<page>_c<cột>_n<nom_idx>_s<syl_idx>.png (28/09: không va tên build, không tái dùng tệp)",
            "bo_qua_khong_co_anh": int(n_skip_no_img),
            "tep_da_co_trung_byte": int(n_same_bytes),
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
    parser.add_argument("--selftest", action="store_true", help="kiểm tên tệp không va + chặn ghi đè (thư mục tạm, 0 dữ liệu thật)")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(selftest())

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


def selftest() -> int:
    """Thư mục tạm: crop giải cứu KHÔNG va tên build, KHÔNG tái dùng tệp build có sẵn, từ chối ghi đè khác byte, chạy lại trùng
    byte thì dùng, ô không có crop thì không giải cứu. In `RESULT:`."""
    import tempfile
    ok, fail = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)
    chk("ten_khong_va_build", rescue_name("stt2", "page_0012", 3, 5, 0) == "rescue_stt2_page_0012_c03_n005_s000.png"
        and not rescue_name("stt2", "page_0012", 3, 5, 0).startswith("stt2_page_0012_c03_005")
        and rescue_name("stt2", "page_0012", 3, 5, None).endswith("_sx.png"))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        gold = td / "gold"; gold.mkdir()
        # tệp build của ô KHÁC mang đúng tên khuôn cũ (va) -> mã mới không được đụng/tái dùng
        build_png = gold / "stt2_page_0001_c01_002.png"
        build_png.write_bytes(b"BUILD-OTHER-CELL")
        X = np.stack([np.full((8, 8), v, np.uint8) for v in (10, 200, 90)])
        np.savez(td / "crops.npz", X=X, book=np.array(["stt2"] * 3), page=np.array(["page_0001"] * 3),
                 column=np.array([1, 1, 1]), nom_idx=np.array([2, 3, 4]))
        (td / "dict.csv").write_text("syllable,nom\nthì,時\nlà,羅\n", encoding="utf-8")
        pd.DataFrame(dict(book="stt2", page="page_0001", column=1, nom_idx=[2, 3, 4, 9], syl_idx=[0, 1, 2, 3],
                          predicted_nom=["時", "羅", "時", "時"], prob=0.99)).to_csv(td / "pseudo.csv", index=False)
        lab = pd.DataFrame(dict(image="review/x.png", book="stt2", page="page_0001", column=1, nom_idx=[2, 3, 4, 9],
                                syl_idx=[0, 1, 2, 3], syllable=["thì", "là", "thì", "thì"], ocr_char="?", label="?", unicode="U+003F",
                                label_level="char", tier="REVIEW", rule="goc", image_md5="abc123def456"))
        lab.to_csv(td / "labels.csv", index=False)
        args = dict(crops_path=td / "crops.npz", dict_path=td / "dict.csv", pseudo_csv=[td / "pseudo.csv"], verbose=False,
                    report_path=td / "rep.json")
        run_rescue(in_csv=td / "labels.csv", out_csv=td / "out.csv", **args)
        out = pd.read_csv(td / "out.csv", keep_default_na=False)
        rep = json.loads((td / "rep.json").read_text(encoding="utf-8"))
        g = out[out.tier == "GOLD"]
        chk("giai_cuu_3_o_co_crop", len(g) == 3 and rep["bo_qua_khong_co_anh"] == 1)
        chk("anh_moi_tep_rieng", g.image.is_unique and all(i.startswith("gold/rescue_") for i in g.image))
        chk("khong_dung_tep_build", build_png.read_bytes() == b"BUILD-OTHER-CELL"
            and "gold/stt2_page_0001_c01_002.png" not in set(out.image))
        chk("md5_la_cua_crop_o_12hex", all(hashlib.md5((td / i).read_bytes()).hexdigest()[:12] == m and len(m) == 12
                                           for i, m in zip(g.image, g.image_md5)))
        # chạy lại trên cùng thư mục (trùng byte) -> dùng lại, không lỗi
        run_rescue(in_csv=td / "labels.csv", out_csv=td / "out2.csv", **args)
        chk("chay_lai_trung_byte", json.loads((td / "rep.json").read_text(encoding="utf-8"))["tep_da_co_trung_byte"] == 3
            and pd.read_csv(td / "out2.csv", keep_default_na=False).image.tolist() == out.image.tolist())
        # tệp đích có sẵn KHÁC byte -> từ chối ghi đè
        (td / g.image.iloc[0]).write_bytes(b"KHAC")
        try:
            run_rescue(in_csv=td / "labels.csv", out_csv=td / "out3.csv", **args)
            chk("tu_choi_ghi_de_khac_byte", False)
        except RescueFileConflict:
            chk("tu_choi_ghi_de_khac_byte", (td / g.image.iloc[0]).read_bytes() == b"KHAC")
    print(json.dumps(dict(passed=ok, failed=fail), ensure_ascii=False))
    print(f"RESULT: {ok} passed, {len(fail)} failed")
    return 0 if not fail else 1


if __name__ == "__main__":
    main()
