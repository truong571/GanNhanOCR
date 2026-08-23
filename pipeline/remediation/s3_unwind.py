"""Gỡ tín hiệu thị giác S3 khỏi mọi quyết định tier — bước tất định, không đoán.

VÌ SAO
------
⚠️ CẬP NHẬT 2026-08-22 — CÁCH PHÁT BIỂU ĐÃ SỬA. Bộ 826 phán quyết dưới đây KHÔNG phải
người chấm (xuất xứ không truy nguyên được, verdict thô đã mất — xem
docs/KE_HOACH_TONG_THE_2026-08-22.md §0). Quyết định gỡ S3 VẪN GIỮ, nhưng lý do đúng là
"CHƯA CHỨNG MINH ĐƯỢC", KHÔNG phải "ĐÃ BÁC BỎ", vì ba lẽ:
  (a) lớp âm chỉ 24 ca -> Hanley-McNeil cho AUC nhỏ nhất phát hiện được = 0,607, nên tín
      hiệu có AUC thật 0,55-0,60 KHÔNG THỂ bị phát hiện bằng mẻ đó;
  (b) đo trên GOLD, nơi S3 không tham gia gán nhãn và chỉ 7,7% số ô có điểm S3;
  (c) 617/826 (74,7%) crop nằm trong split TRAIN của chính lần train ArcFace.
Các số dưới đây giữ lại làm LỊCH SỬ, không còn tư cách bằng chứng.

Đo ngày 2026-08-19 trên 826 verdict (`ArcFace/eval_human_verdicts.py`):

    ArcFace retrain (Kaggle, Sub-center K=3 + SAM)  error-AUC = 0,577  CI95 [0,442-0,706]
    S3 cũ (nom-embed/best.pt)                       error-AUC = 0,566  CI95 [0,459-0,672]

CI của cả hai đều CHỨA 0,5 → CHƯA chứng minh được tín hiệu thị giác phân biệt đúng/sai.
Val head-top1 tăng 0,23 -> 0,806 mà AUC bắt lỗi đứng yên: encoder giỏi XẾP HẠNG chứ
không PHÁT HIỆN SAI. Vậy S3 không được quyền phong hay hạ tier nữa.

BỐN VIỆC, KHÔNG CÓ VIỆC THỨ NĂM
--------------------------------
1. TRẢ VỀ GOLD các ô `s1_inter_s2_similar|demoted_lowcos_s3`.
   Chúng đã thoả S1 ∩ S2 (giao từ điển qua cầu nối tự dạng) và bị hạ CHỈ vì cosine thấp
   — tiêu chí chưa chứng minh được. (Căn cứ cũ là 97,6% (40/41) vs 98,0% (737/752):
   🔴 CẢ HAI ĐÃ HUỶ 2026-08-22 cùng bộ verdict. Quyết định giữ nguyên vì lý do quản
   trị — không cho một tín hiệu chưa chứng minh được quyền HẠ cấp — chứ không còn
   dựa trên hai con số đó.) Mỗi ô trả về vẫn được gắn cờ `readmitted_from_s3_demotion`
   để người dùng dataset lọc được.

2. ĐỔI TIER SILVER -> `SILVER_uncalibrated`. KHÔNG xoá dòng nào, KHÔNG đổi nhãn nào.
   Toàn bộ SILVER do S3 quyết và có ĐÚNG 0 verdict người trên 10.890 ô. Tên tier mới
   nằm ngoài `USABLE_TIERS` nên tự động rơi khỏi bộ giao nộp, trong khi dòng vẫn còn
   trong file để rút mẫu chấm tay (200 ô, Giai đoạn 2) và để phục hồi nếu đo ra tốt.

3. ĐỔI LÝ DO `below_visual_threshold` -> `no_s1_inter_s2`. Tier vẫn REVIEW, chỉ sửa
   cái tên cho đúng bản chất: các ô này ở REVIEW vì KHÔNG có giao S1∩S2, chứ không
   phải vì "điểm thị giác thấp" — lý do cũ viện đến một ngưỡng không còn hiệu lực.

4. CHỐT CHẶN LỚP CONFUSION (thêm 2026-08-23). Không cặp (âm, chữ) nào trong
   `config/confusion_fixes.yaml` được phép nằm ở tier dùng được sau bước này.
   VÌ SAO CẦN: 48 ô 㝵/"người" đã lọt vào bộ công bố theo đúng đường này —
   bước 4 hạ chúng xuống REVIEW theo S3, bước 5 `confusion_fix` chỉ demote tier
   {GOLD, SILVER} nên BỎ QUA (chúng đang ở REVIEW), rồi bước 6 readmit trả thẳng
   về GOLD. Ba chốt an toàn cũ chỉ kiểm số dòng / cột label / hậu tố demote nên
   không ai thấy. Bước này CHỮA (demote) chứ không chỉ chặn, để chạy trên tệp đã
   hỏng sẵn cũng ra kết quả sạch, và bất biến ở cuối hàm trở thành assertion thật.

BƯỚC NÀY KHÔNG SỬA MỘT CHỮ NÀO
------------------------------
Không ô nào đổi `label` (assertion ở cuối hàm). Cái bước này mua là TƯ CÁCH CÔNG BỐ,
không phải độ chính xác — đừng trình bày nó như một phép làm sạch. Tác động lên
precision KHÔNG ĐO ĐƯỢC: hiện chưa có phán quyết người nào.

    .venv/bin/python -m pipeline.remediation.s3_unwind --in dataset_out/labels_final.csv \
        --out dataset_out/labels_s3free.csv --report dataset_out/s3_unwind_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

__all__ = ["DEMOTE_SUFFIX", "SILVER_UNCAL", "unwind", "load_confusion_pairs"]

DEMOTE_SUFFIX = "|demoted_lowcos_s3"
SILVER_UNCAL = "SILVER_uncalibrated"
OLD_REVIEW_REASON = "below_visual_threshold"
NEW_REVIEW_REASON = "no_s1_inter_s2"
FLAG_COL = "readmitted_from_s3_demotion"
USABLE_TIERS = {"GOLD", "SILVER", "SILVER_uncalibrated", "SYLLABLE"}
DEFAULT_FIXES = Path(__file__).resolve().parents[2] / "config" / "confusion_fixes.yaml"


def load_confusion_pairs(path: Path | str | None = None) -> list[dict]:
    """Đọc các cặp (syllable, label) đã chứng minh sai hệ thống từ confusion_fixes.yaml.

    Trả [] nếu tệp không tồn tại — KHÔNG ném lỗi, vì `unwind` vẫn phải chạy được trên
    một cây chỉ có dữ liệu. Nhưng preflight của run_pipeline.sh đã bắt thiếu tệp này.
    """
    import yaml
    p = Path(path) if path is not None else DEFAULT_FIXES
    if not p.exists():
        return []
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return [
        {"syllable": str(f["syllable"]).lower(), "label": str(f["label"]),
         "reason": str(f.get("reason", "demote"))}
        for f in (cfg.get("fixes") or [])
        if f.get("action") == "demote" and f.get("syllable") and f.get("label")
    ]


def _confusion_mask(df: pd.DataFrame, pairs: list[dict]):
    """Mặt nạ các dòng khớp BẤT KỲ cặp confusion nào (so trên cột `label`)."""
    m = pd.Series(False, index=df.index)
    if not pairs or "syllable" not in df.columns or "label" not in df.columns:
        return m          # bảng không đủ cột để phán -> không chặn gì (test tổng hợp)
    syl = df["syllable"].fillna("").astype(str).str.lower()
    lab = df["label"].fillna("").astype(str)
    for fx in pairs:
        m |= (syl == fx["syllable"]) & (lab == fx["label"])
    return m


def unwind(df: pd.DataFrame,
           confusion_pairs: list[dict] | None = None) -> tuple[pd.DataFrame, dict]:
    """Trả (bảng đã gỡ S3, báo cáo). Tất định, không phụ thuộc thứ tự dòng.

    `confusion_pairs=None` -> tự nạp từ config/confusion_fixes.yaml (mặc định AN TOÀN).
    Truyền [] để tắt hẳn chốt chặn (chỉ dùng trong test).
    """
    if confusion_pairs is None:
        confusion_pairs = load_confusion_pairs()
    out = df.copy()
    rule = out["rule"].fillna("").astype(str)

    # --- 1. trả về GOLD -------------------------------------------------
    readmit = rule.str.endswith(DEMOTE_SUFFIX)
    if FLAG_COL not in out.columns:
        out[FLAG_COL] = ""
    out.loc[readmit, FLAG_COL] = "1"
    out.loc[readmit, "rule"] = rule[readmit].str.slice(0, -len(DEMOTE_SUFFIX))
    out.loc[readmit, "tier"] = "GOLD"

    # --- 2. SILVER -> SILVER_uncalibrated -------------------------------
    silver = out["tier"] == "SILVER"
    out.loc[silver, "tier"] = SILVER_UNCAL

    # --- 4. CHỐT CHẶN LỚP CONFUSION (thêm 2026-08-23) --------------------
    # Không lớp (âm, chữ) nào ĐÃ CHỨNG MINH SAI HỆ THỐNG được phép ở tier dùng được
    # sau bước này. Trước đó 48 ô 㝵/"người" lọt vào bộ công bố theo đúng đường này:
    #   bước 4 hạ chúng xuống REVIEW theo S3
    #   -> bước 5 confusion_fix chỉ demote tier {GOLD, SILVER} nên BỎ QUA (chúng đang REVIEW)
    #   -> bước 6 readmit trả thẳng về GOLD, không ai kiểm lại.
    # Ở đây CHỮA chứ không chỉ chặn, để chạy trên tệp đã hỏng sẵn cũng ra kết quả sạch.
    conf = _confusion_mask(out, confusion_pairs)
    demote = conf & out["tier"].isin(USABLE_TIERS)
    by_pair: dict[str, int] = {}
    if demote.any():
        syl_l = out["syllable"].fillna("").astype(str).str.lower()
        lab_s = out["label"].fillna("").astype(str)
        for fx in confusion_pairs:
            m = demote & (syl_l == fx["syllable"]) & (lab_s == fx["label"])
            if m.any():
                out.loc[m, "rule"] = "confusion_fix:" + fx["reason"]
                out.loc[m, "label_level"] = ""
                by_pair[f"{fx['label']}/{fx['syllable']}"] = int(m.sum())
        out.loc[demote, "tier"] = "REVIEW"
        out.loc[demote, FLAG_COL] = ""      # không được mang cờ "đã trả về GOLD"

    # --- 3. đổi tên lý do REVIEW ----------------------------------------
    reason = out["rule"].fillna("").astype(str)
    renamed = reason.eq(OLD_REVIEW_REASON)
    out.loc[renamed, "rule"] = NEW_REVIEW_REASON

    # KIỂM CHỨNG: không dòng nào bị thêm/mất, không nhãn nào bị đụng.
    if len(out) != len(df):
        raise SystemExit("số dòng đổi — dừng")
    if not out["label"].fillna("").equals(df["label"].fillna("")):
        raise SystemExit("cột label bị đụng — dừng, bước này không được sửa nhãn")
    if (out["rule"].fillna("").astype(str).str.contains(DEMOTE_SUFFIX.lstrip("|"))).any():
        raise SystemExit("còn sót hậu tố demote — dừng")
    # BẤT BIẾN MỚI: sau bước này không lớp confusion nào còn nằm ở tier dùng được.
    leaked = _confusion_mask(out, confusion_pairs) & out["tier"].isin(USABLE_TIERS)
    if leaked.any():
        got = out.loc[leaked, ["label", "syllable", "tier"]].value_counts().to_dict()
        raise SystemExit(f"BẤT BIẾN HỎNG — lớp confusion còn ở tier dùng được: {got}")

    report = {
        "n_rows": int(len(out)),
        "readmitted_to_gold": int(readmit.sum()),
        "confusion_demoted_after_unwind": int(demote.sum()),
        "confusion_demoted_by_pair": by_pair,
        "silver_renamed": int(silver.sum()),
        "review_reason_renamed": int(renamed.sum()),
        "tier_before": {k: int(v) for k, v in df["tier"].value_counts().items()},
        "tier_after": {k: int(v) for k, v in out["tier"].value_counts().items()},
        "evidence": {
            "arcface_retrain_error_auc": 0.577,
            "arcface_retrain_ci95": [0.442, 0.706],
            "s3_old_error_auc": 0.566,
            "s3_old_ci95": [0.459, 0.672],
            "n_verdicts": 826,
            "provenance": "UNVERIFIED_machine_graded__withdrawn_2026-08-22",
            "how_to_state_it": ("CHƯA CHỨNG MINH ĐƯỢC, không phải ĐÃ BÁC BỎ: lớp âm 24 ca "
                                "-> AUC nhỏ nhất phát hiện được 0,607; đo trên GOLD nơi S3 "
                                "không gán nhãn; 74,7% crop rò rỉ vào TRAIN."),
            "rule_precision_similar": "40/41 = 97.6% CI95 [87.1, 99.9]",
            "rule_precision_direct": "737/752 = 98.0% CI95 [96.7, 98.9]",
            "note": ("Bước này KHÔNG sửa nhãn nào. Nó gỡ quyền quyết định của một tín "
                     "hiệu có CI chứa 0,5, và tách phần chưa ai đo ra khỏi bộ công bố."),
        },
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(prog="pipeline.remediation.s3_unwind")
    ap.add_argument("--in", dest="src", default=str(repo / "dataset_out/labels_final.csv"))
    ap.add_argument("--out", default=str(repo / "dataset_out/labels_s3free.csv"))
    ap.add_argument("--report", default=str(repo / "dataset_out/s3_unwind_report.json"))
    ap.add_argument("--apply", action="store_true", help="thiếu cờ này = chỉ in, không ghi")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.src, dtype=str, low_memory=False)
    out, rep = unwind(df)

    print(f"[s3-unwind] {rep['n_rows']:,} dòng")
    print(f"[s3-unwind] trả về GOLD          : {rep['readmitted_to_gold']:,} ô "
          f"(gắn cờ {FLAG_COL})")
    print(f"[s3-unwind] SILVER -> {SILVER_UNCAL}: {rep['silver_renamed']:,} ô "
          f"(rơi khỏi bộ giao nộp, KHÔNG xoá)")
    print(f"[s3-unwind] đổi lý do REVIEW     : {rep['review_reason_renamed']:,} ô "
          f"({OLD_REVIEW_REASON} -> {NEW_REVIEW_REASON})")
    for t in sorted(set(rep["tier_before"]) | set(rep["tier_after"])):
        b, a = rep["tier_before"].get(t, 0), rep["tier_after"].get(t, 0)
        if b != a:
            print(f"           {t:22} {b:>7,} -> {a:>7,}  ({a - b:+,})")

    if args.apply:
        out.to_csv(args.out, index=False, encoding="utf-8")
        Path(args.report).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(f"[s3-unwind] -> {args.out}")
        print(f"[s3-unwind] -> {args.report}")
    else:
        print("[thử] chưa ghi gì — thêm --apply để ghi thật")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
