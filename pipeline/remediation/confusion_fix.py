"""Stage 6 — CONFUSION-FIX: sửa có mục tiêu các confusion hệ thống đã chứng minh.

Đọc labels_remediated.csv (đầu ra Stage 3 defect-remediation) + config/confusion_fixes.yaml
→ áp mỗi fix (demote các dòng (syllable,label) khớp) → ghi labels_final.csv (bản công bố).
Hàm THUẦN, idempotent: labels.csv → [remediation apply] → labels_remediated.csv → [confusion_fix]
→ labels_final.csv. KHÔNG sửa tay, KHÔNG remap codepoint (chỉ demote tier).

Chạy:
  .venv/bin/python -m pipeline.remediation.confusion_fix
  .venv/bin/python -m pipeline.remediation.confusion_fix --measure   # kèm đo lại precision GOLD
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO / "dataset_out" / "labels_remediated.csv"
DEFAULT_OUT = REPO / "dataset_out" / "labels_final.csv"
DEFAULT_FIXES = REPO / "config" / "confusion_fixes.yaml"
DEMOTABLE = {"GOLD", "SILVER"}


def apply_fixes(df: pd.DataFrame, fixes: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    log = []
    for fx in fixes:
        if fx.get("action") != "demote":
            continue
        syl = str(fx["syllable"]).lower()
        lab = str(fx["label"])
        to_tier = fx.get("to_tier", "REVIEW")
        mask = (out["syllable"].astype(str).str.lower() == syl) \
            & (out["label"].astype(str) == lab) \
            & (out["tier"].isin(DEMOTABLE))
        n = int(mask.sum())
        by_book = out.loc[mask, "book"].value_counts().to_dict() if n else {}
        out.loc[mask, "rule"] = "confusion_fix:" + str(fx.get("reason", "demote"))
        out.loc[mask, "tier"] = to_tier
        # nhãn char không còn tin -> hạ label_level như REVIEW (không xoá cột để truy vết)
        if to_tier == "REVIEW":
            out.loc[mask, "label_level"] = ""
        log.append({"syllable": syl, "label": lab, "to_tier": to_tier,
                    "demoted": n, "by_book": by_book, "reason": fx.get("reason")})
    return out, log


_BOOK_ALIAS = re.compile(r"(^|/)yen(\d+)_")


def normalize_image_key(path: str) -> str:
    """Chuẩn hoá tiền tố sách đời cũ `yen*` -> `stt*` trước khi join.

    `verdicts_reanchored.csv` ghi `gold/yen11_page_0018_c05_094.png` trong khi bộ nhãn
    dùng `gold/stt11_...`. Join thô khớp 0/825 nên `--measure` trả `null` TRONG IM LẶNG
    suốt nhiều lần chạy. Sau khi chuẩn hoá: 816/825.
    """
    return _BOOK_ALIAS.sub(lambda m: f"{m.group(1)}stt{m.group(2)}_", str(path))


# Đội chấm ngoài trả verdict về đây, kèm khai xuất xứ. Cùng giao ước với apply_verdicts.py.
NGUON_VERDICT = ("re-dataset", "dataset")
PROV_FILE = "NGUOI_CHAM.md"

# Tệp ĐỜI CŨ đã bị vô hiệu: 846 phán quyết trong đó là MÁY chấm (trùng 846/846 item_id với
# audit_gold/audit_gold.jsonl), verdict thô gốc đã mất. Đã xoá khỏi đĩa ở d867bcc289 nhưng
# VẪN CÒN trong lịch sử git — ai `git checkout` một bản cũ là nó sống lại. Thấy là TỪ CHỐI.
VERDICT_DOI_CU = REPO / "dataset_out" / "human_audit" / "verdicts_reanchored.csv"


def _tim_verdict_nguoi() -> tuple[Path, Path] | None:
    """Chỉ nhận verdict khi CÓ CẢ khai xuất xứ đi kèm — thiếu là coi như không có."""
    for d in NGUON_VERDICT:
        v, prov = REPO / d / "verdicts.csv", REPO / d / PROV_FILE
        if v.exists() and prov.exists():
            return v, prov
    return None


def measure_gold_precision(final: pd.DataFrame) -> dict | None:
    """Precision GOLD, neo trên verdict NGƯỜI (<bộ>/verdicts.csv + NGUOI_CHAM.md).

    Trả None khi CHƯA có verdict người — và đó là câu trả lời ĐÚNG, không phải hỏng hóc.

    Dự án này đã một lần nhầm phán quyết MÁY thành phán quyết NGƯỜI rồi phải huỷ sạch số
    liệu dựng trên đó (precision GOLD 97,98%, Fisher p=5,4e-8, κ=0,13 — vô giá trị hết).
    Nên ở đây thà KHÔNG có số còn hơn có một số không truy được xuất xứ: hàm này không
    bao giờ tự bịa nguồn thay, và từ chối thẳng tệp đời cũ nếu nó quay lại.
    """
    if VERDICT_DOI_CU.exists():
        print(f" 🔴 THẤY LẠI {VERDICT_DOI_CU.relative_to(REPO)} — tệp này là MÁY chấm, đã bị")
        print("    vô hiệu. TỪ CHỐI đo precision từ nó. Xoá nó, hoặc dùng verdict người mới.")
        return None
    tim = _tim_verdict_nguoi()
    if tim is None:
        return None
    vfile, prov = tim
    from pipeline.remediation.apply_verdicts import doc_verdict_phang   # nạp muộn: tránh vòng
    vmap = doc_verdict_phang(vfile)
    tier_by_img = {normalize_image_key(k): t for k, t in zip(final["image"], final["tier"])}
    khop = {k: v for k, v in ((normalize_image_key(k), v) for k, v in vmap.items())
            if k in tier_by_img}
    g = {k: v for k, v in khop.items() if tier_by_img[k] == "GOLD" and v != "unsure"}
    n = len(g)
    correct = sum(1 for v in g.values() if v == "correct")
    return {"gold_audited": n, "correct": correct,
            "precision": round(correct / n, 4) if n else None,
            "wrong": n - correct,
            "joined": len(khop), "verdicts": len(vmap),
            "provenance": f"{vfile.relative_to(REPO)} + {prov.name}"}


# CỘT CỜ / CHỈ SỐ đọc là CHUỖI (xem chú thích trong _load/run): ô rỗng -> float64 -> '1.0'
DTYPE = {"image_md5": str, "label_in_train": str, "crop_w": str, "crop_h": str,
         # cờ/chỉ số v3 (A-7/A-5/A-6): có thể rỗng ở thế hệ cũ -> pandas ép float "1.0"
         "nom_idx": str, "syl_idx": str, "band_touched": str, "n_ocr": str, "n_qn": str,
         "n_det": str, "l1_support": str, "l1_tie": str, "flank_gold": str,
         "qd01_locked": str, "qd01_excluded": str, "am_da_quyet_ngoai_khoa": str,
         "norm_fixed": str}


def run(in_csv: Path, out_csv: Path, fixes_yaml: Path, measure: bool) -> dict:
    df = pd.read_csv(in_csv, dtype=DTYPE, keep_default_na=False, na_values=[""])
    cfg = yaml.safe_load(fixes_yaml.read_text()) if fixes_yaml.exists() else {}
    fixes = (cfg or {}).get("fixes", [])
    before = df["tier"].value_counts().to_dict()
    final, log = apply_fixes(df, fixes)
    after = final["tier"].value_counts().to_dict()
    final.to_csv(out_csv, index=False)

    report = {"n_fixes": len(fixes), "total_demoted": sum(x["demoted"] for x in log),
              "tier_before": before, "tier_after": after, "fixes": log}
    if measure:
        # precision GOLD TRƯỚC (từ labels_remediated) vs SAU (labels_final)
        base = measure_gold_precision(df.assign(image=df["image"]))
        post = measure_gold_precision(final)
        report["precision_gold_before"] = base
        report["precision_gold_after"] = post
        # null TRẦN không phân biệt được "đúng theo thiết kế" với "đã sập". Ghi luôn lý do
        # để ai đọc artifact về sau không phải đoán.
        report["precision_gold_ly_do"] = (
            "chua_co_verdict_nguoi: thieu " + "/".join(NGUON_VERDICT) + "/verdicts.csv + "
            + PROV_FILE if post is None else "do_tren_verdict_nguoi")
    (out_csv.parent / "confusion_fix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))

    print("=" * 60)
    print(f" CONFUSION-FIX: {len(fixes)} fix, demote tổng {report['total_demoted']} crop")
    for x in log:
        print(f"   {x['syllable']}→{x['label']} demote {x['demoted']} → {x['to_tier']} "
              f"{x['by_book']}")
    print(f" tier before: {before}")
    print(f" tier after : {after}")
    if measure:
        # LUÔN nói một câu. Trước đây nhánh này gác bằng `report.get(...)` nên khi CHƯA có
        # verdict thì cả khối bị bỏ qua: report ăn thêm hai trường null mà màn hình im lặng.
        b, a = report.get("precision_gold_before"), report.get("precision_gold_after")
        if a and b and a.get("precision") is not None and b.get("precision") is not None:
            print(f" precision GOLD: {b['precision']:.4f} (n={b['gold_audited']}, sai {b['wrong']}) "
                  f"→ {a['precision']:.4f} (n={a['gold_audited']}, sai {a['wrong']})")
            print(f" join khớp {a['joined']}/{a['verdicts']} verdict | xuất xứ: {a['provenance']}")
        elif a:
            print(f" precision GOLD: có {a['verdicts']} verdict, join khớp {a['joined']}, "
                  f"nhưng 0 ô GOLD chấm được → chưa đo.")
        else:
            print(" precision GOLD: CHƯA ĐO ĐƯỢC — chưa có verdict NGƯỜI "
                  f"({'/'.join(NGUON_VERDICT)}/verdicts.csv + {PROV_FILE}).")
            print("   null trong report là câu trả lời ĐÚNG, không phải lỗi.")
    print(f" -> {out_csv}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.remediation.confusion_fix")
    ap.add_argument("--in", dest="in_csv", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--fixes", default=str(DEFAULT_FIXES))
    ap.add_argument("--measure", action="store_true", help="đo lại precision GOLD trước/sau")
    args = ap.parse_args(argv)
    run(Path(args.in_csv), Path(args.out), Path(args.fixes), args.measure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
