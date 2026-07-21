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


def measure_gold_precision(final: pd.DataFrame) -> dict | None:
    """Precision GOLD sau fix, neo trên verdicts_reanchored.csv (join theo image)."""
    vp = REPO / "dataset_out" / "ground_truth" / "verdicts_reanchored.csv"
    if not vp.exists():
        return None
    v = pd.read_csv(vp, dtype=str)
    v = v[v["status"] == "matched"]
    tier_by_img = dict(zip(final["image"], final["tier"]))
    v = v.assign(tier_now=v["image_new"].map(tier_by_img))
    g = v[(v["tier_now"] == "GOLD") & (v["verdict"] != "unsure")]
    n = len(g)
    correct = int((g["verdict"] == "correct").sum())
    return {"gold_audited": n, "correct": correct,
            "precision": round(correct / n, 4) if n else None,
            "wrong": n - correct}


def run(in_csv: Path, out_csv: Path, fixes_yaml: Path, measure: bool) -> dict:
    df = pd.read_csv(in_csv, dtype={"image_md5": str})
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
    (out_csv.parent / "confusion_fix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))

    print("=" * 60)
    print(f" CONFUSION-FIX: {len(fixes)} fix, demote tổng {report['total_demoted']} crop")
    for x in log:
        print(f"   {x['syllable']}→{x['label']} demote {x['demoted']} → {x['to_tier']} "
              f"{x['by_book']}")
    print(f" tier before: {before}")
    print(f" tier after : {after}")
    if measure and report.get("precision_gold_after"):
        b, a = report["precision_gold_before"], report["precision_gold_after"]
        if b.get("precision") is not None and a.get("precision") is not None:
            print(f" precision GOLD: {b['precision']:.4f} (n={b['gold_audited']}, sai {b['wrong']}) "
                  f"→ {a['precision']:.4f} (n={a['gold_audited']}, sai {a['wrong']})")
        else:
            # precision = None khi CHƯA có verdict NGƯỜI cho GOLD (không có audit để đo).
            # Trước đây format None -> TypeError làm CHẾT bước confusion sau khi đã ghi
            # labels_final.csv, kéo pipeline dừng trước publish.
            print(" precision GOLD: chưa đo được (chưa có verdict NGƯỜI cho GOLD).")
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
