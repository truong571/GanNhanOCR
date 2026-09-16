"""CLI for Giai đoạn 1 — Sửa lỗi đã chứng minh.

    .venv/bin/python -m pipeline.remediation census
    .venv/bin/python -m pipeline.remediation apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import census as census_mod
from . import remediate as remediate_mod

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO / "dataset_out" / "labels.csv"
DEFAULT_OUT = REPO / "dataset_out"


# CỘT CỜ / CHỈ SỐ đọc là CHUỖI (xem chú thích trong _load/run): ô rỗng -> float64 -> '1.0'
DTYPE = {"image_md5": str, "label_in_train": str, "crop_w": str, "crop_h": str,
         # cờ/chỉ số v3 (A-7/A-5/A-6): có thể rỗng ở thế hệ cũ -> pandas ép float "1.0"
         "nom_idx": str, "syl_idx": str, "band_touched": str, "n_ocr": str, "n_qn": str,
         "n_det": str, "l1_support": str, "l1_tie": str, "flank_gold": str,
         "qd01_locked": str, "qd01_excluded": str, "am_da_quyet_ngoai_khoa": str,
         "norm_fixed": str}


def _load(path: str) -> pd.DataFrame:
        # `nan` LÀ MỘT ÂM TIẾNG VIỆT (難). pandas mặc định đọc chuỗi "nan"/"NA"/"null"…
    # thành NaN, nên 3 ô mất hẳn âm khi đi qua bước này (2 trong đó là GOLD trong bộ
    # giao nộp) và 8 mục từ điển biến mất mỗi lần đọc bằng pandas. `na_values=[""]`
    # giữ ô RỖNG vẫn là NaN (s3_cosine cần thế) nhưng cứu mọi chuỗi có nội dung.
    # CỘT CỜ VÀ CỘT KÍCH THƯỚC PHẢI ĐỌC LÀ CHUỖI. Chúng có ô rỗng (tầng SYLLABLE không
    # có label_in_train; hàng REVIEW không có crop_w/h), nên pandas ép cả cột về float64
    # rồi ghi ra '1.0' thay vì '1'. Hậu quả: mọi phép lọc `label_in_train == "1"` trả về
    # RỖNG mà không báo lỗi — đúng lớp lỗi câm đã cắn dự án này nhiều lần.
    return pd.read_csv(path, dtype=DTYPE, keep_default_na=False, na_values=[""])


def cmd_census(args) -> None:
    df = _load(args.labels)
    res = census_mod.run_census(df)
    print("[census]", res.summary())


def cmd_apply(args) -> None:
    df = _load(args.labels)
    out_df, report = remediate_mod.remediate(df, tau_silver=args.tau, s3_demote=args.s3_demote)
    out_csv = Path(args.out) / "labels_remediated.csv"
    out_json = Path(args.out) / "remediation_report.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("[apply]", report.summary())
    print(f"[apply] tier before: {report.tier_before}")
    print(f"[apply] tier after : {report.tier_after}")
    print(f"[apply] -> {out_csv}")
    print(f"[apply] -> {out_json}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.remediation",
                                description="Giai đoạn 1 — Sửa lỗi đã chứng minh")
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("census", help="report the duplicate-crop census")
    c.set_defaults(func=cmd_census)

    a = sub.add_parser("apply", help="apply the remediation and write outputs")
    a.add_argument("--s3-demote", action="store_true",
                   help="bật lại phép hạ cấp similar-bridge theo S3 (TẮT mặc định: "
                        "S3 error-AUC 0,566 CI [0,459-0,672], không phân biệt đúng/sai)")
    a.add_argument("--tau", type=float, default=remediate_mod.TAU_SILVER,
                   help="S3 cosine threshold for demoting similar-bridge GOLD")
    a.set_defaults(func=cmd_apply)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
