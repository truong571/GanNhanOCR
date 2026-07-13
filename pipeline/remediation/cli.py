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


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"image_md5": str})


def cmd_census(args) -> None:
    df = _load(args.labels)
    res = census_mod.run_census(df)
    print("[census]", res.summary())


def cmd_apply(args) -> None:
    df = _load(args.labels)
    out_df, report = remediate_mod.remediate(df, tau_silver=args.tau)
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
