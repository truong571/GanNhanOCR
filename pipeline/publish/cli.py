"""CLI `python -m pipeline.publish` — tập công bố theo GOLD chính xác (bước B9 của run_pipeline.sh --publish).

    .venv/bin/python -m pipeline.publish gold-exact [--all-dir dataset/_ALL] [--out <dir>] [--no-files]

30/09: đã XOÁ công cụ công bố cũ tháng 7 (split page-disjoint/LOBO, Frictionless/Croissant, HF Parquet, validate) —
quyết định A-10 không chia tập. Chỉ còn `gold-exact` (tách tập ẢNH = ô gold_exact ok / tập VĂN BẢN, KHÔNG chia tập).
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def cmd_gold_exact(args) -> None:
    """dataset/_ALL/{labels,gold_exact}.csv -> cong_bo/. Bất biến FAIL -> exit 1."""
    from . import gold_exact_release as GR
    try:
        GR.run(Path(args.all_dir), Path(args.out) if args.out else None, check_files=not args.no_files)
    except GR.ReleaseError as e:
        print(f"[công bố] LỖI: {e}")
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.publish", description="Tập công bố theo GOLD chính xác (không chia tập)")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gold-exact", help="tập công bố theo GOLD chính xác (tập ảnh = ô ok; văn bản = phần còn lại)")
    g.add_argument("--all-dir", default=str(REPO / "dataset" / "_ALL"))
    g.add_argument("--out", default=None, help="mặc định <all-dir>/cong_bo")
    g.add_argument("--no-files", action="store_true", help="không kiểm tệp crop chuẩn trên đĩa")
    g.set_defaults(func=cmd_gold_exact)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
