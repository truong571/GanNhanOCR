"""Tách mẻ audit GỘP thành các mẻ RIÊNG theo tier (GOLD / SILVER / SYLLABLE / REVIEW).

Đọc `manifest.jsonl` của mẻ gộp rồi dựng lại HTML + manifest cho từng tier, giữ NGUYÊN
`item_id`, `stratum`, `design_weight`, `stratum_N` — nên `report_combined` / `estimate`
chạy trên các mẻ tách vẫn ra đúng con số như mẻ gộp.

CẢNH BÁO PHƯƠNG PHÁP
--------------------
Mẻ gộp tồn tại vì một lý do ĐO ĐƯỢC: chấm từng tier riêng thì người chấm biết mình đang ở
tier nào, và tiêu chí trôi giữa các buổi (4,2% -> 16% -> 35% trên cùng dân số, 2026-08-04).
Bản tách này dùng để XEM / đối chiếu / chấm bổ sung theo tier, KHÔNG dùng để lấy lại con
số precision so sánh giữa các tier — số đó phải lấy từ mẻ gộp.

Ô lặp (`stratum == "__repeat__"`) đi theo tier của bản gốc; κ nội tại chỉ có nghĩa khi
chấm trong MỘT dòng, nên mặc định `--drop-repeats` là bật.

REVIEW không có trong mẻ gộp (mẻ gộp chỉ rút GOLD/SILVER/SYLLABLE) nên tách ra sẽ rỗng —
muốn có REVIEW thì phải rút một mẻ mới, đây không phải việc của bộ tách.

CHẠY
----
    .venv/bin/python -m pipeline.ground_truth.split_combined
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import audit_grid
from .cli import _load_config, _paths

REPO = Path(__file__).resolve().parents[2]
GT_DIR = REPO / "dataset_out" / "ground_truth"
DEFAULT_SRC = GT_DIR / "audit_combined"
TIERS = ("GOLD", "SILVER", "SYLLABLE", "REVIEW")


def load_manifest(src: Path) -> pd.DataFrame:
    """Manifest không ghi `audit_order` — nhưng các dòng ĐÃ được xếp theo thứ tự chấm,
    nên số dòng chính là thứ tự đó. Giữ lại để mẻ tách không xáo trộn lại."""
    rows = [json.loads(ln) for ln in
            (src / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    df = pd.DataFrame(rows)
    df["audit_order"] = range(len(df))
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.split_combined")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(GT_DIR), help="thư mục cha; mỗi tier một thư mục con")
    ap.add_argument("--prefix", default="audit_split_", help="tiền tố tên thư mục mỗi tier")
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--keep-repeats", action="store_true",
                    help="giữ ô lặp (mặc định: bỏ — κ chỉ có nghĩa trong mẻ gộp)")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = một file mỗi tier")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    args = ap.parse_args(argv)

    src = Path(args.src)
    df = load_manifest(src)
    if not args.keep_repeats:
        n0 = len(df)
        df = df[df.get("stratum").astype(str) != "__repeat__"]
        print(f"[tách] bỏ {n0 - len(df)} ô lặp (dùng --keep-repeats để giữ)")

    cfg = _load_config(Path(args.config))
    paths = _paths(cfg)
    qn_dict = None
    try:
        from core.text.dictionary import load_qn_to_nom
        if paths["qn_dict"].exists():
            qn_dict = load_qn_to_nom(str(paths["qn_dict"]))
    except Exception as e:                                   # noqa: BLE001
        print(f"[tách] bỏ qua từ điển QN ({e})")

    summary = {}
    for tier in [t.strip().upper() for t in args.tiers.split(",") if t.strip()]:
        part = df[df["tier"].astype(str) == tier].sort_values("audit_order")
        if part.empty:
            print(f"[tách] {tier}: 0 ô trong mẻ nguồn — bỏ qua")
            continue
        out_dir = Path(args.out) / f"{args.prefix}{tier.lower()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        stat = audit_grid.build_audit(
            part.reset_index(drop=True),
            dataset_dir=paths["dataset_dir"],
            prepared_dir=paths["prepared_dir"],
            fd_dir=paths["fd_dir"],
            out_html=out_dir / "audit.html",
            out_manifest=out_dir / "manifest.jsonl",
            qn_dict=qn_dict,
            font_path=paths["font"],
            with_context=not args.no_context,
            title=f"Audit nhãn · tier {tier}",
            batch_size=args.batch_size or None,
            mode="label_only",
        )
        meta = {
            "tier": tier, "source_batch": str(src.relative_to(REPO)),
            "n_items": stat["items"], "repeats_kept": bool(args.keep_repeats),
            "missing_reference_glyph": stat["missing_reference_glyph"],
            "missing_context": stat["missing_context"],
            "warning": ("Mẻ tách theo tier — người chấm BIẾT tier, nên precision đo trên "
                        "mẻ này KHÔNG so sánh được giữa các tier. Số công bố lấy từ "
                        "audit_combined."),
        }
        (out_dir / "plan.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[tier] = stat["items"]
        print(f"[tách] {tier}: {stat['items']} ô -> {out_dir} "
              f"(thiếu glyph {stat['missing_reference_glyph']}, "
              f"thiếu ngữ cảnh {stat['missing_context']})")
    print(f"[tách] xong: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
