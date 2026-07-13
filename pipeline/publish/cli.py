"""CLI for Giai đoạn 3 — Công bố đạt chuẩn quốc tế.

    .venv/bin/python -m pipeline.publish all           # split -> metadata -> datasheet -> validate
    .venv/bin/python -m pipeline.publish split
    .venv/bin/python -m pipeline.publish metadata
    .venv/bin/python -m pipeline.publish datasheet
    .venv/bin/python -m pipeline.publish export --sample 200
    .venv/bin/python -m pipeline.publish validate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import datasheet as ds_mod
from . import export as export_mod
from . import hashing, metadata as md_mod, splits as split_mod, validate as val_mod

REPO = Path(__file__).resolve().parents[2]
DATASET_OUT = REPO / "dataset_out"
RELEASE = DATASET_OUT / "release"


def _labels_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    rem = DATASET_OUT / "labels_remediated.csv"          # prefer the Phase-1 output
    return rem if rem.exists() else DATASET_OUT / "labels.csv"


def _load(args) -> pd.DataFrame:
    return pd.read_csv(_labels_path(args.labels), dtype={"image_md5": str})


def _stats(df: pd.DataFrame) -> dict:
    usable = df["tier"].isin(split_mod.USABLE_TIERS)
    return {
        "n_total": len(df),
        "n_usable": int(usable.sum()),
        "n_classes": len(export_mod.class_names(df)),
        "n_books": df["book"].nunique(),
        "n_pages": df.apply(lambda r: f"{r['book']}|{r['page']}", axis=1).nunique(),
        "tier_counts": {str(k): int(v) for k, v in df["tier"].value_counts().items()},
        "quarantined": int((df["tier"] == "QUARANTINE").sum()),
    }


def _published_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Assign page-disjoint splits and return (full_df_with_split, crops_csv_frame)."""
    split, rep = split_mod.assign_page_disjoint(df)
    df = df.copy()
    df["split"] = split
    return df, rep


def _crops_csv(df: pd.DataFrame) -> pd.DataFrame:
    img = df["image"].fillna("").astype(str)
    rows = df[img.str.len() > 0].copy()
    cols = [f.name for f in md_mod.CROP_FIELDS if f.name in rows.columns]
    return rows[cols]


# --------------------------------------------------------------------------- #
def cmd_split(args) -> None:
    df = _load(args)
    df, rep = _published_frame(df)
    RELEASE.mkdir(parents=True, exist_ok=True)
    df.to_csv(RELEASE / "labels_published.csv", index=False)
    print(f"[split] {rep.summary()}")
    for book in sorted(df["book"].unique()):
        lo = split_mod.lobo_split(df, book)
        n_test = int((lo == "test").sum())
        print(f"[lobo]  holdout {book}: test={n_test}")
    print(f"[split] -> {RELEASE / 'labels_published.csv'}")


def cmd_metadata(args) -> None:
    df = _load(args)
    df, _ = _published_frame(df)
    RELEASE.mkdir(parents=True, exist_ok=True)
    crops = _crops_csv(df)
    crops_path = RELEASE / "crops.csv"
    crops.to_csv(crops_path, index=False)
    sha = hashing.sha256_file(crops_path)
    stats = _stats(df)
    stats["resource_bytes"] = crops_path.stat().st_size
    n_rows = len(crops)
    names = export_mod.class_names(df)
    splits_count = {k: int(v) for k, v in df[df["tier"].isin(split_mod.USABLE_TIERS)]
                    ["split"].value_counts().items() if k}

    dp = md_mod.build_datapackage("crops", "crops.csv", sha, n_rows, stats)
    cr = md_mod.build_croissant("crops.csv", sha, n_rows, stats)
    card = md_mod.build_card_yaml(stats, len(names), splits_count)
    json.dump(dp, open(RELEASE / "datapackage.json", "w"), ensure_ascii=False, indent=2)
    json.dump(cr, open(RELEASE / "croissant.json", "w"), ensure_ascii=False, indent=2)
    (RELEASE / "README.md").write_text(card, encoding="utf-8")
    print(f"[metadata] crops.csv sha256={sha[:16]}... rows={n_rows} classes={len(names)}")
    print(f"[metadata] -> datapackage.json, croissant.json, README.md (card) in {RELEASE}")


def cmd_datasheet(args) -> None:
    df = _load(args)
    sheet = ds_mod.generate_datasheet(_stats(df))
    RELEASE.mkdir(parents=True, exist_ok=True)
    (RELEASE / "DATASHEET.md").write_text(sheet, encoding="utf-8")
    print(f"[datasheet] {len(ds_mod.REQUIRED_SECTIONS)} sections -> {RELEASE / 'DATASHEET.md'}")


def cmd_export(args) -> None:
    df = _load(args)
    df, _ = _published_frame(df)
    if args.sample:
        cl = export_mod.char_labeled(df)
        keep = cl.sample(min(args.sample, len(cl)), random_state=42).index
        df = df.loc[keep].copy()                       # only the sampled crops (consistent)
    names = export_mod.class_names(df)
    dd = DATASET_OUT
    parquet_dir = RELEASE / ("parquet_sample" if args.sample else "parquet")
    dsd = export_mod.build_hf_dataset(df, dd, names, split_col="split")
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for sp, ds in dsd.items():
        ds.to_parquet(str(parquet_dir / f"{sp}.parquet"))
    n_img = export_mod.write_imagefolder_metadata(df, RELEASE / "imagefolder")
    print(f"[export] parquet splits={list(dsd.keys())} classes={len(names)} "
          f"-> {parquet_dir}")
    print(f"[export] imagefolder metadata.csv rows={n_img}")


def cmd_validate(args) -> None:
    df = _load(args)
    df, _ = _published_frame(df)
    dp = cr = None
    if (RELEASE / "datapackage.json").exists():
        dp = json.load(open(RELEASE / "datapackage.json"))
    if (RELEASE / "croissant.json").exists():
        cr = json.load(open(RELEASE / "croissant.json"))
    rep = val_mod.validate_release(df, DATASET_OUT, split_col="split",
                                   datapackage=dp, croissant=cr,
                                   check_files=not getattr(args, "no_files", False))
    print("[validate]", rep.summary())
    if not rep.ok():
        raise SystemExit(1)


def cmd_all(args) -> None:
    cmd_split(args)
    cmd_metadata(args)
    cmd_datasheet(args)
    cmd_validate(args)
    print("[all] release artifacts ready in", RELEASE)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.publish",
                                description="Giai đoạn 3 — Công bố đạt chuẩn quốc tế")
    p.add_argument("--labels", default=None, help="labels csv (default: labels_remediated)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split").set_defaults(func=cmd_split)
    sub.add_parser("metadata").set_defaults(func=cmd_metadata)
    sub.add_parser("datasheet").set_defaults(func=cmd_datasheet)
    e = sub.add_parser("export")
    e.add_argument("--sample", type=int, default=0, help="export only N crops (smoke)")
    e.set_defaults(func=cmd_export)
    v = sub.add_parser("validate")
    v.add_argument("--no-files", action="store_true", help="skip on-disk crop check")
    v.set_defaults(func=cmd_validate)
    sub.add_parser("all").set_defaults(func=cmd_all)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
