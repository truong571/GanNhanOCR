"""HuggingFace export — Parquet with embedded images + typed Features.

HF's own guidance (2025) recommends Parquet with embedded images for small images: it
powers the Dataset Viewer, metadata filtering, and automatic Croissant generation. The
image-classification dataset is the char-labeled crops (GOLD/SILVER); SYLLABLE/REVIEW
stay in the full metadata table but not in the ClassLabel parquet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["char_labeled", "class_names", "PARQUET_COLUMNS", "build_hf_dataset",
           "write_imagefolder_metadata"]

PARQUET_COLUMNS = ("image", "label", "tier", "rule", "book", "page", "column",
                   "syllable", "unicode", "s3_cosine", "split", "bbox")


def char_labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Rows eligible for the image-classification parquet: real crop + single-char label."""
    img = df["image"].fillna("").astype(str)
    lab = df["label"].fillna("").astype(str)
    return df[(img.str.len() > 0) & (lab.str.len() == 1)].copy()


def class_names(df: pd.DataFrame) -> list[str]:
    """Sorted unique single-char labels (the ClassLabel vocabulary)."""
    lab = char_labeled(df)["label"].astype(str)
    return sorted(lab.unique().tolist())


def build_hf_dataset(df: pd.DataFrame, dataset_dir: str | Path, names: list[str],
                     split_col: str = "split"):
    """Build a datasets.DatasetDict (by split) with Image() + ClassLabel() features.

    Requires the `datasets` library. Image column is set to absolute paths and cast to
    the Image feature (bytes embed on to_parquet). Raises if a label is not in `names`.
    """
    from datasets import Dataset, DatasetDict, ClassLabel, Image

    dd = Path(dataset_dir)
    rows = char_labeled(df).copy()
    unknown = set(rows["label"]) - set(names)
    if unknown:
        raise ValueError(f"{len(unknown)} labels not in class vocabulary: "
                         f"{sorted(unknown)[:5]}")
    cols = [c for c in PARQUET_COLUMNS if c in rows.columns]
    rows = rows[cols].copy()
    rows["image"] = rows["image"].map(lambda p: str(dd / p))
    # encode labels to ids first: a string->ClassLabel Arrow cast fails, int->ClassLabel
    # (which just attaches names) is safe.
    name2id = {n: i for i, n in enumerate(names)}
    rows["label"] = rows["label"].map(name2id).astype("int64")

    out = {}
    for sp in ("train", "val", "test"):
        part = rows[rows[split_col] == sp]
        if part.empty:
            continue
        ds = Dataset.from_pandas(part.drop(columns=[split_col]), preserve_index=False)
        ds = ds.cast_column("label", ClassLabel(names=names))
        ds = ds.cast_column("image", Image())
        out[sp] = ds
    if not out:
        raise ValueError("no split partitions produced (check split column)")
    return DatasetDict(out)


def write_imagefolder_metadata(df: pd.DataFrame, out_dir: str | Path) -> int:
    """HF imagefolder metadata.csv (file_name + columns), IMAGE ROWS ONLY.

    Only rows with a non-empty image are written, so `file_name` is a genuine non-null
    key (the old export listed empty-image REVIEW rows and broke the primary key).
    Returns the number of rows written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = df["image"].fillna("").astype(str)
    rows = df[img.str.len() > 0].copy()
    rows = rows.rename(columns={"image": "file_name"})
    cols = ["file_name"] + [c for c in rows.columns if c != "file_name"]
    rows[cols].to_csv(out / "metadata.csv", index=False)
    return len(rows)
