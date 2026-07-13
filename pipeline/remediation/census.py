"""Unified duplicate-crop census (AE-1 ∪ F1) over the committed labels.csv.

Two proven, jointly-exhaustive defect families produce the same signature — one
exported crop image reused under >1 label:

  AE-1  identical bbox on 2+ rows within the SAME (book, page, column)
        (align_production._pick_reseg picked one detector box for two chars)
  F1    identical image_md5 spanning >1 column within the SAME (book, page)
        (infer_centernet.column_boxes x_range overlap exported one box twice)

Verified in the 3-round evaluation: AE-1 = 701 rows / 328 groups, F1 = 1,686 rows /
820 groups, union = 2,321 rows / 1,116 md5 groups, of which 1,094 groups (1,177 rows,
955 GOLD) carry conflicting labels on identical pixels — provably wrong. There is NO
third duplicate signature, so md5-dedup at export closes the whole class.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = ["USABLE_TIERS", "CensusResult", "run_census"]

USABLE_TIERS = ("GOLD", "SILVER", "SYLLABLE")


@dataclass
class CensusResult:
    dup_bbox_rows: int
    dup_bbox_groups: int
    cross_col_rows: int
    cross_col_groups: int
    union_rows: int
    union_groups: int
    conflicting_groups: int
    provably_wrong_rows: int          # minority-label rows within conflicting groups
    by_tier: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"AE-1 dup-bbox: {self.dup_bbox_rows} rows / {self.dup_bbox_groups} grp; "
                f"F1 cross-col: {self.cross_col_rows} rows / {self.cross_col_groups} grp; "
                f"UNION: {self.union_rows} rows / {self.union_groups} md5-grp; "
                f"conflicting: {self.conflicting_groups} grp; "
                f"provably-wrong floor: {self.provably_wrong_rows} rows "
                f"(by tier: {self.by_tier})")


def _usable(df: pd.DataFrame) -> pd.DataFrame:
    img = df["image"].fillna("").astype(str)
    return df[df["tier"].isin(USABLE_TIERS) & (img.str.len() > 0)]


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Return usable rows with boolean columns dup_bbox, cross_col, dup_defect, and a
    stable `dup_group` key (the image_md5 for defect rows, else '')."""
    u = _usable(df).copy()
    if u.empty:
        raise ValueError("no usable rows to census")
    md5 = u["image_md5"].fillna("").astype(str)

    # AE-1: identical bbox appearing on >1 row in the same (book,page,column)
    u["dup_bbox"] = (
        u.groupby(["book", "page", "column", "bbox"], dropna=False)["image"]
        .transform("size") > 1
    )
    # F1: same md5 spanning >1 distinct column in the same (book,page)
    ncols = (
        u.assign(_md5=md5)
        .groupby(["book", "page", "_md5"], dropna=False)["column"]
        .transform("nunique")
    )
    u["cross_col"] = (md5.str.len() > 0) & (ncols > 1)
    u["dup_defect"] = u["dup_bbox"] | u["cross_col"]
    u["dup_group"] = md5.where(u["dup_defect"], "")
    return u


def run_census(df: pd.DataFrame) -> CensusResult:
    """Compute the unified census counts from a labels.csv frame."""
    u = annotate(df)
    md5 = u["image_md5"].fillna("").astype(str)

    dup_bbox_rows = int(u["dup_bbox"].sum())
    dup_bbox_groups = int(
        u[u["dup_bbox"]].groupby(["book", "page", "column", "bbox"], dropna=False).ngroups
    )
    cross_rows = int(u["cross_col"].sum())
    cross_groups = int(u[u["cross_col"]].assign(_m=md5[u["cross_col"]]).groupby(
        ["book", "page", "_m"], dropna=False).ngroups)

    defect = u[u["dup_defect"]].copy()
    defect["_md5"] = md5[u["dup_defect"]]
    union_rows = len(defect)
    groups = defect.groupby("_md5", dropna=False)
    union_groups = groups.ngroups

    conflicting = 0
    provably_wrong = 0
    for _, g in groups:
        labels = g["label"].fillna("").astype(str)
        if labels.nunique() > 1:
            conflicting += 1
            # minority-label rows: everything except the single largest label class
            top = labels.value_counts().idxmax()
            provably_wrong += int((labels != top).sum())

    by_tier = defect["tier"].value_counts().to_dict()
    return CensusResult(
        dup_bbox_rows=dup_bbox_rows, dup_bbox_groups=dup_bbox_groups,
        cross_col_rows=cross_rows, cross_col_groups=cross_groups,
        union_rows=union_rows, union_groups=union_groups,
        conflicting_groups=conflicting, provably_wrong_rows=provably_wrong,
        by_tier={str(k): int(v) for k, v in by_tier.items()},
    )
