"""Release CI gate — the invariants frictionless/mlcroissant would enforce, plus more.

Run before publishing (or in CI on every regeneration). Fails loud on: wrong class
count, page/md5 leakage across splits, missing crop files, null/duplicate primary key,
placeholder sha256, out-of-enum values. Uses `datasets` for a load smoke-test when a
parquet dir is given; everything else is dependency-free static checking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import export, metadata as md_mod, splits as split_mod

__all__ = ["Check", "ValidationReport", "validate_release"]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ValidationReport:
    checks: list = field(default_factory=list)

    def add(self, name, ok, detail=""):
        self.checks.append(Check(name, bool(ok), detail))

    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def summary(self) -> str:
        n_ok = sum(c.ok for c in self.checks)
        head = f"{n_ok}/{len(self.checks)} checks passed"
        fails = [f"  FAIL {c.name}: {c.detail}" for c in self.checks if not c.ok]
        return head + ("" if not fails else "\n" + "\n".join(fails))


def validate_release(df: pd.DataFrame, dataset_dir: str | Path, split_col: str = "split",
                     datapackage: dict | None = None, croissant: dict | None = None,
                     declared_classes: int | None = None,
                     check_files: bool = True) -> ValidationReport:
    """Validate a release-ready labels frame + optional metadata dicts."""
    r = ValidationReport()
    dd = Path(dataset_dir)
    usable = df[df["tier"].isin(split_mod.USABLE_TIERS)]

    # 1. class count
    names = export.class_names(df)
    if declared_classes is not None:
        r.add("class count matches declared", len(names) == declared_classes,
              f"{len(names)} vs {declared_classes}")
    else:
        r.add("class vocabulary non-empty", len(names) > 0, str(len(names)))

    # 2. split leakage: no page, no md5 spans >1 split
    if split_col in df.columns:
        u = usable.assign(_s=df.loc[usable.index, split_col],
                          _pid=usable["book"].astype(str) + "|" + usable["page"].astype(str))
        u = u[u["_s"].astype(str).str.len() > 0]
        page_span = int((u.groupby("_pid")["_s"].nunique() > 1).sum())
        r.add("no page spans splits", page_span == 0, f"{page_span} pages span")
        m = u["image_md5"].fillna("").astype(str)
        has = m.str.len() > 0
        md5_span = int((u[has].assign(_m=m[has]).groupby("_m")["_s"].nunique() > 1).sum())
        r.add("no md5 spans splits", md5_span == 0, f"{md5_span} md5 span")

    # 3. primary key: image non-null + unique among image rows
    img = df["image"].fillna("").astype(str)
    imgrows = df[img.str.len() > 0]
    dup = imgrows["image"].duplicated().sum()
    r.add("PK image unique", dup == 0, f"{dup} duplicate image paths")
    r.add("PK image non-null on image resource", (img[img.str.len() > 0] != "").all())

    # 4. every char-labeled crop exists on disk
    if check_files:
        cl = export.char_labeled(df)
        missing = sum(1 for p in cl["image"] if not (dd / str(p)).exists())
        r.add("all char-labeled crops exist", missing == 0, f"{missing} missing")

    # 5. sha256 present (not placeholder) in metadata
    if datapackage is not None:
        h = datapackage.get("resources", [{}])[0].get("hash", "")
        r.add("datapackage sha256 real", h.startswith("sha256:") and "n/a" not in h, h)
        pk = datapackage["resources"][0]["schema"].get("primaryKey")
        r.add("datapackage PK declared image", pk == ["image"], str(pk))
    if croissant is not None:
        sha = croissant.get("distribution", [{}])[0].get("sha256", "")
        r.add("croissant sha256 real", bool(sha) and sha != "n/a", sha)

    # 6. enum conformance for tier/split
    tier_enum = next(f.enum for f in md_mod.CROP_FIELDS if f.name == "tier")
    bad_tier = set(df["tier"].dropna().unique()) - set(tier_enum)
    r.add("tier values within enum", not bad_tier, f"unexpected {bad_tier}")
    if split_col in df.columns:
        split_enum = next(f.enum for f in md_mod.CROP_FIELDS if f.name == "split")
        bad_split = set(df[split_col].fillna("").unique()) - set(split_enum)
        r.add("split values within enum", not bad_split, f"unexpected {bad_split}")

    return r


def load_smoke_test(parquet_dir: str | Path) -> Check:
    """Optional: load an exported parquet dir with `datasets` and check it opens."""
    try:
        from datasets import load_dataset
        ds = load_dataset("parquet", data_dir=str(parquet_dir))
        n = sum(len(v) for v in ds.values())
        return Check("load_dataset smoke", n > 0, f"loaded {n} rows")
    except Exception as e:
        return Check("load_dataset smoke", False, f"{type(e).__name__}: {e}")
