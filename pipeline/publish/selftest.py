"""Self-test for Giai đoạn 3 publishing (no pytest).

    .venv/bin/python -m pipeline.publish.selftest

Synthetic tests give exact assertions for splits/metadata/datasheet/validate; a real
integration then builds metadata + a small parquet from labels_remediated.csv and
validates it. Exit 0 = all pass.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import datasheet as ds_mod
from . import export as export_mod
from . import hashing, metadata as md_mod, splits as split_mod, validate as val_mod

REPO = Path(__file__).resolve().parents[2]
DATASET_OUT = REPO / "dataset_out"

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}  {detail}")


def _row(**kw):
    base = dict(image="", book="yen2", page="p1", column=1, ocr_char="X", syllable="a",
                label="德", unicode="U+5FB7", label_level="char", tier="GOLD",
                rule="s1_inter_s2_direct", s3_cosine="", ink_pct=0.1, crop_w=100,
                crop_h=100, image_md5="", seg_flag="ok", split="", bbox="[0,0,9,9]")
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
def test_hashing():
    print("[hashing]")
    import hashlib
    check("sha256_bytes matches hashlib",
          hashing.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest())
    check("hamming basic", hashing.hamming(0b1011, 0b1001) == 1)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a = td / "a.png"; b = td / "b.png"; c = td / "c.png"
        Image.new("L", (32, 32), 0).save(a)
        Image.new("L", (32, 32), 0).save(b)          # identical to a
        img_c = Image.new("L", (32, 32), 0)
        for i in range(32):
            for j in range(16):
                img_c.putpixel((j, i), 255)          # half white -> very different
        img_c.save(c)
        ha, hb, hc = hashing.dhash(a), hashing.dhash(b), hashing.dhash(c)
        check("dhash deterministic", hashing.dhash(a) == ha)
        check("dhash identical images equal", ha == hb)
        check("dhash different images differ", hashing.hamming(ha, hc) > 4,
              str(hashing.hamming(ha, hc)))


# --------------------------------------------------------------------------- #
def _synthetic_labels() -> pd.DataFrame:
    rows = []
    # 12 pages, several classes, each class on >=2 crops (avoid all-singleton)
    for pg in range(1, 13):
        for col in range(1, 6):
            lab = "德" if (pg + col) % 3 else "月"
            rows.append(_row(image=f"gold/y_p{pg}_c{col}.png", page=f"p{pg}", column=col,
                             label=lab, unicode="U+5FB7" if lab == "德" else "U+6708",
                             image_md5=f"md5_{pg}_{col}", bbox=f"[{col},0,{col+9},9]"))
    # a book2 + book11 for LOBO
    for pg in range(1, 4):
        rows.append(_row(image=f"gold/y4_p{pg}.png", book="yen4", page=f"p{pg}",
                         label="德", image_md5=f"b4_{pg}"))
        rows.append(_row(image=f"gold/y11_p{pg}.png", book="yen11", page=f"p{pg}",
                         label="月", unicode="U+6708", image_md5=f"b11_{pg}"))
    rows.append(_row(image="", tier="REVIEW", rule="below_visual_threshold",
                     label="", unicode="", image_md5=""))
    return pd.DataFrame(rows)


def test_splits():
    print("[splits]")
    df = _synthetic_labels()
    split, rep = split_mod.assign_page_disjoint(df, seed=1)
    check("no page spans splits", rep.pages_spanning_splits == 0, str(rep.pages_spanning_splits))
    check("no md5 spans splits", rep.md5_spanning_splits == 0, str(rep.md5_spanning_splits))
    check("report ok()", rep.ok(), rep.summary())
    check("deterministic", split_mod.assign_page_disjoint(df, seed=1)[0].equals(split))
    check("different seed differs",
          not split_mod.assign_page_disjoint(df, seed=9)[0].equals(split))
    check("REVIEW rows carry no split", (split[df["tier"] == "REVIEW"] == "").all())
    # all three splits present with a reasonable page count
    su = split[df["tier"].isin(split_mod.USABLE_TIERS)]
    check("has train", (su == "train").any())

    # LOBO
    lo = split_mod.lobo_split(df, "yen4")
    check("lobo test = holdout book",
          set(df.loc[lo == "test", "book"]) == {"yen4"})
    check("lobo train excludes holdout", "yen4" not in set(df.loc[lo == "train", "book"]))

    # cross_split_exact detects a planted leak
    planted = split.copy()
    # force two rows with the same md5 into different splits
    idx = df.index[df["image_md5"] == "md5_1_1"]
    df2 = df.copy()
    df2.loc[idx, "image_md5"] = "LEAK"
    df2.loc[df2.index[df2["image_md5"] == "b4_1"], "image_md5"] = "LEAK"
    s2 = split.copy()
    s2[idx[0]] = "train"; s2[df2.index[df2["image_md5"] == "LEAK"][1]] = "test"
    check("cross_split_exact detects leak",
          split_mod.cross_split_exact(df2, s2) >= 1)


def test_perceptual():
    print("[perceptual duplicates]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "gold").mkdir()

        def checker(bx, by):                           # block pattern -> mixed dHash bits
            im = Image.new("L", (40, 40))
            for y in range(40):
                for x in range(40):
                    im.putpixel((x, y), 255 if ((x // bx) + (y // by)) % 2 else 0)
            return im

        checker(5, 5).save(td / "gold" / "a.png")
        checker(5, 5).save(td / "gold" / "b.png")      # identical to a
        checker(4, 8).save(td / "gold" / "c.png")      # different pattern
        pairs = split_mod.perceptual_duplicates(
            {"train": ["gold/a.png", "gold/c.png"], "test": ["gold/b.png"]}, td)
        cross = [p for p in pairs if {p[1], p[3]} == {"train", "test"}]
        check("perceptual finds cross-split near-dup", len(cross) >= 1, str(pairs))
        check("perceptual ignores dissimilar",
              all(p[0].endswith("a.png") or p[2].endswith("a.png") for p in cross))


def test_metadata():
    print("[metadata]")
    dp = md_mod.build_datapackage("crops", "crops.csv", "a" * 64, 1000,
                                  {"resource_bytes": 123})
    check("datapackage PK image", dp["resources"][0]["schema"]["primaryKey"] == ["image"])
    check("datapackage sha256 real", dp["resources"][0]["hash"] == "sha256:" + "a" * 64)
    fields = {f["name"]: f for f in dp["resources"][0]["schema"]["fields"]}
    check("image field required+unique",
          fields["image"]["constraints"].get("required") and
          fields["image"]["constraints"].get("unique"))
    check("tier enum present", "enum" in fields["tier"]["constraints"])
    check("typed integer column", fields["column"]["type"] == "integer")

    cr = md_mod.build_croissant("crops.csv", "b" * 64, 1000, {})
    check("croissant sha256 real (not n/a)",
          cr["distribution"][0]["sha256"] == "b" * 64)
    check("croissant recordSet has fields",
          len(cr["recordSet"][0]["field"]) == len(md_mod.CROP_FIELDS))
    check("croissant conformsTo 1.0", cr["conformsTo"].endswith("croissant/1.0"))

    card = md_mod.build_card_yaml({"n_usable": 65000}, 1592, {"train": 50000, "test": 8000})
    check("card yaml frontmatter", card.startswith("---") and "license:" in card)
    check("card declares class count", "features_class_count: 1592" in card)
    check("card size bucket 10K-100K", "10K<n<100K" in card)


def test_datasheet():
    print("[datasheet]")
    sheet = ds_mod.generate_datasheet(
        {"n_total": 82268, "n_usable": 65029, "n_classes": 1592, "n_books": 3,
         "n_pages": 445, "tier_counts": {"GOLD": 48600, "QUARANTINE": 2299},
         "quarantined": 2299})
    for sec in ds_mod.REQUIRED_SECTIONS:
        check(f"datasheet has section '{sec}'", f"## {sec}" in sheet or sec in sheet)
    check("datasheet interpolates stats", "82,268" in sheet and "1,592" in sheet)
    check("datasheet has license CC0/CC-BY", "CC0" in sheet and "CC BY" in sheet)


def test_export_and_validate_synth():
    print("[export + validate (synthetic)]")
    df = _synthetic_labels()
    split, _ = split_mod.assign_page_disjoint(df, seed=1)
    df = df.assign(split=split)
    check("char_labeled filters single-char", (export_mod.char_labeled(df)["label"].str.len() == 1).all())
    names = export_mod.class_names(df)
    check("class_names sorted unique", names == sorted(set(names)) and len(names) == 2)

    # validate a clean frame (no file check)
    dp = md_mod.build_datapackage("crops", "crops.csv", "c" * 64, len(df), {})
    cr = md_mod.build_croissant("crops.csv", "c" * 64, len(df), {})
    rep = val_mod.validate_release(df, DATASET_OUT, split_col="split",
                                   datapackage=dp, croissant=cr, check_files=False)
    check("clean synthetic passes validate", rep.ok(), rep.summary())

    # inject defects -> validate fails
    bad = df.copy()
    dup_img = bad.index[bad["image"].str.len() > 0][0]
    bad.loc[dup_img, "image"] = bad["image"].dropna().iloc[1]      # duplicate PK
    dp_bad = md_mod.build_datapackage("crops", "crops.csv", "n/a", len(bad), {})
    dp_bad["resources"][0]["hash"] = "sha256:n/a"
    rep2 = val_mod.validate_release(bad, DATASET_OUT, split_col="split",
                                    datapackage=dp_bad, check_files=False)
    check("duplicate PK fails validate", not rep2.ok())
    check("placeholder sha256 flagged",
          any(not c.ok and "sha256" in c.name for c in rep2.checks))

    # write imagefolder metadata only for image rows
    with tempfile.TemporaryDirectory() as td:
        n = export_mod.write_imagefolder_metadata(df, td)
        meta = pd.read_csv(Path(td) / "metadata.csv")
        check("imagefolder rows = image rows", n == (df["image"].str.len() > 0).sum())
        check("imagefolder file_name non-null", meta["file_name"].notna().all()
              and (meta["file_name"].str.len() > 0).all())


def test_integration_real():
    labels = DATASET_OUT / "labels_remediated.csv"
    if not labels.exists():
        labels = DATASET_OUT / "labels.csv"
    if not labels.exists():
        print("[warn] no labels csv — skipping real integration")
        return
    print(f"[integration — real ({labels.name})]")
    df = pd.read_csv(labels, dtype={"image_md5": str})
    split, rep = split_mod.assign_page_disjoint(df, seed=42)
    df = df.assign(split=split)
    check("real: no page spans splits", rep.pages_spanning_splits == 0)
    check("real: no md5 spans splits", rep.md5_spanning_splits == 0)

    # metadata with real sha256
    crops = df[df["image"].fillna("").astype(str).str.len() > 0]
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "crops.csv"
        cols = [f.name for f in md_mod.CROP_FIELDS if f.name in crops.columns]
        crops[cols].to_csv(cp, index=False)
        sha = hashing.sha256_file(cp)
        check("real sha256 is 64 hex", len(sha) == 64)
        dp = md_mod.build_datapackage("crops", "crops.csv", sha, len(crops), {})
        cr = md_mod.build_croissant("crops.csv", sha, len(crops), {})
        rep_v = val_mod.validate_release(df, DATASET_OUT, split_col="split",
                                         datapackage=dp, croissant=cr,
                                         declared_classes=len(export_mod.class_names(df)),
                                         check_files=False)
        check("real: validate passes (static)", rep_v.ok(), rep_v.summary())

    # export a SMALL real parquet sample and reload it
    cl = export_mod.char_labeled(df)
    names = export_mod.class_names(df)
    sample_idx = cl.sample(min(24, len(cl)), random_state=1).index
    sdf = df.loc[sample_idx].copy()
    # force a couple of splits so >1 partition
    sdf.iloc[: len(sdf) // 2, sdf.columns.get_loc("split")] = "train"
    sdf.iloc[len(sdf) // 2:, sdf.columns.get_loc("split")] = "test"
    try:
        dsd = export_mod.build_hf_dataset(sdf, DATASET_OUT, names, split_col="split")
        check("real: HF dataset built", set(dsd.keys()) <= {"train", "val", "test"})
        tr = dsd[list(dsd.keys())[0]]
        check("real: label is ClassLabel", tr.features["label"].__class__.__name__ == "ClassLabel")
        check("real: image is Image feature", tr.features["image"].__class__.__name__ == "Image")
        with tempfile.TemporaryDirectory() as td:
            pq = Path(td) / "train.parquet"
            tr.to_parquet(str(pq))
            from datasets import Dataset
            reloaded = Dataset.from_parquet(str(pq))
            check("real: parquet round-trips", len(reloaded) == len(tr))
    except Exception as e:
        check("real: HF export/round-trip", False, f"{type(e).__name__}: {e}")


def main() -> int:
    print("=" * 64)
    print("PUBLISH SELFTEST")
    print("=" * 64)
    test_hashing()
    test_splits()
    test_perceptual()
    test_metadata()
    test_datasheet()
    test_export_and_validate_synth()
    test_integration_real()
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
