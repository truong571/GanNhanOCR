"""Export the dataset manifest to 3 international standards (non-destructive).

Reads dataset_out/labels.csv + summary.json and writes, alongside them:
  1. metadata.csv      — Hugging Face `imagefolder` convention (column `file_name`
                          + metadata) -> load_dataset("imagefolder", data_dir=...).
  2. datapackage.json  — Frictionless Data Package + Table Schema (typed columns).
  3. croissant.json    — MLCommons Croissant (JSON-LD; HF/Kaggle/Google Dataset
                          Search read it).
Originals (labels.csv, crops) are untouched.

Run:
  cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
  .venv/bin/python evaluation/ver_new/to_standard.py            # --dataset <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (column, frictionless-type, croissant-dataType, description)
COLUMNS = [
    ("file_name", "string", "sc:Text", "Relative path to the crop PNG (renamed from 'image' for the HF imagefolder convention). Empty for REVIEW rows that have no crop."),
    ("book", "string", "sc:Text", "Source book id (SachThanhTruyen 2/4/11)."),
    ("page", "string", "sc:Text", "Source page id (page_XXXX)."),
    ("column", "integer", "sc:Integer", "Column index on the page (1-9, right-to-left)."),
    ("ocr_char", "string", "sc:Text", "Raw SinoNom character from the HCMUS OCR (may be a misread)."),
    ("syllable", "string", "sc:Text", "Quốc-Ngữ syllable from VietOCR (the reading)."),
    ("label", "string", "sc:Text", "Confirmed SinoNom character (target). Empty when char-level is not confirmed (label_level=syllable)."),
    ("unicode", "string", "sc:Text", "Unicode code point of `label`, e.g. U+56FA."),
    ("label_level", "string", "sc:Text", "Supervision level: 'char' (label is the character), 'syllable' (only the syllable is trusted), or '' (REVIEW)."),
    ("tier", "string", "sc:Text", "Trust tier: GOLD | SILVER | SYLLABLE | REVIEW."),
    ("rule", "string", "sc:Text", "Decision rule that produced the label (e.g. s1_inter_s2_direct, s1_inter_s2_similar, s2_inter_s3_corrected, nghia_consensus)."),
    ("s3_cosine", "number", "sc:Float", "Visual-match cosine (trained Nôm embedder) for SILVER rows; empty otherwise."),
    ("ink_pct", "number", "sc:Float", "Fraction of dark (ink) pixels in the crop (quality signal)."),
    ("crop_w", "integer", "sc:Integer", "Crop width in pixels."),
    ("crop_h", "integer", "sc:Integer", "Crop height in pixels."),
    ("image_md5", "string", "sc:Text", "12-hex MD5 of the crop file (dedup/provenance)."),
    ("seg_flag", "string", "sc:Text", "Segmentation flag: 'ok' or 'tall' (possible merged/tall glyph)."),
    ("split", "string", "sc:Text", "Dataset split: train | val | test (leakage-safe, grouped by book/page/column)."),
    ("split_group", "string", "sc:Text", "Group key (book|page|column) guaranteeing a physical column never spans two splits."),
    ("bbox", "string", "sc:Text", "Character bounding box on the full page, JSON [x1,y1,x2,y2] (xyxy)."),
]
RENAME = {"image": "file_name"}


def write_hf_metadata(rows, out: Path):
    """HF imagefolder metadata.csv — one row per image, column `file_name` first."""
    fields = [c[0] for c in COLUMNS]
    img_rows = [r for r in rows if r.get("image")]
    with open(out / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in img_rows:
            row = {RENAME.get(k, k): v for k, v in r.items()}
            w.writerow({k: row.get(k, "") for k in fields})
    return len(img_rows)


def write_datapackage(rows, summary, out: Path):
    """Frictionless Data Package + Table Schema describing labels.csv."""
    dp = {
        "profile": "tabular-data-package",
        "name": "nom-quocngu-character-crops",
        "title": "Chu-Nom character crops aligned to Quoc-Ngu syllables",
        "description": "Per-character woodblock chu-Nom crops cross-labeled with SinoNom "
                       "character + Unicode + Quoc-Ngu syllable, from a parallel Nom-QuocNgu "
                       "corpus (Sach Cac Thanh Truyen 2/4/11).",
        "licenses": [{"name": "research-use", "title": "Academic/research use"}],
        "resources": [{
            "name": "labels",
            "path": "labels.csv",
            "profile": "tabular-data-resource",
            "format": "csv",
            "mediatype": "text/csv",
            "encoding": "utf-8",
            "schema": {
                "fields": [
                    {"name": "image" if c[0] == "file_name" else c[0],
                     "type": c[1], "description": c[3]}
                    for c in COLUMNS
                ],
                "primaryKey": ["image"],
                "missingValues": [""],
            },
        }],
        "count_of_rows": len(rows),
        "stats": summary.get("tiers", {}),
    }
    json.dump(dp, open(out / "datapackage.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def write_croissant(rows, summary, out: Path):
    """MLCommons Croissant (JSON-LD)."""
    ctx = {
        "@language": "en", "@vocab": "https://schema.org/",
        "sc": "https://schema.org/", "cr": "http://mlcommons.org/croissant/",
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "data": {"@id": "cr:data", "@type": "@json"},
        "references": {"@id": "cr:references", "@type": "@id"},
        "source": {"@id": "cr:source", "@type": "@id"},
        "field": {"@id": "cr:field", "@type": "@id"},
        "fileProperty": {"@id": "cr:fileProperty", "@type": "@vocab"},
        "format": {"@id": "cr:format"}, "includes": {"@id": "cr:includes"},
        "extract": {"@id": "cr:extract"}, "column": {"@id": "cr:column"},
        "recordSet": {"@id": "cr:recordSet", "@type": "@id"},
        "examples": {"@id": "cr:examples", "@type": "@json"},
        "repeated": {"@id": "cr:repeated"},
    }
    fields = []
    for col, ftype, dtype, desc in COLUMNS:
        fields.append({
            "@type": "cr:Field", "@id": f"records/{col}", "name": col,
            "description": desc, "dataType": dtype,
            "source": {"fileObject": {"@id": "labels.csv"},
                       "extract": {"column": "image" if col == "file_name" else col}},
        })
    croissant = {
        "@context": ctx,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "nom-quocngu-character-crops",
        "description": "Per-character woodblock chu-Nom crops cross-labeled with SinoNom "
                       "character + Unicode + Quoc-Ngu syllable.",
        "license": "research-use",
        "url": "local",
        "version": "1.0.0",
        "distribution": [
            {"@type": "cr:FileObject", "@id": "labels.csv", "name": "labels.csv",
             "contentUrl": "labels.csv", "encodingFormat": "text/csv",
             "sha256": "n/a"},
            {"@type": "cr:FileSet", "@id": "crops", "name": "crops",
             "encodingFormat": "image/png", "includes": "*/*.png"},
        ],
        "recordSet": [{
            "@type": "cr:RecordSet", "@id": "records", "name": "records",
            "description": f"{len(rows)} labeled character crops.",
            "field": fields,
        }],
    }
    json.dump(croissant, open(out / "croissant.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    args = ap.parse_args()
    D = Path(args.dataset)
    rows = list(csv.DictReader(open(D / "labels.csv", encoding="utf-8")))
    summary = json.load(open(D / "summary.json", encoding="utf-8")) if (D / "summary.json").exists() else {}

    n_img = write_hf_metadata(rows, D)
    write_datapackage(rows, summary, D)
    write_croissant(rows, summary, D)
    print(f"Standards written to {D}/:")
    print(f"  metadata.csv     (HF imagefolder) — {n_img} image rows, column 'file_name'")
    print(f"  datapackage.json (Frictionless Table Schema) — {len(COLUMNS)} fields typed")
    print(f"  croissant.json   (MLCommons Croissant 1.0)")


if __name__ == "__main__":
    main()
