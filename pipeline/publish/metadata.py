"""International-standard metadata: Frictionless Data Package + Croissant + HF card.

Fixes the audit findings directly:
  - primaryKey used to be ['image'] but 14,192 REVIEW rows had an empty image -> the PK
    was null and the package was invalid. Here the published crop resource contains ONLY
    rows with a non-empty image, so `image` is a genuine non-null unique key.
  - croissant sha256 used to be the literal 'n/a' -> here it is the real sha256 of the
    exported data file (see hashing.sha256_file).

Built by hand to the specs (frictionless/mlcroissant are not installed); validate.py
checks the invariants the libraries would.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Field", "CROP_FIELDS", "build_datapackage", "build_croissant", "build_card_yaml"]


@dataclass(frozen=True)
class Field:
    name: str
    ftype: str            # frictionless Table Schema type
    croissant: str        # croissant/schema.org dataType
    desc: str
    enum: tuple = ()


CROP_FIELDS = (
    Field("image", "string", "sc:Text", "Relative path to the crop PNG (primary key)."),
    Field("book", "string", "sc:Text", "Source book code (yen2/yen4/yen11)."),
    Field("page", "string", "sc:Text", "Page id within the book."),
    Field("column", "integer", "sc:Integer", "1-based column index (RTL)."),
    Field("ocr_char", "string", "sc:Text", "SinoNom (S1) OCR character."),
    Field("syllable", "string", "sc:Text", "Aligned Quốc-ngữ syllable."),
    Field("label", "string", "sc:Text", "Assigned Nôm character (target)."),
    Field("unicode", "string", "sc:Text", "Unicode codepoint of the label (U+XXXX)."),
    Field("tier", "string", "sc:Text", "Confidence tier.",
          ("GOLD", "SILVER", "SYLLABLE", "REVIEW", "QUARANTINE")),
    Field("rule", "string", "sc:Text", "Consensus rule that produced the label."),
    Field("split", "string", "sc:Text", "Page-disjoint split.",
          ("train", "val", "test", "")),
    Field("s3_cosine", "number", "sc:Float", "S3 glyph-verifier cosine (may be empty)."),
    Field("ink_pct", "number", "sc:Float", "Ink fraction of the crop."),
    Field("crop_w", "integer", "sc:Integer", "Crop width in px."),
    Field("crop_h", "integer", "sc:Integer", "Crop height in px."),
    Field("image_md5", "string", "sc:Text", "12-hex prefix of the crop md5."),
    Field("seg_flag", "string", "sc:Text", "Segmentation flag (ok/tall)."),
    Field("bbox", "string", "sc:Text", "Fullpage [x1,y1,x2,y2] JSON."),
)


def build_datapackage(resource_name: str, resource_path: str, sha256: str, n_rows: int,
                      stats: dict, fields=CROP_FIELDS, version: str = "1.0.0") -> dict:
    """Frictionless Data Package v2 with a typed Table Schema. PK = image (non-null)."""
    schema_fields = []
    for f in fields:
        d = {"name": f.name, "type": f.ftype, "description": f.desc}
        if f.enum:
            d["constraints"] = {"enum": list(f.enum)}
        if f.name == "image":
            d.setdefault("constraints", {})["required"] = True
            d["constraints"]["unique"] = True
        schema_fields.append(d)
    return {
        "profile": "data-package",
        "name": "han-nom-auto-labeled-crops",
        "title": "Han-Nom handwritten character crops (auto-labeled from Quốc-ngữ)",
        "version": version,
        "licenses": [
            {"name": "CC0-1.0", "title": "CC0 1.0 (crop images — public-domain scans)",
             "path": "https://creativecommons.org/publicdomain/zero/1.0/"},
            {"name": "CC-BY-4.0", "title": "CC BY 4.0 (labels & metadata)",
             "path": "https://creativecommons.org/licenses/by/4.0/"},
        ],
        "resources": [{
            "name": resource_name,
            "path": resource_path,
            "profile": "tabular-data-resource",
            "format": "csv",
            "mediatype": "text/csv",
            "encoding": "utf-8",
            "hash": f"sha256:{sha256}",
            "bytes": stats.get("resource_bytes", 0),
            "schema": {
                "fields": schema_fields,
                "primaryKey": ["image"],
                "missingValues": [""],
            },
        }],
        "count_of_rows": n_rows,
        "_stats": stats,
    }


def build_croissant(resource_path: str, sha256: str, n_rows: int, stats: dict,
                    fields=CROP_FIELDS, version: str = "1.0.0") -> dict:
    """MLCommons Croissant 1.0 JSON-LD with a real sha256 file object."""
    field_defs = []
    for f in fields:
        field_defs.append({
            "@type": "cr:Field",
            "@id": f"crops/{f.name}",
            "name": f.name,
            "description": f.desc,
            "dataType": f.croissant,
            "source": {"fileObject": {"@id": "crops-csv"},
                       "extract": {"column": f.name}},
        })
    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "dct": "http://purl.org/dc/terms/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        },
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "han-nom-auto-labeled-crops",
        "description": "Auto-labeled handwritten chữ-Nôm character crops from three "
                       "woodblock books, aligned to their Quốc-ngữ translation.",
        "version": version,
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "citation": "MSc thesis, Han-Nom auto-labeling pipeline (2026).",
        "distribution": [{
            "@type": "cr:FileObject",
            "@id": "crops-csv",
            "name": "crops-csv",
            "description": "Per-crop label table.",
            "contentUrl": resource_path,
            "encodingFormat": "text/csv",
            "sha256": sha256,
        }],
        "recordSet": [{
            "@type": "cr:RecordSet",
            "@id": "crops",
            "name": "crops",
            "description": f"{n_rows} labeled character crops.",
            "field": field_defs,
        }],
        "_stats": stats,
    }


def build_card_yaml(stats: dict, n_classes: int, splits: dict, version: str = "1.0.0") -> str:
    """Hugging Face dataset-card YAML front-matter."""
    lines = [
        "---",
        "pretty_name: Han-Nom Handwritten Character Crops (auto-labeled)",
        "license:",
        "- cc0-1.0",
        "- cc-by-4.0",
        "language:",
        "- vi",
        "- lzh",
        "task_categories:",
        "- image-classification",
        "tags:",
        "- han-nom",
        "- ocr",
        "- historical-documents",
        "- chu-nom",
        "size_categories:",
        f"- {_size_bucket(stats.get('n_usable', 0))}",
        f"config_name: default",
        "dataset_info:",
        f"  features_class_count: {n_classes}",
        "  splits:",
    ]
    for name, n in splits.items():
        lines.append(f"  - name: {name}")
        lines.append(f"    num_examples: {n}")
    lines += [f"version: {version}", "---", "",
              "# Han-Nom Handwritten Character Crops",
              "",
              "Auto-labeled handwritten chữ-Nôm glyph crops aligned from the Quốc-ngữ "
              "translation. See the datasheet for provenance, tiers and known limitations."]
    return "\n".join(lines) + "\n"


def _size_bucket(n: int) -> str:
    for hi, tag in [(1e3, "n<1K"), (1e4, "1K<n<10K"), (1e5, "10K<n<100K"),
                    (1e6, "100K<n<1M")]:
        if n < hi:
            return tag
    return "1M<n<10M"
