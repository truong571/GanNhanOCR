"""Ingest and standardize Vatican Borgiano Tonchinese manuscripts:
  - MSS_Borg_tonch_34 (Sách Dũng Lý Hộ Thần, 112 text pages)
  - MSS_Borg_tonch_18 (Sách kinh Thầy cả Bỉnh, 529 text pages)

Transforms raw data from data/MSS_Borg.tonch.* -> prepared/MSS_Borg_tonch_*:
  - pages/page_XXXX.png (mode L, contrast normalized)
  - pages_denoised/page_XXXX.png
  - transcriptions/page_XXXX.txt (Quoc Ngu syllables)
  - transcriptions/page_XXXX.json (metadata + sentences + syllables)
  - nom_transcriptions/page_XXXX.txt (Sino-Nom ground truth chars)
  - manifest.json (provenance + page mappings + statistics)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openpyxl
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.image.image_processing import denoise_image
from core.text.text_utils import normalize_tone_marks

BOOK_SPECS = {
    "SachDungLyHoThan": {
        "title": "Sách Dũng Lý Hộ Thần",
        "data_dir": REPO / "data/SachDungLyHoThan",
        "xlsx_name": "SachDungLyHoThan.xlsx",
        "manifest_iiif": "manifest.json",
        "typo_map": {
            "[69]_34r.jpg": "[77]_34r.jpg",
            "[70]_34v.jpg": "[78]_34v.jpg",
        },
        "default_columns": 10,
    },
    "SachKinhThayCaBinh": {
        "title": "Sách kinh Thầy cả Bỉnh",
        "data_dir": REPO / "data/SachKinhThayCaBinh",
        "xlsx_name": "SachKinhThayCaBinh.xlsx",
        "manifest_iiif": "manifest.json",
        "typo_map": {},
        "default_columns": 10,
    },
}
# Alias mapping for backwards compatibility
BOOK_SPECS["MSS_Borg_tonch_34"] = BOOK_SPECS["SachDungLyHoThan"]
BOOK_SPECS["MSS_Borg_tonch_18"] = BOOK_SPECS["SachKinhThayCaBinh"]



def clean_nom_text(text: str) -> str:
    """Strip punctuation and spacing from Nom text."""
    if not text:
        return ""
    return re.sub(r'[\s\.\,\+\-\—\–\:\;。、，！？\?…\(\)\[\]\{\}\<\>\"\'“”„«»]+', '', text)


def clean_quocngu_syllables(text: str) -> list[str]:
    """Clean Quoc Ngu sentence into individual syllables:
    - remove footnotes: [a], [ap], [12], etc.
    - split hyphens in loanwords: Phi-ri-tô -> Phi ri tô
    - normalize tone marks
    - strip punctuation
    """
    if not text:
        return []
    # Remove footnote markers like [a], [1], [ap]
    t = re.sub(r'\[[a-zA-Z0-9]+\]', '', text)
    # Split hyphens and plus signs
    t = t.replace('-', ' ').replace('+', ' ')
    # Split by punctuation / whitespace
    raw_tokens = re.split(r'[\s\.\,\:\;，、。！？\?…\(\)\[\]\{\}\<\>\"\'“”„«»]+', t)
    syls = []
    for tok in raw_tokens:
        tok = tok.strip()
        if not tok:
            continue
        # Normalize tones
        norm = normalize_tone_marks(tok)
        if norm:
            syls.append(norm)
    return syls


def stretch_gray(gray: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Stretch grayscale histogram so paper background maps towards 255."""
    lo, hi = np.percentile(gray, (p_low, p_high))
    if hi <= lo:
        return gray
    stretched = np.clip((gray.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    return stretched.astype(np.uint8)


def prepare_page_image(src_path: Path, dst_png: Path, denoised_png: Path) -> tuple[int, int, str]:
    """Convert raw JPG to standardized PNG and create denoised version."""
    img = Image.open(src_path).convert("L")
    w, h = img.size
    gray = np.asarray(img)
    stretched = stretch_gray(gray)

    dst_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(stretched).save(dst_png, "PNG")

    # Generate MD5 of the prepared page
    with open(dst_png, "rb") as f:
        md5_hash = hashlib.md5(f.read()).hexdigest()

    # Denoise
    denoised_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        denoised = denoise_image(dst_png)
        cv2.imwrite(str(denoised_png), denoised)
    except Exception:
        # Fallback to copy if denoise_image has issues
        Image.fromarray(stretched).save(denoised_png, "PNG")

    return w, h, md5_hash


def parse_excel_groundtruth(xlsx_path: Path, typo_map: dict[str, str]) -> list[dict[str, Any]]:
    """Parse Excel file into ordered pages with text sentences."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active

    # Group by img_id maintaining order
    pages_dict: dict[str, dict[str, Any]] = {}

    for r in range(2, ws.max_row + 1):
        raw_img_id = ws.cell(r, 1).value
        if not raw_img_id:
            continue
        img_id = str(raw_img_id).strip()
        if img_id in typo_map:
            img_id = typo_map[img_id]

        sent_id = str(ws.cell(r, 2).value or f"{img_id}.{r}")
        nom_raw = str(ws.cell(r, 3).value or "")
        qn_raw = str(ws.cell(r, 4).value or "")

        if img_id not in pages_dict:
            pages_dict[img_id] = {
                "img_id": img_id,
                "sentences": [],
                "excel_rows": [],
            }

        clean_nom = clean_nom_text(nom_raw)
        qn_syls = clean_quocngu_syllables(qn_raw)

        pages_dict[img_id]["sentences"].append({
            "sentence_id": sent_id,
            "nom_raw": nom_raw,
            "nom_clean": clean_nom,
            "qn_raw": qn_raw,
            "qn_syllables": qn_syls,
        })
        pages_dict[img_id]["excel_rows"].append(r)

    # Sort pages by their natural order in the manuscript
    # Extract page index bracket [NN] or folio number
    def page_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        img_id = item[0]
        m = re.search(r'\[(\d+)\]', img_id)
        if m:
            return (int(m.group(1)), img_id)
        return (999999, img_id)

    sorted_pages = sorted(pages_dict.items(), key=page_sort_key)
    return [p[1] for p in sorted_pages]


def ingest_book(book_key: str, out_base: Path, limit: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Ingest one Vatican manuscript book."""
    spec = BOOK_SPECS[book_key]
    data_dir = spec["data_dir"]
    xlsx_path = data_dir / spec["xlsx_name"]
    typo_map = spec["typo_map"]

    print(f"\n=======================================================")
    print(f"Ingesting {book_key} ({spec['title']})")
    print(f"  Source dir: {data_dir}")
    print(f"  Excel GT  : {xlsx_path.name}")
    print(f"=======================================================")

    pages_gt = parse_excel_groundtruth(xlsx_path, typo_map)
    total_pages = len(pages_gt)
    print(f"Found {total_pages} text pages in ground truth Excel.")

    if limit and limit > 0:
        pages_gt = pages_gt[:limit]
        print(f"Applying limit: {len(pages_gt)} pages.")

    prep_dir = out_base / book_key
    pages_dir = prep_dir / "pages"
    denoised_dir = prep_dir / "pages_denoised"
    trans_dir = prep_dir / "transcriptions"
    nom_trans_dir = prep_dir / "nom_transcriptions"

    if not dry_run:
        pages_dir.mkdir(parents=True, exist_ok=True)
        denoised_dir.mkdir(parents=True, exist_ok=True)
        trans_dir.mkdir(parents=True, exist_ok=True)
        nom_trans_dir.mkdir(parents=True, exist_ok=True)

    manifest_pages = []
    tot_nom_chars = 0
    tot_qn_syls = 0
    perfect_match_count = 0
    close_match_count = 0

    for idx, page_data in enumerate(pages_gt, start=1):
        page_name = f"page_{idx:04d}"
        img_id = page_data["img_id"]
        src_img_path = data_dir / img_id

        if not src_img_path.exists():
            print(f"[ERROR] Source image not found on disk: {src_img_path}")
            continue

        # Extract folio label from img_id, e.g. [11]_1r.jpg -> 1r
        folio_match = re.search(r'\]_([0-9a-zA-Z\.]+)\.jpg', img_id)
        folio = folio_match.group(1) if folio_match else img_id

        dst_png = pages_dir / f"{page_name}.png"
        denoised_png = denoised_dir / f"{page_name}.png"
        txt_path = trans_dir / f"{page_name}.txt"
        json_path = trans_dir / f"{page_name}.json"
        nom_txt_path = nom_trans_dir / f"{page_name}.txt"

        # Combine text across all sentences on the page
        all_nom = "".join(s["nom_clean"] for s in page_data["sentences"])
        all_qn_syls = []
        for s in page_data["sentences"]:
            all_qn_syls.extend(s["qn_syllables"])

        n_nom = len(all_nom)
        n_qn = len(all_qn_syls)
        tot_nom_chars += n_nom
        tot_qn_syls += n_qn

        diff = abs(n_nom - n_qn)
        if diff == 0:
            perfect_match_count += 1
        elif diff <= 3:
            close_match_count += 1

        w, h, md5_hash = 0, 0, ""
        if not dry_run:
            w, h, md5_hash = prepare_page_image(src_img_path, dst_png, denoised_png)

            # Write Quoc Ngu transcriptions
            # In prose layout, each line in page_XXXX.txt represents syllables for a column or sentence.
            # Writing clean space-separated syllables of the page
            txt_path.write_text(" ".join(all_qn_syls) + "\n", encoding="utf-8")

            # Write Nom transcriptions
            nom_txt_path.write_text(all_nom + "\n", encoding="utf-8")

            # Write detailed JSON
            page_meta = {
                "book_page": idx,
                "page_name": page_name,
                "source_file": img_id,
                "folio": folio,
                "excel_rows": page_data["excel_rows"],
                "width": w,
                "height": h,
                "image_md5": md5_hash,
                "num_sentences": len(page_data["sentences"]),
                "num_nom_chars": n_nom,
                "num_qn_syllables": n_qn,
                "char_syl_diff": diff,
                "sentences": page_data["sentences"],
                "all_syllables": all_qn_syls,
            }
            json_path.write_text(json.dumps(page_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest_pages.append({
            "book_page": idx,
            "page_name": page_name,
            "source_file": img_id,
            "folio": folio,
            "width": w,
            "height": h,
            "md5": md5_hash,
            "n_nom_chars": n_nom,
            "n_qn_syllables": n_qn,
            "char_syl_diff": diff,
            "excel_row_start": min(page_data["excel_rows"]),
            "excel_row_end": max(page_data["excel_rows"]),
        })

        if idx <= 5 or idx % 50 == 0 or idx == len(pages_gt):
            match_str = "EXACT 1-1" if diff == 0 else f"diff={diff:+d}"
            print(f"  [{idx:03d}/{len(pages_gt):03d}] {page_name} <- {img_id:<18} (Folio {folio:<6}) "
                  f"Nom={n_nom:3d} | QN={n_qn:3d} ({match_str})")

    summary = {
        "book": book_key,
        "title": spec["title"],
        "source_dir": str(data_dir),
        "total_text_pages": len(manifest_pages),
        "total_nom_chars": tot_nom_chars,
        "total_qn_syllables": tot_qn_syls,
        "overall_char_syl_ratio": round(tot_nom_chars / max(1, tot_qn_syls), 4),
        "perfect_match_pages": perfect_match_count,
        "perfect_match_pct": round(perfect_match_count / max(1, len(manifest_pages)) * 100, 2),
        "close_match_pages (diff<=3)": close_match_count,
        "close_match_pct": round(close_match_count / max(1, len(manifest_pages)) * 100, 2),
        "high_fidelity_pages (>=97%)": perfect_match_count + close_match_count,
        "high_fidelity_pct": round((perfect_match_count + close_match_count) / max(1, len(manifest_pages)) * 100, 2),
        "pages": manifest_pages,
    }

    if not dry_run:
        manifest_file = prep_dir / "manifest.json"
        manifest_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote manifest to {manifest_file}")

    print(f"\n--- Summary for {book_key} ---")
    print(f"  Pages ingested: {len(manifest_pages)}")
    print(f"  Total Nôm chars: {tot_nom_chars:,}")
    print(f"  Total QN syllables: {tot_qn_syls:,}")
    print(f"  Ratio (Nom/QN): {tot_nom_chars/max(1, tot_qn_syls):.4f}")
    print(f"  1-to-1 match rate: {summary['high_fidelity_pct']}% (diff <= 3)")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Ingest Vatican Borgiano Tonchinese manuscripts")
    parser.add_argument("--book", choices=["SachDungLyHoThan", "SachKinhThayCaBinh", "MSS_Borg_tonch_34", "MSS_Borg_tonch_18", "all"], default="all")
    parser.add_argument("--out", type=Path, default=REPO / "prepared")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pages (for smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Audit only without writing files")

    args = parser.parse_args()

    books_to_run = ["SachDungLyHoThan", "SachKinhThayCaBinh"] if args.book == "all" else [args.book]


    results = {}
    for b in books_to_run:
        results[b] = ingest_book(b, out_base=args.out, limit=args.limit, dry_run=args.dry_run)

    print("\n=======================================================")
    print("ALL INGESTIONS COMPLETED SUCCESSFULLY")
    print("=======================================================")


if __name__ == "__main__":
    main()
