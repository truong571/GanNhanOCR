"""Test if Kimhannom param `lang_type=2` gives better tier-1 hits than default.

Probe showed lang_type=2 returned 17 boxes vs 15 with default on page 12.
Maybe it uses a different OCR model. Worth testing on 10 pages.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.ocr.ocr_api import upload_image, _get_ocr_token, _SN_DOMAIN, boxes_to_columns
from core.text.alignment import levenshtein_align

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


def call_api(file_name: str, lang_type: int) -> list[dict]:
    url = f"https://{_SN_DOMAIN}/api/web/clc-sinonom/image-ocr"
    body = {"file_name": file_name, "ocr_id": 1,
            "lang_type": lang_type, "reading_direction": 1, "font_type": 1}
    headers = {"User-Agent": "Mozilla/5.0",
               "Authorization": f"Bearer {_get_ocr_token()}",
               "Content-Type": "application/json; charset=utf-8"}
    resp = requests.post(url, json=body, headers=headers, verify=False, timeout=60)
    resp.raise_for_status()
    return resp.json().get("data", {}).get("details", {}).get("details", [])


def load_dict() -> dict[str, set[str]]:
    df = pd.read_csv(REPO / "Dict" / "QuocNgu_SinoNom_TongHop3.csv")
    out: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        qn = str(r["QuocNgu"]).strip().lower()
        nom = str(r["SinoNom"]).strip()
        if qn and nom != "nan":
            out.setdefault(qn, set()).update(c for c in nom if not c.isspace())
    return out


def chars_from_columns(cols: list[list[dict]]) -> list[dict]:
    """Flatten column list into char dicts compatible with levenshtein_align."""
    out = []
    for col in cols:
        for i, c in enumerate(col):
            bb = c["bbox"]
            out.append({
                "char_idx": len(out),
                "bbox": [int(b) for b in bb],
                "height": int(bb[3] - bb[1]),
                "width": int(bb[2] - bb[0]),
                "ocr_char": c.get("char"),
            })
    return out


def score_alignment(chars: list[dict], all_syllables: list[str],
                    qn_to_nom: dict[str, set[str]]) -> tuple[int, int]:
    """Run Levenshtein alignment + count tier-1 hits."""
    pairs = levenshtein_align(chars, all_syllables, qn_to_nom=qn_to_nom)
    hits, total = 0, 0
    for p in pairs:
        if p.get("type") != "match":
            continue
        ch = p.get("char") or {}
        syl = (p.get("syllable") or "").strip().lower()
        ocr_ch = ch.get("ocr_char", "")
        if not syl or not ocr_ch:
            continue
        total += 1
        if ocr_ch in qn_to_nom.get(syl, set()):
            hits += 1
    return hits, total


def collect_pages(n: int) -> list[tuple[str, str]]:
    pages = []
    for book_dir in sorted((REPO / "prepared").iterdir()):
        if not book_dir.is_dir() or book_dir.name.startswith("_"):
            continue
        for trans_path in sorted((book_dir / "transcriptions").glob("page_*.json")):
            pages.append((book_dir.name, trans_path.stem))
    random.seed(11)
    return random.sample(pages, min(n, len(pages)))


def main() -> None:
    qn_to_nom = load_dict()
    pages = collect_pages(10)
    print(f"Testing lang_type=1 vs lang_type=2 on {len(pages)} pages\n")
    print(f"{'BOOK':22s} {'PAGE':12s} {'lang=1':>14s} {'lang=2':>14s}  Δ")
    print("-" * 75)

    s1_h = s1_t = s2_h = s2_t = 0
    for book, page in pages:
        img = REPO / "prepared" / book / "pages" / f"{page}.png"
        if not img.exists():
            continue
        # Use the same QN syllables for both lang_type runs
        trans = json.loads((REPO / "prepared" / book / "transcriptions" /
                            f"{page}.json").read_text())
        all_syls = [s for col in trans.get("columns", [])
                    for s in col.get("syllables", [])]
        if not all_syls:
            continue

        fname = upload_image(str(img))
        if not fname:
            continue

        # lang_type=1 (default)
        boxes1 = call_api(fname, lang_type=1)
        cols1 = boxes_to_columns(boxes1)
        chars1 = chars_from_columns(cols1)
        h1, t1 = score_alignment(chars1, all_syls, qn_to_nom)

        # lang_type=2 (experimental)
        boxes2 = call_api(fname, lang_type=2)
        cols2 = boxes_to_columns(boxes2)
        chars2 = chars_from_columns(cols2)
        h2, t2 = score_alignment(chars2, all_syls, qn_to_nom)

        s1_h += h1; s1_t += t1
        s2_h += h2; s2_t += t2

        pct1 = h1 / max(t1, 1) * 100
        pct2 = h2 / max(t2, 1) * 100
        print(f"  {book:20s} {page:12s} "
              f"{h1:>3}/{t1:<3} {pct1:>4.1f}%  "
              f"{h2:>3}/{t2:<3} {pct2:>4.1f}%  "
              f"{pct2 - pct1:+5.1f}pp")
        time.sleep(1)

    print("-" * 75)
    p1 = s1_h / max(s1_t, 1) * 100
    p2 = s2_h / max(s2_t, 1) * 100
    print(f"  TOTAL                       "
          f"{s1_h:>3}/{s1_t:<3} {p1:>4.1f}%  "
          f"{s2_h:>3}/{s2_t:<3} {p2:>4.1f}%  {p2-p1:+5.1f}pp")

    out_path = OUT / "lang_type_test.json"
    out_path.write_text(json.dumps({
        "lang_type_1": {"hits": s1_h, "total": s1_t, "pct": round(p1, 2)},
        "lang_type_2": {"hits": s2_h, "total": s2_t, "pct": round(p2, 2)},
        "delta_pp": round(p2 - p1, 2),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
