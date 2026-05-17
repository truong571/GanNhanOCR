"""Per-page ensemble: try multiple Kimhannom param combinations + per-page
keep BEST tier-1 result. Also sweep more lang_type / ocr_id values.

Hypothesis: lang_type=1 and lang_type=2 are two different OCR models. Each is
better on different page styles. Smart selection per-page could give the
union of their strengths.
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

# Variants — start with the 2 known to work, plus a few exploratory
VARIANTS = [
    ("v1_default",     {"ocr_id": 1, "lang_type": 1, "reading_direction": 1, "font_type": 1}),
    ("v2_lang2",       {"ocr_id": 1, "lang_type": 2, "reading_direction": 1, "font_type": 1}),
    ("v3_lang3",       {"ocr_id": 1, "lang_type": 3, "reading_direction": 1, "font_type": 1}),
    ("v4_ocr2_lang1",  {"ocr_id": 2, "lang_type": 1, "reading_direction": 1, "font_type": 1}),
    ("v5_ocr2_lang2",  {"ocr_id": 2, "lang_type": 2, "reading_direction": 1, "font_type": 1}),
]


def call_api(file_name: str, params: dict) -> list[dict] | None:
    url = f"https://{_SN_DOMAIN}/api/web/clc-sinonom/image-ocr"
    body = {"file_name": file_name, **params}
    try:
        resp = requests.post(url, json=body, headers={
            "User-Agent": "Mozilla/5.0",
            "Authorization": f"Bearer {_get_ocr_token()}",
            "Content-Type": "application/json; charset=utf-8",
        }, verify=False, timeout=60)
        resp.raise_for_status()
        j = resp.json()
        if not j.get("is_success"):
            return None
        return j.get("data", {}).get("details", {}).get("details", [])
    except Exception:
        return None


def load_dict() -> dict[str, set[str]]:
    df = pd.read_csv(REPO / "Dict" / "QuocNgu_SinoNom_TongHop3.csv")
    out: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        qn = str(r["QuocNgu"]).strip().lower()
        nom = str(r["SinoNom"]).strip()
        if qn and nom != "nan":
            out.setdefault(qn, set()).update(c for c in nom if not c.isspace())
    return out


def chars_from_columns(cols):
    out = []
    for col in cols:
        for c in col:
            bb = c["bbox"]
            out.append({"char_idx": len(out),
                        "bbox": [int(b) for b in bb],
                        "height": int(bb[3] - bb[1]),
                        "width": int(bb[2] - bb[0]),
                        "ocr_char": c.get("char")})
    return out


def score(chars, syls, qn_to_nom):
    pairs = levenshtein_align(chars, syls, qn_to_nom=qn_to_nom)
    hits, total = 0, 0
    for p in pairs:
        if p.get("type") != "match":
            continue
        ch = (p.get("char") or {}).get("ocr_char", "")
        sy = (p.get("syllable") or "").strip().lower()
        if not ch or not sy:
            continue
        total += 1
        if ch in qn_to_nom.get(sy, set()):
            hits += 1
    return hits, total


def main() -> None:
    qn_to_nom = load_dict()
    # Pick 8 pages random
    pages_all = []
    for book_dir in sorted((REPO / "prepared").iterdir()):
        if not book_dir.is_dir() or book_dir.name.startswith("_"):
            continue
        for tp in sorted((book_dir / "transcriptions").glob("page_*.json")):
            pages_all.append((book_dir.name, tp.stem))
    random.seed(22)
    pages = random.sample(pages_all, 8)
    print(f"Testing {len(VARIANTS)} variants on {len(pages)} random pages\n")

    totals = {name: {"h": 0, "t": 0, "fails": 0} for name, _ in VARIANTS}
    per_page = []
    ensemble_h = ensemble_t = 0

    for book, page in pages:
        img = REPO / "prepared" / book / "pages" / f"{page}.png"
        trans_path = REPO / "prepared" / book / "transcriptions" / f"{page}.json"
        if not img.exists() or not trans_path.exists():
            continue
        trans = json.loads(trans_path.read_text())
        syls = [s for col in trans.get("columns", []) for s in col.get("syllables", [])]
        if not syls:
            continue

        fname = upload_image(str(img))
        if not fname:
            continue

        row = {"book": book, "page": page, "syllables": len(syls)}
        per_variant_hits = {}
        for vname, vparams in VARIANTS:
            boxes = call_api(fname, vparams)
            if boxes is None:
                totals[vname]["fails"] += 1
                row[vname] = "API fail"
                per_variant_hits[vname] = None
                time.sleep(1)
                continue
            cols = boxes_to_columns(boxes)
            chars = chars_from_columns(cols)
            h, t = score(chars, syls, qn_to_nom)
            totals[vname]["h"] += h
            totals[vname]["t"] += t
            row[vname] = f"{h}/{t} ({h/max(t,1)*100:.1f}%)"
            per_variant_hits[vname] = (h, t)
            time.sleep(1)

        # Ensemble: take the best variant for THIS page
        valid = {k: v for k, v in per_variant_hits.items() if v is not None}
        if valid:
            best_name, (best_h, best_t) = max(valid.items(),
                                              key=lambda kv: kv[1][0])
            row["ensemble_best"] = best_name
            ensemble_h += best_h
            ensemble_t += best_t

        per_page.append(row)
        print(f"  {book:20s} {page:12s}  " +
              "  ".join(f"{vname}={row[vname]}" for vname, _ in VARIANTS))
        if "ensemble_best" in row:
            print(f"     → ensemble_best = {row['ensemble_best']}")

    print(f"\n{'='*70}")
    print(f"{'Variant':18s} {'Hits/Total':>15s} {'%':>7s}  fails")
    print("-" * 60)
    for vname, _ in VARIANTS:
        t = totals[vname]
        pct = t["h"] / max(t["t"], 1) * 100
        print(f"  {vname:16s} {t['h']:>5}/{t['t']:<5}  {pct:>5.2f}%  {t['fails']}")
    print("-" * 60)
    en_pct = ensemble_h / max(ensemble_t, 1) * 100
    print(f"  ENSEMBLE (per-page best)   {ensemble_h:>5}/{ensemble_t:<5}  {en_pct:>5.2f}%")

    out_path = OUT / "ensemble_test.json"
    out_path.write_text(json.dumps({
        "variants": [{"name": n, "params": p,
                      "total_hits": totals[n]["h"], "total": totals[n]["t"],
                      "pct": round(totals[n]["h"]/max(totals[n]["t"],1)*100, 2),
                      "fails": totals[n]["fails"]}
                     for n, p in VARIANTS],
        "ensemble_best_per_page": {"hits": ensemble_h, "total": ensemble_t,
                                    "pct": round(en_pct, 2)},
        "per_page": per_page,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
