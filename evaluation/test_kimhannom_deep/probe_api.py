"""Deep probe of Kimhannom OCR API.

Tests:
  A. Re-call API on page 12 → compare with cached output (verify cache fresh).
  B. Show RAW boxes (transcription strings), not the per-char split.
  C. Sweep `ocr_id`, `font_type`, `reading_direction` parameters to see if
     any give better output.
  D. Try multiple pages to see if Kimhannom is consistently weak or just
     weak on certain page types.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.ocr.ocr_api import upload_image, _get_ocr_token, _SN_DOMAIN
import requests

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


def recognize_with_params(file_name: str, params: dict) -> dict | None:
    """Direct API call with custom params — bypass core.ocr_api default body."""
    url = f"https://{_SN_DOMAIN}/api/web/clc-sinonom/image-ocr"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Authorization": f"Bearer {_get_ocr_token()}",
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {"file_name": file_name, **params}
    try:
        resp = requests.post(url, json=body, headers=headers, verify=False, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def show_raw_boxes(boxes: list[dict], max_show: int = 30) -> str:
    """Format raw API boxes for human inspection."""
    lines = []
    for i, box in enumerate(boxes[:max_show], 1):
        text = box.get("transcription", "").strip()
        pts = box.get("points", [])
        x = pts[0][0] if pts else 0
        y = pts[0][1] if pts else 0
        lines.append(f"  box{i:3d}  at ({x:4d},{y:4d})  text={text!r}")
    if len(boxes) > max_show:
        lines.append(f"  ... and {len(boxes) - max_show} more boxes")
    return "\n".join(lines)


def test_page(book: str, page: str, also_sweep: bool = False) -> dict:
    page_img = REPO / "prepared" / book / "pages" / f"{page}.png"
    cache_path = REPO / "prepared" / book / "detected" / f"{page}_ocr_cache.json"
    print(f"\n{'='*70}\n{book}/{page}\n{'='*70}")

    if not page_img.exists():
        return {"error": f"missing {page_img}"}

    # A: re-call API NOW with default params
    print("[A] Re-calling Kimhannom API with default params...")
    file_name = upload_image(str(page_img))
    if not file_name:
        return {"error": "upload failed"}
    t0 = time.time()
    fresh = recognize_with_params(file_name, {
        "ocr_id": 1, "lang_type": 1,
        "reading_direction": 1, "font_type": 1,
    })
    elapsed = time.time() - t0
    if "error" in fresh:
        print(f"   API error: {fresh['error']}")
        return fresh
    fresh_boxes = fresh.get("data", {}).get("details", {}).get("details", [])
    print(f"   {len(fresh_boxes)} boxes returned ({elapsed:.1f}s)")
    print(f"\n[B] Sample of RAW API boxes (with full transcription strings):")
    print(show_raw_boxes(fresh_boxes, max_show=15))

    # Compare with cache
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        cached_boxes = cache.get("boxes_raw") or []
        cache_texts = sorted([b.get("transcription", "").strip()
                              for b in cached_boxes])
        fresh_texts = sorted([b.get("transcription", "").strip()
                              for b in fresh_boxes])
        identical = cache_texts == fresh_texts
        print(f"\n   Cache vs fresh API output: {'IDENTICAL' if identical else 'DIFFERENT'}")
        if not identical:
            only_cache = set(cache_texts) - set(fresh_texts)
            only_fresh = set(fresh_texts) - set(cache_texts)
            print(f"     in cache only: {len(only_cache)}  e.g. {sorted(only_cache)[:3]}")
            print(f"     in fresh only: {len(only_fresh)}  e.g. {sorted(only_fresh)[:3]}")

    out: dict = {
        "book": book, "page": page,
        "n_boxes_default": len(fresh_boxes),
        "default_boxes": fresh_boxes[:50],
    }

    if also_sweep:
        # C: sweep alternative params
        print(f"\n[C] Sweeping params (ocr_id, lang_type, font_type)...")
        # Try a handful of plausible variants. Kimhannom API doc isn't public
        # so these are guesses based on common conventions.
        variants = [
            {"ocr_id": 1, "lang_type": 1, "reading_direction": 1, "font_type": 1},  # default
            {"ocr_id": 1, "lang_type": 1, "reading_direction": 1, "font_type": 2},  # try font 2
            {"ocr_id": 1, "lang_type": 1, "reading_direction": 2, "font_type": 1},  # horizontal
            {"ocr_id": 2, "lang_type": 1, "reading_direction": 1, "font_type": 1},  # ocr 2
            {"ocr_id": 1, "lang_type": 2, "reading_direction": 1, "font_type": 1},  # lang 2
        ]
        sweep = []
        for v in variants:
            r = recognize_with_params(file_name, v)
            if "error" in r:
                sweep.append({"params": v, "error": r["error"][:60]})
                continue
            bx = r.get("data", {}).get("details", {}).get("details", [])
            sweep.append({"params": v, "n_boxes": len(bx)})
            print(f"   {v}  → {len(bx)} boxes")
        out["sweep"] = sweep

    return out


def main() -> None:
    # 3 different pages from same book to see consistency
    results = []
    for i, (book, page, sweep) in enumerate([
        ("SachThanhTruyen2", "page_0012", True),   # first page of chapter, with date
        ("SachThanhTruyen2", "page_0100", False),  # mid-book body
        ("SachThanhTruyen4", "page_0050", False),  # different book
    ]):
        r = test_page(book, page, also_sweep=sweep)
        results.append(r)

    out_path = OUT / "probe_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n\nWrote {out_path}")


if __name__ == "__main__":
    main()
