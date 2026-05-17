"""Test free web OCR APIs that handle Chinese — verify if they work better on
this Han-Nom handwritten corpus.

Tests:
  1. OCR.space free API (no signup, 500 req/day, supports `chs`/`cht`)
  2. Larger context: send WHOLE PAGE instead of tiny per-char crops

Picks 3 sample pages from SachThanhTruyen2 and compares:
  - Kimhannom (existing) — full page OCR
  - OCR.space chs (Simplified Chinese)
  - OCR.space cht (Traditional Chinese)

Output: prints chars detected per service + intersection with expected chars
(from VietOCR QN syllables → dict lookup).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

OCRSPACE_URL = "https://api.ocr.space/parse/image"
OCRSPACE_APIKEY = "helloworld"  # public demo key, very limited; works for quick test


def ocrspace_call(image_path: str, lang: str) -> str:
    """Call OCR.space free API. Returns extracted text (may be empty)."""
    with open(image_path, "rb") as f:
        try:
            resp = requests.post(
                OCRSPACE_URL,
                files={"file": f},
                data={"language": lang, "apikey": OCRSPACE_APIKEY,
                      "isOverlayRequired": False},
                timeout=60,
            )
            resp.raise_for_status()
            j = resp.json()
            parsed = j.get("ParsedResults") or []
            if not parsed:
                err = j.get("ErrorMessage", j.get("ErrorDetails", "?"))
                return f"<ERROR: {err}>"
            return parsed[0].get("ParsedText", "")
        except Exception as e:
            return f"<EXCEPTION: {type(e).__name__}: {e}>"


def count_cjk(text: str) -> int:
    return sum(1 for c in text if 0x3400 <= ord(c) <= 0x9FFF
               or 0x20000 <= ord(c) <= 0x2FFFF)


def cjk_chars(text: str) -> set[str]:
    return {c for c in text if 0x3400 <= ord(c) <= 0x9FFF
            or 0x20000 <= ord(c) <= 0x2FFFF}


def main() -> None:
    test_pages = [
        ("SachThanhTruyen2", "page_0012"),
        ("SachThanhTruyen2", "page_0100"),
        ("SachThanhTruyen4", "page_0050"),
    ]

    md = ["# Web OCR APIs test — Chinese OCR services on full page", "",
          "Service: OCR.space free API (`helloworld` demo key, very limited).",
          "Hypothesis: web Chinese OCR might work better than Tesseract/Paddle local",
          "because they use bigger models + full-page context (not tiny crop).", ""]

    for book, page in test_pages:
        page_img = REPO / "prepared" / book / "pages" / f"{page}.png"
        cache_path = REPO / "prepared" / book / "detected" / f"{page}_ocr_cache.json"
        if not page_img.exists():
            print(f"!! {page_img} missing")
            continue

        # Reference: Kimhannom's per-char output (existing data)
        kim_cache = json.loads(cache_path.read_text())
        kim_chars = []
        for col in kim_cache.get("columns", []):
            for c in col:
                kim_chars.append(c["char"])
        kim_set = set(c for c in kim_chars if c)
        kim_n_unique = len(kim_set)

        print(f"\n=== {book}/{page} ===")
        print(f"Page image: {page_img} ({page_img.stat().st_size/1024:.0f} KB)")
        print(f"Kimhannom found {len(kim_chars)} chars ({kim_n_unique} unique)")

        # Test OCR.space with both Chinese variants
        for lang, name in [("chs", "Simplified"), ("cht", "Traditional")]:
            t0 = time.time()
            text = ocrspace_call(str(page_img), lang)
            elapsed = time.time() - t0
            chars = cjk_chars(text)
            overlap_with_kim = chars & kim_set
            print(f"  OCR.space {name} ({lang}): {len(chars)} unique CJK chars, "
                  f"{count_cjk(text)} total, {len(overlap_with_kim)} overlap Kimhannom "
                  f"({elapsed:.1f}s)")
            print(f"    sample: {text[:200].strip()!r}")

            md += [f"### {book} / {page} — OCR.space {name}",
                   f"- detected: {count_cjk(text)} CJK chars ({len(chars)} unique)",
                   f"- overlap with Kimhannom: {len(overlap_with_kim)} chars",
                   f"- raw output (first 200 chars):",
                   f"  ```",
                   f"  {text[:300].strip()}",
                   f"  ```",
                   ""]

            time.sleep(2)  # be nice to free API

    (OUT / "web_apis_test.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT / 'web_apis_test.md'}")


if __name__ == "__main__":
    main()
