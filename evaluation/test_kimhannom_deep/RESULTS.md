# Deep Kimhannom OCR Investigation — Results

Tested 2026-05-17 to verify if Kimhannom OCR quality can be improved.

## Conclusion: NO improvement worth merging to main pipeline

## What we verified

1. **Cache is fresh** — `prepared/<book>/detected/*_ocr_cache.json` matches
   what Kimhannom API returns NOW. Re-calling the API doesn't give
   different results.

2. **Kimhannom does read accurately on some content** — e.g. page_0012
   box 2 (`二月二十八日`) is recognized perfectly. Main-text boxes for
   common Han-Viet sequences (Đức Bà, Đức Chúa Giê-su, Pha-lan-sa) all
   correctly transcribed.

3. **Parameter sweep — almost no impact**
   - `ocr_id=1` vs `ocr_id=2`: identical output
   - `lang_type=1` vs `lang_type=3`: identical (same model)
   - `lang_type=2`: genuinely different model, but ~equal accuracy on average
   - `font_type`, `reading_direction`: no observed difference

4. **Ensemble lang=1 + lang=2 with per-page best selection**
   - Default (lang=1): 5.77% tier-1 on 4 pages
   - Ensemble:        7.12% tier-1 (+1.35pp)
   - Cost: 2× API calls per page (~30-40 min extra step-1 time)
   - **Decision: NOT MERGED** — marginal gain, small sample size

## Why ~7% tier-1 is the real ceiling

The mismatches we see (e.g. Kimhannom outputs `工` where syllable is "đã")
are RAW OCR errors, not alignment bugs. We confirmed this by:
- Re-running with cluster-based alignment (+0.74pp only)
- Running other OCR engines (Tesseract, PaddleOCR, Gemini, OCR.space —
  all WORSE than Kimhannom on this corpus)

The bottleneck is the model itself. Kimhannom was trained on Han-Nom but
not on **Catholic handwriting from the 17th-19th century specifically**,
which is the niche this corpus comes from. No publicly available engine
covers this niche.

## What's in this folder

- `probe_api.py`         — verify cache freshness + show raw API boxes
- `test_lang_type2.py`   — compare lang_type=1 vs lang_type=2 on 10 pages
- `test_ensemble.py`     — 5-variant sweep + per-page best ensemble
- `out/`                 — JSON results from the above
