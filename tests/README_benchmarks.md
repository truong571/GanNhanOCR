# OCR Preprocessing / Denoise Benchmarks

Two scripts under `tests/` that empirically answer:
1. What preprocessing should be applied before calling the HCMUS SinoNom OCR API?
2. Does the current `denoise_image` step help OCR accuracy?

Both score against the Quốc-Ngữ transcription (`prepared/<book>/transcriptions/`)
using column-wise Levenshtein alignment and the QN→Nôm dictionary. Primary
metric is `coverage` = fraction of QN syllables that got paired with an OCR
character whose transcription is listed for that syllable in the dictionary.

Outputs live under:
- `tests/bench_cache/` — cached OCR API responses (keyed by image hash)
- `tests/bench_images/` — exact PNG uploaded to the API
- `tests/bench_results/*.json` — aggregated scores per run

## Env setup

The benchmark scripts only need a tiny subset of the project's deps — the
full `requirements.txt` currently fails on Python 3.13 because `vietocr`
transitively pins `Pillow==10.2.0` (which has a broken `setup.py` on 3.13).

Use the minimal file instead:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r tests/requirements_bench.txt
```

That installs only `numpy, opencv-python, scipy, PyYAML, requests`.

Make sure `.env` contains `SN_OCR_TOKEN=...`.

## 1. Preprocessing variants (A/B/C)

```bash
.venv/bin/python tests/bench_ocr_preprocessing_variants.py --book CacThanhTruyen2 --pages 3
```

Tests 3 variants and prints a ranked table:
| Variant | What it does |
|---|---|
| `raw` | Send the page as-is. |
| `crop_frame` | `detect_text_box` → crop to inside the rectangular outer frame. |
| `crop_erase_cols` | `crop_frame` + inpaint long thin vertical ruling lines. |

Each variant issues **1 upload + 1 OCR call per page** unless the result is
already cached.

## 2. Denoise vs raw comparison

```bash
.venv/bin/python tests/bench_denoise_models.py --book CacThanhTruyen2 --pages 10 --preprocess raw
```

Built-in labels:
- `raw` — original page
- `current` — `lib.image_processing.denoise_image` (baseline)

## Historical result (committed findings)

Over 3 books × 13 pages:
| variant | coverage | note |
|---|---|---|
| raw | **0.566** | winner — used by `step1_extract.py` |
| current_denoise | 0.559 | −0.7pp |
| crop_frame | 0.543 | −2.3pp |

Conclusion applied to production: [step1_extract.py:109-115](../pipeline/step1_extract.py#L109-L115)
sends the raw page directly to the OCR API (no denoise step).

## Interpreting the table

- **coverage** — primary metric; higher is better.
- **hit_rate** — of OCR chars that aligned to a syllable, what fraction had a
  valid QN→Nôm dictionary match.
- **cols Δ / chars Δ** — over/under detection. Close to 0 is ideal.

## Notes

- Each uncached variant costs 1 API call. Cache key is SHA1 of the uploaded PNG.
- Scoring is a proxy, not ground truth (no Nôm-level GT exists in the repo).
- Column pairing is index-based (OCR col i ↔ QN col i).
