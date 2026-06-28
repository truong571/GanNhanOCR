# Lexicon config — Catholic transliteration data

These plain-data files are the **single source of truth** for the saint names,
place names, loan phrases and OCR-confusion fixes used while labeling Hán-Nôm
Catholic books. They were extracted out of the Python source
(`core/text/text_utils.py`, `core/text/loanword.py`) so a new book can be
supported by **editing JSON only — no code change**.

Loaded by `core/text/lexicon.py`. Override the directory with the
`GANNHANOCR_LEXICON_DIR` environment variable.

| File | Type | Meaning |
|------|------|---------|
| `saint_names.json`   | `{fused_qn: "syllable separated"}` | Tên Thánh: `"maria": "ma ri a"` — fused QN form → space-separated syllables matching 1 Hán/Nôm char per syllable. |
| `toponyms.json`      | `{fused_qn: "syllable separated"}` | Địa danh phiên âm: `"rôma": "rô ma"`. |
| `loan_phrases.json`  | `["space separated", ...]` | Accent-stripped loan phrases used to detect transliteration spans. |
| `ocr_confusions.json`| `{mistake: correct}` | VietOCR confusions, applied only when the mistake is missing from the QN dict. |

## Rules
- Keys/values are matched **case-insensitively, lowercased** by the pipeline.
- In the mapping files, any key starting with `_` is ignored (use it for notes).
- The values are written in the **modern tone-mark convention**
  (`hòa`, not `hoà`) — the pipeline canonicalizes tone placement before lookup
  (`core.text.text_utils.normalize_tone_marks`).
- If a file is missing or malformed, the pipeline falls back to the built-in
  defaults (`_DEFAULT_*` in the source) and logs one warning — it never crashes.

## Adding a new saint / place for a new book
1. Open the relevant JSON.
2. Add `"fusedqnform": "the syllables one per nom char"`.
3. Re-run the pipeline. Done — no source edit.
