"""Find suspect syllable pairs from existing transcriptions.

Idea: for each adjacent pair (prev, curr) that appears in the corpus, if a
NEAR-IDENTICAL pair (prev, alt) dominates by a large frequency ratio, then
(prev, curr) is likely an OCR error of (prev, alt).

Example:
    count("nói rằng") = 1,247
    count("nói răng") = 8
    → "nói răng" is suspect, suggested fix: "nói rằng"

Output: evaluation/test_llm_postfix/out/suspects.json

Usage:
    .venv/bin/python evaluation/test_llm_postfix/bigram_finder.py
    .venv/bin/python evaluation/test_llm_postfix/bigram_finder.py --min-ratio 50
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"


def _strip_diacritics(s: str) -> str:
    """Remove diacritics, lowercase. Used as similarity key for fuzzy matching."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d")


def _read_transcriptions(books_dir: Path) -> list[list[str]]:
    """Return list of column-syllable-lists from every page json."""
    sentences = []
    for book_dir in sorted(books_dir.iterdir()):
        trans_dir = book_dir / "transcriptions"
        if not trans_dir.exists():
            continue
        for jf in sorted(trans_dir.glob("page_*.json")):
            data = json.loads(jf.read_text())
            for col in data.get("columns", []):
                syls = col.get("syllables", [])
                if syls:
                    sentences.append(syls)
    return sentences


def build_bigram_counts(sentences: list[list[str]]) -> Counter:
    """Count adjacent syllable pairs across all columns."""
    cnt: Counter = Counter()
    for s in sentences:
        for a, b in zip(s, s[1:]):
            cnt[(a, b)] += 1
    return cnt


def find_suspects(bigrams: Counter, min_ratio: float = 30.0,
                  min_dominant: int = 5) -> list[dict]:
    """For each (prev, curr) bigram, check if a near-identical variant
    (prev, alt) is much more frequent. If ratio > min_ratio → curr is suspect.

    Near-identical = same prev + alt has same diacritic-stripped form as curr.
    """
    # Group bigrams by (prev, stripped_curr)
    groups: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for (prev, curr), c in bigrams.items():
        key = (prev, _strip_diacritics(curr))
        groups[key].append((curr, c))

    suspects = []
    for (prev, _stripped), variants in groups.items():
        if len(variants) < 2:
            continue
        variants.sort(key=lambda x: -x[1])
        dominant_word, dominant_count = variants[0]
        if dominant_count < min_dominant:
            continue
        for alt_word, alt_count in variants[1:]:
            if alt_count == 0 or alt_word == dominant_word:
                continue
            ratio = dominant_count / alt_count
            if ratio >= min_ratio:
                suspects.append({
                    "prev": prev,
                    "wrong": alt_word,
                    "suggested": dominant_word,
                    "wrong_count": alt_count,
                    "dominant_count": dominant_count,
                    "ratio": round(ratio, 1),
                    "bigram_wrong": f"{prev} {alt_word}",
                    "bigram_correct": f"{prev} {dominant_word}",
                })
    # Sort by impact (how many fixes this rule would apply)
    suspects.sort(key=lambda x: -x["wrong_count"])
    return suspects


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--books-dir", default=str(REPO / "prepared"))
    p.add_argument("--min-ratio", type=float, default=30.0,
                   help="Min freq ratio dominant:variant to mark as suspect")
    p.add_argument("--min-dominant", type=int, default=5,
                   help="Dominant bigram must appear at least this many times")
    p.add_argument("--max-show", type=int, default=40)
    args = p.parse_args()

    print(f"Reading transcriptions from {args.books_dir}/...")
    sentences = _read_transcriptions(Path(args.books_dir))
    total_syls = sum(len(s) for s in sentences)
    print(f"  {len(sentences):,} columns, {total_syls:,} syllables total")

    bigrams = build_bigram_counts(sentences)
    print(f"  {len(bigrams):,} unique bigrams\n")

    suspects = find_suspects(bigrams, args.min_ratio, args.min_dominant)
    print(f"Found {len(suspects):,} suspect bigrams "
          f"(ratio ≥ {args.min_ratio}, dominant ≥ {args.min_dominant})")
    print()
    total_fixes = sum(s["wrong_count"] for s in suspects)
    print(f"Total fix opportunities: {total_fixes:,} occurrences")
    print()
    print(f"Top {min(args.max_show, len(suspects))} suspect bigrams:")
    print(f"{'WRONG':25s} → {'CORRECT':25s} | {'count':>5s} → {'×':>5s} | ratio")
    print("-" * 85)
    for s in suspects[:args.max_show]:
        print(f"{s['bigram_wrong']:25s} → {s['bigram_correct']:25s} | "
              f"{s['wrong_count']:>5d} → {s['dominant_count']:>5d} | {s['ratio']:>5.1f}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "suspects.json"
    out_path.write_text(json.dumps(suspects, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print()
    print(f"Wrote {out_path}")
    print()
    print(f"Next step: review suspects, then run llm_fixer.py to confirm with Gemini.")


if __name__ == "__main__":
    main()
