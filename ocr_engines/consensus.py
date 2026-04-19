"""Voting / consensus across OCR engines.

Input: per-engine page JSON (written by each engine's `recognize_page`).
Output: merged page JSON with one consensus decision per (column, char_idx).

Decision rule:
    - If all valid votes agree                   → tier=0, strong
    - If majority (>= ceil(n/2)+1) agrees         → tier=0, strong
    - If a plurality exists (e.g. 1-1-1 but one
      engine is higher-weight)                    → tier=0, medium
    - If no agreement                              → tier=None (defer)

Each engine can carry a weight (default 1.0). Kimhannom is the only
engine with per-page detection responsibility, so it is the tie-breaker
by default.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_WEIGHTS = {
    "kimhannom": 1.2,
    "paddleocrv5_nom": 1.0,
    "nomna_ocr": 1.0,
}


@dataclass
class ConsensusDecision:
    char: str | None
    confidence: float       # 0.0 - 1.0
    agreement: str          # "unanimous" | "majority" | "plurality" | "none"
    votes: dict             # {engine_name: {"char": ..., "confidence": ...}}


def vote(
    engine_results: dict,
    weights: dict | None = None,
) -> ConsensusDecision:
    """Vote across {engine_name: {"char": str|None, "confidence": float}}."""
    weights = weights or DEFAULT_WEIGHTS
    votes = {
        name: r for name, r in engine_results.items()
        if r.get("char")
    }

    if not votes:
        return ConsensusDecision(
            char=None, confidence=0.0, agreement="none",
            votes=engine_results,
        )

    # Weighted sum per candidate char
    score: Counter = Counter()
    for name, r in votes.items():
        w = weights.get(name, 1.0) * float(r.get("confidence", 1.0))
        score[r["char"]] += w

    winner, winner_score = score.most_common(1)[0]
    total = sum(score.values())
    ratio = winner_score / total if total > 0 else 0.0

    n_agreeing = sum(1 for r in votes.values() if r["char"] == winner)
    n_votes = len(votes)

    if n_agreeing == n_votes and n_votes >= 2:
        agreement = "unanimous"
    elif n_agreeing > n_votes / 2:
        agreement = "majority"
    elif n_agreeing >= 1 and ratio > 0.5:
        agreement = "plurality"
    else:
        agreement = "none"
        winner = None
        ratio = 0.0

    return ConsensusDecision(
        char=winner,
        confidence=ratio,
        agreement=agreement,
        votes=engine_results,
    )


def merge_page(
    engine_pages: dict,
    weights: dict | None = None,
) -> dict:
    """Merge {engine_name: page_dict} into one consensus page dict.

    All input pages must come from the same detection (same columns and
    char_idx layout). Mismatched columns are skipped with a warning.
    """
    engines = list(engine_pages.keys())
    if not engines:
        return {"page": None, "columns": []}

    # Anchor on the first engine's column structure
    anchor_name = engines[0]
    anchor = engine_pages[anchor_name]
    page_name = anchor.get("page")

    # Build quick lookup: engine -> (col_num, char_idx) -> char info
    lookup: dict = {}
    for name, page in engine_pages.items():
        idx: dict = {}
        for col in page.get("columns", []):
            for ch in col.get("chars", []):
                idx[(col["column"], ch["char_idx"])] = ch
        lookup[name] = idx

    out = {
        "page": page_name,
        "engines": engines,
        "columns": [],
    }

    for col in anchor.get("columns", []):
        col_num = col["column"]
        col_out = {"column": col_num, "chars": []}
        for ch in col.get("chars", []):
            key = (col_num, ch["char_idx"])
            votes = {}
            for name in engines:
                hit = lookup[name].get(key)
                if hit is None:
                    votes[name] = {"char": None, "confidence": 0.0}
                else:
                    votes[name] = {
                        "char": hit.get("char"),
                        "confidence": float(hit.get("confidence", 0.0)),
                    }
            decision = vote(votes, weights=weights)
            col_out["chars"].append({
                "char_idx": ch["char_idx"],
                "char": decision.char,
                "confidence": decision.confidence,
                "agreement": decision.agreement,
                "votes": decision.votes,
            })
        out["columns"].append(col_out)

    return out


def write_consensus(consensus: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(consensus, f, ensure_ascii=False, indent=2)


def summarize(consensus: dict) -> dict:
    """Quick stats on a consensus page (for CLI reporting)."""
    counts = Counter()
    total = 0
    for col in consensus.get("columns", []):
        for ch in col.get("chars", []):
            total += 1
            counts[ch.get("agreement", "none")] += 1
    return {"total": total, **dict(counts)}
