"""Three-signal consensus labeling + GOLD/SILVER/REVIEW tiering.

Per Nôm crop we have up to three INDEPENDENT noisy signals:
  S1  ocr_char         — the SinoNom OCR's proposed character.
  S2  qn_to_nom[syl]   — the dictionary's valid Nôm readings R of the syllable.
  S3  visual rank      — DINOv2/FontDiffusion cosine of the crop vs the
                          candidate glyphs ({ocr_char} ∪ R), OPTIONAL.

A label is trustworthy when independent signals AGREE; we never emit a label
from one signal alone. Tiers:

  GOLD   — column anchored AND S1∩S2 agree (ocr_char is a dict reading of the
           syllable). DICTIONARY-ONLY, so it is S3-independent and is the safe
           floor any thesis can report. ~99% precise where it fires.
  SILVER — S1∩S2 do NOT agree, but S3 (vision) breaks the tie with enough
           confidence:
             (a) a dict reading r∈R wins the visual rank  -> vision CORRECTED
                 the OCR; label = r  (rule s2_inter_s3).
             (b) syllable not in dict (Ext-B variant) and ocr_char wins
                 visually -> label = ocr_char (rule s1_inter_s3).
           Needs S3; without it these go to REVIEW.
  REVIEW — diverged column, no ocr_char, or S3 below threshold. Never a label.

Signal-combination only; WHICH crop a syllable belongs to is decided upstream
by the banded anchored alignment (anchor_align.py).
"""
from __future__ import annotations

from dataclasses import dataclass

# Visual thresholds — PLACEHOLDERS, calibrate on a held-out set (FLOW.md §9).
# DINOv2 cosine is mapped to [0,1] via (cos+1)/2; project default "match" = 0.75.
TAU_SILVER = 0.60      # min visual score of the winning candidate
DELTA_SILVER = 0.04    # min (winner − runner-up) margin


@dataclass
class S3:
    """Optional visual signal: best candidate over {ocr_char} ∪ dict-readings."""
    top_char: str       # argmax candidate
    cosine: float       # its score in [0,1]
    margin: float       # winner − runner-up
    top_in_dict: bool   # is top_char a dict reading R of the syllable?


@dataclass
class LabelDecision:
    label: str | None
    syllable: str
    tier: str           # GOLD | SILVER | REVIEW
    rule_id: str
    confirmed: bool      # dict-confirmed (S1∩S2)


def decide_label(ocr_char: str | None,
                 syllable: str,
                 column_count_matched: bool,
                 qn_to_nom: dict[str, list[str]],
                 similar_dict: dict[str, list[str]] | None = None,
                 s3: S3 | None = None,
                 anchored: bool = False) -> LabelDecision:
    """Combine S1/S2/(S3) into a label + tier for one aligned pair.

    `anchored` = the pair is inside a dict-confirmed run, so its local register
    is certain even if the whole column's counts diverged -> GOLD-eligible too.
    """
    syl = (syllable or "").lower()
    R = qn_to_nom.get(syl, [])
    gold_ok = column_count_matched or anchored

    # --- GOLD : S1 ∩ S2 dictionary confirmation (S3-independent) ----------
    direct = bool(ocr_char and ocr_char in R)
    similar = False
    if not direct and ocr_char and similar_dict:
        sims = similar_dict.get(ocr_char, [])
        similar = any(s in R for s in sims)
    if gold_ok and (direct or similar):
        return LabelDecision(ocr_char, syllable, "GOLD",
                             "s1_inter_s2_direct" if direct else "s1_inter_s2_similar",
                             True)

    # --- SILVER : vision breaks the tie (needs S3) ------------------------
    if gold_ok and s3 is not None and s3.cosine >= TAU_SILVER and s3.margin >= DELTA_SILVER:
        if s3.top_in_dict:
            # vision picked a valid dict reading of the syllable -> OCR corrected
            return LabelDecision(s3.top_char, syllable, "SILVER",
                                 "s2_inter_s3_corrected", False)
        if ocr_char and s3.top_char == ocr_char and not R:
            # syllable absent from dict (Ext-B variant); vision backs the OCR char
            return LabelDecision(ocr_char, syllable, "SILVER",
                                 "s1_inter_s3_out_of_dict", False)

    # --- REVIEW -----------------------------------------------------------
    reason = "diverged_column" if not gold_ok else (
        "below_visual_threshold" if s3 is not None else "unconfirmed_no_s3")
    return LabelDecision(None, syllable, "REVIEW", reason, False)
