"""Self-test for the QN-syllable anchor validation + its consensus guard.

    .venv/bin/python -m core.text.syllable_validation_selftest

The QN syllable is the anchor that ties a crop to its Hán-Nôm reading. A garbage
OCR token ('0', '2017', 'ch', 'giêgiung') has no dictionary readings (R = ∅), so
before this guard it could still earn a SILVER label off S1∩S3 alone via the
`s1_inter_s3_out_of_dict` rule. This test pins:
  1. is_plausible_qn_syllable rejects the measured garbage and keeps real syllables.
  2. decide_label demotes a garbage out-of-dict pair to REVIEW while a plausible
     out-of-dict pair still gets SILVER (and the normal in-dict GOLD path is intact).
Exit 0 = pass.
"""
from __future__ import annotations

from core.text.text_utils import is_plausible_qn_syllable
from pipeline.align_engine.consensus import decide_label, S3

_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}  {detail}")


# --- 1. is_plausible_qn_syllable -------------------------------------------
# Real Vietnamese syllables (incl. single-vowel words + longest 7-letter forms).
PLAUSIBLE = ["nhị", "nguyệt", "atina", "giu", "biết", "ê", "ừ", "ở", "gì",
             "nghiêng", "nghuyễn", "quý", "hòa", "Bảy", "THÁNH"]
# Measured OCR garbage that must never anchor a label.
GARBAGE = ["0", "1", "19", "2017", "0001000000", "r1", "038struyen", "tinh%",
           "đ", "ch", "g", "giêgiung", "thoànghoàng", "intexpleman", "", "  "]

for s in PLAUSIBLE:
    check(f"plausible: {s!r}", is_plausible_qn_syllable(s) is True)
for s in GARBAGE:
    check(f"garbage:   {s!r}", is_plausible_qn_syllable(s) is False)

# case/diacritic insensitivity is orthogonal to the verdict
check("case-insensitive", is_plausible_qn_syllable("NHỊ") == is_plausible_qn_syllable("nhị"))

# --- 2. decide_label out-of-dict SILVER guard ------------------------------
# Minimal dict: 二 is a reading of 'nhị' (in-dict); 烏 has NO QN entry (out-of-dict).
QN2NOM = {"nhị": ["二"]}


def s3_backs(ocr_char):
    """A clean S3 whose top == the OCR char, not a dict reading, not rejected."""
    return S3(top_char=ocr_char, cosine=0.9, margin=0.3, top_in_dict=False, reject=False)


# garbage out-of-dict syllable -> guard fires -> REVIEW (not SILVER)
for garbage in ["0", "2017", "ch", "giêgiung"]:
    d = decide_label("烏", garbage, True, QN2NOM, None, s3=s3_backs("烏"))
    check(f"garbage out-of-dict -> REVIEW: {garbage!r}",
          d.tier == "REVIEW", f"got {d.tier}/{d.rule_id}")

# plausible out-of-dict syllable -> SILVER via s1_inter_s3_out_of_dict (unchanged)
d = decide_label("烏", "atina", True, QN2NOM, None, s3=s3_backs("烏"))
check("plausible out-of-dict -> SILVER",
      d.tier == "SILVER" and d.rule_id == "s1_inter_s3_out_of_dict",
      f"got {d.tier}/{d.rule_id}")

# regression: normal in-dict pair still GOLD (guard must not touch this path)
d = decide_label("二", "nhị", True, QN2NOM, None, s3=None)
check("in-dict pair -> GOLD (regression)",
      d.tier == "GOLD" and d.rule_id == "s1_inter_s2_direct",
      f"got {d.tier}/{d.rule_id}")

# regression: even an implausible syllable stays GOLD if the OCR char IS a dict
# reading (the guard only gates the out-of-dict branch, not S1∩S2).
d = decide_label("二", "nhị", True, QN2NOM, None, s3=None)
check("guard scoped to out-of-dict only", d.tier == "GOLD")


# Tiền tố RESULT: = hợp đồng chung với scripts/run_all_selftests.sh (nó grep '^RESULT:').
print(f"\nRESULT: {_passed} passed, {_failed} failed")
raise SystemExit(1 if _failed else 0)
