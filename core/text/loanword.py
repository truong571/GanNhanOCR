"""Catholic loanword (transliteration) detection for Han-Nom labeling.

Bộ corpus chứa các cụm phiên âm tên thánh / địa danh / nghi lễ Công giáo:
    Bà Ma ri a (Maria), Giê su (Jesus), Ina xio (Ignatius),
    An ti ô ki (Antiochia), Rô ma (Roma), Mi sa (Misa),
    sa se do tê (sacerdote), Pha ri sêu, Ki tô, Phê rô, Phao lô, ...

Mỗi cụm là chuỗi âm tiết 1-2 chữ cái; mỗi âm tiết riêng KHÔNG có nghĩa
Hán-Việt độc lập. Tuy nhiên trong bản chép tay, người dịch thường dùng
chữ Hán phiên âm cho từng âm: "Bà" → 婆, "Ma" → 嗎, "Rô" → 嚕, "sa" → 沙.
Những chữ Hán đó CÓ trong dict QN→Nôm.

Quy ước demote (gọi `should_demote_loan_syllable`):
  - syllable nằm trong cụm phiên âm Latin (find_loan_spans phát hiện)
  - VÀ qn_to_nom không có entry cho syllable đó
  → demote thành matched=False, tier=0

Khi dict CÓ entry → giữ nguyên nhãn (bản chép Hán phiên âm là hợp lệ).
"""
import re
import unicodedata

from core.text.lexicon import load_list

_DEFAULT_LOAN_PHRASES = [
    "ba ma ri a",      # Maria
    "ma ri a",
    "gie su",          # Jesus
    "ina xio",         # Ignatius
    "i na xio",
    "ina",             # standalone — VietOCR sometimes drops the "xio" tail
                       # ("ina" is not a valid Vietnamese syllable on its own,
                       # so the standalone form is unambiguous)
    "an ti o ki a",    # Antiochia
    "an ti o ki",
    "ro ma",           # Roma
    "mi sa",           # Misa
    "sa se do te",     # sacerdote
    "pha ri seu",      # Pharisaeus
    "ki to",           # Kitô (Christ)
    "phe ro",          # Phêrô (Peter)
    "phao lo",         # Phaolô (Paul)
]

# Loaded from config/lexicon/loan_phrases.json (fallback: the list above).
LOAN_PHRASES = load_list("loan_phrases.json", _DEFAULT_LOAN_PHRASES)


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "")
        if unicodedata.category(c) != "Mn"
    ).lower()


def find_loan_spans(syllables: list[str]) -> set[int]:
    """Return indices (within `syllables`) covered by any loan phrase.

    Detection is substring-match on the accent-stripped, space-joined
    syllable sequence with word-boundary anchors.
    """
    if not syllables:
        return set()
    flat = [_strip_accents(s or "") for s in syllables]
    joined = " ".join(flat)

    # Map each char offset in `joined` back to syllable index
    char_to_idx: list[int] = []
    for i, tok in enumerate(flat):
        for _ in tok:
            char_to_idx.append(i)
        if i < len(flat) - 1:
            char_to_idx.append(i)  # space; index doesn't matter once outside any match

    covered: set[int] = set()
    for phrase in LOAN_PHRASES:
        for m in re.finditer(
            rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", joined
        ):
            for k in range(m.start(), m.end()):
                if k < len(char_to_idx):
                    covered.add(char_to_idx[k])
    return covered


def should_demote_loan_syllable(syllable: str,
                                qn_to_nom: dict[str, list[str]]) -> bool:
    """Return True iff this loan syllable has NO entry in qn_to_nom.

    Caller must already have verified the syllable is in a loan span
    (use `find_loan_spans` over the column's syllable list first).
    """
    if not syllable:
        return True
    s = syllable.strip()
    return not (qn_to_nom.get(s.lower()) or qn_to_nom.get(s))
