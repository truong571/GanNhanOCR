"""Text cleaning, syllable splitting, and saint name normalization."""
from __future__ import annotations

import re
import unicodedata

from core.text.lexicon import load_mapping


# ---------------------------------------------------------------------------
# Vietnamese tone-mark placement normalization  ("hoà"/"khoẻ"/"uý"  vs
# "hòa"/"khỏe"/"úy").  Old ("kiểu cũ") and new ("kiểu mới") conventions
# differ ONLY for the open-syllable glide diphthongs  oa / oe / uy, where the
# old style puts the tone on the 2nd vowel (h-o-à) and the modern style puts
# it on the 1st vowel (h-ò-a).  Every other tone placement is identical in
# both conventions, so we touch nothing else.
#
# The QN→Nôm dictionary stores these almost entirely in the OLD style
# (hoà/uý/thuỷ) while modern VietOCR emits the NEW style (hòa/úy/thủy); without
# canonicalising both sides to one convention the exact dict lookup silently
# misses and the downstream Levenshtein alignment drifts.  We canonicalise to
# the MODERN convention (the project standard) on BOTH the OCR side
# (normalize_syllables) and the dictionary side (load_qn_to_nom).
# ---------------------------------------------------------------------------

# The five Vietnamese tone combining marks (NFD): grave/acute/tilde/hook/dot.
_TONE_MARKS = "\u0300\u0301\u0303\u0309\u0323"  # grave acute tilde hook dot

# Glide diphthongs whose tone position differs old↔new. (v1, v2) lowercased:
# the tone sits on v2 in the old style, on v1 in the modern style.
_GLIDE_PAIRS = {("o", "a"), ("o", "e"), ("u", "y")}

# Run of letters (plus any combining marks). Applied to NFC text, where every
# Vietnamese tone+vowel has a precomposed code point, so a run is a clean word.
_LETTER_RUN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _retone_token(token: str) -> str:
    """Canonicalise tone placement of ONE word-run to the modern convention.

    Moves the tone mark from the 2nd to the 1st vowel only for an *open*
    syllable ending in oa/oe/uy (old style). Leaves closed syllables
    ("hoàng", "khoét"), triphthongs ("khoái", "khuỷu") and the qu-glide
    ("quý" → stays, the u is part of "qu") untouched. Idempotent.
    """
    chars = list(unicodedata.normalize("NFD", token))
    # locate the single tone combining mark
    tone_idx = next((i for i, c in enumerate(chars) if c in _TONE_MARKS), None)
    if tone_idx is None:
        return unicodedata.normalize("NFC", token)

    # base letter currently carrying the tone (v2)
    j = tone_idx - 1
    while j >= 0 and unicodedata.category(chars[j]) == "Mn":
        j -= 1
    if j < 0:
        return unicodedata.normalize("NFC", "".join(chars))

    # previous base letter (v1)
    k = j - 1
    while k >= 0 and unicodedata.category(chars[k]) == "Mn":
        k -= 1
    if k < 0:
        return unicodedata.normalize("NFC", "".join(chars))

    pair = (chars[k].lower(), chars[j].lower())
    if pair not in _GLIDE_PAIRS:
        return unicodedata.normalize("NFC", "".join(chars))

    # v1 must be a plain vowel (no circumflex/horn/breve of its own)
    if any(unicodedata.category(chars[t]) == "Mn" for t in range(k + 1, j)):
        return unicodedata.normalize("NFC", "".join(chars))

    # open syllable: no letter (coda consonant) may follow v2
    if any(c.isalpha() for c in chars[j + 1:]):
        return unicodedata.normalize("NFC", "".join(chars))

    # qu-guard: in "qu+y" the u is a glide, tone stays on y ("quý", not "qúy")
    if pair == ("u", "y"):
        p = k - 1
        while p >= 0 and unicodedata.category(chars[p]) == "Mn":
            p -= 1
        if p >= 0 and chars[p].lower() == "q":
            return unicodedata.normalize("NFC", "".join(chars))

    # move the tone mark from after v2 to immediately after v1
    mark = chars.pop(tone_idx)
    chars.insert(k + 1, mark)
    return unicodedata.normalize("NFC", "".join(chars))


def normalize_tone_marks(text: str) -> str:
    """Canonicalise Vietnamese tone-mark placement to the modern convention.

    "hoà"→"hòa", "khoẻ"→"khỏe", "uý"→"úy", "thuý"→"thúy". Closed syllables,
    triphthongs and the qu-glide are preserved. Safe on already-modern and
    non-Vietnamese text (returns it NFC-normalised, otherwise unchanged).
    """
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    return _LETTER_RUN_RE.sub(lambda m: _retone_token(m.group(0)), text)


# ---------------------------------------------------------------------------
# Saint name mapping (religious names -> syllable-separated form)
# ---------------------------------------------------------------------------

_DEFAULT_SAINT_NAMES = {
    # Core Christian figures (each entry maps "fused QN" -> "syllable-separated
    # form matching how the Nom book writes the name, 1 syllable per Han/Nom char)
    "marxiô": "ma rơ xi ô", "marơxiô": "ma rơ xi ô",
    "maria": "ma ri a", "giêsu": "giê su", "phêrô": "phê rô",
    "giuse": "diu xê", "giusê": "diu xê", "giuong": "giu ong",
    "antôniô": "an tô ni ô", "dominhgô": "do minh cô",
    "đominhgô": "do minh cô", "dôminhgô": "do minh cô",
    "dominhcô": "do minh cô", "phaola": "phao la",
    "phanchicô": "phan chi cô", "catarina": "ca ta ri na",
    "kirixitô": "ki ri xi tô",
    "nicolao": "ni cô lao", "nicôlao": "ni cô lao",
    "amen": "a men",
    "anrê": "an rê", "anre": "an rê",
    "lêô": "lê ô", "marcô": "mac cô",
    "basiliô": "ba si li ô", "linô": "li nô",
    "valêrianô": "va lê ri a nô",
    "ghêrêgôriô": "ghê rê gô ri ô",
    "atanaxiô": "a ta na xi ô",
    "giêrônimô": "khê rô ni mô",
    "giêđônimô": "giê đô ni mô",
    "bênêđichtô": "bê nê đich tô",
    "constantino": "con stan ti nô",
    "contăngtinô": "con tăng ti nô",
    "amrôxiô": "am rô xay ô", "ambrôxiô": "am bô xi ô",
    "aucutinh": "ao cu tinh",
    "rosariô": "ro sa ri ô",
    "matthêu": "ma thêu",
    "evan": "ê van",
    "vít vồ": "viết vồ", "vít": "viết",
    "batôlamiêu": "ba tô la miêu",
    "stanilao": "sờ ta ni lao", "stanislaghai": "sờ ta ni sờ lao",
    "aphôcalípsi": "a phô ca líp xi",
    "bảolộc": "bảo lộc",
    # Added from corpus analysis (SachThanhTruyen scans):
    "inaxu": "i na xu",
    "tragianô": "tra gia nô", "tragiano": "tra gia nô",
    "đêuphêrô": "đê u phê rô", "dêuphêrô": "đê u phê rô",
    "pháppha": "pháp pha", "phapha": "pha pha",
    "ucô": "u cô",
    "garanôbi": "ga ra nô bi",
    "akilipha": "a ki li pha",
    "bonipphasiô": "bô ni pha si ô", "bônipphasiô": "bô ni pha si ô",
    "boniphasiô": "bô ni pha si ô", "bôniphasiô": "bô ni pha si ô",
    "ighêrêgia": "i ghê rê gia",
}


# Place names / toponyms — written phonetically with 1 Han char per syllable
# in 17th-19th century Vietnamese Catholic Nom books.
_DEFAULT_TOPONYMS = {
    "rôma": "rô ma", "roma": "rô ma",
    "italia": "i ta li a",
    "giêrusalem": "giê ru sa lem", "jêrusalem": "giê ru sa lem",
    "galilêa": "ga li lê a", "galilê": "ga li lê",
    "nadarét": "na da rết", "nadaret": "na da rết",
    "antiôki": "an ti ô ki", "antiôkia": "an ti ô ki a",
    "phalansa": "pha lan sa", "phalansô": "pha lan sa",
    "phêxia": "phê xi a", "perxia": "phê xi a",
    "milanô": "mi la nô", "milano": "mi la nô",
    "rômanô": "rô man ô",
    "mônrôviđô": "môn rô vi đô",
    "betlem": "bê lem", "bétlem": "bê lem",
    "naxarét": "na xa rết",
    "ghêrêgia": "ghê rê gia",
}


# Public tables — loaded from config/lexicon/*.json (the single source of truth
# for adding names/places without code edits); the _DEFAULT_* dicts above are
# the built-in fallback used only when the JSON is missing or malformed.
SAINT_NAMES = load_mapping("saint_names.json", _DEFAULT_SAINT_NAMES)
TOPONYMS = load_mapping("toponyms.json", _DEFAULT_TOPONYMS)


# ---------------------------------------------------------------------------
# Text cleaning functions
# ---------------------------------------------------------------------------

def remove_footnote_markers(text: str) -> str:
    """Strip footnote markers attached to words.

    Two forms handled:
      (1) Digits from PDF text: 'Vit-vo1' -> 'Vit-vo'
      (2) VietOCR substitutes for superscript digits: 'Vít-vồ?', "Vít-vồ'",
          'Vít-vồ*', 'Vít-vồ¹' -> 'Vít-vồ'
    """
    text = re.sub(
        r"(?<=[a-zA-ZÀ-ỹ\u0300-\u036f])\d{1,2}(?=[\s.,;:!?\)\]\"\'»]|$)",
        "", text,
    )
    text = re.sub(
        r"(?<=[a-zA-ZÀ-ỹ\u0300-\u036f])[?*'+¹²³⁴⁵⁶⁷⁸⁹⁰]+(?=[\s.,;:!?\)\]\"»]|$)",
        "", text,
    )
    return text


def remove_punctuation(text: str) -> str:
    """Remove punctuation, keep letters and whitespace."""
    return re.sub(r"[.,;:!?\"\'()\[\]{}«»…–—]", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces into one."""
    return " ".join(text.split())


def clean_ocr_artifacts(text: str) -> str:
    """Remove common OCR noise characters: |, ¬, _, °, etc."""
    text = re.sub(r"[|¬_°`~©®™•§¶†‡]", "", text)
    return normalize_whitespace(text)


# Common Vietnamese OCR confusions from VietOCR vgg_transformer on this corpus.
# Maps mistake -> correct. Both sides must be valid Vietnamese, so this is only
# applied LAST-RESORT when the OCR'd syllable is not in the QN dictionary.
_DEFAULT_OCR_CONFUSION_FIXES = {
    "răng": "rằng",
    "lien": "liền",
    "nguời": "người", "nguơi": "người",
    "duoc": "được", "đuoc": "được",
    "troi": "trời", "trơi": "trời",
    "đuc": "đức", "duc": "đức",
    "nuoc": "nước",
    "moi": "mỗi",
    "tron": "trọn",
    "phai": "phải",
    "may": "mày",
    "vi": "vì",
    "mong": "mồng",
    "hom": "hôm",
    "lam": "làm", "lăm": "lắm",
}

# Loaded from config/lexicon/ocr_confusions.json (fallback: the dict above).
OCR_CONFUSION_FIXES = load_mapping("ocr_confusions.json", _DEFAULT_OCR_CONFUSION_FIXES)


def fix_common_ocr_confusion(syllable: str) -> str | None:
    """Return corrected form if syllable is a known VietOCR mistake, else None.

    Caller must verify corrected form is in the QN dict before substituting.
    """
    return OCR_CONFUSION_FIXES.get(syllable.lower())


def clean_line_text(text: str) -> str:
    """Full cleaning pipeline for one QN text line.

    NFC-normalize first so the footnote-marker regex sees pre-composed
    characters (VietOCR sometimes emits decomposed form NFD).
    """
    text = unicodedata.normalize("NFC", text)
    text = remove_footnote_markers(text)
    text = clean_ocr_artifacts(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text


def split_to_syllables(text: str) -> list[str]:
    """Split text into syllables (each = 1 Nom character).

    Also splits hyphenated words: 'I-na-xu' -> ['I', 'na', 'xu']
    """
    syllables = []
    for word in text.split():
        for part in word.split("-"):
            part = part.strip()
            if part:
                syllables.append(part)
    return syllables


def normalize_syllables(syllables: list[str], qn_dict: set[str] | None = None) -> list[str]:
    """Normalize syllable list for Levenshtein alignment.

    Steps per syllable:
      1. Strip residual punctuation.
      2. If matches a fused saint name -> expand to space-separated form
         (e.g. "marxiô" -> ["ma","rơ","xi","ô"], matching how the Nom book
         writes the name with 1 char per syllable).
      3. If matches a toponym -> expand similarly.
      4. If not in the QN dict (passed in) AND looks like a known OCR
         confusion (răng vs rằng, duoc vs được, ...) -> fix it.
         Skipped if `qn_dict` is None.
      5. Otherwise keep as-is.

    `qn_dict` lets us only apply OCR-confusion fixes when the syllable would
    otherwise miss dict-lookup. Pass a set of lowercased QN syllables.
    """
    result = []
    for syl in syllables:
        cleaned = re.sub(r'["""\'()[\]{}«»,.;:!?…–—\-]', '', syl).strip()
        if not cleaned:
            continue
        # Canonicalise tone-mark placement to the modern convention BEFORE any
        # dict / saint / toponym lookup, so OCR ("hòa"/"úy") and the dictionary
        # (canonicalised the same way) always agree.
        cleaned = normalize_tone_marks(cleaned)
        lower = cleaned.lower()
        if lower in SAINT_NAMES:
            result.extend(SAINT_NAMES[lower].split())
            continue
        if lower in TOPONYMS:
            result.extend(TOPONYMS[lower].split())
            continue
        if qn_dict is not None and lower not in qn_dict:
            fixed = fix_common_ocr_confusion(cleaned)
            if fixed and fixed.lower() in qn_dict:
                result.append(fixed)
                continue
        result.append(cleaned)
    return result


def has_vietnamese_diacritics(text: str) -> bool:
    """Check if text has Vietnamese diacritics (detect poor OCR)."""
    viet_chars = set(
        "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
        "ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụ"
        "ưứừửữựỳýỷỹỵđ"
    )
    viet_chars |= {c.upper() for c in viet_chars}
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return False
    count = sum(1 for c in text.lower() if c in viet_chars)
    return (count / total_alpha) > 0.05


def fold_text(text: str) -> str:
    """Normalize text: lowercase, remove diacritics, d->d."""
    text = unicodedata.normalize("NFC", text.lower())
    text = text.replace("đ", "d").replace("Đ", "d")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


# The longest phonotactically-valid Vietnamese syllable ("nghiêng", "nghuyễn") is
# 7 letters: onset ≤3 (ngh) + nucleus ≤3 (uyê/ươ) + coda ≤2 (ng/nh). Longer runs are
# OCR merges of ≥2 syllables ("giêgiung", "thoànghoàng").
_MAX_QN_SYLLABLE_LEN = 7
_QN_VOWELS = frozenset("aeiouy")


def is_plausible_qn_syllable(s: str) -> bool:
    """True if `s` could be a real Vietnamese Quốc-Ngữ syllable.

    The QN syllable is the ANCHOR tying a crop to its Hán-Nôm reading, so a token
    that is not a plausible syllable must never confirm a label. Rejects the OCR
    garbage measured in this corpus:
      - digit / symbol tokens      '0', '2017', 'r1', 'tinh%'  (leaked markers/refs)
      - vowel-less consonant stubs 'đ', 'ch', 'g'              (truncated glyphs)
      - merged multi-syllable runs 'giêgiung', 'thoànghoàng'   (len > 7)
    Plausible = 1..7 characters, all alphabetic, ≥1 vowel. Folds first, so it is
    case/diacritic-insensitive and orthogonal to tone/NFC canonicalisation.
    """
    s = (s or "").strip()
    if not (1 <= len(s) <= _MAX_QN_SYLLABLE_LEN):
        return False
    if not all(c.isalpha() for c in s):
        return False
    return any(c in _QN_VOWELS for c in fold_text(s))


def simple_levenshtein(s1: str, s2: str) -> int:
    """Simple Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return simple_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            ))
        prev_row = curr_row
    return prev_row[-1]
