"""Text cleaning, syllable splitting, and saint name normalization."""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Saint name mapping (religious names -> syllable-separated form)
# ---------------------------------------------------------------------------

SAINT_NAMES = {
    "marxiô": "ma rơ xi ô", "marơxiô": "ma rơ xi ô",
    "maria": "ma ri a", "giêsu": "giê su", "phêrô": "phê rô",
    "giuse": "diu xê", "giusê": "diu xê",
    "antôniô": "an tô ni ô", "dominhgô": "do minh cô",
    "đominhgô": "do minh cô", "dôminhgô": "do minh cô",
    "dominhcô": "do minh cô", "phaola": "phao la",
    "phanchicô": "phan chi cô", "catarina": "ca ta ri na",
    "kirixitô": "ki ri xi tô", "rôma": "rô ma",
    "nicolao": "ni cô lao", "nicôlao": "ni cô lao",
    "italia": "i ta li a", "amen": "a men",
    "giêrusalem": "giê ru sa lem",
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
    "rômanô": "rô man ô", "milanô": "mi la nô",
    "amrôxiô": "am rô xay ô", "ambrôxiô": "am bô xi ô",
    "aucutinh": "ao cu tinh",
    "rosariô": "ro sa ri ô",
    "matthêu": "ma thêu",
    "evan": "ê van",
    "vít vồ": "viết vồ", "vít": "viết",
    "batôlamiêu": "ba tô la miêu",
    "stanilao": "sờ ta ni lao", "stanislaghai": "sờ ta ni sờ lao",
    "galilêa": "ga li lê a", "nadarét": "na da rết",
    "aphôcalípsi": "a phô ca líp xi",
    "bảolộc": "bảo lộc",
    "mônrôviđô": "môn rô vi đô",
}


# ---------------------------------------------------------------------------
# Text cleaning functions
# ---------------------------------------------------------------------------

def remove_footnote_markers(text: str) -> str:
    """Remove footnote numbers attached to words: 'Vit-vo1' -> 'Vit-vo'."""
    return re.sub(
        r"(?<=[a-zA-ZÀ-ỹ\u0300-\u036f])\d{1,2}(?=[\s.,;:!?\)\]\"\'»]|$)",
        "", text,
    )


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


def clean_line_text(text: str) -> str:
    """Full cleaning pipeline for one QN text line."""
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


def normalize_syllables(syllables: list[str]) -> list[str]:
    """Normalize syllable list: clean punctuation, expand saint names."""
    result = []
    for syl in syllables:
        cleaned = re.sub(r'["""\'()[\]{}«»,.;:!?…–—\-]', '', syl).strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        if lower in SAINT_NAMES:
            result.extend(SAINT_NAMES[lower].split())
        else:
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
