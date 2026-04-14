"""Dictionary loading, reverse lookup, and character specificity."""

import ast
import csv

from lib.text_utils import fold_text, simple_levenshtein


def load_qn_to_nom(dict_path: str, encoding: str = "utf-8-sig") -> dict[str, list[str]]:
    """Load QuocNgu -> SinoNom dictionary.

    Returns: {qn_word_lower: [nom_char1, nom_char2, ...]}
    """
    trans_dict: dict[str, list[str]] = {}
    with open(dict_path, "r", encoding=encoding) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                word = row[0].strip().lower()
                char = row[1].strip()
                if word and char:
                    trans_dict.setdefault(word, []).append(char)
    return trans_dict


def build_nom_to_qn(qn_to_nom: dict[str, list[str]]) -> dict[str, list[str]]:
    """Build reverse dictionary: SinoNom char -> [QN readings].

    Derived from qn_to_nom by inverting the mapping.
    """
    nom_to_qn: dict[str, list[str]] = {}
    for qn, chars in qn_to_nom.items():
        for c in chars:
            nom_to_qn.setdefault(c, [])
            if qn not in nom_to_qn[c]:
                nom_to_qn[c].append(qn)
    return nom_to_qn


def load_similarity_dict(dict_path: str, encoding: str = "utf-8-sig") -> dict[str, list[str]]:
    """Load SinoNom similar characters dictionary.

    Returns: {nom_char: [similar1, similar2, ...]}  (up to 20 similar chars)
    """
    similar_dict: dict[str, list[str]] = {}
    with open(dict_path, "r", encoding=encoding) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                char = row[0].strip()
                try:
                    similars = ast.literal_eval(row[1].strip())
                    if isinstance(similars, list):
                        similar_dict[char] = similars
                except (ValueError, SyntaxError):
                    similar_dict.setdefault(char, []).append(row[1].strip())
    return similar_dict


def build_char_specificity(qn_to_nom: dict[str, list[str]]) -> dict[str, int]:
    """Count how many QN words map to each Nom character.

    Characters mapping to fewer words are more specific -> higher priority.
    """
    char_word_count: dict[str, int] = {}
    for _word, chars in qn_to_nom.items():
        for c in chars:
            char_word_count[c] = char_word_count.get(c, 0) + 1
    return char_word_count


def cjk_block_score(char: str) -> float:
    """Unicode block priority score. CJK Unified > Extension > PUA."""
    cp = ord(char)
    if 0x4E00 <= cp <= 0x9FFF:
        return 1.0   # CJK Unified (most common)
    if 0xF900 <= cp <= 0xFAFF:
        return 0.8   # CJK Compatibility
    if 0x3400 <= cp <= 0x4DBF:
        return 0.5   # CJK Extension A
    if 0x20000 <= cp <= 0x2A6DF:
        return 0.3   # CJK Extension B
    if 0xE000 <= cp <= 0xF8FF:
        return 0.05  # Private Use Area (avoid)
    return 0.2


def fuzzy_dict_lookup(
    qn_word: str,
    qn_to_nom: dict[str, list[str]],
    max_dist: int = 1,
) -> list[str]:
    """Find candidates via fuzzy matching when exact lookup fails.

    Searches keys in qn_to_nom with Levenshtein distance <= max_dist.
    Uses folded text (no diacritics) to avoid false positives.
    """
    target = fold_text(qn_word)
    if len(target) <= 1:
        return []

    candidates = []
    seen: set[str] = set()
    for key in qn_to_nom:
        key_folded = fold_text(key)
        if abs(len(key_folded) - len(target)) > max_dist:
            continue
        dist = simple_levenshtein(target, key_folded)
        if 0 < dist <= max_dist:
            for c in qn_to_nom[key]:
                if c not in seen:
                    seen.add(c)
                    candidates.append(c)
    return candidates
