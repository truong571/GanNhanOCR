"""Shared helpers for OCR preprocessing / denoise benchmark scripts.

- Loads a page image + its Quốc-Ngữ transcription.
- Preprocessing variants (frame crop, vertical ruling line erase).
- Calls OCR API with local disk caching keyed by image hash.
- Scores OCR columns against QN syllables via Levenshtein (lib.alignment).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from lib.alignment import levenshtein_align
from lib.dictionary import load_qn_to_nom
from lib.image_processing import load_and_binarize, detect_text_box
from lib.ocr_api import upload_image, recognize, boxes_to_columns


REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANT_CACHE_DIR = REPO_ROOT / "tests" / "bench_cache"
VARIANT_IMG_DIR = REPO_ROOT / "tests" / "bench_images"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_page(book: str, page_stem: str) -> tuple[np.ndarray, list[list[str]]]:
    """Load grayscale page + QN syllable columns from prepared/{book}/."""
    page_path = REPO_ROOT / "prepared" / book / "pages" / f"{page_stem}.png"
    trans_path = REPO_ROOT / "prepared" / book / "transcriptions" / f"{page_stem}.json"
    gray = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(page_path)
    with open(trans_path, "r", encoding="utf-8") as f:
        trans = json.load(f)
    qn_columns = [c["syllables"] for c in trans["columns"]]
    return gray, qn_columns


def list_available_pages(book: str, limit: int | None = None) -> list[str]:
    """Pages that have both a PNG and a transcription JSON."""
    pages_dir = REPO_ROOT / "prepared" / book / "pages"
    trans_dir = REPO_ROOT / "prepared" / book / "transcriptions"
    stems = []
    for p in sorted(pages_dir.glob("page_*.png")):
        if (trans_dir / f"{p.stem}.json").exists():
            stems.append(p.stem)
    return stems[:limit] if limit else stems


# ---------------------------------------------------------------------------
# Preprocessing variants
# ---------------------------------------------------------------------------

def variant_raw(gray: np.ndarray) -> np.ndarray:
    """A: unchanged — baseline."""
    return gray


def variant_crop_frame(gray: np.ndarray, margin: int = 8) -> np.ndarray:
    """B: crop to detected text box (removes rectangular outer frame).

    Falls back to input if detection collapses.
    """
    try:
        _, binary = load_and_binarize_from_array(gray)
        l, t, r, b = detect_text_box(binary)
    except Exception:
        return gray
    h, w = gray.shape
    l = max(0, l - margin)
    t = max(0, t - margin)
    r = min(w, r + margin)
    b = min(h, b + margin)
    if (r - l) < w * 0.3 or (b - t) < h * 0.3:
        return gray
    return gray[t:b, l:r].copy()


def variant_crop_and_erase_columns(gray: np.ndarray) -> np.ndarray:
    """C: crop frame, then inpaint long vertical ruling lines inside."""
    cropped = variant_crop_frame(gray)
    return erase_vertical_ruling_lines(cropped)


def erase_vertical_ruling_lines(gray: np.ndarray) -> np.ndarray:
    """Detect long thin vertical black lines and inpaint them out.

    Strategy:
      1. Binarize (Otsu on inverted → ink=1).
      2. Morph open with tall thin kernel (1, h*0.4) → only ruling lines survive.
      3. Reject components whose bbox width > expected stroke (keeps real strokes).
      4. Dilate mask by 2px, inpaint with TELEA.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 50 or w < 50:
        return gray

    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    bg_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bg = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, bg_k)
    norm = cv2.divide(blurred, bg, scale=255)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    v_len = max(int(h * 0.4), 80)
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_k)

    # Keep only thin components (ruling lines are <= ~6 px wide at 300 dpi)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(v_lines, 8)
    mask = np.zeros_like(v_lines)
    for i in range(1, num):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if cw <= 8 and ch >= v_len * 0.6:
            mask[labels == i] = 255
    if mask.sum() == 0:
        return gray

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


def load_and_binarize_from_array(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-memory variant of lib.image_processing.load_and_binarize."""
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    bg_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bg = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, bg_k)
    norm = cv2.divide(blurred, bg, scale=255)
    _, binv = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = (binv > 0).astype(np.uint8)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
    return gray, binary


# ---------------------------------------------------------------------------
# OCR with disk cache (keyed by SHA1 of bytes written to PNG)
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    columns: list  # list[list[{char, y_center, bbox}]]
    n_columns: int
    n_chars: int
    raw_boxes: list


def _image_hash(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return hashlib.sha1(buf.tobytes()).hexdigest()[:16]


def ocr_variant(
    img: np.ndarray,
    label: str,
    verbose: bool = False,
) -> OCRResult | None:
    """Save image, upload, OCR, cache by (label, image-hash)."""
    VARIANT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VARIANT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    h = _image_hash(img)
    cache_f = VARIANT_CACHE_DIR / f"{label}_{h}.json"
    img_f = VARIANT_IMG_DIR / f"{label}_{h}.png"

    if cache_f.exists():
        with open(cache_f) as f:
            data = json.load(f)
        if verbose:
            print(f"    [cache] {cache_f.name}")
        return OCRResult(
            columns=data["columns"],
            n_columns=len(data["columns"]),
            n_chars=sum(len(c) for c in data["columns"]),
            raw_boxes=data.get("boxes_raw", []),
        )

    cv2.imwrite(str(img_f), img)
    if verbose:
        print(f"    [upload] {img_f.name}")
    file_name = upload_image(str(img_f))
    if not file_name:
        return None
    boxes = recognize(file_name)
    if boxes is None:
        return None
    columns = boxes_to_columns(boxes)

    with open(cache_f, "w", encoding="utf-8") as f:
        json.dump({"columns": columns, "boxes_raw": boxes},
                  f, ensure_ascii=False, indent=2)
    return OCRResult(
        columns=columns,
        n_columns=len(columns),
        n_chars=sum(len(c) for c in columns),
        raw_boxes=boxes,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_QN_TO_NOM = None


def get_qn_to_nom() -> dict:
    global _QN_TO_NOM
    if _QN_TO_NOM is None:
        path = REPO_ROOT / "dict" / "QuocNgu_SinoNom_TongHop3.csv"
        if not path.exists():
            path = REPO_ROOT / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"
        _QN_TO_NOM = load_qn_to_nom(str(path))
    return _QN_TO_NOM


def _as_char_dicts(ocr_col: list[dict]) -> list[dict]:
    """Adapt OCR column (char/y_center/bbox) to alignment char dicts (height)."""
    chars = []
    for c in ocr_col:
        x1, y1, x2, y2 = c["bbox"]
        chars.append({
            "height": max(1, int(y2 - y1)),
            "width": max(1, int(x2 - x1)),
            "ocr_char": c["char"],
        })
    return chars


def score_ocr_vs_transcription(
    ocr_columns: list[list[dict]],
    qn_columns: list[list[str]],
) -> dict:
    """Per-column Levenshtein alignment + dict-candidate match rate.

    A char is 'dict-matched' if its OCR char appears in qn_to_nom[syllable]
    for the syllable it was aligned to. This is a proxy for OCR correctness
    without needing a Nôm ground-truth.
    """
    qn2nom = get_qn_to_nom()
    n_cols_paired = min(len(ocr_columns), len(qn_columns))

    total_ocr_chars = sum(len(c) for c in ocr_columns)
    total_qn_syls = sum(len(c) for c in qn_columns)

    total_match_pairs = 0
    total_dict_hits = 0

    for i in range(n_cols_paired):
        ocr_chars = _as_char_dicts(ocr_columns[i])
        syls = qn_columns[i]
        aligned = levenshtein_align(ocr_chars, syls, qn_to_nom=qn2nom)
        for pair in aligned:
            if pair["type"] != "match":
                continue
            total_match_pairs += 1
            nom_ch = pair["char"]["ocr_char"]
            syl = pair["syllable"].lower()
            candidates = qn2nom.get(syl, [])
            if nom_ch in candidates:
                total_dict_hits += 1

    return {
        "ocr_cols": len(ocr_columns),
        "qn_cols": len(qn_columns),
        "col_count_match": len(ocr_columns) == len(qn_columns),
        "ocr_chars": total_ocr_chars,
        "qn_syls": total_qn_syls,
        "char_count_ratio": total_ocr_chars / max(total_qn_syls, 1),
        "aligned_pairs": total_match_pairs,
        "dict_hits": total_dict_hits,
        "dict_hit_rate": total_dict_hits / max(total_match_pairs, 1),
        # Primary single-number metric: fraction of QN syllables that got
        # paired with an OCR char whose transcription is in the dict for
        # that syllable. Higher = better OCR.
        "coverage": total_dict_hits / max(total_qn_syls, 1),
    }


def fmt_score(s: dict) -> str:
    return (
        f"cols {s['ocr_cols']:>2}/{s['qn_cols']:<2} "
        f"chars {s['ocr_chars']:>4}/{s['qn_syls']:<4} "
        f"ratio {s['char_count_ratio']:.2f}  "
        f"aligned {s['aligned_pairs']:>3}  "
        f"dict_hits {s['dict_hits']:>3}  "
        f"coverage {s['coverage']:.3f}  "
        f"hit_rate {s['dict_hit_rate']:.3f}"
    )
