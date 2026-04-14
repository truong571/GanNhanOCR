"""3-tier label ranking: dictionary -> similar chars -> FontDiffusion+DINOv2."""

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from lib.dictionary import (
    build_nom_to_qn, cjk_block_score, fuzzy_dict_lookup,
)


# ---------------------------------------------------------------------------
# Font rendering
# ---------------------------------------------------------------------------

_pygame_font = None
_pygame_font_path = None


def _load_pygame_font(ttf_path: str, size: int = 64):
    try:
        import pygame
        import pygame.freetype
        if not pygame.get_init():
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
        return pygame.freetype.Font(ttf_path, size=size)
    except Exception:
        return None


def render_char(char: str, font_path: str, size: int = 64) -> np.ndarray | None:
    """Render one Nom character as a binary image using NomNaTong font."""
    global _pygame_font, _pygame_font_path

    if _pygame_font_path != font_path:
        _pygame_font = _load_pygame_font(font_path, size=size)
        _pygame_font_path = font_path

    if _pygame_font is not None:
        try:
            import pygame
            surface, _ = _pygame_font.render(char)
            imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)
            imo = 255 - np.array(imo)
            bg = np.full((size, size), 255, dtype=np.uint8)
            h, w = imo.shape[:2]
            if h <= 0 or w <= 0:
                return None
            if h > size:
                w = round(w * size / h)
                h = size
                imo = cv2.resize(imo, (w, h))
            if w > size:
                h = round(h * size / w)
                w = size
                imo = cv2.resize(imo, (w, h))
            x = round((size - w) / 2)
            y = round((size - h) / 2)
            bg[y:h + y, x:x + w] = imo
            _, binarized = cv2.threshold(bg, 128, 255, cv2.THRESH_BINARY)
            return binarized
        except Exception:
            pass

    # Fallback: PIL
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(font_path, size - 10)
        img = Image.new("L", (size, size), 255)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            return None
        x = (size - w) // 2 - bbox[0]
        y = (size - h) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
        arr = np.array(img)
        _, binarized = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY)
        return binarized
    except Exception:
        return None


_render_cache: dict[str, np.ndarray | None] = {}


def get_rendered(char: str, font_path: str, size: int = 64) -> np.ndarray | None:
    """Render with cache."""
    key = f"{char}_{font_path}_{size}"
    if key not in _render_cache:
        _render_cache[key] = render_char(char, font_path, size)
    return _render_cache[key]


# ---------------------------------------------------------------------------
# Visual similarity (classical CV)
# ---------------------------------------------------------------------------

def visual_similarity(crop_img: np.ndarray, rendered: np.ndarray) -> float:
    """Shape similarity between handwritten crop and font-rendered image.

    Components: Hu moments + IoU (30%) + structural features (30%)
                + projection correlation (40%)
    """
    if crop_img is None or rendered is None:
        return 0.0

    size = rendered.shape[0]
    crop_resized = cv2.resize(crop_img, (size, size))
    _, crop_bin = cv2.threshold(crop_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    crop_fg = (crop_bin == 0).astype(np.uint8)
    rend_fg = (rendered == 0).astype(np.uint8)

    # 1. Shape matching
    shape_score = 0.0
    crop_contours = []
    rend_contours = []
    try:
        crop_contours, _ = cv2.findContours(crop_fg * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rend_contours, _ = cv2.findContours(rend_fg * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if crop_contours and rend_contours:
            crop_sorted = sorted(crop_contours, key=cv2.contourArea, reverse=True)[:3]
            rend_sorted = sorted(rend_contours, key=cv2.contourArea, reverse=True)[:3]

            crop_mask = np.zeros_like(crop_fg)
            rend_mask = np.zeros_like(rend_fg)
            cv2.drawContours(crop_mask, crop_sorted, -1, 1, thickness=cv2.FILLED)
            cv2.drawContours(rend_mask, rend_sorted, -1, 1, thickness=cv2.FILLED)

            crop_hu = cv2.HuMoments(cv2.moments(crop_mask)).flatten()
            rend_hu = cv2.HuMoments(cv2.moments(rend_mask)).flatten()
            eps = 1e-10
            crop_log = -np.sign(crop_hu) * np.log10(np.abs(crop_hu) + eps)
            rend_log = -np.sign(rend_hu) * np.log10(np.abs(rend_hu) + eps)
            hu_dist = np.linalg.norm(crop_log - rend_log)
            shape_score = max(0.0, 1.0 - min(hu_dist, 10.0) / 10.0)

            intersection = (crop_fg & rend_fg).sum()
            union = (crop_fg | rend_fg).sum()
            iou = intersection / max(union, 1)
            shape_score = 0.5 * shape_score + 0.5 * iou
    except Exception:
        pass

    # 2. Structural features
    struct_score = 0.0
    try:
        crop_density = crop_fg.sum() / max(1, crop_fg.size)
        rend_density = rend_fg.sum() / max(1, rend_fg.size)
        density_sim = 1.0 - min(abs(crop_density - rend_density) / max(0.01, max(crop_density, rend_density)), 1.0)

        crop_m = cv2.moments(crop_fg)
        rend_m = cv2.moments(rend_fg)
        sz = crop_fg.shape[0]
        if crop_m["m00"] > 0 and rend_m["m00"] > 0:
            com_dist = (
                ((crop_m["m10"] / crop_m["m00"] / sz - rend_m["m10"] / rend_m["m00"] / sz) ** 2 +
                 (crop_m["m01"] / crop_m["m00"] / sz - rend_m["m01"] / rend_m["m00"] / sz) ** 2) ** 0.5
            )
            com_sim = max(0.0, 1.0 - com_dist * 3.0)
        else:
            com_sim = 0.0

        min_area = max(1, crop_fg.size * 0.005)
        n_crop_cc = sum(1 for c in crop_contours if cv2.contourArea(c) > min_area)
        n_rend_cc = sum(1 for c in rend_contours if cv2.contourArea(c) > min_area)
        cc_sim = 1.0 - min(abs(n_crop_cc - n_rend_cc), 5) / 5.0

        struct_score = 0.4 * density_sim + 0.3 * com_sim + 0.3 * cc_sim
    except Exception:
        pass

    # 3. Projection correlation
    proj_score = 0.0
    try:
        def _ncc(a, b):
            a, b = a - a.mean(), b - b.mean()
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-8 or nb < 1e-8:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        h_corr = max(0, _ncc(crop_fg.sum(axis=1).astype(float), rend_fg.sum(axis=1).astype(float)))
        v_corr = max(0, _ncc(crop_fg.sum(axis=0).astype(float), rend_fg.sum(axis=0).astype(float)))
        proj_score = (h_corr + v_corr) / 2.0
    except Exception:
        pass

    return 0.30 * shape_score + 0.30 * struct_score + 0.40 * proj_score


# ---------------------------------------------------------------------------
# DINOv2 ranker (lazy-loaded)
# ---------------------------------------------------------------------------

_dinov2_ranker = None


def get_dinov2_ranker(font_path: str | None = None):
    """Lazy-load DINOv2 ranker singleton."""
    global _dinov2_ranker
    if _dinov2_ranker is None:
        try:
            from lib.dinov2_ranker import DINOv2Ranker
            _dinov2_ranker = DINOv2Ranker(font_path=font_path)
        except Exception as e:
            print(f"[RANKER] DINOv2 not available: {e}", file=sys.stderr)
            _dinov2_ranker = False
    return _dinov2_ranker if _dinov2_ranker is not False else None


# ---------------------------------------------------------------------------
# FontDiffusion image generation
# ---------------------------------------------------------------------------

def generate_fontdiffusion_image(
    char: str,
    font_path: str,
    ckpt_dir: str,
    output_dir: str,
    style_image: str | None = None,
) -> str | None:
    """Generate ancient-style character image using FontDiffusion.

    Returns path to generated image, or None on failure.
    """
    output_path = Path(output_dir) / f"{ord(char):04X}_fd.png"
    if output_path.exists():
        return str(output_path)

    # Render source character from font
    rendered = render_char(char, font_path, size=256)
    if rendered is None:
        return None

    content_path = Path(output_dir) / f"{ord(char):04X}_content.png"
    cv2.imwrite(str(content_path), rendered)

    try:
        cmd = [
            sys.executable, "FontDiffusion/run_inference.py",
            "--mode", "sample_optimized",
            "--ckpt_dir", ckpt_dir,
            "--characters", str(content_path),
            "--output_dir", output_dir,
        ]
        if style_image:
            cmd.extend(["--style_images", style_image])

        subprocess.run(cmd, capture_output=True, timeout=60, check=True)

        if output_path.exists():
            return str(output_path)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# 3-Tier label assignment
# ---------------------------------------------------------------------------

def tier1_dictionary_lookup(
    ocr_char: str | None,
    qn_syllable: str,
    qn_to_nom: dict[str, list[str]],
    nom_to_qn: dict[str, list[str]] | None = None,
) -> tuple[str | None, bool, list[str]]:
    """Tier 1: Bidirectional dictionary lookup.

    Returns: (chosen_char, matched: bool, candidates)
      matched=True  -> label is black (correct/confirmed)
      matched=False -> label is red (unconfirmed/wrong)
    """
    if nom_to_qn is None:
        nom_to_qn = build_nom_to_qn(qn_to_nom)

    qn_lower = qn_syllable.lower()
    s2 = qn_to_nom.get(qn_lower, [])  # QN -> Nom candidates
    s1 = nom_to_qn.get(ocr_char, []) if ocr_char else []  # Nom -> QN readings

    # Unique candidate
    if len(s2) == 1:
        return s2[0], True, s2

    # Bidirectional confirmation
    if ocr_char and ocr_char in s2 and qn_lower in s1:
        return ocr_char, True, s2

    # OCR char is a valid candidate but not confirmed bidirectionally
    if ocr_char and ocr_char in s2:
        return ocr_char, True, s2

    # Multiple candidates, no OCR match -> need tier 2
    if s2:
        return None, False, s2

    # No candidates -> try fuzzy
    fuzzy = fuzzy_dict_lookup(qn_lower, qn_to_nom)
    if fuzzy:
        if len(fuzzy) == 1:
            return fuzzy[0], True, fuzzy
        return None, False, fuzzy

    return None, False, []


def tier2_similar_expansion(
    ocr_char: str | None,
    candidates_s2: list[str],
    similar_dict: dict[str, list[str]],
) -> tuple[str | None, bool]:
    """Tier 2: Expand via similar characters dictionary.

    Returns: (chosen_char, matched: bool)
    """
    if not ocr_char or not similar_dict:
        return None, False

    similar_chars = similar_dict.get(ocr_char, [])
    for sim_char in similar_chars:
        if sim_char in candidates_s2:
            return sim_char, True

    return None, False


def tier3_visual_comparison(
    crop_path: str,
    candidates: list[str],
    similar_chars: list[str],
    font_path: str,
    dinov2_ranker=None,
    fontdiffusion_ckpt: str | None = None,
    threshold: float = 0.75,
) -> tuple[str | None, bool, float]:
    """Tier 3: Visual comparison via FontDiffusion + DINOv2.

    Compare crop image against rendered candidates using DINOv2 embeddings.
    Union of S2 candidates + similar chars as comparison set.

    Returns: (chosen_char, matched: bool, score)
      matched=True if score > threshold (visually confirmed)
    """
    all_candidates = list(dict.fromkeys(candidates + similar_chars))
    if not all_candidates:
        return None, False, 0.0

    # Filter PUA characters
    filtered = [c for c in all_candidates if cjk_block_score(c) > 0.1]
    if not filtered:
        filtered = all_candidates

    # Try DINOv2 ranking
    if dinov2_ranker is not None:
        try:
            results = dinov2_ranker.rank_candidates(crop_path, filtered)
            if results:
                best_char, best_score = results[0]
                return best_char, best_score > threshold, best_score
        except Exception:
            pass

    # Fallback: classical visual similarity
    crop_img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
    if crop_img is None:
        return filtered[0] if filtered else None, False, 0.0

    best_char = None
    best_score = 0.0
    for char in filtered[:20]:
        rendered = get_rendered(char, font_path)
        if rendered is None:
            continue
        score = visual_similarity(crop_img, rendered)
        if score > best_score:
            best_score = score
            best_char = char

    return best_char, best_score > threshold, best_score


def assign_label(
    ocr_char: str | None,
    qn_syllable: str,
    crop_path: str | None,
    qn_to_nom: dict[str, list[str]],
    nom_to_qn: dict[str, list[str]],
    similar_dict: dict[str, list[str]],
    font_path: str | None = None,
    dinov2_ranker=None,
) -> dict:
    """Full 3-tier label assignment for one character.

    Returns: {nom_char, matched: bool, candidates, tier}
      matched=True  -> correct (display in BLACK)
      matched=False -> incorrect/unconfirmed (display in RED)
    """
    # Tier 1: Dictionary
    char, matched, candidates = tier1_dictionary_lookup(
        ocr_char, qn_syllable, qn_to_nom, nom_to_qn
    )
    if matched and char:
        return {
            "nom_char": char,
            "matched": True,
            "nom_candidates": candidates[:10],
            "tier": 1,
        }

    # Tier 2: Similar expansion (only useful when ocr_char is available)
    if ocr_char:
        sim_char, sim_matched = tier2_similar_expansion(ocr_char, candidates, similar_dict)
        if sim_char:
            return {
                "nom_char": sim_char,
                "matched": sim_matched,
                "nom_candidates": candidates[:10],
                "tier": 2,
            }

    # Tier 3: Visual comparison (rank candidates by image similarity)
    if crop_path and font_path and candidates:
        similar_chars = similar_dict.get(ocr_char, []) if ocr_char else []
        vis_char, vis_matched, vis_score = tier3_visual_comparison(
            crop_path, candidates, similar_chars, font_path,
            dinov2_ranker=dinov2_ranker,
        )
        if vis_char:
            return {
                "nom_char": vis_char,
                "matched": vis_matched,
                "nom_candidates": candidates[:10],
                "tier": 3,
                "visual_score": round(vis_score, 3),
            }

    # Tier 1 fallback: have candidates but couldn't rank visually
    # Pick first candidate based on specificity (CJK block score)
    if candidates:
        # Sort by CJK block score (prefer standard CJK over PUA/extensions)
        scored = sorted(candidates[:10], key=lambda c: cjk_block_score(c), reverse=True)
        return {
            "nom_char": scored[0],
            "matched": True,  # dictionary confirmed this char maps to the QN syllable
            "nom_candidates": candidates[:10],
            "tier": 1,
        }

    # No candidates at all
    return {
        "nom_char": None,
        "matched": False,
        "nom_candidates": [],
        "tier": 0,
    }
