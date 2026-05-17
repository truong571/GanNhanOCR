"""3-tier label ranking: dictionary -> similar chars -> FontDiffusion+DINOv2."""

import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from core.text.dictionary import (
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


def get_dinov2_ranker(font_path: str | None = None,
                      model_name: str = "dinov2_vitb14_reg",
                      embedding_cache_dir: str | None = None):
    """Lazy-load DINOv2 ranker singleton.

    embedding_cache_dir: if given, the ranker persists crop+fd_cache embeddings
    there so subsequent step-3 re-runs skip ~50k+22k model inferences.
    """
    global _dinov2_ranker
    if _dinov2_ranker is None:
        try:
            from core.ranking.dinov2_ranker import DINOv2Ranker
            _dinov2_ranker = DINOv2Ranker(
                font_path=font_path, model_name=model_name,
                embedding_cache_dir=embedding_cache_dir,
            )
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
    """Generate ancient-style character image using the in-process batch wrapper.

    Returns path to generated image, or None on failure.
    """
    if not style_image:
        return None

    output_path = Path(output_dir) / f"{ord(char):04X}_fd.png"
    if output_path.exists():
        return str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from core.ranking.fontdiffusion_gen import FontDiffusionGenerator

        generator = FontDiffusionGenerator(
            ckpt_dir=ckpt_dir,
            font_path=font_path,
            cache_dir=output_dir,
            batch_size=1,
        )
        generated = generator.generate([char], style_image, style_name="single")
        generated_path = generated.get(char)
        if generated_path:
            shutil.copy2(generated_path, output_path)
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

    # Unique candidate from QN dict — accept (visual sanity in assign_label)
    if len(s2) == 1:
        return s2[0], True, s2

    # OCR char is among the QN dict candidates — accept it
    if ocr_char and ocr_char in s2:
        return ocr_char, True, s2

    # Multiple candidates, no OCR match -> defer to tier 2
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
) -> tuple[str | None, bool, list[str]]:
    """Tier 2: Expand via similar characters dictionary.

    Returns: (chosen_char, matched: bool, all_matching_similars)
      The third element lists every char that was in BOTH similar_dict[ocr_char]
      AND candidates_s2, so callers can do visual tie-breaking when len > 1.
    """
    if not ocr_char or not similar_dict:
        return None, False, []

    similar_chars = similar_dict.get(ocr_char, [])
    matches = [c for c in similar_chars if c in candidates_s2]
    if not matches:
        return None, False, []
    return matches[0], True, matches


def tier3_visual_comparison(
    crop_path: str,
    candidates: list[str],
    font_path: str,
    dinov2_ranker=None,
    fontdiffusion_ckpt: str | None = None,
    fd_cache: dict[str, str] | None = None,
    dinov2_threshold: float = 0.75,
    classical_threshold: float = 0.55,
    require_fontdiffusion: bool = True,
) -> tuple[str | None, bool, float]:
    """Tier 3: Visual comparison via FontDiffusion + DINOv2.

    Ranks ONLY within `candidates` (the QN->Nom dict candidates for the
    syllable). This guarantees the chosen char is a linguistically valid
    Nom mapping for the QN syllable.

    Comparison MUST be against FontDiffusion-generated images (handwritten
    style matched to the manuscript). When `require_fontdiffusion=True`
    (default), candidates without an fd_cache image are dropped from the
    ranking — never substituted with a font-rendered image, which would
    introduce a domain gap between handwritten crop and printed-style
    reference and make the score meaningless.

    Fallbacks (only when require_fontdiffusion=False):
      - font-rendered DINOv2 (printed style, lower accuracy)
      - classical CV (Hu moments + IoU + projection)

    Returns: (chosen_char, matched: bool, score)
      Returns (None, False, 0.0) when no candidate has an fd_cache image
      and require_fontdiffusion=True — caller must handle this honestly
      (e.g. assign tier=0 / matched=False).
    """
    if not candidates:
        return None, False, 0.0

    # Filter PUA characters
    filtered = [c for c in candidates if cjk_block_score(c) > 0.1]
    if not filtered:
        filtered = list(candidates)

    # PRIMARY (mandatory when require_fontdiffusion): FD-cache + DINOv2
    if fd_cache and dinov2_ranker is not None:
        candidate_images = {}
        for char in filtered[:20]:
            if char in fd_cache:
                candidate_images[char] = fd_cache[char]

        if candidate_images:
            try:
                results = dinov2_ranker.rank_candidates_from_paths(
                    crop_path, candidate_images,
                )
                if results:
                    best_char, best_score = results[0]
                    return best_char, best_score > dinov2_threshold, best_score
            except Exception:
                pass

    # If FontDiffusion is required, do NOT fall back to font-rendered or
    # classical CV — the domain gap would produce misleading scores.
    if require_fontdiffusion:
        return None, False, 0.0

    # Optional fallback: DINOv2 with font-rendered images (printed style).
    if dinov2_ranker is not None:
        try:
            results = dinov2_ranker.rank_candidates(crop_path, filtered)
            if results:
                best_char, best_score = results[0]
                return best_char, best_score > dinov2_threshold, best_score
        except Exception:
            pass

    # Last-resort fallback: classical CV (Hu+IoU+projection)
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

    return best_char, best_score > classical_threshold, best_score


def _visual_sanity_score(
    crop_path: str | None,
    char: str,
    dinov2_ranker,
    fd_cache: dict[str, str] | None = None,
) -> float | None:
    """DINOv2 cosine similarity between crop and a single proposed char.

    Used to sanity-check tier-1 decisions when dict mapping disagrees with OCR.
    Returns score in [0, 1], or None if it cannot be computed.
    """
    if not crop_path or dinov2_ranker is None:
        return None
    try:
        if fd_cache and char in fd_cache:
            results = dinov2_ranker.rank_candidates_from_paths(
                crop_path, {char: fd_cache[char]},
            )
        else:
            results = dinov2_ranker.rank_candidates(crop_path, [char])
        if results:
            return results[0][1]
    except Exception:
        pass
    return None


# Tier-1 sanity threshold: demote matched=True when DINOv2 score is below this.
# Generous (0.4) — only catches egregious mismatches, not strict matches.
TIER1_SANITY_THRESHOLD = 0.4


def assign_label(
    ocr_char: str | None,
    qn_syllable: str,
    crop_path: str | None,
    qn_to_nom: dict[str, list[str]],
    nom_to_qn: dict[str, list[str]],
    similar_dict: dict[str, list[str]],
    font_path: str | None = None,
    dinov2_ranker=None,
    fontdiffusion_ckpt: str | None = None,
    fd_cache: dict[str, str] | None = None,
    dinov2_threshold: float = 0.75,
    classical_threshold: float = 0.55,
    require_fontdiffusion: bool = True,
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
        # Sanity: when dict has exactly one candidate but OCR read something
        # different, verify the dict candidate visually matches the crop.
        # Covers both direct unique mapping AND fuzzy unique fallback.
        sanity_demoted = False
        sanity_score = None
        if (len(candidates) == 1
                and ocr_char and ocr_char != char):
            sanity_score = _visual_sanity_score(
                crop_path, char, dinov2_ranker, fd_cache,
            )
            if sanity_score is not None and sanity_score < TIER1_SANITY_THRESHOLD:
                matched = False
                sanity_demoted = True
        out = {
            "nom_char": char,
            "matched": matched,
            "nom_candidates": candidates[:10],
            "tier": 1,
        }
        if sanity_demoted:
            out["sanity_score"] = round(sanity_score, 3)
        return out

    # Tier 2: Similar expansion (only useful when ocr_char is available)
    if ocr_char:
        sim_char, sim_matched, sim_matches = tier2_similar_expansion(
            ocr_char, candidates, similar_dict,
        )
        if sim_char:
            # Visual tie-break when multiple similars match the dict
            if (len(sim_matches) > 1
                    and crop_path and dinov2_ranker is not None):
                try:
                    if fd_cache:
                        candidate_images = {
                            c: fd_cache[c] for c in sim_matches if c in fd_cache
                        }
                        if candidate_images:
                            ranked = dinov2_ranker.rank_candidates_from_paths(
                                crop_path, candidate_images,
                            )
                        else:
                            ranked = dinov2_ranker.rank_candidates(
                                crop_path, sim_matches,
                            )
                    else:
                        ranked = dinov2_ranker.rank_candidates(
                            crop_path, sim_matches,
                        )
                    if ranked:
                        sim_char = ranked[0][0]
                except Exception:
                    pass
            return {
                "nom_char": sim_char,
                "matched": sim_matched,
                "nom_candidates": candidates[:10],
                "tier": 2,
            }

    # Tier 3: Visual comparison — strict within QN->Nom dict candidates only
    if crop_path and font_path and candidates:
        vis_char, vis_matched, vis_score = tier3_visual_comparison(
            crop_path, candidates, font_path,
            dinov2_ranker=dinov2_ranker,
            fontdiffusion_ckpt=fontdiffusion_ckpt,
            fd_cache=fd_cache,
            dinov2_threshold=dinov2_threshold,
            classical_threshold=classical_threshold,
            require_fontdiffusion=require_fontdiffusion,
        )
        if vis_char:
            return {
                "nom_char": vis_char,
                "matched": vis_matched,
                "nom_candidates": candidates[:10],
                "tier": 3,
                "visual_score": round(vis_score, 3),
            }

    # Final fallback: dict has candidates but no tier could verify which one.
    # Pick the most likely candidate by CJK block score, but DO NOT claim
    # matched=True — visual verification was not possible. Reviewer must confirm.
    if candidates:
        scored = sorted(candidates[:10], key=lambda c: cjk_block_score(c), reverse=True)
        return {
            "nom_char": scored[0],
            "matched": False,
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
