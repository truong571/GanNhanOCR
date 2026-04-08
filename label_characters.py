#!/usr/bin/env python3
"""
label_characters.py - Gán nhãn tự động cho ký tự Nôm viết tay

Pipeline:
  Bước 4: Levenshtein alignment ký tự detected ↔ âm tiết Quốc ngữ
  Bước 5: Tra từ điển gán Unicode Nôm + confidence scoring
  Bước 5.5: Render ảnh Nôm đánh máy từ font NomNaTong
  Bước 6: Xuất dataset JSON + summary + review images

Input:
  data/prepared/SachThanhTruyen2/
    ├── detected/crops/          (ảnh crop từ detect_characters.py)
    ├── detected/crops_cleaned/  (ảnh cleaned từ clean_crops.py)
    ├── detected/page_XXXX_detection.json
    ├── transcriptions/*.txt     (9 dòng QN, mỗi dòng = 1 cột)
    └── manifest.json

Output:
  data/prepared/SachThanhTruyen2/labeled/
    ├── dataset.json             (toàn bộ nhãn ~82,000 ký tự)
    ├── typed_nom/               (ảnh Nôm đánh máy render từ font)
    ├── review/                  (ảnh debug: viết tay | đánh máy | nhãn)
    └── summary.json             (thống kê accuracy)

Usage:
  python label_characters.py data/prepared/SachThanhTruyen2

  # Chỉ 1 trang:
  python label_characters.py data/prepared/SachThanhTruyen2 --page 12

  # Với review images:
  python label_characters.py data/prepared/SachThanhTruyen2 --review

  # Tuỳ chỉnh font:
  python label_characters.py data/prepared/SachThanhTruyen2 --font FontDiffusion/fonts/NomNaTong-Regular.ttf

  # Dùng OCR API để cải thiện độ chính xác:
  python label_characters.py data/prepared/SachThanhTruyen2 --ocr --excel
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Từ điển tên riêng tôn giáo (từ OCR/align/saint_name.py)
# Mapping: tên dính → tên tách âm tiết
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
# Load từ điển
# ---------------------------------------------------------------------------

def load_translation_dict(dict_path: str, encoding: str = "utf-8-sig") -> dict:
    """Load từ điển QuốcNgữ → SinoNom Unicode.

    Returns: {qn_word_lower: [unicode1, unicode2, ...]}
    """
    trans_dict = {}
    with open(dict_path, "r", encoding=encoding) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                word = row[0].strip().lower()
                char = row[1].strip()
                if word and char:
                    trans_dict.setdefault(word, []).append(char)
    return trans_dict


def load_similarity_dict(dict_path: str, encoding: str = "utf-8-sig") -> dict:
    """Load từ điển ký tự Nôm tương tự.

    Returns: {nom_char: [similar1, similar2, ...]}
    """
    import ast
    similar_dict = {}
    with open(dict_path, "r", encoding=encoding) as f:
        reader = csv.reader(f)
        header = next(reader, None)
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


def is_compatible(nom_char: str, qn_word: str, trans_dict: dict,
                  similar_dict: dict | None = None) -> bool:
    """Kiểm tra ký tự Nôm có tương thích với âm QN không (dùng từ điển + similar)."""
    candidates = trans_dict.get(qn_word.lower(), [])
    if nom_char in candidates:
        return True
    if similar_dict:
        similar_chars = similar_dict.get(nom_char, [])
        if set(candidates) & set(similar_chars):
            return True
    return False


# ---------------------------------------------------------------------------
# Corpus Frequency & Visual Ranking (cải tiến chọn ứng viên)
# ---------------------------------------------------------------------------

def build_char_specificity(trans_dict: dict) -> dict[str, int]:
    """Tính độ cụ thể của mỗi ký tự Nôm: bao nhiêu âm QN map đến nó.

    Ký tự map ít QN words → cụ thể hơn → ưu tiên hơn khi xếp hạng.

    Returns: {nom_char: number_of_qn_words_it_maps_to}
    """
    char_word_count: dict[str, int] = {}
    for word, chars in trans_dict.items():
        for c in chars:
            char_word_count[c] = char_word_count.get(c, 0) + 1
    return char_word_count


def _load_pygame_font(ttf_path: str, size: int = 64):
    """Load font bằng pygame.freetype (chất lượng tốt hơn PIL)."""
    try:
        import pygame
        import pygame.freetype
        if not pygame.get_init():
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
        return pygame.freetype.Font(ttf_path, size=size)
    except Exception:
        return None


_pygame_font = None
_pygame_font_path = None


def _render_candidate(char: str, font_or_path, size: int = 64) -> np.ndarray | None:
    """Render 1 ký tự Nôm thành ảnh nhị phân.

    Ưu tiên dùng pygame.freetype (từ FontDiffusion/ttf2im) cho chất lượng
    render tốt hơn: alpha channel, aspect-ratio preserving, centering.
    Fallback sang PIL nếu pygame không khả dụng.
    """
    global _pygame_font, _pygame_font_path

    # --- Thử pygame.freetype (chất lượng cao) ---
    font_path = None
    if isinstance(font_or_path, str):
        font_path = font_or_path
    elif hasattr(font_or_path, 'path'):
        font_path = font_or_path.path

    if font_path:
        if _pygame_font_path != font_path:
            _pygame_font = _load_pygame_font(font_path, size=size)
            _pygame_font_path = font_path

        if _pygame_font is not None:
            try:
                import pygame
                surface, _ = _pygame_font.render(char)
                imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)
                imo = 255 - np.array(imo)  # Invert: nét đen trên nền trắng

                bg = np.full((size, size), 255, dtype=np.uint8)
                h, w = imo.shape[:2]
                if h <= 0 or w <= 0:
                    return None
                # Aspect-ratio preserving resize
                if h > size:
                    w = round(w * size / h)
                    h = size
                    imo = cv2.resize(imo, (w, h))
                if w > size:
                    h = round(h * size / w)
                    w = size
                    imo = cv2.resize(imo, (w, h))
                # Center
                x = round((size - w) / 2)
                y = round((size - h) / 2)
                bg[y:h + y, x:x + w] = imo
                _, binarized = cv2.threshold(bg, 128, 255, cv2.THRESH_BINARY)
                return binarized
            except Exception:
                pass

    # --- Fallback: PIL ---
    from PIL import Image, ImageDraw
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), char, font=font_or_path)
    except Exception:
        return None
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return None
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font_or_path)
    arr = np.array(img)
    _, binarized = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY)
    return binarized


_render_cache: dict[str, np.ndarray | None] = {}


def _get_rendered(char: str, font, size: int = 64) -> np.ndarray | None:
    """Render với cache."""
    if char not in _render_cache:
        _render_cache[char] = _render_candidate(char, font, size)
    return _render_cache[char]


def visual_similarity(crop_img: np.ndarray, rendered: np.ndarray) -> float:
    """Tính độ tương đồng hình dạng giữa ảnh crop và ảnh render.

    Cải tiến (Issue #3):
      1. Multi-contour matching: so sánh TẤT CẢ contours chính (top-3 by area),
         không chỉ contour lớn nhất → capture bộ thủ phức tạp
      2. Pixel IoU trực tiếp trên ảnh binary → robust hơn contour-based
      3. Projection profile correlation
      4. Structural features (density, CoM, #components)

    Returns: float 0.0 → 1.0 (1.0 = giống nhất)
    """
    if crop_img is None or rendered is None:
        return 0.0

    size = rendered.shape[0]
    crop_resized = cv2.resize(crop_img, (size, size))

    # Binarize crop with Otsu
    _, crop_bin = cv2.threshold(crop_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    crop_fg = (crop_bin == 0).astype(np.uint8)
    rend_fg = (rendered == 0).astype(np.uint8)

    # --- 1. Multi-contour shape matching (Issue #3 cải tiến) ---
    shape_score = 0.0
    crop_contours = []
    rend_contours = []
    try:
        crop_contours, _ = cv2.findContours(
            crop_fg * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rend_contours, _ = cv2.findContours(
            rend_fg * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if crop_contours and rend_contours:
            # So sánh top-3 contours (by area) thay vì chỉ largest
            crop_sorted = sorted(crop_contours, key=cv2.contourArea, reverse=True)[:3]
            rend_sorted = sorted(rend_contours, key=cv2.contourArea, reverse=True)[:3]

            # Tạo combined mask từ top contours để so sánh tổng thể
            crop_mask = np.zeros_like(crop_fg)
            rend_mask = np.zeros_like(rend_fg)
            cv2.drawContours(crop_mask, crop_sorted, -1, 1, thickness=cv2.FILLED)
            cv2.drawContours(rend_mask, rend_sorted, -1, 1, thickness=cv2.FILLED)

            # matchShapes trên combined contour masks (Hu moments)
            crop_hu = cv2.HuMoments(cv2.moments(crop_mask)).flatten()
            rend_hu = cv2.HuMoments(cv2.moments(rend_mask)).flatten()
            # Log-transform Hu moments cho so sánh tốt hơn
            eps = 1e-10
            crop_hu_log = -np.sign(crop_hu) * np.log10(np.abs(crop_hu) + eps)
            rend_hu_log = -np.sign(rend_hu) * np.log10(np.abs(rend_hu) + eps)
            hu_dist = np.linalg.norm(crop_hu_log - rend_hu_log)
            shape_score = max(0.0, 1.0 - min(hu_dist, 10.0) / 10.0)

            # Bonus: Pixel IoU trực tiếp (robust hơn contour matching)
            intersection = (crop_fg & rend_fg).sum()
            union = (crop_fg | rend_fg).sum()
            iou = intersection / max(union, 1)
            shape_score = 0.5 * shape_score + 0.5 * iou
    except Exception:
        pass

    # --- 2. Structural features comparison ---
    struct_score = 0.0
    try:
        # Ink density
        crop_density = crop_fg.sum() / max(1, crop_fg.size)
        rend_density = rend_fg.sum() / max(1, rend_fg.size)
        density_sim = 1.0 - min(abs(crop_density - rend_density) / max(0.01, max(crop_density, rend_density)), 1.0)

        # Center of mass
        crop_moments = cv2.moments(crop_fg)
        rend_moments = cv2.moments(rend_fg)
        sz = crop_fg.shape[0]

        if crop_moments["m00"] > 0 and rend_moments["m00"] > 0:
            crop_cx = crop_moments["m10"] / crop_moments["m00"] / sz
            crop_cy = crop_moments["m01"] / crop_moments["m00"] / sz
            rend_cx = rend_moments["m10"] / rend_moments["m00"] / sz
            rend_cy = rend_moments["m01"] / rend_moments["m00"] / sz

            com_dist = ((crop_cx - rend_cx)**2 + (crop_cy - rend_cy)**2) ** 0.5
            com_sim = max(0.0, 1.0 - com_dist * 3.0)
        else:
            com_sim = 0.0

        # Number of connected components (significant ones only)
        min_area = max(1, crop_fg.size * 0.005)  # Ignore tiny noise
        n_crop_cc = sum(1 for c in crop_contours if cv2.contourArea(c) > min_area)
        n_rend_cc = sum(1 for c in rend_contours if cv2.contourArea(c) > min_area)
        cc_sim = 1.0 - min(abs(n_crop_cc - n_rend_cc), 5) / 5.0

        struct_score = 0.4 * density_sim + 0.3 * com_sim + 0.3 * cc_sim
    except Exception:
        pass

    # --- 3. Projection profile correlation ---
    proj_score = 0.0
    try:
        h_crop = crop_fg.sum(axis=1).astype(float)
        h_rend = rend_fg.sum(axis=1).astype(float)
        v_crop = crop_fg.sum(axis=0).astype(float)
        v_rend = rend_fg.sum(axis=0).astype(float)

        def _ncc(a, b):
            a, b = a - a.mean(), b - b.mean()
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-8 or nb < 1e-8:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        h_corr = max(0, _ncc(h_crop, h_rend))
        v_corr = max(0, _ncc(v_crop, v_rend))
        proj_score = (h_corr + v_corr) / 2.0
    except Exception:
        pass

    # Combined: Shape 30% + Structural 30% + Projection 40%
    # (Issue #3: giảm weight shape vì handwriting ≠ font,
    #  tăng projection vì phân bố nét ổn định hơn giữa handwriting/font)
    return 0.30 * shape_score + 0.30 * struct_score + 0.40 * proj_score


def _cjk_block_score(char: str) -> float:
    """Điểm ưu tiên theo Unicode block. CJK cơ bản > Extension > PUA."""
    cp = ord(char)
    if 0x4E00 <= cp <= 0x9FFF:    # CJK Unified (phổ biến nhất)
        return 1.0
    if 0xF900 <= cp <= 0xFAFF:    # CJK Compatibility
        return 0.8
    if 0x3400 <= cp <= 0x4DBF:    # CJK Extension A
        return 0.5
    if 0x20000 <= cp <= 0x2A6DF:  # CJK Extension B
        return 0.3
    if 0xE000 <= cp <= 0xF8FF:    # Private Use Area → tránh
        return 0.05
    return 0.2


def rank_candidates(
    candidates: list[str],
    syllable: str,
    crop_path: str | None,
    font_path: str | None,
    corpus_freq: dict,
    embed_ranker=None,
) -> list[tuple[str, float]]:
    """Xếp hạng ứng viên bằng visual similarity + specificity + CJK block.

    Scoring (khi có ảnh crop + font):
      1. Visual similarity (40%): so hình dạng crop viết tay vs render font
      2. Specificity (30%): ký tự map ít QN words → cụ thể hơn → điểm cao
      3. CJK block (30%): ưu tiên CJK cơ bản, tránh PUA/Extension

    Fallback (không có ảnh crop hoặc font):
      1. Specificity (60%) + CJK block (40%)

    Args:
        candidates: danh sách ký tự Nôm ứng viên
        syllable: âm QN
        crop_path: đường dẫn ảnh crop viết tay (hoặc None)
        font_path: đường dẫn font Nôm (hoặc None)
        corpus_freq: {nom_char: word_count} từ build_char_specificity

    Returns: [(char, score)] sorted by score descending
    """
    if not candidates:
        return []

    # Lọc bỏ PUA characters
    filtered = [c for c in candidates if _cjk_block_score(c) > 0.1]
    if not filtered:
        filtered = candidates

    # Specificity: ký tự map ít words hơn → cụ thể hơn → điểm cao
    max_words = max((corpus_freq.get(c, 1) for c in filtered), default=1)
    if max_words == 0:
        max_words = 1

    # --- Deep embedding ranking (ưu tiên nếu có) ---
    embed_scores = {}
    use_embedding = False
    if embed_ranker is not None and crop_path:
        try:
            embed_results = embed_ranker.rank_candidates(crop_path, filtered)
            embed_scores = {char: score for char, score in embed_results}
            use_embedding = True
        except Exception:
            pass

    # --- Fallback: visual similarity bằng IoU/projection ---
    crop_img = None
    use_visual = False
    if not use_embedding and crop_path and font_path:
        crop_img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if crop_img is not None:
            use_visual = True

    scored = []
    for char in filtered:
        n_words = corpus_freq.get(char, max_words)  # Unknown = treat as ambiguous
        specificity = 1.0 - (n_words - 1) / max_words  # 1 word = 1.0, many words = low
        specificity = max(0.0, specificity)

        block_score = _cjk_block_score(char)

        if use_embedding:
            # Deep embedding: visual chiếm ưu thế (0.65) vì embedding đáng tin hơn
            vis_score = embed_scores.get(char, 0.0)
            total = 0.65 * vis_score + 0.20 * specificity + 0.15 * block_score
        elif use_visual:
            # Issue #3: Giảm visual weight (0.65→0.45) vì handwriting ≠ font render.
            # Tăng specificity (0.20→0.35) vì dictionary specificity đáng tin hơn
            # visual matching giữa chữ viết tay và font.
            rendered = _get_rendered(char, font_path)
            vis_score = visual_similarity(crop_img, rendered) if rendered is not None else 0.0
            total = 0.45 * vis_score + 0.35 * specificity + 0.20 * block_score
        else:
            # Không có ảnh: 0.6 specificity + 0.4 CJK block
            total = 0.6 * specificity + 0.4 * block_score

        scored.append((char, total))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# OCR API Integration (tools.clc.hcmus.edu.vn)
# ---------------------------------------------------------------------------

_SN_DOMAIN = os.environ.get("SN_DOMAIN", "tools.clc.hcmus.edu.vn")

# Issue #10: Token từ environment variable thay vì hardcode.
# Đặt: export SN_OCR_TOKEN="eyJhbGci..." trước khi chạy.
# Nếu không có env var → fallback sang token cũ (có thể hết hạn).
_OCR_TOKEN_FALLBACK = (
    "eyJhbGciOiJSUzI1NiIsImtpZCI6IjQ3YWU0OWM0YzlkM2ViODVhNTI1NDA3MmMz"
    "MGQyZThlNzY2MWVmZTEiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3"
    "VyZXRva2VuLmdvb2dsZS5jb20vY2xjLWhhbS1ub24iLCJhdWQiOiJjbGMtaGFtLW5v"
    "biIsImF1dGhfdGltZSI6MTc1MjA0MDU4NCwidXNlcl9pZCI6InRrS3JOcmVFVVBjND"
    "JLMVBoZVloeDU4THR2cjEiLCJzdWIiOiJ0a0tyTnJlRVVQYzQySzFQaGVZaHg1OEx0"
    "dnIxIiwiaWF0IjoxNzUyMDQwNTg0LCJleHAiOjE3NTIwNDQxODQsImVtYWlsIjoiZG"
    "V0YWlAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJmaXJlYmFzZSI6"
    "eyJpZGVudGl0aWVzIjp7ImVtYWlsIjpbImRldGFpQGdtYWlsLmNvbSJdfSwic2lnbl"
    "9pbl9wcm92aWRlciI6InBhc3N3b3JkIn19.L1w9bt5qh8Hm6BMC091bw6GiswtaMYlE"
    "3XgE_euN4c-HNHaq5Pfk6HwU8ggTVuxJCmQg1tRdaQm3NGovjPHucDzB2VWwKCgW05"
    "lUz7622-bY-FzOt0TB11Abhe2ldzBDy5LIgVcafZ7AsIwUrOQbVPScqSyhcgaFEvaQ4"
    "W24kCOfis2qiLwiXuiHvVLvJEgZQvzDcGCoxZe37bu05D1QOV0-qG_JKJhaXdSbVjBt"
    "OCakZCTJ0W9ax_XBzgqywsfHOB-4qqm4YKVuxLLl0UQCa9627rvNfdumE-YZcuNLCyS"
    "WO_KRD8E3TuM38h6cMNuoqgX-eQDvO2qbKJTNZl08bg"
)


def _get_ocr_token() -> str:
    """Lấy OCR token từ env var hoặc fallback. Kiểm tra expiry."""
    token = os.environ.get("SN_OCR_TOKEN", _OCR_TOKEN_FALLBACK)

    # Kiểm tra JWT expiry (decode payload không cần verify signature)
    try:
        import base64
        parts = token.split(".")
        if len(parts) >= 2:
            # Decode payload (phần 2, base64url)
            payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            exp = payload.get("exp", 0)
            import time
            now = time.time()
            if exp > 0 and now > exp:
                remaining = now - exp
                print(f"[OCR WARNING] Token đã hết hạn {remaining/3600:.0f} giờ trước!",
                      file=sys.stderr)
                print(f"[OCR WARNING] Đặt SN_OCR_TOKEN env var với token mới:",
                      file=sys.stderr)
                print(f"  export SN_OCR_TOKEN=\"<new_token>\"", file=sys.stderr)
            elif exp > 0 and (exp - now) < 300:
                print(f"[OCR WARNING] Token sắp hết hạn trong {(exp-now)/60:.0f} phút!",
                      file=sys.stderr)
    except Exception:
        pass  # Không block pipeline nếu decode thất bại

    return token


def _ocr_upload_image(image_path: str) -> str | None:
    """Upload ảnh lên server OCR, trả về file_name trên server."""
    url = f"https://{_SN_DOMAIN}/api/web/clc-sinonom/image-upload"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Authorization": f"Bearer {_get_ocr_token()}",
    }

    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                url, files={"image_file": f}, headers=headers,
                verify=False, timeout=30,
            )
        resp.raise_for_status()
        result = resp.json()
        if result.get("is_success"):
            return result["data"]["file_name"]
        print(f"[OCR] Upload failed: {result.get('message')}", file=sys.stderr)
    except Exception as e:
        print(f"[OCR] Upload error: {e}", file=sys.stderr)
    return None


def _ocr_recognize(file_name: str) -> list[dict] | None:
    """Gọi OCR API, trả về list of boxes [{points, transcription}, ...]."""
    url = f"https://{_SN_DOMAIN}/api/web/clc-sinonom/image-ocr"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Authorization": f"Bearer {_get_ocr_token()}",
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {
        "file_name": file_name,
        "ocr_id": 1,
        "lang_type": 1,
        "reading_direction": 1,
        "font_type": 1,
    }

    try:
        resp = requests.post(
            url, json=body, headers=headers, verify=False, timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("is_success"):
            return result["data"]["details"]["details"]
        print(f"[OCR] OCR failed: {result.get('message')}", file=sys.stderr)
    except Exception as e:
        print(f"[OCR] OCR error: {e}", file=sys.stderr)
    return None


def _ocr_boxes_to_columns(boxes: list[dict]) -> list[list[dict]]:
    """Chuyển OCR boxes thành list of columns, mỗi column = list of {char, y_center}.

    Sắp xếp: cột phải → trái, trong cột trên → dưới.
    Mỗi box chứa nhiều ký tự → chia đều bbox theo chiều dọc để ước lượng y.
    """
    if not boxes:
        return []

    # Sắp xếp theo x giảm dần (phải → trái)
    sorted_boxes = sorted(boxes, key=lambda b: b["points"][0][0], reverse=True)

    cols = []
    for box in sorted_boxes:
        if not cols:
            cols.append([box])
            continue
        last_box = cols[-1][-1]
        if abs(last_box["points"][0][0] - box["points"][0][0]) < 15:
            cols[-1].append(box)
        else:
            cols.append([box])

    # Sắp xếp trong mỗi cột theo y, tách ký tự với toạ độ ước lượng
    result = []
    for col in cols:
        col_sorted = sorted(col, key=lambda b: b["points"][0][1])
        chars_with_pos = []
        for box in col_sorted:
            text = box.get("transcription", "").strip()
            valid_chars = [ch for ch in text if ch.strip()]
            n = len(valid_chars)
            if n == 0:
                continue
            # Ước lượng y cho mỗi ký tự: chia đều box theo chiều dọc
            y_top = box["points"][0][1]
            y_bot = box["points"][2][1]
            x_left = box["points"][0][0]
            x_right = box["points"][1][0]
            char_h = (y_bot - y_top) / n
            for idx, ch in enumerate(valid_chars):
                cy = y_top + char_h * (idx + 0.5)
                chars_with_pos.append({
                    "char": ch,
                    "y_center": cy,
                    "bbox": [x_left, int(y_top + char_h * idx),
                             x_right, int(y_top + char_h * (idx + 1))],
                })
        result.append(chars_with_pos)

    return result


def ocr_page(image_path: str, cache_path: str | None = None,
             verbose: bool = False) -> list[list[dict]] | None:
    """OCR toàn bộ 1 trang ảnh, trả về columns of char dicts.

    Args:
        image_path: path tới ảnh trang (pages/page_XXXX.png)
        cache_path: nếu có, lưu/đọc cache JSON
        verbose: in trạng thái

    Returns:
        list of columns, mỗi column = list of {char, y_center, bbox}
        None nếu OCR thất bại
    """
    # Đọc cache nếu có
    if cache_path:
        cache_file = Path(cache_path)
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if verbose:
                print(f"    [OCR] Loaded cache: {cache_file.name}")
            return cached.get("columns")

    if verbose:
        print(f"    [OCR] Uploading {Path(image_path).name}...")

    file_name = _ocr_upload_image(image_path)
    if not file_name:
        return None

    if verbose:
        print(f"    [OCR] Running OCR...")

    boxes = _ocr_recognize(file_name)
    if boxes is None:
        return None

    columns = _ocr_boxes_to_columns(boxes)

    if verbose:
        total_chars = sum(len(c) for c in columns)
        print(f"    [OCR] Got {len(columns)} columns, {total_chars} chars")

    # Lưu cache (bao gồm raw boxes để không cần gọi API lại)
    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Hash ảnh để detect thay đổi
        import hashlib
        with open(image_path, "rb") as fh:
            img_hash = hashlib.md5(fh.read()).hexdigest()
        cache_data = {
            "image": image_path,
            "image_hash": img_hash,
            "columns": columns,
            "boxes_raw": boxes,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    return columns


def _find_best_ocr_column(
    ocr_columns: list[list[dict]],
    det_chars: list[dict],
    used_indices: set[int],
) -> int | None:
    """Tìm cột OCR phù hợp nhất dựa trên bbox x-overlap với detected chars.

    So sánh x trung tâm cột OCR vs x trung tâm detected chars.
    """
    if not det_chars:
        return None

    # X trung tâm của detected column
    det_x = sum((c["bbox"][0] + c["bbox"][2]) / 2 for c in det_chars) / len(det_chars)

    best_idx = None
    best_dist = float("inf")
    for i, col in enumerate(ocr_columns):
        if i in used_indices:
            continue
        if len(col) < 3:
            continue
        # X trung tâm cột OCR
        ocr_x = sum((c["bbox"][0] + c["bbox"][2]) / 2 for c in col) / len(col)
        dist = abs(ocr_x - det_x)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    # Chỉ match nếu khoảng cách x hợp lý (< 100px)
    if best_dist > 100:
        return None
    return best_idx


def _match_ocr_bbox(
    ocr_col: list[dict],
    aligned: list[dict],
    trans_dict: dict,
) -> int:
    """So khớp ký tự OCR với aligned pairs bằng bbox y-overlap.

    Với mỗi detected char (matched pair), tìm ký tự OCR có y_center
    gần nhất. Upgrade confidence dựa trên mức độ phù hợp:
    - medium + OCR in candidates → "high" (xác nhận)
    - medium + OCR not in candidates → ghi nhận ocr_char để tham khảo
    - low + OCR char hợp lệ → upgrade "medium" (OCR cung cấp nhãn khi dict thiếu)

    Returns: số lượng upgraded
    """
    upgraded = 0
    matched_pairs = [p for p in aligned if p["type"] == "match" and p.get("char")]

    for pair in matched_pairs:
        char_info = pair["char"]
        det_y1 = char_info["bbox"][1]
        det_y2 = char_info["bbox"][3]
        det_cy = (det_y1 + det_y2) / 2

        # Tìm OCR char có y_center gần nhất
        best_ocr = None
        best_dist = float("inf")
        for oc in ocr_col:
            dist = abs(oc["y_center"] - det_cy)
            if dist < best_dist:
                best_dist = dist
                best_ocr = oc

        if best_ocr is None:
            continue

        # Kiểm tra overlap: khoảng cách y_center < chiều cao ký tự
        det_h = det_y2 - det_y1
        if best_dist > max(det_h, 50):
            continue

        ocr_char = best_ocr["char"]
        if not ocr_char or not ocr_char.strip():
            continue

        candidates = pair.get("nom_candidates", [])
        conf = pair.get("confidence", "")

        if conf == "medium" and candidates:
            if ocr_char in candidates:
                pair["nom_char"] = ocr_char
                pair["confidence"] = "high"
                pair["ocr_source"] = True
                upgraded += 1
            else:
                # OCR không khớp candidates nhưng vẫn ghi nhận
                pair["ocr_char"] = ocr_char
        elif conf == "low" and ocr_char:
            # Dict không có candidates → dùng OCR char trực tiếp
            # Kiểm tra OCR char có phải CJK hợp lệ không
            cp = ord(ocr_char)
            is_cjk = (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
                       or 0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF)
            if is_cjk:
                pair["nom_char"] = ocr_char
                pair["nom_unicode"] = f"U+{cp:04X}"
                pair["nom_candidates"] = [ocr_char]
                pair["confidence"] = "medium"
                pair["ocr_source"] = True
                upgraded += 1
            else:
                pair["ocr_char"] = ocr_char
        elif conf == "high":
            # Ghi nhận OCR char cho cross-validation (không thay đổi nhãn)
            pair["ocr_char"] = ocr_char

    return upgraded


# ---------------------------------------------------------------------------
# Chuẩn hoá text Quốc ngữ
# ---------------------------------------------------------------------------

def normalize_syllables(syllables: list[str]) -> list[str]:
    """Chuẩn hoá danh sách âm tiết: xoá ký tự dính, tách tên riêng."""
    result = []
    for syl in syllables:
        # Xoá dấu ngoặc, quote, dấu câu dính vào âm tiết
        cleaned = re.sub(r'["""\'()[\]{}«»,.;:!?…–—\-]', '', syl).strip()
        if not cleaned:
            continue

        lower = cleaned.lower()
        if lower in SAINT_NAMES:
            parts = SAINT_NAMES[lower].split()
            result.extend(parts)
        else:
            result.append(cleaned)
    return result


# ---------------------------------------------------------------------------
# Levenshtein Alignment (Bước 4)
# Adapted from OCR/align/align.py - levenshtein_align_boxes()
# ---------------------------------------------------------------------------

def levenshtein_align(chars: list[dict], syllables: list[str],
                      deletion_cost_fn=None,
                      trans_dict: dict | None = None) -> list[dict]:
    """Căn chỉnh Levenshtein giữa ký tự detected và âm tiết QN.

    Cải tiến: Dùng dictionary-based substitution cost thay vì cost=0.
    Nếu âm tiết QN có candidates trong từ điển → cost thấp (có thể match).
    Nếu không có candidates → cost cao (mismatch, DP sẽ ưu tiên skip).
    Điều này giúp DP tự sửa khi detection bị lệch 1 ký tự.

    Args:
        chars: list of char dicts từ detection.json
               [{char_idx, bbox, width, height, crop_file}, ...]
        syllables: list of QN syllables ["quốc", "âm", ...]
        deletion_cost_fn: hàm tính chi phí xoá (nhận char dict, trả về float)
                          mặc định: ký tự nhỏ → chi phí thấp (dễ xoá)
        trans_dict: từ điển QN→Nôm để tính substitution cost.
                    Nếu None → fallback cost=0 (hành vi cũ).

    Returns:
        list of aligned pairs:
        [{"char": char_dict|None, "syllable": str|None, "type": "match"|"deletion"|"insertion"}, ...]
    """
    m = len(chars)
    n = len(syllables)

    # Trường hợp đặc biệt
    if m == 0 and n == 0:
        return []
    if m == 0:
        return [{"char": None, "syllable": s, "type": "insertion"} for s in syllables]
    if n == 0:
        return [{"char": c, "syllable": None, "type": "deletion"} for c in chars]

    # Tính chi phí xoá cho từng ký tự dựa trên kích thước
    if deletion_cost_fn is None:
        # Tính median height để phân biệt ký tự thật vs nhiễu
        heights = [c.get("height", 50) for c in chars]
        median_h = sorted(heights)[len(heights) // 2] if heights else 50

        def deletion_cost_fn(c):
            ratio = c.get("height", 50) / median_h if median_h > 0 else 1
            if ratio < 0.3:
                return 0.3    # Ký tự rất nhỏ → rẻ để xoá (nhiễu)
            elif ratio < 0.5:
                return 0.6    # Ký tự nhỏ
            else:
                return 1.2    # Ký tự bình thường → đắt để xoá

    # Pre-compute substitution costs cho mỗi âm tiết dựa trên từ điển.
    # Âm tiết có trong từ điển → cost thấp (nhiều khả năng match).
    # Âm tiết không có → cost cao (mismatch, DP ưu tiên deletion/insertion).
    MATCH_COST = 0.0         # Âm tiết có candidate → likely match
    MISMATCH_COST = 0.8      # Âm tiết không có candidate → likely mismatch
    syl_has_candidates = [False] * n
    if trans_dict:
        for j_idx, syl in enumerate(syllables):
            syl_has_candidates[j_idx] = bool(trans_dict.get(syl.lower(), []))

    # DP matrix
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    bt = [[""] * (n + 1) for _ in range(m + 1)]  # backtrace

    dp[0][0] = 0
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + deletion_cost_fn(chars[i - 1])
        bt[i][0] = "U"  # Up = deletion
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + 1  # insertion cost = 1
        bt[0][j] = "L"  # Left = insertion

    # Fill DP
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Match/mismatch (diagonal) — dictionary-aware cost:
            # - Âm tiết có candidate trong từ điển → cost thấp (likely valid match)
            # - Âm tiết không có candidate → cost cao (likely mismatch,
            #   DP sẽ ưu tiên skip char hoặc skip syllable thay vì match sai)
            if trans_dict:
                subst_cost = MATCH_COST if syl_has_candidates[j - 1] else MISMATCH_COST
            else:
                subst_cost = 0  # Fallback: hành vi cũ khi không có từ điển

            diag_cost = dp[i - 1][j - 1] + subst_cost

            # Deletion (skip char)
            del_cost = dp[i - 1][j] + deletion_cost_fn(chars[i - 1])

            # Insertion (skip syllable)
            ins_cost = dp[i][j - 1] + 1

            best = min(diag_cost, del_cost, ins_cost)
            dp[i][j] = best

            if best == diag_cost:
                bt[i][j] = "D"  # Diagonal
            elif best == del_cost:
                bt[i][j] = "U"  # Up
            else:
                bt[i][j] = "L"  # Left

    # Backtrack
    aligned = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == "D":
            aligned.append({
                "char": chars[i - 1],
                "syllable": syllables[j - 1],
                "type": "match",
            })
            i -= 1
            j -= 1
        elif i > 0 and bt[i][j] == "U":
            aligned.append({
                "char": chars[i - 1],
                "syllable": None,
                "type": "deletion",
            })
            i -= 1
        elif j > 0:
            aligned.append({
                "char": None,
                "syllable": syllables[j - 1],
                "type": "insertion",
            })
            j -= 1
        else:
            break

    aligned.reverse()
    return aligned


# ---------------------------------------------------------------------------
# Gán Unicode Nôm (Bước 5)
# ---------------------------------------------------------------------------

def assign_unicode(
    aligned: list[dict],
    trans_dict: dict,
    similar_dict: dict | None = None,
    corpus_freq: dict | None = None,
    font_path: str | None = None,
    crops_base: Path | None = None,
    embed_ranker=None,
) -> list[dict]:
    """Gán Unicode Nôm cho mỗi cặp (char, syllable) đã aligned.

    Chiến lược:
      1. Tra từ điển QN → danh sách Unicode candidates
      2. Nếu 1 candidate → gán luôn (confidence = "high")
      3. Nếu nhiều candidates → xếp hạng bằng visual + frequency, chọn tốt nhất
      4. Nếu không tìm thấy trong từ điển → confidence = "low"
      5. Gap (insertion/deletion) → confidence = "gap"

    Returns: updated aligned list với thêm fields:
      nom_char, nom_candidates, confidence, ranking_score
    """
    missing_crops = 0
    for pair in aligned:
        syl = pair.get("syllable")

        if pair["type"] == "deletion":
            pair["nom_char"] = None
            pair["nom_candidates"] = []
            pair["confidence"] = "gap"
            continue

        if pair["type"] == "insertion":
            candidates = trans_dict.get(syl.lower(), []) if syl else []
            pair["nom_char"] = candidates[0] if len(candidates) == 1 else None
            pair["nom_candidates"] = candidates[:10]
            pair["confidence"] = "gap"
            continue

        # type == "match"
        if not syl:
            pair["nom_char"] = None
            pair["nom_candidates"] = []
            pair["confidence"] = "gap"
            continue

        candidates = trans_dict.get(syl.lower(), [])

        if len(candidates) == 1:
            pair["nom_char"] = candidates[0]
            pair["nom_candidates"] = candidates
            pair["confidence"] = "high"
        elif len(candidates) > 1:
            # --- CẢI TIẾN: xếp hạng ứng viên ---
            crop_path = None
            crop_file = ""
            if crops_base and pair.get("char"):
                crop_file = pair["char"].get("crop_file", "")
                if crop_file:
                    # Ưu tiên ảnh cleaned
                    cleaned = crop_file.replace("crops/", "crops_cleaned/")
                    p = crops_base / cleaned
                    if p.exists():
                        crop_path = str(p)
                    else:
                        p = crops_base / crop_file
                        if p.exists():
                            crop_path = str(p)

            if crop_path is None and crop_file:
                missing_crops += 1

            ranked = rank_candidates(
                candidates[:10],
                syl,
                crop_path,
                font_path,
                corpus_freq or {},
                embed_ranker=embed_ranker,
            )

            if ranked:
                pair["nom_char"] = ranked[0][0]
                pair["nom_candidates"] = [r[0] for r in ranked]
                pair["ranking_score"] = round(ranked[0][1], 3)
                # Ghi score gap giữa top-1 và top-2 (dùng cho post-processing)
                if len(ranked) >= 2:
                    pair["ranking_gap"] = round(ranked[0][1] - ranked[1][1], 3)
                else:
                    pair["ranking_gap"] = 1.0
            else:
                pair["nom_char"] = candidates[0]
                pair["nom_candidates"] = candidates[:10]

            pair["confidence"] = "medium"
        else:
            # --- Issue #5: Fuzzy matching khi exact match thất bại ---
            # Thử Levenshtein distance ≤1 trên QN word (cover lỗi OCR/typo)
            fuzzy_candidates = _fuzzy_dict_lookup(syl.lower(), trans_dict)
            if fuzzy_candidates:
                pair["nom_candidates"] = fuzzy_candidates[:10]
                if len(fuzzy_candidates) == 1:
                    pair["nom_char"] = fuzzy_candidates[0]
                    pair["confidence"] = "medium"
                    pair["fuzzy_match"] = True
                else:
                    # Rank fuzzy candidates
                    crop_path_f = None
                    if crops_base and pair.get("char"):
                        crop_file_f = pair["char"].get("crop_file", "")
                        if crop_file_f:
                            cleaned_f = crop_file_f.replace("crops/", "crops_cleaned/")
                            p_f = crops_base / cleaned_f
                            if p_f.exists():
                                crop_path_f = str(p_f)
                            elif (crops_base / crop_file_f).exists():
                                crop_path_f = str(crops_base / crop_file_f)
                    ranked_f = rank_candidates(
                        fuzzy_candidates[:10], syl, crop_path_f, font_path,
                        corpus_freq or {}, embed_ranker=embed_ranker,
                    )
                    if ranked_f:
                        pair["nom_char"] = ranked_f[0][0]
                        pair["nom_candidates"] = [r[0] for r in ranked_f]
                        pair["ranking_score"] = round(ranked_f[0][1], 3)
                    else:
                        pair["nom_char"] = fuzzy_candidates[0]
                    pair["confidence"] = "medium"
                    pair["fuzzy_match"] = True
            else:
                pair["nom_char"] = None
                pair["nom_candidates"] = []
                pair["confidence"] = "low"

    if missing_crops > 0:
        print(f"    [WARNING] {missing_crops} chars missing crop files → ranking without visual similarity")

    return aligned


def _fuzzy_dict_lookup(qn_word: str, trans_dict: dict,
                       max_dist: int = 1) -> list[str]:
    """Tìm candidates bằng fuzzy matching khi exact lookup thất bại.

    Tìm tất cả key trong trans_dict có Levenshtein distance ≤ max_dist
    với qn_word. Trả về union của candidates từ tất cả matched keys.
    Chỉ fuzzy match trên folded text (bỏ dấu) để tránh false positive
    do khác dấu thanh.
    """
    import unicodedata

    def _fold(text):
        text = unicodedata.normalize("NFC", text.lower())
        text = text.replace("đ", "d").replace("Đ", "d")
        nfd = unicodedata.normalize("NFD", text)
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    target_folded = _fold(qn_word)
    if len(target_folded) <= 1:
        return []  # Quá ngắn → fuzzy match không đáng tin

    candidates = []
    seen = set()
    for key in trans_dict:
        key_folded = _fold(key)
        # Quick length filter: Levenshtein distance ≥ |len diff|
        if abs(len(key_folded) - len(target_folded)) > max_dist:
            continue
        dist = _simple_levenshtein(target_folded, key_folded)
        if dist <= max_dist and dist > 0:  # dist=0 would be exact match
            for c in trans_dict[key]:
                if c not in seen:
                    seen.add(c)
                    candidates.append(c)

    return candidates


# ---------------------------------------------------------------------------
# Phase 3: Anchor-Based Alignment Refinement
# ---------------------------------------------------------------------------

def _find_anchors(aligned: list[dict], trans_dict: dict) -> list[int]:
    """Tìm anchor positions — cặp (char, syllable) có confidence cao nhất.

    Anchor là các vị trí trong alignment mà ta chắc chắn đúng:
    - Từ QN chỉ có 1 candidate trong từ điển (unambiguous)
    - type == "match" (không phải gap)

    Returns: sorted list of indices in aligned
    """
    anchors = []
    for i, pair in enumerate(aligned):
        if pair["type"] != "match":
            continue
        if pair.get("confidence") != "high":
            continue
        syl = pair.get("syllable", "")
        if not syl:
            continue
        # Chỉ coi là anchor nếu từ điển unambiguous (1 candidate)
        candidates = trans_dict.get(syl.lower(), [])
        if len(candidates) == 1:
            anchors.append(i)
    return anchors


def anchor_refine_alignment(
    aligned: list[dict], trans_dict: dict,
    similar_dict: dict | None = None,
    corpus_freq: dict | None = None,
    font_path: str | None = None,
    crops_base: Path | None = None,
    embed_ranker=None,
) -> list[dict]:
    """Phase 3: Tinh chỉnh alignment dựa trên anchors.

    Ý tưởng: Giữa 2 anchors (vị trí chắc chắn đúng), re-align segment
    con bằng DP cost-aware (dùng từ điển để tính cost thay vì cost=0).

    Cải tiến:
    1. Tìm anchors (unambiguous dictionary matches)
    2. Giữa 2 anchors liên tiếp, nếu có medium/low → re-rank candidates
       dựa trên context (ký tự trước/sau đã biết)
    3. Cross-reference: nếu ký tự A ở vị trí i được gán nom_char X,
       mà X cũng là candidate ở vị trí j gần đó → tăng confidence
    """
    anchors = _find_anchors(aligned, trans_dict)

    if len(anchors) < 2:
        return aligned  # Không đủ anchors để refine

    # Re-rank medium candidates giữa các anchors
    for seg_start_idx in range(len(anchors) - 1):
        a_start = anchors[seg_start_idx]
        a_end = anchors[seg_start_idx + 1]

        # Lấy context từ anchors
        anchor_start_char = aligned[a_start].get("nom_char", "")
        anchor_end_char = aligned[a_end].get("nom_char", "")

        # Re-rank medium items giữa 2 anchors
        for i in range(a_start + 1, a_end):
            pair = aligned[i]
            if pair.get("confidence") != "medium":
                continue
            if pair["type"] != "match":
                continue

            candidates = pair.get("nom_candidates", [])
            if len(candidates) <= 1:
                continue

            syl = pair.get("syllable", "")
            if not syl:
                continue

            # Re-rank với context bonus
            crop_path = None
            if crops_base and pair.get("char"):
                crop_file = pair["char"].get("crop_file", "")
                if crop_file:
                    cleaned = crop_file.replace("crops/", "crops_cleaned/")
                    p = crops_base / cleaned
                    if p.exists():
                        crop_path = str(p)
                    elif (crops_base / crop_file).exists():
                        crop_path = str(crops_base / crop_file)

            ranked = rank_candidates(
                candidates[:10], syl, crop_path, font_path,
                corpus_freq or {}, embed_ranker=embed_ranker,
            )

            if ranked and len(ranked) >= 2:
                # CẢI TIẾN: tăng gap threshold 0.15 → 0.20
                # VÀ require top_score tuyệt đối > 0.35 (tránh upgrade khi
                # cả 2 candidates đều có score thấp)
                top_score = ranked[0][1]
                second_score = ranked[1][1]
                gap = top_score - second_score
                if gap > 0.20 and top_score > 0.35:
                    pair["nom_char"] = ranked[0][0]
                    pair["nom_candidates"] = [r[0] for r in ranked]
                    pair["ranking_score"] = round(top_score, 3)
                    pair["ranking_gap"] = round(gap, 3)
                    pair["confidence"] = "high"
                    pair["anchor_refined"] = True
                elif ranked[0][0] != pair.get("nom_char"):
                    # Update candidate nhưng giữ medium nếu gap chưa đủ lớn
                    pair["nom_char"] = ranked[0][0]
                    pair["nom_candidates"] = [r[0] for r in ranked]
                    pair["ranking_score"] = round(top_score, 3)
                    pair["ranking_gap"] = round(gap, 3)

    return aligned


# ---------------------------------------------------------------------------
# Phase 4: Multi-Tier Post-Processing
# ---------------------------------------------------------------------------

def _fold_text(text: str) -> str:
    """Normalize text: lowercase, bỏ dấu, đ→d."""
    import unicodedata
    text = unicodedata.normalize("NFC", text.lower())
    text = text.replace("đ", "d").replace("Đ", "d")
    # Strip combining characters (tone marks)
    nfd = unicodedata.normalize("NFD", text)
    folded = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return folded


def multi_tier_postprocess(
    all_labels: list[dict],
    trans_dict: dict,
    similar_dict: dict | None = None,
    verbose: bool = True,
) -> int:
    """Phase 4: Multi-tier post-processing cho remaining medium labels.

    3 tầng recovery:
      Tier 1: Dictionary Intersection — nhanh, O(1)
              S1 = readings của nom_char hiện tại
              S2 = candidates cho quoc_ngu (reverse lookup)
              Nếu S1 ∩ S2 ≠ ∅ → chọn best → HIGH

      Tier 2: Reverse Lookup — fallback
              Tra ngược quoc_ngu → tất cả candidates
              Chọn candidate có reading gần nhất với nom_char hiện tại

      Tier 3: Similarity Expansion — dùng similar_dict
              Nếu nom_char có similar chars → check xem similar nào
              khớp với quoc_ngu tốt hơn

    Returns: number of labels upgraded
    """
    # Build reverse dict: quoc_ngu → list of nom chars
    qn_to_nom: dict[str, list[str]] = {}
    for qn, chars in trans_dict.items():
        qn_to_nom[qn.lower()] = chars

    # Build reading map: nom_char → list of quoc_ngu readings
    reading_map: dict[str, list[str]] = {}
    for qn, chars in trans_dict.items():
        for c in chars:
            reading_map.setdefault(c, [])
            if qn not in reading_map[c]:
                reading_map[c].append(qn)

    tier1_upgraded = 0
    tier2_upgraded = 0
    tier3_upgraded = 0

    for lab in all_labels:
        if lab.get("confidence") != "medium":
            continue
        if lab.get("type") != "match":
            continue

        qn = (lab.get("quoc_ngu") or "").lower()
        nom_char = lab.get("nom_char")
        candidates = lab.get("nom_candidates", [])

        if not qn or not nom_char:
            continue

        # ── Tier 1: Dictionary Intersection (với strong signal) ──
        # S1 = set of QN readings cho nom_char hiện tại
        s1 = set(reading_map.get(nom_char, []))
        # S2 = set of nom chars cho QN word
        s2 = set(qn_to_nom.get(qn, []))

        if nom_char in s2 and qn in s1:
            # nom_char vừa là candidate cho qn, VÀ qn là reading của nom_char
            # CẢI TIẾN: chỉ upgrade HIGH nếu có thêm strong signal:
            #   (a) Ít candidates (≤3) → ít ambiguity, HOẶC
            #   (b) ranking_gap lớn (>0.10) → visual rõ ràng, HOẶC
            #   (c) Đã được OCR confirm
            n_cands = len(s2)
            ranking_gap = lab.get("ranking_gap", 0)
            has_ocr = lab.get("ocr_source", False)
            if n_cands <= 3 or ranking_gap > 0.10 or has_ocr:
                lab["confidence"] = "high"
                lab["postprocess_tier"] = 1
                tier1_upgraded += 1
                continue
            # Dict confirm nhưng nhiều candidate + visual không rõ → giữ medium
            # (tránh upgrade sai khi dict quá rộng)

        # Tìm trong candidates: char nào vừa có reading = qn
        intersection_chars = [c for c in candidates if c in s2 and qn in set(reading_map.get(c, []))]
        if intersection_chars:
            # CẢI TIẾN: chỉ thay candidate nếu có strong signal
            # Ưu tiên char có ít readings nhất (cụ thể nhất)
            intersection_chars.sort(key=lambda c: len(reading_map.get(c, [])))
            best = intersection_chars[0]
            best_readings = len(reading_map.get(best, []))
            ranking_gap = lab.get("ranking_gap", 0)
            has_ocr = lab.get("ocr_source", False)

            if best_readings <= 3 or ranking_gap > 0.10 or has_ocr:
                lab["nom_char"] = best
                lab["nom_unicode"] = f"U+{ord(best):04X}"
                lab["confidence"] = "high"
                lab["postprocess_tier"] = 1
                new_cands = [best] + [c for c in candidates if c != best]
                lab["nom_candidates"] = new_cands
                tier1_upgraded += 1
                continue
            elif best != nom_char:
                # Thay candidate cụ thể hơn nhưng giữ medium
                lab["nom_char"] = best
                lab["nom_unicode"] = f"U+{ord(best):04X}"
                lab["postprocess_tier"] = 1
                new_cands = [best] + [c for c in candidates if c != best]
                lab["nom_candidates"] = new_cands
                continue

        # ── Tier 2: Reverse Lookup with Levenshtein ──
        qn_folded = _fold_text(qn)
        best_match = None
        best_edit = float("inf")

        for cand in candidates:
            readings = reading_map.get(cand, [])
            for r in readings:
                r_folded = _fold_text(r)
                # Simple Levenshtein distance
                dist = _simple_levenshtein(qn_folded, r_folded)
                if dist < best_edit:
                    best_edit = dist
                    best_match = cand

        if best_match and best_edit <= 1:
            lab["nom_char"] = best_match
            lab["nom_unicode"] = f"U+{ord(best_match):04X}"
            lab["postprocess_tier"] = 2
            new_cands = [best_match] + [c for c in candidates if c != best_match]
            lab["nom_candidates"] = new_cands
            # edit_dist=0 → exact match → high; edit_dist=1 → fuzzy → medium
            if best_edit == 0:
                lab["confidence"] = "high"
            else:
                lab["confidence"] = "medium"
            tier2_upgraded += 1
            continue

        # ── Tier 3: Similarity Expansion ──
        if similar_dict and nom_char in similar_dict:
            similar_chars = similar_dict[nom_char]
            for sim_char in similar_chars[:5]:
                if sim_char in s2:
                    sim_readings = set(reading_map.get(sim_char, []))
                    if qn in sim_readings:
                        lab["nom_char"] = sim_char
                        lab["nom_unicode"] = f"U+{ord(sim_char):04X}"
                        lab["confidence"] = "high"
                        lab["postprocess_tier"] = 3
                        tier3_upgraded += 1
                        break

    total = tier1_upgraded + tier2_upgraded + tier3_upgraded
    if verbose and total > 0:
        print(f"\n  Post-Processing: {total} upgraded "
              f"(tier1_dict∩={tier1_upgraded}, tier2_reverse={tier2_upgraded}, "
              f"tier3_similar={tier3_upgraded})")

    return total


def _simple_levenshtein(s1: str, s2: str) -> int:
    """Simple Levenshtein distance."""
    if len(s1) < len(s2):
        return _simple_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            dele = curr_row[j] + 1
            sub = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(ins, dele, sub))
        prev_row = curr_row
    return prev_row[-1]


# ---------------------------------------------------------------------------
# Levenshtein Validation (Lượt 2 - validate bằng từ điển)
# ---------------------------------------------------------------------------

def validate_alignment(aligned: list[dict], trans_dict: dict,
                       similar_dict: dict | None = None) -> dict:
    """Validate alignment bằng Levenshtein lượt 2 (dùng từ điển).

    Kiểm tra mỗi cặp (nom_char, syllable) có compatible không.

    Returns: dict thống kê validation
    """
    total = 0
    compatible = 0
    similar_match = 0
    incompatible = 0
    no_unicode = 0

    for pair in aligned:
        if pair["type"] != "match":
            continue

        total += 1
        nom = pair.get("nom_char")
        syl = pair.get("syllable", "")

        if not nom:
            no_unicode += 1
            pair["validation"] = "no_unicode"
            continue

        if is_compatible(nom, syl, trans_dict):
            compatible += 1
            pair["validation"] = "compatible"
        elif similar_dict and is_compatible(nom, syl, trans_dict, similar_dict):
            similar_match += 1
            pair["validation"] = "similar"
        else:
            incompatible += 1
            pair["validation"] = "incompatible"

    return {
        "total_matched": total,
        "compatible": compatible,
        "similar_match": similar_match,
        "incompatible": incompatible,
        "no_unicode": no_unicode,
        "match_rate": (compatible + similar_match) / total if total > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Render ảnh Nôm đánh máy (Bước 5.5)
# ---------------------------------------------------------------------------

def render_nom_char(char: str, font, size: int = 64) -> np.ndarray | None:
    """Render 1 ký tự Nôm thành ảnh grayscale bằng PIL.

    Returns: numpy array (size x size) grayscale, hoặc None nếu font không có glyph.
    """
    from PIL import Image, ImageDraw

    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    # Kiểm tra font có glyph không
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
    except Exception:
        return None

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    if w <= 0 or h <= 0:
        return None

    # Center ký tự
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)

    return np.array(img)


def load_nom_font(font_path: str, size: int = 48):
    """Load font Nôm bằng PIL."""
    from PIL import ImageFont
    try:
        return ImageFont.truetype(font_path, size=size)
    except Exception as e:
        print(f"[WARN] Không load được font {font_path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Xuất Excel (Bước 6)
# ---------------------------------------------------------------------------

def export_excel(dataset_path: str, output_path: str, prepared_dir: str):
    """Xuất dataset ra file Excel đồng bộ với labels.csv.

    Sheet 1: "Dataset" - Chỉ matched entries (giống labels.csv)
      - Cột: image, nom_char, label, reading, confidence, bbox, page, source
      - Màu theo confidence: xanh lá (high), trắng (medium), vàng (low)
    Sheet 2: "Thống kê" - Summary + thống kê theo trang
    """
    try:
        import xlsxwriter
    except ImportError:
        print("[ERROR] Cần cài xlsxwriter: pip install xlsxwriter", file=sys.stderr)
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # dataset.json đã là format chuẩn — dùng trực tiếp
    source_name = Path(prepared_dir).name
    wb = xlsxwriter.Workbook(output_path)

    # --- Formats ---
    header_fmt = wb.add_format({
        "bold": True, "bg_color": "#4472C4", "font_color": "white",
        "border": 1, "text_wrap": True, "valign": "vcenter",
    })
    high_fmt = wb.add_format({"bg_color": "#C6EFCE", "border": 1})
    med_fmt = wb.add_format({"border": 1})
    low_fmt = wb.add_format({"bg_color": "#FFEB9C", "border": 1})
    nom_high = wb.add_format({"bg_color": "#C6EFCE", "border": 1, "font_size": 16, "align": "center"})
    nom_med = wb.add_format({"border": 1, "font_size": 16, "align": "center"})
    nom_low = wb.add_format({"bg_color": "#FFEB9C", "border": 1, "font_size": 16, "align": "center"})
    center_fmt = wb.add_format({"border": 1, "align": "center"})

    # ============================================
    # Sheet 1: Dataset (đồng bộ labels.csv)
    # ============================================
    ws = wb.add_worksheet("Dataset")
    headers = ["image", "nom_char", "label", "reading", "confidence", "bbox", "page", "source"]
    for c, h in enumerate(headers):
        ws.write(0, c, h, header_fmt)

    ws.set_column(0, 0, 45)   # image
    ws.set_column(1, 1, 10)   # nom_char
    ws.set_column(2, 2, 12)   # label (unicode)
    ws.set_column(3, 3, 12)   # reading (QN)
    ws.set_column(4, 4, 12)   # confidence
    ws.set_column(5, 5, 22)   # bbox
    ws.set_column(6, 6, 8)    # page
    ws.set_column(7, 7, 20)   # source

    row = 1
    for entry in data:
        conf = entry.get("confidence", "medium")
        fmt_map = {"high": high_fmt, "medium": med_fmt, "low": low_fmt}
        nom_map = {"high": nom_high, "medium": nom_med, "low": nom_low}
        fmt = fmt_map.get(conf, med_fmt)
        nfmt = nom_map.get(conf, nom_med)

        ws.write(row, 0, entry.get("image", ""), fmt)
        ws.write(row, 1, entry.get("nom_char", ""), nfmt)
        ws.write(row, 2, entry.get("label", ""), center_fmt)
        ws.write(row, 3, entry.get("reading", ""), fmt)
        ws.write(row, 4, conf, fmt)
        ws.write(row, 5, entry.get("bbox", ""), center_fmt)
        ws.write(row, 6, entry.get("page", ""), center_fmt)
        ws.write(row, 7, entry.get("source", source_name), fmt)
        row += 1

    ws.autofilter(0, 0, row - 1, len(headers) - 1)
    ws.freeze_panes(1, 0)

    # ============================================
    # Sheet 2: Thống kê
    # ============================================
    ws2 = wb.add_worksheet("Thống kê")
    title_fmt = wb.add_format({"bold": True, "font_size": 14})
    label_fmt = wb.add_format({"bold": True, "border": 1, "bg_color": "#D9E2F3"})
    val_fmt = wb.add_format({"border": 1, "align": "center", "font_size": 12})
    pct_fmt = wb.add_format({"border": 1, "align": "center", "font_size": 12, "num_format": "0.0%"})

    ws2.set_column(0, 0, 25)
    ws2.set_column(1, 6, 12)

    # --- Tổng kết ---
    ws2.write(0, 0, "Tổng kết gán nhãn", title_fmt)
    ws2.write(1, 0, f"Nguồn: {source_name}", wb.add_format({"italic": True}))

    total_all = len(data)
    total_high = sum(1 for d in data if d.get("confidence") == "high")
    total_med = sum(1 for d in data if d.get("confidence") == "medium")
    total_low = sum(1 for d in data if d.get("confidence") == "low")
    unique_nom = len(set(d.get("nom_char", "") for d in data))
    unique_pages = len(set(d.get("page") for d in data))

    stats_data = [
        ("Tổng entries", total_all),
        ("Tổng trang", unique_pages),
        ("High confidence", total_high),
        ("Medium confidence", total_med),
        ("Low confidence", total_low),
        ("Unicode Nôm unique", unique_nom),
    ]
    for i, (label, val) in enumerate(stats_data):
        ws2.write(3 + i, 0, label, label_fmt)
        ws2.write(3 + i, 1, val, val_fmt)

    # --- Thống kê theo trang ---
    r = 3 + len(stats_data) + 2
    ws2.write(r, 0, "Thống kê theo trang", title_fmt)
    r += 1
    page_headers = ["Trang", "Total", "High", "Medium", "Low"]
    for c, h in enumerate(page_headers):
        ws2.write(r, c, h, header_fmt)

    from collections import defaultdict
    page_stats = defaultdict(lambda: {"total": 0, "high": 0, "medium": 0, "low": 0})
    for entry in data:
        p = entry.get("page", 0)
        conf = entry.get("confidence", "")
        page_stats[p]["total"] += 1
        if conf in ("high", "medium", "low"):
            page_stats[p][conf] += 1

    r += 1
    for page in sorted(page_stats):
        s = page_stats[page]
        ws2.write(r, 0, page, center_fmt)
        ws2.write(r, 1, s["total"], center_fmt)
        ws2.write(r, 2, s["high"], center_fmt)
        ws2.write(r, 3, s["medium"], center_fmt)
        ws2.write(r, 4, s["low"], center_fmt)
        r += 1

    # --- Chú thích ---
    r += 1
    ws2.write(r, 0, "Chú thích màu:", title_fmt)
    ws2.write(r + 1, 0, "High", high_fmt)
    ws2.write(r + 1, 1, "1 ứng viên hoặc OCR/Consistency xác nhận")
    ws2.write(r + 2, 0, "Medium", med_fmt)
    ws2.write(r + 2, 1, "Nhiều ứng viên, xếp hạng bằng visual + frequency")
    ws2.write(r + 3, 0, "Low", low_fmt)
    ws2.write(r + 3, 1, "Âm QN không có trong từ điển")

    wb.close()
    print(f"  Excel: {output_path}")


# ---------------------------------------------------------------------------
# Xuất review image
# ---------------------------------------------------------------------------

def save_review_image(aligned_page: list[dict], output_path: str,
                      crops_dir: str, typed_dir: str | None,
                      max_chars: int = 80):
    """Tạo ảnh review RGB: [viết tay | đánh máy | nhãn] cho mỗi ký tự.

    Mỗi ô gồm 3 phần xếp dọc:
      1. Ảnh viết tay (crop) — 80×80
      2. Ảnh đánh máy (typed_nom) — 80×80
      3. Nhãn text 3 dòng: chữ Nôm, QN reading, Unicode — 44px

    Viền màu theo confidence:
      Xanh lá = high, Cam = medium, Đỏ = low/gap
    """
    from PIL import Image, ImageDraw, ImageFont

    cell_size = 80  # Lớn hơn 64 → dễ nhìn
    pairs = [p for p in aligned_page if p["type"] == "match" and p.get("char")]
    pairs = pairs[:max_chars]

    if not pairs:
        return

    cols = min(12, len(pairs))
    rows = (len(pairs) + cols - 1) // cols

    text_h = 44  # 3 dòng text
    gap = 3  # Khoảng cách giữa các ô
    cell_h = cell_size * 2 + text_h + gap
    cell_w = cell_size + gap

    canvas_h = rows * cell_h + gap * 2
    canvas_w = cols * cell_w + gap * 2

    # Dùng RGB cho viền màu
    pil_img = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(pil_img)

    # Font cho text label
    label_font = None
    nom_font = None
    # Font cho chữ Nôm (cần hỗ trợ CJK)
    cjk_font_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in cjk_font_paths:
        try:
            nom_font = ImageFont.truetype(fp, 16)
            break
        except Exception:
            continue

    try:
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        try:
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except Exception:
            label_font = ImageFont.load_default()

    if nom_font is None:
        nom_font = label_font

    # Màu viền theo confidence
    border_colors = {
        "high": (34, 139, 34),     # Xanh lá đậm
        "medium": (230, 140, 0),   # Cam
        "low": (200, 50, 50),      # Đỏ
        "gap": (180, 180, 180),    # Xám
    }

    for idx, pair in enumerate(pairs):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w + gap
        y0 = r * cell_h + gap

        char_info = pair["char"]
        conf = pair.get("confidence", "")
        nom = pair.get("nom_char", "")
        syl = pair.get("syllable", "")

        # Viền màu nền theo confidence
        border_color = border_colors.get(conf, (180, 180, 180))
        draw.rectangle(
            [x0 - 2, y0 - 2, x0 + cell_size + 1, y0 + 2 * cell_size + text_h],
            outline=border_color, width=2,
        )

        # 1. Ảnh viết tay (crop cleaned hoặc gốc)
        crop_file = char_info.get("crop_file", "")
        crop_path = Path(crops_dir).parent / crop_file.replace("crops/", "crops_cleaned/")
        if not crop_path.exists():
            crop_path = Path(crops_dir).parent / crop_file
        if crop_path.exists():
            hw_img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
            if hw_img is not None:
                hw_img = cv2.resize(hw_img, (cell_size, cell_size))
                hw_pil = Image.fromarray(hw_img).convert("RGB")
                pil_img.paste(hw_pil, (x0, y0))

        # Đường kẻ ngăn cách giữa viết tay và đánh máy
        draw.line([(x0, y0 + cell_size), (x0 + cell_size - 1, y0 + cell_size)],
                  fill=(200, 200, 200), width=1)

        # 2. Ảnh đánh máy (nếu có)
        if typed_dir and nom:
            typed_path = Path(typed_dir) / f"{ord(nom):06X}.png"
            if typed_path.exists():
                tp_img = cv2.imread(str(typed_path), cv2.IMREAD_GRAYSCALE)
                if tp_img is not None:
                    tp_img = cv2.resize(tp_img, (cell_size, cell_size))
                    tp_pil = Image.fromarray(tp_img).convert("RGB")
                    pil_img.paste(tp_pil, (x0, y0 + cell_size))

        # 3. Nhãn text 3 dòng
        text_y = y0 + 2 * cell_size + 2

        # Dòng 1: Chữ Nôm + confidence tag (màu)
        conf_tag = {"high": "H", "medium": "M", "low": "L"}.get(conf, "?")
        conf_text_color = border_colors.get(conf, (100, 100, 100))
        line1 = f"{conf_tag} "
        draw.text((x0 + 2, text_y), line1, fill=conf_text_color, font=label_font)
        # Chữ Nôm Unicode bên cạnh tag
        if nom:
            draw.text((x0 + 18, text_y - 1), nom, fill=(0, 0, 0), font=nom_font)

        # Dòng 2: Quốc ngữ reading
        if syl:
            qn_text = syl if len(syl) <= 10 else syl[:9] + "…"
            draw.text((x0 + 2, text_y + 15), qn_text, fill=(0, 0, 100), font=label_font)

        # Dòng 3: Unicode codepoint
        if nom:
            code_text = f"U+{ord(nom):04X}"
            draw.text((x0 + 2, text_y + 29), code_text, fill=(120, 120, 120), font=label_font)

    pil_img.save(output_path)


# ---------------------------------------------------------------------------
# Xử lý 1 trang
# ---------------------------------------------------------------------------

def process_page(
    page_info: dict,
    prepared_dir: Path,
    trans_dict: dict,
    similar_dict: dict | None,
    nom_font,
    output_dir: Path,
    typed_nom_dir: Path | None,
    review: bool = False,
    use_ocr: bool = False,
    verbose: bool = True,
    corpus_freq: dict | None = None,
    font_path: str | None = None,
    embed_ranker=None,
) -> dict | None:
    """Xử lý gán nhãn cho 1 trang."""
    book_page = page_info["book_page"]

    # Load detection JSON
    det_path = prepared_dir / "detected" / f"page_{book_page:04d}_detection.json"
    if not det_path.exists():
        if verbose:
            print(f"  [SKIP] Trang {book_page}: không có detection")
        return None

    with open(det_path, "r", encoding="utf-8") as f:
        det_data = json.load(f)

    detection = det_data["detection"]

    # Load transcription
    trans_path = prepared_dir / page_info["transcription_file"]
    if not trans_path.exists():
        if verbose:
            print(f"  [SKIP] Trang {book_page}: không có transcription")
        return None

    with open(trans_path, "r", encoding="utf-8") as f:
        trans_lines = f.read().strip().split("\n")

    # --- Issue #9: Column order verification ---
    # Kiểm tra thứ tự cột (R→L) bằng cách so sánh dict match rate
    # giữa column_order hiện tại vs reversed order.
    # Nếu reversed có match rate cao hơn đáng kể → cảnh báo.
    if len(detection["columns"]) >= 3 and len(trans_lines) >= 3 and trans_dict:
        def _quick_match_rate(columns_data, lines, td):
            """Tính nhanh % cặp (char_count, syllable_count) khớp."""
            hits = 0
            total = 0
            for i, col in enumerate(columns_data):
                if i >= len(lines):
                    break
                syls = normalize_syllables(lines[i].split())
                n_chars = len(col["chars"])
                n_syls = len(syls)
                total += 1
                # "Khớp" nếu chênh lệch ≤ 15%
                if n_syls > 0 and n_chars > 0:
                    ratio = abs(n_chars - n_syls) / max(n_chars, n_syls)
                    if ratio <= 0.15:
                        hits += 1
            return hits / total if total > 0 else 0

        cols_data = detection["columns"]
        normal_rate = _quick_match_rate(cols_data, trans_lines, trans_dict)
        reversed_lines = list(reversed(trans_lines))
        reverse_rate = _quick_match_rate(cols_data, reversed_lines, trans_dict)

        if reverse_rate > normal_rate + 0.2 and reverse_rate > 0.5:
            if verbose:
                print(f"  [COLUMN_ORDER] Trang {book_page}: "
                      f"reversed order match rate ({reverse_rate:.0%}) >> "
                      f"normal order ({normal_rate:.0%}) → REVERSING lines")
            trans_lines = reversed_lines
        elif normal_rate < 0.3 and verbose:
            print(f"  [COLUMN_ORDER] Trang {book_page}: "
                  f"low match rate ({normal_rate:.0%}) — column/text alignment may be wrong")

    # --- OCR API (nếu bật) ---
    ocr_columns = None
    if use_ocr:
        page_image = prepared_dir / "pages" / f"page_{book_page:04d}.png"
        if page_image.exists():
            cache_dir = prepared_dir / "labeled" / "ocr_cache"
            cache_file = str(cache_dir / f"page_{book_page:04d}_ocr.json")
            ocr_columns = ocr_page(
                str(page_image), cache_path=cache_file, verbose=verbose,
            )

    # Align từng cột
    page_labels = []
    all_aligned_page = []  # Tích luỹ aligned pairs cho review image
    page_stats = {
        "total_chars": 0, "matched": 0, "gaps": 0,
        "high": 0, "medium": 0, "low": 0,
        "flagged_columns": [],  # Cột có mismatch lớn (Issue #2)
    }
    ocr_used_indices = set()  # track cột OCR đã dùng

    for col_data in detection["columns"]:
        col_num = col_data["column"]
        chars = col_data["chars"]

        # Lấy âm tiết tương ứng
        line_idx = col_num - 1
        if line_idx < len(trans_lines):
            raw_syllables = trans_lines[line_idx].split()
        else:
            raw_syllables = []

        # Chuẩn hoá tên riêng
        syllables = normalize_syllables(raw_syllables)

        # --- Issue #2: Detection validation ---
        # Flag cột có chênh lệch >30% giữa detected chars và syllables
        n_chars = len(chars)
        n_syls = len(syllables)
        if n_syls > 0 and n_chars > 0:
            mismatch_ratio = abs(n_chars - n_syls) / max(n_chars, n_syls)
            if mismatch_ratio > 0.30:
                page_stats["flagged_columns"].append({
                    "column": col_num,
                    "detected_chars": n_chars,
                    "syllables": n_syls,
                    "mismatch_ratio": round(mismatch_ratio, 2),
                })
                if verbose:
                    print(f"    [FLAG] Cột {col_num}: {n_chars} chars vs "
                          f"{n_syls} syllables (mismatch {mismatch_ratio:.0%}) "
                          f"— alignment may be unreliable")

        # Bước 4: Levenshtein alignment (dictionary-aware cost)
        aligned = levenshtein_align(chars, syllables, trans_dict=trans_dict)

        # Bước 5: Gán Unicode (với ranking visual/embedding + frequency)
        crops_base = prepared_dir / "detected"
        aligned = assign_unicode(
            aligned, trans_dict, similar_dict,
            corpus_freq=corpus_freq,
            font_path=font_path,
            crops_base=crops_base,
            embed_ranker=embed_ranker,
        )

        # Phase 3: Anchor-based refinement
        aligned = anchor_refine_alignment(
            aligned, trans_dict, similar_dict,
            corpus_freq=corpus_freq, font_path=font_path,
            crops_base=crops_base, embed_ranker=embed_ranker,
        )

        # Validate bằng từ điển
        validate_alignment(aligned, trans_dict, similar_dict)

        # Bước 5b: So khớp với OCR API bằng bbox overlap (nếu có)
        if ocr_columns:
            ocr_col_idx = _find_best_ocr_column(
                ocr_columns, chars, ocr_used_indices,
            )
            if ocr_col_idx is not None:
                ocr_used_indices.add(ocr_col_idx)
                ocr_col = ocr_columns[ocr_col_idx]
                n_upgraded = _match_ocr_bbox(
                    ocr_col, aligned, trans_dict,
                )
                if verbose and n_upgraded > 0:
                    print(
                        f"    Cột {col_num}: OCR upgraded {n_upgraded} "
                        f"medium→high (ocr_col={ocr_col_idx+1}, "
                        f"{len(ocr_col)} chars)"
                    )

        # Tích luỹ aligned pairs (dùng cho review image)
        all_aligned_page.extend(aligned)

        # Bước 5.5: Render ảnh Nôm đánh máy
        if nom_font and typed_nom_dir:
            for pair in aligned:
                nom = pair.get("nom_char")
                if nom:
                    typed_path = typed_nom_dir / f"{ord(nom):06X}.png"
                    if not typed_path.exists():
                        rendered = render_nom_char(nom, nom_font, size=64)
                        if rendered is not None:
                            cv2.imwrite(str(typed_path), rendered)
                            pair["typed_nom_file"] = f"typed_nom/{ord(nom):06X}.png"
                        else:
                            pair["typed_nom_file"] = None
                    else:
                        pair["typed_nom_file"] = f"typed_nom/{ord(nom):06X}.png"

        # Gom kết quả
        for pair in aligned:
            char_info = pair.get("char")
            nom_char = pair.get("nom_char")
            label = {
                "page": book_page,
                "column": col_num,
                "type": pair["type"],
                "quoc_ngu": pair.get("syllable"),
                "nom_char": nom_char,
                "nom_unicode": f"U+{ord(nom_char):04X}" if nom_char else None,
                "nom_candidates": pair.get("nom_candidates", []),
                "confidence": pair.get("confidence", "gap"),
                "validation": pair.get("validation"),
                "typed_nom_file": pair.get("typed_nom_file"),
                "ocr_source": pair.get("ocr_source", False),
                "ocr_char": pair.get("ocr_char"),
                "ranking_score": pair.get("ranking_score"),
            }

            if char_info:
                label["char_idx"] = char_info["char_idx"]
                label["bbox"] = char_info["bbox"]
                label["crop_file"] = char_info.get("crop_file")
                label["crop_cleaned_file"] = char_info.get("crop_file", "").replace(
                    "crops/", "crops_cleaned/"
                )
                label["position"] = char_info["char_idx"]

            page_labels.append(label)

            # Thống kê
            if pair["type"] == "match":
                page_stats["matched"] += 1
                conf = pair.get("confidence", "low")
                if conf in page_stats:
                    page_stats[conf] += 1
            else:
                page_stats["gaps"] += 1

        page_stats["total_chars"] += len(chars)

    # Review image (dùng aligned pairs đã tính — bao gồm ranking + OCR)
    if review:
        review_dir = output_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = str(prepared_dir / "detected" / "crops")
        save_review_image(
            all_aligned_page,
            str(review_dir / f"page_{book_page:04d}_review.png"),
            crops_dir,
            str(typed_nom_dir) if typed_nom_dir else None,
            max_chars=50,
        )

    if verbose:
        total = page_stats["matched"]
        high = page_stats["high"]
        med = page_stats["medium"]
        low = page_stats["low"]
        gaps = page_stats["gaps"]
        print(
            f"  Trang {book_page:4d}: {total:>3} matched "
            f"[H={high} M={med} L={low}] gaps={gaps}"
        )

    return {
        "book_page": book_page,
        "labels": page_labels,
        "stats": page_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_prepared_dir(
    prepared_dir: Path,
    font_path: str | None = None,
    page_filter: int | None = None,
    review: bool = False,
    excel: bool = False,
    use_ocr: bool = False,
    use_dinov2: bool = False,
    embedding_checkpoint: str | None = None,
    embedding_gallery: str | None = None,
    verbose: bool = True,
):
    """Xử lý gán nhãn cho toàn bộ thư mục prepared."""

    # --- Load embedding ranker (nếu có) ---
    embed_ranker = None
    if use_dinov2:
        try:
            from dinov2_ranker import DINOv2Ranker
            base_dir_d = Path(__file__).parent
            font_d = font_path or str(base_dir_d / "FontDiffusion" / "fonts" / "NomNaTong-Regular.ttf")
            embed_ranker = DINOv2Ranker(
                model_name="dinov2_vits14",
                font_path=font_d,
                device=None,  # auto-detect
            )
            print(f"DINOv2 ranker: ON")
        except Exception as e:
            print(f"[WARN] Không load được DINOv2 ranker: {e}")
            embed_ranker = None
    elif embedding_checkpoint:
        try:
            from embedding.embed_ranker import get_ranker
            base_dir_embed = Path(__file__).parent
            gallery = embedding_gallery or str(base_dir_embed / "embedding" / "data" / "gallery")
            embed_ranker = get_ranker(
                checkpoint_path=embedding_checkpoint,
                gallery_dir=gallery,
                device="cpu",
            )
            if embed_ranker and embed_ranker.is_ready:
                print(f"Embedding ranker: ON ({len(embed_ranker.gallery_codes)} ký tự)")
            else:
                print("[WARN] Embedding ranker không sẵn sàng, dùng ranking cơ bản")
                embed_ranker = None
        except Exception as e:
            print(f"[WARN] Không load được embedding ranker: {e}")
            embed_ranker = None

    # --- Load manifest ---
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] Không tìm thấy manifest: {manifest_path}", file=sys.stderr)
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # --- Load từ điển ---
    base_dir = Path(__file__).parent
    dict_dir = base_dir / "Alignment" / "Code" / "dict"

    trans_dict_path = dict_dir / "QuocNgu_SinoNom_Merged.csv"
    if not trans_dict_path.exists():
        trans_dict_path = dict_dir / "QuocNgu_SinoNom_TongHop3.csv"
    similar_dict_path = dict_dir / "SinoNom_Similar_Dic_v2.csv"

    if not trans_dict_path.exists():
        print(f"[ERROR] Không tìm thấy từ điển: {trans_dict_path}", file=sys.stderr)
        return

    print(f"Loading từ điển QN→Nôm: {trans_dict_path}")
    trans_dict = load_translation_dict(str(trans_dict_path))
    print(f"  → {len(trans_dict)} từ QN, {sum(len(v) for v in trans_dict.values())} mapping")

    similar_dict = None
    if similar_dict_path.exists():
        print(f"Loading từ điển Similar: {similar_dict_path}")
        similar_dict = load_similarity_dict(str(similar_dict_path))
        print(f"  → {len(similar_dict)} ký tự")

    # --- Load font ---
    nom_font = None
    if font_path is None:
        font_path = str(base_dir / "FontDiffusion" / "fonts" / "NomNaTong-Regular.ttf")

    nom_font = load_nom_font(font_path, size=48)
    if nom_font:
        print(f"Font Nôm: {font_path}")
    else:
        print(f"[WARN] Không có font Nôm, bỏ qua render ảnh đánh máy")

    # --- Build character specificity ---
    print("Building character specificity...")
    corpus_freq = build_char_specificity(trans_dict)
    print(f"  → {len(corpus_freq)} ký tự Nôm với specificity score")

    # --- Output directory ---
    output_dir = prepared_dir / "labeled"
    output_dir.mkdir(parents=True, exist_ok=True)

    typed_nom_dir = None
    if nom_font:
        typed_nom_dir = output_dir / "typed_nom"
        typed_nom_dir.mkdir(parents=True, exist_ok=True)

    # --- Process pages ---
    if use_ocr:
        print(f"OCR API: {_SN_DOMAIN} (enabled)")

    if verbose:
        print(f"\nLabel Characters: {prepared_dir.name}")
        print(f"Output: {output_dir}/")
        print("-" * 70)

    all_labels = []
    _all_flagged_columns = []  # Issue #2: track all flagged columns
    total_stats = {
        "total_chars": 0, "matched": 0, "gaps": 0,
        "high": 0, "medium": 0, "low": 0,
    }

    for page_info in manifest["pages"]:
        if page_filter is not None and page_info["book_page"] != page_filter:
            continue

        result = process_page(
            page_info, prepared_dir, trans_dict, similar_dict,
            nom_font, output_dir, typed_nom_dir,
            review=review, use_ocr=use_ocr, verbose=verbose,
            corpus_freq=corpus_freq, font_path=font_path,
            embed_ranker=embed_ranker,
        )

        if result is None:
            continue

        all_labels.extend(result["labels"])
        for k in ("total_chars", "matched", "gaps", "high", "medium", "low"):
            total_stats[k] += result["stats"].get(k, 0)
        # Collect flagged columns (Issue #2)
        for fc in result["stats"].get("flagged_columns", []):
            fc["page"] = result["book_page"]
            _all_flagged_columns.append(fc)

    # --- Bước 4.5: Self-Consistency (lan truyền nhãn thống kê) ---
    # Quan sát: cùng 1 từ QN xuất hiện nhiều lần trong sách.
    # Nếu "trời" = 𡗶 đã được xác nhận ≥2 lần (qua OCR hoặc 1-candidate dict)
    # → tất cả "trời" medium đều nâng lên high.
    #
    # Cải tiến: 2 lượt lan truyền
    #   Lượt 1: Dùng high confidence (dict 1-candidate + OCR confirmed)
    #   Lượt 2: Dùng OCR cross-validation (ocr_char khớp với top candidate)

    # --- Lượt 1: Thu thập cặp (qn, nom_char) truly unambiguous ---
    # Issue #4: CHỈ dùng labels mà:
    #   (a) confidence="high" VÀ chỉ có 1 candidate trong từ điển (truly unambiguous)
    #   (b) HOẶC đã được OCR API xác nhận (ocr_source=True)
    # KHÔNG dùng labels high từ anchor_refined/consistency_source vì có thể sai
    confirmed_freq: dict[str, dict[str, int]] = {}
    for lab in all_labels:
        if lab.get("confidence") != "high":
            continue
        if not (lab.get("quoc_ngu") and lab.get("nom_char")):
            continue
        # Chỉ lấy truly unambiguous sources
        qn = lab["quoc_ngu"].lower()
        candidates = trans_dict.get(qn, [])
        is_unambiguous = len(candidates) == 1
        is_ocr_confirmed = lab.get("ocr_source", False)
        if is_unambiguous or is_ocr_confirmed:
            char = lab["nom_char"]
            confirmed_freq.setdefault(qn, {})
            confirmed_freq[qn][char] = confirmed_freq[qn].get(char, 0) + 1

    # Nâng cấp medium → high nếu cùng từ QN đã được xác nhận đủ nhiều lần
    # CẢI TIẾN: tăng threshold ≥2 → ≥4 để tránh lan truyền lỗi
    # (1 lỗi ranking ban đầu nếu xuất hiện 2 lần sẽ không đủ để lan truyền)
    CONSISTENCY_MIN_COUNT = 4
    consistency_upgraded = 0
    for lab in all_labels:
        if lab.get("confidence") != "medium":
            continue
        qn = (lab.get("quoc_ngu") or "").lower()
        if qn not in confirmed_freq:
            continue
        candidates = lab.get("nom_candidates", [])
        if not candidates:
            continue

        # Tìm ký tự được xác nhận nhiều nhất cho từ QN này
        freq = confirmed_freq[qn]
        best_char = max(freq, key=freq.get)
        best_count = freq[best_char]

        # CẢI TIẾN: chỉ nâng cấp nếu xuất hiện ≥4 lần (thay vì ≥2)
        # VÀ phải chiếm >60% tổng xuất hiện (tránh trường hợp split vote)
        total_for_qn = sum(freq.values())
        dominant = best_count / total_for_qn if total_for_qn > 0 else 0

        if best_char in candidates and best_count >= CONSISTENCY_MIN_COUNT and dominant > 0.6:
            lab["nom_char"] = best_char
            lab["nom_unicode"] = f"U+{ord(best_char):04X}"
            lab["confidence"] = "high"
            lab["consistency_source"] = True
            new_cands = [best_char] + [c for c in candidates if c != best_char]
            lab["nom_candidates"] = new_cands
            consistency_upgraded += 1

    # --- Lượt 2: OCR cross-validation cho remaining medium ---
    # Nếu ocr_char trùng với nom_char hiện tại (top candidate) → upgrade
    ocr_crossval_upgraded = 0
    for lab in all_labels:
        if lab.get("confidence") != "medium":
            continue
        ocr_char = lab.get("ocr_char")
        nom_char = lab.get("nom_char")
        if ocr_char and nom_char and ocr_char == nom_char:
            lab["confidence"] = "high"
            lab["ocr_crossval"] = True
            ocr_crossval_upgraded += 1

    # --- Lượt 3: Lan truyền từ lượt 1+2 cho medium còn lại ---
    # Issue #4: Lượt 3 cũng chỉ dùng truly unambiguous + OCR confirmed
    confirmed_freq2: dict[str, dict[str, int]] = {}
    for lab in all_labels:
        if lab.get("confidence") != "high":
            continue
        if not (lab.get("quoc_ngu") and lab.get("nom_char")):
            continue
        qn = lab["quoc_ngu"].lower()
        candidates_3 = trans_dict.get(qn, [])
        is_unambiguous_3 = len(candidates_3) == 1
        is_ocr_3 = lab.get("ocr_source", False) or lab.get("ocr_crossval", False)
        if is_unambiguous_3 or is_ocr_3:
            char = lab["nom_char"]
            confirmed_freq2.setdefault(qn, {})
            confirmed_freq2[qn][char] = confirmed_freq2[qn].get(char, 0) + 1

    # CẢI TIẾN: lượt 3 cũng cần threshold cao hơn (≥3 thay vì ≥1)
    CONSISTENCY_MIN_COUNT_2 = 3
    consistency_upgraded_2 = 0
    for lab in all_labels:
        if lab.get("confidence") != "medium":
            continue
        qn = (lab.get("quoc_ngu") or "").lower()
        if qn not in confirmed_freq2:
            continue
        candidates = lab.get("nom_candidates", [])
        if not candidates:
            continue
        freq = confirmed_freq2[qn]
        best_char = max(freq, key=freq.get)
        best_count = freq[best_char]
        total_for_qn = sum(freq.values())
        dominant = best_count / total_for_qn if total_for_qn > 0 else 0

        if best_char in candidates and best_count >= CONSISTENCY_MIN_COUNT_2 and dominant > 0.6:
            lab["nom_char"] = best_char
            lab["nom_unicode"] = f"U+{ord(best_char):04X}"
            lab["confidence"] = "high"
            lab["consistency_source"] = True
            new_cands = [best_char] + [c for c in candidates if c != best_char]
            lab["nom_candidates"] = new_cands
            consistency_upgraded_2 += 1

    total_consistency = consistency_upgraded + ocr_crossval_upgraded + consistency_upgraded_2
    if verbose and total_consistency > 0:
        print(f"\n  Self-Consistency: {total_consistency} upgraded "
              f"(lượt1={consistency_upgraded}, ocr_crossval={ocr_crossval_upgraded}, "
              f"lượt2={consistency_upgraded_2})")

    # --- Phase 4: Multi-Tier Post-Processing ---
    multi_tier_postprocess(
        all_labels, trans_dict, similar_dict, verbose=verbose,
    )

    # Cập nhật lại stats sau self-consistency + post-processing
    total_stats["high"] = sum(
        1 for lab in all_labels
        if lab["type"] == "match" and lab.get("confidence") == "high"
    )
    total_stats["medium"] = sum(
        1 for lab in all_labels
        if lab["type"] == "match" and lab.get("confidence") == "medium"
    )

    # --- Summary ---
    total_m = total_stats["matched"]
    total_h = total_stats["high"]
    total_med = total_stats["medium"]
    total_l = total_stats["low"]
    total_g = total_stats["gaps"]

    if verbose:
        print("\n" + "=" * 70)
        print("TỔNG KẾT GÁN NHÃN")
        print("=" * 70)
        print(f"  Tổng ký tự detected : {total_stats['total_chars']}")
        print(f"  Matched (có nhãn)   : {total_m}")
        print(f"    High confidence   : {total_h} ({total_h/total_m*100:.1f}%)" if total_m else "")
        print(f"    Medium confidence : {total_med} ({total_med/total_m*100:.1f}%)" if total_m else "")
        print(f"    Low confidence    : {total_l} ({total_l/total_m*100:.1f}%)" if total_m else "")
        print(f"  Gaps (thừa/thiếu)   : {total_g}")

        # Đếm Unicode unique
        unique_nom = set(
            lab["nom_char"] for lab in all_labels
            if lab.get("nom_char")
        )
        print(f"  Unicode Nôm unique  : {len(unique_nom)}")
        ocr_upgraded = sum(1 for lab in all_labels if lab.get("ocr_source"))
        consist_upgraded = sum(1 for lab in all_labels if lab.get("consistency_source"))
        if ocr_upgraded:
            print(f"  OCR upgraded        : {ocr_upgraded}")
        if consist_upgraded:
            print(f"  Consistency upgraded: {consist_upgraded}")
        if _all_flagged_columns:
            print(f"\n  [WARNING] {len(_all_flagged_columns)} cột có chênh lệch "
                  f"detection/transcription >30%:")
            for fc in _all_flagged_columns[:10]:
                print(f"    Trang {fc['page']} cột {fc['column']}: "
                      f"{fc['detected_chars']} chars vs {fc['syllables']} syls "
                      f"(mismatch {fc['mismatch_ratio']:.0%})")
            if len(_all_flagged_columns) > 10:
                print(f"    ... và {len(_all_flagged_columns) - 10} cột khác")
        print(f"  Output              : {output_dir}/")

    # --- Build dataset chuẩn (đồng bộ JSON, CSV, Excel) ---
    dataset_rows = []
    for lab in all_labels:
        if lab.get("type") != "match" or not lab.get("nom_char"):
            continue
        bbox = lab.get("bbox", [])
        dataset_rows.append({
            "image": lab.get("crop_cleaned_file") or lab.get("crop_file", ""),
            "nom_char": lab.get("nom_char", ""),
            "label": lab.get("nom_unicode", ""),
            "reading": lab.get("quoc_ngu", ""),
            "confidence": lab.get("confidence", ""),
            "bbox": ",".join(str(v) for v in bbox) if bbox else "",
            "page": lab.get("page", ""),
            "source": prepared_dir.name,
        })

    # Save dataset.json
    dataset_path = output_dir / "dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset_rows, f, ensure_ascii=False, indent=1)
    if verbose:
        print(f"\n  Dataset: {dataset_path} ({len(dataset_rows)} entries)")

    # --- Save summary.json ---
    unique_nom = set(
        lab["nom_char"] for lab in all_labels if lab.get("nom_char")
    )
    summary = {
        "source": str(prepared_dir),
        "total_detected": total_stats["total_chars"],
        "total_matched": total_m,
        "total_gaps": total_g,
        "confidence_distribution": {
            "high": total_h,
            "medium": total_med,
            "low": total_l,
        },
        "unique_nom_chars": len(unique_nom),
        "total_labels": len(dataset_rows),
        "ocr_upgraded": sum(1 for lab in all_labels if lab.get("ocr_source")),
        "consistency_upgraded": sum(1 for lab in all_labels if lab.get("consistency_source")),
        "flagged_columns": _all_flagged_columns,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # --- Export labels.csv ---
    labels_csv_path = output_dir / "labels.csv"
    with open(labels_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "nom_char", "label", "reading",
                         "confidence", "bbox", "page", "source"])
        for row in dataset_rows:
            writer.writerow([
                row["image"], row["nom_char"], row["label"], row["reading"],
                row["confidence"], row["bbox"], row["page"], row["source"],
            ])
    if verbose:
        print(f"  Labels CSV: {labels_csv_path} ({len(dataset_rows)} rows)")

    # --- Export Excel ---
    if excel:
        excel_path = output_dir / f"{prepared_dir.name}_labeled.xlsx"
        export_excel(str(dataset_path), str(excel_path), str(prepared_dir))

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Gán nhãn tự động cho ký tự Nôm viết tay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python label_characters.py data/prepared/SachThanhTruyen2
  python label_characters.py data/prepared/SachThanhTruyen2 --page 12 --review
  python label_characters.py data/prepared/SachThanhTruyen2 --font FontDiffusion/fonts/NomNaTong-Regular.ttf
        """,
    )
    parser.add_argument("prepared_dir", type=str, help="Thư mục prepared data")
    parser.add_argument("--font", type=str, default=None,
                        help="Path tới font Nôm TTF (default: FontDiffusion/fonts/NomNaTong-Regular.ttf)")
    parser.add_argument("--page", type=int, default=None,
                        help="Chỉ xử lý 1 trang cụ thể")
    parser.add_argument("--review", action="store_true",
                        help="Tạo ảnh review (viết tay | đánh máy)")
    parser.add_argument("--excel", action="store_true",
                        help="Xuất file Excel với kết quả có màu")
    parser.add_argument("--ocr", action="store_true",
                        help="Dùng API OCR (tools.clc.hcmus.edu.vn) để cải thiện độ chính xác Unicode")
    parser.add_argument("--dinov2", action="store_true",
                        help="Dùng DINOv2 foundation model để ranking (không cần training)")
    parser.add_argument("--embedding", type=str, default=None,
                        help="Path tới embedding checkpoint (best.pt) để dùng deep ranking")
    parser.add_argument("--gallery", type=str, default=None,
                        help="Path tới gallery directory (embedding/data/gallery)")

    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    if not prepared_dir.exists():
        print(f"[ERROR] Không tìm thấy: {prepared_dir}", file=sys.stderr)
        sys.exit(1)

    process_prepared_dir(
        prepared_dir=prepared_dir,
        font_path=args.font,
        page_filter=args.page,
        review=args.review,
        excel=args.excel,
        use_ocr=args.ocr,
        use_dinov2=args.dinov2,
        embedding_checkpoint=args.embedding,
        embedding_gallery=args.gallery,
    )


if __name__ == "__main__":
    main()
