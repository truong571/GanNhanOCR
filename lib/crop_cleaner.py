"""Sauvola binarization + morphological cleanup for character crops."""

from pathlib import Path

import cv2
import numpy as np


class CharacterCleaner:
    """Clean handwritten Nom character crops from old books.

    Pipeline: Denoise -> Background norm -> Sauvola binarize -> Morph cleanup
              -> CC noise removal -> Stroke normalization -> Center+Resize 64x64
    """

    def __init__(
        self,
        target_size: int = 64,
        padding: int = 5,
        sauvola_k: float = 0.2,
        sauvola_window: int = 25,
        sauvola_R: float = 128.0,
        denoise_strength: int = 3,
        min_stroke: int = 2,
    ):
        self.target_size = target_size
        self.padding = padding
        self.sauvola_k = sauvola_k
        self.sauvola_window = sauvola_window | 1
        self.sauvola_R = sauvola_R
        self.denoise_strength = denoise_strength | 1
        self.min_stroke = min_stroke

    def _normalize_background(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        k_size = max(15, min(51, max(h, w) // 3)) | 1
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, bg_kernel)
        normalized = cv2.divide(gray, background, scale=255)
        p_low, p_high = np.percentile(normalized, (2, 98))
        if p_high > p_low:
            normalized = np.clip(
                (normalized.astype(np.float32) - p_low) * 255 / (p_high - p_low),
                0, 255,
            ).astype(np.uint8)
        return normalized

    def _compute_adaptive_window(self, gray: np.ndarray) -> int:
        h, w = gray.shape
        adaptive_w = max(15, min(51, min(h, w) // 3)) | 1
        return max(adaptive_w, self.sauvola_window)

    def _sauvola_binarize(self, gray: np.ndarray, window: int | None = None) -> np.ndarray:
        """Sauvola binarization: T(x,y) = mean * [1 + k * (std/R - 1)]"""
        w = window if window is not None else self.sauvola_window
        k = self.sauvola_k
        R = self.sauvola_R

        gray_f = gray.astype(np.float64)
        mean = cv2.boxFilter(gray_f, -1, (w, w))
        sqmean = cv2.boxFilter(gray_f ** 2, -1, (w, w))
        variance = np.maximum(sqmean - mean ** 2, 0)
        std = np.sqrt(variance)

        threshold = mean * (1.0 + k * (std / R - 1.0))
        binary = np.zeros_like(gray)
        binary[gray_f < threshold] = 255
        return binary

    def _morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, open_k)
        return result

    def _remove_noise_components(self, binary: np.ndarray) -> np.ndarray:
        h, w = binary.shape
        min_area = max(10, int(h * w * 0.005))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 255
        if np.sum(cleaned) == 0:
            return binary
        return cleaned

    def _normalize_stroke(self, binary: np.ndarray) -> np.ndarray:
        if np.sum(binary) == 0:
            return binary
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        ink_dist = dist[binary > 0]
        if len(ink_dist) == 0:
            return binary
        avg_thickness = np.mean(ink_dist) * 2
        if avg_thickness < self.min_stroke:
            k_size = max(2, int(self.min_stroke - avg_thickness) + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            binary = cv2.dilate(binary, kernel, iterations=1)
        elif avg_thickness > self.min_stroke * 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            binary = cv2.erode(binary, kernel, iterations=1)
        return binary

    def _center_and_resize(self, binary: np.ndarray) -> np.ndarray | None:
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        if w == 0 or h == 0:
            return None
        char_crop = binary[y:y + h, x:x + w]
        max_dim = self.target_size - (self.padding * 2)
        ratio = min(max_dim / w, max_dim / h)
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(char_crop, (new_w, new_h), interpolation=interp)
        canvas = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        x_off = (self.target_size - new_w) // 2
        y_off = (self.target_size - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return cv2.bitwise_not(canvas)

    def clean(self, image_path_or_array) -> tuple:
        """Full cleaning pipeline. Returns (cleaned_image, debug_info)."""
        debug_info: dict = {}

        if isinstance(image_path_or_array, (str, Path)):
            gray = cv2.imread(str(image_path_or_array), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return None, {"error": f"Cannot load: {image_path_or_array}"}
        else:
            gray = image_path_or_array
            if len(gray.shape) == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        debug_info["original_shape"] = gray.shape

        if self.denoise_strength > 1:
            denoised = cv2.medianBlur(gray, self.denoise_strength)
        else:
            denoised = gray

        normalized = self._normalize_background(denoised)
        adaptive_w = self._compute_adaptive_window(normalized)
        binary = self._sauvola_binarize(normalized, window=adaptive_w)
        fg_ratio = np.sum(binary > 0) / binary.size

        if fg_ratio < 0.01 or fg_ratio > 0.60:
            _, otsu_binary = cv2.threshold(
                normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            otsu_fg = np.sum(otsu_binary > 0) / otsu_binary.size
            if 0.01 <= otsu_fg <= 0.60:
                binary = otsu_binary
                debug_info["fallback"] = "otsu"

        binary = self._morphological_cleanup(binary)
        binary = self._remove_noise_components(binary)
        binary = self._normalize_stroke(binary)
        output = self._center_and_resize(binary)

        if output is None:
            return None, {"error": "Empty after cleaning"}

        debug_info["success"] = True
        return output, debug_info
