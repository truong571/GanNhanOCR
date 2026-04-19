"""PaddleOCRv5 fine-tuned on Sino-Vietnamese (Han-Nom) manuscripts.

Weights come from MinhDS's Hugging Face Space:
    https://huggingface.co/spaces/MinhDS/Fine-tuned-PaddleOCRv5

Setup (one-time):
    python -m ocr_engines.paddleocrv5_nom.download_weights

This downloads the inference model package and extracts it to
`ocr_engines/paddleocrv5_nom/weights/`. The folder is gitignored.

Requirements (install on first use):
    pip install paddleocr paddlepaddle

This engine is RECOGNITION-only. Kimhannom still provides bboxes/crops
upstream; we just re-recognize each crop as an independent verifier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from ocr_engines.base import OCREngine, RecognitionResult

_WEIGHTS_DIR = Path(__file__).parent / "weights"


class PaddleOCRv5NomEngine(OCREngine):
    name = "paddleocrv5_nom"

    def __init__(
        self,
        weights_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        self.weights_dir = Path(weights_dir) if weights_dir else _WEIGHTS_DIR
        self.device = device
        self._predictor = None

    def _ensure_loaded(self):
        if self._predictor is not None:
            return

        if not self.weights_dir.exists() or not any(self.weights_dir.iterdir()):
            raise RuntimeError(
                f"PaddleOCRv5 weights not found at {self.weights_dir}. "
                "Run: python -m ocr_engines.paddleocrv5_nom.download_weights"
            )

        try:
            # PaddleOCR >= 3.x exposes the recognition predictor directly.
            from paddleocr import TextRecognition
        except ImportError as e:
            print(
                "[paddleocrv5_nom] Missing dependencies. "
                "Install with: pip install paddleocr paddlepaddle",
                file=sys.stderr,
            )
            raise e

        self._predictor = TextRecognition(model_dir=str(self.weights_dir))

    def recognize_crop(
        self,
        crop_path: str,
        context: dict | None = None,
    ) -> RecognitionResult:
        self._ensure_loaded()

        img = cv2.imread(crop_path, cv2.IMREAD_COLOR)
        if img is None:
            return RecognitionResult(char=None, confidence=0.0)

        # Single-character crops are small; upscale to ~64px height so
        # the CTC decoder has enough spatial context.
        h, w = img.shape[:2]
        if h < 48:
            scale = 48.0 / h
            img = cv2.resize(img, (int(w * scale), 48), cv2.INTER_CUBIC)

        try:
            result = self._predictor.predict(img)
        except Exception as e:
            print(f"[paddleocrv5_nom] predict error: {e}", file=sys.stderr)
            return RecognitionResult(char=None, confidence=0.0)

        text, score = _extract_top1(result)
        if not text:
            return RecognitionResult(char=None, confidence=0.0)

        # Keep only the first printable non-space char; crops are single-char.
        for ch in text:
            if ch.strip():
                return RecognitionResult(char=ch, confidence=float(score))
        return RecognitionResult(char=None, confidence=0.0)


def _extract_top1(result) -> tuple[str, float]:
    """Normalize the varying PaddleOCR return shapes into (text, score)."""
    if result is None:
        return "", 0.0

    # Newer PaddleOCR returns a list of dicts or a single dict
    if isinstance(result, list) and result:
        result = result[0]

    if isinstance(result, dict):
        text = result.get("rec_text") or result.get("text") or ""
        score = result.get("rec_score") or result.get("score") or 0.0
        try:
            score = float(np.asarray(score).item())
        except Exception:
            pass
        return str(text), float(score)

    # Legacy tuple: (text, score)
    if isinstance(result, tuple) and len(result) >= 2:
        return str(result[0]), float(result[1])

    return "", 0.0
