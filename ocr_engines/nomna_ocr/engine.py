"""NomNaOCR engine: column-level CRNN recognition on Han-Nom manuscripts.

Uses the CRNN × CTC model from ds4v/NomNaOCR (weights + vocab pulled
from ds4v/NomNaSite). Confirmed handwritten-trained.

This engine operates at COLUMN level (not per-crop): it crops the whole
vertical column from the original page and runs CRNN once, producing a
sequence of N characters which we map back to each char_idx in the
column. That matches how the model was trained.

Setup (one-time):
    pip install tensorflow        # any 2.x; 2.16+ is fine (Keras 3 supported)
    python -m ocr_engines.nomna_ocr.download_assets

Assets land in `ocr_engines/nomna_ocr/assets/` (gitignored).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from ocr_engines.base import OCREngine, RecognitionResult

_ASSETS_DIR = Path(__file__).parent / "assets"
_VOCAB_FILE = _ASSETS_DIR / "vocab.txt"
_WEIGHTS_FILE = _ASSETS_DIR / "CRNN.h5"


class NomNaOCREngine(OCREngine):
    name = "nomna_ocr"

    def __init__(
        self,
        vocab_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        pages_subdir: str = "pages",
    ):
        self.vocab_path = Path(vocab_path) if vocab_path else _VOCAB_FILE
        self.weights_path = Path(weights_path) if weights_path else _WEIGHTS_FILE
        self.pages_subdir = pages_subdir
        self._crnn = None

    def _ensure_loaded(self):
        if self._crnn is not None:
            return

        if not self.vocab_path.exists() or not self.weights_path.exists():
            raise RuntimeError(
                f"NomNaOCR assets missing. Expected:\n"
                f"  {self.vocab_path}\n  {self.weights_path}\n"
                "Run: python -m ocr_engines.nomna_ocr.download_assets"
            )

        try:
            # Import TF lazily — it is a heavy dependency and may not be
            # installed in environments that only use the other engines.
            import tensorflow as tf  # noqa: F401
        except ImportError as e:
            print(
                "[nomna_ocr] TensorFlow not installed. "
                "Install with: pip install tensorflow",
                file=sys.stderr,
            )
            raise e

        from ocr_engines.nomna_ocr.model.crnn import CRNN

        self._crnn = CRNN(vocab_path=self.vocab_path)
        # Force a dummy build so load_weights can align by layer names.
        import numpy as _np
        self._crnn.model.predict(_np.zeros((1, 432, 48, 3), dtype=_np.float32),
                                 verbose=0)
        self._crnn.model.load_weights(str(self.weights_path))

    # The base `recognize_crop` contract does not apply — NomNaOCR needs
    # a whole column strip, not an isolated character. Keep it here for
    # interface compliance but return empty.
    def recognize_crop(self, crop_path, context=None):
        return RecognitionResult(char=None, confidence=0.0)

    def recognize_page(
        self,
        detection: dict,
        crops_base: Path,
        cache_path: Path | None = None,
        verbose: bool = False,
    ) -> dict:
        if cache_path and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        self._ensure_loaded()

        page_name = detection.get("book_page") or detection.get("page") or "unknown"
        out = {"page": page_name, "engine": self.name, "columns": []}

        page_img = self._load_page_image(crops_base, page_name)
        if page_img is None:
            # Without the original page image we cannot crop columns;
            # return empty structure so consensus still works from other
            # engines.
            print(f"[nomna_ocr] page image missing for {page_name}",
                  file=sys.stderr)
            for col in detection.get("columns", []):
                out["columns"].append({
                    "column": col["column"],
                    "chars": [
                        {"char_idx": ch["char_idx"], "char": None, "confidence": 0.0}
                        for ch in col.get("chars", [])
                    ],
                })
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
            return out

        for col in detection.get("columns", []):
            chars = col.get("chars", [])
            col_out = {"column": col["column"], "chars": []}
            if not chars:
                out["columns"].append(col_out)
                continue

            strip = self._crop_column(page_img, chars)
            sequence = "" if strip is None else self._predict(strip)

            if verbose:
                print(f"    [nomna_ocr] col{col['column']:02d}: "
                      f"{len(chars)} expected, {len(sequence)} decoded "
                      f"→ {sequence!r}")

            for i, ch in enumerate(chars):
                predicted = sequence[i] if i < len(sequence) else None
                col_out["chars"].append({
                    "char_idx": ch["char_idx"],
                    "char": predicted,
                    "confidence": 0.85 if predicted else 0.0,
                })
            out["columns"].append(col_out)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_page_image(self, crops_base: Path, page_name: str):
        """Original page is stored one level up from `detected/`."""
        book_dir = crops_base.parent if crops_base.name == "detected" else crops_base
        page_path = book_dir / self.pages_subdir / f"{page_name}.png"
        if not page_path.exists():
            return None
        img = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _crop_column(self, page_img, chars: list[dict]):
        """Crop the vertical column enclosing all chars.

        Kimhannom returns tight character bboxes, but NomNaOCR was
        trained on column-level patches from DBNet which include some
        whitespace around the text. We expand the crop horizontally by
        ~30% of the column width (half on each side) so the aspect ratio
        is closer to the 432:48 (9:1) training target; vertical margins
        stay small since kimhannom already spans the full column top to
        bottom.
        """
        xs = [b for ch in chars for b in (ch["bbox"][0], ch["bbox"][2])]
        ys = [b for ch in chars for b in (ch["bbox"][1], ch["bbox"][3])]
        if not xs or not ys:
            return None

        h, w = page_img.shape[:2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        col_w = x_max - x_min
        pad_x = max(6, int(col_w * 0.05))
        x1 = max(0, x_min - pad_x)
        x2 = min(w, x_max + pad_x)
        y1 = max(0, y_min - 6)
        y2 = min(h, y_max + 6)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = page_img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _predict(self, strip_img: np.ndarray) -> str:
        try:
            return self._crnn.predict_one_patch(strip_img) or ""
        except Exception as e:
            print(f"[nomna_ocr] predict error: {e}", file=sys.stderr)
            return ""
