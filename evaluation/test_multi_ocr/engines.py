"""Engine wrappers — pluggable OCR backends for per-crop char recognition.

Each engine: input = path to crop image, output = (char_str, confidence).

Engines:
  - kimhannom_cached : read from existing pipeline output (no compute, baseline)
  - tesseract_chi    : pytesseract with chi_tra / chi_sim language
  - tesseract_chi_vert: vertical Chinese (for column orientation)

Future: paddleocr_chinese_hw (when paddleocr install works), or
        a Vision-LLM wrapper (gradio_client to a HF Space).
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image


def _strip_to_one_cjk(s: str) -> str:
    """Tesseract often returns 多 chars + spaces. Pick first CJK char."""
    s = (s or "").strip()
    for ch in s:
        cp = ord(ch)
        if (0x3400 <= cp <= 0x9FFF or 0x20000 <= cp <= 0x2FFFF
                or 0xF900 <= cp <= 0xFAFF):
            return ch
    return ""


class TesseractEngine:
    """pytesseract with configurable language. ~100ms/crop CPU."""

    def __init__(self, lang: str = "chi_tra", psm: int = 10):
        self.lang = lang
        self.psm = psm  # 10 = single character; 8 = single word
        self.name = f"tesseract-{lang}"
        try:
            import pytesseract
            self.pyt = pytesseract
        except ImportError:
            raise RuntimeError("pip install pytesseract")

    def recognize(self, crop_path: str) -> tuple[str, float]:
        img = Image.open(crop_path)
        try:
            text = self.pyt.image_to_string(
                img, lang=self.lang,
                config=f"--psm {self.psm}",
            )
        except Exception:
            return "", 0.0
        ch = _strip_to_one_cjk(text)
        # Tesseract psm=10 doesn't give conf, use confidence_data
        try:
            data = self.pyt.image_to_data(
                img, lang=self.lang,
                config=f"--psm {self.psm}",
                output_type=self.pyt.Output.DICT,
            )
            confs = [int(c) for c in data.get("conf", []) if int(c) >= 0]
            conf = max(confs) / 100.0 if confs else 0.0
        except Exception:
            conf = 0.0
        return ch, conf


class KimhannomCachedEngine:
    """Baseline — reads pre-computed Kimhannom OCR char from sample metadata.

    Doesn't actually call the API; assumes you packaged the kimhannom_ch
    field into the sample's metadata dict (see sample_crops.py).
    """

    name = "kimhannom-cached"

    def recognize(self, crop_path: str, *, kimhannom_ch: str = "",
                  **_) -> tuple[str, float]:
        return kimhannom_ch, 1.0 if kimhannom_ch else 0.0


class PaddleOCRChineseEngine:
    """PaddleOCR PP-OCRv5 Chinese model. Lazy-init (200MB download first time).

    Output is text + score per detected line. Since we feed a SINGLE CHAR
    crop, we expect 1 detection. Take the first rec_texts[0].
    """

    def __init__(self, lang: str = "ch"):
        self.lang = lang
        self.name = f"paddleocr-{lang}"
        self._ocr = None

    def _ensure_ocr(self):
        if self._ocr is not None:
            return
        # Suppress paddle's verbose startup logs
        import os, logging, contextlib
        os.environ.setdefault("FLAGS_call_stack_level", "0")
        from paddleocr import PaddleOCR
        # Inference-only flags — skip doc preprocessing for speed on tiny crops
        self._ocr = PaddleOCR(use_angle_cls=False, lang=self.lang)

    def recognize(self, crop_path: str, **_) -> tuple[str, float]:
        self._ensure_ocr()
        try:
            results = self._ocr.ocr(crop_path)
            if not results:
                return "", 0.0
            first = results[0]
            texts = first.get("rec_texts") or []
            scores = first.get("rec_scores") or []
            if not texts:
                return "", 0.0
            ch = _strip_to_one_cjk(texts[0])
            conf = float(scores[0]) if scores else 0.0
            return ch, conf
        except Exception as e:
            return "", 0.0


class GeminiVisionEngine:
    """Free Gemini Vision — multimodal LLM understands Han-Nom from web training.

    Uses GEMINI_API_KEY from .env (you already set this up for llm_postfix).
    Free tier 1500 req/day — 200 crops fits easily.
    Rate-limited to 15 req/min on free tier (~13 min for 200).
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.name = f"gemini-{model_name}"
        self.model_name = model_name
        self._model = None  # lazy init
        self._configured = False
        # Throttle to stay under 15 req/min
        self._last_call_time = 0.0
        self._min_interval = 4.1  # seconds between calls

    def _ensure_model(self):
        if self._model is not None:
            return
        import os
        # Load API key from .env (handle quotes)
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip().strip("'").strip('"')
                    os.environ.setdefault(k.strip(), v)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY in .env")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.model_name)

    PROMPT = (
        "Đây là 1 chữ Hán hoặc chữ Nôm Việt Nam viết tay trong sách Công giáo "
        "thế kỷ 17-19. Trả về CHÍNH XÁC 1 ký tự Unicode CJK duy nhất, không "
        "kèm giải thích, không kèm dấu nháy. Nếu không đọc được, trả về '?'."
    )

    def recognize(self, crop_path: str, **_) -> tuple[str, float]:
        self._ensure_model()
        import time as _t
        elapsed_since = _t.time() - self._last_call_time
        if elapsed_since < self._min_interval:
            _t.sleep(self._min_interval - elapsed_since)
        try:
            img = Image.open(crop_path)
            resp = self._model.generate_content([self.PROMPT, img])
            self._last_call_time = _t.time()
            text = (resp.text or "").strip()
            ch = _strip_to_one_cjk(text)
            return ch, 1.0 if ch else 0.0
        except Exception as e:
            self._last_call_time = _t.time()
            print(f"  [gemini] {Path(crop_path).name}: {type(e).__name__}: {str(e)[:60]}")
            return "", 0.0


# Registry helper. Subset registries by --engines flag.
ENGINES = {
    "kimhannom":          KimhannomCachedEngine(),
    "paddleocr_ch":       PaddleOCRChineseEngine(lang="ch"),
    "tesseract_chi_tra":  TesseractEngine(lang="chi_tra"),
    "tesseract_chi_sim":  TesseractEngine(lang="chi_sim"),
    "gemini_flash":       GeminiVisionEngine("gemini-2.5-flash"),
}
