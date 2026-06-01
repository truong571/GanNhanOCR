"""Pluggable "second OCR" backends for cross-OCR consensus (compare vs classifier).

The project's S1 is the HCMUS SinoNom OCR (core/ocr/ocr_api.py). A SECOND,
INDEPENDENT reader of the same crop lets us confirm/correct a label by consensus
— an alternative to training a visual classifier.

Landscape (why these):
  - No ready-made *chữ-Nôm* OCR exists besides HCMUS (=S1). Generic CJK OCRs
    (PaddleOCR / EasyOCR / cnocr / Tesseract chi_tra) are trained on MODERN
    Chinese and mis-read Nôm-specific Ext-B glyphs -> weak on the hard cases.
  - A vision-LLM reads arbitrary characters zero-shot and is "call-and-run".
    GEMINI_API_KEY is present in .env, so GeminiOCR is the ready backend.

Backends implement: pick(png_bytes, candidates) -> chosen char (or "") and
read(png_bytes) -> free char. GeminiOCR via REST (uses `requests`, no SDK dep).
"""
from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path

import requests

_REPO = Path(__file__).resolve().parent.parent.parent.parent


def _load_env_key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    env = _REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if k.strip() == name:
                    return v.strip().strip("'\"")
    return ""


class GeminiOCR:
    """Vision-LLM second reader (Google Gemini). Forced-choice + free read."""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        self.model = model
        self.key = api_key or _load_env_key("GEMINI_API_KEY")
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY not found (env or .env)")

    def _call(self, prompt: str, png_bytes: bytes, max_retry: int = 4) -> str:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.key}")
        body = {"contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png",
                             "data": base64.b64encode(png_bytes).decode()}},
        ]}], "generationConfig": {"temperature": 0, "maxOutputTokens": 16}}
        for attempt in range(max_retry):
            r = requests.post(url, json=body, timeout=40)
            if r.status_code == 429:                 # free-tier rate limit -> backoff
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code != 200:                 # redact key from any error
                raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
            try:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError):
                return ""
        raise RuntimeError("Gemini 429 rate-limited after retries (free-tier quota)")

    def pick(self, png_bytes: bytes, candidates: list[str]) -> str:
        """Forced choice: which candidate char matches the crop? '' if none."""
        cand = " ".join(candidates)
        prompt = (
            "Ảnh dưới là MỘT ký tự chữ Hán-Nôm cắt từ bản khắc mộc bản (có thể mờ, "
            "nhoè, nét đậm). Trong các ứng viên sau, ký tự nào KHỚP NHẤT với ảnh?\n"
            f"Ứng viên: {cand}\n"
            "Chỉ in DUY NHẤT một ký tự ứng viên khớp nhất. Nếu không ký tự nào khớp, in: none"
        )
        out = self._call(prompt, png_bytes)
        for ch in candidates:                      # robust parse: first candidate present
            if ch in out:
                return ch
        return ""

    def read(self, png_bytes: bytes) -> str:
        """Open read: what single character is this? '' if unsure."""
        prompt = ("Ảnh là MỘT ký tự chữ Hán-Nôm khắc mộc bản. In DUY NHẤT một ký tự "
                  "Hán/Nôm bạn đọc được, không giải thích.")
        out = self._call(prompt, png_bytes)
        m = re.search(r"[㐀-鿿豈-﫿\U00020000-\U0002ebef]", out)
        return m.group(0) if m else ""


# Optional local backend (modern-Chinese-trained -> weak on Nôm; kept for fairness).
class TesseractOCR:
    def __init__(self, lang: str = "chi_tra"):
        import pytesseract  # noqa: F401  (raises if not installed)
        self.lang = lang

    def read(self, png_bytes: bytes) -> str:
        import io
        import pytesseract
        from PIL import Image
        txt = pytesseract.image_to_string(Image.open(io.BytesIO(png_bytes)),
                                          lang=self.lang, config="--psm 10")
        m = re.search(r"[㐀-鿿豈-﫿]", txt)
        return m.group(0) if m else ""

    def pick(self, png_bytes: bytes, candidates: list[str]) -> str:
        r = self.read(png_bytes)
        return r if r in candidates else ""


def get_backend(name: str = "gemini", **kw):
    return {"gemini": GeminiOCR, "tesseract": TesseractOCR}[name](**kw)
