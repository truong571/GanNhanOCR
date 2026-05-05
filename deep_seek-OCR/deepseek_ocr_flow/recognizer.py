"""DeepSeek-OCR Transformers wrapper.

The official model exposes a custom `model.infer(...)` method when loaded with
`trust_remote_code=True`. This wrapper keeps all model-specific behavior in one
place and returns a normalized single-character result for GanNhanOCR crops.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


HAN_RANGES = (
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0xF900, 0xFAFF),    # Compatibility Ideographs
    (0x20000, 0x2EBEF),  # CJK Extensions B-I
    (0xE000, 0xF8FF),    # Private Use Area, present in Nom datasets
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)


@dataclass
class DeepSeekOCRConfig:
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    device: str = "auto"
    dtype: str = "auto"
    attention: str = "auto"
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True
    test_compress: bool = True
    save_results: bool = False
    crop_canvas: int = 384
    crop_prompt: str = "<image>\nFree OCR."
    page_prompt: str = "<image>\n<|grounding|>OCR this image."


class DeepSeekOCRRecognizer:
    def __init__(self, config: DeepSeekOCRConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.image_dir = work_dir / "_model_inputs"
        self.raw_dir = work_dir / "_raw_model_outputs"
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    def recognize_crop(self, crop_path: Path, cache_key: str) -> dict:
        prepared = self._prepare_crop(crop_path, cache_key)
        raw_text = self.infer_image(prepared, prompt=self.config.crop_prompt)
        char = extract_first_hannom_char(raw_text)
        return {
            "char": char,
            "confidence": 1.0 if char else 0.0,
            "raw_text": normalize_space(raw_text),
            "prepared_image": str(prepared),
        }

    def recognize_page(self, page_image: Path, page_name: str) -> str:
        out_text = self.infer_image(
            page_image,
            prompt=self.config.page_prompt,
            output_name=f"page_{page_name}",
        )
        return normalize_space(out_text)

    def infer_image(
        self,
        image_path: Path,
        prompt: str,
        output_name: str | None = None,
    ) -> str:
        self._ensure_loaded()
        output_name = output_name or image_path.stem
        output_dir = self.raw_dir / safe_name(output_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()

        result = self._model.infer(
            self._tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=str(output_dir),
            base_size=self.config.base_size,
            image_size=self.config.image_size,
            crop_mode=self.config.crop_mode,
            save_results=self.config.save_results,
            test_compress=self.config.test_compress,
        )
        text = stringify_result(result)
        if text:
            return text
        return read_new_text_output(output_dir, started)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing DeepSeek-OCR dependencies. Install torch and transformers "
                "in the active environment first."
            ) from exc

        self._torch = torch
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "use_safetensors": True,
        }
        attention = self._resolve_attention()
        if attention:
            kwargs["_attn_implementation"] = attention

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            self._model = AutoModel.from_pretrained(self.config.model_name, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load {self.config.model_name}. Make sure the model is "
                "downloaded in the Hugging Face cache or allow network access for "
                "the first run. DeepSeek-OCR is large and the official examples "
                "target CUDA, so CPU-only inference may be impractically slow. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        self._model = self._model.eval()

        dtype = self._resolve_dtype()
        if self.device != "cpu":
            self._model = self._model.to(self.device)
        if dtype is not None:
            self._model = self._model.to(dtype)

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _resolve_dtype(self):
        if self._torch is None:
            return None
        if self.config.dtype == "auto":
            if self.device == "cuda":
                return self._torch.bfloat16
            return None
        if self.config.dtype in {"none", "float32", "fp32"}:
            return None
        if self.config.dtype in {"bfloat16", "bf16"}:
            return self._torch.bfloat16
        if self.config.dtype in {"float16", "fp16"}:
            return self._torch.float16
        raise ValueError(f"Unknown dtype: {self.config.dtype}")

    def _resolve_attention(self) -> str | None:
        if self.config.attention == "default":
            return None
        if self.config.attention != "auto":
            return self.config.attention
        return "flash_attention_2" if self.device == "cuda" else "eager"

    def _prepare_crop(self, crop_path: Path, cache_key: str) -> Path:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.image_dir / f"{safe_name(cache_key)}.png"
        if out_path.exists():
            return out_path

        img = Image.open(crop_path).convert("L")
        img = ImageOps.autocontrast(img)
        canvas_size = int(self.config.crop_canvas)
        canvas = Image.new("RGB", (canvas_size, canvas_size), "white")

        max_side = max(img.size) or 1
        target = int(canvas_size * 0.78)
        scale = target / max_side
        new_size = (
            max(1, int(img.width * scale)),
            max(1, int(img.height * scale)),
        )
        img = img.resize(new_size, Image.Resampling.LANCZOS).convert("RGB")
        x = (canvas_size - img.width) // 2
        y = (canvas_size - img.height) // 2
        canvas.paste(img, (x, y))
        canvas.save(out_path)
        return out_path


def stable_key(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", value)
    return value.strip("._") or "item"


def stringify_result(result: object) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        return "\n".join(stringify_result(x) for x in result if x is not None)
    if isinstance(result, dict):
        for key in ("text", "result", "content", "markdown"):
            if key in result:
                return stringify_result(result[key])
        return "\n".join(f"{k}: {stringify_result(v)}" for k, v in result.items())
    return str(result)


def read_new_text_output(output_dir: Path, started: float) -> str:
    candidates: list[Path] = []
    for suffix in ("*.txt", "*.md", "*.markdown", "*.json"):
        candidates.extend(output_dir.rglob(suffix))
    candidates = [
        p for p in candidates
        if p.is_file() and p.stat().st_mtime >= started - 1
    ]
    if not candidates:
        return ""
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        return newest.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return newest.read_text(errors="ignore")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_first_hannom_char(text: str) -> str | None:
    text = clean_model_text(text)
    for ch in text:
        code = ord(ch)
        if any(start <= code <= end for start, end in HAN_RANGES):
            return ch
    for ch in text:
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if category[0] in {"L", "N"}:
            return ch
    return None


def clean_model_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("```", " ")
    return normalize_space(text)
