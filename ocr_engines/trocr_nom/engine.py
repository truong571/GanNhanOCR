"""TrOCR fine-tuned on Vietnamese Nom (tt1225 HuggingFace).

Local, no API, no rate limit. Weights auto-download on first use and
cache to `~/.cache/huggingface/`.

Variants (HF_MODEL env override):
    tt1225/finetuned-trocr-tiny-v2-vietnamese-nom   ~60 MB,  fastest
    tt1225/finetuned-trocr-small-vietnamese-nom     ~240 MB, balanced (default)
    tt1225/finetuned-trocr-base-vietnamese-nom      ~550 MB, most accurate

NOTE: The model card on HF is empty, so handwritten-vs-printed training
is not documented. Run `python -m ocr_engines.trocr_nom.probe` against
real crops to verify suitability before relying on it.

Requirements:
    pip install transformers torch pillow
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ocr_engines.base import OCREngine, RecognitionResult

_DEFAULT_MODEL = "tt1225/finetuned-trocr-small-vietnamese-nom"


class TrOCRNomEngine(OCREngine):
    name = "trocr_nom"

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
    ):
        self.model_id = model_id or os.environ.get("TROCR_NOM_MODEL", _DEFAULT_MODEL)
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            import torch  # noqa: F401
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as e:
            print(
                "[trocr_nom] Missing deps. "
                "Install with: pip install transformers torch pillow",
                file=sys.stderr,
            )
            raise e

        self._processor = _load_processor_compat(self.model_id, TrOCRProcessor)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()

    def recognize_crop(
        self,
        crop_path: str,
        context: dict | None = None,
    ) -> RecognitionResult:
        self._ensure_loaded()

        try:
            import torch
            from PIL import Image
        except ImportError:
            return RecognitionResult(char=None, confidence=0.0)

        try:
            img = Image.open(crop_path).convert("RGB")
        except Exception as e:
            print(f"[trocr_nom] open error: {e}", file=sys.stderr)
            return RecognitionResult(char=None, confidence=0.0)

        try:
            pixel_values = self._processor(
                images=img, return_tensors="pt"
            ).pixel_values.to(self.device)
            gen_kwargs = {
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            # Override whichever length knob the checkpoint's generation_config
            # already sets — prefer max_new_tokens so single-char crops stay
            # short. Checking with `getattr` avoids the "both set" warning.
            gen_cfg = getattr(self._model, "generation_config", None)
            if gen_cfg is not None and getattr(gen_cfg, "max_length", None):
                gen_kwargs["max_length"] = 8
            else:
                gen_kwargs["max_new_tokens"] = 8
            with torch.no_grad():
                out = self._model.generate(pixel_values, **gen_kwargs)
            ids = out.sequences
            text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
            confidence = _mean_token_prob(out)
        except Exception as e:
            print(f"[trocr_nom] generate error: {e}", file=sys.stderr)
            return RecognitionResult(char=None, confidence=0.0)

        char = _first_cjk_char(text)
        if char is None:
            return RecognitionResult(char=None, confidence=0.0)
        return RecognitionResult(char=char, confidence=confidence)


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_processor_compat(model_id: str, processor_cls):
    try:
        return processor_cls.from_pretrained(model_id)
    except ValueError as e:
        # Some legacy TrOCR checkpoints still advertise deprecated
        # feature extractor names that recent transformers releases no
        # longer auto-upgrade correctly.
        if "Unrecognized image processor" not in str(e):
            raise
        return _load_legacy_trocr_processor(model_id, processor_cls)


def _load_legacy_trocr_processor(model_id: str, processor_cls):
    from huggingface_hub import hf_hub_download
    from transformers import (
        AutoTokenizer,
        DeiTImageProcessor,
        ViTImageProcessor,
    )

    cfg_path = hf_hub_download(repo_id=model_id, filename="preprocessor_config.json")
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))

    image_processor_type = (
        cfg.get("image_processor_type") or cfg.get("feature_extractor_type") or ""
    )
    processor_map = {
        "ViTFeatureExtractor": ViTImageProcessor,
        "DeiTFeatureExtractor": DeiTImageProcessor,
        "ViTImageProcessor": ViTImageProcessor,
        "DeiTImageProcessor": DeiTImageProcessor,
    }

    image_processor_cls = processor_map.get(image_processor_type)
    if image_processor_cls is None:
        lowered = image_processor_type.lower()
        if "deit" in lowered:
            image_processor_cls = DeiTImageProcessor
        else:
            image_processor_cls = ViTImageProcessor

    image_processor = image_processor_cls.from_dict(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    return processor_cls(image_processor=image_processor, tokenizer=tokenizer)


def _mean_token_prob(generate_output) -> float:
    """Average softmax prob of the generated tokens as a cheap confidence."""
    try:
        import torch
        if not getattr(generate_output, "scores", None):
            return 0.9
        ids = generate_output.sequences[0][-len(generate_output.scores):]
        probs = []
        for step, logits in enumerate(generate_output.scores):
            p = torch.softmax(logits[0], dim=-1)[ids[step]].item()
            probs.append(p)
        if not probs:
            return 0.9
        return float(sum(probs) / len(probs))
    except Exception:
        return 0.9


def _first_cjk_char(text: str) -> str | None:
    for ch in text:
        cp = ord(ch)
        if (
            0x3400 <= cp <= 0x4DBF
            or 0x4E00 <= cp <= 0x9FFF
            or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2EBEF
            or 0x2F800 <= cp <= 0x2FA1F
            or 0x2E80 <= cp <= 0x2EFF
            or 0x2F00 <= cp <= 0x2FDF
        ):
            return ch
    return None
