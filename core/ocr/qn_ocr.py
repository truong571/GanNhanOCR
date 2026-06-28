"""QN text recognition using VietOCR + deep / deskew-aware line detection.

Line detection is delegated to core.ocr.line_detector (DBNet via PaddleOCR when
installed, otherwise a skew-estimating projection fallback). The old raw
horizontal-projection profile remains available there as the ``projection``
backend for A/B comparison.

Used by pipeline/step1_extract.py when book has `reocr: true`.

Two-pass decoding (one VietOCR instance, toggled per call):
  1. beamsearch=True  -> better text (built-in LM picks best path)
  2. beamsearch=False, return_prob=True  -> per-line confidence

If beam-decoded and greedy-decoded texts disagree, the confidence is capped
(disagreement = ambiguous line). Greedy probability stands in as the
confidence signal even though we publish the beam text.

Verified in evaluation/test_qn_ocr/ (POC + 143-page stress test on
SachThanhTruyen11): dict-hit 55% -> 99% vs PyMuPDF text, ~3-5 s/page CPU
(beam roughly doubles greedy time).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
from PIL import Image

from core.ocr.line_detector import detect_line_crops, resolve_backend

_PREDICTOR = None

# Bump if changing decoder logic / model — invalidates all cached results. The
# resolved line-detector backend is appended at runtime so switching detectors
# (projection → deskew/DBNet) also invalidates stale caches.
CACHE_VERSION = "vgg_transformer-2pass-v2"


def _cache_version(backend: str) -> str:
    return f"{CACHE_VERSION}-{resolve_backend(backend)}"

# Below this greedy probability a line is flagged as low-confidence downstream.
# Calibrated empirically: VietOCR on clean print typically gives 0.85-0.99;
# anything <0.65 tends to have a real recognition error.
LOW_CONF_THRESHOLD = 0.65

# When beam and greedy decoders disagree we cap confidence — even if greedy
# was "sure", the disagreement signals real ambiguity worth flagging.
DISAGREEMENT_CONF_CAP = 0.55


def _get_predictor():
    """Lazy singleton — VietOCR weights are ~150 MB and slow to load."""
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
    except ImportError as e:
        print(f"[QN_OCR] ERROR: {e}", file=sys.stderr)
        print("[QN_OCR] Install: pip install --no-deps vietocr && "
              "pip install gdown prefetch_generator pyyaml lmdb einops", file=sys.stderr)
        raise
    cfg = Cfg.load_config_from_name("vgg_transformer")
    cfg["cnn"]["pretrained"] = False
    cfg["device"] = "cpu"
    # Start in greedy mode — we toggle to beamsearch per-call in ocr_qn_page.
    cfg["predictor"]["beamsearch"] = False
    _PREDICTOR = Predictor(cfg)
    return _PREDICTOR


def _predict_with_conf(predictor, crop) -> tuple[str, float]:
    """Two-pass decode: beam text + greedy confidence.

    Returns (final_text, confidence_in_[0,1]). final_text is the beam result;
    confidence is greedy's per-line probability, capped if beam disagrees.
    """
    # Pass 1: beamsearch — best decode (uses internal LM)
    predictor.config["predictor"]["beamsearch"] = True
    try:
        beam_text = predictor.predict(crop)
    finally:
        predictor.config["predictor"]["beamsearch"] = False
    beam_text = (beam_text or "").strip()

    # Pass 2: greedy + return_prob — confidence signal
    greedy_text, greedy_prob = predictor.predict(crop, return_prob=True)
    greedy_text = (greedy_text or "").strip()
    try:
        conf = float(greedy_prob)
    except (TypeError, ValueError):
        conf = 0.0

    # If decoders agree, full greedy confidence stands. Otherwise cap it —
    # disagreement is itself a strong "this line is ambiguous" signal.
    if beam_text != greedy_text:
        conf = min(conf, DISAGREEMENT_CONF_CAP)

    # Prefer beam text if we have it; fall back to greedy if beam emitted empty.
    final_text = beam_text or greedy_text
    return final_text, conf


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_load_cache(cache_path: str, image_path: str, version: str
                    ) -> tuple[str, list[float]] | None:
    """Return cached (text, confs) if cache is valid for this image, else None."""
    cf = Path(cache_path)
    if not cf.exists():
        return None
    try:
        data = json.loads(cf.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("version") != version:
        return None
    if data.get("image_md5") != _md5_file(image_path):
        return None
    text = data.get("text", "")
    confs = data.get("confs", []) or []
    if not isinstance(text, str) or not isinstance(confs, list):
        return None
    return text, [float(c) for c in confs]


def _save_cache(cache_path: str, image_path: str,
                text: str, confs: list[float], version: str) -> None:
    cf = Path(cache_path)
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps({
        "version": version,
        "image_md5": _md5_file(image_path),
        "text": text,
        "confs": confs,
    }, ensure_ascii=False), encoding="utf-8")


def ocr_qn_page(image_path: str, verbose: bool = False,
                cache_path: str | None = None,
                backend: str = "auto"
                ) -> tuple[str, list[float]]:
    """OCR a QN text page.

    Returns:
        (text, line_confidences) — newline-joined text + per-line confidence
        in the SAME order as lines in `text` (one float per line).

    Pipeline:
      [optional cache check] -> load image -> DL / deskew-aware line detection
      (core.ocr.line_detector) -> for each straightened line crop: 2-pass
      VietOCR (beam for text + greedy for confidence) -> [optional cache save].

    backend: line detector backend ("auto"|"dbnet"|"projection_deskew"|
    "projection"). "auto" uses DBNet when PaddleOCR is installed, else the
    deskew-aware projection. The resolved backend is folded into the cache key.

    cache_path: if given, results are cached to this JSON file keyed by
    image-content md5 + decoder/detector version. Re-runs on the same image are
    instantaneous and skip model loading entirely.
    """
    version = _cache_version(backend)
    if cache_path:
        cached = _try_load_cache(cache_path, image_path, version)
        if cached is not None:
            if verbose:
                print(f"  [QN_OCR] {image_path}: cache HIT "
                      f"({len(cached[1])} lines)")
            return cached

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return "", []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    crops = detect_line_crops(img_rgb, backend=backend, verbose=verbose)
    if not crops:
        if verbose:
            print(f"  [QN_OCR] {image_path}: no text lines detected", file=sys.stderr)
        return "", []

    predictor = _get_predictor()
    lines: list[str] = []
    confs: list[float] = []
    n_low = 0

    for i, crop_np in enumerate(crops):
        if crop_np.shape[0] < 10 or crop_np.shape[1] < 10:
            continue
        crop = Image.fromarray(crop_np)
        try:
            text, conf = _predict_with_conf(predictor, crop)
        except Exception as e:
            if verbose:
                print(f"  [QN_OCR] predict failed on line {i}: {e}",
                      file=sys.stderr)
            continue
        if text:
            lines.append(text)
            confs.append(conf)
            if conf < LOW_CONF_THRESHOLD:
                n_low += 1

    if verbose:
        avg = sum(confs) / len(confs) if confs else 0.0
        print(f"  [QN_OCR] {image_path}: {len(lines)} lines, "
              f"avg_conf={avg:.2f}, low_conf={n_low}")

    full_text = "\n".join(lines)
    if cache_path:
        _save_cache(cache_path, image_path, full_text, confs, version)
    return full_text, confs
