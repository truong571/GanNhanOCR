"""QN text recognition using VietOCR + horizontal-projection line detection.

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
import numpy as np
from PIL import Image

_PREDICTOR = None

# Bump if changing decoder logic / model — invalidates all cached results.
CACHE_VERSION = "vgg_transformer-2pass-v1"

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


def _detect_text_lines(img_rgb: np.ndarray, min_height: int = 18,
                       gap: int = 8) -> list[tuple[int, int, int, int]]:
    """Horizontal projection -> rows of text. Returns list of (x1, y1, x2, y2)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    proj = bw.sum(axis=1)
    h, w = bw.shape
    th = max(int(w * 0.01), 5)
    is_ink = proj > th

    runs: list[list[int]] = []
    in_run = False
    start = 0
    for y in range(h):
        if is_ink[y] and not in_run:
            in_run = True
            start = y
        elif not is_ink[y] and in_run:
            in_run = False
            if y - start >= min_height:
                runs.append([start, y])
    if in_run and h - start >= min_height:
        runs.append([start, h])

    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out: list[tuple[int, int, int, int]] = []
    for s, e in merged:
        col = bw[s:e].sum(axis=0)
        nz = np.where(col > 1)[0]
        if not len(nz):
            continue
        x1, x2 = int(nz[0]), int(nz[-1]) + 1
        pad = 5
        out.append((max(0, x1 - pad), max(0, s - pad),
                    min(w, x2 + pad), min(h, e + pad)))
    return out


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_load_cache(cache_path: str, image_path: str
                    ) -> tuple[str, list[float]] | None:
    """Return cached (text, confs) if cache is valid for this image, else None."""
    cf = Path(cache_path)
    if not cf.exists():
        return None
    try:
        data = json.loads(cf.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("version") != CACHE_VERSION:
        return None
    if data.get("image_md5") != _md5_file(image_path):
        return None
    text = data.get("text", "")
    confs = data.get("confs", []) or []
    if not isinstance(text, str) or not isinstance(confs, list):
        return None
    return text, [float(c) for c in confs]


def _save_cache(cache_path: str, image_path: str,
                text: str, confs: list[float]) -> None:
    cf = Path(cache_path)
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps({
        "version": CACHE_VERSION,
        "image_md5": _md5_file(image_path),
        "text": text,
        "confs": confs,
    }, ensure_ascii=False), encoding="utf-8")


def ocr_qn_page(image_path: str, verbose: bool = False,
                cache_path: str | None = None
                ) -> tuple[str, list[float]]:
    """OCR a QN text page.

    Returns:
        (text, line_confidences) — newline-joined text + per-line confidence
        in the SAME order as lines in `text` (one float per line).

    Pipeline:
      [optional cache check] -> load image -> horizontal projection ->
      for each line crop: 2-pass VietOCR (beam for text + greedy for confidence).
      -> [optional cache save]

    cache_path: if given, results are cached to this JSON file keyed by
    image-content md5 + decoder version. Re-runs on the same image are
    instantaneous and skip model loading entirely.
    """
    if cache_path:
        cached = _try_load_cache(cache_path, image_path)
        if cached is not None:
            if verbose:
                print(f"  [QN_OCR] {image_path}: cache HIT "
                      f"({len(cached[1])} lines)")
            return cached

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return "", []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    boxes = _detect_text_lines(img_rgb)
    if not boxes:
        if verbose:
            print(f"  [QN_OCR] {image_path}: no text lines detected", file=sys.stderr)
        return "", []

    predictor = _get_predictor()
    pil = Image.fromarray(img_rgb)
    lines: list[str] = []
    confs: list[float] = []
    n_low = 0

    for (x1, y1, x2, y2) in boxes:
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        crop = pil.crop((x1, y1, x2, y2))
        try:
            text, conf = _predict_with_conf(predictor, crop)
        except Exception as e:
            if verbose:
                print(f"  [QN_OCR] predict failed on box {(x1,y1,x2,y2)}: {e}",
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
        _save_cache(cache_path, image_path, full_text, confs)
    return full_text, confs
