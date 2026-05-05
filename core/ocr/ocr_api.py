"""OCR API clients: HCMUS SinoNom OCR (kimhannom.fit.hcmus.edu.vn)."""

import hashlib
import json
import os
import sys
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Auto-load .env file if exists
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _val = _val.strip().strip("'\"")
                os.environ.setdefault(_key.strip(), _val)


# ---------------------------------------------------------------------------
# HCMUS SinoNom OCR API (kimhannom.fit.hcmus.edu.vn)
# ---------------------------------------------------------------------------

_SN_DOMAIN = os.environ.get("SN_DOMAIN", "kimhannom.fit.hcmus.edu.vn")


def _get_ocr_token() -> str:
    """Get OCR token from environment variable."""
    token = os.environ.get("SN_OCR_TOKEN", "")
    if not token:
        print(
            "[OCR] WARNING: SN_OCR_TOKEN not set. "
            'Set it via: export SN_OCR_TOKEN="<your_token>"',
            file=sys.stderr,
        )
    return token


def upload_image(image_path: str) -> str | None:
    """Upload image to HCMUS OCR server. Returns server file_name."""
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


def recognize(file_name: str) -> list[dict] | None:
    """Call OCR API, returns list of boxes [{points, transcription}, ...]."""
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


def boxes_to_columns(boxes: list[dict]) -> list[list[dict]]:
    """Convert OCR boxes to columns, each column = list of {char, y_center, bbox}.

    Sorted: columns right->left, within column top->bottom.
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b["points"][0][0], reverse=True)

    cols: list[list[dict]] = []
    for box in sorted_boxes:
        if not cols:
            cols.append([box])
            continue
        last_box = cols[-1][-1]
        if abs(last_box["points"][0][0] - box["points"][0][0]) < 15:
            cols[-1].append(box)
        else:
            cols.append([box])

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


def ocr_page(
    image_path: str,
    cache_path: str | None = None,
    verbose: bool = False,
) -> list[list[dict]] | None:
    """OCR a full page image. Returns columns of char dicts with caching."""
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

    file_name = upload_image(image_path)
    if not file_name:
        return None

    if verbose:
        print("    [OCR] Running OCR...")

    boxes = recognize(file_name)
    if boxes is None:
        return None

    columns = boxes_to_columns(boxes)

    if verbose:
        total_chars = sum(len(c) for c in columns)
        print(f"    [OCR] Got {len(columns)} columns, {total_chars} chars")

    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
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
