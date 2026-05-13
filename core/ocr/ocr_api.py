"""OCR API clients: HCMUS SinoNom OCR (kimhannom.fit.hcmus.edu.vn)."""

import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Auto-load .env from project root (3 levels up: core/ocr/ocr_api.py → repo root)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
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

# In-memory cache for refreshed Firebase ID token (idToken lives 1h).
# {'token': <jwt>, 'exp': <epoch_seconds>}
_token_cache: dict[str, object] = {"token": "", "exp": 0.0}


def _jwt_exp(token: str) -> float:
    """Decode JWT payload's exp claim (epoch seconds). Returns 0.0 if not parseable."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        return float(data.get("exp", 0))
    except Exception:
        return 0.0


def _login_hcmus(username: str, password: str) -> tuple[str, float] | None:
    """POST email/password to https://<SN_DOMAIN>/account/login.

    HCMUS backend handles Firebase auth server-side and sets `token` cookie
    with a fresh 1-hour idToken. Returns (id_token, exp_epoch_seconds) on
    success, None on failure.
    """
    url = f"https://{_SN_DOMAIN}/account/login"
    try:
        session = requests.Session()
        # Browser-like headers — server blocks default `python-requests` UA with 403
        session.post(
            url,
            data={"UserName": username, "Password": password},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "Mozilla/5.0",
                "Referer": url,
                "Origin": f"https://{_SN_DOMAIN}",
            },
            verify=False,
            timeout=15,
            allow_redirects=True,
        )
        token = session.cookies.get("token", "")
        if not token:
            print("[OCR] Auto-login failed: no `token` cookie in response", file=sys.stderr)
            return None
        exp = _jwt_exp(token)
        if not exp:
            print("[OCR] Auto-login: token returned but JWT not parseable", file=sys.stderr)
            exp = time.time() + 3600   # fallback 1h
        return token, exp
    except Exception as e:
        print(f"[OCR] Auto-login error: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _get_ocr_token() -> str:
    """Return a valid Firebase ID token for the HCMUS OCR API.

    Resolution order:
      1. Cached token (if still valid for >5 min).
      2. Auto-login via `SN_OCR_USERNAME` + `SN_OCR_PASSWORD` (POSTs to
         /account/login, extracts fresh idToken from `token` cookie).
         HCMUS doesn't expose Firebase API key, so re-login is the only
         way to get a fresh token from script.
      3. Fallback to manual `SN_OCR_TOKEN` (expires in 1h, rotate by hand).
    """
    now = time.time()

    # 1. Cache hit (with 5-min safety margin)
    cached_token = _token_cache.get("token", "")
    cached_exp = float(_token_cache.get("exp", 0.0) or 0.0)
    if cached_token and cached_exp > now + 300:
        return str(cached_token)

    # 2. Auto-login path
    username = os.environ.get("SN_OCR_USERNAME", "").strip()
    password = os.environ.get("SN_OCR_PASSWORD", "").strip()
    if username and password:
        result = _login_hcmus(username, password)
        if result:
            new_token, new_exp = result
            _token_cache["token"] = new_token
            _token_cache["exp"] = new_exp
            ttl_min = (new_exp - now) / 60
            print(
                f"[OCR] Auto-login OK (token valid {ttl_min:.0f} min)",
                file=sys.stderr,
            )
            return new_token

    # 3. Fallback to manual token
    manual_token = os.environ.get("SN_OCR_TOKEN", "").strip()
    if manual_token:
        exp = _jwt_exp(manual_token)
        if exp and exp < now:
            print(
                f"[OCR] WARNING: SN_OCR_TOKEN expired {(now - exp) / 60:.0f} min ago. "
                f"Set SN_OCR_USERNAME + SN_OCR_PASSWORD for auto-login.",
                file=sys.stderr,
            )
        _token_cache["token"] = manual_token
        _token_cache["exp"] = exp
        return manual_token

    print(
        "[OCR] ERROR: No OCR token. Set one of:\n"
        "  • SN_OCR_USERNAME + SN_OCR_PASSWORD  (recommended; auto-login each hour)\n"
        "  • SN_OCR_TOKEN                       (manual, 1h TTL)",
        file=sys.stderr,
    )
    return ""


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
