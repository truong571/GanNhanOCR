"""OCR API clients: HCMUS SinoNom OCR (kimhannom.fit.hcmus.edu.vn)."""
from __future__ import annotations

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


def _merge_fragment_columns(result: list[list[dict]]) -> list[list[dict]]:
    """Hậu xử lý cột OCR để về đúng số cột thật (kinhhannom hay tách/gộp nhầm
    do chữ viết tay lệch x). 5 luật:
      (A) Bỏ CỘT RỖNG (ô trống không ký tự).
      (1) Gộp MẢNH VỤN 1-2 ký tự sát cột bên (< 0.6 nhịp) vào cột đó.
      (2) Gộp CẶP SÁT (gap < 0.35 nhịp, tổng <= 1.3*median) = cột tách đôi giữa thân.
      (3) Gộp box NGẮN (<0.8 cao trung bình) + CHỒNG x vào hàng xóm (detection trùng).
      (4) Bỏ "CỘT MA" GIỮA (chen giữa 2 cột, cả 2 khe < 0.7 nhịp = chú thích/nhiễu).
    Cột thật — kể cả cột chương NGẮN ở rìa — cách hàng xóm ~1 nhịp nên không bị
    gộp/xoá; chỉ mảnh vụn/cột-ma/trùng mới bị xử lý.
    """
    import statistics

    def cx(col):
        return statistics.mean([c["bbox"][0] for c in col]) if col else 0

    def col_h(col):
        ys = [c["bbox"][1] for c in col] + [c["bbox"][3] for c in col]
        return (max(ys) - min(ys)) if ys else 0

    def col_xr(col):
        return (min(c["bbox"][0] for c in col), max(c["bbox"][2] for c in col))

    if len(result) <= 1:
        return result
    sizes = [len(c) for c in result if c]
    if not sizes:
        return result
    med_size = statistics.median(sizes)
    centers = [cx(c) for c in result]
    spacings = sorted(abs(centers[i] - centers[i + 1]) for i in range(len(centers) - 1))
    med_space = statistics.median(spacings) if spacings else 100
    frag_max = max(2, int(0.25 * med_size))

    # (A) BỎ CỘT RỖNG: ô kinhhannom nhận trống (không ký tự) -> không thành cột.
    cols = [list(c) for c in result if len(c) > 0]
    changed = True
    while changed and len(cols) > 1:
        changed = False
        # (1) Mảnh vụn nhỏ -> gộp vào hàng xóm gần nhất
        for i, c in enumerate(cols):
            if 0 < len(c) <= frag_max:
                nbrs = [k for k in (i - 1, i + 1) if 0 <= k < len(cols)]
                j = min(nbrs, key=lambda k: abs(cx(cols[k]) - cx(c)))
                if abs(cx(cols[j]) - cx(c)) < 0.6 * med_space:
                    cols[j] = sorted(cols[j] + c, key=lambda b: b["y_center"])
                    cols.pop(i)
                    changed = True
                    break
        if changed:
            continue
        # (2) Cặp cột SÁT nhau (gap < 0.35 nhịp) mà TỔNG cỡ ~1 cột thường
        # (<= 1.3*median) -> 1 cột bị tách đôi giữa thân -> gộp.
        # (Cột thật cách ~1 nhịp; cặp tách-đôi-L/R mỗi nửa đủ chữ -> tổng lớn,
        #  KHÔNG gộp để tránh nhân đôi ký tự.)
        for i in range(len(cols) - 1):
            gap = abs(cx(cols[i]) - cx(cols[i + 1]))
            if gap < 0.35 * med_space and len(cols[i]) + len(cols[i + 1]) <= 1.3 * med_size:
                cols[i] = sorted(cols[i] + cols[i + 1], key=lambda b: b["y_center"])
                cols.pop(i + 1)
                changed = True
                break
        if changed:
            continue
        # (3) Cột THỪA = NGẮN bất thường (<0.80 chiều cao trung bình) VÀ CHỒNG x
        # với hàng xóm -> detection thừa/trùng của kinhhannom -> gộp vào hàng xóm.
        # (Cột thật đứng cạnh nhau KHÔNG chồng x nên không bị gộp.)
        med_h = statistics.median([col_h(c) for c in cols])
        for i, c in enumerate(cols):
            if col_h(c) < 0.80 * med_h:
                xl, xr = col_xr(c)
                ov = None
                for k in (i - 1, i + 1):
                    if 0 <= k < len(cols):
                        xl2, xr2 = col_xr(cols[k])
                        if (min(xr, xr2) - max(xl, xl2)) > 0.25 * (xr - xl):
                            ov = k
                if ov is not None:
                    cols[ov] = sorted(cols[ov] + c, key=lambda b: b["y_center"])
                    cols.pop(i)
                    changed = True
                    break
        if changed:
            continue
        # (4) "CỘT MA" GIỮA: cột chen giữa 2 cột chính, CẢ HAI khe x < 0.7 nhịp
        # (chú thích nhỏ/nhiễu giữa cột) -> XOÁ khỏi 9 cột chính.
        # (Cột thật — kể cả cột chương ngắn ở rìa — cách hàng xóm ~1 nhịp nên
        #  không bị xoá; chỉ cột chen-giữa-cả-2-bên mới dính.)
        for i in range(1, len(cols) - 1):
            gl = abs(cx(cols[i]) - cx(cols[i - 1]))
            gr = abs(cx(cols[i]) - cx(cols[i + 1]))
            if gl < 0.7 * med_space and gr < 0.7 * med_space:
                cols.pop(i)
                changed = True
                break
    return cols


def boxes_to_columns(boxes: list[dict], merge_fragments: bool = True) -> list[list[dict]]:
    """Convert OCR boxes to columns, each column = list of {char, y_center, bbox}.

    Sorted: columns right->left, within column top->bottom.
    merge_fragments=True: gộp mảnh vụn 1-2 ký tự bị tách nhầm -> không DƯ cột.
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

    if merge_fragments:
        result = _merge_fragment_columns(result)
    # An toàn: bỏ MỌI cột rỗng (ô kinhhannom nhận trống, không ký tự) -> dữ liệu
    # cột sạch để bước sau gán Hán-Nôm <-> Quốc Ngữ chính xác theo số ký tự.
    result = [c for c in result if c]
    return result


def _ocr_one_pass(image_path: str, use_frame: bool, frame_pad: int):
    """1 lượt: (crop khung pad) -> upload -> recognize -> trả boxes thô + có-crop."""
    upload_path = image_path
    tmp_crop = None
    framed = False
    if use_frame:
        try:
            import tempfile
            import cv2
            from core.image.frame_detector import crop_to_frame
            bgr = cv2.imread(str(image_path))
            if bgr is not None:
                crop = crop_to_frame(bgr, pad=frame_pad)
                tmp_crop = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp_crop.close()
                cv2.imwrite(tmp_crop.name, crop)
                upload_path = tmp_crop.name
                framed = True
        except Exception as e:
            print(f"[OCR] frame-crop failed ({e}), dùng ảnh gốc", file=sys.stderr)
    file_name = upload_image(upload_path)
    if tmp_crop is not None:
        try:
            os.unlink(tmp_crop.name)
        except OSError:
            pass
    if not file_name:
        return None, framed
    return recognize(file_name), framed


def ocr_page(
    image_path: str,
    cache_path: str | None = None,
    verbose: bool = False,
    use_frame: bool = True,
    frame_pad: int = 12,
    expected_cols: int = 9,
    retry_pad: int = 30,
) -> list[list[dict]] | None:
    """OCR 1 trang -> các cột char dicts (có cache).

    use_frame=True: crop khung 9 cột (loại số 1-9, số trang, viền) TRƯỚC khi OCR.
    Nếu số cột != expected_cols (mặc định 9), TỰ RETRY với pad lớn hơn (retry_pad)
    -> khôi phục cột chương ngắn ở rìa bị crop sát cắt mất. Giữ kết quả gần 9 nhất.
    """
    if cache_path:
        cache_file = Path(cache_path)
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if verbose:
                print(f"    [OCR] Loaded cache: {cache_file.name}")
            return cached.get("columns")

    if verbose:
        print(f"    [OCR] {Path(image_path).name} (pad={frame_pad})...")
    boxes, framed = _ocr_one_pass(image_path, use_frame, frame_pad)
    if boxes is None:
        return None
    columns = boxes_to_columns(boxes)

    # RETRY với pad lớn nếu chưa đúng số cột (chỉ khi đang crop khung)
    if use_frame and len(columns) != expected_cols and retry_pad > frame_pad:
        if verbose:
            print(f"    [OCR] {len(columns)} cột != {expected_cols} -> retry pad={retry_pad}")
        b2, _ = _ocr_one_pass(image_path, use_frame, retry_pad)
        if b2 is not None:
            c2 = boxes_to_columns(b2)
            # giữ kết quả GẦN expected nhất
            if abs(len(c2) - expected_cols) < abs(len(columns) - expected_cols):
                columns, boxes, frame_pad = c2, b2, retry_pad

    if verbose:
        print(f"    [OCR] -> {len(columns)} cột, {sum(len(c) for c in columns)} chữ")

    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(image_path, "rb") as fh:
            img_hash = hashlib.md5(fh.read()).hexdigest()
        cache_data = {
            "image": image_path, "image_hash": img_hash,
            "framed": framed, "frame_pad": frame_pad,
            "n_columns": len(columns), "columns": columns, "boxes_raw": boxes,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    return columns
