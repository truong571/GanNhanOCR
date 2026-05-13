#!/usr/bin/env python3
"""Check current OCR token state — manual or auto-refresh.

Usage:
    python3 scripts/check_ocr_token.py

Reads .env, reports:
  • Which auth method is configured (refresh vs manual)
  • If manual: TTL remaining
  • If refresh: triggers a refresh + reports new TTL
"""
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Load .env
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return f"EXPIRED {-seconds/60:.0f} min ago"
    if seconds < 3600:
        return f"{seconds/60:.0f} phút"
    return f"{seconds/3600:.2f} giờ"


def decode_jwt_exp(token: str) -> float:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload)).get("exp", 0))
    except Exception:
        return 0.0


refresh_token = os.environ.get("SN_OCR_REFRESH_TOKEN", "").strip()
api_key = os.environ.get("SN_OCR_FIREBASE_API_KEY", "").strip()
manual_token = os.environ.get("SN_OCR_TOKEN", "").strip()

print("=" * 60)
print("HCMUS SinoNom OCR token status")
print("=" * 60)

# Manual token state
print("\n[1] SN_OCR_TOKEN (manual, 1h TTL)")
if manual_token:
    exp = decode_jwt_exp(manual_token)
    if exp:
        eta = exp - time.time()
        exp_dt = datetime.fromtimestamp(exp, timezone.utc).astimezone()
        print(f"    Status:  {'✓ valid' if eta > 0 else '✗ expired'}")
        print(f"    Expires: {exp_dt.isoformat()}")
        print(f"    Còn:     {fmt_eta(eta)}")
    else:
        print(f"    ⚠️  Không decode được JWT (token có thể không hợp lệ)")
else:
    print(f"    (not set)")

# Auto-refresh state
print("\n[2] Auto-refresh (SN_OCR_REFRESH_TOKEN + SN_OCR_FIREBASE_API_KEY)")
if refresh_token and api_key:
    print(f"    SN_OCR_REFRESH_TOKEN:    set (length={len(refresh_token)})")
    print(f"    SN_OCR_FIREBASE_API_KEY: set (prefix={api_key[:8]}...)")
    print(f"\n    → Test refresh call...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.ocr.ocr_api import _refresh_firebase_token
    result = _refresh_firebase_token(refresh_token, api_key)
    if result:
        new_token, new_exp = result
        eta = new_exp - time.time()
        new_exp_dt = datetime.fromtimestamp(new_exp, timezone.utc).astimezone()
        print(f"    ✓ Refresh thành công")
        print(f"    New idToken length: {len(new_token)}")
        print(f"    Hết hạn:            {new_exp_dt.isoformat()}")
        print(f"    Còn:                {fmt_eta(eta)}")
        print(f"\n    Auto-refresh sẽ tự gọi mỗi khi cache token còn <5 phút.")
        print(f"    Refresh token vĩnh viễn (đến khi anh đổi password / logout).")
    else:
        print(f"    ✗ Refresh thất bại (xem stderr ở trên)")
else:
    if not refresh_token:
        print(f"    SN_OCR_REFRESH_TOKEN:    (not set)")
    if not api_key:
        print(f"    SN_OCR_FIREBASE_API_KEY: (not set)")
    print(f"\n    → Chưa setup auto-refresh. Xem README để biết cách lấy.")

# Recommendation
print("\n" + "=" * 60)
if refresh_token and api_key:
    print("✓ Auto-refresh đang active — pipeline chạy mãi không cần thao tác tay.")
elif manual_token:
    eta = decode_jwt_exp(manual_token) - time.time()
    if eta > 0:
        print(f"⚠️  Đang dùng manual token. Hết hạn trong {fmt_eta(eta)}.")
        print(f"    → Setup auto-refresh để không phải rotate mỗi giờ.")
    else:
        print("✗ Manual token đã hết hạn. Rotate hoặc setup auto-refresh.")
else:
    print("✗ Chưa có token nào. OCR sẽ fail.")
print("=" * 60)
