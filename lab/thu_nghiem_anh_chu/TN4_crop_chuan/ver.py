"""ver.py — chọn phiên bản crop chuẩn qua biến môi trường TN4_VER (v1 = đăng ký trước, v2 = sửa sau khi thấy v1)."""
import importlib, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VER = os.environ.get('TN4_VER', 'v1')
assert VER in ('v1', 'v2')
BIGROOT = REPO / 'measure_out/_thu_nghiem_anh_chu/TN4'
BIG = BIGROOT / VER
LABOUT = HERE / f'out_{VER}'
LIBNAME = 'cclib' if VER == 'v1' else 'cclib_v2'
FREEZE = LABOUT / 't00_freeze.json'


def lib():
    return importlib.import_module(LIBNAME)
