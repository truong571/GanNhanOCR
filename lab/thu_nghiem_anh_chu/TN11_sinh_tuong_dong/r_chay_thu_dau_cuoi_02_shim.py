"""Vỏ bọc cho BƯỚC 2 (mô phỏng cấu trúc Kaggle): chạy NGUYÊN VĂN w_runner_khung_chay.py nhưng đổi tiền tố đường dẫn "/kaggle/" trong các lệnh glob
(find_pack hard-code "/kaggle/input/...") sang một gốc giả. Không sửa bộ chạy.

    python r_chay_thu_dau_cuoi_02_shim.py <gốc_giả_thay_/kaggle> <đường dẫn w_runner_khung_chay.py> <đối số của bộ chạy...>
"""
import sys

sys.dont_write_bytecode = True
import glob
import runpy

FAKE = sys.argv.pop(1).rstrip("/")
_orig = glob.glob


def _g(pat, *a, **k):
    if isinstance(pat, str) and pat.startswith("/kaggle/"):
        pat = FAKE + pat[len("/kaggle"):]
    return _orig(pat, *a, **k)


glob.glob = _g
runner = sys.argv[1]
sys.argv = [runner] + sys.argv[2:]
runpy.run_path(runner, run_name="__main__")
