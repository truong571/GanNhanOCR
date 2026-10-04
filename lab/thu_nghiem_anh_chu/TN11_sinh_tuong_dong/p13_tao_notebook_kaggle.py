"""TN11 p13 (04/10) — TẠO notebook Kaggle dựng sẵn (.ipynb) cho gói đủ 10 bộ: preflight (GPU, token, quyền ghi HF) -> run (đẩy HF mỗi 5 phút) -> status.
Notebook KHÔNG chứa token: nó đọc HF_TOKEN từ Kaggle Secrets lúc chạy. Màn hình chỉ có dòng quan trọng ([env]/[tiến độ]/[run]/lỗi); log đầy đủ ở /kaggle/working/tn11_bundles/logs.

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p13_tao_notebook_kaggle.py [--hf-repo mdnt571/gannhanocr-tn11-full] [--out <đường dẫn .ipynb>]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

MD_HEAD = """# TN11 — sinh ảnh glyph theo phong cách TỪNG cuốn (10 bộ) → đẩy lên HuggingFace  (notebook v3.1: màn hình gọn)

**Một lần, trước khi chạy** (cột phải → Settings): Accelerator = **GPU T4 x2** · Internet = **On** · Environment = Pin to original ·
File → Add-ons → **Secrets** → `HF_TOKEN` (quyền Write) → **Attach** · **Add Input** = Dataset chứa gói (`kaggle_pack_….zip`, Kaggle tự giải nén; tên tuỳ ý).

1. **Ca thật** (mặc định `TEST = False`, `BUDGET_HOURS = 7.5`; cần quota còn ≥ ~9 giờ): **Save Version → Save & Run All** một lần (không chạy tương tác, không chạy hai ca cùng lúc).
   Thông lượng đo thật ≈ 38 việc/giờ (T4 x2) ⇒ 7,5 giờ ≈ 265 việc; **chạy lại đúng notebook này** ở ca sau (tự bỏ qua việc đã có ở HF; ca cuối tự dừng khi hết việc) tới khi ô cuối báo **ĐÃ XONG** (≈ 2 ca; `BUDGET_HOURS = 11` thì gần xong trong 1 ca nhưng cần quota ≥ 12,5).
2. Muốn thử lại ngắn: `TEST = True` (≈ 25 phút, 14 việc).
3. Kết quả **đẩy lên HF mỗi 5 phút** (và lần cuối khi kết thúc/huỷ). Màn hình ít dòng: sau `[w0] [fd] sẵn sàng (cuda)` im lặng ≤ 10 phút rồi mỗi 10 phút một dòng `[tiến độ]` — bình thường; log đầy đủ ở `/kaggle/working/tn11_bundles/logs`."""

CELL_CFG = '''# ===== CẤU HÌNH =====
NB_VERSION = "v3.1 (màn hình gọn)"
HF_REPO = "__HF_REPO__"        # repo dataset PRIVATE trên HuggingFace (tự tạo nếu chưa có); đổi nếu tài khoản HF khác
TEST = False                   # False: ca THẬT (ca thử 04/10 đã đạt) · True: chạy thử lại ~25 phút (14 việc)
BUDGET_HOURS = 7.5             # ca thật: min(11, quota còn lại - 1,5); thông lượng đo thật ~38 việc/giờ nên 410 việc ≈ 10,5 giờ

import os, glob, subprocess, sys
# Màn hình gọn: đặt TRƯỚC khi chạy bất kỳ tiến trình Python con nào (các biến này được đọc lúc import thư viện)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"        # tắt khối "Preparing/Uploading/Committing" của huggingface_hub (~9 dòng mỗi lần đẩy)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"          # tắt banner "Hello from the pygame community"
os.environ["PYTHONWARNINGS"] = "ignore:pkg_resources is deprecated:UserWarning"   # chỉ tắt đúng cảnh báo pkg_resources của pygame
print("notebook", NB_VERSION, "| TEST =", TEST, "| BUDGET_HOURS =", BUDGET_HOURS)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")      # chỉ nạp vào môi trường, KHÔNG in ra
except Exception as e:
    raise RuntimeError(f"Chưa đính secret HF_TOKEN (File -> Add-ons -> Secrets -> bật Attach cho HF_TOKEN): {type(e).__name__}") from None
RUNNERS = sorted(glob.glob("/kaggle/input/**/w_runner_khung_chay.py", recursive=True))
if not RUNNERS:                                                  # Dataset còn là tệp .zip (chưa tự giải nén) -> giải vào /kaggle/tmp
    import zipfile
    zs = sorted(glob.glob("/kaggle/input/**/*.zip", recursive=True))
    if zs:
        zipfile.ZipFile(zs[0]).extractall("/kaggle/tmp/pack")
        RUNNERS = sorted(glob.glob("/kaggle/tmp/pack/**/w_runner_khung_chay.py", recursive=True))
assert RUNNERS, "Không thấy gói trong /kaggle/input — đã Add Input Dataset chứa gói chưa?"
RUNNER = RUNNERS[0]
PACK = os.path.dirname(RUNNER)
r = subprocess.run([sys.executable, RUNNER, "preflight", "--pack", PACK, "--remote", f"hf:{HF_REPO}", "--device", "cuda"], text=True, capture_output=True)
print(r.stdout[-2500:], (r.stderr[-800:] if r.returncode else ""))
assert r.returncode == 0, "PREFLIGHT KHÔNG ĐẠT — đọc các dòng ✗ ở trên, sửa rồi chạy lại"
'''

CELL_RUN = '''# ===== CHẠY: màn hình chỉ có dòng quan trọng; log đầy đủ: /kaggle/working/tn11_bundles/logs =====
extra = "--max-tasks 7 --budget-hours 1.5" if TEST else f"--budget-hours {BUDGET_HOURS}"
!python {RUNNER} run --pack {PACK} --device cuda --out /kaggle/tmp/tn11_full --tmp /kaggle/tmp/tn11 --bundle-dir /kaggle/working/tn11_bundles --remote hf:{HF_REPO} --sync-every 300 {extra}
'''

CELL_STATUS = '''# ===== TIẾN ĐỘ TRÊN HF (gọn) =====
!python {RUNNER} status --pack {PACK} --remote hf:{HF_REPO}
'''

MD_TAIL = """**Sau khi `[status]` báo ĐÃ XONG** (trên Mac): `.venv/bin/hf auth login` → `.venv/bin/hf download <HF_REPO> --repo-type dataset --local-dir measure_out/_tn11/full/kaggle_ket_qua`
→ `w_runner_khung_chay.py validate` → `p12_cham_toan_bo.py` (xem docs/KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md)."""


def cell(kind, src):
    base = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default="mdnt571/gannhanocr-tn11-full")
    ap.add_argument("--out", default=str(REPO / "measure_out/_tn11/full/kaggle_notebook/tn11_full_kaggle.ipynb"))
    a = ap.parse_args(argv)
    cells = [cell("markdown", MD_HEAD), cell("code", CELL_CFG.replace("__HF_REPO__", a.hf_repo)), cell("code", CELL_RUN), cell("code", CELL_STATUS), cell("markdown", MD_TAIL)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    # kiểm cú pháp từng ô mã (dòng '!' đổi thành lời gọi hợp lệ)
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            src = re.sub(r"^!(.*)$", lambda m: f"_ = {m.group(1)!r}", src, flags=re.M)
            compile(src, f"cell{i}", "exec")
    print(f"[p13] {out} ({out.stat().st_size} byte), {sum(1 for c in cells if c['cell_type'] == 'code')} ô mã, cú pháp OK; HF_REPO = {a.hf_repo}")


if __name__ == "__main__":
    main()
