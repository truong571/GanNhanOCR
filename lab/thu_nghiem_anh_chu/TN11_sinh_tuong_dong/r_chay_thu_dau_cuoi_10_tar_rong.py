"""Một việc mà MỌI chữ đều bị bỏ qua (vd. OOM ở lô 1 cho cả 128 chữ, phông hỏng) có bị coi là LỖI không? Mô phỏng bằng --stub-fail = toàn bộ chữ của việc đầu tiên của kế hoạch thật.
Kỳ vọng an toàn: việc không có ảnh nào không được ghi là "xong" (hoặc validate báo lỗi). Quan sát: tar rỗng vẫn được ghi, việc tính là xong, validate n_errors = 0.
Ra: .../review/chay_thu_dau_cuoi/10_tar_rong.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import json
import shutil

import r_chay_thu_dau_cuoi_lib as L

K = L.load_runner()
rep = L.Rep("10_tar_rong")
plan = K.load_plan(L.PACK)
t0 = plan["tasks"][0]
chars = "".join(K.task_chars(plan, t0))
out = L.R / "empty_task"
if out.exists():
    shutil.rmtree(out)
code, log, sec = L.sh(["run", "--stub", "--workers", "1", "--stub-ms", "0.1", "--pack", L.PACK, "--out", out, "--max-tasks", "1", "--stub-fail", chars], cwd=L.R / "stub_cwd")
st = L.status_of(out)
rows = L.task_rows(out)
vc, vlog, _ = L.sh(["validate", "--pack", L.PACK, "--out", out], cwd=L.R / "stub_cwd")
vr = json.loads(vlog.strip().splitlines()[-1])
rep.data("worker_log", [dict(id=r["id"], n_gen=r["n_gen"], n_skip=r["n_skip"]) for r in rows])
rep.data("validate", {k: vr[k] for k in ("present", "images", "skipped_by_reason", "n_errors", "missing")})
rep.inv("Z1.viec_khong_sinh_duoc_anh_nao_KHONG_duoc_ghi_la_xong_hoac_validate_bao_loi", not (rows and rows[0]["n_gen"] == 0 and vc == 0 and vr["n_errors"] == 0),
        dict(n_gen=rows[0]["n_gen"] if rows else None, n_skip=rows[0]["n_skip"] if rows else None, validate_exit=vc, n_errors=vr["n_errors"], skipped_by_reason=vr["skipped_by_reason"], tasks_done_local=st["tasks_done_local"]))
rep.save()
