"""BƯỚC 3a — MÔ HÌNH THẬT trên MPS qua đúng lệnh `run` của gói (1 worker, --max-tasks 2) rồi kiểm:
  cổng use_fst (không nổ, không cảnh báo 'Checkpoint for ... not found', có 'Created standard FontDiffuserModelDPM'), ảnh 96x96 RGB không rỗng/trắng tinh,
  _meta.json (sinh + bỏ qua = yêu cầu), validate, tốc độ giây/ảnh (so --est-s mặc định 2,66x128), env.json, dung lượng log,
  và TẤT ĐỊNH: chạy lại ĐÚNG MỘT việc (ca mới, đĩa trống, cùng tmp) -> so băm tar với lần đầu (và nếu khác: số ảnh/pixel lệch).
Chạy lâu (~13 + ~7 phút trên M4/MPS): hãy chạy nền. cwd = review/chay_thu_dau_cuoi/mps_cwd (font_diffusion ghi .log vào cwd).
    PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python lab/.../r_chay_thu_dau_cuoi_03_mps.py [--skip-rerun]
Ra: 03_mps.json, mps_run.log, mps_rerun.log
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import argparse
import io
import json
import re
import shutil
import subprocess
import tarfile
import time

import r_chay_thu_dau_cuoi_lib as L

ap = argparse.ArgumentParser()
ap.add_argument("--skip-rerun", action="store_true")
ap.add_argument("--reuse", action="store_true", help="không chạy lại mô hình: đọc lại mps_out / mps_out_rerun + log đã có (chỉ chạy lại phần kiểm)")
ap.add_argument("--max-tasks", type=int, default=2)
ap.add_argument("--gen-bs", type=int, default=16, help="lô sinh; MẶC ĐỊNH CỦA BỘ CHẠY LÀ 32 nhưng ~12 GB bộ nhớ làm Mac 16 GB hoán đổi")
a = ap.parse_args()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

K = L.load_runner()
rep = L.Rep("03_mps")
R, PACK = L.R, L.PACK
CWD = R / "mps_cwd"
TMP = R / "mps_tmp"
OUT = R / "mps_out"
OUT2 = R / "mps_out_rerun"
plan = K.load_plan(PACK)
CWD.mkdir(parents=True, exist_ok=True)
for p in (OUT, OUT2):
    if p.exists() and not a.reuse:
        shutil.rmtree(p)


def run_real(out, max_tasks, logname):
    if a.reuse and (out / "status" / "status.json").exists() and (R / logname).exists():
        return 0, json.loads((out / "status" / "status.json").read_text(encoding="utf-8"))["seconds"], (R / logname).read_text(encoding="utf-8", errors="replace")
    cmd = [sys.executable, str(L.RUNNER), "run", "--pack", str(PACK), "--workers", "1", "--device", "mps", "--max-tasks", str(max_tasks), "--gen-bs", str(a.gen_bs), "--tmp", str(TMP), "--out", str(out)]
    t0 = time.time()
    with open(R / logname, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=CWD, env=L.env({"PYTORCH_ENABLE_MPS_FALLBACK": "0"}), stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode, round(time.time() - t0, 1), (R / logname).read_text(encoding="utf-8", errors="replace")


code, sec, log = run_real(OUT, a.max_tasks, "mps_run.log")
st = L.status_of(OUT) if (OUT / "status" / "status.json").exists() else {}
rep.inv("M1.exit0_status_max_tasks_2_viec", code == 0 and st.get("worker_exit_codes") == [0] and st.get("stop_reason") == "max_tasks" and st.get("tasks_done_local") == a.max_tasks and st.get("worker_stop") == {"w0": "max_tasks"},
        dict(code=code, giay=sec, status={k: st.get(k) for k in ("tasks_done_local", "worker_exit_codes", "worker_stop", "stop_reason", "complete", "sync_ok")}))
bad_lines = [l for l in log.splitlines() if re.search(r"CỔNG use_fst|Checkpoint for|fst_module|WithFST|No FST checkpoint|using random weights", l)]
# Lưu ý: logger INFO của font_diffusion bị nuốt (tệp inference.sample_optimized.log rỗng) nên KHÔNG thể tìm "Created standard ..." trong log; bằng chứng cổng use_fst = worker không ném FstGuardError
# (không có tệp STOP, mã thoát 0) + 04_font_cascade C0 (guard dict) + w_runner_phat_hien_fst.json F1/F4 (cổng bắt đúng bản lỗi).
rep.inv("M2.khong_co_dong_cong_use_fst_hay_canh_bao_thieu_checkpoint_va_khong_STOP", not bad_lines and "Traceback" not in log and not (OUT / "STOP").exists() and st.get("worker_exit_codes") == [0],
        dict(dong_xau=bad_lines[:3], co_STOP=(OUT / "STOP").exists(), co_dong_Created_standard_trong_log="Created standard FontDiffuserModelDPM" in log, inference_log_bytes=(CWD / "inference.sample_optimized.log").stat().st_size if (CWD / "inference.sample_optimized.log").exists() else None))
m = re.search(r"FontDiffusion loaded in ([\d.]+)s", log)
rep.data("nap_mo_hinh_giay", float(m.group(1)) if m else None)
rep.data("log_bytes", len(log.encode("utf-8")))
rep.data("log_dong_dau", log.splitlines()[:3])
rows = L.task_rows(OUT)
rep.inv("M3.khong_co_tep_loi_errors_w0", not (OUT / "status" / "errors-w0.jsonl").exists(), dict(rows=[(r["id"], r["n_gen"], r["n_skip"], r["sec"]) for r in rows]))
# ảnh
shards = sorted((OUT / "shards").rglob("*.tar"))
img_info, ok_img, n_img, n_blank = [], True, 0, 0
for tp in shards:
    with tarfile.open(tp) as tf:
        meta = json.loads(tf.extractfile("_meta.json").read())
        ink_list = []
        for mm in tf.getmembers():
            if not mm.name.startswith("U+"):
                continue
            im = Image.open(io.BytesIO(tf.extractfile(mm).read()))
            g = np.array(im.convert("L"))
            ink = float((g < 128).mean() * 100)
            ink_list.append(ink)
            n_img += 1
            blank = g.min() > 240 or ink < 1.0
            n_blank += blank
            ok_img &= im.size == (96, 96) and im.mode == "RGB" and not blank
        img_info.append(dict(task=meta["task"], n=len(ink_list), ink_p1=round(float(np.percentile(ink_list, 1)), 1), ink_tv=round(float(np.median(ink_list)), 1), ink_p99=round(float(np.percentile(ink_list, 99)), 1),
                             sinh=len(meta["generated"]), bo_qua=len(meta["skipped"]), du_yeu_cau=sorted(meta["requested"]) == sorted(list(meta["generated"]) + list(meta["skipped"]))))
rep.inv("M4.anh_96x96_RGB_khong_rong_khong_trang_tinh_va_meta_du", ok_img and n_img == 128 * a.max_tasks and n_blank == 0 and all(i["du_yeu_cau"] and i["bo_qua"] == 0 for i in img_info), dict(n_img=n_img, blank=n_blank, theo_lo=img_info))
vc, vlog, vsec = L.sh(["validate", "--pack", PACK, "--out", OUT], cwd=CWD)
vr = json.loads(vlog.strip().splitlines()[-1])
rep.inv("M5.validate_n_errors0_present=max_tasks", vc == 0 and vr["n_errors"] == 0 and vr["present"] == a.max_tasks and vr["images"] == 128 * a.max_tasks, {k: vr[k] for k in ("present", "missing", "images", "n_errors", "skipped_by_reason")})
# tốc độ
secs = [r["sec"] for r in rows]
imgs = [r["n_gen"] for r in rows]
spi = [s / n for s, n in zip(secs, imgs)]
rep.data("toc_do", dict(sec_moi_viec=secs, giay_moi_anh=[round(x, 3) for x in spi], tb_giay_moi_anh=round(float(np.mean(spi)), 3), est_mac_dinh_giay_moi_anh=2.66, est_s_mac_dinh=2.66 * 128,
                        ti_so_do_that_tren_est=round(float(np.mean(spi)) / 2.66, 2), thiet_bi="mps (Apple M4); KHÔNG phải T4"))
env = json.loads((OUT / "status" / "env.json").read_text(encoding="utf-8"))
rep.inv("M6.env_json_ghi_phien_ban", env.get("torch") and env.get("diffusers") and env.get("python") and "gpus" in env, env)
logs_in_cwd = {p.name: p.stat().st_size for p in CWD.glob("*.log")}
rep.data("log_font_diffusion_trong_cwd", logs_in_cwd)

# tất định: chạy lại ĐÚNG MỘT việc (ca mới, đĩa trống)
if not a.skip_rerun:
    code2, sec2, log2 = run_real(OUT2, 1, "mps_rerun.log")
    first = plan["tasks"][0]["id"]
    p1, p2 = OUT / K.rel_path(first), OUT2 / K.rel_path(first)
    same = p1.exists() and p2.exists() and L.sha256_file(p1) == L.sha256_file(p2)
    diff_detail = None
    if p1.exists() and p2.exists() and not same:
        def imgs_of(p):
            with tarfile.open(p) as tf:
                return {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.name.startswith("U+")}
        A, B = imgs_of(p1), imgs_of(p2)
        nd, maxd, mean_d = 0, 0, []
        for k in A:
            if A[k] != B.get(k):
                nd += 1
                x = np.array(Image.open(io.BytesIO(A[k])).convert("L")).astype(int)
                y = np.array(Image.open(io.BytesIO(B[k])).convert("L")).astype(int)
                maxd = max(maxd, int(np.abs(x - y).max())); mean_d.append(float(np.abs(x - y).mean()))
        diff_detail = dict(anh_khac=nd, tren=len(A), lech_pixel_toi_da=maxd, lech_pixel_tb=round(float(np.mean(mean_d)), 3) if mean_d else 0)
    rep.inv("M7.chay_lai_dung_1_viec_cho_tar_byte_giong_(tat_dinh)", code2 == 0 and same, dict(code=code2, giay=sec2, task=first, byte_giong=same, chi_tiet_khac=diff_detail))
    rows2 = L.task_rows(OUT2)
    rep.data("rerun_toc_do", dict(sec=[r["sec"] for r in rows2], giay_moi_anh=[round(r["sec"] / r["n_gen"], 3) for r in rows2]))
rep.save()
