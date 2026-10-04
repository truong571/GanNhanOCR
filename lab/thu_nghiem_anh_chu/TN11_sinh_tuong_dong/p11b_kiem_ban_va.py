"""TN11 p11b (04/10) — KIỂM các bản vá bộ chạy sau phản biện (w_runner_khung_chay.py), 0 GPU, bộ sinh giả. Bổ sung cho w_runner_kiem_khung.py (34 phép thử cũ).

  V1  cổng GPU: run (không --stub, không --device) trên máy không CUDA -> mã thoát 2 + thông báo, KHÔNG nạp mô hình
  V2  việc 0 ảnh / bỏ qua >= 50 % bị coi là LỖI: không ghi tar, không đánh dấu xong; 3 lỗi liên tiếp -> dừng 'errors'
  V3  mã thoát 3 khi không làm được việc nào và chưa xong (hỏng thật), mã 0 khi làm được một phần
  V4  dừng cứng: worker "treo" (bỏ qua hạn) bị watchdog dừng sau hạn nhận việc + --hard-extra-s; status.hard_stop, stop_reason=hard_stop, mã thoát 0
  V5  SIGTERM vào tiến trình run: dừng nhận việc mới sau việc đang làm, flush, ghi status.json (signal=15, stop_reason=signal), mã thoát 0
  V6  assets_sha: kho ngoài đã có assets_sha.txt khác -> mã thoát 2 'TÀI SẢN KHÁC' (không trộn ảnh phong cách/phông khác nhau)
  V7  find_pack: chạy bản sao runner nằm ở gốc gói KHÔNG cần --pack (độc lập độ sâu /kaggle/input)
  V8  Syncer.stop có hạn chờ: remote treo -> trả False sau timeout, không treo tiến trình
  V9  status được chép cạnh bundle (--bundle-dir) và env được in ra log
  V10 log gọn: worker "ồn" (200 dòng cảnh báo/việc) -> màn hình chỉ có dòng quan trọng, log đầy đủ ở tệp
  V11 dòng tiến độ định kỳ [tiến độ] n/N việc, %, ảnh, HF, thời gian
  V12 preflight: gói thật ĐẠT; không --remote / kho khác tài sản / không GPU -> KHÔNG ĐẠT (mã 2), <= 8 dòng
  V13 status: tiến độ gọn từ thư mục cục bộ
  V14 luồng HF end-to-end với huggingface_hub GIẢ (PYTHONPATH): đẩy lô, plan_sha/assets_sha, chạy tiếp, preflight/status qua hf:, + chữ ký hàm khớp thư viện THẬT

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p11b_kiem_ban_va.py [--pack measure_out/_tn11/full/kaggle_pack_...]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import w_runner_khung_chay as K  # noqa: E402

RUN = HERE / "w_runner_khung_chay.py"
OUT = REPO / "measure_out" / "_tn11" / "full" / "kiem_ban_va"
RES = []


def check(name, ok, detail=""):
    RES.append(dict(ten=name, ok=bool(ok), chi_tiet=str(detail)[:300]))
    print(f"{'PASS' if ok else 'FAIL'} {name} {str(detail)[:200]}", flush=True)


def mini_pack(d: Path, chars="ABCDEFGHIJKLMNOP", shard=4, assets_sha=None, runner_copy=False):
    """Gói mini hợp lệ (có ckpt/ rỗng + fonts.json) để preflight không báo 'gói hỏng'; assets_sha=None -> tính đúng, hoặc truyền chuỗi để gây lệch."""
    d.mkdir(parents=True, exist_ok=True)
    plan = K.make_plan({"X": dict(chars=list(chars), styles=1)}, [("base", 0)], shard, fonts=None)
    (d / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    (d / "ckpt").mkdir(exist_ok=True)
    (d / "fonts.json").write_text("{}", encoding="utf-8")
    ash = assets_sha if assets_sha is not None else K.compute_assets_sha(d, {})
    (d / "manifest.json").write_text(json.dumps(dict(plan_sha=plan["sha"], ckpt_sha256={}, assets_sha=ash), ensure_ascii=False), encoding="utf-8")
    if runner_copy:
        shutil.copyfile(RUN, d / "w_runner_khung_chay.py")
    return plan


def run_cli(args, timeout=300, env=None, cwd=None):
    p = subprocess.run([sys.executable, str(RUN), *args], capture_output=True, text=True, timeout=timeout, env=env or dict(os.environ), cwd=cwd)
    return p.returncode, p.stdout + p.stderr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=None)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%H%M%S")
    W = OUT / f"lan_{tag}"
    W.mkdir(parents=True, exist_ok=True)

    # ---- V1 cổng GPU
    pack = Path(a.pack) if a.pack else None
    if pack and (pack / "plan.json").exists():
        code, out = run_cli(["run", "--pack", str(pack), "--out", str(W / "v1_out"), "--tmp", str(W / "v1_tmp")], timeout=120)
        check("V1.cong_GPU_khong_CUDA_thoat_2", code == 2 and "KHÔNG CÓ GPU CUDA" in out and not (W / "v1_tmp" / "font_diffusion").exists(), f"code={code}")
    else:
        check("V1.cong_GPU_khong_CUDA_thoat_2", False, "thiếu --pack")

    # ---- V2 việc rỗng là lỗi (gọi worker_loop trực tiếp)
    plan = mini_pack(W / "v2_pack")
    out2 = W / "v2_out"
    allc = "".join(plan["books"]["X"]["chars"])
    gen = K.StubGen(0.0, fail_chars=allc)                                     # mọi chữ bị bỏ qua -> 0 ảnh
    r = K.worker_loop(plan, W / "v2_pack", out2, "w0", gen, K.now() + 600, 1.0, 600.0)
    tars = list((out2 / "shards").rglob("*.tar")) if (out2 / "shards").exists() else []
    check("V2.viec_rong_khong_ghi_tar_va_dung_errors", len(tars) == 0 and r["stop"] == "errors" and len(r["failed"]) >= 3, f"tar={len(tars)} stop={r['stop']} failed={len(r['failed'])}")
    gen2 = K.StubGen(0.0, fail_chars="A")                                     # 1/4 chữ bị bỏ qua (< 50 %) -> vẫn xong
    out2b = W / "v2b_out"
    r2 = K.worker_loop(plan, W / "v2_pack", out2b, "w0", gen2, K.now() + 600, 1.0, 600.0)
    ok_tar = list((out2b / "shards").rglob("*.tar"))
    check("V2b.bo_qua_it_van_xong", r2["stop"] == "complete" and len(ok_tar) == len(plan["tasks"]), f"stop={r2['stop']} tar={len(ok_tar)}/{len(plan['tasks'])}")

    # ---- V3 mã thoát 3 / 0 (CLI, mini pack, bộ sinh giả)
    p3 = W / "v3_pack"; mini_pack(p3)
    code, out = run_cli(["run", "--stub", "--pack", str(p3), "--out", str(W / "v3_out"), "--tmp", str(W / "v3_tmp"), "--workers", "1", "--stub-fail", allc, "--budget-hours", "0.2", "--safety-s", "0"], timeout=120)
    st = json.loads((W / "v3_out" / "status" / "status.json").read_text(encoding="utf-8"))
    check("V3.thoat_3_khi_khong_lam_duoc_viec_nao", code == 3 and st["tasks_done_local"] == 0 and st["stop_reason"] == "crash_or_error", f"code={code} stop={st['stop_reason']}")
    code, out = run_cli(["run", "--stub", "--pack", str(p3), "--out", str(W / "v3b_out"), "--tmp", str(W / "v3_tmp"), "--workers", "1", "--budget-hours", "0.2", "--safety-s", "0"], timeout=120)
    st = json.loads((W / "v3b_out" / "status" / "status.json").read_text(encoding="utf-8"))
    check("V3b.thoat_0_khi_xong", code == 0 and st["complete"] is True, f"code={code}")

    # ---- V4 dừng cứng: worker coi mỗi việc 1 giây (est-s 1) nhưng thật 128 giây -> bị watchdog dừng
    p4 = W / "v4_pack"; mini_pack(p4, chars="".join(chr(0x4E00 + i) for i in range(64)), shard=32)
    t0 = time.time()
    code, out = run_cli(["run", "--stub", "--pack", str(p4), "--out", str(W / "v4_out"), "--tmp", str(W / "v4_tmp"), "--workers", "1", "--stub-ms", "4000",
                         "--est-s", "1", "--budget-hours", "0.0042", "--safety-s", "0", "--hard-extra-s", "8"], timeout=200)
    dt = time.time() - t0
    st = json.loads((W / "v4_out" / "status" / "status.json").read_text(encoding="utf-8"))
    check("V4.dung_cung_worker_treo", st["hard_stop"] is True and st["stop_reason"] == "hard_stop" and code == 0 and dt < 90, f"dt={dt:.0f}s code={code} stop={st['stop_reason']}")

    # ---- V5 SIGTERM
    p5 = W / "v5_pack"; mini_pack(p5, chars="".join(chr(0x4E00 + i) for i in range(160)), shard=8)
    proc = subprocess.Popen([sys.executable, str(RUN), "run", "--stub", "--pack", str(p5), "--out", str(W / "v5_out"), "--tmp", str(W / "v5_tmp"), "--workers", "2",
                             "--stub-ms", "300", "--budget-hours", "1"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(12)
    proc.send_signal(signal.SIGTERM)
    try:
        outtxt, _ = proc.communicate(timeout=120)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill(); outtxt, code = "", -9
    sp = W / "v5_out" / "status" / "status.json"
    st = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    n_tar = len(list((W / "v5_out" / "shards").rglob("*.tar")))
    check("V5.SIGTERM_dung_sach_ghi_status", code == 0 and st.get("signal") == signal.SIGTERM and st.get("stop_reason") == "signal" and 0 < n_tar < len(json.loads((p5 / "plan.json").read_text())["tasks"]),
          f"code={code} signal={st.get('signal')} stop={st.get('stop_reason')} tar={n_tar}")

    # ---- V6 assets_sha khác kho ngoài
    p6a, p6b = W / "v6_pack_a", W / "v6_pack_b"
    mini_pack(p6a, assets_sha="aaaa"); mini_pack(p6b, assets_sha="bbbb")
    rem = W / "v6_remote"
    c1, o1 = run_cli(["run", "--stub", "--pack", str(p6a), "--out", str(W / "v6_out_a"), "--tmp", str(W / "v6_tmp"), "--workers", "1", "--remote", f"dir:{rem}", "--budget-hours", "0.2", "--safety-s", "0"], timeout=120)
    c2, o2 = run_cli(["run", "--stub", "--pack", str(p6b), "--out", str(W / "v6_out_b"), "--tmp", str(W / "v6_tmp"), "--workers", "1", "--remote", f"dir:{rem}", "--budget-hours", "0.2", "--safety-s", "0"], timeout=120)
    check("V6.assets_sha_khac_kho_ngoai_thoat_2", c1 == 0 and (rem / "assets_sha.txt").exists() and c2 == 2 and "TÀI SẢN KHÁC" in o2, f"c1={c1} c2={c2}")

    # ---- V7 find_pack không cần --pack
    p7 = W / "v7_pack"; mini_pack(p7, runner_copy=True)
    r7 = subprocess.run([sys.executable, str(p7 / "w_runner_khung_chay.py"), "run", "--stub", "--out", str(W / "v7_out"), "--tmp", str(W / "v7_tmp"), "--workers", "1",
                         "--budget-hours", "0.2", "--safety-s", "0"], capture_output=True, text=True, cwd="/tmp", timeout=120)
    c7 = r7.returncode
    st = json.loads((W / "v7_out" / "status" / "status.json").read_text(encoding="utf-8")) if (W / "v7_out" / "status" / "status.json").exists() else {}
    check("V7.find_pack_theo_vi_tri_tep", c7 == 0 and st.get("complete") is True, f"code={c7}")

    # ---- V8 Syncer.stop có hạn chờ
    class HangRemote:
        kind = "hang"
        def push(self, *a, **k):
            time.sleep(60)
    d8 = W / "v8_out"; (d8 / "shards" / "X" / "base_s0").mkdir(parents=True, exist_ok=True)
    (d8 / "shards" / "X" / "base_s0" / "00000.tar").write_bytes(b"x")
    sy = K.Syncer(d8, HangRemote(), 3600, set(), retries=1, backoff=0.0)
    t0 = time.time(); okv = sy.stop(timeout=3.0); dt = time.time() - t0
    check("V8.Syncer_stop_co_han_cho", okv is False and dt < 10, f"ok={okv} dt={dt:.1f}s")

    # ---- V9 status chép cạnh bundle + env in ra log
    p9 = W / "v9_pack"; mini_pack(p9)
    c9, o9 = run_cli(["run", "--stub", "--pack", str(p9), "--out", str(W / "v9_out"), "--tmp", str(W / "v9_tmp"), "--workers", "1", "--bundle-dir", str(W / "v9_bundles"),
                      "--session", "S9", "--budget-hours", "0.2", "--safety-s", "0"], timeout=120)
    check("V9.status_chep_canh_bundle_va_env_trong_log", c9 == 0 and (W / "v9_bundles" / "status_S9" / "status.json").exists() and "[env]" in o9, f"code={c9}")

    # ---- V10 log gọn
    p10 = W / "v10_pack"; mini_pack(p10, chars="".join(chr(0x4E00 + i) for i in range(16)), shard=4)
    env10 = dict(os.environ, TN11_STUB_NOISE="200", TN11_STUB_ERRLINE="1")
    c10, o10 = run_cli(["run", "--stub", "--pack", str(p10), "--out", str(W / "v10_out"), "--tmp", str(W / "v10_tmp"), "--workers", "2", "--bundle-dir", str(W / "v10_b"),
                        "--budget-hours", "0.2", "--safety-s", "0", "--progress-every", "0"], timeout=120, env=env10)
    logs = list((W / "v10_b" / "logs").glob("worker-*.log"))
    n_log = sum(len(f.read_text(encoding="utf-8").splitlines()) for f in logs)
    n_screen = len([ln for ln in o10.splitlines() if "ruido" in ln])
    check("V10.log_gon_man_hinh_chi_dong_quan_trong", c10 == 0 and n_log >= 4 * 200 and n_screen == 0 and "Error: lỗi giả" in o10 and len(o10.splitlines()) < 40,
          f"code={c10} dòng_log={n_log} dòng_ruido_trên_màn_hình={n_screen} tổng_dòng_màn_hình={len(o10.splitlines())}")

    # ---- V11 tiến độ
    p11 = W / "v11_pack"; mini_pack(p11, chars="".join(chr(0x4E00 + i) for i in range(48)), shard=4)
    c11, o11 = run_cli(["run", "--stub", "--pack", str(p11), "--out", str(W / "v11_out"), "--tmp", str(W / "v11_tmp"), "--workers", "2", "--stub-ms", "300",
                        "--budget-hours", "0.2", "--safety-s", "0", "--progress-every", "2"], timeout=120)
    prog = [ln for ln in o11.splitlines() if ln.startswith("[tiến độ]")]
    check("V11.dong_tien_do_dinh_ky", c11 == 0 and len(prog) >= 1 and "việc" in prog[-1] and "%" in prog[-1] and "ảnh" in prog[-1], f"số_dòng={len(prog)} mẫu={prog[-1] if prog else ''}")

    # ---- V12 preflight
    pf_dir = W / "v12_remote"
    c12, o12 = run_cli(["preflight", "--pack", str(pack), "--remote", f"dir:{pf_dir}", "--stub", "--tmp", str(W / "v12_tmp")], timeout=300) if pack else (9, "")
    check("V12a.preflight_goi_that_DAT", c12 == 0 and "PREFLIGHT ĐẠT" in o12 and len(o12.strip().splitlines()) <= 8, f"code={c12} dòng={len(o12.strip().splitlines())}")
    c12b, o12b = run_cli(["preflight", "--pack", str(p3), "--stub", "--tmp", str(W / "v12_tmp")], timeout=120)
    check("V12b.preflight_thieu_remote_KHONG_DAT", c12b == 2 and "KHÔNG có --remote" in o12b, f"code={c12b}")
    pkA, pkB = W / "v12_pkA", W / "v12_pkB"
    mini_pack(pkA, assets_sha="aaaa"); mini_pack(pkB, assets_sha="bbbb")
    rem12 = W / "v12_remote2"
    run_cli(["run", "--stub", "--pack", str(pkA), "--out", str(W / "v12_outA"), "--tmp", str(W / "v12_tmp"), "--workers", "1", "--remote", f"dir:{rem12}", "--budget-hours", "0.2", "--safety-s", "0"], timeout=120)
    c12c, o12c = run_cli(["preflight", "--pack", str(pkB), "--remote", f"dir:{rem12}", "--stub", "--tmp", str(W / "v12_tmp")], timeout=120)
    check("V12c.preflight_kho_khac_tai_san_KHONG_DAT", c12c == 2 and "KHO CHỨA KẾ HOẠCH/TÀI SẢN KHÁC" in o12c, f"code={c12c} {o12c.strip().splitlines()[-2][:120] if o12c.strip() else ''}")
    c12d, o12d = run_cli(["preflight", "--pack", str(p3), "--remote", f"dir:{W / 'v12_remote3'}", "--tmp", str(W / "v12_tmp")], timeout=120)
    check("V12d.preflight_khong_GPU_KHONG_DAT", c12d == 2 and "GPU: KHÔNG CÓ" in o12d, f"code={c12d}")

    # ---- V13 status
    c13, o13 = run_cli(["status", "--pack", str(p3), "--out", str(W / "v3b_out")], timeout=60)
    c13b, o13b = run_cli(["status", "--pack", str(p3), "--out", str(W / "khong_co")], timeout=60)
    check("V13.status_gon_cuc_bo", c13 == 0 and "ĐÃ XONG" in o13 and "[status]" in o13 and c13b == 0 and "chưa xong" in o13b and len(o13.strip().splitlines()) <= 4, f"{o13.strip().splitlines()[0] if o13.strip() else ''}")

    # ---- V14 HF end-to-end với thư viện giả + chữ ký hàm của thư viện THẬT
    fake = W / "fake_hf"
    (fake / "huggingface_hub").mkdir(parents=True, exist_ok=True)
    (fake / "huggingface_hub" / "errors.py").write_text("class EntryNotFoundError(Exception):\n    pass\n", encoding="utf-8")
    (fake / "huggingface_hub" / "__init__.py").write_text(r'''import os, shutil
from pathlib import Path
from .errors import EntryNotFoundError
ROOT = Path(os.environ["FAKE_HF_ROOT"])
def _r(repo_id): return ROOT / repo_id.replace("/", "__")
def get_token(): return None
class HfApi:
    def __init__(self, token=None): self.token = token
    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False): _r(repo_id).mkdir(parents=True, exist_ok=True)
    def whoami(self): return {"name": "fakeuser"}
    def list_repo_files(self, repo_id, repo_type=None): return [p.relative_to(_r(repo_id)).as_posix() for p in _r(repo_id).rglob("*") if p.is_file()]
    def upload_folder(self, folder_path, repo_id, repo_type=None, allow_patterns=None, commit_message=None):
        n = 0
        for rel in allow_patterns or []:
            src = Path(folder_path) / rel
            if src.is_file():
                dst = _r(repo_id) / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst); n += 1
        with open(ROOT / "commits.log", "a") as f: f.write(f"{repo_id}\t{n}\t{commit_message}\n")
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None):
        dst = _r(repo_id) / path_in_repo; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path_or_fileobj if isinstance(path_or_fileobj, bytes) else path_or_fileobj.read())
def hf_hub_download(repo_id, filename, repo_type=None, token=None):
    f = _r(repo_id) / filename
    if not f.exists(): raise EntryNotFoundError(filename)
    return str(f)
''', encoding="utf-8")
    p14 = W / "v14_pack"; mini_pack(p14, chars="".join(chr(0x4E00 + i) for i in range(32)), shard=4)
    froot = W / "v14_fake_root"; froot.mkdir(parents=True, exist_ok=True)
    env14 = dict(os.environ, PYTHONPATH=str(fake) + os.pathsep + os.environ.get("PYTHONPATH", ""), FAKE_HF_ROOT=str(froot), HF_TOKEN="khong-phai-token-that")
    c14, o14 = run_cli(["run", "--stub", "--pack", str(p14), "--out", str(W / "v14_out1"), "--tmp", str(W / "v14_tmp"), "--workers", "2", "--stub-ms", "100", "--remote", "hf:fake/repo",
                        "--sync-every", "1", "--budget-hours", "0.2", "--safety-s", "0", "--progress-every", "0"], timeout=180, env=env14)
    repo = froot / "fake__repo"
    tars = sorted(repo.rglob("*.tar")) if repo.exists() else []
    n_tasks = len(json.loads((p14 / "plan.json").read_text())["tasks"])
    commits = (froot / "commits.log").read_text().splitlines() if (froot / "commits.log").exists() else []
    check("V14a.HF_day_du_tar_va_plan_sha_assets_sha", c14 == 0 and len(tars) == n_tasks and (repo / "plan_sha.txt").exists() and (repo / "assets_sha.txt").exists() and "khong-phai-token-that" not in o14,
          f"code={c14} tar={len(tars)}/{n_tasks} commit={len(commits)}")
    c14b, o14b = run_cli(["run", "--stub", "--pack", str(p14), "--out", str(W / "v14_out2"), "--tmp", str(W / "v14_tmp"), "--workers", "1", "--remote", "hf:fake/repo", "--budget-hours", "0.2",
                          "--safety-s", "0", "--progress-every", "0"], timeout=120, env=env14)
    st14 = json.loads((W / "v14_out2" / "status" / "status.json").read_text(encoding="utf-8"))
    check("V14b.HF_chay_tiep_bo_viec_da_co", c14b == 0 and st14["tasks_done_local"] == 0 and st14["complete"] is True, f"code={c14b} làm_lại={st14['tasks_done_local']}")
    c14c, o14c = run_cli(["preflight", "--pack", str(p14), "--remote", "hf:fake/repo", "--stub", "--tmp", str(W / "v14_tmp")], timeout=120, env=env14)
    c14d, o14d = run_cli(["status", "--pack", str(p14), "--remote", "hf:fake/repo"], timeout=60, env=env14)
    check("V14c.HF_preflight_va_status", c14c == 0 and "tài khoản fakeuser" in o14c and c14d == 0 and "ĐÃ XONG" in o14d and "khong-phai-token-that" not in o14c + o14d, f"preflight={c14c} status={c14d}")
    import inspect
    try:
        import huggingface_hub as H
        sig_ok = (all(k in inspect.signature(H.HfApi.upload_folder).parameters for k in ("folder_path", "repo_id", "repo_type", "allow_patterns", "commit_message"))
                  and all(k in inspect.signature(H.HfApi.create_repo).parameters for k in ("repo_id", "repo_type", "private", "exist_ok"))
                  and all(k in inspect.signature(H.HfApi.list_repo_files).parameters for k in ("repo_id", "repo_type"))
                  and all(k in inspect.signature(H.HfApi.upload_file).parameters for k in ("path_or_fileobj", "path_in_repo", "repo_id", "repo_type"))
                  and all(k in inspect.signature(H.hf_hub_download).parameters for k in ("repo_id", "filename", "repo_type", "token")))
        from huggingface_hub.errors import EntryNotFoundError  # noqa: F401
        check("V14d.chu_ky_ham_khop_huggingface_hub_that", sig_ok, f"huggingface_hub {H.__version__}")
    except Exception as e:  # noqa: BLE001
        check("V14d.chu_ky_ham_khop_huggingface_hub_that", False, f"{type(e).__name__}: {e}")

    n_fail = sum(1 for r in RES if not r["ok"])
    (OUT / f"ket_qua_{tag}.json").write_text(json.dumps(dict(pass_=len(RES) - n_fail, fail=n_fail, n=len(RES), phep_thu=RES), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"TỔNG: pass={len(RES) - n_fail} fail={n_fail} n={len(RES)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
