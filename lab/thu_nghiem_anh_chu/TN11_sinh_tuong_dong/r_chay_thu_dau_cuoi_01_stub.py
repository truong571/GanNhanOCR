"""BƯỚC 1 — chế độ sinh GIẢ (--stub) với KẾ HOẠCH THẬT (410 việc, gói thật) đầu-cuối bằng chính lệnh `run` của gói:
  A. chạy trọn 2 worker: status.complete, validate, xen kẽ giữa cuốn (theo kế hoạch và theo thứ tự chạy thật), mỗi việc đúng một lần, tar tất định
     (so với lần chạy 1 worker và lần chạy 2 worker thứ hai), header tar tất định, không còn khoá/.part;
  B. NGẮT GIỮA CHỪNG: (1) --crash-after (worker chết sau khi nhận việc kế, để lại khoá + .part), chạy lại NGAY với TTL mặc định 1800 s,
     rồi chạy lại khi khoá/.part đã quá TTL -> kết quả cuối byte giống chạy liền; (2) SIGKILL cả nhóm tiến trình của CHÍNH script này
     giữa lúc đang chạy, rồi chạy lại; (3) SIGTERM chỉ gửi cho tiến trình `run` cha (như Kaggle/nhà điều hành gửi) — xem có thoát sạch/đẩy kho/để mồ côi worker.
Mọi tiến trình con do script này tạo, kết thúc theo PID/nhóm PID của chính nó. Ra: .../review/chay_thu_dau_cuoi/01_stub.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import json
import os
import shutil
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path

import r_chay_thu_dau_cuoi_lib as L

K = L.load_runner()
rep = L.Rep("01_stub")
R, PACK = L.R, L.PACK
CWD = R / "stub_cwd"
plan = K.load_plan(PACK)
tasks, ids = plan["tasks"], [t["id"] for t in plan["tasks"]]
STUBMS = "0.2"


def fresh(p: Path) -> Path:
    assert R in p.parents, p                                # chỉ xoá/ghi dưới review/chay_thu_dau_cuoi
    if p.exists():
        shutil.rmtree(p)
    return p


def run_cmd(out: Path, workers=2, *extra, stub_ms=STUBMS) -> tuple:
    return L.sh(["run", "--stub", "--stub-ms", stub_ms, "--workers", workers, "--pack", PACK, "--out", out, *extra], cwd=CWD)


def validate(out: Path) -> tuple:
    code, log, sec = L.sh(["validate", "--pack", PACK, "--out", out], cwd=CWD)
    try:
        return code, json.loads(log.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return code, log[-300:]


# ---------------------------------------------------------------- A. chạy trọn
A = fresh(R / "stub_out")                                    # đúng đường dẫn đề bài
code, log, sec = run_cmd(A, 2)
st = L.status_of(A)
rep.inv("A1.exit0_complete_410", code == 0 and st["complete"] and st["tasks_done"] == 410 == st["tasks_total"] and st["worker_exit_codes"] == [0, 0] and st["stop_reason"] == "complete",
        dict(code=code, giay=sec, status={k: st[k] for k in ("tasks_done", "tasks_total", "complete", "worker_exit_codes", "worker_stop", "stop_reason", "sync_ok")}))
vc, vr = validate(A)
rep.inv("A2.validate_n_errors0_missing0", vc == 0 and isinstance(vr, dict) and vr["n_errors"] == 0 and vr["missing"] == 0 and vr["present"] == 410 and vr["images"] == 51730,
        dict(code=vc, **({k: vr[k] for k in ("present", "missing", "images", "n_errors", "uncovered_no_font", "skipped_by_reason")} if isinstance(vr, dict) else dict(raw=vr))))
rep.data("validate_A_per_book", vr.get("per_book") if isinstance(vr, dict) else None)
rows = L.task_rows(A)
cnt = Counter(r["id"] for r in rows)
perw = Counter(r["worker"] for r in rows)
rep.inv("A3.moi_viec_dung_mot_lan_trong_tasks_w_jsonl", sorted(cnt) == sorted(ids) and max(cnt.values()) == 1 and len(rows) == 410, dict(rows=len(rows), unique=len(cnt), moi_worker=dict(perw)))
rep.inv("A3b.moi_viec_du_n_gen_khong_bo_qua_chu_nao",all(r["n_gen"] == (tasks[ids.index(r["id"])]["hi"] - tasks[ids.index(r["id"])]["lo"]) and r["n_skip"] == 0 for r in rows), None)
# xen kẽ theo kế hoạch: trong 330 việc đầu (33 vòng) mỗi cuốn đúng 1 việc/vòng; chênh số việc giữa các cuốn <= 1 trên MỌI tiền tố tới khi có cuốn hết
nbk = {b: sum(1 for t in tasks if t["book"] == b) for b in plan["books"]}
first_done = min(nbk.values()) * len(nbk)
cnts = Counter()
worst = 0
for i, t in enumerate(tasks[:first_done]):
    cnts[t["book"]] += 1
    if len(cnts) == len(nbk):
        worst = max(worst, max(cnts.values()) - min(cnts.values()))
rep.inv("A4.xen_ke_theo_ke_hoach_chenh_toi_da_1_cuon", worst <= 1 and first_done == 330, dict(so_viec_moi_cuon=nbk, tien_to_kiem=first_done, chenh_max=worst))
# xen kẽ theo THỨ TỰ CHẠY THẬT (thời điểm bắt đầu = t - sec)
st_ord = sorted(rows, key=lambda r: r["t"] - r["sec"])
pos = {tid: i for i, tid in enumerate(ids)}
disp = max(abs(i - pos[r["id"]]) for i, r in enumerate(st_ord))
cnts2, worst2 = Counter(), 0
for r in st_ord[:first_done]:
    cnts2[tasks[pos[r["id"]]]["book"]] += 1
    if len(cnts2) == len(nbk):
        worst2 = max(worst2, max(cnts2.values()) - min(cnts2.values()))
first41 = Counter(tasks[pos[r["id"]]]["book"] for r in st_ord[:41])
rep.inv("A5.xen_ke_theo_thu_tu_chay_that", disp <= 4 and worst2 <= 2 and len(first41) == 10, dict(lech_vi_tri_toi_da_so_voi_ke_hoach=disp, chenh_max_tien_to_330=worst2, sach_trong_41_viec_dau=len(first41)))
# tar tất định: so sánh với 1 worker và 2 worker lần 2
hA = L.shard_hashes(A)
B1 = fresh(R / "stub_out_w1"); c1, _, s1 = run_cmd(B1, 1)
B2 = fresh(R / "stub_out_2"); c2, _, s2 = run_cmd(B2, 2)
h1, h2 = L.shard_hashes(B1), L.shard_hashes(B2)
rep.inv("A6.tar_byte_giong_giua_1_worker_2_worker_va_lan_hai", len(hA) == 410 and hA == h1 == h2 and c1 == 0 and c2 == 0, dict(giay_1w=s1, giay_2w=s2, so_tar=len(hA)))
hdr = [L.tar_header_report(A / K.rel_path(i)) for i in ids[:3] + ids[-3:]]
rep.inv("A7.header_tar_tat_dinh_mtime0_uid0_mode644_meta_cuoi", all(h["last"] == "_meta.json" and not h["bad"] for h in hdr), dict(mau=hdr[:2]))
# meta
import tarfile  # noqa: E402

with tarfile.open(A / K.rel_path(ids[0])) as tf:
    meta = json.loads(tf.extractfile("_meta.json").read())
rep.inv("A8.meta_co_plan_sha_seed_requested_generated", meta["plan_sha"] == plan["sha"] and meta["seed"] == K.seed_of(ids[0]) and len(meta["requested"]) == 128 == len(meta["generated"]), dict(task=meta["task"], seed=meta["seed"]))
rep.inv("A9.khong_con_khoa_part", L.leftover(A) == dict(locks=0, stale_locks=0, parts=0), L.leftover(A))
rep.data("A_status", st)

# ---------------------------------------------------------------- B1. --crash-after
D = fresh(R / "stub_crash")
code, log, sec = run_cmd(D, 2, "--crash-after", "40")
sD, lf = L.status_of(D), L.leftover(D)
rep.inv("B1a.ca1_worker0_chet_17_con_1_khoa_1_part_chua_xong", code == 0 and sD["worker_exit_codes"] == [17, 0] and not sD["complete"] and lf["locks"] == 1 and lf["parts"] == 1 and sD["tasks_done"] == 409,
        dict(code=code, codes=sD["worker_exit_codes"], done=sD["tasks_done"], stop_reason=sD["stop_reason"], worker_stop=sD["worker_stop"], left=lf))
lock_path = next((D / "locks").glob("*.lock"))
part_path = next((D / "shards").rglob("*.part"))
blocked_task = lock_path.name[:-5].replace("__", "/")
# chạy lại NGAY (TTL mặc định 1800 s): khoá/.part còn trẻ -> việc bị kẹt không được làm
code, log, sec = run_cmd(D, 2)
sD2, lf2 = L.status_of(D), L.leftover(D)
rep.inv("B1b.chay_lai_ngay_ttl_1800_viec_bi_khoa_van_ket", code == 0 and not sD2["complete"] and sD2["tasks_done"] == 409 and lf2["locks"] == 1 and lf2["parts"] == 1,
        dict(code=code, done=sD2["tasks_done"], complete=sD2["complete"], stop_reason=sD2["stop_reason"], left=lf2, ket=blocked_task, dong_log="[run] kế hoạch" in log))
rep.data("B1b_stop_reason_khi_ket_khoa", sD2["stop_reason"])
rep.data("B1b_log_dong_cuoi", log.strip().splitlines()[-1][:400] if log.strip() else "")
# làm khoá và .part "già" hơn TTL rồi chạy lại -> dọn + làm nốt
old = time.time() - 2000
for p in (lock_path, part_path):
    os.utime(p, (old, old))
code, log, sec = run_cmd(D, 2)
sD3, lf3 = L.status_of(D), L.leftover(D)
hD = L.shard_hashes(D)
rows3 = L.task_rows(D)
c3 = Counter(r["id"] for r in rows3)
rep.inv("B1c.khoa_part_qua_ttl_duoc_don_va_hoan_tat_byte_giong_chay_lien", code == 0 and sD3["complete"] and lf3 == dict(locks=0, stale_locks=0, parts=0) and hD == hA,
        dict(code=code, complete=sD3["complete"], left=lf3, byte_giong=hD == hA, lam_them=len(rows3)))
rep.inv("B1d.khong_viec_nao_lam_hai_lan_sau_khi_noi_lai", max(c3.values()) == 1 and len(c3) == 410, dict(rows=len(rows3), unique=len(c3)))

# ---------------------------------------------------------------- B2. SIGKILL cả nhóm giữa chừng
E = fresh(R / "stub_kill")
p = subprocess.Popen([sys.executable, str(L.RUNNER), "run", "--stub", "--stub-ms", "8", "--workers", "2", "--pack", str(PACK), "--out", str(E)], cwd=CWD, env=L.env(),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
pgid = os.getpgid(p.pid)
time.sleep(14)
n_before = len(L.shard_hashes(E))
os.killpg(pgid, signal.SIGKILL)                              # chỉ nhóm do CHÍNH script này tạo
p.wait()
time.sleep(0.5)
lfk = L.leftover(E)
n_after = len(L.shard_hashes(E))
rep.inv("B2a.sigkill_giua_chung_de_lai_dau_vet_va_chua_xong", p.returncode == -9 and 0 < n_after < 410 and not (E / "status" / "status.json").exists(),
        dict(rc=p.returncode, tar_truoc_kill=n_before, tar_sau_kill=n_after, left=lfk, co_status_json=(E / "status" / "status.json").exists()))
# chạy lại NGAY (TTL 1800): các khoá của worker đã chết còn trẻ
code, log, sec = run_cmd(E, 2, stub_ms="0.2")
sE1, lfe1 = L.status_of(E), L.leftover(E)
rep.inv("B2b.chay_lai_ngay_sau_sigkill_con_viec_ket_khoa", code == 0 and (not sE1["complete"] or lfe1["locks"] == 0), dict(done=sE1["tasks_done"], complete=sE1["complete"], left=lfe1, stop_reason=sE1["stop_reason"]))
for q in list((E / "locks").glob("*")) + list((E / "shards").rglob("*.part")):
    os.utime(q, (old, old))
code, log, sec = run_cmd(E, 2, stub_ms="0.2")
sE2 = L.status_of(E)
rep.inv("B2c.qua_ttl_hoan_tat_byte_giong_chay_lien", code == 0 and sE2["complete"] and L.shard_hashes(E) == hA and L.leftover(E)["parts"] == 0 and L.leftover(E)["locks"] == 0, dict(complete=sE2["complete"], left=L.leftover(E)))

# ---------------------------------------------------------------- B3. SIGTERM chỉ cho tiến trình `run` cha + remote dir
F = fresh(R / "stub_term")
REM = fresh(R / "stub_term_remote")
p = subprocess.Popen([sys.executable, str(L.RUNNER), "run", "--stub", "--stub-ms", "8", "--workers", "2", "--pack", str(PACK), "--out", str(F), "--remote", f"dir:{REM}", "--sync-every", "6"],
                     cwd=CWD, env=L.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
time.sleep(1.5)
kids = subprocess.run(["pgrep", "-P", str(p.pid)], capture_output=True, text=True).stdout.split()
kids = [int(x) for x in kids]
time.sleep(12)
local_before = len(L.shard_hashes(F)); remote_before = len(K.DirRemote(str(REM)).list_done())
os.kill(p.pid, signal.SIGTERM)                               # chỉ cho tiến trình cha, như `kill <pid>` / nhà điều hành
try:
    p.wait(timeout=10)
    exited = True
except subprocess.TimeoutExpired:
    exited = False
time.sleep(3)
alive_kids = []
for k in kids:
    try:
        os.kill(k, 0); alive_kids.append(k)
    except ProcessLookupError:
        pass
n_loc1 = len(L.shard_hashes(F))
time.sleep(2)
n_loc2 = len(L.shard_hashes(F))                              # worker mồ côi còn ghi thêm?
orphans_working = n_loc2 > n_loc1
remote_after = len(K.DirRemote(str(REM)).list_done())
out_tail = ""
try:
    out_tail = p.stdout.read()[-300:] if p.stdout else ""
except Exception:  # noqa: BLE001
    pass
for k in alive_kids:                                         # dọn: chỉ các PID của CHÍNH script này
    try:
        os.kill(k, signal.SIGKILL)
    except ProcessLookupError:
        pass
time.sleep(0.5)
unsynced = len(L.shard_hashes(F)) - len(K.DirRemote(str(REM)).list_done())
graceful = exited and not alive_kids and unsynced == 0 and (F / "status" / "status.json").exists()
rep.inv("B3.sigterm_thoat_SACH_(worker_dung_kho_duoc_flush_co_status_json)", graceful,
        dict(rc=p.returncode, thoat_trong_10s=exited, worker_con_song_sau_3s=len(alive_kids), worker_mo_coi_van_ghi_them=orphans_working,
             tar_local_khi_gui=local_before, tar_remote_khi_gui=remote_before, remote_sau=remote_after, tar_chua_len_kho_khi_don=unsynced,
             co_status_json=(F / "status" / "status.json").exists()))
rep.data("B3_chi_tiet", dict(kids=len(kids), alive=len(alive_kids), orphans_working=orphans_working, unsynced=unsynced, out_tail=out_tail))
rep.save()
