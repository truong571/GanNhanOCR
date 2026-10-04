"""BƯỚC 2 — mô phỏng cấu trúc Kaggle (thư mục giả, KHÔNG tạo /kaggle thật) và chạy các lệnh trong docs/KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md mục 2 bước 4-5
đúng nguyên văn, chỉ đổi: /kaggle/* -> thư mục giả, hf:<user>/repo -> dir:<kho giả>, thêm --stub --workers 2 (Mac không có 2 GPU).
  S1 `find <input> -name w_runner_khung_chay.py | head -1`: đúng một tệp? (3 bố cục: datasets/<user>/<slug>, <slug>, <slug>/<thư mục>)
  S2 find_pack() không truyền --pack tìm được plan.json ở 3 cấp? thông báo khi không thấy / khi 4 cấp
  S3 ca THỬ (--max-tasks 2 --budget-hours 0.5): status/kho giả như mục "Kết quả mong đợi" của hướng dẫn
  S4 ca THẬT trong container MỚI (đĩa sạch, cùng kho): bỏ qua việc đã có, hoàn tất, bundle, validate trên bản "tải về"
  S5 HF: thiếu token / token giả -> thoát thế nào, có ghi token ra log/tệp không
Ra: .../review/chay_thu_dau_cuoi/02_fake_kaggle.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import r_chay_thu_dau_cuoi_lib as L

K = L.load_runner()
rep = L.Rep("02_fake_kaggle")
R, PACK, HERE = L.R, L.PACK, L.HERE
FK = R / "fake_kaggle"
SHIM = HERE / "r_chay_thu_dau_cuoi_02_shim.py"
PY = sys.executable
plan = K.load_plan(PACK)
tmp_before = {p.name for p in Path("/tmp").glob("tn11*")}


def clone(dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    r = subprocess.run(["cp", "-cR", str(PACK), str(dst)], capture_output=True)       # APFS clone (nhanh, không tốn đĩa)
    if r.returncode != 0:
        shutil.copytree(PACK, dst)


# ---- dựng 3 bố cục (đều là THƯ MỤC thật, không phải symlink: find mặc định không đi qua symlink)
lay = {"new_3cap": FK / "input" / "datasets" / "fakeuser" / "tn11-full-pack",
       "cu_1cap": R / "fake_kaggle_cu" / "input" / "tn11-full-pack",
       "zip_trong_thu_muc_2cap": R / "fake_kaggle_zip" / "input" / "tn11-full-pack" / "kaggle_pack_20261004",
       "sau_4cap": R / "fake_kaggle_4" / "input" / "datasets" / "fakeuser" / "tn11-full-pack" / "v1"}
for p in lay.values():
    clone(p)
roots = {"new_3cap": FK, "cu_1cap": R / "fake_kaggle_cu", "zip_trong_thu_muc_2cap": R / "fake_kaggle_zip", "sau_4cap": R / "fake_kaggle_4"}

# S1 find ... | head -1
res1 = {}
for k, root in roots.items():
    out = subprocess.run(f'find "{root}/input" -name w_runner_khung_chay.py', shell=True, capture_output=True, text=True).stdout.split("\n")
    out = [x for x in out if x]
    first = subprocess.run(f'find "{root}/input" -name w_runner_khung_chay.py | head -1', shell=True, capture_output=True, text=True).stdout.strip()
    res1[k] = dict(n=len(out), first_ok=first.endswith("w_runner_khung_chay.py"))
rep.inv("S1.find_tra_dung_mot_tep_o_ca_4_bo_cuc", all(v["n"] == 1 and v["first_ok"] for v in res1.values()), res1)

# S2 find_pack (glob đã đổi tiền tố, hàm thật của bộ chạy)
import glob as _glob  # noqa: E402

_orig = _glob.glob


def with_root(root: Path):
    def g(pat, *a, **k):
        if pat.startswith("/kaggle/"):
            pat = str(root) + pat[len("/kaggle"):]
        return _orig(pat, *a, **k)
    return g


res2 = {}
for k, root in roots.items():
    _glob.glob = with_root(root)
    try:
        r = K.find_pack(None)
        res2[k] = ("found", str(r.relative_to(root)))
    except SystemExit as e:
        res2[k] = ("SystemExit", str(e))
_glob.glob = _orig
rep.inv("S2.find_pack_thay_plan_json_o_1_2_3_cap", all(res2[k][0] == "found" for k in ("new_3cap", "cu_1cap", "zip_trong_thu_muc_2cap")), res2)
rep.inv("S2b.find_pack_bo_tay_o_4_cap_(gioi_han)", res2["sau_4cap"][0] == "SystemExit" and "plan.json" in res2["sau_4cap"][1], res2["sau_4cap"])
rep.data("find_pack", res2)
os.makedirs(R / "fake_kaggle_trong" / "input", exist_ok=True)
_glob.glob = with_root(R / "fake_kaggle_trong")
try:
    K.find_pack(None); msg = "found?"
except SystemExit as e:
    msg = str(e)
_glob.glob = _orig
rep.inv("S2c.khong_co_goi_thong_bao_ro", "plan.json" in msg, msg)

# S3 ca THỬ: đúng câu lệnh mục 2 bước 4 (đổi đường dẫn giả, remote dir:, --stub --workers 2)
for sub in ("tmp", "tmp2", "working", "hf_remote", "kaggle_ket_qua", "tmp_hf"):
    if (FK / sub).exists():
        shutil.rmtree(FK / sub)
CWD = FK / "working"
CWD.mkdir(parents=True, exist_ok=True)


def guide_cmd(extra: str, tmpname="tmp") -> str:
    return (f'cd "{CWD}" && PYTHONDONTWRITEBYTECODE=1 "{PY}" "{SHIM}" "{FK}" $(find "{FK}/input" -name w_runner_khung_chay.py | head -1) run '
            f'--out "{FK}/{tmpname}/tn11_full" --tmp "{FK}/{tmpname}/tn11" --bundle-dir "{FK}/working/tn11_bundles" '
            f'--remote dir:"{FK}/hf_remote" {extra} --stub --stub-ms 1 --workers 2')


p = subprocess.run(guide_cmd("--max-tasks 2 --budget-hours 0.5"), shell=True, capture_output=True, text=True, env=L.env())
log3 = p.stdout + p.stderr
OUT = FK / "tmp" / "tn11_full"
st = L.status_of(OUT)
remote = FK / "hf_remote"
n_rem = len(list(remote.rglob("*.tar")))
rep.inv("S3a.ca_thu_exit0_status_max_tasks_[0,0]", p.returncode == 0 and st["worker_exit_codes"] == [0, 0] and st["stop_reason"] == "max_tasks" and not st["complete"],
        dict(code=p.returncode, codes=st["worker_exit_codes"], stop_reason=st["stop_reason"], worker_stop=st["worker_stop"]))
rep.inv("S3b.kho_gia_co_plan_sha_txt_va_4_tar", (remote / "plan_sha.txt").read_text().strip() == plan["sha"] and n_rem == 4, dict(tar=n_rem, plan_sha_ok=(remote / "plan_sha.txt").read_text().strip() == plan["sha"]))
rep.inv("S3c.khong_co_dong_CONG_use_fst_va_co_env_json", "CỔNG use_fst" not in log3 and (OUT / "status" / "env.json").exists(), dict(env=json.loads((OUT / "status" / "env.json").read_text())))
rows = L.task_rows(OUT)
rep.inv("S3d.moi_worker_dung_2_viec_128_anh", len(rows) == 4 and all(r["n_gen"] == 128 for r in rows) and {r["worker"] for r in rows} == {"w0", "w1"}, dict(rows=[(r["id"], r["worker"]) for r in rows]))
bd = FK / "working" / "tn11_bundles"
rep.inv("S3e.bundle_cuoi_ca_co_dung_4_viec", len(list(bd.glob("bundle_*.tar"))) == 1 and len(json.loads(next(bd.glob("*.index.json")).read_text())["ids"]) == 4, dict(files=sorted(x.name for x in bd.iterdir())))
rep.data("log_ca_thu", log3[-700:])

# S4 ca THẬT trong container mới (đĩa tmp sạch), cùng kho, --budget-hours 6.5
time.sleep(1.5)                                   # --session mặc định = giờ bắt đầu tính đến giây: tránh trùng tên bundle giữa hai ca chạy sát nhau trong thử nghiệm
p = subprocess.run(guide_cmd("--budget-hours 6.5", tmpname="tmp2"), shell=True, capture_output=True, text=True, env=L.env())
OUT2 = FK / "tmp2" / "tn11_full"
st2 = L.status_of(OUT2)
rows2 = L.task_rows(OUT2)
ids_first = {r["id"] for r in rows}
ids_second = {r["id"] for r in rows2}
rep.inv("S4a.ca_that_bo_qua_4_viec_da_co_lam_406_hoan_tat", p.returncode == 0 and st2["complete"] and st2["tasks_done"] == 410 and st2["tasks_done_local"] == 406 and not (ids_first & ids_second) and len(ids_second) == 406,
        dict(code=p.returncode, status={k: st2[k] for k in ("complete", "tasks_done", "tasks_done_local", "worker_exit_codes", "stop_reason", "sync_ok")}))
n_rem2 = len(list(remote.rglob("*.tar")))
rep.inv("S4b.kho_gia_du_410_va_byte_giong_may_chay", n_rem2 == 410 and all(L.sha256_file(remote / "shards" / x.relative_to(OUT2 / "shards")) == L.sha256_file(x) for x in (OUT2 / "shards").rglob("*.tar")), dict(remote_tar=n_rem2))
# "tải về Mac" = chép kho giả rồi validate (đúng lệnh mục 2 bước 6)
dl = R / "fake_kaggle" / "kaggle_ket_qua"
if dl.exists():
    shutil.rmtree(dl)
shutil.copytree(remote, dl)
code, vlog, sec = L.sh(["validate", "--pack", PACK, "--out", dl], cwd=CWD)
vr = json.loads(vlog.strip().splitlines()[-1])
rep.inv("S4c.validate_tren_ban_tai_ve_n_errors0_missing0_410", code == 0 and vr["n_errors"] == 0 and vr["missing"] == 0 and vr["present"] == 410 and vr["images"] == 51730 and vr["uncovered_no_font"] == {"stt11": 1},
        {k: vr[k] for k in ("present", "missing", "images", "n_errors", "uncovered_no_font")})
bd2 = sorted(x.name for x in bd.iterdir())
idx = [json.loads(x.read_text()) for x in sorted(bd.glob("*.index.json"))]
all_ids = [i for x in idx for i in x["ids"]]
rep.inv("S4d.hai_ca_hai_bundle_hop_lai_du_410_khong_trung_va_it_tep", len(idx) == 2 and sorted(all_ids) == sorted(t["id"] for t in plan["tasks"]) and len(all_ids) == 410 and sum(1 for _ in (FK / "working").rglob("*") if _.is_file()) <= 30,
        dict(files=bd2, ids_moi_bundle=[len(x["ids"]) for x in idx], n_files_working=sum(1 for _ in (FK / "working").rglob("*") if _.is_file())))

# S5 HF: thiếu token / token giả
TOK = "hf" + "_" + "BOGUS1234567890abcdefghijklmnopqrstuv"      # token GIẢ (dựng lúc chạy để không khớp mẫu quét bí mật)
args = [PY, str(PACK / "w_runner_khung_chay.py"), "run", "--stub", "--workers", "2", "--stub-ms", "1", "--pack", str(PACK), "--out", str(R / "fake_kaggle" / "tmp_hf" / "tn11_full"),
        "--remote", "hf:fakeuser/gannhanocr-tn11-full-test"]
shutil.rmtree(R / "fake_kaggle" / "tmp_hf", ignore_errors=True)
p0 = subprocess.run(args, cwd=CWD, env=L.env(), capture_output=True, text=True, timeout=120)
rep.inv("S5a.thieu_token_thoat_1_thong_bao_ro_truoc_khi_ton_cong", p0.returncode == 1 and "Thiếu HF_TOKEN" in (p0.stdout + p0.stderr) and not (R / "fake_kaggle" / "tmp_hf" / "tn11_full" / "shards").exists(),
        dict(code=p0.returncode, tail=(p0.stdout + p0.stderr).strip()[-200:]))
p1 = subprocess.run(args, cwd=CWD, env=L.env({"HF_TOKEN": TOK}), capture_output=True, text=True, timeout=180)
txt = p1.stdout + p1.stderr
leaked_files = []
for f in (R / "fake_kaggle").rglob("*"):
    if f.is_file() and f.stat().st_size < 5_000_000 and f.suffix in (".json", ".jsonl", ".txt", ".log", ""):
        try:
            if TOK in f.read_text(encoding="utf-8", errors="ignore"):
                leaked_files.append(str(f.relative_to(R)))
        except Exception:  # noqa: BLE001
            pass
rep.inv("S5b.token_gia_khong_xuat_hien_trong_stdout_stderr_va_tep_dau_ra", TOK not in txt and not leaked_files,
        dict(code=p1.returncode, token_trong_log=TOK in txt, tep_ro_ri=leaked_files, cuoi_log=txt.strip()[-420:]))
rep.data("S5b_exit_va_loai_loi", dict(code=p1.returncode, last_lines=txt.strip().splitlines()[-4:]))
tmp_after = {p.name for p in Path("/tmp").glob("tn11*")}
rep.inv("S6.khong_tao_/kaggle_va_khong_dung_/tmp", not Path("/kaggle").exists() and tmp_after == tmp_before, dict(truoc=sorted(tmp_before), sau=sorted(tmp_after)))
rep.save()
