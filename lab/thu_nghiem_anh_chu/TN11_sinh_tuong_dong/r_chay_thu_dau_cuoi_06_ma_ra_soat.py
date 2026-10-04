"""BƯỚC 5 — soát mã bộ chạy bằng phép thử có đối chứng (CPU, giả lập; KHÔNG GPU, KHÔNG token thật):
  E1 try_claim: hết hạn TTL -> cướp khoá; hai luồng cùng cướp một khoá hết hạn (cửa sổ đua ABA: cả hai thắng?);
     tỉ số TTL / thời lượng một việc thật; việc dài hơn TTL làm hai worker làm trùng;
  E2 Syncer: stop() trong lúc luồng nền đang đẩy -> đẩy trùng? stop() có hạn chờ không (đẩy treo)? thử lại + chống đẩy trùng sau khi thành công;
  E3 Bundler: stop() trong lúc luồng nền đang gói -> trùng tên .part/ chỉ mục lệch tar? số tệp đầu ra mỗi ca 11 giờ;
  E4 HFRemote với thư viện huggingface_hub THẬT (ẩn danh, repo công khai, chỉ đọc): get_text() cho tệp không tồn tại có trả None? list_done() chạy được?
  E5 mô-đun bên thứ ba mà worker THỰC SỰ nạp (so với danh sách PIP của bộ chạy + ảnh Kaggle thông thường);
  E6 không có GPU: --workers 0 không --device -> rơi về CPU? worker hỏng lúc nạp -> mã thoát của `run`;
  E7 cmd_run in những gì ra log (env.json có in không? tasks-w*.jsonl?) — đối chiếu với mục "Kết quả mong đợi" của hướng dẫn;
  E8 `hf` CLI (hướng dẫn mục 2 bước 6) có trong venv/PATH không.
Ra: .../review/chay_thu_dau_cuoi/06_ma_ra_soat.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
from pathlib import Path

import r_chay_thu_dau_cuoi_lib as L

K = L.load_runner("rv_runner")
rep = L.Rep("06_ma_ra_soat")
R, PACK = L.R, L.PACK
W = R / "rv_work"
if W.exists():
    shutil.rmtree(W)
W.mkdir(parents=True)
os.chdir(R / "mps_cwd")
plan = K.load_plan(PACK)

# ------------------------------------------------------------------------------------------------ E1 try_claim
ld = W / "locks1"
ok1 = K.try_claim(ld, "stt2/base_s0/00000", "wA", 0.3)
ok2 = K.try_claim(ld, "stt2/base_s0/00000", "wB", 0.3)
time.sleep(0.45)
ok3 = K.try_claim(ld, "stt2/base_s0/00000", "wB", 0.3)
stale_files = [p.name for p in ld.glob("*.stale-*")]
rep.inv("E1a.khoa_con_tre_tu_choi_het_han_thi_cuop_va_de_lai_tep_stale", ok1 and not ok2 and ok3 and len(stale_files) == 1, dict(a=ok1, b_ngay=ok2, b_sau_ttl=ok3, stale=stale_files))

# hai luồng cùng cướp một khoá hết hạn: đếm số vòng CẢ HAI cùng "thắng"
N, both, none_ = 4000, 0, 0
for i in range(N):
    d = W / "race" / f"r{i % 50}"
    tid = f"stt2/base_s0/{i:05d}"
    pth = d / (tid.replace("/", "__") + ".lock")
    d.mkdir(parents=True, exist_ok=True)
    pth.write_text("old")
    old = time.time() - 5000
    os.utime(pth, (old, old))                                  # khoá chết từ lâu
    bar = threading.Barrier(2)
    res = [None, None]

    def go(k):
        bar.wait()
        res[k] = K.try_claim(d, tid, f"w{k}", 1800.0)
    ts = [threading.Thread(target=go, args=(k,)) for k in (0, 1)]
    [t.start() for t in ts]; [t.join() for t in ts]
    both += (res[0] and res[1]); none_ += (not res[0] and not res[1])
rep.data("E1b_dua_cuop_khoa_het_han", dict(vong=N, ca_hai_thang=both, khong_ai_thang=none_))
rep.inv("E1b.hai_luong_cung_cuop_khoa_het_han_khong_bao_gio_ca_hai_thang", both == 0, dict(vong=N, ca_hai_thang=both, khong_ai_thang=none_, ghi_chu="cả hai thắng = hai worker làm trùng MỘT việc (kết quả vẫn đúng: tar tất định + ghi nguyên tử)"))
ttl, est = 1800.0, 2.66 * 128
rep.data("E1c_ti_so_TTL_tren_thoi_luong_viec", dict(ttl_s=ttl, viec_T4_s=round(est, 1), ti_so=round(ttl / est, 2), viec_toi_da_an_toan_s=ttl))

# việc dài hơn TTL: hai luồng worker_loop (đồng hồ thật) trên kế hoạch thật; worker A CHẬM (mỗi việc 128 x 20 ms = 2,6 s), worker B nhanh (0,13 s)
def dup_demo(ttl_s):
    out = W / f"dup_{ttl_s}"
    gens = [K.StubGen(20.0), K.StubGen(1.0)]
    res = [None, None]
    sub = dict(plan); sub["tasks"] = plan["tasks"][:6]

    def go(k):
        res[k] = K.worker_loop(sub, PACK, out, f"w{k}", gens[k], K.now() + 600, 0.1, ttl_s, 0, log=lambda *a: None)
    ts = [threading.Thread(target=go, args=(k,)) for k in (0, 1)]
    [t.start() for t in ts]; [t.join() for t in ts]
    rows = L.task_rows(out)
    from collections import Counter
    c = Counter(r["id"] for r in rows)
    return dict(viec=6, lan_lam=len(rows), viec_bi_lam_2_lan=sum(1 for v in c.values() if v > 1))
d_short = dup_demo(0.5)                                         # TTL 0,5 s < 2,6 s/việc của worker chậm
d_long = dup_demo(1800.0)
rep.inv("E1d.trong_mot_ca_khong_viec_nao_bi_lam_2_lan_(worker_duyet_ke_hoach_MOT_luot)_ke_ca_khi_TTL<<thoi_luong_viec", d_short["viec_bi_lam_2_lan"] == 0 and d_long["viec_bi_lam_2_lan"] == 0,
        dict(TTL_0_5s_viec_2_6s=d_short, TTL_1800s=d_long, giai_thich="việc bị khoá khi gặp thì bị bỏ qua và không quay lại trong cùng ca; cướp khoá hết hạn chỉ xảy ra ở ca sau (đĩa mới nên không có khoá)"))

# ------------------------------------------------------------------------------------------------ E2 Syncer
class FakeRemote:
    kind = "fake"

    def __init__(self, delay=0.0, fail_first=0, hang=0.0):
        self.calls, self.delay, self.fail_first, self.hang, self.lock = [], delay, fail_first, hang, threading.Lock()

    def push(self, local_root, rels, message=""):
        t0 = time.time()
        if self.hang:
            time.sleep(self.hang)
        time.sleep(self.delay)
        with self.lock:
            if self.fail_first > 0:
                self.fail_first -= 1
                raise ConnectionError("giả: mạng đứt")
            self.calls.append((t0, list(rels)))


def mk_shards(out, names):
    for n in names:
        p = out / "shards" / "stt2" / "base_s0" / f"{n:05d}.tar"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 100)


o = W / "sy1"; mk_shards(o, range(4))
fr = FakeRemote(delay=0.8)
sy = K.Syncer(o, fr, 0.1, set(), log=lambda *a: None, backoff=0.01)
sy.start()
time.sleep(0.3)                                            # luồng nền đang giữa lần đẩy
fin = sy.stop()                                            # stop() gọi flush() ở luồng chính ngay lúc đó
from collections import Counter  # noqa: E402

cnt = Counter(r for _, rels in fr.calls for r in rels)
rep.data("E2a_dua_flush_stop", dict(so_lan_push=len(fr.calls), dem_theo_tep=dict(cnt), sy_ok=fin))
rep.inv("E2a.stop_trong_luc_luong_nen_dang_day_KHONG_day_trung", len(fr.calls) >= 1 and max(cnt.values()) == 1, dict(so_lan_push=len(fr.calls), tep_day_nhieu_hon_1_lan=sum(1 for v in cnt.values() if v > 1), tong_tep=len(cnt)))
# thử lại + chống đẩy trùng
o = W / "sy2"; mk_shards(o, range(3))
fr2 = FakeRemote(fail_first=2)
sy2 = K.Syncer(o, fr2, 999, set(), log=lambda *a: None, backoff=0.01)
ok_a = sy2.flush(); ok_b = sy2.flush()
rep.inv("E2b.thu_lai_sau_2_loi_roi_thanh_cong_va_lan_hai_khong_day_lai", ok_a and ok_b and len(fr2.calls) == 1 and sy2.fail == 2, dict(calls=len(fr2.calls), fail=sy2.fail))
# stop() có hạn chờ? (đẩy treo): chỉ ghi nhận hành vi + timeout mặc định của huggingface_hub cài sẵn
o = W / "sy3"; mk_shards(o, range(2))
fr3 = FakeRemote(hang=3.0)
sy3 = K.Syncer(o, fr3, 999, set(), log=lambda *a: None, backoff=0.01)
t0 = time.time(); sy3.stop(); dt = time.time() - t0
try:
    import huggingface_hub.utils._http as _hh
    import httpx
    cli = _hh.default_client_factory()
    to = cli.timeout
    hf_to = dict(connect=to.connect, read=to.read, write=to.write, pool=to.pool)
except Exception as e:  # noqa: BLE001
    hf_to = f"không đọc được: {type(e).__name__}"
rep.data("E2c_stop_khi_push_treo", dict(giay_stop_khi_push_treo_3s=round(dt, 1), timeout_mac_dinh_httpx_cua_huggingface_hub=hf_to,
                                       ket_luan="Syncer.stop()/flush() không có hạn chờ; huggingface_hub (bản cài ở venv) dựng httpx.Client(timeout=None) -> kết nối treo = chặn vô hạn (chưa tái hiện treo thật trên HF)"))

# ------------------------------------------------------------------------------------------------ E3 Bundler
o = W / "bd1"; mk_shards(o, range(5))
bdir = W / "bd1_out"
orig = K.bundle_new


def slow_bundle(*a, **k):
    time.sleep(0.6)
    return orig(*a, **k)


K.bundle_new = slow_bundle
logs = []
bu = K.Bundler(o, bdir, "S", 0.1, set(), plan["sha"], log=lambda *a: logs.append(" ".join(map(str, a))))
bu.start()
time.sleep(0.25)                                          # luồng nền đang giữa lần gói
bu.stop()
time.sleep(0.9)
K.bundle_new = orig
tars = sorted(bdir.glob("bundle_*.tar"))
idxs = sorted(bdir.glob("bundle_*.index.json"))
bad_idx = []
for ip in idxs:
    ix = json.loads(ip.read_text())
    tp = ip.with_name(ix["tar"])
    names = []
    if tp.exists():
        with tarfile.open(tp) as tf:
            names = [m.name for m in tf.getmembers()]
    if (not tp.exists()) or tp.stat().st_size != ix["tar_bytes"] or sorted(names) != sorted(K.rel_path(i) for i in ix["ids"]):
        bad_idx.append(ip.name)
rep.data("E3a_dua_flush_stop", dict(tars=[p.name for p in tars], index=[p.name for p in idxs], loi_log=logs[:3], chi_muc_lech=bad_idx, file_con_lai=sorted(p.name for p in bdir.iterdir())))
rep.inv("E3a.stop_trong_luc_luong_nen_dang_goi_khong_lech_tar_chi_muc_khong_loi", not bad_idx and not logs and len(tars) == len(idxs) == 1, dict(tars=len(tars), idx=len(idxs), loi=logs[:2], lech=bad_idx))
# số tệp đầu ra một ca 11 giờ (mỗi 3600 s một bundle + bundle cuối), cộng log font_diffusion ở cwd
n_bundles = 11 + 1
rep.data("E3b_so_tep_ca_11h", dict(bundle_tar=n_bundles, index=n_bundles, log_font_diffusion=2, tong=2 * n_bundles + 2, tran_mem_cua_huong_dan=30))

# ------------------------------------------------------------------------------------------------ E4 HFRemote với huggingface_hub thật (ẩn danh, chỉ đọc)
try:
    import huggingface_hub as hh
    hf = object.__new__(K.HFRemote)
    hf.api = hh.HfApi(token=None)
    hf.repo_id, hf.repo_type, hf.token = "huggingface/documentation-images", "dataset", None
    try:
        miss = hf.get_text("__tn11_khong_ton_tai__.txt")
        rep.inv("E4a.HFRemote.get_text_tep_khong_co_tra_None_(EntryNotFoundError_bat_dung)", miss is None, dict(hf_hub=hh.__version__, ket_qua=miss))
    except Exception as e:  # noqa: BLE001
        rep.inv("E4a.HFRemote.get_text_tep_khong_co_tra_None_(EntryNotFoundError_bat_dung)", False, dict(loi=f"{type(e).__name__}: {str(e)[:200]}"))
    try:
        ld_ = hf.list_done()
        rep.inv("E4b.HFRemote.list_done_chay_duoc_voi_list_repo_files_that", isinstance(ld_, set), dict(so_tar_khop=len(ld_)))
    except Exception as e:  # noqa: BLE001
        rep.inv("E4b.HFRemote.list_done_chay_duoc_voi_list_repo_files_that", False, dict(loi=f"{type(e).__name__}: {str(e)[:200]}"))
except Exception as e:  # noqa: BLE001
    rep.inv("E4.import_huggingface_hub", False, dict(loi=f"{type(e).__name__}: {str(e)[:200]}"))
# E4c allow_patterns = danh sách đường dẫn đúng (HFRemote.push) có lọc đúng tập shards/*.tar (hàm lọc THẬT của huggingface_hub, chạy offline)
try:
    from huggingface_hub.utils import filter_repo_objects
    tree = W / "e4c"
    for f in ("shards/stt2/base_s0/00000.tar", "shards/stt2/base_s0/00001.tar", "shards/B34/base_s0/00000.tar", "shards/stt2/base_s0/.tmp-123-00002.tar.part",
              "locks/stt2__base_s0__00002.lock", "status/env.json", "status/tasks-w0.jsonl"):
        (tree / f).parent.mkdir(parents=True, exist_ok=True)
        (tree / f).write_bytes(b"x")
    rels = sorted(q.relative_to(tree).as_posix() for q in (tree / "shards").rglob("*.tar"))
    allp = [q.relative_to(tree).as_posix() for q in tree.rglob("*") if q.is_file()]
    sel = sorted(filter_repo_objects(allp, allow_patterns=rels))
    specials = [t["id"] for t in plan["tasks"] if any(ch in t["id"] for ch in "*?[]")]
    rep.inv("E4c.allow_patterns=danh_sach_duong_dan_loc_dung_chi_cac_tar_hoan_chinh_va_id_khong_co_ky_tu_glob", sel == rels and not specials, dict(chon=sel, id_co_ky_tu_glob=specials[:3]))
except Exception as e:  # noqa: BLE001
    rep.inv("E4c.allow_patterns=danh_sach_duong_dan_loc_dung_chi_cac_tar_hoan_chinh_va_id_khong_co_ky_tu_glob", False, dict(loi=f"{type(e).__name__}: {str(e)[:200]}"))

# ------------------------------------------------------------------------------------------------ E5 mô-đun bên thứ ba thực sự nạp
code_probe = r"""
import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, %r); sys.path.insert(0, %r)
import fd_wrapper_fix
import inference.sample_optimized as so
from src.tools.utils import ttf2im
std = set(sys.stdlib_module_names)
mods = sorted({m.split('.')[0] for m in sys.modules if not m.startswith('_')} - std)
print(json.dumps(mods))
""" % (str(PACK / "code"), str(PACK / "code" / "font_diffusion"))
pr = subprocess.run([sys.executable, "-c", code_probe], cwd=R / "mps_cwd", env=L.env(), capture_output=True, text=True)
third = []
for line in pr.stdout.splitlines()[::-1]:
    if line.startswith("["):
        third = json.loads(line); break
local_mods = {"core", "src", "inference", "fd_wrapper_fix"}
third = [m for m in third if m not in local_mods]
pip_mods = set(K.PIP)
rep.data("E5_mo_dun_ben_thu_ba_that_su_nap", third)
not_in_pip = [m for m in third if m not in pip_mods]
rep.data("E5_khong_nam_trong_PIP_cua_bo_chay", not_in_pip)
common = {"torch", "torchvision", "numpy", "PIL", "yaml", "tqdm", "packaging", "filelock", "requests", "regex", "psutil", "typing_extensions", "fsspec", "jinja2", "markupsafe", "mpmath", "networkx", "sympy",
          "certifi", "idna", "urllib3", "charset_normalizer", "httpx", "httpcore", "anyio", "h11", "sniffio", "click", "hf_xet", "shellingham", "typer", "rich", "pygments", "markdown_it", "mdurl", "annotated_doc",
          "importlib_metadata", "zipp", "attr", "attrs", "tokenizers", "setuptools", "pkg_resources", "wheel", "cloudpickle", "huggingface_hub", "safetensors", "diffusers", "accelerate", "kornia", "einops", "cv2",
          "skimage", "scipy", "fontTools", "pygame", "info_nce", "kornia_rs"}
unusual = [m for m in not_in_pip if m not in common]
rep.data("E5_ngoai_PIP_va_ngoai_ho_phu_thuoc_pho_bien", unusual)
rep.inv("E5.nap_code_goi_thanh_cong_(rc=0)_va_ghi_danh_sach_mo_dun_(thong_tin)", pr.returncode == 0 and len(third) > 5, dict(rc=pr.returncode, so_mo_dun=len(third), ngoai_PIP=len(not_in_pip), mau=not_in_pip[:12], cuoi_stderr=pr.stderr.strip()[-120:]))

# ------------------------------------------------------------------------------------------------ E6 không GPU
ws = R / "rv_work" / "nogpu"
TMPX = R / "cascade_tmp"
p6 = subprocess.run([sys.executable, str(L.RUNNER), "run", "--pack", str(PACK), "--workers", "0", "--tmp", str(TMPX), "--out", str(ws), "--budget-hours", "0.002", "--safety-s", "0"],
                    cwd=R / "mps_cwd", env=L.env(), capture_output=True, text=True, timeout=600) if (TMPX / "font_diffusion" / "src").exists() else None
if p6 is not None:
    txt = p6.stdout + p6.stderr
    st6 = L.status_of(ws) if (ws / "status" / "status.json").exists() else {}
    rep.inv("E6a.khong_GPU_va_khong_--device_thi_run_TU_CHOI_(khong_chay_im_lang_tren_CPU)", not ("on cpu" in txt and p6.returncode == 0), dict(code=p6.returncode, dong=[l for l in txt.splitlines() if "Loading FontDiffusion" in l][:2], status_stop=st6.get("worker_stop"), n_gpu_workers=len(st6.get("worker_exit_codes", []))))
    p7 = subprocess.run([sys.executable, str(L.RUNNER), "run", "--pack", str(PACK), "--workers", "2", "--device", "cuda", "--max-tasks", "1", "--tmp", str(TMPX), "--out", str(W / "badgpu")],
                        cwd=R / "mps_cwd", env=L.env(), capture_output=True, text=True, timeout=600)
    t7 = p7.stdout + p7.stderr
    st7 = L.status_of(W / "badgpu") if (W / "badgpu" / "status" / "status.json").exists() else {}
    rep.inv("E6b.hai_worker_chet_luc_nap_mo_hinh_thi_run_thoat_KHAC_0", not (p7.returncode == 0 and st7.get("tasks_done_local", 0) == 0 and not st7.get("complete")),
            dict(code=p7.returncode, worker_exit_codes=st7.get("worker_exit_codes"), stop_reason=st7.get("stop_reason"), loi=[l for l in t7.splitlines() if "Error" in l][:2]))
else:
    rep.data("E6", "bỏ qua: cascade_tmp chưa có (chạy script 04 trước)")

# ------------------------------------------------------------------------------------------------ E7 những gì cmd_run in ra
p8 = subprocess.run([sys.executable, str(L.RUNNER), "run", "--stub", "--stub-ms", "0.1", "--workers", "1", "--max-tasks", "1", "--pack", str(PACK), "--out", str(W / "e7")], cwd=R / "mps_cwd", env=L.env(), capture_output=True, text=True)
log8 = p8.stdout + p8.stderr
envj = json.loads((W / "e7" / "status" / "env.json").read_text(encoding="utf-8"))
vals = [str(v) for v in envj.values() if v]
in_log = [k for k, v in envj.items() if v and str(v) in log8]
rep.data("E7_log_run_mau", log8[-900:])
rep.inv("E7.env_json_(phien_ban_thu_vien,_GPU)_duoc_in_ra_log_de_doc_duoc_sau_ca_Save&RunAll", len(in_log) >= 2,
        dict(khoa_env_co_trong_log=in_log, ghi_chu="env.json + tasks-w*.jsonl nằm ở /kaggle/tmp/.../status (scratch, KHÔNG lưu sau ca; gioi_han_kaggle.json: scratch_ngoai_working_khong_luu); log chỉ có '[run] {status}' và '[worker ..] {tasks,images,stop,ema_s}'"))

# ------------------------------------------------------------------------------------------------ E8 hf CLI
which = shutil.which("hf")
venvhf = (L.REPO / ".venv" / "bin" / "hf").exists()
rep.data("E8_hf_cli", dict(trong_PATH=which, trong_venv=venvhf))
rep.inv("E8.lenh_hf_download_co_san_trong_PATH_hoac_venv", bool(which) or venvhf, dict(PATH=which, venv=venvhf))
rep.save()
