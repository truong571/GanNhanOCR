"""r_huong_dan_03_ca_mo_phong.py — REVIEW hướng "huong_dan" (3/5): hướng dẫn nói ca THỬ "--max-tasks 2 --budget-hours 0.5" cho
stop_reason = max_tasks + 4 tệp tar, và "3 ca ~6,5 giờ" (--budget-hours 6.5). Kiểm bằng CHÍNH worker_loop của bộ chạy (đồng hồ giả +
bộ sinh giả, 0 GPU) và bằng CHÍNH `run` (--stub, tỉ lệ thời gian 1/60). Mọi tham số giả định (khởi động E0, giây/ảnh) in rõ trong JSON.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_huong_dan_03_ca_mo_phong.py
Ra: measure_out/_tn11/full/review/huong_dan/03_ca_mo_phong.json (+ real_stub_*/ nhỏ, sim_tmp/ bị xoá khi xong)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import importlib.util
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FULL = REPO / "measure_out" / "_tn11" / "full"
OUT = FULL / "review" / "huong_dan"
PACK = FULL / "kaggle_pack_20261004"
RUNNER = HERE / "w_runner_khung_chay.py"
SIM = OUT / "sim_tmp"

spec = importlib.util.spec_from_file_location("K", RUNNER)
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)

R = {"tao": time.strftime("%Y-%m-%d %H:%M:%S"), "invariants": {}, "gia_dinh": {}}
PLAN = K.load_plan(PACK)
TASKS = PLAN["tasks"]
CLOCK = [0.0]
BASE = 1_000_000.0
PRE_S, POST_S = 90.0, 120.0       # GIẢ ĐỊNH: khởi động phiên Kaggle -> ô code (90 s) và đẩy/ghi cuối ca (120 s) — chưa đo


def inv(name, ok, detail=None):
    R["invariants"][name] = dict(ok=bool(ok), detail=detail)
    print(("PASS" if ok else "FAIL"), name, "" if detail is None else json.dumps(detail, ensure_ascii=False, default=str)[:300], flush=True)


class FakeGen:
    def __init__(self, sec_per_img):
        self.s = sec_per_img

    def generate(self, task, chars, fonts, style_path, seed):
        CLOCK[0] += len(chars) * self.s
        return {c: b"x" for c in chars}, {}


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def patch():
    K.now = lambda: CLOCK[0]
    K.write_shard = lambda out, task, plan, images, skipped: _touch(out / K.rel_path(task["id"]))


def stop_reason(codes, stops, all_done):
    """CHÉP biểu thức ở w_runner_khung_chay.py:cmd_run (dòng 'stop_reason=')."""
    return "complete" if all_done else ("guard" if 2 in codes else ("max_tasks" if "max_tasks" in stops.values() else ("budget" if "budget" in stops.values() else "crash_or_error")))


def session(out: Path, budget_h, e0, sec, max_tasks=0, safety_s=1200.0, est_s=2.66 * 128, ngpu=2):
    """Một phiên `run` bằng worker_loop thật: mỗi worker bắt đầu nhận việc sau e0 giây kể từ đầu cmd_run; các worker chạy lần lượt (đếm việc
    của mỗi worker chỉ phụ thuộc đồng hồ + ema của chính nó nên chạy tuần tự cho cùng tổng số việc)."""
    patch()
    deadline = BASE + budget_h * 3600 - safety_s
    rs, ends = [], []
    for k in range(ngpu):
        CLOCK[0] = BASE + e0
        r = K.worker_loop(PLAN, PACK, out, f"w{k}", FakeGen(sec), deadline, est_s, 1800.0, 0, log=lambda *a, **kw: None, max_tasks=max_tasks)
        rs.append(r)
        ends.append(CLOCK[0] - BASE)
    done = K.done_ids(out)
    all_done = {t["id"] for t in TASKS} <= done
    stops = {f"w{k}": r["stop"] for k, r in enumerate(rs)}
    return dict(tasks=[r["tasks"] for r in rs], images=sum(r["images"] for r in rs), stops=stops, done_total=len(done), all_done=all_done,
                stop_reason=stop_reason([0] * ngpu, stops, all_done), wall_s=round(max(ends), 1), quota_h=round((PRE_S + max(ends) + POST_S) / 3600, 3))


def pilot_table(budget_h, max_tasks, e0s, secs):
    rows = []
    for sec in secs:
        for e0 in e0s:
            out = SIM / f"pilot_{budget_h}_{max_tasks}_{sec}_{e0}"
            shutil.rmtree(out, ignore_errors=True)
            s = session(out, budget_h, e0, sec, max_tasks=max_tasks)
            rows.append(dict(giay_moi_anh=sec, khoi_dong_E0_s=e0, viec_moi_worker=s["tasks"], tar=s["done_total"], dung=s["stops"], stop_reason=s["stop_reason"],
                             dung_nhu_huong_dan=(s["stop_reason"] == "max_tasks" and s["done_total"] == 4)))
            shutil.rmtree(out, ignore_errors=True)
    return rows


def campaign(budget_h, e0, sec, safety_s=1200.0, max_ca=12):
    out = SIM / f"camp_{budget_h}_{e0}_{sec}"
    shutil.rmtree(out, ignore_errors=True)
    cas, quota = [], 0.0
    for k in range(1, max_ca + 1):
        s = session(out, budget_h, e0, sec, safety_s=safety_s)
        cas.append(dict(ca=k, viec=sum(s["tasks"]), tich_luy=s["done_total"], gio_chay=round(s["wall_s"] / 3600, 2), gio_quota=s["quota_h"]))
        quota += s["quota_h"]
        if s["all_done"]:
            break
    shutil.rmtree(out, ignore_errors=True)
    return dict(budget_h=budget_h, E0_s=e0, giay_moi_anh=sec, so_ca=len(cas) if cas[-1]["tich_luy"] == len(TASKS) else None, cac_ca=cas, tong_gio_quota=round(quota, 2),
                xong=cas[-1]["tich_luy"] == len(TASKS))


def min_budget_for(n_ca, e0, sec, lo=1.0, hi=12.0):
    """Ngân sách (giờ) nhỏ nhất, bước 0,05, để xong trong <= n_ca ca."""
    b = lo
    while b <= hi + 1e-9:
        c = campaign(round(b, 2), e0, sec, max_ca=n_ca)
        if c["xong"]:
            return round(b, 2), c["tong_gio_quota"]
        b += 0.05
    return None, None


def scaled_real(tag, budget_h, max_tasks, scale=60.0):
    """Chạy CHÍNH `run --stub` với thời gian chia `scale`: 345,6 s/việc -> 5,76 s; an toàn 1200 s -> 20 s; est_s 340,48 -> 5,675."""
    out = OUT / f"real_stub_{tag}"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [sys.executable, str(RUNNER), "run", "--pack", str(PACK), "--out", str(out), "--stub", "--stub-ms", str(round(2.7 * 1000 / scale, 3)),
           "--est-s", str(round(2.66 * 128 / scale, 3)), "--budget-hours", str(budget_h / scale), "--safety-s", str(round(1200 / scale, 3)), "--max-tasks", str(max_tasks),
           "--workers", "2", "--sync-every", "3600"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(__import__("os").environ, PYTHONDONTWRITEBYTECODE="1"))
    st = json.loads((out / "status" / "status.json").read_text(encoding="utf-8"))
    n_tar = len(list((out / "shards").rglob("*.tar")))
    return dict(tag=tag, lenh_goc=f"--max-tasks {max_tasks} --budget-hours {budget_h} (thời gian chia {int(scale)})", exit=p.returncode, giay_thuc=round(time.time() - t0, 1),
                tar=n_tar, worker_exit_codes=st["worker_exit_codes"], worker_stop=st["worker_stop"], stop_reason=st["stop_reason"], complete=st["complete"])


def main():
    SIM.mkdir(parents=True, exist_ok=True)
    R["gia_dinh"] = dict(giay_moi_anh_huong_dan=2.7, giay_moi_anh_nhanh_gia_dinh=1.4, est_s_mac_dinh=round(2.66 * 128, 2), safety_s_mac_dinh=1200.0, PRE_s=PRE_S, POST_s=POST_S,
                         ghi_chu="E0 = từ đầu `run` tới lúc worker nhận việc đầu (pip + chép gói + kiểm băm ckpt + nạp mô hình). CHƯA ĐO trên Kaggle; quét 60–600 s.")

    # ---- 1. đọc tham số mặc định thật từ argparse (không tin trí nhớ)
    src = RUNNER.read_text(encoding="utf-8")
    inv("P0.mac_dinh_safety_s_1200_est_s_2_66x128_gen_bs_32", 'default=1200.0' in src and "default=2.66 * 128" in src and '"--gen-bs", type=int, default=32' in src, None)

    # ---- 2. ca THỬ đúng như hướng dẫn: --max-tasks 2 --budget-hours 0.5
    tab = pilot_table(0.5, 2, [60, 120, 180, 207, 210, 240, 300, 420], [2.7, 1.4])
    R["ca_thu_dung_nhu_huong_dan"] = tab
    n_ok = sum(r["dung_nhu_huong_dan"] for r in tab)
    inv("P1.ca_thu_0.5h_2.7s_KHONG_bao_gio_cho_4_tar_max_tasks", not any(r["dung_nhu_huong_dan"] for r in tab if r["giay_moi_anh"] == 2.7), [r for r in tab if r["giay_moi_anh"] == 2.7][:4])
    inv("P1b.ca_thu_0.5h_voi_E0_2_phut_tro_len_it_nhat_mot_worker_bi_dung_vi_budget", all(r["stop_reason"] == "budget" for r in tab if r["khoi_dong_E0_s"] >= 120 and r["giay_moi_anh"] == 2.7), None)
    first_fail = next((r["khoi_dong_E0_s"] for r in tab if r["giay_moi_anh"] == 2.7 and sum(r["viec_moi_worker"]) == 0), None)
    R["ca_thu_E0_dau_tien_khong_lam_duoc_viec_nao_s"] = first_fail
    R["ca_thu_nguong_E0_cho_viec_dau_s"] = dict(cong_thuc="deadline - 1,15*est_s = 1800-1200-391,6", gia_tri=round(1800 - 1200 - 1.15 * 2.66 * 128, 1))
    inv("P1c.E0_lon_hon_208s_thi_worker_khong_nhan_viec_nao", first_fail is not None and first_fail <= 240, first_fail)

    # ---- 3. đề xuất sửa: --budget-hours 1.0 (hoặc 0.75)
    tab2 = pilot_table(1.0, 2, [60, 120, 300, 600, 900], [2.7, 1.4])
    R["ca_thu_de_xuat_1h"] = tab2
    inv("P2.ca_thu_1h_cho_4_tar_va_max_tasks_moi_E0_den_900s", all(r["dung_nhu_huong_dan"] for r in tab2), [r["stop_reason"] for r in tab2])
    tab3 = pilot_table(0.75, 2, [60, 120, 300, 480], [2.7])
    R["ca_thu_de_xuat_0_75h"] = tab3
    inv("P2b.ca_thu_0.75h_dung_voi_E0_den_300s_nhung_E0_480s_stop_reason_thanh_budget_vi_kiem_budget_dung_TRUOC_kiem_max_tasks",
        [r["stop_reason"] for r in tab3] == ["max_tasks", "max_tasks", "max_tasks", "budget"] and [r["tar"] for r in tab3] == [4, 4, 4, 4], [(r["khoi_dong_E0_s"], r["stop_reason"], r["tar"]) for r in tab3])
    # thứ tự kiểm trong worker_loop: stop_file -> budget -> max_tasks (đọc thẳng từ mã nguồn)
    i_stop, i_bud, i_max = src.index('stop = "stop_file"'), src.index('stop = "budget"'), src.index('stop = "max_tasks"')
    inv("P2c.worker_loop_kiem_budget_truoc_max_tasks", i_stop < i_bud < i_max, dict(stop_file=src.count("\n", 0, i_stop) + 1, budget=src.count("\n", 0, i_bud) + 1, max_tasks=src.count("\n", 0, i_max) + 1))

    # ---- 4. ca THẬT: 3 ca x 6,5 giờ?
    camp = {}
    for e0 in (180, 600):
        for sec in (2.7, 1.4):
            camp[f"E0={e0}s,{sec}s/anh"] = campaign(6.5, e0, sec)
    R["ba_ca_6_5_gio"] = camp
    c27 = camp["E0=180s,2.7s/anh"]
    inv("C1.ba_ca_x_6.5h_KHONG_du_cho_410_viec_o_2.7s_anh", all(not (v["xong"] and len(v["cac_ca"]) <= 3) for k, v in camp.items() if k.endswith("2.7s/anh")),
        {k: dict(so_ca=len(v["cac_ca"]), tich_luy_sau_3_ca=v["cac_ca"][2]["tich_luy"], quota_h=v["tong_gio_quota"]) for k, v in camp.items() if k.endswith("2.7s/anh")})
    R["ba_ca_6_5_gio_tom_tat"] = {k: dict(so_ca=len(v["cac_ca"]), viec_moi_ca=[c["viec"] for c in v["cac_ca"]], gio_chay_moi_ca=[c["gio_chay"] for c in v["cac_ca"]], tong_quota_h=v["tong_gio_quota"]) for k, v in camp.items()}

    # ---- 5. ngân sách nhỏ nhất cho 3 / 2 ca
    mb = {}
    for e0 in (180, 600):
        for n_ca in (3, 2):
            b, q = min_budget_for(n_ca, e0, 2.7)
            mb[f"{n_ca}_ca,E0={e0}s,2.7s/anh"] = dict(budget_hours_toi_thieu=b, tong_quota_h=q)
    R["ngan_sach_nho_nhat"] = mb
    # bảng theo ngân sách
    sweep = {}
    for b in (6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0):
        c = campaign(b, 300, 2.7)
        sweep[str(b)] = dict(so_ca=len(c["cac_ca"]), viec_moi_ca=[x["viec"] for x in c["cac_ca"]], tong_quota_h=c["tong_gio_quota"])
    R["quet_ngan_sach_E0_300s_2.7s"] = sweep
    inv("C2.11h_thi_2_ca_la_du", sweep["11.0"]["so_ca"] == 2, sweep["11.0"])

    # ---- 6. chạy CHÍNH `run --stub` (thời gian chia 60)
    R["run_stub_ty_le_1_60"] = [scaled_real("pilot_huong_dan_0_5h", 0.5, 2), scaled_real("pilot_de_xuat_1h", 1.0, 2)]
    a, b = R["run_stub_ty_le_1_60"]
    inv("S1.run_stub_dung_huong_dan_0.5h_cho_stop_reason_budget_va_2_tar", a["stop_reason"] == "budget" and a["tar"] == 2, a)
    inv("S2.run_stub_de_xuat_1h_cho_stop_reason_max_tasks_va_4_tar", b["stop_reason"] == "max_tasks" and b["tar"] == 4, b)

    # ---- 7. STOP: hành vi thật của mã
    inv("T1.run_xoa_STOP_cu_o_dau_phien", re.search(r'if \(out / "STOP"\)\.exists\(\):\s*\n\s*\(out / "STOP"\)\.unlink\(\)', src) is not None, "cmd_run xoá STOP tồn tại từ trước")
    inv("T2.worker_chi_kiem_STOP_truoc_moi_viec_moi", src.count("stopfile.exists()") == 1 and "stop = \"stop_file\"" in src, "dừng ở ranh giới việc: trễ tối đa một lô (~340 s)")
    # STOP thật qua worker_loop: tạo tệp STOP từ đầu -> không nhận việc
    out = SIM / "stopfile"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    (out / "STOP").write_text("x")
    patch()
    CLOCK[0] = BASE
    r = K.worker_loop(PLAN, PACK, out, "w0", FakeGen(2.7), BASE + 99999, 340.0, 1800.0, 0, log=lambda *a, **kw: None)
    inv("T3.worker_voi_tep_STOP_dung_stop_file_0_viec", r["stop"] == "stop_file" and r["tasks"] == 0, r)
    shutil.rmtree(SIM, ignore_errors=True)

    R["tong"] = dict(pass_=sum(v["ok"] for v in R["invariants"].values()), fail=sum(not v["ok"] for v in R["invariants"].values()), n=len(R["invariants"]),
                     fail_ten=[k for k, v in R["invariants"].items() if not v["ok"]])
    (OUT / "03_ca_mo_phong.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("TỔNG:", R["tong"])


if __name__ == "__main__":
    main()
