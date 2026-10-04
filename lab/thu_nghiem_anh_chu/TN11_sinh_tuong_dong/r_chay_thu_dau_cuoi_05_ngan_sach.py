"""BƯỚC 4 — ngân sách giờ: chạy CHÍNH worker_loop của gói với ĐỒNG HỒ ẢO (K.now bị thay) và bộ sinh giả tốn T giây ảo/việc, để biết
  (a) --budget-hours 6.5 --safety-s 1200 (mặc định) ca dừng lúc nào, mã thoát, bao nhiêu việc, có chạm mốc 12 giờ của Kaggle không;
  (b) ca THỬ của hướng dẫn (--max-tasks 2 --budget-hours 0.5, safety mặc định 1200 s) làm được mấy việc/worker tuỳ thời gian khởi động U (pip + nạp mô hình);
  (c) ca bị chậm đột ngột (GPU chậm 3x) và đồng hồ nạp mô hình dài: chạm trần 12 giờ không;
  (d) --est-s mặc định 2,66x128 so với tốc độ MPS đo thật (nếu có 03_mps.json).
Không GPU, không ngủ thật (đồng hồ ảo). Ra: .../review/chay_thu_dau_cuoi/05_ngan_sach.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import json
import math
import shutil
from pathlib import Path

import r_chay_thu_dau_cuoi_lib as L

K = L.load_runner("sim_runner")
rep = L.Rep("05_ngan_sach")
plan = K.load_plan(L.PACK)
R = L.R
SIM = R / "budget_sim"
if SIM.exists():
    shutil.rmtree(SIM)
SIM.mkdir(parents=True)
H = 3600.0
EST = 2.66 * 128                                   # --est-s mặc định (bộ chạy: default=2.66 * 128)


class Clock:
    t = 1_000_000.0


class SimGen:
    """Bộ sinh giả: mỗi lô tốn T(k) giây ĐỒNG HỒ ẢO; ảnh nhỏ để ghi nhanh."""
    name = "sim"

    def __init__(self, clock, T):
        self.clock, self.T, self.k = clock, T, 0

    def generate(self, task, chars, fonts, style_path, seed):
        self.clock.t += self.T(self.k, len(chars))
        self.k += 1
        return {c: b"\x89PNG" + bytes([ord(c) & 255]) * 8 for c in chars}, {}


def run_sim(budget_h: float, safety_s: float, startup_s: float, T, max_tasks: int = 0, est: float = EST, ttl: float = 1800.0):
    clock = Clock()
    K.now = lambda: clock.t                          # worker_loop/try_claim/cleanup_stale đều gọi K.now() (hàm mức mô-đun)
    t_start = clock.t
    deadline = t_start + budget_h * H - safety_s
    clock.t += startup_s                             # pip install + chép ckpt + nạp mô hình + CUDA init
    out = SIM / f"o{len(list(SIM.iterdir()))}"
    out.mkdir(parents=True)
    g = SimGen(clock, T)
    r = K.worker_loop(plan, L.PACK, out, "w0", g, deadline, est, ttl, 0, log=lambda *a: None, max_tasks=max_tasks)
    rows = L.read_jsonl(out / "status" / "tasks-w0.jsonl")
    last_end = max((x["t"] for x in rows), default=t_start + startup_s) - t_start
    shutil.rmtree(out)
    return dict(tasks=r["tasks"], stop=r["stop"], dung_luc_h=round(last_end / H, 3), han_dung_h=round((deadline - t_start) / H, 3), vuot_han_s=round(last_end - (deadline - t_start), 1),
                con_lai_toi_het_ngan_sach_phut=round((budget_h * H - last_end) / 60, 1), con_lai_toi_12h_phut=round((12 * H - last_end) / 60, 1))


T0 = lambda k, n: n * 2.66                         # đúng est (T4: 2,66 s/ảnh, 128 chữ = 340,48 s)

# (b) ca THỬ của hướng dẫn: --max-tasks 2 --budget-hours 0.5 (safety 1200 mặc định), U thay đổi
trial = {}
for U in (0, 60, 120, 180, 205, 210, 240, 300, 420):
    trial[f"U={U}s"] = run_sim(0.5, 1200, U, T0, max_tasks=2)
rep.data("ca_thu_theo_huong_dan__moi_worker", trial)
first_check_ok_until = 0.5 * H - 1200 - 1.15 * EST
rep.data("ca_thu__U_toi_da_de_nhan_viec_dau_s", round(first_check_ok_until, 1))
rep.inv("B1.ca_thu_huong_dan_(0,5h_va_safety_1200)_khong_du_cho_2_viec/worker", all(v["tasks"] < 2 for v in trial.values()), dict(U_toi_da_de_nhan_viec_dau=round(first_check_ok_until, 1), so_viec_moi_worker={k: v["tasks"] for k, v in trial.items()}))
rep.inv("B1b.ca_thu_co_the_0_viec_neu_khoi_dong>208s", trial["U=240s"]["tasks"] == 0 and trial["U=240s"]["stop"] == "budget" and trial["U=0s"]["tasks"] == 1, dict(U0=trial["U=0s"], U240=trial["U=240s"]))
# cách sửa đề xuất
fix = {}
for budget, safety in ((0.5, 0), (0.75, 120), (1.0, 300), (1.0, 1200)):
    fix[f"budget={budget}h_safety={safety}s"] = {f"U={U}": run_sim(budget, safety, U, T0, max_tasks=2)["tasks"] for U in (120, 240, 420)}
rep.data("ca_thu_cach_sua_de_xuat__so_viec_moi_worker", fix)
rep.inv("B1c.ca_thu_sua_(--budget-hours 0,75 --safety-s 120)_du_2_viec/worker_voi_U<=420s", all(v == 2 for v in fix["budget=0.75h_safety=120s"].values()), fix["budget=0.75h_safety=120s"])

# (a) ca thật 6,5 h, safety 1200: U = 240 s, T = est
real = {}
for U in (120, 240, 480):
    real[f"6.5h/U={U}s"] = run_sim(6.5, 1200, U, T0)
rep.data("ca_that_6.5h__moi_worker", real)
r = real["6.5h/U=240s"]
rep.inv("B2.ca_that_6.5h_dung_sach_stop=budget_va_xa_12h", r["stop"] == "budget" and r["vuot_han_s"] <= 0 and r["con_lai_toi_12h_phut"] > 300, r)
per_worker = r["tasks"]
n_sess = math.ceil(410 / (2 * per_worker))
rep.data("so_ca_can_voi_6.5h", dict(viec_moi_worker=per_worker, viec_moi_ca_2_GPU=2 * per_worker, so_ca_can=n_sess, ca_cuoi_viec_con_lai=410 - (n_sess - 1) * 2 * per_worker,
                                    gio_phien_tong_uoc=round(((n_sess - 1) * (240 + per_worker * EST) + (240 + math.ceil((410 - (n_sess - 1) * 2 * per_worker) / 2) * EST)) / H, 2)))
rep.inv("B2b.so_ca_can_voi_6.5h_la_4_(huong_dan_noi_~3)", n_sess == 4, rep.d["du_lieu"]["so_ca_can_voi_6.5h"])
# (c) mặc định 11 h và chậm đột ngột 3x sau 10 việc
slow = lambda k, n: n * 2.66 * (3.0 if k >= 10 else 1.0)
res_slow = {f"{b}h": run_sim(b, 1200, 480, slow) for b in (6.5, 11.0)}
rep.data("cham_3x_sau_10_viec", res_slow)
rep.inv("B3.cham_3x_van_xa_12h_(6.5h_va_11h)", all(v["con_lai_toi_12h_phut"] > 30 for v in res_slow.values()), res_slow)
jit = lambda k, n: n * 2.66 * (1.0 + 0.4 * math.sin(k * 1.7))
res_j = {f"{b}h": run_sim(b, 1200, 240, jit) for b in (6.5, 11.0)}
rep.data("dao_dong_±40%", res_j)
rep.inv("B3b.dao_dong_±40%_vuot_han_toi_da_<_1_viec", all(v["vuot_han_s"] < EST for v in res_j.values()), {k: v["vuot_han_s"] for k, v in res_j.items()})
# lãng phí do safety + hệ số 1,15
waste = {b: round((1200 + 1.15 * EST) / 60, 1) for b in (0.5, 6.5, 11)}
rep.data("phut_bo_phi_toi_da_do_safety_va_1.15xema", waste)

# (d) est-s mặc định vs MPS đo thật (nếu có)
mp = R / "03_mps.json"
if mp.exists():
    d = json.loads(mp.read_text(encoding="utf-8"))
    rep.data("mps_do_that", d.get("du_lieu", {}).get("toc_do"))
rep.save()
