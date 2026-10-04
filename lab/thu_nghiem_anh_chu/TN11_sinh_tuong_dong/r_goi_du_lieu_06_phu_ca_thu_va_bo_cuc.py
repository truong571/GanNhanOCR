"""r_goi_du_lieu_06_phu_ca_thu_va_bo_cuc.py — HƯỚNG 5 (tiếp): CA THỬ có chạm các phần rủi ro không, và bộ chạy tìm gói ở bố cục nào.

  6a  `--max-tasks 2` x 2 worker = 4 việc đầu của kế hoạch: gồm những phông/chữ nào? việc đầu tiên chạm từng phông dự phòng, chữ PUA, chữ ngoài BMP; cỡ `--max-tasks` nhỏ nhất chạm HAN NOM A.
  6b  find_pack (w_runner_khung_chay) với ba bố cục giải nén Kaggle có thể (độ sâu 1, 3, 4) — dùng thư mục giả trong review/goi_du_lieu/_fake_input, KHÔNG chạm /kaggle.
  6c  _meta.json của tar có ghi phông từng chữ không (hướng dẫn §5 nói "metadata mỗi tar ghi chữ nào dùng phông nào")? — đọc write_shard + chạy write_shard với bộ sinh giả.

    cd measure_out/_tn11/full/review/goi_du_lieu && PYTHONDONTWRITEBYTECODE=1 ../../../../../.venv/bin/python -B ../../../../../lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_goi_du_lieu_06_phu_ca_thu_va_bo_cuc.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections
import glob as _glob
import json
import math
import tarfile
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_goi_du_lieu_00_lib as Lb  # noqa: E402
import w_runner_khung_chay as R  # noqa: E402


def main():
    t0 = time.time()
    inv = Lb.Inv()
    res: dict = dict(tao_luc=time.strftime("%Y-%m-%d %H:%M:%S"))
    plan = json.loads((Lb.PACK / "plan.json").read_text(encoding="utf-8"))
    fn = plan["font_names"]
    tasks = plan["tasks"]

    def comp(t):
        v = plan["books"][t["book"]]
        ch = v["chars"][t["lo"]:t["hi"]]
        return collections.Counter(fn[i] for i in v["font_idx"][t["lo"]:t["hi"]]), sum(1 for c in ch if Lb.is_pua(c)), sum(1 for c in ch if ord(c) > 0xFFFF)
    first, rows = {}, []
    for i, t in enumerate(tasks):
        c, p, nb = comp(t)
        rows.append(dict(i=i, id=t["id"], phong=dict(c), pua=p, ngoai_BMP=nb))
        for k in c:
            first.setdefault(k, (i, t["id"]))
        if p:
            first.setdefault("PUA", (i, t["id"]))
        if nb:
            first.setdefault("ngoai_BMP", (i, t["id"]))
    trial_n = 4          # --max-tasks 2 x 2 worker, mỗi worker lần lượt nhận việc đầu chưa bị khoá
    c4 = collections.Counter(); p4 = nb4 = 0
    for t in tasks[:trial_n]:
        c, p, nb = comp(t); c4.update(c); p4 += p; nb4 += nb
    idx_hannom = first["HanNomA"][0]
    res["6a_ca_thu"] = dict(giai_doan="--max-tasks 2 (docs/KAGGLE_SINH_ANH_THEO_SACH §2 bước 4) với 2 worker (T4 x2) => các việc đầu theo thứ tự kế hoạch",
                            cac_viec_ca_thu=[t["id"] for t in tasks[:trial_n]], so_chu_theo_phong_trong_ca_thu=dict(c4), chu_PUA_trong_ca_thu=p4, chu_ngoai_BMP_trong_ca_thu=nb4,
                            viec_dau_tien_cham_tung_phong={k: dict(chi_so=v[0], id=v[1]) for k, v in first.items()},
                            phong_KHONG_duoc_ca_thu_cham=[k for k in fn if k not in c4], phong_co_dau_cach_HAN_NOM_A_duoc_cham=("HanNomA" in c4),
                            max_tasks_nho_nhat_cham_HanNomA=math.ceil((idx_hannom + 1) / 2), thoi_gian_phut_neu_340s_moi_viec=round(math.ceil((idx_hannom + 1) / 2) * 340 / 60, 1),
                            viec_cuoi_cung_cham_du_5_phong_du_phong=max(first[k][0] for k in fn if k != "NomNaTong"))
    res["6a_phong_moi_cuon_so_viec_co_phong_du_phong"] = {b: dict(co=sum(1 for r in rows if r["id"].startswith(b + "/") and any(k != "NomNaTong" for k in r["phong"])), tong=sum(1 for r in rows if r["id"].startswith(b + "/"))) for b in Lb.ORDER}
    inv.check("6a_ca_thu_(4_viec_dau)_co_cham_PUA_va_chu_ngoai_BMP", p4 > 0 and nb4 > 0, (p4, nb4))
    inv.check("6a_ca_thu_cham_phong_co_dau_cach_HAN_NOM_A", "HanNomA" in c4, dict(c4))
    inv.check("6a_ca_thu_cham_moi_phong_du_phong", all(k in c4 for k in fn), [k for k in fn if k not in c4])
    # ---- 6b find_pack
    fake = Lb.OUT / "_fake_input"
    layouts = {"A_do_sau_1 (/kaggle/input/<ds>/plan.json)": "tn11-full-pack",
               "B_do_sau_3 (/kaggle/input/datasets/<user>/<ds>/plan.json — đúng như lần chạy trước: /kaggle/input/datasets/truongmdn/tn11-data/...)": "datasets/truongmdn/tn11-full-pack",
               "C_do_sau_4 (thêm một thư mục bọc bên trong dataset)": "datasets/truongmdn/tn11-full-pack/kaggle_pack_20261004"}
    out = {}
    real_glob = _glob.glob

    def fake_glob(pat, *a, **k):
        return real_glob(pat.replace("/kaggle/input", str(fake / "kaggle_input")), *a, **k)
    for name, rel in layouts.items():
        root = fake / "kaggle_input"
        if root.exists():
            import shutil
            shutil.rmtree(root)
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / "plan.json").write_text("{}", encoding="utf-8")
        with mock.patch("glob.glob", side_effect=fake_glob):
            try:
                p = R.find_pack(None)
                out[name] = dict(tim_thay=True, duong=str(p).replace(str(fake / "kaggle_input"), "/kaggle/input"))
            except SystemExit as e:
                out[name] = dict(tim_thay=False, thong_bao=str(e))
    import shutil
    shutil.rmtree(fake / "kaggle_input", ignore_errors=True)
    shutil.rmtree(fake, ignore_errors=True)
    res["6b_find_pack_theo_bo_cuc"] = out
    res["6b_ghi_chu"] = ("lệnh trong hướng dẫn dùng `find /kaggle/input -name w_runner_khung_chay.py` (mọi độ sâu) nhưng find_pack mặc định (không --pack) chỉ duyệt tối đa 3 cấp; "
                         "w_runner_khung_chay.py nằm CÙNG thư mục với plan.json nên Path(__file__).parent là chỗ chắc chắn")
    inv.check("6b_bo_cuc_A_va_B_(kha_nang_cao)_find_pack_tim_thay", out[list(layouts)[0]]["tim_thay"] and out[list(layouts)[1]]["tim_thay"], out)
    res["6b_bo_cuc_C_tim_thay"] = out[list(layouts)[2]]["tim_thay"]
    # ---- 6c _meta.json
    import io
    tmp = Lb.OUT / "_fake_shard"
    shutil.rmtree(tmp, ignore_errors=True)
    task = tasks[0]
    chars = R.task_chars(plan, task)
    images = {c: b"\x89PNG\r\n\x1a\n" + bytes([i % 256]) * 8 for i, c in enumerate(chars[:5])}
    path = R.write_shard(tmp, task, plan, images, {})
    with tarfile.open(path) as tf:
        meta = json.loads(tf.extractfile("_meta.json").read())
    shutil.rmtree(tmp, ignore_errors=True)
    res["6c_meta_tar"] = dict(khoa=sorted(meta.keys()), co_phong_tung_chu=any("font" in k or "phong" in k for k in meta), ghi_chu="phông của từng chữ chỉ nằm ở plan.json (font_idx + font_names); tar _meta.json không ghi")
    doc = (Lb.REPO / "docs/KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md").read_text(encoding="utf-8")
    res["6c_huong_dan_noi"] = ("metadata mỗi tar ghi chữ nào dùng phông nào" in doc)
    inv.check("6c_huong_dan_noi_tar_ghi_phong_tung_chu_va_ma_that_su_ghi (hai điều phải khớp)", res["6c_huong_dan_noi"] == res["6c_meta_tar"]["co_phong_tung_chu"], (res["6c_huong_dan_noi"], res["6c_meta_tar"]["co_phong_tung_chu"]))
    res["bat_bien"] = inv.summary(); res["bat_bien_chi_tiet"] = inv.rows; res["giay"] = round(time.time() - t0, 1)
    Lb.jdump(res, "06_phu_ca_thu_va_bo_cuc.json")
    print(f"[06] xong {res['giay']}s; bất biến {res['bat_bien']['dat']}/{res['bat_bien']['tong']} rớt={res['bat_bien']['rot']}")


if __name__ == "__main__":
    main()
