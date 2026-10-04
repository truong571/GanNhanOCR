"""BƯỚC 5 (bổ sung) — "Phương án B" của hướng dẫn (không HF): ca sau gắn ĐẦU RA notebook ca trước làm Input và dùng --prev.
Mỗi phiên Kaggle bắt đầu với /kaggle/working sạch và Bundler KHÔNG gói lại các việc đã có ở --prev ⇒ đầu ra ca 2 chỉ chứa việc của ca 2.
Nếu ca 3 chỉ gắn đầu ra ca 2 thì ca 3 có biết việc của ca 1 không? Mô phỏng bằng --stub trên KẾ HOẠCH THẬT (410 việc), mỗi ca --max-tasks 60 x 2 worker.
Ra: .../review/chay_thu_dau_cuoi/08_phuong_an_b.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import json
import shutil
from pathlib import Path

import r_chay_thu_dau_cuoi_lib as L

K = L.load_runner()
rep = L.Rep("08_phuong_an_b")
R, PACK = L.R, L.PACK
W = R / "pab"
if W.exists():
    shutil.rmtree(W)
W.mkdir(parents=True)
CWD = W / "cwd"
plan = K.load_plan(PACK)


def session(name, prev=(), tasks=60):
    out, bd = W / f"out_{name}", W / f"bundles_{name}"
    args = ["run", "--stub", "--stub-ms", "0.2", "--workers", "2", "--pack", PACK, "--out", out, "--bundle-dir", bd, "--session", name, "--max-tasks", tasks]
    if prev:
        args += ["--prev", *prev]
    code, log, sec = L.sh(args, cwd=CWD)
    st = L.status_of(out)
    ids = {r["id"] for r in L.task_rows(out)}
    return dict(code=code, ids=ids, bundles=bd, status=st)


s1 = session("s1")
s2 = session("s2", prev=[W / "bundles_s1"])
s3 = session("s3", prev=[W / "bundles_s2"])                                   # CHỈ đầu ra ca 2 (một notebook = một phiên bản đầu ra)
s3b = session("s3b", prev=[W / "bundles_s1", W / "bundles_s2"])               # gắn cả hai
ov13 = s3["ids"] & s1["ids"]
ov23 = s3["ids"] & s2["ids"]
rep.data("so_viec", dict(s1=len(s1["ids"]), s2=len(s2["ids"]), s3_chi_prev_s2=len(s3["ids"]), s3b_prev_s1_va_s2=len(s3b["ids"]),
                         s3_trung_viec_ca1=len(ov13), s3_trung_viec_ca2=len(ov23), s3b_trung=len(s3b["ids"] & (s1["ids"] | s2["ids"]))))
rep.inv("PB1.ca2_bo_qua_viec_ca1_nhờ_prev", not (s2["ids"] & s1["ids"]) and len(s2["ids"]) == 120, dict(trung=len(s2["ids"] & s1["ids"])))
rep.inv("PB2.ca3_chi_gan_dau_ra_ca2_KHONG_lam_lai_viec_ca1", len(ov13) == 0, dict(lam_lai_viec_ca1=len(ov13), ghi_chu="đầu ra ca 2 chỉ có việc của ca 2 (Bundler loại các việc ở --prev) nên ca 3 không biết ca 1 đã làm"))
rep.inv("PB3.gan_ca_hai_dau_ra_thi_khong_lam_lai", len(s3b["ids"] & (s1["ids"] | s2["ids"])) == 0, dict(trung=len(s3b["ids"] & (s1["ids"] | s2["ids"]))))
# unbundle gộp
dest = W / "merged"
code, log, sec = L.sh(["unbundle", "--bundles", W / "bundles_s1", W / "bundles_s2", W / "bundles_s3", W / "bundles_s3b", "--dest", dest], cwd=CWD)
res = json.loads(log.strip().splitlines()[-1])
rep.data("unbundle_gop", res)
rep.inv("PB4.unbundle_gop_khong_xung_dot_byte", code == 0 and not res["conflicts"], res)
rep.save()
