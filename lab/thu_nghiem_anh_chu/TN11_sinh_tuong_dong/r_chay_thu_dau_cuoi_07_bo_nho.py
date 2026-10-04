"""Bộ nhớ thiết bị của RealGen (MPS, fp32) cho một lô 4/8/16 chữ (bs=32 ngoại suy tuyến tính): ước lượng bậc độ lớn cho T4 15 GB (KHÔNG phải số đo CUDA).
Dùng torch.mps.driver_allocated_memory()/current_allocated_memory() sau mỗi lô; chạy từ cwd = review/chay_thu_dau_cuoi/cascade_cwd.
Ra: .../review/chay_thu_dau_cuoi/07_bo_nho.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import os
import time

import r_chay_thu_dau_cuoi_lib as L

os.chdir(L.R / "cascade_cwd")
import torch  # noqa: E402

K = L.load_runner()
rep = L.Rep("07_bo_nho")
plan = K.load_plan(L.PACK)
TMP = L.R / "cascade_tmp"
fmap = __import__("json").loads((L.PACK / "fonts.json").read_text(encoding="utf-8"))
fpaths = {n: str(TMP / "font_diffusion" / "fonts" / f) for n, f in fmap.items()}
gen = K.RealGen(TMP, TMP / "font_diffusion" / "ckpt" / "PROD", fpaths, "mps", 32, cache_dir=TMP / "_unused_cache")
style = L.PACK / "data" / "B34" / "style_0.png"
chars = list(plan["books"]["B34"]["chars"][:32])
fonts = ["NomNaTong"] * 32
res = {}
base = torch.mps.driver_allocated_memory() / 1e9
for bs in (4, 8, 16):                              # KHÔNG chạy bs=32 trực tiếp: lần thử đầu (lô 32 liên tục) đã đẩy Mac 16 GB vào hoán đổi bộ nhớ (RSS ≈ 12 GB)
    gen.batch = bs
    torch.mps.empty_cache()
    t0 = time.time()
    imgs, sk = gen.generate(dict(book="B34", model="base", style=0), chars[:bs], fonts[:bs], style, 12345)
    dt = time.time() - t0
    res[f"bs={bs}"] = dict(giay_moi_anh=round(dt / bs, 2), driver_GB=round(torch.mps.driver_allocated_memory() / 1e9, 2), hien_tai_GB=round(torch.mps.current_allocated_memory() / 1e9, 2), n=len(imgs), bo_qua=len(sk))
rep.data("sau_nap_mo_hinh_driver_GB", round(base, 2))
d16 = res["bs=16"]["driver_GB"]
rep.data("ngoai_suy_bs32_driver_GB", round(base + (d16 - base) * 2, 2))
rep.data("theo_lo", res)
rep.save()
