"""w_hien_trang_du_lieu_06_cache_crop_chuan.py — CROP CHUẨN CÓ SẴN CHO MỌI Ô GOLD? (bộ nhớ đệm của bước gold_exact)

bước gold_exact dựng crop chuẩn (pipeline/gold_exact/signals_geom.run_crop_chuan) cho MỌI ô GOLD vào prepared/_gold_exact/crop_chuan/{sq,sq128}/<book_set>/<book>/<page>/c.._n.._s...png
(đường = common.uid_path(cell_uid)) rồi chỉ XUẤT BẢN ô ok sang dataset/_ALL/crops_chuan. Nếu nhớ đệm còn đủ và khớp bản xuất bản, ảnh phong cách / cặp tinh chỉnh có thể dùng crop chuẩn
của MỌI ô neo sạch hình học (không cần trạng thái ok phụ thuộc nhãn người).
Kiểm: (1) độ phủ nhớ đệm trên ô neo khuyến nghị / mọi ô neo GOLD; (2) md5 nhớ đệm == md5 bản xuất bản (cột crop_chuan_md5) trên MỌI ô ok; (3) mtime tệp nhớ đệm; (4) bản 128;
(5) kích thước crop chuẩn (cạnh) của ô neo khuyến nghị.
Ra: measure_out/_tn11/full/hien_trang_du_lieu/cache_crop_chuan.json. Chỉ ĐỌC. 0 API/GPU.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_06_cache_crop_chuan.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import w_hien_trang_du_lieu_lib as L  # noqa: E402

CACHE = L.REPO / "prepared" / "_gold_exact" / "crop_chuan"
T_FINAL = time.mktime(time.strptime("2026-10-01 19:00", "%Y-%m-%d %H:%M"))


def uid_path(uid: str) -> str:          # == pipeline.gold_exact.common.uid_path
    a = uid.split("/")
    return "/".join(a[:3]) + "/" + "_".join(a[3:]) + ".png"


def pmap(fn, items, w=8):
    with ThreadPoolExecutor(w) as ex:
        return list(ex.map(fn, items, chunksize=128))


def md5(p):
    try:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return ""


def size(p):
    try:
        with Image.open(p) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return None


def main():
    t0 = time.time()
    out, inv = {}, L.Inv()
    inv.check("cache_crop_chuan_ton_tai", CACHE.exists(), str(CACHE))
    for ma in L.ORDER:
        C = L.CFG[ma]
        G = L.load_ge(ma)
        Gi = G.set_index("image")
        N = pd.read_csv(L.OUT / "neo" / f"{ma}.csv", dtype=str, keep_default_na=False)
        N = N[N.tier == "GOLD"].copy()
        R = pd.read_csv(L.OUT / "neo_khuyen_nghi" / f"{ma}.csv", dtype=str, keep_default_na=False)
        for X in (N, R):
            X["uid"] = X.image.map(Gi.cell_uid)
            X["cache"] = [str(CACHE / "sq" / uid_path(u)) for u in X.uid]
            X["cache128"] = [str(CACHE / "sq128" / uid_path(u)) for u in X.uid]
            X["ex"] = np.array(pmap(os.path.isfile, list(X.cache)), bool)
            X["ex128"] = np.array(pmap(os.path.isfile, list(X.cache128)), bool)
        # md5 cache == bản xuất bản trên MỌI ô ok (cả không phải neo)
        ok = G[(G.gold_exact == "ok")]
        pc = [str(CACHE / "sq" / uid_path(u)) for u in ok.cell_uid]
        m_cache = pmap(md5, pc)
        same = int(sum(a == b for a, b in zip(m_cache, ok.crop_chuan_md5)))
        pc128 = [str(CACHE / "sq128" / uid_path(u)) for u in ok.cell_uid]
        m128 = pmap(md5, pc128)
        same128 = int(sum(a == b for a, b in zip(m128, ok.crop_chuan_128_md5)))
        # mtime tệp nhớ đệm của ô neo khuyến nghị
        mt = np.array([os.path.getmtime(p) if e else np.nan for p, e in zip(R.cache, R.ex)])
        st = [size(p) if e else None for p, e in zip(R.cache, R.ex)]
        side = np.array([s[0] for s in st if s], float)
        sq = np.array([s[0] == s[1] for s in st if s], bool)
        # ô GOLD có crop_status ok (theo gold_exact) -> phải có tệp nhớ đệm
        st_ok = G[G.crop_status == "ok"]
        ex_ok = np.array(pmap(os.path.isfile, [str(CACHE / "sq" / uid_path(u)) for u in st_ok.cell_uid]), bool)
        res = dict(
            neo_GOLD=int(len(N)), neo_GOLD_co_cache=int(N.ex.sum()), neo_GOLD_co_cache128=int(N.ex128.sum()),
            neo_khuyen_nghi=int(len(R)), neo_khuyen_nghi_co_cache=int(R.ex.sum()), neo_khuyen_nghi_co_cache128=int(R.ex128.sum()),
            o_crop_status_ok=int(len(st_ok)), o_crop_status_ok_co_cache=int(ex_ok.sum()),
            o_ok=int(len(ok)), o_ok_md5_cache_bang_ban_xuat_ban=same, o_ok_md5_128_bang_ban_xuat_ban=same128,
            mtime_cache_neo_khuyen_nghi=dict(min=time.strftime("%Y-%m-%d %H:%M", time.localtime(np.nanmin(mt))), max=time.strftime("%Y-%m-%d %H:%M", time.localtime(np.nanmax(mt))),
                                              ti_le_truoc_01_10_19h=round(float(np.mean(mt[~np.isnan(mt)] < T_FINAL)), 4)),
            canh_crop_chuan_neo_khuyen_nghi=L.stats(side), ti_le_vuong=round(float(sq.mean()), 4) if len(sq) else None)
        inv.check(f"{ma}_neo_khuyen_nghi_co_du_cache", res["neo_khuyen_nghi_co_cache"] == res["neo_khuyen_nghi"], f"{res['neo_khuyen_nghi_co_cache']}/{res['neo_khuyen_nghi']}")
        inv.check(f"{ma}_o_ok_md5_cache_bang_xuat_ban", same == len(ok) and same128 == len(ok), f"{same}/{len(ok)} ; 128: {same128}/{len(ok)}")
        inv.check(f"{ma}_o_crop_status_ok_co_cache", res["o_crop_status_ok_co_cache"] == res["o_crop_status_ok"], f"{res['o_crop_status_ok_co_cache']}/{res['o_crop_status_ok']}")
        out[ma] = res
        print(f"[06] {ma}: neo GOLD {len(N)} có cache {res['neo_GOLD_co_cache']} | khuyến nghị {len(R)} có cache {res['neo_khuyen_nghi_co_cache']} (128: {res['neo_khuyen_nghi_co_cache128']}) | "
              f"ok {len(ok)} md5 khớp {same} (128: {same128}) | mtime cache {res['mtime_cache_neo_khuyen_nghi']['min']}…{res['mtime_cache_neo_khuyen_nghi']['max']} "
              f"({res['mtime_cache_neo_khuyen_nghi']['ti_le_truoc_01_10_19h']:.1%} trước 01/10 19h) | cạnh median {res['canh_crop_chuan_neo_khuyen_nghi'].get('median')}", flush=True)
    # tổng tệp nhớ đệm / thư mục
    tot = {}
    for sub in sorted((CACHE / "sq").iterdir()):
        tot[sub.name] = sum(len(fn) for _, _, fn in os.walk(sub))
    out["_tong_tep_nho_dem_sq"] = tot
    out["_nho_dem_duong"] = L.rel(CACHE)
    out["_ghi_chu"] = ("nhớ đệm do run_crop_chuan ghi (gitignore: prepared/); khoá trang gồm chữ ký mã, stat trang nguồn, hộp ô, hộp cột -> tệp của ô hiện tại khớp hộp hiện tại; "
                       "clean_rebuild_all.sh có thể xoá/dựng lại thư mục này")
    out["bat_bien"] = inv.summary(); out["bat_bien_chi_tiet"] = inv.rows
    L.jdump(out, L.OUT / "cache_crop_chuan.json")
    print(f"[06] bất biến {out['bat_bien']['dat']}/{out['bat_bien']['tong']} {[r['ten'] for r in out['bat_bien']['rot']]} [{time.time() - t0:.0f}s]")
    print("->", L.rel(L.OUT / "cache_crop_chuan.json"))


if __name__ == "__main__":
    main()
