"""r_goi_du_lieu_02b_dung_lai_crop_chuan.py — HƯỚNG 2 (tiếp): ảnh phong cách của gói = crop chuẩn ĐÚNG Ô, kiểm bằng DỰNG LẠI bằng chính mã pipeline hiện hành.

Với 20 ô phong cách của gói (10 cuốn x 2) và các ô ĐỀ XUẤT thay (4 bộ evaluation_only, do r_goi_du_lieu_02 sinh): chạy lại signals_geom._page_worker (chỉ ĐỌC mã pipeline; ghi PNG vào
review/goi_du_lieu/_dung_lai_crop_chuan/, KHÔNG chạm prepared/_gold_exact) trên trang của ô (cùng bộ ô GOLD của trang như 06b), rồi so:
  md5(crop chuẩn 128 dựng lại) == md5(tệp nhớ đệm đã dùng)   và   style_image(dựng lại) == ảnh trong gói (từng điểm ảnh).
Khác 06b (mẫu 4 trang/cuốn): ở đây là ĐÚNG các ô của gói.

    cd measure_out/_tn11/full/review/goi_du_lieu && PYTHONDONTWRITEBYTECODE=1 ../../../../../.venv/bin/python -B ../../../../../lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_goi_du_lieu_02b_dung_lai_crop_chuan.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_goi_du_lieu_00_lib as Lb  # noqa: E402
from r_goi_du_lieu_02_anh_phong_cach import style_image  # noqa: E402


def md5f(p) -> str:
    import hashlib
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    t0 = time.time()
    inv = Lb.Inv()
    from pipeline.gold_exact import common as CM
    from pipeline.gold_exact import crop_chuan as CC
    from pipeline.gold_exact import signals_geom as SG
    cfg = yaml.safe_load(open(Lb.REPO / "config/gold_exact.yaml", encoding="utf-8"))
    band = cfg["core_loss"]["band_pitch"]
    G = CM.load_gold(Lb.REPO / "dataset/_ALL")
    SG._init()
    WORK = Lb.OUT / "_dung_lai_crop_chuan"
    sq_root, s128_root = WORK / "sq", WORK / "sq128"
    d2 = json.loads((Lb.OUT / "02_anh_phong_cach.json").read_text(encoding="utf-8"))
    cells = []     # (nhãn, cuốn, uid, ảnh trong gói hoặc ảnh đề xuất)
    for b in Lb.ORDER:
        for e in d2["sach"][b]["anh"]:
            cells.append((f"goi:{b}_s{e['j']}", b, e["cell_uid_ky_vong"], Lb.PACK / "data" / b / f"style_{e['j']}.png"))
    for b, pr in d2["tong_hop"].get("de_xuat_thay_anh", {}).items():
        for r in pr["de_xuat"]:
            cells.append((f"de_xuat:{b}_s{r['j']}", b, r["cell_uid"], Lb.OUT / "de_xuat_style" / f"{b}_s{r['j']}.png"))
    builds = {}
    out = []
    for tag, b, uid, png in cells:
        bs, bk, pg = uid.split("/")[:3]
        sub = G[(G.book_set == bs) & (G.book == bk) & (G.page == pg)]
        row = dict(nhan=tag, cell_uid=uid, so_o_GOLD_cung_trang=int(len(sub)), co_trong_dataset_ALL=bool((G.cell_uid == uid).any()))
        if not row["co_trong_dataset_ALL"]:
            out.append(row); continue
        if bs not in builds:
            B = SG.load_build(bs)
            builds[bs] = {k: g for k, g in B.groupby(["book", "page"])}
        bp = builds[bs].get((bk, pg))
        colrows = defaultdict(list)
        if bp is not None:
            for c, bb in zip(bp.column, bp.bbox):
                colrows[c].append(json.loads(bb))
        rows = []
        for u, bb, col in zip(sub.cell_uid, sub.bbox, sub.column):
            bb = json.loads(bb)
            pv, nx = CC.neighbours_by_rule(colrows.get(col, []), bb)
            rows.append(dict(uid=u, bbox=bb, column=col, prev_b=pv, next_b=nx))
        task = (bs, bk, pg, str(CM.page_dir(bs, bk)), bs in CM.ORIGINAL, rows, dict(colrows), str(sq_root), str(s128_root), band)
        res = SG._page_worker(task)
        r = next(x for x in res if x["cell_uid"] == uid)
        rel = CM.uid_path(uid)
        mine128, theirs128 = s128_root / rel, Lb.REPO / "prepared/_gold_exact/crop_chuan/sq128" / rel
        row.update(status=r.get("status"), src_kind=r.get("src_kind"), flags=r.get("flags", ""), tall=r.get("tall"), n_comp=r.get("n_comp"), n_cut_comp=r.get("n_cut_comp"),
                   foreign_erased_px=r.get("foreign_erased_px"), new_crop_quality=r.get("new_crop_quality_flag"))
        if r.get("status") == "ok" and mine128.exists():
            row["md5_dung_lai_128"] = md5f(mine128)
            row["md5_cache_128"] = md5f(theirs128) if theirs128.exists() else None
            row["cache_giong_het_dung_lai"] = row["md5_dung_lai_128"] == row["md5_cache_128"]
            g = cv2.imread(str(mine128), cv2.IMREAD_GRAYSCALE)
            import PIL.Image as PI
            arr = np.array(PI.open(png))
            row["anh_goi_bang_style_image_cua_ban_dung_lai"] = bool(np.array_equal(style_image(g), arr))
        out.append(row)
        print(f"[02b] {tag}: {row.get('status')} cache_same={row.get('cache_giong_het_dung_lai')} style_same={row.get('anh_goi_bang_style_image_cua_ban_dung_lai')} flags='{row.get('flags')}'", flush=True)
    # dọn: giữ lại chỉ crop chuẩn 128 của chính các ô kiểm (còn lại là ô GOLD cùng trang, chỉ cần để chạy _page_worker)
    import shutil
    keep = {CM.uid_path(x["cell_uid"]) for x in out}
    shutil.rmtree(sq_root, ignore_errors=True)
    for f in list(s128_root.rglob("*.png")):
        if f.relative_to(s128_root).as_posix() not in keep:
            f.unlink()
    for d in sorted((p for p in s128_root.rglob("*") if p.is_dir()), key=lambda q: -len(q.parts)):
        if not any(d.iterdir()):
            d.rmdir()
    res_all = dict(tao_luc=time.strftime("%Y-%m-%d %H:%M:%S"), o=out)
    pk = [x for x in out if x["nhan"].startswith("goi:")]
    dx = [x for x in out if x["nhan"].startswith("de_xuat:")]
    inv.check("goi_20_o_co_trong_dataset_ALL", len(pk) == 20 and all(x["co_trong_dataset_ALL"] for x in pk))
    inv.check("goi_20_o_dung_lai_status_ok", all(x.get("status") == "ok" for x in pk), [x["nhan"] for x in pk if x.get("status") != "ok"])
    inv.check("goi_20_o_nho_dem_128_giong_het_dung_lai_bang_ma_pipeline", all(x.get("cache_giong_het_dung_lai") for x in pk), [x["nhan"] for x in pk if not x.get("cache_giong_het_dung_lai")])
    inv.check("goi_20_anh_png_bang_style_image(crop chuẩn dựng lại) từng điểm ảnh", all(x.get("anh_goi_bang_style_image_cua_ban_dung_lai") for x in pk), [x["nhan"] for x in pk if not x.get("anh_goi_bang_style_image_cua_ban_dung_lai")])
    inv.check("goi_20_o_khong_co_co_cat_hay_hai_chu (flags crop_chuan rỗng)", all((x.get("flags") or "") == "" for x in pk), {x["nhan"]: x.get("flags") for x in pk if x.get("flags")})
    inv.check("de_xuat_cac_o_dung_lai_ok_va_trung_nho_dem", all(x.get("status") == "ok" and x.get("cache_giong_het_dung_lai") and x.get("anh_goi_bang_style_image_cua_ban_dung_lai") for x in dx), [x["nhan"] for x in dx if not x.get("cache_giong_het_dung_lai")])
    res_all["bat_bien"] = inv.summary(); res_all["bat_bien_chi_tiet"] = inv.rows; res_all["giay"] = round(time.time() - t0, 1)
    Lb.jdump(res_all, "02b_dung_lai_crop_chuan.json")
    print(f"[02b] xong {res_all['giay']}s; bất biến {res_all['bat_bien']['dat']}/{res_all['bat_bien']['tong']} rớt={res_all['bat_bien']['rot']}")


if __name__ == "__main__":
    main()
