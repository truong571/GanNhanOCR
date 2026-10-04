"""w_hien_trang_du_lieu_06b_xac_minh_cache.py — XÁC MINH ĐỘC LẬP nhớ đệm crop chuẩn (prepared/_gold_exact/crop_chuan) khớp với hộp HIỆN TẠI.

Dựng lại crop chuẩn của MỌI ô GOLD trên một mẫu trang (4 trang/cuốn, seed cố định) bằng chính hàm của pipeline (signals_geom._page_worker, chỉ ĐỌC mã; ghi PNG vào
measure_out/_tn11/full/hien_trang_du_lieu/_xac_minh_canon/, KHÔNG chạm nhớ đệm) rồi so md5 với tệp nhớ đệm cùng ô. Khớp 100 % => nhớ đệm đúng hộp/nhãn hiện tại ngay cả ở ô KHÔNG ok.
Ra: .../cache_crop_chuan_xac_minh.json. CPU, 0 API.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_hien_trang_du_lieu_06b_xac_minh_cache.py [--pages 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import w_hien_trang_du_lieu_lib as L  # noqa: E402
from pipeline.gold_exact import common as CM  # noqa: E402
from pipeline.gold_exact import crop_chuan as CC  # noqa: E402
from pipeline.gold_exact import signals_geom as SG  # noqa: E402

CACHE = L.REPO / "prepared" / "_gold_exact" / "crop_chuan"
WORK = L.OUT / "_xac_minh_canon"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(open(L.REPO / "config/gold_exact.yaml", encoding="utf-8"))
    band = cfg["core_loss"]["band_pitch"]
    G = CM.load_gold(L.REPO / "dataset/_ALL")
    SG._init()
    sq_root, s128_root = WORK / "sq", WORK / "sq128"
    rng = np.random.default_rng(20261004)
    inv = L.Inv()
    out = {}
    tot_cmp = tot_same = tot_cmp128 = tot_same128 = 0
    for bs, subs in G.groupby("book_set", sort=False):
        B = SG.load_build(bs)
        Bg = {k: g for k, g in B.groupby(["book", "page"])}
        for bk, sub in subs.groupby("book", sort=False):
            pages = sorted(set(sub.page))
            pick = sorted(rng.choice(len(pages), size=min(a.pages, len(pages)), replace=False))
            n_cmp = n_same = n128 = s128 = n_ok_cells = n_nonok = n_nonok_same = 0
            diff_examples = []
            for i in pick:
                pg = pages[i]
                g = sub[sub.page == pg]
                pdir = CM.page_dir(bs, bk)
                colrows = defaultdict(list)
                bp = Bg.get((bk, pg))
                if bp is not None:
                    for c, b in zip(bp.column, bp.bbox):
                        colrows[c].append(json.loads(b))
                rows = []
                for u, bb, col in zip(g.cell_uid, g.bbox, g.column):
                    bb = json.loads(bb)
                    pv, nx = CC.neighbours_by_rule(colrows.get(col, []), bb)
                    rows.append(dict(uid=u, bbox=bb, column=col, prev_b=pv, next_b=nx))
                task = (bs, bk, pg, str(pdir), bs in CM.ORIGINAL, rows, dict(colrows), str(sq_root), str(s128_root), band)
                res = SG._page_worker(task)
                for r in res:
                    if r.get("status") != "ok":
                        continue
                    rel = CM.uid_path(r["cell_uid"])
                    mine, theirs = sq_root / rel, CACHE / "sq" / rel
                    n_cmp += 1
                    same = theirs.exists() and md5(mine) == md5(theirs)
                    n_same += int(same)
                    m2, t2 = s128_root / rel, CACHE / "sq128" / rel
                    n128 += 1
                    s128 += int(t2.exists() and md5(m2) == md5(t2))
                    if not same and len(diff_examples) < 3:
                        diff_examples.append(r["cell_uid"])
            out[f"{bs}/{bk}"] = dict(trang=[pages[i] for i in pick], o_so_sanh=n_cmp, giong_het=n_same, o_so_sanh_128=n128, giong_het_128=s128, vi_du_lech=diff_examples)
            tot_cmp += n_cmp; tot_same += n_same; tot_cmp128 += n128; tot_same128 += s128
            inv.check(f"{bs}/{bk}_nho_dem_bang_dung_lai", n_cmp > 0 and n_same == n_cmp and s128 == n128, f"{n_same}/{n_cmp} ; 128: {s128}/{n128}")
            print(f"[06b] {bs}/{bk}: {len(pick)} trang, {n_cmp} ô, giống hệt {n_same} (128: {s128}/{n128}) {diff_examples}", flush=True)
    out["_tong"] = dict(o_so_sanh=tot_cmp, giong_het=tot_same, o_so_sanh_128=tot_cmp128, giong_het_128=tot_same128)
    out["bat_bien"] = inv.summary(); out["bat_bien_chi_tiet"] = inv.rows
    out["giay"] = round(time.time() - t0, 1)
    L.jdump(out, L.OUT / "cache_crop_chuan_xac_minh.json")
    print(f"[06b] tổng {tot_same}/{tot_cmp} giống hệt (128: {tot_same128}/{tot_cmp128}) · bất biến {out['bat_bien']['dat']}/{out['bat_bien']['tong']} [{out['giay']}s]")


if __name__ == "__main__":
    main()
