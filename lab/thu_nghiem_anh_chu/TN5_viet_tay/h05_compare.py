#!/usr/bin/env python3
"""TN5 h05 — kiểm SAU khi cài profile handwriting: (a) 6 bộ in/khắc KHÔNG đổi quyết định từng ô + crop chuẩn (so từng cell_uid với
bản trước profile, measure_out/_thu_nghiem_anh_chu/TN5/base_gold_exact.csv, sha256 da5bdc9a…); (b) chuyển trạng thái của bộ viết tay;
(c) Borg: đo ô ok mới trên nhãn người (hai vế / nhãn, CI cụm trang) — B18 = TRONG MẪU (cấu hình chọn trên Kinh; số giữ ngoài
là CV khối của h04), B34 = LOBO Kinh→DungLy; (d) tái lập: số ô ok Borg của pipeline == số h04 dự đoán.

    .venv/bin/python lab/thu_nghiem_anh_chu/TN5_viet_tay/h05_compare.py [--new dataset/_ALL/gold_exact.csv]
Ra: measure_out/_thu_nghiem_anh_chu/TN5/compare.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).parent)]
OUT = REPO / "measure_out/_thu_nghiem_anh_chu/TN5"
HW = ("stt2", "stt4", "stt11", "B18", "B34")
COLS = ["cell_uid", "set8", "gold_exact", "reason", "crop_chuan", "crop_chuan_md5", "crop_chuan_128_md5", "crop_status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default=str(REPO / "dataset/_ALL/gold_exact.csv"))
    a = ap.parse_args()
    rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, usecols=COLS)
    Bs, Nw = rd(OUT / "base_gold_exact.csv"), rd(a.new)
    res = dict(base=str(OUT / "base_gold_exact.csv"), new=a.new, n_base=len(Bs), n_new=len(Nw),
               same_uid_set=bool(set(Bs.cell_uid) == set(Nw.cell_uid)))
    X = Bs.merge(Nw, on="cell_uid", suffixes=("_b", "_n"), how="inner", validate="1:1")
    pr = ~X.set8_b.isin(HW)
    diff = {}
    for c in ("gold_exact", "reason", "crop_chuan", "crop_chuan_md5", "crop_chuan_128_md5", "crop_status"):
        diff[c] = int((X[c + "_b"][pr] != X[c + "_n"][pr]).sum())
    res["printed"] = dict(sets=sorted(set(X.set8_b[pr])), n=int(pr.sum()), diff=diff, PASS=all(v == 0 for v in diff.values()))
    # crop chuẩn của ô ok ở cả hai bản (mọi bộ): md5 không đổi
    both_ok = (X.gold_exact_b == "ok") & (X.gold_exact_n == "ok")
    res["crop_md5_same_on_ok_both"] = dict(n=int(both_ok.sum()),
                                           diff=int((X.crop_chuan_md5_b[both_ok] != X.crop_chuan_md5_n[both_ok]).sum()))
    tr = {}
    for s in HW:
        m = X.set8_b == s
        t = pd.crosstab(X.gold_exact_b[m], X.gold_exact_n[m])
        tr[s] = dict(before={k: int(v) for k, v in X.gold_exact_b[m].value_counts().items()},
                     after={k: int(v) for k, v in X.gold_exact_n[m].value_counts().items()},
                     ok_lost=int(((X.gold_exact_b == "ok") & (X.gold_exact_n != "ok") & m).sum()),
                     ok_gained=int(((X.gold_exact_b != "ok") & (X.gold_exact_n == "ok") & m).sum()),
                     gained_from={k: int(v) for k, v in X.reason_b[(X.gold_exact_b != "ok") & (X.gold_exact_n == "ok") & m]
                                  .value_counts().head(6).items()},
                     matrix={f"{i}->{j}": int(t.loc[i, j]) for i in t.index for j in t.columns if t.loc[i, j]})
    res["handwriting"] = tr
    # Borg trên nhãn người
    from h04_select import boot, load
    _, B = load()
    okn = dict(zip(Nw.cell_uid, Nw.gold_exact == "ok"))
    sel = json.load(open(OUT / "select.json", encoding="utf-8"))
    for s8, ref in (("B18", "kinh_in_sample"), ("B34", "lobo_K_to_D")):
        D = B[B.set8 == s8]
        ok = D.cell_uid.map(okn).fillna(False).values.astype(bool)
        eb = ok & D.has_gt.values & D.keepbox.values; el = ok & D.has_gt.values
        r = dict(ok=int(ok.sum()), n_both=int(eb.sum()), n_lab=int(el.sum()),
                 both=float(D.y_both.values[eb].mean()) if eb.any() else None, err_both=int((~D.y_both.values[eb]).sum()),
                 both_ci=boot(D.y_both.values[eb], D.page_key.values[eb]),
                 lab=float(D.y_lab.values[el].mean()) if el.any() else None, err_lab=int((~D.y_lab.values[el]).sum()),
                 lab_ci=boot(D.y_lab.values[el], D.page_key.values[el]),
                 h04_ok=sel[ref]["ok"], eq_h04=int(ok.sum()) == sel[ref]["ok"],
                 danh_gia="trong mẫu (cấu hình chọn trên Kinh) — số giữ ngoài: cv_khoi_Kinh" if s8 == "B18" else "giữ ngoài (LOBO K→D)")
        res[f"borg_{s8}"] = r
    json.dump(res, open(OUT / "compare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(dict(printed=res["printed"], crop_ok_both=res["crop_md5_same_on_ok_both"], same_uid=res["same_uid_set"]),
                     ensure_ascii=False))
    for s, v in tr.items():
        print(f"  {s}: ok {v['before'].get('ok', 0)} -> {v['after'].get('ok', 0)} (+{v['ok_gained']} / -{v['ok_lost']}) từ {v['gained_from']}")
    for s8 in ("B18", "B34"):
        r = res[f"borg_{s8}"]
        print(f"  {s8}: ok {r['ok']} (h04 {r['h04_ok']}, trùng {r['eq_h04']}) · hai vế {r['both']} {r['both_ci']} n {r['n_both']} lỗi "
              f"{r['err_both']} · nhãn {r['lab']} {r['lab_ci']} n {r['n_lab']} lỗi {r['err_lab']}")


if __name__ == "__main__":
    main()
