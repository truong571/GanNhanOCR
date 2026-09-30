"""t03_stt_lt2.py — chữ kim lt2 (lang_type 2) cho MỌI ô STT (mọi tầng, gồm REVIEW no_context / lop_nham), cùng phương
pháp ghép 'line' (chính) + 'geom' của scripts/measure/stt_lt2_eval.py (chỉ gọi hàm của nó; 0 API, chỉ đọc cache
prepared/SachThanhTruyen*/kim_raw_lt2 + detected + prepared/_gold_exact/kim_pages).
Ra: measure_out/_tn8/stt_lt2/<bộ>.pkl (index = hàng base) với cột lt2, lt2_geom, st_line, st_geom, lt2_status.
Kiểm: trên ô đã có trong measure_out/stt_lt2/cells.csv, lt2 (line) phải trùng 100 %.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
import stt_lt2_eval as SL  # noqa: E402


def main():
    books = ["stt2", "stt4", "stt11"]
    cov = SL.lt2_pages(books)
    kim = SL.KimCols()
    ref = T.rd(T.REPO / "measure_out/stt_lt2/cells.csv", usecols=["cell_uid", "status", "lt2"])
    refm = dict(zip(ref.cell_uid, zip(ref.status, ref.lt2)))
    od = T.OUT / "stt_lt2"; od.mkdir(parents=True, exist_ok=True)
    for b in books:
        D = T.load_base(b)
        res = {}
        for pg, g in D.groupby("page"):
            pair = cov[b]["lt2"].get(pg)
            if pair is None:
                for i in g.index:
                    res[i] = dict(lt2_status="no_lt2_page")
                continue
            r2, r1 = pair
            pp = SL.PagePair(r1.get("boxes_raw"), r2.get("boxes_raw"))
            out = kim.get(SL.BOOKS[b], pg, SL.STT_CFG) or {}
            for i, col, ni, oc in zip(g.index, g.column, g.nom_i, g.ocr_char):
                chars = out.get(int(col))
                if not chars or ni < 0 or ni >= len(chars) or not chars[ni][1]:
                    res[i] = dict(lt2_status="no_lt1_box"); continue
                kc, kb = chars[ni][0], [float(v) for v in chars[ni][1][:4]]
                if kc != oc:
                    res[i] = dict(lt2_status="lt1_mismatch"); continue
                dl, dg = pp.by_line(kc, kb), pp.by_geom(kb)
                res[i] = dict(lt2_status=dl.get("st_line"), lt2=dl.get("oth_line", "") if dl.get("st_line") == "ok" else "",
                              st_geom=dg["st_geom"], lt2_geom=dg["oth_geom"], n1=dl.get("n1"), n2=dl.get("n2"))
        R = pd.DataFrame.from_dict(res, orient="index").reindex(D.index)
        for c in ("lt2", "lt2_geom", "lt2_status", "st_geom"):
            R[c] = R[c].fillna("") if c in R else ""
        R.to_pickle(od / f"{b}.pkl")
        # kiểm với bản đo gốc
        uid = D.cell_uid.str.replace(f"{T.BOOKS[b]['full']}/", "SachThanhTruyen/", regex=False)
        n = k = 0
        for u, st, l2 in zip(uid, R.lt2_status, R.lt2):
            v = refm.get(u)
            if v is None:
                continue
            n += 1
            k += (v[0] == st) and ((v[1] == l2) if st == "ok" else True)
        st = R.lt2_status.value_counts().to_dict()
        print(f"[t03] {b}: {len(D)} ô, trạng thái {st}; trùng bản stt_lt2 gốc {k}/{n}", flush=True)


if __name__ == "__main__":
    main()
