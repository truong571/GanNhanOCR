"""TN10 t01 (30/09) — kim có xác nhận được VỊ TRÍ của chữ người (bộ _BORG_NHAN_NGUOI) không? 0 API.

Bộ chữ người Borg chỉ giữ 39 % số chữ có ảnh; 50,5 % số ô bị loại vì hậu nghiệm gióng < 0,999 (không phải vì detector sót).
Ý tưởng (từ đề xuất "đọc 2 lượt"): chữ kim đọc ra tại đúng hộp là một mốc vị trí ĐỘC LẬP với bộ gióng (encoder).
Ô người (hộp đơn vị x1..y2 của bộ gióng) được "kim xác nhận" khi có một chữ kim (hộp chữ = hộp dòng chia đều) có tâm nằm trong
hộp đơn vị (nới `--pad` mỗi phía) và trùng chữ người theo V1+.

Đo (trên measure_out/_borg_human/cells.csv.gz — mọi ô người, kèm cờ keep/keep_high và hồ sơ Paddle đóng băng):
  * tỉ lệ kim xác nhận theo nhóm (keep_v5 / keep / keep_high / không giữ);
  * độ tin: Paddle p0 (tỉ lệ `paddle_test == ok` trong ô Paddle thử được) của ô kim xác nhận so với ô keep — Paddle độc lập với
    cả kim lẫn encoder;
  * số ô KHÔNG GIỮ mà kim xác nhận = ứng viên nâng lên keep.
Nguồn kim: --src lt2 (mặc định; prepared/_auto/<S>/kim_raw/<trang>_lt2.json, đủ trang) hoặc thư mục biến thể TN7
(measure_out/_tn7_kim/<S>/<biến thể>/, chỉ trang mẫu; hộp chia về toạ độ gốc theo `scale`).

    .venv/bin/python lab/thu_nghiem_kim/TN10_borg_nguoi/t01_kim_anchor.py [--src lt2|<biến thể TN7>] [--pad 0.25]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "measure_out" / "_tn10"
BOOKS = ("SachKinhThayCaBinh", "SachDungLyHoThan")


def kim_chars(book: str, page: str, src: str):
    """[(char, cx, cy)] toạ độ trang gốc; None nếu không có tệp."""
    from pipeline.tools.ingest_lithograph_book import expand_box_chars
    if src == "lt2":
        f = REPO / "prepared" / "_auto" / book / "kim_raw" / f"{page}_lt2.json"
        scale = 1.0
    else:
        f = REPO / "measure_out" / "_tn7_kim" / book / src / f"{page}.json"
        scale = None
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    if scale is None:
        scale = float(d.get("scale", 1) or 1)
    out = []
    for b in d.get("boxes") or []:
        for c in expand_box_chars(b):
            x0, y0, x1, y1 = c["bbox"]
            out.append((c["char"], (x0 + x1) / 2 / scale, (y0 + y1) / 2 / scale))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="lt2")
    ap.add_argument("--pad", type=float, default=0.25)
    a = ap.parse_args(argv)
    from pipeline.gold_exact.common import Assets, set_lexicon, var_eq_plus
    set_lexicon(Assets())
    C = pd.read_csv(REPO / "measure_out" / "_borg_human" / "cells.csv.gz")
    C = C[C.book.isin(BOOKS)].copy()
    C["lvl"] = np.where(C.keep_v5 == 1, "keep_v5", np.where(C.keep == 1, "keep", np.where(C.keep_high == 1, "keep_high", "khong")))
    anchor = np.zeros(len(C), bool)
    has_kim = np.zeros(len(C), bool)
    eq_memo: dict = {}
    for (book, page), g in C.groupby(["book", "page"], sort=False):
        K = kim_chars(book, page, a.src)
        if K is None:
            continue
        idx = C.index.get_indexer(g.index)
        has_kim[idx] = True
        kc = np.array([k[0] for k in K], dtype=object)
        kx = np.array([k[1] for k in K]); ky = np.array([k[2] for k in K])
        for i, r in zip(idx, g.itertuples()):
            if pd.isna(r.x1) or not isinstance(r.char, str):
                continue
            w, h = r.x2 - r.x1, r.y2 - r.y1
            m = (kx >= r.x1 - a.pad * w) & (kx <= r.x2 + a.pad * w) & (ky >= r.y1 - a.pad * h) & (ky <= r.y2 + a.pad * h)
            for ch in kc[m]:
                key = (ch, r.char)
                v = eq_memo.get(key)
                if v is None:
                    v = eq_memo[key] = bool(var_eq_plus(ch, r.char))
                if v:
                    anchor[i] = True
                    break
    C["kim_xn"] = anchor
    C["co_kim"] = has_kim
    E = C[C.co_kim]
    rows = []
    for lvl in ("keep_v5", "keep", "keep_high", "khong"):
        g = E[E.lvl == lvl]
        pt = g[g.paddle_test.notna() & (g.paddle_test != "")]
        pa = pt[pt.kim_xn]
        pn = pt[~pt.kim_xn]
        rows.append(dict(muc=lvl, n=len(g), kim_xac_nhan=round(g.kim_xn.mean(), 4), n_kim_xn=int(g.kim_xn.sum()),
                         paddle_n=len(pt), paddle_ok_khi_kim_xn=round((pa.paddle_test == "ok").mean(), 4) if len(pa) else None,
                         n_paddle_kim_xn=len(pa),
                         paddle_ok_khi_khong_xn=round((pn.paddle_test == "ok").mean(), 4) if len(pn) else None))
    T = pd.DataFrame(rows)
    by_book = {b: dict(n=int((E.book == b).sum()), khong=int(((E.book == b) & (E.lvl == "khong")).sum()),
                       khong_kim_xn=int(((E.book == b) & (E.lvl == "khong") & E.kim_xn).sum()),
                       keep_hien_tai=int(((E.book == b) & E.lvl.isin(["keep_v5", "keep"])).sum()))
               for b in BOOKS}
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(src=a.src, pad=a.pad, n_o_co_kim=int(len(E)), bang=rows, theo_sach=by_book,
               paddle_ok_goc={k: v for k, v in Counter(C.paddle_test.fillna("")).items()})
    (OUT / f"t01_kim_anchor_{a.src}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    C[["book", "page", "idx", "lvl", "kim_xn"]].to_csv(OUT / f"t01_cells_{a.src}.csv.gz", index=False)
    print(T.to_string(index=False))
    print(json.dumps(by_book, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
