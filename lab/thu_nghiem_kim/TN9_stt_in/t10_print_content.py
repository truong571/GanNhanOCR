"""t10_print_content.py — sách in: ô KHÔNG GOLD còn lại nằm ở đâu, và "tin kim" có an toàn không? 0 API.

Hai định nghĩa "đúng" trên IHR (L16, TK — chữ người + hộp cột người + mẫu khe, pipeline.gold_exact.eval_ihr.Slot):
  KHE (như TN8/eval_ihr): nhãn == chữ người của khe (cột, syl_idx) ∧ tâm crop ở đúng khe đó.
  NỘI DUNG: nhãn == chữ người tại TÂM crop (cột người chứa tâm x; chỉ số theo mẫu khe t VÀ dòng kim — hai cách phải trùng,
            không trùng = không xác định -> tính SAI). Đo "crop có đúng là chữ nhãn không" bất kể âm QN của hàng.
Đặc trưng mỗi ô (không nhìn người): nhóm (TN8 group_of), kim ∈ R(âm) / R bỏ thanh / R bỏ dấu, kim ∈ R(âm ±1,±2 cùng cột)
(lệch gióng), kim_geo (chữ kim tại tâm crop, cache detected/ ẢNH CHỤP) == chữ kim của hàng (crop trùng chỗ kim đọc), điểm
bộ chọn TN8 (LOBO với L16/TK: pred/<bộ>__from_<bộ kia>; Chr/L83/KVK: pred/<bộ>__from_print__kim): top1, P.
L83/KVK: chữ DỊ BẢN cùng vị trí (TN8 base.ref) — trùng kim? (ước lượng, dị bản khác chữ hợp lệ ~12–20 % ở GOLD).
Ra measure_out/_tn9/print/<bộ>.pkl + print_census.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402
import tn8model as M8  # noqa: E402
from t06_kimgeo import geo_for  # noqa: E402

OD = T.OUT / "print"
FULL = {"L16": "LucVanTien1916", "TK": "TruyenKieu1872", "L83": "LucVanTien1883", "KVK": "KimVanKieu1884",
        "Chr": "Chrestomathie1872"}
PRED = {"L16": "L16__from_TK", "TK": "TK__from_L16", "L83": "L83__from_print__kim", "KVK": "KVK__from_print__kim",
        "Chr": "Chr__from_print__kim"}


def content_chars(b: str, D: pd.DataFrame):
    """-> (chữ người tại tâm crop, âm QN người tại tâm crop) — '' nếu không xác định."""
    import csv
    from pipeline.gold_exact.eval_ihr import Slot, N1
    gcfg = yaml.safe_load(open(T.REPO / "config/gold_exact.yaml", encoding="utf-8"))
    S = Slot(FULL[b], gcfg["ihr_slot_template"][FULL[b]])
    qn = {}
    for r in csv.DictReader(open(T.REPO / "data" / FULL[b] / "manifest.tsv", encoding="utf-8"), delimiter="\t"):
        if r["col_index"] != "" and r["part"] != "":
            qn[(r["page_id"], int(r["col_index"]), int(r["part"]))] = r["qn_verse"].split()
    out, outq = [], []
    for pg, bb in zip(D.page.values, D.bbox.values):
        try:
            b4 = json.loads(bb)
        except Exception:  # noqa: BLE001
            out.append(""); outq.append(""); continue
        cx, cy = (b4[0] + b4[2]) / 2, (b4[1] + b4[3]) / 2
        pid = S.pm.get(pg, "")
        if pid not in S.pages_gt:
            out.append(""); outq.append(""); continue
        cand = [(k[1], v) for k, v in S.gt.items() if k[0] == pid]
        inside = [(ci, v) for ci, v in cand if v["box"][0] <= cx <= v["box"][0] + v["box"][2]]
        if not inside:
            out.append(""); outq.append(""); continue
        gci, gv = inside[0]
        x, y, w, h = gv["box"]
        t = (cy - y) / h
        k_t = min(range(14), key=lambda j: abs(t - S.mau[j]))
        k_k = None
        if (pid, gci) in S.klines:
            Lq = S.klines[(pid, gci)]
            top, bot, prt, _ = min(Lq, key=lambda Q: 0 if Q[0] <= cy <= Q[1] else min(abs(cy - Q[0]), abs(cy - Q[1])))
            txt = gv["t"].get(prt, "")
            if txt:
                u = (cy - top) / max(1e-6, bot - top) * len(txt)
                sl = min(len(txt) - 1, max(0, math.floor(u)))
                k_k = sl if prt == 1 else N1 + sl
        if k_k is not None and k_k != k_t:
            out.append(""); outq.append(""); continue
        k = k_t
        s = gv["t1"] if k < N1 else gv["t2"]
        pos = k if k < N1 else k - N1
        out.append(s[pos] if 0 <= pos < len(s) else "")
        q = qn.get((pid, gci, 1 if k < N1 else 2), [])
        outq.append(q[pos].strip(".,;:!?\"'()").lower() if 0 <= pos < len(q) else "")
    return np.array(out, dtype=object), np.array(outq, dtype=object)


def build(b: str) -> pd.DataFrame:
    C = T.lex()
    D = pd.read_pickle(T.T8 / "base" / f"{b}.pkl").reset_index(drop=True)
    D["grp"] = M8.group_of(D)
    syl = D.syllable.values; k = D.ocr_char.values
    R = {s: C.R_of(s) for s in set(syl)}
    D["kim_inR"] = [bool(x) and x in R[s] for x, s in zip(k, syl)]
    D["kim_tone"] = [bool(x) and x in T.R_tone(s) for x, s in zip(k, syl)]
    D["kim_fold"] = [bool(x) and x in T.R_fold(s) for x, s in zip(k, syl)]
    key = {(p, c, s): y for p, c, s, y in zip(D.page, D.column, D.syl_i, syl)}
    nb = []
    for p, c, s, x in zip(D.page, D.column, D.syl_i, k):
        ok = False
        if x:
            for d in (-2, -1, 1, 2):
                y = key.get((p, c, s + d))
                if y is not None and x in R.get(y, C.R_of(y)):
                    ok = True; break
        nb.append(ok)
    D["kim_in_nb"] = nb
    cdir = T.SNAP / "prepared" / FULL[b] / "detected"
    D["kim_geo"] = geo_for(D, lambda pg: cdir / f"{pg}_ocr_cache.json")
    D["geo_ok"] = [bool(g) and bool(x) and C.var_eq_plus(g, x) for g, x in zip(D.kim_geo, k)]
    P = pd.read_pickle(T.T8 / "pred" / f"{PRED[b]}.pkl").set_index("i")
    D["top1"] = P.top1.reindex(D.index).fillna("").values
    D["P"] = P.P.reindex(D.index).fillna(0.0).values
    D["p1"] = P.p1.reindex(D.index).fillna(0.0).values
    if b in ("L16", "TK"):
        D["content"], D["content_qn"] = content_chars(b, D)
        rk = lambda x: C.R_key(x) if x else ""  # noqa: E731
        D["syl_ok_content"] = [bool(q) and rk(q) == rk(s) for q, s in zip(D.content_qn, syl)]
        D["triple_ok"] = [bool(g) and bool(x) and C.var_eq_plus(x, g) and so for x, g, so in zip(D.label, D.content, D.syl_ok_content)]
        D["kim_triple_ok"] = [bool(g) and bool(x) and C.var_eq_plus(x, g) and so for x, g, so in zip(k, D.content, D.syl_ok_content)]
        D["kim_ok_slot"] = [bool(g) and bool(x) and C.var_eq_plus(x, g) for x, g in zip(k, D.gt_char)]
        D["kim_ok_content"] = [bool(g) and bool(x) and C.var_eq_plus(x, g) for x, g in zip(k, D.content)]
        D["lab_ok_content"] = [bool(g) and bool(x) and C.var_eq_plus(x, g) for x, g in zip(D.label, D.content)]
        D["lab_ok_slot2"] = [bool(g) and bool(x) and C.var_eq_plus(x, g) and bool(po) for x, g, po in zip(D.label, D.gt_char, D.pos_ok)]
        D["kim_ok_slot2"] = D.kim_ok_slot & D.pos_ok.astype(bool)
        D["top_ok_content"] = [bool(g) and bool(x) and C.var_eq_plus(x, g) for x, g in zip(D.top1, D.content)]
    else:
        D["kim_eq_ref"] = [bool(r) and bool(x) and any(C.var_eq_plus(x, q) for q in r.split("|")) for x, r in zip(k, D.ref)]
        D["lab_eq_ref"] = [bool(r) and bool(x) and any(C.var_eq_plus(x, q) for q in r.split("|")) for x, r in zip(D.label, D.ref)]
    return D


def main():
    OD.mkdir(parents=True, exist_ok=True)
    out = {}
    for b in (sys.argv[1:] or ["L16", "TK", "L83", "KVK", "Chr"]):
        D = build(b)
        D.to_pickle(OD / f"{b}.pkl")
        r = {"N": len(D), "gold_share": round(float((D.tier == "GOLD").mean()), 4)}
        rows = {}
        for g, s in D.groupby("grp"):
            v = dict(n=int(len(s)), share=round(len(s) / len(D), 4), kim=round(float((s.ocr_char != "").mean()), 3),
                     kim_inR=round(float(s.kim_inR.mean()), 3), kim_tone=round(float(s.kim_tone.mean()), 3),
                     kim_in_nb=round(float(s.kim_in_nb.mean()), 3), geo_ok=round(float(s.geo_ok.mean()), 3))
            if b in ("L16", "TK"):
                h = s[s.content != ""]
                v.update(n_content=int(len(h)), kim_ok_content=round(float(h.kim_ok_content.mean()), 3) if len(h) else None,
                         kim_ok_content_geo=round(float(h.kim_ok_content[h.geo_ok].mean()), 3) if h.geo_ok.any() else None,
                         n_geo=int(h.geo_ok.sum()),
                         kim_ok_slot2=round(float(s.kim_ok_slot2.mean()), 3), lab_ok_slot2=round(float(s.lab_ok_slot2.mean()), 3),
                         lab_ok_content=round(float(h.lab_ok_content.mean()), 3) if len(h) else None,
                         top_ok_content=round(float(h.top_ok_content.mean()), 3) if len(h) else None,
                         syl_ok_content=round(float(h.syl_ok_content.mean()), 3) if len(h) else None,
                         kim_triple_ok=round(float(h.kim_triple_ok.mean()), 3) if len(h) else None,
                         kim_triple_ok_geo=round(float(h.kim_triple_ok[h.geo_ok].mean()), 3) if h.geo_ok.any() else None)
            else:
                hr = s[s.ref != ""]
                v.update(n_ref=int(len(hr)), kim_eq_ref=round(float(hr.kim_eq_ref.mean()), 3) if len(hr) else None,
                         lab_eq_ref=round(float(hr.lab_eq_ref.mean()), 3) if len(hr) else None,
                         kim_eq_ref_geo=round(float(hr.kim_eq_ref[hr.geo_ok].mean()), 3) if hr.geo_ok.any() else None)
            rows[g] = v
        r["groups"] = dict(sorted(rows.items(), key=lambda kv: -kv[1]["n"]))
        out[b] = r
        print(f"== {b}: N {len(D)} GOLD {r['gold_share']}", flush=True)
        for g, v in list(r["groups"].items())[:9]:
            print(f"   {g:12s} {v}", flush=True)
    f = T.OUT / "print_census.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(out)
    T.jdump(old, f)


if __name__ == "__main__":
    main()
