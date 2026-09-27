"""eval_ihr.py — đo trên NHÃN NGƯỜI IHR (L16, TK) độ chính xác hai vế của ô gold_exact = ok (0 API, CPU).

Sự thật (chép nguyên kim_bottleneck/harness/h01_build_cells.py + TN4 hslot.py, chỉ đọc data/ + prepared/):
  chữ người   data/<book>/manifest.tsv: nom_text theo (trang, cột GT, part 1|2); ô (column, syl_idx) -> part = 1 nếu syl < 6,
              pos = syl (part 1) | syl − 6 (part 2); gt_char = nom_text[pos].
  khe         hai mô hình khe của harness áp cho TÂM của crop đang xét: mẫu t (vị trí trung vị theo syl_idx trong hộp cột người
              vẽ, config ihr_slot_template) + dòng kim thô (prepared/<book>/kim_raw/<page>_lt2.json); slot_ok = '1' nếu cột đúng và
              cả hai (hoặc chỉ mẫu khi thiếu dòng kim) chỉ đúng syl_idx; '0' nếu sai cột / cả hai cùng lệch; '' nếu bất đồng.
              Crop chuẩn: tâm hộp mực mới (TN4 slot_new); crop cũ: tâm bbox (= slot_ok harness).
  đúng hai vế = V1+(nhãn, chữ người) ∧ khe = '1' (khe chưa xác định = sai); strict = nhãn == chữ người ∧ khe = '1'.
  CI 95 % bootstrap cụm theo TRANG (B = 2000, seed 20260926, thứ tự L16 rồi TK như TN3 measure_ihr).
"""
from __future__ import annotations

import collections as C
import csv
import json
import math

import numpy as np
import pandas as pd

from .common import REPO, var_eq_plus

N1 = 6


def load_gt(book):
    gt = {}
    for r in csv.DictReader(open(REPO / "data" / book / "manifest.tsv", encoding="utf-8"), delimiter="\t"):
        if r["col_index"] == "" or r["part"] == "":
            continue
        x, y, w, h = map(float, r["col_bbox_xywh"].split(","))
        d = gt.setdefault((r["page_id"], int(r["col_index"])), dict(box=(x, y, w, h), t={}))
        d["t"][int(r["part"])] = r["nom_text"]
    for d in gt.values():
        d["t1"] = d["t"].get(1, ""); d["t2"] = d["t"].get(2, "")
    return gt


class Slot:
    """== TN4 hslot.Slot (mô hình khe người của harness cho một tâm bất kỳ)."""

    def __init__(self, book, mau):
        self.gt = load_gt(book)
        self.pm = {p["page_name"]: p["page_id"] for p in json.loads((REPO / "prepared" / book / "manifest.json").read_text())["pages"]}
        self.pages_gt = {k[0] for k in self.gt}
        self.mau = mau
        self.klines = {}
        for pn, pid in self.pm.items():
            f = REPO / "prepared" / book / "kim_raw" / f"{pn}_lt2.json"
            if not f.exists():
                continue
            bx = json.loads(f.read_text())["boxes"]
            cols = [(k[1], v["box"]) for k, v in self.gt.items() if k[0] == pid]
            per = C.defaultdict(list)
            for b in bx:
                xs = [p[0] for p in b["points"]]; ys = [p[1] for p in b["points"]]
                xc = (min(xs) + max(xs)) / 2
                for ci, (x, y, w, h) in cols:
                    if x <= xc <= x + w:
                        per[ci].append((min(ys), max(ys), b.get("transcription", ""))); break
            for ci, L in per.items():
                L.sort()
                if len(L) == 2:
                    self.klines[(pid, ci)] = [(L[0][0], L[0][1], 1, L[0][2]), (L[1][0], L[1][1], 2, L[1][2])]

    def gt_char(self, page, column, syl_idx):
        pid = self.pm.get(page, "")
        si = int(syl_idx); ec = int(column) - 1
        part = 1 if si < N1 else 2
        pos = si if part == 1 else si - N1
        ev = self.gt.get((pid, ec))
        s = (ev["t1"] if part == 1 else ev["t2"]) if ev else ""
        return s[pos] if 0 <= pos < len(s) else ""

    def slot(self, page, column, syl_idx, cx, cy):
        pid = self.pm.get(page, "")
        if pid not in self.pages_gt or cx is None or cy is None or cx != cx or cy != cy:
            return ""
        cand = [(k[1], v) for k, v in self.gt.items() if k[0] == pid]
        inside = [(ci, v) for ci, v in cand if v["box"][0] <= cx <= v["box"][0] + v["box"][2]]
        if inside:
            gci, gv = inside[0]
        else:
            gci, gv = min(cand, key=lambda kv: min(abs(cx - kv[1]["box"][0]), abs(cx - kv[1]["box"][0] - kv[1]["box"][2])))
        x, y, w, h = gv["box"]
        t = (cy - y) / h
        si = int(syl_idx); ec = int(column) - 1
        k_t = min(range(14), key=lambda j: abs(t - self.mau[j]))
        k_k = None
        if (pid, gci) in self.klines:
            L = self.klines[(pid, gci)]
            top, bot, prt, _ = min(L, key=lambda Q: 0 if Q[0] <= cy <= Q[1] else min(abs(cy - Q[0]), abs(cy - Q[1])))
            txt = gv["t"].get(prt, "")
            if txt:
                u = (cy - top) / max(1e-6, bot - top) * len(txt)
                sl = min(len(txt) - 1, max(0, math.floor(u)))
                k_k = sl if prt == 1 else N1 + sl
        col_ok = gci == ec
        on_t = col_ok and k_t == si
        on_k = None if k_k is None else (col_ok and k_k == si)
        if not col_ok:
            return "0"
        if on_t and (on_k is None or on_k):
            return "1"
        if (not on_t) and on_k is False:
            return "0"
        return ""


def ihr_truth(G: pd.DataFrame, D: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Cho mọi ô GOLD L16/TK: gt_char, slot_old (tâm bbox), slot_new (tâm hộp mực crop chuẩn), y_lab (V1+), y_str."""
    rows = []
    Dm = D.set_index("cell_uid")
    for book in ("LucVanTien1916", "TruyenKieu1872"):
        S = Slot(book, cfg["ihr_slot_template"][book])
        g = G[G.book_set == book]
        for u, pg, col, si, bb, lab in zip(g.cell_uid, g.page, g.column, g.syl_idx, g.bbox, g.label):
            b = json.loads(bb)
            cxo, cyo = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            gt = S.gt_char(pg, col, si)
            r = Dm.loc[u] if u in Dm.index else None
            ok = r is not None and r.get("status") == "ok"
            sn = S.slot(pg, col, si, float(r["ink_cx"]), float(r["ink_cy"])) if ok else ""
            rows.append(dict(cell_uid=u, book_set=book, page=pg, gt_char=gt, slot_old=S.slot(pg, col, si, cxo, cyo), slot_new=sn,
                             y_lab=var_eq_plus(lab, gt) if gt else False, y_str=(lab == gt) if gt else False))
    return pd.DataFrame(rows)


def measure(keep: np.ndarray, H: pd.DataFrame, slot_col: str, B: int, seed: int, order=("L16", "TK")) -> dict:
    """Độ chính xác hai vế (V1+ và strict) của tập giữ `keep` (căn theo H), CI bootstrap cụm trang."""
    rng = np.random.default_rng(seed)
    out = {}
    s8 = H.book_set.map({"LucVanTien1916": "L16", "TruyenKieu1872": "TK"}).values
    for bk in order:
        te = (s8 == bk) & (H.gt_char.values != "")
        pg = H.page.values[te]
        up, inv = np.unique(pg, return_inverse=True)
        idx = rng.integers(0, len(up), (B, len(up)))
        k = keep[te]; okimg = H[slot_col].values[te] == "1"
        r = dict(n_keep=int(keep[s8 == bk].sum()), n_eval=int(k.sum()))
        for nm, lab in (("both", H.y_lab.values[te]), ("strict", H.y_str.values[te])):
            arrs = [k, k & ~(lab & okimg), k & ~lab, k & ~okimg]
            P = np.stack([np.bincount(inv, weights=a.astype(float), minlength=len(up)) for a in arrs])
            tot = P.sum(1)
            if tot[0] == 0:
                continue
            BS = P[:, idx].sum(2)
            pb = 1 - BS[1] / np.maximum(BS[0], 1)
            r[f"{nm}_pt"] = float(1 - tot[1] / tot[0])
            r[f"{nm}_lo"], r[f"{nm}_hi"] = float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))
            r[f"{nm}_err"] = int(tot[1]); r[f"{nm}_err_lab"] = int(tot[2]); r[f"{nm}_err_img"] = int(tot[3])
        out[bk] = r
    return out
