#!/usr/bin/env python3
"""TN5 h01 — SỰ THẬT hai vế trên Borg (chữ viết tay) cho MỌI ô GOLD tự động: nhãn (chữ người) + VỊ TRÍ ẢNH (hộp người).

Chuỗi ghép (0 API, CPU, không mở ảnh):
  ô GOLD tự động (dataset/_ALL/labels.csv, book_set Borg)
   -> chỉ số âm của trang J = syl_span[0](cột) + syl_idx   (adapter prepared/_auto/<S>/transcriptions; = borg_endtoend_eval)
   -> (câu sentence_id, vị trí k trong câu) theo số âm của câu (chỉ câu "đếm bằng": #chữ Nôm người == #âm)
   -> chữ người gt (Excel SinoNom_Char, clean_nom_text)  = cột gt của borg_endtoend cells.csv (kiểm trùng)
   -> chỉ số chữ người của trang idx (pipeline.borg_human.geom.human_seq: chữ Lo của nom_clean, theo câu)  — chỉ khi chuỗi
      Lo của câu == chuỗi Excel đã làm sạch (kiểm từng câu)
   -> hộp người (dataset/_BORG_NHAN_NGUOI/labels.csv: bbox, kind, keep_level).
Vế ảnh: slot_old = tâm hộp auto (bbox labels.csv) ∈ hộp người; slot_new = tâm hộp mực CROP CHUẨN (ink_cx, ink_cy của
gold_exact.csv) ∈ hộp người; iou_old; d_old/d_new = (chỉ số chữ người có hộp chứa tâm, gần idx nhất) − idx (0 = đúng khe,
±1 = trượt một ô). Độ tin của hộp người: keep_v5 (trượt ≤ 1,32 %) ⊂ keep (1,32 %) ⊂ keep_high (1,88 %) ⊂ khong (19,4 %).
Vế nhãn: v1p/strict/p4 của borg_endtoend cells.csv (V1+ = biến thể Unihan/OpenCC/kJapanese).

Ra: measure_out/_thu_nghiem_anh_chu/TN5/truth.csv (+ truth_summary.json).
    .venv/bin/python lab/thu_nghiem_anh_chu/TN5_viet_tay/h01_truth.py
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts/measure")]
OUT = REPO / "measure_out/_thu_nghiem_anh_chu/TN5"
BOOKS = {"SachKinhThayCaBinh": "B18", "SachDungLyHoThan": "B34"}


def rd(p, **k):
    return pd.read_csv(p, dtype=str, keep_default_na=False, **k)


def bb(s):
    try:
        v = json.loads(s)
        return [float(x) for x in v[:4]] if v and len(v) >= 4 else None
    except Exception:  # noqa: BLE001
        return None


def human_index(book: str) -> dict:
    """{(page, sentence_id): (offset idx của câu, chuỗi Lo của nom_clean)} theo đúng human_seq (pipeline.borg_human.geom)."""
    out = {}
    for f in sorted((REPO / "prepared" / book / "transcriptions").glob("page_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        off = 0
        for s in d["sentences"]:
            cs = "".join(c for c in s["nom_clean"] if unicodedata.category(c) == "Lo")
            out[(f.stem, s["sentence_id"])] = (off, cs)
            off += len(cs)
    return out


def main():
    import borg_endtoend_eval as BE
    OUT.mkdir(parents=True, exist_ok=True)
    L = rd(REPO / "dataset/_ALL/labels.csv", usecols=["cell_uid", "book_set", "page", "column", "tier", "label", "bbox"])
    GE = rd(REPO / "dataset/_ALL/gold_exact.csv", usecols=["cell_uid", "ink_cx", "ink_cy", "gold_exact", "reason"])
    HB = rd(REPO / "dataset/_BORG_NHAN_NGUOI/labels.csv", usecols=["book", "page", "idx", "sent", "char", "kind", "bbox", "keep_level"])
    rows, summ = [], {}
    for book, s8 in BOOKS.items():
        C = rd(REPO / "measure_out" / book / "borg_endtoend/cells.csv")
        C = C[C.tier == "GOLD"]
        H = BE.human_pages(book)
        spans = BE.load_spans(book)
        hidx = human_index(book)
        hb = HB[HB.book == book].copy()
        hb["i"] = hb.idx.astype(int)
        hbox = {(p, i): (bb(b), k, kl, ch) for p, i, b, k, kl, ch in zip(hb.page, hb.i, hb.bbox, hb.kind, hb.keep_level, hb.char)}
        by_page = {}
        for p, g in hb.groupby("page"):
            B = np.array([bb(b) or [np.nan] * 4 for b in g.bbox], float)
            by_page[p] = (g.i.values, B)
        n_sent_mismatch = 0
        for r in C.itertuples(index=False):
            pg, col, si = r.page, int(r.column), int(r.syl_idx)
            j0 = spans.get((pg, col))
            rec = dict(cell_uid=r.cell_uid, set8=s8, book_set=book, page=pg, column=col, syl_idx=si, label=r.label, gt=r[C.columns.get_loc("gt")],
                       v1p=r.v1p == "True", strict=r.strict == "True", p4=r.p4 == "True", sid="", k=-1, h_idx=-1, h_char="",
                       h_kind="", keep_level="", h_bbox="")
            if j0 is not None and si >= 0:
                J = j0 + si
                acc = 0
                for s in H[pg]["sents"]:
                    if acc <= J < acc + len(s["syl"]):
                        k = J - acc
                        rec["sid"], rec["k"] = s["sid"], k
                        if s["eq"]:
                            off, cs = hidx.get((pg, s["sid"]), (None, ""))
                            if off is not None and cs == s["nom"]:
                                rec["h_idx"] = off + k
                            elif off is not None:
                                n_sent_mismatch += 1
                        break
                    acc += len(s["syl"])
            if rec["h_idx"] >= 0:
                b, kd, kl, ch = hbox.get((pg, rec["h_idx"]), (None, "", "", ""))
                rec.update(h_char=ch, h_kind=kd, keep_level=kl, h_bbox=json.dumps(b) if b else "")
            rows.append(rec)
        summ[book] = dict(n_gold=len(C), n_sent_mismatch=n_sent_mismatch)
    T = pd.DataFrame(rows)
    T = T.merge(L[["cell_uid", "bbox"]], on="cell_uid", how="left").merge(GE, on="cell_uid", how="left")
    assert T.bbox.notna().all() and T.gold_exact.notna().all(), "ô GOLD thiếu trong labels.csv/gold_exact.csv"
    A = np.array([bb(b) or [np.nan] * 4 for b in T.bbox], float)
    Hh = np.array([bb(b) or [np.nan] * 4 for b in T.h_bbox], float)
    cxo, cyo = (A[:, 0] + A[:, 2]) / 2, (A[:, 1] + A[:, 3]) / 2
    cxn = pd.to_numeric(T.ink_cx, errors="coerce").values; cyn = pd.to_numeric(T.ink_cy, errors="coerce").values
    inside = lambda x, y, B: (x >= B[:, 0]) & (x <= B[:, 2]) & (y >= B[:, 1]) & (y <= B[:, 3])
    has_h = ~np.isnan(Hh[:, 0])
    T["has_hbox"] = has_h
    T["slot_old"] = np.where(has_h, inside(cxo, cyo, Hh), False)
    T["slot_new"] = np.where(has_h & ~np.isnan(cxn), inside(cxn, cyn, Hh), False)
    ix0 = np.maximum(A[:, 0], Hh[:, 0]); iy0 = np.maximum(A[:, 1], Hh[:, 1])
    ix1 = np.minimum(A[:, 2], Hh[:, 2]); iy1 = np.minimum(A[:, 3], Hh[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area = lambda B: (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    T["iou_old"] = np.where(has_h, inter / (area(A) + area(Hh) - inter), np.nan)

    def delta(x, y):
        out = np.full(len(T), np.nan)
        for i, (pg, bk, hi) in enumerate(zip(T.page, T.book_set, T.h_idx)):
            if hi < 0 or np.isnan(x[i]):
                continue
            ids, B = PAGES[(bk, pg)]
            m = inside(x[i], y[i], B)
            if m.any():
                out[i] = ids[m][np.argmin(np.abs(ids[m] - hi))] - hi
        return out
    PAGES = {}
    for book in BOOKS:
        hb = HB[HB.book == book]
        for p, g in hb.groupby("page"):
            PAGES[(book, p)] = (g.idx.astype(int).values, np.array([bb(b) or [np.nan] * 4 for b in g.bbox], float))
    T["d_old"] = delta(cxo, cyo)
    T["d_new"] = delta(cxn, cyn)
    # "một chữ" của CROP CHUẨN: hộp mực giữ lại (ink_box, cache crop chuẩn) so với hộp người: own_cov = phần hộp mực nằm
    # trong hộp người của khe; nb_cov = phần lớn nhất của MỘT hộp người khác (cùng trang) bị hộp mực phủ.
    import pickle
    ink = {}
    for book in BOOKS:
        for f in sorted((REPO / "prepared/_gold_exact/crop_chuan/pages" / book).glob("*/*.pkl")):
            for r in pickle.load(open(f, "rb"))["recs"]:
                if r.get("status") == "ok" and r.get("ink_box") not in (None, ""):
                    ink[r["cell_uid"]] = bb(r["ink_box"]) if isinstance(r["ink_box"], str) else list(map(float, r["ink_box"]))
    IB = np.array([ink.get(u) or [np.nan] * 4 for u in T.cell_uid], float)
    own = np.full(len(T), np.nan); nbc = np.full(len(T), np.nan)
    for i, (pg, bk, hi) in enumerate(zip(T.page, T.book_set, T.h_idx)):
        if hi < 0 or np.isnan(IB[i, 0]) or not has_h[i]:
            continue
        ids, Bp = PAGES[(bk, pg)]
        ix = np.clip(np.minimum(IB[i, 2], Bp[:, 2]) - np.maximum(IB[i, 0], Bp[:, 0]), 0, None)
        iy = np.clip(np.minimum(IB[i, 3], Bp[:, 3]) - np.maximum(IB[i, 1], Bp[:, 1]), 0, None)
        inter = np.nan_to_num(ix * iy)
        a_ink = max((IB[i, 2] - IB[i, 0]) * (IB[i, 3] - IB[i, 1]), 1.0)
        a_h = np.maximum((Bp[:, 2] - Bp[:, 0]) * (Bp[:, 3] - Bp[:, 1]), 1.0)
        me = ids == hi
        own[i] = float(inter[me].sum() / a_ink) if me.any() else np.nan
        oth = ~me & ~np.isnan(Bp[:, 0])
        nbc[i] = float((inter[oth] / a_h[oth]).max()) if oth.any() else 0.0
    T["own_cov"] = own; T["nb_cov"] = nbc
    # kiểm: chữ người theo hộp == gt của borg_endtoend (cùng câu, cùng vị trí)
    m = (T.h_idx >= 0) & (T["gt"] != "")
    char_eq = float((T.h_char[m] == T["gt"][m]).mean())
    T.to_csv(OUT / "truth.csv", index=False)
    for book in BOOKS:
        t = T[T.book_set == book]
        g = t[t["gt"] != ""]
        kl = t.keep_level
        summ[book].update(
            n_gt=int(len(g)), v1p=round(float(g.v1p.mean()), 4), n_hidx=int((t.h_idx >= 0).sum()), n_hbox=int(t.has_hbox.sum()),
            keep_level=kl.value_counts().to_dict(),
            slot_old_all=round(float(t.slot_old[t.has_hbox].mean()), 4),
            slot_new_all=round(float(t.slot_new[t.has_hbox].mean()), 4),
            slot_old_keep=round(float(t.slot_old[kl.isin(["keep_v5", "keep", "keep_high"])].mean()), 4),
            slot_new_keep=round(float(t.slot_new[kl.isin(["keep_v5", "keep", "keep_high"])].mean()), 4),
            slot_new_keepv5=round(float(t.slot_new[kl == "keep_v5"].mean()), 4),
            d_new_hist_keep=t.d_new[kl.isin(["keep_v5", "keep", "keep_high"])].value_counts(dropna=False).head(8).to_dict(),
            slot_new_given_v1p=round(float(t.slot_new[t.has_hbox & t.v1p & kl.isin(["keep_v5", "keep", "keep_high"])].mean()), 4),
            slot_new_given_not_v1p=round(float(t.slot_new[t.has_hbox & ~t.v1p & (t["gt"] != "") & kl.isin(["keep_v5", "keep", "keep_high"])].mean()), 4))
    summ["char_eq_hbox_vs_gt"] = round(char_eq, 5)
    json.dump(summ, open(OUT / "truth_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summ, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
