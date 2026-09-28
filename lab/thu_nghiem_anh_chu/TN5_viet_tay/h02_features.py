#!/usr/bin/env python3
"""TN5 h02 — bảng tín hiệu cho MỌI ô GOLD chữ viết tay (Borg B18/B34 + STT stt2/stt4/stt11). 0 API, CPU, không mở ảnh.

Nguồn: dataset/_ALL/gold_exact.csv (cờ + điểm của bước gold_exact, bản chạy trước khi có profile), bản ghi thô mọi tầng
(signals_geom.load_raw: STT dataset_out/labels_final.csv, Borg prepared/_auto/<S>/dataset_out/labels_gated.csv),
dataset/_ALL/columns.csv, và (chỉ Borg — adapter văn xuôi) prepared/_auto/<S>/transcriptions/*.json.
Tín hiệu DÙNG CHUNG (có ở cả STT lẫn Borg, cùng nghĩa) đánh dấu trong COMMON; tín hiệu riêng adapter Borg (dp_ratio …) chỉ
để đo, KHÔNG được dùng ở profile vì STT không có.

Ra: measure_out/_thu_nghiem_anh_chu/TN5/features.pkl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "measure_out/_thu_nghiem_anh_chu/TN5"
HW = ["B18", "B34", "stt2", "stt4", "stt11"]

# tín hiệu có ở CẢ STT và Borg (cùng định nghĩa) — ứng viên cho profile
COMMON = ["cnt", "bc", "ady", "adx", "vis_z", "lobo_pT", "lobo_pL", "lobo_nh", "lobo_cert", "simg", "f_blank", "f_cut",
          "f_two", "f_ink", "bleed_new", "trunc_new", "tall_new", "one_char_ok", "dup_bbox", "ov_heavy", "core_loss",
          "ink_ratio", "d_on", "ad_on", "d_dn", "ad_dn", "d_do", "rel_pos", "n_qn", "run_gold", "flank_gold", "p_register",
          "dict_support", "stray_ink", "border_ink", "R_size", "col_gold_frac", "pitch_ocr"]
BORG_ONLY = ["dp_ratio", "n_match", "d_kim_syl", "page_match_ratio", "page_syll_cov"]


def rd(p, **k):
    return pd.read_csv(p, dtype=str, keep_default_na=False, **k)


def num(s):
    return pd.to_numeric(pd.Series(s).replace("", np.nan), errors="coerce").values


def main():
    from pipeline.gold_exact import signals_geom as SG
    from pipeline.gold_exact.common import Assets, R_of, set_lexicon
    OUT.mkdir(parents=True, exist_ok=True)
    GE = rd(REPO / "dataset/_ALL/gold_exact.csv")
    GE = GE[GE.set8.isin(HW)].reset_index(drop=True)
    F = GE[["cell_uid", "book_set", "book", "set8", "label", "gold_exact", "reason"]].copy()
    for c in ["ady", "adx", "vis_z", "lobo_pT", "lobo_pL", "lobo_nh", "lobo_cert", "core_loss", "ink_ratio", "m_hom"]:
        F[c] = num(GE[c])
    for c in ["cnt", "bc", "simg", "f_blank", "f_cut", "f_two", "f_ink", "bleed_new", "trunc_new", "tall_new", "one_char_ok",
              "dup_bbox", "ov_heavy", "int_foreign", "rescue", "similar", "weak_text", "mocr"]:
        F[c] = GE[c].replace("", "0").astype(int).values
    raw = SG.load_raw(sorted(set(F.book_set)))
    raw = raw.drop_duplicates("key").set_index("key")
    R = raw.reindex(F.cell_uid)
    assert R.tier.notna().all(), "ô GOLD thiếu bản ghi thô"
    n_ocr, n_qn, n_det = num(R.n_ocr), num(R.n_qn), num(R.n_det)
    F["n_qn"] = n_qn
    F["d_on"] = n_ocr - n_qn; F["ad_on"] = np.abs(F.d_on)
    F["d_dn"] = n_det - n_qn; F["ad_dn"] = np.abs(F.d_dn)
    F["d_do"] = n_det - n_ocr
    si = num(R.syl_idx)
    F["syl_idx"] = si
    F["rel_pos"] = si / np.maximum(n_qn - 1, 1)
    F["pitch_ocr"] = (R.count_source.values == "pitch_ocr").astype(int)
    for c in ["flank_gold", "p_register", "dict_support", "stray_ink", "border_ink"]:
        F[c] = num(R[c])
    set_lexicon(Assets())
    F["R_size"] = [len(R_of(s)) for s in R.syllable.values]
    # ngữ cảnh cột (mọi tầng của bản ghi thô): tỉ lệ ô GOLD trong cột, độ dài chuỗi GOLD liên tiếp (theo syl_idx) chứa ô
    rr = raw.reset_index()
    rr["ck"] = rr.book_set + "|" + rr.book + "|" + rr.page + "|" + rr.column.astype(str)
    rr["si"] = pd.to_numeric(rr.syl_idx, errors="coerce")
    gold_by = {k: set(g.si[g.tier == "GOLD"].astype(int)) for k, g in rr.groupby("ck")}
    ncell = rr.groupby("ck").size().to_dict()
    ck = (R.book_set + "|" + R.book + "|" + R.page + "|" + R.column.astype(str)).values
    run = np.zeros(len(F), int); frac = np.zeros(len(F))
    for i, (k, s) in enumerate(zip(ck, si)):
        S = gold_by.get(k, set()); s = int(s)
        a = s
        while a - 1 in S:
            a -= 1
        b = s
        while b + 1 in S:
            b += 1
        run[i] = b - a + 1
        frac[i] = len(S) / max(1, ncell.get(k, 1))
    F["run_gold"] = run; F["col_gold_frac"] = frac
    # riêng adapter Borg: tín hiệu cột/trang của DP ghép cột ↔ âm
    for c in BORG_ONLY:
        F[c] = np.nan
    for bs in ("SachKinhThayCaBinh", "SachDungLyHoThan"):
        m = (F.book_set == bs).values
        if not m.any():
            continue
        colinfo, pginfo = {}, {}
        for f in sorted((REPO / "prepared/_auto" / bs / "transcriptions").glob("page_*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            dp = d.get("dp") or {}
            if isinstance(dp, str):
                import ast
                dp = ast.literal_eval(dp)
            pginfo[f.stem] = (dp.get("match_ratio", np.nan), dp.get("syll_coverage", np.nan))
            for c in d.get("columns") or []:
                colinfo[(f.stem, int(c["column"]))] = (c.get("dp_ratio", np.nan), c.get("n_match", np.nan),
                                                       (c.get("n_chars_kim") or 0) - (c.get("num_syllables") or 0))
        idx = np.nonzero(m)[0]
        for i in idx:
            pg, col = R.page.values[i], int(R.column.values[i])
            a = colinfo.get((pg, col), (np.nan,) * 3); b = pginfo.get(pg, (np.nan, np.nan))
            F.loc[i, ["dp_ratio", "n_match", "d_kim_syl", "page_match_ratio", "page_syll_cov"]] = [*a, *b]
    F["page"] = R.page.values; F["column"] = R.column.values
    F.to_pickle(OUT / "features.pkl")
    print(F.groupby("set8").size().to_dict(), "cột:", len(F.columns))


if __name__ == "__main__":
    main()
