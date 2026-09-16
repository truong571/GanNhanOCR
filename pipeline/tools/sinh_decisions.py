"""Sinh ỨNG VIÊN cho config/decisions.yaml (A-8, flow N5f/N5i) — máy đề xuất, người ký.

VÌ SAO CÓ TỆP NÀY
-----------------
Mỗi mục trong decisions.yaml là một phán quyết theo LỚP; để người ký được thì ứng viên
phải TÁI LẬP ĐƯỢC từ số liệu (không gõ tay). Hai nguồn:

  corpus_readings  ô rule `s1_inter_s2_direct_am_sua_dau` (L1 đường cũ, 755 ô) trong
                   dataset_out_v3/labels.csv mà CỔNG 2-GRAM ÂM–ÂM LOO (tier_v3.syl_bigram_count,
                   dựng từ cols.pkl = syllables sau normalize_column, độc lập DP) ủng hộ âm GỐC
                   (n_gốc > n_mới) — tức L1 đã ghi đè bản in sai; gom theo (ocr_char, âm gốc)
                   ≥ 5 ô. Âm gốc lấy từ cols.pkl[syl][syl_idx] (labels_v3 chưa có syllable_raw;
                   kiểm: 82.003/82.003 ô không-L1 có syl[syl_idx].lower() == syllable).
  di_the           lớp MỘT HÌNH (v == MOT) trong lab/gan_nhan_2026-09-13/e3_all_classes.csv
                   mà hai mã là dị thể Unicode thuần: có quan hệ trong Unihan_Variants
                   (kZ/kSemantic/kSimplified/kTraditional/kSpecializedSemantic) HOẶC nằm trong
                   danh sách DANH_MUC A-8 / DANH_GIA §2.7. Ghi CẢ HAI mã + số ô mỗi mã (đếm
                   trên labels.csv, mọi tier có label); `chuan` ĐỂ TRỐNG — người ký chọn chiều.
                   (cùng, 共/其) không phải dị thể Unicode nhưng E3 đo một hình -> ghi cho_ky
                   để người thấy (N5j hoãn: chỉ áp khi có xuat_xu).

LUỸ ĐẲNG: nếu tệp đích đã có, mục cùng `id` được GIỮ NGUYÊN (kể cả chữ ký người), chỉ
thêm mục mới; lop_nham/khoa_o giữ nguyên nếu đã có. Không bao giờ ghi trang_thai da_ky.

    .venv/bin/python -m pipeline.tools.sinh_decisions \\
        --labels dataset_out_v3/labels.csv --cols lab/gan_nhan_2026-09-13/cols.pkl \\
        --e3 lab/gan_nhan_2026-09-13/e3_all_classes.csv \\
        [--unihan /path/Unihan_Variants.txt] --out config/decisions.yaml
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pipeline.align_engine import tier_v3 as tv3          # noqa: E402
from pipeline.decisions import MUC, nfc, uni              # noqa: E402

NA = dict(dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
RULE_L1 = "s1_inter_s2_direct_am_sua_dau"
MIN_NHOM = 5
# Dị thể Unicode thuần theo DANH_MUC_SUA_DOI_CUOI §A-8 + DANH_GIA §2.7 (không cần Unihan)
DI_THE_TAY = [("徳", "德"), ("别", "別"), ("爲", "為"), ("廪", "廩"), ("强", "強"), ("没", "沒"),
              ("段", "断"), ("事", "亊"), ("庄", "庒")]
# shinjitai / CJK-compat: cùng chữ nhưng KHÔNG có trong Unihan_Variants 17.0 lẫn danh sách
# tài liệu — ghi rõ nguồn kiểm để người ký tự thẩm định
DI_THE_SHINJITAI = [("亜", "亞"), ("実", "实"), ("毎", "每"), ("凛", "凜")]
# E3 một hình nhưng KHÔNG phải dị thể Unicode — chỉ ghi cho người thấy (N5j hoãn)
KHONG_UNICODE_CHO_KY = [("共", "其")]
UNIHAN_FIELDS = ("kZVariant", "kSemanticVariant", "kSimplifiedVariant", "kTraditionalVariant",
                 "kSpecializedSemanticVariant")

HEADER = """\
# QUYẾT ĐỊNH THEO LỚP — người ký, máy chỉ nạp/kiểm/thi hành (A-8, flow N5f/N5h/N5i).
# Nạp bởi pipeline/decisions.py (schema + xuất xứ); build_dataset PASS 1c thi hành.
#
# 🔴 CHỈ MỤC `trang_thai: da_ky` VÀ `xuat_xu` trỏ tới TỆP CÓ THẬT mới được áp.
#    `cho_ky` = ứng viên máy sinh (pipeline/tools/sinh_decisions.py), chỉ được ĐẾM trong
#    decisions_report.json. Máy KHÔNG ký thay người; KHÔNG tự chọn `chuan`.
#
# corpus_readings: (ocr_char, syllable_raw) là cách đọc thật của ngữ liệu dù từ điển không có
#   -> GOLD, rule corpus_reading:<id>, dict_support 'corpus'; KHÔNG vào qn_to_nom, KHÔNG làm neo.
#   Ứng viên = ô L1 (755) mà cổng 2-gram ÂM–ÂM LOO ủng hộ âm GỐC, nhóm ≥ 5 ô.
# di_the: hai mã Unicode cùng một hình -> cột label_canonical (label KHÔNG đổi).
#   `quan_sat` ghi CẢ HAI mã, `so_o` đếm từng mã; người ký điền `chuan` (∈ quan_sat) +
#   `quy_tac` (da_so_ngu_lieu | unicode) + `xuat_xu`; `book: null` = mọi sách.
# lop_nham: ô NGOÀI khoá QĐ-01, âm `syllable`, nhãn máy == `ocr` -> `den` (REVIEW).
# khoa_o: đường dẫn khoá QĐ-01 (build đọc theo --qd01-cells/--qd01a-decisions).
"""


def _doc_unihan(path: str | None) -> dict[tuple, set]:
    rel: dict[tuple, set] = defaultdict(set)
    if not path:
        return rel
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        cp, fld, vals = line.rstrip("\n").split("\t")
        if fld not in UNIHAN_FIELDS:
            continue
        a = chr(int(cp[2:], 16))
        for v in vals.split():
            b = chr(int(v.split("<")[0][2:], 16))
            rel[(a, b)].add(fld)
            rel[(b, a)].add(fld)
    return rel


def sinh_corpus_readings(labels: pd.DataFrame, cols: list) -> tuple[list[dict], dict]:
    cm = {(c["book"], c["page"], int(c["column"])): c for c in cols}
    bsp = tv3.build_bigram_syl_pages(cols)
    l1 = labels[labels["rule"] == RULE_L1]
    st: dict[tuple, Counter] = defaultdict(Counter)
    books: dict[tuple, Counter] = defaultdict(Counter)
    n_bad = 0
    for r in l1.itertuples():
        c = cm.get((r.book, r.page, int(r.column)))
        if c is None or r.syl_idx == "":
            n_bad += 1
            continue
        j, pg = int(r.syl_idx), (r.book, r.page)
        goc, moi = nfc(c["syl"][j]).lower(), nfc(r.syllable).lower()
        n_new = tv3.syl_bigram_count(bsp, c["syl"], j, moi, pg)
        n_old = tv3.syl_bigram_count(bsp, c["syl"], j, goc, pg)
        k = (r.ocr_char, goc, moi)
        st[k]["giu" if n_old > n_new else ("hoa" if n_old == n_new else "doi")] += 1
        if n_old > n_new:
            books[k][r.book] += 1
    out = []
    for (ch, goc, moi), v in sorted(st.items(), key=lambda x: (-x[1]["giu"], x[0])):
        if v["giu"] < MIN_NHOM:
            continue
        bk = dict(books[(ch, goc, moi)])
        out.append({
            "id": f"cr_{uni(ch)[2:].lower()}_{goc}",
            "ocr_char": ch, "unicode": uni(ch), "syllable_raw": goc, "syllable_l1": moi,
            "book": None, "so_o": int(v["giu"]),
            "ghi_chu": (f"L1 nhóm ({ch}, {goc}->{moi}) {sum(v.values())} ô: 2-gram ủng hộ gốc "
                        f"{v['giu']} / đổi {v['doi']} / hoà {v['hoa']}; theo sách (ủng hộ gốc) "
                        + ", ".join(f"{b} {n}" for b, n in sorted(bk.items()))),
            "xuat_xu": "", "trang_thai": "cho_ky",
        })
    tk = {"n_l1": int(len(l1)), "n_khong_join": n_bad,
          "doi": sum(v["doi"] for v in st.values()), "giu": sum(v["giu"] for v in st.values()),
          "hoa": sum(v["hoa"] for v in st.values()), "n_nhom_ge5": len(out),
          "n_o_ge5": sum(x["so_o"] for x in out)}
    return out, tk


def sinh_di_the(labels: pd.DataFrame, e3: pd.DataFrame, rel: dict) -> tuple[list[dict], dict]:
    lab = labels[labels["label"] != ""]
    dem = Counter(zip(lab["book"], lab["label"]))
    tay = {frozenset(p) for p in DI_THE_TAY}
    shin = {frozenset(p) for p in DI_THE_SHINJITAI}
    cho = {frozenset(p) for p in KHONG_UNICODE_CHO_KY}
    pairs: dict[frozenset, list] = defaultdict(list)
    for r in e3.itertuples():
        pairs[frozenset([r.M, r.L])].append(r)
    out, n_mot_bo = [], 0
    for k, rows in pairs.items():
        vs = {r.v for r in rows}
        if "MOT" not in vs:
            continue
        a, b = sorted(k)
        u = sorted(rel.get((a, b), set()))
        if u:
            kiem = "Unihan:" + "+".join(u)
        elif k in tay:
            kiem = "danh_sach_DANH_MUC_A-8 (không có trong Unihan_Variants)"
        elif k in shin:
            kiem = "shinjitai/CJK-compat (máy đề xuất, không có trong Unihan_Variants lẫn tài liệu)"
        elif k in cho:
            kiem = "KHÔNG phải dị thể Unicode — E3 đo một hình, N5j hoãn tới khi có xuat_xu"
        else:
            n_mot_bo += 1
            continue
        mot_books = sorted({r.book for r in rows if r.v == "MOT"})
        hai_books = sorted({r.book for r in rows if r.v == "HAI"})     # 'mo' (mờ) không chặn
        # đơn vị (sách, âm): HAI ở sách khác -> một mục cho từng sách MOT (N5i: (stt2, vì) 為/爲)
        scopes = [None] if not hai_books else mot_books
        for bk in scopes:
            bks = [bk] if bk else ["stt2", "stt4", "stt11"]
            so_o = {c: int(sum(dem.get((x, c), 0) for x in bks)) for c in (a, b)}
            theo_sach = "; ".join(f"{x}: {a} {dem.get((x, a), 0)} / {b} {dem.get((x, b), 0)}" for x in bks)
            e3s = ", ".join(f"({r.book}, {r.syl}) {r.M}/{r.L} {r.v} nB={r.nB}" for r in rows)
            out.append({
                "id": f"dt_{uni(a)[2:].lower()}_{uni(b)[2:].lower()}" + (f"_{bk}" if bk else ""),
                "quan_sat": [a, b], "unicode": {a: uni(a), b: uni(b)}, "chuan": "",
                "book": bk, "quy_tac": "unicode", "so_o": so_o, "kiem_unicode": kiem,
                "ghi_chu": f"E3: {e3s}. Đếm ô có label theo sách: {theo_sach}",
                "xuat_xu": "", "trang_thai": "cho_ky",
            })
    out.sort(key=lambda x: x["id"])
    tk = {"n_cap_e3": len(pairs), "n_cap_mot": sum(1 for v in pairs.values() if any(r.v == "MOT" for r in v)),
          "n_mot_khong_unicode_bo": n_mot_bo, "n_muc": len(out),
          "n_o": sum(sum(x["so_o"].values()) for x in out)}
    return out, tk


def _dump(doc: dict, path: Path) -> None:
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=110, default_flow_style=False)
    path.write_text(HEADER + "\n" + body, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(REPO / "dataset_out_v3" / "labels.csv"))
    ap.add_argument("--cols", default=str(REPO / "lab" / "gan_nhan_2026-09-13" / "cols.pkl"))
    ap.add_argument("--e3", default=str(REPO / "lab" / "gan_nhan_2026-09-13" / "e3_all_classes.csv"))
    ap.add_argument("--unihan", default="", help="Unihan_Variants.txt (tuỳ chọn; thiếu -> chỉ danh sách tay)")
    ap.add_argument("--out", default=str(REPO / "config" / "decisions.yaml"))
    a = ap.parse_args()

    labels = pd.read_csv(a.labels, **NA)
    cols = pickle.load(open(a.cols, "rb"))
    e3 = pd.read_csv(a.e3, dtype=str, keep_default_na=False)
    rel = _doc_unihan(a.unihan or None)

    cr, tk_cr = sinh_corpus_readings(labels, cols)
    dt, tk_dt = sinh_di_the(labels, e3, rel)
    print(f"[corpus_readings] L1 {tk_cr['n_l1']} ô: đổi {tk_cr['doi']} / giữ gốc {tk_cr['giu']} / hoà {tk_cr['hoa']} "
          f"-> {tk_cr['n_nhom_ge5']} nhóm ≥{MIN_NHOM} ô ({tk_cr['n_o_ge5']} ô)")
    print(f"[di_the] E3 {tk_dt['n_cap_e3']} cặp, MOT {tk_dt['n_cap_mot']} (bỏ {tk_dt['n_mot_khong_unicode_bo']} "
          f"không phải dị thể Unicode) -> {tk_dt['n_muc']} mục, {tk_dt['n_o']} ô")

    out = Path(a.out)
    old = yaml.safe_load(out.read_text(encoding="utf-8")) if out.exists() else None
    old = old if isinstance(old, dict) else {}
    doc = {m: (old.get(m) if old.get(m) is not None else []) for m in MUC[:3]}
    n_giu = n_them = 0
    for m, new in (("corpus_readings", cr), ("di_the", dt)):
        co = {x.get("id") for x in doc[m] if isinstance(x, dict)}
        for x in new:
            if x["id"] in co:
                n_giu += 1
            else:
                doc[m].append(x)
                n_them += 1
    if not doc["lop_nham"]:
        doc["lop_nham"] = [{
            "id": "nguoi_2029A_vs_346B",     # tên rule giữ theo flow N5h (S7); 㝵 thực là U+3775
            "syllable": "người", "ocr": "㝵", "unicode": "U+3775", "den": "REVIEW",
            "pham_vi": "ngoai_khoa", "xuat_xu": "docs/NGUOI_CHAM_QUYET_DINH.md", "trang_thai": "da_ky",
            "ghi_chu": "N5h: ô 'người' ngoài khoá QĐ-01 mang nhãn 㝵 -> REVIEW; ô 𠊚 giữ GOLD",
        }]
    doc["khoa_o"] = old.get("khoa_o") or {"cells": "config/qd01_cells.csv",
                                          "decisions": "config/qd01a_decisions.csv"}
    _dump(doc, out)
    print(f"[decisions] -> {out}: giữ {n_giu} mục cũ, thêm {n_them} mục mới "
          f"(corpus_readings {len(doc['corpus_readings'])}, di_the {len(doc['di_the'])}, "
          f"lop_nham {len(doc['lop_nham'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
