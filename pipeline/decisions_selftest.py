"""Selftest pipeline/decisions.py (A-8): schema sai -> lỗi RÕ; mục cho_ky KHÔNG áp; mục
da_ky chỉ áp khi xuat_xu là tệp có thật; unicode lệch -> từ chối; apply_* đúng luật.

    .venv/bin/python -m pipeline.decisions_selftest

Chỉ ghi vào thư mục tạm; đồng thời nạp config/decisions.yaml thật và đòi 0 mục da_ky
ở corpus_readings/di_the khi chưa có tệp xuất xứ người ký (máy không ký thay người).
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline import decisions as D                       # noqa: E402

_passed = _failed = 0


def check(name, cond, extra=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name} {extra}")


def _loi(doc, tmp: Path, name="x.yaml") -> str:
    """Ghi doc ra yaml, load, trả thông báo lỗi ('' nếu nạp được)."""
    p = tmp / name
    p.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        D.load(p)
        return ""
    except D.LoiQuyetDinh as e:
        return str(e)


def main() -> int:
    print("=" * 64)
    print("DECISIONS SELFTEST (A-8)")
    print("=" * 64)
    tmp = Path(tempfile.mkdtemp(prefix="decisions_selftest_"))
    xx = tmp / "nguoi_ky.md"
    xx.write_text("# ai ký, nhìn gì, ngày nào\n", encoding="utf-8")

    base = {
        "corpus_readings": [
            {"id": "cr_vo", "ocr_char": "無", "unicode": "U+7121", "syllable_raw": "vồ", "book": None,
             "so_o": 69, "xuat_xu": "", "trang_thai": "cho_ky"},
            {"id": "cr_nhieu_stt2", "ocr_char": "僥", "unicode": "U+50E5", "syllable_raw": "nhiều",
             "book": "stt2", "so_o": 59, "xuat_xu": str(xx), "trang_thai": "da_ky"},
        ],
        "di_the": [
            {"id": "dt_duc", "quan_sat": ["徳", "德"], "unicode": {"徳": "U+5FB3", "德": "U+5FB7"},
             "chuan": "德", "book": None, "quy_tac": "unicode", "so_o": {"徳": 313, "德": 163},
             "xuat_xu": str(xx), "trang_thai": "da_ky"},
            {"id": "dt_biet", "quan_sat": ["别", "別"], "chuan": "", "book": None, "quy_tac": "unicode",
             "so_o": {"别": 147, "別": 84}, "xuat_xu": "", "trang_thai": "cho_ky"},
            {"id": "dt_vi_stt4", "quan_sat": ["爲", "為"], "chuan": "為", "book": "stt4", "quy_tac": "unicode",
             "so_o": {"爲": 173, "為": 55}, "xuat_xu": str(xx), "trang_thai": "da_ky"},
        ],
        "lop_nham": [
            {"id": "nguoi_2029A_vs_346B", "syllable": "người", "ocr": "㝵", "unicode": "U+3775",
             "den": "REVIEW", "pham_vi": "ngoai_khoa", "xuat_xu": "docs/NGUOI_CHAM_QUYET_DINH.md"},
        ],
        "khoa_o": {"cells": "config/qd01_cells.csv", "decisions": "config/qd01a_decisions.csv"},
    }

    # ---- 1. schema đúng nạp được; ap_* chỉ trả mục da_ky ----
    print("[nạp + phân loại da_ky/cho_ky]")
    err = _loi(base, tmp)
    check("schema đúng -> nạp không lỗi", err == "", err)
    dec = D.load(tmp / "x.yaml")
    check("corpus_readings: chỉ mục da_ky được áp (1/2)",
          [x["id"] for x in dec.ap_corpus_readings()] == ["cr_nhieu_stt2"])
    check("di_the: chỉ mục da_ky được áp (2/3)",
          [x["id"] for x in dec.ap_di_the()] == ["dt_duc", "dt_vi_stt4"])
    check("lop_nham: xuat_xu là tệp thật -> da_ky mặc định", [x["id"] for x in dec.ap_lop_nham()] == ["nguoi_2029A_vs_346B"])
    rep = dec.report()
    check("report đếm đúng n/da_ky/cho_ky + n_cho_ky",
          (rep["corpus_readings"]["n"], rep["corpus_readings"]["da_ky"], rep["corpus_readings"]["cho_ky"],
           rep["di_the"]["da_ky"], rep["di_the"]["cho_ky"], rep["lop_nham"]["da_ky"], rep["n_cho_ky"])
          == (2, 1, 1, 2, 1, 1, 2), str(rep))
    check("report khoa_o kiểm tồn tại tệp", rep["khoa_o"]["cells"]["exists"] is True)

    # ---- 2. schema sai -> lỗi rõ ----
    print("[schema sai -> LoiQuyetDinh rõ ràng]")
    d = copy.deepcopy(base); d.pop("khoa_o")
    check("thiếu mục khoa_o -> lỗi nêu tên mục", "khoa_o" in _loi(d, tmp))
    d = copy.deepcopy(base); d["la"] = []
    check("mục lạ -> lỗi", "mục lạ" in _loi(d, tmp))
    d = copy.deepcopy(base); d["corpus_readings"][0]["them"] = 1
    check("khoá lạ trong mục -> lỗi nêu khoá", "them" in _loi(d, tmp))
    d = copy.deepcopy(base); d["corpus_readings"][0].pop("so_o")
    check("thiếu khoá bắt buộc -> lỗi nêu khoá", "so_o" in _loi(d, tmp))
    d = copy.deepcopy(base); d["corpus_readings"][0]["unicode"] = "U+7122"
    check("unicode lệch một mã điểm -> từ chối", "U+7121" in _loi(d, tmp) and "U+7122" in _loi(d, tmp))
    d = copy.deepcopy(base); d["corpus_readings"][0]["ocr_char"] = "無無"
    check("ocr_char 2 chữ -> lỗi", "một chữ" in _loi(d, tmp))
    d = copy.deepcopy(base); d["corpus_readings"][0]["syllable_raw"] = "Vồ"
    check("syllable_raw viết hoa -> lỗi", "chữ thường" in _loi(d, tmp))
    d = copy.deepcopy(base); d["corpus_readings"][0]["book"] = "yen2"
    check("book ngoài {null,stt2,stt4,stt11} -> lỗi", "book" in _loi(d, tmp))
    d = copy.deepcopy(base); d["corpus_readings"][0]["trang_thai"] = "ky_roi"
    check("trang_thai lạ -> lỗi", "trang_thai" in _loi(d, tmp))
    d = copy.deepcopy(base); d["corpus_readings"][1]["id"] = "cr_vo"
    check("id trùng -> lỗi", "trùng" in _loi(d, tmp))
    d = copy.deepcopy(base); d["corpus_readings"][0]["so_o"] = "69"
    check("so_o không phải int -> lỗi", "so_o" in _loi(d, tmp))
    d = copy.deepcopy(base); d["di_the"][0]["quan_sat"] = ["徳"]
    check("di_the quan_sat < 2 mã -> lỗi", "quan_sat" in _loi(d, tmp))
    d = copy.deepcopy(base); d["di_the"][0]["chuan"] = "得"
    check("di_the chuan ∉ quan_sat -> lỗi", "∉ quan_sat" in _loi(d, tmp))
    d = copy.deepcopy(base); d["di_the"][0]["quy_tac"] = "tuy_y"
    check("di_the quy_tac lạ -> lỗi", "quy_tac" in _loi(d, tmp))
    d = copy.deepcopy(base); d["di_the"][0]["so_o"] = {"得": 1}
    check("di_the so_o khoá ∉ quan_sat -> lỗi", "so_o" in _loi(d, tmp))
    d = copy.deepcopy(base); d["di_the"][0]["unicode"] = {"徳": "U+5FB7"}
    check("di_the unicode lệch -> từ chối", "U+5FB3" in _loi(d, tmp))
    d = copy.deepcopy(base); d["di_the"][0]["chuan"] = ""
    check("di_the da_ky mà chuan trống -> từ chối (người phải chọn CHIỀU)", "CHIỀU" in _loi(d, tmp))
    d = copy.deepcopy(base)
    d["di_the"].append({"id": "dt_duc2", "quan_sat": ["徳", "得"], "chuan": "得", "book": "stt2",
                        "quy_tac": "unicode", "so_o": {}, "xuat_xu": str(xx), "trang_thai": "da_ky"})
    check("một mã thuộc hai mục da_ky cùng sách -> lỗi", "hai mục da_ky" in _loi(d, tmp))
    d = copy.deepcopy(base); d["lop_nham"][0]["den"] = "SYLLABLE"
    check("lop_nham den ∉ {REVIEW} -> lỗi", "den" in _loi(d, tmp))
    d = copy.deepcopy(base); d["lop_nham"][0]["xuat_xu"] = "docs/KHONG_CO.md"
    check("lop_nham xuat_xu không tồn tại -> từ chối", "không tồn tại" in _loi(d, tmp))
    d = copy.deepcopy(base); d["khoa_o"] = {"cells": "a.csv"}
    check("khoa_o thiếu decisions -> lỗi", "khoa_o" in _loi(d, tmp))

    # ---- 3. xuất xứ: da_ky đòi tệp thật; cho_ky có xuat_xu vẫn KHÔNG áp ----
    print("[giao ước xuất xứ]")
    d = copy.deepcopy(base); d["corpus_readings"][1]["xuat_xu"] = ""
    check("da_ky không khai xuat_xu -> từ chối", "không khai `xuat_xu`" in _loi(d, tmp))
    d = copy.deepcopy(base); d["corpus_readings"][1]["xuat_xu"] = str(tmp / "khong_co.md")
    check("da_ky xuat_xu không tồn tại -> từ chối", "không tồn tại" in _loi(d, tmp))
    d = copy.deepcopy(base); d["corpus_readings"][0]["xuat_xu"] = str(xx)     # cho_ky nhưng có tệp
    err = _loi(d, tmp, "y.yaml")
    dec2 = D.load(tmp / "y.yaml")
    check("cho_ky dù có xuat_xu tệp thật -> vẫn KHÔNG áp (máy không ký thay người)",
          err == "" and [x["id"] for x in dec2.ap_corpus_readings()] == ["cr_nhieu_stt2"])

    # ---- 4. apply_corpus_readings / apply_di_the ----
    print("[apply_*]")
    def rec(book, ch, raw, label, tier, rule, locked=0):
        return {"book": book, "ocr_char": ch, "syllable_raw": raw, "syllable": raw.lower(),
                "label": label, "tier": tier, "rule": rule, "qd01_locked": locked, "dict_support": 5,
                "tier_goc": "", "rule_goc": ""}
    recs = [rec("stt2", "僥", "Nhiều", "", "SYLLABLE", "syl_ctx:bigram"),      # da_ky, sách khớp
            rec("stt4", "僥", "nhiều", "", "REVIEW", "no_context"),            # book stt2 ≠ stt4 -> không
            rec("stt2", "無", "vồ", "", "REVIEW", "no_context"),               # cho_ky -> không
            rec("stt2", "僥", "nhiều", "𠊚", "GOLD", "x", locked=1),            # khoá QĐ-01 -> không
            rec("stt2", "僥", "nhiêu", "僥", "GOLD", "s1_inter_s2_direct")]     # âm khác -> không
    n = D.apply_corpus_readings(recs, dec)
    r = recs[0]
    check("corpus_readings da_ky: (ocr_char, syllable_raw.lower(), book) khớp -> GOLD corpus_reading:<id>, "
          "label=ocr_char, dict_support 'corpus', tier_goc/rule_goc giữ",
          dict(n) == {"cr_nhieu_stt2": 1} and (r["tier"], r["rule"], r["label"], r["dict_support"], r["syllable"],
                                              r["tier_goc"], r["rule_goc"])
          == ("GOLD", "corpus_reading:cr_nhieu_stt2", "僥", "corpus", "nhiều", "SYLLABLE", "syl_ctx:bigram"), str(r))
    check("sách khác / cho_ky / ô khoá / âm khác -> không đụng",
          recs[1]["tier"] == "REVIEW" and recs[2]["tier"] == "REVIEW" and recs[3]["rule"] == "x"
          and recs[4]["rule"] == "s1_inter_s2_direct")
    check("dec=None -> không áp gì", D.apply_corpus_readings([rec("stt2", "僥", "nhiều", "", "REVIEW", "r")], None) == {})

    recs = [{"book": "stt2", "label": "徳"}, {"book": "stt4", "label": "德"}, {"book": "stt11", "label": "别"},
            {"book": "stt4", "label": "爲"}, {"book": "stt2", "label": "爲"}, {"book": "stt2", "label": ""}]
    n = D.apply_di_the(recs, dec)
    check("di_the da_ky: 徳->德 (mọi sách); 德 giữ; 别 cho_ky giữ; 爲->為 CHỈ stt4; nhãn rỗng giữ rỗng",
          [x["label_canonical"] for x in recs] == ["德", "德", "别", "為", "爲", ""]
          and dict(n) == {"dt_duc|stt2": 1, "dt_vi_stt4|stt4": 1}, str(recs))
    check("di_the KHÔNG đổi label", [x["label"] for x in recs] == ["徳", "德", "别", "爲", "爲", ""])
    recs = [{"book": "stt2", "label": "徳"}]
    D.apply_di_the(recs, None)
    check("dec=None -> label_canonical = label", recs[0]["label_canonical"] == "徳")

    # ---- 5. tệp thật trong repo ----
    print("[config/decisions.yaml thật]")
    real = D.load(D.DEFAULT_PATH)
    rr = real.report()
    check("config/decisions.yaml nạp không lỗi, đủ 4 mục", rr["corpus_readings"]["n"] > 0 and rr["di_the"]["n"] > 0)
    check("corpus_readings/di_the: mục da_ky có xuất xứ hợp lệ, máy không tự ký",
          all(D.kiem_xuat_xu(x, "di_the") or x.get("trang_thai") == "cho_ky" for x in real.di_the))
    check("di_the thật: mọi mục da_ky có chuan ∈ quan_sat, so_o cả hai chiều",
          all((x["chuan"] in x["quan_sat"] if x.get("trang_thai") == "da_ky" else x["chuan"] == "")
              and len(x["quan_sat"]) == 2 and set(x["so_o"]) == set(x["quan_sat"])
              for x in real.di_the))
    check("lop_nham thật: ('người', 㝵) da_ky với xuat_xu docs/NGUOI_CHAM_QUYET_DINH.md",
          [(x["syllable"], x["ocr"], x["_ap"]) for x in real.lop_nham] == [("người", "㝵", True)])
    lo = [x for x in real.corpus_readings if x["ocr_char"] == "無"]
    check("corpus_readings thật có ứng viên (無, vồ) và (僥, nhiều)",
          len(lo) == 1 and lo[0]["syllable_raw"] == "vồ"
          and any(x["ocr_char"] == "僥" and x["syllable_raw"] == "nhiều" for x in real.corpus_readings))

    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
