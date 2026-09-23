#!/usr/bin/env python3
"""align_audit.py — RÀ CHUẨN "CĂN CHỈNH / QUY HOẠCH GÁN" trên 8 bộ đã chạy (0 token, 0 API).

Chỉ ĐỌC: `<labels>.csv` do pipeline sinh + `prepared*/<book>/transcriptions/<page>.{txt,json}`
(nguồn âm QN mà chính engine đọc qua `pipeline.step2_align._get_qn_lines`) + nhãn người
`data/<book>/manifest.tsv` (2 bộ IHR) + phiên âm dị bản (LVT1916 / Kiều 1871+1872).
KHÔNG sửa `pipeline/`, `core/`, `data/`; không chạy build/export.

Hai nhóm phép đo:

 A. BẤT BIẾN CĂN CHỈNH (mục 1) — mỗi bất biến trả (số vi phạm, mẫu số n):
    A1 o_duy_nhat        mỗi ô âm QN (page, column, syl_idx) có tối đa 1 ô ảnh
    A2 syl_idx_trong_cot syl_idx ∈ [0, số âm QN của cột)
    A3 am_dung_vi_tri    labels.syllable == âm QN ở ĐÚNG vị trí syl_idx (mọi tier).
                         Hai phép CHUẨN HOÁ CÓ CHỦ Ý của pipeline được đếm RIÊNG, không tính vi phạm:
                         (i) chỉ khác DẤU (tầng dời dấu / rule `*_am_sua_dau`); (ii) `qn_fix_kind ==
                         "charfix"` VÀ `qn_charfix.fix_syllable(âm thô, khoá từ điển)` cho ĐÚNG
                         `labels.syllable` (tầng L3, 2026-09-24). Ô khai `charfix` mà KHÔNG tái lập
                         được bằng đúng luật L3 vẫn là VI PHẠM (`A3_charfix_khong_tai_lap`).
    A4 gold_am_dung      A3 giới hạn trong GOLD / GOLD_text_only
    A5 khong_muon_am     ô có âm KHÁC vị trí của mình nhưng TRÙNG âm ở vị trí khác của cột
                         (dấu hiệu lấy âm của câu/chữ khác) — tách riêng "xuyên tầng"
    A6 nom_idx_theo_y    trong cột: thứ tự nom_idx == thứ tự tâm y (trên → dưới)
    A7 syl_idx_tang      trong cột: syl_idx tăng nghiêm ngặt theo nom_idx (căn đơn điệu)
    A8 cot_theo_x        trong trang: chỉ số cột tăng ⇒ tâm x giảm (dọc, phải → trái)
    A9 tang_6_8          (lục bát) cột QN đúng 14 âm và tách 6⧺8
    A10 khong_xuyen_tang (lục bát) mọi ô tầng trên nằm TRÊN mọi ô tầng dưới (theo y)
    A11 parity_cau       (lục bát) verse_odd lẻ, verse_even == verse_odd + 1
    A12 verse_no_duy_nhat(lục bát) mỗi số câu xuất hiện đúng 1 lần trong sách
    A13 span_lien_tuc    (văn xuôi) span âm của cột kế tiếp nối liền cột trước trong cùng truyện

 B. TRÔI CĂN CHỈNH (mục 2) — nhãn đúng chữ nhưng SAI Ô:
    - 2 bộ IHR (nhãn NGƯỜI, `data/<book>/manifest.tsv`): với mỗi ô so nhãn máy với chữ GT ở
      vị trí 0 và ±1, ±2 TRONG CÙNG CÂU; phân loại dung / lech_vi_tri / sai_chu.
      Mức CÂU: dịch s ∈ [-2, 2] làm khớp nhiều nhất ⇒ câu "trôi" khi s ≠ 0 mà s làm tăng khớp.
    - LVT1883 / KVK1884 (không có nhãn người): dùng phiên âm dị bản độc lập
      (auto_precision.match_verses) làm chuẩn thay thế, cùng công thức dịch s.

Chạy:
    .venv/bin/python scripts/measure/align_audit.py --book all
    .venv/bin/python scripts/measure/align_audit.py --book LucVanTien1883 --drift
    .venv/bin/python scripts/measure/align_audit.py --selftest
Đầu ra: measure_out/align_audit/{SUMMARY.json, REPORT.md, <Book>_invariants.csv,
        <Book>_violations.csv, <Book>_drift.csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LITHO_TIER = (6, 8)
SHIFTS = (-2, -1, 0, 1, 2)

# kind: stt (chép tay, 9 cột) | litho (lục bát 6⧺8) | prose (văn xuôi)
BOOKS: dict[str, dict] = {
    "SachThanhTruyen2": dict(kind="stt", code="stt2", prepared="prepared/SachThanhTruyen2",
                             labels="dataset_out/labels_final.csv", n_columns=9),
    "SachThanhTruyen4": dict(kind="stt", code="stt4", prepared="prepared/SachThanhTruyen4",
                             labels="dataset_out/labels_final.csv", n_columns=9),
    "SachThanhTruyen11": dict(kind="stt", code="stt11", prepared="prepared/SachThanhTruyen11",
                              labels="dataset_out/labels_final.csv", n_columns=9),
    "LucVanTien1883": dict(kind="litho", code="lucvantien1883", prepared="prepared/LucVanTien1883",
                           labels="prepared/LucVanTien1883/dataset_out/labels_gated.csv", n_columns=10,
                           cross="LucVanTien1883"),
    "KimVanKieu1884": dict(kind="litho", code="kimvankieu1884", prepared="prepared/KimVanKieu1884",
                           labels="prepared/KimVanKieu1884/dataset_out/labels_gated.csv", n_columns=10,
                           cross="KimVanKieu1884"),
    "Chrestomathie1872": dict(kind="prose", code="chrestomathie1872", prepared="prepared/Chrestomathie1872",
                              labels="prepared/Chrestomathie1872/dataset_out/labels_gated.csv", n_columns=None),
    "LucVanTien1916": dict(kind="litho", code="lucvantien1916", prepared="prepared/LucVanTien1916",
                           labels="prepared/LucVanTien1916/dataset_out/labels_gated.csv", n_columns=10,
                           ihr=True),
    "TruyenKieu1872": dict(kind="litho", code="truyenkieu1872", prepared="prepared/TruyenKieu1872",
                           labels="prepared/TruyenKieu1872/dataset_out/labels_gated.csv", n_columns=10,
                           ihr=True),
}
GOLDISH = ("GOLD", "GOLD_text_only")


# --------------------------------------------------------------------------- #
# tiện ích
# --------------------------------------------------------------------------- #
def canon(t: str) -> str:
    """Chuẩn hoá âm GIỐNG pipeline (NFC + tone canon + lower)."""
    from core.text.text_utils import normalize_tone_marks
    return normalize_tone_marks(unicodedata.normalize("NFC", (t or "").strip().lower()))


_QN_KEYS_CACHE: set[str] = set()


def _qn_keys() -> set[str]:
    """Khoá từ điển QN, chuẩn hoá y hệt pipeline — nạp MỘT lần cho cả lần chạy."""
    global _QN_KEYS_CACHE
    if not _QN_KEYS_CACHE:
        from core.text.dictionary import load_qn_to_nom
        _QN_KEYS_CACHE = {canon(k) for k in load_qn_to_nom(str(REPO / "Dict" / "QuocNgu_SinoNom.csv"))}
    return _QN_KEYS_CACHE


def _charfix_apply(raw: str) -> str:
    """Chạy lại ĐÚNG luật L3 (`pipeline.align_engine.qn_charfix`) trên âm thô; không sửa được -> trả nguyên."""
    from pipeline.align_engine.qn_charfix import fix_syllable
    return fix_syllable(raw, _qn_keys()) or raw


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, (c - r) / d), 4), round(min(1.0, (c + r) / d), 4))


def pct(k: int, n: int) -> float:
    return round(100.0 * k / n, 3) if n else 0.0


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else ["(rong)"])
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def base_syl(t: str) -> str:
    """Âm bỏ HẾT dấu thanh/dấu phụ (so 'bằng' ~ 'băng' ~ 'bảng'): dùng để tách
    'pipeline SỬA DẤU cho âm OCR' (thiết kế, rule *_am_sua_dau) khỏi 'lấy nhầm âm khác'."""
    s = unicodedata.normalize("NFD", (t or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("đ", "d")


def is_pua(ch: str) -> bool:
    o = ord(ch)
    return 0xE000 <= o <= 0xF8FF or 0xF0000 <= o <= 0x10FFFD


# --------------------------------------------------------------------------- #
# nạp dữ liệu
# --------------------------------------------------------------------------- #
def load_labels(book: str, cfg: dict, labels_override: str | None = None) -> list[dict]:
    """Ô nhãn của MỘT sách (lọc theo cột `book` vì STT dùng chung một tệp)."""
    p = REPO / (labels_override or cfg["labels"])
    rows = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r.get("book") != cfg["code"]:
            continue
        try:
            r["_col"] = int(r["column"])
            r["_syl"] = int(r["syl_idx"]) if r.get("syl_idx", "") != "" else None
            r["_nom"] = int(r["nom_idx"]) if r.get("nom_idx", "") != "" else None
        except (TypeError, ValueError):
            r["_col"], r["_syl"], r["_nom"] = None, None, None
        try:
            b = json.loads(r.get("bbox") or "[]")
            r["_cy"] = (b[1] + b[3]) / 2.0 if len(b) >= 4 else None
            r["_cx"] = (b[0] + b[2]) / 2.0 if len(b) >= 4 else None
        except Exception:
            r["_cy"] = r["_cx"] = None
        rows.append(r)
    return rows


def load_qn(cfg: dict) -> tuple[dict[str, dict[int, list[str]]], dict[str, dict[int, dict]]]:
    """(âm QN mà ENGINE đọc, siêu dữ liệu cột trong .json) cho mọi trang của sách.

    Âm QN lấy đúng đường của engine: `pipeline.step2_align._get_qn_lines` (STT đi qua
    cache VietOCR/parse_v5; sách mới rơi về .txt của adapter).
    """
    from pipeline.step2_align import _get_qn_lines
    from core.text.dictionary import load_qn_to_nom
    qn_dict = set(load_qn_to_nom(str(REPO / "Dict/QuocNgu_SinoNom.csv")).keys())
    data_dir = REPO / cfg["prepared"]
    n_col = cfg.get("n_columns") or 9
    lines: dict[str, dict[int, list[str]]] = {}
    meta: dict[str, dict[int, dict]] = {}
    for tf in sorted((data_dir / "transcriptions").glob("page_*.json")):
        page = tf.stem
        if page.endswith("_qn_tmp") or "_qn_ocr_cache" in page:
            continue
        try:
            d = json.load(open(tf, encoding="utf-8"))
        except Exception:
            continue
        cols = {c.get("column", i + 1): c for i, c in enumerate(d.get("columns") or [])
                if isinstance(c, dict)}
        meta[page] = cols
        n_exp = len(cols) if cfg.get("kind") == "prose" else n_col
        try:
            ql, _ = _get_qn_lines(data_dir, page, qn_dict, n_columns=n_exp)
        except Exception:
            ql = {}
        lines[page] = ql
    return lines, meta


# --------------------------------------------------------------------------- #
# A. bất biến căn chỉnh
# --------------------------------------------------------------------------- #
def check_invariants(book: str, cfg: dict, rows: list[dict],
                     qn: dict[str, dict[int, list[str]]],
                     meta: dict[str, dict[int, dict]]) -> tuple[list[dict], list[dict]]:
    litho = cfg["kind"] == "litho"
    prose = cfg["kind"] == "prose"
    inv: list[dict] = []
    bad: list[dict] = []

    def add(name: str, k: int, n: int, note: str = "") -> None:
        inv.append(dict(book=book, bat_bien=name, vi_pham=k, n=n, ti_le_pct=pct(k, n),
                        ket_qua="PASS" if k == 0 else "FAIL", ghi_chu=note))

    def flag(r: dict, name: str, chi_tiet: str) -> None:
        bad.append(dict(book=book, bat_bien=name, page=r.get("page", ""), column=r.get("column", ""),
                        syl_idx=r.get("syl_idx", ""), nom_idx=r.get("nom_idx", ""),
                        syllable=r.get("syllable", ""), label=r.get("label", ""),
                        tier=r.get("tier", ""), rule=r.get("rule", ""), chi_tiet=chi_tiet))

    by_col: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_col[(r["page"], r["_col"])].append(r)

    # A1 — mỗi ô âm QN tối đa 1 ô ảnh
    key = Counter((r["page"], r["_col"], r["_syl"]) for r in rows if r["_syl"] is not None)
    dup = sum(v - 1 for v in key.values() if v > 1)
    add("A1_o_duy_nhat", dup, len(rows))
    for (p, c, s), v in key.items():
        if v > 1:
            bad.append(dict(book=book, bat_bien="A1_o_duy_nhat", page=p, column=c, syl_idx=s,
                            nom_idx="", syllable="", label="", tier="", rule="",
                            chi_tiet=f"{v} ô cùng syl_idx"))

    # A2/A3/A4/A5 — vị trí và nội dung âm
    n_a2 = n_a3 = n_a4 = 0
    k_a2 = k_a3 = k_a4 = 0
    n_a5 = k_a5 = k_a5x = 0
    k_a3d = k_a4d = 0          # chỉ khác DẤU (pipeline sửa dấu — thiết kế, không phải lệch ô)
    k_a3c = k_a4c = 0          # (2026-09-24) L3 qn_charfix tái lập được — thiết kế, không phải lệch ô
    for r in rows:
        ql = qn.get(r["page"], {}).get(r["_col"])
        if ql is None or r["_syl"] is None:
            continue
        n_a2 += 1
        if not (0 <= r["_syl"] < len(ql)):
            k_a2 += 1
            flag(r, "A2_syl_idx_trong_cot", f"syl_idx={r['_syl']} / n_qn={len(ql)}")
            continue
        want = canon(ql[r["_syl"]])
        got = canon(r.get("syllable", ""))
        n_a3 += 1
        ok = (got == want)
        only_tone = (not ok) and base_syl(got) == base_syl(want)
        # (2026-09-24) L3 `qn_charfix`: ô khai `qn_fix_kind == charfix` được CHẤP NHẬN chỉ khi
        # chạy lại ĐÚNG luật L3 trên âm thô của transcriptions cho ra ĐÚNG `labels.syllable`.
        # Khai charfix mà không tái lập được thì vẫn tính vi phạm, với tên riêng để thấy ngay.
        charfix_ok = False
        if (not ok) and not only_tone and (r.get("qn_fix_kind") or "") == "charfix":
            charfix_ok = (_charfix_apply(want) == got)
        if not ok:
            if only_tone:
                k_a3d += 1
            elif charfix_ok:
                k_a3c += 1
            else:
                k_a3 += 1
                name = ("A3_charfix_khong_tai_lap"
                        if (r.get("qn_fix_kind") or "") == "charfix" else "A3_am_dung_vi_tri")
                flag(r, name, f"am_o={got!r} vs QN[{r['_syl']}]={want!r}")
        if r.get("tier") in GOLDISH:
            n_a4 += 1
            if not ok:
                if only_tone:
                    k_a4d += 1
                elif charfix_ok:
                    k_a4c += 1
                else:
                    k_a4 += 1
                    flag(r, "A4_gold_am_dung", f"am_o={got!r} vs QN[{r['_syl']}]={want!r}")
        # A5 — âm sai vị trí NHƯNG trùng một vị trí khác của cùng cột (bỏ ca chỉ khác dấu / L3)
        if not ok and got and not only_tone and not charfix_ok:
            n_a5 += 1
            other = [i for i, s in enumerate(ql) if canon(s) == got and i != r["_syl"]]
            if other:
                k_a5 += 1
                cross = litho and any((i < LITHO_TIER[0]) != (r["_syl"] < LITHO_TIER[0])
                                      for i in other) and len(ql) == sum(LITHO_TIER)
                if cross:
                    k_a5x += 1
                flag(r, "A5_khong_muon_am",
                     f"am {got!r} thuoc vi tri {other}" + (" [XUYEN TANG]" if cross else ""))
    add("A2_syl_idx_trong_cot", k_a2, n_a2)
    add("A3_am_dung_vi_tri", k_a3, n_a3,
        f"them {k_a3d} o chi khac DAU (rule *_am_sua_dau) + {k_a3c} o L3 qn_charfix tai lap duoc")
    add("A4_gold_am_dung", k_a4, n_a4, f"them {k_a4d} o chi khac DAU + {k_a4c} o L3 qn_charfix")
    add("A5_khong_muon_am", k_a5, n_a5,
        f"trong do xuyen tang {k_a5x}" if litho else "")

    # A6/A7 — đơn điệu trong cột
    k_a6 = k_a7 = 0
    n_c = 0
    for (p, c), rs in by_col.items():
        rs2 = [r for r in rs if r["_nom"] is not None]
        if len(rs2) < 2:
            continue
        n_c += 1
        srt = sorted(rs2, key=lambda r: r["_nom"])
        ys = [r["_cy"] for r in srt if r["_cy"] is not None]
        if len(ys) >= 2 and any(ys[i] > ys[i + 1] for i in range(len(ys) - 1)):
            k_a6 += 1
            bad.append(dict(book=book, bat_bien="A6_nom_idx_theo_y", page=p, column=c, syl_idx="",
                            nom_idx="", syllable="", label="", tier="", rule="",
                            chi_tiet="tam y khong tang theo nom_idx"))
        ss = [r["_syl"] for r in srt if r["_syl"] is not None]
        if len(ss) >= 2 and any(ss[i] >= ss[i + 1] for i in range(len(ss) - 1)):
            k_a7 += 1
            bad.append(dict(book=book, bat_bien="A7_syl_idx_tang", page=p, column=c, syl_idx="",
                            nom_idx="", syllable="", label="", tier="", rule="",
                            chi_tiet=f"syl_idx khong tang: {ss}"))
    add("A6_nom_idx_theo_y", k_a6, n_c)
    add("A7_syl_idx_tang", k_a7, n_c)

    # A8 — thứ tự cột theo x (dọc: cột 1 = phải nhất)
    k_a8 = n_a8 = 0
    for p in sorted({r["page"] for r in rows}):
        cx = {}
        for (pp, c), rs in by_col.items():
            if pp != p:
                continue
            v = [r["_cx"] for r in rs if r["_cx"] is not None]
            if v:
                cx[c] = sum(v) / len(v)
        if len(cx) < 2:
            continue
        n_a8 += 1
        seq = [cx[c] for c in sorted(cx)]
        if any(seq[i] <= seq[i + 1] for i in range(len(seq) - 1)):
            k_a8 += 1
            bad.append(dict(book=book, bat_bien="A8_cot_theo_x", page=p, column="", syl_idx="",
                            nom_idx="", syllable="", label="", tier="", rule="",
                            chi_tiet="tam x khong giam theo chi so cot"))
    add("A8_cot_theo_x", k_a8, n_a8, "cot 1 = phai nhat (doc phai->trai)")

    if litho:
        # A9 — cột QN đúng 14 âm, tách 6⧺8
        k9 = n9 = 0
        k9b = 0
        for p, cols in qn.items():
            for c, ql in cols.items():
                n9 += 1
                if len(ql) != sum(LITHO_TIER):
                    k9 += 1
                    bad.append(dict(book=book, bat_bien="A9_tang_6_8", page=p, column=c, syl_idx="",
                                    nom_idx="", syllable="", label="", tier="", rule="",
                                    chi_tiet=f"n_qn={len(ql)} != 14"))
                    continue
                lo = (meta.get(p, {}).get(c) or {}).get("len_odd")
                if lo is not None and lo != LITHO_TIER[0]:
                    k9b += 1
                    bad.append(dict(book=book, bat_bien="A9_tang_6_8", page=p, column=c, syl_idx="",
                                    nom_idx="", syllable="", label="", tier="", rule="",
                                    chi_tiet=f"len_odd={lo} != 6 (du 14 am)"))
        add("A9_tang_6_8", k9 + k9b, n9, f"cot !=14 am: {k9}; du 14 nhung len_odd!=6: {k9b}")

        # A10 — không xuyên tầng (hình học)
        k10 = n10 = 0
        for (p, c), rs in by_col.items():
            up = [r["_cy"] for r in rs if r["_syl"] is not None and r["_cy"] is not None
                  and r["_syl"] < LITHO_TIER[0]]
            dn = [r["_cy"] for r in rs if r["_syl"] is not None and r["_cy"] is not None
                  and LITHO_TIER[0] <= r["_syl"] < sum(LITHO_TIER)]
            if not up or not dn:
                continue
            n10 += 1
            if max(up) >= min(dn):
                k10 += 1
                bad.append(dict(book=book, bat_bien="A10_khong_xuyen_tang", page=p, column=c,
                                syl_idx="", nom_idx="", syllable="", label="", tier="", rule="",
                                chi_tiet=f"y_max(tang tren)={max(up):.0f} >= y_min(tang duoi)={min(dn):.0f}"))
        add("A10_khong_xuyen_tang", k10, n10)

        # A11/A12 — số câu
        k11 = n11 = 0
        seen: Counter = Counter()
        for p, cols in meta.items():
            for c, m in cols.items():
                vo = (m.get("verse_odd") or {}).get("verse_no")
                ve = (m.get("verse_even") or {}).get("verse_no")
                if vo is None and ve is None:
                    continue
                n11 += 1
                okp = (isinstance(vo, int) and isinstance(ve, int) and vo % 2 == 1 and ve == vo + 1)
                if not okp:
                    k11 += 1
                    bad.append(dict(book=book, bat_bien="A11_parity_cau", page=p, column=c, syl_idx="",
                                    nom_idx="", syllable="", label="", tier="", rule="",
                                    chi_tiet=f"verse_odd={vo}, verse_even={ve}"))
                for v in (vo, ve):
                    if isinstance(v, int):
                        seen[v] += 1
        add("A11_parity_cau", k11, n11)
        dup_v = sum(v - 1 for v in seen.values() if v > 1)
        miss = 0
        if seen:
            miss = len(set(range(min(seen), max(seen) + 1)) - set(seen))
        add("A12_verse_no_duy_nhat", dup_v + miss, len(seen),
            f"trung {dup_v}; thieu {miss} so trong [{min(seen) if seen else 0}..{max(seen) if seen else 0}]")
        for v, n in sorted(seen.items()):
            if n > 1:
                bad.append(dict(book=book, bat_bien="A12_verse_no_duy_nhat", page="", column="",
                                syl_idx="", nom_idx="", syllable="", label="", tier="", rule="",
                                chi_tiet=f"cau {v} xuat hien {n} lan"))

    if prose:
        # A13 — span âm liên tục theo THỨ TỰ ĐỌC (trái → phải) trong cùng truyện
        seq: list[tuple] = []
        for p in sorted(meta):
            for c, m in meta[p].items():
                ro = m.get("reading_order")
                sp = m.get("syl_span")
                if ro is None:
                    continue
                seq.append((p, ro, c, m.get("story"), sp, m.get("matched")))
        seq.sort(key=lambda t: (t[0], t[1]))
        k13 = n13 = 0
        prev = None
        for p, ro, c, story, sp, matched in seq:
            if not (isinstance(sp, list) and len(sp) == 2
                    and all(isinstance(x, int) for x in sp)):
                prev = None          # cột giữ chỗ / không ghép được: cắt chuỗi, không kết luận
                continue
            if prev is not None and prev[0] == story:
                n13 += 1
                if sp[0] != prev[1]:
                    k13 += 1
                    bad.append(dict(book=book, bat_bien="A13_span_lien_tuc", page=p, column=c,
                                    syl_idx="", nom_idx="", syllable="", label="", tier="", rule="",
                                    chi_tiet=f"span bat dau {sp[0]} != ket thuc truoc {prev[1]} "
                                             f"({'chong' if sp[0] < prev[1] else 'ho'} "
                                             f"{abs(sp[0] - prev[1])} am)"))
            prev = (story, sp[1])
        add("A13_span_lien_tuc", k13, n13, "cot ke tiep trong thu tu doc cung truyen")

    return inv, bad


# --------------------------------------------------------------------------- #
# B. trôi căn chỉnh
# --------------------------------------------------------------------------- #
def _best_shift(labels: list[str], ref: str) -> tuple[int, int, int]:
    """(dịch tốt nhất, khớp ở dịch đó, khớp ở dịch 0) cho một câu."""
    best_s, best_k = 0, sum(1 for i, ch in enumerate(labels) if ch and i < len(ref) and ch == ref[i])
    k0 = best_k
    for s in SHIFTS:
        if s == 0:
            continue
        k = sum(1 for i, ch in enumerate(labels) if ch and 0 <= i - s < len(ref) and ch == ref[i - s])
        if k > best_k:
            best_s, best_k = s, k
    return best_s, best_k, k0


def _drift_from_verses(book: str, verses: list[dict], rows: list[dict]) -> tuple[dict, list[dict]]:
    """verses = [{page, column, syl_offset, n, ref_nom, ...}]; rows = ô nhãn của sách."""
    by_col: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        if r["_syl"] is not None:
            by_col[(r["page"], r["_col"])][r["_syl"]] = r
    cell_stat = Counter()
    shift_stat = Counter()
    out: list[dict] = []
    n_verse_shift = n_verse = 0
    qn_src = Counter()
    for v in verses:
        col = by_col.get((v["page"], int(v["column"])), {})
        off, n, ref = int(v["syl_offset"]), int(v["n"]), v["ref_nom"]
        if n != len(ref):
            continue
        labs = [(col.get(off + i) or {}).get("label", "") or "" for i in range(n)]
        if not any(labs):
            continue
        n_verse += 1
        qn_src[v.get("qn_source") or "ocr"] += 1
        s, ks, k0 = _best_shift(labs, ref)
        shift_stat[s] += 1
        if s != 0:
            n_verse_shift += 1
        for i in range(n):
            lab = labs[i]
            if not lab:
                cell_stat["khong_co_nhan"] += 1
                continue
            g0 = ref[i]
            if is_pua(g0):
                cell_stat["ref_pua"] += 1
                continue
            cell_stat["n"] += 1
            if lab == g0:
                cell_stat["dung"] += 1
                continue
            hit = [d for d in (-1, 1, -2, 2) if 0 <= i - d < len(ref) and ref[i - d] == lab
                   and ref[i - d] != g0]
            if hit:
                cell_stat["lech_vi_tri"] += 1
                cell_stat[f"lech_{hit[0]:+d}"] += 1
                r = col.get(off + i) or {}
                out.append(dict(book=book, page=v["page"], column=v["column"], syl_idx=off + i,
                                lech=hit[0], label=lab, ref_tai_o=g0, ref_o_lech=ref[i - hit[0]],
                                tier=r.get("tier", ""), rule=r.get("rule", ""),
                                syllable=r.get("syllable", ""), nguon="cau"))
            else:
                cell_stat["sai_chu"] += 1
    # phân bố ô lệch theo VỊ TRÍ trong cột / theo CỘT / theo TRANG (trôi hay dồn cụm?)
    by_syl = Counter(c["syl_idx"] for c in out)
    by_col_i = Counter(int(c["column"]) for c in out)
    by_page = Counter(c["page"] for c in out)
    n_col_any = len({(c["page"], c["column"]) for c in out})
    return dict(cells=dict(cell_stat), verses=n_verse, verses_shift=n_verse_shift,
                shift_hist={str(k): v for k, v in sorted(shift_stat.items())},
                lech_theo_syl_idx={str(k): v for k, v in sorted(by_syl.items())},
                lech_theo_cot={str(k): v for k, v in sorted(by_col_i.items())},
                n_cot_co_o_lech=n_col_any, qn_source_cua_cau=dict(qn_src),
                trang_lech_nhieu_nhat=[[p, n] for p, n in by_page.most_common(5)]), out


def drift_ihr(book: str, cfg: dict, rows: list[dict]) -> tuple[dict, list[dict]]:
    """Nhãn NGƯỜI (IHR-NomDB): mỗi câu (page_id, col_index, part) → chuỗi chữ GT."""
    gt: dict[tuple, str] = {}
    for r in csv.DictReader(open(REPO / "data" / book / "manifest.tsv", encoding="utf-8"),
                            delimiter="\t"):
        if r["col_index"] == "" or r["part"] == "":
            continue
        gt[(r["page_id"], int(r["col_index"]), int(r["part"]))] = r["nom_text"]
    pmap = {p["page_name"]: p["page_id"] for p in
            json.loads((REPO / cfg["prepared"] / "manifest.json").read_text(encoding="utf-8"))
            .get("pages", [])}
    verses = []
    for page, pid in pmap.items():
        for c in range(1, (cfg.get("n_columns") or 10) + 1):
            for part, off, n in ((1, 0, LITHO_TIER[0]), (2, LITHO_TIER[0], LITHO_TIER[1])):
                s = gt.get((pid, c - 1, part))
                if s and len(s) == n:
                    verses.append(dict(page=page, column=c, syl_offset=off, n=n, ref_nom=s))
    res, cells = _drift_from_verses(book, verses, rows)
    res["nguon_chuan"] = "nhan nguoi IHR-NomDB (manifest.tsv)"
    res["n_cau_gt"] = len(verses)
    return res, cells


def _ref_path(jf: Path, cache: Path) -> Path | None:
    """Tệp phiên âm dị bản; nếu đã bị xoá khỏi cây làm việc thì lấy lại từ git HEAD
    vào thư mục đo (KHÔNG ghi vào data/)."""
    if jf.exists():
        return jf
    rel = jf.relative_to(REPO).as_posix()
    dst = cache / rel.replace("/", "__")
    if not dst.exists():
        import subprocess
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO, capture_output=True)
        if r.returncode != 0 or not r.stdout:
            print(f"[align_audit] thieu di ban {rel} (khong co trong cay lam viec lan git HEAD)",
                  file=sys.stderr)
            return None
        dst.write_bytes(r.stdout)
    return dst


def drift_cross(book: str, cfg: dict, rows: list[dict], min_frac: float,
                cache: Path | None = None) -> tuple[dict, list[dict]]:
    """Dị bản độc lập làm chuẩn thay thế (auto_precision.match_verses)."""
    import importlib
    ap = importlib.import_module("scripts.measure.auto_precision")
    c = dict(ap.CROSS_BOOKS[cfg["cross"]])
    c["trans"] = REPO / cfg["prepared"] / "transcriptions"
    cache = cache or (REPO / "measure_out/align_audit/_refs")
    refs = [(n, p) for n, p in ((n, _ref_path(Path(p), cache)) for n, p in c["refs"]) if p]
    if not refs:
        return {"nguon_chuan": "thieu tep di ban"}, []
    c["refs"] = refs
    verses, vstat = ap.match_verses(book, c, min_frac)
    res, cells = _drift_from_verses(book, verses, rows)
    res["nguon_chuan"] = "di ban doc lap: " + ", ".join(n for n, _ in c["refs"])
    res["n_cau_khop_di_ban"] = len(verses)
    res["verse_match_stat"] = vstat
    return res, cells


# --------------------------------------------------------------------------- #
# chạy
# --------------------------------------------------------------------------- #
def audit_book(book: str, out: Path, do_drift: bool, min_frac: float,
               labels_override: str | None = None) -> dict:
    cfg = BOOKS[book]
    lp = REPO / (labels_override or cfg["labels"])
    if not lp.exists():
        return {"loi": f"thieu {lp}"}
    rows = load_labels(book, cfg, labels_override)
    qn, meta = load_qn(cfg)
    inv, bad = check_invariants(book, cfg, rows, qn, meta)
    write_csv(out / f"{book}_invariants.csv", inv)
    write_csv(out / f"{book}_violations.csv", bad[:5000],
              ["book", "bat_bien", "page", "column", "syl_idx", "nom_idx", "syllable",
               "label", "tier", "rule", "chi_tiet"])
    # Nguồn số đếm ô mỗi cột (count_source) + cột kim/QN đếm lệch: chỗ căn chỉnh CÓ THỂ trôi
    cols: dict[tuple, dict] = {}
    for r in rows:
        cols.setdefault((r["page"], r["_col"]), r)
    cs = Counter(r.get("count_source", "") for r in cols.values())
    lech = sum(1 for r in cols.values()
               if (r.get("n_ocr") or "").isdigit() and (r.get("n_qn") or "").isdigit()
               and int(r["n_ocr"]) != int(r["n_qn"]))
    res = {"kind": cfg["kind"], "labels": str(lp.relative_to(REPO)), "n_o": len(rows),
           "n_cot": len(cols), "count_source": dict(cs),
           "cot_n_ocr_khac_n_qn": lech, "cot_n_ocr_khac_n_qn_pct": pct(lech, len(cols)),
           "n_trang": len(qn), "tier": dict(Counter(r.get("tier", "") for r in rows)),
           "bat_bien": {i["bat_bien"]: {"vi_pham": i["vi_pham"], "n": i["n"],
                                        "pct": i["ti_le_pct"]} for i in inv},
           "n_FAIL": sum(1 for i in inv if i["ket_qua"] == "FAIL"),
           "n_bat_bien": len(inv), "n_vi_pham_ghi": len(bad)}
    if do_drift:
        if cfg.get("ihr"):
            d, cells = drift_ihr(book, cfg, rows)
        elif cfg.get("cross"):
            d, cells = drift_cross(book, cfg, rows, min_frac, out / "_refs")
        else:
            d, cells = ({"nguon_chuan": "khong co chuan doc lap"}, [])
        res["troi"] = d
        if cells:
            write_csv(out / f"{book}_drift.csv", cells)
    return res


def write_report(summary: dict, out: Path) -> None:
    L = ["# align_audit — bất biến căn chỉnh & trôi căn chỉnh", ""]
    names = sorted({k for b in summary["books"].values() if "bat_bien" in b for k in b["bat_bien"]})
    L.append("| Bất biến | " + " | ".join(summary["books"]) + " |")
    L.append("|---|" + "---|" * len(summary["books"]))
    for nm in names:
        cells = []
        for b in summary["books"].values():
            v = b.get("bat_bien", {}).get(nm)
            cells.append("—" if v is None else f"{v['vi_pham']}/{v['n']}")
        L.append(f"| {nm} | " + " | ".join(cells) + " |")
    L.append("")
    for bk, b in summary["books"].items():
        if "troi" in b:
            c = b["troi"].get("cells", {})
            n = c.get("n", 0)
            L.append(f"- **{bk}** ({b['troi'].get('nguon_chuan')}): n={n}, đúng "
                     f"{pct(c.get('dung', 0), n)} %, lệch vị trí {pct(c.get('lech_vi_tri', 0), n)} %, "
                     f"sai chữ {pct(c.get('sai_chu', 0), n)} %; câu trôi "
                     f"{b['troi'].get('verses_shift')}/{b['troi'].get('verses')}")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def selftest() -> int:
    ok = [True]

    def ck(name, cond):
        print(("  PASS " if cond else "  FAIL ") + name)
        ok[0] &= bool(cond)

    ck("_best_shift: khong troi -> 0", _best_shift(list("ABCDEF"), "ABCDEF") == (0, 6, 6))
    ck("_best_shift: troi +1 -> +1", _best_shift(["", "A", "B", "C", "D", "E"], "ABCDEF")[0] == 1)
    ck("_best_shift: troi -1 -> -1", _best_shift(list("BCDEFG"), "ABCDEFG")[0] == -1)
    v = [dict(page="page_0001", column=1, syl_offset=0, n=6, ref_nom="甲乙丙丁戊己")]
    rows = [{"page": "page_0001", "_col": 1, "_syl": i, "label": ch, "tier": "GOLD"}
            for i, ch in enumerate("乙丙丁戊己庚")]
    res, cells = _drift_from_verses("T", v, rows)
    ck("lech -1 duoc dem la lech_vi_tri (5/6)", res["cells"].get("lech_vi_tri") == 5)
    ck("cau bi danh dau troi", res["verses_shift"] == 1 and res["shift_hist"].get("-1") == 1)
    ck("canon ha chu thuong + NFC", canon(" Trước ") == canon("trước"))
    ck("wilson(0,0) an toan", wilson(0, 0) == (0.0, 0.0))
    ck("is_pua", is_pua("") and not is_pua("甲"))
    print("SELFTEST", "OK" if ok[0] else "HONG")
    return 0 if ok[0] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--book", default="all", help="all | tên sách (phân tách bằng dấu phẩy)")
    ap.add_argument("--labels", default=None, help="ghi đè đường dẫn tệp nhãn (1 sách)")
    ap.add_argument("--out", default="measure_out/align_audit")
    ap.add_argument("--drift", action="store_true", help="chỉ chạy phần trôi (mặc định: chạy cả hai)")
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--min-frac", type=float, default=0.75)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    books = list(BOOKS) if a.book == "all" else [b.strip() for b in a.book.split(",")]
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {"books": {}, "labels_override": a.labels}
    for b in books:
        if b not in BOOKS:
            print(f"[align_audit] bo qua {b} (khong co trong BOOKS)", file=sys.stderr)
            continue
        print(f"[align_audit] {b} …", file=sys.stderr)
        summary["books"][b] = audit_book(b, out, not a.no_drift, a.min_frac,
                                         a.labels if len(books) == 1 else None)
        r = summary["books"][b]
        if "bat_bien" in r:
            print(f"  {r['n_o']} o | FAIL {r['n_FAIL']}/{r['n_bat_bien']} bat bien", file=sys.stderr)
    (out / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    write_report(summary, out)
    print(f"[align_audit] -> {out}/SUMMARY.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
