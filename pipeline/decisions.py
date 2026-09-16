"""decisions.yaml — quyết định NGƯỜI theo LỚP, máy chỉ nạp/kiểm/thi hành (A-8, flow N5f/N5h/N5i).

VÌ SAO CÓ MÔ-ĐUN NÀY
--------------------
Ba loại phán quyết theo lớp đang nằm rải rác (hard-code `NGUOI_NHAM` trong build, bảng dị
thể trong kế hoạch cũ gộp 4 cặp về mã hiếm, cách đọc ngữ liệu 僥/nhiều 無/vồ chỉ có trong
log L1). Gom về MỘT tệp `config/decisions.yaml` với 4 mục:

  corpus_readings  (ocr_char, syllable_raw) là cách đọc THẬT của ngữ liệu dù từ điển không có
                   -> tier GOLD, rule `corpus_reading:<id>`, dict_support 'corpus'.
                   KHÔNG đưa vào qn_to_nom, KHÔNG làm neo (gold_direct_anchors chỉ đếm
                   rule s1_inter_s2_direct) — tránh vòng tự khẳng định (N5f).
  di_the           hai mã Unicode là dị thể của cùng một hình -> cột `label_canonical`
                   (mặc định = label); KHÔNG đổi `label` (H3: hình quan sát / mã chuẩn, N5i).
  lop_nham         lớp nhầm hệ thống: ô ngoài khoá QĐ-01, âm X, nhãn máy == Y -> REVIEW (N5h).
  khoa_o           đường dẫn khoá QĐ-01 (cells + decisions) — chỉ kiểm tồn tại, build đọc
                   theo cờ riêng.

🔴 GIAO ƯỚC XUẤT XỨ (giống glyph_fix.kiem_xuat_xu): mục chỉ được ÁP khi
`trang_thai: da_ky` VÀ `xuat_xu` trỏ tới một tệp CÓ THẬT trong repo. `da_ky` mà thiếu
tệp -> TỪ CHỐI nạp (không cảnh báo rồi chạy tiếp). Mục `cho_ky` chỉ được ĐẾM trong
report — máy không ký thay người. Mã chữ trong mỗi mục phải khớp `unicode` khai kèm
(𠊚 U+2029A và 𠊛 U+2029B chỉ khác một mã điểm).

    .venv/bin/python -m pipeline.decisions [--config config/decisions.yaml] [--report x.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO / "config" / "decisions.yaml"

MUC = ("corpus_readings", "di_the", "lop_nham", "khoa_o")
TRANG_THAI = ("cho_ky", "da_ky")
BOOKS = (None, "stt2", "stt4", "stt11")
QUY_TAC = ("da_so_ngu_lieu", "unicode")
DEN = ("REVIEW",)
PHAM_VI = ("ngoai_khoa",)
_ID_RE = re.compile(r"^[\w\-]+$")          # \w = chữ Unicode (id có thể mang âm QN)

RULE_CORPUS = "corpus_reading"          # rule = corpus_reading:<id>
RULE_LOP_NHAM = "lop_nham"              # rule = lop_nham:<id>


class LoiQuyetDinh(SystemExit):
    """Schema/xuất xứ sai -> dừng hẳn với thông báo rõ (SystemExit để CLI/exit code 1)."""

    def __init__(self, msg: str):
        super().__init__(f"[decisions] 🔴 TỪ CHỐI: {msg}")


def nfc(s) -> str:
    return unicodedata.normalize("NFC", str(s if s is not None else "").strip())


def uni(ch: str) -> str:
    return f"U+{ord(ch):04X}"


def _chu(v, muc: str, key: str) -> str:
    """Đúng MỘT mã chữ (NFC); rỗng/nhiều chữ -> lỗi."""
    s = nfc(v)
    if len(s) != 1:
        raise LoiQuyetDinh(f"{muc}: `{key}: {v!r}` phải là đúng một chữ (NFC), có {len(s)}")
    return s


def _kiem_unicode(ch: str, khai, muc: str) -> None:
    if khai is None or khai == "":
        return
    if str(khai).upper() != uni(ch):
        raise LoiQuyetDinh(f"{muc}: chữ {ch!r} là {uni(ch)} nhưng khai `unicode: {khai}` "
                           f"— lệch một mã điểm là áp sai cả lớp")


def _kiem_khoa(item: dict, muc: str, bat_buoc: tuple, tuy_chon: tuple = ()) -> None:
    if not isinstance(item, dict):
        raise LoiQuyetDinh(f"{muc}: mỗi mục phải là mapping, gặp {type(item).__name__}")
    thieu = [k for k in bat_buoc if k not in item]
    if thieu:
        raise LoiQuyetDinh(f"{muc}: mục {item.get('id', item)!r} thiếu khoá {thieu}")
    la = [k for k in item if k not in bat_buoc and k not in tuy_chon]
    if la:
        raise LoiQuyetDinh(f"{muc}: mục {item.get('id', item)!r} có khoá lạ {la}")


def kiem_xuat_xu(item: dict, muc: str) -> bool:
    """True nếu mục được ÁP (da_ky + tệp xuat_xu tồn tại); False nếu cho_ky.
    da_ky mà xuat_xu rỗng/không tồn tại -> LoiQuyetDinh (như glyph_fix.kiem_xuat_xu)."""
    tt = item.get("trang_thai", "cho_ky")
    if tt not in TRANG_THAI:
        raise LoiQuyetDinh(f"{muc}: mục {item.get('id')!r} có trang_thai {tt!r} ∉ {TRANG_THAI}")
    if tt != "da_ky":
        return False
    xx = str(item.get("xuat_xu") or "").strip()
    if not xx:
        raise LoiQuyetDinh(f"{muc}: mục {item.get('id')!r} ghi da_ky nhưng không khai `xuat_xu`. "
                           "Một phán quyết không truy được người ký thì không hơn phán quyết máy.")
    p = Path(xx) if Path(xx).is_absolute() else REPO / xx
    if not p.is_file():
        raise LoiQuyetDinh(f"{muc}: mục {item.get('id')!r} `xuat_xu: {xx}` không tồn tại trên đĩa. "
                           "Tạo tệp đó và khai: ai ký, nhìn gì, ngày nào.")
    return True


@dataclass
class Decisions:
    path: Path
    corpus_readings: list[dict] = field(default_factory=list)
    di_the: list[dict] = field(default_factory=list)
    lop_nham: list[dict] = field(default_factory=list)
    khoa_o: dict = field(default_factory=dict)

    # ---- mục được áp (đã kiểm xuất xứ lúc load) ----
    def ap_corpus_readings(self) -> list[dict]:
        return [x for x in self.corpus_readings if x["_ap"]]

    def ap_di_the(self) -> list[dict]:
        return [x for x in self.di_the if x["_ap"]]

    def ap_lop_nham(self) -> list[dict]:
        return [x for x in self.lop_nham if x["_ap"]]

    def report(self, applied: dict | None = None) -> dict:
        """Đếm da_ky/cho_ky từng mục (+ số ô đã áp nếu build truyền `applied`)."""
        def _dem(items):
            return {"n": len(items),
                    "da_ky": sum(1 for x in items if x["_ap"]),
                    "cho_ky": sum(1 for x in items if not x["_ap"]),
                    "ids_da_ky": [x["id"] for x in items if x["_ap"]],
                    "ids_cho_ky": [x["id"] for x in items if not x["_ap"]]}
        rep = {"path": str(self.path),
               "corpus_readings": _dem(self.corpus_readings),
               "di_the": _dem(self.di_the),
               "lop_nham": _dem(self.lop_nham),
               "khoa_o": {k: {"path": v, "exists": (REPO / v).is_file()}
                          for k, v in self.khoa_o.items()}}
        rep["n_cho_ky"] = sum(rep[m]["cho_ky"] for m in ("corpus_readings", "di_the", "lop_nham"))
        if applied:
            rep["ap"] = applied
        return rep


def _load_corpus_readings(items, muc="corpus_readings") -> list[dict]:
    out, seen_id, seen_key = [], set(), set()
    for it in items:
        _kiem_khoa(it, muc, ("id", "ocr_char", "syllable_raw", "book", "so_o", "xuat_xu", "trang_thai"),
                   ("unicode", "ghi_chu", "nguoi_ky", "ngay", "syllable_l1"))
        it = dict(it)
        if not isinstance(it["id"], str) or not _ID_RE.match(it["id"]) or it["id"] in seen_id:
            raise LoiQuyetDinh(f"{muc}: id {it['id']!r} rỗng/không hợp lệ/trùng")
        seen_id.add(it["id"])
        it["ocr_char"] = _chu(it["ocr_char"], muc, "ocr_char")
        _kiem_unicode(it["ocr_char"], it.get("unicode"), muc)
        s = nfc(it["syllable_raw"])
        if not s or s != s.lower():
            raise LoiQuyetDinh(f"{muc}: {it['id']}: syllable_raw {it['syllable_raw']!r} phải là chữ thường NFC, không rỗng")
        it["syllable_raw"] = s
        if it["book"] not in BOOKS:
            raise LoiQuyetDinh(f"{muc}: {it['id']}: book {it['book']!r} ∉ {BOOKS}")
        if not isinstance(it["so_o"], int) or isinstance(it["so_o"], bool) or it["so_o"] < 0:
            raise LoiQuyetDinh(f"{muc}: {it['id']}: so_o phải là số nguyên ≥ 0")
        key = (it["ocr_char"], s, it["book"])
        if key in seen_key:
            raise LoiQuyetDinh(f"{muc}: {it['id']}: trùng (ocr_char, syllable_raw, book) {key}")
        seen_key.add(key)
        it["_ap"] = kiem_xuat_xu(it, muc)
        out.append(it)
    return out


def _load_di_the(items, muc="di_the") -> list[dict]:
    out, seen_id = [], set()
    for it in items:
        _kiem_khoa(it, muc, ("id", "quan_sat", "chuan", "book", "quy_tac", "so_o", "xuat_xu", "trang_thai"),
                   ("unicode", "kiem_unicode", "ghi_chu", "nguoi_ky", "ngay"))
        it = dict(it)
        if not isinstance(it["id"], str) or not _ID_RE.match(it["id"]) or it["id"] in seen_id:
            raise LoiQuyetDinh(f"{muc}: id {it['id']!r} rỗng/không hợp lệ/trùng")
        seen_id.add(it["id"])
        qs = it["quan_sat"]
        if not isinstance(qs, list) or len(qs) < 2:
            raise LoiQuyetDinh(f"{muc}: {it['id']}: quan_sat phải là danh sách ≥ 2 mã (cả hai chiều)")
        qs = [_chu(c, muc, "quan_sat") for c in qs]
        if len(set(qs)) != len(qs):
            raise LoiQuyetDinh(f"{muc}: {it['id']}: quan_sat có mã trùng {qs}")
        it["quan_sat"] = qs
        u = it.get("unicode") or {}
        if not isinstance(u, dict):
            raise LoiQuyetDinh(f"{muc}: {it['id']}: unicode phải là mapping mã -> U+XXXX")
        for c in qs:
            _kiem_unicode(c, u.get(c), muc)
        chuan = nfc(it["chuan"]) if it["chuan"] is not None else ""
        if chuan and chuan not in qs:
            raise LoiQuyetDinh(f"{muc}: {it['id']}: chuan {chuan!r} ∉ quan_sat {qs}")
        it["chuan"] = chuan
        if it["book"] not in BOOKS:
            raise LoiQuyetDinh(f"{muc}: {it['id']}: book {it['book']!r} ∉ {BOOKS}")
        if it["quy_tac"] not in QUY_TAC:
            raise LoiQuyetDinh(f"{muc}: {it['id']}: quy_tac {it['quy_tac']!r} ∉ {QUY_TAC}")
        so = it["so_o"]
        if not isinstance(so, dict) or any(k not in qs for k in so) \
                or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in so.values()):
            raise LoiQuyetDinh(f"{muc}: {it['id']}: so_o phải là mapping {{mã ∈ quan_sat: số ô ≥ 0}}")
        it["_ap"] = kiem_xuat_xu(it, muc)
        if it["_ap"] and not chuan:
            raise LoiQuyetDinh(f"{muc}: {it['id']}: da_ky nhưng `chuan` trống — người ký phải chọn CHIỀU chuẩn")
        out.append(it)
    # một mã không được thuộc hai mục da_ky cùng phạm vi sách (áp mâu thuẫn)
    scope = {}
    for it in out:
        if not it["_ap"]:
            continue
        for c in it["quan_sat"]:
            for b in ([it["book"]] if it["book"] else BOOKS):
                if (c, b) in scope and scope[(c, b)] != it["id"]:
                    raise LoiQuyetDinh(f"{muc}: mã {c!r} thuộc hai mục da_ky {scope[(c, b)]}/{it['id']} cùng sách {b}")
                scope[(c, b)] = it["id"]
    return out


def _load_lop_nham(items, muc="lop_nham") -> list[dict]:
    out, seen_id = [], set()
    for it in items:
        _kiem_khoa(it, muc, ("id", "syllable", "ocr", "den", "pham_vi", "xuat_xu"),
                   ("unicode", "trang_thai", "ghi_chu", "nguoi_ky", "ngay"))
        it = dict(it)
        if not isinstance(it["id"], str) or not _ID_RE.match(it["id"]) or it["id"] in seen_id:
            raise LoiQuyetDinh(f"{muc}: id {it['id']!r} rỗng/không hợp lệ/trùng")
        seen_id.add(it["id"])
        it["syllable"] = nfc(it["syllable"]).lower()
        if not it["syllable"]:
            raise LoiQuyetDinh(f"{muc}: {it['id']}: syllable rỗng")
        it["ocr"] = _chu(it["ocr"], muc, "ocr")
        _kiem_unicode(it["ocr"], it.get("unicode"), muc)
        if it["den"] not in DEN:
            raise LoiQuyetDinh(f"{muc}: {it['id']}: den {it['den']!r} ∉ {DEN}")
        if it["pham_vi"] not in PHAM_VI:
            raise LoiQuyetDinh(f"{muc}: {it['id']}: pham_vi {it['pham_vi']!r} ∉ {PHAM_VI}")
        it.setdefault("trang_thai", "da_ky")     # lớp nhầm là phán quyết đã có xuất xứ
        it["_ap"] = kiem_xuat_xu(it, muc)
        out.append(it)
    return out


def load(path=DEFAULT_PATH) -> Decisions:
    """Nạp + kiểm schema + kiểm xuất xứ. Sai -> LoiQuyetDinh (SystemExit) với thông báo rõ."""
    p = Path(path)
    if not p.is_file():
        raise LoiQuyetDinh(f"không thấy {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise LoiQuyetDinh(f"{p}: gốc phải là mapping có đủ 4 mục {MUC}")
    thieu = [m for m in MUC if m not in raw]
    la = [k for k in raw if k not in MUC]
    if thieu or la:
        raise LoiQuyetDinh(f"{p}: thiếu mục {thieu}, mục lạ {la} (cần đúng {MUC})")
    for m in MUC[:3]:
        if raw[m] is None:
            raw[m] = []
        if not isinstance(raw[m], list):
            raise LoiQuyetDinh(f"{p}: mục {m} phải là danh sách")
    ko = raw["khoa_o"] or {}
    if not isinstance(ko, dict) or set(ko) != {"cells", "decisions"} \
            or not all(isinstance(v, str) and v for v in ko.values()):
        raise LoiQuyetDinh(f"{p}: khoa_o phải là {{cells: <csv>, decisions: <csv>}}")
    return Decisions(path=p,
                     corpus_readings=_load_corpus_readings(raw["corpus_readings"]),
                     di_the=_load_di_the(raw["di_the"]),
                     lop_nham=_load_lop_nham(raw["lop_nham"]),
                     khoa_o=dict(ko))


# --------------------------------------------------------------------------- #
# Thi hành trên records của build_dataset (PASS 1c)
# --------------------------------------------------------------------------- #
def apply_corpus_readings(records, dec: Decisions | None) -> Counter:
    """N5f: ô có (ocr_char, syllable_raw.lower()) khớp mục da_ky (book khớp hoặc null)
    -> tier GOLD, rule corpus_reading:<id>, label = ocr_char, dict_support 'corpus'.
    Ô khoá QĐ-01 không đụng. Mục theo sách ưu tiên hơn mục book: null."""
    n = Counter()
    if dec is None:
        return n
    idx: dict[tuple, dict] = {}
    for it in dec.ap_corpus_readings():
        idx[(it["ocr_char"], it["syllable_raw"], it["book"])] = it
    if not idx:
        return n
    for r in records:
        if r.get("qd01_locked") or not r.get("ocr_char"):
            continue
        s = nfc(r.get("syllable_raw", "")).lower()
        it = idx.get((r["ocr_char"], s, r["book"])) or idx.get((r["ocr_char"], s, None))
        if it is None:
            continue
        if not r.get("tier_goc"):
            r["tier_goc"], r["rule_goc"] = r.get("tier", ""), r.get("rule", "")
        r["tier"], r["rule"] = "GOLD", f"{RULE_CORPUS}:{it['id']}"
        r["label"], r["syllable"] = it["ocr_char"], s
        r["dict_support"] = "corpus"
        n[it["id"]] += 1
    return n


def apply_di_the(records, dec: Decisions | None) -> Counter:
    """N5i: `label_canonical` = label cho MỌI ô; ô có label ∈ quan_sat của mục da_ky
    (book khớp hoặc null) và label ≠ chuan -> label_canonical = chuan. Không đổi `label`."""
    n = Counter()
    m: dict[tuple, tuple] = {}
    if dec is not None:
        for it in dec.ap_di_the():
            for c in it["quan_sat"]:
                if c != it["chuan"]:
                    m[(c, it["book"])] = (it["chuan"], it["id"])
    for r in records:
        lab = r.get("label", "") or ""
        r["label_canonical"] = lab
        if not lab or not m:
            continue
        hit = m.get((lab, r["book"])) or m.get((lab, None))
        if hit:
            r["label_canonical"] = hit[0]
            n[f"{hit[1]}|{r['book']}"] += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Nạp + kiểm config/decisions.yaml, in report")
    ap.add_argument("--config", default=str(DEFAULT_PATH))
    ap.add_argument("--report", default="", help="ghi decisions_report.json ra đường dẫn này")
    a = ap.parse_args()
    dec = load(a.config)
    rep = dec.report()
    txt = json.dumps(rep, ensure_ascii=False, indent=2)
    if a.report:
        Path(a.report).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"[decisions] {a.config}: corpus_readings {rep['corpus_readings']['da_ky']}/{rep['corpus_readings']['n']} da_ky | "
          f"di_the {rep['di_the']['da_ky']}/{rep['di_the']['n']} | lop_nham {rep['lop_nham']['da_ky']}/{rep['lop_nham']['n']} | "
          f"chờ ký {rep['n_cho_ky']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
