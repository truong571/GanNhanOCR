"""CHUẨN NHIỄU LOẠN CÓ ĐÁP ÁN — đo ĐỘ BỀN của căn chỉnh mà không cần một nhãn người nào.

Ý TƯỞNG
-------
Không có ground truth thì không đo được precision. Nhưng ĐỘ BỀN thì đo được: lấy dữ liệu
thật, tiêm vào một hỏng hóc mà ta BIẾT TRƯỚC, rồi xem thuật toán có giữ nguyên những cặp
ghép chắc chắn đúng hay không.

Ô NEO (anchor) = ô có `rule = s1_inter_s2_direct`: chữ OCR nằm trong danh sách đọc âm của
âm tiết, chi phí ghép 0,0. Đây là những cặp gần chắc chắn đúng. Đo được trên corpus: trung
vị **58%** ô mỗi cột là ô neo (p10 42%, p90 73%) — quá đủ tín hiệu.

    ⚠️ KHÔNG dùng "cột sạch tuyệt đối" làm chuẩn: toàn corpus chỉ có 6 cột đạt 100%
    dict-confirmed (54 ô), quá ít. Chuẩn phải neo vào Ô NEO, không phải cột sạch.

CHỈ SỐ: `anchor_retention` = trong số ô neo CÒN SỐNG sau khi tiêm hỏng (âm/chữ của nó
không bị xoá), bao nhiêu phần trăm vẫn được ghép đúng cặp cũ.

    ⚠️ ĐỪNG tối ưu theo "số ô GOLD" hay tỉ lệ dict-confirm — đó chính là luật gán nhãn,
    cấu hình nới lỏng nhất sẽ luôn thắng trong khi sinh nhiều nhãn sai nhất. Dùng biên
    Pareto giữa `anchor_retention` và sản lượng.

BẢY LOẠI HỎNG — mô phỏng lỗi THẬT đã quan sát được trong corpus:

    drop_syl     VietOCR nuốt âm            ins_syl      VietOCR sinh âm thừa
    swap_syl     lỗi thứ tự                 tone_syl     lỗi thanh điệu (nhóm A/B)
    subst_char   S1 đọc nhầm tự dạng        drop_char    detector sót hộp
    split_char   detector cắt đôi hộp
"""
from __future__ import annotations

import collections
import csv
import hashlib
import random
import unicodedata as ud
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANCHOR_RULE = "s1_inter_s2_direct"

PERTURBATIONS = ("drop_syl", "ins_syl", "swap_syl", "tone_syl",
                 "subst_char", "drop_char", "split_char")
LEVELS = (1, 2, 3)


# --------------------------------------------------------------------------- #
def load_columns(labels_csv: Path | str, min_cells: int = 6) -> list[dict]:
    """Dựng lại (chữ Nôm, âm QN, ô neo) cho từng cột từ bộ nhãn.

    Hàng trong labels.csv là ĐẦU RA của align, nên chuỗi dựng lại đã là một phép ghép
    1-1 theo đường chéo. Đó chính là điều ta muốn: mốc xuất phát KHÔNG hỏng, rồi mới
    tiêm hỏng vào và đo mức tụt.
    """
    with open(labels_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_col: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if r.get("column") and r.get("ocr_char") and r.get("syllable"):
            by_col[(r["book"], r["page"], r["column"])].append(r)
    cols = []
    for key, rs in sorted(by_col.items()):
        chars = [r["ocr_char"] for r in rs]
        syls = [str(r["syllable"]).lower() for r in rs]
        anchors = [i for i, r in enumerate(rs)
                   if str(r.get("rule", "")).split("|")[0] == ANCHOR_RULE]
        if len(rs) >= min_cells and anchors:
            cols.append({"key": key, "chars": chars, "syllables": syls,
                         "anchors": anchors})
    return cols


# --------------------------------------------------------------------------- #
def _seed(*parts) -> int:
    """Seed TẤT ĐỊNH giữa các tiến trình.

    KHÔNG dùng `hash()`/`tuple.__hash__()`: hash của chuỗi trong Python được NGẪU NHIÊN
    HOÁ theo tiến trình (PYTHONHASHSEED), nên hai lần chạy cho hai kết quả khác nhau —
    đúng lỗi đã bị phép kiểm tất định của bàn thí nghiệm bắt được ngày 2026-08-23.
    """
    blob = "|".join(str(x) for x in parts).encode()
    return int.from_bytes(hashlib.md5(blob).digest()[:8], "big")


_TONE = "̣̀́̃̉"          # huyền sắc ngã hỏi nặng


def _retone(s: str, rng: random.Random) -> str:
    """Đổi dấu thanh của âm — mô phỏng đúng lớp lỗi VietOCR nhóm A/B."""
    d = ud.normalize("NFD", s)
    base = "".join(c for c in d if c not in _TONE)
    keep = [c for c in _TONE if c in d]
    new = rng.choice([t for t in _TONE if t not in keep] + [""])
    if not new:
        return ud.normalize("NFC", base)
    # gắn dấu vào nguyên âm đầu tiên
    for i, c in enumerate(base):
        if c.lower() in "aeiouyăâêôơư":
            return ud.normalize("NFC", base[:i + 1] + new + base[i + 1:])
    return ud.normalize("NFC", base)


def apply_perturbation(chars: list[str], syls: list[str], anchors: list[int],
                       kind: str, level: int, rng: random.Random,
                       similar: dict | None = None):
    """Trả (chars', syls', map_anchor) — map_anchor[i_gốc] = (i_char', j_syl') hoặc None.

    None = ô neo đó KHÔNG còn sống (chữ hoặc âm của nó đã bị xoá) -> loại khỏi mẫu số.
    """
    c, s = list(chars), list(syls)
    ci = list(range(len(c)))          # ci[k] = chỉ số GỐC của chữ đang ở vị trí k
    si = list(range(len(s)))
    for _ in range(level):
        if kind == "drop_syl" and len(s) > 2:
            j = rng.randrange(len(s)); s.pop(j); si.pop(j)
        elif kind == "ins_syl":
            j = rng.randrange(len(s) + 1)
            s.insert(j, rng.choice(s) if s else "xxx"); si.insert(j, -1)
        elif kind == "swap_syl" and len(s) > 2:
            j = rng.randrange(len(s) - 1)
            s[j], s[j + 1] = s[j + 1], s[j]; si[j], si[j + 1] = si[j + 1], si[j]
        elif kind == "tone_syl" and s:
            j = rng.randrange(len(s)); s[j] = _retone(s[j], rng)
        elif kind == "subst_char" and c:
            k = rng.randrange(len(c))
            alt = (similar or {}).get(c[k]) or []
            c[k] = rng.choice(alt[:5]) if alt else "口"
        elif kind == "drop_char" and len(c) > 2:
            k = rng.randrange(len(c)); c.pop(k); ci.pop(k)
        elif kind == "split_char" and c:
            k = rng.randrange(len(c))
            c.insert(k + 1, c[k]); ci.insert(k + 1, -1)   # hộp bị cắt đôi -> 2 hộp
    pos_c = {orig: k for k, orig in enumerate(ci) if orig >= 0}
    pos_s = {orig: k for k, orig in enumerate(si) if orig >= 0}
    amap = {a: ((pos_c[a], pos_s[a]) if a in pos_c and a in pos_s else None)
            for a in anchors}
    return c, s, amap


# --------------------------------------------------------------------------- #
def anchor_retention(columns: list[dict], qn_to_nom: dict, similar: dict,
                     kind: str, level: int, seed: int = 0,
                     band_slack: int | None = None, cost: dict | None = None) -> dict:
    """Tỉ lệ ô neo còn sống vẫn giữ đúng cặp ghép cũ, sau khi tiêm hỏng `kind` mức `level`.

    `cost` cho phép quét ma trận chi phí mà không phải sửa mã (T3.2): các khoá hợp lệ là
    COST_CONFIRM / COST_SIMILAR / COST_DICTMISS / COST_NODICT / COST_DEL / COST_INS.
    """
    import sys
    sys.path.insert(0, str(REPO))
    from pipeline.align_engine import anchor_align as aa

    saved = {k: getattr(aa, k) for k in
             ("COST_CONFIRM", "COST_SIMILAR", "COST_DICTMISS",
              "COST_NODICT", "COST_DEL", "COST_INS", "BAND_SLACK")}
    try:
        for k, v in (cost or {}).items():
            setattr(aa, k, v)
        if band_slack is not None:
            aa.BAND_SLACK = band_slack
        slack = aa.BAND_SLACK

        kept = alive = 0
        for ci, col in enumerate(columns):
            rng = random.Random(_seed(seed, ci, kind, level))
            c, s, amap = apply_perturbation(
                col["chars"], col["syllables"], col["anchors"], kind, level, rng, similar)
            live = {a: p for a, p in amap.items() if p is not None}
            if not live:
                continue
            ops = aa.realign_column([{"ocr_char": x} for x in c], s,
                                    qn_to_nom, similar, band_slack=slack)
            got = {(o["nom_idx"], o["syl_idx"]) for o in ops if o["op"] == "match"}
            alive += len(live)
            kept += sum(1 for p in live.values() if p in got)
        return {"kind": kind, "level": level, "anchors_alive": alive,
                "anchors_kept": kept,
                "anchor_retention": round(kept / alive, 4) if alive else None}
    finally:
        for k, v in saved.items():
            setattr(aa, k, v)


def run_suite(columns: list[dict], qn_to_nom: dict, similar: dict,
              seeds: int = 3, band_slack: int | None = None,
              cost: dict | None = None) -> dict:
    """Chạy cả 7 loại × 3 mức × `seeds` lần. Trả một dòng số phẳng."""
    out: dict = {}
    tot_k = tot_a = 0
    for kind in PERTURBATIONS:
        kk = aa_ = 0
        for level in LEVELS:
            for sd in range(seeds):
                r = anchor_retention(columns, qn_to_nom, similar, kind, level,
                                     seed=sd, band_slack=band_slack, cost=cost)
                kk += r["anchors_kept"]; aa_ += r["anchors_alive"]
        out[f"ret_{kind}"] = round(kk / aa_, 4) if aa_ else None
        tot_k += kk; tot_a += aa_
    out["anchor_retention"] = round(tot_k / tot_a, 4) if tot_a else None
    out["anchors_tested"] = tot_a
    return out
