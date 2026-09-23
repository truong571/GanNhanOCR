#!/usr/bin/env python
"""qn_syllable_fix.py — PA1: hậu xử lý ÂM TIẾT quốc ngữ (0 API, không đụng ảnh) + đối chứng PA2 (phiên âm chuẩn).

Bối cảnh: sổ kế toán mất mát (docs/CHAN_DOAN_3_SACH_MOI_2026-09-23.md) cho thấy khâu QN là nút thắt thứ hai.
Ô REVIEW `not_plausible` = âm QN vỡ (không phải âm tiếng Việt hợp lệ); ô `no_context` = âm đọc được nhưng
không có chữ kim nào trong tập ứng viên của âm đó.

PA1 (tự lực, không dùng tham chiếu): sửa âm vỡ về âm hợp lệ theo 3 bước, mỗi bước ghi `fix_kind`
  (a) `charmap` — bảng ký tự nhiễu Tesseract HỌC TỪ DỮ LIỆU THẬT (cặp âm OCR ↔ âm tham chiếu ở SÁCH KHÁC,
      xem --learn-from) chứ không đoán tay;
  (b) `edit1`/`edit2` — khoảng cách chỉnh sửa ≤1 rồi ≤2 tới TẬP ÂM HỢP LỆ (khoá của dict/QuocNgu_SinoNom.csv,
      chuẩn hoá y hệt pipeline: NFC + normalize_tone_marks + lower + is_plausible_qn_syllable);
      CHỈ nhận khi ứng viên DUY NHẤT, nhiều ứng viên → `ambiguous`, giữ nguyên âm;
  (c) `lucbat` — chỉ CHẨN ĐOÁN: đếm ô nằm trong câu có số âm ≠ 6/8 (không đủ căn cứ gán nhãn, xem §GIỚI HẠN).

PA2 (đối chứng): thay hẳn âm OCR bằng âm của PHIÊN ÂM CHUẨN dị bản gần
  (`scripts/measure/verses_ref_fix.py` → `measure_out/<book>/qn_ref_fix/verses_b1.tsv`).

Đo (cell-level, KHÔNG chạy build):
  * `prepared/<book>/dataset_out/labels_gated.csv` → ô (page, column, syl_idx, syllable, ocr_char, rule, tier);
  * `prepared/<book>/transcriptions/page_*.json` → (page, column, syl_idx) ↦ (verse_no, vị trí trong câu);
  * âm tham chiếu (nếu có) CHỈ dùng để CHẤM PA1, không dùng để sửa (charmap học từ sách khác → không tự khẳng định).
  * "ô sẽ nhận nhãn" = sau khi sửa, chữ kim `ocr_char` ∈ tập đọc từ điển của âm mới (đúng luật GOLD của
    pipeline/align_engine/anchor_align.py) — đây là CHẶN TRÊN, build thật còn phải qua các cổng khác.

Chạy:
  .venv/bin/python scripts/measure/qn_syllable_fix.py --book LucVanTien1883
  .venv/bin/python scripts/measure/qn_syllable_fix.py --all
  .venv/bin/python scripts/measure/qn_syllable_fix.py --selftest
Chỉ ĐỌC data/, dict/, prepared/, measure_out/; ghi measure_out/<book>/qn_fix/. Không sửa pipeline/ core/.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from core.text.text_utils import (  # noqa: E402
    clean_line_text, fold_text, is_plausible_qn_syllable, normalize_tone_marks,
    simple_levenshtein, split_to_syllables, strip_tone,
)
from core.text.dictionary import load_qn_to_nom  # noqa: E402

DICT_PATH = REPO / "dict" / "QuocNgu_SinoNom.csv"
BOOKS = ["LucVanTien1883", "KimVanKieu1884", "Chrestomathie1872"]
# học bảng ký tự từ SÁCH KHÁC để phép chấm không tự khẳng định
LEARN_PARTNER = {
    "LucVanTien1883": "KimVanKieu1884",
    "KimVanKieu1884": "LucVanTien1883",
    "Chrestomathie1872": "KimVanKieu1884",
}
# token giữ chỗ do ingest chèn: KHÔNG có nội dung âm nào để sửa (xem ingest_lithograph_book.py)
PLACEHOLDERS = {"khongdoc", "khongkhop"}
MIN_CHARMAP_SUPPORT = 3   # một phép thay ký tự chỉ vào bảng khi quan sát được ≥ ngần này lần ở sách kia
MAX_FOREIGN = 3           # số ký tự lạ tối đa trong một âm mà bước (a) chịu thử (chặn bùng nổ tổ hợp)


# --------------------------------------------------------------------------- #
def canon(t: str) -> str:
    return normalize_tone_marks(unicodedata.normalize("NFC", str(t)).strip().lower())


def qn_tokens(text: str) -> list[str]:
    toks = [t for t in split_to_syllables(clean_line_text(text or "")) if any(c.isalpha() for c in t)]
    return [canon(t) for t in toks]


def load_valid(dict_path: Path = DICT_PATH):
    """Tập âm hợp lệ = khoá ÂM của từ điển, chuẩn hoá y hệt pipeline (load_qn_to_nom đã normalize_tone_marks)."""
    qn2nom = load_qn_to_nom(str(dict_path))
    valid = {k for k in qn2nom if is_plausible_qn_syllable(k)}
    by_len = defaultdict(list)
    for v in valid:
        by_len[len(v)].append(v)
    return qn2nom, valid, by_len


def is_broken(syl: str, valid: set[str]) -> bool:
    """Âm 'hỏng' = không phải khoá từ điển (⇒ không sinh được ứng viên chữ kim nào)."""
    c = canon(syl)
    return bool(c) and c not in valid


# --------------------------------------------------------------------------- #
# (a) bảng ký tự nhiễu — HỌC từ cặp (âm OCR, âm tham chiếu) của một sách khác
# --------------------------------------------------------------------------- #
def learn_charmap(pairs, valid: set[str], min_support: int = MIN_CHARMAP_SUPPORT, allow_delete: bool = True):
    """pairs = [(am_ocr, am_ref)]. Trả (charmap, freq): ký tự lạ ↦ [ký tự thay, theo tần suất giảm dần]."""
    freq: Counter = Counter()
    for a, b in pairs:
        a, b = canon(a), canon(b)
        if not a or not b or a == b or a in valid:
            continue           # chỉ học từ âm OCR HỎNG (a ngoài từ điển) mà tham chiếu đọc được
        if b not in valid:
            continue
        for op, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if op == "replace" and (i2 - i1) == (j2 - j1):
                for k in range(i2 - i1):
                    freq[(a[i1 + k], b[j1 + k])] += 1
            elif op == "delete" and (i2 - i1) == 1:
                freq[(a[i1], "")] += 1
            elif op == "replace" and (i2 - i1) == 1:
                freq[(a[i1], b[j1:j2])] += 1
    charmap: dict[str, list[str]] = defaultdict(list)
    for (src, dst), n in freq.most_common():
        if n >= min_support and src != dst and (allow_delete or dst != ""):
            charmap[src].append(dst)
    return dict(charmap), freq


def _charmap_candidates(c: str, charmap: dict, valid: set[str]) -> set[str]:
    """Sinh ứng viên bằng cách thay MỌI tổ hợp ký tự có trong bảng; giữ ứng viên nằm trong tập âm hợp lệ."""
    pos = [i for i, ch in enumerate(c) if ch in charmap]
    if not pos or len(pos) > MAX_FOREIGN:
        return set()
    out = {c}
    for i in pos:
        nxt = set()
        for s in out:
            nxt.add(s)
            for repl in charmap[c[i]]:
                # vị trí i chỉ còn đúng khi mọi phép thay trước đó dài 1 ký tự; dựng lại theo ký tự gốc
                nxt.add(s[:i] + repl + s[i + 1:] if len(s) == len(c) else s)
        out = nxt
    return {s for s in out if s != c and s in valid}


def _edit_candidates(c: str, by_len: dict, max_d: int) -> tuple[int, set[str]]:
    best, cands = 99, set()
    for L in range(max(1, len(c) - max_d), len(c) + max_d + 1):
        for v in by_len.get(L, ()):
            d = simple_levenshtein(c, v)
            if d < best:
                best, cands = d, {v}
            elif d == best:
                cands.add(v)
    return best, cands


def fix_syllable(syl: str, charmap: dict, valid: set[str], by_len: dict):
    """Trả (am_moi | None, fix_kind). fix_kind ∈ {ok, placeholder, charmap, edit1, edit2, ambiguous, none}."""
    c = canon(syl)
    if c in PLACEHOLDERS:
        return None, "placeholder"
    if not c or c in valid:
        return None, "ok"
    # (a) bảng ký tự nhiễu
    cands = _charmap_candidates(c, charmap, valid)
    if len(cands) == 1:
        return next(iter(cands)), "charmap"
    if len(cands) > 1:
        return None, "ambiguous"
    # (b) khoảng cách chỉnh sửa, chỉ nhận ứng viên DUY NHẤT
    best, cands = _edit_candidates(c, by_len, 2)
    if best <= 2 and len(cands) == 1:
        return next(iter(cands)), f"edit{best}"
    if best <= 2:
        return None, "ambiguous"
    return None, "none"


# --------------------------------------------------------------------------- #
# nạp ô + ánh xạ ô ↦ (câu, vị trí) + âm tham chiếu PA2
# --------------------------------------------------------------------------- #
def load_cells(book: str):
    import csv as _csv
    path = REPO / "prepared" / book / "dataset_out" / "labels_gated.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            rows.append(r)
    return rows


def load_verse_map(book: str):
    """(page, column, syl_idx) ↦ (verse_no, pos_trong_cau, n_syll_cau). Từ prepared/<book>/transcriptions/."""
    out = {}
    verse_nsyll = {}
    verse_src = {}
    tdir = REPO / "prepared" / book / "transcriptions"
    for p in sorted(tdir.glob("page_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        page = p.stem
        for col in d.get("columns", []):
            syls = col.get("syllables") or []
            vo, ve = col.get("verse_odd"), col.get("verse_even")
            len_odd = col.get("len_odd")
            if vo is None and ve is None:
                continue
            if len_odd is None:
                len_odd = len(syls) if ve is None else (vo or {}).get("n_syll", 0)
            n_odd = min(len_odd, len(syls))
            n_even = max(0, len(syls) - n_odd)
            for i in range(len(syls)):
                if i < n_odd and vo:
                    v, pos, nslice = vo.get("verse_no"), i, n_odd
                elif ve:
                    v, pos, nslice = ve.get("verse_no"), i - n_odd, n_even
                else:
                    continue
                if v is None:
                    continue
                # nslice = số âm mà CỘT thực sự cấp cho câu đó; phép chấm chỉ nhận khi bằng số âm tham chiếu
                out[(page, int(col["column"]), i)] = (int(v), pos, nslice)
            for vv in (vo, ve):
                if vv and vv.get("verse_no") is not None:
                    verse_nsyll[int(vv["verse_no"])] = vv.get("n_syll")
                    verse_src[int(vv["verse_no"])] = vv.get("qn_source", "ocr")
    return out, verse_nsyll, verse_src


def load_ref_verses(book: str, ref_dir: str | None = None):
    """verse_no ↦ [âm tham chiếu] cho những câu verses_ref_fix khớp được (qn_source != 'ocr')."""
    path = Path(ref_dir) / "verses_b1.tsv" if ref_dir else REPO / "measure_out" / book / "qn_ref_fix" / "verses_b1.tsv"
    if not path.exists():
        return {}
    import csv as _csv
    ref = {}
    with open(path, encoding="utf-8") as f:
        for r in _csv.DictReader(f, delimiter="\t"):
            if (r.get("qn_source") or "ocr") == "ocr":
                continue
            toks_ref = qn_tokens(r.get("line_text") or "")
            toks_ocr = qn_tokens(r.get("line_text_ocr") or "")
            if not toks_ref or len(toks_ref) != len(toks_ocr):
                continue   # chỉ nhận câu căn chỉnh 1-1 để so theo vị trí
            nom = [c for c in (r.get("ref_nom") or "") if c.strip()]
            if len(nom) != len(toks_ref):
                nom = []   # chữ Nôm tham chiếu không căn được theo âm → không dùng để chấm NHÃN
            ref[int(r["verse_no"])] = (toks_ocr, toks_ref, r.get("qn_source"), nom)
    return ref


def learning_pairs(book: str) -> list[tuple[str, str]]:
    """Cặp (âm OCR, âm tham chiếu) của MỘT sách — nguyên liệu học bảng ký tự."""
    out = []
    for _v, (toks_ocr, toks_ref, _src, _nom) in load_ref_verses(book).items():
        out.extend(zip(toks_ocr, toks_ref))
    return out


# --------------------------------------------------------------------------- #
def measure_book(book: str, charmap: dict, valid: set[str], by_len: dict, qn2nom: dict, outdir: Path, ref_dir=None):
    cells = load_cells(book)
    vmap, verse_nsyll, verse_src = load_verse_map(book)
    ref = load_ref_verses(book, ref_dir)

    res = {"book": book, "n_cells": len(cells)}
    tier_c = Counter(r["tier"] for r in cells)
    res["tier"] = dict(tier_c)

    # --- phân loại ô REVIEW hỏng-ở-khâu-QN ---
    target_rules = {"not_plausible", "no_context"}
    pool, fixed_rows = [], []
    for r in cells:
        if r["tier"] != "REVIEW" or r["rule"] not in target_rules:
            continue
        syl = r.get("syllable") or ""
        c = canon(syl)
        entry = {
            "page": r["page"], "column": int(r["column"]), "syl_idx": int(r["syl_idx"] or 0),
            "rule": r["rule"], "syllable": c, "ocr_char": (r.get("ocr_char") or "").strip(),
        }
        if c in PLACEHOLDERS:
            entry["kind"] = "placeholder"
        elif c in valid:
            entry["kind"] = "valid_but_wrong"     # âm hợp lệ, PA1 KHÔNG đụng tới được
        else:
            entry["kind"] = "broken"
        pool.append(entry)

    res["review_qn_pool"] = dict(Counter(e["kind"] for e in pool))
    res["review_qn_pool"]["total"] = len(pool)

    # --- PA1: sửa những ô 'broken' ---
    kinds = Counter()
    recovered, recov_by_kind = 0, Counter()
    for e in pool:
        if e["kind"] != "broken":
            kinds[e["kind"]] += 1
            continue
        new, fk = fix_syllable(e["syllable"], charmap, valid, by_len)
        kinds[fk] += 1
        e["fix"], e["fix_kind"] = new, fk
        if new:
            cand = qn2nom.get(new, [])
            e["kim_in_cands"] = bool(e["ocr_char"] and e["ocr_char"] in cand)
            e["n_cands"] = len(cand)
            if e["kim_in_cands"]:
                recovered += 1
                recov_by_kind[fk] += 1
        fixed_rows.append(e)
    res["pa1_fix_kind"] = dict(kinds)
    res["pa1_recovered_cells"] = recovered
    res["pa1_recovered_by_kind"] = dict(recov_by_kind)

    # --- chấm PA1 bằng âm tham chiếu (chỉ chấm, không dùng để sửa) ---
    score = Counter()
    for e in fixed_rows:
        key = (e["page"], e["column"], e["syl_idx"])
        if key not in vmap:
            continue
        v, pos, nslice = vmap[key]
        if v not in ref:
            continue
        toks_ocr, toks_ref, _, nom_ref = ref[v]
        if nslice != len(toks_ref) or pos >= len(toks_ref):
            score["skip_len_mismatch"] += 1      # cột cấp số âm khác câu tham chiếu → không so theo vị trí được
            continue
        gold = toks_ref[pos]
        score["n_scorable"] += 1
        if e.get("fix"):
            score["fixed_correct" if e["fix"] == gold else "fixed_wrong"] += 1
            if e["fix"] != gold:
                score[f"wrong_{e['fix_kind']}"] += 1
            else:
                score[f"right_{e['fix_kind']}"] += 1
            if e.get("kim_in_cands"):
                score["kim_in_cands_correct" if e["fix"] == gold else "kim_in_cands_wrong"] += 1
                if nom_ref and pos < len(nom_ref):
                    score["label_n"] += 1
                    score["label_eq_ref_nom" if e["ocr_char"] == nom_ref[pos] else "label_ne_ref_nom"] += 1
        else:
            score["not_fixed"] += 1
    res["pa1_scored_vs_ref"] = dict(score)

    # --- chấm PA1 rộng hơn: MỌI âm hỏng trên câu có tham chiếu (không chỉ ô REVIEW) ---
    wide = Counter()
    for v, (toks_ocr, toks_ref, _src, _nom) in ref.items():
        for a, b in zip(toks_ocr, toks_ref):
            if a in PLACEHOLDERS:
                wide["placeholder"] += 1
                continue
            if a in valid:
                wide["already_valid"] += 1
                continue
            wide["broken"] += 1
            new, fk = fix_syllable(a, charmap, valid, by_len)
            if new:
                wide["fixed_correct" if new == b else "fixed_wrong"] += 1
                wide[("right_" if new == b else "wrong_") + fk] += 1
            else:
                wide["not_fixed_" + fk] += 1
    res["pa1_scored_all_broken_syll"] = dict(wide)

    # --- PA2: thay âm bằng phiên âm chuẩn, đếm ô sẽ nhận nhãn ---
    pa2 = Counter()
    for e in pool:
        key = (e["page"], e["column"], e["syl_idx"])
        if key not in vmap:
            pa2["no_verse_map"] += 1
            continue
        v, pos, nslice = vmap[key]
        e["build_qn_source"] = verse_src.get(v, "ocr")
        if v not in ref:
            pa2["no_ref_line"] += 1
            continue
        toks_ref, nom_ref = ref[v][1], ref[v][3]
        if nslice != len(toks_ref) or pos >= len(toks_ref):
            pa2["skip_len_mismatch"] += 1
            continue
        if e["build_qn_source"] != "ocr":
            pa2["already_applied_in_build"] += 1   # PA2 đã áp cho câu này khi build (KVK) → không còn lợi ích mới
            continue
        gold = toks_ref[pos]
        pa2["covered"] += 1
        pa2["covered_" + e["kind"]] += 1
        if gold not in valid:
            pa2["ref_syl_not_in_dict"] += 1
            continue
        if e["ocr_char"] and e["ocr_char"] in qn2nom.get(gold, []):
            pa2["recovered"] += 1
            pa2["recovered_" + e["kind"]] += 1
            if nom_ref and pos < len(nom_ref):
                pa2["label_n"] += 1
                pa2["label_eq_ref_nom" if e["ocr_char"] == nom_ref[pos] else "label_ne_ref_nom"] += 1
    res["pa2_ref"] = dict(pa2)
    res["pa2_ref_lines"] = len(ref)
    res["build_qn_source"] = dict(Counter(verse_src.values()))

    # --- NỀN dị bản: ô GOLD hiện có trùng chữ Nôm tham chiếu bao nhiêu (1883 ≠ 1916 nên < 100 %) ---
    bg = Counter()
    for r in cells:
        if r["tier"] != "GOLD":
            continue
        key = (r["page"], int(r["column"]), int(r["syl_idx"] or 0))
        if key not in vmap:
            continue
        v, pos, nslice = vmap[key]
        if v not in ref:
            continue
        toks_ref, nom_ref = ref[v][1], ref[v][3]
        if not nom_ref or nslice != len(toks_ref) or pos >= len(nom_ref):
            continue
        bg["n"] += 1
        bg["eq" if (r.get("ocr_char") or "").strip() == nom_ref[pos] else "ne"] += 1
    res["gold_vs_ref_nom_background"] = dict(bg)

    # --- (c) chẩn đoán nhịp lục bát ---
    lb = Counter()
    for e in pool:
        key = (e["page"], e["column"], e["syl_idx"])
        n = verse_nsyll.get(vmap[key][0]) if key in vmap else None
        lb["in_verse_6_8" if n in (6, 8) else "in_verse_other"] += 1
    res["lucbat_diag"] = dict(lb)

    outdir.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(outdir / "fixes.csv", "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["page", "column", "syl_idx", "rule", "kind", "syllable",
                                           "ocr_char", "fix", "fix_kind", "kim_in_cands", "n_cands",
                                           "build_qn_source"])
        w.writeheader()
        for e in fixed_rows:
            w.writerow({k: e.get(k, "") for k in w.fieldnames})
    return res


# --------------------------------------------------------------------------- #
def selftest() -> int:
    qn2nom, valid, by_len = load_valid()
    ok = True

    def _ok(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok = ok and bool(cond)

    _ok(len(valid) > 5000, f"tập âm hợp lệ {len(valid)} khoá")
    _ok("ba" in valid and "khongdoc" not in valid, "'ba' hợp lệ, 'khongdoc' không")
    _ok(canon("Hoà") == "hòa", "chuẩn hoá thanh điệu kiểu cũ → mới")
    cm, _ = learn_charmap([("t7ên", "tiên")] * 5 + [("h#ng", "hưng")] * 5, valid, min_support=3)
    _ok("7" in cm and "i" in cm["7"], f"học được 7→i: {cm.get('7')}")
    new, fk = fix_syllable("t7ên", cm, valid, by_len)
    _ok(new == "tiên" and fk == "charmap", f"sửa t7ên → {new} ({fk})")
    new, fk = fix_syllable("khongdoc", cm, valid, by_len)
    _ok(new is None and fk == "placeholder", "token giữ chỗ không sửa")
    new, fk = fix_syllable("trời", cm, valid, by_len)
    _ok(new is None and fk == "ok", "âm đã hợp lệ thì không đụng")
    new, fk = fix_syllable("zzzzzz", cm, valid, by_len)
    _ok(new is None and fk in ("none", "ambiguous"), f"rác không ép sửa ({fk})")
    # ứng viên không duy nhất thì từ chối
    n_amb = sum(1 for s in ["ba", "bo", "bi"] if s in valid)
    new, fk = fix_syllable("b0", cm, valid, by_len)
    _ok(fk in ("ambiguous", "charmap", "edit1"), f"b0 → {new} ({fk}); {n_amb} âm b? hợp lệ")
    _ok(simple_levenshtein("tien", "tiên") == 1, "levenshtein dùng lại của core")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--learn-from", default=None, help="sách dùng để HỌC bảng ký tự (mặc định: sách khác)")
    ap.add_argument("--charmap-mode", default="nodel", choices=["base", "nodel"],
                    help="base = cho phép luật XOÁ ký tự (đo 71,2 %%); nodel = chỉ luật THAY (73,6 %%, mặc định)")
    ap.add_argument("--ref-dir", default=None, help="thư mục qn_ref_fix khác (quét ngưỡng PA2)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    qn2nom, valid, by_len = load_valid()
    books = BOOKS if a.all else [a.book]
    out_all = {"dict_valid_syllables": len(valid), "books": {}}
    for book in books:
        partner = a.learn_from or LEARN_PARTNER.get(book)
        pairs = learning_pairs(partner) if partner else []
        charmap, freq = learn_charmap(pairs, valid, allow_delete=(a.charmap_mode == "base"))
        outdir = Path(a.out) if a.out else (REPO / "measure_out" / book / "qn_fix")
        r = measure_book(book, charmap, valid, by_len, qn2nom, outdir, a.ref_dir)
        r["charmap_mode"] = a.charmap_mode
        r["charmap_learned_from"] = partner
        r["charmap_pairs"] = len(pairs)
        r["charmap_size"] = len(charmap)
        r["charmap_top"] = [f"{s}->{d}:{n}" for (s, d), n in freq.most_common(25) if n >= MIN_CHARMAP_SUPPORT]
        (outdir / "summary.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        out_all["books"][book] = r
        print(f"→ {outdir}/summary.json")
    (REPO / "measure_out" / "qn_fix_ALL.json").write_text(
        json.dumps(out_all, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
