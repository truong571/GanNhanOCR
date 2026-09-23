#!/usr/bin/env python3
"""qn_orthography.py — PHƯƠNG ÁN 3: chuẩn hoá chính tả Quốc ngữ cổ → tiếng Việt chuẩn (2026-09-23).

CÂU HỎI: văn bản in thế kỷ 19 dùng chính tả cũ / phương ngữ Nam Bộ (nhơn/nhân, sanh/sinh, chánh/chính,
đàng/đường, thiệt/thật…) và đặt dấu thanh kiểu cũ (hoà/hòa, thuý/thúy). Chuẩn hoá chúng về tiếng Việt
chuẩn có cứu được ô REVIEW (`not_plausible` / `no_context` / `R(âm)=∅`) mà KHÔNG hạ độ chính xác không?

Bộ đo này KHÔNG sửa pipeline, KHÔNG chạy build; chỉ đọc `Dict/`, `data/`, `measure_out/`, `prepared/`.

BỐN TẦNG CHUẨN HOÁ được đo riêng (mỗi tầng chỉ kích hoạt khi âm hiện tại KHÔNG có trong từ điển —
"only-if-OOV" — nên theo thiết kế không tầng nào gộp mất hai khoá từ điển đang phân biệt được):
  L0  hiện hành  : NFC + lower + normalize_tone_marks (chỉ oa/oe/uy mở) — ĐÃ có trong core/.
  L1  dấu thanh  : dời dấu thanh sang MỌI vị trí nguyên âm khác (phiá→phía, nghiã→nghĩa, ngòai→ngoài,
                   đừơng→đường); nhận khi đúng MỘT biến thể nằm trong từ điển.
  L2  chính tả cổ: bảng cặp (âm in → âm chuẩn) RÚT TỪ DỮ LIỆU (đối chiếu dòng QN OCR với phiên âm chuẩn
                   của dị bản ở các dòng đã khớp), CHỈ giữ cặp có âm nguồn ∉ từ điển.
  L2b MODERN-hoá : bảng cặp mà CẢ HAI âm ∈ từ điển (nhơn→nhân, sanh→sinh, đàng→đường…) — đo RIÊNG vì đây
                   chính là "hiện đại hoá" mà người dùng hỏi; đo cả phần nó PHÁ (ô GOLD đang đúng).
  L3  lỗi ký tự  : thay ký tự tesseract đọc nhầm (e↔c, ð→đ, ñ→n, ‹→∅, ø/ö…) rút từ cùng nguồn dữ liệu.

ĐẦU RA measure_out/qn_orthography/: summary.json, oov_<Book>.csv, pairs_<Book>.csv,
cells_recover_<Book>.csv, ihr_<Book>.csv, invariants.json.

Chạy:
    .venv/bin/python scripts/measure/qn_orthography.py --book all
    .venv/bin/python scripts/measure/qn_orthography.py --book LucVanTien1883
    .venv/bin/python scripts/measure/qn_orthography.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.text.text_utils import (  # noqa: E402
    clean_line_text, is_plausible_qn_syllable, normalize_tone_marks,
    split_to_syllables, strip_all, strip_tone,
)

OUT = REPO / "measure_out" / "qn_orthography"
DICT_CSV = REPO / "Dict" / "QuocNgu_SinoNom.csv"

# Sách thạch bản: sổ kế toán mất mát + dòng QN OCR + dòng tham chiếu đã khớp.
LITHO = {
    "LucVanTien1883": dict(
        cells="measure_out/loss_ledger/LucVanTien1883_cells.csv",
        verses="measure_out/LucVanTien1883/qn_ocr/verses.tsv",
        matches="measure_out/LucVanTien1883/qn_ref_fix/matches.csv"),
    "KimVanKieu1884": dict(
        cells="measure_out/loss_ledger/KimVanKieu1884_cells.csv",
        verses="measure_out/KimVanKieu1884/qn_ocr/verses.tsv",
        matches="measure_out/KimVanKieu1884/qn_ref_fix/matches.csv"),
    "Chrestomathie1872": dict(
        cells="measure_out/loss_ledger/Chrestomathie1872_cells.csv",
        verses=None, matches=None),          # KHÔNG có phiên âm tham chiếu
}
# Hai bộ IHR có NHÃN NGƯỜI — dùng để đo RỦI RO (chuẩn hoá có làm sai nhãn không).
IHR = {
    "LucVanTien1916": "measure_out/LucVanTien1916/ihr_endtoend/cells.csv",
    "TruyenKieu1872": "measure_out/TruyenKieu1872/ihr_endtoend/cells.csv",
}

TONE_MARKS = "̣̀́̃̉"
VOWELS = "aeiouy"
_GARBAGE = re.compile(r"^[\W\d_]+$")


# --------------------------------------------------------------------------- #
# Từ điển
# --------------------------------------------------------------------------- #
def load_dict() -> tuple[dict[str, set[str]], dict[str, set[str]], int]:
    """(R: âm chuẩn hoá L0 → tập chữ Nôm, readings: chữ → tập âm, số khoá thô)."""
    R: dict[str, set[str]] = defaultdict(set)
    raw = set()
    with open(DICT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            q = (row["QuocNgu"] or "").strip().lower()
            n = (row["SinoNom"] or "").strip()
            if not q or not n:
                continue
            raw.add(q)
            R[normalize_tone_marks(q)].add(n)
    readings: dict[str, set[str]] = defaultdict(set)
    for syl, chars in R.items():
        for ch in chars:
            readings[ch].add(syl)
    return dict(R), dict(readings), len(raw)


def norm0(s: str) -> str:
    """Tầng hiện hành: lower + NFC + chuẩn hoá vị trí dấu thanh oa/oe/uy."""
    return normalize_tone_marks(unicodedata.normalize("NFC", (s or "").strip().lower()))


# --------------------------------------------------------------------------- #
# L1 — dời dấu thanh sang mọi nguyên âm (tổng quát hoá normalize_tone_marks)
# --------------------------------------------------------------------------- #
def tone_variants(s: str) -> set[str]:
    """Mọi cách đặt lại DẤU THANH duy nhất của `s` lên một nguyên âm khác (đã chuẩn hoá L0)."""
    d = list(unicodedata.normalize("NFD", s))
    ti = [i for i, c in enumerate(d) if c in TONE_MARKS]
    if len(ti) != 1:
        return set()
    mark = d.pop(ti[0])
    out = set()
    for i, c in enumerate(d):
        if c.lower() in VOWELS:
            j = i + 1
            while j < len(d) and unicodedata.category(d[j]) == "Mn":
                j += 1
            out.add(norm0(unicodedata.normalize("NFC", "".join(d[:j] + [mark] + d[j:]))))
    out.discard(s)
    return out


# --------------------------------------------------------------------------- #
# L3 — lỗi ký tự tesseract (rút từ cặp đã khớp, xem mine_pairs)
# --------------------------------------------------------------------------- #
CHAR_FIX = {"e": "c", "ð": "đ", "ñ": "n", "‹": "", "›": "", "ø": "o", "ö": "ô",
            "ò": "ồ", "¿": "", "œ": "c", "ø": "o"}


def char_variants(s: str) -> set[str]:
    """Mọi biến thể sinh bởi MỘT phép thay ký tự nhầm (e→c, ð→đ, ñ→n, ‹→∅…)."""
    out = set()
    for i, ch in enumerate(s):
        rep = CHAR_FIX.get(ch)
        if rep is None:
            continue
        v = norm0(s[:i] + rep + s[i + 1:])
        if v and v != s:
            out.add(v)
    return out


# --------------------------------------------------------------------------- #
# L2c — bảng "hiện đại hoá" SOẠN TAY (không rút máy): chính tả cổ / phương ngữ Nam Bộ
# thế kỷ 19 → tiếng Việt chuẩn. Soạn tay để đo PHƯƠNG ÁN 3 ở dạng TỐT NHẤT của nó
# (bảng máy rút từ dòng khớp bị lẫn lỗi OCR + dị bản thật, xem `rui_ro_gop`).
# --------------------------------------------------------------------------- #
CURATED_MODERN = {
    "nhơn": "nhân", "sanh": "sinh", "chánh": "chính", "đàng": "đường", "thiệt": "thật",
    "nhứt": "nhất", "ngãi": "nghĩa", "phước": "phúc", "chơn": "chân", "đờn": "đàn",
    "bịnh": "bệnh", "kiểng": "cảnh", "nầy": "này", "mầy": "mày", "thơ": "thư",
    "thời": "thì", "huê": "hoa", "hường": "hồng", "nhựt": "nhật", "tợ": "tựa",
    "dưng": "dâng", "xuơng": "xương", "vưng": "vâng", "dòm": "nhòm", "giêng": "riêng",
    "trển": "trên", "bển": "bên", "hườn": "hoàn", "huình": "hoàng", "nguơn": "nguyên",
    "sưnh": "sinh", "chưn": "chân", "lãnh": "lĩnh", "tánh": "tính", "mạng": "mệnh",
    "hiệp": "hợp", "huỳnh": "hoàng", "ngoàn": "ngoan", "bông": "hoa", "đặng": "được",
}

# --------------------------------------------------------------------------- #
# Khai thác cặp (âm in → âm chuẩn) từ dòng đã khớp dị bản
# --------------------------------------------------------------------------- #
def line_syllables(text: str) -> list[str]:
    return [norm0(t) for t in split_to_syllables(clean_line_text(text or ""))
            if any(c.isalpha() for c in t)]


def mine_pairs(matches_csv: Path, max_diff: int = 2) -> tuple[Counter, int]:
    """Cặp (âm OCR bản in, âm phiên âm chuẩn) từ dòng khớp dị bản lệch ≤ `max_diff` âm.

    Chỉ lấy dòng CÙNG SỐ ÂM và lệch ít âm → cặp được cô lập, không phải nhiễu căn chỉnh.
    """
    pairs: Counter = Counter()
    nlines = 0
    if not matches_csv or not matches_csv.exists():
        return pairs, 0
    with open(matches_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("qn_source") or "ocr") == "ocr":
                continue
            a, b = line_syllables(r["line_text_ocr"]), line_syllables(r["line_text_ref"])
            if not a or len(a) != len(b):
                continue
            diff = [i for i in range(len(a)) if a[i] != b[i]]
            if len(diff) > max_diff:
                continue
            nlines += 1
            for i in diff:
                pairs[(a[i], b[i])] += 1
    return pairs, nlines


# --------------------------------------------------------------------------- #
# Bộ chuẩn hoá theo tầng
# --------------------------------------------------------------------------- #
class Normalizer:
    """Chuẩn hoá "only-if-OOV": chỉ đụng vào âm KHÔNG có trong từ điển (trừ tầng L2b).

    `layers` ⊆ {"L1","L2","L2b","L3"}. Trả (âm mới, tên tầng) hoặc (âm cũ, "").
    """

    def __init__(self, R, layers, spell_map=None, modern_map=None):
        self.R = R
        self.layers = set(layers)
        self.spell = spell_map or {}
        self.modern = modern_map or {}

    def __call__(self, s: str) -> tuple[str, str]:
        s = norm0(s)
        if "L2b" in self.layers and s in self.modern:      # hiện đại hoá: áp cả khi âm CÓ trong dict
            return self.modern[s], "L2b"
        if s in self.R or not s:
            return s, ""
        if "L1" in self.layers:
            c = sorted(v for v in tone_variants(s) if v in self.R)
            if len(c) == 1:
                return c[0], "L1"
        if "L3" in self.layers:
            c = sorted(v for v in char_variants(s) if v in self.R)
            if len(c) == 1:
                return c[0], "L3"
        if "L2" in self.layers and s in self.spell:
            return self.spell[s], "L2"
        return s, ""


# --------------------------------------------------------------------------- #
# ĐO 1 — âm ngoài từ điển (OOV)
# --------------------------------------------------------------------------- #
def step_oov(book, cfg, R, norms) -> dict:
    """Đếm âm OOV trong verses.tsv (nếu có) và trong cells.csv, trước/sau từng bộ chuẩn hoá."""
    res = {}
    for src, path, col, delim in (("verses", cfg.get("verses"), "line_text", "\t"),
                                  ("cells", cfg.get("cells"), "syllable", ",")):
        if not path:
            continue
        p = REPO / path
        if not p.exists():
            continue
        toks = []
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=delim):
                t = row.get(col) or ""
                toks += line_syllables(t) if src == "verses" else [norm0(t)]
        toks = [t for t in toks if t]
        plaus = [t for t in toks if is_plausible_qn_syllable(t)]
        base = [t for t in plaus if t not in R]
        row = {"n_token": len(toks), "n_implausible": len(toks) - len(plaus),
               "n_plausible": len(plaus), "oov_L0": len(base), "oov_L0_types": len(set(base))}
        for name, nz in norms.items():
            after = [t for t in base if nz(t)[0] in R]
            row[f"cuu_{name}"] = len(after)
        res[src] = row
    return res


# --------------------------------------------------------------------------- #
# ĐO 2 — ô cứu được (sách thạch bản) + ô GOLD bị phá
# --------------------------------------------------------------------------- #
def step_cells(book, cfg, R, readings, norms, modern_map, out: Path) -> dict:
    p = REPO / cfg["cells"]
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    res = {"n_cells": len(rows)}
    qn_causes = {"A_qn_am_la", "A_qn_khong_hop_le", "A_qn_sai_dau"}
    res["n_gold"] = sum(1 for r in rows if r["tier"] == "GOLD")
    res["n_cause_qn"] = sum(1 for r in rows if r["cause"] in qn_causes)
    rec_rows = []
    for name, nz in norms.items():
        cuu = amb = 0
        for r in rows:
            if r["cause"] not in qn_causes:
                continue
            s, kim = norm0(r["syllable"]), (r["ocr_char"] or "").strip()
            if not kim:
                continue
            s2, lay = nz(s)
            if not lay or s2 == s:
                continue
            if kim in R.get(s2, ()):          # điều kiện CẦN để thành GOLD
                cuu += 1
                rec_rows.append(dict(book=book, bo=name, page=r["page"], column=r["column"],
                                     nom_idx=r["nom_idx"], cause=r["cause"], tang=lay,
                                     syllable=s, syllable_norm=s2, kim=kim,
                                     n_ung_vien=len(R.get(s2, ()))))
            else:
                amb += 1
        res[f"cuu_{name}"] = cuu
        res[f"doi_am_khong_cuu_{name}"] = amb
    # L2b PHÁ: ô GOLD đang đúng luật mà hiện đại hoá làm kim rơi khỏi R(âm mới)
    pha = pha_giu = 0
    for r in rows:
        if r["tier"] not in ("GOLD", "GOLD_text_only"):
            continue
        s, kim = norm0(r["syllable"]), (r["ocr_char"] or "").strip()
        if s not in modern_map or not kim:
            continue
        if kim in R.get(modern_map[s], ()):
            pha_giu += 1
        else:
            pha += 1
    res["L2b_gold_bi_pha"] = pha
    res["L2b_gold_con_giu"] = pha_giu
    if rec_rows:
        with open(out / f"cells_recover_{book}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rec_rows[0]))
            w.writeheader()
            w.writerows(rec_rows)
    return res


# --------------------------------------------------------------------------- #
# ĐO 3 — RỦI RO đo bằng NHÃN NGƯỜI (2 bộ IHR)
# --------------------------------------------------------------------------- #
def step_ihr(book, path, R, norms, modern_map, out: Path) -> dict:
    p = REPO / path
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    res = {"n_cells": len(rows)}
    det = []
    for name, nz in norms.items():
        cuu = dung = sai = khong_gt = 0
        for r in rows:
            if r["tier"] in ("GOLD", "GOLD_text_only"):
                continue
            s, kim, gt = norm0(r["syllable"]), (r["ocr_char"] or "").strip(), (r["gt"] or "").strip()
            if not kim:
                continue
            s2, lay = nz(s)
            if not lay or s2 == s or kim not in R.get(s2, ()):
                continue
            cuu += 1
            if not gt:
                khong_gt += 1
            elif gt == kim:
                dung += 1
            else:
                sai += 1
            det.append(dict(book=book, bo=name, page=r["page"], column=r["column"],
                            syl_idx=r["syl_idx"], tier=r["tier"], rule=r["rule"],
                            syllable=s, syllable_norm=s2, tang=lay, kim=kim, gt=gt,
                            ket_qua="dung" if gt == kim else ("khong_gt" if not gt else "sai")))
        n = dung + sai
        res[f"{name}_cuu"] = cuu
        res[f"{name}_dung"] = dung
        res[f"{name}_sai"] = sai
        res[f"{name}_khong_gt"] = khong_gt
        res[f"{name}_precision"] = round(dung / n, 4) if n else None
    # L2b phá ô GOLD ĐANG ĐÚNG theo nhãn người
    pha_dung = pha_sai = 0
    for r in rows:
        if r["tier"] not in ("GOLD", "GOLD_text_only"):
            continue
        s, kim, gt = norm0(r["syllable"]), (r["ocr_char"] or "").strip(), (r["gt"] or "").strip()
        if s not in modern_map or not kim or not gt:
            continue
        if kim not in R.get(modern_map[s], ()):
            if gt == kim:
                pha_dung += 1      # ô ĐÚNG bị hiện đại hoá đánh rớt
            else:
                pha_sai += 1
    res["L2b_gold_bi_pha_dang_dung"] = pha_dung
    res["L2b_gold_bi_pha_dang_sai"] = pha_sai
    if det:
        with open(out / f"ihr_{book}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(det[0]))
            w.writeheader()
            w.writerows(det)
    return res


# --------------------------------------------------------------------------- #
# RỦI RO gộp mất phân biệt
# --------------------------------------------------------------------------- #
def step_collision(R, modern_map) -> dict:
    """Bao nhiêu khoá từ điển biến mất và bao nhiêu chữ Nôm rơi khỏi tập ứng viên nếu áp L2b."""
    mat = 0
    rows = []
    for a, b in modern_map.items():
        ra, rb = R.get(a, set()), R.get(b, set())
        if not ra:
            continue
        mat += 1
        rows.append((a, b, len(ra), len(rb), len(ra & rb), len(ra - rb)))
    return {
        "n_cap_ca_hai_trong_dict": mat,
        "n_khoa_dict_bien_mat": len({a for a, b in modern_map.items() if a in R and a != b}),
        "n_chu_rot_khoi_R": sum(r[5] for r in rows),
        "trung_binh_giao_nhau": round(sum(r[4] for r in rows) / max(mat, 1), 2),
        "vi_du": [dict(cu=r[0], moi=r[1], n_R_cu=r[2], n_R_moi=r[3], giao=r[4], mat=r[5])
                  for r in sorted(rows, key=lambda x: -x[5])[:10]],
    }


# --------------------------------------------------------------------------- #
def build_maps(R) -> tuple[dict, dict, dict, list]:
    """(spell_map L2: nguồn ∉ dict, modern_map L2b: cả hai ∈ dict, thống kê, top-cặp)."""
    allp: Counter = Counter()
    stat = {}
    for book, cfg in LITHO.items():
        m = cfg.get("matches")
        pr, nl = mine_pairs(REPO / m) if m else (Counter(), 0)
        stat[book] = {"n_dong_khop": nl, "n_cap": sum(pr.values()), "n_loai_cap": len(pr)}
        allp.update(pr)
    spell, modern = {}, {}
    for (a, b), n in allp.items():
        if a == b or b not in R:
            continue
        if a in R:
            modern.setdefault(a, Counter())[b] = n
        else:
            spell.setdefault(a, Counter())[b] = n
    # mỗi âm nguồn -> đích hay gặp nhất, và phải áp đảo (≥2× đích nhì) để không đoán bừa
    def _pick(d):
        out = {}
        for a, c in d.items():
            top = c.most_common(2)
            if len(top) == 1 or top[0][1] >= 2 * top[1][1]:
                out[a] = top[0][0]
        return out
    spell_m, modern_m = _pick(spell), _pick(modern)
    top = [dict(cu=a, moi=b, n=n, ca_hai_trong_dict=(a in R))
           for (a, b), n in allp.most_common() if b in R and a != b][:30]
    return spell_m, modern_m, stat, top


def step_curated(R, out: Path) -> dict:
    """PA3 ở dạng TỐT NHẤT: bảng hiện đại hoá SOẠN TAY — đo ô cứu được vs ô GOLD bị phá.

    Trên 2 bộ IHR còn đối chiếu NHÃN NGƯỜI: ô GOLD bị phá có thật sự ĐANG ĐÚNG không.
    """
    res = {"n_cap": len(CURATED_MODERN),
           "nguon_trong_dict": sum(1 for a in CURATED_MODERN if a in R),
           "nguon_ngoai_dict": sorted(a for a in CURATED_MODERN if a not in R),
           "n_chu_rot_khoi_R": sum(len(R.get(a, set()) - R.get(b, set()))
                                   for a, b in CURATED_MODERN.items() if a in R),
           "litho": {}, "ihr": {}}
    for b, cfg in LITHO.items():
        pha = cuu = giu = 0
        for r in csv.DictReader(open(REPO / cfg["cells"], encoding="utf-8")):
            s_, kim = norm0(r["syllable"]), (r["ocr_char"] or "").strip()
            if s_ not in CURATED_MODERN or not kim:
                continue
            new = CURATED_MODERN[s_]
            if r["tier"] in ("GOLD", "GOLD_text_only"):
                giu += int(kim in R.get(new, ()))
                pha += int(kim not in R.get(new, ()))
            elif kim not in R.get(s_, ()) and kim in R.get(new, ()):
                cuu += 1
        res["litho"][b] = {"gold_bi_pha": pha, "gold_con_giu": giu, "o_cuu_duoc": cuu}
    for b, p in IHR.items():
        if not (REPO / p).exists():
            continue
        pd_ = ps = cd = cs = 0
        for r in csv.DictReader(open(REPO / p, encoding="utf-8")):
            s_, kim, gt = norm0(r["syllable"]), (r["ocr_char"] or "").strip(), (r["gt"] or "").strip()
            if s_ not in CURATED_MODERN or not kim or not gt:
                continue
            new = CURATED_MODERN[s_]
            if r["tier"] in ("GOLD", "GOLD_text_only"):
                if kim not in R.get(new, ()):
                    pd_ += int(gt == kim)
                    ps += int(gt != kim)
            elif kim not in R.get(s_, ()) and kim in R.get(new, ()):
                cd += int(gt == kim)
                cs += int(gt != kim)
        res["ihr"][b] = {"gold_dung_bi_pha": pd_, "gold_sai_bi_pha": ps,
                         "o_cuu_dung": cd, "o_cuu_sai": cs}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default="all")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    R, readings, n_raw = load_dict()
    spell, modern, mine_stat, top = build_maps(R)

    norms = {
        "L1": Normalizer(R, {"L1"}, spell, modern),
        "L1+L3": Normalizer(R, {"L1", "L3"}, spell, modern),
        "L1+L2+L3": Normalizer(R, {"L1", "L2", "L3"}, spell, modern),
        "L2b_modern": Normalizer(R, {"L2b"}, spell, modern),
    }
    S = {
        "generated_by": "scripts/measure/qn_orthography.py",
        "dict": {"file": str(DICT_CSV.relative_to(REPO)), "n_khoa_tho": n_raw,
                 "n_khoa_sau_L0": len(R),
                 "n_khoa_doi_boi_L0": n_raw - len(R),
                 "n_chu_nom": len(readings)},
        "khai_thac_cap": mine_stat,
        "bang_L2_nguon_ngoai_dict": {"n": len(spell), "vi_du": dict(list(sorted(spell.items()))[:20])},
        "bang_L2b_ca_hai_trong_dict": {"n": len(modern), "vi_du": dict(list(sorted(modern.items()))[:20])},
        "top30_cap": top,
        "rui_ro_gop": step_collision(R, modern),
        "PA3_bang_soan_tay": step_curated(R, OUT),
        "sach": {}, "ihr": {},
    }
    books = list(LITHO) if a.book == "all" else [a.book]
    for b in books:
        if b in LITHO:
            S["sach"][b] = {"oov": step_oov(b, LITHO[b], R, norms),
                            "o": step_cells(b, LITHO[b], R, readings, norms, modern, OUT)}
    if a.book == "all":
        for b, p in IHR.items():
            if (REPO / p).exists():
                S["ihr"][b] = step_ihr(b, p, R, norms, modern, OUT)

    inv = invariants(S, R, norms)
    S["invariants"] = inv
    (OUT / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"invariants": f"{sum(1 for i in inv if i['ok'])}/{len(inv)} PASS",
                      "fail": [i["ten"] for i in inv if not i["ok"]]}, ensure_ascii=False))
    print("→", OUT / "summary.json")
    return 0 if all(i["ok"] for i in inv) else 1


def invariants(S, R, norms) -> list[dict]:
    out = []

    def ck(ten, ok, ghi=""):
        out.append({"ten": ten, "ok": bool(ok), "ghi": ghi})

    ck("L0_idempotent", all(norm0(norm0(k)) == norm0(k) for k in list(R)[:2000]))
    ck("khoa_dict_da_chuan_L0", all(k == norm0(k) for k in R))
    nz = norms["L1+L2+L3"]
    ck("chuan_hoa_khong_dung_am_trong_dict",
       all(nz(k)[1] == "" for k in list(R)[:3000]), "âm ĐÃ có trong từ điển không bị đổi")
    for b, d in S["sach"].items():
        for src, o in d["oov"].items():
            ck(f"{b}_{src}_oov_le_plausible", o["oov_L0"] <= o["n_plausible"])
            for k in o:
                if k.startswith("cuu_"):
                    ck(f"{b}_{src}_{k}_le_oov", o[k] <= o["oov_L0"])
        o = d["o"]
        for k in o:
            if k.startswith("cuu_"):
                ck(f"{b}_o_{k}_le_nguyen_nhan_qn", o[k] <= o["n_cause_qn"])
    c = S.get("PA3_bang_soan_tay")
    if c:
        ck("PA3_bang_soan_tay_dich_deu_trong_dict",
           all(v in R for v in CURATED_MODERN.values()))
        ck("PA3_bang_soan_tay_khong_tu_anh_xa",
           all(k != v for k, v in CURATED_MODERN.items()))
    for b, d in S["ihr"].items():
        for name in ("L1", "L1+L3", "L1+L2+L3"):
            ck(f"{b}_{name}_tong_khop", d[f"{name}_dung"] + d[f"{name}_sai"] + d[f"{name}_khong_gt"]
               == d[f"{name}_cuu"])
    return out


def selftest() -> int:
    ok = True

    def _(m, c):
        nonlocal ok
        ok &= bool(c)
        print(("PASS " if c else "FAIL ") + m)

    _("norm0 hoà→hòa", norm0("Hoà") == "hòa")
    _("norm0 thuý→thúy", norm0("thuý") == "thúy")
    _("tone_variants phiá ∋ phía", "phía" in tone_variants("phiá"))
    _("tone_variants ngòai ∋ ngoài", "ngoài" in tone_variants("ngòai"))
    _("tone_variants không đổi âm không dấu", tone_variants("cho") == set())
    _("char_variants eho→cho", "cho" in char_variants("eho"))
    _("char_variants ðem→đem", "đem" in char_variants("ðem"))
    R = {"phía": {"某"}, "cho": {"朱"}, "nhân": {"人"}, "nhơn": {"仁"}}
    nz = Normalizer(R, {"L1", "L2", "L3"}, {"vưng": "vâng"}, {"nhơn": "nhân"})
    _("L1 kích hoạt trên âm OOV", nz("phiá") == ("phía", "L1"))
    _("L3 kích hoạt trên âm OOV", nz("eho") == ("cho", "L3"))
    _("L2 kích hoạt trên âm OOV", nz("vưng") == ("vâng", "L2"))
    _("KHÔNG đụng âm đã có trong dict", nz("nhơn") == ("nhơn", ""))
    nzb = Normalizer(R, {"L2b"}, {}, {"nhơn": "nhân"})
    _("L2b hiện đại hoá cả âm có trong dict", nzb("nhơn") == ("nhân", "L2b"))
    _("mine_pairs chấp nhận tệp vắng", mine_pairs(Path("/khong/co.csv")) == (Counter(), 0))
    _("line_syllables bỏ token không chữ cái", line_syllables("Trăm năm 1⁄25,") == ["trăm", "năm"])
    col = step_collision({"nhơn": {"仁", "人"}, "nhân": {"人"}}, {"nhơn": "nhân"})
    _("step_collision đếm chữ rơi khỏi R", col["n_chu_rot_khoi_R"] == 1)
    print(("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
