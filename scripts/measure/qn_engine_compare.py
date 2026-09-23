#!/usr/bin/env python
"""qn_engine_compare.py — so kênh QUỐC NGỮ tesseract(psm 4) ↔ VietOCR trên sách in thế kỷ 19 (2026-09-23).

CHỈ ĐỌC, 0 token, 0 lượt gọi API. Đầu vào = hai lần chạy `qn_print_ocr.py` cùng --start/--limit, khác --engine,
ghi ra thư mục riêng (không đụng measure_out/<book>/qn_ocr/ của bản chốt):

  .venv/bin/python scripts/measure/qn_print_ocr.py --book LucVanTien1883 --engine tesseract \\
      --start 10 --limit 10 --no-margin --out measure_out/qn_engine/LucVanTien1883_tess
  .venv/bin/python scripts/measure/qn_print_ocr.py --book LucVanTien1883 --engine vietocr    ... _vietocr
  .venv/bin/python scripts/measure/qn_engine_compare.py            # → measure_out/qn_engine/SUMMARY.json

Ba chỉ số/engine, trên CÙNG tập trang:
  n_lines     số dòng thơ trích được (so với số dòng của bản chốt cùng trang).
  oov         tỉ lệ âm tiết ngoài từ điển QN (Dict/QuocNgu_SinoNom.csv) — proxy lỗi nhận dạng, không cần GT.
  CER         Levenshtein ký tự so PHIÊN ÂM CHUẨN của dị bản (LVT↔1916 Nôm Foundation, KVK↔Kiều 1871 LVĐ),
              căn dòng bằng cửa sổ neo (tìm offset tốt nhất trong danh sách câu tham chiếu) + NW mức dòng.
              CER này GỒM cả khác biệt dị bản thật; chỉ dùng để SO HAI ENGINE trên cùng dòng, không đọc tuyệt đối.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "measure_out" / "qn_engine"

REFS = {
    "LucVanTien1883": REPO / "data/LucVanTien1916/nomfoundation_lvt_phienam.json",
    # 2026-09-23 (vòng 8): trả lại tham chiếu của KVK1884 — bản phiên âm Kiều 1871 (Liễu Văn Đường)
    # nằm trong thư mục data/TruyenKieuPhongTinhCoLuc/ nhưng là dị bản của KimVanKieu1884, không
    # phải dữ liệu của bản chép tay PTCL đã loại khỏi phạm vi. Dòng lọc dưới vẫn bỏ ref thiếu tệp.
    "KimVanKieu1884": REPO / "data/TruyenKieuPhongTinhCoLuc/thamchieu_kieu_1871_LieuVanDuong_phienam.json",
}
REFS = {b: p for b, p in REFS.items() if p.exists()}
PROD = {b: REPO / "measure_out" / b / "qn_ocr" / "verses.tsv" for b in REFS}


def norm(t: str) -> str:
    from core.text.text_utils import clean_line_text, normalize_tone_marks
    t = normalize_tone_marks(unicodedata.normalize("NFC", clean_line_text(t or "")).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\wÀ-ỹ\s]", "", t)).strip()


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sim(a: str, b: str) -> float:
    m = max(len(a), len(b))
    return 1.0 - lev(a, b) / m if m else 0.0


_DICT = None


def qn_dict() -> set:
    global _DICT
    if _DICT is None:
        from core.text.dictionary import load_qn_to_nom
        from core.text.text_utils import normalize_tone_marks
        _DICT = {normalize_tone_marks(k.lower())
                 for k in load_qn_to_nom(str(REPO / "Dict/QuocNgu_SinoNom.csv"))}
    return _DICT


def oov_rate(lines: list[str]) -> tuple[int, int]:
    from core.text.text_utils import normalize_tone_marks, split_to_syllables
    D = qn_dict()
    n = bad = 0
    for t in lines:
        for s in split_to_syllables(norm(t)):
            if not any(c.isalpha() for c in s):
                continue
            n += 1
            bad += int(normalize_tone_marks(s.lower()) not in D)
    return bad, n


def ref_lines(book: str) -> list[str]:
    d = json.loads(REFS[book].read_text(encoding="utf-8"))
    out = []
    for _, rows in d["pages"].items():
        for r in rows:
            out.append(r.get("qn") or "")
    return out


def read_verses(p: Path) -> dict[str, list[str]]:
    by = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            by.setdefault(r["page"], []).append(r["line_text"])
    return by


def nw_cer(hyp: list[str], ref: list[str]) -> tuple[int, int]:
    """Căn dòng đơn điệu (điểm = độ giống chuẩn hoá), trả (tổng sửa ký tự, tổng ký tự tham chiếu)."""
    n, m = len(ref), len(hyp)
    NEG = -10 ** 9
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0], bt[i][0] = -0.6 * i, "up"
    for j in range(1, m + 1):
        dp[0][j], bt[0][j] = -0.6 * j, "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sc = sim(norm(ref[i - 1]), norm(hyp[j - 1]))
            best, arg = dp[i - 1][j - 1] + sc, "diag"
            if dp[i - 1][j] - 0.6 > best:
                best, arg = dp[i - 1][j] - 0.6, "up"
            if dp[i][j - 1] - 0.6 > best:
                best, arg = dp[i][j - 1] - 0.6, "left"
            dp[i][j], bt[i][j] = best, arg
    edits = chars = 0
    i, j = n, m
    while i > 0 or j > 0:
        a = bt[i][j]
        if a == "diag":
            r, h = norm(ref[i - 1]), norm(hyp[j - 1])
            edits += lev(r, h)
            chars += len(r)
            i, j = i - 1, j - 1
        elif a == "up":
            r = norm(ref[i - 1])
            edits += len(r)
            chars += len(r)
            i -= 1
        else:
            edits += len(norm(hyp[j - 1]))
            j -= 1
    return edits, chars


def anchor_window(page_lines: list[str], ref: list[str], slack: int = 0) -> list[str]:
    """Tìm đoạn câu tham chiếu ứng với trang: quét offset, chấm bằng 3 dòng đầu + 2 dòng cuối.

    slack = 0: cửa sổ đúng bằng số dòng của trang (thêm dòng thừa sẽ bị NW tính là XOÁ trọn dòng
    và thổi CER lên giả tạo — đã đo: slack 4 làm CER 0,07 -> 0,51)."""
    k = len(page_lines)
    probe = [norm(x) for x in (page_lines[:3] + page_lines[-2:])]
    best, best_o = -1.0, 0
    for o in range(0, max(1, len(ref) - k)):
        seg = ref[o:o + k]
        if len(seg) < k:
            break
        cand = [norm(x) for x in (seg[:3] + seg[-2:])]
        s = sum(sim(a, b) for a, b in zip(probe, cand)) / max(1, len(probe))
        if s > best:
            best, best_o = s, o
    return ref[max(0, best_o - slack):best_o + k + slack], round(best, 3)


def main():
    res = {}
    for book in REFS:
        prod = read_verses(PROD[book]) if PROD[book].exists() else {}
        ref = ref_lines(book)
        runs = {}
        for eng in ("tess", "vietocr"):
            p = OUT / f"{book}_{eng}" / "verses.tsv"
            if p.exists():
                runs[eng] = read_verses(p)
        pages = sorted(set.intersection(*[set(v) for v in runs.values()])) if runs else []
        per = {e: dict(n_lines=0, oov=0, n_syl=0, edits=0, chars=0, pages=0) for e in runs}
        prod_lines = sum(len(prod.get(pg, [])) for pg in pages)
        page_rows = []
        for pg in pages:
            base = runs["tess"].get(pg, [])
            if len(base) < 3:
                continue
            win, score = anchor_window(base, ref)
            row = dict(page=pg, n_prod=len(prod.get(pg, [])), anchor_sim=score)
            for e, by in runs.items():
                hyp = by.get(pg, [])
                ed, ch = nw_cer(hyp, win)
                b, n = oov_rate(hyp)
                per[e]["n_lines"] += len(hyp)
                per[e]["oov"] += b
                per[e]["n_syl"] += n
                per[e]["edits"] += ed
                per[e]["chars"] += ch
                per[e]["pages"] += 1
                row[f"{e}_n"] = len(hyp)
                row[f"{e}_cer"] = round(ed / ch, 3) if ch else None
            page_rows.append(row)
        for e, s in per.items():
            s["cer"] = round(s["edits"] / s["chars"], 4) if s["chars"] else None
            s["oov_pct"] = round(100 * s["oov"] / s["n_syl"], 2) if s["n_syl"] else None
        res[book] = dict(pages=len(page_rows), prod_lines=prod_lines, engines=per, per_page=page_rows,
                         ref=str(REFS[book].relative_to(REPO)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for b, r in res.items():
        print(f"== {b}  trang={r['pages']}  dòng bản chốt={r['prod_lines']}")
        for e, s in r["engines"].items():
            print(f"   {e:8s} n_lines={s['n_lines']:4d}  CER={s['cer']}  OOV={s['oov_pct']}% "
                  f"({s['oov']}/{s['n_syl']})")


if __name__ == "__main__":
    main()
