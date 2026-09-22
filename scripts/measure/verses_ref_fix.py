#!/usr/bin/env python
"""verses_ref_fix.py — B1': sửa dòng QN OCR (verses.tsv) bằng phiên âm chuẩn của một dị bản gần (2026-09-22).

Với mỗi dòng QN OCR (tesseract, `measure_out/<book>/qn_ocr/verses.tsv`, thứ tự seq_no) tìm câu tham chiếu
(json Nôm Foundation: pages → [{verse_no, nom, qn}]) có QN:
  * GIỐNG HỆT sau chuẩn hoá (clean_line_text + split_to_syllables + normalize_tone_marks + lower, như pipeline)
    → `qn_source = <ref>_exact`, text thay bằng QN tham chiếu (cùng nội dung, chỉ khác dấu câu/hoa-thường);
  * hoặc KHỚP MỜ: độ giống theo âm tiết (căn chỉnh đơn điệu NW; âm giống hệt 1,0; giống sau bỏ dấu 0,9;
    còn lại tỉ số ký tự difflib ≥ 0,5) chia cho max(số âm hai bên) ≥ --fuzzy-min (mặc định 0,9), ứng viên
    tốt nhất DUY NHẤT (hơn ứng viên nhì ≥ --margin), CÙNG PARITY (dòng OCR 6 âm chỉ khớp câu lục, 8 âm chỉ khớp
    câu bát; dòng OCR ≠ 6/8 âm khớp câu 6 hoặc 8) và offset (ref_idx − seq_no) không lệch quá --offset-tol so với
    trung vị offset của các dòng exact lân cận (±--ctx dòng)
    → `qn_source = <ref>_fuzzy`, text thay bằng QN tham chiếu (chữa lỗi OCR chữ in: dấu, âm lệch, thiếu/thừa âm);
  * không thoả → giữ nguyên, `qn_source = ocr`.
Ghi `verses_b1.tsv` (mọi cột cũ + qn_source, ref_idx, ref_sim, ref_nom, line_text_ocr, n_syll_ocr), `matches.csv`,
`summary.json` (số dòng theo nguồn, số dòng ≠6/8 âm trước/sau, độ nhạy theo ngưỡng 0,75/0,8/0,85/0,9/1,0).

CAVEAT (ghi trong summary): thay QN OCR bằng QN 1871 rồi đối chứng nhãn với Nôm 1871 là TỰ KHẲNG ĐỊNH một phần
(câu exact: QN OCR đã bằng 1871 sẵn → không đổi đầu vào; câu fuzzy: đầu vào đã đổi theo 1871). Dòng fuzzy còn có
thể XOÁ dị bản QN thật của 1884 (câu 1884 khác 1871 một âm mà OCR đọc đúng) — chỉ đo được bằng dị bản thứ ba (1872).

Chạy:
  .venv/bin/python scripts/measure/verses_ref_fix.py --book KimVanKieu1884 \
      --ref data/TruyenKieuPhongTinhCoLuc/thamchieu_kieu_1871_LieuVanDuong_phienam.json --ref-name nf1871
  → measure_out/KimVanKieu1884/qn_ref_fix/{verses_b1.tsv,matches.csv,summary.json}
  .venv/bin/python scripts/measure/verses_ref_fix.py --selftest
Chỉ đọc data/ và measure_out/; không sửa pipeline/ core/.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from core.text.text_utils import clean_line_text, fold_text, normalize_tone_marks, split_to_syllables  # noqa: E402

FUZZY_MIN = 0.9
MARGIN = 0.1          # ứng viên nhất phải hơn ứng viên nhì ít nhất bấy nhiêu
OFFSET_TOL = 3        # |offset − trung vị offset exact lân cận| tối đa
CTX = 6               # dòng exact lân cận xét mỗi bên
PREFILTER = 0.4       # tỉ lệ âm (bỏ dấu) chung tối thiểu trước khi tính NW (chỉ để nhanh)


# --------------------------------------------------------------------------- #
REF_STRIP = "“”‘’«»‹›\"'"   # dấu ngoặc kép/đơn kiểu in mà clean_line_text KHÔNG bỏ (tạo token '“xót' ≠ 'xót')


def sanitize_ref(text: str) -> str:
    """QN tham chiếu → bỏ các dấu ngoặc REF_STRIP (giữ dấu câu khác để clean_line_text xử lý như dòng OCR)."""
    return "".join(ch for ch in text if ch not in REF_STRIP)


def canon_syl(t: str) -> str:
    return normalize_tone_marks(unicodedata.normalize("NFC", t).strip().lower())


def qn_tokens(text: str) -> list[str]:
    toks = [t for t in split_to_syllables(clean_line_text(text or "")) if any(ch.isalpha() for ch in t)]
    return [canon_syl(t) for t in toks]


def tok_sim(a: str, b: str) -> float:
    """Độ giống hai âm tiết đã chuẩn hoá: 1 giống hệt; 0,9 giống sau bỏ dấu (lỗi dấu OCR); tỉ số ký tự ≥ 0,5; else 0."""
    if a == b:
        return 1.0
    fa, fb = fold_text(a), fold_text(b)
    if fa == fb:
        return 0.9
    r = SequenceMatcher(None, fa, fb).ratio()
    return r if r >= 0.5 else 0.0


def _nw(o: list[str], r: list[str]) -> list[list[float]]:
    n, m = len(o), len(r)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        oi = o[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = max(prev[j], row[j - 1], prev[j - 1] + tok_sim(oi, r[j - 1]))
    return dp


def line_sim(o: list[str], r: list[str]) -> float:
    """Căn chỉnh đơn điệu (NW, gap 0) tối đa tổng tok_sim, chia cho max(len) → [0, 1]; 1 = giống hệt."""
    if not o or not r:
        return 0.0
    return _nw(o, r)[len(o)][len(r)] / max(len(o), len(r))


def token_diff(o: list[str], r: list[str], qn_dict: set[str] | None = None) -> list[dict]:
    """Truy vết NW → danh sách khác biệt OCR↔ref: kind ∈ {tone (chỉ dấu), near (tỉ số ≥0,5), diff, ins (OCR thừa),
    del (OCR thiếu)}; in_dict = âm OCR có trong từ điển QN (âm hợp lệ → có thể là dị bản QN thật của 1884)."""
    dp = _nw(o, r)
    i, j = len(o), len(r)
    out = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(dp[i][j] - (dp[i - 1][j - 1] + tok_sim(o[i - 1], r[j - 1]))) < 1e-9:
            a, b = o[i - 1], r[j - 1]
            if a != b:
                ts = tok_sim(a, b)
                kind = "tone" if ts == 0.9 else ("near" if ts >= 0.5 else "diff")
                out.append(dict(pos_ref=j - 1, ocr=a, ref=b, kind=kind, in_dict=int(bool(qn_dict) and a in qn_dict)))
            i, j = i - 1, j - 1
        elif i > 0 and abs(dp[i][j] - dp[i - 1][j]) < 1e-9:
            out.append(dict(pos_ref=None, ocr=o[i - 1], ref="", kind="ins", in_dict=int(bool(qn_dict) and o[i - 1] in qn_dict)))
            i -= 1
        else:
            out.append(dict(pos_ref=j - 1, ocr="", ref=r[j - 1], kind="del", in_dict=0))
            j -= 1
    return out[::-1]


def qn_dict_syllables() -> set[str]:
    from core.text.dictionary import load_qn_to_nom
    return {canon_syl(k) for k in load_qn_to_nom(str(REPO / "Dict" / "QuocNgu_SinoNom.csv"))}


def load_ref_lines(jf: Path) -> list[dict]:
    """json Nôm Foundation → [{idx (1-based, thứ tự dòng), qn, nom, tokens, page, verse_no}]."""
    out = []
    for pg, vs in json.load(open(jf, encoding="utf-8"))["pages"].items():
        for v in vs:
            qn = sanitize_ref(v["qn"])
            out.append(dict(idx=len(out) + 1, qn=qn, nom=unicodedata.normalize("NFC", v.get("nom") or ""),
                            tokens=qn_tokens(qn), page=pg, verse_no=v.get("verse_no")))
    return out


def load_ocr_rows(tsv: Path) -> list[dict]:
    rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
    rows.sort(key=lambda r: int(r["seq_no"]))
    return rows


def _parity_ok(n_ocr: int, n_ref: int) -> bool:
    return n_ref == n_ocr if n_ocr in (6, 8) else n_ref in (6, 8)


def match_lines(rows: list[dict], refs: list[dict], fuzzy_min: float = FUZZY_MIN, margin: float = MARGIN,
                offset_tol: int = OFFSET_TOL, ctx: int = CTX, ref_name: str = "ref") -> list[dict]:
    """Trả 1 bản ghi/dòng OCR: qn_source, ref_idx, ref_sim, best/second sim, offset, lý do từ chối."""
    by_tok: dict[tuple, list[int]] = {}
    for r in refs:
        by_tok.setdefault(tuple(r["tokens"]), []).append(r["idx"])
    ref_fold = [set(fold_text(t) for t in r["tokens"]) for r in refs]
    n = len(rows)
    res: list[dict] = []
    # 1) exact
    for r in rows:
        toks = qn_tokens(r["line_text"])
        seq = int(r["seq_no"])
        rec = dict(seq_no=seq, n_ocr=len(toks), tokens=toks, qn_source="ocr", ref_idx=None, ref_sim=None,
                   best_sim=None, second_sim=None, offset=None, reason="")
        cands = by_tok.get(tuple(toks), []) if toks else []
        if cands:
            # trùng nhiều câu giống hệt → chọn câu gần seq nhất (giải ở bước offset sau)
            rec["exact_cands"] = cands
        res.append(rec)
    # offset kỳ vọng từ exact duy nhất
    for rec in res:
        c = rec.get("exact_cands")
        if c and len(c) == 1:
            rec.update(qn_source=f"{ref_name}_exact", ref_idx=c[0], ref_sim=1.0, best_sim=1.0, offset=c[0] - rec["seq_no"])

    def local_offset(i: int) -> float | None:
        offs = []
        for j in range(max(0, i - ctx), min(n, i + ctx + 1)):
            if j != i and res[j]["qn_source"].endswith("_exact") and res[j]["offset"] is not None:
                offs.append(res[j]["offset"])
        return statistics.median(offs) if offs else None

    for i, rec in enumerate(res):
        c = rec.get("exact_cands")
        if c and len(c) > 1:
            lo = local_offset(i)
            pick = min(c, key=lambda k: abs((k - rec["seq_no"]) - (lo if lo is not None else 0)))
            rec.update(qn_source=f"{ref_name}_exact", ref_idx=pick, ref_sim=1.0, best_sim=1.0, offset=pick - rec["seq_no"],
                       reason=f"exact_multi:{len(c)}")
    # 2) fuzzy
    for i, rec in enumerate(res):
        if rec["qn_source"] != "ocr" or not rec["tokens"]:
            if not rec["tokens"]:
                rec["reason"] = "empty"
            continue
        toks = rec["tokens"]
        fo = set(fold_text(t) for t in toks)
        scored: list[tuple[float, int]] = []
        for k, r in enumerate(refs):
            if not r["tokens"] or not _parity_ok(len(toks), len(r["tokens"])):
                continue
            if len(fo & ref_fold[k]) / max(len(fo), len(ref_fold[k])) < PREFILTER:
                continue
            scored.append((line_sim(toks, r["tokens"]), r["idx"]))
        if not scored:
            rec["reason"] = "no_candidate"
            continue
        scored.sort(reverse=True)
        best, bidx = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        rec.update(best_sim=round(best, 4), second_sim=round(second, 4))
        if best < fuzzy_min:
            rec["reason"] = "below_min"
            continue
        if best - second < margin:
            rec["reason"] = "ambiguous"
            continue
        lo = local_offset(i)
        off = bidx - rec["seq_no"]
        if lo is not None and abs(off - lo) > offset_tol:
            rec["reason"] = f"offset_off:{off}vs{lo}"
            continue
        rec.update(qn_source=f"{ref_name}_fuzzy", ref_idx=bidx, ref_sim=round(best, 4), offset=off,
                   reason="" if lo is not None else "no_ctx")
    for rec in res:
        rec.pop("exact_cands", None)
    return res


def build(rows: list[dict], refs: list[dict], matches: list[dict]) -> list[dict]:
    """verses_b1: dòng OCR với line_text/n_syll thay bằng tham chiếu khi khớp; giữ bản gốc ở *_ocr."""
    ref_by = {r["idx"]: r for r in refs}
    out = []
    for r, m in zip(rows, matches):
        o = dict(r)
        o["line_text_ocr"] = r["line_text"]
        o["n_syll_ocr"] = r.get("n_syll", "")
        o["qn_source"] = m["qn_source"]
        o["ref_idx"] = m["ref_idx"] if m["ref_idx"] is not None else ""
        o["ref_sim"] = m["ref_sim"] if m["ref_sim"] is not None else ""
        o["ref_nom"] = ""
        if m["ref_idx"] is not None:
            ref = ref_by[m["ref_idx"]]
            o["line_text"] = ref["qn"]
            o["n_syll"] = str(len(ref["tokens"]))
            # ref_nom chỉ ghi khi số chữ Nôm == số âm (để adapter dùng làm ứng viên đồng âm theo vị trí)
            if len(ref["nom"]) == len(ref["tokens"]):
                o["ref_nom"] = ref["nom"]
        out.append(o)
    return out


def summarize(rows: list[dict], matches: list[dict], fixed: list[dict], refs: list[dict], ref_name: str,
              fuzzy_min: float) -> dict:
    src = Counter(m["qn_source"] for m in matches)
    n_ocr_bad = sum(1 for m in matches if m["n_ocr"] not in (6, 8))
    n_fixed_bad = sum(1 for o in fixed if len(qn_tokens(o["line_text"])) not in (6, 8))
    n_bad_still_ocr = sum(1 for o, m in zip(fixed, matches) if m["qn_source"] == "ocr" and m["n_ocr"] not in (6, 8))
    # dòng fuzzy: đổi gì? (token_diff OCR↔ref)
    fz = [m for m in matches if m["qn_source"].endswith("_fuzzy")]
    fz_len_change = sum(1 for m in fz if m["n_ocr"] != len(refs[m["ref_idx"] - 1]["tokens"]))
    dk = Counter()
    n_lines_possible_variant = 0
    for m in fz:
        d = m.get("diff") or []
        for x in d:
            dk[x["kind"]] += 1
            if x["in_dict"] and x["kind"] in ("near", "diff"):
                dk["near_or_diff_in_dict(possible_1884_variant_erased)"] += 1
        if any(x["in_dict"] and x["kind"] in ("near", "diff") for x in d):
            n_lines_possible_variant += 1
    reasons = Counter(m["reason"].split(":")[0] for m in matches if m["qn_source"] == "ocr")
    # độ nhạy theo ngưỡng (chỉ tính best_sim, chưa kèm điều kiện offset/ambiguous)
    sens = {}
    for th in (0.75, 0.8, 0.85, 0.9, 1.0):
        sens[str(th)] = sum(1 for m in matches if m["best_sim"] is not None and m["best_sim"] >= th
                            and (m["best_sim"] - (m["second_sim"] or 0)) >= MARGIN)
    offs = Counter(m["offset"] for m in matches if m["offset"] is not None)
    return dict(
        book_rows=len(rows), ref_lines=len(refs), ref_name=ref_name, fuzzy_min=fuzzy_min, margin=MARGIN,
        offset_tol=OFFSET_TOL, ctx=CTX,
        by_source=dict(src),
        n_rows_syll_not_6_8_before=n_ocr_bad, n_rows_syll_not_6_8_after=n_fixed_bad,
        n_rows_syll_not_6_8_still_ocr=n_bad_still_ocr,
        n_fuzzy_len_changed=fz_len_change,
        fuzzy_token_diff_kinds=dict(dk), n_fuzzy_lines_possible_variant_erased=n_lines_possible_variant,
        reject_reasons=dict(reasons),
        sensitivity_best_sim_ge=sens,
        offsets_distinct={str(k): v for k, v in sorted(offs.items())},
        ref_lines_with_nom_eq_len=sum(1 for r in refs if len(r["nom"]) == len(r["tokens"])),
        caveat=("exact: QN OCR đã bằng tham chiếu → không đổi đầu vào; fuzzy: đầu vào QN đã đổi theo tham chiếu → "
                "đối chứng nhãn với cùng tham chiếu là tự khẳng định; fuzzy có thể xoá dị bản QN thật của 1884 — "
                "đo bằng dị bản thứ ba"),
    )


def run(book: str, ref: Path, ref_name: str, out_dir: Path, fuzzy_min: float, verses: Path | None) -> dict:
    tsv = verses or (REPO / "measure_out" / book / "qn_ocr" / "verses.tsv")
    rows = load_ocr_rows(tsv)
    refs = load_ref_lines(ref)
    matches = match_lines(rows, refs, fuzzy_min=fuzzy_min, ref_name=ref_name)
    qd = qn_dict_syllables()
    for m in matches:
        if m["qn_source"].endswith("_fuzzy"):
            m["diff"] = token_diff(m["tokens"], refs[m["ref_idx"] - 1]["tokens"], qd)
    fixed = build(rows, refs, matches)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) + ["qn_source", "ref_idx", "ref_sim", "ref_nom", "line_text_ocr", "n_syll_ocr"]
    with open(out_dir / "verses_b1.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(fixed)
    with open(out_dir / "matches.csv", "w", encoding="utf-8", newline="") as f:
        mf = ["seq_no", "n_ocr", "qn_source", "ref_idx", "ref_sim", "best_sim", "second_sim", "offset", "reason",
              "diff", "line_text_ocr", "line_text_ref"]
        w = csv.DictWriter(f, fieldnames=mf, extrasaction="ignore")
        w.writeheader()
        for r, m, o in zip(rows, matches, fixed):
            dtxt = ";".join(f"{x['kind']}:{x['ocr']}>{x['ref']}{'*' if x['in_dict'] else ''}" for x in (m.get("diff") or []))
            w.writerow({**m, "diff": dtxt, "line_text_ocr": r["line_text"], "line_text_ref": o["line_text"] if m["ref_idx"] else ""})
    s = summarize(rows, matches, fixed, refs, ref_name, fuzzy_min)
    s["inputs"] = dict(verses=str(tsv), ref=str(ref))
    s["outputs"] = dict(verses_b1=str(out_dir / "verses_b1.tsv"), matches=str(out_dir / "matches.csv"))
    (out_dir / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return s


# --------------------------------------------------------------------------- #
def mechanism_check(labels_csv: Path, trans_dir: Path, out_json: Path | None = None) -> dict:
    """Kiểm CƠ CHẾ sau build, KHÔNG dùng chữ Nôm tham chiếu (kim độc lập với tham chiếu): tại mỗi âm bị THAY trong dòng
    fuzzy (tone/near/diff; ins/del không có vị trí 1-1), chữ kim của ô đó có thuộc R(âm OCR cũ) / R(âm tham chiếu mới) không?
      kim ∉ R(cũ) & ∈ R(mới)  → sửa làm chữ kim giải thích được (sửa đúng lỗi OCR, ô có thể GOLD)
      kim ∈ R(cũ) & ∉ R(mới)  → sửa làm mất lời giải thích (nghi XOÁ dị bản QN thật của 1884 / 1871 khác văn bản)
      cả hai / không cái nào  → trung tính (dị thể chính tả sanh/sinh… hoặc kim sai)."""
    import glob
    from core.text.dictionary import load_qn_to_nom
    R = {canon_syl(k): set(v) for k, v in load_qn_to_nom(str(REPO / "Dict" / "QuocNgu_SinoNom.csv")).items()}
    lab = {}
    for r in csv.DictReader(open(labels_csv, encoding="utf-8", newline="")):
        if r.get("syl_idx", "") != "":
            lab[(r["page"], int(r["column"]), int(r["syl_idx"]))] = r
    st = Counter()
    bad = []
    for tf in sorted(glob.glob(str(trans_dir / "page_*.json"))):
        t = json.load(open(tf, encoding="utf-8"))
        page = Path(tf).stem
        for c in t["columns"]:
            lo = int(c.get("len_odd") or 0)
            for half, key in ((0, "verse_odd"), (1, "verse_even")):
                v = c.get(key) or {}
                if not (v.get("qn_source") or "").endswith("_fuzzy"):
                    continue
                o, rt = qn_tokens(v.get("line_text_ocr", "")), qn_tokens(v["text"])
                for d in token_diff(o, rt):
                    if d["kind"] not in ("tone", "near", "diff"):   # ins/del không có vị trí 1-1
                        continue
                    si = (0 if half == 0 else lo) + d["pos_ref"]
                    cell = lab.get((page, int(c["column"]), si))
                    if not cell or not cell.get("ocr_char"):
                        st["no_cell"] += 1
                        continue
                    k = cell["ocr_char"]
                    a, b = k in R.get(d["ocr"], set()), k in R.get(d["ref"], set())
                    key = {(False, True): "kim_notR_old_R_new(fix_explains_kim)", (True, False): "kim_R_old_notR_new(fix_loses_kim)",
                           (True, True): "both", (False, False): "neither"}[(a, b)]
                    st[key] += 1
                    st[f"{d['kind']}:{key}"] += 1
                    if a and not b:
                        bad.append(dict(page=page, column=int(c["column"]), syl_idx=si, kim=k, old=d["ocr"], new=d["ref"],
                                        tier=cell.get("tier")))
    n = sum(st[k] for k in ("kim_notR_old_R_new(fix_explains_kim)", "kim_R_old_notR_new(fix_loses_kim)", "both", "neither"))
    res = dict(labels=str(labels_csv), trans=str(trans_dir), n_changed_syllables_with_kim=n, counts=dict(st),
               pct={k: pct_(st[k], n) for k in ("kim_notR_old_R_new(fix_explains_kim)", "kim_R_old_notR_new(fix_loses_kim)", "both", "neither")},
               fix_loses_kim_examples=bad[:40])
    if out_json:
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def pct_(k, n):
    return round(100.0 * k / n, 1) if n else None


# --------------------------------------------------------------------------- #
def selftest() -> int:
    n = 0

    def ok(c, msg):
        nonlocal n
        if not c:
            print("FAIL:", msg)
            sys.exit(1)
        n += 1

    ok(qn_tokens(sanitize_ref("Than ôi! “Sắc nước hương trời,")) == ["than", "ôi", "sắc", "nước", "hương", "trời"]
       and "“" not in sanitize_ref("“xót"), "sanitize_ref bỏ ngoặc kép in")
    ok(tok_sim("trăm", "trăm") == 1.0 and tok_sim("cỗi", "cõi") == 0.9, "tok_sim giống hệt / bỏ dấu")
    ok(0.5 <= tok_sim("đều", "điều") < 0.9 and tok_sim("sốc", "mệnh") == 0.0, "tok_sim tỉ số ký tự / khác hẳn")
    ok(line_sim(["a", "b", "c"], ["a", "b", "c"]) == 1.0, "line_sim giống hệt")
    ok(abs(line_sim(["a", "b", "c", "d", "e", "f"], ["a", "b", "d", "e", "f"]) - 5 / 6) < 1e-9, "line_sim thiếu 1 âm = 5/6")
    refs = [dict(idx=i + 1, qn=q, nom="", tokens=qn_tokens(q), page="p", verse_no=None) for i, q in enumerate([
        "Trăm năm trong cõi người ta,", "Chữ tài chữ mệnh khéo là ghét nhau.", "Trải qua một cuộc bể dâu,",
        "Những điều trông thấy đã đau đớn lòng.", "Lạ gì bỉ sắc tư phong,", "Trời xanh quen thói má hồng đánh ghen.",
        "Cảo thơm lần giở trước đèn,", "Phong tình cổ lục còn truyền sử xanh."])]
    ocr = ["'Trăm năm, trong cỗi người ta,", "Chữ ¿à¿ chữ sốc khéo là ghét nhau!", "Trải qua một cuộc bê dâu;",
           "Những đều trông thấy đã đau đón lòng!", "Lạ gì bỉ sắc tư phong,", "Trời xanh quen thói má hồng đánh ghen.",
           "Cảo thơm lần giở trước đèn", "Câu này hoàn toàn khác không có trong sách."]
    rows = [dict(seq_no=str(i + 1), line_text=t, n_syll=str(len(qn_tokens(t)))) for i, t in enumerate(ocr)]
    m = match_lines(rows, refs, ref_name="nf")
    ok(m[0]["qn_source"] == "nf_fuzzy" and m[0]["ref_idx"] == 1, f"dòng 1 lỗi dấu (cỗi/cõi) → fuzzy {m[0]}")
    ok(m[1]["qn_source"] == "ocr" and m[1]["reason"] == "below_min", f"dòng 2 rác 2 âm → giữ OCR {m[1]}")
    ok(m[2]["qn_source"] == "nf_fuzzy" and m[3]["qn_source"] == "nf_fuzzy", "dòng 3/4 lỗi dấu, đều/điều → fuzzy")
    ok(m[4]["qn_source"] == "nf_exact" and m[5]["qn_source"] == "nf_exact" and m[6]["qn_source"] == "nf_exact",
       "dòng 5–7 giống hệt (bỏ dấu câu) → exact")
    ok(m[7]["qn_source"] == "ocr", "dòng 8 không có trong tham chiếu → ocr")
    fixed = build(rows, refs, m)
    ok(fixed[0]["line_text"] == refs[0]["qn"] and fixed[0]["line_text_ocr"] == ocr[0] and fixed[0]["n_syll"] == "6",
       "build: text thay, bản OCR giữ ở line_text_ocr")
    ok(fixed[7]["line_text"] == ocr[7] and fixed[7]["qn_source"] == "ocr", "build: không khớp giữ nguyên")
    # parity: dòng OCR 6 âm không được khớp câu 8 dù giống 6/8
    rows2 = [dict(seq_no="1", line_text="Chữ tài chữ mệnh khéo là", n_syll="6")]
    m2 = match_lines(rows2, refs, ref_name="nf")
    ok(m2[0]["qn_source"] == "ocr" and m2[0]["reason"] in ("no_candidate", "below_min"), f"parity 6 ≠ 8 → không khớp {m2[0]}")
    # dòng OCR 7 âm (thừa 1) khớp câu 6 nếu sim ≥ min: 6/7 = 0,857 < 0,9 → ocr; với fuzzy_min 0,85 → fuzzy
    rows3 = [dict(seq_no="1", line_text="Trăm năm trong cõi người ta ta", n_syll="7")]
    ok(match_lines(rows3, refs, ref_name="nf")[0]["qn_source"] == "ocr", "thừa 1 âm: 6/7 < 0,9 → ocr")
    ok(match_lines(rows3, refs, fuzzy_min=0.85, ref_name="nf")[0]["qn_source"] == "nf_fuzzy", "thừa 1 âm với min 0,85 → fuzzy")
    # offset guard: dòng fuzzy lệch offset quá xa so với exact lân cận → từ chối
    refs2 = refs + [dict(idx=9, qn="Trăm năm trong cõi người ta,", nom="", tokens=qn_tokens("Trăm năm trong cõi người ta,"),
                         page="p", verse_no=None)] * 0
    rows4 = [dict(seq_no=str(i + 1), line_text=t, n_syll="") for i, t in enumerate(
        ["Lạ gì bỉ sắc tư phong,", "Trời xanh quen thói má hồng đánh ghen.", "Trăm năm trong cỗi người ta,"])]
    m4 = match_lines(rows4, refs2, ref_name="nf", offset_tol=1)
    ok(m4[2]["qn_source"] == "ocr" and m4[2]["reason"].startswith("offset_off"), f"offset lệch → từ chối {m4[2]}")
    d = token_diff(qn_tokens("Những đều trông thấy đã đau đón lòng"), qn_tokens("Những điều trông thấy đã đau đớn lòng"), {"đều"})
    ok([(x["kind"], x["ocr"], x["ref"], x["in_dict"]) for x in d] == [("near", "đều", "điều", 1), ("tone", "đón", "đớn", 0)],
       f"token_diff near/tone + in_dict {d}")
    d = token_diff(["a", "b", "d", "e", "f"], ["a", "b", "c", "d", "e", "f"])
    ok(len(d) == 1 and d[0]["kind"] == "del" and d[0]["ref"] == "c", f"token_diff thiếu âm = del {d}")
    d = token_diff(["a", "x", "b"], ["a", "b"])
    ok(len(d) == 1 and d[0]["kind"] == "ins" and d[0]["ocr"] == "x", f"token_diff thừa âm = ins {d}")
    print(f"verses_ref_fix selftest: {n}/{n} PASS")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default="KimVanKieu1884")
    ap.add_argument("--ref", default=str(REPO / "data/TruyenKieuPhongTinhCoLuc/thamchieu_kieu_1871_LieuVanDuong_phienam.json"))
    ap.add_argument("--ref-name", default="nf1871")
    ap.add_argument("--verses", default=None, help="verses.tsv đầu vào (mặc định measure_out/<book>/qn_ocr/verses.tsv)")
    ap.add_argument("--out", default=None, help="thư mục ra (mặc định measure_out/<book>/qn_ref_fix)")
    ap.add_argument("--fuzzy-min", type=float, default=FUZZY_MIN)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-labels", default=None, help="sau build: labels.csv để kiểm cơ chế (mechanism_check)")
    ap.add_argument("--check-trans", default=None, help="thư mục transcriptions của lần ingest dùng verses_b1")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.check_labels:
        out = Path(a.out) if a.out else REPO / "measure_out" / a.book / "qn_ref_fix"
        out.mkdir(parents=True, exist_ok=True)
        name = Path(a.check_labels).parent.name
        r = mechanism_check(Path(a.check_labels), Path(a.check_trans), out / f"mechanism_check_{name}.json")
        print(json.dumps({k: r[k] for k in ("n_changed_syllables_with_kim", "pct")}, ensure_ascii=False))
        print("→", out / f"mechanism_check_{name}.json")
        return 0
    out = Path(a.out) if a.out else REPO / "measure_out" / a.book / "qn_ref_fix"
    s = run(a.book, Path(a.ref), a.ref_name, out, a.fuzzy_min, Path(a.verses) if a.verses else None)
    print(json.dumps({k: s[k] for k in ("by_source", "n_rows_syll_not_6_8_before", "n_rows_syll_not_6_8_after",
                                        "n_rows_syll_not_6_8_still_ocr", "n_fuzzy_len_changed", "fuzzy_token_diff_kinds",
                                        "n_fuzzy_lines_possible_variant_erased", "reject_reasons",
                                        "sensitivity_best_sim_ge", "offsets_distinct")}, ensure_ascii=False))
    print("→", out / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
