#!/usr/bin/env python3
"""borg_endtoend_eval.py — ĐỘ ĐÚNG trên CHỮ VIẾT TAY Công giáo: nhãn máy (pipeline tự động) vs chữ Nôm NGƯỜI phiên
của hai bản chép tay Vatican Borgiano Tonchinese (SachKinhThayCaBinh = Borg.Tonch.18, SachDungLyHoThan = Borg.Tonch.34).

Khuôn ihr_endtoend_eval. Adapter `pipeline/tools/ingest_borg_book.py` KHÔNG đọc cột `SinoNom_Char` — chỉ phép đo này đọc,
nên phép đo không vòng tròn. Đọc (0 API, CPU):
  data/<Sách>/<Sách>.xlsx                         câu người: img_id, sentence_id, SinoNom_Char, ChuQN_txt
  prepared/_auto/<Sách>/transcriptions/*.json     cột → khoảng âm của trang (syl_span) do adapter ghép
  prepared/_auto/<Sách>/kim_raw/*_lt2.json        chữ kim thô (đo kim trên chữ viết tay)
  prepared/_auto/<Sách>/dataset_out/labels_gated.csv   nhãn máy (mọi tầng)
  dataset/_ALL/gold_exact.csv                      trạng thái gold_exact (nếu bộ gộp đã có sách này)

Ghép ô ↔ chữ người qua VỊ TRÍ ÂM TIẾT: ô (trang, cột, syl_idx) → chỉ số âm của trang J = syl_span[0](cột) + syl_idx →
(câu, vị trí trong câu) theo số âm từng câu (làm sạch y hệt adapter) → chữ Nôm người cùng vị trí. CHỈ câu có số chữ Nôm
(sau clean_nom_text của ingest_borg_tonch) == số âm (câu "đếm bằng") mới được dùng.

Độ đúng nhãn (theo tầng, và của ô gold_exact = ok): strict (trùng hẳn) · V1+ (biến thể Unihan/OpenCC/kJapanese, bản chép
pipeline/gold_exact/common.var_eq_plus) · V1+ + 4 cặp quy ước người phiên đã biết (người phiên gõ 𠸜 cho hình 先, 𢧚/年,
𠀧/巴, 𧘇/意 — measure_out/_audit_2026-09-26/r5/borg_quality) · V1+ + 11 cặp (bảng PAIRS của q14_clean_keep_v5, độ nhạy).
CI 95 %: bootstrap CỤM TRANG (B = 2000, seed cố định) + Wilson (tham khảo).
Độ đúng của CHÍNH KIM trên chữ viết tay (độc lập pipeline): chuỗi chữ kim cả trang (cột phải→trái, trên→dưới — cùng cách
gom cột của adapter) căn đơn điệu tối đa số khớp (LCS) với chuỗi chữ người cả trang (mọi câu) → precision = khớp/chữ kim,
recall = khớp/chữ người (strict, V1+, +4 cặp).

Chạy:
  .venv/bin/python scripts/measure/borg_endtoend_eval.py --book all
  .venv/bin/python scripts/measure/borg_endtoend_eval.py --book SachDungLyHoThan --labels <labels_gated.csv>
  .venv/bin/python scripts/measure/borg_endtoend_eval.py --selftest
Đầu ra: measure_out/<Sách>/borg_endtoend/{summary.json, cells.csv, errors.csv}; thiếu nhãn máy / cache kim -> invariant SKIP.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BOOKS = ("SachKinhThayCaBinh", "SachDungLyHoThan")
AUTO = REPO / "prepared" / "_auto"
PAIRS4 = {"𠸜": "先", "𢧚": "年", "𠀧": "巴", "𧘇": "意"}
PAIRS11 = {"𢧚": "年", "𠸜": "先", "𠀧": "巴", "𧘇": "意", "𠰺": "代", "𧶮": "古", "𠫾": "多", "𢚸": "弄", "𩈘": "末", "𡗊": "饒",
           "𨕭": "連"}
BOOT_B, BOOT_SEED = 2000, 20260927
# Ước lượng STT (SUY ĐOÁN — không có sự thật người) để đối chiếu: docs/GOLD_CHINH_XAC_2026-09-27.md §6 (bộ ước lượng TN3, tập "lai").
STT_EST = {"stt2": "98,39 [96,40–99,24]", "stt4": "98,26 [96,19–99,19]", "stt11": "98,47 [96,48–99,28]"}


# ---------------------------------------------------------------------------- người
def human_pages(book: str) -> dict[str, dict]:
    """{page_XXXX: {img_id, sents: [{sid, nom, syl, eq}]}} — cùng đánh số trang + làm sạch QN như adapter."""
    import openpyxl
    from pipeline.tools.ingest_borg_book import load_pages
    from pipeline.tools.ingest_borg_tonch import BOOK_SPECS, clean_nom_text
    spec = BOOK_SPECS[book]
    typo = spec.get("typo_map") or {}
    wb = openpyxl.load_workbook(Path(spec["data_dir"]) / spec["xlsx_name"], read_only=True, data_only=True)
    ws = wb.active
    it = ws.iter_rows(values_only=True)
    hdr = [str(h or "").strip() for h in next(it)]
    ix = {k: hdr.index(k) for k in ("img_id", "sentence_id", "SinoNom_Char")}
    nom_by = defaultdict(list)
    for r, row in enumerate(it, start=2):
        row = tuple(row or ()) + (None,) * max(0, len(hdr) - len(row or ()))
        img = row[ix["img_id"]]
        if not img:
            continue
        img = typo.get(str(img).strip(), str(img).strip())
        nom_by[img].append(clean_nom_text(str(row[ix["SinoNom_Char"]] or "")))
    wb.close()
    out = {}
    for rec in load_pages(book):
        noms = nom_by.get(rec["img_id"], [])
        if len(noms) != len(rec["sentences"]):
            raise SystemExit(f"{book} {rec['page_name']}: số câu Nôm {len(noms)} ≠ số câu QN {len(rec['sentences'])}")
        sents = []
        for s, nom in zip(rec["sentences"], noms):
            sents.append(dict(sid=s["sentence_id"], nom=nom, syl=s["syllables"], eq=len(nom) == len(s["syllables"])))
        out[rec["page_name"]] = dict(img_id=rec["img_id"], sents=sents, exists=rec["file"].exists())
    return out


def syl_to_char(page: dict) -> list[str | None]:
    """Chỉ số âm của trang -> chữ người (None nếu câu không đếm bằng)."""
    out = []
    for s in page["sents"]:
        out += list(s["nom"]) if s["eq"] else [None] * len(s["syl"])
    return out


def page_syllables(page: dict) -> list[str]:
    return [x for s in page["sents"] for x in s["syl"]]


# ---------------------------------------------------------------------------- so khớp
class Eq:
    def __init__(self):
        from pipeline.gold_exact.common import Assets, set_lexicon, var_eq_plus
        set_lexicon(Assets())
        self.v = var_eq_plus

    def strict(self, a, g):
        return bool(a) and a == g

    def v1p(self, a, g):
        return bool(a) and bool(g) and self.v(a, g)

    def pairs(self, a, g, P):
        if not a or not g:
            return False
        if self.v1p(a, g):
            return True
        return P.get(g) is not None and self.v1p(a, P[g]) or P.get(a) is not None and self.v1p(P[a], g)


def boot_ci(k_by_page: dict, n_by_page: dict, B=BOOT_B, seed=BOOT_SEED) -> tuple[float, float]:
    import numpy as np
    pages = sorted(n_by_page)
    if not pages:
        return (float("nan"), float("nan"))
    k = np.array([k_by_page.get(p, 0) for p in pages], float); n = np.array([n_by_page[p] for p in pages], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pages), size=(B, len(pages)))
    ks, ns = k[idx].sum(1), n[idx].sum(1)
    r = ks / np.maximum(ns, 1)
    return (round(float(np.quantile(r, 0.025)), 5), round(float(np.quantile(r, 0.975)), 5))


def rate_block(rows: list[dict], key: str) -> dict:
    """rows có 'page', 'gt', key∈{strict,v1p,p4,p11}; CI bootstrap cụm trang cho mọi thước."""
    from pipeline.gold_exact.common import wilson
    ev = [r for r in rows if r["gt"]]
    out = dict(n=len(rows), n_eval=len(ev))
    if not ev:
        return out
    n_by = Counter(r["page"] for r in ev)
    for m in ("strict", "v1p", "p4", "p11"):
        kb = Counter(r["page"] for r in ev if r[m])
        k = sum(kb.values())
        lo, hi = boot_ci(kb, n_by)
        wl, wh = wilson(k, len(ev))
        out[m] = dict(k=k, pt=round(k / len(ev), 5), ci_page=[lo, hi], wilson=[round(wl, 5), round(wh, 5)])
    return out


_EQ_MEMO: dict = {}


def levenshtein_pairs(a: list[str], b: list[str], eq) -> list[tuple[int, int]]:
    """Căn đơn điệu TỐI ĐA SỐ CẶP KHỚP (LCS: chèn/xoá = 1, thay = 2 = xoá + chèn, khớp theo eq) → cặp (i, j) khớp.
    Tối đa số khớp để precision/recall không phụ thuộc cách phá hoà của Levenshtein thay = 1. Bảng khớp tính trên chữ DUY NHẤT
    (nhớ theo (eq, x, y)); hàng DP vector hoá (chèn = cummin như align_sequences của ingest_prose_book)."""
    import numpy as np
    n, m = len(a), len(b)
    if not n or not m:
        return []
    ua = sorted(set(a)); ub = sorted(set(b))
    ia = np.array([ua.index(x) for x in a]) if len(ua) < 50 else np.searchsorted(np.array(ua, dtype=object), np.array(a, dtype=object))
    ib = np.array([ub.index(y) for y in b]) if len(ub) < 50 else np.searchsorted(np.array(ub, dtype=object), np.array(b, dtype=object))
    Eu = np.full((len(ua), len(ub)), 2, np.int32)
    for p_, x in enumerate(ua):
        for q_, y in enumerate(ub):
            k = (id(eq), x, y)
            v = _EQ_MEMO.get(k)
            if v is None:
                v = _EQ_MEMO[k] = 0 if eq(x, y) else 2
            Eu[p_, q_] = v
    E = Eu[ia][:, ib]
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    js = np.arange(m + 1)
    for i in range(1, n + 1):
        best = np.minimum(D[i - 1, :-1] + E[i - 1], D[i - 1, 1:] + 1)      # cột 1..m: thay/khớp hoặc xoá
        bb = np.concatenate(([D[i, 0]], best))
        D[i] = np.minimum.accumulate(bb - js) + js                         # chèn: cummin
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + E[i - 1, j - 1]:
            if E[i - 1, j - 1] == 0:
                pairs.append((i - 1, j - 1))
            i -= 1; j -= 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


@lru_cache(maxsize=None)
def _kim_sfx(auto: str, book: str) -> str:
    """Hậu tố cache kim của bản dựng (manifest.kim_cache_suffix; vắng -> _lt2 = tham số config hiện tại)."""
    mf = Path(auto) / book / "manifest.json"
    if mf.exists():
        return json.loads(mf.read_text(encoding="utf-8")).get("kim_cache_suffix", "_lt2")
    return "_lt2"


def kim_page_chars(book: str, page: str) -> list[str] | None:
    """Chữ kim thô của trang theo thứ tự đọc của adapter (cột phải→trái, trên→dưới); None nếu thiếu cache."""
    from pipeline.tools.ingest_borg_book import cluster_columns
    fs = sorted((AUTO / book / "kim_raw").glob(f"{page}{_kim_sfx(str(AUTO), book)}.json"))
    if not fs:
        return None
    d = json.loads(fs[0].read_text(encoding="utf-8"))
    cols, _ = cluster_columns(d.get("boxes") or [])
    return [c["char"] for col in cols for c in col]


def kim_block(book: str, H: dict, E: Eq) -> dict:
    """Kim thô vs chữ người cả trang (căn LCS) — không phụ thuộc pipeline."""
    tot = Counter()
    pages = 0
    k_by = defaultdict(Counter); n_by = {}
    for pg, h in sorted(H.items()):
        kc = kim_page_chars(book, pg)
        if kc is None:
            continue
        hc = [c for s in h["sents"] for c in s["nom"]]
        pages += 1
        tot["n_kim"] += len(kc); tot["n_human"] += len(hc)
        for name, f in (("strict", E.strict), ("v1p", E.v1p), ("p4", lambda a, g: E.pairs(a, g, PAIRS4))):
            m = len(levenshtein_pairs(kc, hc, f))
            tot[f"m_{name}"] += m; k_by[name][pg] = m
        n_by[pg] = len(kc)
    if not pages:
        return dict(n_pages=0)
    out = dict(n_pages=pages, n_kim=tot["n_kim"], n_human=tot["n_human"])
    for name in ("strict", "v1p", "p4"):
        m = tot[f"m_{name}"]
        out[name] = dict(match=m, precision=round(m / max(1, tot["n_kim"]), 5), recall=round(m / max(1, tot["n_human"]), 5),
                         precision_ci_page=boot_ci(k_by[name], n_by))
    return out


def load_labels(p: Path) -> list[dict]:
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_spans(book: str) -> dict[tuple[str, int], int]:
    """{(page, column): j0} của cột đã ghép (transcriptions adapter)."""
    out = {}
    for f in sorted((AUTO / book / "transcriptions").glob("page_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for c in d.get("columns") or []:
            if c.get("matched") and c.get("syl_span") and c["syl_span"][0] is not None:
                out[(f.stem, int(c["column"]))] = int(c["syl_span"][0])
    return out


def load_gold_exact(book: str) -> dict[str, str]:
    for p in (REPO / "dataset/_ALL/gold_exact.csv", REPO / "dataset" / book / "gold_exact.csv"):
        if p.exists():
            with open(p, encoding="utf-8", newline="") as f:
                return {r["cell_uid"]: r["gold_exact"] for r in csv.DictReader(f) if r.get("book_set") == book}
    return {}


def evaluate(book: str, labels: Path, out_dir: Path, E: Eq | None = None) -> dict:
    t0 = time.time()
    E = E or Eq()
    H = human_pages(book)
    s_all = [s for h in H.values() for s in h["sents"]]
    human = dict(n_pages=len(H), n_sentences=len(s_all), n_sent_eq=sum(s["eq"] for s in s_all),
                 n_syll=sum(len(s["syl"]) for s in s_all), n_syll_eq=sum(len(s["syl"]) for s in s_all if s["eq"]))
    human["frac_syll_eq"] = round(human["n_syll_eq"] / max(1, human["n_syll"]), 4)
    # trần của luật GOLD (kim ∈ R(âm)): tỉ lệ cặp NGƯỜI (chữ, âm) của câu đếm bằng có chữ ∈ R(âm) (Dict/QuocNgu_SinoNom.csv)
    from pipeline.gold_exact.common import R_of, var_in
    n_p = k_R = k_Rv = 0
    for s_ in s_all:
        if not s_["eq"]:
            continue
        for ch, sy in zip(s_["nom"], s_["syl"]):
            R = R_of(sy)
            n_p += 1; k_R += ch in R; k_Rv += var_in(ch, R)
    human["pairs_eq"] = n_p
    human["pair_char_in_R"] = round(k_R / max(1, n_p), 4)
    human["pair_char_in_R_v1"] = round(k_Rv / max(1, n_p), 4)
    inv = []

    def iv(name, expected, observed, ok):
        inv.append(dict(name=name, expected=expected, observed=observed, pass_=ok))
    # ánh xạ trang của adapter == ingest_borg_tonch (thư mục nhãn người)
    ref = json.loads((REPO / "prepared" / book / "manifest.json").read_text(encoding="utf-8"))
    ref_map = {p["page_name"]: p["source_file"] for p in ref["pages"]}
    mine = {k: v["img_id"] for k, v in H.items() if v["exists"]}
    iv("page_map_eq_ingest_borg_tonch", True, mine == ref_map, mine == ref_map)
    from pipeline.tools import ingest_borg_book as IB
    iv("adapter_khong_doc_cot_Nom", "SinoNom_Char ∉ XLSX_READ", list(IB.XLSX_READ), "SinoNom_Char" not in IB.XLSX_READ)
    man_p = AUTO / book / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else None
    iv("manifest_evaluation_only", True, None if man is None else man.get("evaluation_only"),
       None if man is None else man.get("evaluation_only") is True)
    kim = kim_block(book, H, E)
    iv("kim_cache_du_trang", human["n_pages"], kim.get("n_pages", 0),
       None if not kim.get("n_pages") else kim["n_pages"] == human["n_pages"])

    cells_out, err_rows = [], []
    tiers: dict[str, dict] = {}
    ge_blk = kim_cells = None
    if labels.exists():
        L = load_labels(labels)
        spans = load_spans(book)
        ge = load_gold_exact(book)
        s2c = {pg: syl_to_char(h) for pg, h in H.items()}
        s2s = {pg: page_syllables(h) for pg, h in H.items()}
        n_bad_syl = n_nospan = 0
        seen = set()
        for r in L:
            pg, col = r["page"], int(float(r["column"]))
            try:
                si = int(float(r["syl_idx"]))
            except (TypeError, ValueError):
                si = -1
            j0 = spans.get((pg, col))
            gt = None; syl_h = ""
            if j0 is None or si < 0:
                n_nospan += 1
            else:
                J = j0 + si
                chars = s2c.get(pg, [])
                if 0 <= J < len(chars):
                    gt = chars[J]; syl_h = s2s[pg][J]
                    if syl_h != r.get("syllable", "").lower():
                        n_bad_syl += 1
            uid = f"{book}/{r['book']}/{pg}/c{col}/n{r.get('nom_idx', '')}/s{r.get('syl_idx', '')}"
            seen.add(uid)
            lab = r.get("label", "")
            row = dict(cell_uid=uid, page=pg, column=col, syl_idx=si, syllable=r.get("syllable", ""), syl_human=syl_h,
                       tier=r.get("tier", ""), label=lab, ocr_char=r.get("ocr_char", ""), gt=gt or "",
                       gold_exact=ge.get(uid, ""))
            row["strict"] = E.strict(lab, gt); row["v1p"] = E.v1p(lab, gt)
            row["p4"] = E.pairs(lab, gt, PAIRS4); row["p11"] = E.pairs(lab, gt, PAIRS11)
            row["kim_strict"] = E.strict(row["ocr_char"], gt); row["kim_v1p"] = E.v1p(row["ocr_char"], gt)
            cells_out.append(row)
            if gt and lab and not row["p11"]:
                err_rows.append(row)
        for t in sorted({r["tier"] for r in cells_out}):
            tiers[t] = rate_block([r for r in cells_out if r["tier"] == t], "v1p")
        ok_rows = [r for r in cells_out if r["gold_exact"] == "ok"]
        ge_blk = dict(n_gold_exact_rows=len(ge), by_state=dict(Counter(ge.values())),
                      ok=rate_block(ok_rows, "v1p") if ok_rows else dict(n=0))
        kc = [dict(page=r["page"], gt=r["gt"], strict=r["kim_strict"], v1p=r["kim_v1p"],
                   p4=E.pairs(r["ocr_char"], r["gt"], PAIRS4), p11=E.pairs(r["ocr_char"], r["gt"], PAIRS11))
              for r in cells_out if r["ocr_char"] and r["tier"] in ("GOLD", "SILVER", "SYLLABLE", "REVIEW")]
        kim_cells = rate_block(kc, "v1p")
        iv("o_khoa_duy_nhat", len(cells_out), len(seen), len(seen) == len(cells_out))
        n_span = len(cells_out) - n_nospan
        iv("am_ghep_trung_am_nguoi", "≥ 0,99 ô có khoảng âm", round(1 - n_bad_syl / max(1, n_span), 4),
           (1 - n_bad_syl / max(1, n_span)) >= 0.99 if n_span else None)
        g = tiers.get("GOLD", {})
        iv("gold_co_gt_du", "≥ 100 ô GOLD có chữ người", g.get("n_eval", 0), g.get("n_eval", 0) >= 100 if g else None)
    else:
        iv("nhan_may_ton_tai", str(labels.relative_to(REPO) if labels.is_relative_to(REPO) else labels), False, None)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["cell_uid", "page", "column", "syl_idx", "syllable", "syl_human", "tier", "label", "ocr_char", "gt", "gold_exact",
            "strict", "v1p", "p4", "p11", "kim_strict", "kim_v1p"]
    for name, rows in (("cells.csv", cells_out), ("errors.csv", err_rows)):
        with open(out_dir / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    summ = dict(book=book, labels=str(labels), api_calls=0, human=human, kim_vs_human_page=kim, tiers=tiers,
                gold_exact=ge_blk, kim_at_cells=kim_cells, pairs4=PAIRS4, stt_estimate_suy_doan=STT_EST,
                invariants=[dict(name=i["name"], expected=i["expected"], observed=i["observed"], **{"pass": i["pass_"]})
                            for i in inv], seconds=round(time.time() - t0, 1))
    (out_dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return summ


def selftest() -> int:
    n = ok = 0

    def chk(name, cond):
        nonlocal n, ok
        n += 1; ok += bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'} {name}")
    E = Eq()
    chk("strict", E.strict("先", "先") and not E.strict("先", "𠸜"))
    chk("cặp quy ước 𠸜/先 chỉ ở p4", E.pairs("先", "𠸜", PAIRS4) and not E.v1p("先", "𠸜"))
    chk("cặp 𠰺/代 chỉ ở p11", E.pairs("代", "𠰺", PAIRS11) and not E.pairs("代", "𠰺", PAIRS4))
    chk("V1+ biến thể phồn/giản", E.v1p("國", "国"))
    pg = dict(sents=[dict(nom="天主", syl=["đức", "chúa"], eq=True), dict(nom="吧", syl=["ba", "ngôi"], eq=False)])
    chk("âm→chữ: câu không đếm bằng = None", syl_to_char(pg) == ["天", "主", None, None])
    chk("LCS: 1 thay 1 chèn", levenshtein_pairs(list("ABCD"), list("AXCDE"), lambda a, b: a == b) == [(0, 0), (2, 2), (3, 3)])
    chk("LCS: chèn đầu + xoá giữa = 4 khớp", len(levenshtein_pairs(list("BCDEF"), list("ABDEF"), lambda a, b: a == b)) == 4)
    import random
    rnd = random.Random(1)
    for _ in range(30):
        x = [rnd.choice("ABCD") for _ in range(rnd.randint(1, 9))]; y = [rnd.choice("ABCD") for _ in range(rnd.randint(1, 9))]
        k = len(levenshtein_pairs(x, y, lambda a, b: a == b))
        # cận: số cặp khớp ≤ min(len) và cặp đơn điệu tăng
        pr = levenshtein_pairs(x, y, lambda a, b: a == b)
        L = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
        for i_ in range(len(x)):
            for j_ in range(len(y)):
                L[i_ + 1][j_ + 1] = L[i_][j_] + 1 if x[i_] == y[j_] else max(L[i_][j_ + 1], L[i_ + 1][j_])
        if k != L[-1][-1] or any(pr[t][0] >= pr[t + 1][0] or pr[t][1] >= pr[t + 1][1] for t in range(len(pr) - 1)):
            chk("LCS: ngẫu nhiên = LCS kinh điển + đơn điệu", False); break
    else:
        chk("LCS: ngẫu nhiên = LCS kinh điển + đơn điệu", True)
    rows = [dict(page="p1", gt="天", strict=True, v1p=True, p4=True, p11=True),
            dict(page="p2", gt="主", strict=False, v1p=False, p4=False, p11=False),
            dict(page="p2", gt="", strict=False, v1p=False, p4=False, p11=False)]
    b = rate_block(rows, "v1p")
    chk("khối tỉ lệ: n_eval bỏ ô không GT", b["n_eval"] == 2 and b["strict"]["pt"] == 0.5)
    lo, hi = boot_ci({"a": 9, "b": 10}, {"a": 10, "b": 10})
    chk("CI cụm trang trong [0,1] và chứa điểm", 0 <= lo <= 0.95 <= hi <= 1)
    print(f"borg_endtoend_eval selftest: {ok}/{n}")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    global AUTO
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--book", default="all", choices=["all", *BOOKS])
    ap.add_argument("--labels", default=None, help="labels_gated.csv (mặc định prepared/_auto/<Sách>/dataset_out/labels_gated.csv)")
    ap.add_argument("--out", default=str(REPO / "measure_out"))
    ap.add_argument("--auto-root", default=str(AUTO), help="gốc prepared của đường tự động (mặc định prepared/_auto)")
    ap.add_argument("--limit", type=int, default=0, help="(không dùng — giữ cho measure.py)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    books = BOOKS if a.book == "all" else (a.book,)
    AUTO = Path(a.auto_root)
    E = Eq()
    rc = 0
    for b in books:
        lab = Path(a.labels) if a.labels else AUTO / b / "dataset_out" / "labels_gated.csv"
        s = evaluate(b, lab, Path(a.out) / b / "borg_endtoend", E)
        h, k, g = s["human"], s["kim_vs_human_page"], s["tiers"].get("GOLD", {})
        fails = [i["name"] for i in s["invariants"] if i["pass"] is False]
        skips = [i["name"] for i in s["invariants"] if i["pass"] is None]
        print(f"[borg_eval] {b}: người {h['n_pages']} trang, câu đếm bằng {h['n_sent_eq']}/{h['n_sentences']} "
              f"({h['frac_syll_eq']:.1%} âm); chữ người ∈ R(âm) {h['pair_char_in_R']:.1%} (biến thể {h['pair_char_in_R_v1']:.1%})")
        if k.get("n_pages"):
            print(f"  kim thô vs người ({k['n_pages']} trang): precision strict {k['strict']['precision']:.4f} · V1+ "
                  f"{k['v1p']['precision']:.4f} · +4 cặp {k['p4']['precision']:.4f} · recall V1+ {k['v1p']['recall']:.4f}")
        else:
            print("  kim thô: CHƯA có cache kim (prepared/_auto/<Sách>/kim_raw) — SKIP")
        if g.get("n_eval"):
            print(f"  GOLD có chữ người {g['n_eval']}: strict {g['strict']['pt']:.4f} · V1+ {g['v1p']['pt']:.4f} "
                  f"{g['v1p']['ci_page']} · +4 cặp {g['p4']['pt']:.4f}")
        ok_ = (s.get("gold_exact") or {}).get("ok") or {}
        if ok_.get("n_eval"):
            print(f"  gold_exact=ok có chữ người {ok_['n_eval']}: V1+ {ok_['v1p']['pt']:.4f} {ok_['v1p']['ci_page']}")
        print(f"  bất biến: {len(s['invariants']) - len(fails) - len(skips)} PASS / {len(fails)} FAIL / {len(skips)} SKIP"
              + (f" — FAIL {fails}" if fails else ""))
        rc |= 1 if fails else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
