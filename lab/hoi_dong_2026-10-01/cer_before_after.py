#!/usr/bin/env python3
"""CER/precision/recall TRƯỚC (kim thô cả trang) và SAU (nhãn pipeline) so chữ người — 0 API, chỉ đọc.

Sách: LucVanTien1916, TruyenKieu1872 (IHR, manifest.tsv), SachKinhThayCaBinh, SachDungLyHoThan (Borg, xlsx).
Chuỗi so sánh theo TRANG (thứ tự đọc cột phải→trái, trên→dưới):
  kim_page   : chữ kim thô của trang (cache kim_raw *_lt2.json, cluster_columns của adapter Borg) — TRƯỚC pipeline
  kim_cells  : ocr_char của mọi ô pipeline (kim sau khi gán vào ô) — trung gian
  lab_all    : nhãn pipeline mọi tầng có nhãn (GOLD, text_only, REVIEW có nhãn ...) — SAU, đầu ra đầy đủ
  lab_gold   : chỉ nhãn GOLD + GOLD_text_only — SAU, tập công bố
Đại lượng: Levenshtein chuẩn (thay = 1) d ⇒ CER = Σd / Σ|người|; LCS (helper levenshtein_pairs) ⇒ P = khớp/|máy|, R = khớp/|người|.
Hai phép bằng: strict (trùng hẳn) và V1+ (biến thể, pipeline.gold_exact.common.var_eq_plus).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/measure"))
import borg_endtoend_eval as BE  # noqa: E402
import ihr_endtoend_eval as IE  # noqa: E402
from pipeline.tools.ingest_borg_book import cluster_columns  # noqa: E402

_MEMO: dict = {}


def lev(a: list[str], b: list[str], eq) -> tuple[int, int, int, int]:
    """Levenshtein chuẩn (chèn/xoá/thay = 1, khớp theo eq) -> (d, S, D, I) với a = máy, b = người.
    D = chữ người bị thiếu (xoá), I = chữ máy thừa (chèn)."""
    n, m = len(a), len(b)
    if not n:
        return m, 0, m, 0
    if not m:
        return n, 0, 0, n
    ua = sorted(set(a)); ub = sorted(set(b))
    pa = {x: k for k, x in enumerate(ua)}; pb = {y: k for k, y in enumerate(ub)}
    Eu = np.ones((len(ua), len(ub)), np.int32)
    for p_, x in enumerate(ua):
        for q_, y in enumerate(ub):
            k = (id(eq), x, y)
            v = _MEMO.get(k)
            if v is None:
                v = _MEMO[k] = 0 if eq(x, y) else 1
            Eu[p_, q_] = v
    E = Eu[np.array([pa[x] for x in a])][:, np.array([pb[y] for y in b])]
    Dm = np.zeros((n + 1, m + 1), np.int32)
    Dm[:, 0] = np.arange(n + 1); Dm[0, :] = np.arange(m + 1)
    js = np.arange(m + 1)
    for i in range(1, n + 1):
        best = np.minimum(Dm[i - 1, :-1] + E[i - 1], Dm[i - 1, 1:] + 1)
        bb = np.concatenate(([Dm[i, 0]], best))
        Dm[i] = np.minimum.accumulate(bb - js) + js
    S = De = I = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and Dm[i, j] == Dm[i - 1, j - 1] + E[i - 1, j - 1]:
            S += int(E[i - 1, j - 1]); i -= 1; j -= 1
        elif i > 0 and Dm[i, j] == Dm[i - 1, j] + 1:
            I += 1; i -= 1          # chữ máy thừa
        else:
            De += 1; j -= 1         # chữ người thiếu
    return int(Dm[n, m]), S, De, I


FS: dict = {}   # comparator BỀN (giữ sống suốt tiến trình) -> id(eq) duy nhất, không tái dùng giữa các lần gọi


def score(pages: dict[str, tuple[list[str], list[str]]], E) -> dict:
    """pages: {page: (máy, người)} -> chỉ số tổng (vi mô) strict + V1+."""
    if not FS:
        FS["strict"] = E.strict; FS["v1p"] = E.v1p
    out = dict(n_pages=len(pages), n_may=sum(len(a) for a, _ in pages.values()),
               n_nguoi=sum(len(b) for _, b in pages.values()))
    for name, f in (("strict", FS["strict"]), ("v1p", FS["v1p"])):
        d = S = De = I = mt = 0
        for a, b in pages.values():
            x = lev(a, b, f); d += x[0]; S += x[1]; De += x[2]; I += x[3]
            mt += len(BE.levenshtein_pairs(a, b, f)) if a and b else 0
        out[name] = dict(CER=round(d / max(1, out["n_nguoi"]), 4), S=S, D=De, I=I,
                         lcs_match=mt, P=round(mt / max(1, out["n_may"]), 4), R=round(mt / max(1, out["n_nguoi"]), 4))
    assert out["v1p"]["lcs_match"] >= out["strict"]["lcs_match"], out
    assert out["v1p"]["CER"] <= out["strict"]["CER"], out
    return out


def seqs_from_labels(rows, keep) -> dict[str, list[str]]:
    cols = defaultdict(list)
    for r in rows:
        try:
            k = (r["page"], int(r["column"]), int(r["syl_idx"]))
        except (TypeError, ValueError):
            continue
        cols[r["page"]].append((k, r))
    out = {}
    for pg, lst in cols.items():
        lst.sort(key=lambda t: t[0])
        out[pg] = [ch for _, r in lst for ch in keep(r)]
    return out


def kim_chars_from(path: Path) -> list[str]:
    d = json.loads(path.read_text(encoding="utf-8"))
    cols, _ = cluster_columns(d.get("boxes") or [])
    return [c["char"] for col in cols for c in col]


def run_ihr(book: str, E) -> dict:
    prep = REPO / "prepared" / book
    gt = IE.load_gt(book)
    pmap = IE.load_page_map(prep)
    by_page = defaultdict(list)
    for (pid, col, part), s in gt.items():
        by_page[pid].append(((col, part), s))
    human = {pid: [ch for _, s in sorted(v) for ch in s] for pid, v in by_page.items()}
    rows = list(csv.DictReader(open(prep / "dataset_out/labels_gated.csv", encoding="utf-8")))
    return assemble(book, E, human, pmap, rows, lambda pg: prep / "kim_raw" / f"{pg}_lt2.json")


def run_borg(book: str, E) -> dict:
    H = BE.human_pages(book)
    human = {pg: [c for s in h["sents"] for c in s["nom"]] for pg, h in H.items()}
    pmap = {pg: pg for pg in H}
    rows = BE.load_labels(BE.AUTO / book / "dataset_out/labels_gated.csv")
    sfx = BE._kim_sfx(str(BE.AUTO), book)
    return assemble(book, E, human, pmap, rows, lambda pg: BE.AUTO / book / "kim_raw" / f"{pg}{sfx}.json")


def assemble(book, E, human, pmap, rows, kim_path) -> dict:
    gold = ("GOLD", "GOLD_text_only")
    S_kc = seqs_from_labels(rows, lambda r: [r["ocr_char"]] if r.get("ocr_char") else [])
    S_all = seqs_from_labels(rows, lambda r: [r["label"]] if r.get("label") else [])
    S_gold = seqs_from_labels(rows, lambda r: [r["label"]] if r.get("label") and r.get("tier") in gold else [])
    common = []
    for pg, pid in pmap.items():
        if pid in human and kim_path(pg).exists():
            common.append(pg)
    res = dict(book=book, pages_common=len(common))
    res["kim_page"] = score({pg: (kim_chars_from(kim_path(pg)), human[pmap[pg]]) for pg in common}, E)
    res["kim_cells"] = score({pg: (S_kc.get(pg, []), human[pmap[pg]]) for pg in common}, E)
    res["lab_all"] = score({pg: (S_all.get(pg, []), human[pmap[pg]]) for pg in common}, E)
    res["lab_gold"] = score({pg: (S_gold.get(pg, []), human[pmap[pg]]) for pg in common}, E)
    return res


def main():
    E = BE.Eq()
    out = {}
    for b in ("LucVanTien1916", "TruyenKieu1872"):
        out[b] = run_ihr(b, E); print(json.dumps(out[b], ensure_ascii=False), flush=True)
    for b in ("SachKinhThayCaBinh", "SachDungLyHoThan"):
        out[b] = run_borg(b, E); print(json.dumps(out[b], ensure_ascii=False), flush=True)
    p = Path(__file__).with_name("cer_before_after.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", p)


if __name__ == "__main__":
    main()
