#!/usr/bin/env python3
"""ihr_endtoend_eval.py — ĐỘ ĐÚNG END-TO-END THẬT của pipeline trên 2 bộ IHR-NomDB có NHÃN NGƯỜI.

Đây là phép đo DUY NHẤT trong dự án so nhãn máy với **nhãn người từng chữ** trên toàn sách:
`data/<book>/manifest.tsv` (cột `nom_text`, IHR-NomDB Nov 2020) ↔ `labels_gated.csv` do pipeline sinh,
ghép theo (trang, cột, VỊ TRÍ CHỮ). Mọi phép đo "precision" khác trong dự án là proxy (khớp dị bản,
kim ∈ R(âm)…). Adapter ingest KHÔNG đọc `nom_text`, nên phép đo này KHÔNG vòng tròn.

Ghép khoá:
    labels.csv  page = page_XXXX      -> page_id gốc qua prepared_ihr/<book>/manifest.json
    labels.csv  column = 1..10 (PHẢI→TRÁI) -> manifest.tsv col_index = column − 1
    labels.csv  syl_idx = 0..13       -> part = 1 (0..5, câu lục) | 2 (6..13, câu bát); vị trí trong câu
So sánh `label` (chữ pipeline gán) với chữ GT cùng vị trí. Phân loại lỗi:
    dung          label == GT
    di_the        khác chữ nhưng CẢ HAI ∈ R(âm QN của ô) — dị thể đồng âm, không phải lỗi đọc
    gan_hinh      khác chữ, gần hình theo Dict/SinoNom_Similar.csv
    gt_pua        chữ GT nằm vùng PUA (U+E000..U+F8FF, U+F0000+) — GT chưa có mã Unicode chuẩn
    khac_han      còn lại

Chạy:
    .venv/bin/python scripts/measure/ihr_endtoend_eval.py --book all
    .venv/bin/python scripts/measure/ihr_endtoend_eval.py --book LucVanTien1916 \\
        --labels prepared_ihr/LucVanTien1916/dataset_out/labels_gated.csv
    .venv/bin/python scripts/measure/ihr_endtoend_eval.py --selftest
Đầu ra: measure_out/<book>/ihr_endtoend/{summary.json, cells.csv, errors.csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BOOKS = {
    "LucVanTien1916": dict(prepared="prepared_ihr/LucVanTien1916",
                           labels="prepared_ihr/LucVanTien1916/dataset_out/labels_gated.csv"),
    "TruyenKieu1872": dict(prepared="prepared_ihr/TruyenKieu1872",
                           labels="prepared_ihr/TruyenKieu1872/dataset_out/labels_gated.csv"),
}
TIER_RULE = (6, 8)
N_PER_COL = sum(TIER_RULE)
# Mốc cũ: precision GOLD đo GIÁN TIẾP trên 200 patch/sách (kim ×3, cắt cột đúng, KHÔNG chạy
# pipeline) — docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md §(1).
BASELINE_PATCH = {"LucVanTien1916": dict(precision=0.893, coverage=0.551, n=768),
                  "TruyenKieu1872": dict(precision=0.846, coverage=0.495, n=689)}


def is_pua(ch: str) -> bool:
    """Chữ trong vùng dùng riêng (GT IHR ghi chữ chưa có mã chuẩn bằng PUA)."""
    o = ord(ch)
    return 0xE000 <= o <= 0xF8FF or 0xF0000 <= o <= 0x10FFFD


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson 95 % cho tỉ lệ k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


def load_gt(book: str) -> dict[tuple[str, int, int], str]:
    """manifest.tsv → {(page_id, col_index, part): chuỗi chữ Nôm GT}."""
    p = REPO / "data" / book / "manifest.tsv"
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
        if r["col_index"] == "" or r["part"] == "":
            continue
        out[(r["page_id"], int(r["col_index"]), int(r["part"]))] = r["nom_text"]
    return out


def load_page_map(prepared: Path) -> dict[str, str]:
    """prepared_ihr/<book>/manifest.json → {page_XXXX: page_id gốc}."""
    m = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    return {p["page_name"]: p["page_id"] for p in m.get("pages", [])}


def dicts():
    """(R: âm chuẩn hoá → tập chữ, SIM: chữ → tập chữ gần hình)."""
    global _D
    try:
        return _D
    except NameError:
        pass
    from core.text.dictionary import load_qn_to_nom, load_similarity_dict
    from core.text.text_utils import normalize_tone_marks
    q2n = load_qn_to_nom(str(REPO / "Dict/QuocNgu_SinoNom.csv"))
    R = {}
    for k, v in q2n.items():
        R.setdefault(normalize_tone_marks(k.lower()), set()).update(v)
    SIM = {k: set(v) for k, v in load_similarity_dict(str(REPO / "Dict/SinoNom_Similar.csv")).items()}
    _D = (R, SIM)
    return _D


def classify(label: str, gt: str, syl: str, R: dict, SIM: dict) -> str:
    """Nhãn máy vs chữ người: đúng / dị thể / gần hình / GT PUA / khác hẳn."""
    from core.text.text_utils import normalize_tone_marks
    if not gt:
        return "khong_co_gt"
    if not label:
        return "khong_co_nhan"
    if label == gt:
        return "dung"
    rs = R.get(normalize_tone_marks((syl or "").lower()), set())
    if label in rs and gt in rs:
        return "di_the"
    if gt in SIM.get(label, ()) or label in SIM.get(gt, ()):
        return "gan_hinh"
    if is_pua(gt):
        return "gt_pua"
    return "khac_han"


def evaluate(book: str, labels_csv: Path, prepared: Path, out: Path) -> dict:
    R, SIM = dicts()
    gt = load_gt(book)
    pmap = load_page_map(prepared)
    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8")))
    per_tier: dict[str, dict] = {}
    cells, errs = [], []
    n_no_gt = 0
    gt_cols = {k for k in gt}
    for r in rows:
        page_id = pmap.get(r["page"], "")
        try:
            col = int(r["column"]) - 1
            si = int(r["syl_idx"])
        except (TypeError, ValueError):
            continue
        part = 1 if si < TIER_RULE[0] else 2
        pos = si if part == 1 else si - TIER_RULE[0]
        s = gt.get((page_id, col, part), "")
        g = s[pos] if 0 <= pos < len(s) else ""
        if not g:
            n_no_gt += 1
        cls = classify(r.get("label", ""), g, r.get("syllable", ""), R, SIM)
        tier = r.get("tier") or "?"
        t = per_tier.setdefault(tier, dict(n=0, with_gt=0, **{k: 0 for k in
                                ("dung", "di_the", "gan_hinh", "gt_pua", "khac_han",
                                 "khong_co_gt", "khong_co_nhan")}))
        t["n"] += 1
        t[cls] += 1
        if g:
            t["with_gt"] += 1
        cells.append(dict(book=book, page=r["page"], page_id=page_id, column=r["column"],
                          syl_idx=si, part=part, pos=pos, syllable=r.get("syllable", ""),
                          label=r.get("label", ""), gt=g, tier=tier, rule=r.get("rule", ""),
                          ocr_char=r.get("ocr_char", ""), ket_qua=cls))
        if cls not in ("dung", "khong_co_gt"):
            errs.append(cells[-1])

    # ô LÝ THUYẾT = số chữ trong nhãn người. `manifest.tsv` KHÔNG có `col_index` cho một số trang
    # (IHR thiếu patch): các trang ấy không có GT nào, nên coverage phải tính trên TRANG CÓ GT,
    # còn ô của trang không có GT đếm riêng (không phải lỗi ghép).
    n_cols_gt = len({(p, c) for (p, c, _) in gt_cols})
    n_theory = sum(len(v) for v in gt.values())
    produced = len(rows)
    pages_with_gt = {p for (p, _, _) in gt_cols}
    n_no_gt_page = sum(1 for c in cells if c["page_id"] not in pages_with_gt)
    produced_on_gt_pages = sum(1 for c in cells if c["page_id"] in pages_with_gt)
    n_no_gt_pos = n_no_gt - n_no_gt_page          # ô ở trang CÓ GT mà vẫn không ghép được vị trí

    def block(names: list[str]) -> dict:
        d = dict(n=0, with_gt=0, dung=0, di_the=0, gan_hinh=0, gt_pua=0, khac_han=0,
                 khong_co_nhan=0, khong_co_gt=0)
        for nm in names:
            for k, v in (per_tier.get(nm) or {}).items():
                d[k] = d.get(k, 0) + v
        n = d["with_gt"]
        d["precision"] = round(d["dung"] / n, 4) if n else None
        lo, hi = wilson(d["dung"], n)
        d["ci95"] = [round(lo, 4), round(hi, 4)]
        n2 = n - d["gt_pua"]
        d["precision_bo_pua"] = round(d["dung"] / n2, 4) if n2 > 0 else None
        n3 = n - d["gt_pua"] - d["di_the"]
        d["precision_bo_pua_dithe"] = round(d["dung"] / n3, 4) if n3 > 0 else None
        return d

    gold_img = block(["GOLD"])
    gold_all = block(["GOLD", "GOLD_text_only"])
    silver = block(["SILVER"])
    syl = block(["SYLLABLE"])
    review = block(["REVIEW"])
    allb = block(list(per_tier))

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)
    with open(out / "errors.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(errs)

    # kim THÔ so GT (không qua luật nào) + nhãn có bằng kim không -> luật từ điển là LỌC hay SỬA?
    wg = [c for c in cells if c["gt"]]
    kim_all = sum(1 for c in wg if c["ocr_char"] == c["gt"])
    gcells = [c for c in wg if c["tier"] in ("GOLD", "GOLD_text_only")]
    kim_gold = sum(1 for c in gcells if c["ocr_char"] == c["gt"])
    lab_eq_kim = sum(1 for c in gcells if c["label"] == c["ocr_char"])
    lo_a, hi_a = wilson(kim_all, len(wg))
    kim_raw = dict(n=len(wg), dung=kim_all,
                   precision=round(kim_all / len(wg), 4) if wg else None,
                   ci95=[round(lo_a, 4), round(hi_a, 4)],
                   n_gold=len(gcells), dung_gold=kim_gold,
                   precision_tren_o_GOLD=round(kim_gold / len(gcells), 4) if gcells else None,
                   nhan_bang_kim_tren_GOLD=round(lab_eq_kim / len(gcells), 4) if gcells else None)

    base = BASELINE_PATCH.get(book) or {}
    s = dict(
        book=book, generated_by="scripts/measure/ihr_endtoend_eval.py",
        labels=str(labels_csv), prepared=str(prepared),
        gt=dict(file=f"data/{book}/manifest.tsv", n_verses=len(gt), n_columns=n_cols_gt,
                n_chars=n_theory,
                n_chars_pua=sum(1 for v in gt.values() for ch in v if is_pua(ch))),
        cells=dict(produced=produced, theory_from_gt=n_theory,
                   produced_on_gt_pages=produced_on_gt_pages,
                   coverage_cells=round(produced_on_gt_pages / n_theory, 4) if n_theory else None,
                   coverage_cells_all=round(produced / n_theory, 4) if n_theory else None,
                   no_gt=n_no_gt, no_gt_page=n_no_gt_page, no_gt_position=n_no_gt_pos,
                   n_pages_with_gt=len(pages_with_gt),
                   n_pages_without_gt=len({c["page_id"] for c in cells} - pages_with_gt)),
        tiers={k: v for k, v in sorted(per_tier.items())},
        kim_raw=kim_raw,
        precision=dict(gold_anh=gold_img, gold_tat_ca=gold_all, silver=silver,
                       syllable=syl, review=review, toan_bo=allb),
        coverage=dict(gold_anh=round(gold_img["with_gt"] / n_theory, 4) if n_theory else None,
                      gold_tat_ca=round(gold_all["with_gt"] / n_theory, 4) if n_theory else None),
        baseline_patch_2026_09_22=base,
        delta_vs_baseline=dict(
            precision=round((gold_all["precision"] or 0) - base.get("precision", 0), 4) if base else None,
            coverage=round((gold_all["with_gt"] / n_theory) - base.get("coverage", 0), 4)
            if base and n_theory else None),
    )
    s["invariants"] = invariants(s)
    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return s


def invariants(s: dict) -> list[dict]:
    iv = []

    def add(name, expected, observed, ok, method):
        iv.append(dict(name=name, expected=expected, observed=observed, method=method, **{"pass": ok}))
    c = s["cells"]
    add("o_trong_trang_CO_GT_deu_ghep_duoc", "<= 2 %",
        round(c["no_gt_position"] / max(1, c["produced_on_gt_pages"]), 4),
        c["no_gt_position"] / max(1, c["produced_on_gt_pages"]) <= 0.02,
        "ô nằm trong trang CÓ nhãn người mà vẫn không ghép được vị trí (trang không có nhãn người "
        "-- manifest.tsv thiếu col_index -- được đếm riêng ở no_gt_page, không tính là lỗi ghép)")
    add("khong_sinh_thua_o", "<= 1,02", c["coverage_cells"],
        c["coverage_cells"] is not None and c["coverage_cells"] <= 1.02,
        "ô sinh trên TRANG CÓ GT / số chữ GT — > 1 nghĩa là ghép lệch cột (cho phép 2 % vì cột QN "
        "đếm lệch có thể sinh 13 hoặc 15 ô)")
    g = s["precision"]["gold_anh"]
    add("precision_GOLD_anh_cao_hon_REVIEW", "GOLD > REVIEW",
        [g["precision"], s["precision"]["review"]["precision"]],
        (g["precision"] or 0) > (s["precision"]["review"]["precision"] or 0),
        "luật GOLD phải phân biệt được đúng/sai, nếu không thì tầng nhãn vô nghĩa")
    add("precision_GOLD_anh_>=_0.90", ">= 0.90", g["precision"], (g["precision"] or 0) >= 0.90,
        "nhãn GOLD có ảnh so chữ người, toàn sách")
    k = s.get("kim_raw") or {}
    add("luat_GOLD_la_BO_LOC_khong_phai_sua", "nhãn == kim ở 100 % ô GOLD",
        k.get("nhan_bang_kim_tren_GOLD"), k.get("nhan_bang_kim_tren_GOLD") == 1.0,
        "luật s1_inter_s2_direct lấy CHÍNH chữ kim khi kim ∈ R(âm); nếu < 1 thì có bước sửa chữ")
    add("loc_tu_dien_lam_tang_do_dung", "precision GOLD > kim thô",
        [s["precision"]["gold_tat_ca"]["precision"], k.get("precision")],
        (s["precision"]["gold_tat_ca"]["precision"] or 0) > (k.get("precision") or 0),
        "so độ đúng của kim TRÊN Ô ĐƯỢC NHẬN với độ đúng của kim trên MỌI ô")
    b = s.get("baseline_patch_2026_09_22") or {}
    if b:
        add("khong_kem_hon_moc_patch_cu", f">= {b['precision']}",
            s["precision"]["gold_tat_ca"]["precision"],
            (s["precision"]["gold_tat_ca"]["precision"] or 0) >= b["precision"] - 0.02,
            "so với mốc đo gián tiếp 200 patch/sách (22/09): kim ×3, cắt cột đúng, không chạy pipeline")
    return iv


def selftest() -> int:
    n = ok = 0

    def chk(name, cond):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'} {name}")

    chk("is_pua U+E000", is_pua(""))
    chk("is_pua U+F8FF", is_pua(""))
    chk("is_pua chữ Hán thường = False", not is_pua("人"))
    chk("is_pua ngoài BMP (U+F0000)", is_pua(chr(0xF0000)))
    chk("is_pua U+24F93 (Nôm thật) = False", not is_pua(chr(0x24F93)))
    lo, hi = wilson(90, 100)
    chk("wilson chứa tỉ lệ quan sát", lo < 0.9 < hi)
    chk("wilson n=0 an toàn", wilson(0, 0) == (0.0, 0.0))
    chk("wilson hẹp dần khi n lớn",
        (wilson(900, 1000)[1] - wilson(900, 1000)[0]) < (wilson(90, 100)[1] - wilson(90, 100)[0]))
    R = {"ta": {"些", "他"}, "nguoi": {"𠊛"}}
    SIM = {"些": {"柴"}}
    chk("classify: đúng", classify("些", "些", "ta", R, SIM) == "dung")
    chk("classify: dị thể (cùng âm, đều trong R)", classify("些", "他", "ta", R, SIM) == "di_the")
    chk("classify: gần hình", classify("些", "柴", "xxx", R, SIM) == "gan_hinh")
    chk("classify: GT là PUA", classify("些", "", "xxx", R, SIM) == "gt_pua")
    chk("classify: khác hẳn", classify("些", "人", "xxx", R, SIM) == "khac_han")
    chk("classify: không có GT", classify("些", "", "ta", R, SIM) == "khong_co_gt")
    chk("classify: không có nhãn", classify("", "些", "ta", R, SIM) == "khong_co_nhan")
    chk("classify: dị thể cần CẢ HAI trong R", classify("些", "人", "ta", R, SIM) != "di_the")
    for bk in BOOKS:
        g = load_gt(bk)
        chk(f"{bk}: GT > 1.900 câu", len(g) > 1900)
        chk(f"{bk}: mọi khoá có part 1|2", all(p in (1, 2) for (_, _, p) in g))
        chk(f"{bk}: câu lục 6 chữ chiếm đa số",
            sum(1 for k, v in g.items() if k[2] == 1 and len(v) == 6) > 0.9 * sum(1 for k in g if k[2] == 1))
        chk(f"{bk}: mốc patch cũ có sẵn", bk in BASELINE_PATCH)
    print(f"ihr_endtoend selftest: {ok}/{n}")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default="all")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    books = list(BOOKS) if a.book == "all" else [b for b in a.book.split(",") if b]
    bad = [b for b in books if b not in BOOKS]
    if bad:
        print(f"sách không biết: {bad}; chọn trong {list(BOOKS)}")
        return 2
    rc = 0
    for b in books:
        labels = Path(a.labels) if a.labels else REPO / BOOKS[b]["labels"]
        prep = Path(a.prepared) if a.prepared else REPO / BOOKS[b]["prepared"]
        outd = Path(a.out) if a.out else REPO / "measure_out" / b / "ihr_endtoend"
        if not labels.exists():
            print(f"{b}: THIẾU {labels} — chạy ./run_pipeline.sh --book {b} trước")
            rc = 2
            continue
        s = evaluate(b, labels, prep, outd)
        g, ga = s["precision"]["gold_anh"], s["precision"]["gold_tat_ca"]
        c0 = s["cells"]
        print(f"\n== {b} ==  ô sinh {c0['produced']} (trên trang CÓ GT: {c0['produced_on_gt_pages']}) / "
              f"GT {s['gt']['n_chars']} → coverage ô {c0['coverage_cells']}; "
              f"{c0['n_pages_without_gt']} trang không có nhãn người ({c0['no_gt_page']} ô)")
        print(f"  GOLD có ảnh : n {g['with_gt']:6d} · ĐÚNG {g['precision']} CI{g['ci95']} · "
              f"bỏ GT PUA {g['precision_bo_pua']} · bỏ PUA+dị thể {g['precision_bo_pua_dithe']}")
        print(f"  GOLD (kể text_only): n {ga['with_gt']:6d} · ĐÚNG {ga['precision']} CI{ga['ci95']} · "
              f"coverage {s['coverage']['gold_tat_ca']}")
        for nm in ("SYLLABLE", "REVIEW"):
            d = s["precision"][nm.lower()]
            print(f"  {nm:9s}: n {d['with_gt']:6d} · ĐÚNG {d['precision']}")
        print(f"  lỗi GOLD có ảnh: dị thể {g['di_the']} · gần hình {g['gan_hinh']} · "
              f"GT PUA {g['gt_pua']} · khác hẳn {g['khac_han']}")
        k = s["kim_raw"]
        print(f"  kim THÔ so GT (mọi ô)      : {k['precision']} CI{k['ci95']} (n {k['n']}) · "
              f"trên ô GOLD {k['precision_tren_o_GOLD']} · nhãn==kim ở ô GOLD {k['nhan_bang_kim_tren_GOLD']}")
        b0 = s["baseline_patch_2026_09_22"]
        if b0:
            print(f"  mốc patch 22/09: precision {b0['precision']} coverage {b0['coverage']} (n {b0['n']}) "
                  f"→ chênh {s['delta_vs_baseline']}")
        for iv in s["invariants"]:
            print(f"  {'PASS' if iv['pass'] else 'FAIL'} {iv['name']} kỳ vọng {iv['expected']} quan sát {iv['observed']}")
        rc = rc or (1 if any(iv["pass"] is False for iv in s["invariants"]) else 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
