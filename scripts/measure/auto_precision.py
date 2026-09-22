#!/usr/bin/env python
"""auto_precision.py — độ đúng của nhãn máy KHÔNG cần người kiểm (2026-09-22).

Ba phép đo tái lập được, ghi vào measure_out/auto_precision/:

  ihr    (1) GT độc lập (mộc bản IHR-NomDB): lấy mẫu phân tầng theo trang các patch cột
             LucVanTien1916 / TruyenKieu1872 (chỉ patch len(nom_text)==n_qn_syll), OCR kim
             (upload+recognize của core.ocr.ocr_api; phóng --scale vì kim đọc thiếu ở 34 px),
             áp luật GOLD/SYLLABLE-1/REVIEW tối thiểu của pipeline (S1 = chữ kim, S2 = tập đọc
             âm từ điển Dict/QuocNgu_SinoNom.csv chuẩn hoá như core.text.dictionary) rồi so với
             nom_text từng vị trí. Kết quả kim được cache theo md5 patch → chạy lại 0 lần gọi.
  cross  (2) Đối chứng chéo dị bản trên thạch bản (0 API): KVK1884 ↔ Nôm 1871 (Liễu Văn Đường)
             [+ 1872 Duy Minh Thị], LVT1883 ↔ Nôm 1916 (Nôm Foundation). Câu QN OCR của sách
             thạch bản khớp câu tham chiếu (exact / ≥75 % âm cùng vị trí, duy nhất) → so nhãn
             GOLD của pipeline với chữ Nôm tham chiếu ở các vị trí âm giống hệt.
  gates  (3) Cổng máy trên labels_final.csv: n_det≠N, M≠N, count_source conflict, box_source,
             luật cầu (kim ∉ dict), p_register, crop_quality_flag → GOLD bị hạ + precision-proxy
             (từ (2)) trước/sau; chọn tổ hợp có proxy cao nhất với coverage ≥ --min-coverage.

Chạy:  .venv/bin/python scripts/measure/auto_precision.py --steps cross,gates      # 0 API
       .venv/bin/python scripts/measure/auto_precision.py --steps ihr --budget 420  # có API (cache)
       .venv/bin/python scripts/measure/auto_precision.py --all --report-only
Không sửa pipeline/ core/ data/; chỉ đọc. Mã thoát 0 = xong, 1 = có bước lỗi.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from core.text.dictionary import load_qn_to_nom, load_similarity_dict  # noqa: E402
from core.text.text_utils import (clean_line_text, fold_text, normalize_tone_marks,  # noqa: E402
                                  split_to_syllables)

OUT_DEFAULT = REPO / "measure_out" / "auto_precision"
DICT_QN = REPO / "Dict" / "QuocNgu_SinoNom.csv"
DICT_SIM = REPO / "Dict" / "SinoNom_Similar.csv"

IHR_BOOKS = {
    "LucVanTien1916": REPO / "data/LucVanTien1916/manifest.tsv",
    "TruyenKieu1872": REPO / "data/TruyenKieu1872/manifest.tsv",
}
# sách thạch bản → (labels_final, transcriptions dir, [(tên tham chiếu, json)])
CROSS_BOOKS = {
    "KimVanKieu1884": dict(
        labels=REPO / "dataset_out_KimVanKieu1884/labels_final.csv",
        trans=REPO / "prepared/KimVanKieu1884/transcriptions",
        refs=[("Kieu1871_LVD", REPO / "data/TruyenKieuPhongTinhCoLuc/thamchieu_kieu_1871_LieuVanDuong_phienam.json"),
              ("Kieu1872_DMT", REPO / "data/TruyenKieu1872/nomfoundation_1872_phienam.json")]),
    "LucVanTien1883": dict(
        labels=REPO / "dataset_out_LucVanTien1883/labels_final.csv",
        trans=REPO / "prepared/LucVanTien1883/transcriptions",
        refs=[("LVT1916_NF", REPO / "data/LucVanTien1916/nomfoundation_lvt_phienam.json")]),
}


# --------------------------------------------------------------------------- #
# tiện ích
# --------------------------------------------------------------------------- #
def canon_syl(t: str) -> str:
    """Chuẩn hoá âm GIỐNG pipeline: NFC + tone canon (normalize_tone_marks) + lower (load_qn_to_nom)."""
    return normalize_tone_marks(unicodedata.normalize("NFC", t).strip().lower())


def qn_tokens(text: str) -> list[str]:
    """Dòng QN → âm tiết chuẩn hoá (clean_line_text + split_to_syllables của pipeline; bỏ token không chữ cái)."""
    toks = [t for t in split_to_syllables(clean_line_text(text or "")) if any(ch.isalpha() for ch in t)]
    return [canon_syl(t) for t in toks]


def is_pua(c: str) -> bool:
    o = ord(c)
    return 0xE000 <= o <= 0xF8FF or 0xF0000 <= o <= 0x10FFFD


def is_cjk(c: str) -> bool:
    o = ord(c)
    return (0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF
            or 0x20000 <= o <= 0x3134F or is_pua(c))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(k, n):
    return round(100.0 * k / n, 1) if n else None


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


_DICTS: dict = {}


def dicts():
    if not _DICTS:
        q2n = load_qn_to_nom(str(DICT_QN))
        _DICTS["q2n"] = {k: list(v) for k, v in q2n.items()}
        _DICTS["R"] = {k: set(v) for k, v in q2n.items()}
        _DICTS["sim"] = load_similarity_dict(str(DICT_SIM))
    return _DICTS


def monotone_align(kim: list[str], syls: list[str], R: dict[str, set]) -> list[str | None]:
    """Căn chỉnh đơn điệu (Needleman–Wunsch) chữ kim ↔ âm QN khi số chữ ≠ N.
    match = +2 nếu kim[j] ∈ R(syl[i]), 0 nếu ghép không xác nhận, gap = −1. Trả chữ kim cho từng âm (None = trống)."""
    n, m = len(syls), len(kim)
    NEG = -10 ** 9
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = -i
        bt[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = -j
        bt[0][j] = "left"
    for i in range(1, n + 1):
        Ri = R.get(syls[i - 1], set())
        for j in range(1, m + 1):
            sc = 2 if kim[j - 1] in Ri else 0
            best, arg = dp[i - 1][j - 1] + sc, "diag"
            if dp[i - 1][j] - 1 > best:
                best, arg = dp[i - 1][j] - 1, "up"
            if dp[i][j - 1] - 1 > best:
                best, arg = dp[i][j - 1] - 1, "left"
            dp[i][j], bt[i][j] = best, arg
    out: list[str | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        a = bt[i][j]
        if a == "diag":
            out[i - 1] = kim[j - 1]
            i, j = i - 1, j - 1
        elif a == "up":
            i -= 1
        else:
            j -= 1
    return out


def decide_min(s1: str | None, syl: str, D) -> tuple[str, str | None, str]:
    """Luật tối thiểu của pipeline (consensus.decide_label rút gọn, không S3):
    GOLD_direct: s1 ∈ R(syl) → nhãn s1; GOLD_bridge: cầu duy nhất sim(s1) ∩ R → nhãn cầu (s1_inter_s2_similar);
    SYLLABLE-1: |R| == 1 → nhãn chữ duy nhất (proxy cho tầng SYLLABLE — pipeline thật KHÔNG gán chữ ở tầng này);
    còn lại REVIEW."""
    R = D["R"].get(syl, set())
    if s1 and s1 in R:
        return "GOLD", s1, "s1_inter_s2_direct"
    if s1:
        br = [c for c in dict.fromkeys(D["sim"].get(s1, [])) if c in R and c != s1]
        if len(br) == 1:
            return "GOLD_bridge", br[0], "s1_inter_s2_similar"
    if len(R) == 1:
        return "SYLLABLE-1", next(iter(R)), "syl_unique_reading"
    return "REVIEW", None, "unconfirmed"


# --------------------------------------------------------------------------- #
# (1) IHR-NomDB: GT độc lập
# --------------------------------------------------------------------------- #
class KimBudget:
    """Đếm lần gọi kim (1 lần = upload + recognize) bền qua các lần chạy; chặn khi vượt --budget."""

    def __init__(self, path: Path, budget: int):
        self.path, self.budget = path, budget
        self.state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"calls": 0, "log": []}

    @property
    def calls(self):
        return self.state["calls"]

    def can(self):
        return self.state["calls"] < self.budget

    def add(self, note: str):
        self.state["calls"] += 1
        self.state["log"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")


def kim_ocr_patch(patch: Path, scale: int, cache_dir: Path, budget: KimBudget, tag: str) -> dict:
    """OCR kim một patch (phóng `scale` lần, LANCZOS). Cache theo md5 patch + scale → 0 lần gọi khi chạy lại."""
    key = f"{md5_file(patch)}__x{scale}"
    cp = cache_dir / f"{key}.json"
    if cp.exists():
        d = json.loads(cp.read_text(encoding="utf-8"))
        d["from_cache"] = True
        return d
    if not budget.can():
        return {"status": "no_budget", "boxes": None, "scale": scale, "from_cache": False}
    import urllib3
    urllib3.disable_warnings()
    from PIL import Image
    from core.ocr import ocr_api
    src = patch
    tmp = None
    if scale != 1:
        im = Image.open(patch).convert("RGB")
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
        tmp = cache_dir / f"_tmp_{key}.png"
        im.save(tmp)
        src = tmp
    budget.add(f"{tag} x{scale} {patch.name}")
    fn = ocr_api.upload_image(str(src))
    boxes = ocr_api.recognize(fn) if fn else None
    if tmp is not None:
        try:
            tmp.unlink()
        except OSError:
            pass
    d = {"status": "ok" if boxes is not None else "api_fail", "patch": str(patch), "md5": key.split("__")[0],
         "scale": scale, "boxes": boxes, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    d["from_cache"] = False
    return d


def kim_chars(boxes: list[dict] | None) -> list[str]:
    """Hộp kim → chuỗi chữ theo thứ tự đọc (trên→dưới theo y đỉnh hộp); chỉ giữ ký tự CJK."""
    if not boxes:
        return []
    bs = sorted(boxes, key=lambda b: min(p[1] for p in b.get("points", [[0, 0]])))
    s = "".join(b.get("transcription", "") for b in bs)
    return [c for c in unicodedata.normalize("NFC", s) if is_cjk(c)]


def stratified_sample(rows: list[dict], n: int, seed: int, key="page_id") -> list[dict]:
    """Phân tầng theo trang, phân bổ tỷ lệ (làm tròn phần dư lớn nhất), rút ngẫu nhiên trong trang (seed cố định)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    pages = sorted(by)
    N = len(rows)
    quota = {p: n * len(by[p]) / N for p in pages}
    alloc = {p: int(math.floor(quota[p])) for p in pages}
    rem = n - sum(alloc.values())
    for p in sorted(pages, key=lambda p: (-(quota[p] - alloc[p]), p))[:rem]:
        alloc[p] += 1
    out = []
    for p in pages:
        pool = sorted(by[p], key=lambda r: r["patch_image"])
        k = min(alloc[p], len(pool))
        out.extend(rng.sample(pool, k))
    return out


def step_ihr(a, out: Path) -> dict:
    D = dicts()
    budget = KimBudget(out / "kim_calls.json", a.budget)
    summary = {"seed": a.seed, "n_per_book": a.n, "scale": a.scale, "budget": a.budget, "books": {}}
    for book, man in IHR_BOOKS.items():
        if a.books and book not in a.books:
            continue
        rows = list(csv.DictReader(open(man, encoding="utf-8"), delimiter="\t"))
        elig, drop = [], Counter()
        for r in rows:
            toks = qn_tokens(r["qn_verse"])
            nom = unicodedata.normalize("NFC", r["nom_text"])
            if r.get("len_match") != "1":
                drop["len_match!=1"] += 1
                continue
            if len(toks) != len(nom) or len(toks) != int(r["n_qn_syll"]):
                drop["token_split!=n"] += 1
                continue
            if not all(is_cjk(c) for c in nom):
                drop["nom_non_cjk"] += 1
                continue
            pp = (REPO / r["patch_image"]) if r.get("patch_image") else None
            if pp is None or not pp.is_file():
                drop["no_patch_image"] += 1      # IHR không cắt patch cho một số câu (trang tựa/thiếu hộp)
                continue
            r["patch_image"] = str(pp)
            r["_toks"], r["_nom"] = toks, nom
            elig.append(r)
        sample = stratified_sample(elig, a.n, a.seed)
        bdir = out / "ihr" / book
        cache = out / "kim_cache" / book
        # (a) thử raw trên vài patch đầu (ghi tỉ lệ kim đọc đủ N ở độ phân giải gốc)
        raw_stat = Counter()
        for r in sample[:a.raw_pilot]:
            d = kim_ocr_patch(Path(r["patch_image"]), 1, cache, budget, f"{book} raw")
            if d["status"] != "ok":
                raw_stat[d["status"]] += 1
                continue
            k = kim_chars(d["boxes"])
            raw_stat["n_kim==N" if len(k) == len(r["_nom"]) else ("empty" if not k else "partial")] += 1
        # (b) đo chính ở --scale
        pos_rows, patch_rows = [], []
        st = Counter()
        for r in sample:
            d = kim_ocr_patch(Path(r["patch_image"]), a.scale, cache, budget, f"{book} main")
            n = len(r["_nom"])
            kim = kim_chars(d["boxes"]) if d["status"] == "ok" else []
            st[f"status:{d['status']}"] += 1
            if d["status"] == "ok":
                st["kim_empty" if not kim else ("n_kim==N" if len(kim) == n else "n_kim!=N")] += 1
            if len(kim) == n:
                s1 = list(kim)
                how = "positional"
            elif kim:
                s1 = monotone_align(kim, r["_toks"], D["R"])
                how = "monotone_dp"
            else:
                s1 = [None] * n
                how = "none"
            prow = {"book": book, "page_id": r["page_id"], "patch": r["patch_image"], "split": r["split"], "n": n,
                    "n_kim": len(kim), "kim": "".join(kim), "gt": r["_nom"], "qn": " ".join(r["_toks"]),
                    "status": d["status"], "align": how}
            patch_rows.append(prow)
            for i, (syl, gt) in enumerate(zip(r["_toks"], r["_nom"])):
                tier, label, rule = decide_min(s1[i], syl, D)
                R = D["R"].get(syl, set())
                gt_pua = is_pua(gt) or ord(gt) >= 0x30000
                ok = label == gt if label else None
                cat = ""
                if label and not ok:
                    if gt_pua:
                        cat = "gt_pua_extG"
                    elif gt in R:
                        cat = "dong_am_di_the"      # GT cũng là một cách đọc của âm → kim chọn dị thể/đồng âm khác
                    elif label in D["sim"].get(gt, []) or gt in D["sim"].get(label, []):
                        cat = "gt_not_in_R_similar"
                    else:
                        cat = "gt_not_in_R"
                pos_rows.append({"book": book, "patch": r["patch_image"], "i": i, "syl": syl, "gt": gt,
                                 "gt_pua": int(gt_pua), "kim": s1[i] or "", "kim_eq_gt": int(s1[i] == gt),
                                 "kim_in_R": int(bool(s1[i]) and s1[i] in R), "R_size": len(R),
                                 "tier": tier, "label": label or "", "rule": rule,
                                 "label_eq_gt": "" if ok is None else int(ok), "err_cat": cat})
        write_csv(bdir / "sample_patches.csv", patch_rows)
        write_csv(bdir / "positions.csv", pos_rows)
        # thống kê
        P = pos_rows
        npos = len(P)
        nonpua = [p for p in P if not p["gt_pua"]]
        def prec(tiers, rows):
            g = [p for p in rows if p["tier"] in tiers]
            k = sum(int(p["label_eq_gt"] or 0) for p in g)
            lo, hi = wilson(k, len(g))
            return {"n": len(g), "correct": k, "precision": pct(k, len(g)),
                    "wilson95": [round(100 * lo, 1), round(100 * hi, 1)], "coverage": pct(len(g), npos)}
        errs = Counter(p["err_cat"] for p in P if p["tier"] == "GOLD" and p["label_eq_gt"] == 0)
        bs = {
            "manifest_rows": len(rows), "eligible": len(elig), "dropped": dict(drop),
            "sampled_patches": len(sample), "pages_in_sample": len({r["page_id"] for r in sample}),
            "raw_pilot_x1": dict(raw_stat), "kim_status": dict(st),
            "n_positions": npos, "n_positions_gt_pua_extG": npos - len(nonpua),
            "kim_raw_correct": {"all": pct(sum(p["kim_eq_gt"] for p in P), npos),
                                "non_pua": pct(sum(p["kim_eq_gt"] for p in nonpua), len(nonpua))},
            "kim_in_R": pct(sum(p["kim_in_R"] for p in P), npos),
            "GOLD_direct": prec({"GOLD"}, P), "GOLD_direct_nonpua": prec({"GOLD"}, nonpua),
            "GOLD_direct_plus_bridge": prec({"GOLD", "GOLD_bridge"}, P),
            "GOLD_bridge_only": prec({"GOLD_bridge"}, P),
            "SYLLABLE1": prec({"SYLLABLE-1"}, P), "SYLLABLE1_nonpua": prec({"SYLLABLE-1"}, nonpua),
            "REVIEW": pct(sum(p["tier"] == "REVIEW" for p in P), npos),
            "GOLD_errors_by_cat": dict(errs),
            "kim_eq_gt_but_not_in_R": sum(1 for p in P if p["kim_eq_gt"] and not p["kim_in_R"]),
            "kim_calls_total_so_far": budget.calls,
        }
        summary["books"][book] = bs
        print(f"[ihr] {book}: patches {len(sample)} pos {npos} | kim đúng thô {bs['kim_raw_correct']['all']} % | "
              f"GOLD prec {bs['GOLD_direct']['precision']} % {bs['GOLD_direct']['wilson95']} cov {bs['GOLD_direct']['coverage']} % | "
              f"SYL-1 {bs['SYLLABLE1']['precision']} % (n {bs['SYLLABLE1']['n']}) | calls {budget.calls}")
    summary["kim_calls_total"] = budget.calls
    (out / "ihr").mkdir(parents=True, exist_ok=True)
    (out / "ihr" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# (2) Đối chứng chéo dị bản (0 API)
# --------------------------------------------------------------------------- #
def load_ref(jf: Path) -> dict[int, list[tuple[list[str], str, str, int | None]]]:
    """json Nôm Foundation → {n_âm: [(âm chuẩn hoá, nôm, trang, verse_no)]} — chỉ câu len(nom)==n âm."""
    ref = defaultdict(list)
    for pg, vs in json.load(open(jf, encoding="utf-8"))["pages"].items():
        for v in vs:
            t = qn_tokens(v["qn"])
            nom = unicodedata.normalize("NFC", v["nom"])
            if t and len(t) == len(nom):
                ref[len(t)].append((t, nom, pg, v.get("verse_no")))
    return ref


def match_verses(book: str, cfg: dict, min_frac: float) -> tuple[list[dict], dict]:
    """Mỗi câu (page, column, half) của sách thạch bản → câu tham chiếu khớp QN (exact / ≥min_frac duy nhất)."""
    out = []
    stat = Counter()
    refs = {name: load_ref(jf) for name, jf in cfg["refs"]}
    for tf in sorted(glob.glob(str(cfg["trans"] / "page_*.json"))):
        t = json.load(open(tf, encoding="utf-8"))
        page = Path(tf).stem
        for col in t["columns"]:
            lo = int(col.get("len_odd") or 0)
            for half, key in ((0, "verse_odd"), (1, "verse_even")):
                v = col.get(key)
                if not v:
                    continue
                q = qn_tokens(v["text"])
                n = len(q)
                stat["verses"] += 1
                for rname, ref in refs.items():
                    cands = ref.get(n, [])
                    if not cands:
                        stat[f"{rname}:no_len"] += 1
                        continue
                    scores = [sum(a == b for a, b in zip(q, rt)) for rt, _, _, _ in cands]
                    mx = max(scores)
                    frac = mx / n
                    if frac < min_frac or scores.count(mx) != 1:
                        stat[f"{rname}:unmatched"] += 1
                        continue
                    rt, rnom, rpg, rno = cands[scores.index(mx)]
                    tier = "exact" if mx == n else "ge%d" % int(100 * min_frac)
                    stat[f"{rname}:{tier}"] += 1
                    out.append({"book": book, "page": page, "column": int(col["column"]), "half": half,
                                "syl_offset": 0 if half == 0 else lo, "n": n, "qn": " ".join(q),
                                "ref": rname, "ref_page": rpg, "ref_verse_no": rno, "ref_qn": " ".join(rt),
                                "ref_nom": rnom, "match_tier": tier, "n_same_syl": mx,
                                "same_pos": "".join("1" if a == b else "0" for a, b in zip(q, rt))})
    return out, dict(stat)


def load_labels(p: Path) -> list[dict]:
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def step_cross(a, out: Path) -> dict:
    D = dicts()
    summary = {"min_frac": a.min_frac, "books": {}}
    for book, cfg in CROSS_BOOKS.items():
        if a.books and book not in a.books:
            continue
        if not cfg["labels"].exists():
            print(f"[cross] {book}: thiếu {cfg['labels']}", file=sys.stderr)
            continue
        verses, vstat = match_verses(book, cfg, a.min_frac)
        labels = load_labels(cfg["labels"])
        by_col = defaultdict(list)
        for r in labels:
            by_col[(r["page"], int(r["column"]))].append(r)
        cells = []
        for v in verses:
            col = by_col.get((v["page"], v["column"]), [])
            for i in range(v["n"]):
                if v["same_pos"][i] != "1":
                    continue                      # chỉ so ở vị trí âm QN giống hệt
                si = v["syl_offset"] + i
                ref_nom = v["ref_nom"][i]
                for r in col:
                    if r.get("syl_idx", "") == "" or int(r["syl_idx"]) != si:
                        continue
                    syl = canon_syl(r["syllable"])
                    R = D["R"].get(syl, set())
                    lab = r.get("label") or ""
                    eq = lab == ref_nom if lab else None
                    di_the = bool(lab) and not eq and (ref_nom in R) and (lab in R)
                    sim = bool(lab) and not eq and (lab in D["sim"].get(ref_nom, []) or ref_nom in D["sim"].get(lab, []))
                    cells.append({**{k: v[k] for k in ("book", "page", "column", "half", "ref", "match_tier", "ref_verse_no")},
                                  "syl_idx": si, "nom_idx": r.get("nom_idx"), "syllable": syl, "ref_nom": ref_nom,
                                  "ref_pua": int(is_pua(ref_nom) or ord(ref_nom) >= 0x30000),
                                  "ocr_char": r.get("ocr_char"), "label": lab, "tier": r["tier"], "rule": r.get("rule"),
                                  "eq": "" if eq is None else int(eq), "di_the_cung_am": int(di_the), "similar": int(sim),
                                  "ocr_eq_ref": int(r.get("ocr_char") == ref_nom),
                                  "n_det": r.get("n_det"), "n_qn": r.get("n_qn"), "n_ocr": r.get("n_ocr"),
                                  "count_source": r.get("count_source"), "box_source": r.get("box_source"),
                                  "p_register": r.get("p_register"), "crop_quality_flag": r.get("crop_quality_flag", ""),
                                  "tier_v3": r.get("tier_v3", ""),
                                  "image": r.get("image")})
        bdir = out / "cross" / book
        write_csv(bdir / "verses_matched.csv", verses)
        write_csv(bdir / "cells.csv", cells)
        bs = {"verse_match": vstat, "n_verses_matched": len(verses), "refs": {}}
        for rname, _ in cfg["refs"]:
            rc = [c for c in cells if c["ref"] == rname]
            def agg(rows):
                g = [c for c in rows if c["tier"] == "GOLD"]
                k = sum(c["eq"] == 1 for c in g)
                kd = sum((c["eq"] == 1) or c["di_the_cung_am"] for c in g)
                lo, hi = wilson(k, len(g))
                gnp = [c for c in g if not c["ref_pua"]]
                knp = sum(c["eq"] == 1 for c in gnp)
                lo2, hi2 = wilson(knp, len(gnp))
                dis = [c for c in g if c["eq"] == 0]
                return {"n_cells": len(rows), "n_GOLD": len(g), "GOLD_eq": k, "GOLD_eq_pct": pct(k, len(g)),
                        "GOLD_wilson95": [round(100 * lo, 1), round(100 * hi, 1)],
                        "GOLD_eq_nonpua_pct": pct(knp, len(gnp)), "GOLD_nonpua_n": len(gnp),
                        "GOLD_nonpua_wilson95": [round(100 * lo2, 1), round(100 * hi2, 1)],
                        "GOLD_eq_or_di_the_pct": pct(kd, len(g)),
                        "GOLD_di_the_cung_am": sum(c["di_the_cung_am"] for c in g),
                        "GOLD_disagree": len(dis),
                        "GOLD_disagree_shape_similar": sum(c["similar"] for c in dis),
                        "GOLD_disagree_shape_similar_pct": pct(sum(c["similar"] for c in dis), len(dis)),
                        "GOLD_ref_pua": sum(c["ref_pua"] for c in g),
                        "ocr_eq_ref_all_pct": pct(sum(c["ocr_eq_ref"] for c in rows), len(rows)),
                        "by_tier": {t: {"n": sum(c["tier"] == t for c in rows),
                                        "ocr_eq_ref_pct": pct(sum(c["ocr_eq_ref"] for c in rows if c["tier"] == t),
                                                              sum(c["tier"] == t for c in rows))}
                                    for t in ("GOLD", "SYLLABLE", "REVIEW", "QUARANTINE")}}
            bs["refs"][rname] = {"all_tiers": agg(rc),
                                 "exact_only": agg([c for c in rc if c["match_tier"] == "exact"]),
                                 "n_verses": sum(v["ref"] == rname for v in verses),
                                 "n_verses_exact": sum(v["ref"] == rname and v["match_tier"] == "exact" for v in verses)}
        # 20 ô bất đồng mẫu (GOLD, khác chữ tham chiếu, ref không PUA), tham chiếu đầu tiên, seed cố định
        r0 = cfg["refs"][0][0]
        dis = [c for c in cells if c["ref"] == r0 and c["tier"] == "GOLD" and c["eq"] == 0 and not c["ref_pua"]]
        rng = random.Random(a.seed)
        smp = rng.sample(dis, min(20, len(dis)))
        write_csv(bdir / "disagreements_sample20.csv", smp)
        bs["n_GOLD_disagree_primary_nonpua"] = len(dis)
        # nền dị bản: hai tham chiếu khác nhau bao nhiêu ở cùng câu/vị trí (âm giống hệt cả ba bên)?
        if len(cfg["refs"]) >= 2:
            r1 = cfg["refs"][1][0]
            key = lambda v: (v["page"], v["column"], v["half"])
            v0 = {key(v): v for v in verses if v["ref"] == r0}
            v1 = {key(v): v for v in verses if v["ref"] == r1}
            n_ = eq_ = eqd_ = 0
            for k in set(v0) & set(v1):
                A, B = v0[k], v1[k]
                for i in range(A["n"]):
                    if A["same_pos"][i] == "1" and B["same_pos"][i] == "1":
                        x, y = A["ref_nom"][i], B["ref_nom"][i]
                        if is_pua(x) or is_pua(y) or ord(x) >= 0x30000 or ord(y) >= 0x30000:
                            continue
                        n_ += 1
                        eq_ += x == y
            bs["inter_ref_baseline"] = {"refs": [r0, r1], "n_positions_nonpua": n_, "eq_pct": pct(eq_, n_),
                                        "note": "tỉ lệ hai dị bản tham chiếu dùng CÙNG chữ ở vị trí âm giống hệt = trần kỳ vọng cho khớp 1884↔tham chiếu"}
        summary["books"][book] = bs
        m = bs["refs"][r0]["all_tiers"]
        print(f"[cross] {book} vs {r0}: câu khớp {bs['refs'][r0]['n_verses']} (exact {bs['refs'][r0]['n_verses_exact']}) | "
              f"ô GOLD {m['n_GOLD']} khớp {m['GOLD_eq_pct']} % {m['GOLD_wilson95']} | khớp+dị thể {m['GOLD_eq_or_di_the_pct']} %")
    (out / "cross").mkdir(parents=True, exist_ok=True)
    (out / "cross" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# (3) Cổng máy
# --------------------------------------------------------------------------- #
GATES = {
    "ndet": ("n_det ≠ N (SPEC §7 I5)", lambda r: r.get("n_det") != r.get("n_qn")),
    "MN": ("n_ocr ≠ n_qn (M ≠ N)", lambda r: r.get("n_ocr") != r.get("n_qn")),
    "conflict": ("count_source == conflict", lambda r: r.get("count_source") == "conflict"),
    "box": ("box_source ≠ detector (midpoint/split)", lambda r: r.get("box_source") not in ("detector", "", None)),
    "bridge": ("luật cầu s1_inter_s2_similar (kim ∉ dict, nhãn = chữ cầu; tier_v3 CHAR_B)",
               lambda r: r.get("rule") == "s1_inter_s2_similar" or r.get("tier_v3") == "CHAR_B"),
    "tonefix": ("luật s1_inter_s2_direct_am_sua_dau (âm QN OCR bị sửa dấu để khớp dict)",
                lambda r: r.get("rule") == "s1_inter_s2_direct_am_sua_dau"),
    "preg": ("p_register < 0,8", lambda r: _f(r.get("p_register")) < 0.8),
    "crop": ("crop_quality_flag ∈ {blank, truncated}", lambda r: (r.get("crop_quality_flag") or "") in ("blank", "truncated")),
    "bleed": ("crop_quality_flag == bleed (mực cột kề lọt vào hộp)", lambda r: (r.get("crop_quality_flag") or "") == "bleed"),
}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 1.0


def step_gates(a, out: Path) -> dict:
    summary = {"min_coverage": a.min_coverage, "gates": {k: v[0] for k, v in GATES.items()},
               "kim_confidence": "KHÔNG có: kim_raw/*.json chỉ có transcription/points/difficult (difficult=false 100 %)",
               "books": {}}
    for book, cfg in CROSS_BOOKS.items():
        if a.books and book not in a.books:
            continue
        cp = out / "cross" / book / "cells.csv"
        if not cfg["labels"].exists() or not cp.exists():
            print(f"[gates] {book}: cần labels_final + cross/cells.csv", file=sys.stderr)
            continue
        labels = load_labels(cfg["labels"])
        gold = [r for r in labels if r["tier"] == "GOLD"]
        cells = [c for c in load_labels(cp) if c["tier"] == "GOLD" and c["ref"] == cfg["refs"][0][0]]
        n_gold = len(gold)
        # kim_raw difficult flag
        ndiff = 0
        for kf in glob.glob(str(REPO / "prepared" / book / "kim_raw" / "*.json")):
            ndiff += sum(1 for b in json.load(open(kf, encoding="utf-8")).get("boxes", []) if b.get("difficult"))
        def evaluate(keys):
            fn = [GATES[k][1] for k in keys]
            demote = lambda r: any(f(r) for f in fn)
            kept = [r for r in gold if not demote(r)]
            ck = [c for c in cells if not demote(c)]
            eq = sum(c["eq"] == "1" for c in ck)
            eqd = sum(c["eq"] == "1" or c["di_the_cung_am"] == "1" for c in ck)
            lo, hi = wilson(eq, len(ck))
            return {"gates": "+".join(keys) or "(none)", "GOLD_kept": len(kept), "GOLD_demoted": n_gold - len(kept),
                    "coverage_pct": pct(len(kept), n_gold), "proxy_n": len(ck), "proxy_eq_pct": pct(eq, len(ck)),
                    "proxy_wilson95": [round(100 * lo, 1), round(100 * hi, 1)], "proxy_eq_or_di_the_pct": pct(eqd, len(ck))}
        rows = [evaluate([])] + [evaluate([k]) for k in GATES]
        keys = list(GATES)
        combos = []
        for mask in range(1, 1 << len(keys)):
            ks = [k for i, k in enumerate(keys) if mask >> i & 1]
            combos.append(evaluate(ks))
        ok = [c for c in combos if (c["coverage_pct"] or 0) >= a.min_coverage and c["proxy_n"] > 0]
        best = max(ok, key=lambda c: (c["proxy_eq_pct"], c["coverage_pct"])) if ok else None
        best_d = max(ok, key=lambda c: (c["proxy_eq_or_di_the_pct"], c["coverage_pct"])) if ok else None
        # (iv) bất đồng dị bản → hạ: đếm ô GOLD có nhãn ≠ chữ tham chiếu (ref không PUA); KHÔNG đo proxy (tự khẳng định)
        dis = [c for c in cells if c["eq"] == "0" and c["ref_pua"] != "1"]
        write_csv(out / "gates" / book / "gates_single.csv", rows)
        write_csv(out / "gates" / book / "gates_combos.csv", sorted(combos, key=lambda c: -(c["proxy_eq_pct"] or 0)))
        bs = {"n_GOLD": n_gold, "n_proxy_cells_GOLD": len(cells), "kim_difficult_boxes": ndiff,
              "baseline": rows[0], "single": rows[1:], "best_by_proxy_eq": best, "best_by_eq_or_di_the": best_d,
              "gate_iv_disagree_with_variant": {"GOLD_demoted": len(dis), "of_proxy_GOLD_pct": pct(len(dis), len(cells)),
                                                "covers_GOLD_pct": pct(len(cells), n_gold),
                                                "note": "chỉ áp được cho ô nằm trong câu khớp dị bản; proxy sau = 100 % theo định nghĩa (tự khẳng định) — không báo"}}
        summary["books"][book] = bs
        print(f"[gates] {book}: GOLD {n_gold}; baseline proxy {rows[0]['proxy_eq_pct']} % (n {rows[0]['proxy_n']}); "
              f"best ≥{a.min_coverage} %: {best['gates'] if best else None} → {best['proxy_eq_pct'] if best else None} % cov {best['coverage_pct'] if best else None} %")
    (out / "gates").mkdir(parents=True, exist_ok=True)
    (out / "gates" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
def write_report(out: Path):
    S = {}
    for k in ("ihr", "cross", "gates"):
        p = out / k / "summary.json"
        if p.exists():
            S[k] = json.loads(p.read_text(encoding="utf-8"))
    (out / "SUMMARY.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# auto_precision — báo cáo tự sinh", f"_{time.strftime('%Y-%m-%d %H:%M')}_", ""]
    if "ihr" in S:
        s = S["ihr"]
        L += [f"## (1) GT độc lập IHR-NomDB (seed {s['seed']}, {s['n_per_book']} patch/sách, phóng ×{s['scale']}, kim calls {s.get('kim_calls_total')})",
              "| sách | patch | vị trí | GT PUA/ExtG | kim đúng thô | GOLD n / cov | GOLD precision [Wilson] | GOLD (bỏ PUA) | SYL-1 n / precision | lỗi GOLD theo loại |", "|---|---|---|---|---|---|---|---|---|---|"]
        for b, v in s["books"].items():
            g, gn, sy = v["GOLD_direct"], v["GOLD_direct_nonpua"], v["SYLLABLE1"]
            L.append(f"| {b} | {v['sampled_patches']} | {v['n_positions']} | {v['n_positions_gt_pua_extG']} | {v['kim_raw_correct']['all']} % ({v['kim_raw_correct']['non_pua']} % bỏ PUA) | {g['n']} / {g['coverage']} % | **{g['precision']} %** {g['wilson95']} | {gn['precision']} % {gn['wilson95']} (n {gn['n']}) | {sy['n']} / {sy['precision']} % {sy['wilson95']} | {v['GOLD_errors_by_cat']} |")
        L.append("")
    if "cross" in S:
        s = S["cross"]
        L += [f"## (2) Đối chứng dị bản (0 API; câu khớp exact hoặc ≥{int(100*s['min_frac'])} % âm, duy nhất; so ở vị trí âm giống hệt)",
              "| sách | tham chiếu | câu khớp (exact) | ô GOLD | GOLD khớp [Wilson] | GOLD khớp bỏ ref PUA [Wilson] (n) | bất đồng: hình gần (SinoNom_Similar) | ref PUA | kim==ref theo tier |", "|---|---|---|---|---|---|---|---|---|"]
        for b, v in s["books"].items():
            for rn, rv in v["refs"].items():
                m = rv["all_tiers"]
                bt = {t: x["ocr_eq_ref_pct"] for t, x in m["by_tier"].items() if x["n"]}
                L.append(f"| {b} | {rn} | {rv['n_verses']} ({rv['n_verses_exact']}) | {m['n_GOLD']} | **{m['GOLD_eq_pct']} %** {m['GOLD_wilson95']} | **{m['GOLD_eq_nonpua_pct']} %** {m['GOLD_nonpua_wilson95']} ({m['GOLD_nonpua_n']}) | {m['GOLD_disagree_shape_similar']}/{m['GOLD_disagree']} = {m['GOLD_disagree_shape_similar_pct']} % | {m['GOLD_ref_pua']} | {bt} |")
            if v.get("inter_ref_baseline"):
                ib = v["inter_ref_baseline"]
                L.append(f"| {b} | nền {ib['refs'][0]} ↔ {ib['refs'][1]} | — | {ib['n_positions_nonpua']} vị trí | **{ib['eq_pct']} %** (hai dị bản tham chiếu cùng chữ) | | | | |")
        L.append("")
    if "gates" in S:
        s = S["gates"]
        L += [f"## (3) Cổng máy (coverage ≥ {s['min_coverage']} %) — {s['kim_confidence']}",
              "| sách | cổng | GOLD hạ | coverage | proxy n | proxy khớp [Wilson] | khớp+dị thể |", "|---|---|---|---|---|---|---|"]
        for b, v in s["books"].items():
            for r in [v["baseline"]] + v["single"] + [v["best_by_proxy_eq"]]:
                if r:
                    L.append(f"| {b} | {r['gates']} | {r['GOLD_demoted']} | {r['coverage_pct']} % | {r['proxy_n']} | {r['proxy_eq_pct']} % {r['proxy_wilson95']} | {r['proxy_eq_or_di_the_pct']} % |")
            L.append(f"| {b} | (iv) bất đồng dị bản | {v['gate_iv_disagree_with_variant']['GOLD_demoted']} | — | — | tự khẳng định | — |")
        L.append("")
    (out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="cross,gates", help="tập con của ihr,cross,gates")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--books", default="", help="lọc sách (a,b); mặc định mọi sách của bước")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--n", type=int, default=200, help="patch mỗi sách IHR")
    ap.add_argument("--scale", type=int, default=3, help="phóng patch trước khi gọi kim (34 px → kim đọc thiếu)")
    ap.add_argument("--raw-pilot", type=int, default=4, help="số patch đầu mỗi sách thử thêm ở ×1 để ghi tỉ lệ đọc đủ")
    ap.add_argument("--budget", type=int, default=420, help="tổng lần gọi kim (upload+recognize) tích luỹ, bền qua lần chạy")
    ap.add_argument("--min-frac", type=float, default=0.75, help="cross: tỉ lệ âm cùng vị trí tối thiểu để nhận câu")
    ap.add_argument("--min-coverage", type=float, default=80.0)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args(argv)
    a.books = [b for b in a.books.split(",") if b]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = ["ihr", "cross", "gates"] if a.all else [s for s in a.steps.split(",") if s]
    rc = 0
    if not a.report_only:
        for s in steps:
            try:
                {"ihr": step_ihr, "cross": step_cross, "gates": step_gates}[s](a, out)
            except Exception as e:  # noqa: BLE001
                print(f"[{s}] LỖI {type(e).__name__}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                rc = 1
    write_report(out)
    print(f"→ {out / 'SUMMARY.json'} ; {out / 'REPORT.md'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
