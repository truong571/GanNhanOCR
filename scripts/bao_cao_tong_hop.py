#!/usr/bin/env python3
"""bao_cao_tong_hop.py — sinh BÁO CÁO TỔNG HỢP 10 bộ từ đầu ra ĐANG CÓ trên đĩa (0 API, 0 token, chỉ đọc).

    .venv/bin/python scripts/bao_cao_tong_hop.py                      # -> docs/BAO_CAO_TONG_HOP_<hôm nay>.md
    .venv/bin/python scripts/bao_cao_tong_hop.py --timing logs/clean_rebuild_<ts>_thoi_gian.tsv
    .venv/bin/python scripts/bao_cao_tong_hop.py --out /tmp/x.md --date 2026-09-28

Nguồn (thiếu tệp nào -> ô tương ứng ghi "chưa có", KHÔNG suy ra số):
  dataset_out/labels_final.csv (STT, cột book) · prepared/<sách>/dataset_out/labels_gated.csv (dataset_out theo config)
  dataset/<bộ>/labels.csv · dataset/_ALL/{gold_exact.csv,SOURCES.json} · measure_out/SUMMARY.json
  measure_out/<IHR>/ihr_endtoend/summary.json · measure_out/<Borg>/borg_endtoend/{summary.json,cells.csv}
  measure_out/gold_exact/summary.json · <dataset_out LVT1883|KVK1884>/auto_precision_verify|_gated/SUMMARY.json
  thời gian: --timing (TSV của clean_rebuild_all.sh --run; mặc định bản mới nhất trong logs/), CHECKSUMS.txt (thoi_gian),
  dataset/_ALL/THOI_GIAN.txt, logs/run_GOLD_EXACT_*.log.
Mức chắc của mỗi số độ chính xác được GHI RÕ: ĐO (nhãn người) · ƯỚC LƯỢNG (dị bản người) · SUY ĐOÁN (không có sự thật).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
csv.field_size_limit(1 << 30)

# (khoá set8 trong gold_exact.csv, tên hiển thị, book_set, vai trò, loại bản)
SETS = [
    ("stt2", "STT2", "SachThanhTruyen", "giao nộp", "viết tay Công giáo"),
    ("stt4", "STT4", "SachThanhTruyen", "giao nộp", "viết tay Công giáo"),
    ("stt11", "STT11", "SachThanhTruyen", "giao nộp", "viết tay Công giáo"),
    ("L83", "LucVanTien1883", "LucVanTien1883", "giao nộp", "thạch bản"),
    ("KVK", "KimVanKieu1884", "KimVanKieu1884", "giao nộp", "thạch bản"),
    ("Chr", "Chrestomathie1872", "Chrestomathie1872", "giao nộp", "sách in văn xuôi"),
    ("L16", "LucVanTien1916", "LucVanTien1916", "đánh giá (IHR)", "mộc bản"),
    ("TK", "TruyenKieu1872", "TruyenKieu1872", "đánh giá (IHR)", "mộc bản"),
    ("B18", "SachKinhThayCaBinh", "SachKinhThayCaBinh", "đánh giá (Borg)", "viết tay Công giáo"),
    ("B34", "SachDungLyHoThan", "SachDungLyHoThan", "đánh giá (Borg)", "viết tay Công giáo"),
]
EVID = {"do_tren_nhan_nguoi": "ĐO", "uoc_luong": "ƯỚC LƯỢNG", "suy_doan": "SUY ĐOÁN"}
TIERS = ["GOLD", "GOLD_text_only", "SYLLABLE", "REVIEW", "QUARANTINE"]
GX = ["ok", "text_only", "uncertified", "review"]
MISSING: list[str] = []
NA = "chưa có"


# ------------------------------------------------------------------ định dạng
def n_(x) -> str:
    return NA if x is None else f"{int(x):,}".replace(",", ".")


def p_(x, d=2) -> str:
    """tỉ lệ 0..1 -> '98,05 %'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return NA
    return f"{100 * float(x):.{d}f} %".replace(".", ",")


def ci_(lo, hi, d=2, scale=100) -> str:
    if lo is None or hi is None:
        return ""
    return f"[{scale * float(lo):.{d}f}–{scale * float(hi):.{d}f}]".replace(".", ",")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def jload(p: Path, what: str):
    if not p.exists():
        MISSING.append(f"{what}: `{p.relative_to(REPO) if p.is_absolute() else p}`")
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        MISSING.append(f"{what}: đọc lỗi {e}")
        return None


def dig(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def table(hdr: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(hdr) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ nguồn
def book_ds_out(book: str) -> Path:
    p = REPO / "config" / f"pipeline_{book}.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if cfg.get("run_config"):
        cfg = yaml.safe_load((REPO / str(cfg["run_config"])).read_text(encoding="utf-8")) or {}
    return REPO / str((cfg.get("run") or {}).get("dataset_out") or f"prepared/{book}/dataset_out")


def tier_counts(path: Path, by_book: bool = False):
    if not path.exists():
        MISSING.append(f"nhãn: `{path.relative_to(REPO)}`")
        return None
    c: dict = defaultdict(Counter)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c[r.get("book", "") if by_book else "_"][r.get("tier", "")] += 1
    return c


def export_counts(path: Path, by_book: bool = False):
    if not path.exists():
        MISSING.append(f"bản xuất: `{path.relative_to(REPO)}`")
        return None
    c: dict = Counter()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c[r.get("book", "") if by_book else "_"] += 1
    return c


def gold_exact_counts():
    p = REPO / "dataset/_ALL/gold_exact.csv"
    if not p.exists():
        MISSING.append("GOLD chính xác: `dataset/_ALL/gold_exact.csv`")
        return None, None, None, {}
    st, ev, pol, cnt = defaultdict(Counter), {}, None, {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s8 = r["set8"]
            st[s8][r["gold_exact"]] += 1
            ev[s8] = r.get("evidence_level", "")
            pol = pol or (r.get("policy_version"), r.get("config_sha16"))
            if s8 in ("B18", "B34"):
                cnt[r["cell_uid"]] = r.get("cnt", "")
    return st, ev, pol, cnt


def cross_summary(book: str):
    ds = book_ds_out(book)
    for sub in ("auto_precision_verify", "auto_precision_gated", "auto_precision"):
        p = ds / sub / "SUMMARY.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            return dig(s, "cross", "books", book), p
    MISSING.append(f"dị bản {book}: `{(ds / 'auto_precision_verify/SUMMARY.json').relative_to(REPO)}`")
    return None, None


def borg_cnt_split(book: str, cnt: dict):
    """GOLD có chữ người: tỉ lệ đúng V1+ theo cờ CNT của gold_exact (1 = số chữ OCR của cột ≠ số âm QN)."""
    p = REPO / "measure_out" / book / "borg_endtoend" / "cells.csv"
    if not p.exists() or not cnt:
        return None
    agg = defaultdict(lambda: [0, 0])
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("tier") != "GOLD" or not (r.get("gt") or "").strip():
                continue
            k = cnt.get(r["cell_uid"], "?")
            agg[k][0] += r.get("v1p") == "True"
            agg[k][1] += 1
    return dict(agg)


def timing_rows(tsv: Path | None):
    rows = []
    if tsv and tsv.exists():
        with open(tsv, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                rows.append([r["buoc"], r["lenh"], r["ma_thoat"], f"{int(r['giay']) // 60} ph {int(r['giay']) % 60} s",
                             r["bat_dau"], r["ket_thuc"], f"`{r['log']}`"])
    return rows


def _ts(s: str) -> float:
    return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%S"))


def last_block(checksums: Path) -> list[tuple[str, str, int]]:
    """Lượt gần nhất trong CHECKSUMS.txt: [(thời điểm, bước, giây)] từ dòng 'thoi_gian setup' cuối cùng tới hết."""
    if not checksums.exists():
        return []
    lines = [ln.split() for ln in checksums.read_text(encoding="utf-8").splitlines() if "  thoi_gian  " in ln]
    lines = [ln for ln in lines if len(ln) >= 4 and ln[3].endswith("s") and ln[3][:-1].isdigit()]
    start = max((i for i, ln in enumerate(lines) if ln[2] == "setup"), default=0)
    return [(ln[0], ln[2], int(ln[3][:-1])) for ln in lines[start:]]


def run_start(tag: str, first_tick: str) -> float | None:
    """Thời điểm bắt đầu lượt = tên log logs/run_<tag>_<YYYYmmdd_HHMMSS>.log mới nhất KHÔNG muộn hơn tick đầu tiên.
    Cần vì dưới --book all mỗi sách chạy trong subshell kế thừa đồng hồ T_STEP của cha: tick 'setup' là giây CỘNG DỒN."""
    best = None
    for p in (REPO / "logs").glob(f"run_{tag}_*.log"):
        try:
            t = time.mktime(time.strptime(p.stem[len(f"run_{tag}_"):][:15], "%Y%m%d_%H%M%S"))
        except ValueError:
            continue
        if t <= _ts(first_tick) and (best is None or t > best):
            best = t
    return best


def fmt_s(s: float | int | None) -> str:
    return NA if s is None else f"{int(s) // 60} ph {int(s) % 60} s"


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "?"
    except Exception:  # noqa: BLE001
        return "?"


# ------------------------------------------------------------------ báo cáo
def build(a) -> str:
    L: list[str] = []
    day = a.date or time.strftime("%Y-%m-%d")
    L += [f"# BÁO CÁO TỔNG HỢP 10 BỘ — {day}", "",
          f"> Tệp SINH TỰ ĐỘNG bởi `scripts/bao_cao_tong_hop.py` lúc {time.strftime('%Y-%m-%d %H:%M:%S')} (git `{git_head()}`),",
          "> chỉ ĐỌC đầu ra đang có trên đĩa (0 API, 0 token). Ô ghi **chưa có** = tệp nguồn thiếu, KHÔNG suy ra số.",
          "> Mức chắc của số độ chính xác: **ĐO** (so nhãn người từng chữ) · **ƯỚC LƯỢNG** (so dị bản do người số hoá — "
          "khác chữ hợp lệ bị tính sai) · **SUY ĐOÁN** (không có sự thật người). Đừng sửa tay — chạy lại script.", ""]

    # ---------- 1. bảng 10 bộ
    stt = tier_counts(REPO / "dataset_out/labels_final.csv", by_book=True)
    stt_exp = export_counts(REPO / "dataset/SachThanhTruyen/labels.csv", by_book=True)
    gx, ev, pol, cnt = gold_exact_counts()
    rows, tot = [], defaultdict(int)
    gold_n = {}
    for s8, name, bset, role, kind in SETS:
        if bset == "SachThanhTruyen":
            c = (stt or {}).get(s8) if stt else None
            e = (stt_exp or {}).get(s8) if stt_exp else None
        else:
            cc = tier_counts(book_ds_out(bset) / "labels_gated.csv")
            c = cc.get("_") if cc else None
            ee = export_counts(REPO / "dataset" / bset / "labels.csv")
            e = ee.get("_") if ee else None
        g = (gx or {}).get(s8) if gx else None
        vals = [sum(c.values()) if c else None] + [c.get(t, 0) if c else None for t in TIERS] + [e] + \
               [g.get(k, 0) if g else None for k in GX]
        for i, v in enumerate(vals):
            if v is not None:
                tot[i] += v
        gold_n[name] = (c or {}).get("GOLD") if c else None
        rows.append([name, role, kind] + [n_(v) for v in vals] + [EVID.get((ev or {}).get(s8, ""), NA) if ev else NA])
    rows.append(["**TỔNG 10 bộ**", "", ""] + [f"**{n_(tot[i])}**" if i in tot else NA for i in range(len(vals))] + [""])
    L += ["## 1. Bảng 10 bộ", "",
          "Số ô = mọi tầng của bản dựng (`labels_final` STT / `labels_gated` sách). \"xuất\" = dòng `dataset/<bộ>/labels.csv`. "
          "GX = trạng thái GOLD chính xác (`dataset/_ALL/gold_exact.csv`" +
          (f", policy {pol[0]}, config {pol[1]}" if pol else "") + "); ok+text_only+uncertified+review = GOLD ảnh của bộ gộp.", "",
          table(["bộ", "vai trò", "loại bản", "ô", "GOLD", "text_only", "SYLLABLE", "REVIEW", "QUARANT.", "xuất",
                 "GX ok", "GX text_only", "GX uncert.", "GX review", "mức chắc"], rows, "lll" + "r" * 11 + "l"), ""]
    src = jload(REPO / "dataset/_ALL/SOURCES.json", "bộ gộp")
    if src:
        bb = src.get("bat_bien") or {}
        L += [f"Bộ gộp `dataset/_ALL`: {n_(src.get('n_dong'))} dòng · {n_(src.get('n_tep_crop'))} tệp crop · "
              f"{n_(src.get('eval_only_dong'))} dòng `evaluation_only` · {src.get('n_bo')} thư mục bộ "
              f"({', '.join(src.get('bo') or [])}) · bất biến gộp {sum(1 for v in bb.values() if v is True)}/{len(bb)} PASS.", ""]
    same = defaultdict(list)
    for k, v in gold_n.items():
        if v:
            same[v].append(k)
    for v, ks in same.items():
        if len(ks) > 1:
            L += [f"_Ghi chú kiểm: {', '.join(ks)} có cùng số ô GOLD ({n_(v)}) — trùng số tình cờ, hai bản dựng khác nhau "
                  f"(tổng ô và tầng khác xem bảng)._", ""]

    # ---------- 2. độ chính xác
    L += ["## 2. Độ chính xác — ĐO / ƯỚC LƯỢNG / SUY ĐOÁN", ""]
    acc = []
    ge_sum = jload(REPO / "measure_out/gold_exact/summary.json", "gold_exact_eval")
    for s8, name in (("L16", "LucVanTien1916"), ("TK", "TruyenKieu1872")):
        s = jload(REPO / "measure_out" / name / "ihr_endtoend" / "summary.json", f"ihr_endtoend {name}")
        g = dig(s, "precision", "gold_anh")
        if g:
            acc.append([name, "**ĐO** (nhãn người IHR)", "GOLD ảnh: nhãn = chữ người", p_(g["precision"]),
                        ci_(*g["ci95"]), n_(g["with_gt"]), f"`measure_out/{name}/ihr_endtoend/summary.json`"])
            acc.append([name, "**ĐO**", "GOLD ảnh bỏ PUA + dị thể", p_(g.get("precision_bo_pua_dithe")), "",
                        n_(g["with_gt"]), "cùng tệp"])
        r = dig(ge_sum, "ihr", s8)
        if r and "both_pt" in r:
            acc.append([name, "**ĐO** (chỉ báo)", "gold_exact = ok: đúng hai vế (V1+ ∧ crop chuẩn)", p_(r["both_pt"]),
                        ci_(r.get("both_lo"), r.get("both_hi")), n_(r.get("n_eval")), "`measure_out/gold_exact/summary.json`"])
    for s8, name in (("B18", "SachKinhThayCaBinh"), ("B34", "SachDungLyHoThan")):
        s = jload(REPO / "measure_out" / name / "borg_endtoend" / "summary.json", f"borg_endtoend {name}")
        g = dig(s, "tiers", "GOLD")
        if g and g.get("v1p"):
            acc.append([name, "**ĐO** (nhãn người Borg)", "GOLD: nhãn = chữ người, V1+", p_(g["v1p"]["pt"]),
                        ci_(*g["v1p"]["ci_page"]), n_(g["n_eval"]), f"`measure_out/{name}/borg_endtoend/summary.json`"])
            acc.append([name, "**ĐO**", "GOLD strict (trùng hẳn)", p_(g["strict"]["pt"]), ci_(*g["strict"]["ci_page"]),
                        n_(g["n_eval"]), "cùng tệp"])
        ok = dig(s, "gold_exact", "ok")
        if ok and ok.get("v1p"):
            acc.append([name, "**ĐO**", f"gold_exact = ok (n ok {n_(ok['n'])}): V1+", f"{ok['v1p']['k']}/{ok['n_eval']}",
                        ci_(*ok["v1p"]["wilson"]) + " Wilson", n_(ok["n_eval"]), "cùng tệp"])
        elif s is not None:
            acc.append([name, "**ĐO**", "GOLD", "chưa có nhãn máy (SKIP)", "", "", ""])
    for name in ("LucVanTien1883", "KimVanKieu1884"):
        c, src_p = cross_summary(name)
        for ref, blk in ((c or {}).get("refs") or {}).items():
            at = blk.get("all_tiers") or {}
            if "GOLD_eq_pct" not in at:
                continue
            w = at.get("GOLD_wilson95") or [None, None]
            acc.append([name, "**ƯỚC LƯỢNG** (dị bản người)", f"GOLD = chữ dị bản `{ref}` (cận dưới)",
                        f"{at['GOLD_eq_pct']:.1f} %".replace(".", ","), ci_(w[0], w[1], 1, 1), n_(at.get("n_GOLD")),
                        f"`{src_p.relative_to(REPO)}`"])
            acc.append([name, "**ƯỚC LƯỢNG**", f"GOLD = chữ `{ref}` hoặc dị thể cùng âm",
                        f"{at.get('GOLD_eq_or_di_the_pct', float('nan')):.1f} %".replace(".", ","), "",
                        n_(at.get("n_GOLD")), "cùng tệp"])
    stt_est = None          # hằng số TN3 do borg_endtoend_eval ghi kèm (không phải phép đo trên STT)
    for name in ("SachKinhThayCaBinh", "SachDungLyHoThan"):
        p = REPO / "measure_out" / name / "borg_endtoend" / "summary.json"
        if p.exists():
            stt_est = json.loads(p.read_text(encoding="utf-8")).get("stt_estimate_suy_doan")
            break
    for k in ("stt2", "stt4", "stt11"):
        v = (stt_est or {}).get(k)
        val, ci = (v.split(" [", 1) + [""])[:2] if v else (NA, "")
        acc.append([k.upper(), "**SUY ĐOÁN**", "tập \"lai\" (bộ ước lượng TN3, không có sự thật người)",
                    f"{val} %" if v else NA, f"[{ci}" if ci else "", "",
                    "`borg_endtoend_eval.STT_EST` ← docs/GOLD_CHINH_XAC_2026-09-27.md §6"])
    acc.append(["Chrestomathie1872", "**SUY ĐOÁN**", "không có nhãn người lẫn dị bản số hoá", NA, "", "", "—"])
    L += [table(["bộ", "mức chắc", "đại lượng", "giá trị", "CI 95 %", "n", "nguồn"], acc, "lllrrrl"), "",
          "CI: IHR = Wilson; Borg = bootstrap cụm trang (B = 2000) trừ khi ghi Wilson; dị bản = Wilson (điểm %).", ""]

    # ---------- 3. kim theo loại sách
    L += ["## 3. Độ đúng kim (OCR chữ Nôm HCMUS) theo loại sách", ""]
    km = []
    for name in ("LucVanTien1916", "TruyenKieu1872"):
        p = REPO / "measure_out" / name / "ihr_endtoend" / "summary.json"
        k = dig(json.loads(p.read_text(encoding="utf-8")), "kim_raw") if p.exists() else None
        if k:
            km.append([name, "mộc bản", "**ĐO**", "kim ở ô (mọi ô có GT)", p_(k["precision"]), ci_(*k["ci95"]), n_(k["n"])])
            km.append([name, "mộc bản", "**ĐO**", "kim ở ô GOLD", p_(k["precision_tren_o_GOLD"]), "", n_(k["n_gold"])])
    for name in ("SachKinhThayCaBinh", "SachDungLyHoThan"):
        p = REPO / "measure_out" / name / "borg_endtoend" / "summary.json"
        s = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        k = s.get("kim_at_cells") or {}
        if k.get("strict"):
            km.append([name, "viết tay", "**ĐO**", "kim ở ô strict", p_(k["strict"]["pt"]), ci_(*k["strict"]["ci_page"]),
                       n_(k["n_eval"])])
            km.append([name, "viết tay", "**ĐO**", "kim ở ô V1+", p_(k["v1p"]["pt"]), ci_(*k["v1p"]["ci_page"]), n_(k["n_eval"])])
        pg = s.get("kim_vs_human_page") or {}
        if pg.get("strict"):
            km.append([name, "viết tay", "**ĐO**", "kim cả trang (LCS): precision / recall strict",
                       f"{p_(pg['strict']['precision'])} / {p_(pg['strict']['recall'])}", ci_(*pg["strict"]["precision_ci_page"]),
                       n_(pg.get("n_kim"))])
    for name in ("LucVanTien1883", "KimVanKieu1884"):
        c, _ = cross_summary(name)
        for ref, blk in ((c or {}).get("refs") or {}).items():
            at = blk.get("all_tiers") or {}
            if "ocr_eq_ref_all_pct" in at:
                km.append([name, "thạch bản", "**ƯỚC LƯỢNG**", f"kim ở ô = chữ dị bản `{ref}` (mọi tầng)",
                           f"{at['ocr_eq_ref_all_pct']:.1f} %".replace(".", ","), "", n_(at.get("n_cells"))])
    km.append(["STT2/4/11, Chrestomathie1872", "viết tay / sách in", "—", "không có nhãn người -> không đo", NA, "", ""])
    L += [table(["bộ", "loại bản", "mức chắc", "đại lượng", "giá trị", "CI 95 %", "n"], km, "lllllrr"), ""]
    split_rows = []
    for s8, name in (("B18", "SachKinhThayCaBinh"), ("B34", "SachDungLyHoThan")):
        sp = borg_cnt_split(name, cnt)
        if not sp:
            continue
        for key, lab in (("1", "CNT = 1 (số chữ OCR cột ≠ số âm QN)"), ("0", "CNT = 0")):
            if key in sp:
                k, n = sp[key]
                lo, hi = wilson(k, n)
                split_rows.append([name, lab, n_(n), p_(k / n if n else None), ci_(lo, hi)])
    if split_rows:
        L += ["**Cờ CNT của gold_exact trên chữ viết tay (ĐO, Borg).** GOLD có chữ người, tỉ lệ đúng V1+ tách theo cờ `cnt` "
              "(`measure_out/<Borg>/borg_endtoend/cells.csv` ⋈ `dataset/_ALL/gold_exact.csv`, Wilson):", "",
              table(["bộ", "nhóm", "n", "đúng V1+", "CI 95 %"], split_rows), "",
              "Gần như mọi ô GOLD Borg mang CNT = 1 (cột viết tay hiếm khi đếm bằng), nên cờ này chủ yếu dồn ô sang "
              "`text_only` chứ không tách được ô sai khỏi ô đúng; so hai nhóm ở bảng trên (nhóm CNT = 0 rất nhỏ).", ""]

    # ---------- 4. bất biến
    L += ["## 4. Bất biến (invariants)", ""]
    ms = jload(REPO / "measure_out/SUMMARY.json", "bộ đo measure.py")
    if ms:
        t = ms.get("totals") or {}
        L += [f"`measure_out/SUMMARY.json` ({ms.get('generated_at')}, `{ms.get('cli')}`): **{t.get('inv_pass')} PASS · "
              f"{t.get('inv_fail')} FAIL · {t.get('inv_soft_fail')} FAIL mềm · {t.get('inv_skip')} SKIP** trên "
              f"{t.get('steps')} bước (sập {t.get('crashed')}).", ""]
        rr = [[r.get("step"), r.get("book"), r.get("rc"), r.get("n_pass"), r.get("n_fail"), r.get("n_soft_fail"),
               r.get("n_skip"), f"{r.get('seconds', 0):.0f}"] for r in ms.get("results") or []]
        L += [table(["bước", "bộ", "mã", "PASS", "FAIL", "mềm", "SKIP", "giây"], rr, "llrrrrrr"), ""]
    inv_rows = []
    if ge_sum:
        inv = ge_sum.get("invariants") or []
        inv_rows.append(["gold_exact_eval (bản giao dataset/_ALL/gold_exact.csv)",
                         f"{sum(1 for i in inv if i.get('pass') is True)}/{len(inv)}",
                         sum(1 for i in inv if i.get("pass") is False)])
    for name, sub in (("LucVanTien1916", "ihr_endtoend"), ("TruyenKieu1872", "ihr_endtoend"),
                      ("SachKinhThayCaBinh", "borg_endtoend"), ("SachDungLyHoThan", "borg_endtoend")):
        p = REPO / "measure_out" / name / sub / "summary.json"
        if p.exists():
            inv = json.loads(p.read_text(encoding="utf-8")).get("invariants") or []
            inv_rows.append([f"{sub} {name}", f"{sum(1 for i in inv if i.get('pass') is True)}/{len(inv)}",
                             sum(1 for i in inv if i.get("pass") is False)])
    if inv_rows:
        L += [table(["phép đo", "PASS", "FAIL"], inv_rows), ""]

    # ---------- 5. thời gian
    L += ["## 5. Thời gian chạy", ""]
    tsv = Path(a.timing) if a.timing else max((REPO / "logs").glob("clean_rebuild_*_thoi_gian.tsv"), default=None)
    trows = timing_rows(tsv)
    if trows:
        L += [f"Lượt `clean_rebuild_all.sh --run` (`{tsv.relative_to(REPO) if tsv.is_absolute() else tsv}`):", "",
              table(["bước", "lệnh", "mã", "thời gian", "bắt đầu", "kết thúc", "log"], trows, "llrrlll"), ""]
    else:
        L += ["Chưa có lượt `clean_rebuild_all.sh --run` (logs/clean_rebuild_*_thoi_gian.tsv) — dưới đây là lượt GẦN NHẤT "
              "của từng bộ theo `CHECKSUMS.txt` (dòng `thoi_gian`).", ""]
    tb = []
    for s8, name, bset, _, _ in SETS:
        if s8 in ("stt4", "stt11"):
            continue
        stt_ = bset == "SachThanhTruyen"
        ck = (REPO / "dataset_out/CHECKSUMS.txt") if stt_ else book_ds_out(bset) / "CHECKSUMS.txt"
        blk = last_block(ck)
        lab = "STT (3 quyển chung)" if stt_ else name
        if not blk:
            tb.append([lab, "", NA, NA, ""])
            continue
        t_start = run_start("STT" if stt_ else bset, blk[0][0])
        steps = [(k, v) for _, k, v in blk]
        if t_start is not None:           # bước đầu tính lại từ giờ bắt đầu log (tick 'setup' cộng dồn dưới --book all)
            steps[0] = (steps[0][0], int(_ts(blk[0][0]) - t_start))
            total = _ts(blk[-1][0]) - t_start
        else:
            total = None
        note = ""
        if not stt_:
            try:
                g = json.loads((REPO / ("prepared/_auto" if bset in ("SachKinhThayCaBinh", "SachDungLyHoThan") else "prepared")
                                / bset / "manifest.json").read_text(encoding="utf-8")).get("gates") or {}
                calls = int(g.get("ocr_calls", 0) or 0) + int(g.get("n_pages_new_kim", 0) or 0)
                if calls:
                    note = f"lượt nạp này GỌI API kim {calls} trang (lượt từ cache sẽ ngắn hơn nhiều)"
            except Exception:  # noqa: BLE001
                pass
        tb.append([lab, blk[0][0] if t_start is None else time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
                   " · ".join(f"{k} {v}s" for k, v in steps), fmt_s(total) if total is not None else
                   f"≥ {fmt_s(sum(v for _, v in steps[1:]))} (không thấy log bắt đầu)", note])
    mg = t0 = None
    tg = REPO / "dataset/_ALL/THOI_GIAN.txt"
    if tg.exists():
        for ln in tg.read_text(encoding="utf-8").splitlines():
            if "thoi_gian  merge" in ln:
                t0, mg = ln.split()[0], int(ln.split()[-1][:-1])
    tb.append(["gộp dataset/_ALL", t0 or "", f"merge {mg}s" if mg is not None else NA, fmt_s(mg), ""])
    gl = max((REPO / "logs").glob("run_GOLD_EXACT_*.log"), default=None)
    if gl:
        tail = [ln for ln in gl.read_text(encoding="utf-8", errors="replace").splitlines() if ln.startswith("# mã thoát")]
        sec = tail[-1].rsplit("·", 1)[-1].strip().rstrip("s") if tail else None
        tb.append(["GOLD chính xác (B8)", gl.stem[len("run_GOLD_EXACT_"):], tail[-1][2:] if tail else NA,
                   fmt_s(int(sec)) if sec and sec.isdigit() else NA, ""])
    L += [table(["bộ / bước", "bắt đầu", "các bước (giây)", "tổng", "ghi chú"], tb, "lllll"), ""]

    # ---------- 6. giới hạn
    L += ["## 6. Giới hạn", "",
          "1. Độ chính xác chỉ **ĐO** được ở 2 bộ mộc bản IHR (LucVanTien1916, TruyenKieu1872) và 2 bộ chép tay Borg "
          "(SachKinhThayCaBinh, SachDungLyHoThan) — đều là TẬP ĐÁNH GIÁ, không giao nộp. 3 bộ STT và Chrestomathie1872 (giao "
          "nộp) chỉ có **SUY ĐOÁN**; LucVanTien1883/KimVanKieu1884 chỉ có **ƯỚC LƯỢNG** qua dị bản (cận dưới: dị bản khác chữ "
          "hợp lệ bị tính sai).",
          "2. Borg: QN đầu vào là phiên âm NGƯỜI theo trang ⇒ số đo là **cận trên** của phương pháp trên chữ viết tay; chỉ câu "
          "\"đếm bằng\" được chấm. Kim trên chữ viết tay kém hẳn mộc bản (mục 3) — GOLD vẫn cao nhờ luật `kim ∈ R(âm)` lọc.",
          "3. Số ô `gold_exact = ok` phụ thuộc ngưỡng bộ kiểm ảnh↔chữ (`config/gold_exact.yaml`); ở Borg rất ít ô ok có chữ "
          "người (xem mục 2, n nhỏ ⇒ CI rộng).",
          "4. Bảng thời gian ghép từ nhiều lượt khi chưa có lượt `clean_rebuild_all.sh --run` trọn vẹn.", ""]
    if MISSING:
        L += ["**Nguồn thiếu khi sinh báo cáo** (các ô tương ứng ghi \"chưa có\"):", ""]
        L += [f"- {m}" for m in dict.fromkeys(MISSING)]
        L += [""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="mặc định docs/BAO_CAO_TONG_HOP_<ngày>.md")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD cho tên tệp/tiêu đề (mặc định hôm nay)")
    ap.add_argument("--timing", default=None, help="TSV thời gian của clean_rebuild_all.sh --run (mặc định bản mới nhất)")
    a = ap.parse_args(argv)
    md = build(a)
    out = Path(a.out) if a.out else REPO / "docs" / f"BAO_CAO_TONG_HOP_{a.date or time.strftime('%Y-%m-%d')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    print(f"[bao_cao_tong_hop] -> {out.relative_to(REPO) if out.is_relative_to(REPO) else out} "
          f"({len(md.splitlines())} dòng; nguồn thiếu: {len(dict.fromkeys(MISSING))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
