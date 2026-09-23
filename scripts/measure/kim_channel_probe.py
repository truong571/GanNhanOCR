#!/usr/bin/env python
"""kim_channel_probe.py — kiểm chuẩn KÊNH OCR Hán-Nôm (kimhannom HCMUS) cho 3 sách mới (2026-09-23).

CHỈ ĐỌC (không sửa pipeline/core/data). Mọi kết quả kim được cache theo (md5 ảnh, cấu hình) nên chạy lại = 0 lượt gọi.
Ngân sách gọi ghi ở measure_out/kim_channel/kim_calls.json (chặn cứng khi vượt --budget).

Ba phép đo (tương ứng nhiệm vụ I.1-I.3):
  params   1 trang/sách × N cấu hình body image-ocr (ocr_id/lang_type/reading_direction/font_type/epitaph,
           có/không bước image-preprocessing). Giá trị hợp lệ đọc từ /js/hannom.js + <select> của trang chủ:
             ocr_id           -1 Tự động · 1 Văn bản thông thường · 2 Hành chính · 3 Ngoại cảnh · 4 Y học dân tộc
                              · 5 Văn bia · 6 Kinh Phật   (web quy 4/5/6 → 1, riêng 5 bật epitaph=1)
             lang_type        0 Tự động · 1 Hán · 2 Nôm
             reading_direction 0 Tự động · 1 Dọc · 2 Ngang
             font_type        0 Tự động · 1 In · 2 Viết tay   (web: font_type==2 → ocr_id=5 → epitaph=1)
  crop     10 cột/sách: ảnh CẢ TRANG (cache sẵn, 0 lượt) vs ảnh 1 CỘT vs ảnh 1 TẦNG-CỘT.
  contrast 5 cột/sách × {none, stretch, otsu} trên ảnh cột.

Ba tiêu chí chấm (2 và 3 KHÔNG vòng tròn):
  (1) kim == nhãn labels_final  — nhãn GOLD sinh TỪ kim ở mức trang nên cấu hình mặc định ≈ 100 % theo định nghĩa;
      chỉ đọc là "đồng thuận với nhãn đang giao nộp".
  (2) kim ∈ R(âm QN)            — tỉ lệ chữ kim là một cách đọc từ điển của âm QN cùng vị trí = tỉ lệ đủ điều kiện GOLD.
  (3) kim == ref_nom            — chữ Nôm của DỊ BẢN độc lập (Kiều 1871 / LVT 1916) ở đúng vị trí âm
      (measure_out/auto_precision/cross/<book>/cells.csv). Chresto không có dị bản → chỉ (1)(2).

CLI:  .venv/bin/python scripts/measure/kim_channel_probe.py --steps params,crop,contrast --budget 120
      .venv/bin/python scripts/measure/kim_channel_probe.py --report-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = REPO / "measure_out" / "kim_channel"
CACHE = OUT / "kim_cache"
LEDGER = OUT / "kim_calls.json"

BOOKS = {
    "LucVanTien1883": dict(
        prepared=REPO / "prepared/LucVanTien1883",
        dataset=(REPO / "prepared/LucVanTien1883/dataset_out" if (REPO / "prepared/LucVanTien1883/dataset_out").exists() else REPO / "dataset_out_LucVanTien1883"),
        src_dir=REPO / "data/LucVanTien1883/nom_pages",
        cross=REPO / "measure_out/auto_precision/cross/LucVanTien1883/cells.csv",
        prod_contrast="stretch", layout="lithograph"),
    "KimVanKieu1884": dict(
        prepared=REPO / "prepared/KimVanKieu1884",
        dataset=(REPO / "prepared/KimVanKieu1884/dataset_out" if (REPO / "prepared/KimVanKieu1884/dataset_out").exists() else (REPO / "prepared/KimVanKieu1884/dataset_out" if (REPO / "prepared/KimVanKieu1884/dataset_out").exists() else REPO / "dataset_out_KimVanKieu1884_b1")),
        src_dir=REPO / "data/KimVanKieu1884/nom_pages",
        cross=REPO / "measure_out/auto_precision/cross/KimVanKieu1884/cells.csv",
        prod_contrast="otsu", layout="lithograph"),
    "Chrestomathie1872": dict(
        prepared=REPO / "prepared/Chrestomathie1872",
        dataset=(REPO / "prepared/Chrestomathie1872/dataset_out" if (REPO / "prepared/Chrestomathie1872/dataset_out").exists() else REPO / "dataset_out_Chrestomathie1872"),
        src_dir=REPO / "data/Chrestomathie1872/nom_pages",
        cross=None,
        prod_contrast="none", layout="prose"),
}

# ---- cấu hình body image-ocr ------------------------------------------------
CFG_DEFAULT = dict(ocr_id=1, lang_type=1, reading_direction=1, font_type=1)
CONFIGS = {
    # tên              body thêm/ghi đè                              preprocess
    "c0_default":     (dict(), False),
    "c1_lang_nom":    (dict(lang_type=2), False),
    "c2_epitaph":     (dict(font_type=2, epitaph=1), False),
    "c3_preprocess":  (dict(), True),
}


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def md5_file(p: Path) -> str:
    return md5_bytes(Path(p).read_bytes())


class Ledger:
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


# --------------------------------------------------------------------------- #
# gọi API (dùng lại xác thực/Guest Mode của core.ocr.ocr_api, KHÔNG sửa tệp đó)
# --------------------------------------------------------------------------- #
def _post(path: str, body: dict):
    import requests
    from core.ocr import ocr_api

    url = f"https://{ocr_api._SN_DOMAIN}{path}"

    def do(token: str = ""):
        h = ocr_api._base_headers()
        h["Content-Type"] = "application/json; charset=utf-8"
        if token:
            h["Authorization"] = f"Bearer {token}"
        return requests.post(url, json=body, headers=h, verify=False, timeout=90)

    resp = ocr_api._authed_request(do, path.rsplit("/", 1)[-1])
    if resp is None:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def kim_call(image: Path, cfg_name: str, ledger: Ledger, tag: str) -> dict:
    """1 lượt kim với cấu hình `cfg_name` trên `image`. Cache theo (md5 ảnh, cfg_name)."""
    key = f"{md5_file(image)}__{cfg_name}"
    cp = CACHE / f"{key}.json"
    if cp.exists():
        d = json.loads(cp.read_text(encoding="utf-8"))
        d["from_cache"] = True
        return d
    if not ledger.can():
        return {"status": "no_budget", "boxes": None, "from_cache": False}
    import urllib3
    urllib3.disable_warnings()
    from core.ocr import ocr_api

    extra, preprocess = CONFIGS[cfg_name]
    ledger.add(f"{tag} [{cfg_name}] {image.name}")
    fname = ocr_api.upload_image(str(image))
    boxes, note = None, ""
    if fname:
        use = fname
        if preprocess:
            pre = _post("/api/web/clc-sinonom/image-preprocessing", {"file_name": fname})
            if pre and pre.get("is_success"):
                use = pre["data"].get("new_file_name") or fname
                note = "preprocessed"
            else:
                note = f"preprocess_fail:{(pre or {}).get('message')}"
        body = dict(CFG_DEFAULT, file_name=use, **extra)
        res = _post("/api/web/clc-sinonom/image-ocr", body)
        if res and res.get("is_success"):
            boxes = res["data"]["details"]["details"]
        else:
            note += f" ocr_fail:{(res or {}).get('message')}"
    d = {"status": "ok" if boxes is not None else "api_fail", "image": str(image), "cfg": cfg_name,
         "body": dict(CFG_DEFAULT, **extra), "preprocess": preprocess, "note": note, "boxes": boxes,
         "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    CACHE.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    d["from_cache"] = False
    return d


# --------------------------------------------------------------------------- #
# dữ liệu tham chiếu
# --------------------------------------------------------------------------- #
def is_cjk(c: str) -> bool:
    o = ord(c)
    return (0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF
            or 0x20000 <= o <= 0x3134F or 0xE000 <= o <= 0xF8FF or 0xF0000 <= o <= 0x10FFFD)


def boxes_to_chars(boxes):
    """Hộp kim → chuỗi chữ theo thứ tự đọc DỌC (trên→dưới theo đỉnh hộp); chỉ ký tự CJK."""
    if not boxes:
        return []
    bs = sorted(boxes, key=lambda b: min(p[1] for p in b.get("points", [[0, 0]])))
    s = "".join(b.get("transcription", "") or "" for b in bs)
    return [c for c in unicodedata.normalize("NFC", s) if is_cjk(c)]


_D = {}


def dicts():
    if not _D:
        from core.text.dictionary import load_qn_to_nom
        from core.text.text_utils import normalize_tone_marks
        q2n = load_qn_to_nom(str(REPO / "Dict/QuocNgu_SinoNom.csv"))
        R = {}
        for k, v in q2n.items():
            R.setdefault(normalize_tone_marks(k.lower()), set()).update(v)
        _D["R"] = R
        _D["norm"] = normalize_tone_marks
    return _D


def canon(s: str) -> str:
    D = dicts()
    return D["norm"](unicodedata.normalize("NFC", (s or "").strip().lower()))


def pnum(v) -> int:
    """'page_0007' | '7' → 7."""
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else -1


def load_cells(book: str) -> dict:
    """(page, column, syl_idx) → hàng labels_final (+ ref_nom từ cross nếu có)."""
    cfg = BOOKS[book]
    cells = {}
    with open(cfg["dataset"] / "labels_final.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                k = (pnum(r["page"]), int(r["column"]), int(r["syl_idx"]))
            except (ValueError, KeyError):
                continue
            cells[k] = r
    if cfg["cross"] and Path(cfg["cross"]).exists():
        with open(cfg["cross"], encoding="utf-8") as f:
            for r in csv.DictReader(f):
                k = (pnum(r["page"]), int(r["column"]), int(r["syl_idx"]))
                if k in cells and r.get("ref") == (("Kieu1871_LVD") if book == "KimVanKieu1884" else "LVT1916_NF"):
                    cells[k]["ref_nom"] = r.get("ref_nom", "")
                    cells[k]["ref_pua"] = r.get("ref_pua", "")
    return cells


def page_cache(book: str, page: int) -> dict:
    p = BOOKS[book]["prepared"] / "detected" / f"page_{page:04d}_ocr_cache.json"
    return json.loads(p.read_text(encoding="utf-8"))


def kim_page_image(book: str, page: int) -> Path:
    """Ảnh mà kim ĐÃ dùng ở mức trang (LVT/KVK: prepared png; Chresto: jpg gốc)."""
    d = page_cache(book, page)
    src = d.get("kim_source_image") or d.get("image")
    return REPO / src


def col_geom(book: str, page: int, column: int):
    """(x0, x1, tiers) của cột từ hộp kim trong cache trang; tiers = [(y0,y1), ...] hoặc [] (prose)."""
    d = page_cache(book, page)
    cols = d["columns"]
    if column - 1 >= len(cols) or not cols[column - 1]:
        return None
    bb = [c["bbox"] for c in cols[column - 1]]
    x0, x1 = min(b[0] for b in bb), max(b[2] for b in bb)
    tiers = [tuple(t) for t in d.get("tiers") or []]
    split = d.get("tier_split")
    nt = split[column - 1][0] if split and column - 1 < len(split) else None
    return dict(x0=int(x0), x1=int(x1), tiers=tiers, n_top=nt,
                y0=min(b[1] for b in bb), y1=max(b[3] for b in bb))


# --------------------------------------------------------------------------- #
# chấm điểm
# --------------------------------------------------------------------------- #
def nw_align(kim: list[str], gold: list[str]) -> list[str | None]:
    """Căn chỉnh đơn điệu kim ↔ N ô (match = kim[j] == gold[i]); trả chữ kim cho từng ô."""
    n, m = len(gold), len(kim)
    NEG = -10 ** 9
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0], bt[i][0] = -i, "up"
    for j in range(1, m + 1):
        dp[0][j], bt[0][j] = -j, "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sc = 2 if kim[j - 1] == gold[i - 1] else 0
            best, arg = dp[i - 1][j - 1] + sc, "diag"
            if dp[i - 1][j] - 1 > best:
                best, arg = dp[i - 1][j] - 1, "up"
            if dp[i][j - 1] - 1 > best:
                best, arg = dp[i][j - 1] - 1, "left"
            dp[i][j], bt[i][j] = best, arg
    out = [None] * n
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


def score(kim_chars: list[str], rows: list[dict]) -> dict:
    """rows = ô theo thứ tự đọc (có syllable / label / ref_nom). Trả 3 tiêu chí."""
    D = dicts()
    N = len(rows)
    st = dict(N=N, n_kim=len(kim_chars), count_ok=int(len(kim_chars) == N),
              eq_label=0, n_label=0, in_R=0, n_R=0, eq_ref=0, n_ref=0)
    assigned = ([kim_chars[i] if i < len(kim_chars) else None for i in range(N)]
                if len(kim_chars) == N else nw_align(kim_chars, [r.get("label") or "" for r in rows]))
    for r, k in zip(rows, assigned):
        lab = (r.get("label") or "").strip()
        if lab:
            st["n_label"] += 1
            st["eq_label"] += int(k == lab)
        syl = canon(r.get("syllable") or "")
        if syl:
            st["n_R"] += 1
            st["in_R"] += int(bool(k) and k in D["R"].get(syl, set()))
        ref = (r.get("ref_nom") or "").strip()
        if ref and r.get("ref_pua") not in ("1", "True", "true"):
            st["n_ref"] += 1
            st["eq_ref"] += int(k == ref)
    return st


def agg(sts: list[dict]) -> dict:
    o = dict(n_units=len(sts))
    for k in ("N", "n_kim", "count_ok", "eq_label", "n_label", "in_R", "n_R", "eq_ref", "n_ref"):
        o[k] = sum(s[k] for s in sts)
    o["count_ok_pct"] = round(100 * o["count_ok"] / o["n_units"], 1) if o["n_units"] else None
    o["eq_label_pct"] = round(100 * o["eq_label"] / o["n_label"], 1) if o["n_label"] else None
    o["in_R_pct"] = round(100 * o["in_R"] / o["n_R"], 1) if o["n_R"] else None
    o["eq_ref_pct"] = round(100 * o["eq_ref"] / o["n_ref"], 1) if o["n_ref"] else None
    return o


# --------------------------------------------------------------------------- #
# chọn mẫu
# --------------------------------------------------------------------------- #
def pick_columns(book: str, n: int) -> list[tuple[int, int]]:
    """n cột 'nhãn GOLD ổn định', rải đều theo trang: M==N, count_source != conflict, >= 70 % ô GOLD,
    không ô QUARANTINE, N >= 5; ưu tiên cột có nhiều ô ref_nom (dị bản) để dùng được tiêu chí (3)."""
    cells = load_cells(book)
    by_col = defaultdict(list)
    for (p, c, s), r in cells.items():
        by_col[(p, c)].append((s, r))
    cand = []
    for (p, c), lst in by_col.items():
        lst.sort()
        rows = [r for _, r in lst]
        if len(rows) < 5:
            continue
        if any((r.get("tier") or "") == "QUARANTINE" for r in rows):
            continue
        if any(r.get("count_source") == "conflict" for r in rows):
            continue
        if rows[0].get("n_ocr") != rows[0].get("n_qn"):
            continue
        n_gold = sum(1 for r in rows if (r.get("tier") or "") == "GOLD")
        if n_gold < 0.7 * len(rows):
            continue
        n_ref = sum(1 for r in rows if (r.get("ref_nom") or "").strip())
        cand.append((p, c, len(rows), n_gold, n_ref))
    if not cand:
        return []
    cand.sort(key=lambda t: (-t[4], -t[3], t[0], t[1]))
    top = cand[: max(n * 8, n)]
    top.sort(key=lambda t: (t[0], t[1]))
    if len(top) <= n:
        return [(t[0], t[1]) for t in top]
    idx = sorted({round(i * (len(top) - 1) / (n - 1)) for i in range(n)})
    return [(top[i][0], top[i][1]) for i in idx]


def col_rows(book: str, page: int, column: int) -> list[dict]:
    cells = load_cells(book)
    rows = [(s, r) for (p, c, s), r in cells.items() if p == page and c == column]
    rows.sort()
    return [r for _, r in rows]


def pick_page(book: str) -> int:
    """Trang có nhiều ô GOLD + ref nhất (cho bước params)."""
    cells = load_cells(book)
    sc = Counter()
    for (p, c, s), r in cells.items():
        if (r.get("tier") or "") == "GOLD":
            sc[p] += 1 + (2 if (r.get("ref_nom") or "").strip() else 0)
    return sc.most_common(1)[0][0] if sc else 1


# --------------------------------------------------------------------------- #
# ảnh
# --------------------------------------------------------------------------- #
def page_variant(book: str, page: int, contrast: str) -> Path:
    """Trang gốc → PNG mode L với phép kéo nền của ingest (none|stretch|otsu). Cache ở OUT/img/."""
    import cv2
    d = page_cache(book, page)
    src = REPO / (d.get("kim_source_image") or d.get("image"))
    if src.suffix.lower() == ".png" and "prepared" in str(src):
        # LVT/KVK: kim chạy trên prepared png -> ảnh gốc là jpg cùng tên trong data/
        man = json.loads((BOOKS[book]["prepared"] / "manifest.json").read_text(encoding="utf-8"))
        rec = next((r for r in man.get("pages", []) if r.get("page") == page), None)
        cand = BOOKS[book]["src_dir"] / (rec or {}).get("file", "")
        if cand.is_file():
            src = cand
    dst = OUT / "img" / f"{book}_p{page:04d}_{contrast}.png"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    from pipeline.tools.ingest_lithograph_book import otsu_gray, stretch_gray
    gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    out = {"stretch": stretch_gray, "otsu": otsu_gray}.get(contrast, lambda g: g)(gray)
    cv2.imwrite(str(dst), out)
    return dst


def crop(src: Path, box, dst: Path, pad: int = 8) -> Path:
    import cv2
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    h, w = im.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
    x1, y1 = min(w, int(x1) + pad), min(h, int(y1) + pad)
    cv2.imwrite(str(dst), im[y0:y1, x0:x1])
    return dst


# --------------------------------------------------------------------------- #
# các bước
# --------------------------------------------------------------------------- #
def step_params(ledger: Ledger, books: list[str]) -> dict:
    res = {}
    for book in books:
        page = pick_page(book)
        img = kim_page_image(book, page)
        cells = load_cells(book)
        d = page_cache(book, page)
        cols = d["columns"]
        rows_by_col = {}
        for c in range(1, len(cols) + 1):
            rr = [(s, r) for (p, cc, s), r in cells.items() if p == page and cc == c]
            rr.sort()
            rows_by_col[c] = [r for _, r in rr]
        book_res = dict(page=page, image=str(img.relative_to(REPO)), n_cols=len(cols), configs={})
        # struct-classification: server tự đoán ocr_id/lang_type/reading_direction
        sc_path = CACHE / f"{md5_file(img)}__structclass.json"
        if sc_path.exists():
            book_res["structure_classification"] = json.loads(sc_path.read_text(encoding="utf-8"))
        elif ledger.can():
            CACHE.mkdir(parents=True, exist_ok=True)
            import urllib3
            urllib3.disable_warnings()
            from core.ocr import ocr_api
            ledger.add(f"{book} structure-classification p{page}")
            fn = ocr_api.upload_image(str(img))
            r = _post("/api/web/clc-sinonom/structure-classification",
                      {"file_name": fn, "lang_type": 0, "ocr_id": -1, "reading_direction": 0}) if fn else None
            got = {}
            if r and r.get("is_success"):
                dd = r["data"]
                got = {k: dd.get(k) for k in ("ocr_id", "lang_type", "reading_direction", "is_sino_nom")}
                got["has_result_bbox"] = bool((dd.get("result") or {}).get("result_bbox"))
            else:
                got = {"error": (r or {}).get("message")}
            sc_path.write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
            book_res["structure_classification"] = got
        for name in CONFIGS:
            dd = kim_call(img, name, ledger, f"{book} params p{page}")
            if dd["status"] != "ok":
                book_res["configs"][name] = dict(status=dd["status"], note=dd.get("note", ""))
                continue
            # gán hộp vào cột bằng đúng hàm của adapter khi là thạch bản; prose → theo x gần nhất
            per_col = assign_to_columns(book, page, dd["boxes"])
            sts = []
            for c, rows in rows_by_col.items():
                if not rows:
                    continue
                sts.append(score(per_col.get(c, []), rows))
            book_res["configs"][name] = dict(status="ok", n_boxes=len(dd["boxes"]),
                                             n_chars=sum(len(boxes_to_chars([b])) for b in dd["boxes"]),
                                             from_cache=dd.get("from_cache"), **agg(sts))
        res[book] = book_res
    return res


def assign_to_columns(book: str, page: int, boxes) -> dict[int, list[str]]:
    """Hộp kim toàn trang → chữ theo cột, dùng lại hình học cột của cache trang (tâm x gần nhất, tầng theo y)."""
    d = page_cache(book, page)
    cols = d["columns"]
    centers = []
    for i, col in enumerate(cols, start=1):
        if not col:
            continue
        xs = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in col]
        centers.append((i, sum(xs) / len(xs), min(b["bbox"][0] for b in col), max(b["bbox"][2] for b in col)))
    tiers = d.get("tiers") or []
    out: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for b in boxes:
        pts = b.get("points") or []
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2
        y0, y1 = min(ys), max(ys)
        chars = [c for c in unicodedata.normalize("NFC", b.get("transcription", "") or "") if is_cjk(c)]
        if not chars or not centers:
            continue
        h = (y1 - y0) / len(chars)
        i, _, _, _ = min(centers, key=lambda t: abs(t[1] - cx))
        for j, ch in enumerate(chars):
            cy = y0 + h * (j + 0.5)
            if tiers:
                (a0, a1), (b0, b1) = tiers[0], tiers[-1]
                m = 0.5 * (b0 - a1) if b0 > a1 else 0
                if cy < a0 - m or cy > b1 + m:
                    continue            # số câu in / rác ngoài hai tầng
            out[i].append((cy, ch))
    return {k: [c for _, c in sorted(v)] for k, v in out.items()}


def step_tierdp(books: list[str]) -> dict:
    """(II) Rào TẦNG cho DP văn bản: hiện `realign_column` chạy trên CẢ CỘT (6⧺8 nối liền) nên một
    khe ở câu lục có thể kéo sang câu bát. Đo mức phơi nhiễm: số ô bị ghép XUYÊN tầng, và số ô đổi
    cặp khi chạy DP RIÊNG từng tầng (dùng chính tách tầng hình học của kim: cache `tier_split`).
    0 lượt gọi API."""
    import glob
    from core.text.dictionary import load_qn_to_nom, load_similarity_dict
    from pipeline.align_engine.anchor_align import matched_pairs, realign_column
    R = load_qn_to_nom(str(REPO / "Dict/QuocNgu_SinoNom.csv"))
    SIM = load_similarity_dict(str(REPO / "Dict/SinoNom_Similar.csv"))
    res = {}
    for book in books:
        if BOOKS[book]["layout"] != "lithograph":
            continue
        refname = "Kieu1871_LVD" if book == "KimVanKieu1884" else "LVT1916_NF"
        ref = {}
        cp = BOOKS[book]["cross"]
        if cp and Path(cp).exists():
            with open(cp, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["ref"] == refname and (r.get("ref_nom") or "") and r.get("ref_pua") != "1":
                        ref[(pnum(r["page"]), int(r["column"]), int(r["syl_idx"]))] = r["ref_nom"]
        c = Counter()
        for pth in sorted(glob.glob(str(BOOKS[book]["prepared"] / "detected" / "page_*_ocr_cache.json"))):
            page = pnum(Path(pth).name)
            d = json.loads(Path(pth).read_text(encoding="utf-8"))
            tj = BOOKS[book]["prepared"] / "transcriptions" / f"page_{page:04d}.json"
            if not tj.exists():
                continue
            T = json.loads(tj.read_text(encoding="utf-8"))
            split = d.get("tier_split") or []
            for k, (col, tcol) in enumerate(zip(d["columns"], T["columns"])):
                if k >= len(split):
                    continue
                nt, nb = split[k]
                syl, lo = tcol["syllables"], tcol["len_odd"]
                if len(col) != nt + nb or not (0 < lo < len(syl)):
                    continue
                c["cols"] += 1
                c["cells"] += len(col)
                pa = {m["nom_idx"]: m["syl_idx"] for m in matched_pairs(realign_column(col, syl, R, SIM))}
                cross = sum(1 for i, j in pa.items() if (i < nt) != (j < lo))
                c["cols_cross"] += int(cross > 0)
                c["cells_cross"] += cross
                pb = {m["nom_idx"]: m["syl_idx"] for m in matched_pairs(realign_column(col[:nt], syl[:lo], R, SIM))}
                pb.update({m["nom_idx"] + nt: m["syl_idx"] + lo
                           for m in matched_pairs(realign_column(col[nt:], syl[lo:], R, SIM))})
                for i in set(pa) | set(pb):
                    if pa.get(i) == pb.get(i):
                        continue
                    c["cells_diff"] += 1
                    ch = col[i]["char"]
                    ra = ref.get((page, k + 1, pa.get(i))) if pa.get(i) is not None else None
                    rb = ref.get((page, k + 1, pb.get(i))) if pb.get(i) is not None else None
                    if ra or rb:
                        c["diff_with_ref"] += 1
                        c["diff_col_eq_ref"] += int(ra == ch)
                        c["diff_tier_eq_ref"] += int(rb == ch)
        res[book] = dict(c)
    return res


def step_lang(ledger: Ledger, books: list[str], n_pages: int) -> dict:
    """Mở rộng phát hiện lang_type trên nhiều trang: c0 = hộp kim_raw CÓ SẴN (0 lượt), c1 = 1 lượt/trang."""
    res = {}
    for book in books:
        cells = load_cells(book)
        sc = Counter()
        for (p, c, s), r in cells.items():
            sc[p] += (1 if (r.get("tier") or "") == "GOLD" else 0) + (2 if (r.get("ref_nom") or "").strip() else 0)
        pages = [p for p, _ in sc.most_common(n_pages)]
        per = {"c0_default": [], "c1_lang_nom": []}
        for page in sorted(pages):
            rows_by_col = defaultdict(list)
            for (p, c, sidx), r in cells.items():
                if p == page:
                    rows_by_col[c].append((sidx, r))
            rows_by_col = {c: [r for _, r in sorted(v)] for c, v in rows_by_col.items()}
            raw = BOOKS[book]["prepared"] / "kim_raw" / f"page_{page:04d}.json"
            srcs = {}
            if raw.exists():
                d = json.loads(raw.read_text(encoding="utf-8"))
                srcs["c0_default"] = d.get("boxes") if isinstance(d, dict) else d
            img = kim_page_image(book, page)
            dd = kim_call(img, "c1_lang_nom", ledger, f"{book} lang p{page}")
            if dd["status"] == "ok":
                srcs["c1_lang_nom"] = dd["boxes"]
            for name, boxes in srcs.items():
                if boxes is None:
                    continue
                pc = assign_to_columns(book, page, boxes)
                for c, rows in rows_by_col.items():
                    if rows:
                        per[name].append(score(pc.get(c, []), rows))
        res[book] = dict(pages=sorted(pages), variants={k: agg(v) for k, v in per.items() if v})
    return res


def step_crop(ledger: Ledger, books: list[str], n_cols: int) -> dict:
    res = {}
    for book in books:
        picks = pick_columns(book, n_cols)
        lvl = {"page": [], "column": [], "tier": []}
        detail = []
        for (page, column) in picks:
            rows = col_rows(book, page, column)
            if not rows:
                continue
            g = col_geom(book, page, column)
            if not g:
                continue
            img = kim_page_image(book, page)
            # (a) cả trang — từ cache có sẵn, 0 lượt
            d = page_cache(book, page)
            page_chars = [c["char"] for c in d["columns"][column - 1]]
            s_page = score(page_chars, rows)
            lvl["page"].append(s_page)
            # (b) 1 cột
            cimg = crop(img, (g["x0"], g["tiers"][0][0] if g["tiers"] else g["y0"],
                              g["x1"], g["tiers"][-1][1] if g["tiers"] else g["y1"]),
                        OUT / "img" / f"{book}_p{page:04d}_c{column}_col.png")
            dd = kim_call(cimg, "c0_default", ledger, f"{book} col p{page}c{column}")
            s_col = score(boxes_to_chars(dd.get("boxes")), rows) if dd["status"] == "ok" else None
            if s_col:
                lvl["column"].append(s_col)
            # (c) 1 tầng-cột (chỉ thạch bản có 2 tầng)
            s_tier = None
            if g["tiers"] and g["n_top"]:
                nt = g["n_top"]
                timg = crop(img, (g["x0"], g["tiers"][0][0], g["x1"], g["tiers"][0][1]),
                            OUT / "img" / f"{book}_p{page:04d}_c{column}_t0.png")
                dt = kim_call(timg, "c0_default", ledger, f"{book} tier p{page}c{column}")
                if dt["status"] == "ok":
                    s_tier = score(boxes_to_chars(dt.get("boxes")), rows[:nt])
                    lvl["tier"].append(s_tier)
            detail.append(dict(page=page, column=column, N=s_page["N"],
                               page_n_kim=s_page["n_kim"], col_n_kim=(s_col or {}).get("n_kim"),
                               tier_n_kim=(s_tier or {}).get("n_kim")))
        res[book] = dict(picks=[list(p) for p in picks], levels={k: agg(v) for k, v in lvl.items() if v},
                         detail=detail)
    return res


def step_contrast(ledger: Ledger, books: list[str], n_cols: int) -> dict:
    res = {}
    for book in books:
        picks = pick_columns(book, n_cols)[:5]
        per = {v: [] for v in ("none", "stretch", "otsu")}
        for (page, column) in picks:
            rows = col_rows(book, page, column)
            g = col_geom(book, page, column)
            if not rows or not g:
                continue
            box = (g["x0"], g["tiers"][0][0] if g["tiers"] else g["y0"],
                   g["x1"], g["tiers"][-1][1] if g["tiers"] else g["y1"])
            for v in per:
                pimg = page_variant(book, page, v)
                cimg = crop(pimg, box, OUT / "img" / f"{book}_p{page:04d}_c{column}_col_{v}.png")
                dd = kim_call(cimg, "c0_default", ledger, f"{book} contrast[{v}] p{page}c{column}")
                if dd["status"] == "ok":
                    per[v].append(score(boxes_to_chars(dd.get("boxes")), rows))
        res[book] = dict(prod_contrast=BOOKS[book]["prod_contrast"],
                         variants={k: agg(v) for k, v in per.items() if v})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="params,crop,contrast")
    ap.add_argument("--books", default=",".join(BOOKS))
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--n-cols", type=int, default=10)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(LEDGER, 0 if a.report_only else a.budget)
    books = [b for b in a.books.split(",") if b in BOOKS]
    steps = a.steps.split(",")
    sp = OUT / "SUMMARY.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    t0 = time.time()
    if "params" in steps:
        summary["params"] = step_params(ledger, books)
        sp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if "crop" in steps:
        summary["crop"] = step_crop(ledger, books, a.n_cols)
        sp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if "contrast" in steps:
        summary["contrast"] = step_contrast(ledger, books, 5)
    if "lang" in steps:
        summary["lang"] = step_lang(ledger, books, 5)
    if "tier_dp" in steps:
        summary["tier_dp"] = step_tierdp(books)
    summary["meta"] = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), calls_total=ledger.calls,
                           budget=a.budget, t_s=round(time.time() - t0, 1),
                           configs={k: dict(body=dict(CFG_DEFAULT, **v[0]), preprocess=v[1])
                                    for k, v in CONFIGS.items()})
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary.get("meta"), ensure_ascii=False))
    print(f"[kim_channel] lượt gọi tích luỹ = {ledger.calls}/{a.budget}; SUMMARY → {sp}")


if __name__ == "__main__":
    main()
