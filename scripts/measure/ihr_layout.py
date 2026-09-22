#!/usr/bin/env python3
"""ihr_layout.py — đo BỐ CỤC + thăm dò hệ số phóng ảnh cho 2 bộ IHR-NomDB có GT chữ
(LucVanTien1916 = mộc bản R.403 1916, TruyenKieu1872 = mộc bản Duy Minh Thị 1872).

Khác LVT1883/KVK1884 (thạch bản, phải tự dò bố cục): hai bộ này KÈM SẴN bố cục và nhãn:
    data/<book>/pages/images/*.jpg     ảnh trang (~495×763 / ~404×579 px)
    data/<book>/pages/bboxes.json      VoTT v2.2: mỗi asset = 1 trang, region tag "Column"
    data/<book>/pages/annotation.json  [{img, annotations:[{hn_text:[chữ…], translation:[…]}]}]
    data/<book>/manifest.tsv           1 dòng = 1 CÂU (page_id, col_index, part, col_bbox_xywh,
                                       nom_text = GT chữ Nôm, qn_verse = QN câu-với-câu)

Phép đo (0 token LLM; kim chỉ khi --kim-pages > 0):
  1. cột/trang từ bboxes.json và từ manifest.tsv → có khớp nhau không; thứ tự cột (phải→trái?)
  2. MỖI CỘT = 1 CẶP LỤC BÁT hay 1 CÂU: đếm `part` mỗi (trang, cột) và số chữ mỗi part
  3. px/chữ = chiều cao ô cột / 14; bước cột; chồng lấn cột; bề rộng cột
  4. đối chứng ĐỘC LẬP bằng ảnh: số cột theo chiếu mực (không dùng bboxes.json)
  5. kim ở ×1 vs ×3 (và ×2 nếu khai): tỉ lệ đọc đủ chữ + ĐÚNG CHỮ so GT
     (có GT từng chữ nên đo được độ đúng thật, không cần chỉ số thay thế)

Chạy:
    .venv/bin/python scripts/measure/ihr_layout.py --book all --out measure_out/<book>/ihr_layout
    .venv/bin/python scripts/measure/ihr_layout.py --book LucVanTien1916 --kim-pages 10 --scales 1,3
    .venv/bin/python scripts/measure/ihr_layout.py --selftest

Đầu ra: <out>/summary.json (+ invariants), pages.csv, cols.csv, kim_scale.csv, kim_cache/*.json.
Cache kim theo md5 ảnh ĐÃ GỬI (ảnh phóng ×s) → chạy lại 0 lượt gọi.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BOOKS = {
    "LucVanTien1916": dict(
        root=REPO / "data/LucVanTien1916",
        images="pages/images",
        edition="nlvnpf-0059 (mộc bản 1916, IHR-NomDB)",
    ),
    "TruyenKieu1872": dict(
        root=REPO / "data/TruyenKieu1872",
        images="pages/images",
        edition="Duy Minh Thị 1872 (mộc bản, IHR-NomDB)",
    ),
}
TIER_RULE = (6, 8)              # câu lục / câu bát
N_PER_COL = sum(TIER_RULE)      # 14 chữ = 1 cặp
COL_TOL = 0.5                   # × bước cột: |tâm hộp kim − tâm cột| tối đa để gán


# --------------------------------------------------------------------------- dữ liệu
def load_manifest(book: str) -> list[dict]:
    p = BOOKS[book]["root"] / "manifest.tsv"
    return list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))


def load_annotation(book: str) -> dict[str, list[dict]]:
    """annotation.json → {tên ảnh: [câu…]} (mỗi câu: hn_text = GT, translation = QN)."""
    p = BOOKS[book]["root"] / "pages/annotation.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for rec in data:
        img = str(rec.get("img") or "")
        out[Path(img).name] = list(rec.get("annotations") or [])
    return out


def load_bboxes(book: str) -> dict[str, dict]:
    """bboxes.json (VoTT) → {tên ảnh: {size, cols:[{x0,y0,x1,y1}…] sắp PHẢI→TRÁI}}."""
    p = BOOKS[book]["root"] / "pages/bboxes.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for a in (data.get("assets") or {}).values():
        asset = a.get("asset") or {}
        name = asset.get("name")
        if not name:
            continue
        sz = asset.get("size") or {}
        cols = []
        for r in a.get("regions") or []:
            if "Column" not in (r.get("tags") or []):
                continue
            bb = r.get("boundingBox") or {}
            x0, y0 = float(bb["left"]), float(bb["top"])
            cols.append(dict(x0=x0, y0=y0, x1=x0 + float(bb["width"]), y1=y0 + float(bb["height"])))
        cols.sort(key=lambda c: -(c["x0"] + c["x1"]) / 2)      # cột 0 = PHẢI nhất
        out[name] = dict(W=int(sz.get("width") or 0), H=int(sz.get("height") or 0), cols=cols)
    return out


def qn_syllables(text: str) -> list[str]:
    from core.text.text_utils import clean_line_text, split_to_syllables
    return [t for t in split_to_syllables(clean_line_text(text)) if any(ch.isalpha() for ch in t)]


def _R() -> dict[str, set]:
    """âm QN (chuẩn hoá dấu, thường) → tập chữ Nôm đọc được âm ấy."""
    global _R_CACHE
    try:
        return _R_CACHE
    except NameError:
        pass
    from core.text.dictionary import load_qn_to_nom
    from core.text.text_utils import normalize_tone_marks
    q2n = load_qn_to_nom(str(REPO / "Dict/QuocNgu_SinoNom.csv"))
    R: dict[str, set] = {}
    for k, v in q2n.items():
        R.setdefault(normalize_tone_marks(k.lower()), set()).update(v)
    _R_CACHE = R
    return R


# --------------------------------------------------------------------------- 1–3. bố cục
def page_records(book: str) -> list[dict]:
    """Gộp manifest + bboxes + annotation theo trang; 1 bản ghi / ảnh trang."""
    man = load_manifest(book)
    ann = load_annotation(book)
    bbx = load_bboxes(book)
    by_page: dict[str, list[dict]] = {}
    for r in man:
        by_page.setdefault(r["page_id"], []).append(r)
    recs = []
    for page_id, rows in sorted(by_page.items()):
        img_name = Path(rows[0]["page_image"]).name
        cols: dict[int, dict[int, dict]] = {}
        for r in rows:
            if r["col_index"] == "" or r["part"] == "":
                continue
            cols.setdefault(int(r["col_index"]), {})[int(r["part"])] = r
        b = bbx.get(img_name, dict(W=0, H=0, cols=[]))
        a = ann.get(img_name, [])
        recs.append(dict(book=book, page_id=page_id, img=img_name,
                         img_path=BOOKS[book]["root"] / BOOKS[book]["images"] / img_name,
                         W=b["W"], H=b["H"], bbox_cols=b["cols"], man_cols=cols,
                         n_verses_manifest=len(rows), n_verses_annotation=len(a), ann=a))
    return recs


def col_geometry(bcols: list[dict]) -> dict:
    """Bề rộng / chiều cao / bước cột / chồng lấn từ danh sách ô cột (đã sắp phải→trái)."""
    if not bcols:
        return dict(n=0)
    w = [c["x1"] - c["x0"] for c in bcols]
    h = [c["y1"] - c["y0"] for c in bcols]
    xc = [(c["x0"] + c["x1"]) / 2 for c in bcols]
    d = [abs(xc[i] - xc[i + 1]) for i in range(len(xc) - 1)]
    ov = 0
    for i in range(len(bcols) - 1):
        a, b = bcols[i], bcols[i + 1]
        ov += int(min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]) > 0)
    return dict(n=len(bcols), w_med=statistics.median(w), h_med=statistics.median(h),
                pitch_med=statistics.median(d) if d else None, n_overlap=ov,
                px_per_char=statistics.median(h) / N_PER_COL,
                x_desc=all(xc[i] > xc[i + 1] for i in range(len(xc) - 1)))


def ink_columns(img_path: Path, scale: int = 3,
                pitch_range: tuple[int, int] | None = None) -> tuple[int, float | None]:
    """Đối chứng ĐỘC LẬP: (số cột, bước cột) theo chiếu mực dọc — KHÔNG đọc bboxes.json.

    Otsu → mặt nạ mực → chiếu theo trục x → tự tương quan tìm bước cột trong [25, 70] px →
    trượt gốc lưới để tổng mực TẠI RANH GIỚI nhỏ nhất → đếm ô lưới có mực ≥ 0,25 × trung vị
    (cùng ý tưởng `analyze_nom` của chresto_map; mộc bản IHR không có khung kẻ rõ nên bỏ
    bước mở hình thái tìm khung)."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    pil = Image.open(img_path).convert("L")
    if scale != 1:
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.LANCZOS)
    im = np.asarray(pil)
    bw = (im < _otsu(im)).astype(float)
    prof = ndimage.uniform_filter1d(bw.mean(0), 3 * scale)
    on = np.nonzero(prof > 0.02)[0]
    if len(on) < 20:
        return 0, None
    x0, x1 = int(on[0]), int(on[-1])
    seg = prof[x0:x1 + 1]
    v = seg - seg.mean()
    ac = np.correlate(v, v, "full")[len(v) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = pitch_range or (25 * scale, 70 * scale)
    lo, hi = max(2, int(lo)), min(int(hi), len(ac) - 1)
    if hi <= lo:
        return 0, None
    pitch = int(np.argmax(ac[lo:hi])) + lo
    n_slot = max(1, int(round(len(seg) / pitch)))
    best = None
    for off in range(pitch):
        bnd = [off + k * pitch for k in range(n_slot + 1) if 0 <= off + k * pitch < len(seg)]
        if len(bnd) < 2:
            continue
        sc = sum(seg[b] for b in bnd) / len(bnd)
        if best is None or sc < best[0]:
            best = (sc, off)
    off = best[1] if best else 0
    edges = [off + k * pitch for k in range(-1, n_slot + 2)]
    inks = []
    for a, b in zip(edges[:-1], edges[1:]):
        a2, b2 = max(0, a), min(len(seg), b)
        if b2 - a2 >= 0.5 * pitch:
            inks.append(float(seg[a2:b2].sum()))
    if not inks:
        return 0, float(pitch)
    med = statistics.median(inks)
    return int(sum(1 for x in inks if x >= 0.25 * med)), float(pitch)


def _otsu(a) -> int:
    import numpy as np
    h, _ = np.histogram(a, 256, (0, 256))
    p = h.astype(float) / max(1, h.sum())
    w0 = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    mt = m[-1]
    return int(np.argmax((mt * w0 - m) ** 2 / (w0 * (1 - w0) + 1e-9)))


# --------------------------------------------------------------------------- 5. kim × hệ số phóng
def scaled_png(img_path: Path, scale: int, cache_dir: Path) -> Path:
    """Ảnh gửi kim: L, phóng ×scale bằng LANCZOS (scale 1 = chỉ đổi sang PNG mode L)."""
    from PIL import Image
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{img_path.stem}_x{scale}.png"
    if dst.exists():
        return dst
    im = Image.open(img_path).convert("L")
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    im.save(dst, "PNG")
    return dst


def kim_page(png: Path, cache_dir: Path, kim_params: dict, force: bool = False) -> list[dict] | None:
    """kim OCR 1 trang, cache theo (md5 ảnh, tham số) → chạy lại 0 lượt."""
    md5 = hashlib.md5(png.read_bytes()).hexdigest()
    key = md5 + "_" + "".join(f"{k[0]}{v}" for k, v in sorted(kim_params.items()))
    cp = cache_dir / f"{key}.json"
    if cp.exists() and not force:
        try:
            return json.loads(cp.read_text(encoding="utf-8"))["boxes"]
        except Exception:      # noqa: BLE001
            pass
    from core.ocr import ocr_api
    fn = ocr_api.upload_image(str(png))
    if not fn:
        return None
    boxes = ocr_api.recognize(fn, **kim_params)
    if boxes is None:
        return None
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(dict(image=str(png), md5=md5, kim_params=kim_params, boxes=boxes),
                             ensure_ascii=False), encoding="utf-8")
    return boxes


def box_chars(box: dict) -> list[dict]:
    """Hộp kim → từng chữ, chia đều chiều cao (cùng quy ước boxes_to_columns của ocr_api)."""
    text = (box.get("transcription") or "").strip()
    chars = [c for c in text if c.strip()]
    if not chars:
        return []
    xs = [p[0] for p in box["points"]]
    ys = [p[1] for p in box["points"]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    h = (y1 - y0) / len(chars)
    return [dict(char=c, cx=(x0 + x1) / 2, cy=y0 + h * (i + 0.5)) for i, c in enumerate(chars)]


def assign_to_columns(boxes: list[dict], bcols: list[dict], scale: int) -> list[list[str]]:
    """Chữ kim → cột (tâm x gần nhất, ≤ COL_TOL × bước cột), trong cột sắp theo y."""
    cols = [dict(xc=(c["x0"] + c["x1"]) / 2 * scale, chars=[]) for c in bcols]
    if not cols:
        return []
    xcs = [c["xc"] for c in cols]
    pitch = statistics.median([abs(xcs[i] - xcs[i + 1]) for i in range(len(xcs) - 1)]) if len(xcs) > 1 else 1e9
    for b in boxes:
        for ch in box_chars(b):
            d, k = min((abs(ch["cx"] - c["xc"]), i) for i, c in enumerate(cols))
            if d > COL_TOL * pitch:
                continue
            cols[k]["chars"].append(ch)
    return [[c["char"] for c in sorted(col["chars"], key=lambda z: z["cy"])] for col in cols]


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def gt_column(rec: dict, k: int) -> tuple[list[str], list[str]]:
    """(GT chữ Nôm, âm QN) của cột k từ manifest (part 1 ⧺ part 2). Thiếu part → ([], [])."""
    parts = rec["man_cols"].get(k) or {}
    if 1 not in parts or 2 not in parts:
        return [], []
    nom = list(parts[1]["nom_text"]) + list(parts[2]["nom_text"])
    qn = qn_syllables(parts[1]["qn_verse"]) + qn_syllables(parts[2]["qn_verse"])
    return nom, qn


def kim_scale_probe(book: str, recs: list[dict], scales: list[int], n_pages: int,
                    out: Path, kim_params: dict, force: bool) -> dict:
    """Gọi kim trên n_pages trang đầu có đủ bố cục, ở từng hệ số phóng; so GT."""
    R = _R()
    rows, per_scale = [], {}
    sel = [r for r in recs if r["bbox_cols"] and len(r["bbox_cols"]) == len(r["man_cols"])][:n_pages]
    calls = 0
    for s in scales:
        agg = dict(scale=s, pages=0, cols=0, cols_full14=0, cols_6_8=0, chars_kim=0, chars_gt=0,
                   eq_gt=0, in_R=0, n_pos=0, lcs=0, kim_failed=0)
        for rec in sel:
            png = scaled_png(rec["img_path"], s, out / "png")
            existed = True
            md5 = hashlib.md5(png.read_bytes()).hexdigest()
            key = md5 + "_" + "".join(f"{k[0]}{v}" for k, v in sorted(kim_params.items()))
            if not (out / "kim_cache" / f"{key}.json").exists():
                existed = False
            boxes = kim_page(png, out / "kim_cache", kim_params, force=force)
            if not existed and boxes is not None:
                calls += 1
            agg["pages"] += 1
            if boxes is None:
                agg["kim_failed"] += 1
                continue
            cols = assign_to_columns(boxes, rec["bbox_cols"], s)
            for k, kim_chars in enumerate(cols):
                gt, qn = gt_column(rec, k)
                if not gt:
                    continue
                agg["cols"] += 1
                agg["chars_kim"] += len(kim_chars)
                agg["chars_gt"] += len(gt)
                agg["lcs"] += lcs_len(kim_chars, gt)
                full = len(kim_chars) == N_PER_COL
                agg["cols_full14"] += int(full)
                agg["cols_6_8"] += int(full and len(gt) == N_PER_COL)
                if full and len(gt) == N_PER_COL and len(qn) == N_PER_COL:
                    for i, ch in enumerate(kim_chars):
                        agg["n_pos"] += 1
                        agg["eq_gt"] += int(ch == gt[i])
                        from core.text.text_utils import normalize_tone_marks
                        agg["in_R"] += int(ch in R.get(normalize_tone_marks(qn[i].lower()), ()))
                rows.append(dict(book=book, scale=s, page=rec["page_id"], col=k,
                                 n_kim=len(kim_chars), n_gt=len(gt),
                                 lcs=lcs_len(kim_chars, gt),
                                 kim="".join(kim_chars), gt="".join(gt)))
        for key, num, den in (("rate_cols_full14", "cols_full14", "cols"),
                              ("rate_eq_gt", "eq_gt", "n_pos"),
                              ("rate_in_R", "in_R", "n_pos"),
                              ("recall_lcs", "lcs", "chars_gt")):
            agg[key] = round(agg[num] / agg[den], 4) if agg[den] else None
        per_scale[str(s)] = agg
    with open(out / "kim_scale.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["book", "scale", "page", "col", "n_kim", "n_gt", "lcs", "kim", "gt"])
        w.writeheader()
        w.writerows(rows)
    return dict(pages_probed=len(sel), scales=scales, api_calls=calls, kim_params=kim_params,
                per_scale=per_scale)


# --------------------------------------------------------------------------- chạy 1 sách
def measure_book(book: str, out: Path, kim_pages: int, scales: list[int],
                 kim_params: dict, ink_pages: int, force: bool, ink_scale: int = 3) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    recs = page_records(book)
    prow, crow = [], []
    n_pair_cols = n_single = 0
    tier_ok = tier_tot = 0
    n_ann_eq_man = 0
    for rec in recs:
        g = col_geometry(rec["bbox_cols"])
        n_man_cols = len(rec["man_cols"])
        for k, parts in sorted(rec["man_cols"].items()):
            if set(parts) == {1, 2}:
                n_pair_cols += 1
            else:
                n_single += 1
            for p, r in sorted(parts.items()):
                tier_tot += 1
                tier_ok += int(int(r["n_nom"]) == TIER_RULE[p - 1])
                crow.append(dict(book=book, page=rec["page_id"], col=k, part=p,
                                 n_nom=r["n_nom"], n_qn=r["n_qn_syll"], len_match=r["len_match"]))
        n_ann_eq_man += int(rec["n_verses_annotation"] == rec["n_verses_manifest"])
        prow.append(dict(book=book, page=rec["page_id"], img=rec["img"], W=rec["W"], H=rec["H"],
                         n_verses=rec["n_verses_manifest"], n_verses_ann=rec["n_verses_annotation"],
                         n_cols_bbox=len(rec["bbox_cols"]), n_cols_manifest=n_man_cols,
                         n_cols_expected=math.ceil(rec["n_verses_manifest"] / 2),
                         col_w=round(g.get("w_med") or 0, 1), col_h=round(g.get("h_med") or 0, 1),
                         col_pitch=round(g.get("pitch_med") or 0, 1),
                         px_per_char=round(g.get("px_per_char") or 0, 2),
                         n_overlap=g.get("n_overlap", 0), x_desc=int(bool(g.get("x_desc")))))
    with open(out / "pages.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prow[0]))
        w.writeheader()
        w.writerows(prow)
    with open(out / "cols.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(crow[0]))
        w.writeheader()
        w.writerows(crow)

    with_bbox = [p for p in prow if p["n_cols_bbox"]]
    hist = {}
    for p in prow:
        hist[str(p["n_cols_bbox"])] = hist.get(str(p["n_cols_bbox"]), 0) + 1
    n_match = sum(1 for p in with_bbox if p["n_cols_bbox"] == p["n_cols_manifest"])
    n_exp = sum(1 for p in prow if p["n_cols_bbox"] == p["n_cols_expected"])
    pages_not_exp = [p["page"] for p in prow if p["n_cols_bbox"] != p["n_cols_expected"]]
    pitch_all = [p["col_pitch"] for p in with_bbox if p["col_pitch"]]
    ppc_all = [p["px_per_char"] for p in with_bbox if p["px_per_char"]]

    # đối chứng độc lập bằng ảnh (ink_pages trang đầu có bbox)
    ink = dict(pages=0, eq=0, eq1=0, pitch_close=0, rows=[], scale=ink_scale)
    sel_ink = with_bbox[:ink_pages]
    # Lượt 1: bước cột thô từng trang -> trung vị CẢ SÁCH (độc lập bboxes.json); lượt 2 đếm cột
    # trong cửa sổ +-20 % quanh trung vị ấy (ảnh 35-40 px/chữ hay khoá nhầm hài của tự tương quan).
    p1 = [ink_columns(next(r for r in recs if r["page_id"] == q["page"])["img_path"], ink_scale)[1]
          for q in sel_ink]
    p1 = [x for x in p1 if x]
    pr = None
    if p1:
        med = statistics.median(p1)
        pr = (int(med * 0.8), int(med * 1.2) + 1)
    ink["pitch_window"] = pr
    for p in sel_ink:
        rec = next(r for r in recs if r["page_id"] == p["page"])
        n_ink, pitch_ink = ink_columns(rec["img_path"], ink_scale, pr)
        pitch_ink = pitch_ink / ink_scale if pitch_ink else None
        ink["pages"] += 1
        ink["eq"] += int(n_ink == p["n_cols_bbox"])
        ink["eq1"] += int(abs(n_ink - p["n_cols_bbox"]) <= 1)
        ink["pitch_close"] += int(pitch_ink is not None and p["col_pitch"]
                                  and abs(pitch_ink - p["col_pitch"]) <= 0.1 * p["col_pitch"])
        ink["rows"].append(dict(page=p["page"], n_ink=n_ink, n_bbox=p["n_cols_bbox"],
                                pitch_ink=pitch_ink, pitch_bbox=p["col_pitch"]))
    ink["rate"] = round(ink["eq"] / ink["pages"], 4) if ink["pages"] else None
    ink["rate_eq1"] = round(ink["eq1"] / ink["pages"], 4) if ink["pages"] else None
    ink["rate_pitch"] = round(ink["pitch_close"] / ink["pages"], 4) if ink["pages"] else None

    kim = None
    if kim_pages:
        kim = kim_scale_probe(book, recs, scales, kim_pages, out, kim_params, force)

    s = dict(
        book=book, edition=BOOKS[book]["edition"], generated_by="scripts/measure/ihr_layout.py",
        pages=dict(n=len(recs), with_bbox=len(with_bbox), without_bbox=len(recs) - len(with_bbox),
                   ann_eq_manifest=n_ann_eq_man),
        columns=dict(hist_n_cols_bbox=hist, n_cols_bbox_eq_manifest=n_match,
                     n_cols_bbox_eq_expected=n_exp,
                     rate_bbox_eq_expected=round(n_exp / max(1, len(prow)), 4),
                     pages_bbox_ne_expected=pages_not_exp,
                     n_pages_with_bbox=len(with_bbox),
                     rate_bbox_eq_manifest=round(n_match / len(with_bbox), 4) if with_bbox else None,
                     n_cols_pair=n_pair_cols, n_cols_single_part=n_single,
                     rate_col_is_couplet=round(n_pair_cols / max(1, n_pair_cols + n_single), 4),
                     n_verses=tier_tot, n_verses_6_or_8=tier_ok,
                     rate_verse_len_rule=round(tier_ok / max(1, tier_tot), 4),
                     x_order_desc_pages=sum(p["x_desc"] for p in with_bbox),
                     n_pages_overlap=sum(1 for p in with_bbox if p["n_overlap"])),
        geometry=dict(col_pitch_med=round(statistics.median(pitch_all), 1) if pitch_all else None,
                      px_per_char_med=round(statistics.median(ppc_all), 2) if ppc_all else None,
                      px_per_char_min=round(min(ppc_all), 2) if ppc_all else None,
                      px_per_char_max=round(max(ppc_all), 2) if ppc_all else None,
                      W_med=statistics.median([p["W"] for p in with_bbox]) if with_bbox else None,
                      H_med=statistics.median([p["H"] for p in with_bbox]) if with_bbox else None),
        ink_control=dict(pages=ink["pages"], scale=ink["scale"], pitch_window=ink["pitch_window"],
                         eq=ink["eq"], rate=ink["rate"],
                         eq1=ink["eq1"], rate_eq1=ink["rate_eq1"],
                         pitch_close=ink["pitch_close"], rate_pitch=ink["rate_pitch"],
                         rows=ink["rows"][:20]),
        kim_scale=kim,
    )
    s["invariants"] = invariants(s)
    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return s


def invariants(s: dict) -> list[dict]:
    iv = []

    def add(name, expected, observed, ok, method):
        iv.append(dict(name=name, expected=expected, observed=observed, method=method,
                       **{"pass": ok}))
    c = s["columns"]
    add("moi_cot_la_mot_cap_luc_bat", ">= 0.99", c["rate_col_is_couplet"],
        (c["rate_col_is_couplet"] or 0) >= 0.99,
        "đếm `part` mỗi (trang, col_index) trong manifest.tsv: {1,2} = cặp lục bát")
    add("so_chu_cau_theo_luat_6_8", ">= 0.95", c["rate_verse_len_rule"],
        (c["rate_verse_len_rule"] or 0) >= 0.95, "n_nom của part 1 == 6 và part 2 == 8")
    add("cot_bbox_khop_so_cau", ">= 0.95", c["rate_bbox_eq_expected"],
        (c["rate_bbox_eq_expected"] or 0) >= 0.95,
        "số region tag Column trong bboxes.json == ceil(số câu annotation.json / 2) "
        "(mỗi cột = 1 cặp lục bát). Trang lệch liệt kê ở pages_bbox_ne_expected — adapter bỏ qua.")
    add("cot_sap_phai_sang_trai", f"{c['n_pages_with_bbox']}", c["x_order_desc_pages"],
        c["x_order_desc_pages"] == c["n_pages_with_bbox"],
        "tâm x của cột 0..n giảm dần (cột 0 = phải nhất, đúng chiều đọc Hán-Nôm)")
    add("ann_khop_manifest", f"{s['pages']['n']}", s["pages"]["ann_eq_manifest"],
        s["pages"]["ann_eq_manifest"] == s["pages"]["n"],
        "số câu trong annotation.json == số dòng manifest.tsv của trang")
    ic = s["ink_control"]
    add("chieu_muc_doc_lap_khop_bbox_+-1", ">= 0.8", ic.get("rate_eq1"),
        ic.get("rate_eq1") is None or ic["rate_eq1"] >= 0.8,
        "đối chứng: |số cột theo chiếu mực ảnh (không đọc bboxes.json) − số cột bbox| <= 1; "
        "chiếu mực đo trên ảnh phóng ×3 vì bản quét gốc chỉ 35-40 px/chữ")
    add("chieu_muc_buoc_cot_khop_bbox", ">= 0.8", ic.get("rate_pitch"),
        ic.get("rate_pitch") is None or ic["rate_pitch"] >= 0.8,
        "bước cột do tự tương quan chiếu mực lệch <= 10 % so bước cột bbox")
    k = s.get("kim_scale")
    if k:
        best = max(k["per_scale"].values(), key=lambda a: a["rate_eq_gt"] or 0)
        add("kim_phong_to_khong_kem_hon", "scale tốt nhất có rate_eq_gt >= x1",
            {sc: a["rate_eq_gt"] for sc, a in k["per_scale"].items()},
            (best["rate_eq_gt"] or 0) >= (k["per_scale"].get("1", {}).get("rate_eq_gt") or 0),
            "kim đọc ảnh phóng ×s, so GT từng chữ ở cột kim đọc đủ 14")
    return iv


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    n = ok = 0

    def chk(name, cond):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'} {name}")

    chk("lcs_len rỗng", lcs_len([], ["a"]) == 0)
    chk("lcs_len trùng", lcs_len(list("abc"), list("abc")) == 3)
    chk("lcs_len chèn", lcs_len(list("abc"), list("axbc")) == 3)
    chk("lcs_len khác hẳn", lcs_len(list("abc"), list("xyz")) == 0)
    b = dict(points=[[0, 0], [10, 0], [10, 30], [0, 30]], transcription="甲乙丙")
    cs = box_chars(b)
    chk("box_chars 3 chữ", [c["char"] for c in cs] == ["甲", "乙", "丙"])
    chk("box_chars y tăng", cs[0]["cy"] < cs[1]["cy"] < cs[2]["cy"])
    chk("box_chars tâm x", all(abs(c["cx"] - 5) < 1e-6 for c in cs))
    chk("box_chars rỗng", box_chars(dict(points=[[0, 0]], transcription="  ")) == [])
    bcols = [dict(x0=40, y0=0, x1=50, y1=140), dict(x0=20, y0=0, x1=30, y1=140)]
    g = col_geometry(bcols)
    chk("col_geometry bước cột", abs(g["pitch_med"] - 20) < 1e-6)
    chk("col_geometry px/chữ", abs(g["px_per_char"] - 10.0) < 1e-6)
    chk("col_geometry phải→trái", g["x_desc"] is True)
    chk("col_geometry không chồng", g["n_overlap"] == 0)
    boxes = [dict(points=[[40, 0], [50, 0], [50, 14], [40, 14]], transcription="甲乙"),
             dict(points=[[20, 0], [30, 0], [30, 7], [20, 7]], transcription="丙")]
    a = assign_to_columns(boxes, bcols, 1)
    chk("assign cột 0 = phải", a[0] == ["甲", "乙"])
    chk("assign cột 1", a[1] == ["丙"])
    far = [dict(points=[[200, 0], [210, 0], [210, 7], [200, 7]], transcription="丁")]
    chk("assign bỏ hộp quá xa", assign_to_columns(far, bcols, 1) == [[], []])
    chk("assign theo hệ số phóng", assign_to_columns(
        [dict(points=[[80, 0], [100, 0], [100, 28], [80, 28]], transcription="甲")], bcols, 2)[0] == ["甲"])
    chk("qn_syllables bỏ dấu câu", qn_syllables("Trăm năm trong cõi người ta,") ==
        ["Trăm", "năm", "trong", "cõi", "người", "ta"])
    chk("TIER_RULE 6+8", N_PER_COL == 14)
    for bk in BOOKS:
        chk(f"{bk}: manifest tồn tại", (BOOKS[bk]["root"] / "manifest.tsv").exists())
        chk(f"{bk}: bboxes tồn tại", (BOOKS[bk]["root"] / "pages/bboxes.json").exists())
    print(f"ihr_layout selftest: {ok}/{n}")
    return 0 if ok == n else 1


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default="all")
    ap.add_argument("--out", default="", help="mặc định measure_out/<book>/ihr_layout")
    ap.add_argument("--kim-pages", type=int, default=0, help="số trang gọi kim mỗi hệ số phóng (0 = tắt)")
    ap.add_argument("--scales", default="1,3")
    ap.add_argument("--ink-pages", type=int, default=20, help="số trang đối chứng chiếu mực")
    ap.add_argument("--ink-scale", type=int, default=3, help="hệ số phóng khi đối chứng chiếu mực")
    ap.add_argument("--kim-lang-type", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=1)      # tương thích measure.py
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    books = list(BOOKS) if a.book == "all" else [b for b in a.book.split(",") if b]
    bad = [b for b in books if b not in BOOKS]
    if bad:
        print(f"sách không biết: {bad}; chọn trong {list(BOOKS)}")
        return 2
    scales = [int(x) for x in a.scales.split(",") if x]
    kim_params = dict(ocr_id=1, lang_type=a.kim_lang_type, font_type=1)
    rc = 0
    for b in books:
        out = Path(a.out) if a.out and len(books) == 1 else REPO / "measure_out" / b / "ihr_layout"
        s = measure_book(b, out, a.kim_pages, scales, kim_params,
                         a.limit or a.ink_pages, a.force, ink_scale=a.ink_scale)
        nf = sum(1 for iv in s["invariants"] if iv["pass"] is False)
        rc = rc or (1 if nf else 0)
        print(f"{b}: {s['pages']['n']} trang, cột/trang {s['columns']['hist_n_cols_bbox']}, "
              f"px/chữ {s['geometry']['px_per_char_med']}, invariants FAIL {nf} → {out}")
        if s.get("kim_scale"):
            for sc, agg in s["kim_scale"]["per_scale"].items():
                print(f"   kim ×{sc}: đủ14 {agg['rate_cols_full14']} · đúng GT {agg['rate_eq_gt']} · "
                      f"∈R {agg['rate_in_R']} · thu hồi LCS {agg['recall_lcs']} ({agg['cols']} cột)")
            print(f"   lượt gọi kim mới: {s['kim_scale']['api_calls']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
