"""Adapter ingest sách VĂN XUÔI in đá (Chrestomathie1872) → `prepared/<book>/` đúng hợp đồng engine.

Khác thạch bản lục bát (ingest_lithograph_book: 10 cột × 14 âm cố định), văn xuôi có
số cột mỗi trang biến thiên (4–7), mỗi cột ~16–21 chữ, KHÔNG có số âm cố định và không
có số câu in để neo. Cách ghép QN ↔ Nôm:

  1. Bố cục cột từng trang: thuật toán khung/ô cột của bộ đo `scripts/measure/chresto_map.py`
     (`analyze_nom`: khung mở hình thái, pitch cột tự tương quan, lưới 7 ô) — nhập trực tiếp
     để adapter và bộ đo dùng CÙNG một quy tắc ô cột (bang_truyen_trang.csv đánh số ô theo đó).
  2. Kim OCR toàn trang (core.ocr.ocr_api, 1 lượt/trang, cache kim_raw/): mỗi hộp kim →
     chữ chia đều theo chiều cao (như boxes_to_columns) → gom vào ô cột theo tâm x với
     ngưỡng 0,5 × bước cột (không dùng x_tol 15 px cứng của boxes_to_columns).
  3. Truyện ↔ trang: `measure_out/<book>/chresto_map/bang_truyen_trang.csv` (truyện bắt đầu ở
     (canvas, ô)); cột thuộc truyện có (canvas, ô) bắt đầu lớn nhất ≤ (canvas, ô) của cột.
     Văn bản QN của truyện dựng lại từ `qn_lines.csv` + `qn_stories.csv` bằng đúng quy tắc
     `build_stories` của bộ đo (dòng thân, trang VN), thêm tiền tố tiêu đề
     "truyện thứ <số Hán-Việt> <tiêu đề>" (bản Nôm mở đầu bằng 傳次壹…).
  4. Ghép chuỗi chữ kim của cả truyện (các cột theo thứ tự đọc: trang tăng, ô TRÁI→PHẢI —
     sách in kiểu Tây, đã đo) với chuỗi âm tiết QN của truyện bằng DP đơn điệu kiểu
     Needleman–Wunsch: khớp = +3 khi chữ ∈ R(âm) (từ điển QuocNgu_SinoNom), cặp không tra
     được = 0, chèn/xoá = −1, hai đầu QN tự do. Âm tiết ghép cho từng cột = đoạn âm nằm giữa
     âm đầu và âm cuối ghép với chữ của cột (âm kẽ giữa hai cột chia theo trung điểm).
  5. Cột không đủ điểm (n_match < PROSE_MIN_MATCH hoặc tỉ lệ < PROSE_MIN_RATIO), cột ngoài
     truyện (đầu đề chạy 傳制…), hoặc --ocr none → dòng giữ chỗ `khongkhop` × số chữ →
     engine: mọi ô của cột chỉ có thể là REVIEW (token không hợp lệ).

Đầu ra (`--out prepared` → prepared/<book>/), engine đọc như STT:
  pages/page_XXXX.png            ảnh L, nền kéo về 255 (--contrast stretch|otsu|none), giữ kích thước
  pages_denoised/page_XXXX.png   denoise_image như step1
  detected/page_XXXX_ocr_cache.json  coords_space=fullpage, image_hash/pixel_hash của pages/*.png,
                                 columns theo quy ước engine (cột 1 = PHẢI nhất), mỗi chữ bbox toàn trang
  transcriptions/page_XXXX.txt   1 dòng / cột (cùng thứ tự phải→trái), = âm tiết đã ghép hoặc giữ chỗ
  transcriptions/page_XXXX.json  columns [{column, reading_order, slot, story, syllables, num_syllables,
                                 n_chars_kim, n_match, dp_ratio, syl_span, matched, raw_text}], stories, canvas
  kim_raw/page_XXXX.json         hộp thô kim (cache theo md5 ảnh đã OCR; engine không đọc)
  manifest.json                  gates: cột/trang, tỉ lệ cột ghép được, điểm DP từng truyện,
                                 ranh giới truyện giữa trang, kiểm thứ tự đọc (LTR vs RTL), cache ok

Trang: canvas 106–170 của data/Chrestomathie1872/nom_pages → page_0001…page_0065
(page = canvas − 105; ánh xạ ghi trong manifest và transcriptions/*.json).

Chạy:
  .venv/bin/python -m pipeline.tools.ingest_prose_book --book Chrestomathie1872 --limit 5 --ocr kim
  .venv/bin/python -m pipeline.tools.ingest_prose_book --book Chrestomathie1872 --ocr kim
  .venv/bin/python -m pipeline.tools.ingest_prose_book --book Chrestomathie1872 --pages 1-3 --ocr none   # khói
Test: .venv/bin/python -m pipeline.tools.ingest_prose_selftest
Lõi tái dùng từ ingest_lithograph_book: từ điển chữ→âm, tách âm tiết, chia chữ trong hộp kim,
ảnh, gọi kim có cache, token giữ chỗ. Không sửa lõi `--verse-map content` dùng chung.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.tools.ingest_lithograph_book import (  # noqa: E402  (lõi dùng chung)
    CONTENT_PLACEHOLDER, _nom_to_qn_readings, _rel, expand_box_chars, box_rect,
    kim_boxes, kim_cache_suffix, kim_params_of, book_kim_params, lithograph_syllables, prepare_image)

BOOKS = {
    "Chrestomathie1872": dict(
        src=REPO / "data/Chrestomathie1872/nom_pages",
        canvases=list(range(106, 171)),           # 65 trang chữ Nôm (105/171 trắng, 104 tựa)
        canvas_of_page=lambda k: k + 105,         # page_0001 = canvas 106
        page_of_canvas=lambda c: c - 105,
        measure="chresto_map",
        read_ltr=True,                            # cột đọc TRÁI→PHẢI (PIPELINE_SACH_MOI §1.6, kim 106 xác nhận)
        title_prefix=("truyện", "thứ"),           # bản Nôm: 傳次壹 + tiêu đề
    ),
}
COL_TOL = 0.5              # × col_pitch: |tâm hộp kim − tâm ô cột| tối đa để gán
PROSE_MATCH = 3.0          # điểm khớp (chữ kim ∈ R(âm))
PROSE_SUB = 0.0            # cặp không tra được từ điển (vẫn tiến cả hai chuỗi)
PROSE_GAP = -1.0           # chèn/xoá một phần tử
PROSE_MIN_MATCH = 3        # cột ghép được: ít nhất bấy nhiêu chữ khớp từ điển …
PROSE_MIN_RATIO = 0.25     # … và tỉ lệ chữ khớp / chữ kim ≥ ngưỡng; dưới → giữ chỗ (REVIEW)
SINO_DIGITS = {1: "nhất", 2: "nhị", 3: "tam", 4: "tứ", 5: "ngũ", 6: "lục", 7: "thất", 8: "bát", 9: "cửu"}


# ---------------------------------------------------------------------------
# 0. Bộ đo (nhập quy tắc ô cột + dựng truyện từ scripts/measure/chresto_map.py)
# ---------------------------------------------------------------------------
def _measure_module():
    """Nạp scripts/measure/chresto_map.py như mô-đun (không có side effect khi import)."""
    global _CM
    try:
        return _CM
    except NameError:
        pass
    path = REPO / "scripts" / "measure" / "chresto_map.py"
    spec = importlib.util.spec_from_file_location("gn_measure_chresto_map", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _CM = mod
    return mod


def sino_number(n: int) -> list[str]:
    """Số đếm Hán-Việt: 1→nhất … 10→thập, 11→thập nhất, 20→nhị thập (chữ 壹…拾 trong 傳次…)."""
    if n < 10:
        return [SINO_DIGITS[n]]
    tens, ones = divmod(n, 10)
    out = ([SINO_DIGITS[tens]] if tens > 1 else []) + ["thập"]
    if ones:
        out.append(SINO_DIGITS[ones])
    return out


def load_story_table(measure_dir: Path, book: str) -> list[dict]:
    """bang_truyen_trang.csv → [{so, tieu_de, qn_canvas, nom_canvas_dau, nom_slot_dau, ranh_gioi, ...}] theo so."""
    p = measure_dir / book / BOOKS[book]["measure"] / "bang_truyen_trang.csv"
    rows = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        rows.append(dict(so=int(r["so"]), tieu_de=r["tieu_de"], qn_canvas=int(r["qn_canvas"]),
                         nom_canvas_dau=int(r["nom_canvas_dau"]), nom_slot_dau=int(r["nom_slot_dau"]),
                         nom_trang=r["nom_trang"], nom_n_cols=int(r["nom_n_cols"] or 0),
                         ranh_gioi=r["ranh_gioi"], can_nguoi_ra=r.get("can_nguoi_ra", "") == "True"))
    rows.sort(key=lambda r: r["so"])
    return rows


def story_of_column(canvas: int, slot: int, table: list[dict]) -> int | None:
    """Truyện chứa cột (canvas, ô) = truyện có (canvas đầu, ô đầu) lớn nhất ≤ (canvas, ô); None = trước truyện 1."""
    best = None
    for r in table:
        if (r["nom_canvas_dau"], r["nom_slot_dau"]) <= (canvas, slot):
            if best is None or (r["nom_canvas_dau"], r["nom_slot_dau"]) >= (best["nom_canvas_dau"], best["nom_slot_dau"]):
                best = r
    return None if best is None else best["so"]


def load_story_syllables(measure_dir: Path, book: str, table: list[dict]) -> dict[int, dict]:
    """Dựng văn bản QN từng truyện từ qn_lines.csv + qn_stories.csv bằng đúng `build_stories` của bộ đo.

    Trả {so: {syllables: [âm tiết gốc], title_syllables: [...], n_body_lines, qn_pages}}; âm tiết =
    tiền tố tiêu đề ("truyện thứ <Hán-Việt>" + tiêu đề) ⧺ thân truyện (lithograph_syllables: bỏ token không chữ)."""
    cm = _measure_module()
    mdir = measure_dir / book / BOOKS[book]["measure"]
    lines = []
    for r in csv.DictReader(open(mdir / "qn_lines.csv", encoding="utf-8")):
        lines.append(dict(page=int(r["page"]), x0=int(r["x0"]), x1=int(r["x1"]), y0=int(r["y0"]), y1=int(r["y1"]),
                          w=int(r["w"]), cx=float(r["cx"]), text=r["text"], conf=float(r["conf"])))
    anchors = []
    for r in csv.DictReader(open(mdir / "qn_stories.csv", encoding="utf-8")):
        anchors.append((int(r["so"]), int(r["qn_canvas"]), int(r["qn_y"]), r["tieu_de"]))
    anchors.sort()
    summ = json.load(open(mdir / "summary.json", encoding="utf-8"))
    vn_pages = set(summ["qn"]["vn_pages"])
    _, lex = cm.load_dict()
    stories = cm.build_stories(lines, anchors, lex, vn_pages)
    prefix = list(BOOKS[book]["title_prefix"])
    out: dict[int, dict] = {}
    for st in stories:
        so = st["so"]
        body_text = " ".join(l["text"] for l in _story_body_lines(lines, anchors, lex, vn_pages, so))
        title = list(prefix) + sino_number(so) + lithograph_syllables(st["tieu_de"])
        body = lithograph_syllables(body_text)
        out[so] = dict(so=so, tieu_de=st["tieu_de"], title_syllables=title, body_syllables=body,
                       syllables=title + body, n_body_lines=st["n_lines"], qn_pages=st["qn_trang"],
                       oov_rate=st.get("oov_rate"))
    return out


def _story_body_lines(lines, anchors, lex, vn_pages, so) -> list[dict]:
    """Dòng thân của truyện `so` theo đúng lọc của build_stories (w>600, y0≥200, trang VN, không cước chú Pháp)."""
    import unicodedata as ud
    cm = _measure_module()

    def is_vn_line(l):
        ws = [ud.normalize("NFC", w.lower()) for w in cm.WORD_RE.findall(l["text"])]
        return len(ws) < 5 or sum(1 for w in ws if w in lex) / len(ws) >= 0.4
    body = sorted((l for l in lines if l["w"] > 600 and l["y0"] >= 200 and l["page"] in vn_pages and is_vn_line(l)),
                  key=lambda l: (l["page"], l["y0"]))
    idx = next(i for i, a in enumerate(anchors) if a[0] == so)
    n, p, y, t = anchors[idx]
    nxt = anchors[idx + 1] if idx + 1 < len(anchors) else (None, 10 ** 6, 0, None)
    ls = [l for l in body if (p, y + 60) < (l["page"], l["y0"]) < (nxt[1], nxt[2])]
    if ls and ls[0]["w"] < 750 and ls[0]["y0"] < y + 200 and any(w in ls[0]["text"].lower() for w in t.lower().split()[:2]):
        ls = ls[1:]
    return ls


# ---------------------------------------------------------------------------
# 1. Bố cục cột theo trang (quy tắc ô cột của bộ đo, toạ độ toàn trang)
# ---------------------------------------------------------------------------
def page_columns(canvas: int) -> dict:
    """analyze_nom(canvas) → {canvas, frame, col_pitch, nslot, cols: [{slot, x0, x1, xc, y0, y1, slot_x0, slot_x1,
    n_est, n_runs, short}]} với toạ độ TOÀN TRANG (bộ đo trả toạ độ trong khung)."""
    cm = _measure_module()
    L = cm.analyze_nom(canvas)
    if L.get("blank"):
        return dict(canvas=canvas, blank=True, cols=[], col_pitch=None, nslot=0, frame=None)
    top, bot, left, right = L["frame"]
    cols = []
    for c in L["cols"]:
        x0, x1 = left + int(c["x0"]), left + int(c["x1"])
        cols.append(dict(slot=int(c["slot"]), x0=x0, x1=x1, xc=(x0 + x1) / 2.0,
                         y0=top + int(c["y0"]), y1=top + int(c["y1"]),
                         slot_x0=left + int(c["slot_x0"]), slot_x1=left + int(c["slot_x1"]),
                         n_est=float(c["n_est"]), n_runs=int(c["n_runs"]), short=bool(c["short"])))
    cols.sort(key=lambda c: c["slot"])
    return dict(canvas=canvas, blank=False, frame=[int(top), int(bot), int(left), int(right)],
                col_pitch=int(L["col_pitch"]), nslot=int(L["nslot"]), char_pitch=L.get("char_pitch"),
                empty_internal_slots=L.get("empty_internal_slots", []), cols=cols)


def assign_boxes(boxes: list[dict], cols: list[dict], col_pitch: int, tol: float = COL_TOL) -> tuple[dict[int, list[dict]], dict]:
    """Hộp kim → ô cột gần tâm x nhất (|cx − xc| ≤ tol × col_pitch); chữ chia đều theo chiều cao hộp.
    Trả ({slot: [chữ theo y tăng]}, stats)."""
    per: dict[int, list[dict]] = {}
    stats = dict(n_boxes=len(boxes), n_chars=0, n_boxes_unassigned=0, n_chars_unassigned=0)
    for box in boxes:
        chars = expand_box_chars(box)
        stats["n_chars"] += len(chars)
        if not chars or not cols:
            stats["n_boxes_unassigned"] += int(bool(chars))
            stats["n_chars_unassigned"] += len(chars)
            continue
        x0, _, x1, _ = box_rect(box)
        cx = (x0 + x1) / 2.0
        d, slot = min((abs(cx - c["xc"]), c["slot"]) for c in cols)
        if d > tol * col_pitch:
            stats["n_boxes_unassigned"] += 1
            stats["n_chars_unassigned"] += len(chars)
            continue
        per.setdefault(slot, []).extend(chars)
    for s in per:
        per[s].sort(key=lambda c: c["y_center"])
    return per, stats


def projection_chars(col: dict) -> list[dict]:
    """--ocr none: n_runs ô đều nhau trong dải mực của cột, char=None (khói; engine không gán nhãn được)."""
    n = max(1, int(col["n_runs"]))
    h = (col["y1"] - col["y0"]) / n
    return [dict(char=None, y_center=col["y0"] + h * (i + 0.5),
                 bbox=[col["x0"], int(col["y0"] + h * i), col["x1"], int(col["y0"] + h * (i + 1))]) for i in range(n)]


# ---------------------------------------------------------------------------
# 2. DP đơn điệu chữ kim ↔ âm tiết (cả truyện)
# ---------------------------------------------------------------------------
def _norm_syl(s: str) -> str:
    from core.text.text_utils import normalize_tone_marks
    return normalize_tone_marks(s.lower())


def align_sequences(chars: list[str | None], syls: list[str], n2q: dict[str, set[str]],
                    match: float = PROSE_MATCH, sub: float = PROSE_SUB, gap: float = PROSE_GAP,
                    free_qn_ends: bool = True) -> tuple[list[tuple[int, int, bool]], float]:
    """Needleman–Wunsch đơn điệu: chữ i ↔ âm j (khớp khi chữ ∈ n2q và âm chuẩn hoá ∈ n2q[chữ]).
    Hai đầu chuỗi ÂM tự do (free_qn_ends): truyện có thể chưa đủ trang (--limit) hoặc thiếu cột.
    Trả ([(i, j, is_match)] theo i tăng, điểm)."""
    import numpy as np
    n, m = len(chars), len(syls)
    if n == 0 or m == 0:
        return [], 0.0
    ns = [_norm_syl(s) for s in syls]
    S = np.full((n, m), sub, dtype=np.float32)
    for i, ch in enumerate(chars):
        rd = n2q.get(ch) if ch else None
        if rd:
            for j, s in enumerate(ns):
                if s in rd:
                    S[i, j] = match
    H = np.empty((n + 1, m + 1), dtype=np.float32)
    H[0, :] = 0.0 if free_qn_ends else gap * np.arange(m + 1)
    H[:, 0] = gap * np.arange(n + 1)
    js = np.arange(0, m + 1, dtype=np.float32)          # j = 0..m
    for i in range(1, n + 1):
        a = np.maximum(H[i - 1, :-1] + S[i - 1], H[i - 1, 1:] + gap)      # j=1..m: chéo hoặc xoá chữ
        # trái (chèn âm): H[i][j] = max_{k≤j} (a'[k] + gap·(j−k)) với a'[0] = H[i][0], a'[k] = a[k−1]
        #               = gap·j + cummax_k≤j (a'[k] − gap·k)
        b = np.concatenate(([H[i, 0]], a - gap * js[1:]))
        H[i, 1:] = (np.maximum.accumulate(b) + gap * js)[1:]
    j_end = int(np.argmax(H[n])) if free_qn_ends else m
    score = float(H[n, j_end])
    pairs: list[tuple[int, int, bool]] = []
    i, j = n, j_end
    eps = 1e-4
    while i > 0 and j > 0:
        h = H[i, j]
        if abs(h - (H[i - 1, j - 1] + S[i - 1, j - 1])) < eps:
            pairs.append((i - 1, j - 1, bool(S[i - 1, j - 1] >= match)))
            i -= 1; j -= 1
        elif abs(h - (H[i - 1, j] + gap)) < eps:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs, score


def split_syllables_by_column(col_sizes: list[int], pairs: list[tuple[int, int, bool]], n_syl: int,
                              include_ends: bool = False) -> list[dict]:
    """Chia chuỗi âm cho từng cột theo cặp DP. Cột k có chữ [c0, c1); âm ghép = [j_min, j_max] của các cặp
    trong cột; âm kẽ giữa hai cột chia theo trung điểm (nửa đầu về cột trước). Âm trước cột đầu / sau cột
    cuối bỏ (include_ends=False; truyện thiếu trang). Trả [{j0, j1 (exclusive), n_match, n_pair}] theo cột."""
    bounds = []
    c0 = 0
    for sz in col_sizes:
        bounds.append((c0, c0 + sz)); c0 += sz
    per = [dict(j0=None, j1=None, n_match=0, n_pair=0) for _ in col_sizes]
    for i, j, ok in pairs:
        k = next((kk for kk, (a, b) in enumerate(bounds) if a <= i < b), None)
        if k is None:
            continue
        d = per[k]
        d["n_pair"] += 1; d["n_match"] += int(ok)
        d["j0"] = j if d["j0"] is None else min(d["j0"], j)
        d["j1"] = j + 1 if d["j1"] is None else max(d["j1"], j + 1)
    # âm kẽ: chia trung điểm giữa cột có cặp trước và cột có cặp sau
    have = [k for k, d in enumerate(per) if d["j0"] is not None]
    for a, b in zip(have, have[1:]):
        ja, jb = per[a]["j1"], per[b]["j0"]
        if jb > ja:
            mid = ja + (jb - ja) // 2
            per[a]["j1"] = mid; per[b]["j0"] = mid
    if include_ends and have:
        per[have[0]]["j0"] = 0; per[have[-1]]["j1"] = n_syl
    return per


def align_story_columns(cols_chars: list[list[str | None]], syls: list[str], n2q: dict[str, set[str]],
                        min_match: int = PROSE_MIN_MATCH, min_ratio: float = PROSE_MIN_RATIO) -> tuple[list[dict], dict]:
    """Ghép cả truyện: cột theo thứ tự đọc → [{syllables | None, n_match, n_chars, ratio, j0, j1}], info."""
    flat = [ch for col in cols_chars for ch in col]
    pairs, score = align_sequences(flat, syls, n2q)
    spans = split_syllables_by_column([len(c) for c in cols_chars], pairs, len(syls))
    out = []
    n_ok = 0
    for col, sp in zip(cols_chars, spans):
        n = len(col)
        ratio = sp["n_match"] / n if n else 0.0
        good = sp["j0"] is not None and sp["n_match"] >= min_match and ratio >= min_ratio and sp["j1"] > sp["j0"]
        n_ok += int(good)
        out.append(dict(syllables=(syls[sp["j0"]:sp["j1"]] if good else None), n_match=sp["n_match"], n_pair=sp["n_pair"],
                        n_chars=n, ratio=round(ratio, 3), j0=sp["j0"], j1=sp["j1"], matched=good))
    n_match_total = sum(p[2] for p in pairs)
    covered = {j for _, j, _ in pairs}
    info = dict(score=round(score, 1), n_chars=len(flat), n_syll=len(syls), n_pairs=len(pairs), n_match=n_match_total,
                match_ratio=round(n_match_total / len(flat), 3) if flat else None,
                syll_coverage=round(len(covered) / len(syls), 3) if syls else None,
                n_cols=len(cols_chars), n_cols_matched=n_ok)
    return out, info


def placeholder_line(n: int) -> list[str]:
    return [CONTENT_PLACEHOLDER] * max(1, n)


# ---------------------------------------------------------------------------
# 3. Chạy
# ---------------------------------------------------------------------------
def _sel_pages(book: str, pages: list[int] | None, limit: int | None) -> list[int]:
    cfg = BOOKS[book]
    allp = [cfg["page_of_canvas"](c) for c in cfg["canvases"]]
    sel = allp if not pages else [p for p in pages if p in allp]
    if pages and len(sel) != len(pages):
        raise SystemExit(f"[ingest_prose] trang ngoài 1..{len(allp)}: {sorted(set(pages) - set(sel))}")
    return sel[:limit] if limit else sel


def ingest(book: str, pages: list[int] | None, limit: int | None, ocr: str, force: bool, out_root: Path,
           measure_dir: Path, contrast: str = "stretch", kim_src: str = "orig", verbose: bool = True,
           kim: dict | None = None) -> dict:
    """Chạy adapter cho các trang chọn; trả manifest (đã ghi ra out_root/<book>/manifest.json)."""
    import cv2
    from core.image.image_processing import denoise_image
    from core.ocr.ocr_api import _file_md5, _pixel_hash, verify_cache_image

    cfg = BOOKS[book]
    kim = kim_params_of(kim)
    kim_sfx = kim_cache_suffix(kim)
    sel = _sel_pages(book, pages, limit)
    table = load_story_table(measure_dir, book)
    stories = load_story_syllables(measure_dir, book, table)
    n2q = _nom_to_qn_readings() if ocr == "kim" else {}
    out = out_root / book
    dirs = {k: out / k for k in ("pages", "pages_denoised", "detected", "transcriptions", "kim_raw")}
    for k, d in dirs.items():
        if k != "kim_raw" or ocr == "kim":
            d.mkdir(parents=True, exist_ok=True)

    # --- giai đoạn 1: ảnh + bố cục + kim từng trang ---
    P: dict[int, dict] = {}
    g = dict(n_pages=0, n_pages_blank=0, cols_per_page={}, n_cols_measure=0, n_cols_out=0, n_cols_no_kim=0,
             n_cols_matched=0, n_cols_placeholder=0, n_cols_no_story=0, n_chars_kim=0, n_chars_matched=0,
             n_boxes_unassigned=0, n_chars_unassigned=0, n_cache_ok=0, ocr_calls=0, kim_failed_pages=[],
             boundary_mid_page=[], reading_order={}, stories={})
    for p in sel:
        name = f"page_{p:04d}"
        canvas = cfg["canvas_of_page"](p)
        src = cfg["src"] / f"canvas_{canvas:04d}.jpg"
        png = dirs["pages"] / f"{name}.png"
        if force or not png.exists():
            prepare_image(src, png, contrast)
        den = dirs["pages_denoised"] / f"{name}.png"
        if force or not den.exists():
            gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(den), denoise_image(gray))
        img_hash = _file_md5(str(png))
        lay = page_columns(canvas)
        flags: list[str] = []
        if lay["blank"]:
            flags.append("blank_page")
        boxes_raw: list[dict] = []
        per: dict[int, list[dict]] = {}
        stats: dict = {}
        box_source = "projection"
        if ocr == "kim" and not lay["blank"]:
            ocr_img = src if kim_src == "orig" else png
            ocr_hash = _file_md5(str(ocr_img))
            raw_path = dirs["kim_raw"] / f"{name}{kim_sfx}.json"
            had = raw_path.exists() and not force
            boxes = kim_boxes(ocr_img, raw_path, ocr_hash, force, kim=kim)
            if not had:
                g["ocr_calls"] += 1
            if boxes is None:
                flags.append("kim_failed")
                g["kim_failed_pages"].append(name)
            else:
                boxes_raw = boxes
                per, stats = assign_boxes(boxes, lay["cols"], lay["col_pitch"])
                box_source = "kim"
                g["n_boxes_unassigned"] += stats["n_boxes_unassigned"]
                g["n_chars_unassigned"] += stats["n_chars_unassigned"]
        cols_out = []          # theo thứ tự đọc (slot tăng = trái→phải)
        for c in lay["cols"]:
            if box_source == "kim":
                chars = per.get(c["slot"], [])
                if not chars:
                    flags.append(f"slot{c['slot']}:no_kim_chars(n_est={c['n_est']})")
                    g["n_cols_no_kim"] += 1
                    continue
            else:
                chars = projection_chars(c)
            so = story_of_column(canvas, c["slot"], table)
            cols_out.append(dict(slot=c["slot"], story=so, chars=chars, layout=c))
            if so is None:
                flags.append(f"slot{c['slot']}:no_story")
        for r in table:
            if r["nom_canvas_dau"] == canvas and r["nom_slot_dau"] > 0:
                g["boundary_mid_page"].append(dict(page=name, canvas=canvas, slot=r["nom_slot_dau"], story=r["so"]))
        g["n_cols_measure"] += len(lay["cols"])
        P[p] = dict(name=name, canvas=canvas, png=png, img_hash=img_hash, lay=lay, cols=cols_out, flags=flags,
                    boxes_raw=boxes_raw, stats=stats, box_source=box_source, ocr_img=str(src if kim_src == "orig" else png))
        if verbose:
            print(f"  {name} (canvas {canvas}): {len(lay['cols'])} ô cột, {len(cols_out)} cột có chữ, "
                  f"chữ={sum(len(c['chars']) for c in cols_out)}, src={box_source}", flush=True)

    # --- giai đoạn 2: ghép QN theo truyện (DP trên toàn bộ cột của truyện trong tập trang chọn) ---
    if not cfg["read_ltr"]:
        for d in P.values():
            d["cols"].reverse()
    by_story: dict[int, list[tuple[int, int]]] = {}
    for p in sorted(P):
        for k, c in enumerate(P[p]["cols"]):
            if c["story"] is not None:
                by_story.setdefault(c["story"], []).append((p, k))
    for so, refs in sorted(by_story.items()):
        st = stories.get(so)
        if st is None or any(P[p]["box_source"] != "kim" for p, _ in refs):
            for p, k in refs:
                P[p]["cols"][k]["qn"] = None
                P[p]["cols"][k]["dp"] = dict(matched=False, reason="no_kim" if st else "no_story_text")
            continue
        cols_chars = [[ch["char"] for ch in P[p]["cols"][k]["chars"]] for p, k in refs]
        res, info = align_story_columns(cols_chars, st["syllables"], n2q)
        # kiểm thứ tự đọc: đảo cột trong từng trang rồi ghép lại (chỉ báo, không đổi kết quả)
        idx_of = {ref: r for r, ref in enumerate(refs)}
        rev = []
        for p in sorted({p for p, _ in refs}):
            ks = sorted(k for pp, k in refs if pp == p)
            rev += [cols_chars[idx_of[(p, k)]] for k in reversed(ks)]
        _, info_rev = align_story_columns(rev, st["syllables"], n2q)
        info["score_reversed_in_page"] = info_rev["score"]
        info["pages"] = sorted({P[p]["name"] for p, _ in refs})
        info["n_syll_title"] = len(st["title_syllables"])
        g["stories"][so] = info
        for (p, k), r in zip(refs, res):
            P[p]["cols"][k]["qn"] = r["syllables"]
            P[p]["cols"][k]["dp"] = {kk: v for kk, v in r.items() if kk != "syllables"}
    for p in sorted(P):
        for c in P[p]["cols"]:
            if c["story"] is None:
                c["qn"] = None
                c["dp"] = dict(matched=False, reason="no_story")
            elif "qn" not in c:
                c["qn"] = None
                c["dp"] = dict(matched=False, reason="no_story_text")

    # --- giai đoạn 3: ghi cache / transcriptions theo quy ước engine (cột 1 = phải nhất) ---
    results = []
    flagged: dict[str, list[str]] = {}
    for p in sorted(P):
        d = P[p]
        name = d["name"]
        flags = d["flags"]
        cols_rtl = sorted(d["cols"], key=lambda c: -c["layout"]["xc"])
        n_read = len(d["cols"])
        columns_cache = []
        cols_txt = []
        for k, c in enumerate(cols_rtl, start=1):
            chars = c["chars"]
            columns_cache.append([dict(char=ch["char"], y_center=ch["y_center"], bbox=ch["bbox"]) for ch in chars])
            reading_order = d["cols"].index(c) + 1
            if c["qn"]:
                syl = list(c["qn"])
                raw = " ".join(syl)
                g["n_cols_matched"] += 1
                g["n_chars_matched"] += int(c["dp"].get("n_match", 0))
            else:
                syl = placeholder_line(len(chars))
                raw = ""
                g["n_cols_placeholder"] += 1
                if c["story"] is None:
                    g["n_cols_no_story"] += 1
                flags.append(f"col{k}(slot{c['slot']}):placeholder:{c['dp'].get('reason', 'lowscore')}:"
                             f"match={c['dp'].get('n_match')}/{len(chars)}")
            g["n_chars_kim"] += len(chars) if d["box_source"] == "kim" else 0
            cols_txt.append(dict(column=k, reading_order=reading_order, slot=c["slot"], story=c["story"],
                                 raw_text=raw, syllables=syl, num_syllables=len(syl),
                                 n_chars_kim=len(chars), n_match=c["dp"].get("n_match"), dp_ratio=c["dp"].get("ratio"),
                                 syl_span=[c["dp"].get("j0"), c["dp"].get("j1")], matched=bool(c["qn"]),
                                 x_range=[c["layout"]["x0"], c["layout"]["x1"]], y_range=[c["layout"]["y0"], c["layout"]["y1"]]))
        cache_path = dirs["detected"] / f"{name}_ocr_cache.json"
        cache = dict(image=_rel(d["png"]), image_hash=d["img_hash"], pixel_hash=_pixel_hash(str(d["png"])),
                     framed=False, frame_pad=0, coords_space="fullpage",
                     n_columns=len(columns_cache), columns=columns_cache, boxes_raw=d["boxes_raw"],
                     layout="prose", box_source=d["box_source"], reading_direction="ltr" if cfg["read_ltr"] else "rtl",
                     kim_source_image=_rel(Path(d["ocr_img"])) if d["box_source"] == "kim" else None,
                     canvas=d["canvas"], col_pitch=d["lay"]["col_pitch"], frame=d["lay"]["frame"],
                     slots=[c["slot"] for c in cols_rtl])
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        vstat = verify_cache_image(str(cache_path), str(d["png"]))
        if vstat == "ok":
            g["n_cache_ok"] += 1
        else:
            flags.append(f"verify_cache={vstat}")
        (dirs["transcriptions"] / f"{name}.txt").write_text(
            "".join(" ".join(c["syllables"]) + "\n" for c in cols_txt), encoding="utf-8")
        (dirs["transcriptions"] / f"{name}.json").write_text(json.dumps(dict(
            book_page=p, canvas=d["canvas"], columns=cols_txt, qn_line_confidences=[], qn_page_confidence=None,
            layout="prose", reading_direction="ltr" if cfg["read_ltr"] else "rtl", n_columns=len(cols_txt),
            stories=sorted({c["story"] for c in d["cols"] if c["story"] is not None})),
            ensure_ascii=False, indent=1), encoding="utf-8")
        g["n_pages"] += 1
        g["n_pages_blank"] += int(d["lay"]["blank"])
        g["n_cols_out"] += len(cols_txt)
        g["cols_per_page"][name] = len(cols_txt)
        if flags:
            flagged[name] = flags
        results.append(dict(book_page=p, page_name=name, canvas=d["canvas"], source_file=f"canvas_{d['canvas']:04d}.jpg",
                            num_columns=len(cols_txt), n_cols_measure=len(d["lay"]["cols"]), n_read_order=n_read,
                            total_syllables=sum(c["num_syllables"] for c in cols_txt),
                            ocr_chars=sum(len(c) for c in columns_cache), box_source=d["box_source"],
                            n_cols_matched=sum(1 for c in cols_txt if c["matched"]),
                            stories=sorted({c["story"] for c in d["cols"] if c["story"] is not None}),
                            kim_stats=d["stats"] or None, flags=flags))
    g["match_ratio"] = round(g["n_chars_matched"] / g["n_chars_kim"], 3) if g["n_chars_kim"] else None
    g["cols_matched_ratio"] = round(g["n_cols_matched"] / g["n_cols_out"], 3) if g["n_cols_out"] else None
    g["reading_order"] = dict(configured="ltr" if cfg["read_ltr"] else "rtl",
                              n_stories_ltr_better=sum(1 for s in g["stories"].values() if s["score"] > s["score_reversed_in_page"]),
                              n_stories=len(g["stories"]))
    manifest = dict(book=book, source="images", layout="prose", n_columns="auto", contrast=contrast, ocr=ocr, kim_src=kim_src,
                    kim_params=kim, kim_cache_suffix=kim_sfx,
                    reading_direction="ltr" if cfg["read_ltr"] else "rtl", page_of_canvas="page = canvas - 105",
                    measure_dir=_rel(measure_dir), dp=dict(match=PROSE_MATCH, sub=PROSE_SUB, gap=PROSE_GAP,
                                                              min_match=PROSE_MIN_MATCH, min_ratio=PROSE_MIN_RATIO),
                    pages=results, total_pages=len(results), total_syllables=sum(r["total_syllables"] for r in results),
                    gates=dict(**g, pages_flagged=flagged, n_pages_flagged=len(flagged)))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def _parse_pages(s: str | None) -> list[int] | None:
    if not s:
        return None
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            out += list(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--book", required=True, choices=sorted(BOOKS))
    ap.add_argument("--limit", type=int, default=None, help="N trang đầu (page_0001…)")
    ap.add_argument("--pages", default=None, help="vd 1,2,5-8 (số trang page_XXXX, không phải canvas)")
    ap.add_argument("--ocr", choices=["kim", "none"], default="none")
    ap.add_argument("--force", action="store_true", help="ghi đè ảnh/cache kim đã có")
    ap.add_argument("--out", default=str(REPO / "prepared"))
    ap.add_argument("--measure-dir", default=str(REPO / "measure_out"))
    ap.add_argument("--contrast", choices=["stretch", "otsu", "none"], default="stretch")
    ap.add_argument("--kim-src", choices=["orig", "prepared"], default="orig",
                    help="ảnh gửi kim: orig = JPG gốc (kim đọc thạch bản tốt hơn), prepared = pages/*.png (cùng kích thước)")
    ap.add_argument("--kim-lang-type", type=int, default=None, choices=[0, 1, 2],
                    help="lang_type gửi kênh kim: 0 Tự động · 1 Hán · 2 Nôm. Vắng -> books[].kim_lang_type "
                         "của config/pipeline_<book>.yaml (vắng nữa -> 1 = bộ cũ); cache kim_raw/ tách theo tham số")
    ap.add_argument("--kim-ocr-id", type=int, default=None, choices=[-1, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--kim-font-type", type=int, default=None, choices=[0, 1, 2])
    ap.add_argument("--kim-config", default=None,
                    help="config đọc books[].kim_* (mặc định config/pipeline_<book>.yaml)")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    kim = book_kim_params(a.book, dict(ocr_id=a.kim_ocr_id, lang_type=a.kim_lang_type,
                                       font_type=a.kim_font_type),
                          Path(a.kim_config) if a.kim_config else None)
    if a.ocr == "kim":
        print(f"[ingest] kim: {kim} · cache kim_raw/*{kim_cache_suffix(kim)}.json", file=sys.stderr)
    m = ingest(a.book, _parse_pages(a.pages), a.limit, a.ocr, a.force, Path(a.out), Path(a.measure_dir),
               a.contrast, a.kim_src, kim=kim)
    gates = {k: v for k, v in m["gates"].items() if k not in ("pages_flagged", "cols_per_page", "stories")}
    print(json.dumps(dict(out=str(Path(a.out) / a.book), gates=gates,
                          stories={k: dict(score=v["score"], rev=v["score_reversed_in_page"], match=v["match_ratio"],
                                           cov=v["syll_coverage"], cols=f"{v['n_cols_matched']}/{v['n_cols']}")
                                   for k, v in m["gates"]["stories"].items()},
                          pages_flagged=list(m["gates"]["pages_flagged"])), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
