"""Adapter ingest sách thạch bản (LVT1883, KVK1884) → `prepared/<book>/` đúng hợp đồng engine.

Thay thế toàn bộ Bước 1 (step1_extract chỉ nhận PDF) cho sách ảnh JPG có bố cục
lục bát 2 tầng: mỗi trang 10 cột vật lý, mỗi cột = 1 cặp (6 chữ tầng trên + 8 chữ
tầng dưới). Phương án B1: engine thấy 10 cột/trang, mỗi cột 14 chữ, QN mỗi cột =
câu lẻ ⧺ câu chẵn (14 âm tiết). Không sửa mã engine, không đụng đường STT.

Đầu vào (đã đo bằng scripts/measure, 0 token LLM):
  data/<book>/pages/*.jpg
  measure_out/<book>/layout/layout_pages.csv   tầng/pitch/first_seq từng trang
  measure_out/<book>/layout/pages_full.json    x-range từng cột/tầng + char_boxes chiếu
  measure_out/<book>/qn_ocr/verses.tsv         verse_no → dòng QN (tesseract)

Đầu ra (`--out prepared` → prepared/<book>/):
  pages/page_XXXX.png            ảnh L đã kéo nền → 255 (stretch|otsu|none), giữ kích thước
  pages_denoised/page_XXXX.png   core.image.image_processing.denoise_image (như step1)
  detected/page_XXXX_ocr_cache.json  coords_space=fullpage, columns = 10 cột (phải→trái),
                                 mỗi cột = chữ tầng trên rồi tầng dưới (bbox toàn trang)
  transcriptions/page_XXXX.json + .txt   10 dòng = câu lẻ ⧺ câu chẵn
  kim_raw/page_XXXX.json         hộp thô kim (chỉ khi --ocr kim; engine không đọc)
  manifest.json                  khoá step1 + layout/gates/cờ từng trang

Chạy:
  .venv/bin/python -m pipeline.tools.ingest_lithograph_book --book LucVanTien1883 --limit 5 --ocr kim
  .venv/bin/python -m pipeline.tools.ingest_lithograph_book --book KimVanKieu1884 --pages 1,2,3 --ocr none
  --verse-map anchor : ghép QN theo đoạn neo (seq_no liên tiếp quanh neo số in gần nhất, xem anchor_rows_for_page)
  --verse-map content: (cần --ocr kim) ghép QN theo NỘI DUNG: mỗi cột Nôm (chữ kim) chọn cặp dòng QN liên tiếp (seq_no)
                       có nhiều chữ tra được âm nhất trong từ điển QuocNgu_SinoNom, đơn điệu theo cột (DP, xem
                       content_rows_for_page); cột không khớp (cột chú nhỏ, dòng QN thiếu) → cột QN rỗng + cờ.
                       Dùng khi số câu in của bản Nôm và bản QN đánh khác nhau (KVK1884 từ trang 55: Nôm = QN + 4).
  --plan-only        : chỉ so 2 cách ghép (formula/anchor) mọi trang chọn → verse_map_plan.json, không ghi ảnh/cache
Test: .venv/bin/python -m pipeline.tools.ingest_lithograph_selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# --- Cấu hình sách: mọi số lấy từ measure_out (KET_QUA_DO_CUOI_2026-09-21.md) ---
BOOKS = {
    "LucVanTien1883": dict(
        src=REPO / "data/LucVanTien1883/pages",
        pdf="data/LucVanTien1883/LucVanTien_1883_TranNguyenHanh.pdf",  # chỉ để khoá `pdf` tồn tại
        page_of_uid=lambda uid: uid,            # uid = số thứ tự tệp = trang
        total_verses=2088,
    ),
    "KimVanKieu1884": dict(
        src=REPO / "data/KimVanKieu1884/pages",
        pdf="data/KimVanKieu1884/KimVanKieu_1884.pdf",
        page_of_uid=lambda uid: 167 - uid,      # uid = canvas; đo 163/163
        total_verses=3256,
    ),
}
N_COLUMNS = 10            # cột vật lý mỗi trang (B1)
EXPECT_TIER = (6, 8)      # chữ tầng trên / dưới = câu lục / câu bát
TIER_MARGIN = 0.25        # × row_pitch: biên nhận chữ ngoài dải tầng
COL_TOL = 0.5             # × col_pitch: |cx − xc| tối đa để gán hộp vào cột
STRONG_ANCHORS = ("ocr_num", "margin_digits")   # neo = số câu in đọc được thật (không tính position_mod5/chain_interp)
ANCHOR_WINDOW = 100       # câu: neo cách dải câu của trang tối đa bấy nhiêu mới được xét
CONTENT_WINDOW = 60       # câu: --verse-map content xét dòng QN trong [first_seq − W, first_seq + 2n + W]
CONTENT_SKIP_PEN = 2      # điểm phạt khi bỏ một cột (không gán dòng QN nào)
CONTENT_MIN_COL = 4       # cột có điểm < ngưỡng (số chữ tra được âm / 14) → coi như không khớp


# ---------------------------------------------------------------------------
# 1. Layout + QN
# ---------------------------------------------------------------------------
def load_layout(book: str, measure_dir: Path) -> dict[int, dict]:
    """Đọc layout_pages.csv + pages_full.json → {page: layout}; chỉ trang chữ (category=text)."""
    lay_dir = measure_dir / book / "layout"
    rows = list(csv.DictReader(open(lay_dir / "layout_pages.csv", encoding="utf-8")))
    full = {int(p["uid"]): p for p in json.load(open(lay_dir / "pages_full.json", encoding="utf-8"))}
    chain = {int(r["page"]): r for r in csv.DictReader(open(lay_dir / "verse_chain.csv", encoding="utf-8"))
             if r.get("page")}
    page_of_uid = BOOKS[book]["page_of_uid"]
    out: dict[int, dict] = {}
    for r in rows:
        if r["category"] != "text" or not r["page"]:
            continue
        uid, page = int(r["uid"]), int(r["page"])
        if page_of_uid(uid) != page:
            raise ValueError(f"{book}: uid {uid} → page {page} khác quy tắc page_of_uid={page_of_uid(uid)}")
        fp = full[uid]
        first_seq = int(chain[page]["first_seq"]) if page in chain else int(fp["first_seq"])
        out[page] = dict(
            uid=uid, page=page, file=r["file"], W=int(r["W"]), H=int(r["H"]),
            tiers=[(int(r["tier0_y0"]), int(r["tier0_y1"])), (int(r["tier1_y0"]), int(r["tier1_y1"]))],
            row_pitch=int(r["row_pitch"]), col_pitch=int(r["col_pitch"]),
            first_seq=first_seq, n_pairs=int(r["n_pairs"]),
            cols=[{k: c[k] for k in ("tier", "slot", "x0", "x1", "xc", "y0", "y1", "n_chars", "expect", "char_boxes")}
                  for c in fp["cols"]],
        )
    return out


def build_pairs(cols: list[dict]) -> list[dict]:
    """Ghép cột tầng trên/dưới cùng `slot` → cặp; slot 1 = phải nhất. Slot chỉ có 1 tầng (cột tựa) bị bỏ."""
    by = {}
    for c in cols:
        by.setdefault(int(c["slot"]), {})[int(c["tier"])] = c
    return [dict(slot=s, top=t[0], bottom=t[1]) for s, t in sorted(by.items()) if 0 in t and 1 in t]


def verse_pairs_for_page(first_seq: int, n_pairs: int) -> list[tuple[int, int]]:
    """Cột k (0-based, phải→trái) ↔ (câu lẻ, câu chẵn) = first_seq + 2k, +1."""
    return [(first_seq + 2 * k, first_seq + 2 * k + 1) for k in range(n_pairs)]


def load_verses(tsv: Path) -> tuple[dict[int, dict], set[int]]:
    """verses.tsv → ({verse_no: dòng}, {verse_no trùng}); verse_no trùng giữ dòng đầu và bị cờ."""
    verses: dict[int, dict] = {}
    dups: set[int] = set()
    for r in csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"):
        v = int(r["verse_no"])
        if v in verses:
            dups.add(v)
            continue
        verses[v] = r
    return verses, dups


def check_coverage(pairs: list[tuple[int, int]], verses: dict[int, dict], dups: set[int]) -> dict:
    """Cổng cứng: mọi câu của trang phải có đúng 1 dòng trong verses.tsv."""
    need = [v for p in pairs for v in p]
    return dict(missing=[v for v in need if v not in verses], duplicated=[v for v in need if v in dups])


def load_verse_rows(tsv: Path) -> list[dict]:
    """verses.tsv → danh sách dòng theo seq_no (1..N liên tiếp, = thứ tự dòng vật lý trong bản QN)."""
    rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
    rows.sort(key=lambda r: int(r["seq_no"]))
    if [int(r["seq_no"]) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"{tsv}: seq_no không liên tiếp 1..{len(rows)}")
    return rows


def anchor_rows_for_page(first_seq: int, n_pairs: int, rows: list[dict],
                         window: int = ANCHOR_WINDOW) -> tuple[list[dict] | None, dict]:
    """--verse-map anchor: trang Nôm cần 2·n_pairs câu từ first_seq → lấy đúng bấy nhiêu DÒNG LIÊN TIẾP (seq_no)
    của verses.tsv, bắt đầu tại seq = seq(neo) + (first_seq − verse_no(neo)); neo = dòng có số câu in đọc được
    (STRONG_ANCHORS) trong ±window câu. Chọn neo theo (điểm nhịp lục bát: dòng lẻ 6 âm / chẵn 8 âm cao nhất,
    rồi neo gần dải câu nhất, rồi |offset| nhỏ nhất). Trả (dòng đã chọn | None nếu không có neo, thông tin neo);
    `differs` = vị trí mà verse_no trong tsv ≠ first_seq + i (khác cách ghép công thức)."""
    need = 2 * n_pairs
    lo, hi = first_seq, first_seq + need - 1
    info = dict(mode="anchor", first_seq=first_seq, need=need, n_candidates=0, offsets_in_range=[], conflict=False)
    best = None
    for a in rows:
        if a.get("anchor_source") not in STRONG_ANCHORS:
            continue
        v, sq = int(a["verse_no"]), int(a["seq_no"])
        if v < lo - window or v > hi + window:
            continue
        start = sq + (first_seq - v)
        if start < 1 or start + need - 1 > len(rows):
            continue
        sel = rows[start - 1:start - 1 + need]
        parity = sum(1 for i, r in enumerate(sel) if int(r["n_syll"] or 0) == EXPECT_TIER[i % 2])
        dist = 0 if lo <= v <= hi else min(abs(v - lo), abs(v - hi))
        key = (parity, -dist, -abs(v - sq))
        info["n_candidates"] += 1
        if dist == 0 and (v - sq) not in info["offsets_in_range"]:
            info["offsets_in_range"].append(v - sq)
        if best is None or key > best[0]:
            best = (key, sel, dict(anchor_verse=v, anchor_seq=sq, offset=v - sq, start_seq=start, parity=parity,
                                   anchor_source=a["anchor_source"], dist=dist))
    info["offsets_in_range"].sort()
    info["conflict"] = len(info["offsets_in_range"]) > 1
    if best is None:
        info["reason"] = "no_anchor"
        return None, info
    sel = best[1]
    info.update(best[2])
    info["differs"] = [i for i, r in enumerate(sel) if int(r["verse_no"]) != first_seq + i]
    return sel, info


def _nom_to_qn_readings() -> dict[str, set[str]]:
    """Từ điển chữ Nôm → tập âm (đã chuẩn hoá dấu) từ Dict/QuocNgu_SinoNom.csv của pipeline (nạp 1 lần)."""
    global _N2Q
    try:
        return _N2Q
    except NameError:
        pass
    from core.text.dictionary import build_nom_to_qn, load_qn_to_nom
    n2q = build_nom_to_qn(load_qn_to_nom(str(REPO / "Dict/QuocNgu_SinoNom.csv")))
    _N2Q = {k: set(v) for k, v in n2q.items()}
    return _N2Q


def _line_syllable_set(row: dict) -> set[str]:
    from core.text.text_utils import normalize_tone_marks
    return {normalize_tone_marks(t.lower()) for t in lithograph_syllables(row["line_text"])}


def content_rows_for_page(first_seq: int, cols_chars: list[tuple[list[str], list[str]]], rows: list[dict],
                          n2q: dict[str, set[str]] | None = None, window: int = CONTENT_WINDOW,
                          skip_pen: int = CONTENT_SKIP_PEN, min_col: int = CONTENT_MIN_COL,
                          line_sets: list[set[str]] | None = None) -> tuple[list[tuple[dict, dict] | None], dict]:
    """--verse-map content: cột k (chữ kim tầng trên, tầng dưới) ↔ 2 dòng QN liên tiếp (seq a, a+1) sao cho tổng
    số chữ tra được âm (n2q[chữ] ∩ âm của dòng) lớn nhất, với ràng buộc a tăng dần theo cột (a_{k+1} ≥ a_k + 2);
    một cột có thể bị bỏ (phạt skip_pen) khi không dòng nào khớp (cột chú nhỏ, dòng QN thiếu). Ứng viên a trong
    [first_seq − window, first_seq + 2n + window] ∩ [1, len(rows) − 1]. Trả ([(dòng lẻ, dòng chẵn) | None] theo cột,
    info: start_seq/offset từng cột (offset = a − (first_seq + 2k)), điểm từng cột, tổng điểm, cột bỏ)."""
    n2q = n2q if n2q is not None else _nom_to_qn_readings()
    n = len(cols_chars)
    lo = max(1, first_seq - window)
    hi = min(len(rows) - 1, first_seq + 2 * n + window)
    cand = list(range(lo, hi + 1))
    if line_sets is None:
        line_sets = [_line_syllable_set(r) for r in rows]

    def sc(chars, syls):
        return sum(1 for ch in chars if n2q.get(ch, set()) & syls)

    S = [[sc(t, line_sets[a - 1]) + sc(b, line_sets[a]) for a in cand] for (t, b) in cols_chars]
    M = len(cand)
    NEG = -10 ** 9
    # dp[k][j]: tổng điểm tốt nhất cho cột 0..k khi cột k gán cand[j]; skip[k]: cột 0..k với cột k bị bỏ,
    # kèm chỉ số j cuối cùng đã gán (để giữ đơn điệu) → lưu dạng dp_last[k][j'] = tốt nhất khi "cột gán cuối" là j'.
    # Dùng bảng best_end[k][j] = điểm tốt nhất cho cột 0..k mà cột ĐÃ GÁN CUỐI CÙNG có chỉ số j (j = -1: chưa gán).
    best = [dict() for _ in range(n + 1)]
    best[0] = {-1: (0, [])}
    for k in range(n):
        nxt: dict[int, tuple[int, list]] = {}
        for j, (v, path) in best[k].items():
            # bỏ cột k
            cand_v = v - skip_pen
            if j not in nxt or cand_v > nxt[j][0]:
                nxt[j] = (cand_v, path + [None])
            # gán cột k = cand[jj] > j (và cách ≥ 2 dòng)
            start = 0 if j < 0 else j + 1
            for jj in range(start, M):
                if j >= 0 and cand[jj] < cand[j] + 2:
                    continue
                if S[k][jj] < min_col:
                    continue
                cand_v = v + S[k][jj]
                if jj not in nxt or cand_v > nxt[jj][0]:
                    nxt[jj] = (cand_v, path + [jj])
        best[k + 1] = nxt
    tot, path = max(best[n].values(), key=lambda t: t[0]) if best[n] else (NEG, [None] * n)
    sel: list[tuple[dict, dict] | None] = []
    starts, offsets, scores = [], [], []
    for k, jj in enumerate(path):
        if jj is None:
            sel.append(None); starts.append(None); offsets.append(None); scores.append(None)
        else:
            a = cand[jj]
            sel.append((rows[a - 1], rows[a])); starts.append(a); offsets.append(a - (first_seq + 2 * k)); scores.append(S[k][jj])
    info = dict(mode="content", first_seq=first_seq, n_cols=n, total_score=tot, n_chars=sum(len(t) + len(b) for t, b in cols_chars),
                start_seq=starts, offset=offsets, col_score=scores, skipped=[k + 1 for k, x in enumerate(sel) if x is None],
                offsets_distinct=sorted({o for o in offsets if o is not None}))
    return sel, info


CONTENT_PLACEHOLDER = "khongkhop"   # 9 ký tự -> is_plausible_qn_syllable() = False -> ô chỉ có thể là REVIEW


def _placeholder_row(verse_no: int, n_syll: int) -> dict:
    """Dòng QN giữ chỗ cho cột không khớp nội dung (cột chú nhỏ / dòng QN thiếu): đủ n_syll token
    CONTENT_PLACEHOLDER (engine bỏ dòng .txt rỗng nên phải giữ số cột; token không hợp lệ -> mọi ô của cột
    thành REVIEW/not_plausible, không lọt GOLD/SYLLABLE), cờ content_unmatched."""
    return dict(verse_no=verse_no, page="", line_text=" ".join([CONTENT_PLACEHOLDER] * n_syll), n_syll=n_syll,
                anchor_source="", confidence="", seq_no=0, expect_syll="", num_read="", margin_read="",
                line_flag="content_unmatched", page_flag="content_unmatched")


def lithograph_syllables(text: str) -> list[str]:
    """QN dòng thơ → âm tiết: clean_line_text + split_to_syllables của pipeline, bỏ token không có chữ cái
    (số câu in, ký hiệu tesseract như '1⁄25')."""
    from core.text.text_utils import clean_line_text, split_to_syllables
    return [t for t in split_to_syllables(clean_line_text(text)) if any(ch.isalpha() for ch in t)]


def make_column_texts(vpairs: list[tuple[int, int]], verses: dict[int, dict]) -> tuple[list[dict], list[str]]:
    """Mỗi cột = câu lẻ ⧺ câu chẵn → dict như step1 (column, raw_text, cleaned_text, syllables, num_syllables)
    + verse_odd/verse_even; cờ khi số âm tiết ≠ 6/8."""
    from core.text.text_utils import clean_line_text
    cols, flags = [], []
    for k, (vo, ve) in enumerate(vpairs, start=1):
        parts = []
        for v, exp in ((vo, EXPECT_TIER[0]), (ve, EXPECT_TIER[1])):
            row = verses[v]
            syl = lithograph_syllables(row["line_text"])
            info = dict(verse_no=v, text=row["line_text"], n_syll=len(syl), expect=exp,
                        n_syll_tsv=int(row.get("n_syll") or 0), page_qn=row.get("page", ""),
                        anchor_source=row.get("anchor_source", ""), page_flag=row.get("page_flag", ""),
                        line_flag=row.get("line_flag", ""),
                        verse_no_tsv=int(row.get("verse_no") or v), seq_no=int(row.get("seq_no") or 0))
            if len(syl) != exp:
                flags.append(f"col{k}:verse{v}:n_syll={len(syl)}!={exp}")
            parts.append((row["line_text"], syl, info))
        raw = " ".join(p[0] for p in parts)
        syllables = parts[0][1] + parts[1][1]
        cols.append(dict(column=k, raw_text=raw, cleaned_text=clean_line_text(raw), syllables=syllables,
                         num_syllables=len(syllables), verse_odd=parts[0][2], verse_even=parts[1][2],
                         len_odd=len(parts[0][1])))
    return cols, flags


# ---------------------------------------------------------------------------
# 2. Hộp kim → cột/tầng
# ---------------------------------------------------------------------------
def box_rect(box: dict) -> tuple[int, int, int, int]:
    """Tứ giác `points` → hộp thẳng (x_left, y_top, x_right, y_bot)."""
    xs = [p[0] for p in box["points"]]
    ys = [p[1] for p in box["points"]]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def expand_box_chars(box: dict) -> list[dict]:
    """Chia đều chiều cao hộp cho từng ký tự (cùng quy ước boxes_to_columns của ocr_api)."""
    text = box.get("transcription", "") or ""
    chars = [ch for ch in text.strip() if ch.strip()]
    if not chars:
        return []
    x0, y0, x1, y1 = box_rect(box)
    h = (y1 - y0) / len(chars)
    return [dict(char=ch, y_center=y0 + h * (i + 0.5),
                 bbox=[x0, int(y0 + h * i), x1, int(y0 + h * (i + 1))]) for i, ch in enumerate(chars)]


def tier_of_y(cy: float, tiers: list[tuple[int, int]], row_pitch: int) -> int | None:
    """Tầng theo tâm y: ngoài [tier0_y0−m, tier1_y1+m] → None (số câu in lề, rác); còn lại chia tại giữa khe tầng."""
    m = TIER_MARGIN * row_pitch
    (a0, a1), (b0, b1) = tiers
    if cy < a0 - m or cy > b1 + m:
        return None
    return 0 if cy < (a1 + b0) / 2 else 1


def is_number_box(n_chars: int, y_bot: int, tiers: list[tuple[int, int]], row_pitch: int) -> bool:
    """Số câu in trên đỉnh tầng (kim đọc thành 1–2 chữ Hán như '品', '口', '四'): hộp ≤ 2 chữ có đáy nằm
    trên tier_y0 + 0,25×row_pitch của một tầng (đo LVT: số cách đỉnh tầng 6–49 px, KVK 25–100 px)."""
    if n_chars > 2:
        return False
    return any(y_bot < y0 + TIER_MARGIN * row_pitch for y0, _ in tiers)


def assign_boxes_to_columns(boxes: list[dict], layout: dict, pairs: list[dict]) -> tuple[list[list[dict]], dict]:
    """Gán hộp kim vào (slot, tầng) theo tâm hộp; trả cột = chữ tầng trên rồi tầng dưới (trên→dưới).

    Không ép số chữ về 6/8: hộp thật được giữ, chỉ đếm vào `stats` để adapter gắn cờ.
    """
    tiers, rp, cp = layout["tiers"], layout["row_pitch"], layout["col_pitch"]
    slot_cols = {(int(c["tier"]), int(c["slot"])): c for c in layout["cols"]}
    pair_slots = {p["slot"] for p in pairs}
    per: dict[tuple[int, int], list[dict]] = {}
    stats = dict(n_boxes=len(boxes), n_chars=0, n_unassigned_tier=0, n_unassigned_col=0, n_chars_nonpair=0,
                 n_number_boxes=0)
    for box in boxes:
        chars = expand_box_chars(box)
        stats["n_chars"] += len(chars)
        if not chars:
            continue
        x0, y0, x1, y1 = box_rect(box)
        cx = (x0 + x1) / 2
        if is_number_box(len(chars), y1, tiers, rp):
            stats["n_number_boxes"] += 1
            continue
        for ch in chars:
            t = tier_of_y(ch["y_center"], tiers, rp)
            if t is None:
                stats["n_unassigned_tier"] += 1
                continue
            cands = [(abs(cx - int(c["xc"])), int(c["slot"])) for (tt, _), c in slot_cols.items() if tt == t]
            if not cands:
                stats["n_unassigned_col"] += 1
                continue
            d, slot = min(cands)
            if d > COL_TOL * cp:
                stats["n_unassigned_col"] += 1
                continue
            if slot not in pair_slots:
                stats["n_chars_nonpair"] += 1
                continue
            per.setdefault((slot, t), []).append(ch)
    columns, per_col = [], []
    for p in pairs:
        top = sorted(per.get((p["slot"], 0), []), key=lambda c: c["y_center"])
        bot = sorted(per.get((p["slot"], 1), []), key=lambda c: c["y_center"])
        columns.append(top + bot)
        per_col.append((len(top), len(bot)))
    stats["chars_per_tier"] = per_col
    return columns, stats


def projection_columns(pairs: list[dict]) -> list[list[dict]]:
    """--ocr none: char=None, bbox từ char_boxes chiếu mực của measure (x = x-range cột)."""
    columns = []
    for p in pairs:
        col = []
        for c in (p["top"], p["bottom"]):
            x0, x1 = int(c["x0"]), int(c["x1"])
            for y0, y1 in c.get("char_boxes") or []:
                col.append(dict(char=None, y_center=(y0 + y1) / 2, bbox=[x0, int(y0), x1, int(y1)]))
        columns.append(col)
    return columns


# ---------------------------------------------------------------------------
# 3. Ảnh
# ---------------------------------------------------------------------------
def stretch_gray(gray, lo_pct: float = 2.0, hi_pct: float = 90.0):
    """Kéo tương phản tuyến tính: mực (p2) → 0, nền (p90) → 255 (cùng scripts/measure/detector_transfer.py)."""
    import numpy as np
    lo, hi = np.percentile(gray, lo_pct), np.percentile(gray, hi_pct)
    if hi - lo < 10:
        return gray
    return np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def otsu_gray(gray):
    """Nền → 255 theo ngưỡng Otsu, giữ mức xám của mực (không nhị phân hoá toàn bộ)."""
    import cv2
    import numpy as np
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = gray.copy()
    out[gray > thr] = 255
    return np.ascontiguousarray(out)


def prepare_image(src: Path, dst: Path, contrast: str):
    """JPG → PNG mode L cùng kích thước; nền kéo về 255 theo `contrast`. Trả (gray_gốc, gray_ghi)."""
    import numpy as np
    from PIL import Image
    gray = np.asarray(Image.open(src).convert("L"))
    if contrast == "stretch":
        out = stretch_gray(gray)
    elif contrast == "otsu":
        out = otsu_gray(gray)
    else:
        out = gray
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst, "PNG")
    return gray, out


# ---------------------------------------------------------------------------
# 4. Kim
# ---------------------------------------------------------------------------
def kim_boxes(png: Path, raw_path: Path, image_hash: str, force: bool) -> list[dict] | None:
    """upload_image + recognize của core.ocr.ocr_api (không tự viết client); cache hộp thô ở kim_raw/."""
    if raw_path.exists() and not force:
        try:
            d = json.loads(raw_path.read_text(encoding="utf-8"))
            if d.get("image_hash") == image_hash and isinstance(d.get("boxes"), list):
                return d["boxes"]
        except Exception:
            pass
    from core.ocr import ocr_api
    fname = ocr_api.upload_image(str(png))
    if not fname:
        return None
    boxes = ocr_api.recognize(fname)
    if boxes is None:
        return None
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(dict(image=str(png), image_hash=image_hash, boxes=boxes),
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    return boxes


# ---------------------------------------------------------------------------
# 5. Chạy
# ---------------------------------------------------------------------------
def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


def ingest(book: str, pages: list[int] | None, limit: int | None, ocr: str, force: bool, out_root: Path,
           measure_dir: Path, contrast: str, skip_uncovered: bool, verbose: bool = True,
           verse_map: str = "formula", plan_only: bool = False) -> dict:
    """Chạy adapter cho các trang chọn; trả manifest (đã ghi ra out_root/<book>/manifest.json).

    verse_map: "formula" (mặc định, hành vi cũ: câu = first_seq + 2k theo verse_no) | "anchor" (đoạn neo, xem
    anchor_rows_for_page). plan_only: chỉ ghi verse_map_plan.json so sánh 2 cách, không ghi ảnh/cache."""
    from core.ocr.ocr_api import _file_md5, _pixel_hash, verify_cache_image

    cfg = BOOKS[book]
    layout = load_layout(book, measure_dir)
    verses, dups = load_verses(measure_dir / book / "qn_ocr" / "verses.tsv")
    vrows = load_verse_rows(measure_dir / book / "qn_ocr" / "verses.tsv")
    sel = sorted(layout) if not pages else [p for p in pages if p in layout]
    if pages and len(sel) != len(pages):
        raise SystemExit(f"[ingest] trang không có trong layout: {sorted(set(pages) - set(sel))}")
    if limit:
        sel = sel[:limit]

    # Cổng cứng: phủ câu QN trước khi ghi bất cứ gì
    plan, uncovered, vmap_plan = {}, {}, {}
    for p in sel:
        L = layout[p]
        pairs = build_pairs(L["cols"])
        vpairs = verse_pairs_for_page(L["first_seq"], len(pairs))
        cov_formula = check_coverage(vpairs, verses, dups)
        arows, ainfo = anchor_rows_for_page(L["first_seq"], len(pairs), vrows)
        if arows is not None:
            verses_anchor = {L["first_seq"] + i: r for i, r in enumerate(arows)}
            cov_anchor = dict(missing=[], duplicated=[])
        else:
            verses_anchor, cov_anchor = verses, cov_formula   # không có neo → rơi về công thức
        differs = arows is None or bool(ainfo.get("differs")) or bool(cov_formula["missing"] or cov_formula["duplicated"])
        vmap_plan[p] = dict(page=p, first_seq=L["first_seq"], n_pairs=len(pairs), formula=cov_formula,
                            anchor=ainfo, differs=differs)
        if verse_map == "anchor":
            verses_page, cov = verses_anchor, cov_anchor
        elif verse_map == "content":
            # phủ câu do nội dung quyết từng cột sau khi có chữ kim; kế hoạch neo/công thức chỉ là dự phòng khi kim hỏng
            verses_page, cov = verses_anchor, dict(missing=[], duplicated=[])
        else:
            verses_page, cov = verses, cov_formula
        if cov["missing"] or cov["duplicated"]:
            uncovered[p] = cov
        plan[p] = (pairs, vpairs, verses_page, vmap_plan[p])
    if plan_only:
        out = out_root / book
        out.mkdir(parents=True, exist_ok=True)
        diff_pages = sorted(p for p, d in vmap_plan.items() if d["differs"])
        summary = dict(book=book, verse_map_compared=["formula", "anchor"], n_pages=len(sel),
                       pages_differ=diff_pages, n_pages_differ=len(diff_pages),
                       pages_formula_uncovered=sorted(p for p, d in vmap_plan.items()
                                                      if d["formula"]["missing"] or d["formula"]["duplicated"]),
                       pages_anchor_none=sorted(p for p, d in vmap_plan.items() if d["anchor"].get("reason")),
                       pages_anchor_conflict=sorted(p for p, d in vmap_plan.items() if d["anchor"].get("conflict")),
                       pages=vmap_plan)
        (out / "verse_map_plan.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        return dict(book=book, plan_only=True, gates=dict(pages_flagged={}, cols_per_page={},
                    **{k: v for k, v in summary.items() if k != "pages"}))
    if verse_map == "content" and ocr != "kim":
        raise SystemExit("[ingest] --verse-map content cần --ocr kim (ghép theo chữ kim)")
    if uncovered and not skip_uncovered:
        raise SystemExit(f"[ingest] CỔNG CỨNG: verses.tsv không phủ đủ câu của trang {sorted(uncovered)}: "
                         f"{json.dumps(uncovered, ensure_ascii=False)[:400]} (dùng --skip-uncovered để bỏ qua)")

    out = out_root / book
    dirs = {k: out / k for k in ("pages", "pages_denoised", "detected", "transcriptions", "kim_raw")}
    for k, d in dirs.items():
        if k != "kim_raw" or ocr == "kim":
            d.mkdir(parents=True, exist_ok=True)
    import cv2
    from core.image.image_processing import denoise_image

    results, flagged = [], {}
    g = dict(n_pages=0, n_pages_10cols=0, n_cols=0, n_cols_syll14=0, n_cols_kim14=0, n_cols_kim_6_8=0,
             n_boxes_unassigned=0, n_number_boxes=0, n_chars_nonpair=0, n_cache_ok=0, cols_per_page={}, n_pages_skipped=0,
             ocr_calls=0, n_cols_content_unmatched=0, n_pages_content_offset=0)
    line_sets = [_line_syllable_set(r) for r in vrows] if verse_map == "content" else None
    for p in sel:
        name = f"page_{p:04d}"
        if p in uncovered:
            g["n_pages_skipped"] += 1
            flagged[name] = [f"uncovered:{json.dumps(uncovered[p])}"]
            continue
        L = layout[p]
        pairs, vpairs, verses_page, vinfo = plan[p]
        flags: list[str] = []
        if verse_map == "anchor":
            if vinfo["anchor"].get("reason"):
                flags.append("verse_map:no_anchor→formula")
            elif vinfo["anchor"].get("differs"):
                flags.append(f"verse_map:anchor≠formula:offset={vinfo['anchor']['offset']}:"
                             f"start_seq={vinfo['anchor']['start_seq']}")
            if vinfo["anchor"].get("conflict"):
                flags.append(f"verse_map:anchor_conflict:offsets={vinfo['anchor']['offsets_in_range']}")
        if len(pairs) != N_COLUMNS:
            flags.append(f"n_pairs={len(pairs)}!={N_COLUMNS}")
        if len(pairs) != L["n_pairs"]:
            flags.append(f"n_pairs={len(pairs)}!=layout_n_pairs={L['n_pairs']}")

        # ảnh
        png = dirs["pages"] / f"{name}.png"
        src = cfg["src"] / L["file"]
        if force or not png.exists():
            prepare_image(src, png, contrast)
        den = dirs["pages_denoised"] / f"{name}.png"
        if force or not den.exists():
            gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(den), denoise_image(gray))
        img_hash = _file_md5(str(png))

        # Nôm
        stats = {}
        box_source = "projection"
        boxes_raw: list[dict] = []
        if ocr == "kim":
            raw_path = dirs["kim_raw"] / f"{name}.json"
            had = raw_path.exists() and not force
            boxes = kim_boxes(png, raw_path, img_hash, force)
            if not had:
                g["ocr_calls"] += 1
            if boxes is None:
                flags.append("kim_failed")
                columns = projection_columns(pairs)
            else:
                boxes_raw = boxes
                columns, stats = assign_boxes_to_columns(boxes, L, pairs)
                box_source = "kim"
                g["n_boxes_unassigned"] += stats["n_unassigned_tier"] + stats["n_unassigned_col"]
                g["n_chars_nonpair"] += stats["n_chars_nonpair"]
                g["n_number_boxes"] += stats["n_number_boxes"]
                for k, (nt, nb) in enumerate(stats["chars_per_tier"], start=1):
                    if (nt, nb) == EXPECT_TIER:
                        g["n_cols_kim_6_8"] += 1
                    else:
                        flags.append(f"col{k}:kim={nt}+{nb}!=6+8")
                    if nt + nb == sum(EXPECT_TIER):
                        g["n_cols_kim14"] += 1
        else:
            columns = projection_columns(pairs)
        for k, col in enumerate(columns, start=1):
            if len(col) < 4:
                flags.append(f"col{k}:n_chars={len(col)}<4")

        # QN (content: ghép theo chữ kim sau khi đã có cột Nôm)
        cinfo = None
        if verse_map == "content" and stats:
            cols_chars = []
            for col, (nt, nb) in zip(columns, stats["chars_per_tier"]):
                chars = [b["char"] for b in col]
                cols_chars.append((chars[:nt], chars[nt:]))
            csel, cinfo = content_rows_for_page(L["first_seq"], cols_chars, vrows, line_sets=line_sets)
            verses_page = {}
            for k, (vo, ve) in enumerate(vpairs):
                if csel[k] is None:
                    verses_page[vo], verses_page[ve] = _placeholder_row(vo, EXPECT_TIER[0]), _placeholder_row(ve, EXPECT_TIER[1])
                else:
                    verses_page[vo], verses_page[ve] = csel[k]
            flags.append(f"verse_map:content:offset={cinfo['offset']}")
            if cinfo["skipped"]:
                flags.append(f"verse_map:content_unmatched:cols={cinfo['skipped']}")
            if cinfo["total_score"] < CONTENT_MIN_COL * max(1, len(columns)) // 2:
                flags.append(f"verse_map:content_lowscore={cinfo['total_score']}")
            g["n_cols_content_unmatched"] += len(cinfo["skipped"])
            g["n_pages_content_offset"] += int(any(o for o in cinfo["offset"] if o))
        elif verse_map == "content":
            flags.append("verse_map:content_no_kim→anchor/formula")
        cols_txt, tflags = make_column_texts(vpairs, verses_page)
        flags += tflags
        for c in cols_txt:
            for key in ("page_flag", "line_flag"):
                for v in (c["verse_odd"], c["verse_even"]):
                    if v.get(key):
                        flags.append(f"col{c['column']}:verse{v['verse_no']}:{key}={v[key]}")

        tier_split = stats["chars_per_tier"] if stats else [
            (len(q["top"].get("char_boxes") or []), len(q["bottom"].get("char_boxes") or [])) for q in pairs]

        # cache OCR (hợp đồng §2.2)
        cache_path = dirs["detected"] / f"{name}_ocr_cache.json"
        cache = dict(image=_rel(png), image_hash=img_hash, pixel_hash=_pixel_hash(str(png)),
                     framed=False, frame_pad=0, coords_space="fullpage",
                     n_columns=len(columns), columns=columns, boxes_raw=boxes_raw,
                     layout="lithograph", box_source=box_source, tiers=L["tiers"],
                     tier_split=tier_split,
                     verse_pairs=vpairs)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        vstat = verify_cache_image(str(cache_path), str(png))
        if vstat == "ok":
            g["n_cache_ok"] += 1
        else:
            flags.append(f"verify_cache={vstat}")

        # transcriptions
        (dirs["transcriptions"] / f"{name}.txt").write_text(
            "".join(" ".join(c["syllables"]) + "\n" for c in cols_txt), encoding="utf-8")
        (dirs["transcriptions"] / f"{name}.json").write_text(json.dumps(dict(
            book_page=p, columns=cols_txt, qn_line_confidences=[], qn_page_confidence=None,
            layout="lithograph", first_seq=L["first_seq"], verse_pairs=vpairs, verse_map=verse_map,
            verse_map_content=cinfo),
            ensure_ascii=False, indent=1),
            encoding="utf-8")

        n14 = sum(1 for c in cols_txt if c["num_syllables"] == sum(EXPECT_TIER))
        g["n_pages"] += 1
        g["n_pages_10cols"] += int(len(pairs) == N_COLUMNS)
        g["n_cols"] += len(pairs)
        g["n_cols_syll14"] += n14
        g["cols_per_page"][name] = len(pairs)
        if flags:
            flagged[name] = flags
        results.append(dict(book_page=p, page_name=name, source_file=L["file"], uid=L["uid"],
                            num_columns=len(pairs), total_syllables=sum(c["num_syllables"] for c in cols_txt),
                            ocr_chars=sum(len(c) for c in columns), qn_page_confidence=None,
                            first_seq=L["first_seq"], n_cols_syll14=n14, box_source=box_source,
                            kim_stats=stats or None, flags=flags,
                            verse_map=dict(mode=verse_map, differs_from_formula=vinfo["differs"],
                                           anchor=vinfo["anchor"], content=cinfo)))
        if verbose:
            print(f"  {name}: {len(pairs)} cột, syl14={n14}, chữ={sum(len(c) for c in columns)}, "
                  f"src={box_source}, cờ={len(flags)}", flush=True)

    g["n_pages_verse_map_differs"] = sum(1 for r in results if r["verse_map"]["differs_from_formula"])
    g["pages_verse_map_differs"] = [r["book_page"] for r in results if r["verse_map"]["differs_from_formula"]]
    manifest = dict(book=book, pdf=cfg["pdf"], source="images", layout="lithograph", n_columns=N_COLUMNS,
                    tiers=2, qn_per_column="couplet", contrast=contrast, ocr=ocr, verse_map=verse_map,
                    measure_dir=_rel(measure_dir),
                    pages=results, total_pages=len(results),
                    total_syllables=sum(r["total_syllables"] for r in results),
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
    ap.add_argument("--limit", type=int, default=None, help="N trang đầu (theo số trang đọc)")
    ap.add_argument("--pages", default=None, help="vd 1,2,5-8 (số trang đọc, không phải canvas)")
    ap.add_argument("--ocr", choices=["kim", "none"], default="none")
    ap.add_argument("--force", action="store_true", help="ghi đè ảnh/cache kim đã có")
    ap.add_argument("--out", default=str(REPO / "prepared"))
    ap.add_argument("--measure-dir", default=str(REPO / "measure_out"))
    ap.add_argument("--contrast", choices=["stretch", "otsu", "none"], default="stretch")
    ap.add_argument("--skip-uncovered", action="store_true",
                    help="bỏ qua (và cờ) trang verses.tsv không phủ đủ thay vì dừng")
    ap.add_argument("--verse-map", choices=["formula", "anchor", "content"], default="formula",
                    help="formula: câu = first_seq+2k theo verse_no (mặc định); anchor: 2n dòng seq_no liên tiếp theo neo số in; "
                         "content: từng cột chọn 2 dòng QN khớp chữ kim nhất (từ điển), đơn điệu theo cột (cần --ocr kim)")
    ap.add_argument("--plan-only", action="store_true",
                    help="chỉ so sánh formula/anchor → <out>/<book>/verse_map_plan.json, không ghi ảnh/cache")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    m = ingest(a.book, _parse_pages(a.pages), a.limit, a.ocr, a.force, Path(a.out), Path(a.measure_dir),
               a.contrast, a.skip_uncovered, verse_map=a.verse_map, plan_only=a.plan_only)
    gates = {k: v for k, v in m["gates"].items() if k not in ("pages_flagged", "cols_per_page")}
    print(json.dumps(dict(out=str(Path(a.out) / a.book), gates=gates,
                          pages_flagged=list(m["gates"]["pages_flagged"])), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
