"""Selftest adapter thạch bản (dữ liệu giả, không đọc data/ hay API).

    .venv/bin/python -m pipeline.tools.ingest_lithograph_selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # chạy trực tiếp không cần PYTHONPATH=.
from pipeline.tools import ingest_lithograph_book as ing  # noqa: E402

N_PASS = 0


def _ok(cond: bool, msg: str) -> None:
    global N_PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    N_PASS += 1


def _fake_layout(n_slots: int = 3, title_slot: bool = False) -> dict:
    """Trang giả: tầng trên y 100–700 (6 hàng pitch 100), dưới y 900–1700 (8 hàng); cột pitch 150, slot 1 phải nhất."""
    cols = []
    for s in range(1, n_slots + 1):
        x0 = 1000 - s * 150
        for t, (y0, y1, n) in enumerate(((100, 700, 6), (900, 1700, 8))):
            if title_slot and s == 1 and t == 1:
                continue  # slot 1 chỉ có tầng trên = cột tựa
            cols.append(dict(tier=t, slot=s, x0=x0, x1=x0 + 130, xc=x0 + 65, y0=y0, y1=y1, n_chars=n, expect=n,
                             char_boxes=[[y0 + 100 * i, y0 + 100 * i + 90] for i in range(n)]))
    return dict(tiers=[(100, 700), (900, 1700)], row_pitch=100, col_pitch=150, cols=cols)


def _box(x0: int, y0: int, x1: int, y1: int, text: str) -> dict:
    return dict(points=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], transcription=text)


def test_pairs_and_verses() -> None:
    L = _fake_layout(3)
    pairs = ing.build_pairs(L["cols"])
    _ok([p["slot"] for p in pairs] == [1, 2, 3], "pairs theo slot tăng dần (phải→trái)")
    _ok(ing.verse_pairs_for_page(19, 3) == [(19, 20), (21, 22), (23, 24)], "verse_pairs first_seq=19")
    Lt = _fake_layout(3, title_slot=True)
    _ok([p["slot"] for p in ing.build_pairs(Lt["cols"])] == [2, 3], "slot tựa (1 tầng) bị bỏ khỏi cặp")

    verses = {v: dict(verse_no=str(v), line_text=t, n_syll=str(len(t.split())), page="p", anchor_source="ocr_num",
                      page_flag="", line_flag="")
              for v, t in {19: "Trước đèn xem truyện Tây 1⁄25,", 20: "Gẫm cười hai chữ nhân tình éo le!",
                           21: "a b c d e", 22: "1. a b c d e f g h", 23: "x y z w u v",
                           24: "một hai ba bốn năm sáu bảy tám"}.items()}
    dups = set()
    cov = ing.check_coverage([(19, 20), (21, 22)], verses, dups)
    _ok(cov == dict(missing=[], duplicated=[]), "phủ đủ")
    cov = ing.check_coverage([(25, 26)], verses, {26})
    _ok(cov["missing"] == [25, 26] and cov["duplicated"] == [26], "thiếu/trùng được liệt kê")

    _ok(ing.lithograph_syllables("Trước đèn xem truyện Tây 1⁄25,") == ["Trước", "đèn", "xem", "truyện", "Tây"],
        "bỏ token không chữ cái, bỏ dấu câu")
    _ok(ing.lithograph_syllables("Vân-Tiên nghe 12. rằng") == ["Vân", "Tiên", "nghe", "rằng"], "tách gạch nối")

    cols, flags = ing.make_column_texts([(19, 20), (21, 22), (23, 24)], verses)
    _ok(len(cols) == 3 and cols[0]["num_syllables"] == 13 and cols[0]["len_odd"] == 5, "cột 1 = 5 + 8 âm")
    _ok(cols[2]["num_syllables"] == 14 and cols[2]["syllables"][6] == "một", "cột 3 = 6 ⧺ 8, câu chẵn nối sau")
    _ok(any(f.startswith("col1:verse19:n_syll=5!=6") for f in flags), "cờ n_syll≠6")
    _ok(any(f.startswith("col2:verse21:n_syll=5!=6") for f in flags) and not any("verse22" in f for f in flags),
        "'1.' bị bỏ, câu 22 đủ 8")
    _ok(cols[1]["verse_odd"]["verse_no"] == 21 and cols[1]["verse_even"]["verse_no"] == 22, "verse_odd/even")
    _ok(cols[0]["column"] == 1 and set(cols[0]) >= {"raw_text", "cleaned_text", "syllables", "num_syllables"},
        "khoá per-column như step1")


def test_assign_boxes() -> None:
    L = _fake_layout(3)
    pairs = ing.build_pairs(L["cols"])
    boxes = [
        _box(850, 100, 980, 700, "一二三四五六"),          # slot 1 tầng trên, 6 chữ
        _box(852, 900, 978, 1700, "七八九十甲乙丙丁"),     # slot 1 tầng dưới, 8 chữ
        _box(700, 100, 830, 700, "ＡＢＣ"),                # slot 2 trên: chỉ 3 chữ (không ép)
        _box(700, 900, 830, 1300, "ＤＥＦＧ"),             # slot 2 dưới: 2 mảnh, mảnh sau đến trước trong list
        _box(700, 1300, 830, 1700, "ＨＩＪＫ"),
        _box(560, 20, 600, 60, "一五"),                    # số câu in trên lề (đáy < tier0_y0) → bỏ
        _box(560, 60, 600, 110, "品"),                     # số câu in sát đỉnh tầng, trong biên 0,25 pitch → bỏ
        _box(710, 860, 750, 915, "口"),                    # số câu in trên tầng dưới → bỏ
        _box(540, 100, 690, 1700, "壹貳參肆伍陸柒捌玖拾拾壹貳參"),  # slot 3: 1 hộp xuyên 2 tầng, 14 chữ
        _box(100, 100, 230, 700, "零零零"),                # x cách cột gần nhất > 0,5 pitch → bỏ
    ]
    cols, st = ing.assign_boxes_to_columns(boxes, L, pairs)
    _ok([len(c) for c in cols] == [14, 11, 14], f"số chữ từng cột {[len(c) for c in cols]}")
    _ok(st["chars_per_tier"] == [(6, 8), (3, 8), (6, 8)], f"tách tầng {st['chars_per_tier']}")
    _ok("".join(c["char"] for c in cols[0]) == "一二三四五六七八九十甲乙丙丁", "trên rồi dưới, trên→dưới")
    _ok("".join(c["char"] for c in cols[1][3:]) == "ＤＥＦＧＨＩＪＫ", "2 mảnh tầng dưới sắp theo y")
    _ok(st["n_unassigned_tier"] == 0 and st["n_number_boxes"] == 3 and st["n_unassigned_col"] == 3,
        "số câu in (3 hộp ≤2 chữ, đáy trên đỉnh tầng) + hộp lạc cột bị loại")
    ys = [c["y_center"] for c in cols[0]]
    _ok(ys == sorted(ys) and cols[0][5]["bbox"][3] <= 700 and cols[0][6]["bbox"][1] >= 900, "bbox theo tầng")
    _ok(all(len(c["bbox"]) == 4 and c["bbox"][0] < c["bbox"][2] and c["bbox"][1] < c["bbox"][3]
            for col in cols for c in col), "bbox hợp lệ")
    # hộp xuyên 2 tầng: 6 chữ đầu rơi vào tầng trên theo y
    _ok("".join(c["char"] for c in cols[2][:6]) == "壹貳參肆伍陸", "hộp xuyên tầng chia theo y_center")

    Lt = _fake_layout(3, title_slot=True)
    cols_t, st_t = ing.assign_boxes_to_columns(boxes[:2], Lt, ing.build_pairs(Lt["cols"]))
    _ok(len(cols_t) == 2 and st_t["n_chars_nonpair"] == 6 and st_t["n_unassigned_col"] == 8,
        "chữ ở cột tựa (tầng trên) không vào cặp; tầng dưới không có cột gần → lạc cột")

    proj = ing.projection_columns(pairs)
    _ok([len(c) for c in proj] == [14, 14, 14] and all(c["char"] is None for c in proj[0]), "--ocr none: char=None")
    _ok(ing.tier_of_y(50, L["tiers"], 100) is None and ing.tier_of_y(80, L["tiers"], 100) == 0
        and ing.tier_of_y(790, L["tiers"], 100) == 0 and ing.tier_of_y(810, L["tiers"], 100) == 1, "tier_of_y biên")


def test_stretch() -> None:
    import numpy as np
    g = np.full((50, 50), 128, dtype=np.uint8)
    g[10:20, 10:20] = 30
    s = ing.stretch_gray(g)
    _ok(int(np.median(s)) == 255 and s.shape == g.shape and s.dtype == np.uint8, "stretch: nền → 255, giữ kích thước")
    o = ing.otsu_gray(g)
    _ok(int(np.median(o)) == 255 and o[15, 15] == 30, "otsu: nền → 255, mực giữ mức xám")
    _ok(ing._parse_pages("1,3-5,9") == [1, 3, 4, 5, 9] and ing._parse_pages(None) is None, "_parse_pages")


def _fake_rows(n: int = 40, jump_at: int = 12, anchors: dict[int, int] | None = None) -> list[dict]:
    """40 dòng seq 1..n, nhịp 6/8 theo seq lẻ/chẵn; số in nhảy +1 từ seq `jump_at` (verse_no = seq+1);
    anchors {seq: verse_no in} → anchor_source=ocr_num."""
    anchors = anchors or {}
    rows = []
    for sq in range(1, n + 1):
        v = sq + (1 if sq >= jump_at else 0)
        rows.append(dict(seq_no=str(sq), verse_no=str(anchors.get(sq, v)), n_syll=str(6 if sq % 2 else 8),
                         line_text=f"dong {sq}", anchor_source="ocr_num" if sq in anchors else "chain_interp",
                         page="p", page_flag="", line_flag=""))
    return rows


def test_verse_map_anchor() -> None:
    rows = _fake_rows(anchors={5: 5, 14: 15, 25: 26})   # neo 5 offset 0; neo 15/26 offset +1 (sau chỗ nhảy)
    _ok(ing.load_verse_rows.__doc__ and len(rows) == 40, "fake rows")
    # trang first_seq 11, 3 cặp = 6 câu: neo 5 (offset 0) giữ nhịp 6/8 → thắng neo 15 (offset +1, nhịp đảo)
    sel, info = ing.anchor_rows_for_page(11, 3, rows)
    _ok(sel is not None and info["anchor_verse"] == 5 and info["offset"] == 0 and info["start_seq"] == 11,
        f"anchor: chọn neo giữ nhịp ({info})")
    _ok(info["parity"] == 6 and [r["seq_no"] for r in sel] == [str(x) for x in range(11, 17)], "anchor: 6 dòng seq liên tiếp")
    _ok(info["differs"] == [1, 2, 3, 4, 5] and info["offsets_in_range"] == [1] and not info["conflict"],
        f"anchor: differs = dòng sau chỗ nhảy ({info['differs']})")
    # trang không có neo trong ±window → None
    sel2, info2 = ing.anchor_rows_for_page(11, 3, _fake_rows(anchors={35: 35}), window=2)
    _ok(sel2 is None and info2.get("reason") == "no_anchor", "anchor: không neo → None")
    # trang vượt cuối tsv → neo bị loại (thiếu dòng)
    sel3, info3 = ing.anchor_rows_for_page(35, 5, rows)
    _ok(sel3 is None, "anchor: thiếu dòng cuối → None")
    # công thức (verse_no) trên cùng dữ liệu: câu 12 không có (nhảy 11→13) → cổng cứng báo thiếu
    verses = {int(r["verse_no"]): r for r in rows}
    cov = ing.check_coverage(ing.verse_pairs_for_page(11, 3), verses, set())
    _ok(cov["missing"] == [12], f"formula: thiếu câu 12 ({cov})")
    # ghép cột từ dòng neo: verse_no gán = first_seq+i, verse_no_tsv giữ số in
    cols, flags = ing.make_column_texts(ing.verse_pairs_for_page(11, 3), {11 + i: r for i, r in enumerate(sel)})
    _ok(cols[0]["verse_odd"]["verse_no"] == 11 and cols[0]["verse_even"]["verse_no_tsv"] == 13
        and cols[0]["verse_even"]["seq_no"] == 12, "anchor: verse_no gán vs verse_no_tsv")


# ---------------------------------------------------------------------------
# --verse-map content (B1', 22/09): dữ liệu giả — dòng QN seq i có 6/8 âm "t<i>_<j>", chữ Nôm "c<i>_<j>" đọc đúng 1 âm
# ---------------------------------------------------------------------------
def _tok(sq: int, j: int) -> str:
    """Âm giả toàn chữ cái (clean_line_text bỏ chữ số): dòng sq, vị trí j → vd 'qbkc'."""
    return "q" + chr(97 + sq // 26) + chr(97 + sq % 26) + chr(97 + j)


def _content_rows(n: int = 40) -> tuple[list[dict], dict[str, set[str]]]:
    rows, n2q = [], {}
    for sq in range(1, n + 1):
        k = 6 if sq % 2 else 8
        toks = [_tok(sq, j) for j in range(k)]
        rows.append(dict(seq_no=str(sq), verse_no=str(sq), n_syll=str(k), line_text=" ".join(toks),
                         anchor_source="chain_interp", page="p", page_flag="", line_flag=""))
        for j in range(k):
            n2q[f"c{sq}_{j}"] = {_tok(sq, j)}
    return rows, n2q


def _chars(sq: int, drop: int = 0, extra: int = 0) -> list[str]:
    k = 6 if sq % 2 else 8
    out = [f"c{sq}_{j}" for j in range(k - drop)]
    return out + ["rac"] * extra


def _col(a: int, **kw) -> tuple[list[str], list[str]]:
    """Cột Nôm khớp cặp dòng (a, a+1)."""
    return _chars(a, **kw.get("top", {})), _chars(a + 1, **kw.get("bot", {}))


def test_verse_map_content() -> None:
    rows, n2q = _content_rows()
    ls = [ing._line_syllable_set(r) for r in rows]
    # 1. offset 0: 3 cột đúng công thức first_seq=11 → dòng (11,12),(13,14),(15,16), điểm 14/cột
    sel, info = ing.content_rows_for_page(11, [_col(11), _col(13), _col(15)], rows, n2q, line_sets=ls)
    _ok(info["offset"] == [0, 0, 0] and info["col_score"] == [14, 14, 14] and info["total_score"] == 42
        and [int(r[0]["seq_no"]) for r in sel] == [11, 13, 15], f"content: offset 0 ({info})")
    _ok(info["skipped"] == [] and info["offsets_distinct"] == [0] and info["n_chars"] == 42, "content: không cột bỏ")
    # 2. offset −4 (số in Nôm = seq QN + 4): chữ kim là dòng 7,9,11 nhưng first_seq 11 → offset −4 mọi cột
    sel, info = ing.content_rows_for_page(11, [_col(7), _col(9), _col(11)], rows, n2q, line_sets=ls)
    _ok(info["offset"] == [-4, -4, -4] and [int(r[0]["seq_no"]) for r in sel] == [7, 9, 11], f"content: offset −4 ({info['offset']})")
    # 3. offset −5 (mất thêm 1 dòng QN → parity đảo: cột bắt đầu ở seq chẵn vẫn ghép được)
    sel, info = ing.content_rows_for_page(11, [_col(6), _col(8), _col(10)], rows, n2q, line_sets=ls)
    _ok(info["offset"] == [-5, -5, -5] and [int(r[0]["seq_no"]) for r in sel] == [6, 8, 10], f"content: offset −5 ({info['offset']})")
    # 4. kim THIẾU 1 chữ tầng trên (5 chữ) → vẫn ghép đúng, điểm 13
    sel, info = ing.content_rows_for_page(11, [_col(11, top=dict(drop=1)), _col(13)], rows, n2q, line_sets=ls)
    _ok(info["offset"] == [0, 0] and info["col_score"] == [13, 14], f"content: kim thiếu 1 chữ ({info['col_score']})")
    # 5. kim THỪA 1 chữ rác tầng dưới (9 chữ) → vẫn ghép đúng, điểm 14 (chữ rác không tra được âm)
    sel, info = ing.content_rows_for_page(11, [_col(11), _col(13, bot=dict(extra=1))], rows, n2q, line_sets=ls)
    _ok(info["offset"] == [0, 0] and info["col_score"] == [14, 14] and info["n_chars"] == 29, "content: kim thừa 1 chữ rác")
    # 6. cột toàn chữ rác (cột chú nhỏ) → None + skipped, cột hai bên vẫn đúng và đơn điệu
    sel, info = ing.content_rows_for_page(11, [_col(11), (["rac"] * 6, ["rac"] * 8), _col(15)], rows, n2q, line_sets=ls)
    _ok(sel[1] is None and info["skipped"] == [2] and info["offset"] == [0, None, 0]
        and info["start_seq"] == [11, None, 15], f"content: cột rác → placeholder ({info})")
    # 7. cột có dòng QN THIẾU trong tsv (dòng 13–14 không tồn tại: chữ kim của dòng 13 không tra được) → cột bỏ,
    #    cột sau dịch offset: kim (11), (13*), (15) với dòng 13/14 xoá khỏi rows → cột 3 ghép seq 13 (= dòng 15 cũ) offset −2
    rows2 = [r for r in rows if int(r["seq_no"]) not in (13, 14)]
    for i, r in enumerate(rows2, start=1):
        r = dict(r); r["seq_no"] = str(i)
        rows2[i - 1] = r
    ls2 = [ing._line_syllable_set(r) for r in rows2]
    sel, info = ing.content_rows_for_page(11, [_col(11), _col(13), _col(15)], rows2, n2q, line_sets=ls2)
    _ok(sel[1] is None and info["offset"] == [0, None, -2] and info["skipped"] == [2],
        f"content: dòng QN thiếu → cột bỏ, cột sau offset −2 ({info['offset']})")
    # 8. cửa sổ: dòng đúng nằm ngoài [first_seq − W, first_seq + 2n + W] → không ghép (None), không ghép nhầm
    sel, info = ing.content_rows_for_page(31, [_col(11), _col(13)], rows, n2q, window=5, line_sets=ls)
    _ok(sel == [None, None] and info["skipped"] == [1, 2], f"content: ngoài cửa sổ → None ({info['offset']})")
    sel, info = ing.content_rows_for_page(31, [_col(11), _col(13)], rows, n2q, window=25, line_sets=ls)
    _ok(info["offset"] == [-20, -20], "content: trong cửa sổ → ghép")
    # 9. đơn điệu: hai cột giống hệt nhau (cùng chữ) → không được gán cùng dòng; cột thứ hai bỏ hoặc dịch ≥ 2 dòng
    sel, info = ing.content_rows_for_page(11, [_col(11), _col(11), _col(13)], rows, n2q, line_sets=ls)
    st = [x for x in info["start_seq"] if x is not None]
    _ok(st == sorted(st) and all(b - a >= 2 for a, b in zip(st, st[1:])) and 11 in st and 13 in st,
        f"content: đơn điệu a_k+1 ≥ a_k + 2 ({info['start_seq']})")
    # 10. min_col: cột chỉ 3 chữ tra được → dưới ngưỡng 4 → bỏ; hạ min_col=3 → ghép
    weak = (_chars(11)[:3] + ["rac"] * 3, ["rac"] * 8)
    sel, info = ing.content_rows_for_page(11, [weak, _col(13)], rows, n2q, line_sets=ls)
    _ok(sel[0] is None and info["offset"] == [None, 0], f"content: điểm 3 < min_col 4 → bỏ ({info['col_score']})")
    sel, info = ing.content_rows_for_page(11, [weak, _col(13)], rows, n2q, min_col=3, line_sets=ls)
    _ok(sel[0] is not None and info["col_score"] == [3, 14], "content: min_col 3 → ghép")
    # 11. parity/tầng: điểm tính THEO TẦNG (trên ↔ dòng a, dưới ↔ dòng a+1), không trộn 14 chữ; cột đảo tầng
    #     (trên = chữ dòng 12, dưới = chữ dòng 11) chỉ được điểm ở tầng dưới khi a = 10 (dòng 10 ⧺ 11) → 0 + 6 = 6,
    #     KHÔNG được 14 ở a = 11 (dòng 11 ⧺ 12) vì tầng trên phải khớp câu lục
    sel, info = ing.content_rows_for_page(11, [(_chars(12), _chars(11)), _col(13)], rows, n2q, line_sets=ls)
    _ok(info["start_seq"] == [10, 13] and info["col_score"] == [6, 14], f"content: điểm theo tầng, không trộn ({info})")
    # 12. skip_pen: bỏ cột rẻ hơn ghép sai; với skip_pen rất lớn DP vẫn không được ghép cột rác (điểm < min_col)
    sel, info = ing.content_rows_for_page(11, [(["rac"] * 6, ["rac"] * 8), _col(13)], rows, n2q, skip_pen=1000, line_sets=ls)
    _ok(sel[0] is None and info["total_score"] == 14 - 1000, "content: skip_pen chỉ là phạt, không ép ghép cột rác")
    # 13. line_sets tự tính = truyền sẵn
    a = ing.content_rows_for_page(11, [_col(11), _col(13)], rows, n2q)
    b = ing.content_rows_for_page(11, [_col(11), _col(13)], rows, n2q, line_sets=ls)
    _ok(a[1] == b[1], "content: line_sets mặc định = truyền sẵn")
    # 14. placeholder: 6/8 token khongkhop → num_syllables 14, không cờ n_syll, không hợp lệ QN, qn_source 'ocr'
    from core.text.text_utils import is_plausible_qn_syllable
    ph = {11: ing._placeholder_row(11, 6), 12: ing._placeholder_row(12, 8)}
    cols, flags = ing.make_column_texts([(11, 12)], ph)
    _ok(cols[0]["num_syllables"] == 14 and not flags and cols[0]["verse_odd"]["qn_source"] == "ocr"
        and set(cols[0]["syllables"]) == {ing.CONTENT_PLACEHOLDER} and not is_plausible_qn_syllable(ing.CONTENT_PLACEHOLDER),
        f"content: placeholder 6+8 khongkhop ({flags})")


def test_verses_b1_and_dict_boost() -> None:
    """B1': verses_b1.tsv (qn_source/ref_nom) đi vào make_column_texts và apply_dict_boost."""
    rows, n2q = _content_rows()
    r11, r12 = dict(rows[10]), dict(rows[11])
    r11.update(qn_source="nf1871_exact", ref_idx="11", ref_sim="1.0", ref_nom="甲乙丙丁戊己", line_text_ocr="x")
    r12.update(qn_source="nf1871_fuzzy", ref_idx="12", ref_sim="0.95", ref_nom="庚辛壬癸子丑寅卯")
    cols, _ = ing.make_column_texts([(11, 12)], {11: r11, 12: r12})
    vo, ve = cols[0]["verse_odd"], cols[0]["verse_even"]
    _ok(vo["qn_source"] == "nf1871_exact" and vo["ref_idx"] == "11" and vo["line_text_ocr"] == "x"
        and ve["qn_source"] == "nf1871_fuzzy" and ve["ref_sim"] == "0.95", "verses_b1: qn_source/ref_idx/ref_sim vào transcriptions")
    cols0, _ = ing.make_column_texts([(13, 14)], {13: rows[12], 14: rows[13]})
    _ok(cols0[0]["verse_odd"]["qn_source"] == "ocr" and cols0[0]["verse_odd"]["ref_idx"] == "", "verses.tsv cũ: qn_source mặc định ocr")

    # apply_dict_boost: R theo âm t11_j; kim tầng trên = [甲, 乙', 丙', 丁', 戊, X]; ref = 甲乙丙丁戊己
    R = {_tok(11, j): set(c) for j, c in enumerate([["甲"], ["乙", "乙'"], ["丙"], ["丁", "丁'"], ["戊"], ["己"]])}
    R.update({_tok(12, j): {"庚辛壬癸子丑寅卯"[j]} for j in range(8)})
    sim = {"乙'": {"乙"}, "丁": {"丁'"}}          # 乙' gần hình 乙 (chiều thuận); 丁' gần hình 丁 (chiều ngược)
    col = [dict(char=c, bbox=[0, 0, 1, 1]) for c in ["甲", "乙'", "丙'", "丁'", "戊", "X"]] + \
          [dict(char=c, bbox=[0, 0, 1, 1]) for c in "庚辛壬癸子丑寅卯"]
    st = ing.apply_dict_boost([col], [(6, 8)], [(r11, r12)], R, sim)
    _ok(st["boosted"] == 2 and col[1]["char"] == "乙" and col[1]["char_kim"] == "乙'" and col[1]["dict_boost"] == 1
        and col[3]["char"] == "丁" and col[3]["char_kim"] == "丁'", f"dict_boost: kim ∈ R, ≠ ref, gần hình (2 chiều) → thay ({st})")
    _ok(col[2]["char"] == "丙'" and st["kim_not_in_R"] == 2 and "char_kim" not in col[2], "dict_boost: kim ∉ R → giữ (丙', X)")
    _ok(col[0]["char"] == "甲" and st["agree"] == 2 + 8 and st["n_tier_checked"] == 2, "dict_boost: kim == ref (甲, 戊 + 8 tầng dưới) → giữ, đếm agree")
    # ref ∉ R → giữ; đồng âm không gần hình → giữ
    R2 = dict(R); R2[_tok(11, 1)] = {"乙'"}                  # ref 乙 ∉ R(âm 1)
    col2 = [dict(char=c, bbox=[0, 0, 1, 1]) for c in ["甲", "乙'", "丙", "丁'", "戊", "己"]] + \
           [dict(char=c, bbox=[0, 0, 1, 1]) for c in "庚辛壬癸子丑寅卯"]
    st2 = ing.apply_dict_boost([col2], [(6, 8)], [(r11, r12)], R2, {})
    _ok(st2["boosted"] == 0 and st2["ref_not_in_R"] == 1 and st2["di_the_not_similar"] == 1 and col2[1]["char"] == "乙'"
        and col2[3]["char"] == "丁'", f"dict_boost: ref ∉ R / không gần hình → không đổi ({st2})")
    # dòng OCR (qn_source ocr) hoặc ref_nom lệch số chữ hoặc kim ≠ 6/8 → bỏ qua tầng
    r11o = dict(r11); r11o["qn_source"] = "ocr"
    r12s = dict(r12); r12s["ref_nom"] = "庚辛壬"
    col3 = [dict(char=c, bbox=[0, 0, 1, 1]) for c in ["甲", "乙'", "丙", "丁'", "戊", "己"]] + \
           [dict(char=c, bbox=[0, 0, 1, 1]) for c in "庚辛壬癸子丑寅卯"]
    st3 = ing.apply_dict_boost([col3], [(6, 8)], [(r11o, r12s)], R, sim)
    _ok(st3["n_tier_skipped_ocr"] == 1 and st3["n_tier_skipped_len"] == 1 and st3["boosted"] == 0 and col3[1]["char"] == "乙'",
        f"dict_boost: bỏ qua tầng ocr / ref_nom lệch ({st3})")
    st4 = ing.apply_dict_boost([col3[:5] + col3[6:]], [(5, 8)], [(r11, r12)], R, sim)
    _ok(st4["n_tier_skipped_len"] == 1 and st4["n_tier_checked"] == 1, "dict_boost: kim 5 ≠ 6 chữ → bỏ qua tầng")
    st5 = ing.apply_dict_boost([col3], [(6, 8)], [None], R, sim)
    _ok(st5["n_tier_checked"] == 0 and st5["boosted"] == 0, "dict_boost: cột placeholder (None) → bỏ qua")


if __name__ == "__main__":
    test_pairs_and_verses()
    test_assign_boxes()
    test_stretch()
    test_verse_map_anchor()
    test_verse_map_content()
    test_verses_b1_and_dict_boost()
    print(f"ingest_lithograph_selftest: {N_PASS}/{N_PASS} PASS")
