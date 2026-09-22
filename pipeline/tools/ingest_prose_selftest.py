"""Selftest adapter văn xuôi (dữ liệu giả, không đọc data/, measure_out/ hay API).

    .venv/bin/python -m pipeline.tools.ingest_prose_selftest

Kiểm: số Hán-Việt; truyện của cột theo (canvas, ô); gán hộp kim → ô cột theo tâm x (0,5 × bước);
DP đơn điệu (khớp hoàn hảo, chèn/xoá, hai đầu QN tự do, chữ None); chia âm theo cột (âm kẽ chia
trung điểm); cột điểm thấp → None (giữ chỗ); ingest() trên sách giả (page_columns/kim/bộ đo vá) →
cache fullpage verify 'ok', cột 1 = phải nhất, .txt = số cột, giữ chỗ cho cột ngoài truyện; rồi
align_production._detect với layout=prose n_columns=auto → page_ok True và đủ cặp cột.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.tools import ingest_prose_book as ing  # noqa: E402
from pipeline.tools.ingest_lithograph_book import CONTENT_PLACEHOLDER  # noqa: E402

N_PASS = 0


def _ok(cond: bool, msg: str) -> None:
    global N_PASS
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    N_PASS += 1


N2Q = {"A": {"a"}, "B": {"b"}, "C": {"c"}, "D": {"d"}, "E": {"e"}, "F": {"f"}, "T": {"truyện"}, "N": {"nhất"},
       "H": {"thứ"}}


def _box(x0, y0, x1, y1, text):
    return dict(points=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], transcription=text)


def test_sino_and_story() -> None:
    _ok(ing.sino_number(1) == ["nhất"] and ing.sino_number(10) == ["thập"] and ing.sino_number(11) == ["thập", "nhất"]
        and ing.sino_number(20) == ["nhị", "thập"] and ing.sino_number(9) == ["cửu"], "sino_number 1/9/10/11/20")
    table = [dict(so=1, nom_canvas_dau=106, nom_slot_dau=1), dict(so=2, nom_canvas_dau=110, nom_slot_dau=0),
             dict(so=3, nom_canvas_dau=112, nom_slot_dau=4)]
    _ok(ing.story_of_column(106, 0, table) is None, "cột trước truyện 1 (đầu đề chạy) -> None")
    _ok(ing.story_of_column(106, 1, table) == 1 and ing.story_of_column(109, 6, table) == 1, "truyện 1 từ (106,1) tới hết 109")
    _ok(ing.story_of_column(110, 0, table) == 2 and ing.story_of_column(112, 3, table) == 2, "truyện 2 tới (112,3)")
    _ok(ing.story_of_column(112, 4, table) == 3 and ing.story_of_column(200, 0, table) == 3, "ranh giới giữa trang (112,4) -> truyện 3")


def test_assign_boxes() -> None:
    cols = [dict(slot=0, xc=100.0), dict(slot=1, xc=240.0), dict(slot=2, xc=380.0)]
    boxes = [_box(80, 100, 130, 400, "ABC"), _box(220, 100, 270, 300, "DE"), _box(230, 320, 265, 420, "F"),
             _box(700, 100, 750, 300, "ZZ")]
    per, st = ing.assign_boxes(boxes, cols, col_pitch=140)
    _ok(sorted(per) == [0, 1] and [c["char"] for c in per[0]] == ["A", "B", "C"], "hộp -> ô 0 theo tâm x, chữ chia đều")
    _ok([c["char"] for c in per[1]] == ["D", "E", "F"], "2 hộp cùng ô 1 gộp và sắp theo y")
    _ok(st["n_boxes_unassigned"] == 1 and st["n_chars_unassigned"] == 2 and st["n_chars"] == 8,
        "hộp xa mọi ô (> 0,5 × bước) bị bỏ và đếm")
    _ok(per[0][1]["bbox"] == [80, 200, 130, 300] and abs(per[0][1]["y_center"] - 250) < 1e-6, "bbox chữ giữa = 1/3 hộp")
    col = dict(x0=10, x1=60, y0=100, y1=400, n_runs=3)
    pc = ing.projection_chars(col)
    _ok(len(pc) == 3 and pc[0]["char"] is None and pc[2]["bbox"][3] == 400, "projection_chars: n_runs ô, char None")


def test_dp() -> None:
    pairs, sc = ing.align_sequences(list("ABCDE"), list("abcde"), N2Q)
    _ok(pairs == [(i, i, True) for i in range(5)] and sc == 15.0, "DP khớp hoàn hảo 5 × 3 = 15")
    pairs, sc = ing.align_sequences(list("ABZCDE"), "q q a b c x d e".split(), N2Q)
    _ok([(i, j) for i, j, _ in pairs] == [(0, 2), (1, 3), (3, 4), (4, 6), (5, 7)] and sc == 13.0,
        "DP: đầu QN tự do (q q), chữ lạ Z xoá (−1), âm thừa x chèn (−1) -> 15 − 2")
    pairs, sc = ing.align_sequences(list("ABCDE"), "a b c d e x y z".split(), N2Q)
    _ok(len(pairs) == 5 and sc == 15.0, "DP: đuôi QN tự do (x y z không tốn)")
    pairs, sc = ing.align_sequences([None, "A", None], list("ab"), N2Q)
    _ok(all(not ok for _, _, ok in pairs) or any(ok for _, _, ok in pairs), "DP chịu chữ None (không ném)")
    _ok(ing.align_sequences([], list("ab"), N2Q) == ([], 0.0) and ing.align_sequences(list("A"), [], N2Q) == ([], 0.0),
        "DP chuỗi rỗng -> ([], 0)")
    spans = ing.split_syllables_by_column([2, 3, 1], [(0, 2, True), (1, 3, True), (3, 4, True), (4, 6, True), (5, 7, True)], 8)
    _ok([(s["j0"], s["j1"]) for s in spans] == [(2, 4), (4, 7), (7, 8)], "chia âm theo cột, âm kẽ x về cột chứa (4..7)")
    spans = ing.split_syllables_by_column([1, 1], [(0, 0, True), (1, 5, True)], 6)
    _ok([(s["j0"], s["j1"]) for s in spans] == [(0, 3), (3, 6)], "4 âm kẽ giữa 2 cột chia trung điểm (1+2 / 2+1)")
    res, info = ing.align_story_columns([list("AB"), list("ZZZZ"), list("CDE")], "a b c d e".split(), N2Q,
                                        min_match=2, min_ratio=0.5)
    _ok(res[0]["syllables"] == ["a", "b"] and res[2]["syllables"] == ["c", "d", "e"], "cột tốt nhận đúng đoạn âm")
    _ok(res[1]["syllables"] is None and res[1]["matched"] is False and info["n_cols_matched"] == 2,
        "cột 4 chữ lạ (0 khớp) -> None (giữ chỗ), không kéo âm của cột kề")
    _ok(ing.placeholder_line(0) == [CONTENT_PLACEHOLDER] and len(ing.placeholder_line(5)) == 5, "placeholder_line ≥ 1 token")
    from core.text.text_utils import is_plausible_qn_syllable
    _ok(not is_plausible_qn_syllable(CONTENT_PLACEHOLDER), "token giữ chỗ không phải âm hợp lệ -> chỉ REVIEW")


def _fake_page_image(path: Path, cols_x: list[tuple[int, int]], n_chars: int, W=1400, H=1800):
    import numpy as np
    from PIL import Image
    img = np.full((H, W), 255, np.uint8)
    for x0, x1 in cols_x:
        for j in range(n_chars):
            y = 100 + j * 100
            img[y:y + 80, x0:x1] = 0
    Image.fromarray(img).save(path, "JPEG")


def test_ingest_and_engine() -> None:
    from pipeline.align_engine import book_layout as BL
    from pipeline.align_engine import align_production as AP
    cols_x = [(200, 260), (400, 460), (600, 660), (800, 860)]        # 4 cột, trái->phải, pitch 200
    n = 6
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src"; src.mkdir()
        _fake_page_image(src / "canvas_0001.jpg", cols_x, n)
        ing.BOOKS["FakeProse"] = dict(src=src, canvases=[1], canvas_of_page=lambda k: k, page_of_canvas=lambda c: c,
                                      measure="x", read_ltr=True, title_prefix=("truyện", "thứ"))
        lay = dict(canvas=1, blank=False, frame=[50, 1750, 150, 950], col_pitch=200, nslot=4, char_pitch=100,
                   empty_internal_slots=[],
                   cols=[dict(slot=k, x0=x0, x1=x1, xc=(x0 + x1) / 2, y0=100, y1=100 + n * 100, slot_x0=x0 - 70, slot_x1=x1 + 70,
                              n_est=float(n), n_runs=n, short=False) for k, (x0, x1) in enumerate(cols_x)])
        texts = ["ZZ", "THNABC", "DEFABC", "DEFABC"]     # ô 0 = đầu đề chạy (ngoài truyện), ô 1 mở đầu truyện
        boxes = [_box(x0, 100, x1, 100 + len(t) * 100, t) for (x0, x1), t in zip(cols_x, texts)]
        story_syl = "truyện thứ nhất a b c d e f a b c d e f a b c".split()
        saved = (ing.page_columns, ing.kim_boxes, ing.load_story_table, ing.load_story_syllables, ing._nom_to_qn_readings)
        ing.page_columns = lambda canvas: lay
        ing.kim_boxes = lambda png, raw, h, force, kim=None: boxes
        ing.load_story_table = lambda md, b: [dict(so=1, tieu_de="t", qn_canvas=1, nom_canvas_dau=1, nom_slot_dau=1,
                                                   nom_trang="1", nom_n_cols=3, ranh_gioi="sau ô", can_nguoi_ra=True)]
        ing.load_story_syllables = lambda md, b, t: {1: dict(so=1, tieu_de="t", title_syllables=story_syl[:3],
                                                              body_syllables=story_syl[3:], syllables=story_syl)}
        ing._nom_to_qn_readings = lambda: N2Q
        try:
            m = ing.ingest("FakeProse", None, None, "kim", True, td / "prep", td / "measure", contrast="none",
                           kim_src="prepared", verbose=False)
        finally:
            (ing.page_columns, ing.kim_boxes, ing.load_story_table, ing.load_story_syllables,
             ing._nom_to_qn_readings) = saved
            ing.BOOKS.pop("FakeProse", None)
        g = m["gates"]
        _ok(g["n_pages"] == 1 and g["n_cols_out"] == 4 and g["n_cache_ok"] == 1 and g["ocr_calls"] == 1,
            f"ingest giả: 1 trang, 4 cột, cache ok, 1 lượt kim ({g['n_cols_out']}, {g['n_cache_ok']})")
        _ok(g["n_cols_matched"] == 3 and g["n_cols_placeholder"] == 1 and g["n_cols_no_story"] == 1,
            f"3 cột truyện ghép được, 1 cột đầu đề giữ chỗ ({g['n_cols_matched']}/{g['n_cols_placeholder']})")
        _ok(g["boundary_mid_page"] == [dict(page="page_0001", canvas=1, slot=1, story=1)], "ranh giới truyện giữa trang được ghi")
        _ok(g["reading_order"]["n_stories_ltr_better"] == 1 and g["stories"][1]["score"] > g["stories"][1]["score_reversed_in_page"],
            "kiểm thứ tự đọc: LTR hơn RTL trên truyện giả")
        out = td / "prep" / "FakeProse"
        cache = json.loads((out / "detected" / "page_0001_ocr_cache.json").read_text(encoding="utf-8"))
        _ok(cache["coords_space"] == "fullpage" and cache["n_columns"] == 4 and cache["slots"] == [3, 2, 1, 0]
            and cache["image_hash"] and cache["pixel_hash"], "cache: fullpage, cột 1 = ô phải nhất (slot 3), có hash")
        _ok(["".join(c["char"] for c in col) for col in cache["columns"]] == ["DEFABC", "DEFABC", "THNABC", "ZZ"],
            "cache columns phải->trái đúng chữ kim")
        lines = (out / "transcriptions" / "page_0001.txt").read_text(encoding="utf-8").splitlines()
        _ok(len(lines) == 4 and lines[3] == " ".join([CONTENT_PLACEHOLDER] * 2), ".txt 4 dòng, dòng 4 (ô 0) giữ chỗ 2 token")
        _ok(lines[2].split() == ["truyện", "thứ", "nhất", "a", "b", "c"] and lines[0].split() == "d e f a b c".split(),
            f".txt cột truyện = âm đã ghép ({lines[2]!r} / {lines[0]!r})")
        tj = json.loads((out / "transcriptions" / "page_0001.json").read_text(encoding="utf-8"))
        _ok([c["reading_order"] for c in tj["columns"]] == [4, 3, 2, 1] and tj["columns"][3]["story"] is None
            and tj["columns"][0]["matched"] is True, "json: reading_order, story None cho đầu đề, matched")
        # engine: _detect với prose auto trên thư mục vừa ghi
        lay_pr = BL.book_layout({"name": "FakeProse", "layout": "prose"})
        gate: dict = {}
        det = AP._detect("page_0001", out, set(), layout=lay_pr, gate_out=gate)
        _ok(det is not None and det[4] is True and len(det[0]) == 4 and gate["n_nom_cols"] == 4 and gate["n_qn_cols"] == 4,
            f"engine _detect prose auto: page_ok True, 4/4 cột ({gate})")
        _ok(det is not None and det[2] == [(i, i + 1) for i in range(4)]
            and [len(det[1][k]) for k in (1, 2, 3, 4)] == [6, 6, 6, 2], "iter_pairs 4 cặp; số âm mỗi cột 6/6/6/2 (không ép 14)")
        # cùng dữ liệu, layout STT mặc định: vẫn đi đường cũ (n_expected 9, không ghi gate_out)
        gs: dict = {}
        det_stt = AP._detect("page_0001", out, set(), gate_out=gs)
        _ok(gs == {} and (det_stt is None or len(det_stt) == 5), "layout STT mặc định: không ghi gate_out, 5 phần tử như cũ")


def main() -> int:
    for t in (test_sino_and_story, test_assign_boxes, test_dp, test_ingest_and_engine):
        t()
    print(f"OK ingest_prose_selftest: {N_PASS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
