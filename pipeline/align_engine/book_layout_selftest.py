"""Selftest tham số hoá số cột (pipeline.align_engine.book_layout + đường _detect).
Run:  .venv/bin/python -m pipeline.align_engine.book_layout_selftest
Kiểm:
  1. book_layout: vắng khoá == STT (9 cột, không cổng); n_columns=10 + layout=lithograph
     đọc đúng; giá trị sai -> ValueError.
  2. lithograph_gate / expected_qn_counts: cổng 10 cột × 14 âm, num_syllables trong JSON
     ghi đè 14, qn_syllables_per_column=0 bỏ kiểm âm.
  3. _get_qn_lines: cache 10 dòng đánh số -> mặc định vẫn ép 9 (STT), n_columns=10 -> 10.
  4. _detect trên trang giả (ảnh + ocr_cache + .txt): mặc định gọi detect_nom_columns_v3
     với 9, trả 5 phần tử như cũ và không ghi gate_out; layout lithograph n_columns=10
     -> 10 cột, page_ok, gate_out đủ; trang 10 cột với mặc định 9 -> gộp về 9 (hành vi
     cũ giữ nguyên). pipeline/lab/extract_columns.py unpack 5 phần tử vẫn chạy.
  5. Chữ ký align_page/build_dataset có tham số layout mặc định None.
Exit 0 = all pass.
"""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.align_engine import book_layout as BL                      # noqa: E402
from pipeline.align_engine import align_production as AP                 # noqa: E402
from pipeline.step2_align import _get_qn_lines                           # noqa: E402

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}  {detail}")


def test_book_layout():
    print("[1] book_layout")
    d = BL.book_layout(None)
    check("None -> DEFAULT_LAYOUT (cùng đối tượng)", d is BL.DEFAULT_LAYOUT)
    check("mặc định n_columns=9, layout=stt, không lithograph, qn_per_column=0",
          d.n_columns == 9 and d.layout == "stt" and not d.is_lithograph and d.qn_per_column == 0)
    stt = BL.book_layout({"name": "SachThanhTruyen2", "pdf": "x.pdf", "reocr": True})
    check("mục STT (name/pdf/reocr) -> DEFAULT_LAYOUT", stt is BL.DEFAULT_LAYOUT)
    check("khai báo tường minh layout=stt, n_columns=9 -> DEFAULT_LAYOUT",
          BL.book_layout({"name": "a", "layout": "stt", "n_columns": 9}) is BL.DEFAULT_LAYOUT)
    lv = BL.book_layout({"name": "LucVanTien1883", "layout": "lithograph", "n_columns": 10})
    check("lithograph n_columns=10 -> 10, qn_per_column mặc định 14",
          lv.layout == "lithograph" and lv.n_columns == 10 and lv.qn_per_column == 14 and lv.is_lithograph)
    lv0 = BL.book_layout({"name": "b", "layout": "lithograph", "n_columns": 10,
                          "qn_syllables_per_column": 0})
    check("qn_syllables_per_column=0 giữ 0 (tắt kiểm âm)", lv0.qn_per_column == 0)
    check("n_columns=10 không lithograph -> layout stt, 10 cột, không kiểm âm",
          BL.book_layout({"name": "c", "n_columns": 10}) == BL.BookLayout("stt", 10, 0))
    for bad in ({"name": "x", "layout": "couplet10"}, {"name": "x", "n_columns": 0},
                {"name": "x", "n_columns": "10"}, {"name": "x", "n_columns": True},
                {"name": "x", "layout": "lithograph", "qn_syllables_per_column": -1}):
        try:
            BL.book_layout(bad)
            check(f"giá trị sai {bad} -> ValueError", False)
        except ValueError:
            check(f"giá trị sai {bad} -> ValueError", True)


def test_det_params():
    print("[1b] det_xmargin / det_thr theo sách (2026-09-22)")
    d = BL.book_layout(None)
    check("STT: det_xmargin None, det_thr None (= step2 toàn cục)", d.det_xmargin is None and d.det_thr is None)
    check("STT tường minh (layout stt, 9 cột) vẫn là DEFAULT_LAYOUT",
          BL.book_layout({"name": "s", "layout": "stt", "n_columns": 9}) is BL.DEFAULT_LAYOUT)
    lv = BL.book_layout({"name": "L", "layout": "lithograph", "n_columns": 10})
    check("lithograph vắng khoá -> det_xmargin = LITHO_DET_XMARGIN 0,05, det_thr None",
          lv.det_xmargin == BL.LITHO_DET_XMARGIN == 0.05 and lv.det_thr is None)
    lv2 = BL.book_layout({"name": "L", "layout": "lithograph", "n_columns": 10,
                          "det_xmargin": 0.1, "det_thr": 0.15})
    check("khai tường minh det_xmargin 0,1 / det_thr 0,15 -> đúng giá trị (float)",
          lv2.det_xmargin == 0.1 and lv2.det_thr == 0.15 and isinstance(lv2.det_thr, float))
    st = BL.book_layout({"name": "s", "det_xmargin": 0.25})
    check("STT khai det_xmargin -> BookLayout riêng (không phải DEFAULT), layout stt 9 cột",
          st is not BL.DEFAULT_LAYOUT and st.layout == "stt" and st.n_columns == 9 and st.det_xmargin == 0.25)
    for bad in ({"name": "x", "det_xmargin": 2}, {"name": "x", "det_xmargin": -0.1},
                {"name": "x", "det_xmargin": "0.1"}, {"name": "x", "det_thr": True},
                {"name": "x", "det_thr": 1.5}):
        try:
            BL.book_layout(bad)
            check(f"giá trị sai {bad} -> ValueError", False)
        except ValueError:
            check(f"giá trị sai {bad} -> ValueError", True)
    check("hằng engine mặc định không đổi: DETECTOR_THR 0,2 / DETECTOR_XMARGIN 0,25",
          (AP.DETECTOR_THR, AP.DETECTOR_XMARGIN) == (0.2, 0.25))
    src = (REPO / "pipeline" / "align_engine" / "build_dataset.py").read_text(encoding="utf-8")
    check("build_dataset ghi đè DETECTOR_* theo lay.det_thr/det_xmargin rồi khôi phục toàn cục",
          "lay.det_thr" in src and "lay.det_xmargin" in src and "det_xmargin_global" in src
          and "detector_params_by_book" in src)


def test_gate():
    print("[2] lithograph_gate / expected_qn_counts")
    lay = BL.BookLayout("lithograph", 10, 14)
    cols = [{"chars": []} for _ in range(10)]
    qn = {i + 1: ["a"] * 14 for i in range(10)}
    ok, g = BL.lithograph_gate(cols, qn, lay)
    check("10 cột × 14 âm -> PASS", ok and g["n_nom_cols"] == 10 and g["n_qn_cols"] == 10 and not g["bad_syl_cols"])
    ok, g = BL.lithograph_gate(cols[:9], qn, lay)
    check("9 cột Nôm -> FAIL", not ok and g["n_nom_cols"] == 9)
    qn2 = dict(qn); qn2[3] = ["a"] * 13
    ok, g = BL.lithograph_gate(cols, qn2, lay)
    check("cột 3 có 13 âm -> FAIL, bad_syl_cols ghi cột 3 muốn 14",
          not ok and g["bad_syl_cols"] == [{"column": 3, "n_syl": 13, "want": 14}])
    ok, g = BL.lithograph_gate(cols, qn2, lay, expected_counts={3: 13})
    check("num_syllables JSON = 13 cho cột 3 -> PASS (câu lẻ 5/7 âm ghi rõ vẫn qua)", ok)
    ok, g = BL.lithograph_gate(cols, qn2, BL.BookLayout("lithograph", 10, 0))
    check("qn_per_column=0 -> không kiểm âm -> PASS", ok)
    ok, g = BL.lithograph_gate(cols, {k: v for k, v in qn.items() if k <= 9}, lay)
    check("9 dòng QN -> FAIL", not ok and g["n_qn_cols"] == 9)
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "transcriptions").mkdir()
        check("thiếu transcriptions/<page>.json -> {}", BL.expected_qn_counts(td, "page_0001") == {})
        js = {"book_page": 1, "columns": [{"column": 1, "num_syllables": 14},
                                          {"column": 2, "num_syllables": 12},
                                          {"column": 3}, "rác"]}
        (Path(td) / "transcriptions" / "page_0001.json").write_text(json.dumps(js), encoding="utf-8")
        check("đọc num_syllables theo column, bỏ cột thiếu khoá/rác",
              BL.expected_qn_counts(td, "page_0001") == {1: 14, 2: 12})


def test_get_qn_lines():
    print("[3] _get_qn_lines n_columns")
    text = "\n".join(f"{i}. " + " ".join(["xa"] * 14) for i in range(1, 11))
    with tempfile.TemporaryDirectory() as td:
        tr = Path(td) / "transcriptions"; tr.mkdir()
        (tr / "page_0001_qn_ocr_cache.json").write_text(json.dumps({"text": text}), encoding="utf-8")
        d9, src9 = _get_qn_lines(Path(td), "page_0001", None)
        check("mặc định (STT): 10 dòng đánh số -> ép về 9 khoá như cũ", len(d9) == 9 and max(d9) == 9, (len(d9), src9))
        d10, src10 = _get_qn_lines(Path(td), "page_0001", None, n_columns=10)
        check("n_columns=10: 10 khoá 1..10", sorted(d10) == list(range(1, 11)), (len(d10), src10))
        check("n_columns=10: dòng 10 đủ 14 âm", len(d10[10]) == 14, len(d10.get(10, [])))
        # đường .txt (v1) không phụ thuộc n_columns
        (tr / "page_0001_qn_ocr_cache.json").unlink()
        (tr / "page_0001.txt").write_text("\n".join(" ".join(["xa"] * 14) for _ in range(10)), encoding="utf-8")
        dv, srcv = _get_qn_lines(Path(td), "page_0001", None)
        check("đường .txt: 10 dòng -> 10 khoá bất kể n_columns", len(dv) == 10 and srcv == "v1")


def _fake_page(root: Path, n_cols: int, n_chars: int = 14, n_lines: int | None = None,
               syl_per_line: int = 14):
    """Trang giả: ảnh trắng có ô mực đen theo cột, ocr_cache fullpage, .txt + .json."""
    import numpy as np
    import cv2
    W, H = 1400, 1800
    img = np.full((H, W, 3), 255, np.uint8)
    pitch_x = 120
    x0 = W - 100
    columns = []
    for k in range(n_cols):
        x1 = x0 - k * pitch_x - 60
        col = []
        for j in range(n_chars):
            y1 = 100 + j * 110
            cv2.rectangle(img, (x1, y1), (x1 + 60, y1 + 80), (0, 0, 0), -1)
            col.append({"char": "丁", "y_center": y1 + 40.0, "bbox": [x1, y1, x1 + 60, y1 + 80]})
        columns.append(col)
    (root / "pages").mkdir(parents=True); (root / "detected").mkdir(); (root / "transcriptions").mkdir()
    cv2.imwrite(str(root / "pages" / "page_0001.png"), img)
    cache = {"image": "x.png", "framed": False, "frame_pad": 0, "coords_space": "fullpage",
             "n_columns": n_cols, "columns": columns, "boxes_raw": []}
    (root / "detected" / "page_0001_ocr_cache.json").write_text(json.dumps(cache), encoding="utf-8")
    n_lines = n_cols if n_lines is None else n_lines
    (root / "transcriptions" / "page_0001.txt").write_text(
        "\n".join(" ".join(["đinh"] * syl_per_line) for _ in range(n_lines)), encoding="utf-8")
    (root / "transcriptions" / "page_0001.json").write_text(json.dumps(
        {"book_page": 1, "columns": [{"column": i + 1, "num_syllables": syl_per_line} for i in range(n_lines)]}),
        encoding="utf-8")


def test_detect():
    print("[4] _detect với trang giả")
    calls = []
    orig = AP.detect_nom_columns_v3

    def spy(binary, kim_columns, n_expected=9):
        calls.append(n_expected)
        return orig(binary, kim_columns, n_expected)
    AP.detect_nom_columns_v3 = spy
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "B9"; _fake_page(root, 9, n_chars=14, syl_per_line=14)
            g0: dict = {}
            det = AP._detect("page_0001", root, set(), gate_out=g0)
            check("mặc định: detect_nom_columns_v3 nhận n_expected=9", calls[-1] == 9, calls)
            check("mặc định: 5 phần tử như cũ, 9 cột, page_ok, gate_out không bị ghi",
                  det is not None and len(det) == 5 and len(det[0]) == 9 and det[4] is True and g0 == {},
                  None if det is None else (len(det), len(det[0]), det[4], g0))
            det_nog = AP._detect("page_0001", root, set())
            check("mặc định không truyền gate_out: vẫn chạy, 5 phần tử",
                  det_nog is not None and len(det_nog) == 5)
            root = Path(td) / "B10"; _fake_page(root, 10, n_chars=14, syl_per_line=14)
            det_old = AP._detect("page_0001", root, set())
            check("trang 10 cột, mặc định 9: vẫn gọi với 9 và gộp về 9 cột (hành vi cũ)",
                  calls[-1] == 9 and det_old is not None and len(det_old[0]) == 9, None if det_old is None else len(det_old[0]))
            check("trang 10 cột, mặc định 9: qn_parse_ok sai vì 10 dòng != 9 -> page_ok False",
                  det_old is not None and det_old[4] is False)
            lay = BL.BookLayout("lithograph", 10, 14)
            g10: dict = {}
            det10 = AP._detect("page_0001", root, set(), layout=lay, gate_out=g10)
            check("layout lithograph n_columns=10: gọi với 10, ra 10 cột, page_ok True, 5 phần tử",
                  calls[-1] == 10 and det10 is not None and len(det10) == 5 and len(det10[0]) == 10 and det10[4] is True,
                  None if det10 is None else (len(det10[0]), det10[4], g10))
            check("gate_out ghi 10/10 cột, không cột lệch âm, col_method hybrid",
                  g10.get("n_nom_cols") == 10 and g10.get("n_qn_cols") == 10
                  and g10.get("bad_syl_cols") == [] and str(g10.get("col_method", "")).startswith("hybrid"), g10)
            check("iter_pairs 10 cặp (nom_idx i, line_id i+1)",
                  det10 is not None and det10[2] == [(i, i + 1) for i in range(10)])
            # cột QN thiếu âm -> page_ok False, gate nêu cột
            root = Path(td) / "B10s"; _fake_page(root, 10, n_chars=14, syl_per_line=13)
            (root / "transcriptions" / "page_0001.json").write_text(json.dumps(
                {"book_page": 1, "columns": [{"column": i + 1, "num_syllables": 14} for i in range(10)]}), encoding="utf-8")
            g13: dict = {}
            det13 = AP._detect("page_0001", root, set(), layout=lay, gate_out=g13)
            check("mỗi cột 13 âm (JSON ghi 14): page_ok False, 10 cột lệch trong gate",
                  det13 is not None and det13[4] is False and len(g13["bad_syl_cols"]) == 10)
            # JSON ghi đúng 13 -> qua cổng ("như transcriptions ghi")
            (root / "transcriptions" / "page_0001.json").write_text(json.dumps(
                {"book_page": 1, "columns": [{"column": i + 1, "num_syllables": 13} for i in range(10)]}), encoding="utf-8")
            det13b = AP._detect("page_0001", root, set(), layout=lay)
            check("JSON ghi num_syllables=13 khớp .txt -> page_ok True", det13b is not None and det13b[4] is True)
            # 9 cột kim / 10 dòng QN: detect_nom_columns_v3 (mã cũ) rơi xuống projection_fallback
            # và ÉP ảnh về n_expected=10 cột -> gate phải ghi col_method để build đếm được
            # tỉ lệ trang fallback (SPEC §7: hybrid >= 95 % trang).
            root = Path(td) / "B9l"; _fake_page(root, 9, n_chars=14, n_lines=10, syl_per_line=14)
            g9: dict = {}
            det9 = AP._detect("page_0001", root, set(), layout=lay, gate_out=g9)
            check("9 cột kim / 10 dòng QN với n_columns=10: col_method=projection_fallback -> page_ok False",
                  det9 is not None and g9.get("col_method") == "projection_fallback"
                  and g9["n_nom_cols"] == 10 and det9[4] is False, g9)
            # cache 8 cột thật (5 cột/2 tầng tựa trang 105 LVT) + 8 dòng: thiếu so với 10 -> FAIL
            (root / "detected" / "page_0001_ocr_cache.json").write_text(json.dumps(
                {"coords_space": "fullpage", "framed": False, "columns": []}), encoding="utf-8")
            gz: dict = {}
            det0 = AP._detect("page_0001", root, set(), layout=lay, gate_out=gz)
            check("cache 0 cột với lithograph: page_ok False (fallback ép 10 cột từ ảnh, không hybrid)",
                  det0 is not None and det0[4] is False and gz["col_method"] == "projection_fallback", gz)
            # ảnh không nhị phân hoá được -> col_method="hybrid_no_image" (cột chỉ từ cache):
            # lithograph KHÔNG qua cổng; STT mặc định giữ hành vi cũ (page_ok theo đếm cột).
            root = Path(td) / "B10ni"; _fake_page(root, 10, n_chars=14, syl_per_line=14)
            _lab = AP.load_and_binarize
            def _boom(_p):
                raise RuntimeError("giả lập ảnh hỏng")
            AP.load_and_binarize = _boom
            try:
                gni: dict = {}
                detni = AP._detect("page_0001", root, set(), layout=lay, gate_out=gni)
                check("ảnh hỏng (hybrid_no_image) với lithograph: page_ok False, gate ghi col_method",
                      detni is not None and detni[4] is False and gni.get("col_method") == "hybrid_no_image", gni)
                root9 = Path(td) / "B9ni"; _fake_page(root9, 9, n_chars=14, n_lines=9, syl_per_line=14)
                detni9 = AP._detect("page_0001", root9, set())
                check("ảnh hỏng với STT mặc định: page_ok như cũ (9 cột == 9 dòng -> True)",
                      detni9 is not None and detni9[4] is True, detni9 and detni9[4])
            finally:
                AP.load_and_binarize = _lab
            # align_page: bản ghi STT không có khoá layout_gate; lithograph có
            root = Path(td) / "B10p"; _fake_page(root, 10, n_chars=14, syl_per_line=14)
            rec10 = AP.align_page("page_0001", root, set(), {}, {}, "old", layout=lay)
            check("align_page lithograph: bản ghi có layout_gate + page_ok True",
                  rec10 is not None and rec10.get("page_ok") is True and "layout_gate" in rec10
                  and rec10["layout_gate"].get("n_nom_cols") == 10,
                  None if rec10 is None else (rec10.get("page_ok"), rec10.get("layout_gate")))
            root = Path(td) / "B9p"; _fake_page(root, 9, n_chars=14, syl_per_line=14)
            rec9 = AP.align_page("page_0001", root, set(), {}, {}, "old")
            check("align_page mặc định (STT): bản ghi KHÔNG có khoá layout_gate",
                  rec9 is not None and "layout_gate" not in rec9 and rec9.get("page_ok") is True,
                  None if rec9 is None else sorted(rec9))
    finally:
        AP.detect_nom_columns_v3 = orig


def test_signatures():
    print("[5] chữ ký / nối dây")
    sig = inspect.signature(AP.align_page)
    check("align_page có tham số layout mặc định None",
          "layout" in sig.parameters and sig.parameters["layout"].default is None)
    check("_get_qn_lines n_columns mặc định 9",
          inspect.signature(_get_qn_lines).parameters["n_columns"].default == 9)
    src = (REPO / "pipeline" / "align_engine" / "build_dataset.py").read_text(encoding="utf-8")
    check("build_dataset PASS 1: lay = book_layout(b) và align_page(..., layout=lay)",
          "lay = book_layout(b)" in src and "layout=lay)" in src)
    src0 = (REPO / "pipeline" / "step0_setup.py").read_text(encoding="utf-8")
    src1 = (REPO / "pipeline" / "step1_extract.py").read_text(encoding="utf-8")
    check("step0/step1: chỉ nhánh layout=lithograph được vắng pdf",
          'book.get("layout") == "lithograph" and "pdf" not in book' in src0
          and 'book_cfg.get("layout") == "lithograph" and "pdf" not in book_cfg' in src1)
    import yaml
    cfg = yaml.safe_load((REPO / "config" / "pipeline.yaml").read_text(encoding="utf-8"))
    check("config/pipeline.yaml: 3 sách STT không khai layout/n_columns -> DEFAULT_LAYOUT",
          all(BL.book_layout(b) is BL.DEFAULT_LAYOUT for b in cfg["books"]) and len(cfg["books"]) == 3)


def main():
    for t in (test_book_layout, test_det_params, test_gate, test_get_qn_lines, test_detect,
              test_signatures):
        try:
            t()
        except Exception as e:      # noqa: BLE001
            check(f"{t.__name__} ném {type(e).__name__}: {e}", False)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
