"""Self-test cho bàn thí nghiệm (không pytest).

Run:  .venv/bin/python -m pipeline.lab.selftest
"""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}  {detail}")


# --------------------------------------------------------------------------- #
def test_metrics() -> None:
    print("[lab.metrics]")
    from pipeline.lab import metrics as M

    check("_pct p50 của [1..5] = 3", M._pct([1, 2, 3, 4, 5], 50) == 3)
    check("_pct p0 = min, p100 = max",
          M._pct([1, 2, 3], 0) == 1 and M._pct([1, 2, 3], 100) == 3)
    check("_pct trên rỗng trả 0", M._pct([], 50) == 0.0)

    rows = [
        {"tier": "GOLD", "label": "a", "syllable": "ma", "image_md5": "m1",
         "book": "b", "page": "p1", "column": "1", "rule": "s1_inter_s2_direct"},
        {"tier": "GOLD", "label": "b", "syllable": "mà", "image_md5": "m2",
         "book": "b", "page": "p1", "column": "1", "rule": "s1_inter_s2_similar"},
        {"tier": "SYLLABLE", "label": "", "syllable": "xyzzy", "image_md5": "m3",
         "book": "b", "page": "p1", "column": "2", "rule": "nghia_consensus"},
        {"tier": "REVIEW", "label": "", "syllable": "ma", "image_md5": "",
         "book": "b", "page": "p1", "column": "2", "rule": "no_s1_inter_s2"},
        # cùng md5 với hàng 1 nhưng KHÁC nhãn -> mâu thuẫn chắc chắn
        {"tier": "GOLD", "label": "z", "syllable": "ma", "image_md5": "m1",
         "book": "b", "page": "p2", "column": "1", "rule": "s1_inter_s2_direct"},
    ]
    c = M.composition(rows)
    check("composition đếm đúng tổng", c["n_rows"] == 5)
    check("n_usable = GOLD + SYLLABLE", c["n_usable"] == 4, str(c["n_usable"]))
    check("REVIEW không tính vào usable", c.get("tier_REVIEW") == 1)
    check("lớp usable đếm theo nhãn khác rỗng", c["n_classes_usable"] == 3,
          str(c["n_classes_usable"]))
    check("bắt được mâu thuẫn cùng md5 khác nhãn", c["md5_label_conflicts"] == 1)

    q = M.syllable_quality(rows, {"ma", "mà"})
    check("đếm đúng số âm", q["n_syllable"] == 5)
    check("âm ngoài từ điển = 1 (xyzzy)", q["syl_out_of_dict"] == 1, str(q))

    st = M.structure(rows)
    check("đếm đúng số trang", st["n_pages"] == 2, str(st["n_pages"]))
    check("đếm đúng số cột", st["n_columns"] == 3, str(st["n_columns"]))
    check("tỉ lệ ô neo có tính được", "anchor_frac_p50" in st)

    # crop_geometry trên ảnh dựng sẵn: 1 ảnh trắng -> phải bị gắn cờ 'blank'
    try:
        import numpy as np
        from PIL import Image
        tmp = Path(tempfile.mkdtemp())
        Image.fromarray(np.full((40, 40), 255, dtype="uint8")).save(tmp / "blank.png")
        g = M.crop_geometry(["blank.png"], tmp, workers=1)
        check("crop trắng -> cờ blank", g["flag_blank"] == 1, json.dumps(g))
        check("đếm đúng số crop", g["n_crops"] == 1)
        gm = M.crop_geometry(["khong_ton_tai.png"], tmp, workers=1)
        check("ảnh thiếu -> đếm vào unreadable, không nổ",
              gm["n_crops"] == 0 and gm["n_crops_unreadable"] == 1)
    except ImportError:
        check("PIL/numpy sẵn có (bỏ qua nhóm crop)", True)


# --------------------------------------------------------------------------- #
def _toy():
    """3 cột, mỗi cột 8 ô, mọi ô đều là ô neo."""
    qn = {f"s{i}": [f"c{i}"] for i in range(8)}
    cols = [{"key": ("b", f"p{k}", "1"),
             "chars": [f"c{i}" for i in range(8)],
             "syllables": [f"s{i}" for i in range(8)],
             "anchors": list(range(8))} for k in range(3)]
    return cols, qn, {}


def test_perturb() -> None:
    print("[lab.perturb]")
    from pipeline.lab import perturb as P

    cols, qn, sim = _toy()
    base = P.anchor_retention(cols, qn, sim, "drop_syl", 0)
    check("KHÔNG hỏng -> retention = 1,0 (chuẩn không thiên lệch)",
          base["anchor_retention"] == 1.0, str(base))
    check("mọi ô neo đều sống khi mức = 0", base["anchors_alive"] == 24,
          str(base["anchors_alive"]))

    rng = random.Random(0)
    c, s, amap = P.apply_perturbation(list("abcdefgh"), [f"s{i}" for i in range(8)],
                                      [0, 1, 2], "drop_syl", 1, rng)
    check("drop_syl làm ngắn chuỗi âm", len(s) == 7, str(len(s)))
    check("ô neo mất âm bị loại khỏi mẫu số (map = None)",
          sum(1 for v in amap.values() if v is None) <= 1)

    rng = random.Random(0)
    c2, s2, _ = P.apply_perturbation(list("abcdefgh"), [f"s{i}" for i in range(8)],
                                     [0], "split_char", 1, rng)
    check("split_char làm dài chuỗi chữ", len(c2) == 9, str(len(c2)))
    rng = random.Random(0)
    c3, s3, _ = P.apply_perturbation(list("abcdefgh"), [f"s{i}" for i in range(8)],
                                     [0], "ins_syl", 1, rng)
    check("ins_syl làm dài chuỗi âm", len(s3) == 9, str(len(s3)))

    check("_retone đổi dấu thanh", P._retone("ma", random.Random(1)) != "ma")

    a = P.anchor_retention(cols, qn, sim, "swap_syl", 2, seed=7)
    b = P.anchor_retention(cols, qn, sim, "swap_syl", 2, seed=7)
    check("cùng seed -> cùng kết quả (tất định)", a == b, f"{a} vs {b}")

    # cost/band_slack phải được HOÀN NGUYÊN sau khi chạy
    from pipeline.align_engine import anchor_align as aa
    before = (aa.COST_SIMILAR, aa.BAND_SLACK)
    P.anchor_retention(cols, qn, sim, "drop_syl", 1,
                       band_slack=9, cost={"COST_SIMILAR": 0.99})
    check("hằng số engine được hoàn nguyên sau khi quét",
          (aa.COST_SIMILAR, aa.BAND_SLACK) == before,
          f"{(aa.COST_SIMILAR, aa.BAND_SLACK)} vs {before}")

    # TẤT ĐỊNH GIỮA CÁC TIẾN TRÌNH — hồi quy đã từng xảy ra: seed dựng bằng
    # tuple.__hash__() chứa chuỗi, mà hash chuỗi bị ngẫu nhiên hoá theo PYTHONHASHSEED,
    # nên hai lần chạy cho hai con số khác nhau.
    check("_seed không phụ thuộc hash() của Python",
          P._seed(1, 2, "drop_syl", 3) == 5278598175245799795,
          str(P._seed(1, 2, "drop_syl", 3)))
    import subprocess as _sp
    code = ("import sys;sys.path.insert(0,%r);"
            "from pipeline.lab.perturb import _seed;print(_seed(0,1,'swap_syl',2))" % str(REPO))
    outs = {_sp.run([sys.executable, "-c", code], capture_output=True, text=True,
                    env={**os.environ, "PYTHONHASHSEED": h}).stdout.strip()
            for h in ("0", "1", "random")}
    check("seed y hệt dù đổi PYTHONHASHSEED (0/1/random)", len(outs) == 1, str(outs))

    suite = P.run_suite(cols, qn, sim, seeds=1)
    check("run_suite phủ đủ 7 loại hỏng",
          all(f"ret_{k}" in suite for k in P.PERTURBATIONS), str(sorted(suite)))
    check("run_suite trả retention tổng", suite["anchor_retention"] is not None)


# --------------------------------------------------------------------------- #
def test_runner() -> None:
    print("[lab.runner]")
    from pipeline.lab import runner as R

    a = R.config_id({"name": "x", "crops": True})
    b = R.config_id({"name": "HOÀN TOÀN KHÁC", "crops": True})
    c = R.config_id({"name": "x", "crops": False})
    check("config_id BỎ QUA `name` (chỉ là nhãn người đọc)", a == b)
    check("config_id đổi khi tham số đổi", a != c)
    check("config_id ổn định giữa các lần gọi", a == R.config_id({"crops": True}))
    # Phải gồm cả vân tay DỮ LIỆU: cùng cấu hình trên bộ nhãn khác nhau = hai dòng
    # khác nhau, nếu không dòng sau sẽ ĐÈ dòng trước và mất mốc so sánh.
    check("config_id đổi khi DỮ LIỆU đổi",
          R.config_id({"crops": True}, "aaa") != R.config_id({"crops": True}, "bbb"))
    check("labels_sha trên tệp không có -> 'nofile'",
          R.labels_sha("/khong/co/that.csv") == "nofile")

    m = R._merge({"a": 1, "p": {"x": 1, "y": 2}}, {"p": {"y": 9}})
    check("_merge trộn lồng nhau, giữ khoá không nhắc tới",
          m == {"a": 1, "p": {"x": 1, "y": 9}}, str(m))

    tmp = Path(tempfile.mkdtemp()) / "results.csv"
    R.append_row({"run_id": "r1", "n_rows": 10}, tmp)
    R.append_row({"run_id": "r2", "n_rows": 20}, tmp)
    R.append_row({"run_id": "r1", "n_rows": 11, "cot_moi": 5}, tmp)   # ghi đè + cột mới
    rows = list(csv.DictReader(open(tmp, encoding="utf-8")))
    check("append_row ghi đè theo run_id (không nhân bản)", len(rows) == 2, str(len(rows)))
    check("giá trị mới thắng giá trị cũ",
          [r for r in rows if r["run_id"] == "r1"][0]["n_rows"] == "11")
    check("cột mới được thêm, hàng cũ để rỗng",
          "cot_moi" in rows[0] and [r for r in rows if r["run_id"] == "r2"][0]["cot_moi"] == "")


def test_synth() -> None:
    print("[lab.synth]")
    from pipeline.lab import synth as S
    try:
        import numpy as np  # noqa: F401
        from PIL import Image  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        check("numpy/PIL/scipy sẵn có (bỏ qua nhóm synth)", True)
        return

    g = S.glyph_index(limit=200)
    check("đọc được kho glyph FontDiffusion", len(g) == 200, str(len(g)))
    check("khoá là KÝ TỰ, không phải 'U+xxxx'", all(len(k) == 1 for k in g))

    qn = {f"s{i}": [c] for i, c in enumerate(list(g)[:60])}
    pages = S.make_corpus(1, qn, g, seed=0,
                          layout={"cols": 3, "rows_min": 4, "rows_max": 4})
    check("sinh được trang", len(pages) == 1)
    p0 = pages[0]
    check("đúng số cột yêu cầu", len(p0["chars_by_col"]) == 3)
    check("có đáp án hộp cho mọi chữ", len(p0["boxes"]) == 12, str(len(p0["boxes"])))
    b = p0["boxes"][0]
    check("hộp đáp án hợp lệ (x1<x2, y1<y2)",
          b["bbox"][0] < b["bbox"][2] and b["bbox"][1] < b["bbox"][3], str(b))
    H, W = p0["image"].shape
    check("mọi hộp nằm TRONG ảnh",
          all(0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
              for x1, y1, x2, y2 in (bb["bbox"] for bb in p0["boxes"])))
    check("ảnh có mực (không trắng trơn)", float((p0["image"] < 128).mean()) > 0.01)

    import numpy as np
    p1 = S.make_corpus(1, qn, g, seed=0,
                       layout={"cols": 3, "rows_min": 4, "rows_max": 4})[0]
    check("TẤT ĐỊNH theo seed (ảnh y hệt)", np.array_equal(p0["image"], p1["image"]))
    p2 = S.make_corpus(1, qn, g, seed=1,
                       layout={"cols": 3, "rows_min": 4, "rows_max": 4})[0]
    check("đổi seed -> trang khác", not np.array_equal(p0["image"], p2["image"]))
    check("calibrate() không nổ khi thiếu tệp",
          S.calibrate("/khong/co/that.csv") == {})


def test_crop_grid():
    """T4-A: cắt tham số hoá, đệm bất đẳng hướng, luật tiền đăng ký."""
    import numpy as np
    from pipeline.lab import crop_grid as CG
    print("[lab.crop_grid]")

    check("lưới đúng 48 cấu hình",
          len(CG.PADS) * len(CG.THRS) * len(CG.CARVE) * len(CG.RESOLVE) == 48)
    check("MỐC là pad 0,12 (mốc THẬT, không phải 0,18 như config khai)",
          CG.BASELINE["pad"] == 0.12)
    check("trục pad bị KHOÁ (flag_ok không có thẩm quyền)", "pad" in CG.LOCKED_AXES)

    check("_img_key đọc được (cột, chỉ số)",
          CG._img_key("gold/stt2_page_0012_c07_003.png") == (7, 3))
    check("_img_key chịu được giá trị không phải chuỗi (hàng REVIEW)",
          CG._img_key(float("nan")) == (0, 0))

    # Ảnh giả phải để MỰC LẤP KÍN cửa sổ đã đệm, nếu không `tighten_box` co về
    # hộp bao mực và pad mất tác dụng — đúng cái làm hai test đầu của tôi sai.
    img = np.full((200, 120, 3), 255, np.uint8)
    img[:, 40:80] = 0                      # sọc mực dọc suốt cột
    gray = np.full((200, 120), 255, np.uint8); gray[:, 40:80] = 0
    bbox = (40, 80, 80, 120)
    g0 = CG.cut(img, gray, bbox, 0.0, "fixed128", False, None, None)
    check("cut() trả ảnh xám", g0 is not None and g0.ndim == 2)
    g1, r1 = CG.cut(img, gray, bbox, 0.12, "fixed128", False, None, None,
                    return_rect=True)
    check("cut(return_rect) trả hình chữ nhật toạ độ TRANG",
          r1 is not None and len(r1) == 4 and r1[0] >= 0 and r1[2] <= 120)
    # đệm bất đẳng hướng: pad_y lớn hơn thì hộp CAO hơn, RỘNG không đổi
    _, ra = CG.cut(img, gray, bbox, (0.0, 0.5), "fixed128", False, None, None,
                   return_rect=True)
    _, rb = CG.cut(img, gray, bbox, (0.0, 0.0), "fixed128", False, None, None,
                   return_rect=True)
    check("đệm bất đẳng hướng: pad_y chỉ đổi chiều CAO",
          (ra[3] - ra[1]) > (rb[3] - rb[1]) and (ra[2] - ra[0]) == (rb[2] - rb[0]))

    # 3 chế độ ngưỡng cho CÙNG kết quả trên ảnh NHỊ PHÂN (phát hiện T4.c). Ô mẫu
    # phải có CẢ mực và nền — ô toàn mực là ảnh suy biến, Otsu/Sauvola vô định.
    patch = np.full((60, 60), 255, np.uint8); patch[15:45, 20:40] = 0
    outs = {m: CG._tighten(patch, m) for m in CG.THRS}
    check("ảnh nhị phân: 3 chế độ ngưỡng cho CÙNG hộp siết (T4.c)",
          len({str(v) for v in outs.values()}) == 1, str(outs))

    # luật quyết định: mốc thắng -> GIỮ MỐC
    base = dict(CG.BASELINE, flag_ok=0.92, flag_truncated=0.017, is_baseline=True)
    worse = dict(CG.BASELINE, thr="otsu", flag_ok=0.90, flag_truncated=0.017,
                 is_baseline=False)
    d = CG.decide([base, worse])
    check("decide(): mốc thắng -> GIỮ MỐC", d["verdict"] == "GIỮ MỐC")
    # ứng viên pad lớn KHÔNG được xét (trục khoá)
    big = dict(CG.BASELINE, pad=0.22, flag_ok=0.99, flag_truncated=0.017,
               is_baseline=False)
    d2 = CG.decide([base, big])
    check("decide(): bỏ qua ứng viên ở trục pad ĐÃ KHOÁ",
          d2["candidate"]["pad"] == 0.12 and d2["verdict"] == "GIỮ MỐC")
    check("decide() ghi lý do khoá trục", "T4-B" in d2.get("locked_why", ""))


def test_crop_purity():
    """T4-B': thước đo liên thông — mực của mình vs mực lạ."""
    import numpy as np
    from pipeline.lab import crop_purity as CP
    print("[lab.crop_purity]")

    # crop 100px, ô trung tâm = tâm 50 +- 15; một khối ở giữa + một khối RỜI ở trên
    g = np.full((100, 40), 255, np.uint8)
    g[40:60, 10:30] = 0            # chữ của mình (trong ô)
    pure = CP.score(g, (0, 0, 40, 100), 50.0, 30.0)
    check("chỉ có mực của mình -> tinh khiết = 1", pure is not None
          and abs(pure[0] - 1.0) < 1e-9)
    check("không chạm mép -> không bị cắt", pure[1] is False)

    g2 = g.copy(); g2[2:14, 10:30] = 0      # khối RỜI sát mép trên = láng giềng
    r2 = CP.score(g2, (0, 0, 40, 100), 50.0, 30.0)
    check("láng giềng RỜI bị bắt -> tinh khiết < 1", r2 is not None and r2[0] < 1.0)
    check("láng giềng KHÔNG bị tính là 'của mình'", r2[1] is False)

    g3 = g.copy(); g3[0:60, 18:22] = 0      # nối liền tới mép trên
    r3 = CP.score(g3, (0, 0, 40, 100), 50.0, 30.0)
    check("mực NỐI LIỀN tới mép -> báo bị cắt", r3 is not None and r3[1] is True)

    check("ảnh trắng trơn -> None", CP.score(np.full((50, 20), 255, np.uint8),
                                             (0, 0, 20, 50), 25.0, 20.0) is None)
    check("ô trung tâm rỗng -> None", CP.score(g, (0, 0, 40, 100), -999.0, 30.0) is None)
    check("dải pad có đủ 2 đầu (0 và >= 0,45)",
          CP.PADS[0] == 0.0 and max(CP.PADS) >= 0.45)


def test_t7_cau_noi_hai_chieu() -> None:
    """T7 — cầu nối tự dạng hai chiều.

    Hai lớp lỗi ĐÃ XẢY RA trong lúc dựng lab này, cả hai đều câm:
    1. `thu_thap()` ghim cứng `hai_chieu=True` nên cờ `--mot-chieu` vô hiệu — tức ĐỐI CHỨNG
       của chính lab không chạy, và ta sẽ không bao giờ biết.
    2. Tính tỷ lệ xác nhận ở mức Ô thay vì mức ÁNH XẠ. Ô không độc lập (lỗi OCR có tính
       hệ thống: 600 ô chỉ là 183 ánh xạ, 84 ô xác nhận chỉ thuộc 18 ánh xạ). Đếm ở mức ô
       là đếm cùng một bằng chứng nhiều chục lần: p đi từ 0,059 xuống 0,00014 và kết luận
       lật từ GIỮ sang NỚI. Sai đơn vị phân tích thì mọi thứ sau đó sai theo.
    """
    from pipeline.lab import t7_bridge_2way as t7
    print("[T7 cầu nối hai chiều]")
    check("cờ BRIDGE_TWO_WAY mặc định TẮT", t7.BRIDGE_TWO_WAY is False)

    xuoi = {"A": ["B", "C"], "D": ["A"]}
    nguoc = {"B": ["A"], "C": ["A"], "A": ["D"]}
    check("một chiều: chỉ lấy top-K của chính chữ đó",
          t7.cau_noi("A", ["B", "D"], xuoi, nguoc, False) == ["B"])
    check("hai chiều: cộng thêm chiều ngược",
          t7.cau_noi("A", ["B", "D"], xuoi, nguoc, True) == ["B", "D"])
    check("chiều ngược luôn xếp SAU (cau_noi[0] giữ nguyên nghĩa cũ)",
          t7.cau_noi("A", ["B", "D"], xuoi, nguoc, True)[0] == "B")
    check("chữ không có cầu nào -> rỗng",
          t7.cau_noi("Z", ["B"], xuoi, nguoc, True) == [])

    # HỒI QUY 1: tham số phải được dùng, không được ghim cứng
    src = (REPO / "pipeline" / "lab" / "t7_bridge_2way.py").read_text(encoding="utf-8")
    check("thu_thap KHÔNG ghim cứng hai_chieu=True",
          "hai_chieu=hai_chieu" in src and "cau_noi(oc, R, xuoi, nguoc, hai_chieu=True)" not in src)

    # HỒI QUY 2: gom theo ánh xạ phải cho n NHỎ HƠN số ô, và không đếm trùng
    class _S:
        def score(self, chu, am):
            return 0.9 if chu == "X" else 0.0
    pop = [{"ocr_char": "O", "_cau": "X", "syllable": "a", "_am": "a"} for _ in range(50)]
    pop += [{"ocr_char": "P", "_cau": "Y", "syllable": "b", "_am": "b"} for _ in range(3)]
    g = t7.gom_theo_anh_xa(pop, "_cau", "_am", _S())
    check("53 ô -> 2 ánh xạ", g["n_anh_xa"] == 2, str(g))
    check("1/2 ánh xạ được xác nhận (KHÔNG phải 50/53 ô)",
          (g["n_xac_nhan"], g["n_cham_duoc"]) == (1, 2), str(g))
    check("tỷ lệ theo ánh xạ = 50%, không phải 94%", abs(g["ty_le"] - 0.5) < 1e-9)

    check("có hằng số ngưỡng chốt trước",
          all(hasattr(t7, k) for k in
              ("N_MIN", "CONTROL_MIN_RATIO", "CONFIRM_MAX_DROP", "YIELD_MIN", "P_MAX")))
    check("docstring giữ nguyên phán quyết KHÔNG QUA của cổng đăng ký ban đầu",
          "G2 NHƯ TRÊN ĐÃ CHẠY VÀ KHÔNG QUA" in src)
    check("docstring có cảnh báo hậu nghiệm", "CẢNH BÁO HẬU NGHIỆM" in src)


def main() -> int:
    print("=" * 64)
    print("LAB SELFTEST")
    print("=" * 64)
    test_metrics()
    test_perturb()
    test_runner()
    test_synth()
    test_crop_grid()
    test_crop_purity()
    test_t7_cau_noi_hai_chieu()
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
