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


def main() -> int:
    print("=" * 64)
    print("LAB SELFTEST")
    print("=" * 64)
    test_metrics()
    test_perturb()
    test_runner()
    test_synth()
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
