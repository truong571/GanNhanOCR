"""Self-test cho pipeline/tools (không pytest).

Run:  .venv/bin/python -m pipeline.tools.selftest

Trước 2026-08-23 thư mục `tools/` KHÔNG có một test nào, dù `fix_tone` đã chạy thật và
`sem_score` được dùng để xếp hạng ô đáng chấm.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def _frame(pairs):
    return pd.DataFrame({"ocr_char": [c for c, _ in pairs],
                         "syllable": [s for _, s in pairs],
                         "tier": ["REVIEW"] * len(pairs)})


def test_fix_tone() -> None:
    print("[tools.fix_tone]")
    from pipeline.tools import fix_tone as F

    check("strip_tone bỏ THANH nhưng GIỮ dấu tạo chữ",
          F.strip_tone("trẩy") == "trây" and F.strip_tone("bụt") == "but",
          f'{F.strip_tone("trẩy")} / {F.strip_tone("bụt")}')
    check("strip_tone KHÔNG gộp vừa/vua (nếu gộp sẽ nối hai từ khác hẳn)",
          F.strip_tone("vừa") != "vua")
    check("strip_all bỏ MỌI dấu, kể cả đ -> d",
          F.strip_all("đến") == "den" and F.strip_all("ấy") == "ay",
          f'{F.strip_all("đến")} / {F.strip_all("ấy")}')

    # TẦNG 1 — chỉ lệch thanh, ứng viên duy nhất
    qn = {"trẩy": ["礼"], "bụt": ["孛"]}
    f, extra = F.plan(_frame([("礼", "trấy"), ("孛", "but")]), qn)
    check("tone_unique: sửa được cả hai", len(f) == 2, str(len(f)))
    check("tone_unique: đích đúng", sorted(f["syllable"]) == ["bụt", "trẩy"],
          str(sorted(f["syllable"])))
    check("nhãn luật là tone_unique", all(r.startswith("tone_unique") for r in f["rule"]),
          str(list(f["rule"])))

    # TẦNG 2 — rụng hẳn dấu tạo chữ (chỉ được thử khi tầng 1 không ra ứng viên)
    f2, _ = F.plan(_frame([("衣", "ay")]), {"ấy": ["衣"]})
    check("diacritic_unique: 'ay' -> 'ấy'",
          len(f2) == 1 and f2["syllable"].iloc[0] == "ấy", str(list(f2["syllable"])))
    f3, _ = F.plan(_frame([("旦", "den")]), {"đến": ["旦"]})
    check("diacritic_unique: 'den' -> 'đến' (qua đ/d)",
          len(f3) == 1 and f3["syllable"].iloc[0] == "đến", str(list(f3["syllable"])))

    # CHỐT AN TOÀN — âm là TỪ CÓ THẬT thì KHÔNG đụng (nhóm B: sai chữ hay sai thanh
    # là hai giả thuyết ngang nhau; sửa máy sẽ CHE mất lỗi chữ)
    f4, _ = F.plan(_frame([("異", "là")]), {"lạ": ["異"], "là": ["羅"]})
    check("KHÔNG đụng khi âm là từ có thật (nhóm B)", len(f4) == 0, str(len(f4)))

    # Nhập nhằng -> để nguyên, ghi vào báo cáo
    f5, e5 = F.plan(_frame([("係", "hê")]), {"hề": ["係"], "hệ": ["係"]})
    check("nhập nhằng thì KHÔNG sửa", len(f5) == 0)
    check("nhập nhằng được ghi vào báo cáo", e5["ambiguous"] and e5["ambiguous"][0]["n"] == 1,
          str(e5["ambiguous"]))

    # Corpus chốt: cùng chữ đã đi với MỘT ứng viên ở chỗ khác trong corpus
    df = _frame([("噲", "goi"), ("噲", "gọi"), ("噲", "gọi")])
    f6, _ = F.plan(df, {"gọi": ["噲"], "gỏi": ["噲"]})
    check("corpus chốt được khi 1 ứng viên đã có bằng chứng",
          len(f6) == 1 and f6["syllable"].iloc[0] == "gọi", str(list(f6["syllable"])))
    check("nhãn luật là *_corpus", all("_corpus" in r for r in f6["rule"]), str(list(f6["rule"])))

    # Rác marker: KHÔNG sửa, chỉ gắn cờ
    f7, e7 = F.plan(_frame([("白", "3"), ("景", "000"), ("祖", "%")]), {"ba": ["白"]})
    check("rác marker KHÔNG bị sửa", len(f7) == 0)
    check("rác marker được đếm", e7["marker_garbage_cells"] == 3,
          str(e7["marker_garbage_cells"]))

    # Luỹ đẳng: chạy trên bảng đã đúng thì không sửa gì
    f8, _ = F.plan(_frame([("礼", "trẩy")]), {"trẩy": ["礼"]})
    check("luỹ đẳng (âm đã khớp -> 0 sửa)", len(f8) == 0)


def test_sem_score() -> None:
    print("[tools.sem_score]")
    from pipeline.tools import sem_score as S

    # Corpus phải ĐỦ LỚN để IDF có nghĩa: từ xuất hiện ở 2/3 số mục thì
    # idf = log(3/3) = 0 và mọi điểm đều bằng 0 — đó là tính chất đúng của IDF,
    # không phải lỗi. Thêm nhiễu để "water" thành từ hiếm.
    uni = {"A": {"water", "river"}, "B": {"water", "stream"}, "C": {"fire", "flame"}}
    uni.update({f"N{i}": {f"filler{i}", "common"} for i in range(40)})
    qn = {"nuoc": ["A", "B"], "lua": ["C"]}
    sc = S.SemScorer(uni, qn)

    v_in = sc.score("A", "nuoc")     # leave-one-out: hồ sơ còn B (water, stream)
    v_out = sc.score("C", "nuoc")
    check("chữ cùng trường nghĩa -> điểm > 0", v_in and v_in > 0, str(v_in))
    check("chữ khác trường nghĩa -> điểm 0", v_out == 0, str(v_out))
    check("điểm cùng trường > khác trường", v_in > v_out)
    check("leave-one-out: KHÔNG tự khớp với chính mình",
          sc._profile("lua", exclude="C") == {} and sc.score("C", "lua") is None)
    check("chữ không có nghĩa Hán -> None", sc.score("ZZZ", "nuoc") is None)

    v, tag = sc.verdict("C", "nuoc")
    check("điểm 0 -> 'khong_ket_luan', KHÔNG phải 'sai'", tag == "khong_ket_luan", tag)
    check("ngưỡng xác nhận cao hơn ngưỡng khả năng", S.TAU_CONFIRM > S.TAU_LIKELY)


def test_syllable_normalize() -> None:
    """Chuẩn hoá TRƯỚC align — dùng đọc âm của MỌI chữ trong cột (cặp ghép chưa có)."""
    print("[align_engine.syllable_normalize]")
    from pipeline.align_engine import syllable_normalize as SN

    qn = {"trẩy": ["礼"], "ấy": ["衣"], "đến": ["旦"], "lạ": ["異"], "là": ["羅"],
          "hề": ["係"], "hệ": ["係"]}
    R = SN.build_readings(qn)
    keys = set(qn)
    ch = lambda *cs: [{"ocr_char": c} for c in cs]

    out, log = SN.normalize_column(ch("礼"), ["trấy"], keys, R)
    check("tone_unique: 'trấy' -> 'trẩy'", out == ["trẩy"], str(out))
    check("nhật ký ghi luật", log and log[0]["rule"] == "tone_unique", str(log))

    out, _ = SN.normalize_column(ch("衣"), ["ay"], keys, R)
    check("diacritic_unique: 'ay' -> 'ấy'", out == ["ấy"], str(out))
    out, _ = SN.normalize_column(ch("旦"), ["den"], keys, R)
    check("diacritic_unique qua đ/d: 'den' -> 'đến'", out == ["đến"], str(out))

    out, _ = SN.normalize_column(ch("異", "羅"), ["là"], keys, R)
    check("CHỐT 1 — âm là từ có thật thì KHÔNG đụng", out == ["là"], str(out))

    out, log = SN.normalize_column(ch("係"), ["hê"], keys, R)
    check("CHỐT 3 — nhập nhằng thì để nguyên", out == ["hê"], str(out))
    check("nhập nhằng được ghi log", any(e["action"] == "ambiguous_kept" for e in log), str(log))

    out, _ = SN.normalize_column(ch("礼"), ["3"], keys, R)
    check("rác marker không bị đụng (việc của parser)", out == ["3"], str(out))

    out, _ = SN.normalize_column(ch("礼"), ["xyz"], keys, R)
    check("không có ứng viên -> giữ nguyên", out == ["xyz"], str(out))

    out, _ = SN.normalize_column([], ["trấy"], keys, R)
    check("cột không có chữ -> giữ nguyên", out == ["trấy"], str(out))

    o1, _ = SN.normalize_column(ch("礼"), ["trấy"], keys, R)
    o2, l2 = SN.normalize_column(ch("礼"), o1, keys, R)
    check("luỹ đẳng (chạy lại không đổi thêm)",
          o2 == o1 and not [e for e in l2 if e["action"] == "fixed"], str(l2))
    check("build_readings có cache (cùng đối tượng -> cùng kết quả)",
          SN.build_readings(qn) is R)


def test_check_deps() -> None:
    """Kiểm thư viện chạy MỖI LẦN, và tôn trọng ba cạm bẫy của môi trường 3.14."""
    from pathlib import Path as _P
    REPO = _P(__file__).resolve().parents[2]
    print("[kiểm thư viện]")
    from pipeline.tools import check_deps as CD
    req = CD.parse_req()
    check(f"đọc được requirements.txt ({len(req)} gói)", len(req) >= 20)
    names = {n.lower() for n, _, _ in req}
    check("dùng pygame-ce, KHÔNG dùng pygame (3.14 không có wheel)",
          "pygame-ce" in names and "pygame" not in names)
    check("vietocr nằm trong nhóm phải cài --no-deps", "vietocr" in CD.NO_DEPS)
    a = CD.audit()
    check("mọi gói khai báo đều được phân loại",
          len(a["ok"]) + len(a["thieu"]) + len(a["lech"]) == len(req))
    src = (REPO / "pipeline" / "tools" / "check_deps.py").read_text(encoding="utf-8")
    check("--fix cài --no-deps RIÊNG chứ không gộp", '"--no-deps"' in src)
    check("--fix kiểm LẠI sau khi cài", "b = audit()" in src)
    rp = (REPO / "run_pipeline.sh").read_text(encoding="utf-8")
    check("run_pipeline gọi check_deps ở preflight", "pipeline.tools.check_deps" in rp)
    check("thiếu gói thì TỰ VÁ chứ không chỉ báo", "check_deps --fix" in rp)
    check("có đường thoát SKIP_DEPS", "SKIP_DEPS" in rp)


def test_dict_candidates() -> None:
    """Bảng ứng viên từ điển PHẢI có đủ cột phân xử.

    Bản đầu (2026-08-25) bỏ mất `similar_hit` và `char_in_dict` — đúng hai cột dùng để
    phân biệt "TỪ ĐIỂN THIẾU" với "OCR NHẦM TỰ DẠNG" — khiến người duyệt không có căn
    cứ nào để phán. Đây là test chặn việc đó tái diễn.
    """
    from pathlib import Path as _P
    import csv as _csv
    REPO = _P(__file__).resolve().parents[2]
    print("[ứng viên từ điển — đủ cột phân xử]")
    f = REPO / "docs" / "UNG_VIEN_MO_RONG_TU_DIEN.csv"
    if not f.exists():
        print("  [bỏ qua] chưa sinh bảng"); return
    head = next(_csv.reader(open(f, encoding="utf-8")))
    for c in ("similar_hit", "char_in_dict", "gold_count", "nghieng_ve"):
        check(f"có cột {c}", c in head)
    check("KHÔNG dùng cột ảnh mẫu (ô REVIEW không có crop)", "anh_mau_1" not in head)
    check("có toạ độ để mở ảnh trang gốc", "vi_tri_1" in head)
    rows = list(_csv.DictReader(open(f, encoding="utf-8")))
    check("có dòng", len(rows) > 0)
    # bất biến: hễ similar_hit KHÔNG rỗng thì phải xếp vào nhóm nghi OCR nhầm
    sai = [r for r in rows if r["similar_hit"].strip() and r["nghieng_ve"] != "B_OCR_NHAM"]
    check("similar_hit không rỗng => luôn xếp B_OCR_NHAM", not sai,
          f"{len(sai)} dòng lệch")
    check("cột người duyệt để TRỐNG", all(not r["NGUOI_DUYET"].strip() for r in rows))


def test_variant_table() -> None:
    """Bảng dị thể chỉ ĐỀ XUẤT, KHÔNG được sửa bộ nhãn."""
    from pathlib import Path as _P
    import csv as _csv
    REPO = _P(__file__).resolve().parents[2]
    print("[dị thể — máy đề xuất, người quyết]")
    src = (REPO / "pipeline" / "tools" / "variant_table.py").read_text(encoding="utf-8")
    check("KHÔNG ghi vào bộ nhãn", "labels.csv\", \"w\"" not in src and "to_csv" not in src)
    check("đòi HAI điều kiện: cùng âm VÀ nhìn giống", "oneway" in src and "mutual" in src)
    check("bỏ cặp KHÔNG nhìn giống (hai chữ khác nhau thật)", "if not oneway:" in src)
    f = REPO / "docs" / "UNG_VIEN_CHUAN_HOA_DI_THE.csv"
    if not f.exists():
        print("  [bỏ qua] chưa sinh bảng"); return
    rows = list(_csv.DictReader(open(f, encoding="utf-8")))
    check("mã trội luôn nhiều ô hơn mã phụ",
          all(int(r["so_o_troi"]) >= int(r["so_o_phu"]) for r in rows))
    check("cột người duyệt để TRỐNG", all(not r["NGUOI_DUYET"].strip() for r in rows))


def test_dataset_docs() -> None:
    """Tài liệu bộ giao nộp: số ĐỌC TỪ nhãn, và KHÔNG bịa lai lịch thư tịch."""
    from pathlib import Path as _P
    REPO = _P(__file__).resolve().parents[2]
    print("[tài liệu bộ giao nộp]")
    # Đầu ra có HAI chỗ: dataset/ (đã nạp phán quyết) hoặc re-dataset/ (đem chấm).
    # Ghim cứng "dataset" làm 18 test tự bỏ qua sau lần chạy đầu — đúng lỗi vừa vá ở
    # update_bang_so_lieu, và test cũng dính.
    ds = REPO / "dataset"
    if not (ds / "labels.csv").exists():
        ds = REPO / "re-dataset"

    if not (ds / "labels.csv").exists():
        print("  [bỏ qua] chưa có bộ giao nộp"); return
    for n in ("README.md", "DATASHEET.md", "NGUON_THU_TICH.md", "LICENSE.md"):
        check(f"có {n}", (ds / n).exists())
    rd = (ds / "README.md").read_text(encoding="utf-8")
    import csv as _csv
    rows = list(_csv.DictReader(open(ds / "labels.csv", encoding="utf-8")))
    nchar = sum(1 for r in rows if (r.get("label") or "").strip())
    check(f"README nêu đúng số nhãn ký tự ({nchar:,})", f"{nchar:,}" in rd)
    check("README CẢNH BÁO đừng gộp hai loại nhãn", "Đừng phát biểu" in rd)
    ng = (ds / "NGUON_THU_TICH.md").read_text(encoding="utf-8")
    check("lai lịch để TRỐNG chứ không bịa", "⬜ CHƯA ĐIỀN" in ng)
    # HỒI QUY: README từng khẳng định "chia tách theo TRANG, không trang nào ở hai phía"
    # trong khi ĐO ĐƯỢC 360/444 trang có cột ở nhiều phía. Tài liệu nói dối về dữ liệu.
    # README phải mô tả ĐÚNG trạng thái ĐO ĐƯỢC, không phải ý định của mã.
    import csv as _c0, collections as _c1
    _rows = list(_c0.DictReader(open(ds / "labels.csv", encoding="utf-8")))
    _pg = _c1.defaultdict(set)
    for _r in _rows:
        _pg[(_r.get("book"), _r.get("page"))].add(_r.get("split", ""))
    _leak = sum(1 for v in _pg.values() if len(v) > 1)
    if _leak == 0:
        check("0 rò rỉ -> README nói RỜI NHAU THEO TRANG", "rời nhau theo TRANG" in rd)
    else:
        check(f"{_leak} trang rò rỉ -> README phải CẢNH BÁO", "CÓ RÒ RỈ" in rd)
        check("README nói rõ chỉ số là CẬN TRÊN", "CẬN TRÊN" in rd)
    if any("label_in_train" in r for r in _rows[:1]):
        check("có cột label_in_train -> README phải nêu", "label_in_train" in rd)

    # ---- HỒI QUY 2026-08-25: DATASHEET nói NGƯỢC README, và chốt chặn không hề đọc nó ----
    # Test cũ chỉ soi biến `rd` (=README) rồi báo 10/10 xanh, trong khi DATASHEET cùng thư
    # mục, sinh cùng lần chạy, khẳng định "chia tách neo ở mức CỘT — 360/444 trang có cột ở
    # nhiều phía". Câu đó là CHUỖI GHIM CỨNG sót lại từ trước khi đổi sang chia theo trang,
    # nằm lọt giữa một f-string mà mọi số quanh nó đều động, và nằm đúng mục "🔴 Giới hạn —
    # đọc trước khi dùng" mà run_pipeline.sh chỉ người đọc tới. Bộ giao nộp tự mâu thuẫn.
    dsh = (ds / "DATASHEET.md").read_text(encoding="utf-8")
    check("DATASHEET không còn chuỗi ghim cứng '360/444'", "360/444" not in dsh)
    check("DATASHEET không mâu thuẫn README về mức chia tách",
          ("CỘT, không phải TRANG" in dsh) == (_leak > 0))
    import re as _re
    _so_muc = _re.findall(r"^(\d+)\. \*\*", dsh, _re.M)
    check("DATASHEET không đánh trùng số mục", len(_so_muc) == len(set(_so_muc)),
          f"trùng: {[x for x in set(_so_muc) if _so_muc.count(x) > 1]}")
    _lab = [r["label"] for r in _rows if (r.get("label") or "").strip()]
    _ngoai = sum(1 for c in _lab if not ("\u4e00" <= c <= "\u9fff"))
    _pct = f"{100*_ngoai/len(_lab):.2f}".replace(".", ",")
    check(f"DATASHEET nêu tỷ lệ ngoài CJK ĐO ĐƯỢC ({_pct}%)", f"{_pct}%" in dsh,
          "còn ghim cứng 1,63%?" if "1,63%" in dsh else "không thấy số đo")

    # ---- HỒI QUY: README từng bảo "lọc label_in_train == 1" cho CẢ bộ ----
    # Cột này RỖNG ở mọi dòng tầng SYLLABLE, nên điều kiện đó vứt luôn chúng: 7.070 dòng
    # chứ không phải 159. README phải nói rõ phạm vi.
    _syl = sum(1 for r in _rows if not (r.get("label") or "").strip())
    if _syl and any("label_in_train" in r for r in _rows[:1]):
        check("README cảnh báo lọc label_in_train vứt nhầm tầng SYLLABLE",
              'label_level == "char"' in rd and "vứt" in rd)

    # ---- HỒI QUY: label_in_train phải đúng TRÊN CHÍNH BỘ NÀY ----
    # build_dataset tính cột này ở Bước 3 trên cả 82k hàng (GOLD+SILVER+REVIEW) rồi đóng
    # băng; bộ giao nộp chỉ có GOLD+SYLLABLE, và confusion-fix còn hạ 1.988 GOLD sau đó.
    # Đo được 31 ô bị gắn SAI là "lớp có trong train". Nay tính lại ở bước xuất.
    _tr = {r["label"] for r in _rows if r.get("split") == "train"
           and r.get("label_level") == "char" and (r.get("label") or "").strip()}
    _sai = sum(1 for r in _rows
               if r.get("label_level") == "char" and (r.get("label") or "").strip()
               and r.get("label_in_train") != ("1" if r["label"] in _tr else "0"))
    check("label_in_train khớp chính bộ giao nộp (0 ô lệch)", _sai == 0, f"{_sai} ô lệch")
    import csv as _csv, collections as _c
    rows = list(_csv.DictReader(open(ds / "labels.csv", encoding="utf-8")))
    pg = _c.defaultdict(set)
    for r in rows:
        pg[(r["book"], r["page"])].add(r.get("split", ""))

    check("nói rõ vì sao để trống", "cố ý để trống" in ng.lower() or "bịa" in ng)


def test_proto_index_ro_ri_split() -> None:
    """Nguyên mẫu thị giác S3 không được lấy từ trang val/test.

    HỒI QUY 2026-08-25: chia tách chuyển từ mức CỘT sang mức TRANG, `index.csv` không ai
    sinh lại, và 8.615/41.835 nguyên mẫu (20,6%) gắn `split=train` hoá ra nằm trên trang
    NAY thuộc val/test. Phép kiểm `--check` cũ chỉ so TIỀN TỐ SÁCH nên vẫn báo "cùng thế
    hệ" — bảo đảm của build_rows bị phá mà không một tín hiệu nào. Trấn an sai, đúng lớp
    lỗi tệ nhất trong kho này.
    """
    import csv as _csv, tempfile
    from pipeline.tools import rebuild_proto_index as rpi
    print("[nguyên mẫu S3 vs chia tách]")
    lech, tong = rpi.split_lech()
    check("index.csv hiện hành: 0 nguyên mẫu rơi vào val/test", lech == 0,
          f"{lech:,}/{tong:,}")
    check("index.csv không rỗng", tong > 0, str(tong))

    # phép kiểm phải THẬT SỰ bắt được, không chỉ đọc số 0 rồi gật
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "idx.csv").write_text(
            "path,label,unicode,split,source\n"
            "dataset_out/gold/stt2_page_0001_c01_001.png,A,U+0041,train,crop\n"
            "dataset_out/gold/stt2_page_0002_c01_001.png,B,U+0042,train,crop\n",
            encoding="utf-8")
        (d / "lab.csv").write_text(
            "book,page,split\nstt2,page_0001,train\nstt2,page_0002,test\n", encoding="utf-8")
        l2, t2 = rpi.split_lech(d / "lab.csv", d / "idx.csv")
        check("tiêm 1 nguyên mẫu val/test -> phép kiểm BẮT được", (l2, t2) == (1, 2),
              f"{l2}/{t2}")


def test_publish_doc_csv_phong_ve() -> None:
    """publish/ đọc CSV phải phòng vệ y như remediation/ — 'nan' là âm Việt thật."""
    print("[publish đọc CSV]")
    src = (REPO / "pipeline" / "publish" / "cli.py").read_text(encoding="utf-8")
    check("publish/cli.py dùng keep_default_na=False", "keep_default_na=False" in src)
    check("publish/cli.py đọc cột cờ là chuỗi", '"label_in_train": str' in src)


def test_step1_khong_mat_trang() -> None:
    """Số trang IN trên bản quét không duy nhất — trùng thì KHÔNG được nuốt trang.

    HỒI QUY 2026-08-25: `page_name = f"page_{n:04d}"` với n do OCR đọc từ ảnh. Trùng thì
    trang sau giữ nguyên tên trang trước, rồi MỌI bước sau (`if not img_path.exists()`,
    cache OCR Nôm, ảnh QN tạm, cache OCR QN) đều thấy "đã có" nên bỏ qua — nó chép lại y
    nguyên trang trước còn trang thật thì biến mất. results vẫn +1 nên total_pages đếm
    LƯỢT chứ không đếm TRANG. Đo được 3 trang mất: STT2 page_0142 (trang PDF 132),
    STT11 page_0010 (trang PDF 28 và 220).
    """
    print("[bước 1 không nuốt trang]")
    src = (REPO / "pipeline" / "step1_extract.py").read_text(encoding="utf-8")
    check("có sổ tên trang đã dùng", "seen_names" in src)
    check("trùng thì TÁCH TÊN chứ không đè", "_p{page_idx:04d}" in src)
    check("và NÓI RA chứ không nuốt", "trùng SỐ TRANG IN" in src)
    check("Total báo cả số tên trang duy nhất", "tên trang duy nhất" in src)

    # luật đặt tên phải tất định và không bao giờ đụng nhau
    seen: dict[str, int] = {}
    ten = []
    for idx, bp in [(0, 10), (28, 10), (130, 142), (132, 142), (220, 10)]:
        nm = f"page_{bp:04d}"
        if nm in seen:
            nm = f"{nm}_p{idx:04d}"
        seen[nm] = idx
        ten.append(nm)
    check("5 lượt trùng số -> 5 tên PHÂN BIỆT", len(set(ten)) == 5, str(ten))
    check("trang đầu giữ nguyên tên gốc", ten[0] == "page_0010")
    check("trang sau mang hậu tố truy được về trang PDF", ten[1] == "page_0010_p0028")


def test_co_trang_lech_cot() -> None:
    """Trang không đủ 9 cột phải được GẮN CỜ, và cờ phải đếm trên TOÀN BỘ hàng.

    Bố cục ván khắc luôn 9 cột. Lần chạy 2026-08-25 lộ ra 3 trang thiếu cột, trong đó
    stt4/page_0110 có OCR Nôm ĐỦ 9 cột nhưng chỉ 8 cột sống tới bước ghép — nên phép kiểm
    "9 cột" chạy sau extract (đếm cột OCR Nôm) cho nó đi qua. Cờ ở bước xuất là chỗ duy
    nhất nhìn được cả hai phía.

    BẪY: nếu đếm cột trên hàng ĐÃ LỌC (bộ giao nộp bỏ REVIEW/SILVER) thì một trang lành
    có nguyên một cột rơi vào REVIEW sẽ bị đếm hụt và mang tiếng oan. Phải đếm trên all_rows.
    """
    import csv as _csv, collections as _cl
    print("[cờ trang lệch cột]")
    ds = REPO / "dataset"
    if not (ds / "labels.csv").exists():
        ds = REPO / "re-dataset"
    full = REPO / "dataset_out" / "labels_final.csv"
    if not (ds / "labels.csv").exists() or not full.exists():
        print("  [bỏ qua] chưa có bộ giao nộp"); return
    rows = list(_csv.DictReader(open(ds / "labels.csv", encoding="utf-8")))
    check("bộ giao nộp có cột page_cot_lech", "page_cot_lech" in (rows[0] if rows else {}))
    if "page_cot_lech" not in (rows[0] if rows else {}):
        return
    check("giá trị chỉ '0'/'1'", {r["page_cot_lech"] for r in rows} <= {"0", "1"})

    cot_full: dict = {}
    for r in _csv.DictReader(open(full, encoding="utf-8")):
        cot_full.setdefault((r["book"], r["page"]), set()).add(r["column"])
    that_lech = {k for k, v in cot_full.items() if len(v) != 9}
    co_co = {(r["book"], r["page"]) for r in rows if r["page_cot_lech"] == "1"}
    check("cờ khớp CHÍNH XÁC tập trang thiếu cột (đếm trên toàn bộ hàng)",
          co_co == (that_lech & {(r["book"], r["page"]) for r in rows}),
          f"cờ {sorted(co_co)} vs thật {sorted(that_lech)}")

    # BẪY đếm trên hàng đã lọc: dựng một trang lành có 9 cột nhưng chỉ 8 cột lọt bộ xuất
    all_rows = [{"book": "b", "page": "p", "column": str(i), "tier": "GOLD"} for i in range(1, 9)]
    all_rows.append({"book": "b", "page": "p", "column": "9", "tier": "REVIEW"})
    loc = [r for r in all_rows if r["tier"] == "GOLD"]
    dung = len({r["column"] for r in all_rows}) != 9
    sai = len({r["column"] for r in loc}) != 9
    check("đếm trên all_rows -> KHÔNG gắn cờ oan", dung is False)
    check("đếm trên hàng đã lọc -> gắn cờ OAN (bẫy đã tránh)", sai is True)

    for f, ten in ((ds / "README.md", "README"), (ds / "DATASHEET.md", "DATASHEET")):
        check(f"{ten} có nêu page_cot_lech",
              "page_cot_lech" in f.read_text(encoding="utf-8"))


def test_run_pipeline_grep_dem() -> None:
    """`grep -c` không khớp gì vẫn IN "0" rồi thoát mã 1 -> `|| echo 0` in thêm một "0".

    Biến thành "0\n0", và (( )) sặc: chính lỗi in ra ở preflight lần chạy 2026-08-25.
    """
    print("[preflight đếm bằng grep -c]")
    src = (REPO / "run_pipeline.sh").read_text(encoding="utf-8")
    check("không còn `grep -c ... || echo 0`",
          "|| echo 0)" not in src.replace("|| echo 0 )", "|| echo 0)"))
    check("dùng gán rồi mới chữa mã thoát", "|| n_old=0" in src and "|| n_new=0" in src)
    import subprocess
    r = subprocess.run(["bash", "-n", str(REPO / "run_pipeline.sh")], capture_output=True)
    check("run_pipeline.sh qua `bash -n`", r.returncode == 0, r.stderr.decode()[:200])
    # tái hiện lỗi gốc để chứng minh test không rỗng
    bad = subprocess.run(
        ["bash", "-c", 'n=$(grep -c zzz /dev/null || echo 0); [ "$n" = "0" ]'],
        capture_output=True)
    check("tái hiện được: `grep -c || echo 0` KHÔNG cho '0'", bad.returncode != 0)


def test_xlsx_khong_bi_excel_an_kieu() -> None:
    """Bản .xlsx phải là ẢNH CHỤP TRUNG THỰC của labels.csv, không để Excel suy kiểu.

    Kho này đã nhiều lần chảy máu vì ép kiểu: label_in_train '1' -> 1.0, âm Quốc ngữ THẬT
    "nan" -> NaN, crop_w '138' -> 138.0. Excel suy kiểu còn hăng hơn pandas, nên mặc định
    ở đây là CHUỖI; chỉ vài cột đo được mới ghi kiểu số.
    """
    import csv as _csv
    from pipeline.tools import make_xlsx as mx
    print("[xlsx trung thực với csv]")
    ds = REPO / "dataset"
    if not (ds / "labels.csv").exists():
        ds = REPO / "re-dataset"
    src, dst = ds / "labels.csv", ds / "labels.xlsx"
    if not src.exists() or not dst.exists():
        print("  [bỏ qua] chưa có bộ giao nộp / chưa dựng xlsx"); return
    try:
        from openpyxl import load_workbook
    except ImportError:
        print("  [bỏ qua] thiếu openpyxl"); return
    rows = list(_csv.DictReader(open(src, encoding="utf-8")))
    ws = load_workbook(dst, read_only=True)["labels"]
    it = ws.iter_rows(values_only=True)
    hdr = list(next(it))
    xl = list(it)
    check(f"số dòng khớp ({len(rows):,})", len(xl) == len(rows), f"xlsx {len(xl):,}")
    check("cột khớp đúng thứ tự", hdr == list(rows[0].keys()))
    i = {c: k for k, c in enumerate(hdr)}
    lech = 0
    for a, b in zip(rows, xl):
        for c in hdr:
            if c in mx.COT_SO:
                continue
            # ô rỗng trong xlsx đọc ra None — phải quy về "" TRƯỚC khi so, nếu không
            # chính phép kiểm lại báo động giả trên mọi ô trống (đã dính: 194.875 ô).
            v = b[i[c]]
            if ("" if v is None else str(v)) != a[c]:
                lech += 1
    check("0 ô lệch trên mọi cột CHUỖI", lech == 0, f"{lech} ô")
    lit = {str(r[i["label_in_train"]]) for r in xl if r[i["label_in_train"]] is not None}
    check("label_in_train KHÔNG bị hoá số ('1.0')", lit <= {"0", "1"}, str(sorted(lit))[:50])
    cw = {str(r[i["crop_w"]]) for r in xl if r[i["crop_w"]]}
    check("crop_w không có đuôi '.0'", not any("." in v for v in cw), str(sorted(cw)[:3]))
    nan = [r for r in rows if str(r["syllable"]).lower() == "nan"]
    if nan:
        gnan = [r for r in xl if str(r[i["syllable"]]).lower() == "nan"]
        check(f"âm Quốc ngữ 'nan' còn nguyên ({len(nan)} ô)", len(gnan) == len(nan),
              f"xlsx còn {len(gnan)}")


def test_batch_by_rule() -> None:
    """Mẻ chấm PHẢI tách GOLD thành direct/similar — hai lớp rủi ro khác nhau."""
    from pathlib import Path as _P
    import json as _json, collections as _c
    REPO = _P(__file__).resolve().parents[2]
    print("[mẻ chấm — tách theo LUẬT]")
    src = (REPO / "pipeline" / "ground_truth" / "make_combined_batch.py").read_text(encoding="utf-8")
    check("có cờ --by-rule", "--by-rule" in src)
    check("tách được s1_inter_s2_similar", "s1_inter_s2_similar" in src)
    m = REPO / "dataset_out" / "human_audit" / "audit_combined" / "manifest.jsonl"
    if not m.exists():
        print("  [bỏ qua] chưa dựng mẻ"); return
    rows = [_json.loads(l) for l in open(m, encoding="utf-8")]
    st = {str(r.get("stratum", "")).split("|")[0] for r in rows}
    for want in ("GOLD/direct", "GOLD/similar", "SYLLABLE"):
        check(f"có tầng {want}", want in st)
    # BẤT BIẾN ĐÚNG: ô MẪU phải có trọng số; ô LẶP ẨN phải KHÔNG có.
    # Ô lặp dùng đo κ nội tại (người chấm có tự nhất quán không), KHÔNG phải quan sát
    # độc lập — gán trọng số cho nó là ĐẾM TRÙNG dân số và thổi ước lượng.
    mau = [r for r in rows if not r.get("repeat_of")]
    lap = [r for r in rows if r.get("repeat_of")]
    check(f"{len(mau)} ô MẪU đều có design_weight (Horvitz-Thompson)",
          all(r.get("design_weight") for r in mau))
    check(f"{len(lap)} ô LẶP ẨN đều KHÔNG có design_weight (tránh đếm trùng)",
          all(not r.get("design_weight") for r in lap))
    # cảnh báo ảnh hỏng phải tới được manifest
    check("manifest mang crop_quality_flag", "crop_quality_flag" in rows[0])
    h = (m.parent / "audit.html").read_text(encoding="utf-8")
    nbad = sum(1 for r in rows if r.get("crop_quality_flag") in ("blank", "truncated"))
    if nbad:
        check(f"{nbad} ô ảnh hỏng CÓ cảnh báo hiện ra trong HTML",
              "ẢNH TRẮNG" in h or "ẢNH BỊ CẮT" in h)


def main() -> int:
    print("=" * 64)
    print("TOOLS SELFTEST")
    print("=" * 64)
    test_fix_tone()
    test_sem_score()
    test_syllable_normalize()
    test_check_deps()
    test_dict_candidates()
    test_variant_table()
    test_dataset_docs()
    test_proto_index_ro_ri_split()
    test_publish_doc_csv_phong_ve()
    test_step1_khong_mat_trang()
    test_co_trang_lech_cot()
    test_run_pipeline_grep_dem()
    test_xlsx_khong_bi_excel_an_kieu()
    test_batch_by_rule()
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
