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
    ds = REPO / "dataset"
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
    check("nói rõ vì sao để trống", "cố ý để trống" in ng.lower() or "bịa" in ng)


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
    test_batch_by_rule()
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
