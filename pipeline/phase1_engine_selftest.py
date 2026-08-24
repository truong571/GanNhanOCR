"""Self-test for the Giai đoạn 1 SOURCE fixes that prevent the proven errors recurring.

Run:  .venv/bin/python -m pipeline.phase1_engine_selftest
Tests the three pure, engine-level fixes without needing a full pipeline re-run:
  1. align_production._monotone_assign  — monotone 1-1 box assignment (fixes AE-1)
  2. build_dataset.syllable_gate         — case-insensitive SYLLABLE gate (+labels)
  3. ocr_api retry/backoff helpers       — transient retry, reauth, permanent fail

Exit 0 = all pass.
"""
from __future__ import annotations

from pipeline.align_engine.align_production import _monotone_assign
from pipeline.align_engine.build_dataset import syllable_gate
from core.ocr import ocr_api

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


def box(y):
    """A 10px-tall box centred at y (x fixed)."""
    return [0, y - 5, 10, y + 5]


# --------------------------------------------------------------------------- #
def test_monotone_assign():
    print("[_monotone_assign — fixes AE-1 box-sharing]")
    mid = [box(10), box(20), box(30)]

    # equal counts, well separated: each char gets its own box, in order
    boxes = [box(10), box(20), box(30)]
    cys = [10.0, 20.0, 30.0]
    out = _monotone_assign(cys, boxes, mid)
    centers = [(b[1] + b[3]) / 2 for b in out]
    check("3<->3 exact assignment", centers == [10, 20, 30], str(centers))

    # THE bug scenario: two chars near the same box, one box far.
    # Independent argmin would give BOTH chars box@10 (sharing). Monotone must not.
    boxes = [box(10), box(12), box(40)]
    cys = [10.0, 11.0, 40.0]
    out = _monotone_assign(cys, boxes, mid)
    centers = sorted((b[1] + b[3]) / 2 for b in out)
    check("no box shared (distinct assignment)", len(set(map(tuple, out))) == 3, str(centers))

    # monotonicity: char order (by y) maps to non-decreasing box centers
    boxes = [box(5), box(15), box(25), box(35)]
    cys = [6.0, 16.0, 26.0]
    out = _monotone_assign(cys, boxes, mid)
    oc = [(b[1] + b[3]) / 2 for b in out]
    check("monotone non-decreasing", oc == sorted(oc), str(oc))
    check("distinct boxes when m>n", len({tuple(b) for b in out}) == 3, str(oc))

    # fewer boxes than chars -> None (caller falls back to midpoint)
    check("m<n -> None", _monotone_assign([1.0, 2.0, 3.0], [box(1)], mid) is None)

    # guard: a box wildly off (> 0.5*pitch) is replaced by the midpoint box
    mid2 = [box(10), box(20)]
    boxes = [box(10), box(200)]           # 2nd box far from char@20; pitch=10
    out = _monotone_assign([10.0, 20.0], boxes, mid2, guard=0.5)
    check("far box replaced by midpoint", (out[1][1] + out[1][3]) / 2 == 20, str(out[1]))

    # empty input
    check("empty -> []", _monotone_assign([], [], []) == [])

    # determinism
    a = _monotone_assign([10.0, 11.0, 40.0], [box(10), box(12), box(40)], mid)
    b = _monotone_assign([10.0, 11.0, 40.0], [box(10), box(12), box(40)], mid)
    check("deterministic", a == b)


# --------------------------------------------------------------------------- #
def _rec(ocr_char, syllable, book, page, tier="REVIEW", rule="below_visual_threshold"):
    return {"ocr_char": ocr_char, "syllable": syllable, "book": book, "page": page,
            "tier": tier, "rule": rule}


def test_syllable_gate():
    print("[syllable_gate — case-insensitive, fixes case-split]")
    UNCONF = {"unconfirmed_no_s3", "below_visual_threshold"}

    # 'Nhị' x2 + 'nhị' x3 on 5 distinct pages: cased split would give max 3 (<5 occ fail
    # OR <3 pages); merged = 5 occ / 5 pages / purity 1.0 -> PASS.
    recs = [_rec("二", "Nhị", "b", f"p{i}") for i in range(2)]
    recs += [_rec("二", "nhị", "b", f"p{i}") for i in range(2, 5)]
    ok = syllable_gate(recs, UNCONF)
    check("case variants merge and pass", ("二", "nhị") in ok, str(ok))

    # cased-only would fail: build the split explicitly and confirm the lowercase key
    check("key is lowercase", all(k[1] == k[1].lower() for k in ok))

    # below threshold stays out: 4 occ (<5) must fail
    recs2 = [_rec("三", "tam", "b", f"p{i}") for i in range(4)]
    check("below-occ fails", ("三", "tam") not in syllable_gate(recs2, UNCONF))

    # purity: dominant syllable must be >=0.6. 5 'an' + 5 'ba' -> purity 0.5 fails.
    # (Phải dùng HAI âm tiết QN HỢP LỆ: 'x' bị is_plausible_qn_syllable coi là rác nên
    #  bị lọc TRƯỚC khi tính purity, làm test cũ 'x'/'y' không thực sự kiểm ngưỡng purity.)
    recs3 = ([_rec("四", "an", "b", f"p{i}") for i in range(5)]
             + [_rec("四", "ba", "b", f"p{i+5}") for i in range(5)])
    check("low-purity fails", not any(k[0] == "四" for k in syllable_gate(recs3, UNCONF)))

    # non-UNCONF rules ignored
    recs4 = [_rec("五", "ngu", "b", f"p{i}", rule="s1_inter_s2_direct") for i in range(6)]
    check("non-unconf ignored", ("五", "ngu") not in syllable_gate(recs4, UNCONF))

    # distinct-pages requirement: 6 occ but all on ONE page -> fail (needs >=3 pages)
    recs5 = [_rec("六", "luc", "b", "p1") for _ in range(6)]
    check("single-page fails page requirement", ("六", "luc") not in syllable_gate(recs5, UNCONF))


# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, status):
        self.status_code = status


def test_ocr_retry():
    print("[ocr_api retry/backoff]")
    check("classify 429 -> retry", ocr_api.classify_http_status(429) == "retry")
    check("classify 503 -> retry", ocr_api.classify_http_status(503) == "retry")
    check("classify 401 -> reauth", ocr_api.classify_http_status(401) == "reauth")
    check("classify 403 -> reauth", ocr_api.classify_http_status(403) == "reauth")
    check("classify 404 -> fail", ocr_api.classify_http_status(404) == "fail")
    check("backoff exponential", [ocr_api.backoff_delay(i, base=1.0) for i in range(4)]
          == [1.0, 2.0, 4.0, 8.0])
    check("backoff capped", ocr_api.backoff_delay(20, base=1.0, cap=30.0) == 30.0)

    slept = []
    sleep = lambda s: slept.append(s)

    # transient then success: 503, 503, 200 -> returns the 200, slept twice
    seq = [_FakeResp(503), _FakeResp(503), _FakeResp(200)]
    calls = {"n": 0}
    def do_ok():
        r = seq[calls["n"]]; calls["n"] += 1; return r
    resp = ocr_api._request_with_retry(do_ok, "T", sleep=sleep, on_reauth=lambda: None)
    check("retries transient then succeeds", resp is not None and resp.status_code == 200)
    check("slept for each retry", slept == [1.0, 2.0], str(slept))

    # permanent 404: no retry, returns None
    resp = ocr_api._request_with_retry(lambda: _FakeResp(404), "T",
                                       sleep=lambda s: None, on_reauth=lambda: None)
    check("permanent 404 -> None (no retry)", resp is None)

    # reauth on 401 then success; on_reauth called exactly once
    reauth = {"n": 0}
    seq2 = [_FakeResp(401), _FakeResp(200)]
    c2 = {"n": 0}
    def do_auth():
        r = seq2[c2["n"]]; c2["n"] += 1; return r
    resp = ocr_api._request_with_retry(do_auth, "T", sleep=lambda s: None,
                                       on_reauth=lambda: reauth.__setitem__("n", reauth["n"] + 1))
    check("reauth then success", resp is not None and resp.status_code == 200)
    check("reauth called once", reauth["n"] == 1, str(reauth["n"]))

    # exhausts attempts on persistent 500 -> None, sleeps max_attempts-1 times
    slept2 = []
    resp = ocr_api._request_with_retry(lambda: _FakeResp(500), "T", max_attempts=3,
                                       sleep=lambda s: slept2.append(s), on_reauth=lambda: None)
    check("persistent 5xx exhausts -> None", resp is None)
    check("slept max_attempts-1 times", len(slept2) == 2, str(slept2))

    # timeout exception is retried
    import requests
    to = {"n": 0}
    def do_timeout():
        to["n"] += 1
        if to["n"] < 2:
            raise requests.exceptions.Timeout("boom")
        return _FakeResp(200)
    resp = ocr_api._request_with_retry(do_timeout, "T", sleep=lambda s: None,
                                       on_reauth=lambda: None)
    check("timeout retried then success", resp is not None and resp.status_code == 200)

    # _invalidate_token clears the cache
    ocr_api._token_cache["token"] = "x"; ocr_api._token_cache["exp"] = 9e9
    ocr_api._invalidate_token()
    check("_invalidate_token clears cache", ocr_api._token_cache["token"] == ""
          and ocr_api._token_cache["exp"] == 0.0)


def test_ocr_cache_guard():
    """Chốt chặn cache OCR: cache chỉ hợp lệ với ĐÚNG ảnh đã sinh ra nó.

    Trước 2026-08-22 `image_hash` được GHI mà không bao giờ đối chiếu -> đổi
    pages/*.png là bbox cũ áp lên ảnh mới, lệch toạ độ, không một cảnh báo.
    """
    print("[ocr_api chốt chặn cache]")
    import json as _json
    import os as _os
    import tempfile
    from pathlib import Path as _Path
    try:
        from PIL import Image
    except Exception:
        check("PIL sẵn có (bỏ qua nhóm test này nếu không)", True)
        return

    tmp = _Path(tempfile.mkdtemp())
    book = tmp / "prepared" / "BookX"
    (book / "detected").mkdir(parents=True)
    (book / "pages").mkdir()
    img = book / "pages" / "p.png"
    cache = book / "detected" / "p_ocr_cache.json"

    # ảnh nhiễu tất định (seed cố định) để nén PNG không suy biến
    px = bytes((i * 37 + (i // 64) * 11) % 251 for i in range(64 * 64))
    Image.frombytes("L", (64, 64), px).save(img, "PNG", compress_level=9)

    def write_cache(**extra):
        d = {"image": str(img), "columns": [], "coords_space": "fullpage", "framed": False}
        d.update(extra)
        cache.write_text(_json.dumps(d), encoding="utf-8")

    write_cache(image_hash=ocr_api._file_md5(str(img)))
    check("hash khớp -> ok", ocr_api.verify_cache_image(str(cache), str(img)) == "ok")

    write_cache()  # cache đời cũ, không có image_hash
    check("thiếu image_hash -> skipped",
          ocr_api.verify_cache_image(str(cache), str(img)) == "skipped")

    write_cache(image_hash="deadbeef")
    check("ảnh không tồn tại -> skipped",
          ocr_api.verify_cache_image(str(cache), str(img) + ".nope") == "skipped")

    _os.environ["SN_OCR_SKIP_CACHE_VERIFY"] = "1"
    check("cửa thoát hiểm -> skipped",
          ocr_api.verify_cache_image(str(cache), str(img)) == "skipped")
    del _os.environ["SN_OCR_SKIP_CACHE_VERIFY"]

    # nén lại: byte đổi, PIXEL y hệt (mô phỏng đổi phiên bản Pillow/zlib)
    good_hash = ocr_api._file_md5(str(img))
    good_px = ocr_api._pixel_hash(str(img))
    Image.open(img).save(img, "PNG", compress_level=1)
    check("nén khác -> md5 tệp đổi", ocr_api._file_md5(str(img)) != good_hash)
    check("nén khác -> pixel KHÔNG đổi", ocr_api._pixel_hash(str(img)) == good_px)

    write_cache(image_hash=good_hash)          # chưa có pixel_hash -> không phân xử được
    try:
        ocr_api.verify_cache_image(str(cache), str(img))
        check("thiếu pixel_hash + byte lệch -> phải ném lỗi", False)
    except ocr_api.StaleOCRCacheError:
        check("thiếu pixel_hash + byte lệch -> ném lỗi", True)

    write_cache(image_hash=good_hash, pixel_hash=good_px)
    check("có pixel_hash, pixel khớp -> healed",
          ocr_api.verify_cache_image(str(cache), str(img)) == "healed")
    check("sau khi heal -> ok",
          ocr_api.verify_cache_image(str(cache), str(img)) == "ok")

    # đổi PIXEL thật -> phải chặn
    Image.frombytes("L", (64, 64), bytes((i * 5) % 251 for i in range(64 * 64))).save(img, "PNG")
    try:
        ocr_api.verify_cache_image(str(cache), str(img))
        check("pixel đổi thật -> phải ném lỗi", False)
    except ocr_api.StaleOCRCacheError:
        check("pixel đổi thật -> ném lỗi", True)

    # backfill KHÔNG được vá cache mà image_hash đã lệch (đó là ca cần người xem)
    write_cache(image_hash="0" * 32)
    st = ocr_api.backfill_pixel_hash(str(tmp / "prepared"), verbose=False)
    check("backfill bỏ qua cache có hash lệch", st["hash_mismatch"] == 1 and st["added"] == 0,
          str(st))

    write_cache(image_hash=ocr_api._file_md5(str(img)))
    st = ocr_api.backfill_pixel_hash(str(tmp / "prepared"), verbose=False)
    check("backfill thêm pixel_hash khi hash còn khớp", st["added"] == 1, str(st))
    check("pixel_hash đã được ghi",
          bool(_json.loads(cache.read_text(encoding="utf-8")).get("pixel_hash")))

    import shutil as _shutil
    _shutil.rmtree(tmp, ignore_errors=True)


def test_crop_geometry_wired():
    from pathlib import Path as _P
    REPO = _P(__file__).resolve().parents[1]
    """T4.e + T4.a: hình học cắt ảnh do CẤU HÌNH quyết, không ghim cứng trong mã."""
    import yaml
    from pipeline.align_engine import align_production as AP
    print("[hình học cắt ảnh — cấu hình nối vào mã]")

    check("BOX_OVERLAP_FRAC = 0,10 (đã thử 0 và HOÀN NGUYÊN)", AP.BOX_OVERLAP_FRAC == 0.10,
          f"đang là {AP.BOX_OVERLAP_FRAC}")

    cfg = yaml.safe_load((REPO / "config" / "pipeline.yaml").read_text(encoding="utf-8"))
    s2 = cfg.get("step2") or {}
    check("config có box_overlap_frac", "box_overlap_frac" in s2)
    check("config box_overlap_frac = 0,10", float(s2.get("box_overlap_frac", -1)) == 0.10)
    # T4.a: dòng này từng CHẾT (mã dùng 0.12 trong khi cấu hình khai 0.18)
    check("config crop_pad_frac = 0,12 — khớp giá trị THẬT đã đo",
          abs(float(s2.get("crop_pad_frac", -1)) - 0.12) < 1e-9,
          f"đang là {s2.get('crop_pad_frac')}")

    # hộp cao đúng pitch khi F=0, và cao 1,2*pitch khi F=0,10
    chars = [{"bbox": [10, 100 * i, 110, 100 * i + 90]} for i in range(6)]
    old = AP.BOX_OVERLAP_FRAC
    try:
        for F, want in ((0.0, 100.0), (0.10, 120.0)):
            AP.BOX_OVERLAP_FRAC = F
            bx = AP._reseg_column({"chars": chars})
            hs = [b[3] - b[1] for b in bx[1:-1]]          # bỏ hộp đầu/cuối (nửa pitch)
            check(f"F={F}: hộp giữa cao {want:.0f}px (= pitch × {1 + 2 * F:.2f})",
                  all(abs(h - want) <= 1 for h in hs), f"đo được {hs}")
    finally:
        AP.BOX_OVERLAP_FRAC = old

    # build_dataset phải ĐỌC cấu hình, không ghim cứng
    src = (REPO / "pipeline" / "align_engine" / "build_dataset.py").read_text(encoding="utf-8")
    check("build_dataset đọc step2.crop_pad_frac", 'get("crop_pad_frac"' in src)
    check("build_dataset đọc step2.box_overlap_frac", 'get("box_overlap_frac"' in src)
    check("--pad mặc định None (để cấu hình quyết)",
          'ap.add_argument("--pad", type=float, default=None)' in src)


def test_labels_sorted():
    """T6.b: thứ tự dòng KHÔNG phụ thuộc thứ tự sách trong cấu hình."""
    from pathlib import Path as _P
    REPO = _P(__file__).resolve().parents[1]
    print("[T6.b — sắp dòng theo khoá canon]")
    src = (REPO / "pipeline" / "align_engine" / "build_dataset.py").read_text(encoding="utf-8")
    check("có sắp trước khi ghi", 'labels.sort(key=lambda r: r["_sort"])' in src)
    check("khoá = (sách, trang, cột, chỉ-số)", '"_sort": (r["book"], r["page"]' in src)
    check("_sort bị gỡ trước khi ghi CSV", 'r.pop("_sort", None)' in src)
    fields_blk = src[src.index("    fields = ["):src.index("    fields = [") + 400]
    check("_sort KHÔNG nằm trong cột xuất ra", "_sort" not in fields_blk)

    # khoá sắp phải GIỮ NGUYÊN thứ tự trong cột (nếu không, phân tích dựa vào
    # tính liền kề dọc cột sẽ hỏng)
    rows = [{"_sort": ("stt4", "page_0002", 3, 1)}, {"_sort": ("stt2", "page_0009", 1, 0)},
            {"_sort": ("stt2", "page_0009", 1, 2)}, {"_sort": ("stt2", "page_0009", 1, 1)}]
    rows.sort(key=lambda r: r["_sort"])
    check("trong cùng cột, thứ tự chỉ-số tăng dần",
          [r["_sort"][3] for r in rows[:3]] == [0, 1, 2])
    check("sách sắp theo tên, không theo cấu hình", rows[0]["_sort"][0] == "stt2")


def main() -> int:
    print("=" * 64)
    print("PHASE-1 ENGINE-FIX SELFTEST")
    print("=" * 64)
    test_monotone_assign()
    test_syllable_gate()
    test_ocr_retry()
    test_ocr_cache_guard()
    test_crop_geometry_wired()
    test_labels_sorted()
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
