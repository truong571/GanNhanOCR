"""Self-test for the Giai đoạn 1 SOURCE fixes that prevent the proven errors recurring.

Run:  .venv/bin/python -m pipeline.phase1_engine_selftest
Tests the three pure, engine-level fixes without needing a full pipeline re-run:
  1. align_production._monotone_assign  — monotone 1-1 box assignment (fixes AE-1)
  2. build_dataset.syllable_gate         — case-insensitive SYLLABLE gate (+labels)
  3. ocr_api retry/backoff helpers       — transient retry, reauth, permanent fail
  4. anchor_align (A-4, flow N0e/N3d/N3e) — ma trận CALIB, cost_fn, band_touched,
     posterior_matches (argmax == Viterbi, Σp theo hàng ≤ 1)
  5. build_dataset PASS 1b (A-5, flow N3g/N4a–N4c/N4e) — pair_pages LOO từ mọi match,
     cost_fn neo cap 2,0, DP lại + posterior cùng cost_fn, record cờ dày 0/1
  6. assign_boxes 3 nhánh (A-6, flow N3f/N4d) — equal_qn/equal_ocr/conflict, split/
     midpoint, hằng thr/x-margin, cache detector theo thr; --box-rule legacy tái lập
     bbox dataset_out_v3/labels_HEAD_N0c.csv (bản tái lập HEAD N0c) 100% trên 5 trang
     (cần detector + prepared/, nếu thiếu thì BỎ QUA có ghi rõ)
  7. PASS 1c (A-7, flow N5a–N5h) — tier_v3 (feats LOO) -> tier/rule/label, chốt
     is_plausible, L1 có cổng 2-gram ÂM–ÂM, khoá QĐ-01 theo nom_idx (lock/pending/khe/
     bo/khac), 'người' ngoài khoá, flank_gold — trên dữ liệu giả (tier_v3_selftest.py so
     feats với lab thuc_nghiem trên cols.pkl)
  8. A-15 (flow N0e) — enforce_count (M>N / M<N / rỗng / gray None / NMS) trên chính
     train_crop/infer_centernet; _pair_new_state nối 3 nhánh hộp trên cột giả với detector
     giả (a: G[syl_idx], b: G[nom_idx], c: enforce_count chỉ ở đây) + legacy trọn gói +
     _pair_new; bảng chân trị tier_v3/rule_of từng nhánh + chốt is_plausible. (Test
     apply_am_sua_dau có/không 2-gram + hoà nằm ở mục 7; decisions chưa ký không áp ở
     pipeline/decisions_selftest.py; glyph_fix --mode kiem ở remediation/selftest.py.)

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


def test_ocr_guest_mode():
    """Guest Mode (mock, không mạng): token lấy MỚI ở mỗi lần thử; 401 sau re-login
    "not active" -> Guest 1 lần rồi NHỚ; đường thành công y nguyên (1 request, token)."""
    print("[ocr_api guest mode (mock)]")
    import os

    class _Resp:
        def __init__(self, status, text="", payload=None):
            self.status_code = status
            self.text = text
            self._payload = payload if payload is not None else {"is_success": True, "data": {"file_name": "f"}}

        def json(self):
            return self._payload

    def run(server, logins):
        """server(auth_header) -> _Resp; logins = list các token mà login sinh ra."""
        seen = []
        posts = []

        def fake_post(url, **kw):
            auth = kw.get("headers", {}).get("Authorization", "<none>")
            seen.append(auth)
            return server(auth)

        def fake_login(u, p):
            tok = logins.pop(0) if logins else None
            return (tok, 9e9) if tok else None

        saved = (ocr_api.requests.post, ocr_api._login_hcmus, dict(ocr_api._token_cache),
                 ocr_api._guest_mode, os.environ.get("SN_OCR_USERNAME"), os.environ.get("SN_OCR_PASSWORD"),
                 os.environ.get("SN_OCR_TOKEN"))
        ocr_api.requests.post = fake_post
        ocr_api._login_hcmus = fake_login
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tf.write(b"x"); img = tf.name
            r1 = ocr_api.upload_image(img)
            n1 = len(seen)
            r2 = ocr_api.upload_image(img)       # lần 2: kiểm tra có nhớ trạng thái
            os.unlink(img)
            flag = ocr_api.is_guest_mode()
            return r1, r2, seen[:n1], seen[n1:], flag
        finally:
            ocr_api.requests.post, ocr_api._login_hcmus = saved[0], saved[1]
            ocr_api._token_cache.clear(); ocr_api._token_cache.update(saved[2])
            ocr_api._guest_mode = saved[3]
            for k, v in (("SN_OCR_USERNAME", saved[4]), ("SN_OCR_PASSWORD", saved[5]), ("SN_OCR_TOKEN", saved[6])):
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v

    def setup(cached="OLD"):
        os.environ["SN_OCR_USERNAME"] = "u"; os.environ["SN_OCR_PASSWORD"] = "p"
        os.environ.pop("SN_OCR_TOKEN", None)
        ocr_api._token_cache["token"] = cached; ocr_api._token_cache["exp"] = 9e9
        ocr_api.reset_guest_mode()

    # (1) 200 bình thường: 1 request/lần, token cache, không login, không guest
    setup("OLD")
    r1, r2, a1, a2, g = run(lambda auth: _Resp(200), logins=[])
    check("200: upload ok", r1 == "f" and r2 == "f")
    check("200: đúng 1 request mỗi lần, token cache", a1 == ["Bearer OLD"] and a2 == ["Bearer OLD"], f"{a1} {a2}")
    check("200: không bật guest", not g)

    # (2) 401 (token hết hạn) -> re-login -> gửi token MỚI -> 200
    setup("OLD")
    srv = lambda auth: _Resp(200) if auth == "Bearer NEW" else _Resp(401, "token expired")
    r1, r2, a1, a2, g = run(srv, logins=["NEW"])
    check("401->re-login: thành công", r1 == "f" and r2 == "f")
    check("401->re-login: lần thử 2 gửi token MỚI (không phải cũ)", a1 == ["Bearer OLD", "Bearer NEW"], str(a1))
    check("401->re-login: lần gọi sau dùng token mới, không guest", a2 == ["Bearer NEW"] and not g, str(a2))

    # (3) 401 "not active" kể cả sau re-login -> Guest 1 lần rồi NHỚ
    setup("OLD")
    srv = lambda auth: _Resp(200) if auth == "<none>" else _Resp(401, '{"message":"User is not active"}')
    r1, r2, a1, a2, g = run(srv, logins=["NEW", "NEW2", "NEW3"])
    check("not active: guest thành công", r1 == "f" and r2 == "f")
    check("not active: OLD -> re-login NEW -> guest", a1 == ["Bearer OLD", "Bearer NEW", "<none>"], str(a1))
    check("not active: lần gọi sau đi thẳng guest (không login/401 lặp)", a2 == ["<none>"], str(a2))
    check("not active: cờ _guest_mode bật", g)
    ocr_api._guest_mode = True; ocr_api.reset_guest_mode()
    check("reset_guest_mode tắt cờ", not ocr_api.is_guest_mode())

    # (4) 401 KHÔNG phải not-active và guest cũng 401 -> None, không nhớ guest
    setup("OLD")
    srv = lambda auth: _Resp(401, "bad")
    r1, r2, a1, a2, g = run(srv, logins=["NEW", "NEW2"])
    check("401 toàn bộ: trả None", r1 is None and r2 is None)
    check("401 toàn bộ: không bật guest", not g)

    # (5) 500 bền -> None, KHÔNG rơi xuống guest (guest chỉ cho 401)
    setup("OLD")
    real_sleep = ocr_api.time.sleep; ocr_api.time.sleep = lambda s: None
    try:
        r1, r2, a1, a2, g = run(lambda auth: _Resp(500), logins=[])
    finally:
        ocr_api.time.sleep = real_sleep
    check("500 bền: None, không guest", r1 is None and "<none>" not in a1 and not g, str(a1))


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


def test_proto_cache_not_poisoned():
    """HỒI QUY: KHÔNG được cache bộ nguyên mẫu S3 SUY BIẾN.

    Lỗi thật 2026-08-25: clean_build.sh xoá dataset_out/gold/*.png mà index.csv trỏ
    vào -> embed_path() trả None cho MỌI đường dẫn -> proto RỖNG. Bản cũ vẫn ghi nó
    ra cache KÈM CHỮ KÝ HỢP LỆ, nên mọi lần chạy sau nạp lại bộ rỗng và bỏ qua việc
    dựng. S3 chạy với 0 nguyên mẫu crop trong im lặng suốt nhiều lần build.
    Hậu quả đo được: SILVER 10.547 -> 8.044, SYLLABLE 6.991 -> 7.963.

    Chữ ký dựa trên mtime KHÔNG phát hiện được: index.csv và checkpoint đều không
    đổi, chỉ ẢNH bị xoá. Nên phải kiểm chính KẾT QUẢ.
    """
    from pathlib import Path as _P
    import pickle
    REPO = _P(__file__).resolve().parents[1]
    print("[cache nguyên mẫu S3 — không cache bộ suy biến]")
    src = (REPO / "pipeline" / "align_engine" / "visual_signal.py").read_text(encoding="utf-8")
    check("có chốt chặn suy biến trước khi ghi cache",
          "got < max(1, want // 2)" in src)
    check("chốt chặn nằm TRƯỚC pickle.dump",
          src.index("got < max(1, want // 2)") < src.index('pickle.dump({**proto'))
    check("có return sớm để KHÔNG ghi cache", "return proto" in
          src[src.index("got < max(1, want // 2)"):src.index('pickle.dump({**proto')])

    cache = REPO / "pipeline" / "align_engine" / "s3_proto_cache.pkl"
    if cache.exists():
        d = pickle.load(open(cache, "rb"))
        n = len([k for k in d if k != "__sig__"])
        check(f"cache trên đĩa KHÔNG suy biến ({n:,} lớp)", n >= 500, f"chỉ có {n}")


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


def test_anchor_align_calib():
    """A-4 (flow N0e, N3d, N3e): ma trận CALIB + cost_fn + band_touched + posterior."""
    from pipeline.align_engine import anchor_align as aa
    print("[anchor_align — CALIB / cost_fn / band_touched / posterior_matches]")

    # --- hằng số == CALIB (−log tỉ số khả năng, RA_SOAT_GAN_NHAN §2 N1) ---
    check("hằng số CALIB 0/2,5/6,7/5,1/8,6/8,6, băng 2",
          (aa.COST_CONFIRM, aa.COST_SIMILAR, aa.COST_DICTMISS, aa.COST_NODICT,
           aa.COST_DEL, aa.COST_INS, aa.BAND_SLACK) == (0.0, 2.5, 6.7, 5.1, 8.6, 8.6, 2),
          f"{(aa.COST_CONFIRM, aa.COST_SIMILAR, aa.COST_DICTMISS, aa.COST_NODICT, aa.COST_DEL, aa.COST_INS, aa.BAND_SLACK)}")
    check("khe (8,6) đắt hơn một cặp lệch từ điển (6,7) — hết khe giả vì 1 chữ OCR sai",
          aa.COST_INS > aa.COST_DICTMISS > aa.COST_NODICT > aa.COST_SIMILAR > aa.COST_CONFIRM)
    check("ANCHOR_CAP = 2,0 (trần chi phí neo ngữ liệu, flow N4b)", aa.ANCHOR_CAP == 2.0)

    # từ điển giả: 8 âm, mỗi âm một chữ Nôm; 'xa' có trong từ điển nhưng chữ khác
    qn = {"thiên": ["天"], "địa": ["地"], "nhân": ["人"], "sơn": ["山"],
          "thuỷ": ["水"], "hoả": ["火"], "mộc": ["木"], "kim": ["金"], "xa": ["車"]}
    chars = ["天", "地", "人", "山", "水", "火", "木", "金"]
    syls = ["thiên", "địa", "nhân", "sơn", "thuỷ", "hoả", "mộc", "kim"]

    def pairs(ops):
        return [(o["nom_idx"], o["syl_idx"]) for o in ops if o["op"] == "match"]

    def kinds(ops):
        return [o["op"] for o in ops]

    # --- cột m=n sạch: đường chéo, mọi cặp confirmed ---
    ops = aa.realign_column(chars, syls, qn)
    check("cột sạch m=n: đường chéo, 8/8 confirmed",
          pairs(ops) == [(i, i) for i in range(8)] and all(o["confirmed"] for o in ops))

    # --- m=n, 1 chữ OCR sai (âm trong từ điển, chữ không phải cách đọc) → 0 khe ---
    c2 = list(chars); c2[3] = "𠀋"
    ops = aa.realign_column(c2, syls, qn)
    check("m=n, 1 chữ sai từ điển (DICTMISS): 0 khe, vẫn đường chéo",
          kinds(ops) == ["match"] * 8 and pairs(ops) == [(i, i) for i in range(8)], kinds(ops))
    check("  … cặp sai đó confirmed=False, 7 cặp kia True",
          [o["confirmed"] for o in ops] == [True] * 3 + [False] + [True] * 4)
    # --- m=n, 1 âm ngoài từ điển (NODICT) → 0 khe ---
    s2 = list(syls); s2[5] = "zzz"
    ops = aa.realign_column(chars, s2, qn)
    check("m=n, 1 âm ngoài từ điển (NODICT): 0 khe, vẫn đường chéo",
          kinds(ops) == ["match"] * 8 and pairs(ops) == [(i, i) for i in range(8)], kinds(ops))
    # --- m=n, 2 chữ sai liên tiếp vẫn KHÔNG mở khe (2×6,7 = 13,4 < 2×8,6) ---
    c3 = list(chars); c3[3] = "𠀋"; c3[4] = "𠀌"
    ops = aa.realign_column(c3, syls, qn)
    check("m=n, 2 chữ sai liên tiếp: vẫn 0 khe (13,4 < 17,2)",
          kinds(ops) == ["match"] * 8, kinds(ops))

    # --- rụng 1 chữ giữa cột → đúng 1 ins tại chỗ, các cặp khác đúng ---
    drop = 4
    c_drop = chars[:drop] + chars[drop + 1:]
    ops = aa.realign_column(c_drop, syls, qn)
    ins = [o["syl_idx"] for o in ops if o["op"] == "ins"]
    truth = {i: (i if i < drop else i + 1) for i in range(7)}
    check("rụng 1 chữ giữa cột: đúng 1 ins tại syl_idx=4",
          ins == [drop] and kinds(ops).count("del") == 0, f"ins={ins} ops={kinds(ops)}")
    check("  … 7/7 cặp ghép đúng chỗ (không lệch thanh ghi)",
          all(truth[i] == j for i, j in pairs(ops)))
    check("  … cột m=7,n=8 KHÔNG chạm biên băng (|i−j|≤1 < 3)",
          aa.band_touched(ops, 7, 8) is False)

    # --- thừa 1 chữ (hộp thừa) → đúng 1 del tại chỗ ---
    c_extra = chars[:drop] + ["丨"] + chars[drop:]
    ops = aa.realign_column(c_extra, syls, qn)
    dels = [o["nom_idx"] for o in ops if o["op"] == "del"]
    truth = {i: (i if i < drop else i - 1) for i in range(9) if i != drop}
    check("thừa 1 chữ giữa cột: đúng 1 del tại nom_idx=4",
          dels == [drop] and kinds(ops).count("ins") == 0, f"del={dels} ops={kinds(ops)}")
    check("  … 8/8 cặp ghép đúng chỗ", all(truth[i] == j for i, j in pairs(ops)))

    # --- khe hai đầu: rụng chữ đầu VÀ chữ cuối → 2 ins ở j=0 và j=7 ---
    ops = aa.realign_column(chars[1:-1], syls, qn)
    check("rụng chữ đầu + cuối: ins tại j=0 và j=7, 6 cặp giữa đúng",
          [o["syl_idx"] for o in ops if o["op"] == "ins"] == [0, 7]
          and pairs(ops) == [(i, i + 1) for i in range(6)])

    # --- cost_fn ghi đè có tác dụng: ép chi phí 0 cho (人,'xa') đổi đường ghép ---
    # chars [天,地,人] vs syls [thiên,địa,xa,nhân'] với 'nhân' đổi thành âm mà 人 KHÔNG
    # là cách đọc (DICTMISS): mặc định hoà 15,3 = 15,3 → ưu tiên M ở ô cuối → ins 'xa'
    # rồi ghép (2,3). cost_fn ép (人,'xa') = 0 → ghép (2,2) + ins cuối = 8,6 < 15,3.
    qn2 = dict(qn); qn2["nhân"] = ["仁"]
    c_cf, s_cf = ["天", "地", "人"], ["thiên", "địa", "xa", "nhân"]
    ops_def = aa.realign_column(c_cf, s_cf, qn2)
    check("cost_fn=None: mặc định ins 'xa' (j=2) rồi ghép (2,3)",
          pairs(ops_def) == [(0, 0), (1, 1), (2, 3)]
          and [o["syl_idx"] for o in ops_def if o["op"] == "ins"] == [2], pairs(ops_def))

    def cf(c, s):
        return 0.0 if (c, s) == ("人", "xa") else aa.substitution_cost(c, s, qn2)
    ops_cf = aa.realign_column(c_cf, s_cf, qn2, cost_fn=cf)
    check("cost_fn ép (人,'xa')=0: đường ghép đổi thành (2,2) + ins j=3",
          pairs(ops_cf) == [(0, 0), (1, 1), (2, 2)]
          and [o["syl_idx"] for o in ops_cf if o["op"] == "ins"] == [3], pairs(ops_cf))
    check("  … neo KHÔNG đổi cờ confirmed (is_confirmed vẫn theo từ điển)",
          [o["confirmed"] for o in ops_cf if o["op"] == "match"] == [True, True, False])

    # --- posterior_matches: argmax theo hàng == Viterbi (cùng cost_fn), Σ_j p ≤ 1 ---
    def agree(nc, sy, q, cost_fn=None):
        vit = dict(pairs(aa.realign_column(nc, sy, q, cost_fn=cost_fn)))
        post = aa.posterior_matches(nc, sy, q, cost_fn=cost_fn)
        rows = {}
        for (i, j), p in post.items():
            rows.setdefault(i, {})[j] = p
        ok_arg = all(max(rows[i], key=rows[i].get) == vit[i] for i in vit)
        ok_sum = all(sum(r.values()) <= 1 + 1e-9 for r in rows.values())
        ok_pos = all(0.0 <= p <= 1 + 1e-9 for p in post.values())
        return ok_arg, ok_sum, ok_pos, post, vit
    cases = [("sạch", chars, syls, qn, None), ("rụng 1", c_drop, syls, qn, None),
             ("thừa 1", c_extra, syls, qn, None), ("2 chữ sai", c3, syls, qn, None),
             ("cost_fn", c_cf, s_cf, qn2, cf)]
    for name, nc, sy, q, cfn in cases:
        ok_arg, ok_sum, ok_pos, post, vit = agree(nc, sy, q, cfn)
        check(f"posterior[{name}]: argmax hàng == Viterbi", ok_arg,
              {i: max((p, j) for (ii, j), p in post.items() if ii == i) for i in vit})
        check(f"posterior[{name}]: Σ_j p(i,j) ≤ 1 và 0 ≤ p ≤ 1", ok_sum and ok_pos)
    post = aa.posterior_matches(chars, syls, qn)
    check("posterior cột sạch: mọi cặp chéo p > 0,99 (T=1, nat)",
          all(post[(i, i)] > 0.99 for i in range(8)), min(post[(i, i)] for i in range(8)))
    post_cf = aa.posterior_matches(c_cf, s_cf, qn2, cost_fn=cf)
    post_no = aa.posterior_matches(c_cf, s_cf, qn2)
    check("posterior đổi theo cost_fn: p(2,2) tăng khi ép (人,'xa')=0",
          post_cf[(2, 2)] > 0.9 > post_no[(2, 2)], (post_cf[(2, 2)], post_no[(2, 2)]))
    check("posterior m=0 hoặc n=0 → {}",
          aa.posterior_matches([], syls, qn) == {} and aa.posterior_matches(chars, [], qn) == {})
    # T lớn làm phẳng phân bố (p giảm) — kiểm chiều nhiệt độ
    post_hot = aa.posterior_matches(c3, syls, qn, T=5.0)
    post_cold = aa.posterior_matches(c3, syls, qn, T=1.0)
    check("T tăng → p của cặp Viterbi giảm (phân bố phẳng hơn)",
          post_hot[(3, 3)] < post_cold[(3, 3)])

    # --- band_touched ---
    diag = [{"op": "match", "nom_idx": i, "syl_idx": i} for i in range(5)]
    check("band_touched: đường chéo m=n=5 → False", aa.band_touched(diag, 5, 5) is False)
    two_del = ([{"op": "del", "nom_idx": 0}, {"op": "del", "nom_idx": 1}]
               + [{"op": "match", "nom_idx": i + 2, "syl_idx": i} for i in range(3)]
               + [{"op": "ins", "syl_idx": 3}, {"op": "ins", "syl_idx": 4}])
    check("band_touched: 2 del liên tiếp trên m=n=5 (|i−j|=2 == băng 2) → True",
          aa.band_touched(two_del, 5, 5) is True)
    one_del = ([{"op": "del", "nom_idx": 0}]
               + [{"op": "match", "nom_idx": i + 1, "syl_idx": i} for i in range(4)]
               + [{"op": "ins", "syl_idx": 4}])
    check("band_touched: 1 del rồi 1 ins (|i−j|≤1 < 2) → False",
          aa.band_touched(one_del, 5, 5) is False)
    check("band_touched: band_slack=1 làm 1 del chạm biên → True",
          aa.band_touched(one_del, 5, 5, band_slack=1) is True)
    # m≠n: băng = |m−n| + 2; rụng 1 chữ hợp lệ không chạm, rụng 3 chữ liên tiếp (m=5,n=8) thì chạm
    ops = aa.realign_column(chars[:3] + chars[6:], syls, qn)   # rụng 3 chữ giữa: m=5, n=8, băng 5
    check("band_touched trên realign thật: rụng 3 chữ liên tiếp m=5,n=8 → |i−j| tối đa 3 < 5 → False",
          aa.band_touched(ops, 5, 8) is False and kinds(ops).count("ins") == 3, kinds(ops))


def test_two_pass_pass1b():
    """A-5 (flow N3g, N4a–N4c, N4e): pair_pages LOO từ MỌI match, cost_fn neo cap 2,0,
    DP lại + posterior cùng cost_fn, record ghi cờ dày 0/1."""
    from pipeline.align_engine import anchor_align as aa
    from pipeline.align_engine import build_dataset as bd
    from pipeline.align_engine import align_production as AP
    import inspect
    print("[PASS 1b — đệ quy hai lượt / pair_pages LOO / p_register]")

    check("build_dataset.ANCHOR_CAP đọc từ engine (= 2,0)",
          bd.ANCHOR_CAP == aa.ANCHOR_CAP == 2.0, (bd.ANCHOR_CAP, aa.ANCHOR_CAP))
    check("ANCHOR_MIN_OTHER_PAGES = 2 (>= 2 trang KHÁC)", bd.ANCHOR_MIN_OTHER_PAGES == 2)
    # align_page phải trả col_states (N3g) và _pair_new_state trả (out, n_gap, ops, boxes)
    check("align_page trả col_states", '"col_states": col_states' in inspect.getsource(AP.align_page))
    check("_pair_new giữ chữ ký cũ (out, n_gap)",
          inspect.getsource(AP._pair_new).strip().endswith("return out, n_gap"))

    qn = {"thiên": ["天"], "địa": ["地"], "nhân": ["仁"], "xa": ["車"]}
    # --- N4a: pair_pages từ MỌI match (cả confirmed=False), LOO theo (book, page) ---
    def rec(ops):
        return {"col_states": [{"ops1": ops}]}
    ops_p = [{"op": "match", "nom_idx": 0, "syl_idx": 0, "ocr_char": "人", "syllable": "Xa",
              "confirmed": False},
             {"op": "ins", "syl_idx": 1, "syllable": "địa"},
             {"op": "match", "nom_idx": 1, "syl_idx": 2, "ocr_char": "天", "syllable": "thiên",
              "confirmed": True},
             {"op": "del", "nom_idx": 2, "ocr_char": "丨"},
             {"op": "match", "nom_idx": 3, "syl_idx": 3, "ocr_char": None, "syllable": "zzz",
              "confirmed": False}]
    pp = bd.build_pair_pages([("stt2", "page_0001", rec(ops_p)), ("stt2", "page_0002", rec(ops_p)),
                              ("stt4", "page_0001", rec(ops_p))])
    check("pair_pages lấy cả cặp confirmed=False, khoá âm-thường",
          pp.get(("人", "xa")) == {("stt2", "page_0001"), ("stt2", "page_0002"), ("stt4", "page_0001")})
    check("pair_pages: ins/del/ocr_char=None KHÔNG vào", (None, "zzz") not in pp and len(pp) == 2, sorted(pp))

    # --- N4b: cost_fn neo = min(base, cap) khi >= 2 trang KHÁC; LOO trừ trang đang xét ---
    base = aa.substitution_cost("人", "xa", qn)          # 車 là cách đọc, 人 không -> DICTMISS 6,7
    cf_other = bd.anchored_cost_fn(pp, ("stt11", "page_0009"), qn, None)   # trang ngoài tập
    cf_here = bd.anchored_cost_fn(pp, ("stt2", "page_0001"), qn, None)     # trang trong tập (3 -> 2 khác)
    check("neo ở trang ngoài tập: cost (人,'xa') 6,7 -> 2,0",
          base == aa.COST_DICTMISS and cf_other("人", "xa") == 2.0, (base, cf_other("人", "xa")))
    check("neo LOO ở trang trong tập (còn 2 trang khác): 2,0", cf_here("人", "xa") == 2.0)
    pp1 = {("人", "xa"): {("stt2", "page_0001"), ("stt2", "page_0002")}}
    check("chỉ 1 trang KHÁC (LOO trừ chính nó): KHÔNG neo, giữ 6,7",
          bd.anchored_cost_fn(pp1, ("stt2", "page_0001"), qn, None)("人", "xa") == aa.COST_DICTMISS)
    check("2 trang khác, trang ngoài tập: neo", bd.anchored_cost_fn(pp1, ("stt4", "page_0001"), qn, None)("人", "xa") == 2.0)
    check("cặp confirmed (0,0) không bị cap nâng lên: min(0, 2) = 0",
          cf_other("天", "thiên") == 0.0)
    check("cặp không có trong pair_pages: chi phí gốc", cf_other("地", "xa") == aa.COST_DICTMISS)
    check("cap ghi đè theo tham số", bd.anchored_cost_fn(pp, ("x", "y"), qn, None, anchor_cap=0.5)("人", "xa") == 0.5)

    # --- N4b/N4c: DP lại đổi đường ghép khi neo; posterior cùng cost_fn; confirmed giữ ---
    chars, syls = ["天", "地", "人"], ["thiên", "địa", "xa", "nhân"]
    ops_def = aa.realign_column(chars, syls, qn)
    pairs = lambda ops: [(o["nom_idx"], o["syl_idx"]) for o in ops if o["op"] == "match"]
    check("không neo: ins 'xa' rồi ghép (2,3)", pairs(ops_def) == [(0, 0), (1, 1), (2, 3)], pairs(ops_def))
    ops2, post, touched, n_anch = bd.realign_with_anchors(chars, syls, qn, None, pp, ("stt11", "page_0009"))
    check("neo (人,'xa') ở 3 trang khác: đường đổi thành (2,2) + ins 'nhân'",
          pairs(ops2) == [(0, 0), (1, 1), (2, 2)], pairs(ops2))
    check("  … n_anchored = 1: chỉ (人,'xa') được HẠ chi phí; (天,'thiên') confirmed 0 không tính",
          n_anch == 1, n_anch)
    check("  … confirmed vẫn theo từ điển: (人,'xa') = False",
          [o["confirmed"] for o in ops2 if o["op"] == "match"] == [True, True, False])
    check("  … band_touched là bool False (m=3,n=4, băng 3)", touched is False)
    rows = {}
    for (i, j), pv in post.items():
        rows.setdefault(i, {})[j] = pv
    check("  … posterior argmax hàng == đường lượt 2 (cùng cost_fn)",
          all(max(rows[i], key=rows[i].get) == j for i, j in pairs(ops2)))
    check("  … p(2,2) > 0,9 khi neo", post[(2, 2)] > 0.9, post[(2, 2)])
    check("  … 0 <= p <= 1", all(0.0 <= v <= 1 + 1e-9 for v in post.values()))
    ops2b, post_b, _, n_b = bd.realign_with_anchors(chars, syls, qn, None, pp1, ("stt2", "page_0001"))
    check("LOO: trang tự nó chỉ còn 1 trang khác -> đường KHÔNG đổi, n_anchored 0",
          pairs(ops2b) == pairs(ops_def) and n_b == 0, (pairs(ops2b), n_b))

    # --- N4e: _record ghi cờ DÀY 0/1, p_register làm tròn 4, rỗng ở đường cũ ---
    class _Dec:  # thay consensus.Decision
        tier, rule_id, label = "REVIEW", "unconfirmed_no_s3", None
    p_new = {"column": 3, "ocr_char": "人", "syllable": "Xa", "bbox": [0, 0, 1, 1],
             "nom_idx": 2, "syl_idx": 2, "syllable_raw": "Xa", "syllable_ocr": "xa",
             "p_register": 0.987654, "band_touched": True}
    r = bd._record("stt2", "page_0001", "x.png", 7, p_new, _Dec, None, "midpoint")
    check("_record: band_touched -> 1 (int), p_register làm tròn 4, syllable lower",
          r["band_touched"] == 1 and r["p_register"] == 0.9877 and r["syllable"] == "xa"
          and r["syllable_ocr"] == "xa" and r["syllable_raw"] == "Xa", r)
    p_old = {"column": 3, "ocr_char": "人", "syllable": "xa", "bbox": None, "nom_idx": 2, "syl_idx": 2}
    r0 = bd._record("stt2", "page_0001", "x.png", 0, p_old, _Dec, None, "midpoint")
    check("_record đường cũ: p_register '' , band_touched 0 (dày), syllable_ocr/raw ''",
          r0["p_register"] == "" and r0["band_touched"] == 0 and r0["syllable_ocr"] == "" and r0["syllable_raw"] == "")
    check("p_register = 0.0 KHÔNG bị coi là rỗng",
          bd._record("b", "p", "x", 0, dict(p_old, p_register=0.0), _Dec, None, "")["p_register"] == 0.0)
    from pathlib import Path as _P
    src = _P(bd.__file__).read_text(encoding="utf-8")
    fb = src[src.index("    fields = ["):src.index("    fields = [") + 700]
    check("4 cột N4e nằm sau cột cũ, trước 5 cột A-6 (giữ thứ tự cột cũ)",
          '"syllable_ocr", "syllable_raw", "p_register", "band_touched",' in fb
          and fb.index('"band_touched"') > fb.index('"nom_idx", "syl_idx"')
          and fb.index('"n_ocr"') > fb.index('"band_touched"'))
    check("--two-pass mặc định BẬT, có --no-two-pass",
          'action=argparse.BooleanOptionalAction, default=True' in src and '"--two-pass"' in src)
    check("PASS 1 (two-pass) KHÔNG gọi decide_label trong vòng trang",
          "page_states.append((_book_code(book), page, page_png, rec))\n                continue" in src)
    check("build_dataset đọc step2.anchor_cap và gán ngược vào engine",
          'get("anchor_cap"' in src and "aa.ANCHOR_CAP = anchor_cap" in src)


def test_box_rule_a6():
    """A-6 (flow N3f/N4d): hằng thr/x-margin, assign_boxes 3 nhánh, cache detector theo thr,
    build_dataset --box-rule + khoá cột QĐ-01; legacy tái lập bbox bộ v3 trên 5 trang."""
    import inspect
    from pathlib import Path as _P
    from pipeline.align_engine import align_production as AP
    from pipeline.align_engine import build_dataset as bd
    print("[A-6 — hộp thô detector + gán hộp 3 nhánh + --box-rule legacy]")
    REPO = _P(__file__).resolve().parents[1]

    check("DETECTOR_THR 0,2 / DETECTOR_XMARGIN 0,25 / MONOTONE_GUARD 0,35",
          (AP.DETECTOR_THR, AP.DETECTOR_XMARGIN, AP.MONOTONE_GUARD) == (0.2, 0.25, 0.35))
    check("LEGACY_DETECTOR_THR 0,3 / LEGACY_DETECTOR_XMARGIN 0,5 (bộ cũ trọn gói)",
          (AP.LEGACY_DETECTOR_THR, AP.LEGACY_DETECTOR_XMARGIN) == (0.3, 0.5))
    check("_get_detector nhận thr, cache theo thr (dict)",
          "thr" in inspect.signature(AP._get_detector).parameters and isinstance(AP._DETECTORS, dict))
    check("_pick_reseg (đường cũ) gọi column_boxes với x_margin=LEGACY_DETECTOR_XMARGIN",
          "x_margin=LEGACY_DETECTOR_XMARGIN" in inspect.getsource(AP._pick_reseg))
    src_bd = _P(bd.__file__).read_text(encoding="utf-8")
    check("build_dataset có --box-rule {syl_index,legacy} mặc định syl_index",
          '"--box-rule", default="syl_index"' in src_bd and AP.BOX_RULES == ("syl_index", "legacy"))
    check("build_dataset đọc step2.det_thr / det_xmargin gán ngược engine (chỉ syl_index)",
          'get("det_thr"' in src_bd and 'get("det_xmargin"' in src_bd)
    check("PASS 1b gán hộp LẠI theo ops2 qua assign_boxes (nhánh a theo syl_idx)",
          "ap_mod.assign_boxes(\n                        cs[\"G\"], ops2" in src_bd)
    check("labels.csv có 5 cột A-6 sau N4e (A-7 nối tiếp sau)",
          '"n_ocr", "n_qn", "n_det", "count_source", "box_source",' in src_bd
          and src_bd.index('"n_ocr", "n_qn", "n_det", "count_source", "box_source",')
          < src_bd.index("*FIELDS_1C.keys()]"))
    # lọc hộp trang theo ngưỡng cũ = tập hộp của lần chạy thr 0,3 (decode = top-k rồi lọc)
    pb = [(0, 0, 10, 10, 0.95), (0, 20, 10, 30, 0.31), (0, 40, 10, 50, 0.29), (0, 60, 10, 70, 0.2)]
    check("_legacy_page_boxes lọc score >= 0,3 khi thr chạy <= 0,3",
          AP._legacy_page_boxes(pb, 0.2) == [pb[0], pb[1]])

    # --- assign_boxes: 3 nhánh trên dữ liệu giả (G 4-int + score) ---
    G = [[0, 100 * k, 100, 100 * k + 90, 0.9] for k in range(5)]        # 5 hộp thô
    ops_a = [{"op": "match", "nom_idx": 0, "syl_idx": 0}, {"op": "ins", "syl_idx": 1},
             {"op": "match", "nom_idx": 1, "syl_idx": 2}, {"op": "match", "nom_idx": 2, "syl_idx": 3},
             {"op": "match", "nom_idx": 3, "syl_idx": 4}]
    bx, src, cs = AP.assign_boxes(G, ops_a, 4, 5)                        # |G| == n_qn = 5, n_ocr 4
    check("(a) |G|==n_qn: equal_qn, hộp theo SYL_IDX (chữ 1 -> G[2]), nguồn detector",
          cs == "equal_qn" and bx[1] == G[2][:4] and bx[3] == G[4][:4] and src == ["detector"] * 4, (cs, bx, src))
    ops_b = [{"op": "match", "nom_idx": k, "syl_idx": (k if k < 2 else k - 1)} for k in range(5) if k != 2] \
        + [{"op": "del", "nom_idx": 2}]
    bx, src, cs = AP.assign_boxes(G, ops_b, 5, 4)                        # |G| == n_ocr = 5 != n_qn 4
    check("(b) |G|==n_ocr!=n_qn: equal_ocr, hộp theo NOM_IDX (kể cả chữ del)",
          cs == "equal_ocr" and bx == [g[:4] for g in G] and src == ["detector"] * 5, (cs, bx))
    bx, src, cs = AP.assign_boxes(G, ops_a, 5, 5)
    check("|G|==n_qn==n_ocr: ưu tiên (a) equal_qn", cs == "equal_qn")
    # (c) |G|=5 (hộp thứ 2 dính 2 chữ) nhưng n_ocr=n_qn=6: cb = enforce_count bổ đôi hộp dính
    cluster = {"chars": [{"bbox": [0, 100 * k, 100, 100 * k + 100]} for k in range(6)],
               "x_range": (0, 100)}
    G2 = [[0, 0, 100, 90, 0.9], [0, 100, 100, 290, 0.8]] + [[0, 100 * k, 100, 100 * k + 90, 0.9] for k in range(3, 6)]
    cb = [G2[0][:4], [0, 100, 100, 195], [0, 195, 100, 290]] + [g[:4] for g in G2[2:]]
    ops_c = [{"op": "match", "nom_idx": k, "syl_idx": k} for k in range(6)]
    bx, src, cs = AP.assign_boxes(G2, ops_c, 6, 6, cluster=cluster, cb=cb)
    check("(c) còn lại: conflict, đủ 6 hộp, 2 nửa hộp bổ đôi ghi 'split', 4 hộp gốc 'detector'",
          cs == "conflict" and len(bx) == 6 and all(b is not None for b in bx)
          and src == ["detector", "split", "split", "detector", "detector", "detector"]
          and bx[1] == [0, 100, 100, 195] and bx[2] == [0, 195, 100, 290], (cs, src, bx))
    far = [[0, 100 * k, 100, 100 * k + 90] for k in range(5)] + [[0, 5000, 100, 5090]]   # 1 hộp rất xa
    bx, src, cs = AP.assign_boxes(G, ops_c, 6, 6, cluster=cluster, cb=far)
    check("(c) hộp xa tâm OCR > guard×pitch -> 'midpoint' (guard MONOTONE_GUARD)",
          cs == "conflict" and "midpoint" in src, src)
    bx, src, cs = AP.assign_boxes(G, ops_c, 6, 6, cluster=cluster, cb=None)
    check("(c) không cb, không det -> midpoint cả cột (như đường cũ rơi mid)",
          cs == "conflict" and src == ["midpoint"] * 6 and all(b for b in bx))
    bx, src, cs = AP.assign_boxes([], ops_a, 4, 5, cluster=None, cb=None)
    check("G rỗng, không cluster -> conflict, hộp None", cs == "conflict" and bx == [None] * 4)

    # --- khoá cột QĐ-01 từ qd01_cells.csv ---
    import tempfile, os
    fd, tmp = tempfile.mkstemp(suffix=".csv"); os.close(fd)
    _P(tmp).write_text("book,page,column,nom_idx,syllable\nstt2,page_0001,3,4,người\n"
                       "stt2,page_0001,3,7,người\nstt11,page_0009,1,0,người\n", encoding="utf-8")
    lk = bd.load_locked_columns(tmp); os.unlink(tmp)
    check("load_locked_columns: {(book,page): {column}}",
          lk == {("stt2", "page_0001"): {3}, ("stt11", "page_0009"): {1}}, dict(lk))
    check("load_locked_columns(None) rỗng", bd.load_locked_columns(None) == {})
    src_ap = inspect.getsource(AP.align_page)
    check("align_page: cột trong locked_columns -> box_rule 'legacy_locked_col' (trọn cột)",
          'col_rule = "legacy_locked_col"' in src_ap and "line_id in locked_columns" in src_ap)

    # --- legacy tái lập bbox bộ v3 trên 5 trang thật (cần detector + prepared/ + v3) ---
    import csv, glob, json
    # Mốc so là bản TÁI LẬP HEAD (N0c, hộp luật cũ, sha fad8f6fafad79b7a…) chứ KHÔNG phải
    # dataset_out_v3/labels.csv: từ S12 (A-13) thư mục ấy chứa bản build v3 (syl_index) nên
    # hộp ngoài cột khoá khác legacy là ĐÚNG thiết kế. Tệp mốc do S2 sinh, S12 giữ lại dưới tên này.
    ref_csv = REPO / "dataset_out_v3" / "labels_HEAD_N0c.csv"
    ckpt = REPO / "train_crop" / "detector_r34.best.pt"
    if not (ref_csv.exists() and ckpt.exists() and (REPO / "prepared").exists()):
        print("  skip legacy 5 trang: thiếu dataset_out_v3/labels_HEAD_N0c.csv (bản tái lập HEAD N0c), "
              "train_crop/detector_r34.best.pt hoặc prepared/")
        return
    from pipeline.step0_setup import load_config
    from core.text.dictionary import load_qn_to_nom, load_similarity_dict
    cfg = load_config(str(REPO / "config" / "pipeline.yaml")); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    sim = load_similarity_dict(str(REPO / paths["similar_dict"]))
    root = REPO / paths["data_dir"]
    pages = []
    for b, k in (("SachThanhTruyen2", 2), ("SachThanhTruyen4", 2), ("SachThanhTruyen11", 1)):
        ts = [t for t in sorted(glob.glob(str(root / b / "transcriptions" / "page_*.json")))
              if not t.endswith("_qn_ocr_cache.json")]
        pages += [(b, _P(t).stem) for t in ts[:k]]
    want = {(bd._book_code(b), pg) for b, pg in pages}
    ref = {}
    with open(ref_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r["book"], r["page"]) in want:
                ref[(r["book"], r["page"], int(r["column"]), int(r["nom_idx"]))] = json.loads(r["bbox"])
    got, got_new = {}, {}
    for b, pg in pages:
        rec = AP.align_page(pg, root / b, set(qn), qn, sim, "new", reseg_mode="detector", box_rule="legacy")
        for p in rec["pairs"]:
            got[(bd._book_code(b), pg, p["column"], p["nom_idx"])] = p["bbox"]
        check(f"legacy {pg}: mọi cặp box_source='legacy', count_source='legacy'",
              all(p["box_source"] == "legacy" and p["count_source"] == "legacy" for p in rec["pairs"]))
        rec2 = AP.align_page(pg, root / b, set(qn), qn, sim, "new", reseg_mode="detector",
                             box_rule="syl_index", locked_columns={rec["pairs"][0]["column"]})
        for p in rec2["pairs"]:
            got_new[(bd._book_code(b), pg, p["column"], p["nom_idx"])] = p
    common = set(ref) & set(got)
    n_eq = sum(1 for k in common if list(ref[k]) == list(got[k]))
    check(f"--box-rule legacy tái lập bbox HEAD N0c 100% trên 5 trang ({n_eq}/{len(common)} ô chung)",
          len(common) > 500 and n_eq == len(common), f"chỉ ref {len(set(ref) - common)}, chỉ mới {len(set(got) - common)}")
    lock_keys = {k for k, p in got_new.items() if p["box_source"] == "legacy_locked_col"}
    check("syl_index + cột khoá: cột khoá trọn gói legacy_locked_col, bbox == legacy",
          lock_keys and all(list(got_new[k]["bbox"]) == list(got[k]) for k in lock_keys if k in got),
          len(lock_keys))
    check("syl_index + cột khoá: count_source == 'legacy_locked_col' (enum FLOW §172, không phải 'legacy')",
          lock_keys and all(got_new[k]["count_source"] == "legacy_locked_col" for k in lock_keys),
          {got_new[k]["count_source"] for k in lock_keys})
    cs_vals = {p["count_source"] for k, p in got_new.items() if k not in lock_keys}
    check("syl_index: count_source ∈ {equal_qn, equal_ocr, conflict} ở cột không khoá, có equal_qn",
          cs_vals <= {"equal_qn", "equal_ocr", "conflict"} and "equal_qn" in cs_vals, cs_vals)
    # hộp trang thr 0,3 == lọc score >= 0,3 từ lần chạy thr 0,2
    import cv2
    d02, d03 = AP._get_detector(thr=0.2), AP._get_detector(thr=0.3)
    check("cache detector theo thr: hai đối tượng khác nhau, thr đúng",
          d02 is not d03 and d02.thr == 0.2 and d03.thr == 0.3)
    same = 0
    for b, pg in pages:
        img = cv2.imread(str(root / b / "pages" / f"{pg}.png"))
        b02, b03 = d02.boxes_for_page(img), d03.boxes_for_page(img)
        filt = AP._legacy_page_boxes(b02, 0.2)
        same += (len(filt) == len(b03)
                 and all(abs(x - y) < 1e-6 for p1, p2 in zip(filt, b03) for x, y in zip(p1, p2)))
    check("hộp trang lọc >= 0,3 từ thr 0,2 == detector thr 0,3 (5/5 trang)", same == len(pages), same)


def test_pass1c_a7():
    """A-7 (flow N5a–N5h): tier_v3 + L1 có cổng + khoá QĐ-01 + 'người' ngoài khoá."""
    from pathlib import Path as _P
    from pipeline.align_engine import build_dataset as bd
    from pipeline.align_engine import tier_v3 as tv3
    print("[A-7 — PASS 1c: tier_v3, L1 có cổng, khoá QĐ-01, người ngoài khoá]")
    NG = "\U0002029A"
    qn = {"một": ["一"], "hai": ["二"], "ba": ["三"], "bốn": ["四"], "năm": ["五"],
          "người": ["㝵", NG], "chúa": ["主"], "chứa": ["貯"]}
    sim = {"弍": ["二"], "𠄩": ["二"]}
    # 5 trang, cùng một cột 5 ô; p3-p5 ô 2 là chữ lạ (弍) -> cầu có corpus2 LOO; p4/p5 ô 4
    # chữ 𝍠 ngoài từ điển -> bigram LOO; p1 có thêm chữ thứ 6 KHÔNG match (khe QĐ-01)
    syls = ["một", "hai", "ba", "bốn", "năm"]
    base = ["一", "二", "三", "四", "五"]
    pages = {"p1": base + ["六"], "p2": base, "p3": ["一", "弍", "三", "四", "五"],
             "p4": ["一", "弍", "三", "𝍠", "五"], "p5": ["一", "弍", "三", "𝍠", "五"]}
    cols, colmap, records = [], {}, {}
    for pg, chars in pages.items():
        ops = [{"op": "match", "ocr_char": c, "syllable": s, "nom_idx": i, "syl_idx": i}
               for i, (c, s) in enumerate(zip(chars, syls))]
        col = {"book": "b", "page": pg, "column": 1, "chars": chars, "syl": syls, "ops": ops,
               "page_png": "", "boxes": [[0, 10 * i, 10, 10 * i + 10] for i in range(len(chars))],
               "box_source": ["detector"] * len(chars), "count_source": "equal_qn",
               "n_ocr": len(chars), "n_qn": 5, "n_det": len(chars)}
        cols.append(col); colmap[("b", pg, 1)] = col
        for i, (c, s) in enumerate(zip(chars, syls)):
            r = {"book": "b", "page": pg, "column": 1, "idx": i, "page_png": "", "ocr_char": c,
                 "syllable": s, "bbox": col["boxes"][i], "tier": "REVIEW", "rule": "x", "label": "",
                 "nom_idx": i, "syl_idx": i, "syllable_raw": s, "syllable_ocr": s,
                 "p_register": 0.95 if i != 4 else 0.6, **bd.FIELDS_1C}
            records[(pg, i)] = r
    recs = list(records.values())
    C = tv3.CorpusStats(cols, qn, sim)
    cross = bd.apply_tier_v3(recs, C, colmap)
    r = records[("p1", 0)]
    check("direct + p>=0,8 -> CHAR_A / GOLD / s1_inter_s2_direct / label=ocr_char",
          (r["tier_v3"], r["tier"], r["rule"], r["label"]) == ("CHAR_A", "GOLD", "s1_inter_s2_direct", "一"))
    r = records[("p1", 4)]
    check("direct + p<0,8 -> CHAR_B / GOLD / s1_inter_s2_direct_lowp",
          (r["tier_v3"], r["rule"], r["label"]) == ("CHAR_B", "s1_inter_s2_direct_lowp", "五"))
    r = records[("p4", 1)]
    check("sim_unique + ngữ cảnh (corpus2 LOO) + p>=0,8 -> CHAR_B cầu, label = chữ cầu",
          (r["tier_v3"], r["rule"], r["label"]) == ("CHAR_B", "s1_inter_s2_similar", "二"), str(r))
    r = records[("p4", 3)]
    check("ngoài từ điển, bigram ('𝍠','五') thấy ở p5 -> SYL / SYLLABLE / syl_ctx:bigram / label ''",
          (r["tier_v3"], r["tier"], r["rule"], r["label"]) == ("SYL", "SYLLABLE", "syl_ctx:bigram", ""), str(r))
    check("context_evidence là chuỗi cờ đúng theo thứ tự cố định",
          records[("p1", 0)]["context_evidence"] == "bigram|corpus4|corpus2|direct"
          and r["context_evidence"] == "bigram", records[("p1", 0)]["context_evidence"])
    check("dict_support = |R(âm)|", records[("p1", 0)]["dict_support"] == 1 and r["dict_support"] == 1)
    # chốt is_plausible
    bad = {"book": "b", "page": "p1", "column": 1, "idx": 9, "ocr_char": "一", "syllable": "0",
           "nom_idx": 0, "syl_idx": 0, "p_register": 0.99, "tier": "REVIEW", "rule": "", "label": "", **bd.FIELDS_1C}
    bd.apply_tier_v3([bad], C, colmap)
    check("âm không hợp lý -> REVIEW not_plausible (trước tier)", (bad["tier"], bad["rule"]) == ("REVIEW", "not_plausible"))
    # LOO: một trang duy nhất có cặp -> không tự khẳng định
    only = {"book": "b", "page": "p9", "column": 1, "chars": ["丁"], "syl": ["xyz"],
            "ops": [{"op": "match", "ocr_char": "丁", "syllable": "xyz", "nom_idx": 0, "syl_idx": 0}]}
    C2 = tv3.CorpusStats(cols + [only], qn, sim)
    f = C2.feats(["丁"], ["xyz"], 0, 0, ("b", "p9"))
    check("LOO: cặp chỉ có ở trang đang xét -> corpus2 = False", not f["corpus2"] and not f["bigram"])

    # L1 có cổng 2-gram ÂM–ÂM: ô (主,'chứa') ở p6; 'chúa' là đọc âm của 主 cùng khung xương
    for k in range(6):
        cols.append({"book": "b", "page": f"a{k}", "column": 2, "chars": ["主", "三"], "syl": ["chúa", "ba"],
                     "ops": [{"op": "match", "ocr_char": "主", "syllable": "chúa", "nom_idx": 0, "syl_idx": 0},
                             {"op": "match", "ocr_char": "三", "syllable": "ba", "nom_idx": 1, "syl_idx": 1}]})
    anchors = {("主", "chúa"): (6, 6)}
    nom_to_qn = {"主": ["chúa"], "貯": ["chứa"]}
    bsp = tv3.build_bigram_syl_pages(cols)
    C3 = tv3.CorpusStats(cols, qn, sim)
    mk = lambda pg, s2: ({"book": "b", "page": pg, "column": 2, "idx": 0, "ocr_char": "主", "syllable": "chứa",
                          "syllable_raw": "chứa", "syllable_ocr": "chứa", "nom_idx": 0, "syl_idx": 0,
                          "p_register": 0.9, "tier": "REVIEW", "rule": "no_context", "label": "", **bd.FIELDS_1C},
                         {("b", pg, 2): {"book": "b", "page": pg, "column": 2, "chars": ["主", "三"], "syl": ["chứa", s2]}})
    r1, cm1 = mk("z1", "ba")          # (chúa, ba) 6 trang khác; (chứa, ba) 0 -> đổi
    out = bd.apply_am_sua_dau([r1], qn, nom_to_qn, anchors, bigram_syl_pages=bsp, colmap=cm1, corpus=C3)
    check("L1 đổi âm khi n(mới) > n(gốc): syllable 'chúa', GOLD/RULE_L1, l1_support = 6, tier_v3 CHAR_A",
          out[0] == 1 and (r1["syllable"], r1["tier"], r1["rule"], r1["l1_support"], r1["tier_v3"]) ==
          ("chúa", "GOLD", bd.RULE_L1, 6, "CHAR_A"), str(r1))
    check("L1 giữ syllable_raw/syllable_ocr", (r1["syllable_raw"], r1["syllable_ocr"]) == ("chứa", "chứa"))
    r2, cm2 = mk("z2", "xx")          # không 2-gram nào -> hoà 0/0 -> giữ gốc, l1_tie = 1
    out = bd.apply_am_sua_dau([r2], qn, nom_to_qn, anchors, bigram_syl_pages=bsp, colmap=cm2, corpus=C3)
    check("L1 hoà -> giữ gốc + l1_tie = 1", out == (0, 0, 0, 1) and r2["syllable"] == "chứa" and r2["l1_tie"] == 1 and r2["tier"] == "REVIEW")
    cols.append({"book": "b", "page": "y1", "column": 3, "chars": [], "syl": ["chứa", "xx"], "ops": []})
    cols.append({"book": "b", "page": "y2", "column": 3, "chars": [], "syl": ["chứa", "xx"], "ops": []})
    bsp2 = tv3.build_bigram_syl_pages(cols)
    r3, cm3 = mk("z3", "xx")          # (chứa, xx) 2 trang > (chúa, xx) 0 -> thua, giữ gốc
    out = bd.apply_am_sua_dau([r3], qn, nom_to_qn, anchors, bigram_syl_pages=bsp2, colmap=cm3, corpus=C3)
    check("L1 thua -> giữ gốc, l1_support < 0, l1_tie = 0", out == (0, 0, 1, 0) and r3["l1_support"] == -2 and r3["l1_tie"] == 0)
    check("L1 đường cũ (không cổng) vẫn trả (n_moi, n_nang)",
          bd.apply_am_sua_dau([mk("z4", "ba")[0]], qn, nom_to_qn, anchors) == (1, 0))

    # Khoá QĐ-01: cells theo nom_idx; p1/ô1 'hai'≠'người' -> pending; p2/ô1 quyet giu -> lock;
    # p3/ô1 quyet bo -> excluded; p3/ô3 quyet khac:四 -> lock chữ khác; p1/ô5 (khe: chữ có,
    # không match) -> record tổng hợp; p1/cột 2 không có trong build -> không khớp
    cell = lambda pg, i: {"book": "b", "page": pg, "column": "1", "nom_idx": str(i), "bbox_cu": "[1, 2, 3, 4]",
                          "prev_bbox_cu": "[0, 0, 1, 1]", "next_bbox_cu": "", "image_md5_cu": "abc"}
    cells = {("b", "p1", 1, 1): cell("p1", 1), ("b", "p2", 1, 1): cell("p2", 1), ("b", "p3", 1, 1): cell("p3", 1),
             ("b", "p3", 1, 3): cell("p3", 3), ("b", "p1", 1, 5): cell("p1", 5), ("b", "p1", 1, 4): cell("p1", 4),
             ("b", "p1", 2, 0): {**cell("p1", 0), "column": "2"}}
    for r in recs:
        r["tier_goc"] = r["rule_goc"] = ""
    records[("p1", 4)]["syllable_raw"] = "Người"      # khớp nom_idx, âm dòng QN 'người' -> lock
    dec = {("b", "p2", 1, 1): {"quyet": "giu_2029A"}, ("b", "p3", 1, 1): {"quyet": "bo"},
           ("b", "p3", 1, 3): {"quyet": "khac:四"},
           ("b", "p5", 1, 0): {"quyet": "", "bbox_cu": ""}}          # A-3 ngoài cells, chờ người
    n0 = len(recs)
    st, pend = bd.apply_qd01_lock(recs, cells, dec, colmap, {("b", p) for p in pages}, {})
    r = records[("p1", 4)]
    check("lock: label 𠊚 U+2029A, GOLD, rule qd01_cell_lock, qd01_locked=1, bbox=bbox_cu, box_source=qd01_locked, tier_goc giữ",
          (r["label"], r["tier"], r["rule"], r["qd01_locked"], r["bbox"], r["box_source"], r["tier_goc"], r["rule_goc"])
          == (NG, "GOLD", bd.RULE_QD01_LOCK, 1, [1, 2, 3, 4], "qd01_locked", "GOLD", "s1_inter_s2_direct_lowp")
          and r["_prev_bbox_cu"] == [0, 0, 1, 1] and r["_next_bbox_cu"] is None, str(r))
    r = records[("p2", 1)]
    check("quyet giu_2029A dù âm ≠ 'người' -> lock", r["qd01_locked"] == 1 and r["label"] == NG)
    r = records[("p1", 1)]
    check("khớp nom_idx nhưng âm khác -> pending: rule pending, qd01_locked=0, tier giữ, _bbox_cu",
          (r["rule"], r["qd01_locked"], r["tier"], r["_bbox_cu"]) == (bd.RULE_QD01_PENDING, 0, "GOLD", [1, 2, 3, 4]))
    r = records[("p3", 1)]
    check("quyet bo -> giữ tier_v3, qd01_excluded=1", r["qd01_excluded"] == 1 and r["tier"] == "GOLD" and r["rule"] == "s1_inter_s2_similar")
    r = records[("p3", 3)]
    check("quyet khac:四 -> label 四, GOLD, rule qd01a, khoá hộp", (r["label"], r["rule"], r["qd01_locked"]) == ("四", bd.RULE_QD01A, 1))
    khe = [x for x in recs[n0:] if x["page"] == "p1" and x["nom_idx"] == 5]
    kk = [x for x in recs[n0:] if x["column"] == 2]
    check("khe (chữ có, không match) / không khớp -> record tổng hợp REVIEW pending, idx mới, ocr_char từ cột",
          len(khe) == 1 and khe[0]["rule"] == bd.RULE_QD01_PENDING and khe[0]["tier"] == "REVIEW"
          and khe[0]["idx"] == 5 and khe[0]["ocr_char"] == "六" and khe[0]["bbox"] == [0, 50, 10, 60]
          and len(kk) == 1 and kk[0]["bbox"] == [1, 2, 3, 4] and kk[0]["idx"] == 6, str(recs[n0:]))
    check("thống kê: locked 2 + pending 3 + bo 1 + khac 1 == 7 ô trong build; khe 1, không khớp 1",
          (st["n_locked"], st["n_pending"], st["n_bo"], st["n_khac"], st["n_cells_in_build"], st["n_khe"], st["n_khong_khop"])
          == (2, 3, 1, 1, 7, 1, 1), dict(st))
    check("QĐ-01a ngoài cells chờ người -> qd01_excluded=1 + dòng pending 'qd01a_chua_quyet'",
          records[("p5", 0)]["qd01_excluded"] == 1 and st["n_excluded_ngoai_cells"] == 1
          and sum(1 for x in pend if x["ly_do"] == "qd01a_chua_quyet") == 1 and len(pend) == 4)
    # 'người' ngoài khoá
    a = {"syllable": "người", "label": "㝵", "tier": "GOLD", "rule": "s1_inter_s2_direct", "qd01_locked": 0, "qd01_excluded": 0}
    b = {"syllable": "người", "label": NG, "tier": "GOLD", "rule": "s1_inter_s2_similar", "qd01_locked": 0, "qd01_excluded": 0}
    c = {"syllable": "người", "label": "㝵", "tier": "GOLD", "rule": "x", "qd01_locked": 1, "qd01_excluded": 0}
    n_nham, n_ngoai = bd.apply_nguoi_ngoai_khoa([a, b, c])
    check("'người' ngoài khoá: nhãn 㝵 -> REVIEW lop_nham (nhãn xoá); 𠊚 giữ GOLD; ô khoá không đụng",
          (n_nham, n_ngoai) == (1, 2) and (a["tier"], a["rule"], a["label"]) == ("REVIEW", bd.RULE_LOP_NHAM, "")
          and b["tier"] == "GOLD" and a["am_da_quyet_ngoai_khoa"] == 1 and b["am_da_quyet_ngoai_khoa"] == 1
          and c["tier"] == "GOLD" and "am_da_quyet_ngoai_khoa" not in c)
    # flank_gold
    fr = [{"book": "b", "page": "q", "column": 1, "syl_idx": i, "tier_v3": t}
          for i, t in enumerate(["CHAR_A", "SYL", "CHAR_A", "REVIEW"])]
    bd.compute_flank_gold(fr)
    check("flank_gold = số ô kề syl_idx±1 có tier_v3 CHAR_A", [x["flank_gold"] for x in fr] == [0, 2, 0, 1])
    src = _P(bd.__file__).read_text(encoding="utf-8")
    check("labels.csv có đủ cột A-7 (FIELDS_1C ở cuối fields)", "*FIELDS_1C.keys()]" in src
          and all(k in bd.FIELDS_1C for k in ("tier_v3", "dict_support", "context_evidence", "l1_support",
                                              "l1_tie", "flank_gold", "qd01_locked", "qd01_excluded")))
    check("PASS 1c chỉ ở --two-pass; L3/PROMOTE chỉ ở nhánh else (đường cũ)",
          "cross_v3 = apply_tier_v3(records, corpus, colmap)" in src
          and src.index("n_l3 = apply_cot_lech_cau_xuoi") > src.index("    else:\n        # ---------- L1 + L3")
          and src.index("syl_ok = syllable_gate(records, UNCONF)") > src.index("    else:\n        # ---------- L1 + L3"))
    check("build từ chối --out có .FROZEN (trừ FROZEN_OVERRIDE=GHIDE)",
          '(out / ".FROZEN").exists() and os.environ.get("FROZEN_OVERRIDE") != "GHIDE"' in src)
    check("PASS 2: ô khoá cắt bằng prev/next_bbox_cu; dọn gold/silver/syllable/review trước khi cắt",
          'r.get("_has_prev_next_cu")' in src and 'for d in ("gold", "silver", "syllable", "review"):' in src)
    # A-8 (N5f/N5h/N5i): decisions.yaml nối vào PASS 1c — corpus_readings TRƯỚC L1, lop_nham từ
    # yaml, label_canonical cho MỌI ô (cả hai đường), decisions_report.json; chi tiết ở decisions_selftest
    check("A-8: --decisions mặc định config/decisions.yaml; nạp NGAY ĐẦU main (trước align)",
          '"--decisions", default=str(REPO / "config" / "decisions.yaml")' in src
          and src.index("qdl = dcs.load(args.decisions)") < src.index("[align]"))
    check("A-8: corpus_readings áp SAU tier_v3, TRƯỚC L1 (L1 bỏ qua GOLD); lop_nham đọc từ yaml",
          src.index("cross_v3 = apply_tier_v3(records, corpus, colmap)")
          < src.index("n_cr = dcs.apply_corpus_readings(records, qdl)")
          < src.index("anchors = gold_direct_anchors(records)\n        bigram_syl_pages")
          and "lop_nham=(qdl.ap_lop_nham() if qdl is not None else [])" in src)
    check("A-8: label_canonical = apply_di_the cho MỌI ô (ngoài nhánh two_pass) + cột trong fields + report",
          src.index("n_dt = dcs.apply_di_the(records, qdl)") > src.index("# label_level + unicode")
          and '"label_canonical",\n              *FIELDS_1C.keys()]' in src
          and 'out / "decisions_report.json"' in src)
    ln = [{"syllable": "người", "label": "㝵", "tier": "GOLD", "rule": "r", "qd01_locked": 0, "qd01_excluded": 0}]
    bd.apply_nguoi_ngoai_khoa(ln, lop_nham=[{"id": "x1", "syllable": "Người", "ocr": "㝵", "den": "REVIEW"}])
    check("apply_nguoi_ngoai_khoa nhận lop_nham từ yaml -> rule lop_nham:<id>",
          (ln[0]["tier"], ln[0]["rule"], ln[0]["label"]) == ("REVIEW", "lop_nham:x1", ""))
    ln = [{"syllable": "người", "label": "㝵", "tier": "GOLD", "rule": "r", "qd01_locked": 0, "qd01_excluded": 0}]
    check("lop_nham rỗng (--decisions none) -> không hạ ô nào, vẫn cờ am_da_quyet_ngoai_khoa",
          bd.apply_nguoi_ngoai_khoa(ln, lop_nham=[]) == (0, 1) and ln[0]["tier"] == "GOLD"
          and ln[0]["am_da_quyet_ngoai_khoa"] == 1)


# --------------------------------------------------------------------------- #
def _infer_centernet():
    """Nạp train_crop/infer_centernet đúng cách engine nạp (detector_infer._load_centernet)
    để test enforce_count trên CHÍNH hàm sản xuất; None (có ghi rõ) nếu thiếu torch/train_crop."""
    import sys
    try:
        from pipeline.align_engine.char_detector import detector_infer as di
        di._load_centernet()
        return sys.modules["infer_centernet"]
    except Exception as e:  # noqa: BLE001 — thiếu torch/ckpt là lý do bỏ qua, không phải lỗi test
        print(f"  skip: không nạp được train_crop/infer_centernet ({e})")
        return None


def test_enforce_count():
    """A-15 (flow N0e): enforce_count M>N / M<N / rỗng / gray None — hàm thuần
    train_crop/infer_centernet.enforce_count mà DetectorInfer.enforce_count (nhánh (c)
    của assign_boxes) và column_boxes (đường cũ trọn gói) cùng gọi."""
    import numpy as np
    print("[enforce_count — M>N / M<N / rỗng / gray None / NMS]")
    ic = _infer_centernet()
    if ic is None:
        return
    ec = ic.enforce_count
    B = lambda y1, y2, s=0.9: [0, y1, 20, y2, s]

    check("n <= 0 -> []", ec([B(0, 10)], 0) == [] and ec([B(0, 10)], -1) == [])
    check("rỗng + gray None -> [] (không gieo hộp được)", ec([], 3) == [])
    gray = np.full((100, 20), 255, np.uint8)
    out = ec([], 2, gray_image=gray)
    check("rỗng + gray -> gieo 1 hộp bao cả ảnh rồi tách về n=2 liền mạch, phủ [0,100]",
          len(out) == 2 and out[0][1] == 0 and out[0][3] == out[1][1] and out[1][3] == 100, out)
    out = ec([[0, 30, 20, 40], [0, 0, 20, 10], B(15, 25, 0.5)], 3)
    check("M == N: giữ nguyên, sắp theo y, hộp 4-int nhận score 1,0",
          [b[:4] for b in out] == [[0, 0, 20, 10], [0, 15, 20, 25], [0, 30, 20, 40]]
          and out[0][4] == 1.0 and out[1][4] == 0.5, out)
    out = ec([B(0, 10, 0.9), B(15, 25, 0.2), B(30, 40, 0.95), B(45, 55, 0.3)], 2)
    check("M > N: giữ top-N điểm (0,95 + 0,9) rồi sắp theo y (không theo điểm)",
          [b[:4] for b in out] == [[0, 0, 20, 10], [0, 30, 20, 40]], out)
    out = ec([B(0, 10, 0.9), B(20, 60, 0.7)], 3)
    check("M < N, gray None: hộp CAO NHẤT [20,60] bổ đôi tại giữa 40, hai nửa giữ score 0,7",
          [b[:4] for b in out] == [[0, 0, 20, 10], [0, 20, 20, 40], [0, 40, 20, 60]]
          and out[1][4] == out[2][4] == 0.7, out)
    out = ec([B(0, 60)], 4)
    check("M=1 < N=4, gray None: tách lặp hộp cao nhất -> 4 hộp 15px liền mạch phủ [0,60]",
          len(out) == 4 and out[0][1] == 0 and out[-1][3] == 60
          and all(out[k][3] == out[k + 1][1] for k in range(3))
          and [b[3] - b[1] for b in out] == [15, 15, 15, 15], out)
    # có gray: seam/valley cắt vào KẼ TRẮNG giữa hai chữ, không phải giữa hộp
    g = np.full((80, 20), 255, np.uint8)
    g[5:20, 2:18] = 0            # chữ trên: mực hàng 5..19
    g[26:75, 2:18] = 0           # chữ dưới: mực hàng 26..74 (giữa hộp = 40 nằm TRONG mực)
    for m in ("seam", "valley"):
        out = ec([B(0, 80)], 2, gray_image=g, split_method=m)
        cut = out[0][3]
        check(f"M < N, gray + {m}: cắt trong kẽ trắng [20,26] (giữa hộp 40 nằm trong mực)",
              len(out) == 2 and 20 <= cut <= 26 and out[1][1] == cut and out[1][3] == 80, out)
    out = ec([B(0, 80)], 2, gray_image=g, split_method="midpoint")
    check("M < N, gray + midpoint: cắt đúng giữa hộp 40 (bỏ qua mực)", out[0][3] == 40, out)
    out = ec([B(0, 20, 0.6), B(2, 22, 0.9), B(40, 60, 0.8)], 2)
    check("NMS trước hoà giải: 2 hộp chồng y-IoU>0,45 gộp còn 1 (giữ 0,9) -> M=2 == N",
          [b[:4] for b in out] == [[0, 2, 20, 22], [0, 40, 20, 60]], out)
    out = ec([B(0, 20, 0.6), B(2, 22, 0.9)], 2)
    check("NMS gộp xong M=1 < N=2 -> tách hộp còn lại, KHÔNG trả hộp trùng",
          [b[:4] for b in out] == [[0, 2, 20, 12], [0, 12, 20, 22]], out)
    out = ec([B(0, 6)], 2)
    check("điểm cắt clamp ∈ [y1+2, y2−2]: hộp cao 6 -> [0,3] + [3,6]",
          [b[:4] for b in out] == [[0, 0, 20, 3], [0, 3, 20, 6]], out)
    check("tất định", ec([B(0, 60)], 4) == ec([B(0, 60)], 4))
    src = [B(0, 60)]; ec(src, 3)
    check("không sửa đầu vào tại chỗ", src == [B(0, 60)], src)


def test_pair_new_three_branches():
    """A-15 (flow N0e; N3f/N4d): _pair_new_state nối det.raw_column_boxes -> assign_boxes
    đúng 3 nhánh trên CỘT GIẢ với detector giả (duck-type): (a) |G|==n_qn -> G[syl_idx],
    (b) |G|==n_ocr -> G[nom_idx], (c) còn lại -> enforce_count (chỉ gọi ở nhánh này) +
    _monotone_assign; đường legacy trọn gói đi _pick_reseg với ±0,5w; _pair_new giữ
    (out, n_gap)."""
    from pipeline.align_engine import align_production as AP
    print("[_pair_new_state — 3 nhánh hộp trên cột giả + legacy + _pair_new]")
    ic = _infer_centernet()
    if ic is None:
        return

    class _FakeDet:
        """raw_column_boxes trả G cố định (ghi x_margin nhận được); enforce_count = hàm
        sản xuất với gray None (cắt giữa); column_boxes (đường cũ) = enforce_count(G, n)."""
        def __init__(self, G):
            self.G = [list(g) for g in G]
            self.n_enforce = 0
            self.margins = []

        def raw_column_boxes(self, page_boxes, x_range, x_margin=0.25):
            self.margins.append(x_margin)
            return [list(g) for g in self.G]

        def enforce_count(self, boxes, n):
            self.n_enforce += 1
            out = ic.enforce_count([list(b) for b in boxes], n, gray_image=None)
            return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in out]

        def column_boxes(self, page_boxes, x_range, n, x_margin=0.5):
            self.margins.append(x_margin)
            return self.enforce_count(self.G, n)

    qn = {"thiên": ["天"], "địa": ["地"], "nhân": ["人"], "sơn": ["山"],
          "thuỷ": ["水"], "hoả": ["火"], "mộc": ["木"], "kim": ["金"]}
    chars8 = ["天", "地", "人", "山", "水", "火", "木", "金"]
    syls8 = ["thiên", "địa", "nhân", "sơn", "thuỷ", "hoả", "mộc", "kim"]
    G8 = [[0, 100 * k + 5, 100, 100 * k + 95, 0.9] for k in range(8)]
    pb = [[0, 0, 1, 1, 0.5]]                     # page_boxes: chỉ cần khác None

    def cluster(chars):
        return {"chars": [{"char": c, "bbox": [0, 100 * k, 100, 100 * k + 100]}
                          for k, c in enumerate(chars)], "x_range": (0, 100)}

    def run(chars, syls, det, rule="syl_index", reseg=True):
        return AP._pair_new_state(cluster(chars), syls, qn, {}, reseg=reseg, reseg_mode="detector",
                                  det=det, page_boxes=pb, box_rule=rule, legacy_page_boxes=pb)

    pairs = lambda out: [(p["nom_idx"], p["syl_idx"]) for p in out]

    # (a) rụng 1 chữ OCR (水): n_ocr 7, n_qn 8, |G| 8 -> hộp theo SYL_IDX, khe đặt đúng chỗ
    chars7 = chars8[:4] + chars8[5:]
    det = _FakeDet(G8)
    out, n_gap, ops, rb, info = run(chars7, syls8, det)
    check("(a) |G|==n_qn: pairs (i, i<4 ? i : i+1), n_gap 1, count_source equal_qn",
          pairs(out) == [(i, i if i < 4 else i + 1) for i in range(7)] and n_gap == 1
          and info["count_source"] == "equal_qn", (pairs(out), n_gap, info["count_source"]))
    check("(a) bbox = G[syl_idx] (chữ sau khe nhận hộp j=i+1), box_source detector",
          [p["bbox"] for p in out] == [G8[j][:4] for j in (0, 1, 2, 3, 5, 6, 7)]
          and all(p["box_source"] == "detector" for p in out), [p["bbox"] for p in out])
    check("(a) KHÔNG gọi enforce_count, cb None, n_det 8, G == hộp thô, x_margin = DETECTOR_XMARGIN",
          det.n_enforce == 0 and info["cb"] is None and info["n_det"] == 8 and info["G"] == G8
          and det.margins == [AP.DETECTOR_XMARGIN], (det.n_enforce, info["n_det"], det.margins))
    check("(a) record ghi n_ocr/n_qn/n_det = 7/8/8, reseg_boxes theo nom_idx đủ 7",
          all((p["n_ocr"], p["n_qn"], p["n_det"]) == (7, 8, 8) for p in out)
          and rb == [G8[j][:4] for j in (0, 1, 2, 3, 5, 6, 7)], rb)

    # (b) rụng 1 âm (thuỷ): n_ocr 8, n_qn 7, |G| 8 -> hộp theo NOM_IDX, kể cả chữ del
    syls7 = syls8[:4] + syls8[5:]
    det = _FakeDet(G8)
    out, n_gap, ops, rb, info = run(chars8, syls7, det)
    check("(b) |G|==n_ocr!=n_qn: pairs (i<4 ? i : i, j) bỏ nom 4, del 1, n_gap 0, equal_ocr",
          pairs(out) == [(0, 0), (1, 1), (2, 2), (3, 3), (5, 4), (6, 5), (7, 6)] and n_gap == 0
          and [o["op"] for o in ops].count("del") == 1 and info["count_source"] == "equal_ocr",
          (pairs(out), n_gap, info["count_source"]))
    check("(b) bbox = G[nom_idx]; reseg_boxes đủ 8 (cả chữ del giữ hộp); không enforce_count",
          [p["bbox"] for p in out] == [G8[i][:4] for i in (0, 1, 2, 3, 5, 6, 7)]
          and rb == [g[:4] for g in G8] and det.n_enforce == 0 and info["cb"] is None, rb)

    # (c) chữ 3+4 dính thành 1 hộp cao: |G| 7 != 8/8 -> enforce_count 1 lần, bổ đôi 'split'
    G7 = G8[:3] + [[0, 305, 100, 495, 0.8]] + G8[5:]
    det = _FakeDet(G7)
    out, n_gap, ops, rb, info = run(chars8, syls8, det)
    check("(c) conflict: enforce_count gọi ĐÚNG 1 lần với n_qn, cb 8 hộp, n_det 7",
          info["count_source"] == "conflict" and det.n_enforce == 1 and len(info["cb"]) == 8
          and info["n_det"] == 7, (det.n_enforce, info["n_det"]))
    check("(c) 8 cặp chéo, hộp 3/4 = hai nửa bổ đôi tại 400 ghi 'split', 6 hộp kia 'detector'",
          pairs(out) == [(i, i) for i in range(8)]
          and [p["box_source"] for p in out] == ["detector"] * 3 + ["split"] * 2 + ["detector"] * 3
          and out[3]["bbox"] == [0, 305, 100, 400] and out[4]["bbox"] == [0, 400, 100, 495],
          [(p["bbox"], p["box_source"]) for p in out])
    check("(c) hộp không bổ đôi == hộp thô G (không xê dịch)",
          all(out[i]["bbox"] == G7[k][:4] for i, k in ((0, 0), (1, 1), (2, 2), (5, 4), (6, 5), (7, 6))))
    det = _FakeDet([])
    out, n_gap, ops, rb, info = run(chars8, syls8, det)
    check("(c) G rỗng: enforce_count -> [] -> rơi về midpoint cả cột, n_det 0, vẫn 8 cặp",
          info["count_source"] == "conflict" and det.n_enforce == 1 and info["n_det"] == 0
          and [p["box_source"] for p in out] == ["midpoint"] * 8 and len(out) == 8
          and rb == AP._reseg_column(cluster(chars8)), [p["box_source"] for p in out])

    # legacy / legacy_locked_col: trọn gói _pick_reseg (column_boxes ±0,5w), không qua assign_boxes
    for rule in ("legacy", "legacy_locked_col"):
        det = _FakeDet(G7)
        out, n_gap, ops, rb, info = run(chars8, syls8, det, rule=rule)
        check(f"{rule}: count_source == box_source == '{rule}', G/cb None, n_det theo ±0,5w = 7",
              info["count_source"] == rule and all(p["box_source"] == rule for p in out)
              and info["G"] is None and info["cb"] is None and info["n_det"] == 7
              and info["box_rule"] == rule, (info["count_source"], info["n_det"], info["box_rule"]))
        check(f"{rule}: column_boxes gọi với x_margin LEGACY 0,5 và raw_column_boxes ±0,5w cho n_det",
              det.margins == [AP.LEGACY_DETECTOR_XMARGIN] * 2 and det.n_enforce == 1, det.margins)
        check(f"{rule}: hộp 3/4 vẫn là hai nửa bổ đôi (cùng enforce_count), 8 hộp",
              out[3]["bbox"] == [0, 305, 100, 400] and out[4]["bbox"] == [0, 400, 100, 495]
              and len(rb) == 8)

    # reseg=False -> bbox OCR; det=None + midpoint -> mid, box_source 'midpoint', count_source ''
    out, n_gap, ops, rb, info = run(chars8, syls8, _FakeDet(G8), reseg=False)
    check("reseg=False: reseg_boxes None, bbox = hộp OCR, box_source '' , count_source ''",
          rb is None and [p["bbox"] for p in out] == [[0, 100 * k, 100, 100 * k + 100] for k in range(8)]
          and all(p["box_source"] == "" for p in out) and info["count_source"] == "")
    out, n_gap, ops, rb, info = AP._pair_new_state(cluster(chars8), syls8, qn, {}, reseg=True,
                                                   reseg_mode="midpoint")
    check("det=None + midpoint: hộp trung điểm, box_source 'midpoint', box_rule 'midpoint', n_det ''",
          rb == AP._reseg_column(cluster(chars8)) and all(p["box_source"] == "midpoint" for p in out)
          and info["box_rule"] == "midpoint" and info["n_det"] == "" and info["count_source"] == "")
    # _pair_new giữ chữ ký cũ và trả đúng (out, n_gap) của _pair_new_state
    det1, det2 = _FakeDet(G8), _FakeDet(G8)
    st = AP._pair_new_state(cluster(chars7), syls8, qn, {}, reseg=True, reseg_mode="detector",
                            det=det1, page_boxes=pb)
    old = AP._pair_new(cluster(chars7), syls8, qn, {}, reseg=True, reseg_mode="detector",
                       det=det2, page_boxes=pb)
    check("_pair_new == _pair_new_state[:2] (cùng cột, cùng detector giả)", old == st[:2])


def test_tier_v3_branches():
    """A-15 (flow N0e; N5b–N5d): bảng chân trị tier_v3 NGUYÊN VĂN từng nhánh + rule_of +
    TIER_OF + context_evidence; chốt is_plausible đứng TRƯỚC tier trong apply_tier_v3
    (tier_v3_selftest so feats với lab trên cols.pkl; đây là từng nhánh trên cờ giả)."""
    from pipeline.align_engine import tier_v3 as tv3
    from pipeline.align_engine import build_dataset as bd
    from core.text.text_utils import is_plausible_qn_syllable
    print("[tier_v3 — từng nhánh trên cờ giả + not_plausible]")
    check("P_HI 0,8 / P_MID 0,5; TIER_OF 4 hạng -> GOLD/GOLD/SYLLABLE/REVIEW",
          (tv3.P_HI, tv3.P_MID) == (0.8, 0.5)
          and tv3.TIER_OF == {"CHAR_A": "GOLD", "CHAR_B": "GOLD", "SYL": "SYLLABLE", "REVIEW": "REVIEW"})
    Z = dict(direct=False, sim_unique=False, tone=False, corpus2=False, corpus4=False, bigram=False,
             dict_support=0)
    F = lambda **kw: {**Z, **kw}
    cases = [
        # (cờ, p, tier_v3, rule) — mỗi dòng một nhánh của tier_v3()/rule_of()
        (F(direct=True), 0.8, "CHAR_A", "s1_inter_s2_direct"),
        (F(direct=True), 0.79, "CHAR_B", "s1_inter_s2_direct_lowp"),
        (F(direct=True), 0.0, "CHAR_B", "s1_inter_s2_direct_lowp"),
        (F(direct=True, sim_unique=True, bigram=True), 0.9, "CHAR_A", "s1_inter_s2_direct"),
        (F(sim_unique=True, corpus2=True), 0.8, "CHAR_B", "s1_inter_s2_similar"),
        (F(sim_unique=True, bigram=True), 0.9, "CHAR_B", "s1_inter_s2_similar"),
        (F(sim_unique=True, tone=True), 0.9, "CHAR_B", "s1_inter_s2_similar"),
        (F(sim_unique=True), 0.99, "REVIEW", "no_context"),
        (F(sim_unique=True, corpus2=True), 0.79, "REVIEW", "no_context"),
        (F(sim_unique=True, corpus2=True, corpus4=True), 0.79, "SYL", "syl_ctx:corpus4"),
        (F(bigram=True), 0.5, "SYL", "syl_ctx:bigram"),
        (F(corpus2=True, corpus4=True), 0.5, "SYL", "syl_ctx:corpus4"),
        (F(tone=True), 0.5, "SYL", "syl_ctx:tone"),
        (F(bigram=True, corpus4=True, tone=True), 0.6, "SYL", "syl_ctx:bigram"),
        (F(corpus4=True, tone=True), 0.6, "SYL", "syl_ctx:corpus4"),
        (F(bigram=True), 0.49, "REVIEW", "low_posterior"),
        (F(corpus4=True, corpus2=True), 0.0, "REVIEW", "low_posterior"),
        (F(corpus2=True), 0.99, "REVIEW", "no_context"),
        (F(), 0.99, "REVIEW", "no_context"),
    ]
    for f, p, want_t, want_r in cases:
        t = tv3.tier_v3(f, p)
        on = "|".join(k for k in ("direct", "sim_unique", "tone", "corpus2", "corpus4", "bigram") if f[k]) or "∅"
        check(f"tier_v3({on}, p={p}) -> {want_t} / {want_r}",
              t == want_t and tv3.rule_of(t, f, p) == want_r, (t, tv3.rule_of(t, f, p)))
    check("context_evidence theo thứ tự CỐ ĐỊNH bigram|corpus4|tone|corpus2|direct|sim_unique",
          tv3.context_evidence(F(direct=True, sim_unique=True, tone=True, corpus2=True, corpus4=True, bigram=True))
          == "bigram|corpus4|tone|corpus2|direct|sim_unique"
          and tv3.context_evidence(F(corpus2=True, direct=True)) == "corpus2|direct"
          and tv3.context_evidence(F()) == "")
    check("P_HI/P_MID ghi đè theo tham số (direct p=0,7 -> CHAR_A khi P_HI=0,7; bigram p=0,3 -> SYL khi P_MID=0,3)",
          tv3.tier_v3(F(direct=True), 0.7, P_HI=0.7) == "CHAR_A"
          and tv3.tier_v3(F(bigram=True), 0.3, P_MID=0.3) == "SYL")
    # bridge_char: chỉ khi sim(c) ∩ R(âm) đúng 1 chữ
    C = tv3.CorpusStats([], {"hai": ["二", "弍"], "ba": ["三"]}, {"𠄩": ["二", "弍"], "弎": ["三"]})
    check("bridge_char: |sim ∩ R| == 1 -> chữ cầu; == 2 -> ''; == 0 -> ''; âm viết hoa vẫn tra",
          C.bridge_char("弎", "Ba") == "三" and C.bridge_char("𠄩", "hai") == "" and C.bridge_char("弎", "hai") == "")
    # chốt is_plausible đứng TRƯỚC tier: ô direct p=0,99 nhưng âm rác -> REVIEW not_plausible
    col = {"book": "b", "page": "p", "column": 1, "chars": ["三", "三"], "syl": ["ba", "0"],
           "ops": [{"op": "match", "ocr_char": "三", "syllable": "ba", "nom_idx": 0, "syl_idx": 0},
                   {"op": "match", "ocr_char": "三", "syllable": "0", "nom_idx": 1, "syl_idx": 1}]}
    colmap = {("b", "p", 1): col}
    mk = lambda i, s, ch="三": {"book": "b", "page": "p", "column": 1, "idx": i, "ocr_char": ch, "syllable": s,
                               "nom_idx": i, "syl_idx": i, "p_register": 0.99, "tier": "GOLD", "rule": "x",
                               "label": "三", **bd.FIELDS_1C}
    ok, bad, noch = mk(0, "ba"), mk(1, "0"), mk(0, "ba", ch="")
    cross = bd.apply_tier_v3([ok, bad, noch], C, colmap)
    check("is_plausible('0') False, ('ba') True", not is_plausible_qn_syllable("0") and is_plausible_qn_syllable("ba"))
    check("âm rác dù direct + p=0,99 -> REVIEW / not_plausible / label '' / tier_v3 REVIEW (chốt TRƯỚC tier)",
          (bad["tier"], bad["rule"], bad["label"], bad["tier_v3"]) == ("REVIEW", "not_plausible", "", "REVIEW"), bad)
    check("cùng cột, âm hợp lý -> CHAR_A bình thường; không ocr_char -> REVIEW no_context",
          (ok["tier_v3"], ok["rule"], ok["label"]) == ("CHAR_A", "s1_inter_s2_direct", "三")
          and (noch["tier_v3"], noch["rule"], noch["label"]) == ("REVIEW", "no_context", ""))
    check("crosstab đếm theo tier_v3 (kể cả ô bị chốt)", dict(cross) == {"CHAR_A": 1, "REVIEW": 2}, dict(cross))
    r0 = dict(mk(0, "ba"), p_register="")
    bd.apply_tier_v3([r0], C, colmap)
    check("p_register '' -> coi là 0 -> direct thành CHAR_B lowp (không CHAR_A)",
          (r0["tier_v3"], r0["rule"]) == ("CHAR_B", "s1_inter_s2_direct_lowp"), (r0["tier_v3"], r0["rule"]))


def main() -> int:
    print("=" * 64)
    print("PHASE-1 ENGINE-FIX SELFTEST")
    print("=" * 64)
    test_monotone_assign()
    test_syllable_gate()
    test_ocr_retry()
    test_ocr_guest_mode()
    test_ocr_cache_guard()
    test_proto_cache_not_poisoned()
    test_crop_geometry_wired()
    test_labels_sorted()
    test_anchor_align_calib()
    test_two_pass_pass1b()
    test_box_rule_a6()
    test_pass1c_a7()
    test_enforce_count()
    test_pair_new_three_branches()
    test_tier_v3_branches()
    print("=" * 64)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 64)
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
