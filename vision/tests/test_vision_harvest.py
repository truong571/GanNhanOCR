#!/usr/bin/env python3
"""Kiểm thử OFFLINE vision_harvest.py (0 yêu cầu thật): bộ "OCR" giả đọc ẢNH GHÉP THẬT (tìm ô vuông đen bằng thành phần liên thông) rồi trả JSON kiểu Vision,
nên phép thử kiểm cả chuỗi: co/giãn → trần điểm ảnh → xếp hàng/lưới → JPEG → toạ độ ảnh ghép → quy về toạ độ trang gốc; cùng sổ cái/khoá tệp/bộ ngắt/khoá API.

  .venv/bin/python vision/tests/test_vision_harvest.py
"""
from __future__ import annotations

import base64
import csv
import http.client
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
CODE = HERE.parent                      # .../GanNhanOCR/vision
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


def load_vh(env):
    for k, v in env.items():
        os.environ[k] = str(v)
    name = f"vh_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(name, CODE / "vision_harvest.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m          # dataclass cần module đã đăng ký khi chú thích ở dạng chuỗi
    spec.loader.exec_module(m)
    return m


# ───────── kho giả ─────────
SQ = {"A": (200, 280, [(20, 30, 30), (120, 100, 30), (60, 220, 30)], 40),        # (rộng, cao, [(x,y,cạnh)], cao ô) → scale 80/40 = 2,0
      "B": (800, 1100, [(100, 100, 80), (500, 400, 80), (300, 900, 80)], 160)}   # → scale 0,5


def write_labels(root: Path, bdir: str, n: int, cell_h: int):
    with open(root / "dataset" / bdir / "labels.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page", "bbox"])
        for i in range(1, n + 1):
            w.writerow([f"page_{i:04d}", json.dumps([10, 10, 10 + cell_h, 10 + cell_h])])


def make_repo(root: Path, na=9, nb=5):
    for key, bdir, n in (("A", "BkA", na), ("B", "BkB", nb)):
        w, h, sq, ch = SQ[key]
        (root / "prepared" / bdir / "pages_denoised").mkdir(parents=True)
        (root / "dataset" / bdir).mkdir(parents=True)
        for i in range(1, n + 1):
            im = Image.new("L", (w, h), 255)
            px = im.load()
            for x, y, c in sq:
                for yy in range(y, y + c):
                    for xx in range(x, x + c):
                        px[xx, yy] = 0
            im.save(root / "prepared" / bdir / "pages_denoised" / f"page_{i:04d}.png")
        write_labels(root, bdir, n, ch)
    # trang orphan (có ảnh, KHÔNG có ô) ở GIỮA danh sách sắp xếp của sách A: page_0005b
    Image.new("L", (200, 280), 255).save(root / "prepared" / "BkA" / "pages_denoised" / "page_0005b.png")


def fake_ocr_transport(calls, mutate=None, halve=False, shift_dims=None):
    """Nhận payload JSON, đọc ảnh, trả ký hiệu = mỗi ô vuông đen. mutate(call_index) có thể ép lỗi/trạng thái."""
    def transport(body: bytes):
        calls.append(1)
        if mutate:
            r = mutate(len(calls))
            if r is not None:
                return r
        req = json.loads(body.decode())["requests"][0]
        im = Image.open(io.BytesIO(base64.b64decode(req["image"]["content"]))).convert("L")
        a = np.array(im)
        lab, n = ndimage.label(a < 128)
        syms = []
        f = 0.5 if halve else 1.0
        for k, sl in enumerate(ndimage.find_objects(lab)):
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            if (y1 - y0) * (x1 - x0) < 30:
                continue
            xs, ys = [int(x0 * f), int(x1 * f), int(x1 * f), int(x0 * f)], [int(y0 * f), int(y0 * f), int(y1 * f), int(y1 * f)]
            v = [({"x": xs[i]} if xs[i] else {}) | ({"y": ys[i]} if ys[i] else {}) for i in range(4)]       # Vision bỏ toạ độ 0
            syms.append({"text": chr(0x4E00 + k), "confidence": 0.9, "boundingBox": {"vertices": v}})
        W, H = (shift_dims if shift_dims else (int(im.width * f), int(im.height * f)))
        resp = {"responses": [{"fullTextAnnotation": {"text": "x", "pages": [{"width": W, "height": H, "blocks": [{"paragraphs": [{"words": [{"symbols": syms}]}]}]}]}}]}
        return 200, json.dumps(resp).encode()
    return transport


def expected_boxes(key):
    w, h, sq, ch = SQ[key]
    return sorted((x, y, x + c, y + c) for x, y, c in sq)


def page_boxes(store, rec):
    d = json.loads(store.page(rec.book, rec.page).read_text(encoding="utf-8"))
    return sorted((s[2], s[3], s[4], s[5]) for s in d["sym"]), d


def match(got, exp, tol):
    return len(got) == len(exp) and all(max(abs(a - b) for a, b in zip(g, e)) <= tol for g, e in zip(got, exp))


def geometry_ok(store, pages):
    bad = []
    for k, ps in pages.items():
        for rec in ps:
            got, d = page_boxes(store, rec)
            tol = 1.6 / min(d["sx"], d["sy"]) + 1.5
            if not match(got, expected_boxes(k), tol) or d["dropped_in_gap"]:
                bad.append((k, rec.page, got[:2], d["dropped_in_gap"]))
    return bad


def main():
    tmp = Path(tempfile.mkdtemp(prefix="vh_test_"))
    repo, cache = tmp / "repo", tmp / "cache"
    make_repo(repo)
    env = dict(VISION_REPO_ROOT=repo, VISION_CACHE_ROOT=cache, VISION_LEDGER=tmp / "ledger.json", VISION_LEGACY_LEDGER=tmp / "none.json", VISION_LEGACY_DIR=tmp / "none")
    vh = load_vh(env)
    vh.BOOKS = [("A", "BkA"), ("B", "BkB")]
    noop = lambda s: None
    logs = []
    nolog = logs.append

    all_pages = vh.select_pages(None, None)
    pages = {k: [r for r in ps if not r.orphan] for k, ps in all_pages.items()}        # các phép thử hình học dùng trang CÓ Ô (như bản chạy thật)
    gg = vh.plan_groups(all_pages, 2)
    ga = [[r.page for r in g] for g in gg if g[0].book == "A"]
    check("T1.plan_nhom", [len(g) for g in vh.plan_groups(pages, 4)] == [4, 4, 1, 4, 1] and len(vh.plan_groups(pages, 4, mix=True)) == 4,
          f"{[len(g) for g in vh.plan_groups(pages, 4)]} / trộn {[len(g) for g in vh.plan_groups(pages, 4, mix=True)]}")
    check("T1b.trang_orphan_nhom_rieng_cuoi_khong_lam_lech_cap_cu", ga[:5] == [["page_0001", "page_0002"], ["page_0003", "page_0004"], ["page_0005", "page_0006"], ["page_0007", "page_0008"],
                                                                              ["page_0009"]] and ga[-1] == ["page_0005b"] and sum(r.orphan for ps in all_pages.values() for r in ps) == 1, f"{ga}")

    # T2: hình học đầu-cuối cho lưới và hàng, hai tỉ lệ (×2,0 và ×0,5) — dùng run_groups
    for layout in ("grid", "row"):
        cfg = vh.Cfg(n=4, layout=layout, hints=("zh-Hant",), cell_h=80.0, cap_mp=1e9)
        led = vh.Ledger(path=cache / f"led_{layout}.json", month="2026-10")
        calls = []
        res = vh.run_groups(vh.plan_groups(pages, 4), cfg, fake_ocr_transport(calls), led, 950, sleep=noop, log=nolog)
        bad = geometry_ok(vh.Store(cfg.cid()), pages)
        check(f"T2.hinh_hoc_{layout}", not bad and res["api"] == 5 and led.used() == 5, f"api={res['api']} sổ cái={led.used()} lệch={bad[:2]}")

    # T3: trần điểm ảnh ép hạ hệ số cả nhóm mà hình học vẫn đúng
    cfgc = vh.Cfg(n=4, layout="grid", hints=("zh-Hant",), cell_h=80.0, cap_mp=0.30)
    scl = vh.get_scales(cfgc, ["BkA", "BkB"])
    dims = [vh.canvas_dims(g, cfgc, scl) for g in vh.plan_groups(pages, 4)]
    ledc = vh.Ledger(path=cache / "led_cap_mp.json", month="2026-10")
    resc = vh.run_groups(vh.plan_groups(pages, 4), cfgc, fake_ocr_transport([]), ledc, 950, sleep=noop, log=nolog)
    badc = geometry_ok(vh.Store(cfgc.cid()), pages)
    free = [vh.canvas_dims(g, vh.Cfg(n=4, layout="grid", cap_mp=1e9), scl) for g in vh.plan_groups(pages, 4)]
    check("T3.tran_diem_anh_ha_he_so_va_hinh_hoc_van_dung", all(w * h / 1e6 <= 0.30 * 1.01 for w, h, g in dims) and any(g < 0.99 for _, _, g in dims)
          and all(g == 1.0 for _, _, g in free) and not badc and resc["api"] == 5,
          f"MP={[round(w * h / 1e6, 3) for w, h, _ in dims]} g={[round(g, 2) for _, _, g in dims]} không trần g={[g for _, _, g in free]} lệch={badc[:1]}")

    # T4: chạy lại → 0 yêu cầu; T5: dựng lại từ phản hồi thô khi mất JSON trang
    cfg = vh.Cfg(n=4, layout="grid", hints=("zh-Hant",), cell_h=80.0, cap_mp=1e9)
    led = vh.Ledger(path=cache / "led_grid.json", month="2026-10")
    calls = []
    r2 = vh.run_groups(vh.plan_groups(pages, 4), cfg, fake_ocr_transport(calls), led, 950, sleep=noop, log=nolog)
    check("T4.chay_lai_0_yeu_cau", r2["api"] == 0 and r2["cached"] == 5 and not calls and led.used() == 5, f"{r2['api']} api, {r2['cached']} cached")
    store = vh.Store(cfg.cid())
    before = {str(p): p.read_bytes() for p in (store.dir / "pages").rglob("*.json")}
    for p in (store.dir / "pages").rglob("*.json"):
        p.unlink()
    r3 = vh.run_groups(vh.plan_groups(pages, 4), cfg, fake_ocr_transport(calls), led, 950, sleep=noop, log=nolog)
    after = {str(p): p.read_bytes() for p in (store.dir / "pages").rglob("*.json")}
    check("T5.dung_lai_tu_phan_hoi_tho", r3["api"] == 0 and r3["from_raw"] == 5 and before == after and led.used() == 5, f"from_raw={r3['from_raw']} giống={before == after}")

    # T6: sổ cái theo tháng + trần
    cfg2 = vh.Cfg(n=1, layout="row", hints=(), cell_h=80.0)
    led2 = vh.Ledger(path=cache / "led_cap.json", month="2026-10")
    calls = []
    r = vh.run_groups(vh.plan_groups(pages, 1), cfg2, fake_ocr_transport(calls), led2, 2, sleep=noop, log=nolog)
    check("T6a.chan_tran", r["api"] == 2 and "chạm trần" in r["stopped"] and len(calls) == 2, f"api={r['api']} dừng='{r['stopped']}' gọi={len(calls)}")
    led3 = vh.Ledger(path=cache / "led_cap.json", month="2026-11")
    check("T6b.sang_thang_moi_dem_lai", led3.used() == 0 and vh.Ledger(path=cache / "led_cap.json", month="2026-10").used() == 2, f"tháng 11 = {led3.used()}")

    cfg3 = vh.Cfg(n=2, layout="row", hints=("zh-Hant",), cell_h=80.0)
    g = vh.plan_groups(pages, 2)[0]
    sc3 = vh.get_scales(cfg3, ["BkA", "BkB"])

    # T7: lỗi theo ảnh (HTTP 200 + error) → không cache, vẫn tính 1
    led4 = vh.Ledger(path=cache / "led_err.json", month="2026-10")
    err = lambda i: (200, json.dumps({"responses": [{"error": {"code": 3, "message": "Bad image data"}}]}).encode())
    st = vh.Store(cfg3.cid() + "_t7")
    s, _, _ = vh.process_group(g, cfg3, st, led4, fake_ocr_transport([], mutate=err), 950, sc3, sleep=noop)
    check("T7.loi_tung_anh_khong_vao_cache", s.startswith("fail:") and not list((st.dir / "raw").glob("*")) and not list((st.dir / "pages").rglob("*.json")) and led4.used() == 1, s[:60])

    # T8: 429 ×2 rồi 200 → chỉ tính 1; 400 → không thử lại, không tính
    led5 = vh.Ledger(path=cache / "led_429.json", month="2026-10")
    seq = {"n": 0}

    def flaky(i):
        seq["n"] += 1
        return (429, b"{}") if seq["n"] <= 2 else None
    sl = []
    s, _, _ = vh.process_group(g, cfg3, vh.Store("t8"), led5, fake_ocr_transport([], mutate=flaky), 950, sc3, sleep=sl.append)
    check("T8a.429_thu_lai_tinh_1", s == "api" and led5.used() == 1 and len(sl) == 2, f"{s} sổ cái={led5.used()} lần ngủ={len(sl)}")
    led6 = vh.Ledger(path=cache / "led_400.json", month="2026-10")
    s, _, _ = vh.process_group(g, cfg3, vh.Store("t8b"), led6, fake_ocr_transport([], mutate=lambda i: (400, json.dumps({"error": {"message": "Request payload size exceeds"}}).encode())), 950, sc3, sleep=noop)
    check("T8b.400_khong_thu_lai_khong_tinh", s.startswith("fail:HTTP 400") and led6.used() == 0, s[:70])

    # T9: hết giờ (Ambiguous) → giữ tính phí; lỗi trước khi gửi (PreSendError) → hoàn lại
    led7 = vh.Ledger(path=cache / "led_to.json", month="2026-10")
    base = fake_ocr_transport([])
    tseq = {"n": 0}

    def transport_timeout(body):
        tseq["n"] += 1
        if tseq["n"] == 1:
            raise vh.Ambiguous("timeout")
        return base(body)
    s, _, _ = vh.process_group(g, cfg3, vh.Store("t9"), led7, transport_timeout, 950, sc3, sleep=noop)
    check("T9a.het_gio_tinh_la_da_tinh_phi", s == "api" and led7.used() == 2, f"{s} sổ cái={led7.used()}")
    led7b = vh.Ledger(path=cache / "led_pre.json", month="2026-10")

    def pre(body):
        raise vh.PreSendError("gaierror")
    s, _, _ = vh.process_group(g, cfg3, vh.Store("t9b"), led7b, pre, 950, sc3, sleep=noop)
    led7c = vh.Ledger(path=cache / "led_amb3.json", month="2026-10")

    def amb(body):
        raise vh.Ambiguous("timeout")
    s2, _, _ = vh.process_group(g, cfg3, vh.Store("t9c"), led7c, amb, 950, sc3, sleep=noop)
    check("T9b.truoc_khi_gui_khong_tinh_mo_ho_3_lan_tinh_3", s.startswith("fail:lỗi trước khi gửi") and led7b.used() == 0 and s2.startswith("fail:mất kết nối") and led7c.used() == 3,
          f"{s[:40]} used={led7b.used()} | {s2[:30]} used={led7c.used()}")

    # T10: ảnh quá lớn → không gọi
    old = vh.MAX_JSON_BYTES
    vh.MAX_JSON_BYTES = 3000
    calls = []
    s, _, _ = vh.process_group(g, cfg3, vh.Store("t10"), vh.Ledger(path=cache / "led_big.json", month="2026-10"), fake_ocr_transport(calls), 950, sc3, sleep=noop)
    vh.MAX_JSON_BYTES = old
    check("T10.qua_lon_khong_goi", s.startswith("fail:") and not calls, s[:60])

    # T11: Vision báo kích thước xử lý khác ảnh gửi (thu nhỏ một nửa, toạ độ ở khung nhỏ) → quy đổi đúng, gắn cờ 'rescaled'
    cfg4 = vh.Cfg(n=4, layout="grid", hints=("zh-Hant",), cell_h=60.0, cap_mp=1e9)
    st4 = vh.Store(cfg4.cid())
    g4 = vh.plan_groups(pages, 4)[0]
    sc4 = vh.get_scales(cfg4, ["BkA", "BkB"])
    s, ns, an = vh.process_group(g4, cfg4, st4, vh.Ledger(path=cache / "led_half.json", month="2026-10"), fake_ocr_transport([], halve=True), 950, sc4, sleep=noop)
    ok = s == "api" and "rescaled" in an
    for rec in g4:
        got, d = page_boxes(st4, rec)
        ok = ok and match(got, expected_boxes("A"), 2 * (1.6 / min(d["sx"], d["sy"]) + 1.5) + 2)
    check("T11.vision_thu_nho_noi_bo_van_dung_va_co_co", ok, f"{s} cờ='{an[:60]}'")

    # T11b: kích thước báo về KHÔNG khớp và toạ độ vượt khung báo → coi là khung ảnh gửi + cờ anomaly_dims; run dừng ở ảnh đầu tiên, pilot (stop_on_anomaly=False) đi tiếp
    cfg5 = vh.Cfg(n=4, layout="grid", hints=("zh-Hant",), cell_h=60.0, cap_mp=1e9, jpeg_q=89)
    ra = vh.run_groups(vh.plan_groups(pages, 4), cfg5, fake_ocr_transport([], shift_dims=(100, 100)), vh.Ledger(path=cache / "led_an1.json", month="2026-10"), 950, sleep=noop,
                       log=nolog, stop_on_anomaly=True)
    cfg5b = vh.Cfg(n=4, layout="grid", hints=("zh-Hant",), cell_h=60.0, cap_mp=1e9, jpeg_q=88)
    rb = vh.run_groups(vh.plan_groups(pages, 4), cfg5b, fake_ocr_transport([], shift_dims=(100, 100)), vh.Ledger(path=cache / "led_an2.json", month="2026-10"), 950, sleep=noop,
                       log=nolog, stop_on_anomaly=False)
    d0 = json.loads(vh.Store(cfg5.cid()).page("A", "page_0001").read_text(encoding="utf-8"))
    check("T11b.bat_thuong_kich_thuoc_dung_o_anh_dau_tien_pilot_di_tiep", ra["api"] == 1 and "bất thường" in ra["stopped"] and rb["api"] == 5 and d0["frame"] == "anomaly_dims",
          f"run api={ra['api']} dừng='{ra['stopped'][:40]}' | pilot api={rb['api']} frame={d0['frame']}")

    # T12: bảo vệ khoá trong repo
    r = tmp / "gitrepo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    kf = r / "vision-ocr-x.json"
    kf.write_text("{}")
    old_repo = vh.REPO
    vh.REPO = r
    try:
        try:
            vh.key_guard(str(kf))
            refused = False
        except SystemExit as e:
            refused = "TỪ CHỐI" in str(e)
        (r / ".gitignore").write_text("vision-ocr-*.json\n")
        try:
            vh.key_guard(str(kf))
            accepted = True
        except SystemExit:
            accepted = False
    finally:
        vh.REPO = old_repo
    try:
        vh.find_key(str(tmp / "khong_co.json"))
        explicit_missing = False
    except SystemExit:
        explicit_missing = True
    check("T12.khoa_trong_repo_chua_ignore_bi_tu_choi_va_--key_sai_khong_im_lang", refused and accepted and explicit_missing,
          f"từ chối={refused} chấp nhận khi đã ignore={accepted} --key thiếu→thoát={explicit_missing}")

    # T13: không in khoá/URL/tiêu đề (kiểm tĩnh nguồn) + payload + scrub + định dạng khoá API
    src = (CODE / "vision_harvest.py").read_text(encoding="utf-8")
    leak = [ln for ln in src.splitlines() if "print(" in ln and any(k in ln for k in ("auth.url", ".headers()", "api_key", "_tok", "Authorization", "private_key"))]
    p1 = json.loads(vh.build_payload(b"x", ()).decode())["requests"][0]
    p2 = json.loads(vh.build_payload(b"x", ("zh-Hant", "vi")).decode())["requests"][0]
    try:
        vh.Auth(None, "co khoang trang " + "A" * 30)
        badkey = False
    except SystemExit as e:
        badkey = "AAAA" not in str(e)
    sc = vh.scrub("lỗi https://x/y?key=AIzaSyFAKEKEY123456&z=1 Bearer ya29.abcDEF")
    check("T13.khong_ro_ri_khoa_payload_scrub_va_dinh_dang_khoa_API", not leak and "imageContext" not in p1 and p2["imageContext"]["languageHints"] == ["zh-Hant", "vi"]
          and p1["features"][0]["type"] == "DOCUMENT_TEXT_DETECTION" and badkey and "AIzaSy" not in sc and "ya29" not in sc, f"dòng nghi rò: {leak[:1]} scrub={sc}")

    # T14: nhập cache cũ THẬT — số ký hiệu mỗi trang phải trùng phân tích đã làm (tâm trong trang)
    real_repo = HERE.parents[1]
    if (CODE / "legacy" / "cache" / "google_vision" / "usage_ledger.json").exists():
        out = subprocess.run([sys.executable, "-c", (
            "import importlib.util,json,sys,os;"
            f"os.environ['VISION_CACHE_ROOT']={str(tmp / 'cache_legacy')!r};"
            f"spec=importlib.util.spec_from_file_location('vh',{str(CODE / 'vision_harvest.py')!r});m=importlib.util.module_from_spec(spec);sys.modules['vh']=m;spec.loader.exec_module(m);"
            "cid=m.import_legacy(log=lambda *a:None);"
            "d={p.stem:len(json.loads(p.read_text())['sym']) for p in (m.ROOT/cid/'pages').rglob('*.json')};"
            "print(json.dumps(d))")], capture_output=True, text=True, env={k: v for k, v in os.environ.items() if not k.startswith('VISION_')})
        want = {"stt2": 64, "stt4": 70, "stt11": 122, "L83": 104, "KVK": 140, "Chr": 32, "TK": 91, "L16": 78, "B18": 142, "B34": 124}
        try:
            got = json.loads(out.stdout.strip().splitlines()[-1])
            inv = {p: k for k, p in vh.LEGACY10}
            got_b = {inv[k]: v for k, v in got.items()}
            check("T14.nhap_cache_cu_khop_phan_tich_truoc", got_b == want, f"{got_b}")
        except Exception as e:  # noqa: BLE001
            check("T14.nhap_cache_cu_khop_phan_tich_truoc", False, f"{type(e).__name__}: {e} | {out.stderr[-200:]}")
    else:
        print("BỎ QUA T14 (không có cache cũ)")

    # T15: phát lại dùng bố cục ĐÃ LƯU dù labels.csv đổi (cao ô trung vị đổi 10 %) — toạ độ vẫn đúng và không gọi API
    cfg6 = vh.Cfg(n=2, layout="row", hints=("zh-Hant",), cell_h=80.0, cap_mp=1e9, jpeg_q=87)
    led8 = vh.Ledger(path=cache / "led_replay.json", month="2026-10")
    pa = {"A": pages["A"][:4]}
    vh.run_groups(vh.plan_groups(pa, 2), cfg6, fake_ocr_transport([]), led8, 950, sleep=noop, log=nolog)
    st6 = vh.Store(cfg6.cid())
    for p in (st6.dir / "pages").rglob("*.json"):
        p.unlink()
    vh._MED_CACHE.clear()
    write_labels(repo, "BkA", 9, 44)                              # +10 %
    calls = []
    r6 = vh.run_groups(vh.plan_groups(pa, 2), cfg6, fake_ocr_transport(calls), led8, 950, sleep=noop, log=nolog)
    bad6 = geometry_ok(st6, pa)
    write_labels(repo, "BkA", 9, 40)
    vh._MED_CACHE.clear()
    check("T15.phat_lai_dung_bo_cuc_da_luu_khi_labels_doi", r6["from_raw"] == 2 and r6["api"] == 0 and not calls and not bad6, f"from_raw={r6['from_raw']} lệch={bad6[:1]}")

    # T16: sổ cái KHOÁ TỆP — hai tiến trình cùng đặt chỗ, trần 15 → tổng đúng 15, không mất cập nhật
    led9 = cache / "led_race.json"
    code = ("import importlib.util,sys,os;os.environ['VISION_LEGACY_LEDGER']='/nonexistent';"
            f"spec=importlib.util.spec_from_file_location('vh',{str(CODE / 'vision_harvest.py')!r});m=importlib.util.module_from_spec(spec);sys.modules['vh']=m;spec.loader.exec_module(m);"
            f"L=m.Ledger(path={str(led9)!r},month='2026-10');n=0\n"
            "for i in range(10):\n    t=L.reserve('c','g%d'%i,15)\n    n+=1 if t else 0\n"
            "print(n)")
    ps = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True) for _ in range(2)]
    got = [int(p.communicate()[0].strip().splitlines()[-1]) for p in ps]
    final = vh.Ledger(path=led9, month="2026-10").used()
    check("T16.so_cai_khoa_tep_hai_tien_trinh_khong_mat_cap_nhat", sum(got) == 15 and final == 15, f"đặt chỗ được {got}, sổ cái cuối {final}")

    # T17: bộ ngắt — 3 nhóm lỗi liên tiếp thì dừng, 400 được hoàn lại nên sổ cái 0
    led10 = vh.Ledger(path=cache / "led_brk.json", month="2026-10")
    cfg7 = vh.Cfg(n=1, layout="row", hints=(), cell_h=80.0, jpeg_q=86)
    rbk = vh.run_groups(vh.plan_groups(pages, 1), cfg7, fake_ocr_transport([], mutate=lambda i: (400, b'{"error":{"message":"bad"}}')), led10, 950, sleep=noop, log=nolog)
    check("T17.bo_ngat_3_loi_lien_tiep", len(rbk["fails"]) == 3 and "lỗi liên tiếp" in rbk["stopped"] and led10.used() == 0, f"lỗi={len(rbk['fails'])} dừng='{rbk['stopped'][:40]}' sổ cái={led10.used()}")

    # T18: raw .gz hỏng không làm sập cả lượt; sổ cái chính hỏng → dùng .bak; cả hai hỏng → thoát (không tự về 0)
    cfg8 = vh.Cfg(n=2, layout="row", hints=("zh-Hant",), cell_h=80.0, cap_mp=1e9, jpeg_q=84)
    led11 = vh.Ledger(path=cache / "led_corrupt.json", month="2026-10")
    pb = {"A": pages["A"][:4]}
    vh.run_groups(vh.plan_groups(pb, 2), cfg8, fake_ocr_transport([]), led11, 950, sleep=noop, log=nolog)
    st8 = vh.Store(cfg8.cid())
    rawfiles = sorted((st8.dir / "raw").glob("*.json.gz"))
    rawfiles[0].write_bytes(rawfiles[0].read_bytes()[:50])           # cắt cụt
    for p in (st8.dir / "pages").rglob("*.json"):
        p.unlink()
    r8 = vh.run_groups(vh.plan_groups(pb, 2), cfg8, fake_ocr_transport([]), led11, 950, sleep=noop, log=nolog)
    (cache / "led_corrupt.json").write_text("{hong")
    out_ok = vh.Ledger(path=cache / "led_corrupt.json", month="2026-10").used()
    (cache / "led_corrupt.json").write_text("{hong")
    (cache / "led_corrupt.json.bak").write_text("{hong")
    try:
        vh.Ledger(path=cache / "led_corrupt.json", month="2026-10")
        exited = False
    except SystemExit:
        exited = True
    check("T18.tep_hong_khong_sap_luot_so_cai_dung_bak_ca_hai_hong_thoat", len(r8["fails"]) == 1 and r8["from_raw"] == 1 and out_ok >= 1 and exited,
          f"fails={len(r8['fails'])} from_raw={r8['from_raw']} used_from_bak={out_ok} thoát khi cả hai hỏng={exited}")

    # T19: make_transport phân loại lỗi mạng, không để lộ URL/khoá
    class FakeAuth:
        api_key = "AIza" + "x" * 35

        def headers(self): return {}
        def url(self): return "https://vision.googleapis.com/v1/images:annotate?key=" + self.api_key
        def invalidate(self): pass
    orig = vh.urllib.request.urlopen
    outcomes = {}
    for name, exc in (("dns", urllib.error.URLError(socket.gaierror(8, "nodename"))), ("timeout", TimeoutError("timed out")), ("reset", ConnectionResetError("reset")),
                      ("invalid_url", http.client.InvalidURL("URL can't contain control characters: ?key=" + FakeAuth.api_key))):
        def boom(req, timeout=0, _e=exc):
            raise _e
        vh.urllib.request.urlopen = boom
        try:
            vh.make_transport(FakeAuth())(b"{}")
            outcomes[name] = "không lỗi"
        except Exception as e:  # noqa: BLE001
            outcomes[name] = (type(e).__name__, FakeAuth.api_key in str(e) or FakeAuth.api_key in repr(e.__cause__))
    vh.urllib.request.urlopen = orig
    check("T19.phan_loai_loi_mang_va_khong_ro_ri_khoa", outcomes["dns"][0] == "PreSendError" and outcomes["timeout"][0] == "Ambiguous" and outcomes["reset"][0] == "Ambiguous"
          and outcomes["invalid_url"][0] == "Ambiguous" and not any(v[1] for v in outcomes.values()), f"{outcomes}")

    # T21: sổ cái có BẢN SAO: nạp lấy MAX từng tháng; mất một trong hai bản không làm sổ cái về 0
    pa, pm = cache / "led_main.json", cache / "led_mirror.json"
    L1 = vh.Ledger(path=pa, month="2026-10", mirror=str(pm))
    for i in range(7):
        L1.reserve("c", f"g{i}", 950)
    pa.unlink()
    (cache / "led_main.json.bak").unlink(missing_ok=True)
    L2 = vh.Ledger(path=pa, month="2026-10", mirror=str(pm))
    only_mirror = L2.used()                                                            # mất bản chính → còn bản sao = 7
    L2.reserve("c", "gx", 950)                                                         # đặt chỗ tiếp: ghi lại cả bản chính lẫn bản sao (8)
    pm.write_text(json.dumps({"months": {"2026-10": {"calls": 3, "seq": 3, "history": []}}}))   # bản sao cũ/nhỏ hơn
    both = vh.Ledger(path=pa, month="2026-10", mirror=str(pm)).used()                  # lấy MAX → 8
    check("T21.so_cai_ban_sao_lay_max", only_mirror == 7 and both == 8, f"chỉ còn bản sao={only_mirror}; bản sao cũ nhỏ hơn={both}")

    # T20: hạn mức tháng theo giờ Pacific có định dạng đúng
    mo = vh.current_month()
    check("T20.thang_Pacific_dung_dinh_dang", len(mo) == 7 and mo[4] == "-", mo)

    nf = [n for n, ok in RES if not ok]
    print(f"TỔNG: pass={len(RES) - len(nf)} fail={len(nf)} n={len(RES)}")
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
