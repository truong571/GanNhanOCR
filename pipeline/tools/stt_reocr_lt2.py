#!/usr/bin/env python3
"""stt_reocr_lt2.py — OCR LẠI bộ STT (SachThanhTruyen2/4/11) bằng kim ở CHẾ ĐỘ NÔM (lang_type=2)
làm LẦN ĐỌC THỨ HAI, ghi vào CACHE RIÊNG.

Vì sao: STT đã được OCR bằng kim lang_type=1 (Hán, KIM_LANG_TYPE_DEFAULT) — chế độ này gần như
mù chữ Nôm riêng (trên nhãn người IHR: lt1 đọc đúng chữ Nôm riêng 2,4 % so với lt2 96,4 %).
Lần đọc lt2 là tín hiệu độc lập: ô mà hai lần đọc TRÙNG nhau đáng tin hơn (đo bằng
scripts/measure/stt_lt2_eval.py).

AN TOÀN DỮ LIỆU:
  * KHÔNG BAO GIỜ đọc-ghi đè prepared/<Sách>/detected/*_ocr_cache.json (lần đọc lt1 = dữ liệu gốc
    không tái tạo được) — chỉ ĐỌC nó để lấy đúng ảnh đã gửi đi (kiểm md5/pixel như verify_cache_image).
  * Ghi prepared/<Sách>/kim_raw_lt2/<page>_lt2.json (+ sổ kim_calls_lt2.json). Chạy lại = bỏ qua
    trang đã có cache (0 lượt API) → đứt giữa chừng thì chạy lại đúng lệnh cũ.

QUY TRÌNH GIỐNG HỆT lần đọc lt1 (pipeline/step1_extract.py → core.ocr.ocr_api.ocr_page với tham số
mặc định): cắt khung trang (crop_to_frame, pad 12) → upload → recognize → boxes_to_columns; nếu số cột
≠ 9 thì thử lại pad 30 và giữ kết quả gần 9 cột hơn; đổi toạ độ về trang đầy đủ (_frame_offset).
Khác DUY NHẤT: recognize(..., lang_type=2); ocr_id / reading_direction / font_type giữ mặc định như lt1.

Dùng:
  .venv/bin/python -m pipeline.tools.stt_reocr_lt2 --status                 # độ phủ cache (0 API)
  .venv/bin/python -m pipeline.tools.stt_reocr_lt2 --book all --dry-run     # xem sẽ gửi bao nhiêu trang
  .venv/bin/python -m pipeline.tools.stt_reocr_lt2 --book all               # chạy thật (~448 trang)
  .venv/bin/python -m pipeline.tools.stt_reocr_lt2 --book stt2 --pages 12 --budget 1   # thử 1 trang
  .venv/bin/python -m pipeline.tools.stt_reocr_lt2 --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BOOKS = {"stt2": "SachThanhTruyen2", "stt4": "SachThanhTruyen4", "stt11": "SachThanhTruyen11"}
LANG_TYPE = 2                  # Nôm — khác DUY NHẤT so với lần đọc lt1
FRAME_PAD, RETRY_PAD, EXPECTED_COLS = 12, 30, 9     # = mặc định ocr_page() mà step1_extract dùng
SEC_PER_PAGE = 6.5             # đo 27/09 trên kimhannom.clc (1 lượt upload + recognize)
MAX_CONSEC_FAIL = 3            # trang lỗi liên tiếp -> dừng (nghi API/tài khoản)
OUT_DIRNAME = "kim_raw_lt2"


def lt1_dir(book: str) -> Path:
    return REPO / "prepared" / BOOKS[book] / "detected"


def out_dir(book: str) -> Path:
    return REPO / "prepared" / BOOKS[book] / OUT_DIRNAME


def page_of(cache_file: Path) -> str:
    return cache_file.name[: -len("_ocr_cache.json")]


def out_path(book: str, page: str) -> Path:
    return out_dir(book) / f"{page}_lt2.json"


def parse_pages(spec: str | None) -> set[str] | None:
    """'12,15-20' -> {'page_0012', 'page_0015', ...}"""
    if not spec:
        return None
    pages: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            for n in range(int(a), int(b) + 1):
                pages.add(f"page_{n:04d}")
        else:
            pages.add(f"page_{int(part):04d}")
    return pages


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_ok(p: Path) -> bool:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return isinstance(d.get("columns"), list) and d.get("kim_params", {}).get("lang_type") == LANG_TYPE
    except Exception:
        return False


def plan(books: list[str], pages: set[str] | None) -> list[dict]:
    """Mọi trang có cache lt1 (đúng tập trang STT đã OCR) -> việc cần làm."""
    todo = []
    for b in books:
        for cf in sorted(lt1_dir(b).glob("*_ocr_cache.json")):
            pg = page_of(cf)
            if pages is not None and pg not in pages:
                continue
            op = out_path(b, pg)
            todo.append(dict(book=b, page=pg, lt1_cache=cf, out=op, cached=op.exists() and cached_ok(op)))
    return todo


def resolve_image(lt1: dict) -> tuple[Path | None, str]:
    """Ảnh mà lần đọc lt1 đã gửi: đường dẫn trong cache + đối chiếu md5 (rồi pixel) như verify_cache_image."""
    img = Path(lt1.get("image", ""))
    if not img.is_absolute():
        img = REPO / img
    if not img.exists():
        return None, f"thiếu ảnh {img}"
    if lt1.get("image_hash") and md5_file(img) != lt1["image_hash"]:
        from core.ocr import ocr_api as A
        px = A._pixel_hash(str(img))
        if not (lt1.get("pixel_hash") and px == lt1["pixel_hash"]):
            return None, "ảnh đã đổi nội dung so với lúc OCR lt1 (md5 + pixel lệch)"
    return img, "ok"


def one_pass(A, image_path: Path, pad: int):
    """Như ocr_api._ocr_one_pass nhưng recognize(lang_type=2). -> (boxes|None, framed, n_calls)."""
    import cv2
    from core.image.frame_detector import crop_to_frame
    upload_path, tmp, framed = str(image_path), None, False
    bgr = cv2.imread(str(image_path))
    if bgr is not None:
        crop = crop_to_frame(bgr, pad=pad)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        cv2.imwrite(tmp.name, crop)
        upload_path, framed = tmp.name, True
    try:
        file_name = A.upload_image(upload_path)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    if not file_name:
        return None, framed, 1
    return A.recognize(file_name, lang_type=LANG_TYPE), framed, 2


def ocr_page_lt2(A, image_path: Path) -> tuple[dict | None, int, str]:
    """OCR 1 trang theo đúng quy trình ocr_page(); trả (bản ghi cache, số lượt gọi, lỗi)."""
    calls = 0
    boxes, framed, n = one_pass(A, image_path, FRAME_PAD)
    calls += n
    if boxes is None:
        return None, calls, "upload/recognize thất bại"
    pad = FRAME_PAD
    columns = A.boxes_to_columns(boxes)
    if framed and len(columns) != EXPECTED_COLS and RETRY_PAD > FRAME_PAD:
        b2, _, n = one_pass(A, image_path, RETRY_PAD)
        calls += n
        if b2 is not None:
            c2 = A.boxes_to_columns(b2)
            if abs(len(c2) - EXPECTED_COLS) < abs(len(columns) - EXPECTED_COLS):
                columns, boxes, pad = c2, b2, RETRY_PAD
    if framed:
        ox, oy = A._frame_offset(str(image_path), pad)
        A._translate_columns(columns, ox, oy)
        A._translate_boxes(boxes, ox, oy)
    rec = {
        "image": os.path.relpath(image_path, REPO), "image_hash": md5_file(image_path),
        "pixel_hash": A._pixel_hash(str(image_path)),
        "framed": framed, "frame_pad": pad, "coords_space": "fullpage",
        "n_columns": len(columns), "columns": columns, "boxes_raw": boxes,
        "kim_params": {"ocr_id": A.KIM_OCR_ID_DEFAULT, "lang_type": LANG_TYPE,
                       "reading_direction": A.KIM_READING_DIRECTION_DEFAULT,
                       "font_type": A.KIM_FONT_TYPE_DEFAULT},
        "domain": A._SN_DOMAIN, "created": datetime.now().isoformat(timespec="seconds"),
    }
    return rec, calls, ""


def ledger_update(book: str, entry: dict) -> None:
    lp = out_dir(book) / "kim_calls_lt2.json"
    try:
        led = json.loads(lp.read_text(encoding="utf-8"))
    except Exception:
        led = {"calls": 0, "pages_ok": 0, "pages_failed": 0, "log": []}
    led["calls"] += entry.get("calls", 0)
    led["pages_ok" if entry.get("ok") else "pages_failed"] += 1
    led["log"].append(entry)
    tmp = lp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(lp)


def write_atomic(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    # chốt: KHÔNG BAO GIỜ ghi vào thư mục cache lt1
    assert p.parent.name == OUT_DIRNAME, f"từ chối ghi ngoài {OUT_DIRNAME}/: {p}"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def status(books: list[str]) -> int:
    tot_all = tot_done = 0
    for b in books:
        items = plan([b], None)
        done = sum(it["cached"] for it in items)
        tot_all += len(items)
        tot_done += done
        miss = [it["page"] for it in items if not it["cached"]]
        led = out_dir(b) / "kim_calls_lt2.json"
        calls = json.loads(led.read_text(encoding="utf-8")).get("calls", 0) if led.exists() else 0
        print(f"[stt_lt2] {b:6s} ({BOOKS[b]}): cache lt2 {done}/{len(items)} trang · sổ {calls} lượt gọi"
              + (f" · thiếu {len(miss)} (vd {miss[:3]})" if miss else " · ĐỦ"))
    print(f"[stt_lt2] tổng {tot_done}/{tot_all} trang"
          + ("" if tot_done == tot_all else f" · còn {tot_all - tot_done} trang ≈ "
             f"{(tot_all - tot_done) * SEC_PER_PAGE / 60:.0f} phút"))
    return 0 if tot_done == tot_all else 1


def run(books: list[str], pages: set[str] | None, budget: int | None, dry: bool) -> int:
    items = plan(books, pages)
    todo = [it for it in items if not it["cached"]]
    print(f"[stt_lt2] {len(items)} trang trong phạm vi · đã có cache {len(items) - len(todo)} · cần OCR "
          f"{len(todo)}" + (f" · ngân sách lượt này {budget}" if budget is not None else ""))
    if budget is not None:
        todo = todo[:budget]
    print(f"[stt_lt2] sẽ gửi {len(todo)} trang · ước ≈ {len(todo) * SEC_PER_PAGE / 60:.0f} phút "
          f"(+ trang phải thử lại pad {RETRY_PAD})")
    if dry or not todo:
        if dry:
            for it in todo[:5]:
                print(f"    {it['book']} {it['page']} -> {os.path.relpath(it['out'], REPO)}")
            if len(todo) > 5:
                print(f"    … {len(todo) - 5} trang nữa")
        return 0

    from core.ocr import ocr_api as A
    print(f"[stt_lt2] kim: {A._SN_DOMAIN} · lang_type={LANG_TYPE} (Nôm) · ocr_id={A.KIM_OCR_ID_DEFAULT} · "
          f"font_type={A.KIM_FONT_TYPE_DEFAULT} · reading_direction={A.KIM_READING_DIRECTION_DEFAULT}")
    t_all, n_ok, n_fail, consec = time.time(), 0, 0, 0
    for i, it in enumerate(todo, 1):
        t0 = time.time()
        lt1 = json.loads(it["lt1_cache"].read_text(encoding="utf-8"))
        img, why = resolve_image(lt1)
        entry = dict(ts=datetime.now().isoformat(timespec="seconds"), page=it["page"])
        if img is None:
            n_fail += 1
            ledger_update(it["book"], {**entry, "ok": False, "calls": 0, "error": why})
            print(f"  [{i}/{len(todo)}] {it['book']} {it['page']}: BỎ QUA — {why}")
            continue
        rec, calls, err = ocr_page_lt2(A, img)
        if A.is_guest_mode():
            ledger_update(it["book"], {**entry, "ok": False, "calls": calls, "error": "rơi về Guest Mode"})
            print("[stt_lt2] DỪNG: API rơi về Guest Mode (token bị từ chối / tài khoản không hoạt động). "
                  "Kiểm SN_OCR_USERNAME/SN_OCR_PASSWORD trong .env rồi chạy lại đúng lệnh này.", file=sys.stderr)
            return 3
        if rec is None:
            n_fail += 1
            consec += 1
            ledger_update(it["book"], {**entry, "ok": False, "calls": calls, "error": err})
            print(f"  [{i}/{len(todo)}] {it['book']} {it['page']}: LỖI ({err})")
            if consec >= MAX_CONSEC_FAIL:
                print(f"[stt_lt2] DỪNG: {consec} trang lỗi liên tiếp — kiểm kết nối/tài khoản rồi chạy lại.",
                      file=sys.stderr)
                return 2
            continue
        consec = 0
        rec["lt1_cache"] = os.path.relpath(it["lt1_cache"], REPO)
        rec["lt1_n_columns"] = lt1.get("n_columns")
        write_atomic(it["out"], rec)
        n_ok += 1
        dt = time.time() - t0
        ledger_update(it["book"], {**entry, "ok": True, "calls": calls, "seconds": round(dt, 1),
                                   "n_columns": rec["n_columns"], "frame_pad": rec["frame_pad"]})
        el = time.time() - t_all
        eta = el / i * (len(todo) - i)
        n1 = sum(len(c) for c in lt1.get("columns", []))
        n2 = sum(len(c) for c in rec["columns"])
        print(f"  [{i}/{len(todo)}] {it['book']} {it['page']}: {rec['n_columns']} cột · chữ lt2 {n2} "
              f"(lt1 {n1}) · {dt:.1f} s · còn ≈ {eta / 60:.0f} phút", flush=True)
    print(f"[stt_lt2] xong: {n_ok} trang ok · {n_fail} lỗi · {(time.time() - t_all) / 60:.1f} phút")
    return 0 if n_fail == 0 else 1


def selftest() -> int:
    """0 API: kiểm phân tích --pages, lập kế hoạch đúng 448 trang lt1, chốt chặn ghi, và quy trình
    ocr_page_lt2 trên kim GIẢ (không mạng)."""
    ok = 0
    assert parse_pages("12,15-16") == {"page_0012", "page_0015", "page_0016"}; ok += 1
    items = plan(list(BOOKS), None)
    assert all(it["lt1_cache"].parent.name == "detected" for it in items); ok += 1
    assert all(it["out"].parent.name == OUT_DIRNAME for it in items); ok += 1
    try:
        write_atomic(REPO / "prepared" / "x" / "detected" / "page_0000_lt2.json", {})
        raise SystemExit("chốt chặn ghi không hoạt động")
    except AssertionError:
        ok += 1

    class FakeA:
        KIM_OCR_ID_DEFAULT = KIM_READING_DIRECTION_DEFAULT = KIM_FONT_TYPE_DEFAULT = 1
        _SN_DOMAIN = "fake"
        calls = []

        @staticmethod
        def upload_image(p):
            FakeA.calls.append(("up", p))
            return "f.png"

        @staticmethod
        def recognize(fn, lang_type=None):
            FakeA.calls.append(("rec", lang_type))
            return [{"points": [[100 - 10 * k, 0], [110 - 10 * k, 0], [110 - 10 * k, 40], [100 - 10 * k, 40]],
                     "transcription": "字字"} for k in range(9)]

        from core.ocr.ocr_api import boxes_to_columns, _translate_columns, _translate_boxes, _pixel_hash  # noqa
        boxes_to_columns = staticmethod(boxes_to_columns)
        _translate_columns = staticmethod(_translate_columns)
        _translate_boxes = staticmethod(_translate_boxes)
        _pixel_hash = staticmethod(_pixel_hash)

        @staticmethod
        def _frame_offset(p, pad):
            return (5, 7)

    img = None
    for it in items:
        lt1 = json.loads(it["lt1_cache"].read_text(encoding="utf-8"))
        img, why = resolve_image(lt1)
        if img is not None:
            break
    assert img is not None, "không tìm được ảnh STT khớp md5 cache lt1"; ok += 1
    rec, calls, err = ocr_page_lt2(FakeA, img)
    assert rec and not err and all(c[1] == LANG_TYPE for c in FakeA.calls if c[0] == "rec"); ok += 1
    assert rec["kim_params"]["lang_type"] == 2 and rec["coords_space"] == "fullpage"; ok += 1
    assert rec["columns"][0][0]["bbox"][1] >= 7, "chưa đổi toạ độ về trang đầy đủ"; ok += 1
    print(f"[stt_lt2] selftest: {ok}/{ok} PASS · kế hoạch {len(items)} trang lt1 "
          f"({', '.join(f'{b} {sum(1 for x in items if x['book'] == b)}' for b in BOOKS)})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default="all", choices=[*BOOKS, "all"])
    ap.add_argument("--pages", help="vd 12,15-20 (số page_XXXX)")
    ap.add_argument("--budget", type=int, help="số TRANG tối đa gửi kim trong lượt này")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true", help="độ phủ cache lt2 (0 API)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    books = list(BOOKS) if a.book == "all" else [a.book]
    if a.selftest:
        return selftest()
    if a.status:
        return status(books)
    return run(books, parse_pages(a.pages), a.budget, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
