"""Crop khung -> gửi kinhhannom OCR -> KIỂM kết quả có đúng 9 cột không.

Báo trang THIẾU (<9 cột) hoặc DƯ (>9 cột). Dùng ocr_page(use_frame=True) nên
ảnh được crop sát 9 cột (loại số 1-9, số trang, viền) trước khi OCR.

Cache theo trang -> resume được (chạy lại bỏ qua trang đã xong).

Usage:
    python3 evaluation/check_ocr_columns.py --clear-cache          # cả 3 cuốn, xoá cache cũ
    python3 evaluation/check_ocr_columns.py --book SachThanhTruyen4
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.ocr.ocr_api import ocr_page  # noqa: E402

BOOKS = ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]
EXPECTED = 9


def run_book(book: str, clear_cache: bool, out_dir: Path, retries: int = 2) -> list[dict]:
    pages_dir = ROOT / "prepared" / book / "pages"
    cache_dir = ROOT / "prepared" / book / "detected"
    pages = sorted(pages_dir.glob("*.png"))
    rows = []
    print(f"\n===== {book}: {len(pages)} trang =====", flush=True)
    for i, p in enumerate(pages, 1):
        cache_path = cache_dir / f"{p.stem}_ocr_cache.json"
        if clear_cache and cache_path.exists():
            cache_path.unlink()
        cols = None
        for attempt in range(retries + 1):
            try:
                cols = ocr_page(str(p), cache_path=str(cache_path), use_frame=True)
            except Exception as e:
                print(f"  [err] {p.stem} attempt {attempt}: {e}", flush=True)
                cols = None
            if cols is not None:
                break
            time.sleep(2)
        if cols is None:
            rows.append({"page": p.stem, "n_cols": -1, "status": "ERR"})
            print(f"  [{i}/{len(pages)}] {p.stem}: ERR (OCR thất bại)", flush=True)
            continue
        n = len(cols)
        status = "OK" if n == EXPECTED else ("THIEU" if n < EXPECTED else "DU")
        rows.append({"page": p.stem, "n_cols": n, "status": status,
                     "chars": sum(len(c) for c in cols)})
        if status != "OK" or i % 20 == 0:
            print(f"  [{i}/{len(pages)}] {p.stem}: {n} cột [{status}]", flush=True)
    # CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{book}_cols.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page", "n_cols", "status", "chars"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="all")
    ap.add_argument("--clear-cache", action="store_true",
                    help="Xoá cache OCR cũ (OCR ảnh thô) để chạy lại với crop khung")
    ap.add_argument("--out", default="evaluation/_kinhhannom_debug/_col_check")
    args = ap.parse_args()
    books = BOOKS if args.book == "all" else [args.book]
    out_dir = ROOT / args.out

    grand = []
    for book in books:
        rows = run_book(book, args.clear_cache, out_dir)
        grand.append((book, rows))

    print("\n" + "=" * 64)
    print("KẾT QUẢ KIỂM 9 CỘT (sau khi crop khung + OCR kinhhannom)")
    print("=" * 64)
    tot = thieu = du = err = ok = 0
    for book, rows in grand:
        b_ok = sum(1 for r in rows if r["status"] == "OK")
        b_thieu = [r for r in rows if r["status"] == "THIEU"]
        b_du = [r for r in rows if r["status"] == "DU"]
        b_err = [r for r in rows if r["status"] == "ERR"]
        tot += len(rows); ok += b_ok
        thieu += len(b_thieu); du += len(b_du); err += len(b_err)
        print(f"\n{book}: {len(rows)} trang | OK(=9) {b_ok} | "
              f"THIẾU(<9) {len(b_thieu)} | DƯ(>9) {len(b_du)} | ERR {len(b_err)}")
        for r in b_thieu:
            print(f"   THIẾU {r['page']}: {r['n_cols']} cột")
        for r in b_du:
            print(f"   DƯ    {r['page']}: {r['n_cols']} cột")
        for r in b_err:
            print(f"   ERR   {r['page']}")
    print(f"\nTỔNG: {tot} trang | OK {ok} ({100*ok/max(1,tot):.0f}%) | "
          f"THIẾU {thieu} | DƯ {du} | ERR {err}")
    print(f"CSV: {out_dir}/<cuốn>_cols.csv")


if __name__ == "__main__":
    main()
