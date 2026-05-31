"""Lọc cột-ma + gán nhãn 1..N theo R->L, render overlay có số cột.

Cách dùng:
    # 1 trang riêng:
    python3 evaluation/fix_columns_filter.py --cache prepared/SachThanhTruyen2/detected/page_0082_ocr_cache.json
    # toàn bộ 1 sách (so trước/sau lọc):
    python3 evaluation/fix_columns_filter.py --book SachThanhTruyen2 --expected 9
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont


def filter_columns(cols: list[list[dict]], min_chars: int = 3) -> list[list[dict]]:
    """Bỏ các cột-ma (n_chars <= min_chars). Trả về list cột mới (đã đánh chỉ số 1..N theo thứ tự gốc R->L)."""
    return [c for c in cols if len(c) > min_chars]


def render(image_path: Path, cols_kept: list[list[dict]], cols_dropped: list[list[dict]], out_path: Path):
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
        font_big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
    except Exception:
        font = ImageFont.load_default(); font_big = font

    # Cột bị bỏ -> đỏ mờ
    for c in cols_dropped:
        for ch in c:
            b = ch.get("bbox")
            if b:
                draw.rectangle(b, outline=(255, 0, 0, 200), width=3)
        if c:
            b = c[0]["bbox"]
            draw.text((b[0], max(0, b[1] - 30)), "DROP", fill=(220, 0, 0), font=font)

    # Cột giữ + số thứ tự 1..N (R->L)
    for idx, c in enumerate(cols_kept, start=1):
        for ch in c:
            b = ch.get("bbox")
            if b:
                draw.rectangle(b, outline=(0, 170, 0, 220), width=3)
        if c:
            b = c[0]["bbox"]
            draw.text((b[0], max(0, b[1] - 48)), f"#{idx}", fill=(0, 90, 0), font=font_big)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def process_cache(cache_path: Path, expected: int | None, min_chars: int,
                  render_out: Path | None = None) -> dict:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    cols = data["columns"]
    kept = filter_columns(cols, min_chars=min_chars)
    dropped = [c for c in cols if len(c) <= min_chars]
    img = ROOT / data["image"]
    if render_out and img.exists():
        render(img, kept, dropped, render_out)
    return {
        "page": cache_path.stem.replace("_ocr_cache", ""),
        "n_cols_raw": len(cols),
        "n_cols_kept": len(kept),
        "n_cols_dropped": len(dropped),
        "match_expected": (expected is None) or (len(kept) == expected),
        "chars_dropped": sum(len(c) for c in dropped),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", help="1 file *_ocr_cache.json cụ thể")
    ap.add_argument("--book", help="Tên thư mục trong prepared/, chạy toàn sách")
    ap.add_argument("--expected", type=int, default=None,
                    help="Số cột kỳ vọng (vd: 9). Để kiểm tra sau khi lọc")
    ap.add_argument("--min-chars", type=int, default=3,
                    help="Bỏ cột có <= ngưỡng này (mặc định 3)")
    ap.add_argument("--out", default="evaluation/_kinhhannom_debug/_fix")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cache:
        cp = Path(args.cache)
        out_img = out_dir / f"{cp.stem.replace('_ocr_cache','')}_fixed.png" if args.render else None
        r = process_cache(cp, args.expected, args.min_chars, out_img)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    if args.book:
        det = ROOT / "prepared" / args.book / "detected"
        files = sorted(det.glob("*_ocr_cache.json"))
        results = []
        for cp in files:
            out_img = out_dir / args.book / f"{cp.stem.replace('_ocr_cache','')}_fixed.png" if args.render else None
            results.append(process_cache(cp, args.expected, args.min_chars, out_img))

        # Thống kê
        n = len(results)
        matched = sum(1 for r in results if r["match_expected"])
        total_dropped = sum(r["n_cols_dropped"] for r in results)
        total_chars_dropped = sum(r["chars_dropped"] for r in results)
        print(f"Sách: {args.book}")
        print(f"  Số trang: {n}")
        print(f"  Cột TRƯỚC lọc (avg): {sum(r['n_cols_raw'] for r in results)/n:.1f}")
        print(f"  Cột SAU  lọc (avg): {sum(r['n_cols_kept'] for r in results)/n:.1f}")
        if args.expected:
            print(f"  Trang khớp expected={args.expected}: {matched}/{n}  ({100*matched/n:.1f}%)")
        print(f"  Tổng cột-ma đã bỏ: {total_dropped}  ({total_chars_dropped} ký tự rác)")

        # In các trang lệch để soát
        if args.expected:
            bad = [r for r in results if not r["match_expected"]]
            if bad:
                print(f"\n  Trang vẫn lệch ({len(bad)}):")
                for r in bad[:20]:
                    print(f"    {r['page']}: raw={r['n_cols_raw']} kept={r['n_cols_kept']}")

        (out_dir / f"{args.book}_summary.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    ap.error("Provide --cache or --book")


if __name__ == "__main__":
    main()
