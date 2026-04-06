#!/usr/bin/env python3
"""
clean_crops.py - Khử nhiễu ảnh ký tự đã cắt từ detect_characters.py

Sử dụng CharacterImageCleaner từ FontDiffusion với phương pháp Sauvola
(tối ưu cho tài liệu lịch sử viết tay bị nhiễu).

Pipeline:
  1. Đọc ảnh crop grayscale
  2. Sauvola binarization (xử lý tốt uneven illumination)
  3. Morphological cleanup
  4. Connected component noise removal
  5. Stroke thickness normalization
  6. Center + resize về kích thước chuẩn
  7. Output: nét đen trên nền trắng

Usage:
  # Khử nhiễu toàn bộ crops của 1 bộ sách:
  python clean_crops.py data/prepared/SachThanhTruyen2/detected

  # Chỉ 1 trang:
  python clean_crops.py data/prepared/SachThanhTruyen2/detected --page 12

  # Tuỳ chỉnh kích thước và phương pháp:
  python clean_crops.py data/prepared/SachThanhTruyen2/detected --size 64 --method sauvola

  # Xem debug visualization:
  python clean_crops.py data/prepared/SachThanhTruyen2/detected --verify --verify-samples 20
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Import CharacterImageCleaner from FontDiffusion
sys.path.insert(0, str(Path(__file__).parent / "FontDiffusion" / "scripts"))
from clean_image_enhance import CharacterImageCleaner


def clean_page_crops(
    crops_dir: Path,
    output_dir: Path,
    cleaner: CharacterImageCleaner,
    verbose: bool = False,
) -> dict:
    """Khử nhiễu tất cả crop trong 1 thư mục trang.

    Args:
        crops_dir: Thư mục chứa crops (e.g. crops/page_0012/)
        output_dir: Thư mục output tương ứng
        cleaner: CharacterImageCleaner instance
        verbose: In chi tiết

    Returns:
        dict với thống kê {total, success, failed, failed_files}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_files = sorted(crops_dir.glob("*.png"))
    if not crop_files:
        return {"total": 0, "success": 0, "failed": 0, "failed_files": []}

    success = 0
    failed_files = []

    for crop_path in crop_files:
        try:
            cleaned, debug_info = cleaner.clean_image(crop_path)
        except Exception:
            cleaned = None

        if cleaned is not None:
            out_path = output_dir / crop_path.name
            cv2.imwrite(str(out_path), cleaned)
            success += 1
        else:
            # Fallback: lưu ảnh gốc resize về target_size
            orig = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
            if orig is not None and orig.size > 0:
                h, w = orig.shape
                if h > 0 and w > 0:
                    resized = cv2.resize(orig, (cleaner.target_size, cleaner.target_size))
                    out_path = output_dir / crop_path.name
                    cv2.imwrite(str(out_path), resized)
            failed_files.append(crop_path.name)
            if verbose:
                print(f"    [FAIL] {crop_path.name} (fallback to original)")

    return {
        "total": len(crop_files),
        "success": success,
        "failed": len(failed_files),
        "failed_files": failed_files,
    }


def process_detected_dir(
    detected_dir: Path,
    output_name: str = "crops_cleaned",
    size: int = 64,
    method: str = "sauvola",
    denoise: int = 3,
    min_stroke: int = 2,
    padding: int = 5,
    page_filter: int | None = None,
    verify: bool = False,
    verify_samples: int = 10,
    verbose: bool = True,
):
    """Xử lý toàn bộ thư mục detected.

    Args:
        detected_dir: Thư mục detected/ (chứa crops/)
        output_name: Tên thư mục output trong detected/
        size: Kích thước output (square)
        method: Phương pháp binarization
        denoise: Strength khử nhiễu
        min_stroke: Độ dày nét tối thiểu
        padding: Padding quanh ký tự
        page_filter: Chỉ xử lý 1 trang cụ thể
        verify: Tạo ảnh so sánh before/after
        verify_samples: Số sample verify
        verbose: In chi tiết
    """
    crops_base = detected_dir / "crops"
    if not crops_base.exists():
        print(f"[ERROR] Không tìm thấy thư mục crops: {crops_base}", file=sys.stderr)
        return

    output_base = detected_dir / output_name
    output_base.mkdir(parents=True, exist_ok=True)

    cleaner = CharacterImageCleaner(
        target_size=size,
        padding=padding,
        method=method,
        denoise_strength=denoise,
        min_stroke_thickness=min_stroke,
    )

    # Tìm tất cả thư mục trang
    page_dirs = sorted(crops_base.glob("page_*"))
    if page_filter is not None:
        page_dirs = [d for d in page_dirs if d.name == f"page_{page_filter:04d}"]

    if not page_dirs:
        print("[ERROR] Không tìm thấy thư mục trang nào.", file=sys.stderr)
        return

    if verbose:
        parent_name = detected_dir.parent.name
        print(f"\nClean Crops: {parent_name}")
        print(f"  Method: {method}, Size: {size}x{size}, Denoise: {denoise}")
        print(f"  Input:  {crops_base}/")
        print(f"  Output: {output_base}/")
        print("-" * 70)

    total_all = 0
    success_all = 0
    failed_all = 0
    verify_count = 0

    for page_dir in page_dirs:
        page_name = page_dir.name  # e.g. "page_0012"
        out_dir = output_base / page_name

        stats = clean_page_crops(page_dir, out_dir, cleaner, verbose=verbose)

        total_all += stats["total"]
        success_all += stats["success"]
        failed_all += stats["failed"]

        if verbose:
            status = "OK" if stats["failed"] == 0 else f"FAIL={stats['failed']}"
            print(
                f"  {page_name}: {stats['success']}/{stats['total']} cleaned  [{status}]"
            )

        # Tạo verify image (so sánh before/after)
        if verify and verify_count < verify_samples and stats["total"] > 0:
            _save_verify_image(page_dir, out_dir, detected_dir / "verify", page_name)
            verify_count += 1

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("TỔNG KẾT CLEAN CROPS")
        print("=" * 70)
        print(f"  Tổng trang      : {len(page_dirs)}")
        print(f"  Tổng ký tự      : {total_all}")
        print(f"  Thành công      : {success_all}")
        print(f"  Thất bại        : {failed_all}")
        if total_all > 0:
            print(f"  Tỷ lệ thành công: {success_all/total_all:.1%}")
        print(f"  Output          : {output_base}/")

    # Save summary JSON
    summary = {
        "source": str(crops_base),
        "output": str(output_base),
        "settings": {
            "method": method,
            "size": size,
            "padding": padding,
            "denoise": denoise,
            "min_stroke": min_stroke,
        },
        "total_pages": len(page_dirs),
        "total_chars": total_all,
        "success": success_all,
        "failed": failed_all,
    }
    with open(output_base / "clean_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _save_verify_image(
    original_dir: Path, cleaned_dir: Path, verify_dir: Path, page_name: str
):
    """Tạo ảnh so sánh before/after cho verification."""
    verify_dir.mkdir(parents=True, exist_ok=True)

    orig_files = sorted(original_dir.glob("*.png"))[:12]  # Lấy 12 ký tự đầu
    if not orig_files:
        return

    rows = []
    for orig_path in orig_files:
        cleaned_path = cleaned_dir / orig_path.name
        if not cleaned_path.exists():
            continue

        orig = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
        cleaned = cv2.imread(str(cleaned_path), cv2.IMREAD_GRAYSCALE)

        if orig is None or cleaned is None:
            continue

        # Resize original to match cleaned height for comparison
        h_clean = cleaned.shape[0]
        ratio = h_clean / orig.shape[0]
        orig_resized = cv2.resize(
            orig, (int(orig.shape[1] * ratio), h_clean),
            interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC,
        )

        # Thêm separator
        sep = np.full((h_clean, 4), 180, dtype=np.uint8)
        row = np.hstack([orig_resized, sep, cleaned])
        rows.append(row)

    if not rows:
        return

    # Pad all rows to same width
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1]), 255, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    # Thêm separator giữa các hàng
    sep_h = np.full((3, max_w), 180, dtype=np.uint8)
    final_rows = []
    for i, r in enumerate(padded):
        if i > 0:
            final_rows.append(sep_h)
        final_rows.append(r)

    verify_img = np.vstack(final_rows)
    cv2.imwrite(str(verify_dir / f"{page_name}_verify.png"), verify_img)


def main():
    parser = argparse.ArgumentParser(
        description="Khử nhiễu ảnh ký tự Nôm đã cắt từ detect_characters.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python clean_crops.py data/prepared/SachThanhTruyen2/detected
  python clean_crops.py data/prepared/SachThanhTruyen2/detected --method sauvola --size 64
  python clean_crops.py data/prepared/SachThanhTruyen2/detected --page 12 --verify
        """,
    )
    parser.add_argument(
        "detected_dir", type=str,
        help="Thư mục detected/ (chứa crops/)",
    )
    parser.add_argument(
        "--output-name", type=str, default="crops_cleaned",
        help="Tên thư mục output trong detected/ (default: crops_cleaned)",
    )
    parser.add_argument(
        "--size", type=int, default=64,
        help="Kích thước output vuông (default: 64)",
    )
    parser.add_argument(
        "--method", type=str, default="sauvola",
        choices=["auto", "otsu", "adaptive", "sauvola"],
        help="Phương pháp binarization (default: sauvola - tốt nhất cho tài liệu cổ)",
    )
    parser.add_argument(
        "--denoise", type=int, default=3,
        help="Strength khử nhiễu (số lẻ, default: 3)",
    )
    parser.add_argument(
        "--min-stroke", type=int, default=2,
        help="Độ dày nét tối thiểu (default: 2)",
    )
    parser.add_argument(
        "--padding", type=int, default=5,
        help="Padding quanh ký tự (default: 5)",
    )
    parser.add_argument(
        "--page", type=int, default=None,
        help="Chỉ xử lý 1 trang cụ thể",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Tạo ảnh so sánh before/after",
    )
    parser.add_argument(
        "--verify-samples", type=int, default=10,
        help="Số trang tạo ảnh verify (default: 10)",
    )

    args = parser.parse_args()

    detected_dir = Path(args.detected_dir)
    if not detected_dir.exists():
        print(f"[ERROR] Không tìm thấy: {detected_dir}", file=sys.stderr)
        sys.exit(1)

    process_detected_dir(
        detected_dir=detected_dir,
        output_name=args.output_name,
        size=args.size,
        method=args.method,
        denoise=args.denoise,
        min_stroke=args.min_stroke,
        padding=args.padding,
        page_filter=args.page,
        verify=args.verify,
        verify_samples=args.verify_samples,
    )


if __name__ == "__main__":
    main()
