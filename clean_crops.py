#!/usr/bin/env python3
"""
clean_crops.py - Làm sạch ảnh ký tự Nôm đã cắt (Giai đoạn 3)

Công nghệ: Sauvola Binarization + Morphological Cleanup + Connected Component

Pipeline:
  Ảnh crop thô (ký tự viết tay, nền giấy cũ, kích thước đa dạng)
  │
  ├─ Sauvola Binarization:
  │   T(x,y) = mean(x,y) × [1 + k × (std(x,y)/R - 1)]
  │   k = 0.2, R = 128 (dynamic range cố định)
  │   → Ngưỡng cục bộ, xử lý tốt nền sáng tối không đều
  │
  ├─ Morphological Cleanup:
  │   ├─ Close (2×2): nối nét bị đứt
  │   └─ Open (3×3): xóa nhiễu nhỏ
  │
  ├─ Connected Component Noise Removal:
  │   Xóa vùng liên thông quá nhỏ (< ngưỡng)
  │
  ├─ Stroke Normalization:
  │   Chuẩn hóa độ dày nét bút (distance transform)
  │
  └─ Center + Resize → 64×64:
      Đặt ký tự ở giữa, nền trắng, nét đen

Usage:
  python clean_crops.py data/prepared/SachThanhTruyen2/detected
  python clean_crops.py data/prepared/SachThanhTruyen2/detected --page 12
  python clean_crops.py data/prepared/SachThanhTruyen2/detected --size 64 --verify
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CharacterCleaner — Self-contained, không phụ thuộc FontDiffusion
# ---------------------------------------------------------------------------

class CharacterCleaner:
    """Làm sạch ảnh ký tự Nôm viết tay từ sách cổ.

    Tối ưu cho tài liệu lịch sử: giấy ố vàng, mực loang, nền không đều.
    Sử dụng Sauvola binarization thay vì Otsu để xử lý tốt uneven illumination.
    """

    def __init__(
        self,
        target_size: int = 64,
        padding: int = 5,
        sauvola_k: float = 0.2,
        sauvola_window: int = 25,
        sauvola_R: float = 128.0,
        denoise_strength: int = 3,
        min_stroke: int = 2,
    ):
        """
        Args:
            target_size: Kích thước output vuông (pixel)
            padding: Padding quanh ký tự (pixel)
            sauvola_k: Sensitivity parameter (0.2 = mặc định Sauvola)
            sauvola_window: Kích thước cửa sổ cục bộ (pixel, số lẻ)
            sauvola_R: Dynamic range (128 = cố định theo paper gốc)
            denoise_strength: Kích thước kernel median blur (số lẻ)
            min_stroke: Độ dày nét tối thiểu (pixel)
        """
        self.target_size = target_size
        self.padding = padding
        self.sauvola_k = sauvola_k
        self.sauvola_window = sauvola_window | 1  # Đảm bảo số lẻ
        self.sauvola_R = sauvola_R
        self.denoise_strength = denoise_strength | 1  # Đảm bảo số lẻ
        self.min_stroke = min_stroke

    # ----- Sauvola Binarization -----

    def _sauvola_binarize(self, gray: np.ndarray) -> np.ndarray:
        """Sauvola binarization — ngưỡng CỤC BỘ cho từng vùng nhỏ.

        Công thức: T(x,y) = mean(x,y) × [1 + k × (std(x,y)/R - 1)]

        Khác Otsu (1 ngưỡng toàn cục): Sauvola tính ngưỡng riêng cho từng
        vùng dựa trên mean + std cục bộ → hiệu quả với giấy cũ sáng tối
        không đều.

        Args:
            gray: Ảnh grayscale (uint8)

        Returns:
            Binary image (uint8): 255=ink (foreground), 0=background
        """
        w = self.sauvola_window
        k = self.sauvola_k
        R = self.sauvola_R

        # Tính local mean bằng box filter (nhanh nhờ integral image)
        gray_f = gray.astype(np.float64)
        mean = cv2.boxFilter(gray_f, -1, (w, w))

        # Tính local std: std = sqrt(E[X²] - E[X]²)
        sqmean = cv2.boxFilter(gray_f ** 2, -1, (w, w))
        variance = sqmean - mean ** 2
        # Clamp variance >= 0 (floating point errors)
        variance = np.maximum(variance, 0)
        std = np.sqrt(variance)

        # Sauvola threshold: T = mean × (1 + k × (std/R - 1))
        threshold = mean * (1.0 + k * (std / R - 1.0))

        # Pixel < threshold → ink (foreground = 255)
        binary = np.zeros_like(gray)
        binary[gray_f < threshold] = 255

        return binary

    # ----- Morphological Cleanup -----

    def _morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """Morphological close → open.

        Close (2×2): nối nét bị đứt nhỏ (dilation rồi erosion)
        Open (3×3): xóa chấm nhiễu nhỏ (erosion rồi dilation)

        Args:
            binary: Ảnh nhị phân (255=ink)

        Returns:
            Ảnh nhị phân đã cleanup
        """
        # Close — nối nét đứt
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

        # Open — xóa nhiễu nhỏ
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, open_kernel)

        return result

    # ----- Connected Component Noise Removal -----

    def _remove_noise_components(self, binary: np.ndarray) -> np.ndarray:
        """Xóa vùng liên thông quá nhỏ (chấm bẩn, nhiễu).

        Tìm tất cả connected components, chỉ giữ vùng có diện tích
        >= min_area. min_area tỉ lệ với kích thước ảnh.

        Args:
            binary: Ảnh nhị phân (255=ink)

        Returns:
            Ảnh đã xóa nhiễu
        """
        h, w = binary.shape
        # Ngưỡng diện tích tối thiểu: 0.5% diện tích ảnh, tối thiểu 10px
        min_area = max(10, int(h * w * 0.005))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        cleaned = np.zeros_like(binary)
        for i in range(1, num_labels):  # Bỏ qua background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = 255

        # Nếu xóa hết → trả lại bản gốc
        if np.sum(cleaned) == 0:
            return binary

        return cleaned

    # ----- Stroke Normalization -----

    def _normalize_stroke(self, binary: np.ndarray) -> np.ndarray:
        """Chuẩn hóa độ dày nét bút.

        Dùng distance transform để đo độ dày trung bình:
        - Nét quá mảnh (< min_stroke) → dilate
        - Nét quá dày (> 3× min_stroke) → erode

        Args:
            binary: Ảnh nhị phân (255=ink)

        Returns:
            Ảnh với nét đã chuẩn hóa
        """
        if np.sum(binary) == 0:
            return binary

        # Distance transform: khoảng cách từ mỗi pixel ink đến biên gần nhất
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        ink_dist = dist[binary > 0]

        if len(ink_dist) == 0:
            return binary

        # Độ dày trung bình ≈ 2 × mean distance
        avg_thickness = np.mean(ink_dist) * 2

        if avg_thickness < self.min_stroke:
            # Nét quá mảnh → dilate
            k_size = max(2, int(self.min_stroke - avg_thickness) + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            binary = cv2.dilate(binary, kernel, iterations=1)

        elif avg_thickness > self.min_stroke * 3:
            # Nét quá dày → erode nhẹ
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            binary = cv2.erode(binary, kernel, iterations=1)

        return binary

    # ----- Center + Resize -----

    def _center_and_resize(self, binary: np.ndarray) -> np.ndarray | None:
        """Đặt ký tự ở giữa canvas target_size × target_size.

        1. Tìm bounding box chứa ink
        2. Resize giữ tỷ lệ, fit vào canvas (trừ padding)
        3. Đặt ở giữa, nền trắng, nét đen

        Args:
            binary: Ảnh nhị phân (255=ink)

        Returns:
            Ảnh output (uint8, nét đen=0, nền trắng=255) hoặc None nếu rỗng
        """
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        if w == 0 or h == 0:
            return None

        char_crop = binary[y:y + h, x:x + w]

        # Resize giữ tỷ lệ, fit vào vùng cho phép
        max_dim = self.target_size - (self.padding * 2)
        ratio = min(max_dim / w, max_dim / h)
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))

        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(char_crop, (new_w, new_h), interpolation=interp)

        # Đặt ở giữa canvas (ink=255 trên nền 0)
        canvas = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        x_off = (self.target_size - new_w) // 2
        y_off = (self.target_size - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Invert: nét đen (0) trên nền trắng (255)
        output = cv2.bitwise_not(canvas)

        return output

    # ----- Main Pipeline -----

    def clean(self, image_path_or_array) -> tuple:
        """Pipeline làm sạch đầy đủ.

        Ảnh crop thô → Sauvola → Morphological → Noise removal
        → Stroke normalization → Center + Resize → nét đen trên nền trắng

        Args:
            image_path_or_array: Path ảnh hoặc numpy array (grayscale)

        Returns:
            (cleaned_image, debug_info) — cleaned_image=None nếu thất bại
        """
        debug_info = {}

        # 1. Load grayscale
        if isinstance(image_path_or_array, (str, Path)):
            gray = cv2.imread(str(image_path_or_array), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return None, {"error": f"Cannot load: {image_path_or_array}"}
        else:
            gray = image_path_or_array
            if len(gray.shape) == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        debug_info["original_shape"] = gray.shape

        # 2. Denoise — median blur khử nhiễu muối tiêu
        if self.denoise_strength > 1:
            denoised = cv2.medianBlur(gray, self.denoise_strength)
        else:
            denoised = gray

        # 3. Sauvola binarization — ngưỡng cục bộ
        binary = self._sauvola_binarize(denoised)
        debug_info["fg_ratio_after_sauvola"] = round(
            np.sum(binary > 0) / binary.size, 4
        )

        # 4. Morphological cleanup — Close(2×2) → Open(3×3)
        binary = self._morphological_cleanup(binary)

        # 5. Connected component noise removal
        binary = self._remove_noise_components(binary)

        # 6. Stroke normalization
        binary = self._normalize_stroke(binary)

        # 7. Center + resize → target_size × target_size
        output = self._center_and_resize(binary)

        if output is None:
            return None, {"error": "Empty after cleaning"}

        debug_info["success"] = True
        debug_info["fg_ratio_final"] = round(
            np.sum(output < 128) / output.size, 4
        )

        return output, debug_info


# ---------------------------------------------------------------------------
# Page-level processing
# ---------------------------------------------------------------------------

def clean_page_crops(
    crops_dir: Path,
    output_dir: Path,
    cleaner: CharacterCleaner,
    verbose: bool = False,
) -> dict:
    """Làm sạch tất cả crop trong 1 thư mục trang.

    Args:
        crops_dir: Thư mục chứa crops (e.g. crops/page_0012/)
        output_dir: Thư mục output tương ứng
        cleaner: CharacterCleaner instance
        verbose: In chi tiết

    Returns:
        dict với thống kê
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_files = sorted(crops_dir.glob("*.png"))
    if not crop_files:
        return {"total": 0, "success": 0, "failed": 0, "failed_files": []}

    success = 0
    failed_files = []

    for crop_path in crop_files:
        cleaned, debug_info = cleaner.clean(crop_path)

        if cleaned is not None:
            out_path = output_dir / crop_path.name
            cv2.imwrite(str(out_path), cleaned)
            success += 1
        else:
            # Fallback: resize ảnh gốc về target_size
            orig = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
            if orig is not None and orig.size > 0:
                h, w = orig.shape
                if h > 0 and w > 0:
                    resized = cv2.resize(
                        orig, (cleaner.target_size, cleaner.target_size)
                    )
                    out_path = output_dir / crop_path.name
                    cv2.imwrite(str(out_path), resized)
            failed_files.append(crop_path.name)
            if verbose:
                print(f"    [FAIL] {crop_path.name} (fallback to resize)")

    return {
        "total": len(crop_files),
        "success": success,
        "failed": len(failed_files),
        "failed_files": failed_files,
    }


# ---------------------------------------------------------------------------
# Directory-level processing
# ---------------------------------------------------------------------------

def process_detected_dir(
    detected_dir: Path,
    output_name: str = "crops_cleaned",
    size: int = 64,
    sauvola_k: float = 0.2,
    sauvola_window: int = 25,
    denoise: int = 3,
    min_stroke: int = 2,
    padding: int = 5,
    page_filter: int | None = None,
    verify: bool = False,
    verify_samples: int = 10,
    verbose: bool = True,
):
    """Xử lý toàn bộ thư mục detected."""
    crops_base = detected_dir / "crops"
    if not crops_base.exists():
        print(f"[ERROR] Không tìm thấy thư mục crops: {crops_base}", file=sys.stderr)
        return

    output_base = detected_dir / output_name
    output_base.mkdir(parents=True, exist_ok=True)

    cleaner = CharacterCleaner(
        target_size=size,
        padding=padding,
        sauvola_k=sauvola_k,
        sauvola_window=sauvola_window,
        sauvola_R=128.0,  # Cố định theo paper Sauvola gốc
        denoise_strength=denoise,
        min_stroke=min_stroke,
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
        print(f"  Method: Sauvola (k={sauvola_k}, R=128, w={sauvola_window})")
        print(f"  Size: {size}x{size}, Denoise: {denoise}, Min stroke: {min_stroke}")
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
            print(f"  Tỷ lệ thành công: {success_all / total_all:.1%}")
        print(f"  Output          : {output_base}/")

    # Save summary JSON
    summary = {
        "source": str(crops_base),
        "output": str(output_base),
        "settings": {
            "method": "sauvola",
            "sauvola_k": sauvola_k,
            "sauvola_window": sauvola_window,
            "sauvola_R": 128,
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


# ---------------------------------------------------------------------------
# Verification (before/after comparison)
# ---------------------------------------------------------------------------

def _save_verify_image(
    original_dir: Path, cleaned_dir: Path, verify_dir: Path, page_name: str
):
    """Tạo ảnh so sánh before/after cho verification."""
    verify_dir.mkdir(parents=True, exist_ok=True)

    orig_files = sorted(original_dir.glob("*.png"))[:12]  # 12 ký tự đầu
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

        # Resize original to match cleaned height
        h_clean = cleaned.shape[0]
        ratio = h_clean / orig.shape[0] if orig.shape[0] > 0 else 1
        new_w = max(1, int(orig.shape[1] * ratio))
        orig_resized = cv2.resize(
            orig, (new_w, h_clean),
            interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC,
        )

        # Separator
        sep = np.full((h_clean, 4), 180, dtype=np.uint8)
        row = np.hstack([orig_resized, sep, cleaned])
        rows.append(row)

    if not rows:
        return

    # Pad to same width
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1]), 255, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    # Row separators
    sep_h = np.full((3, max_w), 180, dtype=np.uint8)
    final_rows = []
    for i, r in enumerate(padded):
        if i > 0:
            final_rows.append(sep_h)
        final_rows.append(r)

    verify_img = np.vstack(final_rows)
    cv2.imwrite(str(verify_dir / f"{page_name}_verify.png"), verify_img)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Làm sạch ảnh ký tự Nôm (Giai đoạn 3: Sauvola + Morphological)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python clean_crops.py data/prepared/SachThanhTruyen2/detected
  python clean_crops.py data/prepared/SachThanhTruyen2/detected --size 64
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
        "--sauvola-k", type=float, default=0.2,
        help="Sauvola sensitivity k (default: 0.2)",
    )
    parser.add_argument(
        "--sauvola-window", type=int, default=25,
        help="Sauvola window size (default: 25, số lẻ)",
    )
    parser.add_argument(
        "--denoise", type=int, default=3,
        help="Median blur kernel size (số lẻ, default: 3)",
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
        sauvola_k=args.sauvola_k,
        sauvola_window=args.sauvola_window,
        denoise=args.denoise,
        min_stroke=args.min_stroke,
        padding=args.padding,
        page_filter=args.page,
        verify=args.verify,
        verify_samples=args.verify_samples,
    )


if __name__ == "__main__":
    main()
