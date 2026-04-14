"""Test Step 1: Extract — chạy 1 book, kiểm tra output thủ công.

Chạy: python tests/test_step1.py [book_name]
  Mặc định: CacThanhTruyen2

Kiểm tra bằng mắt:
  1. pages/ có ảnh gốc không?
  2. pages_denoised/ có ảnh khử nhiễu không?
  3. Ảnh trong crops/ là crop từ ảnh GỐC (không qua xử lý)?
  4. Ảnh trong crops_cleaned/ là ảnh đã Sauvola + cleanup?
  5. transcriptions/*.txt có nội dung Quốc Ngữ đúng không?
  6. detected/*_detection.json có bbox hợp lý không?
  7. OCR cache dùng ảnh denoised (kiểm tra log)?
"""

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.step0_setup import load_config
from pipeline.step1_extract import process_book

CONFIG_PATH = "config/pipeline.yaml"
BOOK = sys.argv[1] if len(sys.argv) > 1 else "CacThanhTruyen2"


def check_outputs(data_dir: Path):
    """Kiểm tra output Step 1 và in báo cáo."""
    print("\n" + "=" * 60)
    print("STEP 1 OUTPUT CHECK")
    print("=" * 60)

    # 1. Pages (original images)
    pages = sorted((data_dir / "pages").glob("*.png"))
    print(f"\n[1] Original images (pages/): {len(pages)} files")
    for p in pages[:3]:
        img = cv2.imread(str(p))
        if img is not None:
            print(f"    {p.name}: {img.shape[1]}x{img.shape[0]}")
        else:
            print(f"    {p.name}: CANNOT READ")

    # 2. Denoised images
    denoised = sorted((data_dir / "pages_denoised").glob("*.png"))
    print(f"\n[2] Denoised images (pages_denoised/): {len(denoised)} files")
    if pages and denoised:
        orig = cv2.imread(str(pages[0]), cv2.IMREAD_GRAYSCALE)
        dn = cv2.imread(str(denoised[0]), cv2.IMREAD_GRAYSCALE)
        if orig is not None and dn is not None:
            diff = cv2.absdiff(orig, dn).mean()
            print(f"    Mean pixel diff (orig vs denoised): {diff:.1f}")
            if diff < 1:
                print("    WARNING: original and denoised are nearly identical!")
            else:
                print("    OK: denoised differs from original")

    # 3. Transcriptions
    txts = sorted((data_dir / "transcriptions").glob("*.txt"))
    print(f"\n[3] Transcriptions: {len(txts)} files")
    for t in txts[:2]:
        lines = t.read_text(encoding="utf-8").strip().split("\n")
        print(f"    {t.name}: {len(lines)} columns")
        for i, line in enumerate(lines[:2]):
            syls = line.split()
            print(f"      Col {i+1}: {len(syls)} syllables — {line[:80]}...")

    # 4. Detection JSONs
    dets = sorted((data_dir / "detected").glob("*_detection.json"))
    print(f"\n[4] Detection files: {len(dets)} files")
    for d in dets[:2]:
        with open(d) as f:
            det = json.load(f)
        print(f"    {d.name}: {det['num_columns']} cols, {det['total_chars']} chars")

    # 5. Crops — verify they come from ORIGINAL image
    crops_dir = data_dir / "detected" / "crops"
    cleaned_dir = data_dir / "detected" / "crops_cleaned"
    crop_files = list(crops_dir.rglob("*.png"))
    cleaned_files = list(cleaned_dir.rglob("*.png"))
    print(f"\n[5] Crops (original): {len(crop_files)} files")
    print(f"    Crops (cleaned):  {len(cleaned_files)} files")

    if crop_files and cleaned_files:
        c1 = cv2.imread(str(crop_files[0]), cv2.IMREAD_GRAYSCALE)
        c2 = cv2.imread(str(cleaned_files[0]), cv2.IMREAD_GRAYSCALE)
        if c1 is not None and c2 is not None:
            print(f"    Sample crop size:    {c1.shape}")
            print(f"    Sample cleaned size: {c2.shape}")
            if c1.shape == c2.shape:
                diff = cv2.absdiff(c1, c2).mean()
                if diff < 1:
                    print("    WARNING: crop and cleaned look identical!")
                else:
                    print(f"    OK: crop vs cleaned differ (mean diff={diff:.1f})")
            else:
                print("    OK: different sizes (cleaned is resized)")

    # 6. OCR cache — verify it used denoised image
    caches = sorted((data_dir / "detected").glob("*_ocr_cache.json"))
    print(f"\n[6] OCR caches: {len(caches)} files")
    for c in caches[:2]:
        with open(c) as f:
            cache = json.load(f)
        img_used = cache.get("image", "")
        is_denoised = "denoised" in img_used or "pages_denoised" in img_used
        n_cols = len(cache.get("columns", []))
        total = sum(len(col) for col in cache.get("columns", []))
        status = "OK (denoised)" if is_denoised else "WARNING: used original"
        print(f"    {c.name}: {n_cols} cols, {total} chars — {status}")
        print(f"      Image: {img_used}")

    print("\n" + "=" * 60)
    print("Review the above output manually.")
    print("Open pages/, pages_denoised/, crops/ in a file viewer to verify visually.")
    print("=" * 60)


def main():
    config = load_config(CONFIG_PATH)

    print("=" * 60)
    print(f"TEST STEP 1: Extract — {BOOK}")
    print("=" * 60)

    data_dir = Path(config["paths"]["data_dir"]) / BOOK
    process_book(config, BOOK, verbose=True)
    check_outputs(data_dir)


if __name__ == "__main__":
    main()
