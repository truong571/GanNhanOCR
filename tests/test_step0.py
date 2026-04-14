"""Test Step 0: Setup & Validation.

Chạy: python tests/test_step0.py
Kiểm tra:
  - Config load OK
  - Thư mục output được tạo đúng cấu trúc
  - Các file dictionary tồn tại
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.step0_setup import load_config, validate_environment, create_directories

CONFIG_PATH = "config/pipeline.yaml"


def main():
    print("=" * 60)
    print("TEST STEP 0: Setup & Validation")
    print("=" * 60)

    # 1. Load config
    print("\n[1] Load config...")
    config = load_config(CONFIG_PATH)
    print(f"    Books: {[b['name'] for b in config['books']]}")
    print(f"    Data dir: {config['paths']['data_dir']}")
    print(f"    Output dir: {config['paths']['output_dir']}")
    print("    OK")

    # 2. Validate environment
    print("\n[2] Validate environment...")
    ok = validate_environment(config)
    print(f"    Result: {'PASS' if ok else 'FAIL'}")

    # 3. Create directories
    print("\n[3] Create directories...")
    create_directories(config)

    # Verify directory structure
    data_dir = Path(config["paths"]["data_dir"])
    expected_subdirs = [
        "pages", "pages_denoised", "transcriptions",
        "detected/crops", "detected/crops_cleaned",
        "aligned", "labeled",
    ]
    all_ok = True
    for book in config["books"]:
        book_dir = data_dir / book["name"]
        for subdir in expected_subdirs:
            p = book_dir / subdir
            if p.exists():
                print(f"    OK  {p}")
            else:
                print(f"    MISSING  {p}")
                all_ok = False

    print(f"\n    Directory check: {'PASS' if all_ok else 'FAIL'}")
    print("\n" + "=" * 60)
    print(f"STEP 0 RESULT: {'PASS' if ok and all_ok else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
