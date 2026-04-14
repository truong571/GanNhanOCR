"""Step 0: Load config, validate environment, create output directories."""

import argparse
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load and validate pipeline.yaml configuration."""
    path = Path(config_path)
    if not path.exists():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate required keys
    required = ["books", "paths"]
    for key in required:
        if key not in config:
            print(f"[ERROR] Missing required config key: {key}", file=sys.stderr)
            sys.exit(1)

    return config


def validate_environment(config: dict) -> bool:
    """Check that all required files and dependencies exist."""
    paths = config["paths"]
    ok = True

    # Check dictionaries
    for key in ["qn_to_nom_dict", "similar_dict"]:
        p = Path(paths[key])
        if not p.exists():
            print(f"[ERROR] Dictionary not found: {p}", file=sys.stderr)
            ok = False

    # Check font
    font_path = Path(paths["font_path"])
    if not font_path.exists():
        print(f"[WARN] Font not found: {font_path} (visual ranking disabled)")

    # Check PDF files
    for book in config["books"]:
        pdf_path = Path(book["pdf"])
        if not pdf_path.exists():
            print(f"[WARN] PDF not found: {pdf_path}")

    return ok


def create_directories(config: dict):
    """Create output directory structure for all books."""
    data_dir = Path(config["paths"]["data_dir"])
    output_dir = Path(config["paths"]["output_dir"])

    for book in config["books"]:
        book_dir = data_dir / book["name"]
        for subdir in ["pages", "pages_denoised", "transcriptions",
                        "detected/crops", "detected/crops_cleaned",
                        "aligned", "labeled"]:
            (book_dir / subdir).mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)


def run(config_path: str):
    """Execute Step 0."""
    config = load_config(config_path)

    print("=" * 60)
    print("Step 0: Setup & Validation")
    print("=" * 60)

    # Validate
    if not validate_environment(config):
        print("\n[ERROR] Validation failed. Fix issues above.", file=sys.stderr)
        sys.exit(1)

    # Create directories
    create_directories(config)

    print(f"  Books: {len(config['books'])}")
    for book in config["books"]:
        print(f"    - {book['name']}")
    print(f"  Data dir:   {config['paths']['data_dir']}")
    print(f"  Output dir: {config['paths']['output_dir']}")
    print("  Directories created.")
    print("  Validation passed.\n")

    return config


def main():
    parser = argparse.ArgumentParser(description="Step 0: Setup pipeline")
    parser.add_argument("config", type=str, help="Path to pipeline.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
