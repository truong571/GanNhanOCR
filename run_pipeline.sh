#!/usr/bin/env bash
# GanNhanOCR Pipeline — 5 steps (0-4)
#
# Usage:
#   ./run_pipeline.sh                    # Run all steps for all books
#   ./run_pipeline.sh --step 1           # Run only step 1
#   ./run_pipeline.sh --book SachThanhTruyen2  # Run only one book
#   ./run_pipeline.sh --config config/pipeline.yaml

set -euo pipefail

# Pin to the project venv — never fall back to system python3. All pipeline
# dependencies (VietOCR, FontDiffusion, DINOv2, torch, ...) live there.
PY="$(cd "$(dirname "$0")" && pwd)/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[pipeline] .venv not found at $PY" >&2
    echo "[pipeline] create it with: python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

# Load .env (SN_OCR_TOKEN, GEMINI_API_KEY, ...) so child processes see them
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# Sanity-check Python env before doing anything expensive
"$PY" -c "
import sys
miss = []
for m in ('fitz', 'cv2', 'numpy', 'pandas', 'yaml', 'PIL', 'vietocr'):
    try: __import__(m)
    except ImportError: miss.append(m)
if miss:
    print(f'[pipeline] missing modules in venv: {miss}', file=sys.stderr); sys.exit(1)
print(f'[pipeline] using {sys.executable} (Python {sys.version.split()[0]})')
" || exit 1

CONFIG="${CONFIG:-config/pipeline.yaml}"
STEP="${STEP:-all}"
BOOK="${BOOK:-all}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --step)   STEP="$2"; shift 2 ;;
        --book)   BOOK="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--config PATH] [--step N] [--book NAME]"
            echo ""
            echo "Steps:"
            echo "  0     Setup & validation"
            echo "  1     Extract data from PDF"
            echo "  2     Levenshtein alignment"
            echo "  3     3-tier label assignment"
            echo "  4     Export dataset"
            echo "  all   Run all steps (default)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Extract book names from config
if [[ "$BOOK" == "all" ]]; then
    BOOKS=$("$PY" -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
for b in cfg['books']:
    print(b['name'])
")
else
    BOOKS="$BOOK"
fi

echo "================================================================"
echo "  GanNhanOCR Pipeline"
echo "  Config: $CONFIG"
echo "  Steps:  $STEP"
echo "  Books:  $(echo $BOOKS | tr '\n' ' ')"
echo "================================================================"

# Step 0: Setup
if [[ "$STEP" == "all" || "$STEP" == "0" ]]; then
    echo ""
    echo ">>> Step 0: Setup & Validation"
    "$PY" -m pipeline.step0_setup "$CONFIG"
fi

# Step 1: Extract
if [[ "$STEP" == "all" || "$STEP" == "1" ]]; then
    for book in $BOOKS; do
        echo ""
        echo ">>> Step 1: Extract — $book"
        "$PY" -m pipeline.step1_extract "$CONFIG" "$book"
    done
fi

# Step 2: Align
if [[ "$STEP" == "all" || "$STEP" == "2" ]]; then
    for book in $BOOKS; do
        echo ""
        echo ">>> Step 2: Align — $book"
        "$PY" -m pipeline.step2_align "$CONFIG" "$book"
    done
fi

# Step 3: Label
if [[ "$STEP" == "all" || "$STEP" == "3" ]]; then
    for book in $BOOKS; do
        echo ""
        echo ">>> Step 3: Label — $book"
        "$PY" -m pipeline.step3_label "$CONFIG" "$book"
    done
fi

# Step 4: Export
if [[ "$STEP" == "all" || "$STEP" == "4" ]]; then
    echo ""
    echo ">>> Step 4: Export Dataset"
    "$PY" -m pipeline.step4_export "$CONFIG"
fi

echo ""
echo "================================================================"
echo "  Pipeline complete!"
echo "================================================================"
