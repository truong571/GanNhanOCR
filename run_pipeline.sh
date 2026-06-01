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
PY="${PYTHON_BIN:-$(cd "$(dirname "$0")" && pwd)/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "[pipeline] Python not found at: $PY" >&2
    echo "[pipeline] Tạo venv:  python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    echo "[pipeline] hoặc override: PYTHON_BIN=/path/to/python ./run_pipeline.sh ..." >&2
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
FRESH="${FRESH:-0}"   # 1 = xoá cache OCR cũ (số cột sai) trước khi extract lại

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --step)   STEP="$2"; shift 2 ;;
        --book)   BOOK="$2"; shift 2 ;;
        --fresh)  FRESH=1; shift ;;
        --help)
            echo "Usage: $0 [--config PATH] [--step N] [--book NAME] [--fresh]"
            echo ""
            echo "Steps:"
            echo "  0     Setup & validation"
            echo "  1     Extract data from PDF (crop khung -> kinhhannom -> 9 cột)"
            echo "  2     Build dataset (ver_new: banded-DP align + consensus tiers"
            echo "        + re-segment + bbox-fix -> crops + labels.csv + 3 standards)"
            echo "  all   Run all steps (default)"
            echo ""
            echo "Options:"
            echo "  --fresh   Xoá cache OCR cũ (detected/*_ocr_cache.json) trước"
            echo "            khi chạy step 1 -> OCR lại với code mới nhất (9 cột)"
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

# Step -1: Kiểm FontDiffusion cache LOCAL (data đã sinh sẵn ở gannhanocr-fd/,
# tra theo Unicode — KHÔNG sinh/sync gì). Chỉ cần cho step 3/4.
if [[ "$STEP" == "all" || "$STEP" == "3" || "$STEP" == "4" ]]; then
    echo ""
    echo ">>> Step -1: Kiểm FD cache local (gannhanocr-fd, tra theo Unicode)"
    FD_DIR=$("$PY" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['paths']['fd_cache_universal'])")
    FD_N=$(find "$FD_DIR" -name 'U+*.png' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$FD_N" -gt 0 ]]; then
        echo "    OK: $FD_DIR có $FD_N glyph U+*.png"
    else
        echo "    [LỖI] $FD_DIR trống/không có — step 3 sẽ thiếu tier-3." >&2
        echo "          Đặt data đã sinh vào $FD_DIR/ (dạng <hex>/U+XXXX.png)." >&2
    fi
fi

# Step 0: Setup
if [[ "$STEP" == "all" || "$STEP" == "0" ]]; then
    echo ""
    echo ">>> Step 0: Setup & Validation"
    "$PY" -m pipeline.step0_setup "$CONFIG"
fi

# --fresh: xoá cache OCR cũ (số cột không chính xác) -> step 1 sẽ OCR lại với
# code mới nhất (crop khung + 5 luật gộp cột + retry pad -> đúng 9 cột).
if [[ "$FRESH" == "1" && ( "$STEP" == "all" || "$STEP" == "1" ) ]]; then
    echo ""
    echo ">>> [--fresh] Xoá cache OCR cũ (detected/*_ocr_cache.json)"
    for book in $BOOKS; do
        d="prepared/$book/detected"
        n=$(find "$d" -name "*_ocr_cache.json" 2>/dev/null | wc -l | tr -d ' ')
        find "$d" -name "*_ocr_cache.json" -delete 2>/dev/null || true
        echo "    $book: đã xoá $n cache OCR"
    done
fi

# Step 1: Extract
if [[ "$STEP" == "all" || "$STEP" == "1" ]]; then
    for book in $BOOKS; do
        echo ""
        echo ">>> Step 1: Extract — $book"
        "$PY" -m pipeline.step1_extract "$CONFIG" "$book"
    done
    # Kiểm số cột OCR (mong đợi 9 cột/trang) — đọc từ cache vừa tạo
    echo ""
    echo ">>> Kiểm số cột OCR sau extract (9 cột/trang)"
    for book in $BOOKS; do
        "$PY" evaluation/check_ocr_columns.py --book "$book" 2>/dev/null \
            | grep -E "OK\(=9\)|THIẾU|DƯ|≠9" || true
    done
fi

# Step 2: Build dataset (ver_new). Banded dict-anchored alignment + 3-signal
# consensus tiers (GOLD/SYLLABLE/REVIEW) + column re-segment + frame-bbox fix
# -> crops + labels.csv + 3 international standards. SUPERSEDES the old
# step2-align(index) / step3-label(DINOv2, proven non-discriminative) /
# step4-export path; those module files are kept for reference but no longer run.
# See evaluation/ver_new/FLOW.md.
if [[ "$STEP" == "all" || "$STEP" == "2" ]]; then
    echo ""
    echo ">>> Step 2: Build dataset (ver_new: align + consensus + crops + labels)"
    "$PY" evaluation/ver_new/build_dataset.py --config "$CONFIG"
    echo ""
    echo ">>> Step 2b: Export standards (HF imagefolder + Frictionless + Croissant)"
    "$PY" evaluation/ver_new/to_standard.py
fi

echo ""
echo "================================================================"
echo "  Pipeline complete!"
echo "================================================================"
