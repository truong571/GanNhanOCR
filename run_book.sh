#!/usr/bin/env bash
# GanNhanOCR — engine chạy pipeline cho ĐÚNG 1 sách. Dùng qua run_book2/4/11.sh
# (hoặc trực tiếp: ./run_book.sh SachThanhTruyen2 [options]).
#
# 2 MODE cache:
#   --fresh   XOÁ cache cũ (rm -rf prepared/<sách>) rồi chạy lại từ đầu (re-render + re-OCR)
#   --keep    GIỮ cache cũ, tái dùng (mặc định — nhanh, không tốn API)
#
# Usage:
#   ./run_book.sh SachThanhTruyen2                 # giữ cache, cả 3 bước
#   ./run_book.sh SachThanhTruyen4 --fresh         # xoá cache, chạy lại từ đầu
#   ./run_book.sh SachThanhTruyen11 --step 1 --keep
#   ./run_book.sh SachThanhTruyen2 --config config/pipeline_today.yaml --book SachThanhTruyen2_c01
#   ./run_book.sh SachThanhTruyen2 --qwen --qwen-n 5   # + Qwen eval (xoay key .env)

set -euo pipefail
cd "$(dirname "$0")"

BOOK=""
CONFIG="${CONFIG:-config/pipeline.yaml}"
STEP="${STEP:-all}"
MODE="keep"                       # keep | fresh
RUN_QWEN=0
QWEN_N="${QWEN_N:-5}"
QWEN_MODEL="${QWEN_MODEL:-qwen3-vl-235b}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --book)       BOOK="$2"; shift 2 ;;
        --config)     CONFIG="$2"; shift 2 ;;
        --step)       STEP="$2"; shift 2 ;;
        --fresh)      MODE="fresh"; shift ;;
        --keep)       MODE="keep"; shift ;;
        --qwen)       RUN_QWEN=1; shift ;;
        --qwen-n)     QWEN_N="$2"; shift 2 ;;
        --qwen-model) QWEN_MODEL="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 <BOOK> [--fresh|--keep] [--step 0|1|2|all] [--config PATH]"
            echo "          [--book NAME] [--qwen [--qwen-n N] [--qwen-model M]]"
            echo ""
            echo "  --fresh  XOÁ prepared/<sách> rồi chạy lại từ đầu (re-OCR)"
            echo "  --keep   GIỮ cache, tái dùng (mặc định)"
            exit 0 ;;
        -*)  echo "Unknown option: $1" >&2; exit 1 ;;
        *)   if [[ -z "$BOOK" ]]; then BOOK="$1"; shift
             else echo "Chỉ nhận 1 tên sách (đã có '$BOOK', gặp '$1')" >&2; exit 1; fi ;;
    esac
done
[[ -n "$BOOK" ]] || { echo "[book] cần tên sách: $0 <BOOK> [options]" >&2; exit 1; }

# ---- venv + .env (chỉ source biến tên hợp lệ; Qwen3-VL1/2 để Python tự đọc) --
PY="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "[book] Python không thấy ở: $PY" >&2
    echo "[book] Tạo venv: python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi
if [[ -f .env ]]; then
    set -a
    source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env) 2>/dev/null || true
    set +a
fi

"$PY" -c "
import sys, importlib.util
miss = [m for m in ('fitz','cv2','numpy','pandas','yaml','PIL','vietocr')
        if importlib.util.find_spec(m) is None]
if miss: print(f'[book] thiếu module trong venv: {miss}', file=sys.stderr); sys.exit(1)
print(f'[book] {sys.executable} (Python {sys.version.split()[0]})')
" || exit 1

# xác nhận sách có trong config
"$PY" -c "
import yaml, sys
names = [b['name'] for b in yaml.safe_load(open('$CONFIG'))['books']]
if '$BOOK' not in names:
    print(f'[book] \'$BOOK\' không có trong $CONFIG (có: {names})', file=sys.stderr); sys.exit(1)
" || exit 1

# Config CHỈ-1-SÁCH: step0_setup + build_dataset (step 2) duyệt TẤT CẢ books trong
# config; nếu không lọc, chạy 'book2' vẫn setup/build cả 3. Lọc ở đây để mọi bước
# chỉ đụng đúng $BOOK (các paths khác giữ nguyên).
EFFCONFIG="$(mktemp -t run_book).yaml"
trap 'rm -f "$EFFCONFIG"' EXIT
"$PY" -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
cfg['books'] = [b for b in cfg['books'] if b['name'] == '$BOOK']
yaml.safe_dump(cfg, open('$EFFCONFIG','w'), allow_unicode=True, sort_keys=False)
" || exit 1

echo "================================================================"
echo "  GanNhanOCR — sách $BOOK  (config lọc chỉ 1 sách)"
echo "  Config: $CONFIG   Steps: $STEP   Cache: $MODE   Qwen: $RUN_QWEN"
echo "================================================================"

# ---- MODE fresh: xoá cache cũ trước khi extract ----------------------------
if [[ "$MODE" == "fresh" && ( "$STEP" == "all" || "$STEP" == "1" ) ]]; then
    if [[ -n "$BOOK" && -d "prepared/$BOOK" ]]; then
        sz=$(du -sh "prepared/$BOOK" 2>/dev/null | awk '{print $1}')
        echo ">>> [fresh] Xoá cache cũ: prepared/$BOOK ($sz)"
        rm -rf "prepared/$BOOK"
    else
        echo ">>> [fresh] prepared/$BOOK chưa có — chạy sạch từ đầu"
    fi
fi

# ---- Step -1: FD cache (cần cho step 2) ------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "2" ]]; then
    FD_DIR=$("$PY" -c "import yaml;print(yaml.safe_load(open('$EFFCONFIG'))['paths']['fd_cache_universal'])")
    FD_N=$(find "$FD_DIR" -name 'U+*.png' 2>/dev/null | wc -l | tr -d ' ')
    echo ">>> FD cache: $FD_DIR có $FD_N glyph"
    [[ "$FD_N" -gt 0 ]] || echo "    [CẢNH BÁO] FD trống — S3 (step 2) thiếu glyph tham chiếu." >&2
fi

# ---- Step 0 ----------------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "0" ]]; then
    echo ""; echo ">>> Step 0: Setup & Validation"
    "$PY" -m pipeline.step0_setup "$EFFCONFIG"
fi

# ---- Step 1: Extract -------------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "1" ]]; then
    echo ""; echo ">>> Step 1: Extract — $BOOK"
    "$PY" -m pipeline.step1_extract "$EFFCONFIG" "$BOOK"
    echo ">>> Kiểm số cột OCR (mong đợi 9 cột/trang)"
    "$PY" pipeline/check_ocr_columns.py --book "$BOOK" 2>/dev/null \
        | grep -E "OK\(=9\)|THIẾU|DƯ|≠9" || true
fi

# ---- Step 2: Build dataset -------------------------------------------------
if [[ "$STEP" == "all" || "$STEP" == "2" ]]; then
    echo ""; echo ">>> Step 2: Build dataset ($BOOK) | reseg=${RESEG:-detector}"
    "$PY" -m pipeline.align_engine.build_dataset --config "$EFFCONFIG" --use-s3 --reseg "${RESEG:-detector}"
    echo ">>> Step 2b: Export standards"
    "$PY" -m pipeline.align_engine.to_standard
fi

# ---- (tuỳ chọn) Qwen eval, xoay key ----------------------------------------
if [[ "$RUN_QWEN" == "1" ]]; then
    echo ""; echo ">>> Qwen eval — $BOOK ($QWEN_N trang, $QWEN_MODEL, xoay key .env)"
    "$PY" evaluation/tri_consensus/dump_qwen.py \
        --book "$BOOK" --n "$QWEN_N" --model "$QWEN_MODEL" --out qwen_cache_235b
fi

echo ""
echo "================================================================"
echo "  Xong sách $BOOK  (cache mode: $MODE)"
echo "================================================================"
