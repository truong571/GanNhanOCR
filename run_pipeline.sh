#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh - Chạy toàn bộ pipeline cho 3 bộ CacThanhTruyen (2, 4, 11)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
BOOKS=("CacThanhTruyen2" "CacThanhTruyen4" "CacThanhTruyen11")
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USE_PADDLE="${USE_PADDLE:-0}"  # Set USE_PADDLE=1 to use PaddleOCR hybrid detection

# ---------------------------------------------------------------------------
# Màu terminal
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_stage() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}  GIAI ĐOẠN $1: $2${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

log_book() {
    echo -e "${BOLD}  ▸ $1${NC}"
}

elapsed_time() {
    local diff=$(( $(date +%s) - $1 ))
    echo "$((diff / 60))m $((diff % 60))s"
}

# ---------------------------------------------------------------------------
# Kiểm tra file PDF
# ---------------------------------------------------------------------------
for BOOK in "${BOOKS[@]}"; do
    if [[ ! -f "data/${BOOK}.pdf" ]]; then
        echo -e "${RED}[ERROR]${NC} Không tìm thấy: data/${BOOK}.pdf"
        exit 1
    fi
done

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║     PIPELINE GÁN NHÃN: CacThanhTruyen (2, 4, 11)           ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"

PIPELINE_START=$(date +%s)

# =====================================================================
# GIAI ĐOẠN 1: TRÍCH XUẤT DỮ LIỆU TỪ PDF
# =====================================================================
log_stage 1 "TRÍCH XUẤT DỮ LIỆU TỪ PDF"
t=$(date +%s)

for BOOK in "${BOOKS[@]}"; do
    log_book "$BOOK"
    "$PYTHON" "$SCRIPT_DIR/prepare_data.py" "data/${BOOK}.pdf" \
        --dpi 300 \
        --denoise
done

echo -e "${GREEN}[OK]${NC} Giai đoạn 1 hoàn tất ($(elapsed_time $t))"

# =====================================================================
# GIAI ĐOẠN 2: PHÁT HIỆN VÀ CẮT KÝ TỰ
# =====================================================================
log_stage 2 "PHÁT HIỆN VÀ CẮT KÝ TỰ"
t=$(date +%s)

PADDLE_FLAG=""
if [[ "$USE_PADDLE" == "1" ]]; then
    PADDLE_FLAG="--paddle"
    echo -e "  ${CYAN}Mode: PaddleOCR Hybrid${NC}"
fi

for BOOK in "${BOOKS[@]}"; do
    log_book "$BOOK"
    "$PYTHON" "$SCRIPT_DIR/detect_characters.py" "data/prepared/${BOOK}" $PADDLE_FLAG
done

echo -e "${GREEN}[OK]${NC} Giai đoạn 2 hoàn tất ($(elapsed_time $t))"

# =====================================================================
# GIAI ĐOẠN 3: LÀM SẠCH ẢNH KÝ TỰ
# =====================================================================
log_stage 3 "LÀM SẠCH ẢNH KÝ TỰ"
t=$(date +%s)

for BOOK in "${BOOKS[@]}"; do
    log_book "$BOOK"
    "$PYTHON" "$SCRIPT_DIR/clean_crops.py" "data/prepared/${BOOK}/detected" \
        --size 64
done

echo -e "${GREEN}[OK]${NC} Giai đoạn 3 hoàn tất ($(elapsed_time $t))"

# =====================================================================
# GIAI ĐOẠN 4: GÁN NHÃN TỰ ĐỘNG
# =====================================================================
log_stage 4 "GÁN NHÃN TỰ ĐỘNG"
t=$(date +%s)

for BOOK in "${BOOKS[@]}"; do
    log_book "$BOOK"
    "$PYTHON" "$SCRIPT_DIR/label_characters.py" "data/prepared/${BOOK}" \
        --review \
        --ocr \
        --dinov2 \
        --excel
done

echo -e "${GREEN}[OK]${NC} Giai đoạn 4 hoàn tất ($(elapsed_time $t))"

# =====================================================================
# GIAI ĐOẠN 5: XUẤT DATASET (gộp cả 3 bộ)
# =====================================================================
log_stage 5 "XUẤT DATASET (gộp 3 bộ)"
t=$(date +%s)

"$PYTHON" "$SCRIPT_DIR/export_dataset.py" \
    data/prepared/CacThanhTruyen2/labeled \
    data/prepared/CacThanhTruyen4/labeled \
    data/prepared/CacThanhTruyen11/labeled \
    --min-confidence low \
    --split 0.8 0.1 0.1

echo -e "${GREEN}[OK]${NC} Giai đoạn 5 hoàn tất ($(elapsed_time $t))"

# ---------------------------------------------------------------------------
# Tổng kết
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║              PIPELINE HOÀN TẤT!                             ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo -e "${CYAN}  Bộ sách:       CacThanhTruyen2, CacThanhTruyen4, CacThanhTruyen11${NC}"
echo -e "${CYAN}  Tổng thời gian: $(elapsed_time $PIPELINE_START)${NC}"
echo -e "${CYAN}  Output:        data/prepared/CacThanhTruyen*/${NC}"
echo -e "${CYAN}  Dataset:       dataset/${NC}"
echo ""
