#!/usr/bin/env bash
# =============================================================================
# clean_build.sh — DỰNG LẠI TỪ ĐẦU: xoá sạch mọi cache dẫn xuất + hash cũ, rồi
# chạy trọn run_pipeline.sh để ra bộ dữ liệu mới nhất.
#
# RANH GIỚI QUAN TRỌNG NHẤT CỦA TỆP NÀY
# --------------------------------------
# Có ĐÚNG MỘT loại "cache" trong dự án này KHÔNG ĐƯỢC XOÁ:
#
#     prepared/*/detected/*_ocr_cache.json          (16 MB, 445 tệp)
#     prepared/*/transcriptions/*_qn_ocr_cache.json (1,7 MB, 445 tệp)
#
# Đó KHÔNG phải cache theo nghĩa thông thường — đó là DỮ LIỆU GỐC mua bằng tiền
# thật từ API ngoài. Xoá nó thì: (a) phải trả tiền gọi lại, (b) kết quả KHÔNG tái
# lập được vì API bên ngoài có thể đã đổi. Cùng lý do đó, `prepared/*/pages/*.png`
# cũng bị đóng băng tuyệt đối: chính nó là ảnh mà `image_hash` trong cache neo vào.
#
# Script này CỐ Ý không có cờ nào xoá được hai thứ trên. Muốn OCR lại thì chạy
# `run_pipeline.sh` rồi chọn mục 2 và gõ tay chữ XOA — để việc tiêu tiền luôn là
# một hành động có ý thức.
#
# CÁI NÓ XOÁ (đều tự tái sinh, tổng ~700 MB)
# ------------------------------------------
#   dataset_out/{gold,silver,syllable}/   ảnh crop — kể cả 78k tệp MỒ CÔI tích tụ
#                                         qua nhiều lần chạy (53% thư mục)
#   dataset_out/*.csv, *.json             bộ nhãn + báo cáo của lần chạy trước
#   dataset_out/CHECKSUMS.txt             nhật ký hash cũ
#   dataset/                              bản xuất cuối (export tự dựng lại)
#   pipeline/align_engine/s3_proto_cache.pkl   nguyên mẫu S3 (ký bằng mtime)
#   lab/columns.pkl                       cache đầu vào bàn thí nghiệm
#   **/__pycache__/                       bytecode Python
#
#     bash scripts/clean_build.sh              # CHẠY THỬ — chỉ liệt kê, không xoá
#     bash scripts/clean_build.sh --yes        # xoá thật rồi chạy pipeline
#     bash scripts/clean_build.sh --yes --no-run   # chỉ dọn, không chạy pipeline
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; BLD=$'\033[1m'; RST=$'\033[0m'
DO_IT=0; RUN=1
for a in "$@"; do
  case "$a" in
    --yes) DO_IT=1 ;;
    --no-run) RUN=0 ;;
    *) echo "tham số lạ: $a"; exit 2 ;;
  esac
done

# ---- 1. CHỐT CHẶN: dữ liệu gốc phải còn nguyên TRƯỚC khi đụng vào gì --------
n_ocr=$(find prepared -name '*_ocr_cache.json' 2>/dev/null | wc -l | tr -d ' ')
n_png=$(find prepared -path '*/pages/*.png' 2>/dev/null | wc -l | tr -d ' ')
printf '%s--- DỮ LIỆU GỐC (script này KHÔNG đụng tới) ---%s\n' "$BLD" "$RST"
printf '  cache OCR        : %s tệp\n' "$n_ocr"
printf '  ảnh trang        : %s tệp\n' "$n_png"
if (( n_ocr < 800 || n_png < 400 )); then
  printf '%sDỪNG: dữ liệu gốc trông đã thiếu (cần >=890 cache, >=445 ảnh).%s\n' "$RED" "$RST"
  printf 'Khôi phục từ ~/backup_ocr_cache_2026-08-22/ trước khi chạy lại.\n'
  exit 1
fi
BK="$HOME/backup_ocr_cache_2026-08-22"
if [[ -f "$BK/SHA256SUMS.txt" ]]; then
  bad=$( (cd "$BK" && shasum -c SHA256SUMS.txt 2>/dev/null | grep -cv ': OK$') || true )
  printf '  đối chiếu sao lưu: %s tệp hỏng\n' "${bad:-?}"
  if [[ "${bad:-1}" != "0" ]]; then
    printf '%sDỪNG: cache OCR lệch bản sao lưu — điều tra trước khi dựng lại.%s\n' "$RED" "$RST"; exit 1
  fi
else
  printf '  %sKHÔNG thấy bản sao lưu %s — vẫn chạy được, nhưng nên sao lưu trước.%s\n' "$YEL" "$BK" "$RST"
fi

# ---- 2. Liệt kê thứ sẽ xoá --------------------------------------------------
TARGETS=(dataset_out/gold dataset_out/silver dataset_out/syllable dataset
         pipeline/align_engine/s3_proto_cache.pkl lab/columns.pkl)
printf '\n%s--- SẼ XOÁ (đều tự tái sinh) ---%s\n' "$BLD" "$RST"
total=0
for t in "${TARGETS[@]}"; do
  [[ -e "$t" ]] || continue
  sz=$(du -sk "$t" 2>/dev/null | cut -f1); total=$((total+sz))
  printf '  %-44s %6s MB\n' "$t" "$((sz/1024))"
done
n_csv=$(ls dataset_out/*.csv dataset_out/*.json 2>/dev/null | wc -l | tr -d ' ')
printf '  %-44s %6s tệp\n' "dataset_out/*.csv + *.json (nhãn + báo cáo)" "$n_csv"
n_pyc=$(find . -name __pycache__ -type d -not -path './.venv/*' 2>/dev/null | wc -l | tr -d ' ')
printf '  %-44s %6s thư mục\n' "**/__pycache__" "$n_pyc"
printf '  %s---> tổng ~%s MB%s\n' "$BLD" "$((total/1024))" "$RST"

if (( ! DO_IT )); then
  printf '\n%sCHẠY THỬ — chưa xoá gì.%s Chạy thật:\n    bash scripts/clean_build.sh --yes\n' "$YEL" "$RST"
  exit 0
fi

# ---- 3. Xoá -----------------------------------------------------------------
printf '\n%s--- ĐANG XOÁ ---%s\n' "$BLD" "$RST"
for t in "${TARGETS[@]}"; do [[ -e "$t" ]] && rm -rf "$t" && printf '  đã xoá %s\n' "$t"; done
rm -f dataset_out/*.csv dataset_out/*.json 2>/dev/null && printf '  đã xoá dataset_out/*.csv *.json\n'
find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null
printf '  đã xoá __pycache__\n'

# ---- 4. Chốt chặn LẦN HAI: chứng minh không hề đụng dữ liệu gốc ------------
n2=$(find prepared -name '*_ocr_cache.json' 2>/dev/null | wc -l | tr -d ' ')
p2=$(find prepared -path '*/pages/*.png' 2>/dev/null | wc -l | tr -d ' ')
printf '\n%s--- DỮ LIỆU GỐC SAU KHI DỌN ---%s\n' "$BLD" "$RST"
printf '  cache OCR %s -> %s | ảnh trang %s -> %s\n' "$n_ocr" "$n2" "$n_png" "$p2"
if [[ "$n_ocr" != "$n2" || "$n_png" != "$p2" ]]; then
  printf '%sLỖI NGHIÊM TRỌNG: dữ liệu gốc đã bị đụng. DỪNG.%s\n' "$RED" "$RST"; exit 1
fi
printf '  %snguyên vẹn%s\n' "$GRN" "$RST"

(( RUN )) || { printf '\n%sĐã dọn xong (--no-run).%s\n' "$GRN" "$RST"; exit 0; }

# ---- 5. Chạy trọn pipeline --------------------------------------------------
# Chạy từ BẢN SAO ĐÓNG BĂNG: bash đọc script theo từng đoạn và giữ con trỏ byte,
# nên sửa run_pipeline.sh giữa chừng sẽ làm nó đọc lệch. Đây là lỗi đã xảy ra thật.
FROZEN=$(mktemp -t run_pipeline.XXXXXX.sh)
cp run_pipeline.sh "$FROZEN"
trap 'rm -f "$FROZEN"' EXIT
printf '\n%s--- CHẠY PIPELINE (7 bước, dùng cache OCR, 0 đồng) ---%s\n' "$BLD" "$RST"
printf '4\n\n\n' | GANNHANOCR_ROOT="$PWD" bash "$FROZEN"
rc=$?

(( rc == 0 )) || { printf '%sPipeline hỏng (mã %s).%s\n' "$RED" "$rc" "$RST"; exit "$rc"; }

# ---- 6. Kiểm sau khi chạy ---------------------------------------------------
printf '\n%s--- KIỂM SAU KHI DỰNG LẠI ---%s\n' "$BLD" "$RST"
bash scripts/check_consistency.sh
