#!/usr/bin/env bash
# =============================================================================
# check_consistency.sh — MỘT lệnh kiểm mọi thứ phải khớp với bộ nhãn trên đĩa.
#
# VÌ SAO CÓ TỆP NÀY
#   KHỐI 2 viết lại docs/BANG_SO_LIEU_CHINH_THUC.md để diệt lớp lệch "tài liệu nói
#   một đằng, đĩa một nẻo". Rồi T1 và T2 đổi bộ nhãn hai lần và tài liệu LỆCH LẠI
#   NGAY — vì không có gì ÉP kiểm. Cùng lúc, index.csv của crop-proto vẫn trỏ thế hệ
#   sách cũ (yen*) suốt nhiều tháng mà preflight báo xanh vì "tệp vẫn tồn tại".
#
#   Bài học: mỗi lần phát hiện một lớp lệch thì phải để lại MỘT LỆNH bắt được nó,
#   nếu không lần sau vẫn lệch y hệt.
#
#   Chạy TRƯỚC MỖI COMMIT có đụng bộ nhãn:
#       bash scripts/check_consistency.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON_BIN:-.venv/bin/python}"

fail=0
run() {                       # run <mô tả> <lệnh...>
  local desc="$1"; shift
  printf '\n=== %s ===\n' "$desc"
  if "$@"; then :; else fail=$((fail + 1)); fi
}

run "1/4 Bằng chứng SHA256 (bộ đem đo = bộ đem nộp)" \
    bash scripts/check_evidence.sh

run "2/4 Bảng số liệu chính thức khớp đĩa" \
    "$PY" -m pipeline.tools.update_bang_so_lieu --check

run "3/4 Chỉ mục crop-proto cùng thế hệ với bộ nhãn" \
    "$PY" -m pipeline.tools.rebuild_proto_index --check

run "4/4 Tái lập: cây làm việc · cache nguyên mẫu · không RNG (T6)" \
    "$PY" -m pipeline.tools.repro_check --check

echo
echo "================================================================"
if (( fail )); then
  echo "KHÔNG NHẤT QUÁN: $fail/4 phép kiểm hỏng — sửa trước khi commit."
  exit 1
fi
echo "NHẤT QUÁN: 4/4 phép kiểm khớp bộ nhãn trên đĩa."
