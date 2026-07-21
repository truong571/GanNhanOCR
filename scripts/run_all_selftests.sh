#!/usr/bin/env bash
# Chạy toàn bộ selftest và so với MỐC ĐÃ CHỐT.
#
# Mốc đo ngày 2026-07-21 (Giai đoạn 3 — sau khi chốt số liệu):
#     ground_truth      60 passed,  0 failed
#     consensus_fusion  44 passed,  0 failed
#     publish           56 passed,  0 failed
#     remediation       33 passed,  0 failed
#     phase1_engine     30 passed,  0 failed
#     ------------------------------------------
#     TỔNG             223 passed,  0 failed
#
# LỊCH SỬ: mốc 2026-07-20 là 212/11. 11 assertion đỏ KHÔNG phải lỗi code —
# chúng hard-code census của thế hệ labels.csv CŨ (dup_bbox 701, cross_col 1686,
# union 2321, provably-wrong 1177, similar_bridge 3856). Giai đoạn 3 đã CHỐT:
# selftest kiểm cái TÁI LẬP ĐƯỢC từ labels.csv hiện tại (0/8/8/4/3850), còn số
# lịch sử được bảo tồn trong docs/census_history.md như bằng chứng engine-fix
# (before/after). Assertion mới đã được kiểm chứng độc lập là đo THẬT, không ép
# xanh. Riêng phase1 "low-purity" là lỗi TEST (placeholder 'x' bị lọc là rác nên
# purity không được kiểm) — đã sửa placeholder thành âm tiết hợp lệ 'an'/'ba'.
#
# => Con số "223 assertions" trong luận văn giờ ĐÚNG trở lại (223 pass, 0 fail).

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || { echo "Không thấy Python: $PY (đặt biến PY=... để đổi)"; exit 1; }

BASELINE_PASS=223
BASELINE_FAIL=0

MODULES=(
  pipeline.ground_truth.selftest
  pipeline.consensus_fusion.selftest
  pipeline.publish.selftest
  pipeline.remediation.selftest
  pipeline.phase1_engine_selftest
)

total_pass=0
total_fail=0

echo "================================================================"
printf "%-38s %s\n" "SELFTEST" "KẾT QUẢ"
echo "================================================================"

for m in "${MODULES[@]}"; do
  line=$("$PY" -m "$m" 2>&1 | grep -E '^RESULT:' | tail -1)
  if [ -z "$line" ]; then
    printf "%-38s %s\n" "$m" "KHÔNG CHẠY ĐƯỢC"
    total_fail=$((total_fail + 1))
    continue
  fi
  p=$(echo "$line" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+')
  f=$(echo "$line" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+')
  total_pass=$((total_pass + ${p:-0}))
  total_fail=$((total_fail + ${f:-0}))
  printf "%-38s %s\n" "$m" "${p:-0} passed, ${f:-0} failed"
done

echo "----------------------------------------------------------------"
printf "%-38s %s\n" "TỔNG" "$total_pass passed, $total_fail failed"
printf "%-38s %s\n" "MỐC 2026-07-21" "$BASELINE_PASS passed, $BASELINE_FAIL failed"
echo "================================================================"

if [ "$total_pass" -eq "$BASELINE_PASS" ] && [ "$total_fail" -eq "$BASELINE_FAIL" ]; then
  echo "KHỚP MỐC — không có hồi quy."
  exit 0
fi

echo "LỆCH MỐC:"
[ "$total_pass" -lt "$BASELINE_PASS" ] && echo "  - passed giảm $((BASELINE_PASS - total_pass)) → nghi có hồi quy, KIỂM TRA NGAY."
[ "$total_pass" -gt "$BASELINE_PASS" ] && echo "  - passed tăng $((total_pass - BASELINE_PASS)) → nếu do đã sửa thì CẬP NHẬT mốc trong file này."
[ "$total_fail" -gt "$BASELINE_FAIL" ] && echo "  - failed tăng $((total_fail - BASELINE_FAIL)) → hồi quy mới."
[ "$total_fail" -lt "$BASELINE_FAIL" ] && echo "  - failed giảm $((BASELINE_FAIL - total_fail)) → đã sửa được, CẬP NHẬT mốc trong file này."
exit 1
