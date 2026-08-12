#!/usr/bin/env bash
# Chạy toàn bộ selftest và so với MỐC ĐÃ CHỐT.
#
# Mốc đo ngày 2026-08-11 (chốt bộ nhãn công bố + phủ test cho bảng kết quả audit):
#     parser (bước 2)          18 passed,  0 failed
#     syllable_validation      39 passed,  0 failed
#     ground_truth            170 passed,  0 failed   <- +32 cho report_combined
#     consensus_fusion         44 passed,  0 failed
#     publish                  56 passed,  0 failed
#     remediation              35 passed,  0 failed
#     phase1_engine            30 passed,  0 failed
#     -------------------------------------------
#     TỔNG                    392 passed,  0 failed
#
# ĐỔI SO VỚI MỐC 2026-08-03 (360/0):
#   +32  report_combined — module sinh BẢNG HEADLINE của luận văn (precision/CI/acceptance
#        theo tier + κ nội tại + κ liên người) trước đây KHÔNG có một assertion nào.
#        Nay kiểm: ô lặp không lọt vào precision, "không đọc được" bị loại khỏi mẫu số,
#        trọng số lấy từ stratum_N (không phải trung bình cộng), Horvitz–Thompson toàn tập,
#        κ khớp giá trị tính tay 5/9 theo cả hai hướng ma trận, verdict lạc mẻ khác bị chặn,
#        và mẻ thiếu orig_verdict không lặng lẽ trả κ vô nghĩa.
#
# Mốc trước đó — 2026-08-03 (bước A — chuẩn bị audit người), TỔNG 360:
#     ground_truth 138 (+77 cho code bước A-B), các module khác như bảng trên.
#
# ĐỔI SO VỚI MỐC 2026-07-21 (223/0):
#   +57  hai selftest bước 1-2 (core.pdf.parser, core.text.syllable_validation) TRƯỚC ĐÂY
#        BỊ BỎ NGOÀI runner dù vẫn xanh — nên con số "223 assertions" đã bỏ sót bước 1-2.
#   +3   assertion CẤU TRÚC mới (union = hợp 2 lớp con ×2, quarantine ⊆ lớp trùng ×1).
#   +38  test cho code mới của bước A: s3_signals (gắn tín hiệu + chống lệch thế hệ),
#        make_gold_batch (mẻ hai tầng + làm mù), estimate (LOẠI mẫu chủ đích khỏi
#        precision), nạp verdict từ thư mục, và tương thích ngược của suspicion.
#   8 assertion đỏ ngày 03/08 KHÔNG phải hồi quy code: chúng hard-code census của thế hệ
#        labels.csv 21/07 (cross_col 8, union 8, provably-wrong 4, quarantine 8), trong khi
#        labels.csv được sinh lại ngày 22/07 và lớp trùng lặp tụt về 0 — đã kiểm chứng độc
#        lập (dup_bbox 0, cross_col 0, md5 rỗng 0, chỉ 2 hàng chung md5 toàn corpus).
#        Cách sửa: đổi từ SỐ CỨNG sang BẤT BIẾN (== 0) + kiểm cấu trúc, để lần sinh dữ liệu
#        sau không đỏ giả nữa. Lịch sử before/after giữ trong docs/census_history.md.
#   demoted_similar_lowcos 748 -> 925: quarantine = 0 nên không còn cướp hàng của bước
#        demote; nay kiểm theo công thức |GOLD∩bridge∩s3<τ| \ quarantine thay vì khoảng cứng.
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
# => Con số trích dẫn trong luận văn phải là 392 assertions (392 pass, 0 fail), KHÔNG
#    còn là 360 hay 223 — 223 là mốc cũ và đã bỏ sót toàn bộ selftest của bước 1-2.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || { echo "Không thấy Python: $PY (đặt biến PY=... để đổi)"; exit 1; }

BASELINE_PASS=392
BASELINE_FAIL=0

MODULES=(
  core.pdf.parser_selftest
  core.text.syllable_validation_selftest
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
printf "%-38s %s\n" "MỐC 2026-08-11" "$BASELINE_PASS passed, $BASELINE_FAIL failed"
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
