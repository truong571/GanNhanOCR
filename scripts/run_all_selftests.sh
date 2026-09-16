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
#     TỔNG                    722 passed,  0 failed  (mốc 2026-08-25, +43: T4.e + nối cấu hình + T6.b)
#
# ĐỔI SO VỚI MỐC 448 (KHỐI 1):
#   +47  pipeline.lab.selftest — bàn thí nghiệm (metrics/perturb/runner). Gồm chốt
#        TẤT ĐỊNH LIÊN TIẾN TRÌNH: seed từng được dựng bằng tuple.__hash__() chứa
#        chuỗi, mà hash chuỗi bị ngẫu nhiên hoá theo PYTHONHASHSEED -> hai lần chạy
#        ra hai con số khác nhau. Test chạy _seed dưới PYTHONHASHSEED 0/1/random.
#
# ĐỔI SO VỚI MỐC 2026-08-22 (427/0):
#   +12  remediation::test_s3_unwind — chốt chặn lớp confusion: không cặp (âm,chữ) đã
#        chứng minh sai hệ thống nào được quay lại tier dùng được sau s3_unwind. 48 ô
#        㝵/"người" đã lọt vào bộ công bố theo đúng đường này.
#   + 9  remediation::test_confusion_fix_join — chuẩn hoá tiền tố sách yen*->stt*.
#        Trước đó join khớp 0/825 nên --measure trả null TRONG IM LẶNG; nay 816/825.
#
# ĐỔI SO VỚI MỐC 2026-08-19 (414/0):
#   +13  phase1_engine_selftest::test_ocr_cache_guard — chốt đối chiếu cache OCR với
#        ảnh trên đĩa. Trước đó `image_hash` được GHI mà KHÔNG BAO GIỜ đối chiếu, nên
#        đổi prepared/*/pages/*.png mà giữ cache = bbox ảnh cũ áp lên ảnh mới, lệch
#        toạ độ, không một cảnh báo. Phân biệt "đổi nén PNG" (heal) với "đổi pixel"
#        (ném StaleOCRCacheError, KHÔNG tự gọi lại API vì OCR lại tốn tiền).
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
# => Con số trích dẫn trong luận văn phải là BASELINE_PASS hiện hành bên dưới (mốc 722 của
#    2026-08-25 chỉ còn là lịch sử; 360/223 cũ hơn nữa — 223 bỏ sót toàn bộ selftest bước 1-2).
#
# MỐC 2026-09-16 (Khối A, A-1…A-12): TỔNG 922 passed, 0 failed, đo với
#   re-dataset/ THẾ HỆ CŨ (30 cột, đóng băng 25/08) ở gốc kho và CHƯA dựng mẻ mẫu:
#     ground_truth 171 · consensus_fusion 44 · publish 56 · remediation 182 ·
#     phase1_engine 189 · decisions 41 (mới, S8) · tools 101 · lab 81 · parser 18 · syl 39
#   remediation 164 -> 182 (S10, A-11/A-12): +luật MD5_DUP (cùng md5 cùng cột khác bbox —
#   bắt cặp 法/冉 trên cả bộ 25/08 lẫn bản HEAD tái lập), +remediate không cần cột split,
#   +cột cờ v3 (qd01_locked…) đọc là chuỗi; 3 phép "trùng == 0" trên bộ thật đổi thành
#   cấu trúc (union == md5_dup ≤ 2, quarantined == union) vì luật mới THẤY cặp đó.
#   tools.selftest: 4 phép kiểm split/label_in_train cũ bỏ (A-10); phép kiểm SCHEMA 12 CỘT
#   + labels_trace/columns.csv chỉ chạy khi bộ giao nộp là thế hệ mới (DS_OUT=dataset_out_v3
#   -> tools 127, TỔNG 948). Khi bộ v3 thay bộ thật ở gốc kho thì mốc là 948.
#
# MỐC 2026-09-16 (A-15, flow N0e): TỔNG 991 passed, 0 failed (cùng điều kiện dữ liệu như
#   mốc 922 ở trên; với DS_OUT=dataset_out_v3 suy ra 948 + 69 = 1017, chưa đo).
#   phase1_engine 189 -> 251 (+62): enforce_count (M>N / M<N / rỗng / gray None / seam-
#   valley-midpoint / NMS / clamp) trên CHÍNH train_crop/infer_centernet; _pair_new_state
#   nối 3 nhánh hộp (a: G[syl_idx], b: G[nom_idx], c: enforce_count CHỈ gọi ở đây + split/
#   midpoint) trên cột giả với detector giả, legacy/legacy_locked_col trọn gói ±0,5w,
#   _pair_new == _pair_new_state[:2]; bảng chân trị tier_v3/rule_of 19 nhánh + chốt
#   is_plausible TRƯỚC tier. Hai test đầu tự bỏ qua (có ghi) nếu thiếu torch/train_crop.
#   +7  pipeline.align_engine.tier_v3_selftest (suite MỚI vào runner): feats LOO của mã sản
#   xuất == lab thuc_nghiem 100% trên cols.pkl; thiếu cols.pkl (cache lab, gitignored,
#   dựng bằng `thuc_nghiem.py rebuild`) thì in RESULT 0/0 — tổng thấp hơn mốc 7, KHÔNG
#   phải hồi quy.
#   apply_am_sua_dau (có/không 2-gram + hoà), decisions mục chưa ký không áp, glyph_fix
#   --mode kiem trên khung giả: ĐÃ có từ S7–S10 (phase1 mục 7 / decisions / remediation).
#
# MỐC 2026-09-16 (S12, A-13/A-14): TỔNG 1007 passed, 0 failed (cùng điều kiện dữ liệu).
#   ground_truth 171 -> 175 (+4): audit_grid.build_audit(labels_path=) ghi labels_sha256 +
#   labels_path vào TỪNG dòng manifest (bộ đem đo = bộ đem nộp), thiếu tệp -> FileNotFoundError.
#   tools 101 -> 113 (+12): doi_soat_the_he khối v3 (N15 B1–B8, count_source, khe giả) trên
#   dữ liệu giả có đáp án. phase1_engine 251 (không đổi số): mốc "legacy tái lập 100%" đọc
#   dataset_out_v3/labels_HEAD_N0c.csv (bản tái lập HEAD N0c) — dataset_out_v3/labels.csv
#   nay là bản build v3 (syl_index) nên không còn là mốc legacy; thiếu tệp mốc -> BỎ QUA.

# MỐC 2026-09-17 (Khối A thăng cấp chính thức v3, người ký truongmdn): TỔNG 1035 passed, 0 failed.
#   re-dataset/ và dataset_out/ chính thức là bản build v3 (70.326 ô dùng được, 12 cột chuẩn,
#   0 ô pending, 2.012 ô QĐ-01 lock + 2 ô trôi bỏ, cách ly 42 ô conflict MD5_DUP).
#   tools 113 -> 142 (+29): chạy toàn bộ suite kiểm tra schema 12 cột và trace trên re-dataset/ chính thức.
#   phase1_engine 250 passed; ground_truth 175 passed; remediation 182 passed; consensus_fusion 44 passed.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || { echo "Không thấy Python: $PY (đặt biến PY=... để đổi)"; exit 1; }

# MỐC 2026-09-16 (Khối B K4, B-2/B-3 mã): TỔNG 1083 passed, 0 failed = 1035 + 48.
#   +35 pipeline.align_engine.visual_emission_selftest (B-2: trung tính log 0,5, kích thước logP, fold đúng
#       trang == train_oof_cnn_v3, cut() byte-identical với lab, cost_ij/khe, DP+posterior tương thích, thiếu
#       mô hình không crash / --strict ném). realign_column/posterior_matches thêm cost_ij/cost_del/cost_ins:
#       mặc định KHÔNG đổi (ops + posterior y hệt HEAD trên 4.029 cột cols.pkl).
#   +13 pipeline.tools.visual_syl_gate_selftest (B-3: cổng REVIEW→SYL 0,9/0,8 trên khung giả có đáp án, loại
#       QĐ-01/not_plausible, FAR CHAR_A (a)/(b) + Wilson, thiếu OOF không crash, schema cũ bị chặn).
BASELINE_PASS=1083
BASELINE_FAIL=0

MODULES=(
  core.pdf.parser_selftest
  core.text.syllable_validation_selftest
  pipeline.ground_truth.selftest
  pipeline.consensus_fusion.selftest
  pipeline.publish.selftest
  pipeline.remediation.selftest
  pipeline.phase1_engine_selftest
  pipeline.align_engine.tier_v3_selftest
  pipeline.decisions_selftest
  pipeline.tools.selftest
  pipeline.lab.selftest
  pipeline.align_engine.visual_emission_selftest
  pipeline.tools.visual_syl_gate_selftest
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
printf "%-38s %s\n" "MỐC 2026-09-16 (S12)" "$BASELINE_PASS passed, $BASELINE_FAIL failed"
echo "================================================================"

if [ "$total_pass" -eq "$BASELINE_PASS" ] && [ "$total_fail" -eq "$BASELINE_FAIL" ]; then
  echo "KHỚP MỐC — không có hồi quy."
  exit 0
fi

echo "LỆCH MỐC:"
# BÁO ĐỘNG GIẢ ĐÃ XẢY RA: chạy selftest TRONG LÚC pipeline đang dựng lại thì
# dataset_out/*.csv chưa tồn tại, 105 test phụ thuộc dữ liệu tự bỏ qua, và bộ chạy
# kêu "nghi hồi quy" dù 0 test hỏng. Phân biệt hai chuyện đó trước khi kết luận.
if [[ ! -d dataset_out/human_audit/audit_combined ]]; then
  echo
  echo "  ℹ️  chưa dựng mẻ MẪU (đường phụ) — các test tools.selftest đọc manifest của nó tự bỏ qua,"
  echo "      KHÔNG phải hồi quy. Mốc đo lúc CHƯA dựng mẻ; dựng mẻ thì tổng CAO HƠN mốc."
fi
if [[ ! -f lab/gan_nhan_2026-09-13/cols.pkl ]]; then
  echo
  echo "  ℹ️  thiếu lab/gan_nhan_2026-09-13/cols.pkl — 7 test tier_v3_selftest tự bỏ qua (RESULT 0/0),"
  echo "      KHÔNG phải hồi quy. Dựng bằng: .venv/bin/python lab/gan_nhan_2026-09-13/thuc_nghiem.py rebuild"
fi
if [[ ! -f dataset_out/labels_final.csv ]]; then
  echo
  echo "  ⚠️  dataset_out/labels_final.csv KHÔNG CÓ — các test phụ thuộc dữ liệu đã tự bỏ qua."
  echo "      Số 'passed' thấp hơn mốc là BÌNH THƯỜNG lúc này, KHÔNG phải hồi quy."
  echo "      Chạy lại sau khi pipeline dựng xong để so với mốc cho đúng."
fi
[ "$total_pass" -lt "$BASELINE_PASS" ] && echo "  - passed giảm $((BASELINE_PASS - total_pass)) → nghi có hồi quy, KIỂM TRA NGAY."
[ "$total_pass" -gt "$BASELINE_PASS" ] && echo "  - passed tăng $((total_pass - BASELINE_PASS)) → nếu do đã sửa thì CẬP NHẬT mốc trong file này."
[ "$total_fail" -gt "$BASELINE_FAIL" ] && echo "  - failed tăng $((total_fail - BASELINE_FAIL)) → hồi quy mới."
[ "$total_fail" -lt "$BASELINE_FAIL" ] && echo "  - failed giảm $((BASELINE_FAIL - total_fail)) → đã sửa được, CẬP NHẬT mốc trong file này."
exit 1
