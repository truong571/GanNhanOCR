#!/bin/bash
# Chạy chuỗi đo cho MỘT phiên bản crop chuẩn (sau khi s01 đã cắt xong). Dùng: TN4_VER=v2 bash run_ver.sh [from_step]
# Các bước: s02 hình học -> s02b độ phủ lõi -> s03 verifier_ft -> s04b nhúng v1+v2 -> r02/r02b/r03 (viss) -> s05 viết tay
#           -> s06a mặt nạ + ngưỡng LOBO -> s06c ước lượng 8 bộ (bản chép TN3) -> s07 AUC.   0 API; MPS cho CNN.
set -e
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
PY=.venv/bin/python; L=lab/thu_nghiem_anh_chu/TN4_crop_chuan; B=measure_out/_thu_nghiem_anh_chu/TN4/$TN4_VER
S=${1:-0}
[ $S -le 1 ] && $PY $L/s02_geom.py > $B/log_s02.txt 2>&1
[ $S -le 2 ] && $PY $L/s02b_cov_core.py > $B/log_s02b.txt 2>&1
[ $S -le 3 ] && $PY $L/s03_vft.py > $B/log_s03.txt 2>&1
[ $S -le 4 ] && $PY $L/s04b_embed_sv.py > $B/log_s04b.txt 2>&1
[ $S -le 5 ] && $PY $L/s04a_rerank_copies.py > /dev/null && $PY $L/rerank_$TN4_VER/r02_candidates.py mps > $B/log_r02.txt 2>&1 \
   && $PY $L/rerank_$TN4_VER/r02b_self.py > $B/log_r02b.txt 2>&1 && $PY $L/rerank_$TN4_VER/r03_model.py viss > $B/log_r03.txt 2>&1
[ $S -le 6 ] && $PY $L/s05_hand.py > $B/log_s05.txt 2>&1
[ $S -le 7 ] && $PY $L/s06a_masks.py > $B/log_s06a.txt 2>&1
[ $S -le 8 ] && $PY $L/s06c_tn3m.py > $B/log_s06c.txt 2>&1
[ $S -le 9 ] && $PY $L/s07_auc.py > $B/log_s07.txt 2>&1
echo "ALL_DONE $TN4_VER"
