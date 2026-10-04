#!/usr/bin/env bash
# TN11 p11a — LÀM MỚI dữ liệu cho gói Kaggle sau mỗi lần chạy lại pipeline (0 API, CPU). Chạy từ gốc repo:
#   bash lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p11a_lam_moi_du_lieu.sh [tiền tố gói mới]
# Thứ tự: khảo sát dữ liệu (9 bước) -> khảo sát quy mô (chữ ưu tiên + gán phông) -> dựng gói (p11) -> notebook (p13) -> kiểm gói cục bộ.
# Màn hình chỉ có 1 dòng mỗi chặng + FAIL (nếu có) + tóm tắt gói; log đầy đủ ở measure_out/_tn11/full/lam_moi_logs/<giờ>/.
# Thư mục gói phải CHƯA tồn tại (bộ dựng không xoá gì). Mặc định: measure_out/_tn11/full/kaggle_pack_<YYYYMMDD-HHMM>.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
D=lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
OUT="${1:-measure_out/_tn11/full/kaggle_pack_$(date +%Y%m%d-%H%M)}"
LOG="measure_out/_tn11/full/lam_moi_logs/$(date +%Y%m%d-%H%M%S)"
[ -e "$OUT" ] && { echo "✗ $OUT đã tồn tại — chọn tên khác (bộ dựng không ghi đè)"; exit 2; }
mkdir -p "$LOG"

chang() {   # chang <tên> <lệnh...> : chạy, ghi log riêng; thất bại -> in 15 dòng cuối rồi dừng
  local ten=$1; shift
  if "$@" >"$LOG/$ten.log" 2>&1; then return 0; fi
  echo "✗ $ten (mã $?) — 15 dòng cuối, log: $LOG/$ten.log"; tail -15 "$LOG/$ten.log" | cut -c1-220; exit 1
}

t0=$SECONDS
for s in 01_kiem_ke 02_chat_luong_crop 02b_do_neo_nhan_nguoi 03_tn8_cu 03b_neo_truoc_chon_chu 04_rui_ro 05_tong_hop 06_cache_crop_chuan 06b_xac_minh_cache; do
  chang "khao_sat_$s" $PY $D/w_hien_trang_du_lieu_$s.py
done
echo "✓ khảo sát dữ liệu 9/9 bước ($((SECONDS - t0))s)"
t1=$SECONDS; chang quy_mo $PY $D/w_quy_mo_va_ngan_sach_00_chay_het.py
echo "✓ quy mô & ngân sách ($((SECONDS - t1))s)"
t1=$SECONDS; chang dung_goi $PY $D/p11_dung_goi_toan_bo.py --out "$OUT"
echo "✓ dựng gói ($((SECONDS - t1))s)"
chang notebook $PY $D/p13_tao_notebook_kaggle.py --out "$OUT.ipynb"
echo "✓ notebook"
chang kiem_goi $PY $D/w_runner_khung_chay.py preflight --pack "$OUT" --stub --remote "dir:$LOG/kho_thu"
echo "✓ kiểm gói cục bộ (preflight --stub)"

# bất biến: các chặng tự in tóm tắt riêng → đọc lại từ log; chỉ báo khi LỆCH (X ≠ Y) hoặc có FAIL; hash GIỐNG/KHÁC chỉ là thông tin
inv_gop=$(grep -ho 'bất biến gộp [0-9]*/[0-9]*' "$LOG/khao_sat_05_tong_hop.log" | tail -1 | sed 's/bất biến gộp //')
inv_cache=$(grep -ho 'bất biến [0-9]*/[0-9]*' "$LOG/khao_sat_06b_xac_minh_cache.log" | tail -1 | sed 's/bất biến //')
qm_fail=$(grep -h 'FAIL: ' "$LOG/quy_mo.log" | grep -vc 'FAIL: không' || true)
qm_n=$(grep -c 'FAIL: ' "$LOG/quy_mo.log" || true)
nkhac=$(grep -c 'KHÁC (lần đầu' "$LOG/quy_mo.log" || true)
bad=0
[ -n "$inv_gop" ] && [ "${inv_gop%/*}" = "${inv_gop#*/}" ] || bad=1
[ -n "$inv_cache" ] && [ "${inv_cache%/*}" = "${inv_cache#*/}" ] || bad=1
[ "$qm_fail" = "0" ] && [ "$qm_n" != "0" ] || bad=1
echo "· bất biến: khảo sát ${inv_gop:-?} · cache crop ${inv_cache:-?} · quy mô $((qm_n - qm_fail))/${qm_n} bước không FAIL · tệp đầu ra đổi hash so với lần trước: ${nkhac}"
[ "$bad" = "1" ] && { echo "⚠ có bất biến lệch/FAIL — xem log:"; grep -h 'FAIL: ' "$LOG/quy_mo.log" | grep -v 'FAIL: không' | cut -c1-220 | head -6; }
grep -h '^\[p11\] gói:' "$LOG/dung_goi.log" | tail -1
echo "→ Đưa lên Kaggle: $OUT.zip (làm Dataset) + $OUT.ipynb (Import Notebook) · log: $LOG"
[ "$bad" = "0" ]
