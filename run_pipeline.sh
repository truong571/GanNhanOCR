#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — GanNhanOCR · sinh bộ dataset, THUẦN TƯƠNG TÁC (1 lệnh, hỏi-đáp)
#
#   ./run_pipeline.sh
#     -> hỏi 1) chọn sách   2) chạy cache cũ hay xoá-cache-chạy-mới
#     -> chạy: setup -> extract -> build -> remediate -> confusion -> export
#     -> ra:   dataset/labels.csv (+ ảnh crop copy hẳn) = BẢN CUỐI CÙNG, TỰ CHỨA
#              dataset/ bị XOÁ SẠCH và ghi lại mỗi lần chạy — luôn là bản MỚI
#              NHẤT, không cộng dồn. dataset_out/ vẫn giữ nguyên làm nơi làm
#              việc trung gian (labels_remediated.csv đầy đủ tier + report...).
#
# 7 BƯỚC (đúng thứ tự, KHÔNG có cờ dòng lệnh — mọi lựa chọn hỏi qua stdin):
#   1 setup       pipeline.step0_setup — kiểm cấu hình/đường dẫn
#   2 extract     PDF -> khung -> OCR (cache) -> 9 cột/trang  (CHỈ sách đã chọn)
#   3 build       align_engine.build_dataset -> labels.csv + crops (LUÔN cả 3 sách
#                 trong config — build không lọc theo sách vừa extract, xem lưu ý
#                 ở step_build())
#   4 remediate   pipeline.remediation -> labels_remediated.csv + remediation_report.json
#                 (census AE-1/F1 + MD5_DUP cùng cột — A-11)
#   5 confusion   pipeline.remediation.confusion_fix -> labels_final.csv (BẢN CÔNG BỐ)
#   6 gỡ S3       pipeline.remediation.s3_unwind — chạy VÔ ĐIỀU KIỆN (no-op khi S3 tắt)
#                 để s3_unwind_report.json luôn là bản mới (N11)
#   7 kiểm QĐ-01  pipeline.remediation.glyph_fix --mode kiem — KHÔNG gán; chỉ đối chiếu
#                 khoá bền config/qd01_cells.csv + qd01a_decisions.csv (N12); khoá đã
#                 được build áp ở PASS 1c. assert_qd01 chạy sau các bước 4/5/6/7 (N13)
#   8 export      pipeline/export_final_dataset.py -> dataset/ (chỉ tier
#                 GOLD+SILVER+SYLLABLE = usable; XOÁ SẠCH dataset/ cũ trước khi ghi)
#
# S3 TẮT từ 16/09 (A-10 giai đoạn 2): build không còn --use-s3; bộ nhãn KHÔNG có cột
#   split/split_group/label_in_train (bỏ chia train/val/test); proto-index/s3_proto_cache
#   ngoài flow. Thời gian từng bước ghi vào log và $DS_OUT/CHECKSUMS.txt (dòng `thoi_gian`).
#
# ⚠️ BƯỚC 5 KHÔNG ĐƯỢC BỎ. Trước 2026-08-11 script này export thẳng từ
#   labels_remediated.csv, trong khi confusion_fix chỉ được chạy tay một lần hồi
#   21/07 -> bộ giao nộp mang 1.926 crop 㝵/'người' ở tier GOLD/SILVER dù chính
#   lớp lỗi đó đã được chứng minh sai hệ thống (Fisher p=5,4e-8), CÒN mẻ audit
#   lại rút mẫu từ labels_final.csv. Hai tập khác nhau = số precision đo được
#   không áp cho bộ thật. Nối bước 5 vào chuỗi để nhánh đó không tái diễn.
#
# TẠM BỎ QUA theo yêu cầu (chưa cần cho việc sinh dataset thô):
#   audit người · fuse · publish
#   Bản đầy đủ 8 bước (có cờ --only/--from/--until/--all, audit người, publish
#   quốc tế...) vẫn còn nguyên trong lịch sử git: `git log -- run_pipeline.sh`
#   (commit 47dfbfe0f trở về trước) — khôi phục bằng
#   `git show 47dfbfe0f:run_pipeline.sh > run_pipeline.sh` khi cần dùng lại.
#
# BIẾN MÔI TRƯỜNG (N0a, 2026-09-16) — không phải cờ dòng lệnh:
#   DS_OUT=dataset_out_v3   dựng bản THỬ NGHIỆM song song (mặc định dataset_out = bộ thật);
#                           re-dataset/ + dataset/ khi đó nằm dưới $DS_OUT, KHÔNG ghi docs/
#   NONINTERACTIVE=1        bỏ mọi câu hỏi: cả 3 sách, dùng cache OCR cũ, bắt đầu ngay;
#                           gặp $DS_OUT/.FROZEN thì chỉ tiếp tục khi FROZEN_OVERRIDE=GHIDE
#   ALLOW_PENDING=1         CHỈ khi DS_OUT != dataset_out: export dù còn ô QĐ-01 pending (bản
#                           thử nghiệm để đối soát N15 / thăm dò QĐ-01a); bộ thật luôn bị chặn
#
# Viết cho bash 3.2 (bash mặc định của macOS): không dùng mảng kết hợp.
# =============================================================================
set -euo pipefail

REPO_ROOT="${GANNHANOCR_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_ROOT"

# Giảm dao động số học của S3 / torch / tokenizers giữa các lần chạy.
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

# ------------------------------- MẶC ĐỊNH -----------------------------------
PY="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-config/pipeline.yaml}"
RESEG="${RESEG:-detector}"
TAU_REMEDIATE="${TAU_REMEDIATE:-0.62}"
# DS_OUT — thư mục làm việc trung gian (N0a). Mặc định `dataset_out` = bộ giao nộp.
# Đặt DS_OUT=dataset_out_v3 để dựng bản THỬ NGHIỆM song song mà không đụng bộ đã
# đóng băng: mọi bước ghi/đọc đều đi qua biến này, evidence() chỉ ghi vào docs/ khi
# DS_OUT là bộ thật. (KHÔNG đặt tên OUT_DIR — va với `local OUT_DIR` ở step_export.)
DS_OUT="${DS_OUT:-dataset_out}"
LABELS_RAW="$DS_OUT/labels.csv"
LABELS_REMED="$DS_OUT/labels_remediated.csv"
LABELS_FINAL="$DS_OUT/labels_final.csv"   # BẢN CÔNG BỐ — nguồn của export + audit
CONFUSION_FIXES="${CONFUSION_FIXES:-config/confusion_fixes.yaml}"
GLYPH_DECISIONS="${GLYPH_DECISIONS:-config/quyet_dinh_glyph.yaml}"
# ---- HAI ĐẦU RA, TÙY CÓ PHÁN QUYẾT NGƯỜI HAY CHƯA --------------------------
#   re-dataset/  bộ ĐEM CHẤM  — chưa có phán quyết người, chưa kiểm chứng
#   dataset/     bộ CUỐI CÙNG — đã nạp phán quyết, có bảng precision
# Tách hai thư mục để KHÔNG BAO GIỜ nhầm bộ chưa kiểm chứng thành bộ cuối. Bước 7 tự
# chọn dựa trên việc verdicts*.jsonl đã có hay chưa — không cần cờ tay, không quên được.
REDATASET_DIR="re-dataset"
FINAL_DIR="dataset"
# Bản thử nghiệm: hai thư mục xuất nằm DƯỚI $DS_OUT để export không rmtree bộ thật.
if [[ "$DS_OUT" != "dataset_out" ]]; then
  REDATASET_DIR="$DS_OUT/re-dataset"
  FINAL_DIR="$DS_OUT/dataset"
fi
AUDIT_DIR="$DS_OUT/human_audit/audit_combined"
EVIDENCE="docs/EVIDENCE_INDEX.md"
CHECKSUMS="$DS_OUT/CHECKSUMS.txt"
# NONINTERACTIVE=1 — chạy không tay (N0c/N19): bỏ mọi `read -r -p`, lấy mặc định AN
# TOÀN: cả 3 sách, dùng cache OCR cũ (KHÔNG BAO GIỜ xoá cache), .FROZEN -> tiếp tục
# (chỉ có nghĩa khi DS_OUT != dataset_out — xem MAIN), Enter bắt đầu -> tự chạy.
NONINTERACTIVE="${NONINTERACTIVE:-0}"

BOOKS=""            # tên đầy đủ trong config, cách nhau bởi dấu cách — điền ở ask_book_choice
BOOKS_LABEL=""       # nhãn ngắn để in log
FRESH_OCR=0          # 0 = dùng cache cũ | 1 = xoá cache rồi OCR lại

RED=""; YEL=""; GRN=""; CYA=""; BLD=""; RST=""
if [[ -t 1 ]]; then
  RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; CYA=$'\033[36m'
  BLD=$'\033[1m'; RST=$'\033[0m'
fi

N_WARN=0

log()  { printf '%s\n' "$*"; }
info() { printf '%s[i]%s %s\n' "$CYA" "$RST" "$*"; }
ok()   { printf '%s[OK]%s %s\n' "$GRN" "$RST" "$*"; }
warn() { N_WARN=$((N_WARN + 1)); printf '%s[CẢNH BÁO]%s %s\n' "$YEL" "$RST" "$*" >&2; }
die()  { printf '%s[LỖI]%s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }

banner() {   # banner <số> <tên bước> <mô tả>
  log ""
  log "${BLD}================================================================${RST}"
  printf '%s>>> BƯỚC %s/8 · %s%s — %s\n' "$BLD" "$1" "$2" "$RST" "$3"
  log "${BLD}================================================================${RST}"
}

# X — chạy 1 lệnh có tham số rời, in ra trước khi chạy.
X() {
  printf '    %s$%s %s\n' "$CYA" "$RST" "$*"
  "$@"
}

# ============================ HỎI-ĐÁP (interactive) ==========================
ask_book_choice() {
  local choice
  if [[ "$NONINTERACTIVE" == "1" ]]; then
    BOOKS="SachThanhTruyen2 SachThanhTruyen4 SachThanhTruyen11"
    BOOKS_LABEL="STT2+STT4+STT11"
    info "NONINTERACTIVE=1 -> chọn sách: cả 3 ($BOOKS_LABEL)"
    return 0
  fi
  while true; do
    log ""
    log "${BLD}Chọn sách cần OCR (bước extract):${RST}"
    log "  1) STT2   — SachThanhTruyen2"
    log "  2) STT4   — SachThanhTruyen4"
    log "  3) STT11  — SachThanhTruyen11"
    log "  4) Cả 3 sách (STT2 + STT4 + STT11)"
    read -r -p "Nhập lựa chọn [1-4]: " choice
    case "$choice" in
      1) BOOKS="SachThanhTruyen2";  BOOKS_LABEL="STT2";  return 0 ;;
      2) BOOKS="SachThanhTruyen4";  BOOKS_LABEL="STT4";  return 0 ;;
      3) BOOKS="SachThanhTruyen11"; BOOKS_LABEL="STT11"; return 0 ;;
      4) BOOKS="SachThanhTruyen2 SachThanhTruyen4 SachThanhTruyen11"
         BOOKS_LABEL="STT2+STT4+STT11"; return 0 ;;
      *) warn "Lựa chọn không hợp lệ: '$choice' — nhập 1, 2, 3 hoặc 4." ;;
    esac
  done
}

confirm_fresh_delete() {
  local b typed sz
  log ""
  log "${RED}${BLD}Sắp XOÁ cache OCR cho: $BOOKS_LABEL${RST}"
  for b in $BOOKS; do
    if [[ -d "prepared/$b" ]]; then
      sz=$(du -sh "prepared/$b" 2>/dev/null | awk '{print $1}')
      log "  prepared/$b/detected/*_ocr_cache.json  (thư mục sách ~$sz)"
    fi
  done
  log "${RED}Cache OCR là PRIMARY DATA của luận văn:${RST}"
  log "${RED}  · còn cache -> bước extract TÁI LẬP ĐƯỢC, 0 đồng${RST}"
  log "${RED}  · xoá cache -> gọi lại API ngoài: TỐN TIỀN + KHÔNG tái lập${RST}"
  if [[ "$NONINTERACTIVE" == "1" ]]; then
    warn "NONINTERACTIVE=1 -> KHÔNG xoá cache OCR (huỷ, dùng cache cũ)"; return 1
  fi
  read -r -p "Gõ đúng chữ XOA để xác nhận (Enter/khác = huỷ, quay về dùng cache cũ): " typed
  [[ "$typed" == "XOA" ]]
}

ask_cache_choice() {
  local choice
  if [[ "$NONINTERACTIVE" == "1" ]]; then
    FRESH_OCR=0; info "NONINTERACTIVE=1 -> cache OCR: dùng cache cũ (0 đồng, tái lập được)"
    return 0
  fi
  while true; do
    log ""
    log "${BLD}Cache OCR (prepared/<sách>/detected/*_ocr_cache.json):${RST}"
    log "  1) Dùng cache cũ           — mặc định, KHUYẾN NGHỊ (nhanh, 0 đồng, tái lập được)"
    log "  2) Xoá cache & OCR lại mới — gọi API ngoài, TỐN TIỀN, KHÔNG tái lập"
    read -r -p "Nhập lựa chọn [1-2] (Enter = 1): " choice
    choice="${choice:-1}"
    case "$choice" in
      1) FRESH_OCR=0; return 0 ;;
      2) if confirm_fresh_delete; then FRESH_OCR=1; return 0; fi ;;
      *) warn "Lựa chọn không hợp lệ: '$choice' — nhập 1 hoặc 2." ;;
    esac
  done
}

confirm_frozen_override() {
  local typed
  log ""
  log "${RED}${BLD}$DS_OUT/.FROZEN tồn tại${RST} — bản đã đóng băng để trích số vào luận văn."
  log "${RED}Chạy tiếp (bước build) sẽ GHI ĐÈ bằng chứng đã đóng băng đó.${RST}"
  if [[ "$NONINTERACTIVE" == "1" ]]; then
    # Không có người gõ GHIDE: chỉ tiếp tục khi biến môi trường nói rõ ý đó, vì một
    # lần chạy nền vô ý (quên đặt DS_OUT) mà ghi đè bộ đã đóng băng thì không hồi được.
    [[ "${FROZEN_OVERRIDE:-}" == "GHIDE" ]] \
      || die "NONINTERACTIVE=1 nhưng $DS_OUT/.FROZEN tồn tại — đặt FROZEN_OVERRIDE=GHIDE
      để ghi đè có chủ đích, hoặc DS_OUT=dataset_out_v3 để dựng bản thử nghiệm."
    warn "NONINTERACTIVE=1 + FROZEN_OVERRIDE=GHIDE -> tiếp tục, sẽ ghi đè $DS_OUT/"
    return 0
  fi
  read -r -p "Gõ đúng chữ GHIDE để tiếp tục (Enter/khác = huỷ chạy): " typed
  [[ "$typed" == "GHIDE" ]] || die "Đã huỷ — xoá $DS_OUT/.FROZEN nếu thật sự muốn dựng lại."
}

# ============================== PREFLIGHT ====================================
# Chỉ kiểm những gì 6 bước setup/extract/build/remediate/confusion/export cần —
# KHÔNG kiểm verdicts người / mẻ audit / v.v. (thuộc các bước tạm bỏ qua).
book_pdf() {   # book_pdf <tên-sách-trong-config>
  "$PY" -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
for b in cfg['books']:
    if b['name'] == '$1':
        print(b.get('pdf', ''))
        break
"
}

preflight() {
  log ""
  log "${BLD}--- PREFLIGHT ---------------------------------------------------${RST}"

  if [[ -x "$PY" ]]; then
    ok "venv: $("$PY" -c 'import sys;print(sys.executable, sys.version.split()[0])')"
  else
    die "Không thấy Python ở: $PY
      Tạo venv : python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
      Hoặc     : PYTHON_BIN=/path/to/python ./run_pipeline.sh"
  fi

  # --- THƯ VIỆN: kiểm MỖI LẦN CHẠY, và tự vá ---------------------------------
  # Thiếu một gói giữa chừng thì pipeline chết ở phút thứ 20, SAU KHI đã xoá
  # dataset_out/. Kiểm ở đây tốn 1 giây và chặn được cả lần chạy hỏng.
  # Đặt SKIP_DEPS=1 để bỏ qua (ví dụ khi đang chạy ngoại tuyến).
  if [[ "${SKIP_DEPS:-0}" != "1" ]]; then
    if "$PY" -m pipeline.tools.check_deps >/dev/null 2>&1; then
      ok "thư viện: đủ và đúng phiên bản ghim ($("$PY" -m pipeline.tools.check_deps 2>&1 | grep -oE '[0-9]+/[0-9]+' | head -1) gói)"
    else
      warn "thư viện THIẾU hoặc LỆCH phiên bản — đang vá tự động:"
      "$PY" -m pipeline.tools.check_deps 2>&1 | sed 's/^/    /'
      # check_deps tôn trọng ba cạm bẫy của môi trường 3.14: pygame-ce thay pygame,
      # vietocr phải --no-deps, safetensors ghim bản rc có wheel.
      "$PY" -m pipeline.tools.check_deps --fix --python "$PY" 2>&1 | sed 's/^/    /'
      "$PY" -m pipeline.tools.check_deps >/dev/null 2>&1 \
        || die "thư viện vẫn chưa đủ sau khi vá — xem log ở trên, sửa tay rồi chạy lại"
      ok "thư viện: đã vá xong"
    fi
  else
    warn "BỎ QUA kiểm thư viện (SKIP_DEPS=1)"
  fi

  if [[ -f .env ]]; then
    set -a; source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env) 2>/dev/null || true; set +a
    ok ".env đã nạp (chỉ các key hợp lệ bash)"
  else
    warn "không có .env — SN_OCR_TOKEN/GEMINI_API_KEY sẽ thiếu nếu bước extract cần gọi API"
  fi

  local miss
  miss=$("$PY" - <<'PYEOF' 2>/dev/null || true
mods = "fitz cv2 numpy pandas yaml PIL vietocr scipy torch".split()
bad = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        bad.append(m)
print(" ".join(bad))
PYEOF
)
  if [[ -z "${miss// /}" ]]; then
    ok "module có mặt (fitz cv2 numpy pandas yaml PIL vietocr scipy torch)"
  else
    die "thiếu module trong venv: $miss
      -> $PY -m pip install -r requirements.txt"
  fi

  if [[ ! -f "$CONFIG" ]]; then
    die "không thấy $CONFIG — mọi bước đọc cấu hình sẽ hỏng."
  fi
  ok "config: $CONFIG"

  # Bước 5 đọc file này. Thiếu nó thì confusion-fix chạy rỗng và bộ công bố lặng lẽ
  # giữ lại các nhãn đã chứng minh sai — báo ngay ở preflight, đừng để tới bước 5.
  if [[ -f "$CONFUSION_FIXES" ]]; then
    ok "confusion fixes: $CONFUSION_FIXES ($("$PY" -c "
import yaml,sys
print(len((yaml.safe_load(open('$CONFUSION_FIXES')) or {}).get('fixes', [])))" 2>/dev/null || echo '?') fix)"
  else
    warn "thiếu $CONFUSION_FIXES -> bước 5 sẽ DỪNG. Bộ công bố cần file này (dù là 'fixes: []')."
  fi

  # Từ điển: bước build đọc hai file này qua config. Đổi tên/di chuyển chúng là kiểu
  # hỏng lặng lẽ nhất — step0_setup KHÔNG kiểm, nên trước 2026-08-19 việc đổi tên
  # Dict/*.csv chỉ lộ ra khi build đã chạy được vài phút rồi mới ném FileNotFoundError.
  local d_qn d_sim
  d_qn=$(sed -n 's/^[[:space:]]*qn_to_nom_dict:[[:space:]]*//p' "$CONFIG" | sed -n 1p)
  d_sim=$(sed -n 's/^[[:space:]]*similar_dict:[[:space:]]*//p' "$CONFIG" | sed -n 1p)
  for d in "$d_qn" "$d_sim"; do
    if [[ -z "$d" ]]; then
      warn "config thiếu khai báo từ điển (qn_to_nom_dict / similar_dict) -> build sẽ hỏng."
    elif [[ -f "$d" ]]; then
      ok "từ điển: $d ($(( $(wc -l <"$d") - 1 )) dòng)"
    else
      die "KHÔNG thấy từ điển '$d' khai trong $CONFIG.
      Bước build đọc file này; thiếu nó là hỏng cả mẻ.
      Kiểm nhanh:  ls Dict/   ·  grep -n 'dict' $CONFIG"
    fi
  done

  if [[ -f nom-embed/best.pt ]]; then
    ok "checkpoint S3: nom-embed/best.pt"
  elif [[ -f nom-embed/last.pt ]]; then
    warn "chỉ có nom-embed/last.pt (không có best.pt) — S3 chạy trên checkpoint KHÔNG tốt nhất."
  else
    warn "thiếu nom-embed/best.pt — S3 tắt => tier SILVER SẬP ÂM THẦM về REVIEW."
  fi

  local fd_dir fd_first
  fd_dir=$(sed -n 's/^[[:space:]]*fd_cache_universal:[[:space:]]*//p' "$CONFIG" 2>/dev/null | sed -n 1p)
  fd_dir="${fd_dir:-gannhanocr-fd}"
  fd_first=$(find "$fd_dir" -name 'U+*.png' -print -quit 2>/dev/null || true)
  if [[ -n "$fd_first" ]]; then
    ok "kho glyph FD: $fd_dir (ví dụ $(basename "$fd_first"))"
  else
    warn "kho glyph FD '$fd_dir' TRỐNG (không có U+*.png) -> S3 mất glyph tham chiếu -> tier SILVER hỏng."
  fi

  # crop-proto — index.csv trỏ dataset_out/gold/*.png; thiếu/hụt -> luật
  # s2_inter_s3_corrected biến mất -> SILVER −32% mà KHÔNG ném lỗi nào.
  local idx_csv first_crop
  idx_csv="pipeline/align_engine/data/index.csv"
  if [[ ! -f "$idx_csv" ]]; then
    warn "thiếu $idx_csv -> crop-protos = 0 -> SILVER tụt ~32%, KHÔNG có lỗi nào được ném ra."
  else
    first_crop=$(awk -F, 'NR==2{print $1; exit}' "$idx_csv" 2>/dev/null || true)
    if [[ -z "$first_crop" ]]; then
      warn "$idx_csv RỖNG (không có dòng dữ liệu) -> crop-protos = 0 -> SILVER tụt ~32%."
    elif [[ -f "$first_crop" ]]; then
      ok "crop-proto: $idx_csv -> $first_crop (có thật, $(wc -l <"$idx_csv" | tr -d ' ') dòng)"
      # THẾ HỆ: tệp tồn tại chưa đủ — index đời cũ trỏ sách tên `yen*` trong khi bộ
      # nhãn hiện hành dùng `stt*`, giao nhau = 0 nên crop-proto rỗng trên thực tế
      # mà preflight cũ vẫn báo xanh.
      local n_old n_new
      # `grep -c` KHÔNG khớp gì thì vẫn IN "0" rồi thoát mã 1, nên `|| echo 0` in thêm
      # một "0" nữa: biến thành "0\n0" và (( )) sặc cú pháp. Gán rồi mới chữa mã thoát.
      n_old=$(grep -c '/yen[0-9]*_' "$idx_csv" 2>/dev/null) || n_old=0
      n_new=$(grep -c '/stt[0-9]*_' "$idx_csv" 2>/dev/null) || n_new=0
      if (( n_old > 0 && n_new == 0 )); then
        warn "crop-proto LỆCH THẾ HỆ: $idx_csv có $n_old dòng trỏ sách 'yen*' và 0 dòng 'stt*',
      trong khi bộ nhãn hiện hành dùng 'stt*' -> giao nhau = 0 -> crop-protos RỖNG
      trên thực tế dù tệp vẫn tồn tại. Sinh lại index sau lần build tới."
      fi
    else
      warn "crop-proto TRỎ HỤT: $idx_csv dòng 2 = '$first_crop' KHÔNG có trên đĩa
      -> crop-protos = 0 -> SILVER tụt ~32% ÂM THẦM.
      Cách sửa: chạy build một lần (PASS2 dựng lại crops), rồi chạy tiếp."
    fi
  fi

  log "${BLD}--- HẾT PREFLIGHT: $N_WARN cảnh báo ------------------------------${RST}"
}

# ============================== CÁC BƯỚC =====================================
# ---- 1/7 setup --------------------------------------------------------------
step_setup() {
  banner 1 setup "kiểm cấu hình, đường dẫn, tài nguyên (pipeline.step0_setup)"
  X "$PY" -m pipeline.step0_setup "$CONFIG"
}

# ---- 2/7 extract ------------------------------------------------------------
# Cache OCR trong prepared/*/detected/*_ocr_cache.json = PRIMARY DATA.
# Còn cache = tái lập được. Xoá cache = gọi API ngoài = KHÔNG tái lập + tốn tiền.
step_extract() {
  banner 2 extract "PDF -> khung -> OCR (cache) -> 9 cột/trang | sách: $BOOKS_LABEL"
  local b
  if (( FRESH_OCR )); then
    for b in $BOOKS; do
      find "prepared/$b/detected" -name '*_ocr_cache.json' -delete 2>/dev/null || true
    done
    warn "cache OCR đã bị xoá cho: $BOOKS_LABEL — lần extract này SẼ GỌI API NGOÀI."
  fi
  for b in $BOOKS; do
    X "$PY" -m pipeline.step1_extract "$CONFIG" "$b"
  done
  log ""
  info "kiểm số cột OCR sau extract (mong đợi ĐÚNG 9 cột/trang)"
  for b in $BOOKS; do
    "$PY" pipeline/check_ocr_columns.py --book "$b" 2>/dev/null \
      | grep -E "OK\(=9\)|THIẾU|DƯ|≠9" || true
  done
}

# ---- 3/7 build --------------------------------------------------------------
# -> dataset_out/labels.csv + crops gold/silver/syllable
# LƯU Ý: build_dataset.py duyệt TOÀN BỘ config["books"] (cả 3 sách), KHÔNG lọc
# theo sách vừa chọn ở bước extract. Đây là chủ ý: nếu chỉ chọn 1 sách để
# extract hôm nay, build vẫn gộp dữ liệu 2 sách kia từ prepared/ của các lần
# chạy trước — KHÔNG ghi đè labels.csv thành bản thiếu sách (khác run_book*.sh,
# vốn lọc config chỉ còn 1 sách nên có bẫy ghi đè — xem chú thích trong đó).
step_build() {
  banner 3 build "align_engine.build_dataset: banded-DP align + consensus tier + crops (cả 3 sách trong config)"
  # S3 TẮT (A-10 giai đoạn 2, 16/09): KHÔNG truyền --use-s3. Lý do: error-AUC 0,566/0,577
  # (CI chứa 0,5) — không tín hiệu thị giác nào được quyết tier; bỏ luôn split nên
  # proto-index GOLD∧train không còn nguồn. Muốn tái lập thế hệ ≤25/08 thì thêm tay --use-s3.
  X "$PY" -m pipeline.align_engine.build_dataset --config "$CONFIG" --reseg "$RESEG" \
      --out "$DS_OUT"
  [[ -f "$LABELS_RAW" ]] || die "bước build không sinh $LABELS_RAW"
  # --- CHẤM CHIỀU CROP BẰNG MÁY, KHÔNG BẰNG MẮT NGƯỜI ------------------------
  # Mẻ chấm 2026-08-04 hỏng ở chiều CROP (κ = 0,14) chứ không ở chiều NHÃN. Người
  # không nhất quán khi vừa phải đọc chữ vừa phán xét ảnh cắt; máy đo hình học thì
  # nhất quán tuyệt đối. Ba cột này tách bạch hai việc đó — đặc tả T4.4 đặt hàng
  # nhưng chưa từng được làm. Chạy ngay sau build nên luôn khớp crop vừa cắt; các
  # bước 4-7 giữ nguyên cột lạ (remediate dùng pandas, export dùng reader.fieldnames).
  X "$PY" -m pipeline.tools.enrich_crop_quality --labels "$LABELS_RAW" --src-root "$DS_OUT"
}


# ---- CHỐT CHẶN sha256 --------------------------------------------------------
# Ghi vân tay của mọi tạo phẩm nhãn sau mỗi bước quan trọng. Đây là chốt chặn cho
# lỗi B8: mẻ audit người phải rút từ ĐÚNG bản labels_final.csv được export, và cách
# duy nhất chứng minh điều đó là so sha256. Không có file này thì "bộ đem đo" và
# "bộ đem nộp" lại trôi khỏi nhau như hồi trước 08-11.
checkpoint() {
  local tag="$1"; shift
  local f
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    printf '%s  %s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$tag" \
        "$(shasum -a 256 "$f" | awk '{print $1"  "$2}')" >> "$CHECKSUMS"
  done
  log "  ${CYA}sha256 -> $CHECKSUMS ($tag)${RST}"
}

# ---- THỜI GIAN TỪNG BƯỚC ----------------------------------------------------
# tick <tên bước>: in số giây bước vừa chạy (từ mốc T_STEP) và ghi một dòng `thoi_gian`
# vào CHECKSUMS để lần chạy nào cũng có bằng chứng thời gian thật (flow N19 "thời gian
# thật") — không ai còn phải ước lượng "build ~40 phút" từ trí nhớ.
T_STEP=$SECONDS
tick() {
  local dt=$((SECONDS - T_STEP)); T_STEP=$SECONDS
  log "  ${CYA}⏱ $1: ${dt}s${RST}"
  mkdir -p "$(dirname "$CHECKSUMS")"
  printf '%s  thoi_gian  %s  %ss\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$1" "$dt" >> "$CHECKSUMS"
}

# ---- CHỐT CHẶN QĐ-01 (N13) ---------------------------------------------------
# assert_qd01 <labels.csv> <nhãn bước> [final]
# Đếm ô mang rule tiền tố `quyet_dinh_nguoi:` (lock + pending + qd01a) trong tệp nhãn và
# so với KỲ VỌNG suy từ config: số dòng qd01_cells.csv (2.014, N0d) TRỪ ô trong khoá mà
# người đã quyết `bo` (mất tiền tố, qd01_excluded=1) CỘNG ô NGOÀI khoá người quyết
# `giu_2029A`/`khac:` (được khoá thêm, A-3) — đúng ngữ nghĩa apply_qd01_lock (N5g).
# Gọi SAU MỖI bước hậu xử lý (remediate, confusion, s3_unwind, glyph) để bước nào làm rơi
# ô QĐ-01 thì dừng ngay tại bước đó, không đợi tới export. Với `final` (trước export) đòi
# thêm pending == 0: ô pending là ô người CHƯA quyết (QĐ-01a) — export bộ có ô chưa quyết
# là giao nộp phán quyết chưa tồn tại. Đếm bằng Python theo ĐÚNG CỘT rule (grep -c đếm cả
# dòng có chuỗi ở cột khác) và giữ tiền tố qua hậu tố `|quarantine_dup`… của remediate.
QD01_CELLS="${QD01_CELLS:-config/qd01_cells.csv}"
QD01A_DECISIONS="${QD01A_DECISIONS:-config/qd01a_decisions.csv}"
assert_qd01() {
  local f="$1" tag="$2" final="${3:-}"
  [[ -f "$QD01_CELLS" ]] || die "thiếu $QD01_CELLS — sinh bằng python -m pipeline.tools.sinh_qd01_cells (N0d)"
  [[ -f "$f" ]] || die "assert_qd01($tag): không thấy $f"
  local n_qd n_pending n_exp n_cells
  # bash 3.2 (macOS) không phân tích được heredoc lồng trong <( ) -> mã Python đi qua biến.
  local _py='
import csv, sys
def key(r):
    return (r["book"], r["page"], int(float(r["column"])), int(float(r["nom_idx"])))
labels, cells_csv, dec_csv = sys.argv[1:4]
with open(cells_csv, encoding="utf-8", newline="") as fh:
    cells = {key(r) for r in csv.DictReader(fh) if r.get("nom_idx", "") != ""}
dec = {}
try:
    with open(dec_csv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("nom_idx", "") != "":
                dec[key(r)] = (r.get("quyet") or "").strip()
except FileNotFoundError:
    pass
n_bo = sum(1 for k, q in dec.items() if k in cells and q == "bo")
n_ngoai = sum(1 for k, q in dec.items() if k not in cells and (q == "giu_2029A" or q.startswith("khac:")))
n = p = 0
with open(labels, encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        rule = (r.get("rule") or "")
        if rule.startswith("quyet_dinh_nguoi:"):
            n += 1
            if rule.split("|", 1)[0] == "quyet_dinh_nguoi:pending":
                p += 1
print(n, p, len(cells) - n_bo + n_ngoai, len(cells))
'
  read -r n_qd n_pending n_exp n_cells < <("$PY" -c "$_py" "$f" "$QD01_CELLS" "$QD01A_DECISIONS")
  if [[ "$n_qd" != "$n_exp" ]]; then
    die "assert_qd01($tag): $f có $n_qd ô rule 'quyet_dinh_nguoi:*' nhưng kỳ vọng $n_exp
      (= $n_cells ô khoá trong $QD01_CELLS − ô 'bo' + ô ngoài khoá đã quyết trong $QD01A_DECISIONS).
      Bước '$tag' đã làm rơi/thêm ô QĐ-01 — DỪNG, không chạy tiếp."
  fi
  if [[ -n "$final" && "$n_pending" -gt 0 ]]; then
    # ALLOW_PENDING=1 — CHỈ cho bản THỬ NGHIỆM (DS_OUT != dataset_out): cho export đi tiếp
    # dù còn ô pending, để có labels_final/re-dataset tạm cho đối soát thế hệ (N15) và
    # thăm dò trước QĐ-01a (N16). Bộ thật KHÔNG BAO GIỜ được nới: ô chưa quyết mà vào bộ
    # giao nộp là giao nộp phán quyết chưa tồn tại.
    if [[ "${ALLOW_PENDING:-0}" == "1" && "$DS_OUT" != "dataset_out" ]]; then
      warn "assert_qd01($tag): còn $n_pending ô QĐ-01 PENDING — ALLOW_PENDING=1 (bản thử nghiệm $DS_OUT) nên đi tiếp; bộ này KHÔNG được thăng cấp"
    else
      die "assert_qd01($tag): còn $n_pending ô QĐ-01 PENDING (rule quyet_dinh_nguoi:pending).
      Điền phán quyết người vào $QD01A_DECISIONS rồi chạy lại (rebuild) — KHÔNG sửa tay CSV.
      Danh sách ô: $DS_OUT/qd01a_pending.csv · report: $DS_OUT/glyph_fix_kiem_report.json"
    fi
  fi
  ok "QĐ-01 ($tag): $n_qd/$n_exp ô khoá còn nguyên (cells $n_cells) · pending $n_pending"
}

# ---- 4/7 remediate ----------------------------------------------------------
# -> labels_remediated.csv + remediation_report.json
step_remediate() {
  banner 4 remediate "kiểm kê trùng lặp + cách ly/hạ tier -> $LABELS_REMED"
  X "$PY" -m pipeline.remediation --labels "$LABELS_RAW" --out "$DS_OUT" census
  X "$PY" -m pipeline.remediation --labels "$LABELS_RAW" --out "$DS_OUT" apply --tau "$TAU_REMEDIATE"
  [[ -f "$LABELS_REMED" ]] || die "bước remediate không sinh $LABELS_REMED"
}

# ---- 5/7 confusion ----------------------------------------------------------
# -> labels_final.csv + confusion_fix_report.json
# Hàm thuần, idempotent: đọc labels_remediated.csv, hạ tier các cặp (âm tiết, chữ)
# liệt trong config/confusion_fixes.yaml -> ghi BẢN CÔNG BỐ. KHÔNG remap codepoint.
# Đây là bộ nhãn mà export VÀ mẻ audit người CÙNG đọc — một nguồn duy nhất.
step_confusion() {
  banner 5 confusion "hạ tier confusion hệ thống đã chứng minh -> $LABELS_FINAL (bản công bố)"
  [[ -f "$CONFUSION_FIXES" ]] || die "không thấy $CONFUSION_FIXES.
      Bước này quyết định bộ nhãn công bố nên KHÔNG được bỏ qua âm thầm:
      thiếu file = mọi confusion đã chứng minh sẽ lặng lẽ ở lại tier GOLD/SILVER.
      Nếu thật sự chưa có fix nào, tạo file với đúng nội dung:  fixes: []"
  X "$PY" -m pipeline.remediation.confusion_fix \
      --in "$LABELS_REMED" --out "$LABELS_FINAL" --fixes "$CONFUSION_FIXES" --measure
  [[ -f "$LABELS_FINAL" ]] || die "bước confusion không sinh $LABELS_FINAL"
}

# ---- 6/7 gỡ S3 -------------------------------------------------------------
# -> ghi ĐÈ labels_final.csv (luỹ đẳng) + s3_unwind_report.json
# Đo 2026-08-19 trên 826 verdict NGƯỜI: S3 cũ error-AUC 0,566 [0,459-0,672];
# ArcFace retrain (K=3+SAM, val top-1 0,806) 0,577 [0,442-0,706] — CI của cả hai
# đều CHỨA 0,5. Không tín hiệu thị giác nào được quyền phong/hạ tier nữa.
#   * 1.185 ô `|demoted_lowcos_s3` -> trả về GOLD (cờ readmitted_from_s3_demotion)
#   * 10.890 ô SILVER -> SILVER_uncalibrated (0 verdict người; rơi khỏi bộ giao nộp,
#     KHÔNG bị xoá — vẫn rút mẫu chấm tay được)
#   * 10.934 ô below_visual_threshold -> no_s1_inter_s2 (đổi tên lý do, giữ REVIEW)
# Bước này KHÔNG sửa nhãn nào và tự dừng nếu cột label bị đụng.
step_s3unwind() {
  banner 6 "gỡ S3" "gỡ tín hiệu thị giác khỏi mọi quyết định tier -> $LABELS_FINAL"
  X "$PY" -m pipeline.remediation.s3_unwind \
      --in "$LABELS_FINAL" --out "$LABELS_FINAL" \
      --report "$DS_OUT/s3_unwind_report.json" --apply
  [[ -f "$DS_OUT/s3_unwind_report.json" ]] || die "bước gỡ S3 không sinh báo cáo"
}

# ---- 7/8 quyết định glyph -----------------------------------------------------
# -> ghi ĐÈ labels_final.csv (luỹ đẳng) + glyph_fix_report.json
# Đường để một PHÁN QUYẾT NGƯỜI đi vào bộ nhãn. Có những chữ mà bộ OCR Nôm mù hẳn — rõ
# nhất là "người": âm phổ biến nhất cả ba cuốn (2.281 ô) nhưng 94% bị giữ lại, trong khi
# tỷ lệ rớt chung là 31%. Từ điển BIẾT đáp án (𠊚 nằm trong 20 ứng viên, 37/49 ô GOLD dùng
# nó); bộ OCR chỉ là không đọc nổi glyph, nên S1∩S2 không bao giờ khớp.
# Mọi đường tự động đã thử và bác: Unihan kVietnamese 0 ô · hvdic 0 ô · khôi phục dấu 0 ô ·
# cầu tự dạng hai chiều (T7) dưới ngưỡng · dịch vụ đọc ngoài 0/16 sống sót phản biện.
# CHẠY SAU gỡ S3 để không tín hiệu thị giác nào đụng lại nhãn người đã quyết.
# Mỗi quyết định BẮT BUỘC khai xuất xứ; thiếu -> module TỪ CHỐI chạy, không chạy tiếp.
# TỪ 16/09 (A-2/A-12, flow N12): chế độ `kiem` — KHÔNG GÁN GÌ. Khoá QĐ-01 đã được build áp
# trong PASS 1c theo config/qd01_cells.csv (khoá bền book,page,column,nom_idx) +
# qd01a_decisions.csv (phán quyết người cho ô trôi). Bước này chỉ ĐỐI CHIẾU và ghi
# $DS_OUT/glyph_fix_kiem_report.json (n_khop/n_pending/n_mat/n_bbox_doi/n_md5_doi); exit 1
# nếu khớp + pending != số ô khoá. Vẫn truyền --config để kiểm xuất xứ của phán quyết âm.
# Chế độ `am` cũ (gán theo âm, --apply) chỉ dùng tay khi tái lập thế hệ ≤25/08.
step_glyph() {
  banner 7 "kiểm QĐ-01" "đối chiếu khoá QĐ-01 (không gán) -> $DS_OUT/glyph_fix_kiem_report.json"
  [[ -f "$QD01_CELLS" ]] || die "thiếu $QD01_CELLS — sinh bằng python -m pipeline.tools.sinh_qd01_cells (N0d)"
  X "$PY" -m pipeline.remediation.glyph_fix --mode kiem \
      --in "$LABELS_FINAL" --cells "$QD01_CELLS" --decisions "$QD01A_DECISIONS" \
      --config "$GLYPH_DECISIONS" --report "$DS_OUT/glyph_fix_kiem_report.json"
}

# ---- 8/8 export -------------------------------------------------------------
# -> dataset/labels.csv + ảnh crop copy hẳn (chỉ tier usable: GOLD+SILVER+SYLLABLE)
# Nguồn là labels_final.csv (SAU confusion-fix), KHÔNG phải labels_remediated.csv —
# xem cảnh báo ở đầu file. XOÁ SẠCH dataset/ trước khi ghi -> luôn là bản MỚI NHẤT,
# không cộng dồn qua các lần chạy trước. dataset_out/ KHÔNG bị đụng — vẫn còn
# labels_remediated.csv đầy đủ mọi tier (kể cả REVIEW/QUARANTINE) để tra cứu sau.
step_export() {
  # --- CỔNG CHẶN QĐ-01: ĐẾM TRƯỚC KHI XOÁ (N13, final) ---------------------------
  # export_final_dataset.py chạy shutil.rmtree(thư mục đích) RỒI mới ghi lại. Nếu một bước
  # trước đó làm rơi ô QĐ-01 (trước 2026-09-09 step_glyph được định nghĩa mà KHÔNG AI GỌI)
  # thì lệnh xoá đó nuốt mất 2.014 ô phán quyết NGƯỜI và không hồi được từ chính lần chạy
  # này. Đếm TRƯỚC KHI XOÁ, và đòi pending == 0 — ô chưa quyết không được vào bộ giao nộp.
  assert_qd01 "$LABELS_FINAL" "trước export" final

  # --- CÓ PHÁN QUYẾT NGƯỜI CHƯA? ----------------------------------------------
  # Hai đường nạp phán quyết, dò cả hai:
  #   (A) re-dataset/verdicts.csv  — ĐỘI NGOÀI chấm trên chính bộ đem chấm. Đường CHÍNH.
  #   (B) human_audit/.../verdicts*.jsonl — mẻ mẫu, dùng khi chỉ chấm mẫu để ước lượng.
  local _v; _v=$(ls "$REDATASET_DIR/verdicts.csv" 2>/dev/null | head -1 || true)
  [[ -n "$_v" ]] || _v=$(ls "$AUDIT_DIR"/verdicts*.jsonl 2>/dev/null | head -1 || true)
  local OUT_DIR NHAN
  if [[ -n "$_v" ]]; then
    OUT_DIR="$FINAL_DIR"; NHAN="CUỐI CÙNG (đã nạp phán quyết người)"
    banner "7a" "nạp phán quyết" "áp verdict của người + ước lượng precision theo tầng"
    # Bước này TỪ CHỐI chạy tiếp nếu κ liên-người < 0,60 — khi hai người không cùng
    # tiêu chí thì con số precision là tiêu chí của MỘT NGƯỜI, không phải chất lượng dữ liệu.
    # Bản thử nghiệm: bảng precision đi theo $DS_OUT, KHÔNG ghi docs/BANG_PRECISION.md.
    # (bash 3.2 + set -u: mảng rỗng phải nở bằng ${_rep[@]+"${_rep[@]}"})
    local _rep=()
    [[ "$DS_OUT" == "dataset_out" ]] || _rep=(--report "$DS_OUT/BANG_PRECISION.md")
    X "$PY" -m pipeline.remediation.apply_verdicts \
        --in "$LABELS_FINAL" --out "$LABELS_FINAL" \
        --batch "$AUDIT_DIR" --redataset "$REDATASET_DIR" ${_rep[@]+"${_rep[@]}"}
  else
    OUT_DIR="$REDATASET_DIR"; NHAN="ĐEM CHẤM (CHƯA kiểm chứng)"
    log ""
    log "  ${YEL}chưa có $AUDIT_DIR/verdicts*.jsonl${RST}"
    log "  ${YEL}-> xuất ra $REDATASET_DIR/ để đem chấm tay, KHÔNG phải bộ cuối cùng${RST}"
  fi

  banner 8 export "xuất bộ $NHAN -> $OUT_DIR/ (tự chứa)"
  X "$PY" pipeline/export_final_dataset.py \
      --labels "$LABELS_FINAL" --src-root "$DS_OUT" --out "$OUT_DIR"
  [[ -f "$OUT_DIR/labels.csv" ]] || die "bước export không sinh $OUT_DIR/labels.csv"
  FINAL_OUT="$OUT_DIR"

  # --- TÀI LIỆU ĐI KÈM BỘ GIAO NỘP -----------------------------------------
  # Với ngành Hán Nôm, một bộ dữ liệu KHÔNG có lai lịch thư tịch là KHÔNG TRÍCH DẪN
  # ĐƯỢC: người đọc không biết ba cuốn này là bản in nào, lưu ở đâu, ký hiệu gì, nên
  # không kiểm lại được ô nhãn nào. Mọi con số trong DATASHEET đọc TỪ labels.csv nên
  # không thể lệch với dữ liệu. Các mục chỉ người biết được để `⬜ CHƯA ĐIỀN`.
  X "$PY" -m pipeline.tools.make_dataset_docs --dataset "$FINAL_OUT"
  # bản .xlsx đọc bằng Excel của chính labels.csv (ngoài git — đầu ra dựng lại được)
  X "$PY" -m pipeline.tools.make_xlsx --labels "$FINAL_OUT/labels.csv"

  # --- THĂNG CẤP (N19) — chỉ khi dựng BỘ THẬT ----------------------------------
  # S3 tắt -> cache nguyên mẫu s3_proto_cache.pkl (ký bằng mtime index.csv) là tạo phẩm
  # của thế hệ cũ: xoá để repro_check R2 không đối chiếu với một cache không còn ai dựng
  # (KHÔNG chạy rebuild_proto_index: nó đổi mtime index.csv -> R2 lệch, và cần cột split).
  # Rồi đồng bộ docs/BANG_SO_LIEU_CHINH_THUC.md từ đúng bộ vừa export để MỘT commit là đủ.
  if [[ "$DS_OUT" == "dataset_out" ]]; then
    if [[ -f pipeline/align_engine/s3_proto_cache.pkl ]]; then
      X rm -f pipeline/align_engine/s3_proto_cache.pkl
    fi
    X "$PY" -m pipeline.tools.update_bang_so_lieu
  else
    info "bỏ qua rm s3_proto_cache.pkl + update_bang_so_lieu (DS_OUT thử nghiệm: $DS_OUT)"
  fi
}

# ====================== FREEZE / EVIDENCE ====================================
evidence() {
  # nom-embed/best.pt: checkpoint S3 nằm trong SUBMODULE và đang có thay đổi CHƯA
  # COMMIT (mục 0.4) -> con trỏ submodule KHÔNG nhận diện được mô hình thật đã sinh ra
  # cột s3_cosine. Băm thẳng tệp là cách duy nhất hiện có để chỉ đúng mô hình đã dùng.
  # HOA/THƯỜNG: git lưu `Dict/` (13 tệp) còn thư mục trên đĩa máy này tên `dict/`.
  # macOS không phân biệt nên cả hai cùng trỏ một chỗ; trên Linux thì CHỈ MỘT cái tồn
  # tại. Ghim cứng `dict/` sẽ làm 2/10 dòng băm thành "(chưa có)" và check_evidence.sh
  # exit 1 ngay trên clone Linux — tức phá đúng tiêu chí "clone sạch ra cùng sha256".
  # Giải như core/text/dictionary.py:dict_dir(): dùng cái CÓ THẬT.
  local DICT_DIR="Dict"; [[ -d "$DICT_DIR" ]] || DICT_DIR="dict"
  # TỪ ĐIỂN PHẢI CÓ TRONG CHUỖI BẰNG CHỨNG (thêm 2026-08-24, khi làm T5).
  # Mọi nhãn GOLD đều do QuocNgu_SinoNom.csv quyết (s1_inter_s2_direct: ocr_char phải
  # là một âm đọc trong từ điển) và SinoNom_Similar.csv quyết luật bắc cầu — thế mà
  # hash của chúng chưa từng xuất hiện trong EVIDENCE_INDEX.md. Chúng CÓ trong git
  # (dưới `Dict/`, xem dict_dir() về chuyện hoa/thường), nên có lịch sử phiên bản;
  # cái thiếu là mối nối giữa MỘT LẦN CHẠY cụ thể và BẢN từ điển nó đã dùng. Băm ở
  # đây khép mối nối đó, cùng cơ chế với nom-embed/best.pt và index.csv.
  # Băm ĐẦU RA THẬT của lần chạy này: re-dataset/ khi chưa có phán quyết, dataset/ khi
  # đã có. Ghim cứng "$FINAL_DIR" sẽ băm một thư mục có thể còn chưa tồn tại.
  local _out="${FINAL_OUT:-$FINAL_DIR}"
  local files=("$LABELS_RAW" "$LABELS_REMED" "$LABELS_FINAL" "$_out/labels.csv" \
               "nom-embed/best.pt" "pipeline/align_engine/data/index.csv" \
               "$DICT_DIR/QuocNgu_SinoNom.csv" "$DICT_DIR/SinoNom_Similar.csv" \
               "train_crop/detector_r34.best.pt" "config/pipeline.yaml")
  local sha_cmd=""
  if command -v shasum >/dev/null 2>&1; then sha_cmd="shasum -a 256"
  elif command -v sha256sum >/dev/null 2>&1; then sha_cmd="sha256sum"; fi
  [[ -n "$sha_cmd" ]] || { warn "không có shasum/sha256sum — bỏ qua bảng bằng chứng"; return 0; }
  log ""
  log "${BLD}--- BẰNG CHỨNG (sha256) -> $EVIDENCE ---------------------${RST}"
  mkdir -p "$(dirname "$EVIDENCE")"
  {
    printf '\n## Lần chạy %s\n\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'sách: %s | reseg=%s | config=%s\n\n' "$BOOKS_LABEL" "$RESEG" "$CONFIG"
    printf '| file | sha256 |\n|---|---|\n'
  } >>"$EVIDENCE"
  local f h
  for f in "${files[@]}"; do
    if [[ -f "$f" ]]; then
      h=$($sha_cmd "$f" | awk '{print $1}')
      printf '    %-42s %s\n' "$f" "$h"
      printf '| `%s` | `%s` |\n' "$f" "$h" >>"$EVIDENCE"
    else
      printf '    %-42s %s\n' "$f" "(chưa có)"
      printf '| `%s` | (chưa có) |\n' "$f" >>"$EVIDENCE"
    fi
  done
  ok "bảng sha256 đã ghi vào $EVIDENCE (nhật ký)"

  # --- BẢN HIỆN HÀNH: khối luôn được GHI ĐÈ, không phải nhật ký ----------------
  # Trước 2026-08-23 evidence() CHỈ append mục "## Lần chạy", nên bảng §3 "đóng băng
  # 2026-07-20" chắc chắn lệch sau mỗi lần chạy, và hash của labels_final.csv hiện
  # hành KHÔNG xuất hiện ở đâu trong tệp. Khối dưới đây là chỗ DUY NHẤT bảo đảm phản
  # ánh trạng thái hiện tại — mọi phép đối chiếu phải dùng nó.
  local blk_start='<!-- HIEN_HANH:START -->' blk_end='<!-- HIEN_HANH:END -->'
  local tmp; tmp="$(mktemp)"
  {
    printf '%s\n' "$blk_start"
    printf '## BẢN HIỆN HÀNH (tự sinh — ghi đè mỗi lần chạy, ĐỪNG sửa tay)\n\n'
    printf -- '- sinh lúc: `%s`\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf -- '- commit  : `%s`\n' "$(git rev-parse --short HEAD 2>/dev/null || echo 'không phải git')"
    # COMMIT SHA MỘT MÌNH KHÔNG ĐỦ ĐỊNH DANH LẦN CHẠY. Nếu cây làm việc bẩn thì mã
    # thực thi KHÁC mã ở commit đó, và hai lần chạy cùng hash vẫn không chứng minh
    # được gì về phiên bản mã. Bỏ qua submodule vì `nom-embed` luôn báo bẩn do
    # artefact con trỏ LFS (xem ghi chú bên dưới) chứ không phải do nội dung đổi.
    _dirty="$(git status --porcelain --untracked-files=no --ignore-submodules=all 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$_dirty" == "0" ]]; then
      printf -- '- cây làm việc: SẠCH — commit ở trên định danh đúng mã đã chạy\n'
    else
      printf -- '- cây làm việc: 🔴 BẨN (%s tệp đã sửa chưa commit) — commit ở trên KHÔNG\n' "$_dirty"
      printf -- '  định danh được mã đã chạy. Muốn tái lập thì phải commit trước khi chạy.\n'
      git status --porcelain --untracked-files=no --ignore-submodules=all 2>/dev/null \
        | head -12 | sed 's/^/    /'
    fi
    printf -- '- sách    : %s | reseg=%s\n\n' "$BOOKS_LABEL" "$RESEG"
    printf '| file | sha256 |\n|---|---|\n'
    local f h
    for f in "${files[@]}"; do
      if [[ -f "$f" ]]; then h=$($sha_cmd "$f" | awk '{print $1}')
      else h='(chưa có)'; fi
      printf '| `%s` | `%s` |\n' "$f" "$h"
    done
    printf '\n**Bộ dò ký tự**: `mdnt571/nom-char-det` @ HuggingFace,\n'
    printf -- '  tệp `detector_r34.PROD_e38_img1024.best.pt`, oid `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694`\n'
    printf -- '  ⚠️ CÙNG REPO có `detector_r34.best.pt` là MỘT MÔ HÌNH KHÁC (epoch 41, img 1280,\n'
    printf -- '  F1 0,8298 so với bản sản xuất epoch 38, img 1024, F1 0,8436) — 259/259 tensor khác\n'
    printf -- '  nhau. infer_centernet.py:198 lấy độ phân giải TỪ checkpoint nên thay nhầm sẽ\n'
    printf -- '  letterbox ở 1280 và cho hộp khác MÀ KHÔNG BÁO LỖI. Luôn dùng tệp tên PROD.\n'
    printf '\n**Checkpoint S3**: `mdnt571/nom-embed` @ HuggingFace, revision `7ff74f57c4be`\n'
    printf -- '- `best.pt` LFS oid = `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0`\n'
    printf -- '- `last.pt` LFS oid = `c05dd1723c751059`… (xem repo HF)\n'
    printf -- '- git báo `nom-embed` "modified" là ARTEFACT: huggingface_hub tải tệp THẬT ghi đè\n'
    printf -- '  con trỏ LFS 134 byte, nên git so 134 byte với 140 MB. `oid` trong con trỏ KHỚP\n'
    printf -- '  sha256 tệp trên đĩa VÀ khớp bản trên HF -> chuỗi xuất xứ NGUYÊN VẸN.\n'
    printf -- '  ĐỪNG `git checkout` các tệp này: sẽ thay tệp thật bằng con trỏ và làm sập S3.\n'
    printf '\nKiểm lại: `bash scripts/check_evidence.sh`\n'
    printf '%s\n' "$blk_end"
  } >"$tmp"

  if grep -qF "$blk_start" "$EVIDENCE" 2>/dev/null; then
    awk -v s="$blk_start" -v e="$blk_end" -v f="$tmp" '
      index($0,s){ while ((getline l < f) > 0) print l; close(f); skip=1; next }
      index($0,e){ skip=0; next }
      !skip' "$EVIDENCE" >"$EVIDENCE.new" && mv "$EVIDENCE.new" "$EVIDENCE"
  else
    cat "$tmp" >>"$EVIDENCE"
  fi
  rm -f "$tmp"
  ok "khối BẢN HIỆN HÀNH đã cập nhật trong $EVIDENCE"
}

# =============================== MAIN ========================================
log "${BLD}================================================================${RST}"
log "${BLD}  GanNhanOCR — sinh bộ dataset (setup -> extract -> build -> remediate -> confusion -> export)${RST}"
log "${BLD}================================================================${RST}"

preflight
ask_book_choice
ask_cache_choice

if [[ -f "$DS_OUT/.FROZEN" ]]; then
  confirm_frozen_override
fi

log ""
log "${BLD}Sẽ chạy:${RST} setup -> extract($BOOKS_LABEL) -> build(cả 3 sách, S3 TẮT) -> remediate -> confusion -> gỡ S3 -> kiểm QĐ-01 -> export"
log "  cache OCR : $([[ $FRESH_OCR == 1 ]] && echo 'XOÁ & OCR lại mới' || echo 'dùng cache cũ')"
log "  ${YEL}export sẽ XOÁ SẠCH thư mục đích rồi ghi lại bản mới nhất:${RST}"
log "  ${YEL}  chưa có phán quyết người -> $REDATASET_DIR/  (bộ ĐEM CHẤM)${RST}"
log "  ${YEL}  đã có phán quyết         -> $FINAL_DIR/      (bộ CUỐI CÙNG)${RST}"
if [[ "$NONINTERACTIVE" == "1" ]]; then
  info "NONINTERACTIVE=1 -> bắt đầu ngay (DS_OUT=$DS_OUT)"
else
  read -r -p "Enter để bắt đầu, Ctrl-C để huỷ... " _
fi

T_STEP=$SECONDS
step_setup;   tick setup
step_extract; tick extract
step_build
checkpoint build "$LABELS_RAW"; tick build
step_remediate
checkpoint remediate "$LABELS_REMED"; tick remediate
assert_qd01 "$LABELS_REMED" remediate
step_confusion
checkpoint confusion "$LABELS_FINAL"; tick confusion
assert_qd01 "$LABELS_FINAL" confusion
# N11: s3_unwind chạy VÔ ĐIỀU KIỆN (no-op tất định khi S3 tắt) để report luôn được ghi mới
step_s3unwind
checkpoint s3unwind "$LABELS_FINAL"; tick s3unwind
assert_qd01 "$LABELS_FINAL" s3unwind
# BƯỚC 7 PHẢI NẰM Ở ĐÂY, KHÔNG PHẢI CHỖ KHÁC. glyph_fix áp phán quyết NGƯỜI, nên nó
# phải chạy SAU bước máy cuối cùng đụng tier (s3_unwind) và TRƯỚC export. Đã chứng
# minh bằng phép chạy lại chứ không bằng chú thích: thứ tự này tái lập ĐÚNG BYTE cả
# dataset_out/labels_final.csv (550a9726…) lẫn re-dataset/labels.csv (c346a032…);
# đảo lại (glyph trước s3_unwind) cho tệp KHÁC — lệch 3 ô ở cột tier_goc và đổi cả
# thứ tự cột. Trước 2026-09-09 hàm này được ĐỊNH NGHĨA ở dòng 446 nhưng KHÔNG AI GỌI,
# nên mỗi lần chạy script lại sinh bộ 57.357 dòng, thiếu đúng 2.014 ô phán quyết người.
step_glyph
checkpoint glyph "$LABELS_FINAL"; tick glyph
assert_qd01 "$LABELS_FINAL" glyph
step_export
checkpoint export "${FINAL_OUT:-$FINAL_DIR}/labels.csv"; tick export
# Bằng chứng vào docs/ CHỈ khi dựng bộ thật: bản thử nghiệm (DS_OUT khác) mà ghi
# EVIDENCE_INDEX.md thì khối BẢN HIỆN HÀNH sẽ trỏ sang tệp không phải bộ giao nộp.
if [[ "$DS_OUT" == "dataset_out" ]]; then
  evidence
else
  info "bỏ qua evidence (DS_OUT thử nghiệm: $DS_OUT — không ghi docs/)"
fi

log ""
log "${BLD}================================================================${RST}"
if [[ "${FINAL_OUT:-}" == "$REDATASET_DIR" ]]; then
  log "${YEL}${BLD}  Xong — bộ ĐEM CHẤM (CHƯA kiểm chứng, tự chứa):${RST}"
else
  log "${GRN}${BLD}  Xong — bộ CUỐI CÙNG (đã nạp phán quyết người, tự chứa):${RST}"
fi
log "  ${FINAL_OUT:-$FINAL_DIR}/labels.csv  (kèm ảnh crop copy hẳn, README + DATASHEET)"
if [[ "${FINAL_OUT:-}" == "$REDATASET_DIR" ]]; then
  log ""
  log "  ${YEL}${BLD}ĐÂY LÀ BỘ ĐEM CHẤM, CHƯA PHẢI BỘ CUỐI CÙNG.${RST}"
  log ""
  log "  Giao CẢ thư mục ${BLD}$REDATASET_DIR/${RST} cho đội chấm — nó TỰ ĐỦ."
  log "  Cột nào nghĩa gì: $REDATASET_DIR/README.md · giới hạn: DATASHEET.md"
  log ""
  log "  Nhận về, đặt ĐÚNG hai tệp vào $REDATASET_DIR/ :"
  log "     verdicts.csv     cột image,verdict,nguoi_cham,ghi_chu"
  log "                      verdict chỉ 3 giá trị: dung · sai · khong_doc_duoc"
  log "     NGUOI_CHAM.md    khai ai chấm — THIẾU thì pipeline TỪ CHỐI nạp"
  log ""
  log "  Rồi CHẠY LẠI script này -> bộ CUỐI vào $FINAL_DIR/ + docs/BANG_PRECISION.md"
  log ""
  log "  (Chỉ muốn chấm MẪU ~960 ô để có khoảng tin cậy thay vì chấm cả 57 nghìn:"
  log "   $PY -m pipeline.ground_truth.make_combined_batch --by-rule \\"
  log "     --n-gold 380 --n-similar 260 --n-silver 0 --n-syllable 260 --n-repeat 60 --seed 2026)"
fi
log ""
log "  Bản làm việc trung gian (đủ mọi tier kể cả REVIEW/QUARANTINE, không bị đụng):"
log "  $LABELS_REMED   (trước confusion-fix)"
log "  $LABELS_FINAL   (BẢN CÔNG BỐ — nguồn của bộ xuất và của mẻ audit người)"
log "  $DS_OUT/{gold,silver,syllable}/"
log "  cảnh báo    : $N_WARN"
log "  thời gian   : tổng ${SECONDS}s (từng bước: grep thoi_gian $CHECKSUMS)"
log ""
log "  ${YEL}Nếu có dùng mẻ MẪU (đường phụ): nhãn vừa đổi -> mẻ dựng từ bản cũ đã hết hiệu lực.${RST}"
log "  Dựng lại: rm -rf $AUDIT_DIR && \\"
log "            $PY -m pipeline.ground_truth.make_combined_batch --seed 2026"
log ""
log "  TẠM BỎ QUA (theo yêu cầu): audit người · fuse · publish của bản 8-bước cũ"
log "  Bản đầy đủ 8 bước còn trong lịch sử git: git log -- run_pipeline.sh"
log "${BLD}================================================================${RST}"
