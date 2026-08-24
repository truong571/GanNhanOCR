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
#   5 confusion   pipeline.remediation.confusion_fix -> labels_final.csv (BẢN CÔNG BỐ)
#                 hạ tier các confusion HỆ THỐNG đã chứng minh bằng audit người
#   7 export      pipeline/export_final_dataset.py -> dataset/ (chỉ tier
#                 GOLD+SILVER+SYLLABLE = usable; XOÁ SẠCH dataset/ cũ trước khi ghi)
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
LABELS_RAW="dataset_out/labels.csv"
LABELS_REMED="dataset_out/labels_remediated.csv"
LABELS_FINAL="dataset_out/labels_final.csv"   # BẢN CÔNG BỐ — nguồn của export + audit
CONFUSION_FIXES="${CONFUSION_FIXES:-config/confusion_fixes.yaml}"
FINAL_DIR="dataset"
EVIDENCE="docs/EVIDENCE_INDEX.md"
CHECKSUMS="dataset_out/CHECKSUMS.txt"

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
  printf '%s>>> BƯỚC %s/7 · %s%s — %s\n' "$BLD" "$1" "$2" "$RST" "$3"
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
  read -r -p "Gõ đúng chữ XOA để xác nhận (Enter/khác = huỷ, quay về dùng cache cũ): " typed
  [[ "$typed" == "XOA" ]]
}

ask_cache_choice() {
  local choice
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
  log "${RED}${BLD}dataset_out/.FROZEN tồn tại${RST} — bản đã đóng băng để trích số vào luận văn."
  log "${RED}Chạy tiếp (bước build) sẽ GHI ĐÈ bằng chứng đã đóng băng đó.${RST}"
  read -r -p "Gõ đúng chữ GHIDE để tiếp tục (Enter/khác = huỷ chạy): " typed
  [[ "$typed" == "GHIDE" ]] || die "Đã huỷ — xoá dataset_out/.FROZEN nếu thật sự muốn dựng lại."
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
      Tạo venv : python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
      Hoặc     : PYTHON_BIN=/path/to/python ./run_pipeline.sh"
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
      n_old=$(grep -c '/yen[0-9]*_' "$idx_csv" 2>/dev/null || echo 0)
      n_new=$(grep -c '/stt[0-9]*_' "$idx_csv" 2>/dev/null || echo 0)
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
  X "$PY" -m pipeline.align_engine.build_dataset --config "$CONFIG" --use-s3 --reseg "$RESEG"
  [[ -f "$LABELS_RAW" ]] || die "bước build không sinh $LABELS_RAW"
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

# ---- 4/7 remediate ----------------------------------------------------------
# -> labels_remediated.csv + remediation_report.json
step_remediate() {
  banner 4 remediate "kiểm kê trùng lặp + cách ly/hạ tier -> $LABELS_REMED"
  X "$PY" -m pipeline.remediation --labels "$LABELS_RAW" --out dataset_out census
  X "$PY" -m pipeline.remediation --labels "$LABELS_RAW" --out dataset_out apply --tau "$TAU_REMEDIATE"
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
      --report dataset_out/s3_unwind_report.json --apply
  [[ -f dataset_out/s3_unwind_report.json ]] || die "bước gỡ S3 không sinh báo cáo"
}

# ---- 7/7 export -------------------------------------------------------------
# -> dataset/labels.csv + ảnh crop copy hẳn (chỉ tier usable: GOLD+SILVER+SYLLABLE)
# Nguồn là labels_final.csv (SAU confusion-fix), KHÔNG phải labels_remediated.csv —
# xem cảnh báo ở đầu file. XOÁ SẠCH dataset/ trước khi ghi -> luôn là bản MỚI NHẤT,
# không cộng dồn qua các lần chạy trước. dataset_out/ KHÔNG bị đụng — vẫn còn
# labels_remediated.csv đầy đủ mọi tier (kể cả REVIEW/QUARANTINE) để tra cứu sau.
step_export() {
  banner 7 export "xuất bộ dataset CUỐI CÙNG (GOLD+SYLLABLE; SILVER_uncalibrated bị loại) -> $FINAL_DIR/ (tự chứa)"
  X "$PY" pipeline/export_final_dataset.py \
      --labels "$LABELS_FINAL" --src-root dataset_out --out "$FINAL_DIR"
  [[ -f "$FINAL_DIR/labels.csv" ]] || die "bước export không sinh $FINAL_DIR/labels.csv"
}

# ====================== FREEZE / EVIDENCE ====================================
evidence() {
  # nom-embed/best.pt: checkpoint S3 nằm trong SUBMODULE và đang có thay đổi CHƯA
  # COMMIT (mục 0.4) -> con trỏ submodule KHÔNG nhận diện được mô hình thật đã sinh ra
  # cột s3_cosine. Băm thẳng tệp là cách duy nhất hiện có để chỉ đúng mô hình đã dùng.
  # TỪ ĐIỂN PHẢI CÓ TRONG CHUỖI BẰNG CHỨNG (thêm 2026-08-24, khi làm T5).
  # Mọi nhãn GOLD đều do QuocNgu_SinoNom.csv quyết (s1_inter_s2_direct: ocr_char phải
  # là một âm đọc trong từ điển) và SinoNom_Similar.csv quyết luật bắc cầu — thế mà
  # hash của chúng chưa từng xuất hiện trong EVIDENCE_INDEX.md. Chúng CÓ trong git
  # (dưới `Dict/`, xem dict_dir() về chuyện hoa/thường), nên có lịch sử phiên bản;
  # cái thiếu là mối nối giữa MỘT LẦN CHẠY cụ thể và BẢN từ điển nó đã dùng. Băm ở
  # đây khép mối nối đó, cùng cơ chế với nom-embed/best.pt và index.csv.
  local files=("$LABELS_RAW" "$LABELS_REMED" "$LABELS_FINAL" "$FINAL_DIR/labels.csv" \
               "nom-embed/best.pt" "pipeline/align_engine/data/index.csv" \
               "dict/QuocNgu_SinoNom.csv" "dict/SinoNom_Similar.csv")
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
    printf -- '- sách    : %s | reseg=%s\n\n' "$BOOKS_LABEL" "$RESEG"
    printf '| file | sha256 |\n|---|---|\n'
    local f h
    for f in "${files[@]}"; do
      if [[ -f "$f" ]]; then h=$($sha_cmd "$f" | awk '{print $1}')
      else h='(chưa có)'; fi
      printf '| `%s` | `%s` |\n' "$f" "$h"
    done
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

if [[ -f dataset_out/.FROZEN ]]; then
  confirm_frozen_override
fi

log ""
log "${BLD}Sẽ chạy:${RST} setup -> extract($BOOKS_LABEL) -> build(cả 3 sách) -> remediate -> confusion -> gỡ S3 -> export"
log "  cache OCR : $([[ $FRESH_OCR == 1 ]] && echo 'XOÁ & OCR lại mới' || echo 'dùng cache cũ')"
log "  ${YEL}export sẽ XOÁ SẠCH $FINAL_DIR/ hiện có rồi ghi lại bản mới nhất${RST}"
read -r -p "Enter để bắt đầu, Ctrl-C để huỷ... " _

step_setup
step_extract
step_build
checkpoint build "$LABELS_RAW"
step_remediate
checkpoint remediate "$LABELS_REMED"
step_confusion
checkpoint confusion "$LABELS_FINAL"
step_s3unwind
checkpoint s3unwind "$LABELS_FINAL"
step_export
checkpoint export "$FINAL_DIR/labels.csv"
evidence

log ""
log "${BLD}================================================================${RST}"
log "${GRN}${BLD}  Xong — bộ dataset CUỐI CÙNG (tự chứa, đã ghi đè bản cũ):${RST}"
log "  $FINAL_DIR/labels.csv  (chỉ GOLD+SILVER+SYLLABLE, kèm ảnh crop copy hẳn)"
log ""
log "  Bản làm việc trung gian (đủ mọi tier kể cả REVIEW/QUARANTINE, không bị đụng):"
log "  $LABELS_REMED   (trước confusion-fix)"
log "  $LABELS_FINAL   (BẢN CÔNG BỐ — nguồn của $FINAL_DIR/ và của mẻ audit người)"
log "  dataset_out/{gold,silver,syllable}/"
log "  cảnh báo    : $N_WARN"
log ""
log "  ${YEL}Nhãn vừa đổi -> mẻ audit người dựng từ bản cũ đã hết hiệu lực.${RST}"
log "  Dựng lại: rm -rf dataset_out/ground_truth/audit_combined && \\"
log "            $PY -m pipeline.ground_truth.make_combined_batch --seed 2026"
log ""
log "  TẠM BỎ QUA (theo yêu cầu): audit người · fuse · publish của bản 8-bước cũ"
log "  Bản đầy đủ 8 bước còn trong lịch sử git: git log -- run_pipeline.sh"
log "${BLD}================================================================${RST}"
