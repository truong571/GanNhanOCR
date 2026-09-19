#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — GanNhanOCR · sinh bộ dataset tự động 100% (6 BƯỚC TINH GỌN)
#
#   ./run_pipeline.sh
#     -> hỏi 1) chọn sách   2) chạy cache cũ hay xoá-cache-chạy-mới
#     -> chạy: setup -> extract -> build -> remediate -> rescue -> export
#     -> ra:   dataset/labels.csv (+ ảnh crop copy hẳn) = BẢN CUỐI CÙNG, TỰ CHỨA
#              dataset/ bị XOÁ SẠCH và ghi lại mỗi lần chạy — luôn là bản MỚI NHẤT.
#
# 6 BƯỚC CỐT LÕI (Tự động hoàn toàn, tích hợp Self-Training In-domain Rescue):
#   1 setup       pipeline.step0_setup — kiểm cấu hình, từ điển, detector
#   2 extract     PDF -> khung -> OCR (cache) -> 9 cột/trang  (CHỈ sách đã chọn)
#   3 build       align_engine.build_dataset -> labels.csv + crops (100% tự động)
#   4 remediate   pipeline.remediation (census AE-1/F1) + confusion_fix
#   5 rescue      pipeline.remediation.self_training_rescue (MỚI: giải cứu REVIEW bằng mô hình nội bộ)
#   6 export      pipeline/export_final_dataset.py -> dataset/ (usable: GOLD+SYLLABLE)
#
# Viết cho bash 3.2 (bash mặc định của macOS).
# =============================================================================
set -euo pipefail

REPO_ROOT="${GANNHANOCR_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_ROOT"

export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

# ------------------------------- MẶC ĐỊNH -----------------------------------
PY="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-config/pipeline.yaml}"
RESEG="${RESEG:-detector}"
TAU_REMEDIATE="${TAU_REMEDIATE:-0.62}"
DS_OUT="${DS_OUT:-dataset_out}"
LABELS_RAW="$DS_OUT/labels.csv"
LABELS_REMED="$DS_OUT/labels_remediated.csv"
LABELS_FINAL="$DS_OUT/labels_final.csv"
CONFUSION_FIXES="${CONFUSION_FIXES:-config/confusion_fixes.yaml}"
FINAL_DIR="dataset"
REDATASET_DIR="re-dataset"
FINAL_OUT="${FINAL_DIR}"
# Kiến trúc 2 đầu ra (bước 7 tự chọn theo verdicts*.jsonl -> gọi pipeline.remediation.apply_verdicts khi có)
if [[ "$DS_OUT" != "dataset_out" ]]; then
  FINAL_DIR="$DS_OUT/dataset"
  REDATASET_DIR="$DS_OUT/re-dataset"
  FINAL_OUT="$FINAL_DIR"
fi
EVIDENCE="docs/EVIDENCE_INDEX.md"
CHECKSUMS="$DS_OUT/CHECKSUMS.txt"
NONINTERACTIVE="${NONINTERACTIVE:-0}"

BOOKS=""
BOOKS_LABEL=""
FRESH_OCR=0

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
  printf '%s>>> BƯỚC %s/6 · %s%s — %s\n' "$BLD" "$1" "$2" "$RST" "$3"
  log "${BLD}================================================================${RST}"
}

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
  log "${RED}${BLD}$DS_OUT/.FROZEN tồn tại${RST} — bản đã đóng băng."
  if [[ "$NONINTERACTIVE" == "1" ]]; then
    [[ "${FROZEN_OVERRIDE:-}" == "GHIDE" ]] \
      || die "NONINTERACTIVE=1 nhưng $DS_OUT/.FROZEN tồn tại — đặt FROZEN_OVERRIDE=GHIDE để ghi đè có chủ đích."
    warn "NONINTERACTIVE=1 + FROZEN_OVERRIDE=GHIDE -> tiếp tục, sẽ ghi đè $DS_OUT/"
    return 0
  fi
  read -r -p "Gõ đúng chữ GHIDE để tiếp tục (Enter/khác = huỷ chạy): " typed
  [[ "$typed" == "GHIDE" ]] || die "Đã huỷ — xoá $DS_OUT/.FROZEN nếu thật sự muốn dựng lại."
}

# ============================== PREFLIGHT ====================================
preflight() {
  log ""
  log "${BLD}--- PREFLIGHT ---------------------------------------------------${RST}"

  if [[ -x "$PY" ]]; then
    ok "venv: $("$PY" -c 'import sys;print(sys.executable, sys.version.split()[0])')"
  else
    die "Không thấy Python ở: $PY"
  fi

  if [[ "${SKIP_DEPS:-0}" != "1" ]]; then
    if "$PY" -m pipeline.tools.check_deps >/dev/null 2>&1; then
      ok "thư viện: đủ và đúng phiên bản ghim"
    else
      warn "thư viện THIẾU hoặc LỆCH phiên bản — đang vá tự động:"
      "$PY" -m pipeline.tools.check_deps --fix --python "$PY" 2>&1 | sed 's/^/    /'
    fi
  fi

  if [[ -f .env ]]; then
    set -a; source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env) 2>/dev/null || true; set +a
    ok ".env đã nạp"
  fi

  if [[ ! -f "$CONFIG" ]]; then
    die "không thấy $CONFIG — cấu hình pipeline bị thiếu."
  fi
  ok "config: $CONFIG"

  # Kiểm từ điển
  local d_qn d_sim
  d_qn=$(sed -n 's/^[[:space:]]*qn_to_nom_dict:[[:space:]]*//p' "$CONFIG" | sed -n 1p)
  d_sim=$(sed -n 's/^[[:space:]]*similar_dict:[[:space:]]*//p' "$CONFIG" | sed -n 1p)
  for d in "$d_qn" "$d_sim"; do
    if [[ -f "$d" ]]; then
      ok "từ điển: $d ($(( $(wc -l <"$d") - 1 )) dòng)"
    else
      die "KHÔNG thấy từ điển '$d' khai trong $CONFIG."
    fi
  done

  # Kiểm bộ dò ký tự
  local det_model="train_crop/detector_r34.best.pt"
  if [[ -f "$det_model" ]]; then
    ok "detector: $det_model"
  else
    warn "không thấy $det_model — kiểm tra đường dẫn checkpoint detector."
  fi

  # Kiểm tra mô hình / dữ liệu giải cứu Self-Training
  if [[ -f "lab/enhanced_self_training_v3/best_enhanced_nom_ocr.pt" ]]; then
    ok "self-training model: lab/enhanced_self_training_v3/best_enhanced_nom_ocr.pt (Enhanced SE-ResNet v3)"
  elif [[ -f "lab/self_training_v2/best_nom_ocr.pt" ]]; then
    ok "self-training model: lab/self_training_v2/best_nom_ocr.pt (NomOCRNet v2)"
  elif [[ -f "lab/self_training_v2/review_pseudo_labels.csv" ]]; then
    ok "self-training data: lab/self_training_v2/review_pseudo_labels.csv (Sẵn sàng giải cứu 2.172 ô)"
  else
    warn "chưa có mô hình/nhãn giải cứu Self-Training — bước giải cứu sẽ bỏ qua nếu chưa có."
  fi

  log "${BLD}--- HẾT PREFLIGHT: $N_WARN cảnh báo ------------------------------${RST}"
}

# ============================== CÁC BƯỚC =====================================
# ---- 1/6 setup --------------------------------------------------------------
step_setup() {
  banner 1 setup "kiểm cấu hình, đường dẫn, tài nguyên (pipeline.step0_setup)"
  X "$PY" -m pipeline.step0_setup "$CONFIG"
}

# ---- 2/6 extract ------------------------------------------------------------
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

# ---- 3/6 build --------------------------------------------------------------
step_build() {
  banner 3 build "align_engine.build_dataset: banded-DP align + consensus tier (100% tự động, --qd01-cells none)"
  X "$PY" -m pipeline.align_engine.build_dataset --config "$CONFIG" --reseg "$RESEG" \
      --qd01-cells none --force --out "$DS_OUT"
  [[ -f "$LABELS_RAW" ]] || die "bước build không sinh $LABELS_RAW"

  X "$PY" -m pipeline.tools.enrich_crop_quality --labels "$LABELS_RAW" --src-root "$DS_OUT"
}

# ---- CHỐT CHẶN VÂN TAY & THỜI GIAN ------------------------------------------
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

T_STEP=$SECONDS
tick() {
  local dt=$((SECONDS - T_STEP)); T_STEP=$SECONDS
  log "  ${CYA}⏱ $1: ${dt}s${RST}"
  mkdir -p "$(dirname "$CHECKSUMS")"
  printf '%s  thoi_gian  %s  %ss\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$1" "$dt" >> "$CHECKSUMS"
}

assert_qd01() {
  local f="$1" tag="$2"
  [[ -f "$f" ]] || die "assert_qd01($tag): không thấy $f"
  local n_qd
  n_qd=$("$PY" -c 'import csv, sys; print(sum(1 for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if (r.get("rule") or "").startswith("quyet_dinh_nguoi:")))' "$f")
  if [[ "$n_qd" -ne 0 ]]; then
    die "assert_qd01($tag): $f có $n_qd ô rule 'quyet_dinh_nguoi:*' — bài toán tự động đòi hỏi 0 ô can thiệp người!"
  fi
  ok "Tự động 100% ($tag): 0 ô quyet_dinh_nguoi (hoàn toàn do máy suy diễn)"
}

# ---- 4/6 remediate & confusion ----------------------------------------------
step_remediate() {
  banner 4 "remediate & confusion" "khử trùng lặp AE-1/F1 + sửa lỗi nhầm hệ thống -> $LABELS_FINAL"
  X "$PY" -m pipeline.remediation --labels "$LABELS_RAW" --out "$DS_OUT" census
  X "$PY" -m pipeline.remediation --labels "$LABELS_RAW" --out "$DS_OUT" apply --tau "$TAU_REMEDIATE"
  [[ -f "$LABELS_REMED" ]] || die "bước remediate không sinh $LABELS_REMED"

  [[ -f "$CONFUSION_FIXES" ]] || die "không thấy $CONFUSION_FIXES"
  X "$PY" -m pipeline.remediation.confusion_fix \
      --in "$LABELS_REMED" --out "$LABELS_FINAL" --fixes "$CONFUSION_FIXES" --measure
  [[ -f "$LABELS_FINAL" ]] || die "bước confusion không sinh $LABELS_FINAL"
}

# ---- 5/6 rescue (Self-Training In-domain) ------------------------------------
step_rescue() {
  banner 5 rescue "tự động giải cứu REVIEW bằng mô hình Self-Training -> $LABELS_FINAL"

  local _model=""
  if [[ -f "lab/enhanced_self_training_v3/best_enhanced_nom_ocr.pt" ]]; then
    _model="lab/enhanced_self_training_v3/best_enhanced_nom_ocr.pt"
  elif [[ -f "lab/self_training_v2/best_nom_ocr.pt" ]]; then
    _model="lab/self_training_v2/best_nom_ocr.pt"
  fi

  local _pseudo=()
  if [[ -f "lab/enhanced_self_training_v3/enhanced_pseudo_labels.csv" ]]; then
    _pseudo+=("lab/enhanced_self_training_v3/enhanced_pseudo_labels.csv")
  fi
  if [[ -f "lab/self_training_v2/review_pseudo_labels.csv" ]]; then
    _pseudo+=("lab/self_training_v2/review_pseudo_labels.csv")
  fi

  local _crops="KhoiB/v3/crops_v3.npz"
  [[ -f "$_crops" ]] || _crops="lab/gan_nhan_2026-09-13/crops.npz"

  local _args=(
    --in "$LABELS_FINAL"
    --out "$LABELS_FINAL"
    --crops "$_crops"
    --dict "Dict/QuocNgu_SinoNom.csv"
    --tau 0.70
    --report "$DS_OUT/self_training_rescue_report.json"
  )
  if [[ -n "$_model" ]]; then
    _args+=(--model "$_model")
  fi
  if [[ ${#_pseudo[@]} -gt 0 ]]; then
    _args+=(--pseudo-csv "${_pseudo[@]}")
  fi

  X "$PY" -m pipeline.remediation.self_training_rescue "${_args[@]}"
}

# ---- 6/6 export -------------------------------------------------------------
step_export() {
  banner 6 export "xuất bộ dữ liệu CUỐI CÙNG (100% tự động) -> $FINAL_DIR/ (tự chứa)"
  assert_qd01 "$LABELS_FINAL" "trước export"

  X "$PY" pipeline/export_final_dataset.py \
      --labels "$LABELS_FINAL" --src-root "$DS_OUT" --out "$FINAL_DIR"
  [[ -f "$FINAL_DIR/labels.csv" ]] || die "bước export không sinh $FINAL_DIR/labels.csv"
  FINAL_OUT="$FINAL_DIR"

  # Sinh tài liệu và bảng tính Excel
  X "$PY" -m pipeline.tools.make_dataset_docs --dataset "$FINAL_OUT"
  X "$PY" -m pipeline.tools.make_xlsx --labels "$FINAL_OUT/labels.csv"

  if [[ "$DS_OUT" == "dataset_out" ]]; then
    X "$PY" -m pipeline.tools.update_bang_so_lieu
  fi
}

# ====================== FREEZE / EVIDENCE ====================================
evidence() {
  local _out="${FINAL_OUT:-$FINAL_DIR}"
  local DICT_DIR="Dict"; [[ -d "$DICT_DIR" ]] || DICT_DIR="dict"
  local files=("$LABELS_RAW" "$LABELS_REMED" "$LABELS_FINAL" "$_out/labels.csv" \
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
    printf '\n## Lần chạy %s (100%% Tự động - Tích hợp Self-Training)\n\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
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
  ok "bảng sha256 đã ghi vào $EVIDENCE"
}

# =============================== MAIN ========================================
log "${BLD}================================================================${RST}"
log "${BLD}  GanNhanOCR — Pipeline Tự Động 100% (Tích Hợp Self-Training Rescue)${RST}"
log "${BLD}================================================================${RST}"

preflight
ask_book_choice
ask_cache_choice

if [[ -f "$DS_OUT/.FROZEN" ]]; then
  confirm_frozen_override
fi

log ""
log "${BLD}Sẽ chạy 6 bước:${RST} setup -> extract($BOOKS_LABEL) -> build(100% tự động) -> remediate & confusion -> rescue (Self-Training) -> export"
log "  cache OCR : $([[ $FRESH_OCR == 1 ]] && echo 'XOÁ & OCR lại mới' || echo 'dùng cache cũ')"
log "  ${YEL}export sẽ xuất bộ dữ liệu tự động hoàn toàn -> $FINAL_DIR/${RST}"
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
checkpoint remediate "$LABELS_FINAL"; tick remediate
assert_qd01 "$LABELS_FINAL" remediate
step_rescue
checkpoint rescue "$LABELS_FINAL"; tick rescue
assert_qd01 "$LABELS_FINAL" rescue
step_export
checkpoint export "${FINAL_OUT:-$FINAL_DIR}/labels.csv"; tick export

if [[ "$DS_OUT" == "dataset_out" ]]; then
  evidence
else
  info "bỏ qua evidence (DS_OUT thử nghiệm: $DS_OUT)"
fi

log ""
log "${BLD}================================================================${RST}"
log "${GRN}${BLD}  Hoàn tất Pipeline Tự Động 100% (Đã giải cứu REVIEW bằng Self-Training):${RST}"
log "  $FINAL_DIR/labels.csv  (kèm ảnh crop copy, labels.xlsx, README, DATASHEET)"
log "  Tổng thời gian: ${SECONDS}s"
log "${BLD}================================================================${RST}"
