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
# SÁCH MỚI (thạch bản/văn xuôi, 2026-09-22 — xem khối "SÁCH MỚI" dưới, đường STT trên KHÔNG đổi):
#   ./run_pipeline.sh --book LucVanTien1883 | KimVanKieu1884 | Chrestomathie1872 | all-new
#                            | LucVanTien1916 | TruyenKieu1872 | all-ihr   (tập ĐÁNH GIÁ, có nhãn người)
#       [--dry-run] [--skip-ingest] [--no-api] [--no-auto-precision] [--suffix _rp]
#   ./run_pipeline.sh --dry-run        # STT: chỉ in chuỗi lệnh 6 bước, không chạy
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

# ====================== SÁCH MỚI: --book <Book|all-new> ======================
# Đường tắt B0→B6 của docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md §2 cho thạch bản (LucVanTien1883,
# KimVanKieu1884 — chính thức B1') và văn xuôi (Chrestomathie1872). Toàn bộ mã đường mới nằm trong
# khối này + khối THAM SỐ ngay dưới; các hàm preflight/ask_*/step_*/checkpoint/tick/assert_qd01/evidence
# và khối MAIN của đường STT ở trên/dưới KHÔNG đổi. `./run_pipeline.sh` không tham số = STT như cũ.
#
#   ./run_pipeline.sh --book LucVanTien1883            # 1 sách mới, trọn B0→B6, kim cache -> 0 gọi API
#   ./run_pipeline.sh --book KimVanKieu1884            # config chính có run_config: -> pipeline_KimVanKieu1884_b1.yaml
#   ./run_pipeline.sh --book all-new                   # cả 3 sách mới (NEW_BOOKS_ALL)
#   cờ: --dry-run (chỉ in lệnh) · --skip-ingest (dùng prepared*/<Book> sẵn có) · --no-api (ingest --ocr none)
#       --no-auto-precision (bỏ auto_precision: không --cross cho B4'(d), không B6) · --suffix _rp (ra
#       dataset_out_<Book>_rp + dataset_<Book>_rp để so với bản chốt, không ghi đè)
#   ./run_pipeline.sh --dry-run                        # STT: in đúng chuỗi lệnh 6 bước cũ, không chạy gì
#
# Hồ sơ chạy mỗi sách đọc từ config/pipeline_<Book>.yaml (book_profile): khoá tuỳ chọn `run_config:` chuyển
# sang config chính thức (KVK -> _b1) và khối `run:` {ingest, ingest_args, verses_ref_fix{ref,ref_name,
# fuzzy_min}, dataset_out, n_columns, cross, measure_steps}; vắng khoá -> mặc định theo books[].layout.
# Chuỗi bước: 0 setup (step0 + bộ đo measure.py nếu thiếu measure_out/<Book> + B1' verses_ref_fix)
#   -> 1 ingest (ingest_lithograph_book | ingest_prose_book, --ocr kim, cache kim_raw/)
#   -> 2 build (build_dataset --use-s3 --two-pass --box-rule syl_index + enrich_crop_quality)
#   -> 3 remediate (census; apply --tau 0,62 --out dataset_out_<Book>; confusion_fix)
#   -> 4 gates (auto_precision cross,gates trên labels_final -> mechanism_gates --cross ... -> labels_gated.csv)
#   -> 5 export (export_final_dataset --n-columns; make_dataset_docs; make_xlsx)
#   -> 6 measure (auto_precision cross trên labels_gated = B6, chỉ sách có CROSS_BOOKS)
# Mỗi lệnh thật ghi vào logs/run_<Book>_<thời điểm>.log (kèm stdout/stderr); sha256 vào dataset_out_<Book>/CHECKSUMS.txt.
NEW_BOOKS_ALL="LucVanTien1883 KimVanKieu1884 Chrestomathie1872"
# 2026-09-23: 2 bộ IHR-NomDB (tập ĐÁNH GIÁ, có nhãn người) chạy riêng, KHÔNG gộp vào all-new
# để không lẫn vào bộ giao nộp: ./run_pipeline.sh --book LucVanTien1916|TruyenKieu1872
EVAL_BOOKS_IHR="LucVanTien1916 TruyenKieu1872"
NEW_BOOKS=""
DRY_RUN=0
SKIP_INGEST=0
NO_API=0
NO_AUTO_PRECISION=0
OUT_SUFFIX=""
BK_LOG=/dev/null
BK_NAME=""
N_BK_STEPS=6

usage() {
  cat <<'EOF'
run_pipeline.sh — GanNhanOCR
  (không tham số)                 đường STT 6 bước (hỏi sách/cache), ra dataset/
  --dry-run                       STT: chỉ in chuỗi lệnh, không chạy
  --book <Book> [--book <Book>…]  sách mới (config/pipeline_<Book>.yaml): LucVanTien1883 | KimVanKieu1884 | Chrestomathie1872
  --book all-new                  cả 3 sách mới (bộ giao nộp)
  --book all-ihr                  2 bộ IHR-NomDB có nhãn người (TẬP ĐÁNH GIÁ, không giao nộp)
    --dry-run                     chỉ in lệnh B0→B6
    --skip-ingest                 bỏ bước ingest (dùng prepared*/<Book> đã có)
    --no-api                      ingest --ocr none (không gọi kim; --verse-map content -> formula)
    --no-auto-precision           bỏ auto_precision (mechanism_gates không --cross; không B6)
    --suffix <s>                  hậu tố thư mục ra: prepared/<Book>/dataset_out<s>, dataset/<Book><s>
EOF
}

banner_bk() {   # banner_bk <số> <tên bước> <mô tả>
  log ""
  log "${BLD}================================================================${RST}"
  printf '%s>>> [%s] BƯỚC %s/%s · %s%s — %s\n' "$BLD" "$BK_NAME" "$1" "$N_BK_STEPS" "$2" "$RST" "$3"
  log "${BLD}================================================================${RST}"
}

R() {   # R <lệnh…>: in lệnh, ghi lệnh thật + toàn bộ output vào $BK_LOG, chạy (trừ --dry-run)
  local q
  q=$(printf '%q ' "$@")
  printf '    %s$%s %s\n' "$CYA" "$RST" "$*"
  if (( DRY_RUN )); then return 0; fi
  printf '\n%s  $ %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$q" >> "$BK_LOG"
  "$@" 2>&1 | tee -a "$BK_LOG"
}

need_file() {   # need_file <tệp> <tên bước>: bỏ qua khi --dry-run
  (( DRY_RUN )) && return 0
  [[ -f "$1" ]] || die "bước $2 không sinh $1"
}

bk_tick() {   # tick() của STT ghi thời gian vào $CHECKSUMS -> chỉ khi chạy thật
  (( DRY_RUN )) || tick "$1"
}

book_profile() {   # book_profile <Book> -> các dòng BK_*=<giá trị đã shell-quote> để eval
  "$PY" - "$1" <<'PYEOF'
import shlex, sys
from pathlib import Path
import yaml

book = sys.argv[1]


def emit(k, v):
    print(f"{k}={shlex.quote('' if v is None else str(v))}")


p = Path("config") / f"pipeline_{book}.yaml"
if not p.exists():
    emit("BK_ERR", f"không thấy {p} — sách mới cần config riêng (HUONG_DAN_CHAY_SACH_MOI §2 B0)"); sys.exit(0)
cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
if cfg.get("run_config"):                      # config chính trỏ sang config chính thức (KVK -> _b1)
    p = Path(str(cfg["run_config"]))
    if not p.exists():
        emit("BK_ERR", f"run_config trỏ tới {p} không tồn tại"); sys.exit(0)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
b = next((x for x in (cfg.get("books") or []) if str(x.get("name", "")).lower() == book.lower()), None)
if b is None:
    emit("BK_ERR", f"{p}: books[] không có {book}"); sys.exit(0)
layout = str(b.get("layout", "stt"))
if layout not in ("lithograph", "prose"):
    emit("BK_ERR", f"{p}: books[{book}].layout = {layout!r} — --book chỉ cho lithograph/prose (STT: chạy không tham số)")
    sys.exit(0)
run = cfg.get("run") or {}
paths = cfg.get("paths") or {}
try:
    sys.path.insert(0, str(Path("scripts") / "measure"))
    from auto_precision import CROSS_BOOKS      # sách có dị bản số hoá -> cross (0 API)
    in_cross = book in CROSS_BOOKS
except Exception:                                # noqa: BLE001
    in_cross = False
cross = bool(run.get("cross", in_cross)) and in_cross
rf = run.get("verses_ref_fix") or {}
emit("BK_CONFIG", p)
emit("BK_LAYOUT", layout)
emit("BK_INGEST", run.get("ingest") or layout)
emit("BK_INGEST_ARGS", run.get("ingest_args") or "")
emit("BK_DATA_DIR", paths.get("data_dir", "prepared"))
emit("BK_OUT_DIR", paths.get("output_dir") or f"dataset/{book}")
emit("BK_DS_OUT", run.get("dataset_out") or f"prepared/{book}/dataset_out")
emit("BK_NCOL", run.get("n_columns") or (10 if layout == "lithograph" else 7))
emit("BK_CROSS", "1" if cross else "0")
emit("BK_REF", rf.get("ref") or "")
emit("BK_REF_NAME", rf.get("ref_name") or "nf1871")
emit("BK_REF_FUZZY", rf.get("fuzzy_min") or "")
emit("BK_MEASURE_STEPS", run.get("measure_steps") or ("layout,qn_ocr" if layout == "lithograph" else "chresto_map"))
PYEOF
}

run_new_book() {   # run_new_book <Book>: B0→B6 cho một sách mới
  local book="$1" prof
  local BK_ERR="" BK_CONFIG="" BK_LAYOUT="" BK_INGEST="" BK_INGEST_ARGS="" BK_DATA_DIR="" BK_OUT_DIR=""
  local BK_DS_OUT="" BK_NCOL="" BK_CROSS="" BK_REF="" BK_REF_NAME="" BK_REF_FUZZY="" BK_MEASURE_STEPS=""
  prof=$(book_profile "$book") || die "book_profile($book) lỗi"
  eval "$prof"
  [[ -z "$BK_ERR" ]] || die "$BK_ERR"
  BK_NAME="$book"

  local ds_out="${BK_DS_OUT}${OUT_SUFFIX}" final_dir="${BK_OUT_DIR}${OUT_SUFFIX}"
  local labels_raw="$ds_out/labels.csv" labels_remed="$ds_out/labels_remediated.csv"
  local labels_final="$ds_out/labels_final.csv" labels_gated="$ds_out/labels_gated.csv"
  local trans_dir="$BK_DATA_DIR/$book/transcriptions"
  local ap_pre="$ds_out/auto_precision" ap_post="$ds_out/auto_precision_gated"
  local ocr="kim" verses_tsv=""
  local -a cmd
  (( NO_API )) && ocr="none"
  CHECKSUMS="$ds_out/CHECKSUMS.txt"      # checkpoint()/tick() của đường STT ghi vào đây

  if (( DRY_RUN )); then
    BK_LOG=/dev/null
  else
    mkdir -p logs "$ds_out"
    BK_LOG="logs/run_${book}_$(date +%Y%m%d_%H%M%S).log"
    {
      printf '# run_pipeline.sh --book %s  (%s)\n' "$book" "$(date +%Y-%m-%dT%H:%M:%S)"
      printf '# git HEAD %s · cwd %s · python %s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo '?')" "$REPO_ROOT" "$PY"
      printf '# cờ: skip_ingest=%s no_api=%s no_auto_precision=%s suffix=%q\n' "$SKIP_INGEST" "$NO_API" "$NO_AUTO_PRECISION" "$OUT_SUFFIX"
      printf '# hồ sơ: %s\n' "$(printf '%s' "$prof" | tr '\n' ' ')"
    } >> "$BK_LOG"
  fi

  log ""
  log "${BLD}[$book] config=$BK_CONFIG layout=$BK_LAYOUT ingest=$BK_INGEST data_dir=$BK_DATA_DIR${RST}"
  log "  dataset_out=$ds_out  export=$final_dir  n_columns=$BK_NCOL  cross=$BK_CROSS  log=$BK_LOG"
  [[ -n "$BK_INGEST_ARGS" ]] && log "  ingest_args: $BK_INGEST_ARGS"
  [[ -n "$BK_REF" ]] && log "  B1' verses_ref_fix: ref=$BK_REF ($BK_REF_NAME)"

  # ---- 0/6 setup ------------------------------------------------------------
  banner_bk 0 setup "step0_setup + bộ đo (nếu thiếu measure_out/$book) + B1' verses_ref_fix"
  R "$PY" -m pipeline.step0_setup "$BK_CONFIG"
  local need_measure=0
  if [[ "$BK_INGEST" == "prose" ]]; then
    [[ -f "measure_out/$book/chresto_map/bang_truyen_trang.csv" ]] || need_measure=1
  elif [[ "$BK_INGEST" == "ihr" ]]; then
    # IHR-NomDB: bố cục nằm sẵn trong data/<book>/pages/bboxes.json; bộ đo chỉ để KIỂM + lấy số
    [[ -f "measure_out/$book/ihr_layout/summary.json" ]] || need_measure=1
  else
    [[ -f "measure_out/$book/layout/layout_pages.csv" && -f "measure_out/$book/qn_ocr/verses.tsv" ]] || need_measure=1
  fi
  if (( need_measure )); then
    info "thiếu measure_out/$book -> chạy bộ đo (CPU, 0 token): --steps $BK_MEASURE_STEPS"
    R "$PY" scripts/measure/measure.py --book "$book" --steps "$BK_MEASURE_STEPS"
  else
    ok "bộ đo đã có: measure_out/$book"
  fi
  # 2026-09-23: tệp tham chiếu có thể đã bị gỡ khỏi data/ (vd bản 1871 trong thư mục PTCL) —
  # bỏ B1' kèm cảnh báo THẤY ĐƯỢC thay vì để verses_ref_fix chết vì không mở được tệp.
  if [[ -n "$BK_REF" && ! -f "$BK_REF" ]]; then
    log "  ⚠️  B1' BỎ QUA: không có tệp tham chiếu $BK_REF (đã gỡ khỏi data/). Bản dựng này KHÔNG có bước sửa QN theo dị bản."
    BK_REF=""
  fi
  if [[ -n "$BK_REF" && "$BK_INGEST" != "prose" ]]; then
    cmd=("$PY" scripts/measure/verses_ref_fix.py --book "$book" --ref "$BK_REF" --ref-name "$BK_REF_NAME")
    [[ -n "$BK_REF_FUZZY" ]] && cmd+=(--fuzzy-min "$BK_REF_FUZZY")
    R "${cmd[@]}"
    verses_tsv="measure_out/$book/qn_ref_fix/verses_b1.tsv"
    need_file "$verses_tsv" "verses_ref_fix"
  fi
  bk_tick "setup"

  # ---- 1/6 ingest -----------------------------------------------------------
  banner_bk 1 ingest "ảnh -> pages/detected/transcriptions (--ocr $ocr, cache kim_raw/) -> $BK_DATA_DIR/$book"
  if (( SKIP_INGEST )); then
    info "--skip-ingest: dùng $BK_DATA_DIR/$book có sẵn"
    (( DRY_RUN )) || [[ -d "$BK_DATA_DIR/$book/transcriptions" ]] || die "--skip-ingest nhưng thiếu $BK_DATA_DIR/$book/transcriptions"
  else
    local ingest_args="$BK_INGEST_ARGS"
    if [[ "$BK_INGEST" == "prose" ]]; then
      cmd=("$PY" -m pipeline.tools.ingest_prose_book --book "$book" --ocr "$ocr" --out "$BK_DATA_DIR"
           --kim-config "$BK_CONFIG")
    elif [[ "$BK_INGEST" == "ihr" ]]; then
      # IHR-NomDB (LucVanTien1916 / TruyenKieu1872): ô cột + QN lấy từ data/<book>/pages/*.json,
      # KHÔNG đọc nhãn chữ Nôm. Bộ ra là TẬP ĐÁNH GIÁ (manifest.evaluation_only = true).
      cmd=("$PY" -m pipeline.tools.ingest_ihr_book --book "$book" --ocr "$ocr" --out "$BK_DATA_DIR"
           --kim-config "$BK_CONFIG")
    else
      if (( NO_API )) && [[ " $ingest_args " == *" content "* ]]; then
        warn "--no-api: --verse-map content cần kim -> thay bằng formula (kết quả KHÁC bản chốt)"
        ingest_args=$(printf '%s' "$ingest_args" | sed 's/--verse-map content/--verse-map formula/')
      fi
      # --kim-config: adapter đọc books[].kim_lang_type/kim_ocr_id/kim_font_type của ĐÚNG config
      # đang chạy (KVK: pipeline_KimVanKieu1884_b1.yaml qua run_config:), không đoán theo tên sách.
      cmd=("$PY" -m pipeline.tools.ingest_lithograph_book --book "$book" --ocr "$ocr" --out "$BK_DATA_DIR"
           --kim-config "$BK_CONFIG")
      [[ -n "$verses_tsv" ]] && cmd+=(--verses "$verses_tsv")
    fi
    # shellcheck disable=SC2206  # ingest_args cố ý tách theo khoảng trắng (chuỗi cờ trong config run.ingest_args)
    [[ -n "$ingest_args" ]] && cmd+=($ingest_args)
    R "${cmd[@]}"
    need_file "$BK_DATA_DIR/$book/manifest.json" "ingest"
  fi
  bk_tick "ingest"

  # ---- 2/6 build ------------------------------------------------------------
  banner_bk 2 build "build_dataset (--use-s3 --two-pass --box-rule syl_index, det_* theo sách) + enrich_crop_quality -> $ds_out"
  R "$PY" -m pipeline.align_engine.build_dataset --config "$BK_CONFIG" --reseg detector \
      --qd01-cells none --decisions none --use-s3 --two-pass --box-rule syl_index --force --out "$ds_out"
  need_file "$labels_raw" "build"
  R "$PY" -m pipeline.tools.enrich_crop_quality --labels "$labels_raw" --src-root "$ds_out"
  (( DRY_RUN )) || checkpoint build "$labels_raw"
  bk_tick "build"

  # ---- 3/6 remediate --------------------------------------------------------
  banner_bk 3 remediate "census + apply --tau $TAU_REMEDIATE (--out $ds_out, KHÔNG đụng dataset_out/ STT) + confusion_fix -> $labels_final"
  R "$PY" -m pipeline.remediation --labels "$labels_raw" --out "$ds_out" census
  R "$PY" -m pipeline.remediation --labels "$labels_raw" --out "$ds_out" apply --tau "$TAU_REMEDIATE"
  need_file "$labels_remed" "remediate"
  [[ -f "$CONFUSION_FIXES" ]] || die "không thấy $CONFUSION_FIXES"
  R "$PY" -m pipeline.remediation.confusion_fix \
      --in "$labels_remed" --out "$labels_final" --fixes "$CONFUSION_FIXES" --measure
  need_file "$labels_final" "confusion_fix"
  (( DRY_RUN )) || { assert_qd01 "$labels_final" remediate; checkpoint remediate "$labels_final"; }
  bk_tick "remediate"

  # ---- 4/6 gates (B4') ------------------------------------------------------
  banner_bk 4 gates "mechanism_gates (a)(b)(c)[(d) --cross] -> $labels_gated"
  cmd=("$PY" -m pipeline.remediation.mechanism_gates --in "$labels_final" --out "$labels_gated"
       --config "$BK_CONFIG" --book "$book" --report "$ds_out/mechanism_gates_report.json")
  if [[ "$BK_CROSS" == "1" ]]; then
    if (( NO_AUTO_PRECISION )); then
      warn "--no-auto-precision: bỏ auto_precision cross -> mechanism_gates KHÔNG --cross (cổng (d) tắt, khác bản chốt)"
    else
      R "$PY" scripts/measure/auto_precision.py --steps cross,gates --books "$book" \
          --labels "$labels_final" --trans "$trans_dir" --out "$ap_pre"
      need_file "$ap_pre/cross/$book/cells.csv" "auto_precision cross"
      cmd+=(--cross "$ap_pre/cross/$book/cells.csv")
    fi
  else
    info "sách không có dị bản số hoá (CROSS_BOOKS) -> không --cross"
  fi
  R "${cmd[@]}"
  need_file "$labels_gated" "mechanism_gates"
  (( DRY_RUN )) || checkpoint gates "$labels_gated"
  bk_tick "gates"

  # ---- 5/6 export -----------------------------------------------------------
  banner_bk 5 export "export_final_dataset --n-columns $BK_NCOL -> $final_dir/ + README/DATASHEET + xlsx"
  R "$PY" pipeline/export_final_dataset.py \
      --labels "$labels_gated" --src-root "$ds_out" --out "$final_dir" --n-columns "$BK_NCOL"
  need_file "$final_dir/labels.csv" "export"
  R "$PY" -m pipeline.tools.make_dataset_docs --dataset "$final_dir" --n-columns "$BK_NCOL"
  R "$PY" -m pipeline.tools.make_xlsx --labels "$final_dir/labels.csv"
  # Bộ IHR-NomDB có NHÃN NGƯỜI -> đóng dấu TẬP ĐÁNH GIÁ vào thư mục export (không sửa labels.csv).
  # Chặn rủi ro R1 (rò rỉ tập đánh giá vào tập huấn luyện) — docs/CHAY_3_BO_CON_LAI_2026-09-23.md §5.
  if [[ "$BK_INGEST" == "ihr" ]]; then
    R "$PY" -m pipeline.tools.mark_eval_dataset --dataset "$final_dir" --book "$book" \
        --gt "data/$book/manifest.tsv"
  fi
  (( DRY_RUN )) || checkpoint export "$final_dir/labels.csv"
  bk_tick "export"

  # ---- 6/6 measure (B6) -----------------------------------------------------
  banner_bk 6 measure "auto_precision cross trên labels_gated (B6, 0 API) -> $ap_post"
  if [[ "$BK_CROSS" == "1" ]] && (( ! NO_AUTO_PRECISION )); then
    R "$PY" scripts/measure/auto_precision.py --steps cross --books "$book" \
        --labels "$labels_gated" --trans "$trans_dir" --out "$ap_post"
  else
    info "bỏ qua (không CROSS_BOOKS hoặc --no-auto-precision)"
  fi
  bk_tick "measure"

  log ""
  log "${GRN}${BLD}[$book] xong:${RST} $final_dir/labels.csv (+ ảnh crop GOLD/SYLLABLE, labels.xlsx, README, DATASHEET)"
  log "  trung gian: $ds_out/{labels,labels_remediated,labels_final,labels_gated}.csv · $CHECKSUMS · log $BK_LOG"
}

stt_dry_run() {   # STT --dry-run: in đúng chuỗi 6 bước cũ; X/die/assert_qd01 chỉ in trong subshell, không ghi gì
  BOOKS="SachThanhTruyen2 SachThanhTruyen4 SachThanhTruyen11"; BOOKS_LABEL="STT2+STT4+STT11"; FRESH_OCR=0
  log ""
  log "${BLD}[DRY-RUN STT] sách: $BOOKS_LABEL · cache OCR: dùng cache cũ · DS_OUT=$DS_OUT · không chạy gì${RST}"
  log "${BLD}Sẽ chạy 6 bước:${RST} setup -> extract($BOOKS_LABEL) -> build(100% tự động) -> remediate & confusion -> rescue (Self-Training) -> export"
  (
    X() { printf '    %s$%s %s\n' "$CYA" "$RST" "$*"; }
    die() { printf '    %s(dry-run: sẽ dừng nếu thiếu)%s %s\n' "$YEL" "$RST" "$*"; }
    assert_qd01() { printf '    %s(dry-run)%s assert_qd01 %s (%s)\n' "$CYA" "$RST" "$1" "$2"; }
    step_setup; step_extract; step_build; step_remediate; step_rescue; step_export
    log ""
    log "  (sau export: checkpoint/evidence sha256 -> $CHECKSUMS, $EVIDENCE)"
  )
}

# ============================== THAM SỐ ======================================
while (( $# )); do
  case "$1" in
    --book)       [[ $# -ge 2 ]] || die "--book cần tên sách"; NEW_BOOKS="$NEW_BOOKS $2"; shift 2 ;;
    --book=*)     NEW_BOOKS="$NEW_BOOKS ${1#--book=}"; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --skip-ingest) SKIP_INGEST=1; shift ;;
    --no-api)     NO_API=1; shift ;;
    --no-auto-precision) NO_AUTO_PRECISION=1; shift ;;
    --suffix)     [[ $# -ge 2 ]] || die "--suffix cần giá trị"; OUT_SUFFIX="$2"; shift 2 ;;
    --suffix=*)   OUT_SUFFIX="${1#--suffix=}"; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            usage >&2; die "tham số không hiểu: $1" ;;
  esac
done

if [[ -n "$NEW_BOOKS" ]]; then
  _books=""
  for _b in $NEW_BOOKS; do
    if [[ "$_b" == "all-new" ]]; then _books="$_books $NEW_BOOKS_ALL"
    elif [[ "$_b" == "all-ihr" ]]; then _books="$_books $EVAL_BOOKS_IHR"
    else _books="$_books $_b"; fi
  done
  log "${BLD}================================================================${RST}"
  log "${BLD}  GanNhanOCR — sách mới (thạch bản / văn xuôi) B0→B6:${_books}${RST}"
  log "${BLD}================================================================${RST}"
  (( DRY_RUN )) && info "--dry-run: chỉ in lệnh, không chạy, không ghi log"
  for _b in $_books; do
    [[ -f "config/pipeline_${_b}.yaml" ]] || die "không thấy config/pipeline_${_b}.yaml (sách: $_b)"
  done
  export PYTHONUNBUFFERED=1
  if (( DRY_RUN )); then SKIP_DEPS=1; fi
  preflight
  T_STEP=$SECONDS
  for _b in $_books; do
    run_new_book "$_b"
  done
  log ""
  log "${BLD}================================================================${RST}"
  log "${GRN}${BLD}  Hoàn tất sách mới:${_books} · tổng ${SECONDS}s${RST}"
  log "${BLD}================================================================${RST}"
  exit 0
fi

if (( DRY_RUN )); then
  stt_dry_run
  exit 0
fi

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
