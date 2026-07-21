#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — GanNhanOCR · 8 bước có TÊN, "1 lệnh ra kết quả cuối"
#
#   ./run_pipeline.sh                        # MẶC ĐỊNH: setup..confusion
#                                            #   -> dataset_out/labels_final.csv
#                                            #   (KHÔNG audit, KHÔNG publish)
#   ./run_pipeline.sh --only publish         # chỉ đóng gói dataset_out/release/
#   ./run_pipeline.sh --from remediate --strict
#   ./run_pipeline.sh --only audit --audit-tier SILVER --audit-n 400
#   ./run_pipeline.sh --only estimate --verdicts dataset_out/ground_truth/verdicts_001.jsonl
#   ./run_pipeline.sh --all --strict         # 8 bước, YÊU CẦU verdicts NGƯỜI đã có sẵn
#   ./run_pipeline.sh --check-only --strict  # chỉ chạy preflight
#   ./run_pipeline.sh --dry-run --all        # in lệnh, KHÔNG thực thi bước nào
#
# 8 BƯỚC:  setup extract build remediate audit fuse confusion publish
# Tương thích ngược: --step 0|1|2|3  ->  setup | extract | build | fuse
#
# Viết cho bash 3.2 (bash mặc định của macOS): không dùng mảng kết hợp,
# không bung mảng có thể rỗng khi đang set -u.
# =============================================================================
set -euo pipefail

# Bình thường: chạy tại thư mục chứa script (= repo root khi file nằm ở repo root).
# GANNHANOCR_ROOT chỉ dùng để KIỂM THỬ bản nháp từ nơi khác; đặt file vào repo
# root là hành vi giống hệt `cd "$(dirname "$0")"`. Xoá phần ${...:-} nếu không cần.
REPO_ROOT="${GANNHANOCR_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_ROOT"

# Giảm dao động số học của S3 / torch / tokenizers giữa các lần chạy.
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

# ------------------------------- MẶC ĐỊNH -----------------------------------
PY="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-config/pipeline.yaml}"
BOOK="${BOOK:-all}"
RESEG="${RESEG:-detector}"
LABELS_FINAL="${LABELS_FINAL:-dataset_out/labels_final.csv}"
LABELS_RAW="dataset_out/labels.csv"
LABELS_REMED="dataset_out/labels_remediated.csv"
GT_DIR="dataset_out/ground_truth"
RELEASE_DIR="dataset_out/release"
EVIDENCE="docs/EVIDENCE_INDEX.md"
FIXES_YAML="config/confusion_fixes.yaml"

STRICT=0; DRY=0; CHECK_ONLY=0; CROP_REVIEW=0; RAW_STANDARD=0; FRESH_OCR=0
ALL=0; AUDIT_OPTIN=0
FROM=""; UNTIL=""; ONLY=""
AUDIT_TIER="GOLD"; AUDIT_N=846; AUDIT_DESIGN="srs"; VERDICTS=""
# Manifest cho `--only estimate`: PHẢI khớp lô verdict đang chấm. GOLD -> grid ghi
# $GT_DIR/manifest.jsonl; SILVER/SYLLABLE -> make_audit_batch ghi audit_<TIER>/manifest.jsonl.
# Ghép verdict lô này với manifest lô khác => sai item_id. Override bằng --manifest.
MANIFEST="${MANIFEST:-$GT_DIR/manifest.jsonl}"
EXPORT_SAMPLE="${EXPORT_SAMPLE:-0}"     # 0 = parquet đầy đủ; >0 = smoke test
TAU_REMEDIATE="${TAU_REMEDIATE:-0.62}"
TAU_FUSE="${TAU_FUSE:-0.90}"; L2_FUSE="${L2_FUSE:-1.0}"

# THỨ TỰ CHUẨN 8 BƯỚC — dùng TÊN, không dùng số.
#   `--step 3` cũ = consensus-fusion, còn "Giai đoạn 3" của FLOW = remediation
#   -> giữ số sẽ nhập nhằng vĩnh viễn.
STEPS=(setup extract build remediate audit fuse confusion publish)
PSEUDO_STEPS=(preflight estimate)

RED=""; YEL=""; GRN=""; CYA=""; BLD=""; RST=""
if [[ -t 1 ]]; then
  RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; CYA=$'\033[36m'
  BLD=$'\033[1m'; RST=$'\033[0m'
fi

N_WARN=0; N_FAIL=0

log()  { printf '%s\n' "$*"; }
info() { printf '%s[i]%s %s\n' "$CYA" "$RST" "$*"; }
ok()   { printf '%s[OK]%s %s\n' "$GRN" "$RST" "$*"; }
warn() { N_WARN=$((N_WARN + 1)); printf '%s[CẢNH BÁO]%s %s\n' "$YEL" "$RST" "$*" >&2; }
die()  { printf '%s[LỖI]%s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }

# Mục preflight đỏ: --strict => lỗi cứng ở cuối preflight; không --strict => cảnh báo.
hard_or_warn() {
  if (( STRICT )); then
    N_FAIL=$((N_FAIL + 1)); printf '%s[ĐỎ]%s %s\n' "$RED" "$RST" "$*" >&2
  else
    warn "$*"
  fi
}

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }

banner() {   # banner <số> <tên bước> <mô tả>
  log ""
  log "${BLD}================================================================${RST}"
  printf '%s>>> BƯỚC %s/8 · %s%s — %s\n' "$BLD" "$1" "$2" "$RST" "$3"
  log "${BLD}================================================================${RST}"
}

# X — chạy (hoặc CHỈ IN nếu --dry-run) một lệnh có tham số rời.
X() {
  if (( DRY )); then
    printf '    %s$%s ' "$CYA" "$RST"; printf '%q ' "$@"; printf '\n'
    return 0
  fi
  printf '    %s$%s %s\n' "$CYA" "$RST" "$*"
  "$@"
}

# XSH — chạy (hoặc in) một lệnh shell tự do (glob/pipe/điều kiện).
XSH() {
  if (( DRY )); then printf '    %s$%s %s\n' "$CYA" "$RST" "$*"; return 0; fi
  printf '    %s$%s %s\n' "$CYA" "$RST" "$*"
  eval "$@"
}

usage() {
  cat <<'EOF'
run_pipeline.sh — GanNhanOCR, 8 bước, "1 lệnh ra kết quả cuối"

CÁCH DÙNG
  ./run_pipeline.sh [CỜ]

8 BƯỚC (theo TÊN, đúng thứ tự)
  1 setup       pipeline.step0_setup — kiểm cấu hình/đường dẫn
  2 extract     PDF -> khung -> kinhhannom/VietOCR -> 9 cột (cache OCR = PRIMARY DATA)
  3 build       align_engine.build_dataset -> dataset_out/labels.csv + crops
  4 remediate   pipeline.remediation -> labels_remediated.csv + remediation_report.json
  5 audit       ĐIỂM DỪNG CÓ NGƯỜI #1 — dựng grid HTML chấm mù rồi DỪNG (OPT-IN)
  6 fuse        consensus_fusion.fuse_stage (lớp phủ) -> dataset_out/fusion/
  7 confusion   ĐIỂM DỪNG CÓ NGƯỜI #2 — cần config/confusion_fixes.yaml -> labels_final.csv
  8 publish     pipeline.publish -> dataset_out/release/ (split+metadata+datasheet+export+validate)

MẶC ĐỊNH (không cờ) = setup, extract, build, remediate, fuse, confusion
  -> dataset_out/labels_final.csv
  Bước 5 audit BỎ QUA (cần người chấm; opt-in bằng --only/--from audit hoặc --all).
  Bước 8 publish BỎ QUA (cần chốt bản công bố).

CỜ CHỌN BƯỚC
  --from  BƯỚC        bắt đầu từ bước này (mặc định: setup)
  --until BƯỚC        dừng sau bước này  (mặc định: confusion; alias "labels")
  --only  BƯỚC[,...]  chỉ chạy các bước liệt kê (kèm 2 bước ảo: preflight, estimate)
  --all               chạy đủ 8 bước setup..publish (YÊU CẦU verdicts NGƯỜI đã có sẵn)
  --step  0|1|2|3     TƯƠNG THÍCH NGƯỢC: setup | extract | build | fuse

CỜ CHUNG
  --config PATH       cấu hình pipeline (mặc định config/pipeline.yaml)
  --book   NAME       chỉ xử lý 1 sách ở bước extract (mặc định: all)
  --reseg  MODE       chế độ tái phân đoạn cột cho build (mặc định: detector)
  --labels PATH       đường dẫn labels_final.csv (mặc định dataset_out/labels_final.csv)
  --strict            mọi mục preflight đỏ => exit 1; truyền --strict xuống build_dataset;
                      fuse/confusion không được phép SKIP âm thầm
  --crop-review       build thêm crop tier REVIEW
  --raw-standard      (TUỲ CHỌN) chạy to_standard vào dataset_out/_raw_standard/
                      — MẶC ĐỊNH TẮT: to_standard ghi đè metadata/croissant/datapackage
                      theo labels.csv THÔ, mâu thuẫn vĩnh viễn với release/
  --dry-run           chỉ IN lệnh của các bước, KHÔNG thực thi bước nào
                      (preflight VẪN chạy vì chỉ đọc, nhưng KHÔNG bao giờ die)
  --check-only        chỉ chạy preflight rồi thoát
  --help              bản trợ giúp này

CỜ AUDIT (bước 5) / ESTIMATE
  --audit-tier TIER   GOLD | SILVER | SYLLABLE (mặc định GOLD)
  --audit-n N         cỡ mẫu (mặc định 846 — kế hoạch chấp nhận p0=0.97)
  --audit-design D    srs | stratified (mặc định srs)
  --verdicts FILE     file verdicts_*.jsonl cho `--only estimate`
  --manifest FILE     manifest.jsonl KHỚP lô verdict cho `--only estimate`
                      (mặc định dataset_out/ground_truth/manifest.jsonl = lô GOLD;
                      SILVER/SYLLABLE phải trỏ audit_<TIER>/manifest.jsonl đúng lô)
  --export-sample N   bước publish chỉ export N crop (smoke test; 0 = đầy đủ)

CỜ NGUY HIỂM
  --fresh-ocr-i-know-this-costs-api
                      XOÁ cache OCR trong prepared/*/detected/*_ocr_cache.json.
                      Cache OCR = PRIMARY DATA: còn cache = tái lập được; xoá cache
                      = gọi lại API ngoài = TỐN TIỀN + MẤT tính tái lập.

BIẾN MÔI TRƯỜNG
  PYTHON_BIN, CONFIG, BOOK, RESEG, LABELS_FINAL, EXPORT_SAMPLE,
  TAU_REMEDIATE, TAU_FUSE, L2_FUSE

VÍ DỤ
  ./run_pipeline.sh --strict
  ./run_pipeline.sh --only audit --audit-tier SILVER --audit-n 400
  ./run_pipeline.sh --only estimate --verdicts dataset_out/ground_truth/verdicts_001.jsonl
  ./run_pipeline.sh --from fuse --strict
  ./run_pipeline.sh --only publish --labels dataset_out/labels_final.csv
EOF
}

# ------------------------------- ARGS ---------------------------------------
valid_step() {
  local s
  for s in "${STEPS[@]}" "${PSEUDO_STEPS[@]}" labels; do
    [[ "$1" == "$s" ]] && return 0
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)        CONFIG="${2:?--config cần giá trị}"; shift 2 ;;
    --book)          BOOK="${2:?--book cần giá trị}"; shift 2 ;;
    --reseg)         RESEG="${2:?--reseg cần giá trị}"; shift 2 ;;
    --labels)        LABELS_FINAL="${2:?--labels cần giá trị}"; shift 2 ;;
    --from)          FROM="${2:?--from cần giá trị}"; shift 2 ;;
    --until)         UNTIL="${2:?--until cần giá trị}"; shift 2 ;;
    --only)          ONLY="${2:?--only cần giá trị}"; shift 2 ;;
    --all)           ALL=1; FROM="setup"; UNTIL="publish"; AUDIT_OPTIN=1; shift ;;
    --strict)        STRICT=1; shift ;;
    --crop-review)   CROP_REVIEW=1; shift ;;
    --raw-standard)  RAW_STANDARD=1; shift ;;
    --dry-run)       DRY=1; shift ;;
    --check-only)    CHECK_ONLY=1; shift ;;
    --audit-tier)    AUDIT_TIER="${2:?--audit-tier cần giá trị}"; shift 2 ;;
    --audit-n)       AUDIT_N="${2:?--audit-n cần giá trị}"; shift 2 ;;
    --audit-design)  AUDIT_DESIGN="${2:?--audit-design cần giá trị}"; shift 2 ;;
    --verdicts)      VERDICTS="${2:?--verdicts cần giá trị}"; shift 2 ;;
    --manifest)      MANIFEST="${2:?--manifest cần giá trị}"; shift 2 ;;
    --export-sample) EXPORT_SAMPLE="${2:?--export-sample cần giá trị}"; shift 2 ;;
    --fresh-ocr-i-know-this-costs-api) FRESH_OCR=1; shift ;;
    --fresh)
      die "--fresh đã bị ĐỔI TÊN. Cờ này xoá cache OCR = gọi lại API ngoài
      = TỐN TIỀN + MẤT tính tái lập. Nếu thật sự muốn:
        ./run_pipeline.sh --fresh-ocr-i-know-this-costs-api" ;;
    --step)
      case "${2:?--step cần giá trị}" in
        0)   ONLY="setup"   ;;
        1)   ONLY="extract" ;;
        2)   ONLY="build"   ;;
        3)   ONLY="fuse"    ;;
        all) FROM="setup"; UNTIL="confusion" ;;
        *)   die "--step chỉ nhận 0|1|2|3|all (alias cũ). Dùng --only <tên bước> cho bản mới." ;;
      esac
      warn "--step là alias TƯƠNG THÍCH NGƯỢC; bản mới dùng --only/--from/--until theo TÊN."
      shift 2 ;;
    --help|-h)       usage; exit 0 ;;
    *) log "Tuỳ chọn không hiểu: $1"; log ""; usage; exit 1 ;;
  esac
done

is_uint "$AUDIT_N"       || die "--audit-n phải là số nguyên không âm (nhận: '$AUDIT_N')"
is_uint "$EXPORT_SAMPLE" || die "--export-sample phải là số nguyên không âm (nhận: '$EXPORT_SAMPLE')"
case "$AUDIT_DESIGN" in srs|stratified) ;; *) die "--audit-design chỉ nhận srs|stratified" ;; esac
case "$AUDIT_TIER" in GOLD|SILVER|SYLLABLE) ;; *) die "--audit-tier chỉ nhận GOLD|SILVER|SYLLABLE" ;; esac

# --------------------- CHUẨN HOÁ PHẠM VI BƯỚC -------------------------------
[[ -z "$UNTIL" ]] && UNTIL="labels"          # MẶC ĐỊNH dừng TRƯỚC publish
[[ "$UNTIL" == "labels" ]] && UNTIL="confusion"
[[ -z "$FROM"  ]] && FROM="setup"

idx() {
  local i
  for i in "${!STEPS[@]}"; do
    [[ "${STEPS[$i]}" == "$1" ]] && { printf '%s' "$i"; return 0; }
  done
  return 1
}

if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a _only_arr <<<"$ONLY"
  for s in "${_only_arr[@]}"; do
    valid_step "$s" || die "--only: bước không hợp lệ '$s' (hợp lệ: ${STEPS[*]} ${PSEUDO_STEPS[*]})"
  done
fi
valid_step "$FROM"  || die "--from: bước không hợp lệ '$FROM'"
valid_step "$UNTIL" || die "--until: bước không hợp lệ '$UNTIL'"
FROM_I=$(idx "$FROM")   || die "--from '$FROM' không nằm trong 8 bước"
UNTIL_I=$(idx "$UNTIL") || die "--until '$UNTIL' không nằm trong 8 bước"
(( FROM_I <= UNTIL_I )) || die "--from ($FROM) đứng SAU --until ($UNTIL)"

# audit là bước CÓ NGƯỜI: nó nằm GIỮA khoảng mặc định (setup..confusion) nhưng
# KHÔNG được chạy khi chạy trơn — nếu chạy, pipeline sẽ dừng ở audit và
# labels_final.csv KHÔNG BAO GIỜ ra đời. Chỉ bật khi được gọi ĐÍCH DANH.
case ",$ONLY," in (*",audit,"*) AUDIT_OPTIN=1 ;; esac
[[ "$FROM" == "audit" ]] && AUDIT_OPTIN=1

want_step() {
  local name="$1" i
  if [[ "$name" == "audit" ]] && (( ! AUDIT_OPTIN )); then return 1; fi
  if [[ -n "$ONLY" ]]; then
    case ",$ONLY," in (*",$name,"*) return 0 ;; (*) return 1 ;; esac
  fi
  i=$(idx "$name") || return 1
  (( i >= FROM_I && i <= UNTIL_I ))
}

PLAN=()
for s in "${STEPS[@]}"; do want_step "$s" && PLAN+=("$s"); done
want_step estimate && PLAN+=("estimate")

log "${BLD}================================================================${RST}"
log "${BLD}  GanNhanOCR — pipeline 8 bước${RST}"
log "  Config       : $CONFIG"
log "  Python       : $PY"
log "  Sách         : $BOOK        reseg=$RESEG"
log "  Nhãn cuối    : $LABELS_FINAL"
if [[ -n "$ONLY" ]]; then
  log "  Phạm vi      : --only $ONLY"
else
  log "  Phạm vi      : $FROM .. $UNTIL"
fi
log "  Sẽ chạy      : ${PLAN[*]:-(không có bước nào)}"
if ! want_step audit; then
  log "  audit        : BỎ QUA (opt-in: --only audit | --from audit | --all)"
fi
log "  strict=$STRICT  dry-run=$DRY  check-only=$CHECK_ONLY  crop-review=$CROP_REVIEW"
log "${BLD}================================================================${RST}"

if (( FRESH_OCR )); then
  log ""
  log "${RED}${BLD}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${RST}"
  log "${RED}${BLD}  --fresh-ocr-i-know-this-costs-api${RST}"
  log "${RED}  Bạn sắp XOÁ cache OCR (prepared/*/detected/*_ocr_cache.json).${RST}"
  log "${RED}  Cache OCR là PRIMARY DATA của luận văn:${RST}"
  log "${RED}    · còn cache  -> bước extract TÁI LẬP ĐƯỢC, 0 đồng${RST}"
  log "${RED}    · xoá cache  -> gọi lại API ngoài: TỐN TIỀN + KHÔNG tái lập${RST}"
  log "${RED}  Ctrl-C NGAY nếu không chắc chắn. Chờ 3 giây...${RST}"
  log "${RED}${BLD}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${RST}"
  (( DRY )) || sleep 3
fi

# ------------------- VERDICTS: đếm ĐÚNG như code Python -----------------------
# ĐÍNH CHÍNH quan trọng (đã đối chiếu mã nguồn, đừng chép lại lời đồn cũ):
#   · fuse_stage.py:62, mine_confusions.py:54, score_s3.py:93 đều dùng
#       glob.glob(gt/"**"/verdicts_*.jsonl, recursive=True)
#     => glob ĐỆ QUY: file trong dataset_out/ground_truth/audit_SILVER/ VẪN ĐƯỢC THẤY.
#     (Không cần copy lên cấp phẳng — lời khuyên "glob không đệ quy" là SAI.)
#   · Cửa chặn THẬT là NGUỒN verdict: fuse_stage.py:43 AI_VERDICT_SOURCES={"ai_vision"};
#     fuse_stage.py:69 loại mọi bản ghi có source="ai_vision" trừ khi --include-ai-verdicts.
#     => verdicts_ai.jsonl KHÔNG phải ground truth, KHÔNG mở khoá được bước fuse.
N_VERDICT_HUMAN=-1     # -1 = chưa đếm
count_human_verdicts() {
  local n=""
  if [[ -x "$PY" && -d "$GT_DIR" ]]; then
    n=$("$PY" - "$GT_DIR" <<'PYEOF' 2>/dev/null || true
import glob, json, sys
AI = {"ai_vision"}          # = fuse_stage.AI_VERDICT_SOURCES
n = 0
for f in glob.glob(f"{sys.argv[1]}/**/verdicts_*.jsonl", recursive=True):
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if str(rec.get("source") or "human").strip().lower() not in AI:
                n += 1
print(n)
PYEOF
)
  fi
  is_uint "${n:-}" || n=0
  N_VERDICT_HUMAN="$n"
}
has_human_verdicts() {
  (( N_VERDICT_HUMAN < 0 )) && count_human_verdicts
  (( N_VERDICT_HUMAN > 0 ))
}

# ====================== PREFLIGHT (fail-loud) ================================
# Biến các chế độ HỎNG-ÂM-THẦM thành lỗi ở phút 0 thay vì phút 90.
#   --strict : mọi mục đỏ => exit 1  (trừ khi --dry-run: preflight chỉ để xem)
# Preflight CHỈ ĐỌC nên vẫn chạy dưới --dry-run — nhưng KHÔNG BAO GIỜ die ở đó.
preflight() {
  log ""
  log "${BLD}--- PREFLIGHT ---------------------------------------------------${RST}"

  # (a) venv -----------------------------------------------------------------
  if [[ -x "$PY" ]]; then
    ok "venv: $("$PY" -c 'import sys;print(sys.executable, sys.version.split()[0])')"
  elif (( DRY )); then
    hard_or_warn "không thấy Python ở: $PY (dry-run nên chỉ báo, không dừng)"
  else
    die "Không thấy Python ở: $PY
      Tạo venv : python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
      Hoặc     : PYTHON_BIN=/path/to/python ./run_pipeline.sh ..."
  fi

  # (b) .env — chỉ source biến có TÊN HỢP LỆ bash (key có dấu '-' sẽ vỡ + lộ key)
  if [[ -f .env ]]; then
    set -a; source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env) 2>/dev/null || true; set +a
    ok ".env đã nạp (chỉ các key hợp lệ bash)"
  else
    warn "không có .env — SN_OCR_TOKEN/GEMINI_API_KEY sẽ thiếu nếu bước extract cần gọi API"
  fi

  # (c) 11 module ------------------------------------------------------------
  # Sanity-check CŨ chỉ kiểm 7 module -> publish/export.py import `datasets`
  # và chết ở phút 90. Kiểm đủ 11 ngay từ phút 0.
  if [[ -x "$PY" ]]; then
    local miss
    miss=$("$PY" - <<'PYEOF' 2>/dev/null || true
mods = "fitz cv2 numpy pandas yaml PIL vietocr scipy torch datasets pyarrow".split()
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
      ok "11/11 module có mặt (fitz cv2 numpy pandas yaml PIL vietocr scipy torch datasets pyarrow)"
    else
      hard_or_warn "thiếu module trong venv: $miss
      -> $PY -m pip install -r requirements.txt
      (datasets/pyarrow là bắt buộc cho bước publish; thiếu = chết ở phút 90)"
    fi
  fi

  # (d) checkpoint S3 --------------------------------------------------------
  if [[ -f nom-embed/best.pt ]]; then
    ok "checkpoint S3: nom-embed/best.pt"
  elif [[ -f nom-embed/last.pt ]]; then
    warn "chỉ có nom-embed/last.pt (không có best.pt) — S3 chạy trên checkpoint KHÔNG tốt nhất."
  else
    hard_or_warn "thiếu nom-embed/best.pt — S3 tắt => tier SILVER SẬP ÂM THẦM về REVIEW.
      Train ở pipeline/align_engine/nom_classifier rồi đặt best.pt vào nom-embed/."
  fi

  # (e) config phải tồn tại trước khi đọc ------------------------------------
  if [[ ! -f "$CONFIG" ]]; then
    hard_or_warn "không thấy $CONFIG — mọi bước đọc cấu hình sẽ hỏng."
  fi

  # (f) kho glyph FontDiffusion ---------------------------------------------
  local fd_dir fd_first
  fd_dir=$(sed -n 's/^[[:space:]]*fd_cache_universal:[[:space:]]*//p' "$CONFIG" 2>/dev/null | sed -n 1p)
  fd_dir="${fd_dir:-gannhanocr-fd}"
  fd_first=$(find "$fd_dir" -name 'U+*.png' -print -quit 2>/dev/null || true)
  if [[ -n "$fd_first" ]]; then
    ok "kho glyph FD: $fd_dir (ví dụ $(basename "$fd_first"))"
  else
    hard_or_warn "kho glyph FD '$fd_dir' TRỐNG (không có U+*.png)
      -> S3 mất glyph tham chiếu cho mọi ứng viên => tier SILVER hỏng.
      Đặt data đã sinh vào $fd_dir/ theo dạng <hex>/U+XXXX.png"
  fi

  # (g) CROP-PROTO — BẪY QUAN TRỌNG NHẤT ------------------------------------
  # index.csv trỏ dataset_out/gold/*.png. Repo sạch (đã xoá dataset_out/gold/)
  # -> crop-protos = 0 -> luật `s2_inter_s3_corrected` BIẾN MẤT -> SILVER −32%
  # mà KHÔNG ném lỗi nào. Phải kiểm CẢ file index lẫn file crop đầu tiên.
  local idx_csv first_crop
  idx_csv="pipeline/align_engine/data/index.csv"
  if [[ ! -f "$idx_csv" ]]; then
    hard_or_warn "thiếu $idx_csv -> crop-protos = 0 -> luật s2_inter_s3_corrected biến mất
      -> SILVER tụt ~32%, KHÔNG có lỗi nào được ném ra."
  else
    first_crop=$(awk -F, 'NR==2{print $1; exit}' "$idx_csv" 2>/dev/null || true)
    if [[ -z "$first_crop" ]]; then
      hard_or_warn "$idx_csv RỖNG (không có dòng dữ liệu) -> crop-protos = 0 -> SILVER tụt ~32%."
    elif [[ -f "$first_crop" ]]; then
      ok "crop-proto: $idx_csv -> $first_crop (có thật, $(wc -l <"$idx_csv" | tr -d ' ') dòng)"
    else
      hard_or_warn "crop-proto TRỎ HỤT: $idx_csv dòng 2 = '$first_crop' KHÔNG có trên đĩa
      -> crop-protos = 0 -> luật s2_inter_s3_corrected biến mất -> SILVER tụt ~32% ÂM THẦM.
      Cách sửa: chạy lại bước build MỘT lần (PASS2 dựng lại crops), rồi chạy tiếp."
    fi
  fi

  # (h) config trỏ đúng PDF có thật -----------------------------------------
  if [[ -f "$CONFIG" ]]; then
    local pdf n_bad_pdf=0
    while read -r pdf; do
      [[ -z "$pdf" ]] && continue
      if [[ ! -f "$pdf" ]]; then
        hard_or_warn "$CONFIG trỏ PDF KHÔNG tồn tại: $pdf
      (file trên đĩa đã đổi tên STT{2,4,11}.pdf — phải vá books[].pdf trong $CONFIG)"
        n_bad_pdf=$((n_bad_pdf + 1))
      fi
    done < <(sed -n 's/^[[:space:]]*pdf:[[:space:]]*//p' "$CONFIG")
    (( n_bad_pdf == 0 )) && ok "mọi PDF khai trong $CONFIG đều có thật trên đĩa"
  fi

  # (i) transcriptions -------------------------------------------------------
  local n_tr
  n_tr=$(find prepared -path '*/transcriptions/page_*.json' 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${n_tr:-0}" -gt 0 ]]; then
    ok "prepared/*/transcriptions/page_*.json : $n_tr file"
  else
    hard_or_warn "KHÔNG có prepared/*/transcriptions/page_*.json
      -> bước build sẽ sinh labels.csv RỖNG mà KHÔNG raise. Chạy bước extract trước."
  fi

  # (j) confusion_fixes.yaml -------------------------------------------------
  if [[ -f "$FIXES_YAML" ]]; then
    ok "$FIXES_YAML có mặt (bảng sửa nhầm lẫn do NGƯỜI soạn)"
  else
    hard_or_warn "thiếu $FIXES_YAML
      -> bước confusion thành no-op: labels_final.csv == labels_remediated.csv ÂM THẦM."
  fi

  # (k) verdicts NGƯỜI (không phải verdicts_ai) ------------------------------
  count_human_verdicts
  local n_files
  n_files=$(find "$GT_DIR" -name 'verdicts_*.jsonl' 2>/dev/null | wc -l | tr -d ' ')
  if (( N_VERDICT_HUMAN > 0 )); then
    ok "verdicts NGƯỜI: $N_VERDICT_HUMAN bản ghi trong $n_files file (quét ĐỆ QUY, đúng như fuse_stage)"
  else
    warn "0 verdict NGƯỜI trong $GT_DIR (thấy $n_files file verdicts_*.jsonl) -> bước fuse sẽ SKIP.
      ĐÍNH CHÍNH: glob của fuse_stage/mine_confusions/score_s3 LÀ ĐỆ QUY
      (glob.glob(gt/'**'/verdicts_*.jsonl, recursive=True)) — file trong audit_SILVER/
      VẪN được thấy, KHÔNG cần copy lên cấp phẳng.
      Thứ bị loại là NGUỒN: bản ghi source='ai_vision' (verdicts_ai.jsonl) bị bỏ theo
      fuse_stage.py:43,69 — verdict MÁY không được dùng làm ground truth."
  fi

  # (l) dataset_out/.FROZEN --------------------------------------------------
  # Quy ước của repo này: chạm tay tạo file rỗng dataset_out/.FROZEN sau khi đã
  # trích số vào luận văn. Không có file đó thì mục này im lặng.
  if [[ -f dataset_out/.FROZEN ]]; then
    if want_step build; then
      hard_or_warn "dataset_out/.FROZEN tồn tại (bản đã đóng băng để trích luận văn)
      nhưng phạm vi chạy có bước 'build' -> sẽ GHI ĐÈ bằng chứng.
      Xoá .FROZEN nếu thật sự muốn dựng lại."
    else
      info "dataset_out/.FROZEN có mặt — bản đóng băng; phạm vi hiện tại không đụng bước build."
    fi
  fi

  log "${BLD}--- HẾT PREFLIGHT: $N_WARN cảnh báo, $N_FAIL mục đỏ ------------------${RST}"
  if (( STRICT && N_FAIL > 0 )); then
    if (( DRY )); then
      warn "--dry-run: bỏ qua exit 1 của --strict ($N_FAIL mục đỏ) để in tiếp kế hoạch."
    else
      die "--strict: $N_FAIL mục preflight ĐỎ. Sửa xong rồi chạy lại."
    fi
  fi
  return 0
}

# ------------------ danh sách sách cần xử lý --------------------------------
resolve_books() {
  if [[ "$BOOK" != "all" ]]; then printf '%s\n' "$BOOK"; return 0; fi
  if (( DRY )) || [[ ! -x "$PY" ]]; then
    # dry-run: KHÔNG gọi python. Đọc thô tên sách bằng awk.
    awk '/^books:/{f=1;next} f&&/^[a-z_]+:/{f=0} f&&/- name:/{print $3}' "$CONFIG" 2>/dev/null
    return 0
  fi
  "$PY" -c "
import yaml
for b in yaml.safe_load(open('$CONFIG'))['books']:
    print(b['name'])
"
}

# ============================== CÁC BƯỚC =====================================
# CÚ PHÁP ARGPARSE đã đối chiếu TRỰC TIẾP mã nguồn (dòng ghi kèm). Quy tắc chung:
# optional khai ở parser CHA phải đứng TRƯỚC subcommand, nếu không argparse báo lỗi.

# ---- 1/8 setup --------------------------------------------------------------
step_setup() {
  banner 1 setup "kiểm cấu hình, đường dẫn, tài nguyên (pipeline.step0_setup)"
  X "$PY" -m pipeline.step0_setup "$CONFIG"
}

# ---- 2/8 extract ------------------------------------------------------------
# Cache OCR trong prepared/*/detected/*_ocr_cache.json = PRIMARY DATA.
# Còn cache = tái lập được. Xoá cache = gọi API ngoài = KHÔNG tái lập + tốn tiền.
step_extract() {
  banner 2 extract "PDF -> khung -> OCR (cache) -> 9 cột/trang | sách: $(echo $BOOKS | tr '\n' ' ')"
  local b
  if (( FRESH_OCR )); then
    for b in $BOOKS; do
      XSH "find prepared/$b/detected -name '*_ocr_cache.json' -delete 2>/dev/null || true"
    done
    warn "cache OCR đã bị xoá cho: $(echo $BOOKS | tr '\n' ' ') — lần extract này SẼ GỌI API NGOÀI."
  fi
  for b in $BOOKS; do
    X "$PY" -m pipeline.step1_extract "$CONFIG" "$b"
  done
  log ""
  info "kiểm số cột OCR sau extract (mong đợi ĐÚNG 9 cột/trang)"
  for b in $BOOKS; do
    if (( DRY )); then
      X "$PY" pipeline/check_ocr_columns.py --book "$b"
    else
      "$PY" pipeline/check_ocr_columns.py --book "$b" 2>/dev/null \
        | grep -E "OK\(=9\)|THIẾU|DƯ|≠9" || true
    fi
  done
}

# ---- 3/8 build --------------------------------------------------------------
# -> dataset_out/labels.csv + crops gold/silver/syllable (+ review nếu --crop-review)
# build_dataset.py:130-142 — mọi cờ dưới đây là optional của MỘT parser phẳng.
step_build() {
  banner 3 build "align_engine.build_dataset: banded-DP align + consensus tier + crops"
  BUILD_ARGS=(--config "$CONFIG" --use-s3 --reseg "$RESEG")
  # README khuyên dùng --strict nhưng bản cũ QUÊN truyền xuống -> truyền lại ở đây.
  if (( STRICT )); then BUILD_ARGS+=(--strict); fi
  if (( CROP_REVIEW )); then BUILD_ARGS+=(--crop-review); fi
  X "$PY" -m pipeline.align_engine.build_dataset "${BUILD_ARGS[@]}"
  if (( ! DRY )) && [[ ! -f "$LABELS_RAW" ]]; then die "bước build không sinh $LABELS_RAW"; fi

  # to_standard ĐÃ BỊ BỎ khỏi đường chạy mặc định: nó ghi đè dataset_out/{metadata.csv,
  # croissant.json,datapackage.json} theo labels.csv THÔ -> mâu thuẫn VĨNH VIỄN với
  # dataset_out/release/. Bộ chuẩn quốc tế DUY NHẤT trích trong luận văn = release/.
  # to_standard.py:151-155 chỉ có --dataset (vừa ĐỌC labels.csv vừa GHI ra cùng thư mục)
  # -> muốn tách in/out thì bắt buộc phải copy sang sân riêng trước.
  if (( RAW_STANDARD )); then
    info "--raw-standard: xuất bộ chuẩn THÔ vào dataset_out/_raw_standard/ (KHÔNG đụng release/)"
    XSH "mkdir -p dataset_out/_raw_standard"
    XSH "cp dataset_out/labels.csv dataset_out/_raw_standard/labels.csv"
    XSH "[ -f dataset_out/summary.json ] && cp dataset_out/summary.json dataset_out/_raw_standard/ || true"
    X "$PY" -m pipeline.align_engine.to_standard --dataset dataset_out/_raw_standard
  fi
}

# ---- 4/8 remediate ----------------------------------------------------------
# -> labels_remediated.csv + remediation_report.json
# remediation/cli.py:51-53 — --labels/--out ở parser CHA, `census`/`apply` là subcommand;
# viết `apply --labels ...` = LỖI NGAY.  apply có --tau riêng (cli.py:59).
step_remediate() {
  banner 4 remediate "kiểm kê trùng lặp + cách ly/hạ tier -> $LABELS_REMED"
  X "$PY" -m pipeline.remediation --labels "$LABELS_RAW" --out dataset_out census
  X "$PY" -m pipeline.remediation --labels "$LABELS_RAW" --out dataset_out apply --tau "$TAU_REMEDIATE"
  if (( ! DRY )) && [[ ! -f "$LABELS_REMED" ]]; then die "bước remediate không sinh $LABELS_REMED"; fi
}

# nhãn tốt nhất đang có để đem đi chấm — labels_final.csv chỉ ra đời ở BƯỚC 7,
# nên audit (bước 5) không được hardcode nó, nếu không `--all` trên repo sạch sẽ chết.
audit_labels() {
  local f
  for f in "$LABELS_FINAL" "$LABELS_REMED" "$LABELS_RAW"; do
    [[ -f "$f" ]] && { printf '%s' "$f"; return 0; }
  done
  printf '%s' "$LABELS_RAW"
}

# ---- 5/8 audit — ĐIỂM DỪNG CÓ NGƯỜI #1 --------------------------------------
# 5a MÁY   : rank -> plan -> sample -> grid  ==> DỪNG
# 5b NGƯỜI : mở audit_*.html, chấm mù, xuất verdicts_*.jsonl
# 5c MÁY   : estimate  (gọi lại bằng --only estimate --verdicts <file>)
# KHÔNG tự động hoá 5b. Đây là ranh giới giữa "AI tự chấm" và "ground truth".
# ground_truth/cli.py:175-178 — --labels/--out/--config/--conf ở parser CHA.
step_audit() {
  banner 5 audit "dựng lưới chấm mù (tier=$AUDIT_TIER, n=$AUDIT_N, design=$AUDIT_DESIGN) rồi DỪNG"
  local L; L=$(audit_labels)
  [[ "$L" == "$LABELS_FINAL" ]] || warn "chưa có $LABELS_FINAL -> audit trên $L"
  # labels_final.csv CÓ nhưng CŨ HƠN labels.csv = rút mẫu audit từ dân số CŨ (chưa
  # qua bước confusion). Kiểm ở đây, KHÔNG trong audit_labels: hàm đó chạy trong
  # command-substitution `L=$(audit_labels)` (subshell) nên die/warn bị nuốt, phải
  # kiểm ở shell chính để --strict thật sự dừng TRƯỚC khi dựng lưới trên dân số cũ.
  if [[ -f "$LABELS_FINAL" && "$LABELS_FINAL" -ot "$LABELS_RAW" ]]; then
    if (( STRICT && ! DRY )); then
      die "labels_final.csv cũ hơn labels.csv — chạy lại bước confusion trước khi audit"
    else
      warn "labels_final.csv cũ hơn labels.csv — chạy lại bước confusion trước khi audit"
    fi
  fi
  info "chấm trên nhãn: $L"

  if [[ "$AUDIT_TIER" == "GOLD" ]]; then
    X "$PY" -m pipeline.ground_truth --labels "$L" --out "$GT_DIR" rank
    # plan là thống kê thuần, không đọc labels (cli.py:74-84)
    X "$PY" -m pipeline.ground_truth plan --p0 0.97 --p-assumed 0.985
    # --force: né cache labels_ranked.csv CŨ (cli.py:50-51) -> nếu không, mẫu lệch dân số
    X "$PY" -m pipeline.ground_truth --labels "$L" --out "$GT_DIR" \
        sample --n "$AUDIT_N" --design "$AUDIT_DESIGN" --seed 42 --force
    # grid đọc paths từ --config (cli.py:100-101) -> --config PHẢI đứng trước subcommand
    X "$PY" -m pipeline.ground_truth --labels "$L" --out "$GT_DIR" --config "$CONFIG" \
        grid --sample "$GT_DIR/sample_${AUDIT_DESIGN}.csv" --batch-size 150
    # audit_grid.py:228-239 — >batch-size thì tách audit_001.html, audit_002.html, ...
    AUDIT_HTML="$GT_DIR/audit.html (hoặc audit_001.html, audit_002.html, ... nếu chia lô)"
    AUDIT_MANIFEST="$GT_DIR/manifest.jsonl"
  else
    # SILVER / SYLLABLE dùng MODULE RIÊNG, không phải subcommand của cli.py
    # (make_audit_batch.py:150-160 ghi ra dataset_out/ground_truth/audit_<TIER>/)
    X "$PY" -m pipeline.ground_truth.make_audit_batch --tier "$AUDIT_TIER" --n "$AUDIT_N" \
        --labels "$L" --min-per-rule 40 --seed 42 --batch-size 150 --config "$CONFIG"
    X "$PY" -m pipeline.ground_truth.batch_json --all
    AUDIT_HTML="$GT_DIR/audit_${AUDIT_TIER}/audit_*.html"
    AUDIT_MANIFEST="$GT_DIR/audit_${AUDIT_TIER}/manifest.jsonl"
  fi

  cat <<MSG

${BLD}================= ĐIỂM DỪNG CÓ NGƯỜI #1 (audit) =================${RST}
  Máy đã dựng xong lưới chấm. Phần còn lại KHÔNG thể tự động hoá:
  ground truth theo định nghĩa là do NGƯỜI chấm, mù, trên ảnh gốc.

  1. Mở lưới HTML và chấm mù từng crop (tier = $AUDIT_TIER):
        open $AUDIT_HTML

  2. Bấm "Xuất verdicts.jsonl" và LƯU THÀNH:
        $(dirname "$AUDIT_MANIFEST")/verdicts_<lô>.jsonl
     ${YEL}Đặt CẠNH manifest.jsonl của chính lô đó.${RST}
     Glob của fuse_stage/mine_confusions/score_s3 LÀ ĐỆ QUY
     (glob.glob(gt/'**'/verdicts_*.jsonl, recursive=True)) nên thư mục con VẪN được thấy.
     ${YEL}Điều thật sự quan trọng: KHÔNG đặt "source": "ai_vision" trong bản ghi.${RST}
     Bản ghi source=ai_vision bị fuse_stage.py:43,69 LOẠI (verdict MÁY ≠ ground truth).

  3. Nếu dataset vừa được BUILD LẠI sau khi chấm, phải neo lại verdict:
        $PY -m pipeline.ground_truth.reanchor_verdicts \\
          --old-manifest $AUDIT_MANIFEST \\
          --verdicts $GT_DIR --new-labels $LABELS_RAW

  4. Chạy tiếp:
        ./run_pipeline.sh --only estimate --verdicts $GT_DIR/verdicts_001.jsonl
        ./run_pipeline.sh --from fuse --strict
${BLD}=================================================================${RST}

MSG

  # DỪNG chờ người chấm. Ngoại lệ: --dry-run (chỉ in kế hoạch) và --all
  # (người đã tuyên bố verdicts có sẵn — vẫn kiểm chứng bằng số verdict NGƯỜI).
  if (( DRY )); then return 0; fi
  N_VERDICT_HUMAN=-1        # đếm lại: lô vừa dựng có thể đã kèm verdict cũ
  if has_human_verdicts; then
    info "đã thấy $N_VERDICT_HUMAN verdict NGƯỜI -> chạy tiếp các bước sau."
    return 0
  fi
  if (( ALL )); then
    die "--all: chưa có verdict NGƯỜI nào trong $GT_DIR.
      --all giả định audit ĐÃ chấm xong. Chấm đi rồi chạy: ./run_pipeline.sh --from fuse --strict"
  fi
  log "${YEL}DỪNG tại đây: chưa có verdict NGƯỜI trong $GT_DIR.${RST}"
  log "${YEL}Chấm xong rồi chạy: ./run_pipeline.sh --from fuse${RST}"
  exit 0
}

# ---- 5c estimate (bước ảo) --------------------------------------------------
# cli.py:209-219 — estimate cần --verdicts và --manifest (đều required); --out ở
# parser CHA nên đứng TRƯỚC subcommand. KHÔNG truyền --include-ai-verdicts:
# cli.py:216 mặc định TẮT -> chỉ verdict NGƯỜI mới được tính precision (giữ guard).
# --manifest PHẢI khớp lô verdict (biến $MANIFEST, override bằng cờ --manifest);
# hardcode $GT_DIR/manifest.jsonl sẽ ghép verdict SILVER/SYLLABLE với manifest GOLD.
step_estimate() {
  banner "5c" estimate "verdicts -> precision + CI + kế hoạch chấp nhận (+PPI)"
  [[ -n "$VERDICTS" ]] || die "--only estimate cần --verdicts <file .jsonl>"
  if (( ! DRY )) && [[ ! -f "$VERDICTS" ]]; then die "không thấy file verdicts: $VERDICTS"; fi
  if (( ! DRY )) && [[ ! -f "$MANIFEST" ]]; then die "không thấy manifest: $MANIFEST (đúng lô của verdict?)"; fi
  X "$PY" -m pipeline.ground_truth --out "$GT_DIR" \
      estimate --verdicts "$VERDICTS" --manifest "$MANIFEST" \
      --p0 0.97 --design "$AUDIT_DESIGN"
}

# ---- 6/8 fuse ---------------------------------------------------------------
# -> dataset_out/fusion/{channels,fused,labels_fused}.csv + summary.json
# LỚP PHỦ: KHÔNG đụng crops/gold/silver/labels_remediated.csv.
# fuse_stage.py:245-252 — parser phẳng: --config/--tau/--l2/--include-ai-verdicts.
step_fuse() {
  banner 6 fuse "consensus_fusion: hợp nhất kênh (s3+dict) hiệu chỉnh trên verdicts"
  if (( STRICT )) && (( ! DRY )); then
    has_human_verdicts || die "--strict: bước fuse cần verdict NGƯỜI trong $GT_DIR.
      (glob ĐỆ QUY nên thư mục con vẫn được thấy; thứ bị loại là bản ghi source='ai_vision'.)"
  fi
  if (( DRY )) || (( STRICT )); then
    X "$PY" -m pipeline.consensus_fusion.fuse_stage --config "$CONFIG" \
        --tau "$TAU_FUSE" --l2 "$L2_FUSE"
  else
    "$PY" -m pipeline.consensus_fusion.fuse_stage --config "$CONFIG" \
        --tau "$TAU_FUSE" --l2 "$L2_FUSE" \
      || warn "fuse SKIP/lỗi — bỏ qua (lớp phủ, không chặn pipeline). Dùng --strict để bắt lỗi cứng."
  fi
  # gợi ý cặp nhầm lẫn cho bước 7 (người soạn confusion_fixes.yaml đọc bảng này)
  # mine_confusions.py:150-152 — parser phẳng: --top/--include-ai-verdicts.
  if (( DRY )); then
    X "$PY" -m pipeline.consensus_fusion.mine_confusions --top 40
  else
    "$PY" -m pipeline.consensus_fusion.mine_confusions --top 40 \
      || warn "mine_confusions lỗi/skip — không chặn."
  fi
}

# ---- 7/8 confusion — ĐIỂM DỪNG CÓ NGƯỜI #2 ----------------------------------
# config/confusion_fixes.yaml PHẢI do NGƯỜI soạn từ kết luận audit
# (neo trên verdicts_reanchored.csv, 3/3 kiểm chéo đối kháng CONFIRMED).
# -> dataset_out/labels_final.csv = BẢN CÔNG BỐ + confusion_fix_report.json
# confusion_fix.py:105-109 — parser phẳng: --in/--out/--fixes/--measure.
step_confusion() {
  banner 7 confusion "áp bảng sửa nhầm lẫn do NGƯỜI soạn -> $LABELS_FINAL"
  if [[ ! -f "$FIXES_YAML" ]]; then
    cat <<MSG

${BLD}================= ĐIỂM DỪNG CÓ NGƯỜI #2 (confusion) =============${RST}
  Thiếu ${BLD}$FIXES_YAML${RST} — file này KHÔNG được sinh tự động.
  Nội dung là kết luận đã ĐƯỢC NGƯỜI XÁC NHẬN từ audit:
    · đọc bảng gợi ý: $PY -m pipeline.consensus_fusion.mine_confusions --top 40
    · đối chiếu verdicts đã neo: $GT_DIR/verdicts_reanchored.csv
    · chỉ đưa vào cặp đã 3/3 kiểm chéo đối kháng = CONFIRMED
  Soạn xong, chạy lại: ./run_pipeline.sh --only confusion
${BLD}=================================================================${RST}

MSG
    if (( STRICT )); then
      die "--strict: thiếu $FIXES_YAML (bước 7 sẽ thành no-op âm thầm)."
    fi
    warn "bỏ qua bước confusion — $LABELS_FINAL sẽ KHÔNG được sinh/cập nhật."
    return 0
  fi
  X "$PY" -m pipeline.remediation.confusion_fix \
      --in "$LABELS_REMED" --out "$LABELS_FINAL" \
      --fixes "$FIXES_YAML" --measure
  if (( ! DRY )) && [[ ! -f "$LABELS_FINAL" ]]; then die "bước confusion không sinh $LABELS_FINAL"; fi
}

# ---- 8/8 publish ------------------------------------------------------------
# -> dataset_out/release/{labels_published.csv,crops.csv,datapackage.json,
#    croissant.json,README.md,DATASHEET.md,parquet/,imagefolder/}
# BẮT BUỘC truyền --labels labels_final.csv: publish/cli.py:27-36 (_labels_path) tự dò
#   labels_final -> labels_remediated -> labels; nếu labels_final chưa có, release sẽ
#   lặng lẽ đóng gói bản CHƯA sửa nhầm lẫn.
# Gọi 5 lệnh RỜI thay vì `all` để log/chẩn đoán rõ và để --export-sample điều khiển được.
# publish/cli.py:168 — --labels ở parser CHA -> đứng TRƯỚC subcommand.
step_publish() {
  banner 8 publish "đóng gói bản công bố quốc tế -> $RELEASE_DIR/"
  if (( ! DRY )) && [[ ! -f "$LABELS_FINAL" ]]; then
    die "bước publish cần $LABELS_FINAL.
      Chạy bước 7 trước (./run_pipeline.sh --only confusion) hoặc trỏ --labels <file>."
  fi
  PUB=("$PY" -m pipeline.publish --labels "$LABELS_FINAL")
  X "${PUB[@]}" split
  X "${PUB[@]}" metadata
  X "${PUB[@]}" datasheet
  if (( EXPORT_SAMPLE > 0 )); then
    warn "--export-sample $EXPORT_SAMPLE: parquet chỉ là SMOKE TEST, KHÔNG dùng để công bố."
    X "${PUB[@]}" export --sample "$EXPORT_SAMPLE"
  else
    X "${PUB[@]}" export
  fi
  X "${PUB[@]}" validate            # exit 1 nếu fail -> cổng CI
}

# ====================== FREEZE / EVIDENCE ====================================
# 0 chi phí tính toán. Trả lời câu "chạy lại có giống không" bằng SỐ, không cảm tính.
evidence() {
  local files=("$LABELS_RAW" "$LABELS_REMED" "$LABELS_FINAL"
               "$RELEASE_DIR/crops.csv" "$RELEASE_DIR/datapackage.json"
               "$RELEASE_DIR/croissant.json")
  local sha_cmd=""
  if command -v shasum >/dev/null 2>&1; then sha_cmd="shasum -a 256"
  elif command -v sha256sum >/dev/null 2>&1; then sha_cmd="sha256sum"; fi
  log ""
  log "${BLD}--- BẰNG CHỨNG (sha256) -> $EVIDENCE ---------------------${RST}"
  if (( DRY )); then
    printf '    %s$%s %s %s\n' "$CYA" "$RST" "${sha_cmd:-shasum -a 256}" "${files[*]}"
    printf '    %s$%s (ghi bảng sha256 vào %s)\n' "$CYA" "$RST" "$EVIDENCE"
    return 0
  fi
  [[ -n "$sha_cmd" ]] || { warn "không có shasum/sha256sum — bỏ qua bảng bằng chứng"; return 0; }
  mkdir -p "$(dirname "$EVIDENCE")"
  {
    printf '\n## Lần chạy %s\n\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'phạm vi: %s | strict=%s | reseg=%s | config=%s\n\n' \
           "${PLAN[*]:-none}" "$STRICT" "$RESEG" "$CONFIG"
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
preflight

if (( CHECK_ONLY )) || [[ ",$ONLY," == *",preflight,"* ]]; then
  log ""
  ok "check-only: preflight xong ($N_WARN cảnh báo, $N_FAIL mục đỏ). Thoát."
  exit 0
fi

BOOKS="$(resolve_books)"
[[ -n "${BOOKS// /}" ]] || warn "không đọc được danh sách sách từ $CONFIG"

# 8 dòng dưới đây LÀ toàn bộ luồng. Dùng `if ... then ... fi` chứ KHÔNG dùng
# `A && B || true` để một bước lỗi là dừng thật (set -e không bị nuốt).
if want_step setup;     then step_setup;     fi
if want_step extract;   then step_extract;   fi
if want_step build;     then step_build;     fi
if want_step remediate; then step_remediate; fi
if want_step audit;     then step_audit;     fi
if want_step estimate;  then step_estimate;  fi
if want_step fuse;      then step_fuse;      fi
if want_step confusion; then step_confusion; fi
if want_step publish;   then step_publish;   fi

evidence

log ""
log "${BLD}================================================================${RST}"
if (( DRY )); then
  log "${BLD}  DRY-RUN xong — KHÔNG có bước nào được thực thi.${RST}"
else
  log "${GRN}${BLD}  Pipeline xong: ${PLAN[*]:-(không bước nào)}${RST}"
  log "  Nhãn công bố : $LABELS_FINAL"
  log "  Bộ chuẩn     : $RELEASE_DIR/ (nếu đã chạy bước publish)"
  log "  cảnh báo     : $N_WARN"
fi
log "${BLD}================================================================${RST}"

# ---------------------------------------------------------------------------
# GHI CHÚ IDEMPOTENT (đưa nguyên vào README + Phụ lục luận văn):
#  TẤT ĐỊNH       : bước remediate/confusion/publish — pandas thuần, seed 42 cố định
#                   (splits.py:59, publish/cli.py:118, ground_truth/cli.py:193)
#  BÁN TẤT ĐỊNH   : bước build — phụ thuộc (a) cache OCR trong prepared/,
#                   (b) crop-proto đọc từ align_engine/data/index.csv trỏ
#                   dataset_out/gold/*.png, (c) device torch (MPS/CPU) lệch ~1e-3
#  KHÔNG TẤT ĐỊNH : bước extract khi xoá cache OCR (gọi API ngoài)
#  BẪY CACHE      : ground_truth/cli.py:50-51 dùng labels_ranked.csv CŨ nếu không --force
#  VERDICT        : glob verdicts_*.jsonl LÀ ĐỆ QUY (fuse_stage.py:62); cửa chặn thật
#                   là source='ai_vision' bị loại (fuse_stage.py:43,69)
#  KHÔNG DÙNG     : run_book*.sh chỉ là bản 1-sách của bước setup/extract/build;
#                   chạy các bước 4-8 trên 1 sách sẽ ghi đè labels.csv bằng bản thiếu 2 sách
# ---------------------------------------------------------------------------
