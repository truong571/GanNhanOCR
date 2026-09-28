#!/usr/bin/env bash
# =============================================================================
# clean_rebuild_all.sh — XOÁ SẠCH dữ liệu DẪN XUẤT rồi (tuỳ chọn) CHẠY LẠI TOÀN BỘ 10 BỘ bằng mã mới nhất
# (28/09/2026; thay scripts/clean_build.sh — bản thời chỉ có STT, nay đã LỖI THỜI).
#
#   bash scripts/clean_rebuild_all.sh                # CHẠY THỬ (mặc định): kiểm cache OCR 10 bộ + in GIỮ / SẼ XOÁ
#   bash scripts/clean_rebuild_all.sh --yes          # kiểm kê sha256 phần GIỮ -> xoá dẫn xuất -> kiểm lại kiểm kê
#   bash scripts/clean_rebuild_all.sh --yes --run    # ... rồi chạy lại 10 bộ + bộ nhãn người Borg + bộ đo + báo cáo
#   bash scripts/clean_rebuild_all.sh --deep [...]   # xoá THÊM cache dẫn xuất nặng (0 API, nhưng dựng lại lâu)
#   bash scripts/clean_rebuild_all.sh --root <thư mục> [--yes]   # thử trên BẢN GIẢ/bản sao (mã Python vẫn lấy từ repo này)
#
# 10 BỘ: 3 STT (SachThanhTruyen2/4/11, config/pipeline.yaml) + 3 sách giao nộp (LucVanTien1883, KimVanKieu1884,
# Chrestomathie1872) + 2 IHR đánh giá (LucVanTien1916, TruyenKieu1872) + 2 Borg đánh giá (SachKinhThayCaBinh,
# SachDungLyHoThan). Chạy lại = `./run_pipeline.sh --book all --yes --borg require --no-verify` (gộp dataset/_ALL,
# GOLD chính xác) -> `python -m pipeline.borg_human --stage all` -> `scripts/measure/measure.py --all` ->
# `./run_pipeline.sh --verify` -> `scripts/bao_cao_tong_hop.py` (docs/BAO_CAO_TONG_HOP_<ngày>.md). Log: logs/clean_rebuild_<thời điểm>_*.
#
# RANH GIỚI (xác định bằng MÃ, không đoán — xem khối Python bên dưới, hàm protected_roots/targets):
#   GIỮ, KHÔNG BAO GIỜ XOÁ (kiểm kê sha256 trước/sau khi xoá -> measure_out/_clean_manifest/<ts>.txt):
#     prepared/<sách>/ trừ dataset_out/ — cache kim thô kim_raw/ (+ prepared/_auto/<Borg>/kim_raw/, kim_calls.json:
#       641 lượt API ~2 giờ), cache OCR trang detected/*_ocr_cache.json, cache QN transcriptions/*qn*cache*, ảnh trang
#       pages/ mà image_hash của cache neo vào (PNG ghi lại có thể KHÁC byte -> trượt cache -> GỌI API), bản nạp người
#       Borg prepared/SachKinhThayCaBinh|SachDungLyHoThan; data/; models/; measure_out/_* (kho nguồn gold_exact,
#       _borg_human_src, kho lưu _audit_*, _thu_nghiem_anh_chu mà gold_exact đọc TN4…); MỌI thư mục bộ đo có cache kim
#       (measure_out/auto_precision, kim_channel, <IHR>/ihr_layout — measure.py --all gọi kim 10 trang/hệ số ở ihr_layout
#       nếu trượt cache); dataset_out/human_audit/, dataset_out/CHECKSUMS.txt (nhật ký có trong git), .env; nguyên mẫu S3
#       đóng băng pipeline/align_engine/s3_proto_cache.pkl (+ data/index.csv — index chỉ còn trỏ tới một phần ảnh crop
#       nên dựng lại sẽ ra bộ nguyên mẫu KHÁC -> nhãn sách mới đổi); đầu vào mô hình train_crop/, nom-embed/, lab/…
#   XOÁ (--run sinh lại, 0 API): dataset/* (cả _ALL, _BORG_NHAN_NGUOI; GIỮ dataset/SachThanhTruyen{2,4,11} cho web), dataset_out/* (trừ human_audit, CHECKSUMS.txt),
#     prepared/*/dataset_out, prepared/_auto/*/dataset_out, đầu ra đo mà --run sinh lại (measure_out/SUMMARY.json,
#     REPORT.md, logs/, code_facts/, align_audit/, gold_exact/, borg_human/, detector_transfer/, box_ref/, _gold_exact/,
#     <LVT1883|KVK1884>/qn_ref_fix/, <IHR>/ihr_endtoend/, <Borg>/borg_endtoend/), **/__pycache__.
#   --deep XOÁ THÊM: prepared/_gold_exact (cache crop chuẩn + nhúng, ~4 GB), measure_out/_borg_human (trung gian
#     borg_human), measure_out/<LVT1883|KVK1884>/{layout,qn_ocr}, measure_out/Chrestomathie1872/chresto_map (đầu vào
#     tiền xử lý — run_pipeline tự đo lại ở bước 0). ⚠ Với --deep, tập trang của lần chạy sau do mã đo MỚI quyết định:
#     phép kiểm cache dưới đây dựa trên tập trang HIỆN TẠI; trang mới (nếu có) sẽ bị CHẶN ở lượt chạy (xem dưới).
#   Kết quả thí nghiệm cũ trong measure_out/ mà --run KHÔNG sinh lại (box_ref_v9_*, qn_engine, loss_ledger…) và logs/
#     được GIỮ nguyên.
#
# CHỐT CHẶN (không bao giờ để chạy lại phải GỌI API kim):
#   1. Trước khi xoá: mọi trang của 10 bộ phải có cache hợp lệ theo ĐÚNG khoá mà mã dùng (STT: detected/<trang>_ocr_cache
#      .json + image_hash == md5 ảnh, cache QN đúng phiên bản + md5 ảnh; 7 bộ kim: kim_raw/<trang><hậu tố>.json với md5
#      ảnh GỬI + tham số kim của config HIỆN TẠI; Borg thêm `ingest_borg_book --status`; ihr_layout: mỗi ảnh png/ có
#      kim_cache). Thiếu/lệch -> --yes DỪNG CỨNG (chạy thử chỉ báo).
#   2. --run: mọi tiến trình Python con nạp một sitecustomize tạm CHẶN mọi yêu cầu HTTP tới *.hcmus.edu.vn / $SN_DOMAIN
#      (ném lỗi, ghi logs/clean_rebuild_<ts>_api_bi_chan.txt) — lỡ có trang trượt cache thì bộ đó HỎNG, không gọi API.
#   3. Sau --run: tệp cache gốc từ API (kim_raw, kim_calls.json, cache OCR/QN của STT, kim_cache + png của bộ đo) phải
#      TRÙNG sha256 với kiểm kê và không có tệp mới; manifest ingest: ocr_calls = 0, Borg n_pages_new_kim = 0.
#
# Viết cho bash 3.2 (bash mặc định của macOS).
# =============================================================================
set -uo pipefail

CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$CODE_DIR"
PY="${PYTHON_BIN:-$CODE_DIR/.venv/bin/python}"
DO_IT=0; RUN=0; DEEP=0

RED=""; YEL=""; GRN=""; CYA=""; BLD=""; RST=""
if [[ -t 1 ]]; then
  RED=$'\033[31m'; YEL=$'\033[33m'; GRN=$'\033[32m'; CYA=$'\033[36m'; BLD=$'\033[1m'; RST=$'\033[0m'
fi
die()  { printf '%s%s[DỪNG]%s %s\n' "$RED" "$BLD" "$RST" "$*" >&2; exit 1; }
ok()   { printf '%s[OK]%s %s\n' "$GRN" "$RST" "$*"; }
info() { printf '%s[i]%s %s\n' "$CYA" "$RST" "$*"; }
warn() { printf '%s[CẢNH BÁO]%s %s\n' "$YEL" "$RST" "$*" >&2; }
banner() { printf '\n%s================================================================%s\n%s  %s%s\n%s================================================================%s\n' \
  "$BLD" "$RST" "$BLD" "$*" "$RST" "$BLD" "$RST"; }

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; }

while (( $# )); do
  case "$1" in
    -y|--yes) DO_IT=1 ;;
    --run)    RUN=1 ;;
    --deep)   DEEP=1 ;;
    --root)   [[ $# -ge 2 ]] || die "--root cần một thư mục"; ROOT="$(cd "$2" 2>/dev/null && pwd)" || die "không vào được $2"; shift ;;
    --root=*) ROOT="$(cd "${1#--root=}" 2>/dev/null && pwd)" || die "không vào được ${1#--root=}" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "tham số lạ: $1" ;;
  esac
  shift
done
(( RUN && ! DO_IT )) && die "--run phải đi kèm --yes (xoá dẫn xuất rồi mới chạy lại)."
(( RUN )) && [[ "$ROOT" != "$CODE_DIR" ]] && die "--run chỉ dùng trên repo thật (không đi với --root)."
[[ -x "$PY" ]] || die "không thấy Python ở $PY"
[[ "$ROOT" == "/" ]] && die "--root không được là /"
if [[ "$ROOT" != "$CODE_DIR" ]]; then
  [[ -d "$ROOT/prepared" || -d "$ROOT/dataset" ]] || die "$ROOT không giống một cây GanNhanOCR (thiếu prepared/ và dataset/)."
fi
cd "$ROOT" || die "không vào được $ROOT"

TMPD="$(mktemp -d -t clean_rebuild.XXXXXX)" || die "mktemp lỗi"
trap 'rm -rf "$TMPD"' EXIT
HELPER="$TMPD/clean_rebuild_helper.py"
TARGETS="$TMPD/targets.txt"
export CR_CODE="$CODE_DIR" CR_ROOT="$ROOT" CR_DEEP="$DEEP" PYTHONHASHSEED=0

# ------------------------------------------------------------------------------------------------------------------
# Trợ lý Python (ghi ra tệp tạm; mọi phép kiểm 0 API, chỉ đọc — trừ lệnh con `delete`)
# ------------------------------------------------------------------------------------------------------------------
cat > "$HELPER" <<'PYEOF'
# -*- coding: utf-8 -*-
import argparse, hashlib, json, os, shutil, stat, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE = Path(os.environ["CR_CODE"]).resolve()
ROOT = Path(os.environ["CR_ROOT"]).resolve()
DEEP = os.environ.get("CR_DEEP") == "1"
sys.path.insert(0, str(CODE))
TTY = sys.stdout.isatty()
RED, YEL, GRN, BLD, RST = (("\033[31m", "\033[33m", "\033[32m", "\033[1m", "\033[0m") if TTY else ("",) * 5)

STT_BOOKS = ["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"]
KIM_BOOKS = ["LucVanTien1883", "KimVanKieu1884", "Chrestomathie1872", "LucVanTien1916", "TruyenKieu1872",
             "SachKinhThayCaBinh", "SachDungLyHoThan"]
BORG_BOOKS = ["SachKinhThayCaBinh", "SachDungLyHoThan"]
LITHO = ["LucVanTien1883", "KimVanKieu1884"]
IHR = ["LucVanTien1916", "TruyenKieu1872"]
SKIP = {".DS_Store", "__pycache__"}
# thư mục measure_out/_* là DẪN XUẤT (không phải kho lưu): _gold_exact = thư mục làm việc bước B8; _borg_human = trung
# gian borg_human (--deep); _clean_manifest = kiểm kê của chính script này (không băm chính nó)
MO_DERIVED_UNDERSCORE = {"_gold_exact", "_borg_human", "_clean_manifest"}


def rel(p: Path) -> str:
    return os.path.relpath(str(p), str(ROOT))


def fmt_n(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt_b(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.1f} GB".replace(".", ",")
    if b >= 1 << 20:
        return f"{b / (1 << 20):.0f} MB"
    return f"{b / 1024:.0f} KB"


def has_symlink_component(p: Path) -> bool:
    cur = ROOT
    for part in p.relative_to(ROOT).parts:
        cur = cur / part
        if cur.is_symlink():
            return True
    return False


# ---------------------------------------------------------------- phạm vi XOÁ
def targets() -> list[tuple[str, str]]:
    """[(đường dẫn tương đối, lý do)] — CHỈ những gì chuỗi --run sinh lại (0 API)."""
    T = []

    def add(r, why):
        p = ROOT / r
        if (p.exists() or p.is_symlink()) and not has_symlink_component(p.parent if p.is_symlink() else p):
            T.append((r, why))

    # dataset/SachThanhTruyen{2,4,11}: bản xuất STT theo quyển (23/09) mà web/server.py còn đọc và KHÔNG bước nào
    # sinh lại -> GIỮ (nằm trong protected_roots); mọi thư mục/tệp khác của dataset/ được xoá từng mục.
    ds = ROOT / "dataset"
    if ds.is_dir():
        for p in sorted(ds.iterdir()):
            if p.name in STT_BOOKS or p.name in SKIP:
                continue
            add(rel(p), "bản xuất bộ / _ALL / _BORG_NHAN_NGUOI (run_pipeline, borg_human)")
    dso = ROOT / "dataset_out"
    if dso.is_dir():
        for p in sorted(dso.iterdir()):
            if p.name in ("human_audit", "CHECKSUMS.txt", ".DS_Store"):
                continue
            add(rel(p), "đầu ra trung gian STT (đường STT 6 bước)")
    for pat in ("prepared/*/dataset_out", "prepared/_auto/*/dataset_out"):
        for p in sorted(ROOT.glob(pat)):
            add(rel(p), "đầu ra trung gian sách (run_new_book B2–B6)")
    mo = "measure_out"
    for r, why in ((f"{mo}/SUMMARY.json", "measure.py"), (f"{mo}/REPORT.md", "measure.py"), (f"{mo}/logs", "measure.py"),
                   (f"{mo}/code_facts", "measure.py"), (f"{mo}/detector_transfer", "measure.py"),
                   (f"{mo}/box_ref", "measure.py"), (f"{mo}/gold_exact", "--verify + measure.py (gold_exact_eval)"),
                   (f"{mo}/borg_human", "measure.py (borg_human_eval)"), (f"{mo}/align_audit", "--verify (align_audit)"),
                   (f"{mo}/_gold_exact", "thư mục làm việc bước GOLD chính xác (B8)")):
        add(r, why)
    for b in LITHO:
        add(f"{mo}/{b}/qn_ref_fix", "run_new_book bước 0 (verses_ref_fix) sinh lại mỗi lượt")
    for b in IHR:
        add(f"{mo}/{b}/ihr_endtoend", "--verify + measure.py (ihr_endtoend_eval)")
    for b in BORG_BOOKS:
        add(f"{mo}/{b}/borg_endtoend", "run_new_book B6 + --verify + measure.py (borg_endtoend_eval)")
    if DEEP:
        add("prepared/_gold_exact", "--deep: cache crop chuẩn + nhúng + hộp kim của gold_exact (dựng lại 0 API, lâu)")
        add(f"{mo}/_borg_human", "--deep: trung gian borg_human (dựng lại 0 API)")
        for b in LITHO:
            add(f"{mo}/{b}/layout", "--deep: đầu vào tiền xử lý (run_new_book bước 0 đo lại)")
            add(f"{mo}/{b}/qn_ocr", "--deep: đầu vào tiền xử lý (run_new_book bước 0 đo lại; cache tesseract _cache/ giữ)")
        add(f"{mo}/Chrestomathie1872/chresto_map", "--deep: đầu vào tiền xử lý (run_new_book bước 0 đo lại)")
    # __pycache__ (ngoài .venv*/.git/tf_env), không đi theo symlink
    for dp, dns, _ in os.walk(ROOT):
        dns[:] = [d for d in dns if not (d.startswith(".venv") or d in (".git", "tf_env", "node_modules"))
                  and not os.path.islink(os.path.join(dp, d))]
        if "__pycache__" in dns:
            T.append((rel(Path(dp) / "__pycache__"), "__pycache__"))
            dns.remove("__pycache__")
    return T


# ---------------------------------------------------------------- phạm vi GIỮ bất khả tái tạo (băm sha256)
def measure_kim_dirs() -> list[str]:
    """Thư mục bộ đo (ngoài measure_out/_*) có cache kim từ API -> giữ NGUYÊN CẢ thư mục bước đo (kim_cache + ảnh gửi)."""
    out = set()
    mo = ROOT / "measure_out"
    if not mo.is_dir():
        return []
    for dp, dns, fns in os.walk(mo):
        r = Path(dp).relative_to(mo)
        if r.parts and r.parts[0].startswith("_"):
            dns[:] = []
            continue
        dns[:] = [d for d in dns if d not in SKIP]
        if "kim_cache" in dns or "kim_calls.json" in fns:
            out.add(rel(Path(dp)))
            dns[:] = []
    return sorted(out)


def protected_roots() -> list[str]:
    R = ["prepared", "data", "models", ".env", "dataset_out/human_audit", "dataset_out/CHECKSUMS.txt",
         "pipeline/align_engine/s3_proto_cache.pkl", "pipeline/align_engine/data/index.csv",
         "train_crop", "nom-embed", "lab/enhanced_self_training_v3", "lab/self_training_v2", "KhoiB/v3",
         "lab/gan_nhan_2026-09-13/crops.npz"]
    R += [f"dataset/{b}" for b in STT_BOOKS]   # bản xuất STT theo quyển (web viewer), không sinh lại được
    mo = ROOT / "measure_out"
    if mo.is_dir():
        R += [rel(p) for p in sorted(mo.iterdir()) if p.name.startswith("_") and p.name not in MO_DERIVED_UNDERSCORE]
    R += measure_kim_dirs()
    return [r for r in R if (ROOT / r).exists() or (ROOT / r).is_symlink()]


def excluded_from_protected() -> list[str]:
    """Nằm TRONG gốc giữ nhưng là dẫn xuất (không băm): đích xoá + prepared/_gold_exact (cache dựng lại được)."""
    ex = [t for t, _ in targets()]
    ex.append("prepared/_gold_exact")
    return ex


def under(r: str, base: str) -> bool:
    return r == base or r.startswith(base.rstrip("/") + "/")


def walk_protected():
    """Sinh (rel, kind, st) cho mọi tệp/symlink thuộc phạm vi GIỮ (không theo symlink thư mục).
    os.walk tỉa thư mục bị loại ngay tại chỗ, nên so KHỚP ĐÚNG với tập loại trừ là đủ (__pycache__ đã nằm trong SKIP)."""
    ex = [e for e in excluded_from_protected() if not e.endswith("__pycache__")]
    ex_set = set(ex)
    seen = set()
    for r0 in protected_roots():
        p0 = ROOT / r0
        if any(under(r0, e) for e in ex):
            continue
        if p0.is_symlink() or p0.is_file():
            if r0 not in seen:
                seen.add(r0)
                yield r0, ("link" if p0.is_symlink() else "file"), p0.lstat()
            continue
        for dp, dns, fns in os.walk(p0):
            rd = rel(Path(dp))
            keep_d = []
            for d in dns:
                rr = f"{rd}/{d}" if rd != "." else d
                if d in SKIP or rr in ex_set:
                    continue
                if os.path.islink(os.path.join(dp, d)):
                    if rr not in seen:
                        seen.add(rr)
                        yield rr, "link", os.lstat(os.path.join(dp, d))
                    continue
                keep_d.append(d)
            dns[:] = keep_d
            for f in fns:
                if f in SKIP:
                    continue
                rr = f"{rd}/{f}" if rd != "." else f
                if rr in seen or rr in ex_set:
                    continue
                seen.add(rr)
                fp = os.path.join(dp, f)
                st = os.lstat(fp)
                yield rr, ("link" if stat.S_ISLNK(st.st_mode) else "file"), st


def category(r: str) -> str:
    n = r.rsplit("/", 1)[-1]
    if r.startswith("prepared/"):
        if "/kim_raw/" in r:
            return "A1 cache kim thô (prepared/**/kim_raw/)"
        if n == "kim_calls.json":
            return "A2 sổ lượt gọi kim (prepared/_auto/*/kim_calls.json)"
        if "/detected/" in r and n.endswith("_ocr_cache.json"):
            return "A3 cache OCR trang (prepared/*/detected/*_ocr_cache.json)"
        if "/transcriptions/" in r and "qn" in n and "cache" in n:
            return "A4 cache OCR Quốc ngữ (prepared/*/transcriptions/*qn*cache*)"
        if r.startswith(("prepared/SachKinhThayCaBinh/", "prepared/SachDungLyHoThan/")):
            return "A6 bản nạp NGƯỜI Borg (ingest_borg_tonch: prepared/SachKinhThayCaBinh|SachDungLyHoThan)"
        if "/pages/" in r or "/pages_denoised/" in r or n.endswith("_qn_tmp.png"):
            return "A5 ảnh trang — neo image_hash của cache (pages/, pages_denoised/, *_qn_tmp.png)"
        return "A7 phần còn lại prepared/<sách> (manifest, transcriptions, aligned, labeled, symlink bí danh…)"
    if r.startswith("data/") or r == "data":
        return "B1 dữ liệu nguồn data/"
    if r.startswith("models/"):
        return "B2 tài sản models/ (gold_exact…)"
    if r.startswith("measure_out/_gold_exact_assets_src/"):
        return "C1 kho nguồn tài sản gold_exact (measure_out/_gold_exact_assets_src/)"
    if r.startswith("measure_out/_borg_human_src/"):
        return "C2 kho nguồn borg_human (measure_out/_borg_human_src/)"
    if r.startswith("measure_out/_audit_"):
        return "C3 kho lưu measure_out/_audit_*"
    if r.startswith("measure_out/_"):
        return "C4 kho lưu/thí nghiệm khác measure_out/_* (_thu_nghiem_anh_chu: gold_exact đọc TN4; _gold_exact_run1; _cache…)"
    if r.startswith("measure_out/"):
        return "C5 bộ đo có cache kim từ API (measure_out/{auto_precision,kim_channel,<IHR>/ihr_layout,…})"
    if r in (".env", "dataset_out/CHECKSUMS.txt") or r.startswith("dataset_out/human_audit"):
        return "D1 .env · dataset_out/CHECKSUMS.txt (nhật ký, git) · dataset_out/human_audit/ (phán quyết người)"
    if r.startswith("pipeline/align_engine/"):
        return "D2 nguyên mẫu S3 đóng băng (s3_proto_cache.pkl + data/index.csv)"
    return "D3 đầu vào mô hình (train_crop/, nom-embed/, lab/*self_training*, KhoiB/v3, crops.npz)"


def du(p: Path) -> tuple[int, int]:
    """(số tệp, byte trên đĩa) không theo symlink."""
    if p.is_symlink() or p.is_file():
        st = p.lstat()
        return 1, st.st_blocks * 512
    n = b = 0
    for dp, dns, fns in os.walk(p):
        for f in fns:
            try:
                st = os.lstat(os.path.join(dp, f))
            except OSError:
                continue
            n += 1
            b += st.st_blocks * 512
    return n, b


# ---------------------------------------------------------------- kiểm cache OCR (0 API)
def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def book_config(book: str) -> tuple[Path, dict]:
    import yaml
    p = ROOT / "config" / f"pipeline_{book}.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if cfg.get("run_config"):
        q = Path(str(cfg["run_config"]))
        p = q if q.is_absolute() else ROOT / q
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return p, cfg


def cmd_check(a) -> int:
    rows, bad = [], 0
    # --- 3 STT: cache kim cả trang + cache QN (VietOCR) theo từng ảnh pages/*.png
    qn_ver = None
    try:
        import yaml
        from core.ocr.qn_ocr import _cache_version
        base = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text(encoding="utf-8")) or {}
        qn_ver = _cache_version((base.get("step1") or {}).get("qn_line_detector", "auto"))
    except Exception as e:  # noqa: BLE001
        print(f"  {YEL}không tính được phiên bản cache QN ({e}) -> chỉ kiểm tồn tại + md5{RST}")
    try:
        from core.ocr.ocr_api import _pixel_hash
    except Exception:  # noqa: BLE001
        _pixel_hash = None
    for b in STT_BOOKS:
        d = ROOT / "prepared" / b
        pngs = sorted((d / "pages").glob("*.png"))
        miss, stale, healable, qmiss, qstale = [], [], 0, [], []
        for png in pngs:
            c = d / "detected" / f"{png.stem}_ocr_cache.json"
            if not c.exists():
                miss.append(png.stem)
            else:
                try:
                    j = json.loads(c.read_text(encoding="utf-8"))
                    h = j.get("image_hash")
                    if h and h != md5(png):
                        if _pixel_hash and j.get("pixel_hash") and j.get("pixel_hash") == _pixel_hash(str(png)):
                            healable += 1        # verify_cache_image sẽ "heal" (ghi lại image_hash), không gọi API
                        else:
                            stale.append(png.stem)   # verify_cache_image ném StaleOCRCacheError
                except Exception:  # noqa: BLE001
                    stale.append(png.stem)
            q = d / "transcriptions" / f"{png.stem}_qn_ocr_cache.json"
            qi = d / "transcriptions" / f"{png.stem}_qn_tmp.png"
            if not q.exists() or not qi.exists():
                qmiss.append(png.stem)
            else:
                try:
                    jq = json.loads(q.read_text(encoding="utf-8"))
                    if (qn_ver and jq.get("version") != qn_ver) or jq.get("image_md5") != md5(qi):
                        qstale.append(png.stem)
                except Exception:  # noqa: BLE001
                    qstale.append(png.stem)
        nb = len(miss) + len(stale) + len(qmiss) + len(qstale) + (0 if pngs else 1)
        bad += nb
        note = f"kim {len(pngs) - len(miss) - len(stale)}/{len(pngs)} · QN {len(pngs) - len(qmiss) - len(qstale)}/{len(pngs)}"
        if healable:
            note += f" · {healable} cache 'heal' được (PNG nén khác, pixel trùng)"
        ex = (miss + stale + qmiss + qstale)[:4]
        rows.append((b, "STT: kim cả trang + QN", len(pngs), nb, note + (f" · vd {ex}" if ex else "")))
    # --- 7 bộ kim (thạch bản / văn xuôi / IHR / Borg): mọi trang của manifest lần nạp trước
    from pipeline.tools.ingest_lithograph_book import book_kim_params, kim_cache_suffix, kim_params_of
    try:
        from pipeline.tools.ingest_prose_book import BOOKS as PROSE
    except Exception:  # noqa: BLE001
        PROSE = {}
    for b in KIM_BOOKS:
        try:
            cfgp, cfg = book_config(b)
            data_dir = (cfg.get("paths") or {}).get("data_dir") or "prepared"
            k = book_kim_params(b, config=cfgp)
            sfx = kim_cache_suffix(k)
        except Exception as e:  # noqa: BLE001
            rows.append((b, "kim", 0, 1, f"không đọc được config ({e})")); bad += 1
            continue
        d = ROOT / data_dir / b
        mf = d / "manifest.json"
        if not mf.exists():
            rows.append((b, f"kim{sfx}", 0, 1, f"THIẾU {rel(mf)} — không biết tập trang, chạy lại sẽ phải gọi API")); bad += 1
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        pages = [p for p in (m.get("pages") or []) if "blank_page" not in (p.get("flags") or [])]
        orig = m.get("kim_src") == "orig"
        src_dir = None
        if orig:
            if m.get("orig_dir"):
                src_dir = ROOT / m["orig_dir"]
            elif b in PROSE:
                s = Path(PROSE[b]["src"])
                try:
                    src_dir = ROOT / s.relative_to(CODE)
                except ValueError:
                    src_dir = s
        miss = []
        for p in pages:
            name = p["page_name"]
            img = (src_dir / p["source_file"]) if orig and src_dir else d / "pages" / f"{name}.png"
            raw = d / "kim_raw" / f"{name}{sfx}.json"
            okk = False
            if raw.exists() and img.exists():
                try:
                    j = json.loads(raw.read_text(encoding="utf-8"))
                    okk = (j.get("image_hash") == md5(img) and isinstance(j.get("boxes"), list)
                           and kim_params_of(j.get("kim_params")) == k)
                except Exception:  # noqa: BLE001
                    okk = False
            if not okk:
                miss.append(name)
        note = f"{len(pages) - len(miss)}/{len(pages)} trang · khoá = md5 ảnh {'gốc' if orig else 'pages/'} + {k}"
        if not pages:
            miss.append("(manifest 0 trang)")
        if b in BORG_BOOKS and ROOT == CODE:
            try:
                from pipeline.tools.ingest_borg_book import page_status
                s = page_status(b, ROOT / data_dir, config=cfgp)
                note += f" · adapter --status {s['n_cached']}/{s['n_pages']}"
                if not s["complete"]:
                    miss.append(f"adapter: thiếu {s['n_missing']} trang")
            except Exception as e:  # noqa: BLE001
                note += f" · adapter --status lỗi ({e})"
                miss.append("adapter-status")
        bad += len(miss)
        rows.append((b, f"kim{sfx}" + (" (Borg)" if b in BORG_BOOKS else ""), len(pages), len(miss),
                     note + (f" · vd {miss[:4]}" if miss else "")))
    # --- bộ đo: ihr_layout (measure.py --all gọi kim --ihr-kim-pages 10 × hệ số) — mỗi ảnh png/ phải có kim_cache
    for b in IHR:
        d = ROOT / "measure_out" / b / "ihr_layout"
        pngs = sorted((d / "png").glob("*.png")) if d.is_dir() else []
        keys = {f.name.split("_", 1)[0] for f in (d / "kim_cache").glob("*.json")} if d.is_dir() else set()
        miss = [p.name for p in pngs if md5(p) not in keys]
        nb = len(miss) + (0 if pngs else 1)
        bad += nb
        rows.append((f"{b}/ihr_layout", "bộ đo: kim ×hệ số", len(pngs), nb,
                     f"{len(pngs) - len(miss)}/{len(pngs)} ảnh có kim_cache" + ("" if pngs else " · THIẾU png/ hoặc thư mục")))
    w = [max(len(str(r[i])) for r in rows + [("bộ", "loại cache", "trang", "thiếu", "")]) for i in range(4)]
    print(f"  {'bộ'.ljust(w[0])}  {'loại cache'.ljust(w[1])}  {'trang'.rjust(w[2])}  {'thiếu'.rjust(w[3])}  ghi chú")
    for r in rows:
        col = RED if r[3] else GRN
        print(f"  {r[0].ljust(w[0])}  {r[1].ljust(w[1])}  {str(r[2]).rjust(w[2])}  {col}{str(r[3]).rjust(w[3])}{RST}  {r[4]}")
    if bad:
        print(f"  {RED}{BLD}THIẾU/LỆCH {bad} mục cache — chạy lại sẽ GỌI API hoặc hỏng: KHÔNG được xoá/chạy lại.{RST}")
        return 3
    print(f"  {GRN}đủ cache cho 10 bộ + bộ đo ihr_layout (theo tập trang hiện tại) -> chạy lại 0 lượt API{RST}")
    return 0


# ---------------------------------------------------------------- liệt kê GIỮ / XOÁ
def cmd_plan(a) -> int:
    T = targets()
    prot = protected_roots()
    # kiểm chéo: không đích xoá nào chứa hay nằm dưới một gốc GIỮ trừ trường hợp đã loại trừ có chủ đích
    for t, _ in T:
        for r in prot:
            if under(r, t):
                print(f"{RED}LỖI THIẾT KẾ: đích xoá {t} chứa gốc GIỮ {r}{RST}")
                return 2
    cats = {}
    for r, kind, st in walk_protected():
        c = category(r)
        n, b = cats.get(c, (0, 0))
        cats[c] = (n + 1, b + st.st_blocks * 512)
    print(f"{BLD}--- GIỮ · BẤT KHẢ TÁI TẠO / ĐẦU VÀO NEO (băm sha256 trước & sau khi xoá) ---{RST}")
    tn = tb = 0
    for c in sorted(cats):
        n, b = cats[c]
        tn += n; tb += b
        print(f"  {c[3:]:<104} {fmt_n(n):>9} tệp {fmt_b(b):>9}")
    print(f"  {BLD}{'tổng GIỮ bất khả tái tạo':<104} {fmt_n(tn):>9} tệp {fmt_b(tb):>9}{RST}")
    if not (ROOT / "dataset_out/human_audit").exists():
        print("  (dataset_out/human_audit/ hiện KHÔNG có trên đĩa — nếu xuất hiện sẽ được giữ)")
    idx = ROOT / "pipeline/align_engine/data/index.csv"
    if idx.exists():
        import csv
        with open(idx, encoding="utf-8") as f:
            paths = [r.get("path") for r in csv.DictReader(f)]
        have = sum(1 for p in paths if p and (ROOT / p).exists())
        print(f"  ghi chú S3: index.csv trỏ {fmt_n(len(paths))} ảnh, hiện còn {fmt_n(have)} -> s3_proto_cache.pkl GIỮ "
              f"(dựng lại từ crop hiện có sẽ ra bộ nguyên mẫu khác)")
    tset = {t for t, _ in T}
    print(f"\n{BLD}--- GIỮ · dẫn xuất nhưng KHÔNG xoá ở chế độ này ---{RST}")
    kept_derived = []
    if not DEEP:
        kept_derived += [("prepared/_gold_exact", "cache crop chuẩn + nhúng + hộp kim gold_exact (--deep để xoá; 0 API)"),
                         ("measure_out/_borg_human", "trung gian borg_human (--deep để xoá)")]
        kept_derived += [(f"measure_out/{b}/{s}", "đầu vào tiền xử lý của ingest (--deep để xoá)")
                         for b in LITHO for s in ("layout", "qn_ocr")]
        kept_derived.append(("measure_out/Chrestomathie1872/chresto_map", "đầu vào tiền xử lý của ingest (--deep để xoá)"))
    kept_derived.append(("logs", "nhật ký các lượt chạy (lịch sử thời gian)"))
    for r, why in kept_derived:
        p = ROOT / r
        if p.exists():
            n, b = du(p)
            print(f"  {r:<48} {why:<62} {fmt_n(n):>9} tệp {fmt_b(b):>9}")
    mo = ROOT / "measure_out"
    if mo.is_dir():
        special = tset | {r for r in prot if r.startswith("measure_out/")} | {k for k, _ in kept_derived} \
            | {f"measure_out/{x}" for x in MO_DERIVED_UNDERSCORE}
        old, n_old, b_old = [], 0, 0

        def touches(r):
            return any(under(x, r) or under(r, x) for x in special)

        for p in sorted(mo.iterdir()):
            r = rel(p)
            if p.name == ".DS_Store":
                continue
            if not touches(r):                              # cả thư mục/tệp là kết quả cũ
                n, b = du(p); n_old += n; b_old += b; old.append(p.name)
            elif p.is_dir() and r not in special:           # thư mục sách: lẫn đích xoá/giữ -> xét từng con
                for q in sorted(p.iterdir()):
                    if q.name != ".DS_Store" and not touches(rel(q)):
                        n, b = du(q); n_old += n; b_old += b; old.append(rel(q)[len("measure_out/"):])
        if old:
            print(f"  {'measure_out/… kết quả đo/thí nghiệm cũ':<48} {'--run KHÔNG sinh lại -> giữ nguyên':<62} "
                  f"{fmt_n(n_old):>9} tệp {fmt_b(b_old):>9}")
            line = ", ".join(old)
            print("      " + (line if len(line) < 400 else line[:400] + " …"))
    print(f"\n{BLD}--- SẼ XOÁ (--run sinh lại, 0 API){' + --deep' if DEEP else ''} ---{RST}")
    tot = 0
    pyc_n = pyc_b = 0
    with open(a.targets_out, "w", encoding="utf-8") as f:
        for t, why in T:
            f.write(t + "\n")
            n, b = du(ROOT / t)
            tot += b
            if why == "__pycache__":
                pyc_n += 1; pyc_b += b
                continue
            print(f"  {t:<48} {why:<62} {fmt_n(n):>9} tệp {fmt_b(b):>9}")
    if pyc_n:
        print(f"  {'**/__pycache__':<48} {f'{pyc_n} thư mục (ngoài .venv)':<62} {'':>13} {fmt_b(pyc_b):>9}")
    print(f"  {BLD}{'tổng SẼ XOÁ':<111}       {fmt_b(tot):>9}{RST}")
    old_stt = [b for b in STT_BOOKS if (ROOT / "dataset" / b).is_dir()]
    if old_stt:
        print(f"  {GRN}giữ: dataset/{{{','.join(old_stt)}}}/ — bản xuất STT theo quyển (23/09, web/server.py đọc; không bước "
              f"nào sinh lại) — nằm trong kiểm kê GIỮ, không bị xoá{RST}")
    return 0


# ---------------------------------------------------------------- kiểm kê sha256
def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def hash_entries(entries) -> dict:
    """{rel: (sha|LINK, size|target)} cho các (rel, kind, st)."""
    out, files = {}, []
    for r, kind, st in entries:
        if kind == "link":
            out[r] = ("LINK", os.readlink(ROOT / r))
        else:
            files.append((r, st.st_size))
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as ex:
        for (r, size), h in zip(files, ex.map(lambda x: sha256(str(ROOT / x[0])), files, chunksize=64)):
            out[r] = (h, str(size))
            done += 1
            if done % 50000 == 0:
                print(f"    … {fmt_n(done)}/{fmt_n(len(files))} tệp ({time.time() - t0:.0f} s)", flush=True)
    return out


def write_manifest(path: Path, M: dict, note: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"# clean_rebuild_all.sh — kiểm kê sha256 phần GIỮ · {note}\n# sha256|LINK<TAB>byte|đích<TAB>đường dẫn\n")
        for r in sorted(M):
            h, s = M[r]
            f.write(f"{h}\t{s}\t{r}\n")
    tmp.replace(path)


def read_manifest(path: Path) -> dict:
    M = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln or ln.startswith("#"):
            continue
        h, s, r = ln.split("\t", 2)
        M[r] = (h, s)
    return M


def cmd_manifest(a) -> int:
    t0 = time.time()
    M = hash_entries(walk_protected())
    write_manifest(Path(a.out), M, f"{time.strftime('%Y-%m-%dT%H:%M:%S')} · root {ROOT} · deep={int(DEEP)}")
    tb = sum(int(s) for h, s in M.values() if h != "LINK")
    print(f"  kiểm kê: {fmt_n(len(M))} mục ({fmt_b(tb)}) -> {rel(Path(a.out))} ({time.time() - t0:.0f} s)")
    return 0


def diff_manifest(old: dict, new: dict, label: str) -> int:
    miss = sorted(set(old) - set(new))
    add = sorted(set(new) - set(old))
    chg = sorted(r for r in set(old) & set(new) if old[r] != new[r])
    if not (miss or add or chg):
        print(f"  {GRN}{label}: {fmt_n(len(old))} mục — TRÙNG 100 % (không mất, không thêm, không đổi){RST}")
        return 0
    print(f"  {RED}{BLD}{label}: MẤT {len(miss)} · THÊM {len(add)} · ĐỔI {len(chg)}{RST}")
    for tag, L in (("MẤT", miss), ("THÊM", add), ("ĐỔI", chg)):
        for r in L[:15]:
            print(f"    {RED}{tag}{RST} {r}")
        if len(L) > 15:
            print(f"    … còn {len(L) - 15} mục {tag}")
    return 1


def cmd_verify(a) -> int:
    old = read_manifest(Path(a.manifest))
    new = hash_entries(walk_protected())
    return diff_manifest(old, new, "kiểm lại phần GIỮ")


def cmd_delete(a) -> int:
    import bisect
    T = [ln.strip() for ln in Path(a.targets).read_text(encoding="utf-8").splitlines() if ln.strip()]
    kept = read_manifest(Path(a.manifest))
    keys = sorted(kept)
    link_tgts = [(r, os.path.realpath(ROOT / r)) for r, (h, _) in kept.items() if h == "LINK"]
    prot = protected_roots()
    errs = 0
    for t in T:
        p = ROOT / t
        rp = os.path.realpath(p) if not p.is_symlink() else os.path.realpath(p.parent)
        if t in ("", ".", "..") or ".." in Path(t).parts or not (rp == str(ROOT) or rp.startswith(str(ROOT) + os.sep)) \
                or os.path.realpath(p) == str(ROOT):
            print(f"  {RED}BỎ (ngoài gốc/không hợp lệ): {t}{RST}"); errs += 1; continue
        if any(under(r, t) for r in prot):
            print(f"  {RED}BỎ (chứa gốc GIỮ): {t}{RST}"); errs += 1; continue
        i = bisect.bisect_left(keys, t + "/")
        hit = t if t in kept else (keys[i] if i < len(keys) and keys[i].startswith(t + "/") else None)
        if hit:
            print(f"  {RED}BỎ (chứa tệp đã kiểm kê {hit}): {t}{RST}"); errs += 1; continue
        rt = os.path.realpath(p)
        bad_link = next((r for r, tg in link_tgts if tg == rt or tg.startswith(rt + os.sep)), None)
        if bad_link:                                         # symlink trong phần GIỮ trỏ vào đích xoá -> không xoá
            print(f"  {RED}BỎ ({bad_link} là symlink trỏ vào đích): {t}{RST}"); errs += 1
        else:
            try:
                if p.is_symlink() or p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
                else:
                    continue
                if not t.endswith("__pycache__"):
                    print(f"  đã xoá {t}")
            except OSError as e:
                print(f"  {RED}LỖI xoá {t}: {e}{RST}"); errs += 1
    n_pyc = sum(1 for t in T if t.endswith("__pycache__"))
    if n_pyc:
        print(f"  đã xoá {n_pyc} thư mục __pycache__")
    return 1 if errs else 0


# ---------------------------------------------------------------- sau --run: cache gốc từ API KHÔNG được đổi
def api_derived(r: str) -> bool:
    n = r.rsplit("/", 1)[-1]
    if r.startswith("prepared/"):
        if "/kim_raw/" in r or n == "kim_calls.json":
            return True
        stt = r.split("/")[1] in STT_BOOKS
        return stt and (("/detected/" in r and n.endswith("_ocr_cache.json"))
                        or ("/transcriptions/" in r and ("_qn_ocr_cache" in n or n.endswith("_qn_tmp.png"))))
    if r.startswith("measure_out/") and not r.startswith("measure_out/_"):
        return "/kim_cache/" in r or n == "kim_calls.json" or "/ihr_layout/png/" in r
    return False


def cmd_postrun(a) -> int:
    old = {r: v for r, v in read_manifest(Path(a.manifest)).items() if api_derived(r)}
    ents = [(r, k, st) for r, k, st in walk_protected() if api_derived(r)]
    new = hash_entries(ents)
    rc = diff_manifest(old, new, "cache gốc từ API (kim_raw, kim_calls, cache OCR/QN STT, kim_cache bộ đo)")
    for b in KIM_BOOKS:
        try:
            _, cfg = book_config(b)
            mf = ROOT / ((cfg.get("paths") or {}).get("data_dir") or "prepared") / b / "manifest.json"
            g = (json.loads(mf.read_text(encoding="utf-8")).get("gates") or {})
        except Exception as e:  # noqa: BLE001
            print(f"  {YEL}{b}: không đọc được manifest ({e}){RST}")
            continue
        calls = int(g.get("ocr_calls", 0) or 0) + (int(g.get("n_pages_new_kim", 0) or 0) if b in BORG_BOOKS else 0)
        if calls:
            print(f"  {RED}{b}: manifest ghi {calls} lượt gọi kim mới{RST}"); rc = 1
    log = os.environ.get("GANNHANOCR_CAM_API_LOG")
    if log and Path(log).exists() and Path(log).stat().st_size:
        print(f"  {RED}có yêu cầu API bị CHẶN — xem {log}{RST}"); rc = 1
    if rc == 0:
        print(f"  {GRN}0 lượt API: manifest ingest ocr_calls = 0, Borg n_pages_new_kim = 0, không yêu cầu nào bị chặn{RST}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("plan"); p.add_argument("--targets-out", required=True)
    p = sub.add_parser("manifest"); p.add_argument("--out", required=True)
    p = sub.add_parser("verify"); p.add_argument("--manifest", required=True)
    p = sub.add_parser("delete"); p.add_argument("--targets", required=True); p.add_argument("--manifest", required=True)
    p = sub.add_parser("postrun"); p.add_argument("--manifest", required=True)
    a = ap.parse_args()
    return dict(check=cmd_check, plan=cmd_plan, manifest=cmd_manifest, verify=cmd_verify, delete=cmd_delete,
                postrun=cmd_postrun)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
PYEOF

H() { "$PY" "$HELPER" "$@"; }

# ------------------------------------------------------------------------------------------------------------------
banner "clean_rebuild_all · gốc $ROOT$( [[ "$ROOT" != "$CODE_DIR" ]] && printf ' (BẢN THỬ; mã: %s)' "$CODE_DIR")$( (( DEEP )) && printf ' · --deep')"
printf '%s' "$( (( DO_IT )) && printf '%sCHẾ ĐỘ THẬT (--yes)%s' "$RED$BLD" "$RST" || printf '%sCHẠY THỬ — không xoá gì%s' "$YEL$BLD" "$RST")"
(( RUN )) && printf ' + chạy lại 10 bộ (--run)'
printf '\n'

banner "1 · KIỂM CACHE OCR CỦA 10 BỘ (0 API, theo đúng khoá của mã ingest/extract)"
H check; RC_CHECK=$?
BLOCKED=0
if (( RC_CHECK != 0 )); then
  BLOCKED=1
  (( DO_IT )) && die "thiếu/lệch cache OCR (xem bảng trên) — KHÔNG xoá gì. Khôi phục cache hoặc chạy bộ thiếu RIÊNG có xác nhận API trước."
  warn "chạy thử: có mục THIẾU/LỆCH — bản --yes sẽ DỪNG CỨNG ở đây."
fi

banner "2 · GIỮ / SẼ XOÁ"
H plan --targets-out "$TARGETS" || die "lập danh sách lỗi"

if (( ! DO_IT )); then
  printf '\n%sCHẠY THỬ — chưa xoá gì.%s Chạy thật:\n' "$YEL$BLD" "$RST"
  printf '    bash scripts/clean_rebuild_all.sh --yes          # chỉ dọn (kiểm kê sha256 trước/sau)\n'
  printf '    bash scripts/clean_rebuild_all.sh --yes --run    # dọn rồi chạy lại 10 bộ + borg_human + measure + báo cáo\n'
  exit $(( BLOCKED ? 3 : 0 ))
fi

TS="$(date +%Y%m%d_%H%M%S)"
MAN="$ROOT/measure_out/_clean_manifest/$TS.txt"
banner "3 · KIỂM KÊ sha256 PHẦN GIỮ -> XOÁ -> KIỂM LẠI"
info "kiểm kê -> ${MAN#$ROOT/}"
H manifest --out "$MAN" || die "không lập được kiểm kê sha256 — KHÔNG xoá gì."
cp "$TARGETS" "${MAN%.txt}_xoa.txt" 2>/dev/null || true
H delete --targets "$TARGETS" --manifest "$MAN"; RC_DEL=$?
info "kiểm lại kiểm kê sau khi xoá"
if ! H verify --manifest "$MAN"; then
  printf '%s%sLỖI NGHIÊM TRỌNG: phần GIỮ đã khác kiểm kê (%s). DỪNG — điều tra trước khi làm gì tiếp.%s\n' \
    "$RED" "$BLD" "${MAN#$ROOT/}" "$RST" >&2
  exit 1
fi
(( RC_DEL == 0 )) || die "có đích xoá bị bỏ/lỗi (xem trên) — phần GIỮ vẫn nguyên; sửa rồi chạy lại."
ok "đã dọn dẫn xuất; phần GIỮ nguyên vẹn (kiểm kê ${MAN#$ROOT/})"

if (( ! RUN )); then
  printf '\n%sĐã dọn xong (không --run).%s Chạy lại toàn bộ:\n    ./run_pipeline.sh --book all --yes --borg require --no-verify\n' "$GRN$BLD" "$RST"
  printf '    .venv/bin/python -m pipeline.borg_human --stage all\n    .venv/bin/python scripts/measure/measure.py --all\n'
  printf '    ./run_pipeline.sh --verify\n    .venv/bin/python scripts/bao_cao_tong_hop.py\n'
  exit 0
fi

# ------------------------------------------------------------------------------------------------------------------
banner "4 · CHẠY LẠI 10 BỘ + BORG NHÃN NGƯỜI + BỘ ĐO + BÁO CÁO (API kim bị CHẶN trong mọi tiến trình con)"
mkdir -p logs
TIMING="logs/clean_rebuild_${TS}_thoi_gian.tsv"
BLOCK_LOG="$ROOT/logs/clean_rebuild_${TS}_api_bi_chan.txt"
GUARD="$TMPD/guard"; mkdir -p "$GUARD"
cat > "$GUARD/sitecustomize.py" <<'PYGUARD'
# sitecustomize TẠM của scripts/clean_rebuild_all.sh --run — CẤM gọi API kim (HCMUS) trong mọi tiến trình Python con.
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _p in list(sys.path):                      # chạy tiếp sitecustomize gốc (vd Homebrew) mà tệp này che mất
    _d = os.path.abspath(_p or os.curdir)
    _f = os.path.join(_d, "sitecustomize.py")
    if _d != _here and os.path.isfile(_f):
        try:
            with open(_f, encoding="utf-8") as _fh:
                exec(compile(_fh.read(), _f, "exec"), {"__name__": "sitecustomize", "__file__": _f})
        except Exception as _e:                # noqa: BLE001
            sys.stderr.write(f"[cam_api] sitecustomize gốc {_f} lỗi: {_e}\n")
        break

if os.environ.get("GANNHANOCR_CAM_API") == "1":
    class KimApiBiChan(BaseException):
        """BaseException: không bị `except Exception` nuốt -> tiến trình dừng, bộ đó HỎNG, 0 lượt API."""

    try:
        import requests.sessions as _rs
        from urllib.parse import urlsplit as _us
        _orig = _rs.Session.request

        def _request(self, method, url, *a, **k):
            host = (_us(str(url)).hostname or "").lower()
            dom = (os.environ.get("SN_DOMAIN") or "").strip().lower()
            if host.endswith("hcmus.edu.vn") or (dom and host == dom):
                msg = f"CẤM GỌI API kim (clean_rebuild_all --run): {method} {url} · pid {os.getpid()} · {' '.join(sys.argv)[:160]}"
                log = os.environ.get("GANNHANOCR_CAM_API_LOG")
                if log:
                    try:
                        with open(log, "a", encoding="utf-8") as fh:
                            fh.write(msg + "\n")
                    except OSError:
                        pass
                sys.stderr.write("\033[31m[CẤM API] " + msg + "\033[0m\n")
                raise KimApiBiChan(msg)
            return _orig(self, method, url, *a, **k)

        _rs.Session.request = _request
    except Exception:                          # noqa: BLE001  (không có requests -> không có gì để chặn)
        pass
PYGUARD
export GANNHANOCR_CAM_API=1 GANNHANOCR_CAM_API_LOG="$BLOCK_LOG"
export PYTHONPATH="$GUARD${PYTHONPATH:+:$PYTHONPATH}"
# Tự thử chốt chặn KHÔNG ra mạng: phiên có bộ chuyển (adapter) giả — lọt tới lớp gửi = chốt chặn hỏng -> mã 1.
if "$PY" -c 'import requests
class A(requests.adapters.BaseAdapter):
    def send(self, *a, **k): raise SystemExit(1)
    def close(self): pass
s = requests.Session(); s.mount("https://", A())
try:
    s.get("https://kimhannom.clc.hcmus.edu.vn/khong-goi-that")
except BaseException as e:
    raise SystemExit(0 if type(e).__name__ == "KimApiBiChan" else 1)
raise SystemExit(1)' 2>/dev/null; then
  : > "$BLOCK_LOG"
  ok "chốt chặn API kim đang bật (thử chặn 1 yêu cầu giả: bị chặn trước khi ra mạng)"
else
  die "không bật được chốt chặn API kim — KHÔNG chạy lại (dữ liệu đã dọn; chạy tay từng lệnh nếu chắc chắn)."
fi
printf 'buoc\tlenh\tma_thoat\tgiay\tbat_dau\tket_thuc\tlog\n' > "$TIMING"
FROZEN="$TMPD/run_pipeline.frozen.sh"
cp run_pipeline.sh "$FROZEN"     # bash đọc script theo đoạn: sửa run_pipeline.sh giữa chừng sẽ làm lệch — chạy bản đóng băng
N_FAIL=0
step() {   # step <số> <tên> <lệnh…>
  local n="$1" name="$2"; shift 2
  local lf="logs/clean_rebuild_${TS}_${n}_${name}.log" t0=$SECONDS s0 rc
  s0="$(date +%Y-%m-%dT%H:%M:%S)"
  printf '\n%s>>> [%s] %s%s\n    $ %s\n    log: %s\n' "$BLD" "$n" "$name" "$RST" "$*" "$lf"
  { printf '# clean_rebuild_all.sh --run · bước %s %s · %s · git %s\n$ %s\n' "$n" "$name" "$s0" \
      "$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')" "$*"; } > "$lf"
  "$@" 2>&1 | tee -a "$lf"
  rc=${PIPESTATUS[0]}
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$name" "$rc" "$((SECONDS - t0))" "$s0" "$(date +%Y-%m-%dT%H:%M:%S)" "$lf" >> "$TIMING"
  if (( rc == 0 )); then ok "[$n] $name: xong ($((SECONDS - t0)) s)"; else
    N_FAIL=$((N_FAIL + 1)); printf '%s[LỖI]%s [%s] %s: mã thoát %s — xem %s\n' "$RED" "$RST" "$n" "$name" "$rc" "$lf" >&2; fi
  if [[ -s "$BLOCK_LOG" ]]; then
    die "có yêu cầu API kim bị CHẶN trong bước [$n] (xem $BLOCK_LOG) — dừng chuỗi chạy lại; cache thiếu so với tập trang mới."
  fi
  return 0
}
# THỨ TỰ (sửa 2026-09-28): nghiệm thu --verify chỉ ĐỌC kết quả đo (measure.py --all --report-only) mà bước dọn đã
# xoá -> phải chạy measure.py --all TRƯỚC rồi mới nghiệm thu, nếu không code_facts/detector_transfer/box_ref báo ERR.
step 1 run_pipeline_all  env GANNHANOCR_ROOT="$ROOT" bash "$FROZEN" --book all --yes --borg require --no-verify
step 2 borg_human        "$PY" -m pipeline.borg_human --stage all
step 3 measure_all       "$PY" scripts/measure/measure.py --all
step 4 nghiem_thu        env GANNHANOCR_ROOT="$ROOT" bash "$FROZEN" --verify
step 5 bao_cao_tong_hop  "$PY" scripts/bao_cao_tong_hop.py --timing "$TIMING"

banner "5 · KIỂM SAU KHI CHẠY: 0 lượt API, cache gốc không đổi"
H postrun --manifest "$MAN"; RC_POST=$?
if git -C "$CODE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  info "dataset_out/ (bộ STT có trong git) so với HEAD — trống = tái lập TRÙNG TỪNG BYTE:"
  git -C "$CODE_DIR" diff --stat -- dataset_out/ | tail -5 | sed 's/^/    /'
fi
printf '\n'
column -t -s $'\t' "$TIMING" 2>/dev/null | sed 's/^/    /' || cat "$TIMING"
if (( N_FAIL || RC_POST )); then
  printf '%s%s[LỖI] %s bước lỗi · kiểm sau chạy %s — xem logs/clean_rebuild_%s_*%s\n' "$RED" "$BLD" "$N_FAIL" \
    "$( (( RC_POST )) && echo FAIL || echo PASS)" "$TS" "$RST" >&2
  exit 1
fi
ok "XONG: dọn + chạy lại 10 bộ + borg_human + measure + báo cáo · kiểm sau chạy PASS · ${SECONDS} s"
exit 0
