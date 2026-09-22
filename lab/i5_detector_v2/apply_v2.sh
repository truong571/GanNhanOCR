#!/usr/bin/env bash
# apply_v2.sh <best.pt> [--no-build] [--no-eval] [--pages 27] [--resize area|linear] [--ckpt-only]
#
# Đưa detector v2 (tải từ Kaggle) vào pipeline sách mới và ĐO LẠI, không sửa run_pipeline.sh, không đụng STT/dataset_out:
#   1. kiểm best.pt (i5v2.decode.load_ckpt) → copy train_crop/detector_r34_v2_litho.pt
#   2. ghi books[].detector_ckpt + detector_resize vào config 3 sách mới (LVT, KVK chính + _b1 (run_config), Chresto)
#   3. scripts/measure/box_ref_eval.py 3 lần: v1 linear (mốc), v1 area (chỉ đổi phép thu ảnh), v2 area → bảng so sánh
#   4. ./run_pipeline.sh --book all-new --suffix _v2 → compare_builds.py (tier, GOLD ảnh, I5, khớp dị bản) — bỏ bằng --no-build
#   5. in hướng dẫn quyết định giữ / hoàn nguyên
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LAB=lab/i5_detector_v2
DST=train_crop/detector_r34_v2_litho.pt
PAGES=27; RESIZE=area; BUILD=1; EVAL=1; CKPT_ONLY=0
BEST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) BUILD=0; shift ;;
    --no-eval)  EVAL=0; shift ;;
    --ckpt-only) CKPT_ONLY=1; BUILD=0; EVAL=0; shift ;;
    --pages)    PAGES="$2"; shift 2 ;;
    --resize)   RESIZE="$2"; shift 2 ;;
    -h|--help)  sed -n 2,12p "$0"; exit 0 ;;
    *)          BEST="$1"; shift ;;
  esac
done
[[ -n "$BEST" && -f "$BEST" ]] || { echo "cần đường dẫn best.pt (tải từ Kaggle /kaggle/working/best.pt)"; exit 1; }

echo "== 1) kiểm $BEST"
$PY - "$BEST" <<'EOF'
import sys; sys.path.insert(0, "lab/i5_detector_v2")
from i5v2.decode import load_ckpt
d = load_ckpt(sys.argv[1])
assert "model" in d and d.get("arch") == "resnet34_fpn", "không phải ckpt CenterNet r34"
v, b = d.get("val") or {}, d.get("base_v1") or {}
print(f"  v2 epoch {d.get('epoch')} | img {d.get('img')} | use_dcn {d.get('use_dcn')} | keys {len(d['model'])}")
for k in ("litho_ok50", "litho_tiers_eq_015", "litho_cut", "stt_F1_020", "chresto_tiers_eq_015"):
    print(f"  {k:22s} v1 {b.get(k)}  ->  v2 {v.get(k)}")
if v.get("stt_F1_020") is not None and b.get("stt_F1_020") is not None and v["stt_F1_020"] < b["stt_F1_020"] - 0.01:
    raise SystemExit("  STT F1 v2 < v1 − 0,01 trong ckpt: KHÔNG áp dụng (guard).")
EOF
cp -f "$BEST" "$DST"; echo "  -> $DST ($(du -h "$DST" | cut -f1))"

echo "== 2) ghi detector_ckpt/detector_resize vào config sách mới"
for pair in "config/pipeline_LucVanTien1883.yaml:LucVanTien1883" "config/pipeline_KimVanKieu1884.yaml:KimVanKieu1884" \
            "config/pipeline_KimVanKieu1884_b1.yaml:KimVanKieu1884" "config/pipeline_Chrestomathie1872.yaml:Chrestomathie1872"; do
  cfg="${pair%%:*}"; book="${pair##*:}"
  [[ -f "$cfg" ]] || continue
  $PY $LAB/set_book_keys.py "$cfg" --book "$book" --set "detector_ckpt=$DST" --set "detector_resize=$RESIZE"
done
$PY -m pipeline.align_engine.book_layout_selftest 2>&1 | tail -1
(( CKPT_ONLY )) && { echo "--ckpt-only: xong (chưa đo)."; exit 0; }

if (( EVAL )); then
  echo "== 3) box_ref_eval trên ảnh gốc, $PAGES trang/sách (≈ 80 s mỗi lần)"
  [[ -f measure_out/box_ref_v1/summary.json ]] || $PY scripts/measure/box_ref_eval.py --book all --pages "$PAGES" --out measure_out/box_ref_v1 2>&1 | tail -3
  $PY scripts/measure/box_ref_eval.py --book all --pages "$PAGES" --resize "$RESIZE" --out measure_out/box_ref_v1_area 2>&1 | tail -3
  $PY scripts/measure/box_ref_eval.py --book all --pages "$PAGES" --ckpt "$DST" --resize "$RESIZE" --out measure_out/box_ref_v2 2>&1 | tail -3
  $PY $LAB/box_ref_compare.py measure_out/box_ref_v1 measure_out/box_ref_v1_area measure_out/box_ref_v2
fi

if (( BUILD )); then
  echo "== 4) run_pipeline --book all-new --suffix _v2 (dài: LVT ~1 h, KVK ~1,5 h, Chresto ~30 phút; log logs/)"
  ./run_pipeline.sh --book all-new --suffix _v2 --skip-ingest
  $PY $LAB/compare_builds.py --books LucVanTien1883 KimVanKieu1884 Chrestomathie1872 --suffix _v2
fi

cat <<EOF
== 5) QUYẾT ĐỊNH
  GIỮ v2 khi (bảng box_ref): I5 n_det==N và ok50 của v2 ≥ v1_area, cắt-thân-chữ giảm, và (compare_builds) GOLD ảnh /
  khớp dị bản không giảm > 0,5 điểm, box_source ink_cut+detector_low giảm. Khi đó bản chốt = dataset_out_<Book>_v2.
  HOÀN NGUYÊN ckpt (giữ resize area nếu v1_area tốt hơn v1):
    for c in config/pipeline_LucVanTien1883.yaml config/pipeline_KimVanKieu1884.yaml config/pipeline_KimVanKieu1884_b1.yaml \\
             config/pipeline_Chrestomathie1872.yaml; do $PY $LAB/set_book_keys.py \$c --book <Book> --remove detector_ckpt; done
  HOÀN NGUYÊN hẳn v1: thêm --remove detector_resize. STT không bị đụng trong mọi trường hợp (config/pipeline.yaml không đổi).
EOF
