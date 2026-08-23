#!/usr/bin/env bash
# =============================================================================
# check_evidence.sh — đối chiếu bảng sha256 trong docs/EVIDENCE_INDEX.md với đĩa.
#
# VÌ SAO CÓ TỆP NÀY
#   Kiểm ngày 2026-08-22: 29 mục hash trong EVIDENCE_INDEX -> khớp 6 · lệch 16 ·
#   thiếu tệp 7. Nhưng phải TÁCH BẠCH hai loại:
#     - khối "## Lần chạy ..."  = NHẬT KÝ lịch sử; lệch ở đó là ĐÚNG THIẾT KẾ
#     - khối "BẢN HIỆN HÀNH"    = trạng thái hiện tại; lệch ở đây là LỖI THẬT
#   Gộp hai loại vào một con số là cách chắc chắn nhất để mất niềm tin vào cả tệp.
#
#   Chạy:  bash scripts/check_evidence.sh          (0 = khối hiện hành khớp)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

EVIDENCE="docs/EVIDENCE_INDEX.md"
[[ -f "$EVIDENCE" ]] || { echo "không thấy $EVIDENCE"; exit 2; }

if command -v shasum >/dev/null 2>&1; then SHA="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"
else echo "không có shasum/sha256sum"; exit 2; fi

if ! grep -qF '<!-- HIEN_HANH:START -->' "$EVIDENCE"; then
  echo "CHƯA CÓ khối BẢN HIỆN HÀNH trong $EVIDENCE."
  echo "  -> chạy ./run_pipeline.sh (bước cuối gọi evidence()) để sinh."
  exit 1
fi

ok=0; bad=0; missing=0
while IFS= read -r line; do
  f=$(sed -n 's/^| `\([^`]*\)` | `.*`.*/\1/p' <<<"$line")
  h=$(sed -n 's/^| `[^`]*` | `\([^`]*\)`.*/\1/p' <<<"$line")
  [[ -n "$f" && -n "$h" ]] || continue
  if [[ "$h" == "(chưa có)" ]]; then
    printf '  %-46s %s\n' "$f" "ghi (chưa có)"; ((missing++)); continue
  fi
  if [[ ! -f "$f" ]]; then
    printf '  %-46s %s\n' "$f" "THIẾU TỆP"; ((missing++)); continue
  fi
  real=$($SHA "$f" | awk '{print $1}')
  if [[ "$real" == "$h" ]]; then printf '  %-46s KHỚP\n' "$f"; ((ok++))
  else printf '  %-46s LỆCH (ghi %s… / thật %s…)\n' "$f" "${h:0:12}" "${real:0:12}"; ((bad++)); fi
done < <(awk '/<!-- HIEN_HANH:START -->/{p=1} /<!-- HIEN_HANH:END -->/{p=0} p' "$EVIDENCE")

echo "----------------------------------------------------------------"
echo "BẢN HIỆN HÀNH: khớp $ok · lệch $bad · thiếu $missing"
if (( bad || missing )); then
  echo "  -> bộ nhãn trên đĩa KHÔNG còn ứng với bảng bằng chứng."
  echo "     Chạy lại pipeline, hoặc khôi phục tệp, rồi kiểm lại."
  exit 1
fi
echo "  -> bộ đem đo = bộ đem nộp (theo bảng hiện hành)."
