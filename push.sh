#!/bin/bash
# Đẩy toàn bộ mã nguồn + tài liệu + bằng chứng lên GitHub, để không có thứ gì
# quan trọng chỉ nằm trên một cái máy.
#
# Dùng:
#   ./push.sh "thông điệp commit"
#   ./push.sh                      # mặc định "update code"
#   ./push.sh -n                   # chạy thử: chỉ xem sẽ đẩy gì, KHÔNG commit
#
# Script làm 5 việc, dừng ngay nếu có việc nào hỏng:
#   1. Chặn file rác / file quá nặng lọt vào commit
#   2. Cảnh báo submodule có thay đổi chưa đẩy (đây là kiểu MẤT CODE hay gặp nhất:
#      repo cha trỏ vào một commit mà submodule chưa push -> máy khác clone về hỏng)
#   3. Commit + push nhánh đang đứng
#   4. Xác nhận nhánh trên GitHub đã bắt kịp máy
#   5. Liệt kê những thứ CHỈ CÒN trên máy này (bị .gitignore chặn) để biết mà sao lưu
#
# Không commit: *.png *.jpg *.pdf *.xlsx *.zip *.pt, dataset/, prepared/,
# dataset_out/{gold,silver,syllable,review}/, HTML công cụ chấm, release/.
# Xem .gitignore để biết đầy đủ.

set -euo pipefail

export GIT_PAGER=cat PAGER=cat LESS=FRX
unset GIT_EXTERNAL_DIFF

GIT=$(command -v git)                      # bỏ qua alias/function tên `git`
MAX_MB=45                                  # GitHub cảnh báo từ 50 MB, chặn ở 100 MB

DRY=0
if [[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]]; then DRY=1; shift; fi
MSG="${1:-update code}"

cd "$(dirname "$0")"

hr() { printf '%s\n' "------------------------------------------------------------"; }

# ---- 2/5 submodule: kiểm TRƯỚC khi commit repo cha ------------------------
# Repo cha chỉ lưu một con trỏ commit. Nếu submodule có thay đổi chưa commit/chưa
# push thì con trỏ đó vô nghĩa với người clone về — và code trong submodule chỉ
# còn tồn tại trên máy này. Không tự động commit hộ vì trong đó thường là
# checkpoint nặng (nom-embed đẩy lên HuggingFace, không phải GitHub).
SUB_WARN=0
if [[ -f .gitmodules ]]; then
    hr; echo "SUBMODULE"
    while read -r name path; do
        [[ -d "$path/.git" || -f "$path/.git" ]] || { printf '  %-22s chưa init (bỏ qua)\n' "$path"; continue; }
        # Tôn trọng `ignore = dirty|all` trong .gitmodules — vd gannhanocr-fd là kho
        # cache glyph ~90k ảnh tải từ HuggingFace, luôn "bẩn" và không cần đẩy.
        ign=$("$GIT" config --file .gitmodules --get "submodule.$name.ignore" 2>/dev/null || true)
        if [[ "$ign" == "dirty" || "$ign" == "all" ]]; then
            printf '  %-22s bỏ qua theo .gitmodules (ignore = %s)\n' "$path" "$ign"
            continue
        fi
        dirty=$("$GIT" -C "$path" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
        ahead=$("$GIT" -C "$path" log --oneline @{u}..HEAD 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$dirty" != "0" || "$ahead" != "0" ]]; then
            SUB_WARN=1
            printf '  %-22s ⚠ %s file đổi, %s commit chưa đẩy\n' "$path" "$dirty" "$ahead"
            printf '     đẩy bằng:  git -C %s add -A && git -C %s commit -m "%s" && git -C %s push\n' \
                   "$path" "$path" "$MSG" "$path"
        else
            printf '  %-22s ok\n' "$path"
        fi
    done < <("$GIT" config --file .gitmodules --get-regexp '^submodule\..*\.path$' \
             | sed -E 's/^submodule\.(.*)\.path (.*)$/\1 \2/')
fi

# ---- 1/5 staging + chặn file không được phép -----------------------------
"$GIT" add -A

hr; echo "SẼ COMMIT"
"$GIT" --no-pager diff --cached --name-status | cat
CHANGED=$("$GIT" --no-pager diff --cached --name-only | wc -l | tr -d ' ')
echo "  ($CHANGED file)"

BLOCK=$("$GIT" --no-pager diff --cached --name-only --diff-filter=d \
        | grep -Ei '\.(png|jpe?g|pdf|xlsx|zip|pt|ckpt)$' || true)
if [[ -n "$BLOCK" ]]; then
    hr
    echo "DỪNG: các file này thuộc loại không được commit (ảnh/checkpoint/gói nén):"
    echo "$BLOCK" | sed 's/^/  /'
    echo "  Gỡ khỏi git bằng:  git rm --cached <file>"
    exit 1
fi

BIG=""
while read -r f; do
    [[ -z "$f" || ! -f "$f" ]] && continue
    sz=$(( $(wc -c < "$f") / 1048576 ))
    (( sz >= MAX_MB )) && BIG+="  ${sz} MB  $f"$'\n'
done < <("$GIT" --no-pager diff --cached --name-only --diff-filter=d)
if [[ -n "$BIG" ]]; then
    hr
    echo "DỪNG: file vượt ${MAX_MB} MB — đẩy lên sẽ làm phình repo vĩnh viễn:"
    printf '%s' "$BIG"
    echo "  Cách xử lý: thêm vào .gitignore, hoặc đưa lên HuggingFace/Kaggle rồi tải về khi cần."
    exit 1
fi

if "$GIT" diff --cached --quiet; then
    hr; echo "Không có gì để commit."
    [[ "$SUB_WARN" == "1" ]] && echo "Nhưng SUBMODULE vẫn còn thay đổi chưa đẩy — xem cảnh báo ở trên."
    exit 0
fi

if [[ "$DRY" == "1" ]]; then
    hr; echo "CHẠY THỬ — không commit, không push. Bỏ cờ -n để đẩy thật."
    "$GIT" reset -q
    exit 0
fi

# ---- 3/5 commit + push ----------------------------------------------------
BRANCH=$("$GIT" rev-parse --abbrev-ref HEAD)
hr; echo "COMMIT: $MSG   (nhánh $BRANCH)"
"$GIT" commit -q -m "$MSG"
# --recurse-submodules=check: từ chối đẩy nếu repo cha trỏ vào commit submodule
# chưa có trên remote — chặn đúng kiểu hỏng "clone về thiếu code".
"$GIT" push --recurse-submodules=check -u origin "$BRANCH"

# ---- 4/5 xác nhận GitHub đã bắt kịp --------------------------------------
"$GIT" fetch -q origin "$BRANCH"
BEHIND=$("$GIT" log --oneline "origin/$BRANCH..HEAD" | wc -l | tr -d ' ')
hr
if [[ "$BEHIND" == "0" ]]; then
    echo "✓ GitHub đã có đủ: origin/$BRANCH == máy này ($("$GIT" rev-parse --short HEAD))"
else
    echo "✗ CÒN $BEHIND commit chưa lên GitHub — chạy lại hoặc kiểm tra mạng/quyền."
    exit 1
fi

# ---- 5/5 những gì CHỈ còn trên máy này ------------------------------------
hr
echo "CHỈ CÓ TRÊN MÁY NÀY (git không giữ — mất máy là mất)"
for p in dataset dataset_out/gold dataset_out/silver dataset_out/syllable \
         prepared nom-embed report_truong.xlsx; do
    [[ -e "$p" ]] || continue
    printf '  %-26s %s\n' "$p" "$(du -sh "$p" 2>/dev/null | cut -f1)"
done
echo "  → ảnh crop và checkpoint dựng lại được bằng run_pipeline.sh / tải từ HuggingFace."
echo "  → report_truong.xlsx thì KHÔNG dựng lại được: nhớ sao lưu tay (Drive/OneDrive)."
hr
echo "Xong."
