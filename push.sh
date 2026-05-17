#!/bin/bash
# Push code to GitHub.
# Usage:  ./push.sh "commit message"
#         ./push.sh                    # uses default "update code"
#
# Files matching *.png, *.xlsx, *.pdf, prepared/, dataset/ are ignored
# via .gitignore — they will never be staged even with `git add -A`.

set -e

# Disable any pager — some shells alias git to auto-paginate.
export GIT_PAGER=cat
export PAGER=cat
export LESS=FRX
unset GIT_EXTERNAL_DIFF

MSG="${1:-update code}"

# Use raw git path to bypass any shell alias/function called `git`.
GIT=$(command -v git)

"$GIT" add -A
echo ""
echo "=== Files staged ==="
"$GIT" --no-pager diff --cached --name-status | cat
echo ""

# Refuse to commit any *.png / *.xlsx that slipped through (e.g. forced add).
LEAKED=$("$GIT" --no-pager diff --cached --name-only | grep -E '\.(png|xlsx)$' || true)
if [[ -n "$LEAKED" ]]; then
    echo "ERROR: refusing to push — these tracked files match png/xlsx blocklist:"
    echo "$LEAKED"
    echo "Untrack them with:  git rm --cached <file>"
    exit 1
fi

if "$GIT" diff --cached --quiet; then
    echo "Nothing to commit."
    exit 0
fi

echo "Committing: $MSG"
"$GIT" commit -m "$MSG"
"$GIT" push origin main
echo ""
echo "Done."
