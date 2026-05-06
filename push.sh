#!/bin/bash
# Push code to GitHub.
# Usage:  ./push.sh "commit message"
#         ./push.sh                    # uses default "update code"
#
# Files matching *.png, *.xlsx, *.pdf, prepared/, dataset/ are ignored
# via .gitignore — they will never be staged even with `git add -A`.

set -e

MSG="${1:-update code}"

git add -A
echo ""
echo "=== Files staged ==="
git diff --cached --name-status
echo ""

# Refuse to commit any *.png / *.xlsx that slipped through (e.g. forced add).
LEAKED=$(git diff --cached --name-only | grep -E '\.(png|xlsx)$' || true)
if [[ -n "$LEAKED" ]]; then
    echo "ERROR: refusing to push — these tracked files match png/xlsx blocklist:"
    echo "$LEAKED"
    echo "Untrack them with:  git rm --cached <file>"
    exit 1
fi

if git diff --cached --quiet; then
    echo "Nothing to commit."
    exit 0
fi

echo "Committing: $MSG"
git commit -m "$MSG"
git push origin main
echo ""
echo "Done."
