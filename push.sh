#!/bin/bash
# Push code to GitHub
# Usage: ./push.sh "commit message"

set -e

git add -A
git status
echo ""
echo "Committing: update code"
git commit -m "update code"
git push origin main
echo ""
echo "Done."
