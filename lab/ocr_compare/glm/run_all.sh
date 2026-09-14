#!/bin/zsh
# Chạy toàn bộ giao thức, MỘT tiến trình tuần tự, GPU Metal (CPU đo 90 s/ô khi máy đầy tải/swap 11,6 GB → không khả thi).
# Thứ tự: M2 (phép đo chính) → M3 → M1 → M4 → ngữ cảnh cột.
G=/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/glm
PY=/tmp/venv_glm/bin/python
cd /tmp
for S in M2 M3 M1; do
  $PY $G/run_glm.py --device gpu --set $S --prompt a --mode nat   --scale 3
  $PY $G/run_glm.py --device gpu --set $S --prompt a --mode force --scale 1
  $PY $G/run_glm.py --device gpu --set $S --prompt b --mode nat   --scale 3
  $PY $G/run_glm.py --device gpu --set $S --prompt b --mode force --scale 3
  $PY $G/run_glm.py --device gpu --set $S --prompt c --mode nat   --scale 3
  $PY $G/run_glm.py --device gpu --set $S --prompt c --mode force --scale 3
done
$PY $G/run_glm.py --device gpu --set M4 --prompt a   --mode nat --scale 1
$PY $G/run_glm.py --device gpu --set M4 --prompt col --mode nat --scale 1
for S in M2 M3 M1; do $PY $G/run_glm_col.py --device gpu --set $S; done
echo "ALL DONE"
