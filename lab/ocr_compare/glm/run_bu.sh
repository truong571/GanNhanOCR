#!/bin/zsh
# Chạy bù các cấu hình bị GPU Timeout cắt (resume) + đối chứng thiên vị vị trí cho prompt c.
G=/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/glm
PY=/tmp/venv_glm/bin/python
cd /tmp
for i in 1 2 3; do  # lặp tối đa 3 lần vì mỗi lần GPU lỗi liên tục tiến trình phải khởi động lại
  $PY $G/run_glm.py --device gpu --set M2 --prompt a --mode nat   --scale 3
  $PY $G/run_glm.py --device gpu --set M2 --prompt c --mode force --scale 3
  $PY $G/run_glm.py --device gpu --set M3 --prompt a --mode nat   --scale 3
  $PY $G/run_glm.py --device gpu --set M1 --prompt a --mode nat   --scale 3
  $PY $G/run_glm.py --device gpu --set M1 --prompt a --mode force --scale 1
  $PY $G/run_glm.py --device gpu --set M1 --prompt b --mode nat   --scale 3
  $PY $G/run_glm.py --device gpu --set M1 --prompt b --mode force --scale 3
  $PY $G/run_glm.py --device gpu --set M1 --prompt c --mode nat   --scale 3
  $PY $G/run_glm.py --device gpu --set M1 --prompt c --mode force --scale 3
done
# ĐỐI CHỨNG: danh sách R(người) trên 200 crop M1 (không phải 'người'), ép đọc — đo thiên vị vị trí
$PY $G/run_glm.py --device gpu --set M1 --prompt c --mode force --scale 3 --fixed-syl người --limit 200
echo "BU DONE"
