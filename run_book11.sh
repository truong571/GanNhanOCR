#!/usr/bin/env bash
# Chạy pipeline CHỈ cho SachThanhTruyen11.
#   ./run_book11.sh            # GIỮ cache (mặc định, nhanh)
#   ./run_book11.sh --fresh    # XOÁ cache cũ, chạy lại từ đầu
#   ./run_book11.sh --keep --qwen --qwen-n 5
# Mọi option forward sang run_book.sh (--step/--config/--book/--fresh/--keep/--qwen...).
exec "$(dirname "$0")/run_book.sh" SachThanhTruyen11 "$@"
