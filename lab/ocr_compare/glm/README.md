# GLM-OCR trên chữ Nôm chép tay — so sánh với kinhhannom (giao thức chung lab/ocr_compare/mau)

## Engine
- GLM-OCR (zai-org/GLM-OCR): VLM 0,9B = CogViT + connector + GLM-0.5B decoder. Trọng số Mac: `mlx-community/GLM-OCR-bf16`
  (sha 24f15402, 2,2 GB, tải 76 s từ HF). Chạy qua **mlx-vlm 0.7.0** (mlx 0.32.2), Python 3.10.20, macOS 26.6.2, Apple M4 16 GB.
- SDK `glmocr` 0.1.5 chỉ là pipeline tài liệu (layout PP-DocLayoutV3 + gọi mô hình qua HTTP với prompt cố định
  `"Text Recognition:"`, temperature 0, top_k 1, repetition_penalty 1.1, min_pixels 112×112). Không có API prompt tuỳ ý
  → tôi gọi thẳng mô hình qua mlx-vlm với cùng tham số lấy mẫu, và tự nhắc (prompt).
- Tài liệu (README GitHub + HF card): OmniDocBench V1.5 94,62; "8 ngôn ngữ" (zh en fr es ru de ja ko theo tag HF);
  KHÔNG có bất kỳ số liệu nào về chữ viết tay, cổ tịch, chữ dọc, hay Nôm. Layout SDK có nhãn `vertical_text`.

## Cài đặt (venv riêng, không đụng .venv của kho)
```
/opt/homebrew/bin/python3.10 -m venv /tmp/venv_glm
# pip PyPI ~100 kB/s trên máy này → tải wheel mlx_metal (64 MB) bằng curl rồi cài; mlx-vlm --no-deps để bỏ opencv/pymupdf/mlx-audio/fastapi không dùng
/tmp/venv_glm/bin/pip install mlx_metal-0.32.2-*.whl mlx==0.32.2
/tmp/venv_glm/bin/pip install --no-deps mlx-vlm==0.7.0
/tmp/venv_glm/bin/pip install "transformers>=5.14.0" numpy pillow requests tqdm huggingface_hub pandas jinja2 sentencepiece
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/GLM-OCR-bf16')"
```
Lỗi gặp và vá: mlx-vlm 0.7.0 `wired_limit()` đọc `mx.device_info()['max_recommended_working_set_size']` → KeyError khi
default device = CPU; vá bằng context manager no-op (run_glm.py). `import mlx_vlm.generate.common` phải qua importlib
vì `mlx_vlm.generate` bị rebind thành hàm.

## Thiết bị
CPU nhanh ngang GPU khi máy rảnh (0,87 s/ô scale 3). Nhưng lúc chạy chính thức máy đang đầy tải (paddle + 5 tiến trình
huấn luyện khác, load 11-14, swap 11,6/12 GB, RAM trống 21 %) → CPU đo 40-97 s/ô, bất khả thi. **Bản chạy chính thức dùng GPU
Metal (mlx), một tiến trình tuần tự** — nói rõ vì đề bài ưu tiên CPU.

## Phát hiện cấu trúc (không cần chạy)
- Tokenizer 59.282 token, thiên giản thể: chỉ **8,4 %** từ vựng Nôm của Dict/QuocNgu_SinoNom.csv (3.504/41.502) là 1 token;
  㝵 = 3 byte-token, 𠊚 𠊛 𠬠 𡗶 𣈜 = 4 byte-token; ngay phồn thể 沒 cũng 2 token. 74,5 % ô kinhhannom trong corpus là chữ 1 token.
  → mô hình phải "đánh vần" byte UTF-8 cho hầu hết chữ Nôm riêng, gần như không thể phát ra chúng.
- Chữ Nôm 𠊚 in bằng font Plangothic, rõ nét → GLM-OCR đọc thành 得 (không biết chữ).
- Chữ Hán in rõ (Songti) → đọc đúng (天地玄黃宇宙洪荒 → thiếu 黃), cả ngang lẫn dọc → port mlx hoạt động.

## Tệp
- `run_glm.py` — chạy M1-M4 với prompt a/b/c/col × chế độ nat/force (xem docstring). `out/<set>_<prompt>-<mode>-s<scale>.csv`
- `run_glm_col.py` — thí nghiệm bổ sung: đọc nguyên cột chứa ô, căn Levenshtein với kinhhannom → `out/<set>_col-ctx.csv`
- `score_glm.py` — chấm theo giao thức (cùng logic paddle/score.py + chi phí "bịa" + M4) → `scores.json`, `out/*_scored.csv`
- `run_all.sh`, `logs/` — lệnh và log chạy. `build_mau_khong_dung.py` không còn (đã chuyển về mau/build_mau.py).

## Kết quả (chi tiết: ket_qua.md, scores.json, out/*_scored.csv)
Đối chứng ngẫu nhiên M2: P(chữ ngẫu nhiên ∈ R(âm)) = 0,06 % (đều trên 41.502 chữ) / 0,64 % (theo tần suất kinhhannom) / 0,17 % (hoán vị chữ kinhhannom).

| | M1 đối chứng (n=600) | M2 khối trượt (n=600) — CỨU ĐƯỢC | M3 㝵/người (n=300) |
|---|---|---|---|
| a "Text Recognition:" tự nhiên | ∈R 5,7 %, =kinh 4,8 %, rỗng 50 % | ∈R 0,3 % (2/600, lift 0,5×) | 𠊚 0 %, 㝵 0 %, rỗng 60 % |
| a ép-không-EOS (scale 1) | ∈R 12,2 %, =kinh 11,2 % | ∈R 1,2 % (7/600, lift 1,8×) | 𠊚 0 %, 㝵 0 %; ra 一/答/峨 |
| b nói rõ 1 chữ Nôm, tự nhiên | ∈R 2,2 %, rỗng 91 % | ∈R 0 % (0/600), rỗng 91 % | 𠊚 0 %, rỗng 95 % |
| b ép | ∈R 4,3 % | ∈R 0,2 % (1/600) | 0 %; ra 越/峙 (chép prompt) |
| c ĐÓNG (cho R(âm)), tự nhiên | rỗng 78 % | "∈R" 6,8 % nhưng 0/11 trùng nhãn, cùng âm → cùng chữ bất kể ảnh | rỗng 91 % |
| c ĐÓNG, ép | "∈R" 54 %, =kinh 2,8 % | "∈R" 65 % nhưng trùng nhãn 1/129 (ngẫu nhiên 6,3 %) | 𫴮 45 %, 𠊚 23 %, 𠊛 14 % — ĐỐI CHỨNG ảnh không-phải-người cho 𫴮 52 %, 𠊚 30 %, 𠊛 14 % → độc lập với ảnh |
| **col-ctx (đọc nguyên cột, căn với kinhhannom)** | ∈R 60,2 %, =kinh 56,2 % | **∈R 10,3 % (62/600, lift 16×)**; SILVER 36/219 (34/36 trùng nhãn s2∩s3), SYLLABLE 26/381 | 𠊚 0 %, 㝵 0 %; đọc thành 寻 47 %, 导 14 % |

M4 (20 cột dọc): số chữ GLM/kinhhannom = 0,91 (prompt gốc) – 0,96 (prompt cột); đếm đúng 4/20 – 9/20, lệch ≤2: 13/20 – 18/20; CER so kinhhannom ≈ 0,61-0,62; 1,5-6 s/cột.
Chi phí "bịa" (crop đơn, a+b tự nhiên, n=3000): rỗng 71,8 %, rác markdown ``` 14,0 %, có Latin 12,9 %, >1 CJK 0,7 %, đúng 1 CJK 7,2 %.
Ép đọc (n=3000): 23,4 % ký tự Avestan/Thái/Khmer/Bengali, 19,6 % "---", đúng 1 CJK 29,6 %.
Trên M1: chữ kinhhannom là 1 token GLM → trùng 13,8 % (a-force), nhiều token → 0,8 % (1/120).
Thời gian (GPU Metal, máy đầy tải): 0,3-1,1 s/crop đơn; 1,2-1,7 s/cột (~21 chữ) → ~0,07 s/chữ khi đọc cột.
