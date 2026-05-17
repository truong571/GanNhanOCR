# test_llm_postfix

POC: dùng Gemini Free API để fix lỗi context-OCR (răng vs rằng, chăng vs chẳng,...)
mà dict-lookup không catch được. Chỉ gửi **unique bigram pairs** chứ không phải
toàn trang → cực rẻ, ~10k tokens 1 lần cho toàn dataset.

## Workflow

```bash
# 1. Tìm suspect bigrams từ transcriptions hiện có
.venv/bin/python evaluation/test_llm_postfix/bigram_finder.py

# 2a. Xem prompt sẽ gửi (không cần API key)
.venv/bin/python evaluation/test_llm_postfix/llm_fixer.py --dry-run

# 2b. Gửi thật lên Gemini (cần GEMINI_API_KEY trong .env)
.venv/bin/python evaluation/test_llm_postfix/llm_fixer.py

# 3a. Xem fix sẽ áp dụng
.venv/bin/python evaluation/test_llm_postfix/apply_fixes.py --dry-run

# 3b. Apply (auto backup .pre_fix)
.venv/bin/python evaluation/test_llm_postfix/apply_fixes.py

# 3c. Revert nếu hối hận
.venv/bin/python evaluation/test_llm_postfix/apply_fixes.py --restore
```

## Lấy free Gemini API key

1. https://aistudio.google.com/app/apikey (cần Google account, không cần thẻ)
2. Tạo key
3. Thêm vào `.env`:
   ```
   GEMINI_API_KEY=AIza...
   ```

Free tier: 15 RPM, 1500 req/day, 1M TPM. Đủ làm POC + production.

## Outputs

`out/`:
- `suspects.json`     — bigrams từ bigram_finder
- `prompt_sent.txt`   — prompt LLM thấy
- `fix_map.json`      — sau khi LLM verify, danh sách fix confirmed

## Safety

- Mọi fix đều CHỈ apply khi `prev` từ đầy đủ khớp → không bao giờ over-correct ở vị trí khác
- `apply_fixes.py` auto backup `.pre_fix`; có `--restore` để revert
- LLM bị constrain bằng prompt + JSON schema, không sinh text dài tự do
