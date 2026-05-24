# Ensemble Kim OCR (full-page + per-col) — kết quả

Tổng cặp xét: **373**

## 1. Mức đồng ý giữa hai bản đọc

- Cặp có cả 2 OCR: **366**
- Đồng ý (cùng ký tự): **259**  (70.8%)
- Bất đồng: **107**  (29.2%)

## 2. Tier-1 dict match (sau khi đổi gợi ý OCR)

| Nguồn ocr_char | Số cặp Tier-1 dict match | Δ so với full |
|---|---:|---:|
| Full-page (hiện tại) | 216 | — |
| Per-col only         | 204 | -12 |
| **Ensemble**         | **228** | **+12** |

## 3. Promotion / save count

- Promote Tier 2/3 → Tier 1 (nhờ ensemble): **12**
- Per-col cứu được (full không tra ra dict): **12**
- Full cứu được (per-col không tra ra dict): **24**
- Cả 2 cùng tra ra dict nhưng nom khác nhau: **4**

## 4. Nguồn ensemble (source breakdown)

| source | n |
|---|---:|
| agree | 259 |
| disagree_no_dict | 68 |
| prefer_full_dict | 24 |
| prefer_per_dict | 12 |
| both_match_diff_nom | 4 |
| only_per | 4 |
| none | 1 |
| only_full | 1 |

## 5. Promotion list (Tier 2/3 → Tier 1 nhờ ensemble)

| page | col | idx | syl | full_ocr | per_ocr | ens_ocr | source | nom (mới) |
|------|----:|----:|-----|---------|---------|---------|--------|-----------|
| page_0012 | 4 | 20 | An | 女 | 安 | 安 | prefer_per_dict | 安 |
| page_0012 | 5 | 1 | ki | 冥 | 箕 | 箕 | prefer_per_dict | 箕 |
| page_0012 | 6 | 2 | thì | 詩 | 寺 | 寺 | prefer_per_dict | 寺 |
| page_0012 | 6 | 15 | răng | 𫝕 | 浪 | 浪 | prefer_per_dict | 浪 |
| page_0012 | 7 | 12 | bụt | 季 | 孛 | 孛 | prefer_per_dict | 孛 |
| page_0014 | 3 | 5 | Nói | 肉 | 内 | 内 | prefer_per_dict | 内 |
| page_0014 | 5 | 14 | một | 民 | 蔑 | 蔑 | prefer_per_dict | 蔑 |
| page_0014 | 5 | 15 | ngày | 呼 | 時 | 時 | prefer_per_dict | 時 |
| page_0014 | 6 | 12 | vậy | 不 | 丕 | 丕 | prefer_per_dict | 丕 |
| page_0014 | 8 | 18 | kẻo | 冠 | 矯 | 矯 | prefer_per_dict | 矯 |
| page_0014 | 9 | 8 | các | 冬 | 各 | 各 | prefer_per_dict | 各 |
| page_0014 | 9 | 14 | kẻo | 犒 | 矯 | 矯 | prefer_per_dict | 矯 |

## 6. Bất đồng cả 2 cùng dict-match nhưng ra nom khác (cần audit thủ công)

| page | col | idx | syl | full→ | per→ | prev_nom | prev_tier |
|------|----:|----:|-----|------|------|----------|----------:|
| page_0012 | 4 | 4 | a | 亜→亜 | 亞→亞 | 亜 | T1 |
| page_0014 | 3 | 17 | bỏ | 補→補 | 𥙷→𥙷 | 補 | T1 |
| page_0014 | 4 | 7 | Thánh | 垩→垩 | 堊→堊 | 垩 | T1 |
| page_0014 | 7 | 4 | tử | 子→子 | 于→于 | 子 | T1 |