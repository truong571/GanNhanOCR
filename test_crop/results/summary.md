# Crop cleaning A/B/C comparison

Total samples: **36**  (per-book: CacThanhTruyen2=6, CacThanhTruyen4=6, CacThanhTruyen11=6, SachThanhTruyen2=6, SachThanhTruyen4=6, SachThanhTruyen11=6)

## Approaches

| Code | Approach | Description |
|---|---|---|
| **A** | Current full | Sauvola + bg_norm + morph open + stroke_norm + CC noise + border line removal |
| **B** | Sauvola only | Sauvola → trim → center 64×64. No morph / stroke / CC operations |
| **C** | NEW4 paper | Adaptive Gaussian (block 21, C 8) → trim → center 64×64 |

## Counts

- A produced output: 36/36
- B produced output: 36/36
- C produced output: 36/36

## How to evaluate

Open `grid.png` — each row is one sample, columns are original / A / B / C.

Look for:
- **Stroke fidelity**: are thin strokes preserved? Are dots from radicals (心 ⺗, 氵 etc.) intact?
- **Distortion**: does any version make the character look thicker/thinner than the original?
- **Background noise**: is paper texture removed without damaging ink?
- **Border lines**: are vertical/horizontal ruling lines (mép sách) removed cleanly?

## Per-sample details

| # | book | page | col | nom | syl | tier | OCR |
|---|---|---|---|---|---|---|---|
| 00 | CacThanhTruyen2 | page_0018 | 6 | 紇 | hạt | 1 | 紇 |
| 01 | CacThanhTruyen2 | page_0012 | 7 | 除 | thờ | 1 | 除 |
| 02 | CacThanhTruyen2 | page_0012 | 3 | 翁 | ông | 1 | 翁 |
| 03 | CacThanhTruyen2 | page_0020 | 1 | 仍 | nhưng | 1 | 仍 |
| 04 | CacThanhTruyen2 | page_0014 | 7 | 思 | tử | 3 | 爲 |
| 05 | CacThanhTruyen2 | page_0014 | 5 | 湄 | Mi | 3 | 困 |
| 06 | CacThanhTruyen4 | page_0014 | 2 | 体 | thấy | 1 | 体 |
| 07 | CacThanhTruyen4 | page_0012 | 7 | 󱬀 | gánh | 3 | 掩 |
| 08 | CacThanhTruyen4 | page_0018 | 9 | 福 | phúc | 1 | 福 |
| 09 | CacThanhTruyen4 | page_0012 | 6 | 喒 | thánh | 3 | 望 |
| 10 | CacThanhTruyen4 | page_0018 | 6 | 离 | le | 1 | 离 |
| 11 | CacThanhTruyen4 | page_0016 | 8 | 召 | chịu | 1 | 召 |
| 12 | CacThanhTruyen11 | page_0010 | 6 | 賀 | hạ | 3 | 奈 |
| 13 | CacThanhTruyen11 | page_0016 | 3 | 荗 | chưa | 3 | 体 |
| 14 | CacThanhTruyen11 | page_0014 | 4 | 蓮 | sen | 1 | 蓮 |
| 15 | CacThanhTruyen11 | page_0010 | 3 | 詳 | tường | 1 | 詳 |
| 16 | CacThanhTruyen11 | page_0010 | 3 | 󱞿 | Trời | 3 | 丞 |
| 17 | CacThanhTruyen11 | page_0010 | 6 | 羣 | còn | 1 | 羣 |
| 18 | SachThanhTruyen2 | page_0122 | 5 | ? | tlm | - | 宜 |
| 19 | SachThanhTruyen2 | page_0128 | 4 | ? | hanh | - | 朱 |
| 20 | SachThanhTruyen2 | page_0284 | 2 | ? | thi | - | 肢 |
| 21 | SachThanhTruyen2 | page_0024 | 3 | ? | vaa | - | 要 |
| 22 | SachThanhTruyen2 | page_0316 | 2 | ? | lien | - | 百 |
| 23 | SachThanhTruyen2 | page_0112 | 6 | ? | vua | - | 官 |
| 24 | SachThanhTruyen4 | page_0284 | 1 | ? | give | - | 返 |
| 25 | SachThanhTruyen4 | page_0224 | 2 | ? | ba | - | 𭃡 |
| 26 | SachThanhTruyen4 | page_0122 | 1 | ? | nguoi | - | 子 |
| 27 | SachThanhTruyen4 | page_0236 | 9 | ? | it | - | 丑 |
| 28 | SachThanhTruyen4 | page_0304 | 8 | ? | ta | - | 歡 |
| 29 | SachThanhTruyen4 | page_0150 | 4 | ? | Song | - | 死 |
| 30 | SachThanhTruyen11 | page_0252 | 2 | ? | phcii | - | 下 |
| 31 | SachThanhTruyen11 | page_0270 | 1 | ? | khi | - | 丄 |
| 32 | SachThanhTruyen11 | page_0012 | 3 | ? | nguoi | - | 𡿨 |
| 33 | SachThanhTruyen11 | page_0238 | 5 | ? | lam | - | 賑 |
| 34 | SachThanhTruyen11 | page_0250 | 7 | ? | den | - | 畨 |
| 35 | SachThanhTruyen11 | page_0058 | 4 | ? | hinh | - | 天 |
