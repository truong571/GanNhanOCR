# Audit Post-Ship — sau F1' patch + re-run pipeline

Dataset: dataset/all/labels.csv (83,652 records, 5,703 classes)

## A — HAN_BASIC singleton audit

- Tổng HAN_BASIC singleton class: **1415**
- Records nằm trong pool singleton (HAN_BASIC, n=1): **1415**
- Tier distribution: {'1': 123, '3': 1134, '2': 158}
- Matched distribution: {'true': 1329, 'false': 86}
- Có ocr_char: 1387, null OCR: 28
- Kim đọc trùng nhãn (đúng đắn): 121

### Sample 50 — phân loại nghi vấn

| Loại | n | Đánh giá |
|---|---:|---|
| ocr_agrees_with_nom | 5 | ✅ Kim trùng nhãn — OK, không nghi |
| ocr_disagree_but_high_vis | 12 | 🟡 Kim khác, vis ≥ 0.90 — DINOv2 tự tin |
| ocr_disagree_mid_vis | 19 | 🟠 Kim khác, vis 0.80-0.90 — mơ hồ |
| ocr_disagree_low_vis | 13 | 🔴 Kim khác, vis < 0.80 — đáng nghi |
| no_ocr | 1 | ⚪ Kim null — projection fill |
| rank_high_in_cand | 14 | 🟠 nom_char rank >5 trong pool |
| not_in_cand | 13 | 🔴 nom_char KHÔNG trong pool (leak) |

### 20 sample đầu (chi tiết)

| sách | page | col | syl | ocr | nom | tier | matched | vis | rank | crop |
|------|------|---:|-----|-----|-----|----:|---------|----:|----:|------|
| SachThanhTruyen11 | page_0228 | 8 | giơ | 畧 | 剶 | T3 | False | 0.747 | 0 | `crops/page_0228/col08_char002.png` |
| SachThanhTruyen2 | page_0142 | 8 | dấu | 呌 | 鬥 | T3 | False | 0.745 | -1 | `crops/page_0142/col08_char001.png` |
| SachThanhTruyen2 | page_0042 | 9 | tăm | 沿 | 籤 | T3 | True | 0.87 | 8 | `crops/page_0042/col09_char011.png` |
| SachThanhTruyen4 | page_0050 | 1 | cấu | 礼 | 彀 | T3 | True | 0.903 | -1 | `crops/page_0050/col01_char003.png` |
| SachThanhTruyen4 | page_0022 | 8 | hôn | 㸔 | 焄 | T3 | False | 0.729 | 9 | `crops/page_0022/col08_char020.png` |
| SachThanhTruyen2 | page_0312 | 7 | thí | 丄 | 弑 | T3 | True | 0.922 | 6 | `crops/page_0312/col07_char000.png` |
| SachThanhTruyen2 | page_0190 | 3 | lối | 事 | 礌 | T3 | True | 0.943 | 4 | `crops/page_0190/col03_char006.png` |
| SachThanhTruyen2 | page_0130 | 6 | thở | 疝 | 呲 | T3 | True | 0.842 | 2 | `crops/page_0130/col06_char009.png` |
| SachThanhTruyen11 | page_0276 | 9 | giục | 伐 | 豚 | T3 | True | 0.875 | 4 | `crops/page_0276/col09_char015.png` |
| SachThanhTruyen11 | page_0106 | 2 | dỗ | 唯 | 捬 | T3 | True | 0.861 | 5 | `crops/page_0106/col02_char003.png` |
| SachThanhTruyen2 | page_0108 | 6 | ô | 每 | 戽 | T2 | True | None | 8 | `crops/page_0108/col06_char019.png` |
| SachThanhTruyen11 | page_0164 | 3 | Nhất | 瑪 | 搋 | T3 | True | 0.909 | 4 | `crops/page_0164/col03_char006.png` |
| SachThanhTruyen4 | page_0234 | 8 | vò | 捕 | 趶 | T3 | True | 0.87 | -1 | `crops/page_0234/col08_char018.png` |
| SachThanhTruyen2 | page_0050 | 9 | Khi | · | 机 | T3 | True | 0.89 | -1 | `crops/page_0050/col09_char017.png` |
| SachThanhTruyen2 | page_0048 | 5 | phôn | 魄 | 噴 | T3 | True | 0.871 | 1 | `crops/page_0048/col05_char023.png` |
| SachThanhTruyen2 | page_0118 | 7 | bọt | 浡 | 浡 | T1 | True | None | 1 | `crops/page_0118/col07_char009.png` |
| SachThanhTruyen2 | page_0304 | 6 | trao | 梓 | 捞 | T3 | True | 0.854 | 5 | `crops/page_0304/col06_char003.png` |
| SachThanhTruyen2 | page_0326 | 8 | háo | 好 | 好 | T1 | True | None | 1 | `crops/page_0326/col08_char006.png` |
| SachThanhTruyen11 | page_0048 | 1 | rất | 眾 | 栗 | T3 | True | 0.803 | 4 | `crops/page_0048/col01_char016.png` |
| SachThanhTruyen11 | page_0178 | 4 | gái | 妙 | 匄 | T3 | True | 0.873 | 4 | `crops/page_0178/col04_char000.png` |

## B — Unmatched distribution

- Tổng unmatched: **1925**
- Per book: {'SachThanhTruyen2': 447, 'SachThanhTruyen4': 976, 'SachThanhTruyen11': 502}
- Per tier:  {'3': 1925}
- Per bucket (chỉ records có nom_char): {'HAN_BASIC': 1420, 'NOM_EXT_B_PLUS': 335, 'CJK_EXT_A': 169, 'OTHER': 1}
- Records không có syllable: 0
- Tier 0 records: 0

### Tier 3 unmatched — visual_score distribution

- n = 1925
- min/p25/median/p75/max = 0.635/0.713/0.729/0.74/0.75
- < 0.70: 243
- 0.70-0.75: 1650
- ≥ 0.75: 32

### 20 sample/book

#### SachThanhTruyen2

| page | col | syl | ocr | tier | nom |
|------|---:|-----|-----|----:|-----|
| page_0098 | 3 | Đức | 知 | T3 | 䙷 |
| page_0054 | 4 | già | 雜 | T3 | 嗻 |
| page_0118 | 5 | Thầy | 㮣 | T3 | 賴 |
| page_0222 | 9 | giây | 寺 | T3 | 𠔇 |
| page_0022 | 3 | su | 主 | T3 | 樞 |
| page_0030 | 9 | đầy | 弄 | T3 | 台 |
| page_0302 | 3 | ấy | 津 | T3 | 乙 |
| page_0168 | 4 | tao | 𫊫 | T3 | 糙 |
| page_0040 | 4 | nghĩ | 丐 | T3 | 蚁 |
| page_0110 | 3 | cùng | 牙 | T3 | 珙 |
| page_0196 | 5 | vồ | 高 | T3 | 撫 |
| page_0026 | 8 | Rất | 懦 | T3 | 𫇐 |
| page_0156 | 3 | lửa | 恕 | T3 | 火 |
| page_0074 | 7 | chước | 砧 | T3 | 硳 |
| page_0020 | 3 | tử | 每 | T3 | 秄 |
| page_0036 | 6 | A | 眉 | T3 | 哑 |
| page_0126 | 4 | khi | 旦 | T3 | 諆 |
| page_0124 | 2 | tội | 遜 | T3 | 辠 |
| page_0030 | 1 | đạo | 知 | T3 | 盜 |
| page_0082 | 4 | xưa | 𭃡 | T3 | 㧅 |

#### SachThanhTruyen4

| page | col | syl | ocr | tier | nom |
|------|---:|-----|-----|----:|-----|
| page_0028 | 5 | chẳng | · | T3 | 亟 |
| page_0162 | 5 | người | 身 | T3 | 匕 |
| page_0128 | 4 | lại | 渚 | T3 | 又 |
| page_0022 | 5 | Thánh | 皇 | T3 | 㗂 |
| page_0268 | 9 | người | 丄 | T3 | 圤 |
| page_0172 | 1 | khi | 朱 | T3 | 諆 |
| page_0044 | 2 | còn | 丕 | T3 | 存 |
| page_0304 | 6 | nghĩa | 委 | T3 | 义 |
| page_0072 | 8 | những | 事 | T3 | 仙 |
| page_0190 | 5 | le | 爾 | T3 | 𠲥 |
| page_0188 | 9 | kẻ | 蜍 | T3 | 計 |
| page_0178 | 1 | vui | 實 | T3 | 衃 |
| page_0022 | 7 | sang | 丄 | T3 | 戈 |
| page_0176 | 3 | dây | 㕠 | T3 | 移 |
| page_0178 | 8 | lem | · | T3 | 淋 |
| page_0120 | 5 | chực | 身 | T3 | 𠅺 |
| page_0020 | 3 | mà | 竈 | T3 | 罵 |
| page_0072 | 7 | xứ | 丄 | T3 | 処 |
| page_0018 | 6 | khó | 中 | T3 | 瘔 |
| page_0164 | 4 | mẹ | 夷 | T3 | 袄 |

#### SachThanhTruyen11

| page | col | syl | ocr | tier | nom |
|------|---:|-----|-----|----:|-----|
| page_0252 | 2 | Chúa | 秋 | T3 | 炷 |
| page_0054 | 5 | phải | 藥 | T3 | 仕 |
| page_0098 | 9 | Thánh | 他 | T3 | 垩 |
| page_0138 | 9 | dạy | 奴 | T3 | 曳 |
| page_0058 | 4 | pha | 之 | T3 | 葩 |
| page_0162 | 9 | Giê | 主 | T3 | 支 |
| page_0046 | 4 | có | · | T3 | 固 |
| page_0168 | 9 | tha | 固 | T3 | 它 |
| page_0108 | 9 | Phi | 口 | T3 | 伾 |
| page_0166 | 3 | khác | 蜍 | T3 | 吉 |
| page_0240 | 9 | chịu | 時 | T3 | 𠮥 |
| page_0200 | 6 | cho | 補 | T3 | 株 |
| page_0072 | 4 | cô | 眉 | T3 | 喿 |
| page_0044 | 8 | MÔNG | 極 | T3 | 梦 |
| page_0170 | 2 | rô | · | T3 | 鱸 |
| page_0186 | 1 | Đức | 自 | T3 | 䙷 |
| page_0074 | 3 | mà | 庻 | T3 | 女 |
| page_0122 | 5 | trăm | 果 | T3 | 啉 |
| page_0042 | 7 | thầy | 把 | T3 | 賴 |
| page_0162 | 9 | Thánh | 并 | T3 | 𠄵 |

## C — Class-weight + rare-class grouping

- N (total records): **83,652**
- K (unique classes): **5,703**

### Class-weight distribution (balanced = N / K·n_c)

- min / p25 / median / p75 / p95 / max = 0.01 / 1.834 / 4.889 / 14.668 / 14.668 / 14.668
- CSV xuất: [evaluation/audit_post_ship/class_weights_balanced.csv](evaluation/audit_post_ship/class_weights_balanced.csv)

### Rare-class grouping proposal

Gộp các class có ≤ T records thành 1 class `<rare>`.

| Ngưỡng T | # class hiếm | # records hiếm | % class loại | % records → <rare> | K mới |
|---:|---:|---:|---:|---:|---:|
| <= 1 | 2006 | 2006 | 35.17% | 2.4% | 3698 |
| <= 2 | 2817 | 3628 | 49.4% | 4.34% | 2887 |
| <= 3 | 3298 | 5071 | 57.83% | 6.06% | 2406 |
| <= 5 | 3836 | 7430 | 67.26% | 8.88% | 1868 |
| <= 10 | 4512 | 12611 | 79.12% | 15.08% | 1192 |

## Tổng kết

**A — HAN singleton**: trên sample 50, 5 trường hợp Kim trùng nhãn (≈10% probably-correct). Số nghi (vis<0.80 + leak ngoài pool): 26. → Đa số không phải Tier 3 nhầm bừa.

**B — Unmatched 1925**: tier 0 = 0 (loan_demoted = 3 + còn lại do step 2 không tìm được match). Tier 3 unmatched có visual_score median = 0.729 — đúng vùng dưới ngưỡng 0.75.

**C — Class weight**: K=5,703 class, weight max/min ratio ≈ 1,467× (rất lệch). Gộp class ≤2 records → K=2887 (4.34% records → <rare>).