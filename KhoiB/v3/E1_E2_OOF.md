# E1 / E2 OUT-OF-FOLD — B-1' Kaggle (804 lớp, 5 fold theo trang) — K5 16/09/2026

Nguồn: `KhoiB/v3/p_visual_oof_v3_results/{p_visual_oof_v3.csv, probs_oof.npz, summary.json}` (labels_md5 `1a2d8e0ca0a1199f11ff514135051c15` == `dataset_out/labels_final.csv`), script `KhoiB/v3/e1_e2_oof.py` → `e1_e2_oof.json`. Mọi ô đều out-of-fold (mô hình fold k chưa thấy trang của ô). Ô có âm ∈ 804 lớp: 79,889/83,239. Kiểm csv↔npz: max|Δp| = 6.07e-07, top-1 csv 74.42% == npz 74.42%.

## E1 · P(âm v3 | crop) theo tier_v3 (so HUONG_TAN_CONG §3: 706 lớp, chỉ trang test 20%)

| tier_v3 | n (∈ lớp / tổng) | top-1 | top-5 | hạng trung vị | p50 | % p<0,05 | % argmax đúng ∧ p≥0,9 | §3 top-1 (test, 706 lớp) |
|---|---|---|---|---|---|---|---|---|
| CHAR_A | 47,033 / 47,956 | **77.31%** | 91.77% | 1 | 0.6978 | 13.17% | 25.24% | 77.8% |
| CHAR_B | 3,280 / 3,311 | **76.19%** | 90.18% | 1 | 0.6407 | 14.79% | 22.23% | 67.6% |
| SYL | 18,878 / 19,075 | **78.13%** | 92.04% | 1 | 0.6958 | 12.77% | 25.88% | 81.9% |
| REVIEW | 10,698 / 12,897 | **54.62%** | 71.7% | 1 | 0.1959 | 35.61% | 9.63% | 45.6% |
| **tất cả** | 79,889 / 83,239 | 74.42% | 89.08% | 1 | 0.645 | 16.15% | 23.18% | — |

Theo tier (bộ giao nộp `tier`):

| tier | n (∈ lớp / tổng) | top-1 | top-5 | hạng trung vị | p50 |
|---|---|---|---|---|---|
| GOLD | 50,969 / 51,922 | 77.46% | 91.77% | 1 | 0.6976 |
| QUARANTINE | 41 / 42 | 39.02% | 43.9% | 26 | 0.0015 |
| REVIEW | 10,672 / 12,871 | 54.52% | 71.65% | 1 | 0.1936 |
| SYLLABLE | 18,207 / 18,404 | 77.66% | 91.89% | 1 | 0.6893 |

## E1 theo sách × tier_v3

| sách | tier_v3 | n ∈ lớp | top-1 | top-5 | p50 |
|---|---|---|---|---|---|
| stt2 | CHAR_A | 15,849 | 77.29% | 92.03% | 0.6912 |
| stt2 | CHAR_B | 1,064 | 71.43% | 88.25% | 0.5227 |
| stt2 | SYL | 6,360 | 76.87% | 91.78% | 0.6606 |
| stt2 | REVIEW | 3,916 | 55.54% | 73.7% | 0.1986 |
| stt2 | **tất cả** | 27,189 | 73.83% | 89.18% | 0.6174 |
| stt4 | CHAR_A | 15,347 | 75.05% | 90.56% | 0.6719 |
| stt4 | CHAR_B | 1,014 | 74.65% | 89.35% | 0.6294 |
| stt4 | SYL | 6,386 | 77.86% | 91.37% | 0.6943 |
| stt4 | REVIEW | 3,664 | 56.82% | 73.36% | 0.2562 |
| stt4 | **tất cả** | 26,411 | 73.19% | 88.32% | 0.6318 |
| stt11 | CHAR_A | 15,837 | 79.51% | 92.69% | 0.7254 |
| stt11 | CHAR_B | 1,202 | 81.7% | 92.6% | 0.7291 |
| stt11 | SYL | 6,132 | 79.73% | 93.02% | 0.7298 |
| stt11 | REVIEW | 3,118 | 50.87% | 67.25% | 0.147 |
| stt11 | **tất cả** | 26,289 | 76.26% | 89.75% | 0.6839 |

## E1 theo fold (ô mọi tier) và E2b theo fold

| fold | trang | val top-1 (GOLD+SYL, Kaggle) | OOF top-1 mọi ô | OOF top-5 | p50 | E2b AUC | n đúng / n sai |
|---|---|---|---|---|---|---|---|
| 0 | 80 | 77.4% | 74.05% | 88.84% | 0.6482 | **0.9435** | 8,431 / 15,413 |
| 1 | 85 | 78.2% | 74.9% | 89.12% | 0.6506 | **0.9471** | 8,794 / 16,037 |
| 2 | 88 | 77.3% | 74.43% | 89.66% | 0.6469 | **0.9486** | 9,383 / 17,154 |
| 3 | 104 | 78.1% | 75.15% | 89.41% | 0.6513 | **0.9446** | 11,030 / 20,124 |
| 4 | 91 | 76.5% | 73.43% | 88.34% | 0.6269 | **0.9395** | 9,395 / 17,186 |

## E2b · thanh ghi trượt (CHAR_A: P(âm mình) vs P(âm ô kề), loại 58 kề cùng âm)

- AUC toàn bộ **0.9447** (summary.json.e2b: 0.9446; HUONG_TAN_CONG §3: 0,946 trên trang test) — n đúng 47,033, n sai 85,914 (summary.json n sai 87,431: `train_oof_cnn_v3.py:394` lấy cả âm kề của 1,517 lượt ô CHAR_A có âm mình ∉ lớp; ở đây chỉ ô có cả hai vế — AUC lệch 0,0001).
- Ngưỡng p<0,05: bắt 91.06% ô sai, oan 13.17% ô đúng (§3: 93% / 14,3%).
- Theo sách: stt11 0.9497 (n 15,837/29,056) · stt2 0.9453 (n 15,849/28,814) · stt4 0.9385 (n 15,347/28,044).
- Theo fold: f0 0.9435 · f1 0.9471 · f2 0.9486 · f3 0.9446 · f4 0.9395 (min–max 0.9395–0.9486).

## Đối chiếu B-3: summary.json.b3_gate_review vs visual_syl_gate

- summary.json đếm MỌI ô tier_v3 REVIEW (12.897): argmax == âm ∧ p ≥ 0,9 → 1030 (tính lại 1030); ≥ 0,8 → 1846 (tính lại 1846).
- visual_syl_gate loại trước 36 ô REVIEW đã dùng được (QĐ-01 khoá → tier GOLD) + 188 ô âm không hợp lệ; trong số ô bị loại có 5 qua cổng 0,9 và 19 qua cổng 0,8 → còn **1025 / 1827** = đúng số của `visual_syl_gate_report.json` (1.025 / 1.827). Hai số KHỚP sau khi trừ ô bị loại.

## Đọc

- Top-1 OOF toàn bộ 74.42% trên 79,889 ô (mọi trang, 804 lớp) so 77,8% GOLD-trực-tiếp trang test (706 lớp, 4.899 ô) của §3: cùng bậc; CHAR_A 77.31%, SYL 78.13%, CHAR_B 76.19%, REVIEW 54.62%. Thứ tự SYL ≥ CHAR_A > CHAR_B ≫ REVIEW của §3 được tái lập trên toàn ngữ liệu, không chỉ 20% trang test.
- CHAR_B (cầu tự dạng) OOF 76.19% cao hơn 67,6% của §3 — §3 đo trên GOLD-cầu của thế hệ 64.525 (512 ô test), v3 định nghĩa CHAR_B khác (3.311 ô).
- E2b AUC 0.9447 ổn định giữa 5 fold (chênh ≤ 0.009) và 3 sách → kênh ảnh không phụ thuộc trang đã học; đây là điều kiện B-1 ("E2b out-of-fold trên trang train ≈ test, AUC ≈ 0,95") — ĐẠT.
