# Debug fixes — SachThanhTruyen2 page_0012 + page_0014

Tier-3 threshold: **0.75 → 0.9**

## So sánh trước / sau

| Trang | metric | trước | sau | Δ |
|---|---|---:|---:|---:|
| page_0012 | matched | 158 | 97 | -61 ↓ |
| page_0012 | unmatched | 10 | 71 | +61 ↑ |
| page_0012 | rate_pct | 94.05 | 57.74 | -36.31 ↓ |
| page_0012 | tier1 | 88 | 79 | -9 ↓ |
| page_0012 | tier2 | 11 | 9 | -2 ↓ |
| page_0012 | tier3 | 68 | 60 | -8 ↓ |
| page_0012 | tier0 | 1 | 20 | +19 ↑ |
| page_0014 | matched | 203 | 149 | -54 ↓ |
| page_0014 | unmatched | 2 | 56 | +54 ↑ |
| page_0014 | rate_pct | 99.02 | 72.68 | -26.34 ↓ |
| page_0014 | tier1 | 128 | 121 | -7 ↓ |
| page_0014 | tier2 | 17 | 16 | -1 ↓ |
| page_0014 | tier3 | 60 | 54 | -6 ↓ |
| page_0014 | tier0 | 0 | 14 | +14 ↑ |

## Số lượt sửa theo từng fix

| Fix | Mô tả | Số record |
|---|---|---:|
| **F1_loan** | Skip phiên âm Latin (Maria, Antiochia, Roma, Misa, ...) | 33 |
| **F2_low_conf** | Tier 3 vis < 0.9 → unmatched | 55 |
| **F3_out_of_cand** | nom_char ∉ nom_candidates → unmatched | 27 |
| **F4_promoted** | ocr_char=null + 1 candidate → Tier 1 | 0 |

## Chi tiết từng fix

### F1_loan (33)

| page | col | syllable | nom (was) | tier (was) | vis | note |
|------|----:|----------|-----------|-----------:|----:|------|
| page_0012 | 2 | Ina | 哪 | T2 | — | demoted |
| page_0012 | 3 | Giê | 支 | T1 | — | demoted |
| page_0012 | 3 | su | 秋 | T1 | — | demoted |
| page_0012 | 4 | Bà | 婆 | T1 | — | demoted |
| page_0012 | 4 | Ma | 嗎 | T3 | — | demoted |
| page_0012 | 4 | ri | 𪅨 | T3 | — | demoted |
| page_0012 | 4 | a | 亜 | T1 | — | demoted |
| page_0012 | 5 | Rô | 嚕 | T3 | — | demoted |
| page_0012 | 5 | ma | 嫫 | T3 | — | demoted |
| page_0014 | 5 | sa | 沙 | T1 | — | demoted |
| page_0014 | 5 | se | 𠈴 | T3 | — | demoted |
| page_0014 | 5 | do | 由 | T1 | — | demoted |
| page_0014 | 5 | tê | 痺 | T3 | — | demoted |
| page_0014 | 5 | mi | 眉 | T1 | — | demoted |
| page_0014 | 5 | sa | 娑 | T3 | — | demoted |
| page_0012 | 6 | An | 氨 | T3 | — | demoted |
| page_0012 | 6 | ti | 𤰞 | T1 | — | demoted |
| page_0012 | 6 | ô | 烏 | T1 | — | demoted |
| page_0012 | 6 | ki | 璣 | T3 | — | demoted |
| page_0014 | 6 | Giê | 支 | T1 | — | demoted |
| page_0014 | 6 | su | 秋 | T2 | — | demoted |
| page_0014 | 6 | Rô | 鯈 | T3 | — | demoted |
| page_0014 | 6 | ma | 尛 | T3 | — | demoted |
| page_0012 | 7 | phê | 批 | T1 | — | demoted |
| page_0012 | 7 | rô | 𬠩 | T3 | — | demoted |
| page_0012 | 8 | phê | 批 | T1 | — | demoted |
| page_0012 | 8 | rô | 噜 | T3 | — | demoted |
| page_0012 | 8 | Giê | 支 | T1 | — | demoted |
| page_0012 | 8 | su | 秋 | T2 | — | demoted |
| page_0014 | 8 | An | 𠼞 | T3 | — | demoted |
| page_0014 | 8 | ti | 司 | T1 | — | demoted |
| page_0014 | 8 | ô | 烏 | T1 | — | demoted |
| page_0014 | 8 | ki | 其 | T1 | — | demoted |

### F2_low_conf (55)

| page | col | syllable | nom (was) | tier (was) | vis | note |
|------|----:|----------|-----------|-----------:|----:|------|
| page_0014 | 1 | làm | 漤 | T3 | 0.860 | < 0.9 |
| page_0014 | 1 | trời | 𡗶 | T3 | 0.880 | < 0.9 |
| page_0012 | 2 | SƠ | 匹 | T3 | 0.854 | < 0.9 |
| page_0012 | 2 | NHÁT | 戛 | T3 | 0.768 | < 0.9 |
| page_0012 | 2 | Ong | 蜂 | T3 | 0.805 | < 0.9 |
| page_0012 | 2 | vồ | 釪 | T3 | 0.808 | < 0.9 |
| page_0012 | 2 | vì | 爲 | T3 | 0.794 | < 0.9 |
| page_0012 | 2 | truyen | 揎 | T3 | 0.831 | < 0.9 |
| page_0014 | 2 | nước | 箸 | T3 | 0.827 | < 0.9 |
| page_0014 | 2 | Chó | 犬 | T3 | 0.828 | < 0.9 |
| page_0014 | 2 | vua | 𠱭 | T3 | 0.840 | < 0.9 |
| page_0014 | 2 | Vua | 君 | T3 | 0.764 | < 0.9 |
| page_0012 | 3 | đầy | 台 | T3 | 0.779 | < 0.9 |
| page_0012 | 3 | là | 囉 | T3 | 0.779 | < 0.9 |
| page_0012 | 3 | Thánh | 喒 | T3 | 0.783 | < 0.9 |
| page_0014 | 3 | rằng | 喨 | T3 | 0.857 | < 0.9 |
| page_0014 | 3 | làm | 灠 | T3 | 0.866 | < 0.9 |
| page_0014 | 4 | tao | 糙 | T3 | 0.899 | < 0.9 |
| page_0014 | 4 | cho | 咮 | T3 | 0.765 | < 0.9 |
| page_0014 | 4 | sãi | 𠱊 | T3 | 0.844 | < 0.9 |
| page_0014 | 4 | ấy | 意 | T3 | 0.802 | < 0.9 |
| page_0014 | 4 | làm | 㕚 | T3 | 0.838 | < 0.9 |
| page_0012 | 5 | đi | 𠫾 | T3 | 0.899 | < 0.9 |
| page_0014 | 5 | tội | 𩵽 | T3 | 0.838 | < 0.9 |
| page_0014 | 5 | ngày | 匕 | T3 | 0.832 | < 0.9 |
| page_0014 | 5 | làm | 爪 | T3 | 0.839 | < 0.9 |
| page_0014 | 5 | lê | 琍 | T3 | 0.835 | < 0.9 |
| page_0012 | 6 | đòi | 隧 | T3 | 0.868 | < 0.9 |
| page_0012 | 6 | quở | 𠵩 | T3 | 0.875 | < 0.9 |
| page_0014 | 6 | Vua | 君 | T3 | 0.891 | < 0.9 |
| page_0014 | 6 | đem | 耽 | T3 | 0.879 | < 0.9 |
| page_0014 | 6 | đi | 𫺲 | T3 | 0.844 | < 0.9 |
| page_0012 | 7 | thưa | 撪 | T3 | 0.889 | < 0.9 |
| page_0012 | 8 | phải | 塊 | T3 | 0.758 | < 0.9 |
| page_0012 | 8 | Dêu | 搖 | T3 | 0.857 | < 0.9 |
| page_0012 | 8 | vì | 位 | T3 | 0.858 | < 0.9 |
| page_0012 | 8 | hăng | 行 | T3 | 0.868 | < 0.9 |
| page_0014 | 8 | mặt | 湎 | T3 | 0.839 | < 0.9 |
| page_0014 | 8 | kẻo | 矯 | T3 | 0.884 | < 0.9 |
| page_0012 | 9 | hỏi | 噲 | T3 | 0.816 | < 0.9 |
| page_0012 | 9 | băng | 冰 | T3 | 0.879 | < 0.9 |
| page_0012 | 9 | có | 故 | T3 | 0.849 | < 0.9 |
| page_0012 | 9 | tưởng | 奖 | T3 | 0.828 | < 0.9 |
| page_0012 | 9 | làm | 灠 | T3 | 0.862 | < 0.9 |
| page_0012 | 9 | vì | 為 | T3 | 0.828 | < 0.9 |
| page_0012 | 9 | bụt | 桲 | T3 | 0.784 | < 0.9 |
| page_0012 | 9 | chữa | 渚 | T3 | 0.871 | < 0.9 |
| page_0012 | 9 | nước | 著 | T3 | 0.879 | < 0.9 |
| page_0012 | 9 | nữa | 汝 | T3 | 0.844 | < 0.9 |
| page_0012 | 9 | lời | 塁 | T3 | 0.787 | < 0.9 |
| ... | | | (+5 more) | | | |

### F3_out_of_cand (27)

| page | col | syllable | nom (was) | tier (was) | vis | note |
|------|----:|----------|-----------|-----------:|----:|------|
| page_0012 | 3 | này | 𠽬 | T3 | 0.876 | cands=['又', '呢', '尼', '怩', '昵'] |
| page_0012 | 3 | Giu | 稿 | T2 | — | cands=['裱', '㧼', '𩤕', '瓢', '緥'] |
| page_0012 | 3 | cùng | 邛 | T3 | 0.901 | cands=['供', '共', '其', '塘', '廾'] |
| page_0014 | 3 | liền | 連 | T2 | — | cands=['吝', '嗹', '恡', '悋', '槤'] |
| page_0014 | 3 | tao | 蚤 | T2 | — | cands=['\ue077', '傮', '幍', '慅', '慒'] |
| page_0012 | 4 | làm | 𬈋 | T3 | 0.844 | cands=['\ue1d5', '乄', '滥', '漤', '濫'] |
| page_0012 | 4 | An | 𠼞 | T3 | 0.758 | cands=['侒', '媕', '安', '案', '桉'] |
| page_0012 | 4 | ti | 朇 | T3 | 0.775 | cands=['丝', '伺', '俾', '偲', '凘'] |
| page_0014 | 4 | làm | 𣩂 | T3 | 0.932 | cands=['\ue1d5', '乄', '滥', '漤', '濫'] |
| page_0014 | 4 | là | 𨔍 | T3 | 0.809 | cands=['\ue027', '\ue0e6', '𱺵', '卥', '号'] |
| page_0014 | 4 | hồn | 諢 | T3 | 0.853 | cands=['塊', '愧', '捆', '楎', '浑'] |
| page_0012 | 5 | là | 𦲿 | T3 | 0.922 | cands=['\ue027', '\ue0e6', '𱺵', '卥', '号'] |
| page_0012 | 5 | aé | 阿 | T2 | — | cands=['丫', '了', '亜', '亞', '兮'] |
| page_0014 | 5 | là | 買 | T3 | 0.872 | cands=['\ue027', '\ue0e6', '𱺵', '卥', '号'] |
| page_0014 | 5 | một | 沒 | T3 | 0.900 | cands=['\ue029', '\ue134', '\ue298', '\ue2a7', '\ue2ae'] |
| page_0014 | 5 | một | 沒 | T3 | 0.843 | cands=['\ue029', '\ue134', '\ue298', '\ue2a7', '\ue2ae'] |
| page_0012 | 6 | về | 𧗱 | T2 | — | cands=['𱸳', '揮', '撝', '皮', '米'] |
| page_0012 | 6 | thì | 蒔 | T2 | — | cands=['𱢑', '匙', '埘', '塒', '寺'] |
| page_0012 | 7 | bụt | 𡥞 | T3 | 0.831 | cands=['\ue0ba', '佛', '侼', '勃', '孛'] |
| page_0014 | 7 | ấy | 衤 | T3 | 0.919 | cands=['𱍸', '乙', '倚', '噫', '娎'] |
| page_0014 | 8 | một | 没 | T3 | 0.906 | cands=['\ue029', '\ue134', '\ue298', '\ue2a7', '\ue2ae'] |
| page_0014 | 8 | ta | 罝 | T3 | 0.858 | cands=['些', '仨', '低', '偺', '傞'] |
| page_0012 | 9 | Ví | 𡃪 | T3 | 0.877 | cands=['\ue123', '𱒢', '叿', '啻', '喟'] |
| page_0012 | 9 | tao | 蚤 | T3 | 0.898 | cands=['\ue077', '傮', '幍', '慅', '慒'] |
| page_0012 | 9 | chăng | 荘 | T3 | 0.854 | cands=['\ue001', '丕', '仳', '庄', '庒'] |
| page_0014 | 9 | tay | 𢬣 | T3 | 0.925 | cands=['\ue010', '塞', '壦', '思', '拪'] |
| page_0014 | 9 | giữ | 與 | T3 | 0.797 | cands=['𱠎', '𰑚', '与', '佇', '咛'] |
