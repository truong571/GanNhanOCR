# Khối C · C3 — phê bình toàn khối, vá lỗi chặn, thiết kế mẻ chốt (16/09/2026)

Vai: người phản biện luận văn đọc toàn bộ Khối C **trước khi có một verdict nào** (verdict = 0 tại thời điểm rà —
mọi thay đổi dưới đây không làm mất công chấm). Đối tượng rà: `pipeline/ground_truth/make_khoi_c_batch.py`,
`estimate_khoi_c.py`, HTML một phiên (thử máy Chrome headless), `README.md` của mẻ,
`docs/HUONG_DAN_CHAM_KHOI_C_2026-09-16.md`, `docs/DANH_MUC_SUA_DOI_CUOI_2026-09-16.md` §1C, bài học 04/08
(κ = 0,13 vì 'wrong_image' không ngưỡng; memory `audit-rater-reliability-crisis`).

**Kết luận ngắn:** mẻ C1 có **3 lỗi chặn** (lộ nhóm "không mã" ⇒ lộ CHAR vs SYL/REVIEW; mồi dương toàn 𠊚/QĐ-01 →
người chấm là tác giả QĐ-01 nhận ra ngay; lặp ẩn 75 ô SRS cho KTC κ ≈ ±0,28 — không kết luận được) và **1 lỗi thiết
kế** (T1 150 ô chỉ đạt mục tiêu ≤ 3 % khi 0 lỗi, công suất 0,22). Tất cả đã **sửa trong mã và dựng lại** (tất định,
hai lần dựng byte-identical). Mẻ chốt: **1.279 lượt = 1.179 ô + 100 lặp, 5 phiên**, `KHOA.jsonl` sha256
`7cb21bafb604d4c9…`, `labels_final.csv` sha256 `8da7da81fc9c1da2…` (không đổi). Selftest ground_truth 218 → **232/0**,
toàn repo 1.126 → **1.140/0** (`scripts/run_all_selftests.sh`, mốc C3).

---

## 1. Phê bình theo 8 câu hỏi

### (1) Mù có thật không? — KHÔNG hoàn toàn: 2 kênh lộ, 1 kênh cần pha loãng

Đã thử từng kênh trên `_khoa/KHOA.jsonl` mẻ C1 (1.064 lượt) và HTML (`scratchpad/khoi_c/leak1.py`):

| Kênh | Đo | Kết luận |
|---|---|---|
| Thứ tự ô | P(hai ô liền nhau cùng tang) quan sát 0,340 vs kỳ vọng 0,356; cùng tier_v3 0,318 vs 0,317; mỗi phiên có đủ 8 tầng | **không lộ** (`interleave` băm sha1 theo seed) |
| `item_id` | 12 hex từ UUID4 RNG có seed, gán theo (tang, key) nhưng giá trị ngẫu nhiên; phân bố hex đầu theo tang đồng đều | **không lộ** (`assign_ids` `:438`) |
| Kích thước crop theo thư mục | median w×h: file 93×129, cắt lại 94×129; theo tier CHAR_A 94×131 / CHAR_B 92×129 / REVIEW 93,5×129 / SYL 91,5×132; **40/40** ô cắt lại từ scan byte-identical với tệp đã giao (`scratchpad/khoi_c/cutcheck.py`) | **không lộ** |
| Tên tệp / đường dẫn / tier / rule | `check_blind` 26 mẫu sau khi bỏ base64 = 0/5 phiên; item chỉ có `{id, crop, ctx, ref, syl, code, uni, cw, ch}` | **không lộ** |
| **Glyph tham chiếu + "chưa có mã (chỉ hỏi Q1)"** | 425/1.064 ô không mã = REVIEW 50 + SYL 148 + T1 150 + T2 40 + B-2 7 + lặp 30; 0 ô CHAR không mã | **LỘ CHAR vs SYL/REVIEW/T1** — hiện ngay từ đầu, trước khi Q1 được trả lời. Thêm nữa: ô CHAR có glyph "gợi ý" khi trả lời Q1, ô SYL không → Q1 hai nhóm không đo cùng điều kiện thông tin |
| **Mã 𠊚 / âm "người"** | 84 ô hiện 𠊚 = 45 mồi dương + 30 B-5 + 8 T2 + 1 B-2 = **100 % QĐ-01**; trong 89 ô hiện "người", 84 là QĐ-01 (94 %); toàn ngữ liệu 𠊚 = 2.012 QĐ-01 + 26 không khoá | **LỘ QĐ-01**: người chấm (chủ nhiệm, người ký QĐ-01) thấy 𠊚 hay "người" là biết "ô đã quyết" → mồi dương không đo được gì, B-5 chệch về `dung` |
| Ảnh ngữ cảnh | cột ±2,2 ô, khung đỏ; lộ trang/sách nhưng không lộ tier | chấp nhận (cần để phân xử) |

**Vá (đã làm):**

- **Giao diện HAI PHA** (`make_khoi_c_batch.py:777–800`, `:722`, `:731–735`): pha 1 mọi ô trông y hệt nhau — crop ·
  ngữ cảnh · ÂM · ô phải ghi "MÃ (nếu có) hiện sau khi trả lời Q1"; **chỉ sau** khi Q1 được bấm mới lộ MÃ + glyph + Q2
  (hoặc "ô này không có mã (Q2 không áp dụng)" rồi tự sang ô sau). Phím 5·6·7 bị chặn khi chưa lộ. Verdict ghi
  `q1_blind` (Q1 lúc lộ mã), `q1` (cuối), `n_q1_change_after_reveal`, `dwell_q1_ms`, `hist[].after_reveal`
  (`:839–843`). Ước lượng dùng `q1_blind` (`estimate_khoi_c.py:139`); verdict thế hệ cũ không có `q1_blind` → báo động
  "không mù giữa CHAR và SYL". Thử máy Chrome headless trên phiên 4 ô: pha 1 không có `.lab`/"không có mã"; bấm 5
  trước Q1 bị bỏ qua; bấm 1 → lộ mã, `q1_blind=dung`, `dwell_q1=308 ms`; bấm 2 sau lộ → `q1=sai_crop`,
  `q1_blind=dung`, `n_change=1`, hist có `after_reveal:true`; ô không mã: bấm 3 → lộ "không có mã", tự chuyển;
  xuất 4 dòng, khoá, phím sau khoá không đổi verdict.
  Vì sao không "hiện glyph cho cả SYL (chữ đa số của (sách, âm))": glyph đó không phải cái đem nộp (SYL nộp
  không mã) và vẫn là gợi ý — Q1 phải đo "người đọc chữ trên crop có ra âm này không" ở cùng điều kiện cho mọi tier;
  hai pha làm đúng việc đó mà không bỏ Q2.
- **Mồi dương** = SRS **phân tầng theo tier_v3 trên cả 2.012 ô QĐ-01** (CHAR_A 26/1.180 · SYL 11/515 · CHAR_B 5/281 ·
  REVIEW 3/36; `draw_t4_decoys` `:332–360`) → có `stratum_N`/`design_weight`, đồng thời là tầng "QĐ-01" của dân số
  re-dataset. Đạt/rớt **chỉ theo Q1** (mù ở pha 1); Q2 = `dung` là *phán quyết*, chỉ đếm "nhất quán với QĐ-01"
  (`decoy_report` `:262–290`). Ở pha 1 người chấm chỉ thấy "người" + crop.
- **T6 — 40/283 ô "người" CHƯA KHOÁ** (`draw_t6_nguoi` `:412–435`; SYL 28 · CHAR_B 6 · REVIEW 6, w = 7,075): pha loãng
  dấu hiệu "người" ⇒ QĐ-01 từ **94 % → 66 %** (`plan.json.blind_check.nguoi_shown`: 132 ô "người", 87 QĐ-01) và
  đo thẳng khối còn lại của QĐ-01a (chưa từng được người nhìn mù).
- **Giới hạn còn lại (ghi vào luận văn):** (i) mã nằm trong JSON của HTML — mù *thụ động* (giao diện), không chống
  người cố ý mở DevTools; (ii) sau pha 2 người chấm biết ô có/không mã và có thể nhớ mẫu; các lần đổi Q1 sau lộ
  được đếm, và precision công bố dùng `q1_blind`; (iii) 66 % ô "người" vẫn là QĐ-01 — Q1 trên các ô đó vẫn có
  prior "đã quyết", nêu như giới hạn của tầng QĐ-01.

### (2) Mồi âm có "dễ" quá không? — Đáp án chắc, nhưng chỉ là kiểm chú ý

45 mồi âm: hiện ÂM/MÃ của ô kề `syl_idx±1` (khác âm, khác mã, không QĐ-01, cùng cột). Đã kiểm **0/45** cặp
(chữ thật, âm hiển thị) là một cách đọc hợp lệ: 0 trong `dict/QuocNgu_SinoNom.csv` (104.177 cặp), 0 trong chính
ngữ liệu (`scratchpad/khoi_c/decoy_neg.py`). Ví dụ: 饒/nhiều hiện 施/thí; 麻/mà hiện 扒/bắt. → đáp án `sai_am`/`sai`
không thể bị "đúng oan".

Đúng là dễ: chữ khác hẳn, glyph khác hẳn. Nhưng vai của mồi âm là **bắt người bấm máy** (κ 0,13 của 04/08 có phần
do chấm không nhìn), không phải đo năng lực đọc — năng lực đọc đo bằng κ lặp ẩn và bằng chính precision. Với hai
pha, ở Q1 mồi âm **không có glyph** → người chấm phải đọc chữ để thấy nó không phải "thí" → khó hơn bản C1 một bậc.
Không đổi thêm; ghi rõ "mồi = attention check" trong luận văn. (Mồi "khó" kiểu cặp dễ nhầm 㝵/𠊚 bị bác vì không có
đáp án chắc.)

### (3) Lặp ẩn có đủ để κ có KTC hẹp không? — KHÔNG với 75 ô SRS; có với 100 ô phân tầng

SE(κ) ≈ √(pₒ(1−pₒ)/(n(1−pₑ)²)). Với tỉ lệ khác `dung` dự kiến CHAR_A 4 % · CHAR_B 12 % · SYL 10 % · REVIEW 40 % · T1 2 %
· T2 50 % và tự-đồng-thuận pₒ = 0,95:

| Thiết kế | n | P(khác dung) | pₑ | κ | KTC 95 % |
|---|---:|---:|---:|---:|---|
| C1: 75 SRS mẻ chính (½ CHAR_A) | 75 | 0,098 | 0,823 | 0,72 | **±0,28** (pₒ = 0,90: κ 0,44 ± 0,38) |
| C3: 100 phân tầng MAIN CHAR_A 20 · CHAR_B 15 · SYL 15 · REVIEW 15 · T1 15 · T2 20 | 100 | 0,204 | 0,675 | 0,85 | **±0,13** (pₒ = 0,90: 0,69 ± 0,18) |

Muốn ±0,15 bằng SRS phải 250 ô lặp. Vá: `add_hidden_repeats` phân tầng theo (tang, tier_v3) (`:456–494`, `LAP_ALLOC`
`:103`), gap ≥ 100 (đo min 102, median 445; 93/100 lặp rơi vào phiên khác), ghi `lap_tang` để tách κ theo tầng gốc.
Ước lượng thêm **KTC bootstrap** 2.000 lần, seed cố định (`kappa_bootstrap_ci` `estimate_khoi_c.py:318`) — giả lập
nhiễu 5 % trên mẻ thật cho κ Q1 0,79 [0,64; 0,91], đúng cỡ ±0,13. Báo cáo luôn kèm đồng thuận thô (Wilson) và ma
trận, vì κ nhạy tỉ lệ nền.

### (4) Cỡ mẫu: CHAR_A 300, muốn khẳng định ≥ 98 % cần gì?

- n = 300, precision 96 %: Wilson [93,1; 97,7] → ±2,3 điểm. n = 600: [94,1; 97,3]. Bán rộng ±1,0 điểm cần **1.496**
  ô; ±1,5 cần 674 (`stats.required_n_for_halfwidth`).
- Khẳng định một phía "≥ 98 %" ở 95 %: với n = 300, cận dưới CP theo số lỗi 0 → 99,0 % · 1 → 98,4 % · **2 → 97,9 %** →
  chỉ đạt khi **≤ 1 lỗi / 300**. Kế hoạch chấp nhận (`stats.acceptance_plan`): p₀ = 0,98, precision thật 0,995 → **n = 456,
  c = 4** (công suất 0,92); precision thật 0,99 → **n = 1.271, c = 17** (0,91); công suất 0,8 với 0,99 → 969.
- Kết luận cho luận văn: **300 ô CHAR_A đủ để công bố điểm ước lượng ± ~2,3 điểm và để bác/không bác "≥ 98 %" khi
  precision thật rất cao (≥ 99,5 %)**; không đủ để *chứng minh* ≥ 98 % nếu precision thật ≈ 98–99 %. Đúng tinh thần
  DANH_MUC C-3: "nếu GOLD thật 93–96 % thì CI là số trung thực". Không tăng n trong mẻ này; nếu sau mẻ CHAR_A ≥ 98,5 %
  và luận văn cần con số "≥ 98 %", chấm bổ sung 160–450 ô CHAR_A (một mẻ phụ cùng khung, ~30–60 phút).

### (5) Trọng số tầng đúng chưa? — Đúng cho dân số "ngoài QĐ-01"; đã khép kín về re-dataset

- Dân số MAIN = `labels_final` trừ QĐ-01 (2.012) trừ QUARANTINE (42), yêu cầu crop (CHAR_B mất 10 ô không tệp).
  Σ N_h theo tier: CHAR_A 46.743 · CHAR_B 3.017 · SYL 18.554 · REVIEW 12.861. Đối chiếu `re-dataset/labels.csv`
  (70.326 ô): **= CHAR_A 46.743 + CHAR_B 3.017 + SYL 18.554 + QĐ-01 2.012, khớp từng con số** (`scratchpad/khoi_c/pop.py`).
- `stats.stratified_mean_ci` = ước lượng phân tầng chuẩn với w_h = N_h/N, phương sai có FPC — đúng là hậu-phân-tầng
  (post-stratification) vì mẫu được rút SRS **trong** tầng; `_ht` (`estimate_khoi_c.py:415`) lấy N_h từ khoá, n_h = số ô
  đã chấm trừ `khong_ro` (giả định MCAR trong tầng, số `khong_ro` được báo riêng). Tầng nhỏ (5 ô, midpoint_split) có
  p̂ = 1 → s² = 0 → phương sai bị đánh giá thấp một chút — không đáng kể vì tổng trọng số của 6 tầng đó < 1,5 %.
- **Lỗi khép kín đã vá:** bản C1 chỉ ước lượng được "ngoài QĐ-01" (68.314) trong khi bộ giao nộp là 70.326. Nay mồi dương
  có trọng số → pooled `QD01` (N = 2.012, 4 tầng) và `USABLE_RE_DATASET` = USABLE ∪ QD01 (**N = 70.326**, 22 tầng;
  `:485–503`), kiểm bằng tính tay trong selftest.
- Wilson gộp không trọng số vẫn in cạnh HT để người đọc thấy chênh.

### (6) B-3: 0/150 → cận trên 2,0 % có hợp mục tiêu ≤ 3 %? — Hợp nhưng KHÔNG có công suất

Cận trên CP một phía 95 % theo (n, k lỗi):

| n | k=0 | k=1 | k=2 | k=3 | k=4 | k tối đa đạt ≤ 3 % | P(đạt \| lỗi thật 0,5 %) | P(đạt \| 1 %) | P(đạt \| 2 %) |
|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 150 | 2,0 % | 3,1 % | 4,1 % | 5,1 % | 6,0 % | **0** | 0,47 | **0,22** | 0,05 |
| 200 | 1,5 % | 2,3 % | 3,1 % | 3,8 % | 4,5 % | 1 | 0,74 | 0,40 | 0,09 |
| 250 | 1,2 % | 1,9 % | 2,5 % | 3,1 % | 3,6 % | 2 | 0,87 | 0,54 | 0,12 |
| **300** | 1,0 % | 1,6 % | 2,1 % | **2,6 %** | 3,0 % | **3** | **0,93** | **0,65** | 0,15 |
| 400 | 0,7 % | 1,2 % | 1,6 % | 1,9 % | 2,3 % | 6 | 1,00 | 0,89 | 0,31 |

Với 150 ô, một lỗi duy nhất (kể cả lỗi *của người chấm*, tự-đồng-thuận ~95 %) làm rớt cổng; xác suất đạt khi luật
thật sự tốt (1 % lỗi) chỉ 0,22 — thí nghiệm gần như được thiết kế để thất bại. Vá: **T1 = 300** (100/sách,
`N_T1` `:101`; w 2,65 / 3,56 / 4,02 trên 1.023 ô còn lại của 1.025 gate_09), cho phép ≤ 3 lỗi. Ước lượng in cả cận trên
"bảo thủ" (KHÔNG RÕ tính là lỗi) và tỉ lệ `sai_am` riêng (lỗi *âm* là thứ luật gate chịu trách nhiệm; `sai_crop` là lỗi
hộp, không do cổng thị giác). Chi phí: +150 ô ≈ +15–20 phút.

### (7) T2 chuỗi trượt: Q1 có bắt được "trượt 1 ô" không? — Có

Trượt thanh ghi = crop là chữ của ô i nhưng ÂM là của ô i±1. Q1 định nghĩa `sai_am` = "crop là MỘT chữ hoàn chỉnh
nhưng không đọc là âm này" → đúng là lớp đó (crop không hỏng, âm sai). Người chấm không thấy chuỗi (100 ô của 24
chuỗi bị xáo khắp 5 phiên) nên không "đoán theo mẫu"; ô nhập "âm đúng là…" được so với `argmax` thị giác
(`t2_report` `:568`). Chuỗi ≥ 50 % `sai_am` → đề xuất hạ REVIEW theo chuỗi. Điểm yếu nhỏ: 40/100 ô T2 không mã, 60 có
mã — với hai pha điều này không lộ ở Q1 nữa. Giữ nguyên.

### (8) Điểm yếu còn lại và cách vá (≤ 1 ngày)

| # | Điểm yếu | Vá | Công |
|---|---|---|---|
| a | **Một người chấm**, là tác giả pipeline và QĐ-01 → κ nội tại không thay được κ liên người | Người thứ hai chấm **1 phiên** (256 ô, ≈ 35 phút) — cùng HTML, đổi `session_id` hậu tố; `report_combined.cohens_kappa` đã có sẵn cho liên người | 0,5 ngày (tìm người) |
| b | Mã nằm trong JSON HTML — mù thụ động | Chấp nhận + tuyên bố; nếu cần "chống gian" thì tách MÃ sang tệp thứ hai chỉ mở ở pha 2 | 2 giờ, không làm |
| c | 66 % ô "người" vẫn là QĐ-01 (prior "đã quyết") | Nêu là giới hạn của tầng QĐ-01; Q1 đo *crop + âm* chứ không đo mã | 0 |
| d | `khong_ro` giả định MCAR trong tầng | Báo số `khong_ro` theo tầng + cận "bảo thủ" (đã có cho T1; thêm cho tier nếu > 5 %) | 1 giờ |
| e | Trôi giữa buổi (04/08: 4 % → 16 % → 35 %) | Mỗi phiên ≤ 45 phút, 1 phiên/buổi; ước lượng in dwell/κ/mồi theo `session_id` để thấy trôi; nếu phiên 5 khác phiên 1 rõ → chấm lại phiên 5 | 0 |
| f | CHAR_A không đủ n để *chứng minh* ≥ 98 % | Mẻ phụ 160–450 ô CHAR_A cùng khung, chỉ khi cần con số đó | 1 giờ dựng + 30–60 phút chấm |
| g | Không có ô "trắng" (crop rỗng) làm mồi SAI CROP | Không thêm — ví dụ 5 trong hướng dẫn đã dạy; mồi âm đủ vai attention check | 0 |

---

## 2. Đã sửa trong mã (tất cả đã chạy lại, tất định)

| Tệp | Thay đổi | Kiểm |
|---|---|---|
| `pipeline/ground_truth/make_khoi_c_batch.py` (1.239 dòng) | hai pha (`_HTML`: `revealed()`, chặn Q2, `q1_blind`, `dwell_q1_ms`, `after_reveal`); `draw_t4_decoys` mồi dương phân tầng có trọng số; `draw_t6_nguoi`; `add_hidden_repeats` phân tầng + `lap_tang`; `N_T1` 300; `N_SESSIONS` 5; `LEAK_PATTERNS` +4; `_nguoi_leak_stats` vào `plan.json`; README mẻ mô tả hai pha; pilot dùng cùng HTML, lặp phân tầng nhỏ | dựng 2 lần: `KHOA.jsonl` `7cb21bafb604d4c9…`, `phien_1.html` `0d375e88e76406d0…`, `KHOA_pilot.jsonl` `efb7c119cb1ceb4a…`, `pilot_50.html` `13a998ba8bea10b2…` — **byte-identical**; `check_blind` 0/5 phiên |
| `pipeline/ground_truth/estimate_khoi_c.py` (1.126 dòng) | đọc `q1_blind` (ưu tiên) + báo động thế hệ cũ; mồi dương theo Q1; `kappa_bootstrap_ci` + `by_lap_tang` + Wilson đồng thuận; dwell pha 1 + số đổi sau lộ; pooled `QD01`, `USABLE_RE_DATASET`; `t6_report`; giả lập có `q1_blind`/`p_change_after_reveal` | giả lập trên mẻ thật (`--simulate`, nhiễu 5 %): 1.279/1.279, mồi 43/45 · 43/45, κ Q1 0,79 [0,64; 0,91], GOLD HT 92,0 % [89,0; 94,9], USABLE_RE_DATASET N = 70.326, T1 13/292 lỗi → cận trên 6,99 % → CHƯA đủ (đúng như gieo) |
| `pipeline/ground_truth/selftest.py` | +14 assertion (mồi dương có trọng số; T6; lặp phân tầng tất định; HTML hai pha; `_nguoi_leak_stats`; κ bootstrap; QD01/USABLE_RE_DATASET khớp tính tay; T6 HT; `q1_blind` ưu tiên khi 30 % ô đổi Q1 sau lộ; verdict cũ → báo động); test cũ đổi sang can thiệp cả `q1` lẫn `q1_blind` | **232/0**; `scripts/run_all_selftests.sh` **1.140/0** (mốc cập nhật) |
| `pipeline/ground_truth/README.md`, `docs/HUONG_DAN_CHAM_KHOI_C_2026-09-16.md`, `docs/BAO_CAO_KHOI_C_C1_2026-09-16.md` | mô tả hai pha, 5 phiên, đính chính C3 ở đầu báo cáo C1 | — |

Không đụng: `dataset_out/labels*.csv`, `re-dataset/`, `prepared/`, `docs/EVIDENCE_INDEX.md`, `BANG_SO_LIEU`. Phiên C3 không
commit; commit `ba1980fbb5` (17:33, "feat(khoi-c)…") được tạo **ngoài phiên này** khi mã đã ở trạng thái trên — nó chứa cả
`_khoa/KHOA.jsonl` (khoá ô → nguồn) trong lịch sử git, nên người chấm **không được** `git show`/mở `_khoa/` (đã ghi ở
HUONG_DAN §0). Sau commit đó chỉ còn khác: câu chữ README mẻ ("chấm TRƯỚC các phiên chính") và tài liệu này.

---

## 3. Thiết kế mẻ chốt (`dataset_out/human_audit/khoi_c_2026-09-16/`, seed 20260916)

Nguồn: `dataset_out/labels_final.csv` 83.239 dòng, sha256 `8da7da81fc9c1da2…` (= `CHECKSUMS.txt`, `BANG_SO_LIEU` md5
`1a2d8e0c…`). Crop: 880 từ tệp `dataset_out/{gold,syllable}/`, 399 cắt lại từ `prepared/<Sách>/pages/<page>.png` bằng
đúng công thức `build_dataset.save_crop` (REVIEW không có tệp; 40/40 kiểm byte-identical).

| Tầng | n | Dân số N | Cách rút | Trọng số | Đo gì |
|---|---:|---:|---|---|---|
| MAIN CHAR_A | 300 | 46.743 | SRS trong 6 tầng phụ (lớp cột eq/ne × box detector/legacy/mid), sàn 5 | 98–163 | precision Q1/Q2/Q1∧Q2 |
| MAIN CHAR_B | 100 | 3.017 | như trên | 6–38 | — |
| MAIN SYL | 150 | 18.554 | như trên (2 ô có mã vì tier GOLD/tier_v3 SYL) | 42–142 | Q1 (Q2 chỉ đếm) |
| MAIN REVIEW | 50 | 12.861 | như trên, crop cắt lại | 39–383 | đối chứng Q1 |
| T1 cổng B-3 | **300** | 1.025 gate_09 (1.023 khả dụng) | SRS 100/sách | 2,65 / 3,56 / 4,02 | lỗi ≤ 3 % (CP một phía) |
| T2 chuỗi trượt | 100 | 404 ô / 114 chuỗi | 24 chuỗi nguyên: ưu tiên chuỗi có QĐ-01 (8 ô), run ≥ 4 | cụm (NaN) | `sai_am` theo chuỗi |
| T3 B-2 đổi âm | 19 | 19 | census | 1 | âm cũ/mới đúng |
| T3 B-5 hộp lệch | 30 | 117 | SRS | 3,9 | hộp khoá đúng crop? |
| T4 mồi dương (QĐ-01) | 45 | **2.012** | SRS phân tầng tier_v3 (26/5/11/3) | 45,4 / 56,2 / 46,8 / 12,0 | attention (Q1) + tầng QĐ-01 |
| T4 mồi âm | 45 | — | CHAR_A sạch, hiện âm/mã ô kề | NaN | attention (Q1 ∧ Q2) |
| **T6 "người" chưa khoá** | **40** | **283** | SRS | 7,075 | pha loãng + QĐ-01a |
| T5 lặp ẩn | **100** | — | phân tầng MAIN 65 · T1 15 · T2 20, gap ≥ 100 | — | κ nội tại |
| **Tổng** | **1.279** | | 5 phiên: 256 · 256 · 256 · 256 · 255 | | |

Pilot (C-1) riêng: `pilot_50.html` 50 lượt = 40 MAIN (20/7/10/3) + 2 mồi dương + 3 mồi âm + 5 lặp (gap ≥ 15), ô không
trùng mẻ chính, seed 20260917, khoá `_khoa/KHOA_pilot.jsonl`.

Ánh xạ ô → nguồn chỉ trong `_khoa/` (54 trường); `manifest.jsonl` mang `labels_sha256` + `batch_sha256` để ước lượng từ chối
khi bộ nhãn hoặc khoá đổi.

---

## 4. Cách ước lượng (`estimate_khoi_c`, chạy sau từng phiên cũng được)

1. **Toàn vẹn**: sha256 khoá/labels khớp manifest; mọi `item_id` có trong khoá; `has_code` = `has_q2`; `source = human`
   (verdict giả lập/AI bị loại); có `q1_blind`.
2. **Mồi**: dương đạt ⇔ `q1_blind = dung`; âm đạt ⇔ `q1_blind ∈ {sai_am, sai_crop}` ∧ `q2 = sai`; báo động < 90 %.
3. **κ**: Cohen Q1 (4 mức) / Q1 nhị phân / Q2 (3 mức) + KTC bootstrap 95 % + đồng thuận thô Wilson + ma trận + theo tầng gốc;
   κ < 0,4 → báo động "thiết kế chưa ổn"; 0,4–0,8 → công bố kèm κ như giới hạn; ≥ 0,8 → ổn.
4. **Dwell**: p10/p50/p90 cả ô và pha Q1, ô < 1,5 s (cờ, > 10 % → báo động), đổi ý, đổi Q1 sau lộ mã.
5. **Precision**: Wilson theo 24 tầng; Horvitz–Thompson (FPC) theo tier_v3, GOLD, USABLE (68.314), QD01 (2.012),
   **USABLE_RE_DATASET (70.326)**; `khong_ro` loại khỏi mẫu số và báo số.
6. **T1**: cận trên CP một phía 95 % của lỗi (và bản "bảo thủ" KHÔNG RÕ = lỗi); ≤ 3 % → đủ điều kiện `visual_syl_gate`.
7. **T2/T3/T6/QĐ-01**: như §1(7), B-2 ba lớp, B-5 hộp khoá, T6 HT về 283, QĐ-01 nhìn lại mù.
8. Xuất `docs/KET_QUA_KHOI_C_<ngày>.md/.json`. **Không** tự ghi `BANG_SO_LIEU` — chỉ qua `evidence()` (A-13) sau khi người duyệt.

---

## 5. Việc người và thời gian

| Bước | Ai | Việc | Thời gian |
|---|---|---|---|
| 0 | quản lý mẻ | đọc `HUONG_DAN_CHAM_KHOI_C` §0–§4 một lần | 15 phút |
| 1 | người chấm | `pilot_50.html` → xuất → `estimate_khoi_c --pilot`; đạt ⇔ mồi 5/5, κ Q1 ≥ 0,4 (n = 5, chỉ là tín hiệu), dwell p50 3–12 s, ô < 1,5 s < 10 % | 5–8 phút + 1 phút chạy |
| 2–6 | người chấm | phiên 1…5, **mỗi phiên một buổi ≤ 45 phút**, không hai phiên liền; ước tính 6–9 s/ô (hai pha + Q2 ở 51 % ô) → 26–38 phút/phiên | tổng **2,2–3,2 giờ** chấm, rải **3–5 ngày** |
| 7 | quản lý mẻ | `estimate_khoi_c` sau mỗi phiên (xem trôi theo `session_id`); sau phiên 5 → `KET_QUA_KHOI_C` | 5 phút/lần |
| 8 (tuỳ chọn) | người thứ hai | 1 phiên (256 ô) để có κ liên người | 35–45 phút |

---

## 6. Điều kiện dùng kết quả

**Điều kiện chung (không đạt → không trích số nào):** mồi dương ≥ 90 % theo Q1 **và** mồi âm ≥ 90 %; κ Q1 ≥ 0,4 với KTC
bootstrap không chứa 0; ô < 1,5 s ≤ 10 %; `khong_ro` ≤ 10 % ở mỗi tier trích số; sha256 khoá/labels khớp; ≥ 95 % ô đã chấm
(bỏ sót chọn lọc làm mất hiệu lực CI).

**Cho B-3 (thêm rule `visual_syl_gate`, ≈ 894–961 ô REVIEW→SYL):** T1 cận trên CP một phía 95 % của lỗi Q1 ≤ 3 %
(**≤ 3 lỗi / 300**), và bản bảo thủ (KHÔNG RÕ = lỗi) cũng ≤ 3 % hoặc các ô KHÔNG RÕ được xem lại; đồng thời tỉ lệ
`sai_am` (lỗi *âm*) ≤ 2 %. Nếu chỉ `sai_crop` vượt → vấn đề hộp, không phải cổng: vẫn có thể thêm rule nhưng chỉ nhãn
âm, ghi rõ. Rớt → không thêm rule; giữ REVIEW; B-3 ở luận văn thành "đề xuất, chưa xác nhận".

**Cho B-2 (19 ô đổi âm):** census, không suy rộng. Trích ba con số `am_moi_dung / am_cu_dung / ca_hai_sai`; B-2 được coi
"có lợi" nếu `am_moi_dung > am_cu_dung` và `ca_hai_sai` ≤ 3; nếu không, λ = 0,25 giữ như đã chốt nhưng luận văn không
được nói "cải thiện" — chỉ "không hại trên 19 ô".

**Cho T2 (chuỗi trượt) và B-5:** chuỗi ≥ 50 % `sai_am` → hạ REVIEW theo chuỗi ở lần build sau (ghi `decisions`/evidence,
không sửa tay `labels`). B-5: nếu hộp khoá `dung` ≥ 90 % (Wilson dưới ≥ 80 %) thì giữ B-5; dưới đó → xem lại 117 ô bằng
mắt trước khi giữ.

**Cho luận văn (BANG_SO_LIEU qua `evidence()`):** trích **HT + CI** cho CHAR_A, CHAR_B, SYL (Q1), GOLD (Q1, Q2, Q1∧Q2),
USABLE_RE_DATASET (Q1), kèm: n chấm, số `khong_ro`, κ Q1 + KTC, mồi, dwell p50, tên tệp verdict + sha256 khoá + sha256
labels. Câu "≥ 98 %" chỉ được viết khi cận dưới CP một phía 95 % ≥ 0,98 (CHAR_A: ≤ 1 lỗi/300 hoặc mẻ phụ §1(4)); nếu không,
viết điểm ước lượng + CI, và nếu GOLD 93–96 % thì đó là số trung thực (DANH_MUC C-3). Q2 trên ô QĐ-01 **không** được
trích làm precision mã. Precision Q1 dùng `q1_blind`; số ô đổi Q1 sau lộ mã ghi kèm.

---

## 7. Tái lập

```bash
.venv/bin/python -m pipeline.ground_truth.make_khoi_c_batch            # ≈ 23 s, sha KHOA 7cb21bafb604d4c9…
.venv/bin/python -m pipeline.ground_truth.make_khoi_c_batch --pilot    # sha KHOA_pilot efb7c119cb1ceb4a…
.venv/bin/python -m pipeline.ground_truth.selftest                      # 232/0
bash scripts/run_all_selftests.sh                                       # 1140/0
.venv/bin/python -m pipeline.ground_truth.estimate_khoi_c --simulate /tmp/sim && \
.venv/bin/python -m pipeline.ground_truth.estimate_khoi_c --verdicts /tmp/sim --include-ai-verdicts --out-md /tmp/sim.md --out-json /tmp/sim.json
```

Số trong tài liệu này đo bằng `scratchpad/khoi_c/{leak1,leak2,decoy_neg,pop,cutcheck}.py` và `stats.py`
(`cp_upper_bound`, `acceptance_plan`, `required_n_for_halfwidth`); công thức SE(κ) là xấp xỉ mẫu lớn của Fleiss–Cohen–Everitt.
