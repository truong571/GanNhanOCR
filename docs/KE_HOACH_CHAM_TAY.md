# KIỂM ĐỊNH NHÃN BẰNG NGƯỜI — CƠ SỞ, THIẾT KẾ VÀ KẾ HOẠCH THỰC HIỆN

**Lập**: 2026-08-10 · **Dựng lại mẻ**: 2026-08-11 · **Dân số tham chiếu**: `dataset_out/labels_final.csv` (82.274 dòng)
**Phạm vi**: toàn bộ phần công việc bắt buộc phải do **người** thực hiện trong quy trình gán nhãn tự động.

> **Cập nhật 2026-08-11 — mẻ chấm đã được DỰNG LẠI, số khung mẫu đổi.** Bản 10/08 rút mẫu
> từ `labels_final.csv` của thế hệ 21/07, trong khi bộ thực sự xuất ra `dataset/` lại là
> `labels_remediated.csv` của lần chạy 22/07 — hai tập khác nhau, và bộ xuất ra còn mang
> 1.926 crop 㝵/'người' ở tier GOLD/SILVER dù lớp lỗi đó đã bị chứng minh sai hệ thống.
> Đã chạy lại `confusion_fix` trên thế hệ 22/07, xuất lại `dataset/`, nối bước confusion
> vào `run_pipeline.sh` (bước 5/6) để nhánh đó không tái diễn, rồi dựng lại mẻ 860 ô trên
> đúng khung mới. **Chưa ô nào bị chấm trước khi dựng lại**, nên không mất công chấm nào.

> Tài liệu này gom **mọi** hạng mục cần chấm tay vào một chỗ: vì sao cần, cần chấm cái gì,
> bao nhiêu ô, theo thiết kế mẫu nào, chấm theo quy tắc nào, và mỗi kết quả đi vào chương
> nào của luận văn. Dùng trực tiếp làm một mục trong chương *Đánh giá chất lượng bộ dữ liệu*.

> **👉 Mở file này để chấm** — `dataset_out/ground_truth/audit_combined/audit.html`
>
> **860 ô** — GOLD 250 · SILVER 300 · SYLLABLE 250 · 60 ô lặp ẩn — trong **một file duy
> nhất**, ba tier trộn chung và xáo trộn. Chấm xong bấm *Xuất verdicts.jsonl*, lưu vào cùng
> thư mục, rồi chạy `python -m pipeline.ground_truth.report_combined` để ra bảng kết quả.
> Chi tiết ở **Phần C** (thiết kế) và **Phần E** (quy trình).

---

## PHẦN A — VÌ SAO PHẢI CHẤM TAY

### A.1. Bộ dữ liệu hiện không có một nhãn nào do người tạo ra

Toàn bộ 82.274 nhãn trong `labels_final.csv` sinh ra từ **giao của ba tín hiệu máy**:

| Tín hiệu | Nội dung | Nguồn sai số |
|---|---|---|
| **S1** | OCR chữ Nôm (NomNaOCR / kinhhannom) | nhận dạng sai tự dạng |
| **S2** | Âm Quốc ngữ (VietOCR) + tra từ điển Nôm–Quốc ngữ | OCR bản dịch sai, từ điển thiếu mục |
| **S3** | So khớp thị giác crop ↔ glyph tham chiếu (ArcFace) | huấn luyện trên chính nhãn do S1∩S2 sinh |

Ba tín hiệu **đồng thuận** không đồng nghĩa với **đúng**: chúng chia sẻ chung ảnh đầu vào,
chung bộ từ điển và — với S3 — chung cả nhãn huấn luyện. Một lỗi hệ thống (ví dụ cặp
㝵 / 得) làm cả ba cùng sai theo một hướng, và đồng thuận khi đó chỉ **khuếch đại** lỗi
chứ không phát hiện được lỗi.

Hệ quả: **không tồn tại con số chất lượng nào của bộ dữ liệu mà pipeline có thể tự sinh ra.**
Đây không phải chuyện cầu toàn — đây là giới hạn logic của một quy trình khép kín.

### A.2. Ba bằng chứng đã đo được cho thấy metric máy không thay thế được người

| Bằng chứng | Số đo | Ý nghĩa |
|---|---|---|
| **S3 tự chấm điểm mình** | `precision = 0,9517` trong `s3_calibration.json` | Đo trên chính tập GOLD do S1∩S2 sinh ra → **vòng lặp tự xác nhận**, không có giá trị chứng minh |
| **S3 phát hiện lỗi thật** | **AUC = 0,566** [0,459 – 0,672] | Gần mức ngẫu nhiên (0,5). S3 là *bộ xếp hạng ứng viên*, **không** phải cổng kiểm lỗi |
| **NomNaOCR làm trọng tài thứ ba** | vô hiệu | Circular kép: mô hình được tinh chỉnh trên chính nhãn này, và các trang dùng để đánh giá đã nằm trong tập huấn luyện |
| **Verdict SILVER hiện có** | 750 dòng, 100% `source='ai_vision'` | Precision 72,98% là **AI chấm AI** — không được trình bày như kiểm định người |

Nói gọn: mỗi lần thay người bằng máy, con số thu được đều là máy tự khen mình. Chuỗi đó
chỉ cắt được bằng **một quan sát độc lập bên ngoài hệ thống** — tức người đọc bản scan gốc.

### A.3. Chấm tay tạo ra bốn thứ mà không cách nào khác tạo được

1. **Precision + khoảng tin cậy 95% cho từng tier** — con số headline của luận văn.
   Không có nó, mọi phát biểu "bộ dữ liệu chất lượng cao" đều là khẳng định không có căn cứ.
2. **Kiểm định chấp nhận (acceptance sampling)** — cho phép phát biểu ở dạng bảo vệ được:
   *"precision ≥ 97% với độ tin cậy một phía 95%"*, thay vì một con số trần trụi.
3. **Ma trận lỗi mô tả được** — biết lỗi **loại nào**, không chỉ **bao nhiêu**. Chính mẫu
   chấm tay vòng trước đã lộ ra nhầm lẫn hệ thống 㝵 ↔ 得 (Fisher p = 5,4·10⁻⁸), dẫn tới
   hạ cấp 1.926 crop. Đây là case study có giá trị khoa học của luận văn.
4. **Tập kiểm chứng độc lập để đánh giá mọi tín hiệu máy** — AUC của S3, của đầu ArcFace,
   ngưỡng hạ cấp… đều cần nhãn người làm mốc. Không có nó thì mọi ngưỡng đang dùng đều
   là **tiên nghiệm chưa kiểm định**.

### A.4. Yêu cầu bắt buộc từ bên ngoài

- **Công bố bộ dữ liệu**: chuẩn *Datasheets for Datasets* (Gebru et al., 2021) và tạp chí
  *Journal of Open Humanities Data* đều bắt buộc mục **"chất lượng nhãn được kiểm định
  bằng cách nào"**. Không có audit người → mục đó rỗng → hồ sơ không đạt.
- **Bảo vệ trước hội đồng**: câu hỏi *"căn cứ nào nói nhãn đúng 98%?"* chỉ có một câu trả
  lời đứng vững — *rút ngẫu nhiên n ô, người đọc bản scan gốc, đếm m lỗi, khoảng tin cậy
  Clopper–Pearson*. Mọi câu trả lời khác đều quy về "máy tự nói máy đúng".

### A.5. Vì sao là **chấm lại**, không phải chấm mới

Vòng chấm 03–04/08/2026 **đã thực hiện** nhưng kết quả **không tái lập được**. Chi tiết ở
Phần B. Tóm tắt: công cụ chấm cũ trộn hai câu hỏi khác bản chất vào một lần bấm, làm
precision GOLD dao động 95,8% ↔ 84,0% giữa hai buổi trên **cùng một dân số**. Vòng chấm
mới dùng công cụ đã sửa thiết kế, nên phải chấm lại từ đầu trên mẫu mới.

---

## PHẦN B — BÀI HỌC TỪ VÒNG CHẤM TRƯỚC (kiểm định độ tin cậy người chấm)

### B.1. Hiện tượng

Hai mẫu ngẫu nhiên độc lập, rút từ **cùng dân số GOLD**, chấm cách nhau một ngày:

| Buổi | n | Tỷ lệ lỗi tổng | riêng lỗi *"sai ảnh"* |
|---|---:|---:|---:|
| 03/08/2026 | 119 | 4,2 % | 0,8 % |
| 04/08/2026 | 25 | 16,0 % | 12,0 % |

Fisher exact trên chiều *"sai ảnh"*: **p = 0,017**. Dữ liệu không đổi — chỉ tiêu chí chấm
đổi. Suy ra precision GOLD là **95,8 %** hay **84,0 %** tuỳ buổi chấm, chênh 12 điểm phần
trăm — lớn hơn mọi cải tiến đang được tinh chỉnh ở các bước phía trên.

### B.2. Chẩn đoán — mẻ kiểm tra lặp (test–retest), n = 40

Trình bày lại 40 ô **đã từng chấm**, ẩn danh và xáo trộn, so verdict cũ ↔ mới:

| Chiều đánh giá | Cohen's κ | Chi tiết |
|---|---:|---|
| **Tổng hợp (4 mức)** | **0,128** | dưới ngưỡng 0,4 → *"chưa có con số precision nào bảo vệ được"* |
| Chiều **NHÃN** (*nhãn có đúng chữ không*) | 0,184 | **0/20 báo động giả** — không ô đúng nào bị gọi mới là sai |
| Chiều **CROP** (*khung cắt có sạch không*) | 0,140 | 8 gắn mới / 6 gỡ bỏ trên 40 ô — **đảo chiều cả hai hướng** |

**Kết luận chẩn đoán**: bất ổn nằm gần như trọn vẹn ở chiều CROP. Chiều NHÃN tuy κ thấp
(vì mẫu lệch nặng về lớp "đúng" — nghịch lý κ ở tỷ lệ nền cao) nhưng **sai lệch một
chiều và an toàn**: người chấm không bao giờ gắn lỗi mới cho ô đúng, chỉ có xu hướng
"gọi quá tay" rồi tự rút lại (5/6 lần bấm *"sai nhãn"* được đảo lại thành đúng ở lần sau).

Nguyên nhân gốc là **lỗi thiết kế nhiệm vụ**, không phải lỗi người chấm: câu hỏi *"crop có
sạch không"* **không có ngưỡng định lượng**. Bao nhiêu mực thừa của chữ bên cạnh thì tính
là hỏng? Một mẩu nét ở mép? Nửa chữ hàng xóm? Không có câu trả lời khách quan → mỗi buổi
người chấm tự đặt một ngưỡng khác.

### B.3. Đã sửa gì

| Vấn đề | Cách xử lý | Trạng thái |
|---|---|---|
| Chiều CROP không tái lập được | **Bỏ khỏi nhiệm vụ của người.** Chất lượng khung cắt nay **đo bằng hình học** trên toàn bộ 69.440 crop (`pipeline/ground_truth/crop_bleed.py`) | ✅ đã đo |
| Trộn hai câu hỏi | Công cụ chấm thêm chế độ `label_only` — **3 lựa chọn, một câu hỏi duy nhất** | ✅ đã cài |
| Không biết ngưỡng | Ngưỡng nay là tham số của thuật toán, tái lập 100%, không phụ thuộc buổi chấm | ✅ |

**Kết quả đo hình học (n = 69.440 crop, toàn bộ, `dataset_out/crop_bleed.csv`)**

| Chỉ số | Giá trị |
|---|---|
| Tỷ lệ mực dính từ chữ lân cận — trung vị | 0,087 |
| Tỷ lệ mực dính — phân vị 95 | 0,189 |
| Crop **hỏng kết cấu thật** (`detached_frac` > 0,5) | **14 ô = 0,02 %** |
| Crop nghi ngờ (`detached_frac` > 0,3) | 59 ô = 0,08 % |
| Kiểm tra tái lập md5 | 69.440 / 69.440 khớp |

→ Nỗi lo "crop cắt hỏng hàng loạt" **không có cơ sở** (0,02 %). Toàn bộ 12 điểm phần trăm
dao động precision là **nhiễu người chấm trên một câu hỏi không đo được**, chứ không phải
khiếm khuyết dữ liệu. Đây là một phát hiện đáng viết vào luận văn.

> **Câu để viết vào luận văn** — *Kiểm tra lặp cho thấy phán đoán "khung cắt có sạch không"
> của con người không đạt độ tin cậy tối thiểu (κ = 0,14). Chúng tôi đã loại tiêu chí này
> khỏi nhiệm vụ của người chấm và thay bằng phép đo hình học tất định trên toàn bộ 69.440
> crop; nhiệm vụ của người được thu về đúng một câu hỏi có thể tái lập.*

---

## PHẦN C — DANH MỤC VIỆC CẦN CHẤM TAY

### C.0. Bảng tổng hợp — gộp thành MỘT mẻ duy nhất

Toàn bộ việc chấm tay còn lại nằm trong **một thư mục, một file**:

```
dataset_out/ground_truth/audit_combined/audit.html      ← mở là chấm
dataset_out/ground_truth/audit_combined/verdicts.jsonl  ← lưu kết quả về đây
```

| # | Hạng mục | Khung rút mẫu | n | Mục đích |
|---|---|---:|---:|---|
| 1 | Precision nhãn tier **GOLD** | 48.395 | **250** | số headline của luận văn |
| 2 | Precision nhãn tier **SILVER** | 10.887 | **300** | tier chưa từng có kiểm định người |
| 3 | Precision tier **SYLLABLE** | 6.809 | **250** | tier chưa từng có verdict nào |
| 4 | **Ô lặp ẩn** (đo κ nội tại) | — | **60** | chứng minh tiêu chí chấm đã ổn định |
| | **Tổng — bạn chấm** | | **860 ô** | **≈ 5 – 7 giờ**, chia 6 buổi × ~150 ô |
| 5 | **κ liên người** → xem **C.4** | mẫu con của #1–3 | **100** | **người KHÁC** chấm, ~40 phút — đóng lỗ hổng "ai kiểm tra lại tác giả" |

**Vì sao gộp chứ không chấm ba mẻ riêng** — cả hai lý do đều là lý do thống kê, không phải
tiện lợi:

1. **Bỏ thiên lệch kỳ vọng.** Chấm riêng thì người chấm biết "mẻ này là SILVER, chắc nhiều
   lỗi". Kỳ vọng đó tác động thẳng vào chính đại lượng đang đo. Trộn chung và xáo trộn thì
   không ai biết mình đang ở tier nào.
2. **Bỏ trôi tiêu chí giữa các buổi.** Đây không phải giả định: mục B.1 đo được tỷ lệ lỗi
   trôi 4,2 % → 16 % → 35 % qua ba buổi trên **cùng một dân số**. Ba mẻ riêng = ba tiêu chí
   khác nhau, và khi đó chênh lệch precision giữa các tier **không phân biệt được** với
   chênh lệch của người chấm. Gộp lại thì cả ba tier chịu cùng một tiêu chí, cùng một
   trạng thái mệt mỏi — nên bảng so sánh tier mới có nghĩa.

### C.0.a. "Đa dạng" đúng cách — và cái bẫy phải tránh

Có hai loại đa dạng, chỉ một loại hợp lệ:

| | Cách làm | Hệ quả |
|---|---|---|
| ✅ **Hợp lệ** | Phân tầng **(tier × sách)**, phân bổ theo tỷ lệ dân số, rút **ngẫu nhiên** trong từng ô, ghi `design_weight = N_h/n_h` | Cả 3 sách và 3 tier chắc chắn có mặt, mà Horvitz–Thompson vẫn **không chệch** |
| ❌ **Sai** | Cố ý nhặt cho đủ mặt lớp chữ hiếm / các ca khó / các ô "trông đáng ngờ" | Mẫu **không còn đại diện**; Clopper–Pearson tính trên nó vô nghĩa; precision thu được thấp giả |

Mẻ này chỉ dùng loại thứ nhất — 9 tầng `(tier × sách)`, phân bổ tỷ lệ, SRS trong từng
tầng. **Độ phủ thực tế đo được** (ghi trong `plan.json`, trích thẳng vào bài báo):

| Chỉ số | Giá trị |
|---|---|
| Lớp chữ phân biệt trong mẫu | **259** |
| Trang scan khác nhau | **377** |
| Sách | **3/3** (stt2, stt4, stt11) |
| Tầng | 9, mỗi tầng 77 – 109 ô |

> Nếu vẫn muốn soi kỹ các ca khó cho phần *phân tích lỗi* của bài báo, cách đúng là rút
> một mẻ **CHỦ ĐÍCH riêng** với `design_weight` để trống — `estimate` tự động loại nó khỏi
> mọi phép tính precision. Tuyệt đối không trộn ca khó vào mẻ này.

### C.0.b. Ô lặp ẩn — lấy κ mà không cần thêm buổi chấm

**60 ô** được đưa vào mẻ **hai lần** với `item_id` khác nhau, cách nhau **tối thiểu 200 vị
trí** (đo được: khoảng cách thực tế 200 – 804). Người chấm không có cách nào biết ô nào là
ô lặp.

- So verdict hai lần → **Cohen's κ nội tại** ngay trong chính mẻ này. Không phải tổ chức
  buổi test–retest riêng, và không dính lại đúng cái bẫy "hai buổi hai tiêu chí" của vòng
  trước.
- 60 ô lặp phủ cả ba tier (GOLD 19 · SILVER 22 · SYLLABLE 19).
- Ô lặp mang `design_weight` rỗng → `estimate._split_purposive` **tự động loại** khỏi mọi
  phép tính precision. Không có nguy cơ đếm hai lần.

**Ngưỡng diễn giải κ** (chốt **trước** khi chấm, không đổi sau):

| κ | Kết luận |
|---|---|
| **≥ 0,8** | Tiêu chí ổn định → precision công bố được |
| **0,4 – 0,8** | Công bố được nhưng **phải nêu κ kèm theo** như một giới hạn |
| **< 0,4** | Thiết kế vẫn chưa ổn → viết rubric chi tiết hơn rồi chấm lại |

⚠️ κ nhạy với tỷ lệ nền: khi gần như mọi ô đều "đúng" thì κ thấp **dù** đồng thuận thô rất
cao. `report_combined` luôn in κ **cạnh** đồng thuận thô và ma trận đảo verdict — trích cả
ba, đừng trích κ một mình.

### C.0.c. Những gì **đã** chấm — dùng được đến đâu

| Mẻ | n | Ngày | Kết quả | Dùng được? |
|---|---:|---|---|---|
| `verdicts_reanchored.csv` | 846 | 07/2026 | precision GOLD 97,08 % (trước hạ cấp) / 98,00 % (sau) | ⚠️ Con số 98,00 % là **post-hoc** — tính lại trên chính mẫu đã dùng để phát hiện lỗi. Trình bày được như *trình tự phát hiện*, **không** như tuyên bố acceptance |
| `audit_gold_human/` | 200 (120 SRS + 80 chủ đích) | 03–04/08/2026 | precision 97,4 %; **riêng chiều NHÃN 98,3 %** [93,9 – 99,8] | ⚠️ Chỉ **chiều NHÃN** dùng được. Chiều CROP đã bị bác bỏ (Phần B) |
| `audit_retest/` | 40 | 04/08/2026 | κ = 0,128 | ✅ Dùng được — làm **bằng chứng chẩn đoán** ở Phần B, không dùng ước lượng precision |
| `audit_confusion_奴_nó/` | 55 | 04/08/2026 | đối chứng lớp 奴 | ✅ Dùng được |
| `audit_SILVER/verdicts_ai.jsonl` | 750 | — | 72,98 % | ❌ **100 % `source='ai_vision'`** — không phải kiểm định người |
| `audit_gold/`, `audit_SYLLABLE/*.jsonl` | 846 / 299 | — | verdict máy (deepseek) | ❌ Không phải kiểm định người |

> 🔒 **Chốt an toàn đã cài**: `pipeline/ground_truth/estimate.py` mặc định **loại bỏ** mọi
> verdict `source='ai_vision'` và báo lỗi lớn nếu toàn bộ đầu vào là AI. Muốn dùng verdict
> máy phải khai `--include-ai-verdicts` tường minh. Nghĩa là **không thể vô tình** biến
> nhãn máy thành ground truth.

---

### C.1. Thiết kế mẫu chi tiết

| | |
|---|---|
| **Nguồn nhãn** | `dataset_out/labels_final.csv` — **bộ công bố**, không phải thế hệ trung gian |
| **Thiết kế** | Phân tầng `(tier × sách)`, phân bổ theo tỷ lệ dân số, **SRS** trong từng tầng |
| **Trọng số** | `design_weight = N_h / n_h` ghi sẵn từng hàng → Horvitz–Thompson có FPC |
| **Chế độ chấm** | `label_only` — **3 lựa chọn**, một câu hỏi duy nhất |
| **Loại trừ** | 498 ô đã chấm ở các mẻ trước (1,0 % GOLD) — tránh nhiễm trí nhớ |
| **Hạt giống** | `seed = 2026` → chạy lại cho ra **đúng cùng một mẫu** |

**Chín tầng:**

| Tầng | N_h | n_h | design_weight |
|---|---:|---:|---:|
| GOLD × stt2 / stt4 / stt11 | 48.395 tổng | 83 / 83 / 84 | ≈ 192 – 194 |
| SILVER × stt2 / stt4 / stt11 | 10.887 tổng | 111 / 103 / 86 | ≈ 36,2 – 36,4 |
| SYLLABLE × stt2 / stt4 / stt11 | 6.809 tổng | 81 / 76 / 93 | ≈ 27,1 – 27,4 |

**Khoảng tin cậy mỗi tier sẽ đạt được** (độ tin cậy 95 %, Clopper–Pearson):

*GOLD — n = 250*

| Số ô sai nhãn | Precision | CI 95 % | Cận dưới một phía |
|---:|---:|---|---:|
| 0 | 100,0 % | [98,5 % · 100,0 %] | 98,8 % |
| 1 | 99,6 % | [97,8 % · 100,0 %] | 98,1 % |
| 2 | 99,2 % | [97,1 % · 99,9 %] | 97,5 % |
| 3 | 98,8 % | [96,5 % · 99,8 %] | 96,9 % |
| 5 | 98,0 % | [95,4 % · 99,3 %] | 95,8 % |
| 8 | 96,8 % | [93,8 % · 98,6 %] | 94,3 % |

Nếu ≤ 3 ô sai, phát biểu *"precision GOLD ≥ 96,9 % (một phía 95 %)"* đứng vững. So với mẻ
người trước (n = 116, CI [93,9 % · 99,8 %]) — thu hẹp khoảng tin cậy khoảng một nửa.

*SILVER — n = 300.* Precision kỳ vọng ~0,75 (ước lượng AI 72,98 %), tức vùng phương sai
lớn nhất. Ở p ≈ 0,75, nửa độ rộng ±5 % cần n = 287 → chọn 300 (CI ≈ [69,8 % · 79,6 %]).
Muốn ±3 % cần n = 799 — không khuyến nghị, chi phí gấp 2,7 lần cho một tier phụ.

> **Nếu SILVER thật sự thấp (< 85 %) thì đó KHÔNG phải thất bại** — đó là kết quả có giá
> trị: nó chứng minh việc phân tầng GOLD/SILVER là **có căn cứ đo được**, và biện minh cho
> khuyến nghị *"dùng GOLD để huấn luyện, SILVER chỉ để tiền huấn luyện / tăng cường"* trong
> datasheet.

*SYLLABLE — n = 250.* Ở p ≈ 0,90, ±5 % cần n = 141 → chọn 250 cho CI ≈ ±4 %.

### C.2. Tier SYLLABLE — câu hỏi khác, thẻ trông khác

269 thẻ trong mẻ (250 ô mẫu + 19 ô lặp) **không có ảnh "glyph tham chiếu"**: hàng tier
SYLLABLE có `label = null`, `unicode = null` — nhãn của chúng là **âm tiết Quốc ngữ**, không
phải một chữ Nôm cụ thể. Công cụ chấm đã xử lý riêng: ô chữ lớn hiển thị **âm tiết**, kèm
dòng nói rõ *"nhãn ở ô này là ÂM TIẾT"*, và bộ **ứng viên từ điển** (các chữ Nôm mà từ điển
gắn với âm đó) làm căn cứ đối chiếu.

> **Câu hỏi cho các thẻ đó:**
> **"Chữ Nôm trong ô này có đọc là âm tiết ‹X› không?"**
>
> **1** đúng · **2** sai (đọc là âm khác) · **3** không đọc được

Nhiệm vụ này **khó và chậm hơn** hai tier kia vì không có ảnh để so trực tiếp. Đừng bấm
**2** chỉ vì thiếu glyph đối chiếu — không đủ căn cứ thì bấm **3**.

### C.3. Hai lỗi công cụ đã phát hiện và sửa trước khi dựng mẻ

Cả hai nằm trong `audit_grid.py` và **đều đủ sức làm hỏng toàn bộ buổi chấm**. Ghi lại vì
đây là một phần của lịch sử kiểm định, và phần *Giới hạn* của bài báo nên nhắc tới.

| Lỗi | Biểu hiện | Hậu quả nếu không sửa |
|---|---|---|
| **Nút chấm bị hardcode 4 mức** | Chế độ `label_only` chỉ đổi phần chú thích phím, còn thân hàm dựng thẻ vẫn in cứng 4 nút. Nút *"3 · sai ảnh"* thật ra ghi verdict `unsure`; nút *"4 · không chắc"* ghi `undefined` | Dòng xuất ra **mất hẳn trường `verdict`** → `estimate` ném lỗi; và mọi ô bấm nút 3 bị dán nhãn sai ý định. Toàn bộ mẻ 250 ô của `audit_label_only/` sẽ **không dùng được** |
| **Thẻ âm tiết hiển thị chữ `nan`** | `str(NaN or "")` trả về `"nan"` vì NaN là *truthy* trong Python → 269 thẻ in chữ **nan** cỡ 56 px như thể đó là nhãn đề xuất | Người chấm không hiểu đang được hỏi gì trên 31 % số ô |

Ngoài ra hai cải thiện về vận hành: (a) bộ dựng mẻ trước đây giữ **mọi** trang scan đã mở
trong RAM (~13,7 MB/trang → mẻ 860 ô chạm 362 trang sẽ cần nhiều GB) — nay duyệt theo trang
và chỉ giữ một trang; (b) thêm `content-visibility:auto` để file 43 MB không treo trình
duyệt khi mở.

---

### C.4. Giai đoạn 2 — κ LIÊN NGƯỜI (100 ô, người chấm thứ hai) 🔴

**Làm sau khi chấm xong 860 ô.** Đây là hạng mục đóng lỗ hổng phương pháp lớn nhất còn lại.

#### Vì sao κ nội tại chưa đủ

60 ô lặp ẩn đo được người chấm có **tự nhất quán** không. Chúng **không** trả lời được câu
hỏi hội đồng chắc chắn hỏi:

> *"Người chấm chính là tác giả của bộ dữ liệu. Ai kiểm tra lại?"*

Một người hoàn toàn tự nhất quán vẫn có thể **sai hệ thống theo cùng một hướng** ở cả hai
lần chấm — κ nội tại cao **không** loại trừ khả năng đó. Chỉ người thứ hai, chấm độc lập và
mù với verdict của người thứ nhất, mới tách được *"tiêu chí ổn định"* khỏi *"tiêu chí ổn
định nhưng lệch"*.

Đây cũng là hạng mục rẻ nhất trên mỗi đơn vị giá trị bảo vệ: **100 ô ≈ 40 phút** của một
người khác.

#### Thiết kế

| | |
|---|---|
| **n** | 100, rút **ngẫu nhiên phân tầng theo tier** (phân bổ tỷ lệ) từ chính 800 ô người thứ nhất đã chấm |
| **Vì sao ngẫu nhiên thuần** | κ **đọc thẳng được**, không phải hiệu chỉnh theo tỷ trọng dân số. Mẻ kiểm tra lặp cũ phải lấy vượt tỷ lệ ô lỗi vì lỗi quá hiếm; ở đây tier SILVER (~25 % lỗi kỳ vọng) đã cung cấp sẵn ~10–15 ô đủ khả năng gây bất đồng |
| **Mù** | `orig_verdict` đi vào manifest để ghép cặp nhưng nằm trong `_HIDDEN_FIELDS` → **không bao giờ** lọt vào HTML (đã kiểm chứng: 0 lần xuất hiện trong file) |
| **Người chấm thứ hai** | Không cần biết gì về bộ dữ liệu, và tốt nhất là không biết. Nhận cùng rubric ở Phần D. **Tuyệt đối không hỏi người thứ nhất đã chấm gì** |
| **Không được dùng** | ❌ ước lượng precision — đây là mẫu con của mẫu đã chấm, dùng lại sẽ đếm hai lần |

#### Đại lượng đọc

`report_combined --interrater` sinh `KAPPA_LIEN_NGUOI.md` gồm:

1. **Cohen's κ** + đồng thuận thô kèm CI Wilson
2. **Đồng thuận có điều kiện** — bền với thiết kế mẫu, và là thứ cứu được tình huống κ
   thấp: nếu *"người 1 gọi nhãn SAI"* mà người 2 đồng ý ở tỷ lệ cao, thì các ô bị tính là
   lỗi là **lỗi thật**, không phải người 1 gọi quá tay
3. **Bảng chồng lấn lỗi** — hai người có bắt cùng những ô lỗi không, hay mỗi người bắt một
   nhóm khác
4. Ma trận bất đồng + phân rã theo tier

**Thang Landis & Koch (1977)** — chuẩn trích dẫn được: κ 0,41–0,60 trung bình ·
**0,61–0,80 tốt** · 0,81–1,00 rất tốt.

> **Nếu κ thấp thì vẫn phải báo.** Bảng đồng thuận có điều kiện sẽ chỉ ra bất đồng nằm ở
> nhóm nào (thường là ranh giới *"không đọc được"* vs *"đúng"*), và điều đó đưa vào phần
> *Giới hạn* — trung thực hơn nhiều so với việc chỉ có κ nội tại rồi im lặng về câu hỏi
> "ai kiểm tra lại".

---

## PHẦN D — QUY TẮC CHẤM (RUBRIC CHÍNH THỨC)

> Rubric là **một phần của phương pháp nghiên cứu** — chép nguyên vào phụ lục luận văn.

### D.1. Câu hỏi duy nhất (GOLD, SILVER)

> **Nhãn được gán có đúng là chữ viết trong ô này không?**

| Phím | Verdict | Khi nào chọn |
|:---:|---|---|
| **1** | **nhãn ĐÚNG** | Chữ trong ô đúng là chữ được gán |
| **2** | **nhãn SAI** | Chữ trong ô là một chữ **khác** |
| **3** | **không đọc được** | Không đủ căn cứ để kết luận, kể cả sau khi xem ảnh ngữ cảnh |

### D.2. Bốn quy tắc bắt buộc

1. **Khung cắt xấu KHÔNG phải lỗi nhãn.** Crop dính chút mực của chữ bên cạnh mà vẫn đọc
   ra chữ → **nhãn ĐÚNG** (nếu nhãn khớp). Chất lượng khung đã được đo riêng bằng hình học.
   *Câu hỏi ở đây là về **chữ**, không phải về **khung**.*
2. **Crop khó nhìn thì dùng ảnh ngữ cảnh** — khung bbox đỏ trên trang scan gốc, luôn hiển thị.
3. **Lưỡng lự thì chọn "không đọc được", đừng chọn "nhãn SAI".** Ô *không đọc được* bị
   **loại khỏi mẫu số** (không tính là lỗi); ô *nhãn SAI* bị **tính là lỗi**. Dữ liệu vòng
   trước cho thấy xu hướng **gọi quá tay**: 5/6 lần bấm "sai nhãn" thì lần sau chính người
   chấm đảo lại thành đúng.
4. **Chấm hết mọi ô theo đúng thứ tự, không bỏ ô khó.** Bỏ chọn lọc phá tính ngẫu nhiên của
   mẫu và làm mọi khoảng tin cậy mất hiệu lực. Ô khó → bấm **3**, đừng bỏ qua.

### D.3. Kỷ luật thực hiện

| Quy tắc | Lý do |
|---|---|
| Tối đa **150 ô/buổi** | Trôi tiêu chí đã đo được: 4,2 % → 16 % → 35 % qua ba buổi |
| Giữa hai mẻ nghỉ **≥ 24 giờ** | Tránh nhiễm trí nhớ khi kiểm tra lặp |
| **Không xem** tier / rule / điểm S3 khi chấm | Công cụ đã ẩn sẵn — mọi trường gây thiên lệch chỉ ghi vào `manifest.jsonl` để ghép lại sau |
| Ở mẻ kiểm tra lặp: **đừng cố nhớ lần trước** | Nếu cố nhớ, phép đo chỉ đo trí nhớ chứ không đo tiêu chí. **Đổi ý là dữ liệu quý, không phải lỗi** |
| Chấm xong bấm **Download JSON** → lưu vào **đúng thư mục mẻ đó** | `estimate` đọc verdict theo thư mục |

---

## PHẦN E — QUY TRÌNH THỰC HIỆN

### Bước 1 — Chấm (mẻ đã dựng sẵn, mở là chạy)

```bash
open dataset_out/ground_truth/audit_combined/audit.html
```

Một file duy nhất, **43 MB**, mọi ảnh nhúng sẵn — **không cần mạng**. Lần mở đầu mất
khoảng 10–20 giây; sau đó chỉ những thẻ trong tầm nhìn mới được dựng nên cuộn vẫn mượt.

- Bấm **1** / **2** / **3**, hoặc bấm chuột. `←` `→` chuyển ô.
- Tiến độ **tự lưu vào trình duyệt** (localStorage, khoá riêng cho mẻ này) — đóng tab rồi
  mở lại vẫn còn. Chấm nhiều buổi trên **cùng một trình duyệt, cùng một máy**.
- Chấm xong toàn bộ 860 ô → bấm **Xuất verdicts.jsonl** → lưu vào
  `dataset_out/ground_truth/audit_combined/verdicts.jsonl`.

> Có thể xuất giữa chừng để phòng hờ (file chỉ chứa những ô đã chấm). `report_combined` sẽ
> báo rõ còn bao nhiêu ô chưa chấm và cảnh báo rằng số liệu chỉ là tạm thời.

### Bước 2 — Sinh bảng kết quả

```bash
.venv/bin/python -m pipeline.ground_truth.report_combined \
    --dir dataset_out/ground_truth/audit_combined
```

Sinh ra hai file ngay trong thư mục đó:

| File | Nội dung |
|---|---|
| `BANG_KET_QUA.md` | Bảng precision/CI theo từng tier · quyết định chấp nhận · κ nội tại + ma trận đảo verdict — **dán thẳng vào luận văn** |
| `report.json` | Toàn bộ số liệu dạng máy đọc, kèm phân rã theo từng sách |

Ngưỡng chấp nhận mặc định: GOLD `p0 = 0,97` · SILVER `p0 = 0,85` · SYLLABLE `p0 = 0,85`.
Đổi bằng `--p0-gold` / `--p0-silver` / `--p0-syllable`.

### Bước 3 — κ liên người (sau khi bước 1 xong)

```bash
# dựng mẻ 100 ô cho người chấm thứ hai
.venv/bin/python -m pipeline.ground_truth.make_interrater_batch --n 100

# đưa cả thư mục audit_interrater/ cho người thứ hai (họ đọc README.md trong đó,
# mở audit.html, chấm, xuất verdicts.jsonl vào lại thư mục đó)

.venv/bin/python -m pipeline.ground_truth.report_combined \
    --interrater dataset_out/ground_truth/audit_interrater
```

→ `KAPPA_LIEN_NGUOI.md` + `interrater.json`. Xem Phần C.4 để biết đọc con số thế nào.

### Trang demo để chiếu trước hội đồng

```bash
.venv/bin/python -m pipeline.ground_truth.make_demo_page      # đổi --seed nếu thẻ render xấu
open dataset_out/ground_truth/demo_audit.html
```

**442 KB, 9 thẻ** (3 mỗi tier), mở tức thì. Khác công cụ chấm thật ở ba điểm có chủ đích:
thẻ demo **hiện** các trường bị giấu (đóng khung đỏ nhạt "người chấm KHÔNG thấy"), không
bấm chấm được, và rút từ các ô **ngoài** mẻ chấm thật nên chiếu bao nhiêu lần cũng không
làm bẩn mẫu đang đo. Rút ngẫu nhiên có hạt giống cố định — ghi thẳng trên trang, để không
ai hỏi được *"thầy chọn mấy ô đẹp phải không"*.

**Đừng mở `audit.html` (43 MB) trong buổi bảo vệ** — tải chậm, và chiếu công cụ chấm lên
màn hình dễ gây ấn tượng "tự làm tự chấm". Chiếu trang demo, rồi để file thật làm phụ lục
tái lập.

### Nếu cần dựng lại mẻ (chỉ khi thật sự phải)

```bash
.venv/bin/python -m pipeline.ground_truth.make_combined_batch \
    --n-gold 250 --n-silver 300 --n-syllable 250 --n-repeat 60 --seed 2026
```

Cùng `seed` → **đúng cùng một mẫu**, tái lập được 100 %. Dựng lại mất ~10 giây.

> ⚠️ Chạy lại sau khi mẻ đã tồn tại sẽ **loại chính các ô của mẻ cũ** (cơ chế chống chấm
> trùng quét mọi `manifest.jsonl` dưới `ground_truth/`), nên mẫu sẽ **khác**. Muốn tái lập
> đúng mẻ hiện tại thì phải xoá thư mục `audit_combined/` trước.

---

## PHẦN F — KẾT QUẢ (điền sau khi chấm) VÀ NƠI SỬ DỤNG

### F.1. Bảng kết quả — bảng chính thức của chương đánh giá

**Không cần điền tay** — `report_combined` sinh đúng bảng này vào `BANG_KET_QUA.md` với số
liệu thật. Khung dưới đây chỉ để biết bảng sẽ có hình dạng nào:

| Tier | N (khung rút mẫu) | n (chấm được) | Đúng | Sai nhãn | Không đọc được | **Precision** | **CI 95 %** | Cận dưới một phía |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| GOLD | 48.395 | ≤ 250 | | | | | | |
| SILVER | 10.887 | ≤ 300 | | | | | | |
| SYLLABLE | 6.809 | ≤ 250 | | | | | | |
| **Toàn tập có nhãn** | **66.091** | ≤ 800 | | | | *(Horvitz–Thompson có FPC)* | | |

*Cột `n` là số ô **chấm được** — các ô "không đọc được" bị loại khỏi mẫu số và báo riêng,
nên `n` luôn ≤ cỡ mẫu thiết kế.*

**Độ tin cậy người chấm** — κ nội tại từ 60 ô lặp (báo kèm đồng thuận thô và ma trận đảo
verdict): ` ` (do `report_combined` điền)

### F.2. Con số nào đi vào chỗ nào

| Kết quả | Vị trí trong luận văn |
|---|---|
| Precision + CI từng tier (F.1) | Chương *Đánh giá chất lượng bộ dữ liệu* — bảng chính |
| κ nội tại + đồng thuận thô + ma trận đảo verdict | Cùng chương, mục *Độ tin cậy của quy trình kiểm định* |
| **κ liên người** (C.4) + đồng thuận có điều kiện | Cùng mục — đây là câu trả lời cho *"ai kiểm tra lại tác giả"* |
| Thiết kế mẫu C.0 → C.0.b (gộp tier để khử thiên lệch kỳ vọng; phân tầng thay vì "chọn cho đa dạng"; ô lặp ẩn) | Chương *Phương pháp*, mục *Thiết kế lấy mẫu kiểm định* |
| Độ phủ mẫu (253 lớp chữ · 362 trang · 3 sách) | Cùng chương — chứng minh mẫu không dồn vào một góc dữ liệu |
| Phần B (κ = 0,13 → chẩn đoán → sửa thiết kế) | Chương *Phương pháp*, mục *Thiết kế nhiệm vụ kiểm định* — **đây là đóng góp phương pháp luận, không phải thất bại cần giấu** |
| Đo hình học crop (14/69.440 = 0,02 %) | Cùng chương — dẫn chứng cho việc thay phán đoán người bằng phép đo |
| Case study 㝵/người (Fisher p = 5,4·10⁻⁸ → hạ cấp 1.926 crop) | Chương *Kết quả*, mục *Phát hiện nhầm lẫn hệ thống* |
| Precision SILVER (dù thấp) | Datasheet + khuyến nghị sử dụng theo tier |
| Toàn bộ Phần A + D | Phụ lục *Phương pháp kiểm định nhãn* |

### F.3. Cập nhật bắt buộc sau khi có kết quả

- [ ] Ghi số mới vào `docs/BANG_SO_LIEU_CHINH_THUC.md` §2 và §3 (nguồn số liệu duy nhất
      của luận văn — **mọi chương chỉ được trích từ file đó**)
- [ ] Cập nhật cột *"Nguồn kiểm định precision"* trong bảng tier: SILVER 🤖 → 👤, SYLLABLE ⚪ → 👤
- [ ] Cập nhật `docs/EVIDENCE_INDEX.md` — gỡ mục *"Verdict SILVER là AI chấm"*
- [ ] Cập nhật mục *label quality* trong datasheet (`pipeline/publish/`)
- [ ] Gỡ cảnh báo *"98,00 % là post-hoc"* nếu mẫu SRS mới xác nhận

---

## PHẦN G — GHI CHÚ TRUNG THỰC BẮT BUỘC (viết nguyên văn vào luận văn)

1. **Người chấm chính là tác giả bộ dữ liệu.** Đây là xung đột lợi ích phải nêu thẳng, và
   được bù bằng ba thứ: nhiệm vụ **mù** (đo được: GOLD và SILVER không phân biệt được qua
   thẻ, 88,4 % vs 86,0 %), rubric **chốt trước khi chấm**, và κ nội tại từ ô lặp ẩn.
   Nhưng κ nội tại **không** loại trừ được sai hệ thống một chiều → **phải làm C.4 (κ liên
   người, 100 ô, ~40 phút của người khác)**. Nếu vì lý do nào đó không làm được, phải ghi
   nguyên văn vào phần *Giới hạn* rằng độ đồng thuận liên người **chưa được đo**.
2. **Con số 98,00 % hiện tại là post-hoc.** Tính lại trên chính mẫu đã dùng để phát hiện lỗi
   㝵/người. Trước khi có mẫu SRS xác nhận mới, phải trình bày theo **trình tự**:
   *audit → phát hiện (Fisher p = 5,4·10⁻⁸) → hạ cấp 1.926 crop → tính lại*, chứ không
   trình bày như một tuyên bố acceptance độc lập.
3. **Precision không phải recall.** Mọi con số ở đây trả lời *"nhãn đã gán có đúng không"*.
   Câu hỏi *"còn bao nhiêu chữ trên trang chưa được gán nhãn"* (recall) **chưa được đo** và
   cần một thiết kế mẫu khác — lấy mẫu theo **trang**, không theo **crop**.
4. **Phạm vi hẹp.** 3 sách, 445 trang, **một nét chữ duy nhất**. Mọi con số chỉ có giá trị
   trong phạm vi đó; **không** ngoại suy sang văn bản Nôm nói chung.
5. **Verdict AI không bao giờ được trộn vào.** `estimate.py` đã chặn ở mức code, nhưng phải
   nêu trong phần phương pháp rằng chốt chặn này tồn tại và vì sao.

---

### Phụ lục — tệp và mã nguồn liên quan

| Thành phần | Đường dẫn |
|---|---|
| **Mẻ chấm (mở file này)** | `dataset_out/ground_truth/audit_combined/audit.html` |
| **Trang demo để chiếu hội đồng** | `dataset_out/ground_truth/demo_audit.html` (442 KB) |
| **Dựng mẻ gộp** | `pipeline/ground_truth/make_combined_batch.py` |
| **Dựng mẻ κ liên người** | `pipeline/ground_truth/make_interrater_batch.py` |
| **Dựng trang demo** | `pipeline/ground_truth/make_demo_page.py` |
| **Sinh bảng kết quả + κ** | `pipeline/ground_truth/report_combined.py` (`--interrater` cho κ liên người) |
| Công cụ chấm mù (HTML) | `pipeline/ground_truth/audit_grid.py` — `CHOICE_SETS['label_only']` |
| Đo hình học khung cắt | `pipeline/ground_truth/crop_bleed.py` → `dataset_out/crop_bleed.csv` |
| Ước lượng precision + CI (một tier) | `pipeline/ground_truth/estimate.py` |
| Thư viện thống kê | `pipeline/ground_truth/stats.py` — Wilson · Clopper–Pearson · acceptance · Horvitz–Thompson (có FPC) · PPI |
| Dựng mẻ một tier / mẻ kiểm tra lặp riêng | `make_label_batch.py` · `make_retest_batch.py` |
| Nguồn số liệu duy nhất | `docs/BANG_SO_LIEU_CHINH_THUC.md` |
| Kiểm định toàn bộ | `bash scripts/run_all_selftests.sh` → **392 passed, 0 failed** (mốc 11/08; +32 là test cho `report_combined`) |
