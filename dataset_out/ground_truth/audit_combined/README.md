# Mẻ audit GỘP — GOLD + SILVER + SYLLABLE, một dòng chấm duy nhất

**860 ô** trong **một file** `audit.html`
= 800 ô mẫu + 60 ô lặp ẩn.
Nguồn nhãn: `dataset_out/labels_final.csv` (bộ **công bố**).

## Chỉ có MỘT câu hỏi

> **Nhãn hiện trên thẻ có đúng là chữ viết trong ô không?**

| Phím | Nghĩa |
|:---:|---|
| **1** | **nhãn ĐÚNG** — chữ trong ô đúng là chữ được gán |
| **2** | **nhãn SAI** — chữ trong ô là một chữ KHÁC |
| **3** | **không đọc được** — không đủ căn cứ, kể cả sau khi xem ảnh ngữ cảnh |

Lựa chọn "sai ảnh" đã bị **bỏ hẳn**. Kiểm tra lặp đo được κ = 0,14 cho phán đoán "crop có
sạch không" — không tái lập được, và chính nó làm precision GOLD nhảy 95,8% ↔ 84,0% giữa
hai buổi. Chất lượng khung cắt nay đo bằng hình học trên toàn bộ 69.440 crop (chỉ 14 ô =
0,02% hỏng kết cấu thật).

**Bốn quy tắc:**

1. Khung cắt xấu **không phải** lỗi nhãn. Dính chút mực chữ bên cạnh mà vẫn đọc ra chữ →
   bấm **1**. Câu hỏi là về **chữ**, không phải về **khung**.
2. Crop khó nhìn thì đọc bằng **ảnh ngữ cảnh** (khung đỏ trên trang scan).
3. Lưỡng lự → bấm **3**, đừng bấm **2**. Ô *không đọc được* bị loại khỏi mẫu số; ô
   *nhãn SAI* bị tính là lỗi. Dữ liệu cũ cho thấy xu hướng **gọi quá tay**: 5/6 lần bấm
   "sai nhãn" thì lần sau chính bạn đảo lại thành đúng.
4. Chấm **hết** theo đúng thứ tự, không bỏ ô khó — bỏ chọn lọc phá tính ngẫu nhiên và
   làm mọi khoảng tin cậy mất hiệu lực.

## Có ô KHÔNG có glyph tham chiếu — đó là tier âm tiết

Một phần ô hiện dấu **?** ở ô chữ lớn và không có ảnh "glyph tham chiếu". Đó là các hàng
mang nhãn **âm tiết Quốc ngữ** chứ không phải một chữ Nôm cụ thể. Với những ô này câu hỏi
đổi thành:

> **Chữ Nôm trong ô này có đọc là âm tiết ‹âm› hiển thị trên thẻ không?**

Dùng dòng **ứng viên** (các chữ Nôm mà từ điển gắn với âm đó) làm căn cứ. Vẫn ba phím như
trên. Đừng bấm **2** chỉ vì không có glyph để so — không đủ căn cứ thì bấm **3**.

## Ba tier trộn chung và xáo trộn — cố ý

Bạn **không** biết ô đang chấm thuộc tier nào, và điều đó là chủ đích: chấm riêng từng
tier thì kỳ vọng "mẻ này chắc nhiều lỗi" trở thành thiên lệch tác động thẳng vào con số
đang đo. Trộn chung còn bảo đảm cả ba tier chịu **cùng một tiêu chí trong cùng một buổi**,
nên bảng so sánh giữa các tier mới có nghĩa.

Có một số ô **xuất hiện hai lần**, cách nhau ít nhất 200 vị trí. Đừng
tìm chúng và đừng cố nhớ đã bấm gì — chúng dùng để đo độ ổn định của chính bạn. **Đổi ý là
dữ liệu quý, không phải lỗi.**

## Cách chấm

1. Mở `audit.html` (một file duy nhất, mọi ảnh nhúng sẵn, không cần mạng).
2. Bấm phím **1** / **2** / **3**, hoặc bấm chuột. `←` `→` chuyển ô. Tiến độ tự lưu vào
   trình duyệt — đóng tab rồi mở lại vẫn còn.
3. **Không chấm quá ~150 ô mỗi buổi.** Trôi tiêu chí đã đo được: 4,2% → 16% → 35% qua ba
   buổi liên tiếp. Nghỉ rồi quay lại, tiến độ vẫn giữ.
4. Chấm xong **toàn bộ** → bấm **Xuất verdicts.jsonl** → lưu file vào **chính thư mục này**
   (`dataset_out/ground_truth/audit_combined/verdicts.jsonl`).

## Phân tầng (design_weight = N_h / n_h)

| Tầng | N_h | n_h | design_weight |
|---|---:|---:|---:|
| GOLD|stt11 | 16,307 | 84 | 194.131 |
| GOLD|stt2 | 16,122 | 83 | 194.241 |
| GOLD|stt4 | 15,966 | 83 | 192.3614 |
| SILVER|stt11 | 3,111 | 86 | 36.1744 |
| SILVER|stt2 | 4,029 | 111 | 36.2973 |
| SILVER|stt4 | 3,747 | 103 | 36.3786 |
| SYLLABLE|stt11 | 2,523 | 93 | 27.129 |
| SYLLABLE|stt2 | 2,204 | 81 | 27.2099 |
| SYLLABLE|stt4 | 2,082 | 76 | 27.3947 |

Độ phủ thực tế của mẫu: **253 lớp chữ phân biệt** ·
**362 trang** · **3 sách**.

## Mẻ này siết được khoảng tin cậy tới đâu

### GOLD — n = 250 trên dân số 48,893

| Số ô sai nhãn | Precision | CI95 (Clopper–Pearson) | Cận dưới một phía |
|--------------:|----------:|------------------------|------------------:|
| 0 | 100.0% | [98.5% · 100.0%] | 98.8% |
| 1 | 99.6% | [97.8% · 100.0%] | 98.1% |
| 2 | 99.2% | [97.1% · 99.9%] | 97.5% |
| 3 | 98.8% | [96.5% · 99.8%] | 96.9% |
| 5 | 98.0% | [95.4% · 99.3%] | 95.8% |
| 8 | 96.8% | [93.8% · 98.6%] | 94.3% |

### SILVER — n = 300 trên dân số 10,887

| Số ô sai nhãn | Precision | CI95 (Clopper–Pearson) | Cận dưới một phía |
|--------------:|----------:|------------------------|------------------:|
| 0 | 100.0% | [98.8% · 100.0%] | 99.0% |
| 1 | 99.7% | [98.2% · 100.0%] | 98.4% |
| 2 | 99.3% | [97.6% · 99.9%] | 97.9% |
| 3 | 99.0% | [97.1% · 99.8%] | 97.4% |
| 5 | 98.3% | [96.2% · 99.5%] | 96.5% |
| 8 | 97.3% | [94.8% · 98.8%] | 95.2% |

### SYLLABLE — n = 250 trên dân số 6,809

| Số ô sai nhãn | Precision | CI95 (Clopper–Pearson) | Cận dưới một phía |
|--------------:|----------:|------------------------|------------------:|
| 0 | 100.0% | [98.5% · 100.0%] | 98.8% |
| 1 | 99.6% | [97.8% · 100.0%] | 98.1% |
| 2 | 99.2% | [97.1% · 99.9%] | 97.5% |
| 3 | 98.8% | [96.5% · 99.8%] | 96.9% |
| 5 | 98.0% | [95.4% · 99.3%] | 95.8% |
| 8 | 96.8% | [93.8% · 98.6%] | 94.3% |

## Sau khi chấm

```
.venv/bin/python -m pipeline.ground_truth.report_combined --dir dataset_out/ground_truth/audit_combined
```

Sinh ra `report.json` + `BANG_KET_QUA.md` — bảng precision/CI theo từng tier, tổng hợp
Horvitz–Thompson, và κ nội tại từ các ô lặp; dán thẳng vào luận văn.
