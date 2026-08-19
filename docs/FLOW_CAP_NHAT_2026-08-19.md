# Đánh giá tổng thể & flow đề nghị cập nhật — 2026-08-19

Mọi con số trong tài liệu này đều đo lại được bằng lệnh ghi kèm. Chỗ nào là suy đoán thì
ghi rõ là suy đoán.

---

## Phần 1 — Cái gì đang vững, cái gì đang hỏng

### Vững, giữ nguyên

| hạng mục | bằng chứng |
|---|---|
| Chuỗi 6 bước tất định `setup→extract→build→remediate→confusion→export` | `run_pipeline.sh`, chạy lại ra byte giống nhau khi còn cache OCR |
| Cache OCR là primary data, có chốt chặn xoá nhầm | `confirm_fresh_delete()` bắt gõ `XOA` |
| Bước 5 `confusion_fix` đã nối vào chuỗi | trước 08-11 bị bỏ quên → bộ giao nộp lệch với bộ đem audit |
| Tách train/val/test theo `(sách, trang, cột)` | rò rỉ 288 → 0 sau remediation |
| Đóng gói quốc tế (Frictionless + Croissant + datasheet) | `pipeline/publish/`, CI gate 11/11 |
| Mẻ audit GỘP một dòng chấm | `pipeline/ground_truth/make_combined_batch.py`, 860 ô, 60 ô lặp ẩn |

### Hỏng hoặc chưa đủ căn cứ

| # | vấn đề | số đo | hệ quả |
|---|---|---|---|
| B1 | **Độ tin cậy người chấm** κ = 0,13 test-retest; tỷ lệ lỗi trôi 4,2% → 16% → 35% qua ba buổi | 40 ô lặp, 2026-08-04 | mọi precision đo bằng tay đều lung lay, trừ chiều NHÃN |
| B2 | **Nhiệm vụ chấm bị trộn hai chiều** — NHÃN (đọc chữ) và CROP (cắt đúng ô) chấm chung một nút | κ chiều CROP = 0,14 vs chiều NHÃN 0/20 báo động giả | phải tách hẳn: NHÃN cho người, CROP đo bằng hình học |
| B3 | **S3 ngược dấu** trên verdict người | 6/6 AUC < 0,5; bank_cos 0,26 | không được dùng S3 làm cổng hạ cấp GOLD |
| B4 | **1.846 ô lệch DẤU THANH** giữa âm OCR và từ điển | GOLD 167 · SILVER 308 · SYLLABLE 857 · REVIEW 514 | lỗi chuẩn hoá, sửa được tất định, hiện đang bị tính là "không khớp từ điển" |
| B5 | **Tier SYLLABLE 6.809 ô không có căn cứ ký tự** | 316 cặp (chữ, âm); Unihan 17.0 xác nhận **0/316** | không nguồn công khai nào lấp được; phải chấm từ chính corpus |
| B6 | **Nghi OCR thay chữ Nôm hiếm bằng chữ Hán nhìn giống** | 妃 đọc "bà" (chữ chuẩn 妑); 而 đọc "làm" (chữ chuẩn 爫); 27 cặp/581 ô nối được qua similar-dict | nếu đúng thì đây là lớp lỗi hệ thống thứ hai sau 㝵/người |
| B7 | **73.829 crop mồ côi** (113 MB) trong `gold/ silver/ syllable/` | mtime toàn 07/2026 | thư mục crop không phản ánh bộ hiện hành |
| B8 | Bước audit / fuse / publish **không nằm trong** `run_pipeline.sh` | chú thích "TẠM BỎ QUA" đầu file | bộ công bố và bộ đo precision dễ trôi khỏi nhau lần nữa |

### Đã loại trừ (không phải nguyên nhân, đừng đào lại)

- **Lệch dòng khi align**: với 而→"làm" (514 ô), âm ở vị trí ±1/±2 khớp từ điển của 而 đúng
  **1 ô**; 爾/白/希/皇/妃 = **0 ô**. Alignment không bị trượt.
- **Từ điển thiếu bản mới**: `QuocNgu_SinoNom.csv` (104.048 cặp) đã là tập cha của
  Unihan `kVietnamese` (8.655 cặp); Unihan chỉ thêm 109 cặp và xác nhận 0/316 cặp SYLLABLE.
- **Detector còn dư địa**: box-F1 ~0,84 là trần do GT loại tier REVIEW, không phải do mô hình.

---

## Phần 2 — Flow đề nghị

Nguyên tắc xuyên suốt: **bộ đem đo và bộ đem nộp phải là MỘT tập**, và **phán đoán của AI
không bao giờ trở thành nhãn** — chỉ để xếp hạng việc cho người.

```
1 setup ─ 2 extract ─ 3 normalize* ─ 4 build ─ 5 remediate ─ 6 confusion
                                                                  │
                            ┌─────────────────────────────────────┘
                            ▼
        7 gapcheck*  →  8 audit (người, mẻ GỘP)  →  9 export  →  10 publish
        (offline,        (đo precision từng tier      (chỉ chạy khi bước 8
         tất định)        + κ nội tại)                 đã có verdict)
```

`*` = bước mới.

### Bước 3 — `normalize` (MỚI, đặt TRƯỚC build)

Chuẩn hoá âm Quốc ngữ trước khi tier hoá, thay vì vá sau:

- hợp nhất dấu thanh theo từ điển khi phần gốc âm tiết trùng khớp duy nhất
  (rì/ri, lạ/là, mã/ma, lắm/lăm, vô/vồ, chăng/chắng…) — **1.846 ô**;
- ghi cột `syllable_raw` bên cạnh `syllable` để không mất dấu vết bản gốc;
- ô nào sửa thanh xong khớp từ điển thì **đi thẳng vào đường GOLD**, không rơi xuống
  SYLLABLE nữa.

Vì sao đặt trước build: đặt sau thì tier đã chốt, và 857 ô đã mang nhãn cấp âm tiết —
sửa sau là vá, đặt trước là hết bệnh.

### Bước 7 — `gapcheck` (MỚI, offline, không API)

Chạy trên các cặp (chữ, âm) từ điển không có — `pipeline/tools/dict_gap.py` đã dựng:

1. **tầng nghĩa**: đối chiếu `kDefinition` của Unihan (287/316 cặp có) → tách "mượn nghĩa
   hợp lý" (皇 *royal* → thánh; 妃 *wife* → bà) khỏi "vô lý" (而 *and/but* → làm;
   常 *common* → và);
2. **tầng tự dạng**: hỏi ngược "âm này thường viết bằng chữ Nôm nào" rồi so hình với chữ
   OCR đọc được → bắt lớp lỗi B6;
3. đầu ra là **thứ tự ưu tiên chấm tay**, không phải nhãn.

20 cặp đầu phủ 35% số ô SYLLABLE, 50 cặp đầu phủ 55% — nên đây là việc một buổi, không
phải một tháng.

### Bước 8 — `audit` (đưa TRỞ LẠI chuỗi, và tách đôi nhiệm vụ)

| chiều | ai làm | cách đo |
|---|---|---|
| **NHÃN** — chữ trong ô có đúng là chữ được gán không | người, mẻ GỘP một dòng chấm | Clopper–Pearson từng tier + κ nội tại từ ô lặp ẩn |
| **CROP** — ô có cắt đúng một chữ không | **máy**, không phải người | IoU với box detector, tỷ lệ mực, aspect-ratio, `crop_bleed.csv` |

Đây là sửa trực tiếp B2. Chiều CROP có κ = 0,14 khi để người chấm — nó là đại lượng hình
học, có ngưỡng đo được, không phải việc của mắt người.

Ràng buộc bắt buộc: mẻ audit rút từ **đúng file** mà bước 9 export ra
(`labels_final.csv`), và bước 9 **từ chối chạy** nếu chưa có verdict tương ứng với bản
`labels_final.csv` hiện hành (so bằng sha256). Đây là chốt chặn cho B8.

### Bước 9–10 — `export` + `publish`

- `export` dọn luôn crop mồ côi (B7): xoá file không được `labels_final.csv` tham chiếu,
  in ra số đã xoá;
- `publish` giữ nguyên `pipeline/publish/`, nhưng datasheet phải ghi **số precision đo
  được ở bước 8** kèm CI, và ghi thẳng giới hạn κ người chấm — không giấu.

### Cái KHÔNG nên làm

- Không dùng S3 làm cổng hạ cấp GOLD (B3) — sẽ đánh rớt 3.228 hàng sạch.
- Không sinh thêm tier do AI phán. "SILVER = AI audit" đã là một blocker của luận văn;
  thêm một tier nữa cùng kiểu là lặp lại lỗi cũ.
- Không đi tìm từ điển mới cho tier SYLLABLE (B5 đã loại trừ).

---

## Phần 3 — Thứ tự làm

| ưu tiên | việc | công | mở khoá |
|---|---|---|---|
| 1 | Bước 3 `normalize` (dấu thanh) | nửa ngày | 1.846 ô, trong đó 857 ô thoát tier SYLLABLE |
| 2 | Bước 7 `gapcheck` tầng nghĩa (offline) | nửa ngày | thứ tự ưu tiên cho 316 cặp |
| 3 | Chấm tay 50 cặp đầu của `dict_gap_syllable.csv` | một buổi | 55% tier SYLLABLE + xác nhận/bác bỏ B6 |
| 4 | Tách chiều CROP sang đo bằng máy | một ngày | gỡ B2, làm số precision NHÃN dùng được |
| 5 | Chốt chặn sha256 giữa audit và export | 2 giờ | gỡ B8 vĩnh viễn |
| 6 | Dọn crop mồ côi trong `export` | 1 giờ | gỡ B7 |

Ba việc đầu đều **không tốn API và không cần chấm lại từ đầu**.

---

## Lệnh kiểm lại các số trong tài liệu này

```bash
.venv/bin/python -m pipeline.tools.check_dict_duplicates     # quan hệ trùng lặp trong Dict/
.venv/bin/python -m pipeline.tools.dict_gap                  # bộ chữ từ điển không có
.venv/bin/python -m pipeline.ground_truth.split_combined     # tách mẻ gộp theo tier
```
