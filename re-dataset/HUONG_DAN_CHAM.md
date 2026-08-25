# Hướng dẫn chấm — dành cho đội chấm

Thư mục này **tự đủ**: `labels.csv` + hai thư mục ảnh. Không cần công cụ nào của chúng
tôi. Đội chấm **tự làm giao diện** theo cách thuận tiện nhất.

Chúng tôi chỉ chốt **đầu ra**, để nạp ngược vào pipeline được.

## Câu hỏi — KHÁC NHAU theo `label_level`

| `label_level` | số ô | câu hỏi |
|---|---|---|
| `char` | 50,156 | **Chữ trong ảnh có đúng là chữ ở cột `label` không?** |
| `syllable` | 6,911 | **Âm ở cột `syllable` có đúng với chữ trong ảnh không?** |

> ⚠️ Ô `label_level = syllable` có cột **`label` RỖNG** — chúng **không gán chữ Nôm**.
> Hỏi "chữ này đúng không" ở đó là hỏi một thứ không tồn tại.

## Bốn quy tắc

1. **Khung cắt xấu KHÔNG phải lỗi nhãn.** Dính chút mực chữ bên cạnh mà vẫn đọc ra chữ
   → chấm **đúng**. Câu hỏi là về **chữ**, không phải về **khung**.
2. **424 ô có `usable_image = 0`** (ảnh trắng hoặc cắt mất nét) — **bỏ qua**,
   chấm `khong_doc_duoc`. Đừng chấm thành "sai".
3. **Lưỡng lự → `khong_doc_duoc`, đừng chấm `sai`.** Ô *không đọc được* bị loại khỏi mẫu
   số; ô *sai* bị tính là lỗi. Dữ liệu cũ cho thấy xu hướng **gọi quá tay**.
4. Chấm khó thì mở ảnh trang gốc theo cột `book` / `page` / `bbox`.

## Đầu ra cần nộp — ĐÚNG hai tệp, đặt ngay trong thư mục này

### 1. `verdicts.csv`

```csv
image,verdict,nguoi_cham,ghi_chu
gold/stt2_page_0012_c01_003.png,dung,Nguyen Van A,
gold/stt2_page_0012_c01_004.png,sai,Nguyen Van A,nhìn giống chữ khác
syllable/stt4_page_0020_c03_055.png,khong_doc_duoc,Tran Thi B,mực nhoè
```

| cột | bắt buộc | giá trị |
|---|---|---|
| `image` | ✅ | chép **nguyên văn** từ `labels.csv`, không đổi đường dẫn |
| `verdict` | ✅ | **chỉ ba giá trị**: `dung` · `sai` · `khong_doc_duoc` |
| `nguoi_cham` | nên có | tên người chấm ô đó — cần để tính κ liên-người |
| `ghi_chu` | không | tuỳ ý |

**Không cần chấm hết.** Chấm được bao nhiêu nộp bấy nhiêu; pipeline tự tính độ phủ.
Nhưng nếu chấm một phần thì con số precision **chỉ đúng cho phần đã chấm** — sẽ không
suy rộng ra toàn bộ, vì không biết phần đó có đại diện hay không.

### 2. `NGUOI_CHAM.md`

**Bắt buộc.** Không có nó thì pipeline **từ chối nạp**, dù `verdicts.csv` hoàn hảo.

Lý do: dự án này đã một lần tin nhầm verdict **MÁY** là verdict người và phải huỷ toàn
bộ số liệu — và các tệp đó **không hề tự khai là máy**. Nên chúng tôi không suy đoán
xuất xứ từ dữ liệu nữa, mà **bắt khai ra**.

Mẫu ở `docs/QUY_TRINH_CHAM_TAY.md`. Ba điều quyết định: người chấm có **đọc được chữ
Nôm** không · có **ít nhất hai người** không · có cam kết **không dùng máy** không.

## Nộp về rồi thì sao

Chép cả thư mục `re-dataset/` (đã có hai tệp trên) về máy chạy pipeline, rồi:

```bash
bash run_pipeline.sh
```

Pipeline tự phát hiện `verdicts.csv`, nạp phán quyết, và xuất bộ **cuối cùng** ra
`dataset/` kèm `docs/BANG_PRECISION.md`.

*Sinh tự động ngày 2026-08-25 · commit `90320fd1fd`*
