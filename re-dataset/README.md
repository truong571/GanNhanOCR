# Bộ dữ liệu gán nhãn chữ Nôm — Sách Thánh Truyện

**52,441 nhãn cấp KÝ TỰ** + **6,930 chú giải cấp ÂM TIẾT**
= 59,371 dòng · 1,590 lớp chữ · 447 trang · 4,011 cột

> ⚠️ **Đừng phát biểu là "59,371 nhãn".** 6,930 dòng tầng SYLLABLE có cột
> `label` **rỗng** — chúng chỉ ghi âm Quốc ngữ, KHÔNG gán chữ Nôm. Con số đem so với các bộ
> dữ liệu Hán Nôm khác là **52,441**.

## Cấu trúc

| | |
|---|---|
| `labels.csv` | một dòng mỗi ô, đường dẫn ảnh tương đối |
| `gold/` | ảnh crop của các ô có nhãn cấp ký tự |
| `syllable/` | ảnh crop của các ô chỉ có chú giải âm |

## Cột quan trọng

| cột | nghĩa |
|---|---|
| `label` | chữ Nôm được gán. **Rỗng** ở tầng SYLLABLE |
| `label_level` | `char` = nhãn ký tự · `syllable` = chỉ có âm |
| `syllable` | âm Quốc ngữ tương ứng, lấy từ bản dịch song song in kèm |
| `tier` / `rule` | luật nào quyết nhãn này — xem DATASHEET |
| `usable_image` | `0` = ảnh trắng hoặc bị cắt mất nét (446 ô). Nhãn có thể vẫn đúng; đừng chấm chiều ảnh ở các ô này |
| `crop_quality_flag` | `ok` / `bleed` (dính mực chữ bên cạnh) / `truncated` / `blank` |
| `split` / `split_group` | **rời nhau theo TRANG** |
| `label_in_train` | `0` = lớp chữ này **không có mặt trong `train`**. Đánh giá phải lọc theo cột này |
| `page_cot_lech` | `1` = trang này không đủ 9 cột (197 ô / 3 trang). Ghép cột Nôm↔Quốc ngữ có thể đã trượt |

## Chia tách — rời nhau theo TRANG

Đo trên chính `labels.csv`: **0/447 trang** nằm ở hai phía.

Chọn mức trang chứ không phải cột vì hai cột cạnh nhau trên cùng một trang dùng
chung ván khắc, chung mực, chung lần quét — chia theo cột thì mô hình học được
*diện mạo trang* rồi được chấm lại trên chính trang đó.

**Hệ quả phải biết:** 164 ô có lớp chữ **không xuất hiện trong
`train`** — hệ quả của việc không bóp méo chia tách. Cột **`label_in_train`**
đánh dấu chúng: `1` có mặt, `0` không, **rỗng** với dòng tầng SYLLABLE (chúng
không có nhãn cấp ký tự nên câu hỏi không áp dụng).

Khi đánh giá **cấp ký tự**, bỏ các ô `label_in_train == 0` — nếu không, 164 ô đó
bị tính sai 100% dù mô hình chưa từng có cơ hội học lớp chữ ấy.

> ⚠️ Đừng viết `df[df.label_in_train == "1"]` để lọc cả bộ: điều kiện đó vứt
> luôn 6,930 dòng SYLLABLE có ô rỗng, tức 7,094 dòng
> chứ không phải 164. Lọc trong phạm vi `label_level == "char"`.

## 🔴 Trạng thái kiểm định

**Chưa có phép đo precision nào còn hiệu lực.** Mọi con số precision trong các bản trước
đã bị **tước tư cách bằng chứng** vì nguồn phán quyết hoá ra là máy chấm chứ không phải
người.

Nghĩa là: bộ này dùng được để **huấn luyện** và **thăm dò**, nhưng **chưa được trích dẫn
như dữ liệu đã kiểm chứng**.

## Trích dẫn

Xem `NGUON_THU_TICH.md` cho lai lịch ba cuốn sách nguồn.

*Sinh tự động từ `labels.csv` ngày 2026-08-25 · commit `8899d0cb50`*
