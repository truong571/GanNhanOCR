# Bộ dữ liệu gán nhãn chữ Nôm — Sách Thánh Truyện

**50,156 nhãn cấp KÝ TỰ** + **6,911 chú giải cấp ÂM TIẾT**
= 57,067 dòng · 1,583 lớp chữ · 444 trang · 3,985 cột

> ⚠️ **Đừng phát biểu là "57,067 nhãn".** 6,911 dòng tầng SYLLABLE có cột
> `label` **rỗng** — chúng chỉ ghi âm Quốc ngữ, KHÔNG gán chữ Nôm. Con số đem so với các bộ
> dữ liệu Hán Nôm khác là **50,156**.

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
| `usable_image` | `0` = ảnh trắng hoặc bị cắt mất nét (424 ô). Nhãn có thể vẫn đúng; đừng chấm chiều ảnh ở các ô này |
| `crop_quality_flag` | `ok` / `bleed` (dính mực chữ bên cạnh) / `truncated` / `blank` |
| `split` / `split_group` | 🔴 **CÓ RÒ RỈ** — xem dưới |


## 🔴 Chia tách CÓ RÒ RỈ theo trang

Đo trên chính `labels.csv`: **360/444 trang** có ô nằm ở
**hai phía khác nhau**. Hai cột cạnh nhau trên cùng một trang dùng chung ván khắc,
chung mực, chung lần quét, nên mô hình huấn luyện trên `train` học được *diện mạo
trang* rồi được chấm lại trên chính trang đó.

**Mọi chỉ số đo bằng `split` sẵn có là CẬN TRÊN**, không phải hiệu năng thật trên
trang chưa từng thấy. Muốn đánh giá trung thực thì tự chia lại theo `book` + `page`.

## 🔴 Trạng thái kiểm định

**Chưa có phép đo precision nào còn hiệu lực.** Mọi con số precision trong các bản trước
đã bị **tước tư cách bằng chứng** vì nguồn phán quyết hoá ra là máy chấm chứ không phải
người.

Nghĩa là: bộ này dùng được để **huấn luyện** và **thăm dò**, nhưng **chưa được trích dẫn
như dữ liệu đã kiểm chứng**.

## Trích dẫn

Xem `NGUON_THU_TICH.md` cho lai lịch ba cuốn sách nguồn.

*Sinh tự động từ `labels.csv` ngày 2026-08-25 · commit `bfec0ac78b`*
