# Bản cũ đã bị thay thế — ĐO ĐƯỢC là tập con, không mất dữ liệu

Ba file dưới đây được dời ra khỏi `Dict/` ngày 2026-08-19. Không file nào được bất kỳ
đoạn mã nào tham chiếu (`config/pipeline.yaml`, `core/`, `pipeline/`, `ArcFace/` chỉ trỏ
vào hai file còn lại ở thư mục cha). Chỉ giữ làm bằng chứng xuất xứ, không dùng để chạy.

## Nhóm 1 — âm Quốc ngữ ↔ chữ Nôm

| file | số cặp | quan hệ với bản đang dùng |
|---|---:|---|
| `QuocNgu_SinoNom_Dic.xlsx` | 53.097 | **tập con thật sự** của `../QuocNgu_SinoNom.csv` (104.172 cặp): 0 cặp nằm ngoài, bản tổng hợp thêm 51.075 cặp |
| `SinoNom_QuocNgu_Dic.xlsx` | 53.097 | **cùng một bảng**, chỉ đảo thứ tự hai cột. So theo tập cặp (âm, chữ): giống 100%, hiệu hai chiều = 0 |

Nói cách khác cả hai file này chỉ là một bảng duy nhất, và bảng đó đã nằm trọn trong
`QuocNgu_SinoNom.csv`. Cần tra chiều ngược chữ→âm thì đảo cột của bản tổng hợp,
đừng đọc file cũ (sẽ mất 49% dữ liệu mà không có cảnh báo).

## Nhóm 2 — tự dạng gần giống

| file | số khoá | quan hệ với bản đang dùng |
|---|---:|---|
| `SinoNom_Similar_Dic.xlsx` | 26.044 | **trùng khít** `../SinoNom_Similar.csv` (bản TRƯỚC khi gộp 2026-08-19, xem `../_backup/`): cùng 26.044 khoá, 0 khoá lệch, 0 dòng khác nội dung |

Hậu tố "v2" ở đây **không** có nghĩa là bản mới hơn về nội dung — chỉ là bản đổi định dạng
sang CSV. Hai file là một.

## Cách kiểm lại

    .venv/bin/python -m pipeline.tools.check_dict_duplicates
