# Census history — before/after engine-fix + dedup (bằng chứng lịch sử được bảo tồn)

Tài liệu này bảo tồn số liệu census của **thế hệ `labels.csv` CŨ** (trước engine-fix +
dedup upstream). Các con số "before" đó KHÔNG còn tái lập được trên đĩa vì `labels.csv`
đã được tái sinh sạch; chúng được lưu ở đây làm bằng chứng rằng engine-fix + dedup thực
sự đã đóng lớp lỗi trùng lặp / xung đột nhãn.

Selftest (`pipeline/ground_truth/selftest.py`, `pipeline/remediation/selftest.py`) kiểm
cái **TÁI LẬP ĐƯỢC từ `labels.csv` hiện tại**, tức các con số "after". Mỗi assertion đã
được chỉnh về giá trị đo hiện tại kèm comment trỏ về bảng này.

## Bảng before / after

| Chỉ số (census)            | Before (labels.csv cũ) | After (labels.csv hiện tại) | Ý nghĩa                                                        |
|----------------------------|------------------------:|----------------------------:|---------------------------------------------------------------|
| dup_bbox_rows              |                    701  |                          0  | Trùng-bbox cùng cột — dedup đã xoá sạch                        |
| cross_col_rows             |                   1686  |                          8  | Cùng md5 khác cột — còn 4 nhóm xung đột nhãn (2 hàng/nhóm)     |
| union_rows (dup_defect)    |                   2321  |                          8  | Hợp của hai lớp trên; dup_bbox=0 nên union == cross_col        |
| provably_wrong_rows        |                  ~1177  |                          4  | 1 nhãn sai / mỗi nhóm xung đột (4 nhóm)                        |
| md5 split-leak (original)  |                    288  |                          0  | md5 span >1 split ở ĐẦU VÀO — dedup đã đóng, bất biến giữ từ đầu |
| quarantined_rows           |                  >1000  |                          8  | Hàng bị cách ly — chỉ còn 8 hàng cross-col xung đột (0 dup thuần) |

### Chỉ số phụ (biến động nhỏ / trạng thái sau remediation)

| Chỉ số                        | Before  | After  | Ghi chú                                                       |
|-------------------------------|--------:|-------:|--------------------------------------------------------------|
| similar_bridge_rows           |   3856  |  3850  | Dao động 6 hàng theo lần tái sinh labels.csv (KHÔNG phải dedup) |
| quarantined_conflict          |     —   |     8  | Toàn bộ 8 hàng cách ly là xung đột nhãn                       |
| quarantined_duplicate         |     —   |     0  | Không còn bản trùng thuần nào để cách ly (lớp dedup đã đóng)  |
| demoted_similar_lowcos        |    748  |   839  | Similar-bridge cosine thấp bị hạ REVIEW; đổi theo dữ liệu mới |
| usable_after                  |     —   | 68499  | Số hàng dùng được sau remediation (report hiện tại)          |

## Nguồn của các con số

- **"After"** — đo trực tiếp trên `dataset_out/labels.csv` hiện tại qua
  `pipeline.remediation.census.run_census` và `pipeline.remediation.remediate.remediate`;
  xác nhận chéo với `dataset_out/remediation_report.json`
  (`census.dup_bbox_rows=0`, `cross_col_rows=8`, `union_rows=8`, `provably_wrong_rows=4`,
  `quarantined_rows=8`, `quarantined_conflict=8`, `quarantined_duplicate=0`,
  `md5_spanning_splits_original=0`, `demoted_similar_lowcos=839`, `usable_after=68499`).
- **"Before"** — đo trên thế hệ `labels.csv` TRƯỚC engine-fix (không còn trên đĩa). Nguồn
  bảo tồn: git history của `dataset_out/labels.csv`, các báo cáo remediation cũ, và ghi chú
  dự án (round-3 census: 2.321 hàng trùng / 1.177 provably-wrong; phase-1 fixes:
  split-leak 288 -> 0, demote 748).

## Vì sao selftest kiểm "after" chứ không kiểm "before"

Chuẩn tái lập yêu cầu selftest khẳng định cái **tái lập được từ trạng thái hiện tại**. Số
"before" đã bị chính pipeline sửa; ghim chúng vào assertion sẽ biến selftest thành thất bại
vĩnh viễn dù pipeline đúng. Bất biến kiểm định vẫn được giữ nguyên ý nghĩa:
`md5_spanning_splits_after == 0`, lớp trùng đã đóng (quarantine toàn conflict, 0 duplicate),
split-leak == 0. Bảng before/after ở trên là bằng chứng lịch sử cho thấy các con số cao
ban đầu đã được xử lý, không bị xoá khỏi hồ sơ.
