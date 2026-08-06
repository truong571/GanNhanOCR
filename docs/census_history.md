# Census history — before/after engine-fix + dedup (bằng chứng lịch sử được bảo tồn)

Tài liệu này bảo tồn số liệu census của **các thế hệ `labels.csv` đã qua**. Những con số
cũ KHÔNG còn tái lập được trên đĩa vì `labels.csv` đã được tái sinh; chúng được lưu ở đây
làm bằng chứng rằng engine-fix + dedup thực sự đã đóng lớp lỗi trùng lặp / xung đột nhãn.

Selftest (`pipeline/ground_truth/selftest.py`, `pipeline/remediation/selftest.py`) kiểm
cái **TÁI LẬP ĐƯỢC từ `labels.csv` hiện tại**. Từ 2026-08-03, các assertion này được viết
dưới dạng **BẤT BIẾN** (`== 0`) và **kiểm cấu trúc** (công thức tính ra từ chính dữ liệu),
thay cho số cứng — vì số cứng đã hai lần gây đỏ giả khi dữ liệu được sinh lại (11 assertion
ngày 20/07, 8 assertion ngày 03/08) mà không hề có lỗi code nào.

## Bảng before / after

Ba thế hệ `labels.csv` đã được đo. **G3 là bản duy nhất còn trên đĩa** và là bản mà
selftest kiểm.

| Chỉ số (census)           | G1: trước engine-fix | G2: 2026-07-21 | **G3: 2026-07-22 (hiện tại)** | Ý nghĩa                                                    |
|---------------------------|---------------------:|---------------:|------------------------------:|------------------------------------------------------------|
| dup_bbox_rows             |                  701 |              0 |                         **0** | Trùng-bbox cùng cột — dedup đã xoá sạch từ G2              |
| cross_col_rows            |                 1686 |              8 |                         **0** | Cùng md5 khác cột — 4 nhóm xung đột cuối đã hết ở G3       |
| union_rows (dup_defect)   |                 2321 |              8 |                         **0** | Hợp của hai lớp trên — **lớp trùng lặp ĐÃ ĐÓNG HOÀN TOÀN** |
| provably_wrong_rows       |                ~1177 |              4 |                         **0** | 1 nhãn sai / mỗi nhóm xung đột; không còn nhóm nào         |
| md5 split-leak (original) |                  288 |              0 |                         **0** | md5 span >1 split ở ĐẦU VÀO — bất biến giữ từ đầu tới cuối |
| quarantined_rows          |                >1000 |              8 |                         **0** | Không còn hàng nào phải cách ly                            |

### Chỉ số phụ (biến động nhỏ / trạng thái sau remediation)

| Chỉ số                 |   G1 |   G2 |  **G3 (hiện tại)** | Ghi chú                                                    |
|------------------------|-----:|-----:|-------------------:|------------------------------------------------------------|
| similar_bridge_rows    | 3856 | 3850 |           **3850** | Dao động theo lần tái sinh labels.csv (KHÔNG phải dedup)   |
| quarantined_conflict   |    — |    8 |              **0** | Không còn xung đột nhãn cross-column                       |
| quarantined_duplicate  |    — |    0 |              **0** | Không còn bản trùng thuần nào để cách ly                   |
| demoted_similar_lowcos |  748 |  839 |            **925** | Xem giải thích bên dưới — TĂNG là đúng, không phải hồi quy |
| usable_before -> after |    — |    — | **69440 -> 68515** | Số hàng dùng được trước/sau remediation                    |

### Vì sao `demoted_similar_lowcos` TĂNG 748 -> 839 -> 925

Đây **không** phải dấu hiệu chất lượng giảm. Remediation chạy quarantine **trước**, demote
**sau**. Hàng đã bị cách ly không còn tier `GOLD` nên **không đi vào bước demote nữa**. Ở G1
quarantine cướp mất ~72 hàng similar-bridge cosine-thấp; ở G3 quarantine = 0 nên **toàn bộ**
925 hàng `GOLD ∩ s1_inter_s2_similar ∩ s3_cosine < 0.62` đều được demote đúng như thiết kế.

Kiểm chứng trực tiếp trên G3: `|GOLD ∩ similar_bridge| = 3850`, trong đó `s3_cosine < 0.62`
là **925** — trùng khớp tuyệt đối với `demoted_similar_lowcos = 925`.

Vì vậy selftest **không** ghim khoảng cứng `700..850` nữa mà kiểm theo công thức
`|GOLD ∩ bridge ∩ s3<τ| \ quarantine`, đúng với mọi thế hệ dữ liệu về sau.

## Nguồn của các con số

- **G3 (hiện tại)** — đo trực tiếp trên `dataset_out/labels.csv` (mtime 2026-07-22 22:10)
  qua `pipeline.remediation.census.run_census` và `pipeline.remediation.remediate.remediate`.
  Kiểm chứng **độc lập** bằng pandas thuần (không qua module census) ngày 2026-08-03:
  `dup_bbox=0`, `cross_col=0`, `union=0`, `md5 rỗng=0`, và chỉ **2** hàng chung md5 trên
  toàn bộ 69.440 hàng usable.
- **G2 (2026-07-21)** — mốc selftest cũ, giá trị đã ghim trong assertion trước ngày 03/08.
- **G1 (trước engine-fix)** — không còn trên đĩa. Nguồn bảo tồn: git history của
  `dataset_out/labels.csv`, các báo cáo remediation cũ, và ghi chú dự án (round-3 census:
  2.321 hàng trùng / 1.177 provably-wrong; phase-1 fixes: split-leak 288 -> 0, demote 748).

## Vì sao selftest kiểm "after" chứ không kiểm "before"

Chuẩn tái lập yêu cầu selftest khẳng định cái **tái lập được từ trạng thái hiện tại**. Số
"before" đã bị chính pipeline sửa; ghim chúng vào assertion sẽ biến selftest thành thất bại
vĩnh viễn dù pipeline đúng. Bất biến kiểm định vẫn được giữ nguyên ý nghĩa:
`md5_spanning_splits_after == 0`, lớp trùng đã đóng (quarantine toàn conflict, 0 duplicate),
split-leak == 0. Bảng before/after ở trên là bằng chứng lịch sử cho thấy các con số cao
ban đầu đã được xử lý, không bị xoá khỏi hồ sơ.
