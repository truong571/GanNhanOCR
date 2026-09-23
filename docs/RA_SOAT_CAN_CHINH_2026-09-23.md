# RÀ SOÁT "CĂN CHỈNH / QUY HOẠCH GÁN" — 8 bộ đã chạy (23/09/2026)

Mọi số do script sinh, **0 token LLM, 0 lượt API**. Tái lập (≈10 s):

```bash
.venv/bin/python scripts/measure/align_audit.py --selftest      # 8 phép tự kiểm
.venv/bin/python scripts/measure/align_audit.py --book all      # 8 bộ, ≈10 s
```
Đầu ra `measure_out/align_audit/{SUMMARY.json, REPORT.md, <Book>_invariants.csv, <Book>_violations.csv, <Book>_drift.csv}`.
Script chỉ ĐỌC; không sửa `pipeline/`, `core/`, `data/`; không chạy build/export.

**Ảnh chụp bộ nhãn được đo** (md5 8 ký tự đầu · giờ sửa · số ô): `dataset_out/labels_final.csv` `d2def17d` 22/09 09:30 (83.239 ô = STT2+4+11)
· `prepared/LucVanTien1883/dataset_out/labels_gated.csv` `48f2da1c` 22/09 23:41 (14.474) · `prepared_b1/KimVanKieu1884/dataset_out_b1/labels_gated.csv`
`b10e5ec1` 22/09 23:45 (22.750) · `prepared/Chrestomathie1872/dataset_out/labels_gated.csv` `663cff3f` 22/09 23:46 (8.016)
· `prepared_ihr/LucVanTien1916/dataset_out/labels_gated.csv` `5d275ce6` 23/09 00:53 (13.761) · `prepared_ihr/TruyenKieu1872/…` `1b109438` 23/09 01:11 (22.499).
Dùng bản `*_gated` (có cả REVIEW/QUARANTINE) chứ không dùng `dataset/<Book>/labels.csv` đã lọc, để bất biến phủ **mọi** ô pipeline sinh ra.

## 1. Bảng vi phạm bất biến — 8 bộ (vi phạm / mẫu số)

| Bất biến (mục 1 của đề bài) | STT2 | STT4 | STT11 | LVT1883 | KVK1884 | Chresto | LVT1916 | TK1872 |
|---|---|---|---|---|---|---|---|---|
| A1 mỗi âm QN ≤ 1 ô *(a)* | 0/28.485 | 0/27.451 | 0/27.303 | 0/14.474 | 0/22.750 | 0/8.016 | 0/13.761 | 0/22.499 |
| A2 `syl_idx` trong cột *(a)* | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A3 âm đúng vị trí *(e)* | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| A4 GOLD âm đúng vị trí *(d)* | 0/17.678 | 0/17.440 | 0/17.589 | 0/10.980 | 0/20.030 | 0/6.224 | 0/12.207 | 0/20.159 |
| A5 không mượn âm ô khác *(e)* | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A6 `nom_idx` đúng thứ tự y *(b)* | 0/1.440 | 0/1.301 | 0/1.286 | 0/1.041 | 0/1.628 | 0/418 | 0/986 | 0/1.608 |
| A7 `syl_idx` tăng theo `nom_idx` *(b)* | 0/1.440 | 0/1.301 | 0/1.286 | 0/1.041 | 0/1.628 | 0/418 | 0/986 | 0/1.608 |
| A8 cột phải→trái theo x *(b)* | 0/160 | 0/145 | 0/143 | 0/105 | 0/163 | 0/65 | 0/99 | 0/161 |
| A9 cột = 14 âm, tách 6⧺8 *(c)* | — | — | — | **27/1.044** | **26/1.628** | — | **22/988** | **7/1.610** |
| A10 không ghép xuyên tầng *(c)* | — | — | — | 0/1.040 | 0/1.628 | — | 0/985 | 0/1.608 |
| A11 parity câu lẻ/chẵn *(c)* | — | — | — | 0/1.044 | 0/1.628 | — | **61/988** | **1.078/1.610** |
| A12 số câu duy nhất & liên tục | — | — | — | 0/2.088 | 0/3.256 | — | **84/1.975** | **21/3.219** |
| A13 (văn xuôi) span truyện liền | — | — | — | — | — | 0/392 | — | — |

*(a)–(e) = mã bất biến trong đề bài.* Chi tiết từng ca ở `<Book>_violations.csv`.

**Đọc bảng.** Bốn bất biến *cốt lõi* của phép gán ô↔âm (A1, A2, A4, A5) **PASS tuyệt đối trên cả 8 bộ**:
không ô nào chiếm hai chỗ, không ô nào nhận âm của vị trí khác trong cột, mọi ô GOLD mang đúng âm ở đúng chỗ.
Thứ tự đọc (A6–A8) cũng PASS tuyệt đối. Ba nhóm còn FAIL:

* **A3 (3 ô/83.239 ở STT)** — `syllable` rỗng ở ô REVIEW (`page_0072/c7`, `page_0040/c2`, `page_0108/c8`), không phải lệch ô.
  *Ghi chú phương pháp*: A3/A4 loại riêng ca **chỉ khác dấu thanh** (pipeline cố ý sửa dấu, rule `*_am_sua_dau`) — 196/171/174
  (STT) và 264/172/204/6/1 (5 bộ mới) ô; nếu tính cả thì A3 "FAIL" nhưng đó là **thiết kế**, không phải trôi.
* **A9 — số âm QN mỗi cột ≠ 14**: 27 / 26 / 22 / 7 cột. **Không cột nào đủ 14 âm mà chia tầng sai 6/8** ⇒ luật 6/8 (§5.2 bước 3/6) đã ăn trọn.
* **A11/A12 — đánh số câu 2 bộ IHR**: xem §2.3, là lỗi *sổ sách số câu*, không làm lệch ô.

## 2. Trôi căn chỉnh (chỉ số quan trọng nhất)

### 2.1 So nhãn NGƯỜI (2 bộ IHR-NomDB) và so dị bản (2 bộ thạch bản)

| Bộ | chuẩn | n ô | đúng | **lệch VỊ TRÍ** (đúng chữ, sai ô) | sai chữ | câu trôi (dịch ≠ 0) |
|---|---|---|---|---|---|---|
| LucVanTien1916 | nhãn người | 12.682 | 97,92 % | **91 (0,72 %)** | 173 (1,36 %) | 14/1.931 (−1: 6, +1: 8) |
| TruyenKieu1872 | nhãn người | 19.962 | 98,58 % | **32 (0,16 %)** | 251 (1,26 %) | 6/3.043 |
| LucVanTien1883 | dị bản LVT1916 | 3.687 | 72,53 % | 7 (0,19 %) | 1.006 | 0/609 |
| KimVanKieu1884 | dị bản 1871+1872 | 31.866 | 80,58 % | 16 (0,05 %) | 6.172 | 1/4.792 |
| KVK1884 — chỉ câu QN **không** lấy từ 1871 | dị bản | 5.252 | 75,74 % | 10 (0,19 %) | 1.264 | 1/812 |
| Chresto1872, STT2/4/11 | *không có chuẩn độc lập* | — | — | — | — | — |

Trôi **dồn cụm**, không rải đều: 91 ô lệch của LVT1916 nằm trong **25/986 cột**, 32 ô của TK1872 trong **7/1.608 cột**,
7 ô của LVT1883 trong 6 cột, 16 ô của KVK trong 11 cột. Trong cột trôi, số ô lệch tăng dần theo `syl_idx`
(LVT1916: 3 ô ở vị trí 1 → 14 ô ở vị trí 11) và **74/91** (LVT1916) · **24/32** (TK1872) nằm ở **tầng dưới** (câu bát) —
đúng hình ảnh "hở một khe rồi kéo lệch phần còn lại của tầng".

### 2.2 Nguồn gốc của trôi: ĐẾM ÂM QN SAI, không phải hình học

Cắt chéo ô lệch theo tình trạng đếm của cột (nhãn người, bảng `n_ocr`/`n_qn` có sẵn trong `labels_gated.csv`):

| Nhóm cột | LVT1916: cột · cột có ô lệch · ô lệch | TK1872 |
|---|---|---|
| `n_qn ≠ 14` (QN đếm sai) | 22 · **15 (68,2 %)** · **78** | 6 · **3 (50,0 %)** · **24** |
| `n_qn = 14` và `n_ocr = 14` | 950 · 10 (1,05 %) · 13 | 1.534 · 4 (0,26 %) · 8 |
| `n_qn = 14`, `n_ocr ≠ 14` (kim đếm sai) | 14 · **0** · **0** | 68 · **0** · **0** |

⟹ **85,7 % (78/91) và 75,0 % (24/32) toàn bộ trôi nằm trong 22 và 6 cột `n_qn ≠ 14`.** Cột kim đếm sai mà QN đúng: **trôi bằng 0** —
luật 6/8 + `pitch_decode` đã chặn trọn nhánh này. Hệ quả chất lượng, đo trên nhãn người:

| | GOLD trong cột `n_qn ≠ 14` | GOLD trong cột `n_qn = 14` |
|---|---|---|
| LucVanTien1916 | **109/209 = 52,2 %** | 11.355/11.543 = **98,4 %** |
| TruyenKieu1872 | **25/51 = 49,0 %** | 17.804/18.035 = **98,7 %** |

Tức ô GOLD trong cột đếm sai gần như **tung đồng xu**, trong khi phần còn lại đạt 98,4–98,7 %.
Cùng hướng: `count_source = pitch_ocr` (nhánh ép N = n_ocr khi QN lệch) đúng **72,6 %** (n 62, LVT1916) và **94,0 %** (n 117, TK1872)
so với `pitch` 97,7 % / 98,6 %. Phơi nhiễm ở hai bộ chốt: **LVT1883 27/1.041 cột (2,6 %)** và **KVK1884 26/1.628 (1,6 %)** có `n_qn ≠ 14`,
và **cả 27 + cả 26 cột ấy kim đều đọc đủ 14 chữ** ⇒ bên sai chắc chắn là QN. Số ô liên quan: 364 (2,51 % bộ LVT1883, trong đó 250 GOLD)
· 350 (1,54 %, 285 GOLD) · 279 (2,03 %) · 77 (0,34 %).

### 2.3 Lỗi đánh số câu của 2 bộ IHR (A11/A12) — gốc rễ đã định vị

`pipeline/tools/ingest_ihr_book.py:118` `seq += len(verses)` cộng **số câu thô** của trang, còn
`verse_pairs_for_page` (`:138-140`) lại giả định `first_seq` **lẻ** (cột k ↔ câu `first_seq+2k`, `+1`).
Trang nào có **số câu LẺ** sẽ đảo parity cho **mọi trang sau**:

* LucVanTien1916: 4 trang lẻ (ảnh 12 = 21 câu, 13 = 21, **82 = 19**, 89 = 21). Trang 12↔13 và 82↔89 bù nhau từng cặp ⇒ chỉ
  `page_0083…page_0089` hỏng: **61/988 cột** có `verse_odd` là số **chẵn**. Thêm 83 số câu trống vì 5 trang bị cổng loại ⇒ A12 = 84.
* TruyenKieu1872: 1 trang lẻ (`page28b`, ảnh thứ 54, 23 câu) ⇒ **1.078/1.610 cột (67,0 %)** từ `page_0055` đến hết bị đảo parity;
  mỗi lần đảo còn làm **1 số câu bị dùng hai lần** (câu 1084).

Đây là lỗi **sổ sách**, không phải lệch ô: ghép ô↔chữ của nhóm này vẫn đúng 98,58 % (§2.1) vì GT khớp theo (trang, cột, vị trí)
chứ không theo số câu. Nhưng nó làm **`verse_no` công bố trong `transcriptions/*.json` sai từ trang đó trở đi** và khiến
"số câu in" **không dùng được làm neo cứng** cho 2 bộ IHR.

### 2.4 Chrestomathie1872 và STT — vùng KHÔNG đo được trôi

Không bộ nào có chuẩn độc lập; chỉ đo được *phơi nhiễm*: Chresto **234/418 cột (56,0 %)** có `n_ocr ≠ n_qn`
(lệch +1 ở 108 cột, +2 ở 45; QN thường **thừa** âm), **154/417 cột `dp_ratio` < 0,75** (trung vị 0,783), 12 cột giữ chỗ không ghép được,
58/418 cột dùng `pitch_ocr`. STT: cột `n_ocr ≠ n_qn` 340/1.440 (23,6 %) · 322/1.303 (24,7 %) · 437/1.286 (34,0 %),
`count_source = conflict` 89 · 101 · 136 cột (6,2–10,6 %). Bất biến A1–A8 vẫn PASS ở cả bốn bộ này, nhưng **PASS ở đây
chỉ nói "gán tự nhất quán", không nói "gán đúng"** — không có chuẩn thì không kết luận được.

## 3. Đối chiếu thiết kế hiện tại với quy hoạch §5 `CHOT_KENH_OCR_VA_QUY_HOACH_GAN_2026-09-23.md`

*Số dòng trích dẫn ứng với cây làm việc lúc viết (HEAD `014a0733fa` + các sửa đang diễn ra của phiên song song); nếu lệch, tra theo TÊN HÀM.*

| §5.2 | Trạng thái | Bằng chứng / ước lượng phần còn thiếu |
|---|---|---|
| 1. Trang: 10 cột × 2 tầng, cổng `first_seq` | **ĐÃ** | `lithograph_gate` (`book_layout.py:286`); trang đủ 10 cột 103/105 · 162/163 · 97/99 · 159/161 |
| 2. Câu = neo cứng #1 (số câu in) | **ĐÃ với LVT/KVK, CHƯA với IHR** | LVT1883 `anchor_source` ocr_num 81/105 trang, `margin_digits` 24; còn **76 câu `value_conflict`, 4 `no_anchor`**. KVK: 125 + 37 trang, còn **162 `ambiguous` + 92 `no_anchor` + 86 `value_conflict` + 42 `drift`**, **27 trang `verse_map` khác công thức**. IHR: đếm chạy ⇒ §2.3 |
| 3. Tầng-cột = neo cứng #2 | **ĐÃ** | `tier_rule_for`/`expected_tier_counts` (`book_layout.py:340`/`:350`), `column_tiers` (`align_production.py:662`) → A9 "đủ 14 âm mà chia sai" = **0/1.044, 0/1.628, 0/988, 0/1.610** |
| 4. Sửa số đếm QN trước DP | **ĐÃ một phần** | `repair_tier_syllables` (`ingest_lithograph_book.py:310`) + `resplit_couplet` (`:341`) đã sửa **138 câu (LVT1883) / 97 (KVK) / 7 (LVT1916) / 0 (TK1872)**; còn **`count_unfixed` 27 / 29 / 34 / 21 câu** ⇒ đúng 27/26/22/7 cột hỏng ở A9 và 76–86 % trôi (§2.2) |
| 5. DP chạy TRONG từng tầng | **ĐÃ** | `tier_dp: true` trong 4 config; `realign_column_tiered` (`anchor_align.py:232`) → A10 xuyên tầng **0** trên cả 4 bộ lục bát |
| 6. `tier_n` = 6/8 theo luật | **ĐÃ** | như dòng 3 |
| 7. Cờ `tier_barrier_moved` | **CHƯA** | không có trong mã lẫn trong 8 bộ nhãn; `qn_count_unfixed` chỉ tồn tại trong **chú thích** (`ingest_lithograph_book.py:299`), **không cột nào của `labels*.csv` mang cờ này** ⇒ hạ nguồn không lọc được cột đếm hỏng |
| 8. Chresto: DP mức truyện + neo mềm pitch | **DP truyện ĐÃ; neo pitch CHƯA** | A13 span liền mạch 0/392 vi phạm, nhưng 56,0 % cột còn `n_ocr ≠ n_qn` (§2.4) — neo "số chữ/cột = pitch" chưa được dùng làm ràng buộc |

### 3.1 Luật của STT còn dính vào sách mới

* `x_tol = 15 px` cứng (`core/ocr/ocr_api.py:489`) + `_merge_fragment_columns` (`:374`): **chỉ đường STT** dùng
  (`ocr_api.py:789`); 3 adapter sách mới tự gán hộp theo hình học đã đo ⇒ **không còn dính**.
* `step2.det_thr = 0,2` / `det_xmargin = 0,25` (`config/pipeline.yaml`) là hiệu chuẩn chữ thảo; 5 sách mới đã ghi đè
  **0,15 / 0,05** theo sách ⇒ **không còn dính**.
* **Còn dính**: `pitch_target_count` (`align_production.py:626`) vẫn **ưu tiên `n_qn`** — logic sinh ra từ STT nơi QN có
  đánh số dòng đáng tin. Trên sách mới, QN mới là bên yếu: 27/26 cột LVT/KVK có `n_qn ≠ 14` mà kim đọc đúng 14. Nhánh
  `pitch_ocr` (N = n_ocr) chỉ bật khi `n_det == n_ocr ≠ n_qn` nên **không cứu** các cột này; đo được nó cũng kém hơn (§2.2).
* **Còn dính**: cổng `lithograph_gate` nhận `num_syllables` của JSON làm kỳ vọng (`book_layout.py:286-300`), nên cột 13/15 âm
  vẫn **qua cổng** — cổng chỉ kiểm "khớp với chính mình", không kiểm "khớp luật 6/8".

## 4. Đề xuất sửa, xếp theo lợi/chi phí (KHÔNG tự sửa mã)

| # | Sửa gì | Tệp:dòng | Lợi (số đo §2) | Rủi ro STT | Nghiệm thu |
|---|---|---|---|---|---|
| 1 | **Gắn cờ + hạ cấp ô thuộc cột `n_qn ≠ sum(tier_rule)`** (không xoá): thêm cờ `qn_count_unfixed` vào `labels*.csv` (dữ liệu đã có sẵn cột `n_qn`) và loại chúng khỏi tier GOLD | `ingest_lithograph_book.py:310-339` (ghi cờ ra JSON), `align_production.py:838-913` (chuyển cờ vào bản ghi ô), `build_dataset.py` (cổng tier) | Bỏ **85,7 % / 75,0 %** ô trôi; các ô này chỉ đúng **52,2 % / 49,0 %** (n 209/51) so với 98,4/98,7 % phần còn lại. Chi phí: **364 ô (2,51 %) LVT1883 · 350 (1,54 %) KVK · 279 (2,03 %) LVT1916 · 77 (0,34 %) TK1872** | **0** — STT không có `tier_rule` (`tier_rule_for` trả None) nên cờ luôn rỗng; kiểm md5 `labels_final.csv` STT | A9 không đổi; GOLD trong cột `n_qn ≠ 14` phải về **0**; chạy lại `align_audit --book all` → `lech_vi_tri` ≤ 15 ô (LVT1916) và ≤ 10 (TK1872) |
| 2 | **Sửa đánh số câu IHR**: `seq += 2 * ceil(len(verses)/2)` (hoặc `2 * n_cols`) để parity không đảo; ghi cờ trang có số câu lẻ | `ingest_ihr_book.py:118` (+ `:138-140` giữ nguyên) | A11 **1.078/1.610 → 0** (TK1872), **61/988 → 0** (LVT1916); A12 hết **1 số câu trùng** mỗi lần đảo. Không đổi nhãn ô (ghép theo trang/cột/vị trí) | **0** — chỉ adapter IHR | A11 = 0/1.610 và 0/988; A12 chỉ còn phần "thiếu" do trang bị cổng loại; `ihr_endtoend_eval` precision không giảm |
| 3 | **Nới luật sửa đếm QN** cho 4 ca còn lại: cột 13 âm mà kim đủ 14 → chèn `khongdoc` **ở đúng khe mực rộng nhất của tầng**; cột 15 âm → gộp cặp token không phải âm QN kề nhau (nới `merge_unreadable` sang cả tầng bát) | `ingest_lithograph_book.py:310-339` | Cứu một phần **27 + 26 + 22 + 7 = 82 cột** (≈1.070 ô) mà #1 chỉ hạ cấp. Trần lợi ích = đúng lượng ô #1 loại bỏ | 0 (chỉ `layout=lithograph`) | A9 giảm; `lech_vi_tri` không tăng; GOLD trong cột được cứu phải đạt ≥ 95 % trên nhãn người IHR |
| 4 | **Thắt cổng `lithograph_gate`**: khi `tier_rule` có hiệu lực, kỳ vọng lấy **luật 6/8** chứ không lấy `num_syllables` của JSON ⇒ cột 13/15 âm **trượt cổng** thay vì qua ngầm | `book_layout.py:286-300` (`expected_counts`) | Biến "cột đếm hỏng" thành sự kiện **nhìn thấy được ở mức trang** (hiện 27/26/22/7 cột đi lọt); là cổng đôi cho #1 | 0 (`tier_rule_for` = None với STT/prose) | `page_ok` giảm đúng bằng số trang chứa cột `n_qn ≠ 14`; không trang nào khác rớt |
| 5 | **Chresto: dùng pitch cột làm neo mềm** cho DP mức truyện (số chữ/cột từ `pitch_decode` thay vì để `n_qn` tự do) + hạ cấp cột `dp_ratio < 0,75` | `ingest_prose_book.py` (bước ghép truyện), `align_production.py:626` | Chạm **234/418 cột (56,0 %)** `n_ocr ≠ n_qn` và **154/417 cột `dp_ratio` < 0,75`**; chưa ước lượng được lợi vì **không có chuẩn độc lập** cho Chresto | 0 | A13 giữ 0 vi phạm; tỉ lệ cột `n_ocr == n_qn` tăng; `loss_ledger` mục (e) căn chỉnh giảm |

Thứ tự đề nghị: **1 → 2 → 4 → 3 → 5**. #1 và #4 là cùng một lần chạy lại; #2 độc lập (chỉ 2 bộ đánh giá); #3 mới cần đo lại nhãn người.

## 5. Giới hạn của chính các phép đo trên

1. **Chỉ 2/8 bộ có nhãn người.** Mọi số "trôi" đáng tin (0,72 % và 0,16 %) đến từ LVT1916/TK1872. Suy ra cho LVT1883/KVK1884
   là **ngoại suy** dựa trên cùng adapter + cùng config.
2. **Phép đo dị bản mù đúng chỗ cần nhìn**: `match_verses` chỉ nhận câu có **đúng** số âm bằng số chữ của dị bản, nên chỉ
   **7/609** (LVT1883) và **42/4.792** (KVK) câu khớp đến từ cột `n_qn ≠ 14`. Vì vậy "0/609 và 1/4.792 câu trôi" **không**
   bác bỏ được §2.2; nó chỉ nói phần cột đếm đúng thì không trôi.
3. **KVK1884 có vòng tròn**: 3.980/4.792 câu (83,1 %) lấy QN từ chính bản 1871 (B1'), rồi lại so với Nôm 1871. Tách riêng
   812 câu QN độc lập: đúng **75,74 %** (thay vì 80,58 %) nhưng lệch vị trí vẫn **0,19 %** ⇒ chỉ số *trôi* bền, chỉ số *đúng* thì không.
4. **"Sai chữ" ở 2 bộ thạch bản gồm khác biệt dị bản thật** (nền 1871↔1872 = 82,9 %), không đọc là lỗi pipeline.
5. A3/A4/A5 dựa trên `syllable` ghi trong `labels.csv` so với âm QN mà **chính engine** đọc (`step2_align._get_qn_lines`).
   Nếu tệp `transcriptions/` bị ghi đè sau khi build thì phép so này mất hiệu lực — vì vậy §0 ghi md5 + giờ sửa.
6. **Chresto và STT không có bất kỳ chuẩn ngoài nào**; A1–A8 PASS ở đó chỉ chứng minh *tự nhất quán*.
7. n nhỏ ở vài chỗ: GOLD trong cột `n_qn ≠ 14` chỉ n = 209 và 51 (Wilson 95 % ≈ 45–59 % và 35–63 %) — dấu chắc chắn, độ lớn thì không.
