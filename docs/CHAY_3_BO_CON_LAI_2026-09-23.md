# Ba bộ còn lại trong `data/` — đo bố cục, chạy pipeline, ĐỘ ĐÚNG END-TO-END THẬT (23/09/2026)

> 🔴 **CẬP NHẬT 23/09 — PTCL (bản chép tay R.987) LOẠI KHỎI PHẠM VI, NHƯNG TỆP KIỀU 1871 GIỮ LẠI.**
> Bốn tệp mô tả bản chép tay (`README_tai_ve.md`, `SOURCE.md`, `metadata.json`, `reference_kieu_1872.tsv`)
> đã bị xoá khỏi đĩa; lý do đã đo: tầng trên mỗi cột là **lời bình chữ Hán**, không có số câu neo, QN chỉ có dị bản.
> **`data/TruyenKieuPhongTinhCoLuc/thamchieu_kieu_1871_LieuVanDuong_phienam.json` ĐƯỢC GIỮ LẠI** (vòng 8): nó chỉ
> *nằm nhờ* trong thư mục ấy, còn nội dung là **bản phiên âm Truyện Kiều 1871 (Liễu Văn Đường)** — tham chiếu dị bản
> của **KimVanKieu1884**, không phải dữ liệu của PTCL. Vì vậy vòng 8 đã **bật lại**: khối `run.verses_ref_fix` trong
> `config/pipeline_KimVanKieu1884_b1.yaml` (bước B1'), `CROSS_BOOKS["KimVanKieu1884"].refs[0] = Kieu1871_LVD`
> (`auto_precision.py`) và `REFS["KimVanKieu1884"]` (`qn_engine_compare.py`) → "nền dị bản 1871↔1872" tính được trở lại
> và số trôi của KVK trùng khít mốc công bố (n 31.866 · đúng 80,58 % · lệch vị trí 16).
> `measure.py` / `verses_ref_fix.py` / `run_pipeline.sh` vẫn giữ cách xử lý "thiếu tệp → cảnh báo, bỏ bước".
> Chi tiết: `docs/VONG8_CAN_CHINH_VA_CROP_2026-09-23.md` §6. Phần dưới đây giữ nguyên làm hồ sơ lịch sử.

Trên HEAD `12a00ede3e`, nhánh `main`. **Chưa commit gì.** Mọi số trong tài liệu này do script sinh (0 token LLM);
tái lập bằng các lệnh ghi ở §0. `data/` chỉ được ĐỌC, không sửa/ghi đè bất cứ tệp nào.

Ba bộ: **A. `data/LucVanTien1916/`** và **B. `data/TruyenKieu1872/`** (mộc bản, IHR-NomDB, **có nhãn người từng chữ**),
**C. `data/TruyenKieuPhongTinhCoLuc/`** (bản chép tay R.987, chỉ có dị bản 1871/1872 làm tham chiếu).

> **Kết luận một dòng.** A và B chạy trọn pipeline và lần đầu tiên cho **độ đúng end-to-end đo bằng nhãn NGƯỜI trên toàn
> sách**: GOLD 97,2 % (A) / 98,5 % (B) — cao hơn mốc đo gián tiếp 22/09 (89,3 / 84,6 % trên patch) **+7,9** và **+13,9** điểm, coverage
gấp 1,6–1,8 lần; **hai bộ này là TẬP ĐÁNH GIÁ, không được trộn vào tập huấn luyện**. C **KHÔNG chạy trong vòng này**:
> phép đo bố cục (§3) cho thấy bố cục thật khác hẳn giả định ban đầu (mỗi cột có **2 tầng**: tầng trên là **lời bình
> chữ Hán**, tầng dưới mới là 1 cặp lục bát), nên adapter hiện có không dùng được; §3.4 ghi đường chạy đã định lượng.

---

## 0. Tái lập

```bash
PY=.venv/bin/python
# (1) ĐO BỐ CỤC trước khi code — 3 bộ, đã gắn vào runner chung
$PY scripts/measure/measure.py --book LucVanTien1916,TruyenKieu1872,TruyenKieuPhongTinhCoLuc
$PY scripts/measure/ihr_layout.py  --book all --kim-pages 10 --scales 1,3   # A, B: bố cục + hệ số phóng kim
$PY scripts/measure/ptcl_layout.py --kim-pages 6                            # C: bố cục bản chép tay
# (2) CHẠY (B0→B6). kim cache theo md5 -> chạy lại 0 lượt
./run_pipeline.sh --book LucVanTien1916 ; ./run_pipeline.sh --book TruyenKieu1872   # hoặc --book all-ihr
# (3) ĐO ĐỘ ĐÚNG THẬT so nhãn người
$PY scripts/measure/ihr_endtoend_eval.py --book all
# (4) Selftest
$PY -m pipeline.tools.ingest_ihr_selftest ; $PY scripts/measure/ihr_layout.py --selftest
$PY scripts/measure/ptcl_layout.py --selftest ; $PY scripts/measure/ihr_endtoend_eval.py --selftest
```

Đầu ra: `measure_out/<Book>/{ihr_layout,ihr_endtoend,ptcl_layout}/`, `prepared_ihr/<Book>/`,
`prepared_ihr/<Book>/dataset_out/`, `dataset/<Book>/`. Mã mới: `scripts/measure/{ihr_layout,ptcl_layout,ihr_endtoend_eval}.py`,
`pipeline/tools/{ingest_ihr_book,mark_eval_dataset}.py` (+ selftest), `config/pipeline_{LucVanTien1916,TruyenKieu1872}.yaml`,
`run_pipeline.sh` (nhánh `ingest: ihr`, `--book all-ihr`). **Không sửa** `pipeline/align_engine/`, `core/`, `data/`.

---

## 1. ĐO BỐ CỤC — A và B (`scripts/measure/ihr_layout.py`)

### 1.1 Cột và câu

| Chỉ số | A · LucVanTien1916 | B · TruyenKieu1872 |
|---|---|---|
| Trang ảnh / trang có ô cột VoTT | 104 / 103 | 162 / 162 |
| Cột/trang (histogram từ `bboxes.json`) | **10 ở 100 trang**; 11·1, 7·1, 4·1, 0·1 | **10 ở 159 trang**; 11·1, 12·1, 8·1 |
| **Mỗi cột là 1 CẶP lục bát hay 1 CÂU?** | **1 CẶP** — 997/998 cột có đủ `part` 1 và 2 (**99,9 %**) | **1 CẶP** — 1.529/1.530 (**99,9 %**) |
| Số chữ mỗi câu đúng luật 6 (lục) / 8 (bát) | 1.968/1.995 = **98,7 %** | 3.056/3.059 = **99,9 %** |
| Số ô cột == `ceil(số câu/2)` (kiểm chéo độc lập) | 99/104 = **95,2 %** | 161/162 = **99,4 %** |
| Cột 0 nằm PHẢI nhất (đúng chiều đọc) | 103/103 trang | 162/162 trang |
| px/chữ (cao ô cột / 14), trung vị [min–max] | **37,6** [35,8–38,4] | **40,0** [38,5–41,3] |
| Bước cột (px, trung vị) | 37,7 | 36,4 |
| Kích thước trang (trung vị) | 495 × 763 | 397 × 581 |

> **Sửa một giả định của đề bài.** Đề bài giả định "mỗi cột = 1 CÂU (6 hoặc 8 chữ)". Phép đo bác bỏ: `manifest.tsv`
> ghi **hai** dòng (`part` 1 và 2) cho **cùng một** `col_index` và **cùng một** `col_bbox_xywh` ở 99,9 % số cột —
> tức một ô cột chứa **cả cặp 6⧺8 = 14 chữ**, y như thạch bản LVT1883/KVK1884. `patches/` cắt theo CÂU (2 patch/cột)
> nên nhìn vào `patches/` dễ tưởng cột = câu. Nhờ vậy **không cần layout mới**: dùng thẳng `layout: lithograph`.

Đối chứng độc lập bằng chiếu mực (không đọc `bboxes.json`): |số cột chiếu mực − số cột bbox| ≤ 1 ở **17/20** trang (B)
nhưng **0/20** (A) — bản quét 37–40 px/chữ làm tự tương quan khoá vào hài. Đây là **giới hạn của phép đối chứng**, không
phải của bố cục: bố cục chính thức lấy từ ô cột **do người vẽ** và đã được kiểm chéo bằng **số câu** (dòng 4 bảng trên).
Hai invariant này được xếp **mềm** trong `measure.py` kèm lý do.

### 1.2 Hệ số phóng ảnh cho kim (10 trang/sách, có nhãn người nên đo ĐỘ ĐÚNG THẬT, không cần chỉ số thay thế)

| | A ×1 | A ×3 | B ×1 | B ×3 |
|---|---|---|---|---|
| Cột kim đọc đủ 14 chữ | **100 %** | 99,0 % | 95,0 % | **97,0 %** |
| **kim == chữ người gán** (vị trí-với-vị trí) | **97,2 %** | 97,3 % | 98,2 % | **98,5 %** |
| kim ∈ R(âm QN) | 98,8 % | 98,7 % | 98,8 % | 98,8 % |
| Thu hồi LCS (không phụ thuộc số đếm) | 96,8 % | 96,9 % | 98,2 % | 98,4 % |

**Chốt `--scale 1`** (không phóng ảnh): ×3 hơn ×1 **0,1 điểm (A) / 0,3 điểm (B)** — trong nhiễu, mà tốn 9× số điểm ảnh.
Lý do sâu hơn: detector CenterNet **letterbox trang về 1024 px**, nên trang 495×763 cho chữ ~50 px *trong không gian
detector* — **đúng bằng** thạch bản (154 px/chữ trên trang 3.204 px). Phóng ảnh không đổi gì ở khâu detector; nó chỉ
làm ảnh crop xuất ra to hơn. Cờ `--scale N` vẫn có sẵn trong adapter cho ai cần crop độ phân giải cao.

> **Phát hiện đi ngược số cũ — và giải thích được.** Bản chẩn đoán 22–23/09 ghi "kim thô đúng **49,2 %** (LVT1916) /
> **42,3 %** (Kiều1872)" trên GT người. Số ấy đo trên **patch từng CÂU** (41×228 px, 6 hoặc 8 chữ). Ở đây kim đọc **cả
> trang** và đúng **97,2 / 98,2 %**. Hai số không mâu thuẫn: `CHOT_KENH §2` đã đo rằng cắt ảnh xuống mức **tầng-cột**
> làm kim rụng gần nửa số chữ (đọc 34/60 chữ, đúng 4,3 % ở LVT1883) — patch IHR chính là mức tầng-cột ấy.
> ⇒ **Mốc "kim chỉ đúng 42–49 % trên mộc bản" phải bỏ**: đó là chỉ số của *cách đo*, không phải của kênh OCR.

---

## 2. CHẠY A và B — cấu hình, adapter, kết quả

### 2.1 Adapter mới `pipeline/tools/ingest_ihr_book.py` (selftest **64/64**)

Vì sao không dùng `ingest_lithograph_book`: adapter thạch bản đọc bố cục từ `measure_out/<book>/layout/*` (tự dò tầng,
bước cột, số câu in) và ghép câu bằng **số câu in ở lề**. Hai bộ IHR **có sẵn** ô cột người vẽ và QN câu-với-câu, không
có số câu in. Adapter mới:

1. Ô cột từ `pages/bboxes.json` (VoTT, tag `Column`), sắp **phải→trái** → cột 1..10.
2. QN từ `pages/annotation.json` → `translation` (nối các mảnh bằng dấu cách); cột k ↔ câu `2k`, `2k+1`.
3. **Cổng cứng trước khi ghi**: số ô cột phải bằng `ceil(số câu/2)`; không đạt → **bỏ trang** + ghi cờ.
4. Ranh giới tầng **theo từng cột**: 14 chữ chia đều chiều cao ô cột, cắt sau chữ thứ 6. Chữ kim vào cột theo tâm x
   (≤ 0,5 × bước cột), vào tầng theo tâm y.
5. **Cân lại tầng theo luật 6/8**: cột đủ 14 chữ nhưng chia tầng ≠ 6+8 → ép về (6, 8) (luật thể thơ là bất biến của bản
   in, thắng ranh giới hình học). Chạm **1 cột** (A) — ranh giới hình học gần như luôn đúng.
6. Ghi đúng hợp đồng engine của thạch bản (`pages/`, `pages_denoised/`, `detected/*_ocr_cache.json` có `tier_split`,
   `transcriptions/*.json|txt`, `kim_raw/`, `manifest.json`) nên **engine, cổng B4', export không phải sửa một dòng nào**.

**Rào không lẫn nhãn người vào nhãn máy** (3 phép kiểm trong selftest, soi MÃ THẬT sau khi bóc docstring bằng AST):
adapter không tham chiếu `hn_text`, không tham chiếu `nom_text`, và chuỗi `manifest.tsv` chỉ xuất hiện ở khoá metadata
`gt_file` (ghi đường dẫn, không mở đọc). `manifest.json` mang `evaluation_only: true` + ghi chú tập đánh giá. Sau bước export, `run_pipeline.sh` gọi
`pipeline/tools/mark_eval_dataset.py` (selftest 15/15) ghi `TAP_DANH_GIA.md` + `evaluation_only.json` vào
`dataset/<Book>/` — cờ máy đọc được để cổng CI về sau từ chối bộ này khi dựng split huấn luyện.

### 2.2 Config

`config/pipeline_LucVanTien1916.yaml`, `config/pipeline_TruyenKieu1872.yaml`: `layout: lithograph`, `n_columns: 10`,
`qn_syllables_per_column: 14`, `det_xmargin: 0.05`, `det_thr: 0.15`, `box_decoder: pitch`, **`kim_lang_type: 2`**,
`tier_dp: true`, `detector_resize: linear`; `paths.data_dir: prepared_ihr`, `paths.output_dir: dataset/<Book>`;
khối `run:` {`ingest: ihr`, `ingest_args: --qn-count-rule`, `dataset_out: prepared_ihr/<Book>/dataset_out`,
`n_columns: 10`, `cross: false`, `measure_steps: ihr_layout`}. `cross: false` vì hai bộ này **không cần** cổng dị bản —
độ đúng đo thẳng bằng nhãn người (§4), chính xác hơn mọi phép so dị bản.

### 2.3 Kết quả chạy

| | A · LucVanTien1916 | B · TruyenKieu1872 |
|---|---|---|
| Trang ingest / bỏ theo cổng | **99 / 5** | **161 / 2** |
| Lượt gọi kim | 94 (+5 đã có từ bước đo) | 161 |
| Cột ghi ra | 988 | 1.610 |
| Cột kim đọc đủ 14 chữ / tách đúng 6+8 | 972 (98,4 %) / 972 (**100 %**) | 1.541 (95,7 %) / 1.541 (**95,7 %**) |
| Cột QN đủ 14 âm | 966 (97,8 %) | 1.603 (99,6 %) |
| **Ô sinh ra** (`labels.csv`) | **13.761** | **22.499** |
| tier thô: GOLD / SYLLABLE / REVIEW | 13.298 / 86 / 377 | 21.996 / 47 / 442 |
| Sau cổng B4': **GOLD ảnh** / GOLD_text_only / SYLLABLE / REVIEW | **8.783** / 3.424 / 89 / 1.465 | **13.465** / 6.694 / 48 / 2.278 |
| **Ảnh export** | **8.872** | **13.513** |
| Nhãn cấp ký tự trong `dataset/<Book>/labels.csv` | **12.207** | **20.159** |

Trang bị bỏ theo cổng: A `nlvnpf-0059-{007, 013, 014, 090, 105}`, B `page24a`, `page79b` — số ô cột ≠
`ceil(số câu/2)` (bộ IHR thiếu/thừa ô cột ở đúng các trang này); **không** bỏ trang nào vì lỗi ảnh.
Ô không sinh ra so với nhãn người: A 230/13.991 chữ (**1,6 %**), B 31/21.410 (**0,1 %**).

**Một điểm bất đối xứng của B cần nhớ**: 8 trang B được pipeline gán nhãn (**1.120 ô**) nhưng `manifest.tsv` **không có
nhãn người** cho chúng (IHR thiếu `col_index`/patch ở các trang ấy, dù `bboxes.json` vẫn có ô cột). Những ô này **không
đo được** — chúng nằm trong 22.499 ô sinh ra nhưng ngoài 21.379 ô được chấm ở §4. A không có tình huống này
(cổng `ceil(số câu/2)` đã loại đúng các trang thiếu nhãn).

---

## 3. C · `data/TruyenKieuPhongTinhCoLuc/` — ĐO BỐ CỤC (`scripts/measure/ptcl_layout.py`, invariants 7/7 PASS)

### 3.1 Số đo

| Chỉ số | Giá trị |
|---|---|
| Ảnh | 120 tờ, **2000 × 1820** (tờ đôi), 0 tờ trắng |
| Ô cột theo lưới chiếu mực | **19 ở 105/120 tờ** (20·6, 18·3, 15·3, 25·3); bước cột **107 px** |
| Cột có chữ ở nửa trái / nửa phải (trung vị) | 8 / 11 |
| **Ranh giới tầng** (cực tiểu chiếu mực theo hàng, trong dải 0,15–0,45 × cao) | y ≈ **488** (kim cho ≈ 555) |
| Bước chữ **tầng dưới** | **74,8 px** |
| Số chữ tầng dưới mỗi cột — chiếu mực | trung vị **15** |
| Số chữ tầng dưới mỗi cột — **kim** (6 tờ, 92 cột) | trung vị **14,0**; đúng 14 ở **41/88 = 46,6 %**; trong [13,15] ở **75/88 = 85,2 %** |
| Nền dị bản 1871 ↔ 1872 (chữ trùng, cùng chỉ số câu) | **71,3 %** (n 7.490) |

### 3.2 Bố cục thật (đọc được từ chính chuỗi chữ kim, không đoán)

Mỗi ô cột có **HAI TẦNG**, ranh giới là một **đường ngang chung cho cả tờ**:

* **Tầng trên** (y ≲ 555): **lời bình / bài tựa bằng chữ HÁN**, chữ **nhỏ** (cao/chữ ≈ 61 px), thường chia **2 cột con**
  trong cùng một ô cột (kiểu 雙行小註). Ví dụ tờ 004 cột 4: `須知蒼昊之憐` (6 chữ).
* **Tầng dưới** (y ≳ 555): **ĐÚNG MỘT CẶP LỤC BÁT 14 chữ Nôm**, chữ **to** (cao/chữ ≈ 77 px).
  Cùng tờ 004 cột 4: `浪紅顔自刼𠸗丐條薄命固除埃兜` = 14 chữ.

Kiểm chứng số học: gióng chuỗi chữ kim của từng tờ vào chuỗi Nôm bản 1871 bằng LCS trượt cửa sổ cho vị trí **đơn điệu
theo thứ tự tờ** (tờ 001→002→…→006 rơi vào ký tự 0 → 50 → 225 → 525 → 725 → 925), tức **≈ 185 chữ Kiều/tờ ≈ 26 câu
≈ 13 cặp/tờ**; 120 tờ × 13 ≈ 1.560 cặp ≈ 3.120 câu, so với 3.254 câu của Kiều — **khớp**. Cũng chính vì vậy mà "19 ô
cột/tờ" ở §3.1 **không** phải 19 cột chữ: lưới phủ hết bề ngang 2000 px nên có cả ô lề trống; số cột **có chữ** thực tế
là ~14 (tờ 004: ô 3…16) — **khớp với `data/README.md`** ("7 cột/trang"): 7 cột × 2 nửa tờ = 14 cột/tờ đôi.
`data/README.md` cũng đã ghi đúng "2 tầng (tầng trên = tựa/đề vịnh, phải bỏ)"; phép đo ở đây bổ sung **ranh giới
tầng là số đo được** (y ≈ 488–555, chung cả tờ), **tầng trên viết bằng chữ Hán cỡ nhỏ, thường 2 cột con**, và
**tầng dưới chứa CẢ CẶP 14 chữ** (chứ không phải 1 câu) — ba điều quyết định adapter phải viết thế nào.

### 3.3 Vì sao KHÔNG chạy C trong vòng này (4 lý do đo được, không phải cảm tính)

1. **Adapter hiện có không khớp bố cục.** `ingest_lithograph_book` giả định tầng trên = câu lục (6) và tầng dưới =
   câu bát (8). Ở C, tầng trên là **lời bình chữ Hán không thuộc tác phẩm** và tầng dưới chứa **cả cặp 14 chữ**. Chạy
   nguyên trạng sẽ gán âm QN của câu lục cho lời bình chữ Hán — sai hệ thống 100 % ở tầng trên.
2. **Không có neo câu.** Bản chép tay không đánh số câu; `reference_kieu_1872.tsv` và bản 1871 là **dị bản**, chỉ dùng
   để gióng bằng nội dung. Mà kim đọc chép tay yếu: thu hồi LCS so bản 1871 chỉ **35–44 %** (6 tờ), trong khi trần thực
   tế của phép so này — **nền dị bản 1871↔1872 — là 71,3 %**. Tức kim đạt ≈ 0,5–0,6 lần trần ⇒ DP gióng theo nội dung
   sẽ trôi.
3. **Đếm chữ tầng dưới chưa đủ chắc để làm ràng buộc cứng.** kim cho đúng 14 ở **46,6 %** cột (so với **100 %** ở A và
   98,4 % ở B). Ràng buộc 6/8 — thứ đã cứu LVT1883/KVK1884 — ở đây chưa dùng được.
4. **Nhãn sẽ mang rủi ro sai hệ thống, và rủi ro ấy KHÔNG đo được trên chính C.** QN dùng để gán là dị bản; nền dị bản
   71,3 % nghĩa là ~29 % vị trí, bản 1871 và 1872 vốn đã ghi **chữ khác nhau**. Luật GOLD (`kim ∈ R(âm)`) lọc phần lớn
   ca kim đọc sai, nhưng **không** lọc được ca kim đọc đúng chữ của R.987 mà âm dị bản lại trỏ sang chữ khác: khi đó
   `R(âm sai)` vẫn có thể chứa chữ kim và ô lọt GOLD với nhãn của **bản khác**. Cận trên của lớp lỗi này ≈ **1 − 71,3 %
   ≈ 29 %** số vị trí; phần thật sự thành nhãn sai chỉ đo được khi có phiên âm của **chính R.987**, mà bộ này không có.

⇒ Nếu vẫn muốn bộ ra, phải khai `qn_source: di_ban` cho **mọi** ô và đóng dấu "nhãn theo dị bản, chưa kiểm" — nhưng
theo cổng nghiệm thu đang dùng (`không chỉ số nào giảm`), bộ ấy không đủ điều kiện vào bộ giao nộp.

### 3.4 Đường chạy C đã định lượng (việc còn lại, ~1 ngày công + ~130 lượt kim)

1. **Tách tầng ở mức tờ** (đã có: `ptcl_layout.analyze_page` trả `tier_y` cho cả 120 tờ). Thêm cổng: tờ có
   `|tier_y − trung vị| > 0,1 × cao` → xem tay.
2. **Adapter `ingest_ptcl_book.py`**: bỏ toàn bộ chữ tầng trên; mỗi ô cột có chữ ở tầng dưới = **1 cặp**; cột theo
   thứ tự **phải→trái**; ghi `layout: lithograph`, `qn_syllables_per_column: 14`, `tier_split` = (6, 8) theo luật.
3. **Gióng cặp↔câu bằng DP đơn điệu cả sách** (như `ingest_prose_book` làm cho Chrestomathie, nhưng đơn vị là **cặp**):
   điểm = số chữ kim ∈ R(âm) của cặp câu ứng viên; ràng buộc đơn điệu + bước 1 cặp/cột. Neo mềm sẵn có: vị trí LCS
   theo tờ ở §3.2 (≈ 26 câu/tờ) để giới hạn cửa sổ ±40 câu.
4. **Cổng nghiệm thu trước khi tin bộ ra**: (a) ≥ 80 % cột có 13–15 chữ ở tầng dưới; (b) chuỗi cặp gán được **đơn điệu**
   và phủ ≥ 90 % số tờ; (c) khớp dị bản của tier GOLD **≥ 71,3 %** (nền 1871↔1872) — dưới ngưỡng ấy thì bộ nhãn không
   tốt hơn việc chép thẳng bản 1871.
5. Ngân sách: 120 lượt kim (1/tờ) + ~10 lượt kiểm; ~25 phút build.

---

## 4. ĐỘ ĐÚNG END-TO-END THẬT của A và B (`scripts/measure/ihr_endtoend_eval.py`, selftest **24/24**)

Ghép `labels_gated.csv` ↔ `data/<book>/manifest.tsv` theo **(trang, cột, vị trí chữ)**: `page_XXXX` → `page_id` qua
`prepared_ihr/<book>/manifest.json`; `column` 1..10 → `col_index` = column − 1; `syl_idx` 0..5 → `part` 1, 6..13 →
`part` 2. Phép đo **không vòng tròn**: adapter không đọc `nom_text` (kiểm bằng AST trong selftest), còn `nom_text` là
nhãn người của IHR-NomDB.

### 4.1 Bảng chính

| | A · LucVanTien1916 | B · TruyenKieu1872 |
|---|---|---|
| Chữ GT (manifest.tsv) | 13.991 (trong đó 522 chữ PUA) | 21.410 (trong đó 1.158 chữ PUA) |
| Ô pipeline sinh (tổng / trên trang CÓ nhãn người) | 13.761 / 13.761 | 22.499 / 21.379 |
| **Coverage ô** = ô trên trang có GT / chữ GT | **98,4 %** | **99,9 %** |
| Trang pipeline gán nhãn nhưng `manifest.tsv` KHÔNG có nhãn người (ô không đo được) | 0 trang (0 ô) | **8 trang** (1.120 ô) |
| Ô ở trang CÓ GT mà vẫn lệch vị trí | 18 (0,13 %) | 10 (0,05 %) |
| **GOLD có ảnh: n / ĐÚNG [Wilson 95 %]** | 8.775 / **97,07 %** [96,70–97,40] | 12.736 / **98,49 %** [98,27–98,69] |
| **GOLD kể cả `GOLD_text_only`: n / ĐÚNG** | 12.197 / **97,22 %** [96,91–97,50] | 19.128 / **98,47 %** [98,28–98,63] |
| GOLD coverage (trên số chữ GT) | **87,2 %** (GOLD ảnh riêng: 62,7 %) | **89,3 %** (GOLD ảnh riêng: 59,5 %) |
| GOLD, bỏ dị thể đồng âm | **98,89 %** | **99,86 %** |
| SYLLABLE (n) | 89 — không có nhãn ký tự (đúng thiết kế) | 46 — như A |
| REVIEW: n / ĐÚNG | 1.457 / 71,5 % | 2.181 / 87,8 % |
| **Toàn bộ ô có nhãn: ĐÚNG** | 93,87 % | 97,17 % |
| **kim THÔ so GT (không qua luật nào)** | **95,18 %** [94,81–95,53] (n 13.743) | **97,46 %** [97,24–97,66] (n 21.369) |
| kim thô **trên ô được nhận GOLD** | 97,22 % | 98,47 % |
| Nhãn == chữ kim ở ô GOLD | **100 %** | **100 %** |

**Một nguồn bi quan nhỏ, đã lượng hoá.** Phép ghép giả định cột có đúng 14 ô (6 + 8). Ở A có **26/986 cột**
(2,6 %) số ô ≠ 14 (QN đếm lệch, `qn_count_unfixed`); trong các cột ấy vị trí ô lệch nên precision GOLD chỉ 157/214 =
73,4 %, kéo con số chung xuống. Tính riêng **960 cột đủ 14 ô**: GOLD **97,65 %** (n 11.983); ở B là **98,47 %** (n 19.040, 1.599/1.608 cột). Tức 97,22 % là **cận
dưới**; con số "đúng của phương pháp" nằm giữa 97,2 % và 97,7 %.

### 4.2 Phân loại lỗi của tier GOLD có ảnh

| Loại | A | B |
|---|---|---|
| **dị thể đồng âm** (khác chữ nhưng CẢ HAI ∈ R(âm) — không phải lỗi đọc) | 146 (1,66 %) | 171 (1,34 %) |
| gần hình (`SinoNom_Similar`) | 0 | 0 |
| GT là chữ PUA | 0 | 2 |
| **sai hẳn** | 111 (1,27 %) | 19 (0,15 %) |

### 4.3 So với phép đo gián tiếp 22/09 (200 patch/sách, kim ×3, cắt cột đúng, KHÔNG chạy pipeline)

| | A: cũ → mới | B: cũ → mới |
|---|---|---|
| Precision GOLD | 89,3 % → **97,2 %** (**+7,9 điểm**) | 84,6 % → **98,5 %** (**+13,9 điểm**) |
| Coverage | 55,1 % → **87,2 %** (**+32,1 điểm**) | 49,5 % → **89,3 %** (**+39,8 điểm**) |

**Vì sao chênh nhiều đến thế — ba nguyên nhân, đã tách riêng được:**

1. **Mức ảnh gửi kim** (phần lớn của chênh lệch). Phép đo cũ gửi kim **từng patch CÂU**; phép đo mới gửi **cả trang**.
   Đo lại trên cùng 10 trang/sách: kim cả trang đúng **97,2 / 98,2 %** so với **49,2 / 42,3 %** ở patch (§1.2).
2. **`kim_lang_type = 2`** (Nôm) thay vì 1 (Hán) — khoá vòng 5, cả hai bộ đều dùng 2.
3. **n lớn hơn ~16 lần** (12.197 ô thay vì 768): mốc cũ có CI ±2 điểm, mốc mới ±0,3 điểm.

### 4.4 Ba giới hạn PHẢI đọc kèm (nếu không sẽ suy diễn sai sang 3 sách giao nộp)

1. **QN của A/B là bản PHIÊN ÂM CỦA NGƯỜI** (Nôm Foundation), không phải QN do OCR đọc. Ở LVT1883/KVK1884/Chresto, QN
   đến từ tesseract (âm ngoài từ điển 3,5–6,5 %, cột lệch số âm 7–15 %). Vì vậy **97 % KHÔNG phải là độ đúng dự kiến
   của 3 sách giao nộp**; nó là **cận trên của PHƯƠNG PHÁP khi QN đúng và ô cột đúng**. Đọc đúng cách: khoảng cách giữa
   97,2 % (QN người + cột người) và ~78–90 % khớp dị bản của thạch bản chính là **phần lỗi do khâu QN + khâu dò cột**.
2. **Ô cột do người vẽ.** Bước dò cột (khâu hay hỏng nhất ở STT, `x_tol` 15 px + 5 luật vá) được **bỏ qua** ở A/B.
3. **Luật GOLD ở đây chỉ LỌC, không SỬA**: nhãn == chữ kim ở **100 %** ô GOLD; lọc từ điển nâng kim từ 95,18 % lên
   97,22 % (**+2,0 điểm**) bằng cách loại 11,2 % số ô. Không có bước nào "chữa" chữ kim đọc sai.

---

## 5. Đánh giá rủi ro và khuyến nghị dùng

| # | Rủi ro | Mức | Cách chặn đã làm / cần làm |
|---|---|---|---|
| R1 | **Rò rỉ tập đánh giá vào tập huấn luyện** (A, B có nhãn người; nếu trộn vào train thì mọi con số đánh giá về sau vô nghĩa) | **Cao** | 3 lớp đã làm: (a) `prepared_ihr/<Book>/manifest.json` mang `evaluation_only: true`; (b) `dataset/<Book>/` được **đóng dấu bằng máy** — `pipeline/tools/mark_eval_dataset.py` ghi `TAP_DANH_GIA.md` + `evaluation_only.json` (cờ máy đọc được, hàm `is_marked()` cho cổng CI), `run_pipeline.sh` gọi tự động khi `ingest: ihr`; (c) hai sách nằm ngoài `NEW_BOOKS_ALL`, chạy bằng `--book all-ihr` riêng. **CÒN THIẾU**: cổng ở `pipeline/publish/` từ chối `book` đã đóng dấu khi dựng split huấn luyện |
| R2 | Nhãn máy ghi đè nhãn người trong `data/` | **Cao nếu xảy ra** | Adapter chỉ đọc; mọi đầu ra vào `prepared_ihr/` và `dataset/`; selftest kiểm bằng AST rằng mã không mở `manifest.tsv`; `git status -- data` không có mục nào do vòng này sinh |
| R3 | Dùng 97,2 % như "độ đúng của pipeline" nói chung | **Cao** (dễ mắc) | §4.4 — luôn kèm điều kiện "QN người + ô cột người"; trong báo cáo gọi là *cận trên của phương pháp* |
| R4 | C: nhãn sai hệ thống do QN dị bản | **Cao** | **Đã chặn bằng cách KHÔNG chạy C** (§3.3); khi chạy phải khai `qn_source: di_ban` mọi ô và gắn cổng §3.4 bước 4 |
| R5 | kim là dịch vụ ngoài, kết quả có thể trôi | Trung bình | `kim_raw/` + `kim_cache/` giữ theo md5 ảnh và theo tham số; chạy lại 0 lượt; mọi số §1.2/§4 gắn ngày 23/09 |
| R6 | Ô cột IHR sai ở vài trang | Thấp | Cổng `ceil(số câu/2)` bỏ 5 trang (A) / 2 trang (B); ô không sinh ra 1,6 % (A) / 0,1 % (B) |
| R7b | `pipeline/tools/make_dataset_docs.py` ghi CỨNG tiêu đề "Sách Thánh Truyện" (dòng 226) và bảng tên sách chỉ có `stt2/stt4/stt11` (dòng 31–33) ⇒ `dataset/<Book>/README.md` của MỌI sách mới đều mang tên sai — lỗi này có từ trước, chạm cả 3 sách giao nộp | Trung bình (chỉ là nhãn hiển thị, số liệu đúng) | **Chưa sửa** (vòng 6 không đụng `pipeline/`). `TAP_DANH_GIA.md` do `mark_eval_dataset.py` ghi có tên sách ĐÚNG. Đề nghị vòng sau: tham số hoá tiêu đề theo `books[].name` |
| R7 | 522 chữ PUA trong GT của A | Thấp | Đo riêng: `precision_bo_pua`; ở A chỉ **1** ô sai vì GT là PUA ⇒ PUA không phải nguồn lỗi |

### Khuyến nghị

1. **Dùng A và B làm TẬP ĐÁNH GIÁ chuẩn của luận văn** — đây là bộ duy nhất cho phép nói "precision X % so nhãn người"
   với n ~12.000/sách và CI ±0,3 điểm, thay cho mọi chỉ số thay thế (khớp dị bản, `kim ∈ R(âm)`). Không trộn vào train.
2. **Bỏ mốc "kim chỉ đúng 42–49 % trên mộc bản"** khỏi mọi tài liệu; thay bằng "kim đọc **cả trang** đúng 97–98 % trên
   mộc bản in, nhưng rụng gần nửa số chữ nếu cắt ảnh xuống mức câu" (§1.2).
3. **Dùng khoảng cách 97,2 % ↔ khớp dị bản của thạch bản để phân bổ ngân sách lỗi** — nhưng **chỉ như giả thuyết**:
   suy luận này bắc qua hai miền in (mộc bản ↔ thạch bản). Phép kiểm rẻ nhất để biến nó thành kết luận: trên **1 trang
   LVT1883 và 1 trang KVK1884**, thay dòng QN bằng phiên âm chuẩn và thay ô cột tự dò bằng ô vẽ tay, rồi chạy lại
   `build_dataset --limit`. Nếu precision nhảy lên ~97 % thì nút thắt thật là **QN (tesseract) + dò cột**, và xếp hạng
   "kim = 77–81 % mất mát" của `CHAN_DOAN_3_SACH_MOI §5` phải đọc lại là "mất mát **quan sát được** dưới điều kiện QN
   OCR", không phải "kim đọc sai".
4. **C**: làm theo §3.4 ở một vòng riêng, hoặc để ngoài luận văn và ghi rõ lý do đo được (§3.3).

---

## 6. Hồi quy và nghiệm thu (23/09)

| Cổng | Kết quả |
|---|---|
| Hồi quy STT 3 trang (`build_dataset --config config/pipeline.yaml --qd01-cells none --force --pages pages3.csv`, stt2/0024 · stt4/0050 · stt11/0100) | `labels.csv` md5 **`59e436d7641fa849bb6759868ac29259`**, **556 dòng** — **trùng mốc**. Không sửa dòng nào trong `pipeline/align_engine/`, `pipeline/remediation/`, `core/`, `config/pipeline.yaml` (3 tệp mới trong `pipeline/tools/` là THÊM, không ai gọi từ đường STT); `run_pipeline.sh` chỉ THÊM nhánh `ingest: ihr`, bước đóng dấu tập đánh giá và `--book all-ihr` — cả ba nằm trong nhánh `--book <sách>` |
| selftest | `ingest_ihr` **64/64** (mới) · `mark_eval_dataset` **15/15** (mới) · `ihr_layout` **22/22** (mới) · `ptcl_layout` **15/15** (mới) · `ihr_endtoend` **24/24** (mới) · book_layout 115/115 · ingest_lithograph 82/82 · ingest_prose 33/33 · mechanism_gates 94/94 · pitch_decode 22/22 · verses_ref_fix 19/19 · phase1_engine 253/0 |
| `scripts/measure/code_facts.py` | **18/18** invariants |
| `measure.py --all --report-only` (13 bước, 6 sách) | **177 PASS / 0 FAIL / 2 mềm / 0 SKIP**, mã thoát **0** (trước vòng 6: 118 invariant trên 6 bước) |
| `.gitignore` | thêm `prepared_ihr/` (206 MB ảnh + cache kim, sinh lại được) |
| `git status --short -- dataset_out prepared/SachThanhTruyen*` | **trống** |
| `git status --short -- data` | Không trống nhưng **KHÔNG do vòng này**: cùng danh sách `D data/...` đã ghi ở `KE_HOACH_COMMIT_VONG5 §0` (có từ 20–21/09). Vòng này không có lệnh nào ghi vào `data/` |
| Ngân sách kim | A 94 lượt · B 161 lượt · đo bố cục A/B 40 lượt · đo C 6 lượt = **301/450** |

Tệp mới/đổi (chưa commit): xem `docs/KE_HOACH_COMMIT_VONG6_2026-09-23.md`.
