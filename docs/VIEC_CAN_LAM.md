# DANH SÁCH VIỆC — TỔNG HỢP SAU TOÀN BỘ NGHIÊN CỨU

**Chốt ngày**: 2026-08-22 · **Đây là tài liệu điều hành duy nhất.** Bốn tài liệu nghiên cứu bên
dưới là căn cứ, không phải danh sách việc:

| tài liệu | nội dung |
|---|---|
| `docs/PHAN_HOI_GOP_Y_2026-08-22.md` | phản hồi bản góp ý MI3, đã qua kiểm định 199 khẳng định |
| `docs/KE_HOACH_TONG_THE_2026-08-22.md` | §0 chứng minh vì sao mọi số precision cũ bị huỷ |
| `docs/CHUONG_TRINH_THI_NGHIEM_2026-08-22.md` | chi tiết lưới cấu hình từng chương trình T1–T6 |
| `docs/NGHIEN_CUU_UNIHAN_KDEFINITION_2026-08-22.md` | tín hiệu nghĩa Hán, một chiều |

---

## BỐI CẢNH ĐIỀU HÀNH — ba sự thật chi phối mọi việc

1. **Đề tài có 0 phán quyết người.** Toàn bộ "chấm tay" trước đây là máy chấm; verdict thô đã mất.
   Mọi con số precision / error-AUC / κ đã bị huỷ.
2. **Chấm tay đang tạm gác.** Sẽ rút ô chưa chấm từ bộ dữ liệu, chấm, nạp lại sau.
3. ~~Chưa có gì được commit.~~ ✅ Đã chốt mốc `17289d145c` (2026-08-23): `HEAD` giờ khớp đĩa.

---

## ĐƯỜNG GĂNG

```
KHỐI 0 ✅       →  KHỐI 1 (vá lỗi)  →  KHỐI 3 (bàn thí nghiệm)  →  KHỐI 4 (T1→T6)
                            ↓                                              ↓
                     KHỐI 2 (dọn số liệu)                          KHỐI 5 (tín hiệu nghĩa)
                                                                           ↓
                                                    KHỐI 6 (dựng hạ tầng chấm — làm SONG SONG)
                                                                           ↓
                                                    KHỐI 7 (chấm tay)  →  KHỐI 8 (đóng gói)
```

**KHỐI 6 xây được ngay bây giờ, không phải chờ.** Xây sớm thì lúc chấm không phải dừng lại làm công cụ.

---

# KHỐI 0 — CỨU DỮ LIỆU ✅ HOÀN THÀNH 2026-08-23 · commit `17289d145c`

- [x] **0.1 Sao lưu primary data + sha256** → `~/backup_ocr_cache_2026-08-22/` (**1.783 tệp, 67 MB**)
  Checklist ban đầu ghi `cp -r prepared/*/detected/` — **thiếu một nửa**: cache QN nằm ở
  `transcriptions/`, không phải `detected/`. Bản sao thực tế gồm:
  445 cache Nôm (tốn tiền API) · 445 cache QN · 445 cột đã bóc · **445 `pages/*.png`** (ảnh mà
  `image_hash` neo vào — thiếu nó thì không kiểm chứng được chuỗi) · 3 manifest.
  Kèm `SHA256SUMS.txt` + `README.md` (cách kiểm & khôi phục). Đối chiếu từng tệp với bản gốc:
  **khớp tuyệt đối**; `shasum -a 256 -c` → 0 dòng lỗi.

- [x] **0.2 Commit mốc SHA** — commit `17289d145c`, 36 tệp.
  `.gitignore` đã loại `*.png`/`prepared/`/`dataset/` nên chỉ 36 tệp vào commit, không phải
  145.000 ảnh. Unihan được add theo **đúng casing `Dict/`** mà git đang dùng → index **không**
  sinh thư mục trùng `dict/`. Xác minh: `HEAD:dataset_out/labels_final.csv` = 82.269 dòng
  (GOLD 50.063 / SILVER_uncalibrated 10.890 / SYLLABLE 6.761 / REVIEW 14.555) — **khớp đĩa**.
  Commit vào `main` theo quy ước repo (`push.sh:10`: chỉ dùng nhánh main). **Chưa push.**

- [x] **0.3 Chốt đối chiếu cache OCR** — `core/ocr/ocr_api.py`
  Không phải "5 dòng" như dự tính. Nghiên cứu cho thấy `extract_nom_image` lưu ảnh qua
  `PIL.Image.save(..., "PNG")`, tức **mã hoá lại** → byte tệp phụ thuộc phiên bản Pillow/zlib,
  còn pixel thì không. Chốt md5 thô sẽ **báo động giả** khi đổi môi trường. Thiết kế thực hiện:

  | tình huống | xử lý |
  |---|---|
  | md5 tệp khớp | `ok` (đường nhanh, không mở ảnh) |
  | byte lệch, **pixel y hệt** | `healed` — cập nhật `image_hash`, cache vẫn dùng |
  | byte lệch, **pixel đổi** | **ném `StaleOCRCacheError`** |
  | cache chưa có `pixel_hash` | ném lỗi kèm hướng dẫn chạy backfill |
  | cache đời cũ không có `image_hash` | `skipped` |
  | `SN_OCR_SKIP_CACHE_VERIFY=1` | `skipped` (cửa thoát hiểm) |

  **KHÔNG tự gọi lại API khi lệch** — OCR lại tốn tiền, phải do người quyết định.
  `backfill_pixel_hash()` đã vá **445/445** cache Nôm; verify lại toàn bộ → **445 `ok`**.
  445 cache QN được bỏ qua an toàn (không có khoá `image`).
  Đường QN (`qn_ocr.py:129`) **vốn đã** đối chiếu md5 → không cần sửa.
  **+13 assertion**, mốc selftest **414 → 427**, không hồi quy.

## Phát hiện thêm trong lúc làm KHỐI 0

- [ ] **0.4 `nom-embed` submodule đang bẩn** — `best.pt` và `last.pt` **đã đổi nhưng chưa commit
  trong submodule**. Con trỏ submodule ở repo cha vẫn là `7ff74f5`, nên **checkpoint S3 trên đĩa
  KHÁC với thứ mà lịch sử git ghi lại**. Đây là một lỗ hổng tái lập: không ai dựng lại được đúng
  mô hình đã sinh ra `s3_cosine` trong bộ nhãn. Cần quyết định: commit trong submodule, hay đẩy
  checkpoint lên HuggingFace và ghi hash vào `EVIDENCE_INDEX.md`.

# KHỐI 1 — VÁ 9 LỖI CHẶN (2–3 ngày)

| # | lỗi | vá thế nào |
|---|---|---|
| 1.1 | **48 ô 㝵/"người" đang ở GOLD** trong bản công bố | `s3_unwind.py`: sau readmit, chặn mọi ô thuộc lớp trong `confusion_fixes.yaml` |
| 1.2 | `confusion_fix --measure` trả `null` **trong im lặng** (join `yen*` vs `stt*` khớp 0/825) | chuẩn hoá tiền tố trước khi join (`confusion_fix.py:53-60`) |
| 1.3 | Thiếu checkpoint detector ⇒ **rơi ngầm về midpoint**, không ném lỗi, không ghi vết | `align_production.py:183`: `raise FileNotFoundError` + ghi cột `seg_backend` vào labels |
| 1.4 | `CHECKSUMS.txt` **chưa từng được sinh**; hash hiện hành không có trong `EVIDENCE_INDEX.md` | ép `checkpoint()` ghi thật; `evidence()` **thêm bảng mới**, không chỉ append log |
| 1.5 | `banner()` in ra **"BƯỚC 7/6"**; đánh số lệch ở 6 chỗ | `run_pipeline.sh:83` + các chú thích khối |
| 1.6 | Thư mục từ điển: git lưu `Dict/`, đĩa là `dict/` (7 tệp `.py` + 2 config hard-code) | `git mv` hai bước rồi thống nhất mã |
| 1.7 | `index.csv` (crop-proto) thuộc **thế hệ cũ**: 100% hàng trỏ `yen2/yen4/yen11`, giao với bộ hiện hành = 0 | sinh lại từ `stt*`; preflight thêm kiểm "cùng thế hệ", không chỉ "tệp tồn tại" |
| 1.8 | `config/pipeline_today.yaml:29` còn `qn_line_detector: auto` — bẫy copy-paste | xoá tệp |
| 1.9 | 73 ô sửa dấu thanh **mồ côi** (`labels_tonefix.csv` không vào bản công bố) | đưa chuẩn hoá thanh **lên trước build** (xem T1), không vá sau |

**Hoàn thành khi**: chạy lại từ bước 4 ra **56.776 dòng**; đếm ô 㝵/"người" ở GOLD ra **0**;
`CHECKSUMS.txt` tồn tại; selftest xanh.

---

# KHỐI 2 — DỌN SỐ LIỆU (1 ngày, làm song song KHỐI 1)

- [ ] **2.1** Đánh dấu **CHƯA ĐO** mọi số sau, ở mọi tài liệu: precision GOLD 97,98% / 98,00% /
  97,08% · `rule_precision_direct` 737/752 · `rule_precision_similar` 40/41 · error-AUC 0,566 và
  0,577 · κ = 0,13.
- [ ] **2.2** Xoá bảng số cũ trong `BANG_SO_LIEU_CHINH_THUC.md` (dòng 59-70 — không khớp tệp nào
  trên đĩa); sửa `EVIDENCE_INDEX.md:71` (66.589 → **66.529**).
- [ ] **2.3** Giữ Bước 6 (`s3_unwind`) nhưng **viết lại lý do**: "tín hiệu chưa chứng minh được",
  **không** phải "đã bác bỏ".
- [ ] **2.4** Ghi vào datasheet: mẻ phán quyết trước 2026-08-22 **bị huỷ vì không truy nguyên được
  xuất xứ**. Khai thẳng — đây là điểm cộng liêm chính, không phải điểm trừ.

---

# KHỐI 3 — DỰNG BÀN THÍ NGHIỆM (3–4 ngày)

- [ ] **3.1 `pipeline/lab/runner.py`** — chạy một cấu hình (1 YAML) vào `lab/run_<hash>/`, không
  đụng `dataset_out/`
- [ ] **3.2 `pipeline/lab/metrics.py`** — thư viện thước đo hình học; **nối `crop_quality.py` vào
  đây** (lần đầu tiên module 208 dòng này được dùng)
- [ ] **3.3 `pipeline/lab/perturb.py`** — sinh nhiễu loạn có đáp án trên dữ liệu thật (7 loại × 3 mức)
- [ ] **3.4 `pipeline/lab/synth.py`** — sinh trang tổng hợp từ 89.898 glyph FontDiffusion, nhiễu
  hiệu chuẩn theo `ink_pct` thật
- [ ] **3.5 `lab/results.csv`** — mỗi dòng = 1 cấu hình × mọi thước đo

**Hoàn thành khi**: chạy baseline ghi được 1 dòng đầy đủ, chạy lại **byte-identical**.

---

# KHỐI 4 — SÁU CHƯƠNG TRÌNH THÍ NGHIỆM (4 tuần)

Thứ tự **bắt buộc từ thượng nguồn xuống** — chỉnh crop trên hộp xấu là công cốc.

| # | chương trình | cấu hình | thước đo chính | baseline |
|---|---|---|---|---|
| **T1** | Đường Quốc ngữ (tách dòng × 2-pass × chuẩn hoá thanh × parser) | 54 | tỉ lệ âm **ngoài từ điển** | **0,95%** (778 ô) → mục tiêu ~0,6% |
| **T2** | Hình học trang Nôm (DPI × nhị phân hoá × dò cột) | 48 | tỉ lệ trang đủ 9 cột | **439/445** → mục tiêu 445/445 |
| **T3** | **Căn chỉnh** — chuẩn nhiễu loạn + quét ma trận chi phí | 144 | `anchor_retention` dưới nhiễu | **chưa đo bao giờ** |
| **T4** | Tách ký tự & chất lượng crop | 144 | 6 thước đo hình học | **chưa đo bao giờ** |
| **T5** | Độ giòn từ điển + `syllable_gate` + top-K cầu tự dạng | ~30 | độ giòn, nhất quán liên sách | — |
| **T6** | Tái lập & bằng chứng | — | byte-identical ×2, clone sạch | — |

### Ba cảnh báo bắt buộc

1. **T3 — không tối ưu theo "số ô GOLD" hay tỉ lệ dict-confirm.** Đó chính là luật gán nhãn; cấu
   hình nới lỏng nhất sẽ luôn thắng trong khi sinh nhiều nhãn sai nhất. Dùng **biên Pareto**
   `anchor_retention` × sản lượng.
2. **T3 — chuẩn đo neo vào Ô NEO** (`s1_inter_s2_direct`, trung vị 58%/cột), **không** phải "cột
   sạch tuyệt đối" — toàn corpus chỉ có **6 cột** đạt 100% dict-confirmed.
3. **T2 — không dùng công thức `pad_px = pad_frac × pitch × (300/DPI)`.** `pitch` đo bằng pixel của
   chính trang đó nên `pad_frac × pitch` **vốn đã tự chuẩn hoá**. Cách đúng: **resample crop về kích
   thước chuẩn**, hoặc chuẩn hoá theo `pitch`.

### Sản phẩm đi kèm

- [ ] **T3.4** Thêm **≥ 10 assertion** cho `anchor_align` (hiện **0** test phủ ma trận chi phí)
- [ ] **T4.1** Bảng chất lượng hình học của **toàn bộ 66.589 crop** — chưa ai đo bao giờ; tự nó là
  một mục trong luận văn, và thay được chiều CROP vốn phải nhờ mắt người
- [ ] **T5** Cột **`fragility`** trong `labels.csv` = số mục từ điển hậu thuẫn cho nhãn này. Ô
  `fragility = 1` là ứng viên chấm tay số một → **làm mẻ chấm sau này rẻ đi nhiều lần**

---

# KHỐI 5 — TÍN HIỆU NGHĨA HÁN (2 ngày)

Công cụ đã có: `pipeline/tools/sem_score.py` (chạy được, `--bench` tái lập số).

- [ ] **5.1** Ghi 2 cột `sem_score` + `sem_confirmed` vào `labels.csv`.
  **CHỈ PHONG, KHÔNG BAO GIỜ HẠ** — 47,7% cặp đúng cũng cho điểm 0; 㝵/"người" (sai) và 主/"chúa"
  (đúng) **cùng điểm 0,0**. Ngưỡng đã hiệu chuẩn: > 0,05 chính xác 100%; > 0,02 chính xác 98,9%.
  Sản lượng: **451 ô** xác nhận trên luật GOLD yếu nhất `s1_inter_s2_similar`.
- [ ] **5.2** Xuất bảng **"mượn nghĩa vs mượn âm"** cho toàn corpus — đóng góp ngôn ngữ học, chưa
  từ điển nào của ngữ liệu Công giáo cổ này có.
- [ ] **5.3** Tải CHISE `ids.txt`, kiểm giả thuyết **mô hình lỗi OCR**: OCR quy chữ Nôm lạ về chữ
  Hán **chia sẻ thành phần** (韋→喡 rụng bộ 口 · 等→寺 thêm bộ 竹 · 忝→𡗶 cùng 天 · 洋→群 cùng 羊).
  *Đây mới là chỗ IDS có ích thật — để mô hình hoá lỗi, KHÔNG phải để huấn luyện thị giác.*

---

# KHỐI 6 — DỰNG HẠ TẦNG CHẤM TAY (làm SONG SONG với KHỐI 4, chưa chấm)

- [ ] **6.1 `audit_grid` bổ sung 4 cơ chế kiểm soát chất lượng**:
  - **ô mồi 10%** — nửa là ô chắc chắn đúng, nửa **cố ý gán sai**; nếu độ chính xác trên ô mồi
    < 90% thì **huỷ cả buổi chấm**. *Đây là cơ chế trực tiếp phát hiện chấm ẩu hoặc chấm bằng máy —
    đúng vấn đề đã xảy ra.*
  - **ô lặp ẩn 8%** — đo κ nội tại thật
  - **làm mù**: không hiện tier / rule / `s3_cosine`
  - ghi `session_id` + `dwell_ms`; ô nào `dwell_ms < 800ms` gắn cờ xem lại
- [ ] **6.2 Gắn nghĩa Hán vào HTML chấm** (dùng `sem_score.load_meanings`) — 2 giờ, **giá trị cao
  nhất**: người chấm thấy ngay *"㝵 = (dạng cổ của 得) to get / to obstruct"* khi đang chấm ô âm
  "người".
- [ ] **6.3 Tách đôi nhiệm vụ**: chiều **NHÃN** cho người; chiều **CROP** đo bằng máy
  (`crop_quality.py` từ T4). Trộn hai chiều vào một nút bấm là lỗi thiết kế của mẻ cũ.
- [ ] **6.4 Rút mẻ, niêm phong TEST**:
  - **DEV 650** (GOLD 300 · SILVER 200 · SYLLABLE 150) — chỉnh tham số thoải mái
  - **TEST 1.050** (GOLD 600 · SILVER 250 · SYLLABLE 200) — **chỉ mở MỘT LẦN** ở KHỐI 8
  - loại mọi trang từng dùng để phát hiện lớp lỗi 㝵 khỏi TEST (chống post-hoc)
  - ưu tiên ô `fragility = 1` và ô `sem_score` thấp
- [ ] **6.5 Chấm thử 50 ô** để kiểm hướng dẫn chấm **trước khi** cam kết 2.000 ô.

---

# KHỐI 7 — CHẤM TAY & ĐO LẠI (sau KHỐI 4–6)

- [ ] **7.1** Chấm DEV (~650 ô + mồi + lặp), 4–6 buổi ≤ 45 phút/buổi.
  *Chia nhỏ vì mẻ cũ trôi 4,2% → 16% → 35% qua ba buổi dài.*
  **Cổng**: ô mồi ≥ 95%, κ chiều NHÃN ≥ 0,6. Nếu κ < 0,4 → **dừng, sửa hướng dẫn, làm lại**.
- [ ] **7.2** Đo lại toàn bộ: precision từng **tier**, từng **rule**, từng **sách** (bắt buộc vì
  STT4 lệch DPI), kèm Clopper-Pearson + Horvitz-Thompson.
- [ ] **7.3** Khai thác lại lớp nhiễu hệ thống, **có hiệu chỉnh Benjamini-Hochberg** (mẻ cũ không
  hiệu chỉnh đa so sánh). Xác nhận hoặc **hoàn 1.924 ô 㝵 về GOLD**.
- [ ] **7.4** Đo S3 **trên SILVER**, không trên GOLD — tỉ lệ lỗi SILVER 14–28% nên n = 300 cho
  50–80 ca lỗi; GOLD chỉ 2–3% nên cần 2.000–6.000 ô. **Rẻ hơn 10 lần cho cùng kết luận.**
  Bắt buộc holdout page-disjoint sạch (bộ cũ rò rỉ 74,7% vào train).
- [ ] **7.5** Chấm TEST (~1.050 ô), **mở một lần duy nhất**, không chỉnh gì sau khi mở.

---

# KHỐI 8 — ĐÓNG GÓI & VIẾT

- [ ] **8.1** Chốt chặn **Pre-Export**: Bước 7 **từ chối chạy** nếu SHA256 của `labels_final.csv`
  khác hash ghi trong **manifest của mẻ audit**. *(Không phải so hash Bước 6 với Bước 7 — đó chỉ là
  toàn vẹn trong một lần chạy, không phải "bộ đem đo = bộ đem nộp".)*
- [ ] **8.2** Chạy Bước 7 + `pipeline/publish/` (Frictionless + Croissant + datasheet Gebru + HF
  Parquet, CI gate 11/11 — đã viết xong, chỉ chạy ngoài `run_pipeline.sh`).
- [ ] **8.3** Datasheet khai đủ **5 giới hạn**: 3 sách 1 nét chữ · **STT4 ~201 DPI vs 302 DPI** kèm
  precision tách theo sách · SYLLABLE là nhãn cấp âm tiết, 316 cặp không nguồn nào xác nhận ·
  SILVER công bố riêng gắn nhãn "uncalibrated" · κ đo được + lịch sử mẻ phán quyết bị huỷ.
- [ ] **8.4** Viết luận văn theo ba đóng góp ở dưới.

---

# CÁI GÌ DỨT KHOÁT **KHÔNG** LÀM

| hạng mục | lý do (đã đo) |
|---|---|
| **SVTR** | Là mô hình nhận dạng **chuỗi text-line**; tác vụ là phân lớp **một ký tự đã cắt sẵn**. Bộ xếp hạng thật hiện đã là **đầu ArcFace 1.591 lớp**, không phải backbone |
| **IDS cho mô hình thị giác** | Tiền đề sai: **1.582/1.582 lớp đều có codepoint**; ví dụ ⿰口巴 xuất hiện **0 lần**. (IDS **có** ích cho mô hình hoá lỗi OCR — mục 5.3) |
| **Mở từ điển lên 120.000 cặp** | `kDefinition` sinh **0 cặp**; `kVietnamese` chỉ +109 ròng. Đổ `dict_gap` vào từ điển đẩy **20.144 ô** lên GOLD (+40,2%) mà không ô nào được xác nhận |
| **Sửa dấu thanh nhóm B** | Chạm **0 ô** bộ nhãn công bố; áp dụng sẽ **phá 170 cặp đang đúng** |
| **Hợp phiếu có VLM** | Vi phạm ràng buộc gốc: phán đoán của AI không bao giờ trở thành nhãn |
| **Đặt KPI Error-AUC "0,850" hay "0,720–0,750"** | Cả hai đều là số tự chế, không dựa trên công trình nào |
| **Dùng `sem_score` để hạ cấp** | 㝵/"người" (sai) và 主/"chúa" (đúng) cùng điểm 0,0 |

---

# BỐN RÀNG BUỘC BẤT DI BẤT DỊCH

1. **Phán đoán của AI không bao giờ trở thành nhãn** — chỉ để xếp hạng việc cho người.
2. **KPI phải là đại lượng đo được kèm khoảng tin cậy**, không phải số lượng.
3. **Mọi con số phải có lệnh tái sinh**, và phải nói rõ đo trên commit nào.
4. **Nếu một thí nghiệm đã chạy, phải nêu kết quả cũ trước khi đề xuất chạy lại.**

---

# LUẬN ĐỀ CUỐI — ba đóng góp

1. **Phương pháp gán nhãn cấp ký tự bằng cầu song ngữ**: NW có băng + ma trận chi phí từ điển +
   tier hoá theo giao tín hiệu, và đặc biệt **quy trình khai thác lớp nhiễu hệ thống bằng kiểm
   định Fisher có hiệu chỉnh FDR** — chuyển giao được sang bất kỳ corpus nào gán nhãn bằng cầu từ
   điển.
2. **Kết quả âm có đo đạc về tín hiệu thị giác**, kèm phân tích lực thống kê và phát hiện rò rỉ
   74,7% — cộng đồng vẫn mặc định so khớp glyph hoạt động trên chữ Nôm.
3. **Bộ dữ liệu ~57.000 ô có precision đo được kèm CI**, chuẩn quốc tế, chuỗi bằng chứng SHA256
   khép kín, và **datasheet khai cả những gì đã hỏng**.

Điều thứ ba phân biệt luận văn tốt với luận văn trung bình trong lĩnh vực này: phần lớn bộ dữ liệu
di sản được công bố kèm những con số chất lượng không ai tái lập được. Đề tài này đã tự phát hiện
mình ở trong tình trạng đó và sửa — **viết thẳng ra thì đó là chương hay nhất của luận văn**.
