# SỔ ĐĂNG KÝ BẰNG CHỨNG (EVIDENCE INDEX)

**Lập**: 2026-07-20 (Giai đoạn 0 — Cứu bằng chứng)
**Mục đích**: mỗi artifact được dùng làm bằng chứng trong luận văn phải truy nguyên được về (a) commit chứa nó, (b) sha256 tại thời điểm đóng băng, (c) lệnh tái sinh nó.

> **Quy tắc**: không chương nào của luận văn được trích số từ artifact không có trong bảng này.
>
> **Số liệu chốt → `docs/BANG_SO_LIEU_CHINH_THUC.md`** (Giai đoạn 3, tag `dataset-frozen-v1`). Mọi con số của luận văn trích từ đó. Sổ này (EVIDENCE_INDEX) quản lý *provenance + sao lưu*, BANG_SO_LIEU quản lý *giá trị + nguồn kiểm định*.

---

## 1. TUYÊN BỐ TRUNG THỰC VỀ TÌNH TRẠNG BẰNG CHỨNG

Ba điều phải khai báo, không được giấu:

| Vấn đề | Trạng thái |
|---|---|
| **`verdicts_001–006.jsonl` gốc (mẻ audit GOLD n=846) ĐÃ MẤT** | Chỉ còn bản dẫn xuất `verdicts_reanchored.csv` (825/846 neo lại được, 97,5%). Không thể tái kiểm từ verdict thô. |
| **`FLOW_TONG_THE_CHOT_2026-07-14.md` là BẢN PHỤC DỰNG** | Bản gốc bị xoá nhầm 2026-07-20 khi chưa từng được commit. Phục dựng 101/101 dòng từ transcript agent đã đọc trọn file; đối chiếu khớp mọi trích dẫn rời rạc trong 4 workflow journal. Header của file ghi rõ điều này. |
| **Verdict SILVER là AI chấm, KHÔNG phải người** | `audit_SILVER/verdicts_ai.jsonl` = 750 dòng, toàn bộ `source='ai_vision'`. Con số precision SILVER 72,98% **không được trình bày như human audit**. SYLLABLE hiện có **0 verdict**. |

---

## 2. COMMIT MỐC

| Commit | Nội dung | Ghi chú |
|---|---|---|
| `388694a456` | feat: audit-remediation pipeline (Giai đoạn 0–3) + engine fixes | Nền của 4 giai đoạn |
| `f78dbc4da5` | **Chứa toàn bộ artifact bằng chứng** (`dataset_out/human_audit/**` 54 file, `labels_final.csv`, `config/confusion_fixes.yaml`, `dataset_out/fusion/**`) | Message gốc là "update code" — vô nghĩa. Đây là commit phải trích khi nói về bằng chứng audit. |
| `f47f431abd` | chore(cleanup): xoá mã chết + vá bug ghi đè `fused.csv` + chốt mốc selftest | Nhóm A kiểm kê 2026-07-20 |
| tag `freeze-pre-thesis-2026-07-20` | Điểm đóng băng Giai đoạn 0 | Mọi bằng chứng tính đến ngày này |

**Trạng thái remote** (cập nhật 2026-07-21): `main` = `feat/phases-0-3-audit-pipeline` = remote, đều đã push lên `github.com:truong571/GanNhanOCR.git`. Remote có đủ **54 file** `dataset_out/human_audit/` + `labels_final.csv` và **3 tag**.

### 2-bis. Nội dung commit `f78dbc4da5` theo 4 nhóm

Commit này gộp **76 file** vào một lần với message `"update code"`. Không tách lại được (đã push, hash đã trích dẫn, `CODE_FREEZE.md` cấm viết lại SHA), nên liệt kê ở đây để **log vẫn kể được câu chuyện**:

| Nhóm | Nội dung |
|---|---|
| (a) 7 file `.py` | `consensus_fusion/{fuse_stage,mine_confusions,score_s3}.py` · `ground_truth/{batch_json,make_audit_batch,reanchor_verdicts}.py` · `remediation/confusion_fix.py` |
| (b) Cấu hình quyết định nhãn | `config/confusion_fixes.yaml` |
| (c) Bằng chứng | `dataset_out/human_audit/**` (54 file) · `labels_final.csv` · `confusion_fix_report.json` · `dataset_out/fusion/**` |
| (d) Tài liệu | `DE_XUAT_HOAN_THIEN_LUAN_VAN_2026-07-20.md` · `KIEM_KE_FILE_VA_LO_TRINH_2026-07-20.html` |

⚠️ Commit này **đồng thời gỡ `Data/SachThanhTruyen{2,4,11}.pdf` khỏi tracking** (Bin → 0 bytes) — một hành vi khác loại bị trộn chung vào commit "bằng chứng". Ba file PDF vẫn còn trên đĩa dưới tên mới `Data/STT{2,4,11}.pdf` và nằm trong gói `repro_assets_2026-07-20.tgz`.

### 2-ter. Tag

| Tag | Trỏ tới | Ý nghĩa |
|---|---|---|
| `freeze-pre-thesis-2026-07-20` | `9da646c5c4` | Mốc **GĐ0** — cứu bằng chứng |
| `freeze-features-2026-07-20` | `77bb46d31d` | Mốc **code-freeze tính năng** thật sự (`CODE_FREEZE.md` tuyên bố tại đây) |
| `state-post-phase2-2026-07-21` | sau GĐ2 + vá kiểm toán | Trạng thái đã kiểm toán độc lập |

---

## 3. ARTIFACT VÀ SHA256 (tại thời điểm đóng băng 2026-07-20)

### 3.1. Chuỗi nhãn — xếp lớp bất biến

`labels.csv` → *(remediation)* → `labels_remediated.csv` → *(confusion-fix)* → `labels_final.csv`

| Artifact | sha256 | Lệnh tái sinh |
|---|---|---|
| `dataset_out/labels.csv` | `189b61d8801db1d3…c01e1506` | `python -m pipeline.align_engine.build_dataset --config config/pipeline.yaml --use-s3 --reseg detector` |
| `dataset_out/labels_remediated.csv` | `715b98d0ccdb54b0…96814f84` | `python -m pipeline.remediation --labels dataset_out/labels.csv --out dataset_out apply --tau 0.62` |
| `dataset_out/labels_final.csv` | `62f9791bff858a79…9e70a2615` | `python -m pipeline.remediation.confusion_fix --in dataset_out/labels_remediated.csv --out dataset_out/labels_final.csv --fixes config/confusion_fixes.yaml --measure` |
| `dataset_out/summary.json` | `b5b05761b9899088…6950fa5ab` | sinh kèm `build_dataset` |

~~**Số liệu chốt** (đo 2026-08-11, 82.274 dòng): GOLD 48.893 · SILVER 10.887 · SYLLABLE 6.809 · REVIEW 15.685. Dataset có-nhãn = 66.589.~~

🔴 **SỐ TRÊN ĐÃ HUỶ (2026-08-23)** — không khớp tệp nào trên đĩa. Câu "66.589 khớp đúng `dataset/labels.csv`" **sai**: đĩa khi đó là 66.529, và sau KHỐI 1 là **56.776**.
Số hiện hành (commit `2eb51e66f6`): `labels_final.csv` **82.269** dòng — GOLD **50.015** · SILVER_uncalibrated **10.890** · SYLLABLE **6.761** · REVIEW **14.603**; bộ giao nộp **56.776**. Nguồn duy nhất: `docs/BANG_SO_LIEU_CHINH_THUC.md`.

### 3.2. Báo cáo xử lý

| Artifact | sha256 | Sinh bởi |
|---|---|---|
| `dataset_out/remediation_report.json` | `0490277e9787c3ff…0a7f1b71` | `pipeline.remediation apply` |
| `dataset_out/confusion_fix_report.json` | `28189cc49334592a…de167d961` | `pipeline.remediation.confusion_fix --measure` |

`confusion_fix_report`: 1 fix (người→㝵), demote **1.926 crop** → REVIEW; precision GOLD **0,9708 → 0,9800** — tính trên mẫu neo 846 verdict, còn là số **post-hoc**.

### 3.3. Bằng chứng audit (ground truth)

| Artifact | sha256 / số lượng | Ghi chú |
|---|---|---|
| `dataset_out/human_audit/report.json` | `63188b4f80d4386d…da3d8cad8` | ⚠️ precision 0,7298 ở đây là mẫu **suspicion-ranked + stratified SILVER (AI chấm)** — KHÁC khung lấy mẫu với 97–98% GOLD (SRS). Không được trộn hai số. |
| `dataset_out/human_audit/verdicts_reanchored.csv` | `df5a6b1568be2daf…08c51882c` | 825/846 verdict người neo lại được (median IoU 1.0, 93% byte-identical) |
| `audit_gold/` | 8 file JSON + 6 HTML | Grid chấm mù mẻ GOLD |
| `audit_SILVER/verdicts_ai.jsonl` | **750 dòng, 100% `source='ai_vision'`** | ⚠️ AI chấm — xem §1 |
| `audit_SYLLABLE/` | **0 verdict** | Grid đã dựng, chưa ai chấm |

### 3.4. Cấu hình quyết định nhãn

| Artifact | sha256 | Ghi chú |
|---|---|---|
| `config/confusion_fixes.yaml` | `223999dcb49b3363…ae530115` | Đầu vào Stage 6. Quy tắc đã chốt: **DEMOTE chứ không remap** (remap 㝵→𠊛 bị bác: 𠊛=0 trên trang, hỏng ~713 nhãn đúng) |

### 3.5. Tài liệu chiến lược

| Artifact | sha256 |
|---|---|
| `FLOW_TONG_THE_CHOT_2026-07-14.md` | `ad7bdec74bebf0a0…3eebe417` (bản phục dựng — xem §1) |
| `DE_XUAT_HOAN_THIEN_LUAN_VAN_2026-07-20.md` | trong commit docs |
| `THU_TU_THUC_HIEN_TONG_THE.md` | trong commit docs |
| `KIEM_KE_FILE_VA_LO_TRINH_2026-07-20.html` | kiểm kê 318 file, 33 kết luận "nên xoá" bị phản biện bác |

---

## 4. SAO LƯU LẠNH (ngoài repo)

> 🔴 **CẢNH BÁO 2026-08-24 (T6) — MỤC NÀY ĐÃ CHẾT.** <!-- THS_ARCHIVE_CHET -->
> `~/ThS_archive/` **KHÔNG CÒN TỒN TẠI trên máy** (đo bằng `ls -d ~/ThS_archive`), nên mọi
> lệnh khôi phục ở mục này KHÔNG chạy được. Bản sao lưu CÒN HIỆU LỰC:
> `~/backup_ocr_cache_2026-08-22/` (1.783 tệp cache OCR, verify 1783/1783) và
> `~/backup_models_2026-08-24/` (detector, verify OK). Giữ nguyên văn mục dưới đây để
> đối chiếu lịch sử, KHÔNG làm theo.

Vị trí: `~/ThS_archive/backup_2026-07-20/`

| Gói | Dung lượng | sha256 | Nội dung |
|---|---|---|---|
| `evidence_2026-07-20.tgz` | 96 MB | `ff29cf3d370d83c2…27de4674` | 83 file: toàn bộ `ground_truth/`, `fusion/`, 3 bản labels, các report, `release/`, config, 4 tài liệu chiến lược |
| `repo_2026-07-20.bundle` | 1,7 GB | `842a4ed681d32a69…41d38df2` | **Toàn bộ lịch sử git** (101 commit, mọi nhánh + tag) |
| `repro_assets_2026-07-20.tgz` | 321 MB | xem `SHA256SUMS.txt` | **Tài sản KHÔNG có trong git**: `Data/*.pdf` (3 bản scan gốc), `prepared/` (cache OCR), `dataset_out/{gold,silver,syllable}` (72.873 crop) |
| `SHA256SUMS.txt` | — | — | Bảng hash để verify cả 3 gói |

> **Vì sao gói thứ 3 tồn tại** — phát hiện khi chạy kiểm chứng clone sạch ở §4.1: ba tài sản này **không có trong git** (gitignored vì dung lượng) và ban đầu **cũng không có trong sao lưu**. Nếu mất ổ đĩa thì: bản scan gốc `Data/*.pdf` **không tái tạo được**; `prepared/` là cache OCR — theo FLOW đây chính là *primary data* vì OCR Nôm phụ thuộc API Kimhannom bên ngoài (tái tạo tốn tiền và **không tất định**); 72.873 crop là chính bản thân dataset.

**Đã kiểm chứng thật, không tin tưởng mù quáng:**
- `shasum -c SHA256SUMS.txt` → OK cả 2 gói
- Bung tarball ra thư mục tạm → 83 file; sha256 của `labels_final.csv`, `verdicts_reanchored.csv`, `confusion_fixes.yaml`, `FLOW…md` **khớp bản gốc**; 13/13 file audit HTML có mặt
- `git bundle verify` → *"The bundle records a complete history"*
- Clone thử từ bundle → khôi phục 101 commit, HEAD `f78dbc4da5`, có đủ 54 file `dataset_out/human_audit`

**Khôi phục khi cần:**
```bash
git clone ~/ThS_archive/backup_2026-07-20/repo_2026-07-20.bundle <đích>
tar xzf ~/ThS_archive/backup_2026-07-20/evidence_2026-07-20.tgz     -C <đích>
tar xzf ~/ThS_archive/backup_2026-07-20/repro_assets_2026-07-20.tgz -C <đích>   # bắt buộc
```

### 4.1. KIỂM CHỨNG CLONE SẠCH — clone làm được gì, KHÔNG làm được gì

Đã chạy thật ngày 2026-07-20 (clone sạch nhánh `feat/phases-0-3-audit-pipeline` ra thư mục tạm).

**Clone sạch LÀM ĐƯỢC:**
- `git submodule init` đăng ký **đủ 4/4** submodule, exit 0 (trước khi vá `.gitmodules` thì FATAL)
- Có đủ bằng chứng: 54 file `dataset_out/human_audit/`, `labels_final.csv` (82.275 dòng), `confusion_fixes.yaml`, 3 tài liệu chiến lược, `requirements.lock.txt`
- Chạy được 3/5 bộ selftest với **kết quả y hệt** repo gốc: `consensus_fusion` 44/0 · `remediation` 27/6 · `phase1_engine` 29/1

**Clone sạch KHÔNG làm được** (khác biệt đo được so với repo gốc):

| Selftest | Repo gốc | Clone sạch | Nguyên nhân |
|---|---|---|---|
| `publish` | 56/0 | **55/1** | `FAIL real: HF export/round-trip — FileNotFoundError: dataset_out/gold/yen4_page_0174_c09_198.png` |
| `ground_truth` | 56/4 | **không hoàn tất** | `FAIL grid produced items {'items': 0, 'skipped_no_crop': 6}` và `FAIL html embeds crops (data-uri)` |

**Kết luận cho chương Tái lập của luận văn**: mã và bằng chứng **tái lập được từ git**, nhưng **ảnh crop và cache OCR thì không** — chúng bị gitignore vì dung lượng (285 MB + 232 MB). Muốn tái lập trọn vẹn phải: (a) bung `repro_assets_2026-07-20.tgz`, hoặc (b) chạy lại pipeline từ `Data/*.pdf` — mà (b) cần gọi lại API Kimhannom nên **không tất định**. Đây là lý do cache OCR được coi là *primary data*, không phải sản phẩm trung gian.

---

## 4.2. DỮ LIỆU NGOÀI ĐÃ CHUYỂN KHỎI REPO (Giai đoạn 2, 2026-07-20)

Vị trí: `~/ThS_archive/external_data/` — có `MANIFEST.md` ghi nguồn gốc và cách khôi phục cho từng mục.

| Mục | Cỡ | Nguồn | Vì sao chuyển đi |
|---|---|---|---|
| `MTH_TKHMTH2200` | 4,7 GB | HCIILAB (SCUT), `github.com/HCIILAB/TKH_MTH_Datasets_Release` | Dữ liệu pretrain detector (~1,08M box). Đã kết tinh vào checkpoint đang dùng → chỉ cần khi **pretrain lại**. Còn tham chiếu ở `train_crop/build_mth_pretrain.py` nhưng script có cờ `--mth-root` để trỏ lại. |
| `kkanji2` | 549 MB | Kuzushiji-Kanji (CODH), 3.832 lớp | **0 tham chiếu** trong toàn repo (đã grep `*.py *.sh *.ipynb *.yaml`). Tải về cân nhắc pretrain nhưng cuối cùng dùng MTH/TKH vì cùng miền hơn. |
| `font_diffusion_ckpt_failed` | 1,3 GB | Các lần train FontDiffuser **không thành công** (FST mất step cuối 9k/15k) | Checkpoint đang dùng là `font_diffusion/ckpt/PROD/` — **vẫn giữ trong repo**. |

**Giữ lại trong repo có chủ ý**: `MTH/MTHv2_Datasets_Release/` (2 MB — readme + train/test split, cần trích dẫn trong luận văn) và `font_diffusion/ckpt/PROD/` (383 MB — sinh ra kho glyph `gannhanocr-fd`).

**Không ảnh hưởng tín hiệu S3**: kho glyph đã sinh (`gannhanocr-fd`, 89.898 file) là submodule HuggingFace, không nằm trong số đã chuyển đi.

Khôi phục khi cần pretrain lại detector:
```bash
python train_crop/build_mth_pretrain.py --mth-root ~/ThS_archive/external_data/MTH_TKHMTH2200
```

**Kết quả**: repo 15 GB → **8,9 GB** (−6,1 GB). Cổng nghiệm thu đã qua: selftest vẫn **212/11 khớp mốc**, toàn bộ import production sạch, `build_mth_pretrain.py` vẫn import được.

---

## 5. TRẠNG THÁI KIỂM ĐỊNH (selftest)

**Mốc 2026-07-20: 212 passed, 11 failed** — chạy bằng `bash scripts/run_all_selftests.sh`.

Con số **"223 assertions"** từng ghi trong tài liệu là mốc CŨ, **không còn đúng**.

11 assertion đỏ **không phải bug code**. Chúng hard-code census của thế hệ `labels.csv` cũ:

| Assertion mong đợi | Thực tế trên đĩa |
|---|---|
| `dup_bbox == 701` | 0 |
| `cross_col == 1686` | 8 |
| `union == 2321` | 8 |
| `provably-wrong ~ 1177` | 4 |
| `similar_bridge == 3856` | 3850 |

Nghĩa là **các số 701 / 1.686 / 2.321 / 1.177 đang in trong README và luận văn không tái lập được** từ `labels.csv` hiện tại (đã dedup ở lần re-run trước). Đây chính là blocker "số liệu bất nhất" — phải chốt dùng số lịch sử hay số hiện tại rồi sửa assertion + tài liệu cho khớp (xem `THU_TU_THUC_HIEN_TONG_THE.md` GĐ3).

---

## 6. VIỆC CÒN THIẾU

- [x] **Push nhánh + tag lên remote** — xong 2026-07-20, `main` = nhánh = remote, 3 tag đã lên GitHub.
- [x] **BẢN SAO NGOÀI Ổ ĐĨA — ĐÃ LÀM 2026-07-21.** `backup_2026-07-21/` (`repo_2026-07-21.bundle` 1,73 GB + `models_2026-07-21.tgz` 445 MB + `SHA256SUMS.txt`) đã upload lên **Google Drive** của tác giả. Đây là bản sao đầu tiên **khác ổ vật lý** — trước đó repo, `backup_*` và `external_data` đều chung `/dev/disk3s5`. Kích thước cần khớp khi kiểm trên Drive: bundle `1.860.817.760` bytes, models `466.327.314` bytes, SHA256SUMS `177` bytes.
  - ⏳ Còn nên làm khi rảnh (không chặn): (1) upload thêm `backup_2026-07-20/evidence_2026-07-20.tgz` (92 MB) + `repro_assets_2026-07-20.tgz` (321 MB) để có cả 3 bản scan gốc + 72.873 crop ngoài ổ; (2) cân nhắc Zenodo cho `evidence_*.tgz` để có **DOI trích được vào luận văn**.
  - Khi tải bundle về máy khác để dùng: `shasum -a 256 -c SHA256SUMS.txt` tại đích trước khi tin.

---

## 7. KIỂM TOÁN ĐỘC LẬP GĐ0–GĐ2 (2026-07-21)

3 kiểm toán viên độc lập soi lại toàn bộ tuyên bố "đã hoàn tất", kết luận `ready_for_phase3: false` với 4 việc chặn. Đã xử lý:

| Phát hiện | Xử lý |
|---|---|
| `requirements.lock.txt:99` để `vietocr==0.3.13` chưa comment → `pip install -r` **chắc chắn hỏng** (metadata pin pillow 10.2.0, xung đột pillow 12.2.0 trong chính lock, không có wheel py3.14) | Đã comment + ghi cách cài `--no-deps`. **Kiểm chứng thật**: venv py3.14 trống + `pip install --dry-run -r requirements.lock.txt` → resolve sạch 90 gói, 0 xung đột |
| `config/pipeline.yaml:54` để `qn_line_detector: auto` → máy có paddleocr dùng backend **khác** máy sinh số liệu, âm thầm | Ghim `projection_deskew` (đúng backend đã chạy; giữ nguyên hành vi hiện tại) |
| `ground_truth/selftest.py:199` — `manifest[0]` khi manifest rỗng → **cả suite crash**, không in RESULT, 56 assertion biến mất khỏi báo cáo | Thêm guard fail-có-số. Đo trong clone sạch: trước = crash, sau = `52 passed, 8 failed` |
| Bundle sao lưu thiếu 7 commit và **không có tag nào** (dù §4 khai "mọi nhánh + tag") | Bundle mới tại HEAD, `git bundle list-heads` xác nhận có `refs/tags/*`; clone thử khôi phục 110 commit + tag |
| 4 tài sản **ngoài git VÀ ngoài mọi backup**: `detector_r34.best.pt` (82 MB — sinh ra reseg → sinh ra toàn bộ crop), `font_diffusion/ckpt/PROD` (383 MB — sinh ra kho glyph của SILVER), `dict/*.xlsx`, `docs/refs/` | Gói thứ 5 `models_2026-07-21.tgz` (449 MB), đã verify + bung thử |
| `kkanji2` — mục kế hoạch nêu đích danh — có **0 dòng** sha256 | Đã băm đủ **153.236 file**. Bảng hash phần có ý nghĩa (MTH + ckpt, 12.812 hash) đưa vào `docs/data_manifest/`; phần `kkanji2` (140k file, dữ liệu **không dùng**) giữ ở kho lưu |
| `build_mth_pretrain.py` trỏ `MTH/TKHMTH2200` đã bị dời → chạy hỏng runtime với lỗi khó hiểu | Thêm nấc dự phòng trỏ kho lưu; đã đo: mặc định giờ resolve đúng và **tồn tại** |
| Submodule `gannhanocr-fd` dirty vĩnh viễn làm mọi cổng "cây sạch" mất tác dụng cảnh báo | `submodule.gannhanocr-fd.ignore = dirty` (thay đổi gitlink SHA vẫn được báo) |
| Tag `freeze-pre-thesis-2026-07-20` đứng **trước** mốc code-freeze 3 commit | Thêm `freeze-features-2026-07-20` và `state-post-phase2-2026-07-21`; **không di dời** tag cũ vì đã trích dẫn |

## Lần chạy 2026-07-21T14:21:02Z

phạm vi: publish | strict=0 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `66eba9a81b4f902e8ebe1e6f0eeb30273fb4f9a52a248722ccc4228bb2ab0eab` |
| `dataset_out/labels_remediated.csv` | `02c93faa4cc66b9869e4d19f30719145f9c6e4c65869982722416d3e36ebbe05` |
| `dataset_out/labels_final.csv` | `bace79adb7bc82173aa4e38922546b4d40aa100bba1a65ae358830518330af94` |
| `dataset_out/release/crops.csv` | `232056ab582d589e88d47c4f8b2024c9198f5730a50be8622c0655e1272cf449` |
| `dataset_out/release/datapackage.json` | `e586daed8fb62118a5d6d1e15d510b5918c715bc06fc1f13b69665b7ca87e1e1` |
| `dataset_out/release/croissant.json` | `ba33665225b7c4baffb361f97a1c6643abc5178d39d0ba1d4cec5eb57610b715` |

## Lần chạy 2026-07-21T14:33:36Z

phạm vi: build remediate fuse confusion publish | strict=0 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `16a2375ca54d1d2d8265b033ae416e04cae2f73804fb54b7478cb5aa0c5250dd` |
| `dataset_out/labels_remediated.csv` | `64b9fafd92a092b0f17c9ef1da4c9255c47415bee28aaa8355a2a7dfb5895a95` |
| `dataset_out/labels_final.csv` | `b32599a857e8e1ac39e2fa355b1d9d70e43a7c92ab22ba964e3b0f9988a3fda5` |
| `dataset_out/release/crops.csv` | `0bf40da4b8708a16c0dfe5955452a7bd0c135b5654f147c0e474bda56f2027cf` |
| `dataset_out/release/datapackage.json` | `fc9c165ce644fe7a8925e1636bb1bb643356b912a07467d8f8664a91c4fb90f5` |
| `dataset_out/release/croissant.json` | `3093e4821414ad1bb6a788dfa0038d1ee2ed88ea6238683ab1d6e7518f67c424` |

## Lần chạy 2026-07-22T14:01:38Z

sách: STT2 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `8e214572229ae3fa14eb7f478999d8c68eefc46417199f0fafb2b798c16f0b19` |
| `dataset_out/labels_remediated.csv` | `96ee3741a9a912229b89e244ad3f98e779f85809daab94a4abef923ed3f13b01` |

## Lần chạy 2026-07-22T14:49:00Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `8e214572229ae3fa14eb7f478999d8c68eefc46417199f0fafb2b798c16f0b19` |
| `dataset_out/labels_remediated.csv` | `96ee3741a9a912229b89e244ad3f98e779f85809daab94a4abef923ed3f13b01` |
| `dataset/labels.csv` | `fe286d05804d5ba18754b28237f8b248d367f72da4da2be8143c9019faeaf87f` |

## Lần chạy 2026-08-19T04:51:50Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `7e0a2f3901c84a82a293ff5703c80f24406fd88847252d9a6f4643c1e4eac786` |
| `dataset_out/labels_remediated.csv` | `6232b64d3c147d9f669f834a8c051b4a789e8e6418c8045abe749fd57595963c` |
| `dataset_out/labels_final.csv` | `80bcb2cd3ed2bcdd7002a774f623277099851f580154893e27cfe5639d243d7c` |
| `dataset/labels.csv` | `5a4848648bdee1355613dc28f606100784235f92f65bdb1dcb0239bac55397e6` |

## Lần chạy 2026-08-23T09:17:03Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `7e0a2f3901c84a82a293ff5703c80f24406fd88847252d9a6f4643c1e4eac786` |
| `dataset_out/labels_remediated.csv` | `532f7c7de777fa524b9d045c221bc7d5ea2f974692645b19bbf291fb2b558b31` |
| `dataset_out/labels_final.csv` | `3208cdab6272e154279a4f25ac573489c5912b6bceca45c7dd66e1ec60c2792a` |
| `dataset/labels.csv` | `dbad35e92c76f84882757f74cd50a8670224d2f0d741691ce5f38469459c472a` |
<!-- HIEN_HANH:START -->
## BẢN HIỆN HÀNH (tự sinh — ghi đè mỗi lần chạy, ĐỪNG sửa tay)

- sinh lúc: `2026-08-25T14:41:36Z`
- commit  : `a8bbf959bb`
- cây làm việc: 🔴 BẨN (8 tệp đã sửa chưa commit) — commit ở trên KHÔNG
  định danh được mã đã chạy. Muốn tái lập thì phải commit trước khi chạy.
     M dataset_out/labels_final.csv
     M docs/EVIDENCE_INDEX.md
     M pipeline/remediation/selftest.py
     M re-dataset/DATASHEET.md
     M re-dataset/README.md
     M re-dataset/labels.csv
     M run_pipeline.sh
     M scripts/run_all_selftests.sh
- sách    : STT2+STT4+STT11 | reseg=detector

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `256519f3e2c8f91c875ced6e5cffabf84a0366a538a1a3174b0772b8ee4857da` |
| `dataset_out/labels_remediated.csv` | `9c6371ef2c0c4a26a3744e073e781e65f7fbda459b7c7d790dd16b9129e24462` |
| `dataset_out/labels_final.csv` | `550a9726bc0b1f80534ce4f0a7dca501b81d3120a0faec70a0ae55763eabed2f` |
| `re-dataset/labels.csv` | `c346a0321ecbb1dfc6a03491aaa44a157704281416fe1bb7a4f8364bf8ac4b89` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `48899fa32efb99bd43f8b34c8b05311e5a351e20ca12377eb951baf25bbb8044` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

**Bộ dò ký tự**: `mdnt571/nom-char-det` @ HuggingFace,
  tệp `detector_r34.PROD_e38_img1024.best.pt`, oid `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694`
  ⚠️ CÙNG REPO có `detector_r34.best.pt` là MỘT MÔ HÌNH KHÁC (epoch 41, img 1280,
  F1 0,8298 so với bản sản xuất epoch 38, img 1024, F1 0,8436) — 259/259 tensor khác
  nhau. infer_centernet.py:198 lấy độ phân giải TỪ checkpoint nên thay nhầm sẽ
  letterbox ở 1280 và cho hộp khác MÀ KHÔNG BÁO LỖI. Luôn dùng tệp tên PROD.

**Checkpoint S3**: `mdnt571/nom-embed` @ HuggingFace, revision `7ff74f57c4be`
- `best.pt` LFS oid = `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0`
- `last.pt` LFS oid = `c05dd1723c751059`… (xem repo HF)
- git báo `nom-embed` "modified" là ARTEFACT: huggingface_hub tải tệp THẬT ghi đè
  con trỏ LFS 134 byte, nên git so 134 byte với 140 MB. `oid` trong con trỏ KHỚP
  sha256 tệp trên đĩa VÀ khớp bản trên HF -> chuỗi xuất xứ NGUYÊN VẸN.
  ĐỪNG `git checkout` các tệp này: sẽ thay tệp thật bằng con trỏ và làm sập S3.

Kiểm lại: `bash scripts/check_evidence.sh`
<!-- HIEN_HANH:END -->

## Lần chạy 2026-08-24T01:16:48Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `9f7c0e3560d93076ccb6a1bcd9dcbd1ad8e36723ae512358da439e768331f7d5` |
| `dataset_out/labels_remediated.csv` | `110eea3a5f295217fba9c7dcfc4aab87e5197006ac295139c4b6197feb044336` |
| `dataset_out/labels_final.csv` | `1fc93dfc763391aaad53b296c61684ac40235ce90157d7f88fd0cd72958bb428` |
| `dataset/labels.csv` | `d6fe4676cd4ee7db3e5c3b2b759e7ed50d1d4cd85fe60472c31d812e162b10f8` |

## Lần chạy 2026-08-24T01:46:51Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `461ab90f3d4c0af57cc714e9b8633c45c6b82cf596e7fe8cfd6cf55a3127a976` |
| `dataset_out/labels_remediated.csv` | `2a78d6196a149205521a881da8f212c90b3065a236100b8d191daef1a8787e63` |
| `dataset_out/labels_final.csv` | `9727b4624d3e02943a8eca2a5725c4a486db1bf06b86364f1bbcd7ae6fe23368` |
| `dataset/labels.csv` | `236cbc4fa57f25b32270b89f95914ab62184011ffedf9ccb7c88d53c0b7ae0fc` |

## Lần chạy 2026-08-24T02:00:59Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `461ab90f3d4c0af57cc714e9b8633c45c6b82cf596e7fe8cfd6cf55a3127a976` |
| `dataset_out/labels_remediated.csv` | `2a78d6196a149205521a881da8f212c90b3065a236100b8d191daef1a8787e63` |
| `dataset_out/labels_final.csv` | `9727b4624d3e02943a8eca2a5725c4a486db1bf06b86364f1bbcd7ae6fe23368` |
| `dataset/labels.csv` | `236cbc4fa57f25b32270b89f95914ab62184011ffedf9ccb7c88d53c0b7ae0fc` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |

## Lần chạy 2026-08-24T06:51:04Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `461ab90f3d4c0af57cc714e9b8633c45c6b82cf596e7fe8cfd6cf55a3127a976` |
| `dataset_out/labels_remediated.csv` | `2a78d6196a149205521a881da8f212c90b3065a236100b8d191daef1a8787e63` |
| `dataset_out/labels_final.csv` | `9727b4624d3e02943a8eca2a5725c4a486db1bf06b86364f1bbcd7ae6fe23368` |
| `dataset/labels.csv` | `236cbc4fa57f25b32270b89f95914ab62184011ffedf9ccb7c88d53c0b7ae0fc` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |

## Lần chạy 2026-08-24T07:43:49Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `461ab90f3d4c0af57cc714e9b8633c45c6b82cf596e7fe8cfd6cf55a3127a976` |
| `dataset_out/labels_remediated.csv` | `2a78d6196a149205521a881da8f212c90b3065a236100b8d191daef1a8787e63` |
| `dataset_out/labels_final.csv` | `9727b4624d3e02943a8eca2a5725c4a486db1bf06b86364f1bbcd7ae6fe23368` |
| `dataset/labels.csv` | `236cbc4fa57f25b32270b89f95914ab62184011ffedf9ccb7c88d53c0b7ae0fc` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |

## Lần chạy 2026-08-24T13:05:58Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `461ab90f3d4c0af57cc714e9b8633c45c6b82cf596e7fe8cfd6cf55a3127a976` |
| `dataset_out/labels_remediated.csv` | `2a78d6196a149205521a881da8f212c90b3065a236100b8d191daef1a8787e63` |
| `dataset_out/labels_final.csv` | `9727b4624d3e02943a8eca2a5725c4a486db1bf06b86364f1bbcd7ae6fe23368` |
| `dataset/labels.csv` | `236cbc4fa57f25b32270b89f95914ab62184011ffedf9ccb7c88d53c0b7ae0fc` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `a1225ed806cf6740c81f88524e2b40893c161586f8f51f9996b302430628547d` |

## Lần chạy 2026-08-24T13:55:49Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `461ab90f3d4c0af57cc714e9b8633c45c6b82cf596e7fe8cfd6cf55a3127a976` |
| `dataset_out/labels_remediated.csv` | `2a78d6196a149205521a881da8f212c90b3065a236100b8d191daef1a8787e63` |
| `dataset_out/labels_final.csv` | `9727b4624d3e02943a8eca2a5725c4a486db1bf06b86364f1bbcd7ae6fe23368` |
| `dataset/labels.csv` | `236cbc4fa57f25b32270b89f95914ab62184011ffedf9ccb7c88d53c0b7ae0fc` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `a1225ed806cf6740c81f88524e2b40893c161586f8f51f9996b302430628547d` |

## Lần chạy 2026-08-24T14:25:42Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `ff56c6818315f69646364fad76fdf19350777673b942813e527e02fae950cb57` |
| `dataset_out/labels_remediated.csv` | `2a2c461c0f8199bea2f27553372003f73937c0a01b9b61d2fa911d6c411573ff` |
| `dataset_out/labels_final.csv` | `3112c5221acf06d39ba07354ebb9c97747997be03520d798754d7e220d1b5af0` |
| `dataset/labels.csv` | `258944f50f7496bac1862aa4e8b904aa9ec73ec12ef8e9b5ffda9a23a57371ea` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `a7bd2bd79875c5e4ea72ce4a53a48c5b13d41d5b09f39d6fe2b83b80e2349e2f` |

## Lần chạy 2026-08-24T15:42:19Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `4c309758c6c5b628b344abc8da67bf1ce35769273151e698285aeb903397e844` |
| `dataset_out/labels_remediated.csv` | `1bc34138c5ba6e523bc028ee898b54f12225762087ab92b76792d587d821bab6` |
| `dataset_out/labels_final.csv` | `f7ec09577a72e74db50a84f9ddaf32bc1cbee0bb83519ac3d1db31bad5ec9be6` |
| `dataset/labels.csv` | `20b15db39cc2e0ad3914f9a767a137ccd5fff9ee98d63e20f7765e22e2373cc9` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-24T15:47:41Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `1ae3a383189e94916e57c40bdc39d9a4dd482771bd0d09ca4d9862e590604d9a` |
| `dataset_out/labels_remediated.csv` | `76dfb3ecc7d07c08a53ade399816adc3b68571031c63f4ea131a1645d299674b` |
| `dataset_out/labels_final.csv` | `73b9c7fb25ecb8120688577372d7f77cd9374ca484829fba431d08dc468e54fb` |
| `dataset/labels.csv` | `050327f869f817dd846a265d3267fd76b7d69b7ff59ca37bb2d4de830cd6c147` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T01:27:50Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `5eeb3450a38a22a08b4dff518c5e7d7bfa78bd01232bd277cf2f2875ac634a6f` |
| `dataset_out/labels_remediated.csv` | `7a42c650166302ca2b09be534dec34b0066188b836f6440bb4d0127fabc990bf` |
| `dataset_out/labels_final.csv` | `d121b6d22ac7013646a03e6df89ba5032ec62195d793eef498bc42acaba6f550` |
| `dataset/labels.csv` | `816363b4c5b789d063632a12cba956e6ecbaa613cd14f5a2e0afe2fea3e1b1ed` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T01:32:33Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `5eeb3450a38a22a08b4dff518c5e7d7bfa78bd01232bd277cf2f2875ac634a6f` |
| `dataset_out/labels_remediated.csv` | `7a42c650166302ca2b09be534dec34b0066188b836f6440bb4d0127fabc990bf` |
| `dataset_out/labels_final.csv` | `d121b6d22ac7013646a03e6df89ba5032ec62195d793eef498bc42acaba6f550` |
| `dataset/labels.csv` | `816363b4c5b789d063632a12cba956e6ecbaa613cd14f5a2e0afe2fea3e1b1ed` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T01:53:34Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `1ae3a383189e94916e57c40bdc39d9a4dd482771bd0d09ca4d9862e590604d9a` |
| `dataset_out/labels_remediated.csv` | `0278b6bee659a154c9a1f4ef37e0ace35a6a7290754f42bb3373ed6b894818e2` |
| `dataset_out/labels_final.csv` | `87ac87a990cacc41d79a319c65e5b7d58719024e7b7fa63f1f0cd89570bc4baf` |
| `dataset/labels.csv` | `f92fbb0f4f7b4851e7ede29899544f12cbb7cdfdf2b1533fdd8e336227845d8d` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T02:21:53Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `1ae3a383189e94916e57c40bdc39d9a4dd482771bd0d09ca4d9862e590604d9a` |
| `dataset_out/labels_remediated.csv` | `0278b6bee659a154c9a1f4ef37e0ace35a6a7290754f42bb3373ed6b894818e2` |
| `dataset_out/labels_final.csv` | `87ac87a990cacc41d79a319c65e5b7d58719024e7b7fa63f1f0cd89570bc4baf` |
| `dataset/labels.csv` | `f92fbb0f4f7b4851e7ede29899544f12cbb7cdfdf2b1533fdd8e336227845d8d` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T05:11:15Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `1ae3a383189e94916e57c40bdc39d9a4dd482771bd0d09ca4d9862e590604d9a` |
| `dataset_out/labels_remediated.csv` | `0278b6bee659a154c9a1f4ef37e0ace35a6a7290754f42bb3373ed6b894818e2` |
| `dataset_out/labels_final.csv` | `87ac87a990cacc41d79a319c65e5b7d58719024e7b7fa63f1f0cd89570bc4baf` |
| `dataset/labels.csv` | `072199695dfed2f9c074f02ad82d58ef47e7d2fc48c13274e4cf50a028c66be8` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T05:35:50Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `5eeb3450a38a22a08b4dff518c5e7d7bfa78bd01232bd277cf2f2875ac634a6f` |
| `dataset_out/labels_remediated.csv` | `7a42c650166302ca2b09be534dec34b0066188b836f6440bb4d0127fabc990bf` |
| `dataset_out/labels_final.csv` | `d121b6d22ac7013646a03e6df89ba5032ec62195d793eef498bc42acaba6f550` |
| `re-dataset/labels.csv` | `aecd98a1a8893db96a1e9aa0063bcac91bd8761e4d1bb27da3aa4141cfae0c38` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T05:52:36Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `5eeb3450a38a22a08b4dff518c5e7d7bfa78bd01232bd277cf2f2875ac634a6f` |
| `dataset_out/labels_remediated.csv` | `7a42c650166302ca2b09be534dec34b0066188b836f6440bb4d0127fabc990bf` |
| `dataset_out/labels_final.csv` | `d121b6d22ac7013646a03e6df89ba5032ec62195d793eef498bc42acaba6f550` |
| `re-dataset/labels.csv` | `aecd98a1a8893db96a1e9aa0063bcac91bd8761e4d1bb27da3aa4141cfae0c38` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T06:13:11Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `5eeb3450a38a22a08b4dff518c5e7d7bfa78bd01232bd277cf2f2875ac634a6f` |
| `dataset_out/labels_remediated.csv` | `7a42c650166302ca2b09be534dec34b0066188b836f6440bb4d0127fabc990bf` |
| `dataset_out/labels_final.csv` | `d121b6d22ac7013646a03e6df89ba5032ec62195d793eef498bc42acaba6f550` |
| `re-dataset/labels.csv` | `aecd98a1a8893db96a1e9aa0063bcac91bd8761e4d1bb27da3aa4141cfae0c38` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T06:24:05Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `1ae3a383189e94916e57c40bdc39d9a4dd482771bd0d09ca4d9862e590604d9a` |
| `dataset_out/labels_remediated.csv` | `0278b6bee659a154c9a1f4ef37e0ace35a6a7290754f42bb3373ed6b894818e2` |
| `dataset_out/labels_final.csv` | `87ac87a990cacc41d79a319c65e5b7d58719024e7b7fa63f1f0cd89570bc4baf` |
| `re-dataset/labels.csv` | `072199695dfed2f9c074f02ad82d58ef47e7d2fc48c13274e4cf50a028c66be8` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T06:34:25Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `1ae3a383189e94916e57c40bdc39d9a4dd482771bd0d09ca4d9862e590604d9a` |
| `dataset_out/labels_remediated.csv` | `0278b6bee659a154c9a1f4ef37e0ace35a6a7290754f42bb3373ed6b894818e2` |
| `dataset_out/labels_final.csv` | `87ac87a990cacc41d79a319c65e5b7d58719024e7b7fa63f1f0cd89570bc4baf` |
| `re-dataset/labels.csv` | `925014110e39f6f2b0f7e320a4b22df9bf6050b2cc262c84a2e1834c99a0dd0b` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T06:47:42Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `1ae3a383189e94916e57c40bdc39d9a4dd482771bd0d09ca4d9862e590604d9a` |
| `dataset_out/labels_remediated.csv` | `0278b6bee659a154c9a1f4ef37e0ace35a6a7290754f42bb3373ed6b894818e2` |
| `dataset_out/labels_final.csv` | `87ac87a990cacc41d79a319c65e5b7d58719024e7b7fa63f1f0cd89570bc4baf` |
| `re-dataset/labels.csv` | `925014110e39f6f2b0f7e320a4b22df9bf6050b2cc262c84a2e1834c99a0dd0b` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T07:14:54Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `30a213e653ab9a26def458031a501db65f01bac23c24c1b2932abdb9b6329649` |
| `dataset_out/labels_remediated.csv` | `797fcf6528e4cf893a715e8d812c1ba67b0b033ed8f69574c886a8a97c6c786a` |
| `dataset_out/labels_final.csv` | `133ae2a38d38e24d323f448a97ddb569f35d7632680a39e0fb45bc5f09e8b26e` |
| `re-dataset/labels.csv` | `03b3a04db47f41e6acc601686a10a9685ae94d44cc9d5390f197080d266761d8` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T07:25:48Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `a89b8f9e078bc40b8ed8301d60661a1e297b810cfc12599a6ec003ef5cdb8f81` |
| `dataset_out/labels_remediated.csv` | `eeff0b9756571784ec353f3cf04c361b65efd82582f2b572b54ae23d68ed6b03` |
| `dataset_out/labels_final.csv` | `d9847af0539f124302a6e419fc864696f465f494fbd86eb9a997fbe39eafcd67` |
| `re-dataset/labels.csv` | `23aa32fadb009945701e588a7ac62d518a232be5397d288b7c8b51f7351e460a` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T07:34:42Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `a89b8f9e078bc40b8ed8301d60661a1e297b810cfc12599a6ec003ef5cdb8f81` |
| `dataset_out/labels_remediated.csv` | `b4d913bffbe0a546a23cd1a1048b55a499fc525b1567eafecee873a001147014` |
| `dataset_out/labels_final.csv` | `cb74db2b995ce0dbf6fe297a46a6cc957b7da62f93dcf3b04266495a54ec2413` |
| `re-dataset/labels.csv` | `839a10688d2e2cbef6e5dbf3f38e77f34119be6f8c73cc1ab4c80af25654aed4` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T08:03:31Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `30a213e653ab9a26def458031a501db65f01bac23c24c1b2932abdb9b6329649` |
| `dataset_out/labels_remediated.csv` | `311fdd9f05248acdc7d9f8bd30ccefad8f961e1e9ac93b239a020f2bc8cec514` |
| `dataset_out/labels_final.csv` | `797d78f1f67a628567a0f0487bab59a19bc09296149c8b893935b1392c8a1d4e` |
| `re-dataset/labels.csv` | `954b5e179e397e37511d818cef4a0401183f622435b69bfc113578e56c6f7ff4` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T08:13:55Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `a89b8f9e078bc40b8ed8301d60661a1e297b810cfc12599a6ec003ef5cdb8f81` |
| `dataset_out/labels_remediated.csv` | `b4d913bffbe0a546a23cd1a1048b55a499fc525b1567eafecee873a001147014` |
| `dataset_out/labels_final.csv` | `cb74db2b995ce0dbf6fe297a46a6cc957b7da62f93dcf3b04266495a54ec2413` |
| `re-dataset/labels.csv` | `839a10688d2e2cbef6e5dbf3f38e77f34119be6f8c73cc1ab4c80af25654aed4` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `cca7d2a078b22c83831f8a5fd3e8d3d468a09319266a89606f889a1668020097` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T10:10:10Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `256519f3e2c8f91c875ced6e5cffabf84a0366a538a1a3174b0772b8ee4857da` |
| `dataset_out/labels_remediated.csv` | `9c6371ef2c0c4a26a3744e073e781e65f7fbda459b7c7d790dd16b9129e24462` |
| `dataset_out/labels_final.csv` | `e897fe697dd33deb9d5ea45523eb798c4d7c6a25876e7d279f1971584704b3e4` |
| `re-dataset/labels.csv` | `cf180600f308399214dced97c0e0a48bba38aa714153a39096798a1d2121f56a` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `48899fa32efb99bd43f8b34c8b05311e5a351e20ca12377eb951baf25bbb8044` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T11:55:52Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `256519f3e2c8f91c875ced6e5cffabf84a0366a538a1a3174b0772b8ee4857da` |
| `dataset_out/labels_remediated.csv` | `9c6371ef2c0c4a26a3744e073e781e65f7fbda459b7c7d790dd16b9129e24462` |
| `dataset_out/labels_final.csv` | `e897fe697dd33deb9d5ea45523eb798c4d7c6a25876e7d279f1971584704b3e4` |
| `re-dataset/labels.csv` | `cf180600f308399214dced97c0e0a48bba38aa714153a39096798a1d2121f56a` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `48899fa32efb99bd43f8b34c8b05311e5a351e20ca12377eb951baf25bbb8044` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T12:04:13Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `256519f3e2c8f91c875ced6e5cffabf84a0366a538a1a3174b0772b8ee4857da` |
| `dataset_out/labels_remediated.csv` | `9c6371ef2c0c4a26a3744e073e781e65f7fbda459b7c7d790dd16b9129e24462` |
| `dataset_out/labels_final.csv` | `e897fe697dd33deb9d5ea45523eb798c4d7c6a25876e7d279f1971584704b3e4` |
| `re-dataset/labels.csv` | `83a217251ea45507627b98f07fb8a4eb0ad1c7993d4a58a5bbbf7da998acfc87` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `48899fa32efb99bd43f8b34c8b05311e5a351e20ca12377eb951baf25bbb8044` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |

## Lần chạy 2026-08-25T14:41:35Z

sách: STT2+STT4+STT11 | reseg=detector | config=config/pipeline.yaml

| file | sha256 |
|---|---|
| `dataset_out/labels.csv` | `256519f3e2c8f91c875ced6e5cffabf84a0366a538a1a3174b0772b8ee4857da` |
| `dataset_out/labels_remediated.csv` | `9c6371ef2c0c4a26a3744e073e781e65f7fbda459b7c7d790dd16b9129e24462` |
| `dataset_out/labels_final.csv` | `550a9726bc0b1f80534ce4f0a7dca501b81d3120a0faec70a0ae55763eabed2f` |
| `re-dataset/labels.csv` | `c346a0321ecbb1dfc6a03491aaa44a157704281416fe1bb7a4f8364bf8ac4b89` |
| `nom-embed/best.pt` | `eee2f3e706b08622320b3024ce244b9cbf01ed2df5c2f279d2b5c358ce6ee3d0` |
| `pipeline/align_engine/data/index.csv` | `48899fa32efb99bd43f8b34c8b05311e5a351e20ca12377eb951baf25bbb8044` |
| `Dict/QuocNgu_SinoNom.csv` | `e65b98748e13e41f66e1f5527e4b07e7f89ba24522090a4050decab95f1c87d8` |
| `Dict/SinoNom_Similar.csv` | `2ac4cb6dea38e9a6fc544966051d06932c67d918d06c3a0b640ab4b70b9b01b3` |
| `train_crop/detector_r34.best.pt` | `2c119689debfff01a81fee6bf198181c47fe8ab3ccb1eeaa6b576539dd344694` |
| `config/pipeline.yaml` | `ac7038d9c0b16d28b8c64b25e0904ffb50be85fb9cf97651f57f455e9468d1f0` |
