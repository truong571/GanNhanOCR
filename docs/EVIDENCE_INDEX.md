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
| `f78dbc4da5` | **Chứa toàn bộ artifact bằng chứng** (`dataset_out/ground_truth/**` 54 file, `labels_final.csv`, `config/confusion_fixes.yaml`, `dataset_out/fusion/**`) | Message gốc là "update code" — vô nghĩa. Đây là commit phải trích khi nói về bằng chứng audit. |
| `f47f431abd` | chore(cleanup): xoá mã chết + vá bug ghi đè `fused.csv` + chốt mốc selftest | Nhóm A kiểm kê 2026-07-20 |
| tag `freeze-pre-thesis-2026-07-20` | Điểm đóng băng Giai đoạn 0 | Mọi bằng chứng tính đến ngày này |

**Trạng thái remote** (cập nhật 2026-07-21): `main` = `feat/phases-0-3-audit-pipeline` = remote, đều đã push lên `github.com:truong571/GanNhanOCR.git`. Remote có đủ **54 file** `dataset_out/ground_truth/` + `labels_final.csv` và **3 tag**.

### 2-bis. Nội dung commit `f78dbc4da5` theo 4 nhóm

Commit này gộp **76 file** vào một lần với message `"update code"`. Không tách lại được (đã push, hash đã trích dẫn, `CODE_FREEZE.md` cấm viết lại SHA), nên liệt kê ở đây để **log vẫn kể được câu chuyện**:

| Nhóm | Nội dung |
|---|---|
| (a) 7 file `.py` | `consensus_fusion/{fuse_stage,mine_confusions,score_s3}.py` · `ground_truth/{batch_json,make_audit_batch,reanchor_verdicts}.py` · `remediation/confusion_fix.py` |
| (b) Cấu hình quyết định nhãn | `config/confusion_fixes.yaml` |
| (c) Bằng chứng | `dataset_out/ground_truth/**` (54 file) · `labels_final.csv` · `confusion_fix_report.json` · `dataset_out/fusion/**` |
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

**Số liệu chốt** (`labels_final.csv`, 82.274 dòng): GOLD **48.969** · SILVER **10.856** · SYLLABLE **6.751** · REVIEW **15.690** · QUARANTINE **8**. Dataset có-nhãn ≈ **66.576**.

### 3.2. Báo cáo xử lý

| Artifact | sha256 | Sinh bởi |
|---|---|---|
| `dataset_out/remediation_report.json` | `0490277e9787c3ff…0a7f1b71` | `pipeline.remediation apply` |
| `dataset_out/confusion_fix_report.json` | `28189cc49334592a…de167d961` | `pipeline.remediation.confusion_fix --measure` |

`confusion_fix_report`: 1 fix (người→㝵), demote **1.923 crop** → REVIEW; precision GOLD **0,9708 → 0,9800**.

### 3.3. Bằng chứng audit (ground truth)

| Artifact | sha256 / số lượng | Ghi chú |
|---|---|---|
| `dataset_out/ground_truth/report.json` | `63188b4f80d4386d…da3d8cad8` | ⚠️ precision 0,7298 ở đây là mẫu **suspicion-ranked + stratified SILVER (AI chấm)** — KHÁC khung lấy mẫu với 97–98% GOLD (SRS). Không được trộn hai số. |
| `dataset_out/ground_truth/verdicts_reanchored.csv` | `df5a6b1568be2daf…08c51882c` | 825/846 verdict người neo lại được (median IoU 1.0, 93% byte-identical) |
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
- Clone thử từ bundle → khôi phục 101 commit, HEAD `f78dbc4da5`, có đủ 54 file `dataset_out/ground_truth`

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
- Có đủ bằng chứng: 54 file `dataset_out/ground_truth/`, `labels_final.csv` (82.275 dòng), `confusion_fixes.yaml`, 3 tài liệu chiến lược, `requirements.lock.txt`
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
