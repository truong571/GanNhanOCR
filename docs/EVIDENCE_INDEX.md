# SỔ ĐĂNG KÝ BẰNG CHỨNG (EVIDENCE INDEX)

**Lập**: 2026-07-20 (Giai đoạn 0 — Cứu bằng chứng)
**Mục đích**: mỗi artifact được dùng làm bằng chứng trong luận văn phải truy nguyên được về (a) commit chứa nó, (b) sha256 tại thời điểm đóng băng, (c) lệnh tái sinh nó.

> **Quy tắc**: không chương nào của luận văn được trích số từ artifact không có trong bảng này.

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

**Trạng thái remote** (cập nhật 2026-07-20 23:3x): nhánh `feat/phases-0-3-audit-pipeline` **đã push** lên `github.com:truong571/GanNhanOCR.git` tại `9da646c5c4`, kèm tag `freeze-pre-thesis-2026-07-20`. Xác minh: local HEAD = remote HEAD; remote có đủ **54 file** `dataset_out/ground_truth/` + `labels_final.csv`. `main` vẫn ở `cdf68821e4` (chưa merge — thuộc Giai đoạn 1).

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

## 6. VIỆC CÒN THIẾU CỦA GIAI ĐOẠN 0

- [ ] **Push nhánh + tag lên remote** — hiện bằng chứng chỉ có trên 1 máy + 2 gói sao lưu cùng ổ đĩa. Cần bản sao ở nơi khác về mặt vật lý.
- [ ] Cân nhắc lưu trữ `evidence_2026-07-20.tgz` lên Zenodo/Google Drive để có DOI/bản sao ngoài máy.
