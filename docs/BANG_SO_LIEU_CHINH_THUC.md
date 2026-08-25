# BẢNG SỐ LIỆU CHÍNH THỨC

<!-- AUTO:HEADER:START -->
**Bộ nhãn sinh ngày**: 2026-08-25 · **Commit chạm bộ nhãn gần nhất**: `5462713955` · **Bộ nhãn**: `dataset_out/labels_final.csv` (82.246 dòng)
<!-- AUTO:HEADER:END -->

> **QUY TẮC BẤT DI BẤT DỊCH**: mọi con số trong luận văn (mọi chương, mọi bảng, mọi slide) **chỉ
> được trích từ tệp này**. Mỗi số dưới đây có: giá trị · lệnh tái sinh · **nguồn kiểm định**.

## 🔴 TUYÊN BỐ HUỶ SỐ (2026-08-22)

**Đề tài hiện có 0 (không) phán quyết người dùng được.** Toàn bộ phần "chấm tay" trước đây thực chất
do **máy chấm**. Bằng chứng: `dataset_out/human_audit/verdicts_reanchored.csv` (846 phán quyết —
nguồn của MỌI số precision cũ) trùng **846/846 `item_id`** với `audit_gold/audit_gold.jsonl` (tệp máy
chấm) nhưng chỉ khớp giá trị **47/846**; verdict thô gốc `verdicts_001–006.jsonl` **đã mất**
(`docs/EVIDENCE_INDEX.md:18` tự khai). Chi tiết: `docs/KE_HOACH_TONG_THE_2026-08-22.md` §0.

**Hệ quả**: mọi số ở **§4 (SỐ ĐÃ HUỶ)** không được trích vào luận văn dưới bất kỳ hình thức nào cho
tới khi có mẻ chấm người mới. Các số ở §1–§3 và §5 **không phụ thuộc phán quyết** nên vẫn nguyên giá trị.

Ba nhãn nguồn kiểm định dùng trong tệp này:

| nhãn | nghĩa |
|---|---|
| 🔢 **MÁY ĐẾM** | đại lượng tất định, tái sinh được bằng lệnh; không cần ai phán xét |
| ⚪ **CHƯA ĐO** | cần phán quyết người, hiện chưa có |
| 🔴 **ĐÃ HUỶ** | từng được công bố, nay rút lại vì xuất xứ không truy nguyên được |

---

## 1. NGUỒN GỐC — chuỗi nhãn bất biến 🔢

```
labels.csv --[4 remediate]--> labels_remediated.csv --[5 confusion_fix]--> labels_final.csv
           --[6 s3_unwind, ghi đè]--> labels_final.csv --[7 export]--> dataset/labels.csv
```

<!-- AUTO:NGUON_GOC:START -->
| Tệp | dòng | sha256 (16 đầu) | Lệnh tái sinh |
|---|---|---|---|
| `dataset_out/labels.csv` | 82.246 | `a89b8f9e078bc40b` | `python -m pipeline.align_engine.build_dataset --config config/pipeline.yaml --use-s3 --reseg detector` |
| `dataset_out/labels_remediated.csv` | 82.246 | `b4d913bffbe0a546` | `python -m pipeline.remediation --labels dataset_out/labels.csv --out dataset_out apply --tau 0.62` |
| `dataset_out/labels_final.csv` | 82.246 | `cb74db2b995ce0db` | `python -m pipeline.remediation.confusion_fix … rồi python -m pipeline.remediation.s3_unwind … --apply` |
| `dataset/labels.csv` | (chưa có) | — | `python pipeline/export_final_dataset.py --labels dataset_out/labels_final.csv --src-root dataset_out --out dataset` |
<!-- AUTO:NGUON_GOC:END -->

**Đối chiếu**: `bash scripts/check_evidence.sh` → khớp 4 · lệch 0 · thiếu 0.
Vân tay từng bước: `dataset_out/CHECKSUMS.txt`.

---

## 2. BỘ NHÃN HIỆN HÀNH 🔢

### 2.1 Phân hạng

<!-- AUTO:PHAN_HANG:START -->
| Tier | Số ô | % | Vào bộ giao nộp? | Nguồn kiểm định |
|---|---|---|---|---|
| **GOLD** | **50.156** | 61.0% | ✅ | ⚪ CHƯA ĐO |
| SILVER_uncalibrated | 10.547 | 12.8% | ❌ ngoài `USABLE_TIERS` | ⚪ CHƯA ĐO (verdict **máy**, không dùng làm bằng chứng) |
| SYLLABLE | 6.911 | 8.4% | ✅ (nhãn cấp **âm tiết**) | ⚪ CHƯA ĐO |
| REVIEW | 14.632 | 17.8% | ❌ | — (không phải nhãn) |

**Bộ giao nộp = 50.156 nhãn CẤP KÝ TỰ + 6.911 chú giải CẤP ÂM TIẾT** = 57.067 dòng · 57.067 ảnh đã copy, **0 thiếu**.

> Nguồn: `re-dataset/labels.csv` — **bộ ĐEM CHẤM (CHƯA kiểm chứng)**.

> ⚠️ **KHÔNG phát biểu là “57.067 nhãn”.** 6.911/6.911 dòng tầng SYLLABLE có cột `label` **rỗng** — chúng chỉ ghi ÂM Quốc ngữ, không gán chữ Nôm. Con số dùng khi so với các bộ dữ liệu Hán Nôm khác là **50.156**.
<!-- AUTO:PHAN_HANG:END -->

Tái sinh: `python -c "import csv,collections;r=list(csv.DictReader(open('dataset_out/labels_final.csv')));print(len(r),collections.Counter(x['tier'] for x in r))"`

### 2.2 Luật sinh nhãn

<!-- AUTO:LUAT:START -->
| Rule | Số ô | Tier |
|---|---|---|
| `s1_inter_s2_direct` | 46.327 | GOLD |
| `no_s1_inter_s2` | 11.007 | REVIEW |
| `s2_inter_s3_corrected` | 9.997 | SILVER_uncalibrated |
| `nghia_consensus` | 6.911 | SYLLABLE |
| `s1_inter_s2_similar` | 3.829 | GOLD |
| `confusion_fix:systematic_confusion_nguoi_3775` | 1.988 | REVIEW |
| `diverged_column` | 1.599 | REVIEW |
| `s3_head_bank_consensus` | 293 | SILVER_uncalibrated |
| `s1_inter_s3_out_of_dict` | 257 | SILVER_uncalibrated |
| `unconfirmed_no_s3` | 38 | REVIEW |
<!-- AUTO:LUAT:END -->

### 2.3 Phạm vi và lớp

<!-- AUTO:PHAM_VI:START -->
| Chỉ số | Giá trị |
|---|---|
| Sách | 3 (stt11, stt2, stt4) |
| Trang | 445 |
| **Trang cho đủ 9 cột có nhãn** | **443/445** |
| Lớp ký tự phân biệt (mọi tier có nhãn) | 1.601 |
| **Lớp trong bộ giao nộp** | **1.583** |
| Split bộ giao nộp | test 6.155 · train 45.066 · val 5.846 |
| **Selftest** | **646 passed, 0 failed** |
<!-- AUTO:PHAM_VI:END -->

### 2.4 Vá lỗi (bước 4–6)

<!-- AUTO:VA_LOI:START -->
| Chỉ số | Giá trị | Ghi chú |
|---|---|---|
| Quarantine (bbox trùng, nhãn mâu thuẫn) | **0** | lớp lỗi đã đóng ở gốc engine |
| Đổi split do trùng md5 | **0** | rò rỉ vốn đã bằng 0 trước bước 4 |
| Demote theo S3 (`--s3-demote`) | **0** | **TẮT MẶC ĐỊNH** từ 2026-08-19 (tiêu chí dựa trên S3, chưa chứng minh được) |
| Demote lớp confusion 㝵/"người" | **1.988** | chốt chặn ở `s3_unwind` (KHỐI 1.1) |
| Trả về GOLD sau `s3_unwind` | **0** | 0 = không còn ô nào mang hậu tố `demoted_lowcos_s3` |
<!-- AUTO:VA_LOI:END -->

---

## 3. SỐ ĐO ĐƯỢC KHÁC (không phụ thuộc phán quyết) 🔢

| Chỉ số | Giá trị | Lệnh / nguồn |
|---|---|---|
| Từ điển `QuocNgu_SinoNom.csv` | 104.177 cặp thô · **104.053** sau chuẩn hoá của bộ nạp | `load_qn_to_nom` |
| `SinoNom_Similar.csv` | 33.396 chữ | — |
| Unihan `kDefinition` | 23.285 chữ · phủ **89,4%** lớp GOLD, **96,9%** số ô | `Dict/unihan_kdefinition_full.csv` |
| **DPI hiệu dụng** (ảnh Nôm là ảnh **nhúng bóc ra**, không render) | STT2 p50 **302,7** · **STT4 p50 201,8** · STT11 p50 **302,7** | ⚠️ ~1/3 ngữ liệu ở 2/3 độ phân giải — **phải khai trong datasheet** |
| Ô neo `s1_inter_s2_direct` mỗi cột | trung vị **58%** (p10 42%, p90 73%) | chuẩn đo cho T3 |
| Kho glyph FontDiffusion | 89.898 tệp | `gannhanocr-fd/` |
| Đầu ArcFace trong `nom-embed/best.pt` | 1.591 lớp | ⚠️ thiếu **18 lớp** so với 1.609 lớp của bộ nhãn |
| Điểm nghĩa Hán (kênh xác nhận một chiều) | AUC **0,742**; ngưỡng >0,05 chính xác **100%** trên chuẩn 2.002 cặp | `python -m pipeline.tools.sem_score --bench` |

<!-- AUTO:DO_KHAC:START -->
| Chỉ số phụ thuộc dữ liệu | Giá trị |
|---|---|
| Âm QN hợp lệ (`is_plausible_qn_syllable`) | **99.74%** |
| **Âm QN có trong từ điển** | **99.196%** (661 ô ngoài) |
| Trang cho đủ 9 cột có nhãn | **443/445** |
| Mâu thuẫn tự thân (cùng md5, khác nhãn) | **1** / 62.690 crop |
<!-- AUTO:DO_KHAC:END -->

---

## 4. 🔴 SỐ ĐÃ HUỶ — KHÔNG ĐƯỢC TRÍCH

Mọi con số dưới đây neo vào bộ 846 phán quyết không truy nguyên được xuất xứ.

| Số từng công bố | Nội dung | Trạng thái |
|---|---|---|
| ~~97,98% [96,7–98,8]~~ (777/793) | Precision GOLD | 🔴 ĐÃ HUỶ |
| ~~98,00%~~ · ~~97,08%~~ | Precision GOLD trước/sau `confusion_fix` | 🔴 ĐÃ HUỶ |
| ~~737/752 = 98,0%~~ | Precision luật `s1_inter_s2_direct` | 🔴 ĐÃ HUỶ |
| ~~40/41 = 97,6%~~ | Precision luật `s1_inter_s2_similar` | 🔴 ĐÃ HUỶ |
| ~~0,566 [0,459–0,672]~~ | Error-AUC S3 cũ | 🔴 ĐÃ HUỶ — **và vốn đã không tái lập được**: là hằng số viết cứng ở `s3_unwind.py`, dựng lại bằng `s3_bank_cos` cho 0,504 |
| ~~0,577 [0,442–0,706]~~ | Error-AUC ArcFace retrain | 🔴 ĐÃ HUỶ. AUC 0,577 tái lập được, nhưng CI thì không (logit-DeLong cho [0,442–0,701]) |
| ~~κ = 0,13~~ | Độ tin cậy người chấm | 🔴 ĐÃ HUỶ — không có người chấm để mà đo κ |
| ~~Fisher p = 5,4×10⁻⁸~~ | Lớp 㝵/"người" sai hệ thống | ⚠️ **Số tái lập được** (23 ô, 8 sai, vs nền 16/802), nhưng **nguồn verdict đã huỷ** → giữ quyết định demote, **bỏ tư cách bằng chứng** cho tới mẻ mới |
| ~~66.589~~ · ~~56.824~~ · ~~50.063~~ | Cỡ bộ giao nộp / GOLD | 🔴 SỐ CŨ — số hiện hành ở §1 và §2.1 (tự sinh) |
| ~~1.924~~ · ~~1.926~~ | Số ô 㝵/"người" bị demote | 🔴 SỐ CŨ — số hiện hành ở §2.4 (tự sinh); 48 ô từng lọt về GOLD đã vá ở KHỐI 1 |
| ~~82.274 / GOLD 48.893 / SILVER 10.887 / SYLLABLE 6.809~~ | Bảng "thế hệ 08-11" | 🔴 **không khớp tệp nào trên đĩa** — đã xoá khỏi tài liệu này |

### 4b. Bước 6 `s3_unwind` — sửa lại cách phát biểu

Quyết định gỡ S3 **vẫn giữ**, nhưng lý do phải viết là **"chưa chứng minh được"**, KHÔNG phải
**"đã bác bỏ"**. Ba căn cứ:

1. **Thiếu lực thống kê**: lớp âm chỉ 24 ca. Theo Hanley–McNeil với 802 dương / 24 âm, AUC nhỏ nhất
   mà mẫu phát hiện được (CI95 loại trừ 0,5) là **0,607**. Một tín hiệu AUC thật 0,55–0,60 **về
   thiết kế là không thể phát hiện** bằng mẻ đó.
2. **Đo sai miền**: phép đo chạy trên GOLD, nơi S3 **không tham gia gán nhãn** và chỉ **7,7%** số ô
   có điểm S3 (SILVER_uncalibrated: 100%).
3. **Rò rỉ hiệu chuẩn**: **617/826 (74,7%)** crop verdict nằm trong split TRAIN của chính lần train
   ArcFace → con số vốn đã lạc quan, và mọi ngưỡng khớp trên bộ này đều vô giá trị.

Phát biểu đúng: *"cosine thô của S3 không bắt được lỗi âm đọc, và bộ ground-truth hiện tại quá ít ca
lỗi — lại bị rò rỉ train — để kết luận về khả năng bắt lỗi tự dạng."*

---

## 5. KIỂM ĐỊNH (selftest) 🔢

**448 passed, 0 failed** — `bash scripts/run_all_selftests.sh`
(mốc 2026-08-23; lịch sử 223 → 392 → 414 → 427 → 448).
Con số trích vào luận văn là **448 assertions**.

| Bộ | Assertion |
|---|---|
| `core.pdf.parser_selftest` | 18 |
| `core.text.syllable_validation_selftest` | 39 |
| `pipeline.ground_truth.selftest` | 170 |
| `pipeline.consensus_fusion.selftest` | 44 |
| `pipeline.publish.selftest` | 56 |
| `pipeline.remediation.selftest` | 78 |
| `pipeline.phase1_engine_selftest` | 43 |

---

## 6. QUY TẮC TRÍCH DẪN

1. Số 🔢 **MÁY ĐẾM** → trích tự do, kèm lệnh tái sinh và commit SHA.
2. Số ⚪ **CHƯA ĐO** → chỉ được viết "chưa đo", **không** được thay bằng ước lượng, không được
   mượn số cũ.
3. Số 🔴 **ĐÃ HUỶ** → không trích dưới bất kỳ hình thức nào, kể cả "khoảng", "ước tính", "trước đây".
4. Datasheet phải khai đủ 5 giới hạn: 3 sách 1 nét chữ · **STT4 ~201 DPI** kèm precision tách theo
   sách · SYLLABLE là nhãn cấp âm tiết (316 cặp không nguồn nào xác nhận) · SILVER công bố riêng gắn
   nhãn "uncalibrated" · **lịch sử mẻ phán quyết bị huỷ** (khai thẳng — đây là điểm cộng liêm chính).

## Ghi chú bảo trì

🤖 **Bốn khối giữa mốc `<!-- AUTO:… -->` là TỰ SINH — đừng sửa tay.** Sau mỗi lần đổi bộ nhãn:

```bash
python -m pipeline.tools.update_bang_so_lieu          # ghi lại
python -m pipeline.tools.update_bang_so_lieu --check  # chỉ kiểm (exit 1 nếu lệch)
```

KHỐI 2 viết lại tài liệu này để diệt lớp lệch "tài liệu nói một đằng, đĩa một nẻo". Rồi T1 và
T2 đổi bộ nhãn hai lần và tài liệu **lệch lại ngay** — vì không có gì ÉP cập nhật. Gõ tay thì
sẽ quên; nên mọi khối phụ thuộc dữ liệu nay tự sinh.


Tệp này được viết lại toàn bộ ngày 2026-08-23 (KHỐI 2). Bản trước đó chứa **đồng thời hai bộ số mâu
thuẫn** (hậu và tiền `s3_unwind`), trong đó một bộ không khớp tệp nào trên đĩa. Sau mỗi lần chạy
pipeline: cập nhật §1 (hash), §2 (đếm), §5 (selftest), rồi chạy `bash scripts/check_evidence.sh`.
