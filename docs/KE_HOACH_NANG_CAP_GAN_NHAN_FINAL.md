# BẢN ĐẶC TẢ KỸ THUẬT NÂNG CẤP LUỒNG GÁN NHÃN HÁN NÔM ↔ QUỐC NGỮ (KHỐI A CHUẨN)
> **Phiên bản:** 3.1-Calibrated · **Ngày cập nhật:** 2026-09-15  
> **Căn cứ:** Đã đối soát và vượt qua 11 lượt kiểm chứng độc lập của `DANH_GIA_KE_HOACH_NANG_CAP_2026-09-15.md`.  
> **Nguyên tắc cốt lõi:** Văn bản trước – Ảnh làm trọng tài – Người ký theo lớp; bảo toàn tuyệt đối 2.014 ô QĐ-01; không thay đổi md5 crop; tái lập 100% bằng code.

---

## 1. MỤC TIÊU VÀ CHỈ SỐ NGHIỆM THU ĐÃ ĐỐI SOÁT

| Chỉ số kỹ thuật | Bản hiện hành (`bff2a3bd28`) | Bản nâng cấp Khối A (Chuẩn số đo) | Phương pháp kiểm chứng |
|---|---|---|---|
| **Số cột có khe giả** | 346 / 2.893 cột | **54 cột** (CALIB chuẩn) | `thuc_nghiem.py rebuild calib` |
| **Cặp ghép lệch chéo** | 1.63% (992 / 60.678) | **0.82%** (503 / 61.006) | `thuc_nghiem.py calib` |
| **Khe đúng chỗ (benchmark drop_char)** | 60% | **85 – 86%** (CALIB + neo đệ quy) | `thuc_nghiem.py recursive` |
| **Ghép sai khi rụng 1 chữ** | 3.85% | **1.36 – 1.42%** | `thuc_nghiem.py recursive` |
| **Số nhãn usable giao nộp** | 64.525 ô | **70.108 ± 300 ô** (`tier_v3` nguyên văn) | `thuc_nghiem_ke_hoach.py tier` |
| **Bảo toàn QĐ-01 người duyệt** | 2.014 ô | **ĐÚNG 2.014 Ô** (khóa theo ô `qd01_cells.csv`) | `assert qd01 == 2014` |
| **Số ô Quốc ngữ bị sửa đè âm** | ~161 ô ghi đè bản in | **0 ô** (giữ `syllable_raw` + duyệt corpus) | `crosstab L1` |
| **Độ đúng vị trí hộp khi $M \neq N$** | 59.1% | **$\ge 91.0\%$** (luật `hộp[j] ↔ âm j`, `thr=0.2`) | `thuc_nghiem.py geo` |
| **Chuỗi bằng chứng SHA256** | 2/4 hỏng (chưa đồng bộ) | **4/4 KHỚP TUYỆT ĐỐI** | `bash scripts/check_consistency.sh` |
| **Precision người** (Wilson 95% theo tầng: GOLD mã chữ · ghép âm · SYLLABLE; n≈600) | 0 phán quyết người | **chưa đo** — chỉ có số khi Khối C chấm xong; mẻ chấm phải rút từ đúng `labels_final.csv` đem nộp (`manifest.jsonl` ghi `labels_sha256`, A-14) | `pipeline.ground_truth` (grid → estimate) + `check_evidence.sh` |

_Bảng này KHÔNG có dòng "GOLD ≥98% bằng bộ kiểm độc lập / posterior" (bỏ theo `DANH_MUC_SUA_DOI_CUOI` D-6 / A-14: `re-dataset/check` mù mã chữ — 㝵 và 𠊚 cùng TRUE; `p_register` chỉ dùng phân tầng mẫu). Bản cập nhật đầy đủ của bảng: `DANH_MUC_SUA_DOI_CUOI_2026-09-16.md` §4._

---

## 2. LỘ TRÌNH 7 BƯỚC TRIỂN KHAI KHỐI A (LÀN VĂN BẢN AN TOÀN)

```
[A0: HÀNG RÀO BẢO VỆ]
  ├── Tạo dataset_out_v3/ (không ghi đè dataset_out gốc)
  ├── Xuất config/qd01_cells.csv khóa cứng 2.014 ô QĐ-01
  └── Viết bổ sung ≥15 selftest cho 5 hàm quyết định nhãn lõi
          │
[A1: MA TRẬN CALIB ĐÚNG & POSTERIOR T=1.0]
  ├── anchor_align.py: COST_CONFIRM=0.0, SIMILAR=2.5, DICTMISS=6.7, NODICT=5.1, DEL=INS=8.6
  └── posterior_matches(..., T=1.0)
          │
[A2: ĐỆ QUY HAI LƯỢT (TWO-PASS RECURSIVE)]
  ├── PASS 1: Căn chỉnh sơ bộ -> trích xuất pair_pages[(chữ, âm)] ≥ 2 trang khác
  └── PASS 1b: Căn chỉnh lại với chi phí min(base, 2.0) cho cặp có neo ngữ liệu
          │
[A3: ĐIỀU PHỐI HỘP THEO CHỈ SỐ ÂM]
  ├── Hạ ngưỡng detector thr 0.3 -> 0.2 trong align_production.py (M==N tăng từ 69% -> 92%)
  ├── Gán hộp[j] ↔ âm j (không dùng thứ tự match sai lầm)
  └── Ghi nhận cột metadata: n_ocr, n_qn, n_det, box_source, flank_gold
          │
[A4: PHÂN TẦNG TIER_V3 NGUYÊN VĂN CÓ NGỮ CẢNH]
  ├── CHAR_A : direct ∧ P ≥ 0.8 (47.482 ô)
  ├── CHAR_B : (direct ∧ P < 0.8) ∨ (sim_unique ∧ (bigram ∨ corpus2 ∨ ~thanh) ∧ P ≥ 0.8) (3.338 ô)
  ├── SYL    : P ≥ 0.5 ∧ (bigram ∨ corpus4 ∨ ~thanh) (19.288 ô - bắt buộc có ngữ cảnh)
  └── REVIEW : Các ô còn lại (12.983 ô)
          │
[A5: BẢNG QUY ƯỚC DECISIONS.YAML & VÁ L1]
  ├── config/decisions.yaml: Gộp dị thể (徳=德, 别=別, 為=爲) + khoá QĐ-01 theo ô
  ├── L1 (apply_am_sua_dau): Chỉ sửa khi 2-gram ngoài ô ủng hộ; luôn giữ cột syllable_raw
  └── Sửa BUG-1 trong step2_align.py: Chặn fallback bóc dòng trả nguyên câu
          │
[A6: CHUỖI BẰNG CHỨNG TỰ ĐỘNG]
  ├── Rebuild ra dataset_out_v3/, đối soát 1-1 với dataset_out/
  ├── Bật cờ --crop-review nếu tắt --use-s3 để không rớt 552 ảnh QĐ-01
  └── Tự động gọi update_bang_so_lieu và evidence() ở cuối step_export
```

---

## 3. CHI TIẾT ĐẶC TẢ KỸ THUẬT TỪNG BƯỚC

### BƯỚC A0: THIẾT LẬP HÀNG RÀO AN TOÀN (SAFETY FENCE)
1. **Bảo vệ dữ liệu gốc:** Thêm biến `OUT_DIR="${OUT_DIR:-dataset_out_v3}"` vào `run_pipeline.sh` và `build_dataset.py` để không ghi đè thư mục `dataset_out/` hiện hành khi đang thử nghiệm.
2. **Khóa cứng QĐ-01:** Chạy script xuất danh sách 2.014 ô người duyệt từ `dataset_out/labels_final.csv`:
   ```bash
   python -c '
   import pandas as pd
   df = pd.read_csv("dataset_out/labels_final.csv", dtype=str)
   qd = df[df["rule"].str.startswith("quyet_dinh_nguoi", na=False)]
   assert len(qd) == 2014, f"Lệch QĐ-01: {len(qd)}"
   qd[["book", "page", "column", "bbox", "image_md5", "label"]].to_csv("config/qd01_cells.csv", index=False)
   print("Đã khoá 2.014 ô QĐ-01 vào config/qd01_cells.csv")
   '
   ```
3. **Bổ sung Selftest (Vá BUG-8):** Thêm $\ge 15$ test assertions vào `pipeline/phase1_engine_selftest.py` kiểm thử các hàm: `realign_column`, `decide_label`, `apply_am_sua_dau`, `_pair_new`.

---

### BƯỚC A1: MA TRẬN CALIB ĐÚNG VÀ FORWARD-BACKWARD ($T=1.0$)
- **Tệp chỉnh sửa:** `pipeline/align_engine/anchor_align.py`
- **Hằng số chi phí chuẩn:**
  ```python
  # Hiệu chuẩn chính xác theo thuc_nghiem.py: CALIB
  COST_CONFIRM   = 0.0   # Khớp trực tiếp S1 ∩ S2
  COST_SIMILAR   = 2.5   # Cầu tự dạng xuôi/ngược
  COST_DICTMISS  = 6.7   # OCR đọc sai chữ nhưng đúng thanh ghi (không phải 5.1!)
  COST_NODICT    = 5.1   # Âm không có trong từ điển (không phải 4.5!)
  COST_DEL       = 8.6   # Xóa chữ Nôm (khe đắt nhất)
  COST_INS       = 8.6   # Chèn âm QN (khe đắt nhất)
  BAND_SLACK     = 2
  ```
- **Hàm `posterior_matches`:** Ghim nhiệt độ $T = 1.0$ (với ma trận CALIB, $T=1.0$ cho phân phối chuẩn xác nhất; $T=0.35$ bị bão hòa $99.8\%$ ở 1.0).

---

### BƯỚC A2: CĂN CHỈNH ĐỆ QUY HAI LƯỢT (TWO-PASS RECURSIVE)
- **Tệp chỉnh sửa:** `pipeline/align_engine/build_dataset.py` (trong hàm `align_page` hoặc loop xử lý trang).
- **Cơ chế:**
  - *Lượt 1:* Căn chỉnh toàn bộ các trang bằng ma trận CALIB.
  - *Thu thập neo:* Thống kê `pair_pages[(chữ, âm)]` đếm số trang khác nhau chứa cặp này.
  - *Lượt 2 (PASS 1b):* Căn chỉnh lại từng cột. Khi tính `substitution_cost(c, s)`:
    $$\text{cost} = \min(\text{cost}_{\text{base}}, 2.0) \quad \text{nếu } (c, s) \text{ xuất hiện ở } \ge 2 \text{ trang KHÁC trang hiện tại (Leave-One-Out)}$$
  *(Kết quả: Tỷ lệ đặt khe đúng chỗ tăng từ 64% lên 85–86%, ghép sai giảm từ 3.14% xuống 1.36%).*

---

### BƯỚC A3: ĐIỀU PHỐI HỘP THEO CHỈ SỐ ÂM QUỐC NGỮ
- **Tệp chỉnh sửa:** `pipeline/align_engine/align_production.py`
- **Hành động:**
  1. Hạ ngưỡng phát hiện ký tự trong CenterNet: `thr = 0.2` (thay vì `0.3` ở dòng 252).
  2. Lọc biên ngang: loại bỏ các hộp ngoài lề cột ($|x - x_{\text{col}}| > 0.10 \times w$).
  3. Khi số hộp detector bằng số âm tiết ($|G| = |Q|$): Gán thẳng **`hộp[j] ↔ âm j`** theo thứ tự từ trên xuống dưới.
  4. Ghi nhận các trường metadata vào `labels.csv`: `n_ocr`, `n_qn`, `n_det`, `box_source` (`detector` | `midpoint`), `flank_gold`. Không dùng các trường này để hạ tier, chỉ dùng làm cờ kiểm soát.

---

### BƯỚC A4: PHÂN TẦNG TIER_V3 NGUYÊN VĂN CÓ NGỮ CẢNH
- **Tệp chỉnh sửa:** `pipeline/align_engine/consensus.py` và `build_dataset.py` (chuyển sang PASS 1b/1c sau khi đã có thống kê ngữ cảnh toàn sách).
- **Quy tắc nguyên văn (đã kiểm chứng sinh ra 70.108 ô):**
  ```python
  def tier_v3_exact(f, p):
      # f chứa các cờ: direct, sim_unique, bigram, corpus2, corpus4, tone
      if f["direct"] and p >= 0.8:
          return "CHAR_A"
      if (f["direct"] and p < 0.8) or (f["sim_unique"] and (f["bigram"] or f["corpus2"] or f["tone"]) and p >= 0.8):
          return "CHAR_B"
      if p >= 0.5 and (f["bigram"] or f["corpus4"] or f["tone"]):
          return "SYL"  # BẮT BUỘC CÓ NGỮ CẢNH -> giữ SYL đúng ở 19.288 ô, không bị vỡ đê lên 31k
      return "REVIEW"
  ```
- **Ánh xạ về `USABLE_TIERS` hiện hành:** Giữ nguyên tên gọi để không làm gãy 11 module hạ nguồn:
  - `CHAR_A` và `CHAR_B` $\rightarrow$ ghi `tier = "GOLD"` (kèm rule tương ứng).
  - `SYL` $\rightarrow$ ghi `tier = "SYLLABLE"`.
  - Cột `tier_v3` được ghi riêng vào `labels.csv` để đối soát.

---

### BƯỚC A5: BẢNG QUY ƯỚC DECISIONS.YAML & VÁ LỖI L1
1. **Tạo `config/decisions.yaml`:**
   - Hợp nhất dị thể: `徳 -> 德`, `别 -> 別`, `爲 -> 為`, `廪 -> 廩`.
   - Lớp nhầm hệ thống: `(cùng, 其) -> 共` (theo phát hiện E3: 158 ô cùng hình).
   - Khóa ô QĐ-01: Tích hợp đọc `config/qd01_cells.csv`. Bất kỳ ô nào nằm trong danh sách này tự động nhận nhãn `𠊚 (U+2029A)` và `tier = "GOLD"`, bảo toàn nguyên vẹn 2.014 ô.
2. **Vá lỗi L1 (`apply_am_sua_dau`):**
   - Thêm cột `syllable_raw` giữ 100% nguyên vẹn âm gốc của bản in.
   - Chỉ cho phép L1 đổi âm khi có ngữ cảnh 2-gram ngoài ô ủng hộ âm mới.
   - Với 161 trường hợp dị âm chép tay (như `僥 -> nhiều`, `無 -> vồ`), đưa vào danh mục `corpus_readings` có khai xuất xứ thay vì để L1 sửa ép âm.
3. **Sửa BUG-1 trong `pipeline/step2_align.py`:** Nối qua `parse_v5._split_syllables` ở cả 2 lệnh `return` (dòng 84 và 88).

---

### BƯỚC A6: ĐỒNG BỘ CHUỖI BẰNG CHỨNG VÀ KIỂM ĐỊNH TỰ ĐỘNG
1. Chạy build thử nghiệm sang `dataset_out_v3/`.
2. Nếu tắt `--use-s3` trong `run_pipeline.sh`, bắt buộc phải truyền cờ `--crop-review` để các crop của QĐ-01 vẫn được cắt đầy đủ ra đĩa.
3. Nối lệnh `python -m pipeline.tools.update_bang_so_lieu` và hàm `evidence()` vào cuối `run_pipeline.sh`.
4. Chạy `bash scripts/check_consistency.sh` đảm bảo kết quả đạt **4/4 KHỚP**.

---

## 4. CÁC HẠNG MỤC ĐÃ LOẠI BỎ (TRÁNH LÀM HỎNG DỮ LIỆU)
Theo khuyến nghị của tài liệu đánh giá ngày 15-09, các hạng mục sau **TUYỆT ĐỐI KHÔNG THỰC HIỆN** trong Khối A:
- ❌ **Không dùng khối tâm mực (COM)** ở Bước 2.2 (mù với độ lệch, phá hỏng md5 crop).
- ❌ **Không dùng Pitch DP bổ đôi khi > 1.6 pitch** ở Bước 2.1 (làm 1.443 cột rơi về midpoint).
- ❌ **Không dùng `resolve_overlap`** ở Bước 4.2 (đã đo làm giảm tỷ lệ cờ ok).
- ❌ **Không can thiệp vào mô hình OCR Nôm** (giữ nguyên cache Kimhannom làm Primary Data).
- ❌ **Không đổi tên tier toàn cục** làm gãy các hằng số `USABLE_TIERS` ở 11 file hệ thống.
