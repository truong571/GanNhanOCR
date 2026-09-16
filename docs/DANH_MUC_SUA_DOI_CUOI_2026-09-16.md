# DANH MỤC SỬA ĐỔI CUỐI — sau khi thẩm định "tư vấn MI3" và hai ràng buộc mới

**2026-09-16** · Đầu vào: (1) bản tư vấn ngoài "Cải tiến & khai thác công nghệ tự động (MI3)" gồm 2 tiền đề + 4
cải tiến + 1 lộ trình; (2) hai ràng buộc của người dùng: **không chia train/val/test**, **labels.csv chỉ giữ cột chính**;
(3) toàn bộ nghiên cứu 13–15/09 (`RA_SOAT_*`, `HUONG_TAN_CONG_*`, `DANH_GIA_KE_HOACH_*`, `FLOW_PIPELINE_v3.1_*`).
Cách thẩm định: 8 lượt độc lập (mỗi khẳng định của tư vấn một lượt + 2 lượt rà cột/split), 42 phát hiện chặn/lớn qua
84 phiếu bác bỏ thử → **0 bị bác**; mọi số đo lại được, script sao lưu ở `lab/tham_dinh_2026-09-16/` (20 tệp).

---

## 0. TƯ VẤN NÓI GÌ, ĐO ĐƯỢC GÌ

| Tư vấn | Kết luận | Số |
|---|---|---|
| "MI3 = Milestone 3 / mốc phản biện lần 3" | **sai** | MI3 là nhãn của **bản góp ý PDF** nhận 22/08 ("Tối Ưu Pipeline GanNhanOCR (MI3)", `VIEC_CAN_LAM.md:8`: 199 khẳng định, 21 bác). Bản ấy không nêu "không chấm tay"; ngược lại nó dùng error-AUC tính trên verdict máy như GT |
| "Chấm tay bất khả thi, phá khách quan; hệ thống tự chứng minh bằng SHA256 + posterior" | **sai cả 3 vế** | QĐ-01 = 1 người nhìn 2.063 crop trong ≤31 phút; `pipeline/ground_truth` (171/171 test) dựng sẵn mẻ 600 ô ≈ 3–4 buổi. SHA256 chỉ chứng minh *đồng nhất tệp*. Posterior bão hoà (p≥0,8 ở 98,93% cặp, quyết tier 119 ô) và đo *ghép âm*, không đo *mã chữ*: từ điển có "người,㝵" nên máy xếp 1.178 ô 㝵 vào CHAR_A — đúng lớp mà phán quyết người duy nhất nói là 𠊚; bộ kiểm "độc lập" chấm 2.051/2.051 TRUE cho cả 㝵 lẫn 𠊚 |
| Cải tiến 1 · CNN âm tiết làm phát xạ trong DP (E2a 72%, AUC 0,946) | **đúng hướng, có điều kiện** | E2a/E2b tái lập (trên 89 cột/1.829 ô, ảnh thuần). `emb.npz` hiện **rò**: ô đã học top-1 97,4% vs test 77,5% → phải out-of-fold (5 fold ≈ 88 phút MPS). Đo mới 523 cột/10.612 cặp: CALIB+corpus 0,73% → +ảnh λ=0,25: **0,25%** sai, khe đúng 86,6→95,4%; λ≥1 **hại**; trên hộp OCR thật lợi ≤≈50 ô |
| Cải tiến 2 · gom cụm embedding tự gán `label_canonical` "không cần người" | **sai** | "63 lớp độ thuần >98%" không có ở đâu: k-means trên Z cho độ thuần 0,756 = tỉ lệ nhãn đa số, 0/63 ≥0,98; DBSCAN eps ≤0,25 → 100% nhiễu; 63/63 lớp cả hai mã đều ∈ R(âm); 8 cặp đa số đảo chiều giữa sách → chọn mã là **quy ước người ký**. E4 chạy lại: lan nhãn 43,2% vs tần suất 42,7% |
| Cải tiến 3 · tách từ dính "+200–300 ô GOLD" | **sai 10–20×** | 17 cột tách duy nhất = 321 ô, 214 đã usable; tách thêm 21 cặp → usable **+21–23** (GOLD +12); trần 36 cột ≈ +40 ô (0,06%); 4/17 nghiệm "duy nhất" cắt sai ngôn ngữ (truy+en, A+ra+bia, Conhiều→co+nhi+ề+u tạo direct giả). 0 cột nào nhờ tách mà thêm được hộp detector |
| Cải tiến 4 · hoà giải 3 nguồn đếm (98% / >95% / >92%) | **đã có trong flow** (N4d) | Tư vấn hẹp hơn flow (517 vs 3.704 cột). Số đúng: nhãn hai chiều **96–97%** (DEL 96,3–96,7% chấm chặt; INS 95,9–97,1%); hộp detector **87,7%** với khoá trọn cột QĐ-01 (42,4% ô usable nằm trong cột khoá) hoặc ≈96% nếu khoá ô |
| "Giai đoạn 2 thăng cấp tự động 1.500–2.500 ô" | **không có nguồn** | grep docs + git log -S = 0. Đo out-of-fold trên 1.355 ô REVIEW test: argmax CNN == âm ghép 43,8%; ∧P≥0,9 chỉ 7,6% → ngoại suy **≈960 ô** [780–1.140] **chỉ nhãn âm**; mã chữ tự động = 0/1.109 |

**Nguyên tắc chốt (thay mọi câu "tự động 100%, không chấm tay"):** *Gán nhãn 100% tự động; người chỉ ký quyết định
theo lớp (QĐ-nn, bảng quy ước, tên cụm). Đánh giá KHÔNG tự động: precision công bố = số đo trên mẻ ~600 ô rút mù có
người chấm, kèm CI Wilson 95% và manifest SHA256 "bộ đem đo = bộ đem nộp". SHA256 chứng minh đồng nhất tệp, posterior
chỉ phân xử ~1–2% cặp hoà; cả hai không chứng minh nhãn đúng.*

---

## PHƯƠNG ÁN CUỐI (chốt 16/09)

**Bài toán:** gán cho mỗi ô chữ Nôm viết tay một mã Unicode + một âm Quốc ngữ, tự động, từ bản dịch song hành 9 cột ↔ 9 dòng.

**Nguyên tắc:** gán nhãn 100% máy; người chỉ ký quyết định theo lớp (QĐ-01, bảng dị thể, tên cụm); precision công bố
đo bằng mẻ ~600 ô rút mù có người chấm (Wilson 95%); SHA256 = chuỗi bảo quản, posterior = phân xử cặp hoà — không
phải bằng chứng nhãn đúng.

**Đường chạy (một lệnh, tất định, S3 tắt):** extract (giữ) → PASS 1: DP văn bản ma trận CALIB + hộp thô detector thr 0,2
→ PASS 1b: đệ quy neo cặp lặp ≥2 trang khác (LOO) + posterior cùng chi phí + hộp 3 nhánh → PASS 1c: tier_v3 có ngữ cảnh
(thay decide_label/L2/L3/L5), L1 có cổng 2-gram, `decisions.yaml` → `label_canonical`, khoá QĐ-01 theo `nom_idx` trong
build → PASS 2 crop (ô khoá giữ nguyên byte) → remediate → confusion (0) → glyph_fix kiểm → export **12 cột, không
split** → đối soát thế hệ → QĐ-01a (≤22 ô) → thăng cấp + evidence → 3/3.

**Kết quả kỳ vọng (đã đo trên benchmark có đáp án):** khe giả 346→54 cột; ghép sai khi rụng chữ 3,85→1,3–1,4%; hộp đúng
glyph 59→96–97% trên cột đếm khớp; usable 64.525→70.100–70.450, ~1.300 GOLD yếu lộ ra; QĐ-01 2.014 giữ nguyên; âm bản in
không bị sửa đè. Precision mã chữ: **chưa có số** cho tới Khối C.

**Sau đó (có điều kiện):** CNN âm tiết out-of-fold → phát xạ ảnh λ=0,25 (≤50 ô) và cổng REVIEW→SYL P≥0,9 (≈960 ô âm)
chỉ sau khi Khối C xác nhận; H2-lite ≈60 quyết định lớp người ký. **Không làm:** gom cụm tự gán, DBSCAN, COM, Pitch-DP,
resolve_overlap, tách từ dính (Khối D), mọi kỳ vọng "1.500–2.500 ô".

---

## 1. DANH MỤC SỬA ĐỔI CUỐI

### 1A · KHỐI A — làm ngay, theo thứ tự (≈8 ngày; nút = `FLOW_PIPELINE_v3.1` §3)

| # | Hạng mục | Nguồn | Tệp/hàm | Kỳ vọng đo được |
|---|---|---|---|---|
| A-1 | **Hàng rào + tái lập HEAD** trước khi đổi bất kỳ hằng số: `DS_OUT`, `NONINTERACTIVE`, phát `nom_idx/syl_idx`, so khoá với bộ 64.525 | flow N0a–N0c | `run_pipeline.sh:52,55-57,365…`; `align_production.py:458-461` | 0 lệch; QĐ-01 2.014 |
| A-2 | **Khoá QĐ-01 bền** `qd01_cells.csv` theo `(book,page,column,nom_idx)` + `bbox/prev/next_cu`; áp **trong build, PASS 1c**, ô trôi → `qd01a_decisions.csv` | flow N0d, N5g | `build_dataset.py`, `glyph_fix.py --mode kiem` | locked + pending = 2.014; bbox_cu 2.014/2.014 |
| A-3 | Gộp vào QĐ-01a: **12 ô GOLD 'người' mang mã ≠ 𠊚** (昆 8, 辞 2, 匕 1, 命 1) — ngoài phạm vi 2.014, bộ kiểm 12/12 FALSE | mới (C0) | `qd01a_decisions.csv` | 0 ô GOLD 'người' không có phán quyết người |
| A-4 | Ma trận **CALIB** + `posterior_matches(cost_fn)` port vào engine, T=1,0; `thuc_nghiem.py:76` đọc hằng số engine | flow N3d–N3e | `anchor_align.py:33-39,:81,:127` | khe giả 346→54 |
| A-5 | **Đệ quy hai lượt** LOO, `pair_pages` từ **mọi** match, cap 2,0; posterior cùng `cost_fn` | flow N4a–N4c | `build_dataset.py` sau `:447` | khe đúng 64→85–86% |
| A-6 | **Hộp 3 nhánh** + thr 0,2 + ±0,25w; cột có QĐ-01 chạy luật cũ trọn gói; `--box-rule legacy` trọn gói | flow N3f, N4d | `align_production.py:184,:252,:387-421` | hộp[j] 96–97% trên \|G\|=\|Q\|; detector hiệu dụng 87,7% |
| A-7 | **tier_v3 nguyên văn** (feats LOO) thay `decide_label` + L2 + L3 + L5; chốt `is_plausible`; L1 có cổng 2-gram ÂM–ÂM từ `col_states`; 'người' ngoài khoá chỉ hạ khi nhãn v3 == 㝵 | flow N5a–N5h | `tier_v3.py` (mới), `build_dataset.py:106-160` | usable 70.100–70.450; L1 ≈376/245/134 |
| A-8 | `decisions.yaml`: `corpus_readings` + dị thể → cột **`label_canonical`** (bảng 15 dòng, người ký **chiều** chuẩn); `(cùng,其)` chờ người ký | flow N5f, N5i | `pipeline/decisions.py` (mới) | ≈640–990 ô có `label_canonical` |
| A-9 | **Schema giao nộp 12 cột** (§2) + `labels_trace.csv` + `columns.csv`; `dataset_out/labels_final.csv` giữ nguyên mọi cột cho nội bộ | ràng buộc (b) | `export_final_dataset.py:31`; N7b | labels.csv 8,4 MB thay 14,2 MB |
| A-10 | **Bỏ split** — giai đoạn 1 (lớp giao nộp, làm ngay) + giai đoạn 2 (build/remediate/proto-index, cùng lúc tắt S3) (§3) | ràng buộc (a) | 17 vị trí mã §3 | README ghi công thức hash cũ để ai cần tự chia theo trang |
| A-11 | Census thêm luật **cùng `image_md5` trong cùng cột, khác bbox** (cặp 法/冉 lọt hôm nay; `census.py:60-72` chỉ bắt AE-1/F1) | mới (D1) | `remediation/census.py` | 0 nhóm md5 trùng trong bộ giao nộp |
| A-12 | Hậu xử lý: `remediate` giữ (bỏ vế rò split), `confusion_fix` giữ (kỳ vọng 0), `s3_unwind` giữ no-op, `glyph_fix --mode kiem`, `assert_qd01` sau mỗi bước | flow N9–N13 | `run_pipeline.sh:395-481` | |
| A-13 | Đối soát thế hệ theo `nom_idx` → QĐ-01a (người, ≤10 ô + 12 ô A-3) → **thăng cấp** `DS_OUT=dataset_out`, rm `s3_proto_cache`, `update_bang_so_lieu` (HEADER sha16), `evidence()`, 1 commit → `check_consistency` **3/3** (proto-index bỏ) | flow N15–N19 | `pipeline/tools/doi_soat_the_he.py` (mới) | chuỗi bằng chứng liền; thời gian thật |
| A-14 | **Bảng nghiệm thu `FINAL §1` thêm dòng precision người** (Wilson 95% theo tầng) và **manifest SHA256** nối mẻ chấm với `labels_final.csv`; bỏ dòng "GOLD ≥98% bằng bộ kiểm độc lập" | mới (C0) | `KE_HOACH_..._FINAL.md`; `audit_grid.py:44-64` | "bộ đem đo = bộ đem nộp" kiểm bằng máy |
| A-15 | Selftest cho `realign_column`/`enforce_count` (ngày 1), thêm theo ngày làm nút; `BASELINE_PASS` cập nhật cuối; **script đo** của 3 đợt thẩm định đã sao lưu `lab/tham_dinh_2026-09-16/` | flow N0e; mới | `phase1_engine_selftest.py`, `run_all_selftests.sh:80` | 722 + N pass |

### 1B · KHỐI B — làm có điều kiện (kênh ảnh, chỉ đo; sau Khối A)

| # | Hạng mục | Điều kiện | Kỳ vọng |
|---|---|---|---|
| B-1 | CNN âm tiết **out-of-fold** 5 fold theo `hash(book,page)`, 706 lớp cố định → sidecar `p_visual_oof.csv` (`book,page,column,nom_idx,fold,p_visual,argmax_syl`); **không** thêm cột vào `labels.csv` | chạy nền ≈88 phút MPS; **không dùng** `emb.npz/model.pt` hiện tại cho 49.771 ô đã học | E2b out-of-fold trên trang train ≈ test (AUC ≈0,95) |
| B-2 | **Phát xạ ảnh trong DP** (H1): `cost_fn = text + λ·min(−LP, 12)`, CALIB, **λ=0,25**, khe ảnh 8, trung tính 0,5 cho âm ngoài 706 lớp | sau A-4/A-5 và B-1; gác bằng benchmark 523 cột (λ≥1 hại) | khe đúng 86,6→95,4% mô phỏng; trên hộp thật ≤≈50 ô đổi; không thăng tier |
| B-3 | **Cổng thị giác REVIEW→SYL** (chỉ nhãn âm): `argmax == âm ghép ∧ P≥0,9`, rule `visual_syl_gate` | **sau Khối C**: tầng 100–150 ô "qua cổng" trong mẻ 600, 0/150 lỗi (cận trên 2%) | ≈894–961 ô SYL (+1,4% usable); mã chữ tự động = 0 |
| B-4 | **H2-lite**: giao diện 8–12 crop/lớp cho ≈60 quyết định lớp (51 lớp nhầm khác chữ/1.159 ô + 9 mờ/202 ô); 38 lớp hai hình giữ nguyên; kết quả vào `decisions.yaml` có `xuat_xu` | 1–2 buổi người; k-means có hạt giống GOLD chỉ để **sắp** crop, không để quyết | precision mã GOLD 93–96% → sát 98% trên 12.285 ô thuộc 112 lớp (đo lại bằng Khối C) |
| B-5 | **Khoá ô thay khoá trọn cột** cho QĐ-01 (`bbox/prev/next_cu` từng ô + `legacy_locked_col` chỉ 22 cột) | chạy A-6 thật: 23 cặp trùng bbox → **0** và md5 QĐ-01 2.014/2.014; nếu không → giữ trọn cột | hộp detector 87,7% → ≈96% (61.989/64.525) |

### 1C · KHỐI C — bắt buộc để có số công bố

| # | Hạng mục | Công |
|---|---|---|
| C-1 | Chấm thử 50 ô đo dwell/κ (`VIEC_CAN_LAM` 6.5; `audit_grid` hiện ghi 0 dòng dwell) | 0,5 ngày |
| C-2 | **Mẻ ~600 ô rút mù** từ `labels_final.csv` có sha256, phân tầng ≤8 (tier_v3 × lớp cột × box_source), ô mồi ~10%, lặp ẩn ~8%, hai câu tách (crop đúng glyph? mã đúng?), Wilson 95% theo tầng; thêm tầng "qua cổng thị giác" 100–150 ô cho B-3 | 3–4 buổi ≤45 phút |
| C-3 | Ghi kết quả vào `BANG_SO_LIEU` qua `evidence()`; nếu GOLD thật 93–96% thì CI [92,9; 96,5] là **số trung thực**, không phải lý do bỏ chấm | |

### 1D · KHÔNG LÀM (và không trích số của tư vấn)

| # | Hạng mục | Vì sao (số) |
|---|---|---|
| D-1 | Gom cụm rồi **tự gán** `label_canonical` cho cả cụm "không cần người" | độ thuần 0,756 = base; 0/63 ≥0,98; 8 cặp đảo chiều giữa sách; E4 không hơn tần suất |
| D-2 | DBSCAN / ngưỡng cosine trên Z; cột `glyph_cluster_id` trong `labels.csv` | eps ≤0,25: 100% nhiễu; cụm không tái lập (ARI 0–0,56) |
| D-3 | Dùng `emb.npz/model.pt` hiện tại cho ô trang train | vòng tròn 49.771 ô (60,1%) |
| D-4 | Ghi kỳ vọng "thăng cấp 1.500–2.500 ô" ở bất kỳ tài liệu nào | không nguồn; đo được ≈960 ô SYL-only có điều kiện |
| D-5 | Trích "98% / >95% / >92%" | thay bằng 96–97% nhãn hai chiều; hộp detector 87,7% (khoá cột) / ≈96% (khoá ô) |
| D-6 | Chỉ tiêu "GOLD ≥98% bằng bộ kiểm độc lập / posterior" | bộ kiểm mù mã chữ (㝵 và 𠊚 đều TRUE); `p_register` chỉ để phân tầng mẫu |
| D-7 | Tách từ dính trong Khối A | +21–23 ô (trần ≈40), 4/17 nghiệm sai ngôn ngữ → Khối D "sau", ~28 dòng sau `normalize_column`, cờ `split_token` |
| D-8 | Sửa `pipeline/publish` (splits.py) | tự sinh split riêng, ngoài flow, không đọc cột `labels.csv`; chỉ thêm 1 dòng README |
| D-9 | COM, Pitch-DP bổ đôi, resolve_overlap, đổi tên tier, `--crop-review` (đã bác ở 15/09) | |

---

## 2. SCHEMA BỘ GIAO NỘP — 12 cột cố định (whitelist)

| Cột | Giữ vì |
|---|---|
| `image` | khoá chính (64.525 phân biệt), đường crop |
| `book`, `page`, `column` | khoá bền liên thế hệ; `page` = đơn vị chia tách nếu người dùng cần |
| `ocr_char` | chữ OCR S1 — bằng chứng trực tiếp (0 rỗng) |
| `syllable` | âm QN lower/NFC — nhãn tầng SYLLABLE, khoá lớp (sách, âm) |
| `label` | mã Nôm (rỗng 10.369 ô SYLLABLE) |
| `unicode` | dẫn xuất nhưng người dùng cần (Excel không hiện Ext-B), `hannom_recheck.py:270` dùng |
| `tier` | GOLD / SYLLABLE |
| `rule` | xuất xứ (QĐ-01 = `quyet_dinh_nguoi:*`); 7 chỗ mã so chuỗi |
| `bbox` | toạ độ trên `pages/<page>.png` — không tái lập từ crop |
| `image_md5` | toàn vẹn khi sao chép; bắt trùng crop không cần đọc ảnh (cặp 法/冉 phát hiện nhờ nó) |

Cột 13 tuỳ chọn khi bảng dị thể đã ký: `label_canonical`. Bộ kiểm của người dùng (`hannom_recheck.py:37-39` KEEP)
đang chọn đúng 10 cột đầu — bỏ `bbox/image_md5` cũng được nếu muốn 10.

**Sidecar** (cùng số dòng/thứ tự, khoá `image` + `book,page,column,nom_idx`):
- `labels_trace.csv`: `nom_idx, syl_idx, syllable_ocr, syllable_raw, tier_v3, tier_goc, rule_goc, p_register, dict_support,
  context_evidence, l1_support, l1_tie, box_source, qd01_locked, qd01_excluded, crop_quality_flag, stray_ink, border_ink,
  ink_pct, crop_w, crop_h, seg_flag, s3_cosine` (mọi cột cờ ghi dày 0/1).
- `columns.csv` (4.030 dòng, khoá `book,page,column`): `n_ocr, n_qn, n_det, count_source`.
- `lab/.../p_visual_oof.csv` (Khối B).
- **Bỏ hẳn** (không sidecar): `label_level` (= f(tier) 64.525/64.525), `readmitted_from_s3_demotion` (rỗng 82.780/82.780),
  `seg_backend` (1 giá trị → `summary.json`), `usable_image` (= f(crop_quality_flag)), `page_cot_lech` (3 trang → DATASHEET),
  `split`, `split_group`, `label_in_train`.

**Mã gãy khi cắt cột (phải sửa cùng lúc):** `update_bang_so_lieu.py:142` (`r["split"]` KeyError), `re-dataset/check/hannom_recheck.py:138`
(`label_level`), `tools/selftest.py:309-314, :415-440, :502-504`, `remediation/selftest.py:354-359`, `export_review_excel.py:39`
(ra 0 dòng), `make_dataset_docs.py:54` (`n_leak=0` → README nói sai "rời nhau theo trang"). Các mô-đun `ground_truth/`,
`remediation/`, `publish/`, `lab/` đọc `dataset_out/labels*.csv` (giữ đủ cột) → không gãy.

---

## 3. BỎ CHIA train/val/test — hai giai đoạn

Ba cột là hàm thuần của cột đã có: `split_group == book|page` (64.525/64.525), `split == int(md5(book|page),16)%100`
(64.525/64.525), `label_in_train` = lớp có trong train. Không mất thông tin. S3 tắt nên proto-index `GOLD∧train` không
cần; CNN out-of-fold tự chia theo trang trong lab; mẻ chấm phân tầng theo tier×sách, không dùng split.

**Giai đoạn 1 — lớp giao nộp (≈2 giờ, 0 ô đổi):** `export_final_dataset.py:31` whitelist 12 cột, `:103-119` xoá tính
lại `label_in_train`; `update_bang_so_lieu.py:142,:153`; `make_dataset_docs.py:52-56,:63,:76,:89-117` → mục "Không chia
train/val/test; chia theo trang bằng `book+page`, công thức hash cũ: `int(md5(f'{book}|{page}').hexdigest(),16)%100` <80
train / <90 val / còn lại test (áp dụng tới 25/08)"; `tools/selftest.py:267-275,:283-285,:297-314,:319,:502-503`;
`make_xlsx.py:6,:30-32` (dọn); `run_all_selftests.sh:80` mốc mới; `FLOW` N6 [GIỮ]→[BỎ], N7b 22→19 cột, N9 bỏ vế rò split.

**Giai đoạn 2 — nội bộ, cùng lúc tắt `--use-s3` (`run_pipeline.sh:365`):** `build_dataset.py:518-550` xoá khối SPLIT
(+`:597,:602,:609-610,:628,:644`); `remediate.py:78` bỏ `split` khỏi tuple bắt buộc (nếu không bước 4 chết vì `set -e`),
`:86-90,:139,:152-159` bọc `if 'split' in out.columns`; `remediation/selftest.py:179-185,:403,:411-413`;
`rebuild_proto_index.py` (`:97` lọc train → 0 nguyên mẫu **âm thầm** nếu S3 còn bật) xoá cùng `data/index.csv` + `rm
s3_proto_cache.pkl`; `visual_signal.py:206`; `check_consistency.sh:34-35` (proto-index) → 3 phép; `thi_giac_am_tiet.py:139,
:153,:218,:269,:396` → `load_all()` tự sinh `df['split']` bằng cùng hash (5 dòng; E1/E2 tái lập nguyên xi);
`audit_grid.py:48` bỏ `split` khỏi `_HIDDEN_FIELDS` (tuỳ chọn). **Thứ tự bắt buộc:** giai đoạn 2 chỉ sau khi S3 tắt —
bỏ split ở build mà S3 còn bật thì crop-protos = 0 âm thầm, SILVER tụt ~32%.

---

## 4. BẢNG NGHIỆM THU CẬP NHẬT (thay §1 của `FINAL`)

| Chỉ tiêu | Hiện | Kỳ vọng | Đo bằng |
|---|---|---|---|
| Tái lập HEAD | chưa | 0 lệch, QĐ-01 2.014 | `doi_soat_the_he.py` |
| Khe giả / lệch chéo | 346 / 1,63% | 54 / 0,82% | `thuc_nghiem_ke_hoach.py matrix` + đếm trên build |
| Khe đúng / ghép sai rụng 1 chữ (benchmark) | 60% / 3,85% | 85–86% / 1,3–1,4% (văn bản); 95% / 0,25% nếu B-2 | `thuc_nghiem.py recursive` |
| Ô dùng được | 64.525 | 70.100–70.450 (+≈900 SYL nếu B-3 qua Khối C) | crosstab |
| QĐ-01 | 2.014 | locked + pending = 2.014 (+12 ô A-3 có phán quyết); bbox_cu 2.014/2.014 | assert |
| Hộp detector thật | ~74% | 87,7% (khoá cột) / ≈96% (B-5) | `summary.json` |
| Nhãn hộp hai chiều | 59,1% | 96–97% | `geo` tham số hoá (INS + DEL) |
| L1 | 755 đổi | ≈376 đổi / 245 giữ / 134 hoà; ghi đè bản in 0 (trên `syllable_ocr`) | crosstab |
| `labels.csv` giao nộp | 30 cột | **12 cột**, không split | `export_final_dataset` |
| md5 crop đổi | — | ≤14.117, liệt kê; QĐ-01 0 | N15 B5 |
| **Precision người** | **0 phán quyết** | GOLD mã chữ, ghép âm, SYL — Wilson 95% theo tầng, n≈600 | Khối C |
| Chuỗi bằng chứng | 2/4 | 3/3 sau thăng cấp | `check_consistency.sh` |

---

## 5. CÒN THIẾU — phải làm/đo trước khi công bố

1. Chưa có số precision nào do người đo; `FINAL §1` chưa có dòng precision người (grep `precision|Wilson|600 ô` = 0).
2. Cặp md5 trùng `aa18c3e60447` (法/冉, cùng cột, bbox lệch 60 px, cả hai GOLD direct) — chưa ai nhìn.
3. 218 ô 'người' REVIEW không có crop → không thể quyết; ghi rõ ngoài phạm vi.
4. CNN out-of-fold chưa chạy (88 phút) — mọi số `p_visual` hiện là in-sample với 60% ô.
5. Khoá ô (B-5): số 61.989/64.525 là ước từ mô phỏng, chưa chạy A-6 thật để đếm 23 cặp trùng bbox.
6. Bảng dị thể 15 dòng chưa có chiều chuẩn người ký (yaml cũ gộp 4 cặp về mã hiếm).
7. INS benchmark chiều OCR<Q (95,9–97,1%) chỉ ở `lab/tham_dinh_2026-09-16/c43_ins.py`, chưa vào cột "Kiểm" của N4d.
8. `BANG_SO_LIEU` lệch từ 09/09 → `check_consistency` đỏ độc lập với việc bỏ split; chỉ xanh sau A-13.
