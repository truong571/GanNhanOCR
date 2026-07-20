# THỨ TỰ THỰC HIỆN TỔNG THỂ — HOÀN THIỆN LUẬN VĂN

**Hợp nhất từ**: `DE_XUAT_HOAN_THIEN_LUAN_VAN_2026-07-20.md` (4 blocker + kế hoạch 12 tuần) và `FLOW_TONG_THE_CHOT_2026-07-14.md` (§4 việc còn lại P1.2–P2, §5 cấu trúc 8 bước) + các phát hiện đo được ngày 2026-07-20.

**Lập ngày**: 2026-07-20 · **Quỹ thời gian**: ~2–3 tháng tới bảo vệ

---

## NGUYÊN TẮC XUYÊN SUỐT

1. **Bằng chứng trước, mã sau.** Repo đã mất `FLOW_TONG_THE_CHOT` một lần (phục hồi được nhờ transcript, không nhờ git). Còn ~30 file bằng chứng đang untracked.
2. **Đo một lần, không đo hai lần.** Mọi thay đổi ảnh hưởng mẫu số (thêm crop, demote, rebuild) phải gộp vào MỘT mốc đóng băng, rồi mới rút mẫu audit.
3. **Mọi claim precision neo vào audit NGƯỜI.** Không neo vào S3 (AUC bắt-lỗi 0.566), không neo vào verdict AI.
4. **Không tái cấu trúc lớn.** Đã phân tích 42 đề xuất gộp file → 39 bị bác. Repo giữ nguyên 125 file .py.
5. **Không thêm sách mới trước bảo vệ.** Blocker nằm ở chất lượng, không phải khối lượng.

---

## GIAI ĐOẠN 0 — CỨU BẰNG CHỨNG (hôm nay, 2–4 giờ) 🔴 CHẶN TẤT CẢ

> Đây là việc duy nhất có rủi ro mất trắng thật sự, và đã xảy ra một lần.

- [x] Sao lưu **lạnh** ra ngoài repo TRƯỚC mọi lệnh git (`5 gói`, đã verify + bung/clone thử):
  `tar czf ~/ThS_archive/evidence_2026-07-20.tgz dataset_out/ground_truth dataset_out/fusion dataset_out/labels_final.csv config/confusion_fixes.yaml` + ghi sha256 + bung thử vào `/tmp`
- [x] Commit *(bằng chứng đã nằm sẵn ở `f78dbc4da5`; 4 nhóm được liệt kê trong `EVIDENCE_INDEX` §2-bis vì không rebase được)* (để log kể được câu chuyện, dễ trích commit-hash vào luận văn):
  (a) 7 file .py Giai đoạn 0–2 · (b) `config/confusion_fixes.yaml` · (c) toàn bộ `dataset_out/ground_truth/**` + `labels_final.csv` + `confusion_fix_report.json` + `dataset_out/fusion/**` · (d) tài liệu `.md` + `.html`
- [x] `git tag freeze-pre-thesis-2026-07-20` + push branch kèm tag *(thêm 2 tag: `freeze-features-2026-07-20`, `state-post-phase2-2026-07-21`)*
- [x] Viết `docs/EVIDENCE_INDEX.md`: mỗi artifact → commit-hash + sha256 + lệnh sinh ra nó. **Ghi thẳng dòng**: *"verdicts_001–006.jsonl gốc ĐÃ MẤT, chỉ còn dẫn xuất verdicts_reanchored.csv"* và *"FLOW_TONG_THE_CHOT là bản phục dựng từ transcript"*.

**Xong khi**: `git status --porcelain | grep -c '^??'` = 0 · tag có trên remote · file backup bung thử được, khớp sha256.

---

## GIAI ĐOẠN 1 — TÁI LẬP TỐI THIỂU (0,5–1 ngày)

> Đóng 2 lỗ hổng tái lập thật, không chạm cấu trúc thư mục. Đây là câu Q&A số 7 của hội đồng.

- [x] Xử lý `gannhanocr-fd` đang dirty → `ignore = dirty` (ảo giác bộ lọc LFS, dữ liệu nguyên vẹn) trước (kho 89.898 glyph, phụ thuộc BẮT BUỘC của SILVER)
- [x] Khai đủ 4 gitlink vào `.gitmodules` (hiện chỉ có `font_diffusion`; `git submodule status` đang FATAL)
- [x] Bổ sung `pandas`+`pyarrow` + sinh `requirements.lock.txt` *(và vá dòng `vietocr` khiến lock không cài được)*
- [x] Kiểm chứng bằng **clone sạch** vào `/tmp` rồi chạy 1 selftest trong đó
- [x] Merge branch vào `main` → **TUYÊN BỐ CODE-FREEZE TÍNH NĂNG**

**Xong khi**: `git submodule status` exit 0, in 4 dòng, không dòng nào có tiền tố `-` hay `+` · clone sạch chạy được selftest.

---

## GIAI ĐOẠN 2 — DỌN RÁC AN TOÀN (0,5–1 ngày, có thể bỏ nếu gấp)

> Giữ nguyên **100%** đường dẫn code. Mọi thứ bị xoá/dời đều đã gitignore hoặc 0 tham chiếu.

- [x] ~~Xoá `__pycache__` (17 thư mục), `.DS_Store` (8), `.vscode/`~~ — **ĐÃ LÀM 2026-07-20**
- [x] ~~5 việc Nhóm A: vá bug `--out` đè `fused.csv`; xoá 3 hàm chết `core/image/`; xoá `ranker.generate_fontdiffusion_image`; xoá `run_full.load_similar`; thêm `scripts/run_all_selftests.sh`~~ — **ĐÃ LÀM, −160 LOC, selftest giữ nguyên 212/11**
- [x] Chuyển ra kho ngoài (−6,1 GB; sha256 ở `docs/data_manifest/`): `kkanji2/` (549 MB, 0 tham chiếu), `MTH/TKHMTH2200/` (4,7 GB) → `~/ThS_archive/`, ghi sha256 + URL nguồn vào `EVIDENCE_INDEX.md`. **Giữ** `MTH/MTHv2_Datasets_Release/` (cần trích dẫn).
- [x] Gom file mồ côi bằng `git mv`: `ccrclip.pdf` → `docs/refs/`; `scripts/*ppocrv5*` → `scripts/_retired/` (**vẫn tracked** — là bằng chứng duy nhất cho câu "đã thử và loại Paddle" ở Chương 4.4)

**VÙNG CẤM**: `pipeline/`, `core/`, `config/`, `dict/`, `Data/`, `prepared/`, `dataset_out/`, `train_crop/`, 4 gitlink. Không tạo `src/`, không viết `paths.py`, không dời `s3_proto_cache.pkl` / `s3_calibration.json` / `align_engine/data/index.csv`.

**Xong khi**: `bash scripts/run_all_selftests.sh` vẫn ra **đúng 212/11** (không test nào chuyển từ pass sang skip) · `du -sh .` giảm ≥5 GB.

---

## GIAI ĐOẠN 3 — MỘT NGUỒN SỐ LIỆU DUY NHẤT (1–1,5 tuần) 🔴 CHẶN GĐ4

> Diệt blocker "số liệu bất nhất" (DE_XUAT §2.3). Đây là điều kiện để mọi con số viết vào luận văn không bị trôi.

### 3a. Vá 5 tiền điều kiện — pipeline hiện **KHÔNG chạy lại được** (2–3 giờ)

| # | Lỗi | Hậu quả nếu không vá |
|---|---|---|
| 1 | `config/pipeline.yaml` trỏ `Data/SachThanhTruyen*.pdf`, file thật là `Data/STT*.pdf` | Step 1 không mở được PDF. **Sửa `pdf:`, GIỮ NGUYÊN `name:`** — tên sách đi vào tên crop và cột `book` |
| 2 | `.gitmodules` thiếu 3/4 gitlink | Clone sạch thiếu `gannhanocr-fd` + `nom-embed` → SILVER sập âm thầm |
| 3 | Glob verdict không đệ quy (3 file `consensus_fusion/`) | 3 module in "SKIP" rồi `exit 0` — pipeline "thành công" mà không có bằng chứng |
| 4 | `publish/cli.py` lấy `labels_remediated.csv` | Release dư **1.396 GOLD + 527 SILVER** |
| 5 | `cmd_all` không gọi `export` | `release/` thiếu `parquet/`, `imagefolder/`, `crops.csv` — trong khi croissant đã ghi sha256 của `crops.csv` |

> ⚠️ **Vá #3 phải kèm guard**: file verdict duy nhất còn trên đĩa là `audit_SILVER/verdicts_ai.jsonl` (750 dòng `source='ai_vision'`). Glob đệ quy mà không lọc nguồn sẽ **âm thầm biến nhãn máy thành ground truth** — đúng blocker "SILVER = AI-audit". Mặc định chỉ nhận verdict NGƯỜI; muốn dùng AI phải khai cờ tường minh.

### 3b. `run_pipeline.sh` 8 bước (1 ngày)

Theo FLOW §5, dùng **tên chữ** thay vì số (vì `--step 3` hiện là fusion, còn Stage 3 của FLOW là remediation — giữ số sẽ nhập nhằng vĩnh viễn):

```
setup → extract → build → remediate → audit → fuse → confusion → publish
  0        1         2         3          4       5b      6          7
```

- Mặc định (không cờ) = `setup..confusion` → ra `labels_final.csv`. Không tự publish, không tự audit.
- `preflight()` fail-loud. Quan trọng nhất là **bẫy crop-proto**: `align_engine/data/index.csv` trỏ `dataset_out/gold/*.png`; nếu file đó không có thật → crop-protos = 0 → **SILVER tụt ~32% mà không ném lỗi nào**.
- Truyền `--strict` xuống `build_dataset` (hiện thiếu, dù README khuyên dùng).
- Bỏ `to_standard` khỏi đường mặc định (ghi đè metadata theo `labels.csv` **thô**, mâu thuẫn vĩnh viễn với `release/`).
- 2 **điểm dừng có người** tường minh (bước `audit`, bước `confusion`) — đây là *điểm bán* của luận văn, không phải khuyết điểm.

### 3c. Chốt số liệu (2–3 ngày)

- [ ] Quyết định **số census nào là chính thức**: số lịch sử ("đã từng có 2.321 lỗi" — chứng minh lỗi tồn tại) hay số hiện tại ("còn 8" — chứng minh đã phòng ngừa). Rồi sửa assertion + README + luận văn cho khớp.
- [ ] Chạy end-to-end từ cache **một lần**, kèm `--crop-review` (cách duy nhất lấy 15.690 crop REVIEW mà không rebuild lần hai)
- [ ] Rebuild TOÀN BỘ lớp publish từ `labels_final.csv` (xoá bản cũ trước, không vá tay)
- [ ] Dọn số đã bị bác còn sống trong mã: `consensus.py` (95,9%/97,6%), `channels.py` (`s3_pass=0.29`), mọi chỗ ghi "AUC 0.938" (đó là **retrieval-metric**, không phải error-detection; error-AUC thật là **0.566**)
- [ ] Lập `docs/BANG_SO_LIEU_CHINH_THUC.md` — mỗi số một dòng kèm (giá trị, file nguồn, lệnh tái sinh, ngày đo). **Không chương nào được trích số từ chỗ khác.**
- [ ] `git tag dataset-frozen-v1` + sha256 manifest

> **Lưu ý selftest**: hiện là **212 passed / 11 failed**, KHÔNG phải "223 assertions" như đang ghi trong luận văn. 11 lỗi không phải bug code — chúng hard-code số census của thế hệ `labels.csv` cũ (701/1686/2321/1177/3856) trong khi dữ liệu hiện tại cho 0/8/8/4/3850. Đây chính là bằng chứng của blocker "số liệu bất nhất".

**Xong khi**: `run_pipeline.sh` chạy liền mạch 8 bước · sha256 `labels_final.csv` sau khi chạy lại KHỚP bản trước · `frictionless validate` PASS · grep không còn số đã bị bác · tag tồn tại.

---

## GIAI ĐOẠN 4 — AUDIT NGƯỜI ĐỢT 2 (4–5 tuần lịch, ~30–40 giờ công) 🔴 ĐƯỜNG GĂNG DÀI NHẤT

> Khởi động **ngay sau GĐ3**, chạy song song mọi thứ. Đây là FLOW §4 **P1.2** ("việc mở lớn nhất còn lại") và DE_XUAT §2.1 + §2.2 cùng lúc.

- [ ] **Mẫu SRS xác nhận GOLD MỚI** trên `labels_final.csv` đã đóng băng, n≈450–600, plan acceptance mới. Đây là câu trả lời cho *"vì sao 98% mà không phải 97,08%"* — thiếu nó là câu chết, vì 98,0% hiện là số **post-hoc** tính trên chính mẫu đã dùng để phát hiện lỗi 㝵.
- [ ] **SILVER**: người chấm 750 item trên grid HTML **đã dựng sẵn** (ảnh nhúng base64, tự chứa). Tối thiểu chấp nhận được: 300. Ghi verdict `source='human'` vào file RIÊNG, **tuyệt đối không ghi đè** `verdicts_ai.jsonl`.
- [ ] **SYLLABLE**: hiện 0 verdict, chấm ~300 item.
- [ ] **Người chấm thứ 2** làm chung ~200 item → Cohen's kappa (hiện 1 auditor, 0 kappa).
- [ ] Phân tích độ nhạy `unsure` theo 3 cận (=đúng / loại / =sai).
- [ ] Bảng **đối chiếu AI-vs-người** → trình bày `verdicts_ai` như "AI second opinion" minh bạch. (2 AI hiện bất đồng lớn: verdicts_ai 505/692 vs Qwen 293/308 — dữ liệu tốt cho phần thảo luận.)

**Xong khi**: có verdict `source='human'` n≥450 (GOLD) / ≥300 (SILVER) / ≥300 (SYLLABLE) · `report.json` có precision + CI95 cho **cả ba** tier · có kappa · mẫu GOLD mới **độc lập** với mẫu đã dùng để mine 㝵.

**Phương án lùi** nếu hết tuần 5 chưa xong: SILVER n=300 với CI rộng hơn, khai rõ giới hạn trong luận văn.

---

## GIAI ĐOẠN 5 — RQ3 DOWNSTREAM (2–3 tuần, song song GĐ4)

> Lấp lỗ hổng thực nghiệm lớn nhất: 84% dataset paper được nhận đều có benchmark.

- [ ] Train classifier ký tự (tái dùng ResNet18+ArcFace + notebook Kaggle sẵn có) trên 3 cấu hình: **GOLD-only / GOLD+SILVER / +SYLLABLE**
- [ ] Đánh giá trên: (a) subset đã human-audit từ GĐ4, (b) 1 cấu hình **LOBO** theo sách
- [ ] Dùng splits page-disjoint + LOBO của `pipeline/publish` (đã đo 0 leakage) — **không tự chia lại bằng tay**
- [ ] Mục tiêu là MỘT bảng duy nhất, không cầu toàn SOTA. Không tinh chỉnh hyperparameter quá 2 vòng.

---

## GIAI ĐOẠN 6 — ABLATION (RQ2) + LÀM GIÀU NỘI TẠI + CHỐT CỨNG (2–3 tuần)

### 6a. Ablation — gần như 0 compute (chạy từ cache, không gọi API)
- [ ] banded-DP vs `_pair_old` vs Levenshtein cũ (số giai thoại hiện có: 41.824 vs 4.133 cặp)
- [ ] 3 chế độ reseg · có/không S3 · có/không `syllable_gate`
- [ ] Độ phủ từ điển (7.522 key) trên corpus — số 1 dòng nhưng chưa ai đo, chắc chắn bị hỏi

### 6b. FLOW §4 P1.3–P1.5 + làm giàu **nội tại** (không dùng nguồn ngoài)
- [ ] **P1.3**: audit ≥10 crop/cặp cho các cặp dân số lớn CHƯA kiểm (đời→代 195, mới→買 183, còn→群 173, nới→尼 166, liên→連, phúc→福, viết→曰). Quy tắc đã chốt: **không tin `exp_wrong` khi `a_n=1`**. Cặp nào sai cao → thêm vào `confusion_fixes.yaml`, **DEMOTE chứ không remap** (remap 㝵→𠊛 đã bị bác: 𠊛=0 trên trang, hỏng ~713 nhãn đúng).
- [ ] **P1.4**: soát mẫu 3.850 dòng `s1_inter_s2_similar` (nhóm bắc-cầu rủi ro cao nhất trong GOLD)
- [ ] **P1.5**: phục hồi 15 orphan (nới IoU≥0.3 + trùng nhãn) + re-audit 6 `label_changed`
- [ ] Cứu 1.923 crop 㝵/người bằng human glyph-anchor (~65% vẫn đúng → thu hồi ~1.250 GOLD, đồng thời trả lời câu *"mất lớp người thì dataset còn đại diện không"*)
- [ ] Audit **mức-lớp** cho SYLLABLE → vừa lấp lỗ "0 verdict", vừa sinh thêm ~3.750–6.751 nhãn ký tự với **cùng số giờ công**

### 6c. HẠN CHÓT CỨNG cuối tuần 7
- [ ] Đóng băng `confusion_fixes.yaml` → chạy lại chuỗi GĐ3 **đúng một lần** → `labels_final` v2 + publish v2 + cập nhật `BANG_SO_LIEU_CHINH_THUC.md` một lần duy nhất → `git tag dataset-frozen-final`

> **Sau tag này, mọi thay đổi số liệu đều bị từ chối** — kể cả khi "chỉ tốt hơn một chút". Không viết một dòng số nào của Chương 4 trước khi có tag.

---

## GIAI ĐOẠN 7 — VIẾT LUẬN VĂN (3–4 tuần, 50–75 trang)

Theo mục lục DE_XUAT §4. Bốn điểm phải viết **chủ động**:

1. **Circularity thành một mục riêng, không né.** Nguyên tắc: mọi claim công bố neo vào audit NGƯỜI.
2. **Case study 㝵/người thành mạch phương pháp**: audit → phát hiện (Fisher p=5,4e-8) → mine → demote 1.923 crop → xác nhận lại bằng **mẫu mới** (97,08% → 98,0%). *Chủ động khai trình tự là điểm cộng; bị phát hiện là điểm chết.*
3. **4 negative result như tài sản**: DINOv2 non-discriminative · S3 là **ranker** (retrieval@1 0,89) chứ không phải **error-gate** (AUC 0,566) — nhấn mạnh hai metric đo hai việc khác nhau · Fusion Path A là thiết kế + negative result · NomNaOCR double-circular → LOBO, Qwen-Flash wrong>right → VLM demote-only.
4. **Câu chốt bắt buộc, không được suy rộng** (nguyên văn FLOW §6): *"Tier GOLD (~48,9k crop) có precision char kiểm định bằng human-audit = 98,0% (CI95 ~96,9–98,8%, n=800) sau khi demote confusion hệ thống 㝵/người. SILVER/SYLLABLE là supervision yếu/mức-âm CHƯA kiểm định precision char. Không suy rộng 98% ra toàn bộ crop."*

**Xong khi**: mọi con số đối chiếu 1-1 với `BANG_SO_LIEU_CHINH_THUC.md` (rà thủ công một lượt) · không còn số nào thuộc thế hệ `labels.csv`/`labels_remediated.csv` · mọi URL tham khảo còn sống.

---

## GIAI ĐOẠN 8 — BUFFER + BẢO VỆ THỬ (2 tuần)

- [ ] Slide + demo chạy 1 sách **từ cache** (không gọi API — Kimhannom chết giữa buổi là hỏng demo; cache OCR chính là primary data)
- [ ] Bảo vệ thử với 8 câu Q&A (DE_XUAT §6), đặc biệt câu 3 (circularity) và câu 4 (vì sao 98%)
- [ ] Tuỳ chọn nếu dư: nộp **JOHD data paper** (~1.000 từ, datasheet đã build sẵn, rào cản thấp)

### DANH SÁCH CẤM cho tới khi bảo vệ xong
| Việc | Vì sao cấm |
|---|---|
| src-layout / `paths.py` / dời 25 hằng `parents[N]` | Đụng ~250 file; lỗi biểu hiện là **ghi số sai**, không crash |
| Dời `dataset_out`, `prepared/`, `Data/`, `dict/` | `Data`/`data` cùng inode trên APFS; index git ghi `Dict/` còn đĩa ghi `dict/` |
| Dời `s3_proto_cache.pkl` / `s3_calibration.json` / `index.csv` | 52.786 đường dẫn hard-code — dời là SILVER −32% âm thầm |
| Gộp file (42 đề xuất, 39 bị bác) | Xem `KIEM_KE_FILE_VA_LO_TRINH_2026-07-20.html`. Gom 7 selftest về `tests/` sẽ làm **66 assertion skip âm thầm** mà `exit 0` |
| Quay lại fusion Path B / kênh nặng / PPI | Đã đo AUC 0,54 và coverage 30% — không đổi được điểm bảo vệ |
| Thêm sách mới / mua 5 quyển còn thiếu | Số trang bản in không khớp PDF hiện có (T2: 362 vs 320) → **bản in khác**. Và thêm trang nào cũng buộc đo lại TOÀN BỘ |
| `git filter-repo` rút `.git` 1,8 GB | Viết lại mọi SHA → hỏng commit-hash đã trích trong luận văn |

**Sau bảo vệ**, trên nhánh riêng, theo thứ tự: `pyproject` + `paths.py` → `tests/` gom selftest → mới cân nhắc src-layout.

---

## SƠ ĐỒ PHỤ THUỘC

```
GĐ0 cứu bằng chứng ──┬──> GĐ1 tái lập ──> GĐ2 dọn rác (bỏ được)
                     │
                     └──> GĐ3 một-nguồn-số-liệu ──┬──> GĐ4 audit người (đường găng) ──┐
                              [tag v1]            │                                    │
                                                  └──> GĐ5 downstream ─────────────────┤
                                                                                       │
                                          GĐ6 ablation + làm giàu nội tại [tag final] ─┴──> GĐ7 viết ──> GĐ8 bảo vệ
```

**Ba ràng buộc thứ tự bất di bất dịch:**
1. GĐ0 xong **trước mọi lệnh** `mv`/`rm`/`--fresh`.
2. GĐ3 xong **trước** GĐ4 — mẫu SRS phải rút trên `labels_final` đã đóng băng, nếu không audit lại vô hiệu.
3. GĐ4 khởi động **sớm nhất có thể** — đường găng dài nhất, phụ thuộc người.
