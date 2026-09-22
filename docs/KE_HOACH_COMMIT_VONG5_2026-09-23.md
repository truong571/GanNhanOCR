# Kế hoạch commit vòng 5 (23/09) — kênh kim `lang_type = 2`, luật đếm 6/8, rào tầng DP

Trên HEAD **0997cf57bb**, nhánh `main`. **Chưa commit gì** — tệp này là danh mục đích danh để người ký.

## 0. Trạng thái nghiệm thu lúc chốt

| Cổng | Kết quả |
|---|---|
| Hồi quy STT 3 trang (`build_dataset --config config/pipeline.yaml --qd01-cells none --force --pages pages3.csv`, stt2/0024 · stt4/0050 · stt11/0100) | `labels.csv` md5 **59e436d7641fa849bb6759868ac29259** (556 dòng) — **worktree HEAD 0997cf57bb == mã vòng 5**; `diff -rq` toàn thư mục: 2/13 tệp khác và chỉ khác **đường dẫn REPO** (`summary.json`, `decisions_report.json`). Worktree đã gỡ (`git worktree list` chỉ còn repo chính) |
| selftest | ingest_lithograph **82/82** (61 → +21) · book_layout **115/115** (100 → +15) · ingest_prose 33/33 · phase1_engine 253/0 · pitch_decode 22/22 · mechanism_gates 94/94 · verses_ref_fix 19/19 |
| `scripts/measure/code_facts.py` | **18/18** invariants (pin `book_layout.py` `DEFAULT_N_COLUMNS` 61 → **83**; bất biến `ocr_api_guest_mode_committed` bỏ mệnh đề "không có diff" — xem §3) |
| `./run_pipeline.sh --dry-run` | in đúng 6 bước STT |
| `git status --short -- dataset_out prepared/SachThanhTruyen*` | **trống** |
| `git status --short -- data` | **KHÔNG trống nhưng KHÔNG do vòng 5**: `D data/{CacThanhTruyen1646/SOURCE.md, IHR-NomDB_nlp/*, KimVanKieu1894/SOURCE.md, LyHangCaDao/*, TamTuKinhDienAm/*}` + `?? data/CacThanhTruyen1646_x/`. `.git/index` mtime 21:17 < lúc phiên bắt đầu (22:20) và vòng 5 không có lệnh nào ghi vào `data/` ⇒ đã có từ 20–21/09 (đổi tên/di chuyển thủ công). `data/LucVanTien1916` và `data/TruyenKieu1872` (tham chiếu IHR) còn nguyên |

## 1. Quyết định của vòng 5 (số ở §2)

**Nhận `lang_type = 2` (Nôm) làm bản chốt cho CẢ BA sách**, kèm luật đếm QN 6/8 (LVT/KVK) và rào tầng DP (LVT/KVK).
Bản `lang_type = 1` giữ nguyên tại `dataset/<Book>_v5_lang1/` và `prepared*/<Book>/dataset_out*_v5_lang1/` để đối chiếu;
cache kim của **cả hai** cấu hình cùng tồn tại (`kim_raw/page_XXXX.json` = lang 1, `page_XXXX_lt2.json` = lang 2) nên
quay lại chỉ tốn 0 lượt gọi API: đổi `books[].kim_lang_type` về 1 rồi `./run_pipeline.sh --book all-new`.

## 2. Bảng trước → sau (bản chốt cũ = pitch/linear/lang 1 ⟶ bản chốt mới = pitch/linear/lang 2 + 6/8 + rào tầng)

| Chỉ số | LucVanTien1883 | KimVanKieu1884 (B1') | Chrestomathie1872 |
|---|---|---|---|
| Ô sinh ra | 14.476 → 14.474 | 22.704 → **22.750** | 8.303 → **8.016** (−287) |
| Cột QN đủ 14 âm (ingest) | 893/1.044 = 85,5 % → **1.017 = 97,4 %** | 1.511/1.628 = 92,8 % → **1.602 = 98,4 %** | — (văn xuôi) |
| Cột kim đọc đủ 14 chữ | 1.024 = 98,1 % → 971 = **93,0 %** ↓ | 1.560 = 95,8 % → **1.603 = 98,5 %** | — |
| **M==N** (kim = QN) | 83,8 % → **90,4 %** | 88,8 % → **96,9 %** | 69,5 % → **44,0 %** ↓↓ |
| I5 thô `n_det==N` | 65,2 % → **70,7 %** | 59,2 % → **62,3 %** | 70,7 % → 66,7 % ↓ |
| tier thô GOLD | 9.667 → **11.431** | 15.931 → **20.636** | 5.892 → **6.569** |
| **GOLD ảnh** (sau cổng B4') | 8.650 → **10.831** (+2.181) | 13.908 → **19.303** (+5.395) | 5.359 → **5.998** (+639) |
| GOLD_text_only | 130 → 149 | 510 → 727 | 126 → 226 |
| SYLLABLE / REVIEW | 2.363 / 3.319 → **1.207 / 2.259** | 3.844 / 4.340 → **618 / 1.971** | 1.286 / 1.532 → **856 / 936** |
| QUARANTINE (F1 cross-col) | 14 → 28 | 102 → 131 | 0 → 0 |
| **Ảnh export** | 11.013 → **12.038** | 17.752 → **19.921** | 6.645 → **6.854** |
| `count_source = pitch_ocr` (cột QN lệch) | 1.319 → **319** | 1.088 → **256** | 1.902 → 1.109 |
| crop `bleed` / tổng dòng | 1.988/11.143 = 17,8 % → 2.196/12.187 = **18,0 %** | 1.905/18.262 = 10,4 % → 2.138/20.648 = **10,4 %** | 137/6.771 = 2,02 % → 139/7.080 = **1,96 %** |
| **Khớp dị bản GOLD, bỏ ref PUA — trước cổng** | 1916: 72,6 % (n 2.796) → **78,0 %** (n 3.293) | 1871: 80,6 % (n 11.463) → **90,2 %** (n 14.635); **1872 độc lập**: 73,0 % (n 11.662) → **78,2 %** (n 14.976) | không có dị bản |
| **…B6 (labels_gated)** | 79,1 % → **82,1 %** (n 3.068) | 1871: 85,8 % → **92,0 %**; 1872: 78,5 % → **79,9 %** | — |

Cổng nghiệm thu đã đặt ở `CHOT_KENH_OCR §6 #1` ("khớp dị bản tăng ≥ 5 điểm, `box_ref` không giảm"): **đạt** —
LVT +5,4 điểm, KVK/1872 (**độc lập**) +5,2 điểm, KVK/1871 +9,6 điểm; cờ crop không xấu đi ở cả ba sách.
KVK khớp 1871 **92,0 %** nay **vượt nền dị bản 1871↔1872 = 82,9 %** (1871 là nguồn QN của B1' nên chiều này có
phần tự khẳng định; con số **độc lập** phải đọc là 1872: 78,2 / 79,9 %).

**Chrestomathie1872 — nhận nhưng có nợ**: GOLD +639 và ảnh export +209 nên theo luật quyết định (giữ lang 1 *chỉ khi*
GOLD giảm) thì nhận; **nhưng** M==N 69,5 → 44,0 % và số ô sinh ra −287, và sách này **không có dị bản** để kiểm chéo,
nên bằng chứng chỉ một chiều (hợp từ điển). Đổi lại bằng một khoá: `books[].kim_lang_type: 1` trong
`config/pipeline_Chrestomathie1872.yaml`.

## 3. Commit 1 — kênh kim theo sách (`lang_type` 2) + cache tách theo tham số

```
core/ocr/ocr_api.py                          (M)  recognize(file_name, *, ocr_id, lang_type, reading_direction, font_type)
                                                  — CHỈ TỪ KHOÁ, None = hằng KIM_*_DEFAULT (1,1,1,1) ⇒ body của mọi lời
                                                  gọi cũ BYTE-IDENTICAL; không có biến toàn cục đổi mặc định
pipeline/align_engine/book_layout.py         (M)  books[].kim_lang_type / kim_ocr_id / kim_font_type (+ _enum_key),
                                                  BookLayout.kim_params / kim_is_default; docstring có số đo
pipeline/tools/ingest_lithograph_book.py     (M)  kim_params_of / kim_cache_suffix / book_kim_params; kim_boxes(..., kim=)
                                                  ghi `kim_params` vào cache và CHỈ dùng lại cache trùng tham số;
                                                  CLI --kim-lang-type/--kim-ocr-id/--kim-font-type/--kim-config
pipeline/tools/ingest_prose_book.py          (M)  dùng lại 3 hàm trên; cùng bộ CLI
pipeline/tools/ingest_prose_selftest.py      (M)  stub kim_boxes nhận kw `kim`
run_pipeline.sh                              (2 hunk) truyền --kim-config "$BK_CONFIG" cho cả 2 adapter
```
Kiểm: `ingest_lithograph_selftest` +21 phép (body mặc định bắt tận nơi qua `requests.post` giả; cache cũ không có
`kim_params` vẫn hợp lệ với bộ cũ; cache lang 1 KHÔNG được dùng cho lang 2).

**Lưu ý bất biến `ocr_api_guest_mode_committed`** (`scripts/measure/code_facts.py`): mệnh đề thứ ba
"`ocr_api.py` không có diff chưa commit" sinh ra hồi Guest Mode chỉ sống trong worktree; nay guest đã ở trong git
(65f7ca9) còn `ocr_api.py` vẫn được sửa hợp lệ, nên mệnh đề ấy đã bỏ khỏi **bất biến** (trạng thái diff vẫn được
ghi trong `summary.git`). Nếu người ký muốn giữ nguyên mệnh đề cũ thì chỉ cần commit là bất biến tự xanh lại.

## 4. Commit 2 — luật lục bát 6/8: sửa số đếm âm QN + `tier_n` theo luật + rào tầng cho DP

```
pipeline/tools/ingest_lithograph_book.py     (M)  QN_UNREADABLE, repair_tier_syllables (restore_unreadable /
                                                  drop_verse_number / merge_unreadable), resplit_couplet;
                                                  make_column_texts(qn_rule=, fix_stats=); CLI --qn-count-rule;
                                                  manifest.gates.qn_count_fix
pipeline/align_engine/book_layout.py         (M)  LITHO_TIER_RULE (6,8), tier_rule_for(lay),
                                                  expected_tier_counts(..., tier_rule=) — cột đủ 14 âm lấy LUẬT
                                                  thay len_odd của QN; books[].tier_dp (chỉ lithograph)
pipeline/align_engine/anchor_align.py        (M)  tier_spans / tiers_valid / realign_column_tiered /
                                                  posterior_matches_tiered (DP 6↔6 rồi 8↔8, chỉ số dịch về cột)
pipeline/align_engine/align_production.py    (M)  column_tiers() dùng group_tiers của pitch_decode (cùng ranh giới
                                                  với bộ giải mã hộp); _pair_new_state(tier_dp=); box_info['col_tiers'];
                                                  expected_tier_counts(tier_rule=tier_rule_for(layout))
pipeline/align_engine/build_dataset.py       (M)  PASS 1b: realign_with_anchors(tiers=cs['col_tiers']) — DP và
                                                  posterior CÙNG tiers
pipeline/align_engine/book_layout_selftest.py(M)  +15 phép → 115/115
pipeline/tools/ingest_lithograph_selftest.py (M)  (đã tính ở commit 1)
```
Số đo: luật đếm chạm **165 tầng LVT** (sửa được 138: 64 restore + 60 drop_verse_number + 14 merge; còn 27) và
**126 tầng KVK** (sửa 97 + 2 cột resplit 6/8; còn 29) ⇒ cột đủ 14 âm 85,5 → 97,4 % (LVT), 92,8 → 98,4 % (KVK).
Rào tầng: mặc định **tắt**; bật bằng `books[].tier_dp: true` (chỉ `layout: lithograph`), cột không đủ điều kiện
(số tầng hai bên khác nhau / tầng rỗng) tự rơi về DP cả cột.

**Ablation rào tầng đo riêng** (LVT, `build_dataset --limit 20`, chỉ đổi `tier_dp` true/false, mọi thứ khác giữ nguyên):
2.763 ô, **đổi đúng 2 ô** — 1 cặp ghép xuyên tầng bị gỡ (`page_0003` cột 3: âm "sao" của câu lục ghép với chữ ở tầng
**dưới** khi DP tự do, ghép đúng tầng **trên** khi có rào); 0 ô đổi tier, 0 ô đổi bbox. Đúng như dự đoán ở
`CHOT_KENH_OCR §6 #3` ("lợi nhỏ nhưng là bảo đảm cấu trúc"): rào tầng là **ràng buộc**, không phải nguồn tăng GOLD.

## 5. Commit 3 — config theo sách + bộ đo + tài liệu

```
config/pipeline_LucVanTien1883.yaml      (M)  kim_lang_type: 2 · tier_dp: true · run.ingest_args "--qn-count-rule"
                                              + chú thích `--contrast stretch` là ÁNH XẠ ĐỒNG NHẤT trên bản quét này
config/pipeline_KimVanKieu1884_b1.yaml   (M)  như trên (`--contrast otsu` cũng là ánh xạ đồng nhất)
config/pipeline_KimVanKieu1884.yaml      (M)  kim_lang_type: 2 (để chạy tay cũng đúng kênh)
config/pipeline_Chrestomathie1872.yaml   (M)  kim_lang_type: 2 (kèm cảnh báo M==N trong chú thích)
scripts/measure/code_facts.py            (M)  pin book_layout.py 61 → 83; bất biến guest-mode (xem §3)
docs/PIPELINE_FACTS.json                 (M)  sinh lại (18/18)
scripts/measure/kim_channel_probe.py     (A)  bộ đo kênh kim (107 lượt, cache theo md5+cấu hình) — vòng trước để lại
scripts/measure/qn_engine_compare.py     (A)  so tesseract ↔ VietOCR trên sách in tk 19 — vòng trước để lại
scripts/measure/loss_ledger.py           (A)  sổ kế toán mất mát 3 sách (24/24 invariants) — vòng trước để lại
docs/CHOT_KENH_OCR_VA_QUY_HOACH_GAN_2026-09-23.md (A)  cơ sở đo của vòng 5
docs/CHAN_DOAN_3_SACH_MOI_2026-09-23.md           (A)  sổ kế toán + §5 đã bổ sung "thực tế đạt được"
docs/KE_HOACH_COMMIT_VONG5_2026-09-23.md          (A)  tệp này
docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md      (M)  §0 §3.4 §6 §8 — bản chốt mới
docs/CHAY_LVT1883_2026-09-21.md · CHAY_KVK1884_B1_2026-09-22.md · CHAY_CHRESTO1872_2026-09-22.md (M) khối đầu
```

**KHÔNG thuộc vòng 5** (người dùng sửa song song, để người ký tự quyết):
`config/pipeline_KimVanKieu1884_b1_boost.yaml` (đổi `output_dir` sang `dataset/`),
`lab/i5_detector_v2/kaggle_train_i5_v2.ipynb` (xoá output notebook), `web/` (thư mục rỗng mới),
`nom-embed` (submodule ` m`), và các thay đổi `paths.output_dir` / `run.dataset_out` sang `dataset/<Book>` +
`prepared*/<Book>/dataset_out*` nằm lẫn trong 4 config sách ở trên.

## 6. Sau khi commit

`git add` **đích danh** (không `-A`), không commit `dataset*/`, `prepared*/`, `measure_out/`, `logs/`.
Chạy lại để xác nhận: `./run_pipeline.sh --book all-new` (0 lượt API — cache `kim_raw/*_lt2.json` đã có) rồi
`scripts/measure/code_facts.py` (18/18) và hồi quy STT md5 59e436d7…
