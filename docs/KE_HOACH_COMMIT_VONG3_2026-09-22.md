# Kế hoạch commit vòng 3 (22/09 chiều) — trên HEAD 4e0a314bca, nhánh main

Trạng thái lúc chốt: `git status --short -- dataset_out data prepared/SachThanhTruyen*` **trống**; hồi quy STT 3 trang `labels.csv` md5
**59e436d7641fa849bb6759868ac29259** (556 dòng) worktree HEAD 4e0a314bca == mã mới (`summary.json` chỉ thêm `box_decoder: legacy`, `decisions_report.json` chỉ khác đường dẫn REPO);
`./run_pipeline.sh --dry-run` in đúng 6 bước STT; selftest book_layout 79/79 · ingest_lithograph 61/61 · ingest_prose 33/33 · **mechanism_gates 94/94** · **pitch_decode 22/22** ·
verses_ref_fix 19/19 · phase1_engine 253/0 · tools 139 pass / 3 fail (có sẵn ở HEAD, `KE_HOACH_COMMIT_2026-09-21.md` §4); `scripts/measure/code_facts.py` → FACTS **18/18**.
`git add` **đích danh** (không `-A`; submodule `nom-embed` ` m` không đụng). Sau mỗi commit chạy lại selftest nhóm đó.

## Commit 1 — engine: pitch_decode (box_decoder theo sách) + cổng (a') mechanism_gates; STT byte-identical

```
pipeline/align_engine/char_detector/pitch_decode.py   (??) giải mã hộp theo bước cột, ràng buộc N; selftest 22/22
pipeline/align_engine/book_layout.py                  (M)  khoá books[].box_decoder legacy|pitch; expected_tier_counts
pipeline/align_engine/align_production.py             (M)  pitch_target_count, assign_boxes_pitch, detector ở 0,05 khi pitch, seg_backend +pitch; n_det giữ nghĩa
pipeline/align_engine/build_dataset.py                (M)  summary.detector_params_by_book[].box_decoder, log "box_decoder = pitch", PASS 1b gán lại theo pitch
pipeline/remediation/mechanism_gates.py               (M)  luật (a'): detect_pitch_mode (cli>config>summary>labels), box_low_conf → GOLD_text_only, cờ n_det_mismatch; --box-decoder, --summary
pipeline/remediation/mechanism_gates_selftest.py      (M)  +32 phép [5] → 94/94
pipeline/export_final_dataset.py                      (M)  TRACE + n_det_mismatch (chỉ ghi khi nguồn có cột)
```
Thông điệp gợi ý: `feat(engine): box_decoder pitch (pitch_decode theo bước cột, n_det giữ hộp thô) + cổng B4' luật (a') theo ô ink_cut/detector_low, cờ n_det_mismatch; STT byte-identical`.
Bằng chứng kèm: md5 59e436d7…; 27 trang/sách IoU ≥ 0,5 97,4 → 98,1 / 94,1 → 97,0 % (`measure_out/box_ref/summary.json`); mechanism_gates STT → TẮT sao byte.

## Commit 2 — run_pipeline --book + config 3 sách (pitch) + bộ đo box_ref + train_crop v2

```
run_pipeline.sh                                       (M)  khối SÁCH MỚI: --book <Book>|all-new, --dry-run/--skip-ingest/--no-api/--no-auto-precision/--suffix; hàm STT/MAIN không đổi
config/pipeline_LucVanTien1883.yaml                   (M)  khối run:, box_decoder: pitch (+ lý do có số đo)
config/pipeline_KimVanKieu1884_b1.yaml                (M)  khối run: (verses_ref_fix), box_decoder: pitch
config/pipeline_KimVanKieu1884.yaml                   (M)  run_config → _b1; ghi chú box_decoder ở config chính thức (giữ legacy cho _v2_gates_noB1/_v1)
config/pipeline_Chrestomathie1872.yaml                (M)  khối run: (prose), box_decoder: pitch
scripts/measure/box_ref_eval.py                       (??) ô tham chiếu tự động (kim tầng + chiếu mực) vs legacy/pitch; invariants
scripts/measure/measure.py                            (M)  bước box_ref (ALL_STEPS, SUMMARY)
scripts/measure/code_facts.py                         (M)  pin DEFAULT_N_COLUMNS 43 → 50
scripts/measure/README.md                             (M)  box_ref
train_crop/build_lithograph_manifest.py               (??) nhãn yếu 31.776 ô + ignore_boxes, chia page-disjoint / --lobo
train_crop/train_v2_lithograph.py                     (??) trainer v2 (init v1, cân bằng miền, augment nền xám/kéo dọc), early-stop theo ok50 + STT F1
train_crop/eval_boxes_ref.py                          (??) eval mỗi epoch (ok50, tầng n==N thô, cắt thân chữ, STT P/R/F1)
README.md                                             (M)  mục "Sach moi": lệnh --book
.gitignore                                            (M)  /train_crop/data_lithograph/
```
Thông điệp gợi ý: `feat(run,measure,train_crop): run_pipeline --book <Book>|all-new (B0→B6, log, CHECKSUMS) + config pitch 3 sách + box_ref_eval + kịch bản huấn luyện detector v2 nhãn yếu`.
Bằng chứng kèm: `--suffix _rp` md5 khớp 3/3 sách (HUONG_DAN §2.1); `all-new --suffix _pitch` EXIT 0, 0 API, 169/236/137 s.

## Commit 3 — docs

```
docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md         (??) BÁO CÁO GỘP DUY NHẤT (vòng 1+2+I5+run_pipeline, 128 dòng)
docs/HUONG_DAN_HUAN_LUYEN_I5_2026-09-22.md           (??) I5: chẩn đoán, phương án A (đã chốt), phương án B, cách đo trung thực
docs/KE_HOACH_COMMIT_VONG3_2026-09-22.md             (??) tệp này
docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md           (M)  §2.0 --book, §2.1 tái lập, ghi chú vòng 3 (pitch, a'), §6.1
docs/CHAY_LVT1883_2026-09-21.md, CHAY_KVK1884_B1_2026-09-22.md, CHAY_KVK1884_2026-09-21.md, CHAY_CHRESTO1872_2026-09-22.md  (M)  đầu tệp "chốt cuối: pitch"
docs/BAO_CAO_TONG_THE_SACH_MOI_2026-09-22.md, docs/PHUONG_AN_TU_DONG_2026-09-22.md  (M)  1 dòng "đã gộp…; giữ làm lịch sử"
docs/PIPELINE_FACTS.json                              (M)  18/18 (sinh lại sau commit 2 nếu muốn git_head khớp)
CLAUDE.md                                             (M)  trỏ báo cáo gộp + lệnh --book + kế hoạch commit vòng 3
```
Thông điệp gợi ý: `docs: báo cáo gộp sách mới (pitch + cổng a' là chốt cuối), hướng dẫn I5/huấn luyện v2, CHAY_* chốt pitch, kế hoạch commit vòng 3`.

## KHÔNG commit

| Tệp / thư mục | Lý do |
|---|---|
| `train_crop/data_lithograph/` (?? , **7,3 MB** manifest JSON) | > 5 MB, tái sinh 24 s bằng `build_lithograph_manifest.py` → **thêm `.gitignore`: `/train_crop/data_lithograph/`** |
| `pipeline/tools/kiem_nguoi_grid.py`, `kiem_nguoi_score.py` (??) | chưa selftest/review; đề tài không có người kiểm |
| `docs/KVK1884_TRANG_CAN_XAC_NHAN.csv` (??) | lỗi thời (27 trang formula≠anchor; `content` quyết từng cột) → xoá hoặc để ngoài |
| `nom-embed` (` m`) | submodule có best.pt/last.pt chưa track — không đụng |
| `dataset_*/`, `dataset_out_*/` (kể cả `*_v3_legacy`, `*_rp`), `prepared*/`, `measure_out/`, `logs/*.log` | đã ignore; tái sinh bằng `./run_pipeline.sh --book all-new` (~9 phút, 0 API) / `measure.py --all` |

`.gitignore` **đã thêm** (chưa commit, đưa vào commit 2): `/train_crop/data_lithograph/` (dưới `train_crop/kaggle_pkg/`). `*.pt` đã ignore (detector_v2_litho.pt khi huấn luyện).

## Kiểm trước khi bấm commit

1. `git status --short -- dataset_out data prepared/SachThanhTruyen*` trống; `git diff --stat` chỉ gồm tệp trong 3 nhóm trên + `.gitignore`.
2. Sau commit 1: `book_layout_selftest` 79/79, `mechanism_gates_selftest` 94/94, `pitch_decode --selftest` 22/22, `phase1_engine_selftest` 253/0; hồi quy STT 3 trang md5 59e436d7….
3. Sau commit 2: `step0_setup` 4 config "Validation passed" (book_layout ghi `box_decoder=pitch` ở LVT/KVK_b1/Chresto); `./run_pipeline.sh --book all-new --dry-run` in B0→B6; `./run_pipeline.sh --dry-run` in 6 bước STT.
4. Sau commit 3 (tuỳ chọn): `scripts/measure/code_facts.py` → 18/18 rồi amend FACTS.
