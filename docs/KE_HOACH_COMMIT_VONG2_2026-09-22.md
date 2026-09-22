# Kế hoạch commit vòng 2 (22/09) — trên HEAD 853cadfd9c, nhánh main

Trạng thái lúc chốt (NV-D, 22/09 chiều): `git status --short -- dataset_out data prepared/SachThanhTruyen*` **trống**; hồi quy STT 3 trang
`labels.csv` md5 **59e436d7641fa849bb6759868ac29259** (556 dòng) HEAD == bản sửa (worktree tạm đã gỡ); selftest book_layout 79/79 ·
ingest_lithograph 61/61 · ingest_prose 33/33 · mechanism_gates 62/62 · verses_ref_fix 19/19 · phase1_engine 253/0 · tools 139 pass / 3 fail
(3 fail có sẵn ở HEAD: `KE_HOACH_COMMIT_2026-09-21.md` §4); `scripts/measure/code_facts.py` → `docs/PIPELINE_FACTS.json` **18/18**.
`git add` **đích danh** từng tệp dưới đây, không `-A`. Sau mỗi commit chạy lại selftest của nhóm đó.

## Commit 1 — engine: layout prose + cổng cơ chế B4' + tầng GOLD_text_only (không đổi hành vi STT)

```
pipeline/align_engine/book_layout.py            (M)  layout: prose, n_columns: auto, prose_gate, PROSE_DET_XMARGIN
pipeline/align_engine/book_layout_selftest.py   (M)  +22 phép prose → 79/79
pipeline/align_engine/align_production.py       (M)  nhánh _detect prose (n_exp = số dòng QN), layout_gate; nhánh STT giữ từng dòng
pipeline/step0_setup.py                          (M)  sách không PDF: layout ∈ {lithograph, prose}
pipeline/step1_extract.py                        (M)  như trên
pipeline/remediation/mechanism_gates.py          (??) cổng (a)–(d), bật theo layout lithograph | books[].mechanism_gates
pipeline/remediation/mechanism_gates_selftest.py (??) 62/62
pipeline/export_final_dataset.py                 (M)  tầng GOLD_text_only (nhãn, không ảnh), --n-columns
pipeline/tools/make_dataset_docs.py              (M)  README/DATASHEET nêu GOLD_text_only, --n-columns
```
Thông điệp gợi ý: `feat(engine): layout prose (n_columns auto, prose_gate) + cổng cơ chế B4' mechanism_gates + tầng GOLD_text_only ở export; STT byte-identical`.
Bằng chứng kèm: md5 59e436d7…; mechanism_gates với `config/pipeline.yaml` → TẮT, sao byte; export STT `diff -rq` rỗng (71.592 tệp).

## Commit 2 — tools + config + bộ đo: adapter B1'/prose, config 3 sách, verses_ref_fix, code_facts

```
pipeline/tools/ingest_lithograph_book.py         (M)  --verses PATH (B1'), --dict-boost (không dùng), qn_source vào transcriptions
pipeline/tools/ingest_lithograph_selftest.py     (M)  +26 phép → 61/61
pipeline/tools/ingest_prose_book.py              (??) adapter văn xuôi Chrestomathie
pipeline/tools/ingest_prose_selftest.py          (??) 33/33
config/pipeline_Chrestomathie1872.yaml           (??) prose, auto, mechanism_gates: true
config/pipeline_KimVanKieu1884.yaml              (M)  khoá ghi nhận verses_b1 + chú thích đường chính thức B1'
config/pipeline_KimVanKieu1884_b1.yaml           (??) ĐƯỜNG CHÍNH THỨC KVK (data_dir prepared_b1, output_dir dataset_KimVanKieu1884)
config/pipeline_KimVanKieu1884_b1_boost.yaml     (??) chỉ để tái lập ablation dict-boost (header ghi KHÔNG DÙNG) — có thể bỏ nếu không muốn giữ
scripts/measure/verses_ref_fix.py                (??) B1' verses_b1.tsv, --selftest 19/19, --check-labels
scripts/measure/build_metrics.py                 (??) so tier/cột giữa các bản build
scripts/measure/auto_precision.py                (M)  --labels/--trans/--labels-name, by_qn_source
scripts/measure/code_facts.py                    (M)  KNOWN_PINS dời theo mã mới; invariant guest_mode_committed, dataset_out_tracked_clean
scripts/measure/README.md                        (M)
.gitignore                                       (M)  /prepared_b1*/, /dataset_out_Chrestomathie1872*/, /dataset_Chrestomathie1872*/
```
Thông điệp gợi ý: `feat(tools): ingest_prose_book + B1' verses_ref_fix/--verses + config Chresto/KVK_b1 + auto_precision --labels; code_facts pins 22/09`.

## Commit 3 — docs

```
docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md      (M)  §2 B4 --out + cảnh báo, B1', B4', prose; §3 cổng 3 sách; §7 chọn layout (139 dòng)
docs/BAO_CAO_TONG_THE_SACH_MOI_2026-09-22.md    (M)  §9 vòng 2; §1/§3/§4/§6/§7/§8 nhất quán
docs/CHAY_LVT1883_2026-09-21.md                 (M)  mục sau cổng B4'
docs/CHAY_KVK1884_2026-09-21.md                 (M)  mục sau cổng B4' + ghi chú đường chính thức = B1'
docs/CHAY_KVK1884_B1_2026-09-22.md              (??) B1' + chốt vòng 2 (§0)
docs/CHAY_CHRESTO1872_2026-09-22.md             (??)
docs/KE_HOACH_COMMIT_VONG2_2026-09-22.md        (??) tệp này
docs/PIPELINE_FACTS.json                        (M)  18/18 (git_head ghi 853cadfd9c; sinh lại sau commit 2 nếu muốn khớp HEAD mới)
docs/CHAY_KVK1884_THU_2026-09-21.md             (??) lịch sử (được CHAY_KVK1884 §đầu tham chiếu) — commit cùng nhóm docs
```
Thông điệp gợi ý: `docs: vòng 2 22/09 — hướng dẫn 3 sách (B1', B4', prose, --out), báo cáo tổng thể §9, CHAY KVK B1'/Chresto, FACTS 18/18`.

## KHÔNG commit

| Tệp / thư mục | Lý do |
|---|---|
| `pipeline/tools/kiem_nguoi_grid.py`, `kiem_nguoi_score.py` (??) | chưa selftest/review; đề tài không có người kiểm |
| `docs/KVK1884_TRANG_CAN_XAC_NHAN.csv` (??) | LỖI THỜI (27 trang formula≠anchor; `content` quyết từng cột) → xoá hoặc để ngoài |
| `nom-embed` (` m`) | submodule có nội dung chưa track (best.pt/last.pt) — không đụng |
| `dataset_*/`, `dataset_out_*/`, `prepared*/`, `measure_out/` | đã ignore; tái sinh theo HUONG_DAN §2 (KVK 0 lượt kim, Chresto 60 lượt) |

## Kiểm trước khi bấm commit

1. `git status --short -- dataset_out data` trống; `git diff --stat` chỉ gồm tệp trong 3 nhóm trên.
2. Sau commit 1: `book_layout_selftest` 79/79, `mechanism_gates_selftest` 62/62, `phase1_engine_selftest` 253/0.
3. Sau commit 2: `ingest_lithograph_selftest` 61/61, `ingest_prose_selftest` 33/33, `verses_ref_fix.py --selftest` 19/19, `step0_setup` 3 config "Validation passed".
4. Sau commit 3 (tuỳ chọn): `scripts/measure/code_facts.py` → 18/18 rồi amend FACTS.
