# Chữa I5 (hộp chữ detector trên thạch bản) — chẩn đoán, phương án A (pitch_decode, đã đo), phương án B (huấn luyện v2, kế hoạch)

Ngày 22/09/2026 · HEAD 8c08591e9a + mã chưa commit (khoá `detector_ckpt`/`detector_resize`, lab/i5_detector_v2) · mọi số ở đây sinh bởi script, 0 token LLM, 0 API, **0 hộp GT người**.

## 0. Phát hiện răng cưa (INTER_AREA) — 22/09 tối, đọc trước khi huấn luyện

**Phần lớn lỗi I5 không phải model mà là phép thu ảnh.** `CenterNetDetector._preprocess` thu trang về 1.024 px bằng `cv2.resize` mặc định
(INTER_LINEAR); ảnh thạch bản 3.204 px (LVT) / 2.789 px (KVK) bị thu 3,1× / 2,7× → răng cưa làm rụng nét mảnh → điểm tin cậy thấp,
bỏ sót chữ nhạt, bắt 2 lần. Chỉ đổi sang INTER_AREA (khoá **`books[].detector_resize: area`**, mặc định `linear` = STT không đổi byte;
`train_crop/infer_centernet.py` tham số `resize`, cache detector theo `(ckpt, resize, thr)`), **cùng ckpt v1, không huấn luyện**:

| v1, cùng mã đo | linear (pipeline) | **area** | nguồn |
|---|---|---|---|
| 27 trang thạch bản val ảnh gốc: ok50 / miss / extra÷100 / **tầng n==N @0,15** / cắt thân chữ | 95,0 / 2,1 / 2,6 / **77,0** / 7,1 | **98,1 / 0,8 / 0,9 / 91,9 / 6,4** | `train_crop/eval_boxes_ref.py`, lab README |
| STT 45 trang val F1@0,2 (hồi quy) | 0,8767 | 0,8785 | như trên |
| `box_ref_eval` 27 trang/sách ảnh prepared — LVT: I5 cột / tầng n==N / legacy ok50 / miss / extra÷100 / pitch ok50 | 63,7 / 77,4 / 97,4 / 0,5 / 2,0 / 98,1 | **79,6 / 88,5 / 99,3 / 0,1 / 0,5 / 99,3** | `measure_out/box_ref_area/` (`--resize area`, 78 s) |
| … KVK | 54,4 / 71,1 / 94,1 / 3,0 / 2,3 / 97,0 | **74,8 / 85,4 / 97,8 / 1,0 / 1,2 / 98,6** | 24/24 invariants PASS |
| … "cắt vào thân chữ" legacy / pitch — LVT · KVK | 11,0 / 9,5 · 3,8 / 2,6 | 10,8 / **10,6** · 3,1 / 2,5 | LVT pitch **tăng 1,1 điểm** (đổi cờ 179 ô có / 142 ô mất; CI ô ±0,7) |
| … hộp thô: h/p trung vị · điểm tin cậy trung bình — LVT · KVK | 1,14 · 0,35 · 1,11 · 0,32 | 1,16 · 0,39 · 1,13 · 0,36 | hộp cao hơn ≈ 2 % (+3 px), dịch lên 2 px |
| Build trọn `./run_pipeline.sh --book all-new --suffix _area` (0 API, 174 / 237 / 138 s): **I5 thô toàn sách** LVT · KVK · Chresto | 65,2 · 59,2 · 70,7 | **77,7 · 77,9** · 69,3 | `dataset_out_<Book>[_b1]_area/labels.csv`; Chresto thu 2,2× nên ≈ linear |
| … GOLD ảnh / text_only / ảnh export — LVT · KVK · Chresto | 8.650/130/11.013 · 13.908/510/17.752 · 5.359/126/6.645 | 8.733/43/11.097 · 14.165/201/17.999 · 5.344/134/6.636 | tier văn bản thô y hệt; khớp dị bản sau (a)(b)(c) 73,4 = 73,4 / 81,3 → 81,1 (trong CI) |
| … `bleed` trên ảnh export — LVT · KVK · Chresto | 18,0 · 10,5 · 2,0 % | **21,7** · 10,5 · 2,0 % | LVT +429 ô: cờ đổi dồn vào ô có crop cao thêm > 4 px (+607 / −204), tức do hộp to hơn |

**Quyết định (luật "không chỉ số then chốt nào giảm", `docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md` §3.3): `_area` CHƯA thành bản chốt** —
LVT bleed ảnh export +3,7 điểm và cắt thân chữ pitch +1,1; Chresto không lợi. KVK là sách duy nhất không giảm chỉ số nào (bleed 10,5 = 10,5,
blank/truncated +40/+14 đều bị cổng (c) hạ REVIEW). Config 3 sách ghi `detector_resize: linear` kèm số; đổi một chữ `area` là tái lập `_area`.

**Hệ quả cho phương án B (v2)**: mốc so sánh thật của v2 là **v1 + area**, không phải v1 linear — v1+area đã đạt tầng n==N 91,9 % (val) /
I5 thô 77,7 / 77,9 % (toàn sách), tức phần "đếm lệch ±1" phần lớn là răng cưa. Việc còn lại cho v2 là **kích thước/tâm hộp trên thạch bản**
(h/p 1,16 → gần ô tham chiếu ≈ 1,0–1,1; giảm bleed và cắt cùng lúc) và điểm tin cậy chữ nhạt; bundle Kaggle đo bằng `area`, `best.pt`
chỉ ghi khi vượt v1+area (§3, lab README).

## 0b. Kết luận ngắn (phương án A, 22/09 chiều)

| | LVT1883 | KVK1884 | ghi chú |
|---|---|---|---|
| I5 cột `n_det == N` @0,15 (27 trang/sách, 270 cột) | 63,7 % | 54,4 % | đúng số cũ 65,2 / 59,2 (toàn sách) |
| Ô tham chiếu verified: legacy@0,15 IoU ≥ 0,5 / miss / extra | 97,4 % / 0,5 % / 2,0 | 94,1 % / 3,0 % / 2,3 | ô-level tốt hơn nhiều so với cột-level |
| **pitch_decode** IoU ≥ 0,5 / miss / extra | **98,1 % / 0,1 % / 0,1** | **97,0 % / 0,4 % / 0,4** | miss/extra ≈ 0 **một phần theo cấu tạo** (§4) |
| "cắt vào thân chữ" (mực chạm mép hộp, không phụ thuộc tham chiếu): legacy → pitch | 11,0 → 9,5 % | 3,8 → 2,6 % | ô tham chiếu: 0,9 / 0,2 % |
| build LVT `--limit 10` (98 cột, 1.361 ô): tier | **không đổi** (GOLD 814 / SYL 74 / REVIEW 473) | — | tier do văn bản quyết |
| … box_source midpoint+split → ink_cut+detector_low | 52+1 → 5+3 | — | 69 ô đổi hộp (5,1 %) |
| … F1 cross-col (census) · bleed (crop_quality) | 10 → 8 ô · 154 → 155 | — | trong nhiễu |

Đọc: **I5 thất bại ở mức cột (35–45 %) nhưng ở mức ô chỉ 2–6 % ô lệch**; lỗi là ±1 hộp trong tầng (bắt 2 lần / bỏ sót chữ
nhạt dưới ngưỡng), **không phải** số câu in lọt vào dải x (chỉ 0–0,4 % cột có hộp ngoài tầng). Phương án A sửa được phần
"hoà giải" (chọn đúng N hộp theo bước) mà không cần huấn luyện; phương án B mới sửa được gốc (điểm tin cậy thấp trên thạch bản).

## 1. Chẩn đoán định lượng — `measure_out/box_ref/summary.json` (`scripts/measure/box_ref_eval.py`, 79 s)

**Ô tham chiếu tự động** = hộp TẦNG của kim (adapter: 1 hộp/tầng-cột) cắt tại N−1 khe mực yếu nhất bằng quy hoạch động có phạt
lệch bước (`pitch_decode.ink_cut_cells`, KHÔNG chia đều), N = số âm QN của câu (`transcriptions/*.json`), x kẹp ±0,5 bước cột.
"Verified" = số chữ kim của tầng == N (LVT 90,6 %, KVK 93,5 % tầng). Ảnh = `prepared/<book>/pages/*.png` (đúng ảnh pipeline thấy:
LVT stretch, KVK otsu-giữ-xám); biến thể `otsu` (nhị phân hoàn toàn trên JPG gốc) đo thêm.

| chỉ số (ô verified, ảnh prepared) | LVT legacy@0,15 | LVT @0,2 | LVT pitch | KVK legacy@0,15 | KVK @0,2 | KVK pitch |
|---|---|---|---|---|---|---|
| n ô | 3.412 | 3.412 | 3.412 | 3.527 | 3.527 | 3.527 |
| miss (không hộp IoU ≥ 0,3) | 0,5 % (17 dưới ngưỡng, 1 vắng) | 2,2 % | 0,1 % | 3,0 % (90 dưới ngưỡng, 15 vắng) | 8,1 % | 0,4 % |
| extra / 100 ô (dup : stray) | 2,0 (7 : 60) | 1,3 | 0,1 (0 : 5) | 2,3 (12 : 70) | 1,4 | 0,4 (0 : 13) |
| IoU ≥ 0,5 | 97,4 % | 95,8 % | 98,1 % | 94,1 % | 89,5 % | 97,0 % |
| IoU trung vị | 0,75 | 0,75 | 0,75 | 0,74 | 0,74 | 0,75 |
| \|dy\| tâm: trung vị px / % bước · p90 % bước | 10,0 / 6,6 · 17,0 | 10,0 / 6,6 | 9,8 / 6,4 · 16,3 | 7,0 / 5,5 · 16,1 | 6,5 / 5,4 | 6,7 / 5,4 · 15,4 |
| mực chạm mép hộp thô > 0,20 ("cắt vào thân chữ") | 11,0 % | 10,5 % | 9,5 % | 3,8 % | 3,2 % | 2,6 % |
| hộp/bước (h/p) trung vị | 1,14 | | | 1,11 | | |

Mức cột (270 cột/sách): LVT `n_det − N` @0,15: −1: 24 · 0: 172 · +1: 55 · +2: 12; KVK: −1: 49 · 0: 147 · +1: 48 · +2: 7. Tầng
đúng số hộp: LVT 77,4 %, KVK 71,1 %. Cột có hộp NGOÀI hai tầng: LVT 0,4 %, KVK 0,0 % → **số câu in không phải nguồn +1**.
KVK biến thể nhị phân hoàn toàn (`otsu`) cho I5 65,6 % (prepared 54,4 %) — detector thích ảnh đen-trắng hơn otsu-giữ-xám; chưa khai
thác (ảnh prepared còn dùng để cắt crop; muốn dùng phải tách "ảnh cho detector" khỏi "ảnh để crop" trong engine).

Trường hợp I5 mù (thấy khi so build): `page_0008` cột 7 LVT — tầng trên +1 (hộp trùng điểm 0,24), tầng dưới −1 (chữ đầu chỉ 0,10)
→ `n_det = 14 = N`, I5 "đạt", nhưng legacy gán 5 hộp lệch một chữ; pitch chọn đúng (bỏ hộp trùng, dùng hộp 0,10 làm `detector_low`).

## 2. Phương án A — giải mã theo bước cột với ràng buộc số lượng (đã hiện thực, đã đo)

Mô-đun `pipeline/align_engine/char_detector/pitch_decode.py` (selftest 22/22:
`.venv/bin/python -m pipeline.align_engine.char_detector.pitch_decode --selftest`). Bật bằng khoá config **theo sách**
`books[].box_decoder: pitch` (mặc định `legacy` = hành vi cũ, STT không khai → labels.csv STT không đổi byte; engine selftest 253/0).

Cách làm trong mỗi **tầng-cột**: tầng = nhóm chữ kim liền nhau; N tầng = `len_odd`/`num−len_odd` của transcriptions (hoặc số chữ kim);
ứng viên = hộp detector ở **0,05** trong cửa sổ x = x_range ± 0,05w, y = tầng ± 0,35 bước, cộng N "ô ảo" cắt theo chiếu mực (điểm −0,05);
quy hoạch động chọn đúng N ứng viên tối đa `Σ điểm − 0,6·Σ((Δy−p)/p)² − 1,0·Σ chồng_y/p − 1,0·Σ max(0, h/p−1,5)² − 0,3·(lệch mép tầng)²`.
Nguồn ô ghi vào `box_source`: `detector` (≥ det_thr) | `detector_low` (0,05 ≤ điểm < det_thr) | `ink_cut` (ô ảo); `count_source` =
`pitch` (gán theo syl_idx) hoặc `pitch_ocr` (khi `n_det == n_ocr ≠ n_qn`: kim và detector đồng ý, QN OCR lệch → N = n_ocr, gán theo
nom_idx — nếu ép n_qn thì mọi hộp sau khe lệch một chữ, đo được ở 11/98 cột). **`n_det` giữ nguyên nghĩa** (hộp thô ở det_thr trong
dải x) nên I5 không biến thành hằng đúng; `seg_backend = detector_centernet_v1+pitch`.

Chạy (config tạm, không đụng config repo):
```bash
sed -e 's#^    det_thr: 0.15#    det_thr: 0.15\n    box_decoder: pitch#' -e 's#output_dir: dataset_LucVanTien1883.*#output_dir: <scratch>/ds#' \
    config/pipeline_LucVanTien1883.yaml > <scratch>/pipeline_LVT_pitch.yaml
.venv/bin/python -m pipeline.align_engine.build_dataset --config <scratch>/pipeline_LVT_pitch.yaml --reseg detector \
    --qd01-cells none --decisions none --use-s3 --two-pass --box-rule syl_index --force --limit 10 --out <scratch>/build_pitch/dataset_out
# log phải có "[align] LucVanTien1883: box_decoder = pitch"; so với build legacy cùng --limit:
.venv/bin/python -m pipeline.remediation --labels <...>/labels.csv --out <scratch>/build_pitch/dataset_out census   # LUÔN --out
.venv/bin/python -m pipeline.tools.enrich_crop_quality --labels <...>/labels_cq.csv --src-root <scratch>/build_pitch/dataset_out
```
Kết quả `--limit 10` LVT (98 cột, 1.361 ô; legacy cùng tham số): tier **y hệt** (tier do DP văn bản quyết, không phụ thuộc hộp);
`count_source` pitch 87 / pitch_ocr 11; `box_source` detector 1.353 / ink_cut 5 / detector_low 3 (legacy: midpoint 52, split 1);
69 ô đổi hộp: 48 midpoint→detector (cột conflict), 11 detector→detector trong cột equal_qn (tâm hộp gần tâm kim hơn: 0,15 → 0,05 h;
1 ô legacy lệch 2,94 h → 0,06 h); F1 cross-col 10 → 8 ô; bleed 154 → 155 (không đổi — bleed do hộp cao 1,14 bước, không do chọn hộp).

**Cổng B4'(a) với pitch** (chưa sửa `mechanism_gates.py`, việc của luồng khác): cổng `n_det ≠ n_qn` vẫn chặn cột như cũ (n_det không
đổi nghĩa). Cổng hợp lý hơn khi bật pitch: hạ `text_only` theo Ô — `box_source ∈ {ink_cut, detector_low}` (ô detector không tự tin) —
thay vì theo cột; ước lượng từ 27 trang: ink_cut + detector_low ≈ 1,5 % (LVT) / 4,1 % (KVK) ô thay vì loại 35–45 % ô như hiện nay.

## 3. Phương án B — huấn luyện detector v2 trên nhãn yếu (kế hoạch + script sẵn, CHƯA chạy dài)

**Nhãn yếu** (`train_crop/build_lithograph_manifest.py`, 24 s → `train_crop/data_lithograph/manifest_{all,train,val,test}.json`, 7 MB, đường dẫn ảnh):
- Thạch bản 268 trang (LVT 105 + KVK 163, ảnh prepared): ô tham chiếu §1 ở 4.550/5.343 tầng (85 %) → **31.776 ô**; loại tầng
  không verified (373) và tầng có ô xấu (420: blank ink < 5 % 249, mực chạm mép > 20 % 203) → 793 vùng **`ignore_boxes`** (trainer tô
  trắng trước khi học, để glyph không nhãn không dạy detector bỏ sót). Số câu in/rác lề không nhãn → học KHÔNG bắt.
- STT 445 trang (`train_crop/detect_manifest.json`, hộp kim/pipeline GOLD/SILVER/SYLLABLE = cách v1) **+ 14.172 bbox REVIEW làm ignore**
  (v1 để REVIEW không nhãn → trần F1 0,84 đã ghi trong bộ nhớ dự án).
- Chia **page-disjoint theo sách**: trong mỗi sách trang 10k+3 → val, 10k+7 → test (litho 215/27/26, STT 356/45/44 trang);
  `--lobo KimVanKieu1884` để giữ nguyên 1 sách làm test (tổng quát hoá liên sách).

**Trainer** `train_crop/train_v2_lithograph.py` (dùng lại loss/vòng train/validate của `train_centernet.py`): init từ v1
(`detector_r34.best.pt`, DCN), AdamW lr 1e-4 cosine; **lấy mẫu cân bằng miền 50/50** (WeightedRandomSampler); augment thêm:
**nền xám** (nén dải xám mực→[20,80], nền→[100,210], mô phỏng JPG gốc nền ~128 mà v1 mù — KVK raw 23,5 %), **kéo dọc ±10 %** (jitter
bước), rồi affine/blur/noise của v1. Loss giữ Focal + 0,1·L1 size + 1,0·L1 offset (không đổi — nhãn yếu, không thêm ràng buộc mới).

**Đánh giá mỗi epoch** (`train_crop/eval_boxes_ref.py`, 44 s cho val 27+45 trang): thạch bản val — IoU ≥ 0,5, miss/extra, |dy| % bước,
**% tầng n == N** (đếm hộp trong tầng, không ép N → không tautology), mực chạm mép; STT val — P/R/F1 @IoU 0,5 (hồi quy).
Mốc v1 trên val (img 1024, thr 0,15): linear — litho ok50 95,0 %, miss 2,1 %, extra 2,6, tầng n==N 77,0 %, cắt 7,1 %, STT F1 0,8767;
**v1 + area (mốc thật, §0) — ok50 98,1 %, miss 0,8 %, extra 0,9, tầng n==N 91,9 %, cắt 6,4 %, STT F1 0,8785.** Trong Kaggle mốc = `epoch 0`
trên bundle (ok50 98,71 / tầng 96,1 / cắt 11,4 — ảnh bundle đã thu 1536 bằng area nên khác ảnh gốc; chỉ so trong cùng cột).
Lưu best theo ok50_litho **với điều kiện** F1_STT ≥ F1_STT(v1+area) − 0,01; **dừng** khi 4 epoch không tăng ≥ 0,2 điểm (không trước epoch 6) hoặc STT tụt.
Nghiệm thu cuối (không phải val, **so với v1+area**): (i) `scripts/measure/detector_transfer.py` với `NOM_DETECTOR_CKPT=<v2>`: STT đối chứng
`stt_control_pct_cols_eq_pipeline_cfg` ≥ 90,1 không giảm; (ii) `box_ref_eval.py --ckpt <v2> --resize area` 27 trang/sách: I5 cột ≥ 79,6 / 74,8,
ok50 pitch ≥ 99,3 / 98,6, và **cắt thân chữ ≤ 10,6 / 2,5 % đồng thời h/p giảm về ≈ 1,0–1,1** (v2 phải sửa kích thước hộp, không chỉ điểm);
(iii) build trọn `apply_v2.sh` → `--suffix _v2`: `detector_low`/`ink_cut` ≤ 97 / 350 ô, GOLD ảnh ≥ 8.733 / 14.165, **bleed ảnh export ≤ 18,0 / 10,5 %**
(mức chốt linear) — điều kiện bleed là thứ v1+area chưa qua được ở LVT (§0).

**Thời gian** (đo 22/09, Mac 10 nhân CPU): img 1024 batch 2 ≈ 2–3 s/ảnh → 1 epoch (2×215 mẫu) ≈ 15–25 phút + eval 40 s; 20 epoch
≈ 5–8 h CPU. Kaggle T4 ≈ 0,3 s/ảnh → 2–3 phút/epoch, 20 epoch ≈ 1 h (đóng gói: `pack_for_kaggle.py` + thêm `data_lithograph/` và ảnh
`prepared/{LucVanTien1883,KimVanKieu1884}/pages`).
```bash
.venv/bin/python train_crop/build_lithograph_manifest.py                      # nhãn yếu (đã chạy)
.venv/bin/python train_crop/eval_boxes_ref.py --ckpt train_crop/detector_r34.best.pt \
    --manifest train_crop/data_lithograph/manifest_val.json --thr 0.15,0.2     # mốc v1 (đã chạy)
.venv/bin/python train_crop/train_v2_lithograph.py --limit 3 --img 512 --epochs 1 --batch 2 \
    --workers 0 --threads 6 --device cpu --eval-pages 2 --out /tmp/v2_smoke.pt  # chứng minh chạy: 5 s (đã chạy; 1024: 12 s/4 ảnh)
.venv/bin/python train_crop/train_v2_lithograph.py --img 1024 --epochs 20 --batch 2 --threads 8 \
    --out train_crop/detector_v2_litho.pt                                       # CHẠY THẬT (5–8 h CPU / 1 h T4) — chưa chạy
NOM_DETECTOR_CKPT=train_crop/detector_v2_litho.best.pt .venv/bin/python scripts/measure/detector_transfer.py --book all --pages 27
NOM_DETECTOR_CKPT=train_crop/detector_v2_litho.best.pt .venv/bin/python scripts/measure/box_ref_eval.py --book all --pages 27
```
Kết quả 1 epoch nhỏ (3+3 trang, img 512) chỉ chứng minh pipeline chạy: loss 3,85 → 2,56, STT F1 0,797 → 0,859 trên 2 trang — **không phải bằng chứng**.

## 4. Cách đo trung thực

1. **Không dùng `n_det == N` khi ép N**: decoder luôn trả đúng N hộp/tầng (hằng đúng). `n_det` trong labels.csv vẫn là số hộp thô ở
   det_thr; chỉ số thay thế là `% tầng n == N` của **hộp thô** (eval_boxes_ref) và, cho hộp đã giải mã, sai số tâm/IoU so ô tham chiếu.
2. **Ô `ink_cut` trùng ô tham chiếu theo cấu tạo** (cùng hàm cắt) → `summary.json` tách `*_honest(det_src_only)` (chỉ ô nguồn
   detector/detector_low): LVT 98,3 %, KVK 97,3 % ok50 — hơn legacy 0,9 / 3,2 điểm; phần miss → 0 của pitch một nửa là "theo cấu tạo".
3. Chỉ số **không phụ thuộc tham chiếu** duy nhất: mực chạm mép hộp thô (`crop_quality.BORDER_INK_MAX` 0,20) — dùng làm chỉ số cuối;
   bleed/stray_ink của crop thật (`enrich_crop_quality`) đo sau tighten/carve, đã bão hoà (không phân biệt được legacy/pitch).
4. Tham chiếu là **proxy**: kim tầng có thể cụt/rộng (x kẹp ±0,5 bước), khe mực yếu nhất có thể là khe trong chữ ⿱ (phạt lệch bước
   1,5 hạn chế); tầng không verified (7–9 %) bị loại nên bảng chỉ nói về 91–93 % tầng "dễ".
5. CI: 27 trang/sách ≈ ±3,5 điểm ở mức cột; mức ô (3.4–3.5 k ô) ≈ ±0,7 điểm.

## 5. "Tốt nhất" nghĩa là gì — và điều không thể

- Trần của mọi phương án ở đây là **ô tham chiếu**, không phải chữ thật: không có người vẽ hộp thì không đo được precision crop;
  mọi số là proxy và phải ghi đúng tên (như §5 PHUONG_AN_TU_DONG). Tự huấn luyện v2 trên nhãn yếu rồi đo bằng cùng loại nhãn yếu
  held-out là **vòng tròn có kiểm soát**: page-disjoint + STT hồi quy + chỉ số không phụ thuộc tham chiếu (mực chạm mép) là ba hàng rào.
- Với phương án A: mức ô 98 / 97 % IoU ≥ 0,5 và ±1 hộp hoà giải bằng bước; phần còn lại (≈ 2–3 % ô lệch > 0,5 IoU, p90 |dy| ≈ 16 % bước)
  là hộp detector lệch nhẹ, chỉ B (học lại kích thước/tâm trên thạch bản) mới sửa.
- Với phương án B: v1+area đã cho tầng n == N thô 91,9 % (val) / I5 thô 77,7 · 77,9 % (toàn sách), ok50 98,1 % — kỳ vọng hợp lý của v2 là
  **thắng v1+area** ở cả ba: ok50 ≥ 98 và > mốc, tầng n==N > 91,9 %, và **cắt + bleed cùng giảm** (hộp học lại kích thước thạch bản, h/p ≈ 1,0–1,1),
  STT F1 ≥ 0,8785 − 0,01. Không hứa hơn: nhãn yếu chia đều-theo-mực có sai số hệ thống ở chữ ⿱ và chữ nhạt.
- Không làm: self-training từ chính hộp detector (vòng tròn không kiểm soát), sửa `run_pipeline.sh`, đổi STT, commit.

## 6. Huấn luyện v2 trên Kaggle (gói sẵn, 0 sửa tay) — `lab/i5_detector_v2/README.md`

6 bước: `make_bundle.py` (bundle 777 ảnh 1536 px + manifest nhãn yếu + v1 + mã `i5v2/` + notebook, zip 243 MB) → upload Dataset → import
`kaggle_train_i5_v2.ipynb` (T4, Persistence Files only) → Run All (20 epoch ≈ 1–1,5 h; `epoch 0` = mốc v1+area trên bundle; `best.pt` chỉ ghi khi
vượt mốc và qua guard STT) → tải `best.pt` → `apply_v2.sh ~/Downloads/best.pt` (cài ckpt + ghi `detector_ckpt`/`detector_resize: area` vào 4 config,
`box_ref_eval` ×3 v1 · v1+area · v2+area, build `--suffix _v2`, `compare_builds.py`). Mục tiêu số, cách đọc `metrics.csv`, sự cố, hoàn nguyên: README đó.
Mốc trên máy: `--eval-only` v1 bundle = `v1_baseline_val.json` (area) / `v1_baseline_val_linear.json`.

## 7. Tệp

`pipeline/align_engine/char_detector/pitch_decode.py` (mới) · `pipeline/align_engine/{book_layout,align_production,build_dataset}.py`
(khoá `box_decoder`, `pitch_target_count`, `assign_boxes_pitch`, PASS 1b) · `scripts/measure/box_ref_eval.py` (mới, đăng ký bước `box_ref`
trong `measure.py`) · `measure_out/box_ref/{summary.json,box_ref_cells.csv,box_ref_columns.csv,debug_*.jpg}` ·
`train_crop/{build_lithograph_manifest,train_v2_lithograph,eval_boxes_ref}.py` (mới) · `train_crop/data_lithograph/` (manifest, không ảnh) ·
**22/09 tối**: `book_layout.py` (khoá `detector_ckpt`, `detector_resize`, `resolve_detector_ckpt`), `align_production.py` (`DETECTOR_CKPT/RESIZE`, cache
`(ckpt, resize, thr)`, `detector_backend_name`), `build_dataset.py` (fail fast + gán/khôi phục theo sách), `char_detector/detector_infer.py` +
`train_crop/infer_centernet.py` (`resize`), `scripts/measure/box_ref_eval.py` (`--ckpt`, `--resize`) · `lab/i5_detector_v2/` (bundle, notebook,
trainer Kaggle, `apply_v2.sh`, `set_book_keys.py`, `box_ref_compare.py`, `compare_builds.py`, `v1_baseline_val*.json`) · `measure_out/box_ref_area/` ·
`dataset_out_<Book>[_b1]_area/`, `dataset_<Book>_area/` (bản so sánh, không chốt).
