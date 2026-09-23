# Kế hoạch commit CUỐI — siết theo hướng ưu tiên precision (23/09/2026)

Nền: HEAD `e1a7d6a572` (vòng 9 đã commit). **Vòng này chưa commit gì** — tôi không commit, người dùng commit.
Tài liệu số liệu: `docs/CHOT_CUOI_2026-09-23.md`.

> 🔴 **Ba thứ TUYỆT ĐỐI không đưa vào commit này**
> 1. `nom-embed` (submodule, trạng thái `m`) — **không đụng**.
> 2. **Mọi `*.pt`** — `train_crop/detector_r34_v2_litho.pt` (86 MB) và `lab/i5_detector_v2/last.pt`
>    (330 MB). `.gitignore:37` đã có `*.pt`; kiểm bằng
>    `git check-ignore -v train_crop/detector_r34_v2_litho.pt` (phải in `.gitignore:37:*.pt`).
> 3. `dataset/`, `prepared/`, `measure_out/` — đã gitignore; **không** `git add -f`.

---

## 1. Danh mục tệp ĐÍCH DANH — 2 `M` + 1 `M` (tự sinh) + 2 tệp mới

| Tệp | Δ | Nội dung |
|---|---|---|
| `config/pipeline_Chrestomathie1872.yaml` | **+10 −0** | thêm `books[].qn_count_gate: review` + 9 dòng chú thích nêu **hiệu chuẩn trên nhãn người** (50,93 % / 49,06 %), **giá phải trả** (GOLD ảnh 5.998 → 4.229; ảnh export 6.854 → 4.978) và **giới hạn** (cờ văn xuôi tính theo `dp_ratio`, khác cơ chế cờ thạch bản) |
| `config/pipeline_LucVanTien1916.yaml` | **+8 −1** | `detector_resize: linear → area`; thêm `detector_ckpt: train_crop/detector_r34_v2_litho.pt` + 6 dòng chú thích ghi rõ **đảo quyết định vòng 9** kèm số (97,96 → 98,04 %, n 8.629 → 11.496, tập chung 0/8.271 ô đổi nhãn, McNemar p = 1) |
| `docs/PIPELINE_FACTS.json` | **+6 −19** | **tự sinh** bởi `scripts/measure/code_facts.py --check` — chỉ cập nhật `generated_at`, `git_head 1c44d200a9 → e1a7d6a572` và xoá mục `dirty_pipeline_core` (nay rỗng, vì vòng 9 đã commit). **Không sửa tay.** |
| `docs/CHOT_CUOI_2026-09-23.md` | **mới, 180 dòng** | bảng 8 bộ, 13 đánh đổi đã rà kèm số và quyết định, nghiệm thu, giới hạn trung thực |
| `docs/KE_HOACH_COMMIT_CUOI_2026-09-23.md` | **mới** | tệp này |

**KHÔNG đổi**: `config/pipeline.yaml` (**STT — cấm đụng**), `config/pipeline_LucVanTien1883.yaml`,
`config/pipeline_KimVanKieu1884*.yaml`, `config/pipeline_TruyenKieu1872.yaml`, toàn bộ `pipeline/`,
`core/`, `scripts/measure/`, `run_pipeline.sh`, `.gitignore`. Vòng này **không sửa một dòng mã nào** —
toàn bộ thay đổi nằm trong 2 khoá cấu hình đã có sẵn cơ chế đọc (`mechanism_gates.qn_count_mode`,
`books[].detector_ckpt`).

---

## 2. Lệnh commit đề nghị

```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
git add config/pipeline_Chrestomathie1872.yaml config/pipeline_LucVanTien1916.yaml \
        docs/PIPELINE_FACTS.json docs/CHOT_CUOI_2026-09-23.md docs/KE_HOACH_COMMIT_CUOI_2026-09-23.md
git status --short            # phải CHỈ còn ` m nom-embed` ngoài 5 tệp đã add
git commit -F - <<'MSG'
feat(chốt cuối): siết 2 cổng theo ƯU TIÊN PRECISION + đo công bằng v1↔v2 trên cùng tập ô

Phát hiện gốc: detector KHÔNG tạo và KHÔNG sửa nhãn — trên cùng tập ô, v1 và v2 cho
nhãn giống hệt nhau (0/12.341 TK1872, 0/8.271 LVT1916, 0/11.915 LVT1883, 0/20.300 KVK;
McNemar 0↔0, p = 1). Chọn detector vì vậy là câu hỏi coverage ẢNH, không có rủi ro precision.

- LucVanTien1916: ĐỔI SANG detector v2 + area (đảo quyết định vòng 9, vốn dựa trên chỉ số
  hộp I5). Nhãn NGƯỜI: 97,96 % (n 8.629) -> 98,04 % (n 11.496); GOLD ảnh 8.633 -> 11.498;
  3.225 ô v2 thêm vào đúng 98,23 % (cao hơn tập chung 97,97 %).
- Chrestomathie1872: bật qn_count_gate: review. Hiệu chuẩn trên 2 bộ IHR có nhãn người —
  ô trong cột "đếm âm QN hỏng" mà VẪN thoả đúng luật GOLD s1_inter_s2_direct chỉ đúng
  50,93 % (n 216) / 49,06 % (n 53), phần còn lại 97,96 / 98,61 %. Giá: GOLD ảnh
  5.998 -> 4.229, ảnh export 6.854 -> 4.978.
- GIỮ NGUYÊN (đo được là siết không đem lại điểm precision nào): (a') n_det_mismatch trong
  GOLD (98,60 vs 98,67 % và 98,04 vs 98,70 %), (d) cross chỉ hạ ô gần hình (100 % ô bất
  đồng dị bản còn lại là dị thể CÙNG ÂM; lỗi gần hình trên nhãn người = 0/18.550 và
  0/11.496), tầng GOLD_text_only (96,34 % / 99,23 %, CI chồng GOLD-ảnh), detector v2 của
  TK1872 / LVT1883 / KVK1884.

Nghiệm thu: hồi quy STT md5 59e436d7641fa849bb6759868ac29259 (555 dòng × 42 cột);
selftest 157/288/81/113/82/33/22/19/24 đều 0 FAIL; code_facts 18/18;
measure.py --all --report-only 170 PASS / 0 FAIL; align_audit FAIL 1/1/0/2/2 y hệt vòng 9;
crop_source_samples 8 ảnh 0 FAIL; git status -- data dataset_out prepared/SachThanhTruyen* trống.

Số liệu đầy đủ: docs/CHOT_CUOI_2026-09-23.md

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
```

---

## 3. Kiểm trước khi commit (5 lệnh, ≈ 3 phút)

```bash
PY=.venv/bin/python
git check-ignore -v train_crop/detector_r34_v2_litho.pt lab/i5_detector_v2/last.pt   # phải in .gitignore:37:*.pt
git status --short -- data dataset_out 'prepared/SachThanhTruyen*'                   # phải TRỐNG
$PY -c "import yaml; [yaml.safe_load(open(f)) for f in ('config/pipeline_Chrestomathie1872.yaml','config/pipeline_LucVanTien1916.yaml')]; print('yaml ok')"
$PY -m pipeline.remediation.mechanism_gates_selftest | tail -1                        # 113 passed, 0 failed
$PY scripts/measure/code_facts.py --check | tail -1                                   # invariants_pass 18/18
```

---

## 4. Dọn dẹp sau khi commit (KHÔNG bắt buộc, tiết kiệm ~525 MB)

Các bản dựng đối chứng của vòng này **không** nằm trong git và có thể xoá; lệnh dựng lại nằm ở
`docs/CHOT_CUOI_2026-09-23.md` §5 "Tái lập" bước (3).

```bash
rm -rf prepared/TruyenKieu1872/dataset_out_v1probe   # 208 MB — bản v1 của TK1872 (phép so công bằng)
rm -rf prepared/LucVanTien1916/dataset_out_v2probe   # 180 MB — bản v2 của LVT1916
rm -rf dataset/TruyenKieu1872_v1probe                #  62 MB
rm -rf dataset/LucVanTien1916_v2probe                #  75 MB
# measure_out/<book>/ihr_endtoend_{v1probe,v2probe}/ : GIỮ LẠI (nhẹ, là bằng chứng của §1–§2)
```

**GIỮ** `dataset/<Book>_v9/` (bản trước khi siết) cho tới khi người hướng dẫn duyệt xong — đó là
bản duy nhất còn 1.769 ô GOLD-ảnh của Chrestomathie và bản v1 của LucVanTien1916.
`dataset/<Book>_v5_lang1/`, `_v7/`, `_v8/` là hồ sơ ablation của các vòng trước, xoá được nếu cần chỗ.

---

## 5. Việc CÒN MỞ — không thuộc commit này

1. **Chrestomathie1872 cần detector riêng** (v2 làm I5 tụt 67,6 → 50,6 % vì văn xuôi không nằm trong
   tập train). Chừng nào chưa có, sách này giữ v1 linear và "cắt thân chữ" 78,3 %.
2. **`truncated` tăng ở 2 bộ IHR khi dùng v2** (TK1872 767 → 1.195, LVT1916 700 → 1.065). Các ô ấy ở
   REVIEW nên không giao nộp; hướng cứu là nới `det_xmargin` riêng — chưa thử.
3. **Đề xuất #3 của `RA_SOAT_CAN_CHINH_2026-09-23.md`** (cứu 82 cột ≈ 1.070 ô bằng cách nới luật sửa
   đếm âm QN) là việc **tăng coverage**, cố ý hoãn. Nghiệm thu bắt buộc nếu làm: ô được cứu phải đạt
   **≥ 95 %** trên nhãn người IHR.
4. **Train lại detector từ `best_litho.pt`** thay vì `last.pt` (epoch 6) — bảng Kaggle cho thấy epoch 3
   tốt hơn ở `litho_tiers_eq_015`.
5. **Khai trong luận văn**: LucVanTien1916 được chỉnh dựa trên chính bộ ĐÁNH GIÁ của nó (chọn siêu
   tham số trên tập đánh giá). Đây là giới hạn phương pháp, phải nói rõ, dù §1 của
   `CHOT_CUOI` chứng minh detector không chạm vào nhãn nên rủi ro khớp nhiễu là rất thấp.
