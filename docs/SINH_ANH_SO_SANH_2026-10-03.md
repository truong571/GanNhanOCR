# Sinh ảnh tương đồng rồi so sánh ảnh — hiện trạng, thử nghiệm TN11, thẩm định báo cáo HQC (03–04/10/2026)

Mọi số do script sinh (`lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/`, đầu ra `measure_out/_tn11/`); 0 gọi API kim. Sự thật chỉ dùng để ĐO: chữ người
Borg (B18 = Kinh Thầy cả Bỉnh, B34 = Dung lý hộ thân) và GT IHR (L16, TK); mô hình không bao giờ học trên nhãn người của sách đang đo.
Các tệp `p03_truc_tiep_sinh_so_sanh.py`, `p04_he_quy_chieu_mau_giay.py`, `p05_full_corpus_he_quy_chieu.py` cùng thư mục do MỘT PHIÊN KHÁC ghi
(không thuộc dãy p01–p09 của tài liệu này); xem §5.

## 0. Hai lỗi nền phát hiện 04/10 — ảnh hưởng cách đọc mọi số về FD

| # | Lỗi | Bằng chứng | Hệ quả | Việc cần làm |
|---|---|---|---|---|
| 1 | Kho ảnh FD trên máy chỉ có con trỏ | `gannhanocr-fd` là submodule HF (`mdnt571/gannhanocr-fd-cache-v2`); 89.813/89.898 tệp là con trỏ Git-LFS (129 B); `Glyphs` bỏ tệp < 1 KB ⇒ pipeline chỉ nạp **1.646** ảnh FD thật (1.562 ở `ArcFace/data/glyphs` + 85) | f_fd = NaN ở ≈ 80 % ứng viên; ảnh FD chỉ phủ chữ thường gặp (48–70 % chữ đúng nhưng 15–20 % mọi ứng viên) ⇒ f_fd lẫn **tiên nghiệm tần suất** với so ảnh. Số thô "FD hơn phông 22–26 điểm" ở Borg là hiệu ứng độ phủ | Đã thử kéo đủ kho vào thư mục tạm (TN12, §4): **không nâng bộ chọn chữ** (|Δ| ≤ 0,12 điểm) ⇒ không cần `git lfs pull` trong repo để cải thiện nhãn; chỉ cần ghi đúng trong tài liệu rằng pipeline dùng 1.646 ảnh FD. visual_dp/gold_exact chưa thử (đã gần trần) |
| 2 | Wrapper sinh ảnh dựng FST ngẫu nhiên | `core/ranking/fontdiffusion_gen.py:76` đặt `use_fst=False` (commit da8d15f4f3, 06/05) nhưng `:101` vẫn gọi `use_fst=True` (commit 8e985d9845, 15/04); log: "Checkpoint for 'fst_module' not found" | `mss_encoder` + `fst_module` (≈ 68 triệu tham số) khởi tạo ngẫu nhiên ⇒ truyền phong cách yếu (chênh mực do ảnh phong cách +5,0 so với +17,9 ở đường chuẩn; nội dung chữ giữ). Mọi ảnh sinh p02/p03/p04 đi đường này; kho FD dựng trên Kaggle nhiều khả năng cùng khiếm khuyết (chưa có log để kiểm) | Dùng `fd_wrapper_fix.py` (không sửa `core/`); sửa một dòng trong `core/` nếu đồng ý |

## 1. Pipeline hiện tại đã "sinh ảnh rồi so ảnh" như thế nào

Với mỗi ô, từ âm Quốc ngữ lấy tập ứng viên R(âm) (≈ 26 chữ, `Dict/QuocNgu_SinoNom.csv`). Với mỗi ứng viên c:

1. **Ảnh mẫu của c**: render phông (NomNaTong → Plangothic P1/P2) và ảnh FontDiffuser sinh sẵn (chỉ 1.646 ảnh thật trên máy này, §0).
2. **So ảnh**: crop và ảnh mẫu cùng qua encoder ResNet-18 ArcFace (v1 + v2) → vector 512 chiều (ghép hai encoder) → cos.
3. Điểm so ảnh dùng ở ba chỗ: gióng hộp `visual_dp` (phát xạ = max cos crop–glyph trong R(âm)), bộ chọn chữ `pipeline/chon_chu/`
   (cùng nguyên mẫu ảnh thật cùng sách, nguyên mẫu người sách kia, CNN kiểm), và kiểm cuối `gold_exact` (chỉ hạ).

## 2. Đo hiện trạng (Top-1 khi chỉ dùng MỘT tín hiệu, mọi ô có chữ đúng, ≈ 26 ứng viên/ô)

| Tín hiệu so ảnh | B18 | B34 | L16 | TK |
|---|---:|---:|---:|---:|
| crop ↔ glyph phông in | 43,0 | 34,6 | 63,3 | 62,5 |
| crop ↔ ảnh FD sinh sẵn — số thô (ứng viên không có ảnh = −9) | 65,4 | 60,4 | 49,8 | 48,2 |
| cùng tập ứng viên có ảnh FD: phông / FD (p08) | 66,3 / 72,4 | 60,3 / 66,6 | 85,4 / 81,4 | 84,9 / 81,7 |
| FD − phông cùng tập (CI95 % cụm trang) | +6,1 [5,7; 6,4] | +6,4 [5,5; 7,2] | −4,0 [−4,8; −3,2] | −3,2 [−3,9; −2,6] |
| chiếu Ridge theo sách (p01) | 70,3 | 61,9 | 82,8 | 78,6 |
| nguyên mẫu ảnh THẬT cùng sách (ô neo tự động, khối trang khác) | 77,7 | 67,5 | 81,5 | 84,0 |
| nguyên mẫu ảnh người sách Borg kia | 85,8 | 84,2 | — | — |
| **Bộ chọn chữ đầy đủ (TN8, LOBO)** | **94,6** | **94,1** | **97,5** | **98,4** |

Lỗi của bộ chọn dồn vào **chữ hiếm** (chữ đúng chưa có ô neo trong sách): B18 89,7 %, B34 87,2 % so với 96,8 / 98,7 % ở chữ có ≥ 5 mẫu thật.
Đây là chỗ duy nhất ảnh sinh có thể thay ảnh thật — nên các thử nghiệm sinh ảnh đo trên chữ hiếm.

## 3. Công nghệ hiện đại liên quan (đã mở trang gốc xác minh)

| Hướng | Công trình | Ý chính | Hợp với đề tài? |
|---|---|---|---|
| Sinh phông few-shot | FontDiffuser (Yang et al., AAAI 2024); CF-Font (CVPR 2023); VQ-Font (ICCV 2023); MSD-Font (CVPR 2024); HFH-Font (SIGGRAPH Asia 2024, suy luận 1 bước); IF-Font (NeurIPS 2024, sinh từ chuỗi IDS) | Sinh chữ bất kỳ theo phong cách vài ảnh mẫu | Đang dùng FontDiffuser; IF-Font sinh được chữ thiếu trong phông; HFH-Font nhanh |
| Bắt chước nét viết tay | SDT (CVPR 2023); One-DM (ECCV 2024); DiffusionPen (ECCV 2024) | Sinh chữ viết tay theo người viết | Chủ yếu chữ Latinh / nét online |
| Ảnh sinh thay dữ liệu thật cho lớp chưa thấy | Gui et al. (ICDAR 2023): DDPM học 2.000 lớp đã thấy → lớp chưa thấy 93,0–94,7 % (thật 97,9 %); Ao et al. (ICASSP 2024; Pattern Recognition 2025): hiệu chỉnh nguyên mẫu bằng mẫu sinh; CMPL (Pattern Recognition 2022) | Bộ sinh phải **học trên chữ thật cùng miền** rồi mới khái quát | Đúng bài toán chữ hiếm; cần GPU |
| Giới hạn của FontDiffuser | MegaHan97K (Pattern Recognition 2025) | FontDiffuser sinh 97 nghìn lớp: lỗi đặt nét ở chữ phức tạp; nhân đôi mẫu chữ khó chỉ 92,32 → 92,35 % | Cảnh báo kỳ vọng |
| Diffusion làm bộ phân loại | Li et al. (ICCV 2023); Clark & Jaini (NeurIPS 2023) | Chọn lớp có sai số khử nhiễu nhỏ nhất | Chưa ai áp cho chữ; thử ở p03 |
| Khớp ảnh ↔ cấu tạo chữ | CCR-CLIP (ICCV 2023, ảnh ↔ IDS); stroke-level (IJCAI 2021) | Không cần ảnh mẫu của chữ hiếm | Chữ Nôm phần lớn ghép từ bộ kiện Hán |
| Đặc trưng nền tảng | DINOv3 (arXiv 2508.10104); SigLIP 2 (arXiv 2502.14786) | Đặc trưng ảnh tổng quát | DINOv2 đã đo không phân biệt được chữ Nôm |
| Hán-Nôm | Diesbach et al. (ICFHR 2022, CUT sinh chữ Nôm văn bia) | Sinh phong cách văn bia | Không đo lợi ích cho gán nhãn; chưa thấy công trình diffusion chữ Nôm 2024–2026 |

Liên kết gốc: AAAI 2024 doi 10.1609/aaai.v38i7.28482 · github.com/yeungchenwa/FontDiffuser · github.com/grovessss/HFH-Font · github.com/Stareven233/IF-Font ·
github.com/diffusion-classifier/diffusion-classifier · github.com/SCUT-DLVCLab/MegaHan97K · arXiv 2305.15660 (Gui et al.) ·
github.com/bad-meets-joke/BCCSS (Ao et al.) · github.com/FudanVI/FudanOCR (CCR-CLIP).

## 4. Thử nghiệm TN11 (0 API, không sửa pipeline) — kết quả đã hiệu chỉnh

Bản 03/10 của tài liệu này kết luận "ảnh FD sinh sẵn hơn phông 22–26 điểm" và "ảnh sinh theo sách kém kho FD (34–38 % so với 54 %)": **cả hai bị rút**
vì so không cùng tập ứng viên (§0 lỗi 1). Số đúng ở bảng dưới (p09 = so cùng tập ứng viên + kiểm định cặp).

| Mã | Cách | Đo trên | Kết quả |
|---|---|---|---|
| p01 | Ridge: học phép biến glyph → nguyên mẫu ảnh thật của sách (ô neo tự động khối khác), áp cho chữ chưa có mẫu | mọi ô, Top-1 một tín hiệu | Ridge hơn phông +27,3 / +27,3 / +19,6 / +16,2 điểm (B18/B34/L16/TK); thấp hơn nguyên mẫu ảnh thật cùng sách 5,6–7,4 điểm (Borg) ⇒ tín hiệu thay thế khi thiếu glyph, không phải bước nâng bộ chọn |
| p05 | Thêm Ridge vào bộ chọn chữ TN8 (cùng t06, LOBO) | Top-1 bộ chọn | Không đổi: B18 94,65 → 94,70; B34 94,09 → 94,11; L16 97,48 → 97,45; TK 98,44 → 98,45 |
| HQC | Đồng bộ quang học glyph ↔ crop (nền giấy thật 16 trang/sách, độ nhoè/đậm/độ phân giải chọn trên 250 ô neo tự động; nhị phân hoá cả hai phía), `v_thu_hqc_*` | mọi ô, 4 bộ nhãn người, CI cụm trang | f_font: **+3,39 / +1,53 / +5,83 / +6,69** (CI > 0 cả 4; tái lập với mảng giấy khác +2,4…+5,2); f_fd: +0,32 / −0,61 / +2,03 / +1,12. Cơ chế là **quang học** (nhoè/đậm/độ phân giải), không phải nền giấy (chỉ-giấy ≈ 0). Nhị phân hoá **hại** f_font −7…−11. Thêm vào bộ chọn TN8: Δ ≤ +0,09 (B18 +0,01; B34 +0,09; L16 0,00; TK −0,04) |
| p02 | FD one-shot, ảnh phong cách = ô neo B34 (罪), đường FST ngẫu nhiên | 200 ô, top-8 theo CNN; p09 cùng tập FD (n = 111) | gen ≈ phông ≈ FD: Top-1 63,1 (nét thường) / 64,9 (nét đậm) so với 61,3 / 62,2; Δ so với phông +1,8 / +3,6, CI chứa 0. Ghép với CNN không tăng so với ghép phông (vW+gen − vW+phông = −1,6 trên 188 ô; +0,9 trên n = 111; CI đều chứa 0) |
| p03dc | Diffusion làm bộ phân loại (ICCV 2023), FontDiffuser có sẵn (đường FST ngẫu nhiên) | như p02 | 21 % (ngẫu nhiên 12,5 %); mẫu quá nhỏ để kết luận, chưa chạy lại đường chuẩn |
| p04 | Tinh chỉnh nhẹ (1.200 bước × lô 4 ≈ 0,9 vòng) trên 5.461 cặp ô neo B34 (loại mọi chữ đúng của mẫu đo), đường FST ngẫu nhiên | như p02 | **Xấu đi**: Top-1 55,0 / 45,9 (cùng tập FD) so với 63,1 / 64,9 trước khi tinh chỉnh; nét đậm kém phông −15,3 [−26,1; −4,5], p = 0,0095 |
| p02 chuẩn | Như p02 nhưng `use_fst=False` (`fd_wrapper_fix.py`): 790 ảnh, 3,0 giây/ảnh (nhanh gần gấp đôi đường lỗi) | như p02; p09 | **Ngang phông**: cùng tập FD (n = 111) Top-1 61,3 % (nét thường) / 55,9 % (nét đậm) so với phông 61,3 % và FD 62,2 %; Δ so với phông 0,0 [−8,1; +8,1] / −5,4 [−14,4; +2,7]. Trên 188 ô top-8: +5,9 [−1,1; +12,8] / +6,4 [−1,6; +14,4], không có ý nghĩa thống kê. Ghép với CNN không hơn ghép phông (−0,5 / −3,2, CI chứa 0). Sửa lỗi `use_fst` **không** tạo ra lợi ích đo được |
| TN12 | Kho FD đầy đủ: `git lfs pull` vào thư mục tạm ngoài repo (89.898 ảnh thật, 216 MB), nhúng 32.895 ảnh của mọi chữ ứng viên; f_fd_full phủ 100 % chữ đúng và 96 % ứng viên (trước: 48–70 % và 15–20 %). Hiệu chuẩn: tái tạo f_fd cũ sai khác ≤ 2,2e-6 | 4 bộ nhãn người; bộ chọn TN8 LOBO | So ảnh một tín hiệu, cùng ≈ 22 ứng viên: FD − phông **+4,6 / +5,1 điểm** (Borg) và **−9,6 / −9,7** (mộc bản); Top-1 thô của FD chỉ còn 46,5 / 38,7 / 53,4 / 52,4 % (số cũ 65,4 / 60,4 / 49,8 / 48,2 % phần lớn là tiên nghiệm tần suất do độ phủ). Bộ chọn TN8 với f_fd := f_fd_full: Δ mọi ô +0,01 / −0,11 / +0,02 / −0,03, chữ hiếm +0,05 / −0,12 / +0,09 / −0,09 ⇒ **không đổi** |
| p10 | **Ablation phần sinh ảnh ĐANG chạy**: tắt cos với glyph phông (f_font), với ảnh FD (f_fd) hoặc cả hai trong bộ chọn TN8 (cột = NaN ⇒ đặc trưng hằng số ở cả hai tầng); cùng khung t06 LOBO | 4 cặp LOBO, CI cụm trang | Δ Top-1 mọi ô so với gốc: bỏ FD −0,10…+0,02; bỏ phông +0,02…+0,06 (TK<-L16 +0,06 [+0,01; +0,11]); **bỏ cả hai −0,01…+0,05**; chữ hiếm −0,19…+0,18. Nghĩa là bộ chọn không còn cần cos với glyph sinh/render: thông tin glyph đã vào qua CNN kiểm (bảng W học từ glyph phông) và nguyên mẫu ảnh thật cùng sách/người. Chưa đo cho `visual_dp` (nơi so ảnh với glyph là bắt buộc vì chưa có nhãn nào) và `gold_exact` |

## 5. Thẩm định báo cáo "tổng hợp công nghệ HQC / sinh ảnh thích ứng sách" (dán từ phiên khác)

Quy trình: 4 hướng kiểm chứng song song (nguồn gốc/tái lập, truy nguồn số, kỹ thuật + tài liệu, thử HQC quy mô lớn) → mỗi hướng một người phản biện
độc lập → một bước rà soát chung (9 tác tử; kết quả đầy đủ `measure_out/_tn11/verify/_workflow_result.json`, script `v_*.py`).

**Không dùng được làm bằng chứng**
- Bảng "Top-1 trước → sau" và dòng TỔNG: thực đo **185 ô (0,067 % của 275.136)**, 3 ứng viên/ô (top-3 theo CNN), chọn điều kiện tốt nhất trong 3 cho cả 8/8 bộ,
  6/10 bộ chấm bằng nhãn của chính pipeline (vòng tròn), cột "Trước" trộn 4 loại số (tỉ lệ ô GOLD của Chr/L83, Top-1 toàn quần thể, Top-1 top-3…);
  trung bình đúng là 59,4 → 84,7 (+25,4), không phải 54,3 → 84,7 (+30,4); "Sau" của 4 bộ có nhãn người đều **thấp hơn** bộ chọn TN8 đang chạy (94,6/94,1/97,5/98,4).
- Mẫu lấy bằng `hash(book)` (Python đổi theo từng tiến trình): chạy lại script cho tập ô khác (Jaccard 0). Seed gốc đã khôi phục (10/10 bộ có đúng 1 seed khớp);
  với 10 seed cố định, SD Top-1 mỗi điều kiện 6,0–12,8 điểm — lớn hơn chênh lệch HQC1/2/3 mà báo cáo dùng để kết luận.
- p03: n = 10 ô; "FD chung STT2" thực ra là phông (đọc con trỏ LFS, `cv2.imread` trả None, rơi về phông): baseline không tồn tại; McNemar p = 0,5.
- "Độ chính xác ô neo > 99 %": đo trên nhãn người là Borg 88–91 %, IHR 97,5–98,9 %. "Hồ sơ mẫu giấy đo thực tế": stt2/4/11 `ink = 60` là hằng số mặc định (ảnh chỉ 2 mức xám),
  mảng giấy B34/L16/TK là nhiễu Gauss giả lập; lý do "nền 179 so với 255 làm vector lệch pha" không đứng vững: đổi độ sáng nền phẳng 255 → 179 gần như không đổi embedding (cos 0,9999 khi chuẩn hoá như crop); độ lệch đo được (cos 0,79–0,91 giữa glyph nền trắng và nền giấy) đến từ nhiễu giả lập và từ việc nhúng glyph với `norm=False`, trong khi crop sản xuất đã giãn p2–p98.
- Kỹ thuật sai: K = 3 tâm con **không** ứng với 3 kiểu thư pháp (sub-center ArcFace dùng để hút mẫu nhiễu; 92 % crop về một tâm; v2 là ResNet-18 256 chiều, s = 30, m = 0,30; 512 chiều là
  ghép v1 + v2); "Style-RSI (Residual Style Injection)" — RSI là Reference-Structure Interaction (DCN); MSSE không thuộc bài FontDiffuser; ước lượng "3.000 giờ GPU" phóng đại
  (chữ-theo-sách cần sinh = 244.614 ⇒ 347–747 giờ cho một phong cách); dilate làm nét **mảnh**, không "đậm"; CFG thực chạy ở p03 là 7,5, không phải 2,0.
- "Chứng minh bằng toán học", "đáp ứng toàn diện MI3", "tính mới", "100 % tự động không cần nhãn người", mô tả quy trình 6 bước (Sauvola không chạy; hồ sơ giấy/Ridge W_b/FD thích ứng/HQC
  không có trong `run_pipeline.sh`; chon_chu/gold_exact dùng nhãn người theo LOBO để chọn ngưỡng) — cần viết lại theo bảng trạng thái trong `_workflow_result.json` (khoá `critic.trang_thai`).

**Dùng được (sau khi sửa)**: số kho 10 bộ/1.682 trang/275.136 ô (Chr là thạch bản, không phải chữ chì); khung mức tin (chỉ 4 bộ có nhãn người);
Ridge như tín hiệu đơn (+16…+27 điểm so với phông, nhưng không nâng bộ chọn); quan sát quang học (+1,5…+6,7 điểm ở f_font đơn tín hiệu, không chuyển thành lợi ích ở bộ chọn);
hai lỗi nền §0; kết quả âm có khoảng tin cậy. Ghi chú quy trình: một tác tử ghi đè nhầm tệp scratch dùng chung `scratchpad/draft_findings.py` (không phải tệp repo).

## 6. Kết luận và việc tiếp theo

1. **Phía so ảnh của bộ chọn chữ đã bão hoà**: thêm bất kỳ cải tiến phía glyph nào cũng không đổi Top-1 quá 0,12 điểm — Ridge theo sách (−0,04…+0,05), đồng bộ quang học (≤ +0,09),
   kho FD đầy đủ (−0,11…+0,02). Nhị phân hoá glyph + crop hại (−7…−11 ở tín hiệu đơn). **Không có thành phần nào của báo cáo nên tích hợp vào pipeline.**
1b. **Trạng thái tích hợp (04/10)**: không thành phần TN11 nào (Ridge, quang học, ảnh sinh/tinh chỉnh) được tham chiếu từ `run_pipeline.sh`, `config/`, `pipeline/`, `core/`; chỉ nằm ở `lab/`. "Sinh ảnh" đang chạy trong pipeline = phông render lúc chạy + ảnh FD sinh OFFLINE (`lab/kaggle_diffusion`, chỉ 1.646 ảnh thật nạp được); mã sinh ảnh tại chỗ trong `pipeline/step3_label.py` là mã cũ, không được gọi. Vị trí so ảnh thật: `visual_dp` (bước 3 build), `chon_chu` (4b), `gold_exact` (8). Không bước nào cần train lại khi chạy; trọng số (`nom-embed/best.pt`, `ArcFace/checkpoints/best.pt`, `models/chon_chu`, `models/gold_exact`) chỉ được đọc.
2. **Ảnh sinh theo phong cách sách không hơn phông** (đường chuẩn `use_fst=False`: Δ so với phông 0,0 [−8,1; +8,1] cùng tập FD, n = 111; đường lỗi cho kết quả tương đương), và tinh chỉnh nhẹ làm xấu đi.
   Mẫu nhỏ (một sách, 200 ô) nên chưa bác được hiệu ứng nhỏ, nhưng không có tín hiệu nào đáng đầu tư GPU.
3. Gói Kaggle (`measure_out/_tn11/kaggle/tn11_kaggle_data.zip`) đã dựng lại với `fd_wrapper_fix.py` (bản 03/10 dính lỗi §0) và chạy thử sạch. Với kết quả trên, chỉ nên chạy nếu muốn có
   bằng chứng âm cho "tinh chỉnh FontDiffuser theo sách" trong luận văn; kỳ vọng lợi ích thấp.
3b. **Gói Kaggle đủ 10 bộ, phong cách từng cuốn (04/10, theo yêu cầu bắt buộc của đề tài)**: `measure_out/_tn11/full/kaggle_pack_20261004_v3.zip` + notebook dựng sẵn `kaggle_pack_20261004_v3.ipynb` (cùng thư mục) — 410 việc, 51.730 ảnh, ≈ 38,8 giờ GPU (≈ 3 ca), bộ chạy bền có hàng đợi động/tiếp tục theo lô/đẩy HF mỗi 5 phút (kiểm ghi/đọc HF trước khi chạy)/cổng `use_fst`/cổng GPU/màn hình gọn; ảnh phong cách và danh sách chữ dựng từ nhãn 01/10 không dùng nhãn người (ảnh phong cách chỉ từ ô chưa bị `chon_chu` chạm). Cách làm từng bước: `docs/KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md`; chấm kết quả: `p12_cham_toan_bo.py`. Kỳ vọng thấp (xem trên); chạy để có bằng chứng đủ bộ. Ghi chú về kết quả tinh chỉnh Kaggle lần trước (B18/B34): cặp huấn luyện lấy từ bảng TN8 cũ (chứa 4.022/499 ô neo mà `chon_chu` L5 sau đó đổi nhãn, khoảng 8 %) và đụng quy tắc `evaluation_only` (`TAP_DANH_GIA.md`) — kết luận âm đó có thêm hai nhiễu này.
4. Lỗi hạ tầng cần sửa/ghi nhận: (a) `core/ranking/fontdiffusion_gen.py:101` `use_fst=True` ⇒ `False` (nếu đồng ý; hiện đã có bản sửa `fd_wrapper_fix.py` ngoài `core/`); (b) ghi đúng trong tài liệu hội đồng:
   `docs/TRA_LOI_HOI_DONG_2026-10-01.md:188` ("FontDiffuser chỉ dùng ảnh glyph sinh sẵn (89.898 tệp)" — trên máy này pipeline nạp 1.646 ảnh thật) và `docs/KHAO_SAT_TAI_LIEU_2026-10-01.md:218` (số glyph kho).
5. Chỗ còn sai nhiều là **chữ hiếm** (chữ đúng chưa có ô neo cùng sách: B18 89,7 %, B34 87,2 %); tín hiệu mạnh nhất cho chữ hiếm vẫn là nguyên mẫu ảnh người sách kia (75,9–78,4 %).
   Hướng chưa thử, có tiền lệ trong tài liệu: khớp ảnh ↔ cấu tạo chữ (CCR-CLIP, IDS) cho chữ không có mẫu thật; hoặc bộ sinh được huấn luyện trên nhiều lớp chữ thật cùng miền (Gui et al. 2023) — công sức lớn.
