# IHR-NomDB gộp — bộ đối chứng ngoài (external benchmark) cho phương pháp gán nhãn

Sinh bởi `scripts/build_ihr_manifest.py` (2026-09-20). Nguồn gốc DUY NHẤT: `data/handwritten/`
(IHR-NomDB, Nov 2020). `manifest.tsv`: 1 dòng = 1 câu thơ = 1 patch cột.

| Cột | Ý nghĩa |
|---|---|
| page_image / col_bbox_xywh | ảnh trang gốc + bbox CỘT (VoTT, đã sắp phải→trái); KHÔNG có bbox từng chữ |
| patch_image / patch_preprocessed / split | patch cột của IHR (5.054 câu có patch; train 4.001 / val 1.000 theo IHR) |
| nom_text / nom_unicode / n_nom | nhãn Nôm chuẩn (ground truth) theo thứ tự chữ trong cột; 4,8% ký tự là PUA (chưa có mã Unicode chính thức) |
| qn_verse / n_qn_syll / len_match | quốc ngữ đối dịch câu-với-câu của IHR (95,2% khớp số âm tiết) |
| nomnaocr_label / nna_eq_ihr | nhãn cùng câu trong NomNaOCR/All.txt (5.287 có; 5.231 ≡ IHR) — chứng minh 2 bộ là MỘT nguồn |

Tổng: 5.303 câu · 266 trang · 37.135 chữ · 3.556 chữ phân biệt (99,2% có trong dict/QuocNgu_SinoNom.csv).
Lưu ý: cả hai sách đều là **bản in mộc bản**, không phải viết tay; độ phân giải ~35 px/chữ (STT ≈ 110 px/chữ).
