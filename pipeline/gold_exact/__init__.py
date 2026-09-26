"""pipeline.gold_exact — đánh giá tự động MỌI ô GOLD theo cả hai vế (ảnh + chữ), cấu hình "lai" + luật A.

Nguồn thiết kế (đã đo, không sáng tác lại): lab/thu_nghiem_anh_chu/TN4_crop_chuan/KET_QUA.md §1/§5/§7,
TN3_tat_ca_bo (t02_matrix: CNT, BC, H1..H4, TA_OK), TN1_cong_kiem (ngưỡng C5/C2 LOBO τ = 0,995),
chính sách v3 (luật A + M-OCR). Gói CHỈ HẠ ô / gắn trạng thái — không sửa nhãn, không ghi dataset/.

Trạng thái (luật đầu tiên khớp thắng):
  review       luật A (ảnh của ô khác, rescue, cầu tự dạng, văn bản yếu) + M-OCR t50 (như policy v3)
  text_only    ảnh không phải đúng một chữ / lệch khe: A0_new ∪ B0_new (crop chuẩn) ∪ CNT ∪ BC ∪ core_loss
  uncertified  không qua bộ kiểm ảnh↔chữ (ngưỡng TN1, chấm crop CŨ) hoặc thiếu chứng dị bản người (TA)
  ok           còn lại — ảnh giao = CROP CHUẨN v2

Mô-đun: common (đường dẫn, tài sản + sha256, cache, từ điển V1+), crop_chuan (port nguyên cclib_v2),
signals_geom, signals_img, signals_text, policy, eval_ihr (đo trên nhãn người IHR), compare_ref (đối chiếu
số tham chiếu TN4), export_assets (dựng models/gold_exact từ thư mục thử nghiệm), selftest, __main__ (CLI).
Chạy: .venv/bin/python -m pipeline.gold_exact --all-dir dataset/_ALL --out <dir> [--device mps] [--selftest]
"""
