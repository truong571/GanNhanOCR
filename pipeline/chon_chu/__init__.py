"""pipeline.chon_chu — bước 4b CHỌN CHỮ BẰNG ẢNH (30/09/2026, thực thi chính sách TN8; 0 API).

Nguồn thiết kế + số đo: measure_out/_tn8/KET_QUA.md (mã lab/thu_nghiem_kim/TN8_chon_chu/t00…t17). Bước này là bản chép của
các hàm đã đo ở đó — tái lập ĐÚNG dự đoán TN8 (|ΔP| ≤ 3·10⁻⁷, cùng top-1) trên B34/L16 (kiểm khi cài).

Vị trí: run_pipeline.sh run_new_book  B4 gates -> [B4b chon_chu] -> B5 export (đường STT: sau rescue, TRƯỚC export, TẮT).
Đầu vào/ra: labels_gated.csv sửa TẠI CHỖ (bản trước: labels_gated_truoc_chon_chu.csv, chạy lại không cộng dồn) + báo cáo
<dataset_out>/chon_chu_report.json. labels.csv giao nộp vẫn 12 cột; vết ở rule (hậu tố `|chon_chu:<đòn>`) + labels_trace.csv
(chon_chu, chon_chu_p, chon_chu_truoc).

Thành phần:
  crops.py     crop mọi ô đúng hình học save_crop (pad 0,12, carve, tighten, ảnh gốc)
  embed.py     MultiEnc v1+v2 + CNN kiểm ảnh↔chữ (models/gold_exact, kiểm sha) — cache prepared/_chon_chu/<Sách>/emb
  features.py  ứng viên R(âm) ∪ {kim} (∪ {lt2}); 35 đặc trưng: glyph font/FD, head v1, nguyên mẫu cùng sách (khối trang khác),
               nguyên mẫu người Borg (LOBO), CNN kiểm, tiên nghiệm âm→chữ, cờ kim
  model.py     tầng 1 logit có điều kiện + tầng 2 logistic (46 đặc trưng) — tham số models/chon_chu/chooser_<tên>.npz
  policy.py    config/chon_chu.yaml (kiểm LOBO: không vft trên Borg, không nhãn người của chính sách), quyết định nâng/L5, STT bảo thủ
  stt.py       chữ kim lt2 của ô STT (chỉ dùng khi bật STT)
  export_assets.py  dựng models/chon_chu + MANIFEST (sha256, lệnh dựng lại)
  selftest.py  python -m pipeline.chon_chu --selftest

Đòn bẩy: L1 crop_bad · L2 qn_count_unfixed · L2b cầu/confusion_fix/direct khác · L4 kim ∉ R (SYLLABLE, no_context, low_posterior,
syl_ctx|crop_bad) · L5 sửa nhãn GOLD (viết tay) · L3 STT bảo thủ (lt2 ∧ ảnh). Họ in: chỉ L1 + confusion_fix.
"""
