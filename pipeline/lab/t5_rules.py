"""T5 — LUẬT QUYẾT ĐỊNH TIỀN ĐĂNG KÝ cho 3 thí nghiệm từ điển/tier.

Commit tệp này TRƯỚC khi nhìn bất kỳ kết quả nào, y như đã làm ở T3 (`sweep_t3.decide`)
và T4 (`crop_grid.decide`). Ở cả hai lần, luật viết SAU khi thấy số đã suýt cho ra kết
luận ngược, và cái cứu lại là luật viết TRƯỚC.

BA CẠM BẪY ĐÃ TỪNG LÀM SẬP T3/T4, NAY KHOÁ TRƯỚC
------------------------------------------------
1. THƯỚC ĐO SUY BIẾN — tối ưu ra nghiệm biên. T3: trục sản lượng chỉ tăng theo một
   chiều. T4: `flag_ok` tăng đơn điệu tới pad 0,60. Luật: mọi trục LIÊN TỤC phải chứng
   minh có CỰC ĐẠI NỘI trên dải NỚI RỘNG thì mới được dùng để chốt; không thì trục đó
   bị KHOÁ ở mốc và phải đổi dụng cụ đo.
2. CHỈ SỐ VÒNG TRÒN — chấm điểm bằng chính thứ sinh ra dữ liệu. Nguy cơ lớn nhất của
   T5: đo "chất lượng" nhãn GOLD bằng từ điển, trong khi nhãn GOLD DO từ điển sinh ra.
   Luật: thước đo chất lượng phải ĐỘC LẬP với luật sinh ra nhãn đang chấm.
3. NHẦM CẬN VỚI GIÁ TRỊ — T4 tôi đặt ngưỡng phân loại nằm ngoài dải hình học nên "0%"
   là hằng đúng. Luật: trước khi báo một con số, phải nêu DẢI GIÁ TRỊ KHẢ DĨ của nó và
   xác nhận kết quả không nằm ở biên của dải đó vì lý do cấu trúc.
"""
from __future__ import annotations

# =============================================================================
# THÍ NGHIỆM 1 — ĐỘ GIÒN TỪ ĐIỂN (bỏ ra 10% × 10 lần)
# =============================================================================
# CÂU HỎI: bao nhiêu ô trong bộ GIAO NỘP (GOLD 50.156 + SYLLABLE 6.753) mất nhãn khi
# mất một phần từ điển?
#
# CỔNG HỢP LỆ (phải qua trước, nếu không thì KHÔNG công bố số nào):
#   G1.1 Phản thực phải TÁI LẬP ĐƯỢC. Nếu chỉ tính được CẬN thì mọi con số phải ghi
#        kèm chữ "cận trên"/"cận dưới" và độ rộng, không được nói như giá trị chính xác.
#   G1.2 Định nghĩa `fragility` phải PHÂN BIỆT ĐƯỢC. Nếu > 95% ô cùng một giá trị thì
#        định nghĩa đó VÔ DỤNG, phải loại và nói rõ, không được xuất thành cột.
#   G1.3 Định nghĩa không được VÒNG TRÒN. Đếm số lần cặp xuất hiện TRONG BỘ NHÃN là
#        vòng tròn nếu bộ nhãn do chính luật từ điển sinh — phải nêu rõ và xử lý.
#
# KẾT LUẬN CHO PHÉP: chỉ được đề xuất cột `fragility` khi qua cả G1.1-G1.3 và định
# nghĩa thắng tách được ít nhất 2 nhóm có kích thước >= 1% bộ giao nộp.
FRAGILITY_MIN_DISCRIMINATION = 0.95   # >95% cùng giá trị => loại định nghĩa
FRAGILITY_MIN_GROUP = 0.01            # nhóm nhỏ nhất phải >= 1% bộ giao nộp

# =============================================================================
# THÍ NGHIỆM 2 — NGƯỠNG syllable_gate (5 / 3 / 0,6 hiện hành)
# =============================================================================
# CÂU HỎI: ngưỡng hiện tại nằm đâu trên đường cong đánh đổi sản lượng-chất lượng?
#
# THƯỚC ĐO CHẤT LƯỢNG ĐƯỢC PHÉP DÙNG — và CHỈ một thước đo này:
#   tỷ lệ cặp (chữ, âm) qua cổng mà TỪ ĐIỂN cũng công nhận (chữ nằm trong qn_to_nom[âm]).
#   Đây là phép kiểm GIỮ-LẠI hợp lệ vì `syllable_gate` KHÔNG hề đọc từ điển: nó chỉ
#   đếm tần suất/trang/độ thuần trên các ô REVIEW. Nên từ điển là trọng tài ĐỘC LẬP.
#   (Ngược lại, chấm GOLD bằng từ điển thì VÒNG TRÒN — GOLD do từ điển sinh.)
#
# CỔNG HỢP LỆ:
#   G2.1 Phải tái lập được 6.753 ô SYLLABLE hiện hành bằng ngưỡng hiện hành. Lệch quá
#        2% thì DỪNG: nghĩa là quét ngoại tuyến không nói được gì về production.
#   G2.2 Phải có ĐÁNH ĐỔI thật: tồn tại tổ hợp làm sản lượng tăng VÀ chất lượng giảm.
#        Nếu một tổ hợp trội tuyệt đối ở CẢ HAI thì mốc hiện tại sai hiển nhiên — vẫn
#        báo, nhưng phải nghi ngờ phép đo trước, vì trội tuyệt đối là dấu hiệu lỗi.
#
# LUẬT ĐỔI NGƯỠNG (phải thoả HẾT):
#   (a) chất lượng (tỷ lệ từ-điển-công-nhận) KHÔNG giảm quá 1 điểm phần trăm so với mốc
#   (b) sản lượng tăng >= 5% số ô SYLLABLE (>= ~338 ô) — dưới mức đó không đáng đổi
#       một cấu hình đã đóng băng trong bộ đã công bố
#   (c) lợi thế không đến từ MỘT ngưỡng duy nhất (ép 2 ngưỡng kia về mốc thì vẫn còn
#       >= 30% lợi thế) — điều kiện BỀN, y như T4
# Không thoả đủ -> GIỮ MỐC 5/3/0,6.
SYL_QUALITY_MAX_DROP = 0.01
SYL_YIELD_MIN_GAIN = 0.05
SYL_ROBUST_MIN_SHARE = 0.30

# =============================================================================
# THÍ NGHIỆM 3 — CẦU NỐI top-K (s1_inter_s2_similar, 3.829 ô GOLD)
# =============================================================================
# CỔNG HỢP LỆ QUAN TRỌNG NHẤT:
#   G3.1 "top-K" CHỈ có nghĩa nếu danh sách chữ giống ĐƯỢC SẮP theo độ giống. Nếu
#        SinoNom_Similar.csv không có điểm/thứ hạng thì cắt top-K là cắt NGẪU NHIÊN và
#        thí nghiệm này VÔ HIỆU — phải báo "KHÔNG ĐO ĐƯỢC", tuyệt đối không thay bằng
#        thứ tự xuất hiện rồi coi như thứ tự độ giống.
#
# NẾU G3.1 KHÔNG QUA thì vẫn còn một câu hỏi HỢP LỆ và đáng giá, làm thay:
#   luật đòi ĐÚNG 1 cầu. Vậy phân bố số cầu ra sao, và bao nhiêu ô đang bị đẩy xuống
#   REVIEW chỉ vì có >= 2 cầu? Đó là sản lượng bị mất do NHẬP NHẰNG, đo được, không
#   cần thứ tự.
#
# LUẬT THU HẸP LUẬT BẮC CẦU (nếu G3.1 qua):
#   chỉ được đề xuất hạ K khi số ô GOLD MẤT đi < 10% của 3.829 (< 383 ô) VÀ có lý do
#   độc lập tin rằng số ô mất đi phần lớn là SAI. Không có bằng chứng độc lập đó thì
#   GIỮ NGUYÊN — vì "thu hẹp" mà không biết mình bỏ đi cái đúng hay cái sai là đánh đổi mù.
BRIDGE_MAX_LOSS = 0.10

# =============================================================================
# ĐIỀU KIỆN CHUNG CHO CẢ BA
# =============================================================================
# Không thí nghiệm nào được phép ĐỔI bộ nhãn đã công bố trong lượt này. Đầu ra tối đa
# là (a) một cột THÔNG TIN THÊM (`fragility`) không đổi tier của ai, và (b) khuyến nghị
# ghi lại để bạn quyết. Đổi tier là đổi bộ giao nộp, phải chốt MỘT LẦN trước KHỐI 6 —
# cùng ràng buộc đã áp cho T4.e.
ALLOWED_OUTPUT = ("cot_thong_tin_them", "khuyen_nghi_ghi_lai")
