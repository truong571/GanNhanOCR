# Quy trình chấm tay — mẻ 2026-08-25

> Mẻ đã dựng: `dataset_out/ground_truth/audit_combined/` — **960 ô = 900 mẫu + 60 ô lặp ẩn**.
> Sinh lại y hệt: `python -m pipeline.ground_truth.make_combined_batch --by-rule --n-gold 380 --n-similar 260 --n-silver 0 --n-syllable 260 --n-repeat 60 --seed 2026`

## Vì sao mẻ này khác mọi mẻ trước

Mẻ trước **hỏng ở chiều CROP** (κ = 0,14) chứ không hỏng ở chiều NHÃN, và toàn bộ
"chấm tay" hoá ra là **máy chấm** — nên mọi con số precision cũ đã bị tước tư cách bằng
chứng. Mẻ này sửa ba chỗ đó:

1. **Máy chấm chiều ẢNH, người chấm chiều NHÃN.** Cột `crop_quality_flag` và
   `usable_image` đã có sẵn trong bộ nhãn. Ô `usable_image=0` (424 ô) thì **bỏ qua câu
   hỏi về ảnh**, đừng chấm thành "nhãn sai".
2. **Ba tầng tách bạch khi PHÂN TÍCH, trộn chung khi TRÌNH BÀY.** Người chấm không được
   biết mình đang ở tầng nào — nếu biết, kỳ vọng "tầng này chắc nhiều lỗi" sẽ tự ứng nghiệm.
3. **Hai người, và người thứ hai không phải tác giả.**

## Ba tầng — ba CÂU HỎI KHÁC NHAU

| tầng | dân số | mẫu | câu hỏi đặt cho người chấm |
|---|---|---|---|
| `GOLD/direct` | 46.327 | 380 | Chữ trong ảnh **có đúng là** chữ nhãn không? |
| `GOLD/similar` | 3.829 | 260 | *(như trên)* — **nhưng đây là lớp rủi ro khác** |
| `SYLLABLE` | 6.911 | 260 | **ÂM Quốc ngữ** này có đúng với chữ trong ảnh không? |

**Vì sao phải tách `direct` khỏi `similar`:** `direct` xác nhận **chữ quan sát được**;
`similar` **thay** chữ quan sát được bằng một chữ khác bắc cầu qua "nhìn giống". Sai ở hai
lớp này là **sai kiểu khác nhau**, và gộp lại thì precision của lớp lớn sẽ che lớp nhỏ.
Với 46.327 vs 3.829 ô, gộp lại gần như chỉ đo được lớp `direct`.

**Vì sao `SYLLABLE` hỏi câu khác:** tầng này có cột `label` **RỖNG** — nó không gán chữ Nôm
nào. Hỏi "chữ này đúng không" là hỏi một thứ không tồn tại.

## Ai chấm

**Bắt buộc hai người, cả hai phải đọc được chữ Nôm.** Người thứ hai **không được là tác giả**.

Một người tự nhất quán vẫn có thể **sai hệ thống theo cùng một hướng ở cả hai lần** — nên
κ nội tại (test–retest) cao **không** loại trừ khả năng đó. Chỉ κ **liên-người** mới tách
được "tiêu chí ổn định" khỏi "tiêu chí ổn định nhưng lệch".

    # sau khi người 1 chấm xong:
    python -m pipeline.ground_truth.make_interrater_batch
    # người 2 chấm mẻ đó, MÙ với verdict của người 1

## Đọc kết quả

    python -m pipeline.ground_truth.report_combined

Báo cáo trả về, và **phải công bố cả ba**:

- **precision riêng từng tầng** kèm khoảng tin cậy Clopper–Pearson
- **κ nội tại** (từ 60 ô lặp ẩn) — người chấm có tự nhất quán không
- **κ liên-người** (từ mẻ người thứ hai) — hai người có cùng tiêu chí không

> ⚠️ **Nếu κ liên-người < 0,6 thì DỪNG.** Con số precision khi đó không diễn giải được:
> nó đo tiêu chí riêng của một người, không đo chất lượng dữ liệu. Phải thống nhất lại
> định nghĩa verdict rồi chấm lại, chứ đừng công bố.

## Suy rộng ra dân số

Mỗi hàng mang `stratum` và `design_weight` = N_h / n_h. Ước lượng phải dùng **trọng số
Horvitz–Thompson**, không được lấy trung bình cộng thô — mẫu phân bổ theo sách chứ không
tỷ lệ thuận với dân số.

Precision toàn bộ giao nộp = tổ hợp có trọng số của ba tầng, **không phải** trung bình
của ba con số.

## Điều mẻ này KHÔNG trả lời được

- **Recall.** Mẻ chỉ chấm ô đã gán nhãn, không chấm 14.632 ô bị bỏ ở REVIEW.
- **Ngoại suy sang thể loại khác.** Cả ba cuốn cùng là truyện thánh Công giáo.
- **Chữ Nôm tự tạo hụt.** Bộ này chỉ 1,63% ô ngoài khối CJK cơ bản, so với 4,21% ở ngữ
  liệu NomNaOCR. Câu hỏi "pipeline bóc mất bộ thủ, hay Nôm Công giáo vốn chuộng dạng
  giản" cần một mẻ RIÊNG, lấy mẫu có chủ đích quanh các cặp `包/𠓨`, `弄/𢚸`, `礼/𥙩`.
