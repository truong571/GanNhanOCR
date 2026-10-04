# Google Cloud Vision làm kênh đối chứng độc lập cho kim (04/10/2026, bản 2 sau phản biện độc lập)

> Mục đích của bạn: đọc nhiều trang bằng Vision (ghép 3–4 trang/ảnh để không vượt 950 yêu cầu/tháng), lưu về đĩa, rồi dùng để (1) bắt kim đọc sai, (2) kiểm lại nhãn, (3) chỉnh crop bị lệch so với chữ.
> Trạng thái: **đã OCR TOÀN BỘ 1.682 trang của 10 sách** (ghép 2 trang/ảnh = 845 yêu cầu, 175.029 ký hiệu, 0 lỗi); sổ cái 876/950; chạy lại gọi 0 yêu cầu. Mã + kết quả nằm hết trong thư mục `vision/` (xem `vision/README.md`). Công cụ đã được 3 người phản biện độc lập soát và đã sửa; tiêu chí quyết định đã **đăng ký trước** (`vision/prereg/prereg_vision.json`, sha256 trong `vision/prereg/prereg_vision.sha256`).

## KẾT QUẢ ĐÃ CHẠY (04/10 tối; sổ cái 876/950; 0 lỗi, 0 cờ bất thường)

Pilot 26 yêu cầu (10 trang) rồi thu hoạch **845 yêu cầu → toàn bộ 1.682 trang, 244.514 ô** (cfg `n2-grid-zhHant-h80-d9e1aa`, ghép 2 trang, gợi ý zh-Hant, trần 10,5 MP): L16/TK 130 → 7 sách 448 → B18 + 2 trang không ô 267; sổ cái = 5 (cũ) + 26 + 845 = 876.
Dữ liệu: `vision/cache/` (không commit, **nhớ sao lưu**), phân tích: `vision/ket_qua/n2-grid-zhHant-h80-d9e1aa/`. Toàn cục: Vision phủ 35,1 % ô GOLD (kéo xuống vì B18), khớp 59,4 % trên ô có đọc.

- **Q1** (pilot, 10 trang): n=4 không trần 31,9 % agree/GOLD (Δ +2,4 so với n=1 [−3,9; +11,7]) vượt n=1 nhưng KTC rộng ⇒ trượt (i); n=3 và n=4 trần 5 MP kém hơn (−1,9; −1,1). Theo prereg ⇒ dự phòng **n=2** (lệch D1: trần 10,5 MP thay vì 5 MP; pilot cho thấy trần 5 MP làm giảm số ô agree). Vision không thu nhỏ ảnh tới 10,5 MP. **Gợi ý ngôn ngữ không có tác dụng** (Δ +0,0; 0/10 trang) ⇒ giữ zh-Hant.
- **Q2 (nhãn người IHR, L16 13.659 ô, TK 21.369 ô) — KHÔNG đạt**: ô xác nhận (agree & conf ≥ 0,8) chính xác 99,2 % / 99,0 % nhưng chỉ 5,1 % / 14,2 % số ô GOLD (< 20 %). Nhãn GOLD đúng 97,9 % / 98,6 %; chữ kim thô đúng 95,8 % / 97,5 %.
  **Cờ bất đồng không giàu lỗi hơn nền**: nhãn sai chiếm 1,9 % (L16) và 3,3 % (TK) số ô conf ≥ 0,8 bị cờ, nền 2,05 % / 1,40 %; Vision đúng 4/2.723 ô bất đồng URO không-dị-thể; kim thô sai 1.115 ô, Vision đọc 333 và đúng 40. ⇒ **Vision không giúp bắt kim sai hay kiểm lại nhãn** ở hai sách này.
- **Nhãn người Borg (toàn bộ trang)**: B34 (23.238 ô) Vision đọc 10,8 % ô, đúng 4,0 % mọi ô, conf ≥ 0,8 chỉ 59 % đúng; B18 (116.202 ô) đọc 15,9 %, đúng 7,0 % mọi ô (44 % trên ô có đọc), conf ≥ 0,8 đúng 63 %. Cấu hình tốt hơn kịch bản cũ (5–7 %) nhưng vẫn không thay được kim ở chữ viết tay.
- **Q3 (hiệu chỉnh crop theo cột) — KHÔNG đạt**: bỏ-một-ô ≤ 0,1 bước chỉ 26–67 % ở L16/Chr/STT/B18/B34 (cần ≥ 80 %); mô hình hằng và tuyến tính không hơn.
- **Nhưng "hại hình học" (glyph Vision > 10 % ngoài cửa sổ crop bbox ± 0,12) là có thật**, trên ô Vision đọc được: **STT 21–25 %, Chr 20 %, L16 20 %, B18 17 %, B34 15 %; KVK 0 %, L83 0 %, TK 2 %**. Hai cơ chế khác nhau:
  chữ viết tay (STT/Chr/B34): do **lệch tâm theo từng ô** (|dy| ≤ 0,10 bước: hại 1–8 %; |dy| > 0,10 — 62–70 % số ô — hại 26–35 %), không phải lệch hằng theo cột nên không sửa bằng một phép dịch;
  L16: do **hộp quá thấp** (62 % ô có chiều cao glyph ≥ chiều cao hộp → hại 31 %; còn lại 2,5 %), tức cửa sổ cắt đầu/đuôi chữ.
- Vision bao phủ (ô GOLD): KVK 84 %, L83 81 %, Chr 75 %, TK 57 %, L16 44 %, stt11 33 %, stt2 20 %, B18 18 %, stt4 17 %, B34 12,5 %; xác nhận conf ≥ 0,8: KVK 47 %, L83 41 %, TK 14 %, Chr 8 %, L16 5 %, B18 4 %, B34 3 %, STT 1–2 %.

Quyết định đã đăng ký: **không** đưa Vision vào pipeline làm tín hiệu xác nhận/cứu nhãn, **không** áp hiệu chỉnh crop theo cột. Hai lệch so với prereg (D1, D2) ghi ở `prereg_vision_ghi_chu_lech.json`.

## 0. Điều đã đo từ 5 phản hồi cache cũ (10 trang, MỖI CUỐN 1 TRANG, ghép 2 trang, gợi ý zh-Hant+vi) — chỉ là dấu hiệu, không phải kết luận

- **Độ phủ/khớp**: Vision có ký hiệu ở 42 % ô GOLD (KVK 87, L83 69, TK 60, L16 55, stt11 54, stt2 26, B34 21, stt4 24, Chr 21, B18 13); trên ô có đọc khớp nhãn 61 % (+50 ô dị thể/hình gần).
  Xác nhận ở conf ≥ 0,8: KVK 54 %, L83 35 %, TK 20 %, L16 10 %, còn lại ≤ 7 % ô GOLD. Khớp 47–50 % ở L16/TK chủ yếu là hiệu ứng kho chữ (0/74 khi nhãn ngoài URO).
- **Nhãn người**: ở B18 tr.88 và B34 tr.81 Vision đúng 5,2 % và 7,3 % số ô (= phủ 18–23 % × đúng 29–31 % trên ô có ký hiệu; ở conf ≥ 0,8 đúng 37,5 % [n=16] và 76,9 % [n=13] — mẫu quá nhỏ).
  Kết luận được: **Vision không thay được kim** ở chữ viết tay; không kết luận "vô dụng".
- **"Cứu kim sai" chưa có giá trị**: điều kiện "chữ Vision thuộc R(âm)" vô hiệu vì nhãn GOLD thuộc R(âm) theo cấu tạo (99,8 %); 7 ứng viên ban đầu không có ca nào là lỗi kim (3 dị thể giản/phồn, 4 chữ Nôm bị đọc thành chữ thành tố/đồng âm);
  trên nhãn người IHR (L16, TK) Vision đúng 1/83 ô bất đồng. Công cụ mới đã chuẩn hoá dị thể, chỉ xét nhãn URO, và đổi tên thành `disagree_candidates.csv` (không phải "kim sai").
- **Lệch crop**: `bbox` trong `labels.csv` KHÔNG phải hộp chữ chặt mà là hộp detector hoặc cửa sổ theo bước (cao 1,01–1,23 bước, 67–89 % cặp ô liền kề chồng nhau; crop thật = bbox ± 0,12 rồi carve). Vì vậy "tâm Vision − tâm hộp" một mình chưa phải "crop hỏng".
  Chỉ số đúng hơn: phần hộp glyph Vision nằm **ngoài** cửa sổ crop > 10 % ("hại hình học"), trên ô Vision khớp chữ: **stt11 37 %, stt4 29 %, stt2 25 %, Chr 12 %, B34 8 %, L16 3 %, KVK/L83/TK 0 %** (mẫu 9–83 ô mỗi cuốn, 1 trang/cuốn).
  Mô hình "độ lệch hằng theo cột rồi lan sang ô Vision không đọc được" **chưa được ủng hộ**: bỏ-một-ô đạt ≤ 0,1 bước ở chỉ 11–55 % ô trang viết tay (mô hình hằng: stt11 49, stt2 33, stt4 55, Chr 43 %; cần ≥ 80 %), chỉ ở sách in mới đạt vì "không dịch" đã đạt 89–100 %. Khoảng tin cậy bootstrap theo cột của stt4 và Chr chứa 0 ⇒ "hệ thống +0,15" chưa chứng minh.
- **Dò lệch ảnh–chữ** có độ nhạy thấp (tiêm lệch 1 ô: bắt 46 % ở KVK, 14 % L83, 3 % TK): "không thấy" ≠ "không lệch". Số ô phát hiện ở 10 trang là 4.
- **Gợi ý ngôn ngữ**: đã bỏ nhận định "gợi ý `vi` kéo nét Nôm thành chữ Latin" — 87 %/83 % ký hiệu Latin ở B18/B34 nằm ngoài mọi ô pipeline (chữ nhỏ ở lề, cao ~14 px); A/B gợi ý phải đo bằng số ô Hán đọc ĐÚNG.

## 1. Công cụ (vision/, kiểm offline 25 + 14 phép; bản cũ stitched_vision_audit.py / run_google_vision_audit.py đã bị vô hiệu hoá)

| tệp | việc |
| --- | --- |
| `vision_harvest.py` | `plan` (0 yêu cầu, chỉ đọc) · `pilot --stage 1\|2 --yes` · `run --yes` · `status` · `import-legacy`; chạy `pilot`/`run` **không có --yes** = xem trước |
| `vision_crosscheck.py` | `cells --cfg <id>` → `cells.csv.gz`, `disagree_candidates.csv`, `drift.csv`, `offsets.csv`, `column_model.csv`, `summary.json` (khớp 1–1, chuẩn hoá dị thể, lớp mã nhãn, hại hình học, bỏ-một-ô) · `compare --cfgs a,b,…` |
| `test_vision_harvest.py` · `test_vision_crosscheck.py` | "OCR" giả đọc ảnh ghép thật nên kiểm cả chuỗi co giãn → trần điểm ảnh → lưới/hàng → JPEG → toạ độ trang gốc; kiểm sổ cái khoá tệp, bộ ngắt, phân loại lỗi mạng, khoá API |

Sửa theo phản biện: sổ cái **ngoài repo** (`~/.cache/gannhanocr/vision_ledger.json`, khoá tệp, `.bak`, kỳ tính phí giờ Pacific, trần cứng 990), **đếm trước khi gửi** (hoàn lại khi chắc chắn không tính phí; giữ khi hết giờ/ngắt/HTTP 200), một tiến trình `run` tại một thời điểm, dừng sau 3 nhóm lỗi liên tiếp,
mỗi lần chạy mặc định ≤ 25 yêu cầu (`--max-requests 0` để chạy hết), **trần 5 MP mỗi ảnh ghép** (vùng đã kiểm chứng với Vision thật; ảnh 14 MP chưa có bằng chứng), hệ số co giãn đóng băng theo cấu hình + phát lại từ bố cục đã lưu, dừng ở ảnh đầu tiên có kích thước/toạ độ báo về bất thường,
`--key` sai không lặng lẽ dùng khoá khác, khoá API được kiểm định dạng và mọi chuỗi lỗi được lọc `key=`/`Bearer`. Số yêu cầu cho cả bộ (1.680 trang): **n=2: 843 · n=3: 564 · n=4: 424** (+5 đã dùng). n=2 đã vừa trần 950; ghép 3–4 trang chỉ để giảm yêu cầu và có thể giảm chất lượng đọc.

## 2. Cách chạy (theo thứ tự; mỗi bước tự dừng được)

0. **Điều kiện (P0)**: (a) dùng interpreter có `google-auth`+`requests` (không có trong `.venv`/python3 hệ thống): `python3 -m venv ~/.venvs/vision && ~/.venvs/vision/bin/pip install google-auth requests pillow`;
   (b) chuyển khoá ra ngoài repo: `mkdir -p ~/.config/gcloud && mv vision-ocr-*.json ~/.config/gcloud/ && chmod 600 ~/.config/gcloud/vision-ocr-*.json`; (c) **commit `.gitignore` trước khi `git add`** (đang chỉ sửa chưa commit);
   (d) xem Console → Vision API → Metrics và Billing → Reports để biết dùng ngoài repo (hạn mức tính theo tài khoản thanh toán, tháng Pacific); đặt cảnh báo ngân sách ~1 USD.
1. **Xem trước (không gọi mạng)**: `python3 vision/vision_harvest.py plan` và `python3 vision/vision_harvest.py pilot --stage 1`.
2. **Pilot giai đoạn 1 (20 yêu cầu)**: `~/.venvs/vision/bin/python vision/vision_harvest.py pilot --stage 1 --yes` — 10 trang cũ đọc theo n=1 · n=4 không trần MP (6,5–10,5 MP: thử Vision có thu nhỏ ảnh lớn không) · n=4 trần 5 MP · n=3 trần 5 MP;
   rồi `.venv/bin/python vision/vision_crosscheck.py compare --cfgs <các cfg in ra, tiền tố _pilot/>` và chọn theo Q1 trong `prereg_vision.json`. **Pilot giai đoạn 2 (6 yêu cầu)**: `pilot --stage 2 --yes [--cell-h X]` (gợi ý zh-Hant · không gợi ý).
3. **Giai đoạn đánh giá**: `run --yes --books L16,TK --max-requests 0` + cấu hình thắng (66 yêu cầu ở n=4) rồi `vision_crosscheck.py cells` — chỉ khi đạt Q2 mới tiếp các sách còn lại. Nếu vẫn muốn đọc hết: `run --yes --max-requests 0` (sách in trước); (B18 đã chạy theo yêu cầu "toàn bộ trang"). Lần chạy đầu để mặc định (25 yêu cầu), xem dòng `[...]` và cờ `⚠`, rồi tăng.
4. **Dùng kết quả**: mọi kết luận/áp dụng theo `prereg_vision.json` (Q1 cấu hình · Q2 tín hiệu xác nhận · Q3 hiệu chỉnh crop); chỉ ghi sidecar ở `vision/ket_qua/`, không sửa `labels.csv`/crop. Điểm vào kiểm độc lập việc chỉnh crop có sẵn: `pipeline/tools/eval_crops_v2.py` (CNN OOF + McNemar theo cụm cột),
   `pipeline/gold_exact/signals_img.py` (Glyphs/MultiEnc → cosine crop↔glyph nhãn), `build_dataset.save_crop` / `pipeline/tools/recrop_v2.py` (cắt lại từ bbox dời); harness mới đặt ở `lab/`.

## 3. Giới hạn chưa kiểm chứng

- Hành vi của Vision ở > 4,69 MP / > 2.604 px (cấu hình mặc định nay bị chặn ở 5 MP; pilot sẽ cho biết); thời gian mỗi yêu cầu và độ trễ phía Google chưa đo (`run --max-requests 5` rồi ngoại suy);
  quy ước biên hộp ký hiệu của Vision (chặn trên đo được ở sách in ≤ ~1,5 px trang); Google có tính phí ảnh lỗi theo ảnh hay không (tài liệu không nói — mã giữ tính 1 để thận trọng).
- Toàn bộ số ở §0 là 1 trang/cuốn, đơn vị độc lập thật là cột (3–10 cột đủ mẫu mỗi cuốn); các ngưỡng không được chỉnh trên đó, nhưng cũng không dùng nó để xác nhận.
