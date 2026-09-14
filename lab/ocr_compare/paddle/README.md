# PaddleOCR 3.7 (PP-OCRv6 / PP-OCRv5 / PaddleOCR-VL-1.6) trên chữ Nôm chép tay — giao thức chung M1–M4

Ngày đo: 2026-09-11. Thiết bị: Apple M4, 16 GB, **CPU** (không dùng MPS). Máy đang chạy song song
các workflow khác nên thời gian/ô có nhiễu (±30%).

## Cài đặt (venv riêng, không đụng .venv của kho)
```
/opt/homebrew/bin/python3.10 -m venv /tmp/venv_paddle
/tmp/venv_paddle/bin/pip install paddlepaddle paddleocr        # paddlepaddle 3.3.1, paddleocr 3.7.0, paddlex 3.7.2
/tmp/venv_paddle/bin/pip install opencc-python-reimplemented    # phân loại giản/phồn (phân tích)
/tmp/venv_paddle/bin/pip install sentencepiece jinja2 einops    # chỉ cần cho PaddleOCR-VL native
```
Python 3.14 (mặc định máy) KHÔNG có wheel paddlepaddle → phải dùng python3.10 của Homebrew. Model tải tự động
về ~/.paddlex/official_models/ (PP-OCRv6_medium_rec 76 MB, PP-OCRv5_server_rec 84 MB, PaddleOCR-VL-1.6 1,8 GB).

## Bộ mẫu chung
`lab/ocr_compare/mau/M1.csv M2.csv M3.csv M4.csv + cot/` (seed 20260911; md5 ghi ở `mau_md5.txt`). Trọng tài
R(âm) = cột `R_candidates` (Dict/QuocNgu_SinoNom.csv qua `core.text.dictionary.load_qn_to_nom`).

## Mã
| Tệp | Việc |
|---|---|
| `run_paddle.py` | `rec`: nhận dạng thẳng crop đơn (TextRecognition, không detector) với tiền xử lý raw/pad/up3/padup3; `pipe`: det+rec trên crop; `m4`: cột dọc (det+rec nguyên cột, +textline-orientation, cột xoay 90°, phóng 2×; rec-only xoay CCW/CW) |
| `run_paddle_col.py` | **chế độ CỘT** cho M1–M3: đọc nguyên cột chứa ô (cắt từ trang theo hộp cột của cache kinhhannom, xoay 90° CCW), lấy vị trí CTC từng chữ (`return_word_box`) rồi gióng về ô: `pred` = theo y, `pred_mono` = DP đơn điệu theo toạ độ, `pred_seq` = NW với chuỗi kinhhannom (thiên vị, chỉ để tham khảo) |
| `run_vl.py` | PaddleOCR-VL-1.6-0.9B (backend native, CPU) trên M1–M4 |
| `score.py`, `score_m4.py`, `tong_hop.py`, `phan_tich_them.py`, `analyze_gian_phon.py` | chấm điểm, tổng hợp (`tong_hop.csv/json`, `phan_tich_them.json`, `charset_coverage.json`) |
| `ket_qua/*.csv` | đầu ra thô từng ô (sample_id, pred, score, sec) |

Chạy: `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True /tmp/venv_paddle/bin/python run_paddle.py rec --set M2 --model PP-OCRv6_medium_rec --prep raw`
(chấm bằng `.venv` của kho: `.venv/bin/python tong_hop.py`).

## Phát hiện cấu trúc (trước khi đo)
* Trong PaddleOCR 3.7, `lang="ch"` và `lang="chinese_cht"` ánh xạ về **cùng một model** (PP-OCRv6_medium_rec, hoặc
  PP-OCRv5_server_rec khi `ocr_version="PP-OCRv5"`) — không có model phồn thể riêng. So sánh "ch vs chinese_cht" vô nghĩa;
  thay bằng so v6 vs v5.
* Charset v6 = v5: 18.708 token, 15.906 CJK: URO 15.565, Ext-A 137, **Ext-B 44**, Ext-C–F 160. Phủ 35,6% từ vựng Nôm
  (14.758/41.502; Ext-B chỉ 17/16.264) nhưng 92,8% token `ocr_char` của corpus (vì corpus phần lớn là chữ Hán mượn).
* **㝵, 𠊚, 𠊛 đều KHÔNG có trong charset** → M3 không thể ra 𠊚 lẫn 㝵 bất kể ảnh.
* 99,7% ô M2 có ≥1 ứng viên R(âm) nằm trong charset → trần lý thuyết của "cứu" gần 100%, không bị charset chặn.

## Kết quả (n kèm theo; `in_R` = chữ CJK đầu tiên đọc ra ∈ R(âm))

### M1 — đối chứng (600 ô GOLD kinhhannom đúng theo R)
| Cấu hình | ra đúng 1 CJK | in_R | trùng kinhhannom | s/ô |
|---|---|---|---|---|
| v6 rec crop raw | 69,5% | 34,3% [30,6–38,2] | 31,5% | 0,067 |
| v6 rec crop up3 | 69,2% | 34,5% | 31,7% | 0,078 |
| v6 rec crop pad25% | 59,5% | 28,2% | 25,3% | 0,070 |
| v5 rec crop raw | 77,0% | **37,2%** [33,4–41,1] | 34,0% | 0,103 |
| v6 det+rec crop pad+up3 | 62,3% | 29,2% | 27,2% | 1,08 |
| **v6 CỘT, gióng y** | 90,7% | **52,3%** [48,3–56,3] | 48,0% | 0,167 s/cột (~0,008 s/ô) |
| v6 CỘT, gióng đơn điệu | 95,8% | **57,2%** | 51,8% | " |
| v6 CỘT, gióng theo chuỗi kinhhannom (thiên vị) | 97,0% | 71,0% | 64,8% | " |
| v5 CỘT, gióng y | 61,8% | 31,3% | 28,5% | 0,31 |

Phóng to 3× không đổi (rec resize về cao 48 px). Viền trắng làm giảm. Detector trên crop đơn làm giảm.
Chữ đọc ra: 0% ngoài từ điển Nôm nhưng vì từ điển chứa cả dạng giản thể — chỉ số hữu ích hơn: ở chế độ crop
**14,4% (v6) / 18,5% (v5) chữ đọc ra là giản thể thuần** (为国发动…, không thể có trong bản chép tay TK19);
ở chế độ cột chỉ 5,3% giản-thuần, 25,7% phồn-thuần (ngữ cảnh kéo model về phồn thể). Lưu ý OpenCC xếp
庄/双/几/云 là "giản thể" nhưng chúng là chữ Nôm hợp lệ (chẳng/song/kỉ/vân).

### M2 — PHÉP ĐO CHÍNH (600 ô SILVER/SYLLABLE, kinhhannom ∉ R(âm))
| Cấu hình | in_R = **tỷ lệ cứu được** | lift so ngẫu nhiên theo tần suất (0,64%) | lift so hoán vị | SILVER | SYLLABLE |
|---|---|---|---|---|---|
| v6 rec crop raw | 39/600 = 6,5% [4,8–8,8] | ×10 | ×26 (0,25%) | 19/219 = 8,7% | 20/381 = 5,2% |
| v6 rec crop up3 | 43/600 = 7,2% | ×11 | | 10,0% | 5,5% |
| v5 rec crop raw | 40/600 = 6,7% [4,9–9,0] | ×10,5 | ×32 (0,21%) | 17/219 = 7,8% | 23/381 = 6,0% |
| v6 det+rec crop | 31/600 = 5,2% | ×8 | | 5,9% | 4,7% |
| **v6 CỘT, gióng y** | **54/600 = 9,0% [7,0–11,6]** | **×14** | ×28 (0,33%) | 31/219 = **14,2%** | 23/381 = 6,0% |
| v6 CỘT, gióng đơn điệu | **57/600 = 9,5%** | ×15 | | 14,2% | 6,8% |
| v6 CỘT, gióng chuỗi (thiên vị) | 61/600 = 10,2% | ×16 | | 14,2% | 7,9% |
| v5 CỘT, gióng y | 33/600 = 5,5% | ×8,6 | | 5,5% | 5,5% |

Đối chứng: chữ ngẫu nhiên đều trên 41.502 chữ từ điển rơi vào R với 0,058%; theo tần suất corpus 0,64%;
hoán vị chính đầu ra của engine giữa các ô 0,25–0,33% → lift ×26–×32 là tín hiệu thật, không phải do
engine hay ra chữ thông dụng.
Trên SILVER, 24/31 chữ cứu được (cột) trùng đúng nhãn S2/S3 của pipeline (PaddleOCR chưa hề thấy nhãn này) —
hội tụ độc lập. Theo rule: s2_inter_s3_corrected 30/209 = 14,4%; nghia_consensus 11/226 = 4,9%;
nghia_consensus_tu_silver 12/155 = 7,7%. Theo sách: stt11 13, stt2 20, stt4 21 /200.
Ví dụ cứu được (kinhhannom → Paddle, âm): 杢→吏 lại; 𣣔→歇 hết; 𠬧→双 song; 𠇭→命 mình; 女→共 cũng;
𭃡→初 xưa; 之→代 đời; 馬→為 vì; 𡚶→安 ăn; 達→連 trên.

### M3 — phá sản (300 ô kinhhannom=㝵, âm "người"; người phán đúng là 𠊚)
| Cấu hình | 𠊚 | 㝵 | ∈ R(người) | phân bố đọc ra (top) |
|---|---|---|---|---|
| v6 crop raw | **0%** | **0%** | 0/300 | 寻 15,0%, 导 12,7%, rỗng 10,0%, 等 8,7%, 是 7,0%, 守 3,0%, 子 2,3%, 得 2,0% |
| v5 crop raw | 0% | 0% | 0/300 | rỗng 17,0%, 寻 15,0%, 等 12,0%, 导 9,7%, 享 4,3% |
| v6 CỘT | 0% | 0% | 2/300 = 0,7% (= mức hoán vị 0,67%) | 尋 32,3%, 等 15,3%, 是 10,3%, 得 7,7%, 景 4,0% |

Cả hai mã đích đều ngoài charset nên PaddleOCR không thể phán xử lớp này; nó đọc thành 寻/尋/导/等/是/得 —
chữ có hình gần 㝵 (㝵 = 得 bỏ 彳; 尋 có 彐+寸). Kết quả âm tính dứt khoát cho M3.

### M4 — 20 cột dọc nguyên vẹn (kinhhannom trung bình 21,05 chữ/cột)
| Cấu hình | chữ đọc/cột | tỷ lệ so kinhhannom | cột lệch ≤2 chữ | LCS/kinhhannom | s/cột |
|---|---|---|---|---|---|
| v6 det+rec nguyên cột | 20,85 | 0,99 | **20/20** | **51,5%** | 1,02 |
| v6 det+rec + textline-orientation | 19,25 | 0,92 | 17/20 | 43,7% | 1,05 |
| v6 det+rec, cột đã xoay 90° CCW | 21,0 | 1,00 | 20/20 | 52,0% | 0,95 |
| v6 det+rec, phóng 2× | 20,9 | 0,99 | 20/20 | 49,2% | 3,3 |
| v6 rec-only, xoay 90° CCW | 20,95 | 1,00 | 20/20 | 49,4% | **0,19** |
| v6 rec-only, xoay 90° CW | 10,35 | 0,49 | 0/20 | 0,2% | 0,17 |
| v5 det+rec nguyên cột | 12,3 | 0,58 | 0/20 | 31,8% | 2,7 |
| v5 rec-only xoay CCW | 11,8 | 0,56 | 0/20 | 29,7% | 0,41 |

Cơ chế: detector v6 ra đúng 1 hộp/cột (20/20 hộp cao, h/w ≥ 1,5) → pipeline tự xoay 90° CCW (np.rot90) rồi
nhận dạng như một dòng ngang; chiều xoay CW cho rác → model chỉ hiểu chữ dọc theo hướng CCW. Không có tham số
"văn bản dọc" riêng; `use_textline_orientation` (0/180°) làm kém đi. PP-OCRv6 đọc được cột dài ~21 chữ (số chữ
đúng ±2 ở 20/20 cột); PP-OCRv5 bị cắt còn ~12 chữ (giới hạn bề rộng rec 320 px). Model **không tách ký tự thành
hộp**: chỉ trả chuỗi; vị trí từng chữ lấy được qua `return_word_box=True` (chỉ số bước CTC) → gióng về ô kinhhannom
theo y (73% cột có số chữ bằng đúng kinhhannom, 16% thiếu 1 chữ).

### PaddleOCR-VL-1.6-0.9B (backend native, CPU fp32) — `run_vl.py`, `ket_qua/*_PaddleOCR-VL-1.6.csv`
Cài được sau khi bổ sung sentencepiece/jinja2/einops (pip paddleocr không kéo theo). Nạp model 206–284 s, RSS 3,6–5,7 GB.
Thời gian **2,8 s/ô crop, 7,4 s/cột** khi CPU rảnh (M1 600 + M2 600 + M3 300 + M4 20 = 1.520 ảnh, ~75 phút) (45–159 s/ô khi tranh chấp CPU với mẻ khác — lần đo đầu).
Prompt "OCR:" (prompt nhận dạng chuẩn của PaddleOCR-VL). Đầu ra hay kèm rác: LaTeX (`\frac{1}{2}`), kana, chữ Latin.
| Tập | n | ra đúng 1 CJK | in_R | trùng kinhhannom | s/ô | ghi chú |
|---|---|---|---|---|---|---|
| M1 crop | 600 | 61,7% | 189/600 = 31,5% (228 nếu tính bất kỳ chữ) | 173 = 28,8% | 2,8 | ≈ v6 crop 34,3%, kém v6 cột 52–57% |
| M2 crop | 600 | 60,5% | **33/600 = 5,5%** (46 nếu tính bất kỳ chữ nào trong đầu ra) | 36 | 2,8 | kém v6 crop 6,5% và v6 cột 9,0–9,5%, chi phí ×40 |
| M3 crop | 300 | 70,3% | 0/300; **𠊚 0%, 㝵 0%** | 0 | 2,8 | đọc thành 等 23,7%, 导 15%, 尊 6,7%, 旱 6%, 寻 3% |
| M4 cột | 20 | — | — | LCS 44,7% với kinhhannom; 20,6 chữ/cột, 20/20 cột lệch ≤2 | 7,4/cột | VL đọc được cột dọc, tự ngắt nhóm bằng dấu cách; kém v6 (51,5%) |

Kết luận VL: đọc được cột dọc chữ Nôm nhưng không hơn PP-OCRv6 trên bất kỳ tập nào, chậm hơn 7–40 lần; charset là
từ vựng tokenizer LLM nên về lý thuyết có thể sinh Ext-B, nhưng thực tế trên M3 không ra 𠊚 lần nào.

## Kết luận
1. **PaddleOCR đọc được chữ Nôm chép tay ở mức có tín hiệu thật nhưng thấp**: đối chứng M1 (kinhhannom đúng) chỉ 34–37%
   ở mức crop đơn, 52–57% khi đọc nguyên cột (gióng không thiên vị). Kinhhannom vẫn là kênh chính.
2. **Phép đo chính M2**: PaddleOCR cứu được **9,0–9,5% [7,0–11,6]** khối trượt ở chế độ cột (54–57/600), lift ×14–15 so
   ngẫu nhiên theo tần suất và ×28 so hoán vị đầu ra; 6,5–7,2% ở chế độ crop. Trên SILVER 14,2%, SYLLABLE 6,0–6,8%.
   Ngoại suy thô lên khối ~27.770 ô: ≈2.500–2.600 ô có bằng chứng độc lập mới (chưa tính điều kiện "∈R là cần chứ chưa đủ").
   24/31 chữ cứu được trên SILVER trùng nhãn S2/S3 của pipeline → hội tụ độc lập, tăng độ tin cho SILVER hơn là mở rộng.
3. **M3 phá sản hoàn toàn**: 㝵 và 𠊚 đều ngoài charset (Ext-A/B chỉ 137/44 ký tự) → 0% ở mọi cấu hình. Không dùng
   PaddleOCR cho lớp này hay bất kỳ chữ Nôm tự tạo nào (Ext-B: 16.264 chữ trong từ điển, model có 17).
4. **Chữ dọc (M4)**: PP-OCRv6 xử lý được cột dài (~21 chữ, số chữ đúng ±2 ở 20/20 cột) nhờ detector 1 hộp/cột + xoay 90° CCW;
   PP-OCRv5 bị cắt ~12 chữ. Không tách ký tự thành hộp; vị trí CTC (`return_word_box`) đủ để gióng ô (73% cột khớp đúng số chữ).
5. **Cấu hình tốt nhất**: PP-OCRv6_medium_rec, đọc **nguyên cột** (từ hộp cột kinhhannom), rec-only xoay 90° CCW, gióng đơn điệu
   theo vị trí CTC; 0,17–0,2 s/cột CPU (~0,01 s/ô). Không phóng to, không viền, không detector trên crop đơn.
6. Chế độ crop trả 14–18% giản thể thuần (为国发…) — không có trong bản chép tay; chế độ cột kéo về phồn thể (5% giản-thuần).
7. Phát hiện phụ: 19,3% ô M2 (116/600) Paddle đọc ra **đúng chữ kinhhannom đã đọc** (cả hai ∉ R) → một phần "khối trượt"
   có thể không phải lỗi OCR mà là từ điển thiếu âm hoặc âm QN gióng sai; đáng kiểm tra bằng máy trước khi vứt.
