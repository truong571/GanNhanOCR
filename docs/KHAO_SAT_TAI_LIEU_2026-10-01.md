# Tổng quan tài liệu — "Hệ thống hỗ trợ gán nhãn tự động văn bản Hán Nôm cổ"

Ngày lập: 01/10/2026. Phạm vi: các công trình 2011–2026 (trọng tâm 2018–2026) liên quan trực tiếp tới pipeline GanNhanOCR.

## 0. Cách xác minh

- **Đã xác minh** = đã mở được ít nhất một trang gốc (arXiv, DOI qua Crossref, trang hội nghị/nhà xuất bản, OpenReview,
  PMC) và đối chiếu được **tên bài, tác giả, năm, nơi công bố**. Siêu dữ liệu (DOI, tác giả) của 31 bài đã được đối chiếu
  thêm qua Crossref API, kết quả khớp tên bài 31/31.
- Số liệu trong bảng lấy từ **tóm tắt (abstract) hoặc trang gốc** của bài. Bài nào chỉ đọc được tóm tắt thì không suy
  diễn thêm.
- Cột "Hạn chế liên quan" là **nhận định của người viết** khi đặt bài đó cạnh bài toán của đề tài, không phải lời của
  tác giả bài.
- Mục nào không xác minh đủ thì đưa sang §6 ("chưa xác minh") và **không** đưa vào danh sách chính.

---

## 1. Tài liệu đã được trích trong repo: kết quả đối chiếu (Bước 1)

Nguồn đã quét: `README.md`, `ArcFace/README.md`, `pipeline/consensus_fusion/*`, `pipeline/align_engine/nom_classifier/infer.py`,
`ArcFace/{evaluate,sam}.py`, `pipeline/ground_truth/README.md`, `scripts/_retired/README.md`, `NomNaOCR/HUONG_DAN_SU_DUNG.md`,
`font_diffusion/*`, `train_crop/README.md`, `docs/*.md`, `data/*/SOURCE.md`.

**Tệp không tồn tại:** `docs/DE_XUAT_HOAN_THIEN_LUAN_VAN_2026-07-20.md` và `testsoanh/README.md` không có trong cây làm việc
và cũng không có trong lịch sử git (`git log --all`).

| Trích dẫn trong repo | Nơi trích | Kết quả xác minh | Ghi chú / sai lệch |
|---|---|---|---|
| arXiv 2510.04003 (PaddleOCRv5 Hán-Nôm) | `scripts/_retired/README.md:33`, `NomNaOCR/HUONG_DAN_SU_DUNG.md:1` | Đúng: Nguyen & Nguyen Thiet, 2025 | khớp (37,5 → 50,0 % exact) |
| arXiv 2508.07904 ("kraken 5.x ForcedAlignmentTaskModel") | `pipeline/consensus_fusion/README.md:64` | Bài có thật: **Peer, Scius-Bertrand, Fischer — CTC Transcription Alignment of the Bullinger Letters** (VisionDocs@ICCV 2025) | Chưa xác minh được việc bài dùng kraken; nên trích bài này cho ý "căn chỉnh CTC bằng DP" và bỏ chữ "kraken" nếu chưa kiểm |
| arXiv 2605.11960 ("mọi VLM < 25 % char-F1 trên cổ văn") | `pipeline/consensus_fusion/README.md:76` | Bài có thật: **Chronicles-OCR** (Li et al., 2026) | **Sai nội dung**: bài không nêu ngưỡng "< 25 % char-F1". Mô hình tốt nhất (Kimi K2.5) đạt 31,9 % nhận dạng chữ cổ (exact match, trung bình), NED 0,78 với chữ khải nhưng chỉ 0,06 với giáp cốt văn. Cần sửa câu |
| arXiv 2605.29800 ("Nine Judges, Two Effective Votes") | `consensus_fusion/{fusion,independence}.py`, README | Đúng: Kohli, 2026 | khớp (Kish n_eff ≈ 2 trên 9 giám khảo) |
| arXiv 2601.22336 | `consensus_fusion/fusion.py:4` | Đúng: Balasubramanian et al., 2026 (gộp nhãn có phụ thuộc, mô hình Ising) | khớp |
| arXiv 2410.11302 (sycophancy VLM) | `consensus_fusion/qwen_verifier.py:4` | Đúng: Li et al., 2024 (MM-SY) | khớp |
| arXiv 2410.15393 ("position bias") | `qwen_verifier.py:8` | Bài thật là **CalibraEval** (Li et al., 2024): *selection bias* của LLM-as-judge (mô hình chữ, không phải VLM) | Nên ghi đúng là "selection bias của LLM-giám khảo" |
| arXiv 2312.12142 FontDiffuser | `font_diffusion/gradio_app.py` | Đúng: Yang et al., AAAI 2024 | khớp |
| arXiv 2510.13745 UniCalli "(ICLR 2026)" | `font_diffusion/src/modules/unicalli_improvements.py:5` | Đúng; ICLR 2026 xác nhận qua OpenReview/mlanthology | khớp |
| DINOv2 | `README.md:408` | Đúng: Oquab et al., TMLR 01/2024 | repo đo được DINOv2 không phân biệt được chữ Nôm (kết quả nội bộ) |
| Vaze et al. ICLR'22 (MLS); Liu NeurIPS'20 (Energy); Foret ICLR 2021 (SAM) | `ArcFace/evaluate.py`, `sam.py`, `infer.py` | Cả ba đúng | khớp |
| Angelopoulos et al. 2023 (PPI) | `pipeline/ground_truth/README.md:181` | Đúng: *Science* 382 | khớp |
| Gebru et al. 2021 (Datasheets) | `docs/KE_HOACH_*`, `scripts/_retired/README.md:82` | Đúng: CACM 12/2021 | khớp |
| Dawid–Skene | `consensus_fusion/fusion.py` | Đúng: JRSS-C 28(1), 1979 | khớp |
| Croissant, Frictionless | `pipeline/publish`, `to_standard.py` | Croissant đúng (NeurIPS 2024 D&B) | Frictionless là chuẩn kỹ thuật, không phải bài báo, nên không đưa vào bảng |
| NomNaOCR | nhiều nơi | Đúng: Dang et al., RIVF 2022 | BibTeX trong repo GitHub ds4v bị lỗi (author "ds4v", doi "10.1109/10013842"); DOI đúng là `10.1109/RIVF55975.2022.10013842` |
| IHR-NomDB | `data/LucVanTien1916/SOURCE.md` … | Đúng: Vu, Le, Beurton-Aimar, ICDAR 2021 | Tên bài ghi "Handwritten", nhưng `data/LucVanTien1916/SOURCE.md` đã kiểm ảnh và xác định đây là **mộc bản in**. Luận văn phải gọi là mộc bản |
| MTH/TKH (pretrain detector) | `train_crop/README.md`, `docs/EVIDENCE_INDEX.md:175` | Đúng: Yang et al., IEEE Access 2018 (TKH, MTH); Ma et al., ICFHR 2020 (MTHv2) | repo ghi "cần trích dẫn trong luận văn" mà chưa ghi bài cụ thể; dùng hai bài này |
| Kuzushiji-Kanji (CODH) | `docs/data_manifest/MANIFEST.md:31` | Đúng: Clanuwat et al., 2018 | repo không dùng (0 tham chiếu) |
| Kaggle New-SinoNom, SinoNom Similarity Retrieval | `README.md:405–406` | Chỉ xác minh được tiêu đề trang dataset, chưa đọc được nội dung | xem §6 |
| HF `dzungpham/font-architect` | `README.md:407` | Model card có thật: dựa trên FontDiffuser, học trên dataset `dzungpham/FontTransfer` (> 632 nghìn mẫu), Apache-2.0 | là tài nguyên cộng đồng, **không phải** mô hình do dự án huấn luyện |

---

## 2. (a) Bảng công trình liên quan đã xác minh

### A. OCR Hán-Nôm / chữ Nôm và dữ liệu

| # | Tác giả (năm) | Nơi công bố | Phương pháp chính | Dữ liệu | Hạn chế liên quan đến đề tài | Liên kết |
|---|---|---|---|---|---|---|
| A1 | Phan T.V., Zhu B., Nakagawa M. (2012) | DAS 2012 (IAPR) | Tách chữ bằng projection profile; gom cụm bằng OCR chữ Hán + K-means; người vận hành xác nhận cụm | Hàng trăm trang Nôm viết tay | Không dùng bản phiên âm; nhãn vẫn cần người duyệt | https://doi.org/10.1109/DAS.2012.25 |
| A2 | Phan T.V., Nakagawa M. (2014) | DATeCH 2014 (ACM) | Hệ số hoá: nhị phân, tách chữ, phân loại GLVQ (thô) + MQDF2 (tinh); mẫu huấn luyện **sinh từ 27 phông** Hán/Nhật/Nôm | 7.601 / 32.733 lớp | Học từ phông, chưa có kiểm chứng tự động nhãn | https://doi.org/10.1145/2595188.2595196 |
| A3 | Phan T.V., Nguyen K.C., Nakagawa M. (2015/2016) | IJDAR 19:49–64 | Recursive X–Y cut + Voronoi; k-d tree + GLVQ + MQDF; hai bộ nhận dạng (7.660 và 32.695 lớp) | Tài liệu Nôm lịch sử | Kỹ thuật cổ điển, cần người sửa | https://doi.org/10.1007/s10032-015-0257-8 |
| A4 | Nguyen C.K., Nguyen C.T., Nakagawa M. (2017) | HIP 2017 (ACM) | CNN sâu cho 32.695 lớp Nôm, lớp thô dựng trước bằng K-means | Mẫu chữ Nôm đơn | Chỉ nhận dạng chữ đã tách, không gán nhãn trang | https://doi.org/10.1145/3151509.3151517 |
| A5 | Nguyen K.C., Nguyen C.T., Nakagawa M. (2020) | Pattern Recognition Letters | Tách chữ bằng CNN (IoU 92,08 %, so với 81,23 % của projection + Voronoi) rồi OCR từng chữ; có attention thì đạt 85,05 % | Trang Nôm | Cần nhãn hộp chữ để huấn luyện bộ tách | https://doi.org/10.1016/j.patrec.2020.02.015 |
| A6 | Vu M.T., Le V.L., Beurton-Aimar M. (2021) | ICDAR 2021 (LNCS) | **IHR-NomDB**: hơn 260 trang có hộp đánh tay + 101.621 ảnh chuỗi Nôm tổng hợp; CRNN-CTC đạt 42,70 % câu / 82,28 % ký tự | Trang Nôm từ Nôm Foundation | Hộp ở mức cột/dòng (theo `SOURCE.md` của repo: bbox chỉ ở mức CỘT); tên bài ghi "handwritten" nhưng sách L16 là mộc bản | https://doi.org/10.1007/978-3-030-86334-0_6 · https://morphoboid.labri.fr/ihr-nom.html |
| A7 | Dang H.Q., Nguyen D.A., … Do T.H. (2022) | RIVF 2022 (IEEE) | **NomNaOCR**: 2.953 trang, 38.318 patch dòng; DBNet phát hiện (F1 99,65 %), CRNN nhận dạng (29,41 % chuỗi / 84,73 % ký tự) | 3 tác phẩm, Nôm Foundation | Nhãn ở mức dòng, không có hộp chữ; trùng nguồn với IHR L16 (repo đo: 2.007/2.053 câu giống hệt), nên dùng làm kênh kiểm sẽ rò dữ liệu | https://doi.org/10.1109/RIVF55975.2022.10013842 · https://github.com/ds4v/NomNaOCR |
| A8 | Tran A.T., Le N.Y., Dinh D., Tran T.S. (2022) | ICABDE 2021, LNDECT (Springer) | OCR chữ Nôm mộc bản + **mô hình ngôn ngữ Nôm** xếp lại top-N | Train Kiều 1871; test Kiều 1902 (71,80 % mAP top-1), LVT (72,96 %) | Ngôn ngữ mô hình chỉ thêm +2,72 / +0,14 điểm; giảm mạnh khi đổi bản | https://doi.org/10.1007/978-3-030-97610-1_5 |
| A9 | Dao T.T., Le C.T., Ngo T.D., Le T.H. (2023) | KSE 2023 (IEEE) | Phát hiện + nhận dạng chữ Hán-Nôm mộc bản, bán giám sát; phát hiện F1 > 0,97; nhận dạng top-1 80,1 %, top-5 90,1 % | Ảnh mộc bản | Chỉ khắc in; độ đúng top-1 chưa đủ để dùng thẳng làm nhãn | https://doi.org/10.1109/KSE59128.2023.10299440 |
| A10 | Le Thi Cam Thi, Dinh Si Dien, Nguyen Vinh Tiep (2025) | ICCIES 2025, CCIS (Springer) | OCR Hán-Nôm **ngoài cảnh** (đình, chùa, lăng) kết hợp tri thức ngôn ngữ (âm); độ đúng 80 % | Bộ dữ liệu cảnh Hán-Nôm đầu tiên | Khác miền (ảnh cảnh, không phải sách) | https://doi.org/10.1007/978-3-031-98170-8_9 |
| A11 | Nguyen M.H., Nguyen Thiet S. (2025) | arXiv 2510.04003 | Fine-tune bộ nhận dạng PaddleOCRv5 trên bản thảo Hán-Nôm: exact accuracy 37,5 → 50,0 % | Tập con bản thảo | Độ đúng thấp, chưa đủ làm nhãn | https://arxiv.org/abs/2510.04003 |
| A12 | Vo T.K.T., Nguyen N.H., Ha M.T. (2026) | MAPR 2026 (arXiv 2607.11434) | Dịch thẳng ảnh Hán-Nôm sang tiếng Việt hiện đại: CLIP ViT-L/14 + bert-base-chinese + PhoBERT, căn chỉnh sở thích (PPO/DPO/KTO; DPO tốt nhất) | Bản thảo Hán-Nôm | Đi từ ảnh ra nghĩa, không cho nhãn mức chữ | https://arxiv.org/abs/2607.11434 |
| A13 | Dinh D., Nguyen P., Nguyen L.H.B. (2021) | IJACSA 12(2) | Phiên âm Nôm sang Quốc ngữ bằng dịch máy thống kê (SMT), tổng quan và so sánh mô hình | Văn bản Nôm–QN | Chỉ làm trên văn bản, chiều Nôm→QN; không xử lý ảnh | https://doi.org/10.14569/IJACSA.2021.0120205 |
| A14 | Thai L.H., Nguyen L.H.B., Dinh D. (2022) | ITCC 2022 (ACM) | SMT Nôm→QN có thêm tri thức Hán-Việt | Văn bản | Như A13 | https://doi.org/10.1145/3548636.3548647 |
| A15 | Scius-Bertrand A., Jungo M., Wolf B., Fischer A., Bui M. (2021) | *Applied Sciences* 11(11):4894 | **Căn chỉnh bản phiên âm không cần nhãn người**: YOLOv5 học trên trang in tổng hợp từ phông, rồi tự huấn luyện (self-training) trên kết quả căn chỉnh; độ chính xác phát hiện 96,4 % | Bản thảo viết tay Việt (chữ Nôm) | **Gần nhất với đề tài.** Căn với bản Unicode chữ Nôm có sẵn (cùng hệ chữ), không phải với Quốc ngữ; chỉ đo độ chính xác phát hiện | https://doi.org/10.3390/app11114894 |
| A16 | Scius-Bertrand A., Jungo M., Wolf B., Fischer A., Bui M. (2021) | ICDAR 2021 (LNCS) | Phát hiện chữ Nôm trên văn bia không cần nhãn: bộ phát hiện học trên chữ in rồi tự hiệu chỉnh sang ảnh thật | Ảnh văn bia Việt Nam | Chỉ phát hiện, chưa gán nhãn chữ | https://doi.org/10.1007/978-3-030-86549-8_28 |
| A17 | Scius-Bertrand A., Studer L., Fischer A., Bui M. (2022) | S+SSPR 2022 (LNCS) | Tìm từ khoá (keyword spotting) không cần nhãn: bộ phát hiện học từ phông + so khớp đồ thị; truy vấn bằng mẫu viết tay hoặc chữ in | **Kieu database**: 719 trang Truyện Kiều viết tay | Truy hồi chứ không gán nhãn từng chữ | https://doi.org/10.1007/978-3-031-23028-8_3 |
| A18 | Diesbach J., Fischer A., Bui M., Scius-Bertrand A. (2022) | ICFHR 2022 (LNCS) | Sinh chữ Nôm mang phong cách văn bia bằng dịch ảnh không ghép cặp (CUT/GAN); 26.901 chữ, 21 phong cách; người xem đánh giá giống thật | Văn bia + phông Nôm | Đánh giá chủ yếu bằng cảm nhận, chưa đo lợi ích cho gán nhãn | https://doi.org/10.1007/978-3-031-21648-0_33 |
| A19 | Do T., Tran D.P., Vo A., Kim D. (2025) | AAAI 2025 (AISI) | Hậu xử lý OCR bằng LLM dựa trên **văn bản tham chiếu** (ebook) để tạo dữ liệu có nhãn cho văn bản Việt lịch sử (dấu thanh) | Sách Quốc ngữ cổ | Chữ Latinh có dấu, không phải chữ Nôm; tuy vậy cùng ý tưởng dùng văn bản tham chiếu làm giám sát | https://doi.org/10.1609/aaai.v39i27.35012 |

### B. OCR và phát hiện chữ trên tài liệu chữ Hán/Nhật lịch sử

| # | Tác giả (năm) | Nơi công bố | Phương pháp chính | Dữ liệu | Hạn chế liên quan | Liên kết |
|---|---|---|---|---|---|---|
| B1 | Zhou X., Wang D., Krähenbühl P. (2019) | arXiv 1904.07850 | **CenterNet**: biểu diễn đối tượng bằng tâm hộp, hồi quy kích thước và độ lệch | COCO | Mô hình chung; với chữ dính cần ràng buộc thêm (dự án ràng buộc số hộp N + seam carving) | https://arxiv.org/abs/1904.07850 |
| B2 | Yang H., Jin L., Huang W., Yang Z., Lai S., Sun J. (2018) | IEEE Access 6 | Bộ phát hiện chữ dày đặc có dẫn hướng nhận dạng; công bố bộ **TKH** và **MTH** | Sách cổ chữ Hán (mộc bản) | Chữ Hán, không có Nôm | https://doi.org/10.1109/ACCESS.2018.2840218 |
| B3 | Ma W., Zhang H., Jin L., Wu S., Wang J., Wang Y. (2020) | ICFHR 2020 | Khung end-to-end: nhánh phát hiện/nhận dạng chữ + nhánh bố cục (FCN), cơ chế chấm lại | **MTHv2** | Cần nhãn chữ đầy đủ | https://doi.org/10.1109/ICFHR2020.2020.00017 · https://arxiv.org/abs/2007.06890 |
| B4 | Liu C.-L., Yin F., Wang D.-H., Wang Q.-F. (2011) | ICDAR 2011 | Bộ CASIA-HWDB/OLHWDB: khoảng 3,9 triệu mẫu chữ đơn (7.356 lớp) + 5.090 trang văn bản, chú thích mức ký tự | Chữ Hán viết tay hiện đại | Chữ hiện đại; nhãn có được nhờ người viết theo văn bản cho sẵn | https://doi.org/10.1109/ICDAR.2011.17 |
| B5 | Clanuwat T., Bober-Irizar M., Kitamoto A., Lamb A., Yamamoto K., Ha D. (2018) | arXiv 1812.01718 (workshop NeurIPS 2018) | Bộ KMNIST, Kuzushiji-49, Kuzushiji-Kanji | Sách cổ Nhật (CODH) | Chữ đã cắt sẵn do chuyên gia | https://arxiv.org/abs/1812.01718 |
| B6 | Clanuwat T., Lamb A., Kitamoto A. (2019) | ICDAR 2019 | **KuroNet**: U-Net dư nhận dạng cả trang (vị trí + nhãn), không tiền xử lý; độ đúng > 90 % trên dữ liệu Kaggle | Kuzushiji (CODH) | Cần dữ liệu trang có nhãn chữ | https://doi.org/10.1109/ICDAR.2019.00103 |
| B7 | CODH / Kaggle (2019) | Cuộc thi Kaggle Kuzushiji Recognition (19/07–14/10/2019) | Lời giải top: phát hiện + phân loại; hạng 5 dùng **CenterNet một tầng** (0,940) | Kuzushiji (CODH) | Bằng chứng CenterNet hợp với chữ cổ dày đặc; không phải bài báo bình duyệt | https://codh.rois.ac.jp/competition/kaggle/ |
| B8 | Ueki K., Kojima T. (2020) | arXiv 2007.09637 | Tổng quan nhận dạng Kuzushiji bằng học sâu | — | Tổng quan | https://arxiv.org/abs/2007.09637 |
| B9 | Peng D., Jin L., Liu Y., Luo C., Lai S. (2022) | IJCV | **PageNet**: nhận dạng cả trang chữ Hán viết tay, giám sát yếu: dữ liệu thật chỉ cần bản phiên âm, hộp chữ/dòng sinh tự động qua vòng ghép–cập nhật–tối ưu | Chữ Hán viết tay | Bản phiên âm cùng hệ chữ với ảnh | https://doi.org/10.1007/s11263-022-01654-0 |
| B10 | Shi Y., Liu C., Peng D., Jian C., Huang J., Jin L. (2023) | NeurIPS 2023 D&B | **M5HisDoc**: chuẩn tài liệu cổ Trung Quốc đa phong cách (4.000 ảnh R + 4.000 ảnh H) | Sách cổ chữ Hán | Không có Nôm | https://proceedings.neurips.cc/paper_files/paper/2023/hash/f7b424d242cc6bb7708cff241367334d-Abstract-Datasets_and_Benchmarks.html |
| B11 | Shi Y., Peng D., Zhang Y., Cao J., Jin L. (2025) | *Scientific Data* | **HisDoc1B**: 40.281 cuốn, 3,16 triệu ảnh, 1,08 tỷ chữ, 30.615 lớp. Gán nhãn **bán tự động**: YOLOv7 + ViT + luật xếp chữ; mỗi cuốn gán tay một ảnh (giảm 98,7 % công sức) | Sách cổ chữ Hán | Vẫn có người gán hạt giống/tinh chỉnh; không có kênh văn bản ngoài | https://doi.org/10.1038/s41597-025-04495-x |
| B12 | Gao E.-H., Huang Y.-X., Hu W.-C., Zhu X.-H., Dai W.-Z. (2024) | AAAI 2024 | **KESAR**: học abductive kết hợp tri thức logic bậc nhất cho tách + nhận dạng tài liệu cổ | Tài liệu cổ | Tri thức hình thức hoá thủ công | https://doi.org/10.1609/aaai.v38i8.28683 |
| B13 | Wang P., Zhang K., … Jin L., Bai X., Liu Y. (2024) | *Scientific Data* 11:976 | **HUST-OBC**: 140.053 ảnh giáp cốt văn; pipeline bán tự động (ResNet-50 gán lớp, MoCo khử trùng), kiểm chứng phân loại 94,6 % | Giáp cốt văn | Miền khác; có khâu kiểm chứng người | https://doi.org/10.1038/s41597-024-03807-x |
| B14 | Li J., Chi X., Wang Q., Wang D., Huang K., Liu Y., Liu C.-L. (2024) | arXiv 2411.11354 | Tổng quan nhận dạng chữ giáp cốt: thách thức, chuẩn, phương pháp | — | Tổng quan | https://arxiv.org/abs/2411.11354 |
| B15 | Diao X., Shi D., Li J., … Xu H. (2023) | ACM MM 2023 | **ACCID**: chữ Hán cổ có chú thích mức bộ thủ, nhận dạng zero-shot bằng tách–ghép bộ | Chữ Hán cổ | Cần chú thích bộ thủ | https://doi.org/10.1145/3581783.3612201 |
| B16 | Diao X., Bo R., … Shi D. (2025) | arXiv 2506.19208 | Tổng quan nhận dạng ảnh chữ cổ (chữ tượng hình Ai Cập, giáp cốt…): few-shot, mất cân bằng, ảnh suy giảm | — | Tổng quan | https://arxiv.org/abs/2506.19208 |

### C. Gán nhãn yếu / căn chỉnh ảnh–bản phiên âm

| # | Tác giả (năm) | Nơi công bố | Phương pháp chính | Dữ liệu | Hạn chế liên quan | Liên kết |
|---|---|---|---|---|---|---|
| C1 | Kornfield E.M., Manmatha R., Allan J. (2004; 2007) | DIAL 2004; IJDAR 10(1) | Tách trang thành ảnh từ, ghép với văn bản bằng **DTW** trên đặc trưng hình học hộp từ (văn bản được vẽ bằng một phông đặc biệt); F 68,3 (theo dòng), 57,8 (cả trang) | Bản thảo tiếng Anh viết tay | Chữ Latinh, cùng hệ chữ, đơn vị là từ | https://ciir.cs.umass.edu/pubfiles/mm-320.pdf · https://doi.org/10.1007/s10032-006-0019-8 |
| C2 | Fischer A., Frinken V., Fornés A., Bunke H. (2011) | HIP 2011 (ACM) | Căn bản phiên âm với bản thảo Latinh bằng **HMM** | Bản thảo Latinh | Cùng hệ chữ | https://doi.org/10.1145/2037342.2037348 |
| C3 | Yin F., Wang Q.-F., Liu C.-L. (2013) | Pattern Recognition 46(10) | **Transcript mapping** cho chữ Hán viết tay: bài toán tối ưu Bayes kết hợp mô hình nhận dạng chữ + ngữ cảnh hình học; xử lý chữ dính và khe trong chữ | CASIA-HWDB | Cùng hệ chữ; cần bộ nhận dạng tốt cho chính hệ chữ đó | https://doi.org/10.1016/j.patcog.2013.03.013 |
| C4 | Wilkinson T., Nettelblad C. (2020) | arXiv 2003.11087 | Căn chỉnh bằng HMM không cần huấn luyện để tự sinh dữ liệu nhãn yếu cho word spotting; dùng 1–7 % dữ liệu có nhãn đầy đủ mà gần bằng giám sát đầy đủ | Bản thảo viết tay | Chữ Latinh | https://arxiv.org/abs/2003.11087 |
| C5 | Torras P., Souibgui M.A., Chen J., Fornés A. (2021) | ICDAR 2021 Workshops (LNCS) | Học căn chỉnh qua attention, chỉ cần bản phiên âm | Bản thảo/ký hiệu | Cùng hệ ký hiệu | https://doi.org/10.1007/978-3-030-86198-8_11 |
| C6 | Peer M., Scius-Bertrand A., Fischer A. (2025) | VisionDocs @ ICCV 2025 (arXiv 2508.07904) | Căn chỉnh CTC (quy hoạch động) bản phiên âm đầy đủ với ảnh dòng, tự huấn luyện để sửa lỗi chú thích; CER giảm 1,1 điểm; **mô hình yếu hơn cho căn chỉnh tin cậy hơn** | Thư Bullinger (thế kỷ 16) | Cùng hệ chữ; mức dòng | https://arxiv.org/abs/2508.07904 |
| (A15) | Scius-Bertrand et al. (2021) | *Applied Sciences* | Căn chỉnh không nhãn người cho chữ Nôm (xem A15) | | | |

### D. Học metric / kiểm chứng ảnh–chữ / nhận dạng tập mở

| # | Tác giả (năm) | Nơi công bố | Phương pháp chính | Dữ liệu | Hạn chế liên quan | Liên kết |
|---|---|---|---|---|---|---|
| D1 | Deng J., Guo J., Xue N., Zafeiriou S. (2019) | CVPR 2019 (bản mở rộng TPAMI) | **ArcFace**: lề góc cộng trên siêu cầu | Khuôn mặt | Cần nhãn lớp sạch; dự án học trên nhãn tự động nên có nguy cơ vòng tròn | https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html |
| D2 | Deng J., Guo J., Liu T., Gong M., Zafeiriou S. (2020) | ECCV 2020 (LNCS) | **Sub-center ArcFace**: K tâm con mỗi lớp, tâm trội gom mẫu sạch, tâm phụ hút mẫu nhiễu | Ảnh mặt web nhiễu | Thiết kế cho nhiễu nhãn kiểu web, chưa được kiểm trên chữ cổ | https://doi.org/10.1007/978-3-030-58621-8_43 |
| D3 | Ao X., Zhang X.-Y., Liu C.-L. (2022) | Pattern Recognition | **CMPL**: nguyên mẫu lớp lấy từ ảnh chữ in, chiếu vào không gian sâu; nhận dạng zero-shot chữ viết tay theo nguyên mẫu gần nhất | CASIA (on/offline) | Chữ Hán hiện đại; cần dữ liệu viết tay có nhãn để học ánh xạ | https://doi.org/10.1016/j.patcog.2022.108859 |
| D4 | Yu H., Wang X., Li B., Xue X. (2023) | ICCV 2023 | **CCR-CLIP**: tiền huấn luyện kiểu CLIP ghép ảnh chữ in với chuỗi mô tả IDS; dùng cho nhận dạng dòng | Chữ Hán | Cần IDS cho từng chữ (chữ Nôm hiếm thì có thể thiếu) | https://doi.org/10.1109/ICCV51070.2023.01097 |
| D5 | Vaze S., Han K., Vedaldi A., Zisserman A. (2022) | ICLR 2022 (oral) | Tập mở: điểm max-logit (MLS) của bộ phân loại tập đóng tốt đã đủ mạnh | Ảnh chung | Giả định bộ phân loại tập đóng đã tốt | https://arxiv.org/abs/2110.06207 |
| D6 | Liu W., Wang X., Owens J.D., Li Y. (2020) | NeurIPS 2020 | Điểm năng lượng để phát hiện mẫu ngoài phân phối (OOD) | CIFAR… | Như trên | https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html |
| D7 | Foret P., Kleiner A., Mobahi H., Neyshabur B. (2021) | ICLR 2021 (spotlight) | **SAM**: tối thiểu hoá độ sắc của hàm mất mát; bền với nhiễu nhãn | CIFAR, ImageNet | — | https://openreview.net/forum?id=6Tm1mposlrM |
| D8 | Oquab M., Darcet T., … (2024) | TMLR 01/2024 | **DINOv2**: đặc trưng thị giác tự giám sát | LVD-142M | Không chuyên cho chữ; repo đo được DINOv2 không phân biệt chữ Nôm | https://arxiv.org/abs/2304.07193 |

### E. Sinh glyph bằng phông / diffusion

| # | Tác giả (năm) | Nơi công bố | Phương pháp chính | Dữ liệu | Hạn chế liên quan | Liên kết |
|---|---|---|---|---|---|---|
| E1 | Yang Z., Peng D., Kong Y., Zhang Y., Yao C., Jin L. (2024) | AAAI 2024 | **FontDiffuser**: sinh phông one-shot bằng diffusion, mô-đun gom nội dung đa tỉ lệ + tinh chỉnh phong cách bằng học tương phản | Phông chữ Hán | Mục tiêu là phông đẹp, không phải độ giống chữ viết tay cổ; chưa kiểm cho Nôm | https://doi.org/10.1609/aaai.v38i7.28482 · https://arxiv.org/abs/2312.12142 |
| E2 | Tian Y. (2024) | ICCC 2024 | **DiffCJK**: diffusion có điều kiện sinh chữ CJK theo phong cách từ một chữ tham chiếu | CJK in và viết tay | Chưa đánh giá cho chữ Nôm | https://arxiv.org/abs/2404.05212 |
| E3 | Gui D., Chen K., Ding H., Huo Q. (2023) | arXiv 2305.15660 | DDPM biến glyph in thành mẫu "viết tay" để huấn luyện HCCR zero-shot (3.755 lớp), độ đúng gần mức học trên dữ liệu thật | CASIA-HWDB | Chữ Hán hiện đại | https://arxiv.org/abs/2305.15660 |
| E4 | Li J., Wang Q.-F., Wang S., Zhang R., Huang K., Cambria E. (2023/2024) | arXiv 2312.13631 | **Diff-Oracle**: diffusion có bộ mã hoá phong cách + nội dung, sinh chữ giáp cốt để tăng dữ liệu | Giáp cốt văn | Miền khác | https://arxiv.org/abs/2312.13631 |
| E5 | Xu T., Wang K., Chen Z., Wu L., Wen T., Chao F., Chen Y.-C. (2026) | ICLR 2026 | **UniCalli**: diffusion hợp nhất sinh + nhận dạng thư pháp mức cột; chuyển được sang giáp cốt văn, chữ tượng hình | Thư pháp chữ Hán | Mức cột, chi phí huấn luyện lớn | https://arxiv.org/abs/2510.13745 · https://openreview.net/forum?id=OSIPdrw56X |
| (A18) | Diesbach et al. (2022) | ICFHR 2022 | Sinh chữ Nôm phong cách văn bia (CUT) | | | |

### F. Mô hình đa phương thức lớn (VLM/LMM) cho OCR tài liệu cổ

| # | Tác giả (năm) | Nơi công bố | Phương pháp chính | Dữ liệu | Hạn chế liên quan | Liên kết |
|---|---|---|---|---|---|---|
| F1 | Wei H., Liu C., Chen J., Wang J., Kong L., … (2024) | arXiv 2409.01704 | **GOT-OCR2.0**: mô hình OCR end-to-end 580M tham số | Đa dạng | Không có kho chữ Nôm | https://arxiv.org/abs/2409.01704 |
| F2 | Bai S., Chen K., Liu X., Wang J., Ge W., … (2025) | arXiv 2502.13923 | **Qwen2.5-VL**: ViT độ phân giải động, phân tích tài liệu | Đa dạng | Chưa có đánh giá công bố trên chữ Nôm | https://arxiv.org/abs/2502.13923 |
| F3 | Cui C., Sun T., Liang S., Gao T., Zhang Z., … (2025) | arXiv 2510.14528 | **PaddleOCR-VL** 0,9B, phân tích tài liệu 109 ngôn ngữ | Đa dạng | Repo thử làm kênh phủ quyết kim: precision 47–59 % (đo nội bộ), nên bác | https://arxiv.org/abs/2510.14528 |
| F4 | Yu H., Wu Y., Shi F., … Li B. (2025) | arXiv 2509.09731 | **AncientDoc**: chuẩn VLM trên tài liệu cổ Trung Quốc, từ OCR đến suy luận tri thức; 5 tác vụ, 14 loại tài liệu, khoảng 3.000 trang | Tài liệu cổ chữ Hán | Không có Nôm | https://arxiv.org/abs/2509.09731 |
| F5 | Li G., Peng S., … Hu H. (2026) | arXiv 2605.11960 | **Chronicles-OCR**: 2.800 ảnh qua các giai đoạn chữ Hán. Nhận dạng chữ cổ tốt nhất 31,9 % (Kimi K2.5); giáp cốt văn chỉ 14,0 % (Gemini 3.1 Pro); NED từ 0,78 (khải thư) rơi xuống 0,06 (giáp cốt) | Chữ Hán qua các thời kỳ | Bằng chứng VLM tổng quát yếu với chữ cổ | https://arxiv.org/abs/2605.11960 |
| F6 | Zhou Z., Sun Y., He H., … Jin L. (2026) | arXiv 2608.07917 | **TongGuOCR**: MLLM OCR chuyên tài liệu cổ, tiền xử lý theo bố cục + mở rộng từ vựng chữ hiếm; 93,76 AR trên M5HisDoc | M5HisDoc | Chữ Hán; cần dữ liệu huấn luyện lớn có nhãn | https://arxiv.org/abs/2608.07917 |
| F7 | Chung Y.H.M., Choi D. (2025) | arXiv 2507.06761 | Fine-tune VLM (LLaMA-3.2-11B tốt nhất) làm OCR cho tiếng Mãn: 98,3 % từ trên dữ liệu tổng hợp, 93,1 % trên tài liệu viết tay thật | 60.000 ảnh từ tổng hợp | Ngôn ngữ ít tài nguyên khác; mức từ | https://arxiv.org/abs/2507.06761 |
| F8 | Li S., Ji T., … Huang X. (2024) | arXiv 2410.11302 | Hiện tượng **sycophancy** của VLM (chuẩn MM-SY) và cách giảm | — | Nếu cho VLM thấy nhãn đề xuất, nó dễ thuận theo | https://arxiv.org/abs/2410.11302 |
| F9 | Ren Y. (2026) | arXiv 2605.25781 (ARR 03/2026) | **Double Triangle Annotation**: hai MLLM khác kiến trúc cùng gán; trùng thì tự nhận, lệch thì chuyển người; tự nhận khoảng 85 % của 13.595 trường, WER 0,003 | Danh bạ y khoa Pháp 1887–1906 | **Giả định lỗi độc lập** giữa các mô hình, mâu thuẫn với G3 | https://arxiv.org/abs/2605.25781 |

### G. Hợp nhất nhãn nhiễu, suy luận thống kê, tài liệu hoá bộ dữ liệu (cơ sở phương pháp)

| # | Tác giả (năm) | Nơi công bố | Nội dung | Liên quan | Liên kết |
|---|---|---|---|---|---|
| G1 | Dawid A.P., Skene A.M. (1979) | JRSS Series C 28(1):20–28 | Ước lượng tỉ lệ lỗi của người gán bằng EM khi không có sự thật | Nền của gộp phiếu | https://www.jstor.org/stable/2346806 |
| G2 | Northcutt C.G., Jiang L., Chuang I.L. (2021) | JAIR 70:1373–1411 | Confident learning: ước lượng phân phối chung nhãn nhiễu/nhãn đúng, tìm lỗi nhãn | Tìm ô GOLD sai | https://doi.org/10.1613/jair.1.12125 |
| G3 | Kohli G. (2026) | arXiv 2605.29800 | 9 giám khảo LLM chỉ bằng khoảng 2 phiếu độc lập (Kish n_eff); hội đồng kém kỳ vọng 8–22 điểm | Kênh kim/NomNaOCR/VLM tương quan nhau | https://arxiv.org/abs/2605.29800 |
| G4 | Balasubramanian K., Podkopaev A., Kasiviswanathan S.P. (2026) | arXiv 2601.22336 | Gộp nhãn có tính phụ thuộc (mô hình Ising); Dawid–Skene có thể sai khi các giám khảo tương quan | Như trên | https://arxiv.org/abs/2601.22336 |
| G5 | Li H., Chen J., Ai Q., Chu Z., Zhou Y., Dong Q., Liu Y. (2024) | arXiv 2410.15393 | CalibraEval: giảm selection bias của LLM-giám khảo mà không cần nhãn | Thiết kế câu hỏi trắc nghiệm mù | https://arxiv.org/abs/2410.15393 |
| G6 | Angelopoulos A.N., Bates S., Fannjiang C., Jordan M.I., Zrnic T. (2023) | *Science* 382 | Prediction-powered inference: khoảng tin cậy hợp lệ khi trộn ít nhãn người với nhiều dự đoán máy | Ước lượng độ đúng GOLD | https://doi.org/10.1126/science.adi6000 |
| G7 | Candès E.J., Ilyas A., Zrnic T. (2025) | arXiv 2506.10908 | Probably Approximately Correct labels: gán nhãn tự động có bảo đảm chất lượng | Khung lý thuyết cho "GOLD có bảo đảm" | https://arxiv.org/abs/2506.10908 |
| G8 | Gebru T., Morgenstern J., Vecchione B., Vaughan J.W., Wallach H., Daumé III H., Crawford K. (2021) | CACM 12/2021 | Datasheets for Datasets | Tài liệu hoá bộ công bố | https://arxiv.org/abs/1803.09010 |
| G9 | Akhtar M., Benjelloun O., Conforti C., … (2024) | NeurIPS 2024 D&B | Croissant: siêu dữ liệu ML-ready | Xuất bộ dữ liệu | https://proceedings.neurips.cc/paper_files/paper/2024/file/9547b09b722f2948ff3ddb5d86002bc0-Paper-Datasets_and_Benchmarks_Track.pdf |
| G10 | Unicode Consortium | UAX #38 (Unihan), Unicode 18.0, rev. 41 | Trường `kVietnamese`: "cách đọc tiếng Việt của chữ, theo Quốc ngữ" (Provisional) | Kênh âm độc lập với từ điển của dự án | https://www.unicode.org/reports/tr38/ |

### H. Tài nguyên (không phải bài báo, đã kiểm có tồn tại)

| Tài nguyên | Đã kiểm | Liên kết |
|---|---|---|
| Kim Hán Nôm (VCL/CLC, ĐH KHTN – ĐHQG TP.HCM): OCR Hán-Nôm, chuyển tự Hán Nôm↔Quốc ngữ, có API | Trang giới thiệu của trường/trung tâm. **Không tìm thấy bài báo bình duyệt mô tả mô hình OCR**, nên luận văn phải coi đây là hộp đen | https://www.clc.hcmus.edu.vn/?p=3737 · https://en.hcmus.edu.vn/ai-unlocks-han-nom-heritage-opening-the-door-to-a-millennium-of-knowledge-for-wider-public-access/ |
| Thư viện số Vatican, nhóm Borg.tonch (41 mục) | Danh mục có liệt kê; không tìm thấy bộ dữ liệu phiên âm Borg.tonch.18/34 đã công bố | https://digi.vatlib.it/mss/ |
| Dự án IHR-Nom (LaBRI) | Trang tải dữ liệu | https://morphoboid.labri.fr/ihr-nom.html |
| HF `dzungpham/font-architect` (FontDiffuser cộng đồng) | Model card | https://huggingface.co/dzungpham/font-architect |

---

## 3. (b) Tổng hợp: các công trình trước đã làm gì

1. **OCR chữ Nôm qua ba thế hệ.** (i) Tách chữ theo hình học (projection profile, X–Y cut, Voronoi) rồi phân loại thống kê
   (GLVQ, MQDF), học từ mẫu **sinh bằng phông** do thiếu nhãn thật (A1–A3). (ii) CNN sâu cho 32 nghìn lớp và tách chữ bằng CNN
   (A4–A5). (iii) Nhận dạng mức dòng CRNN-CTC trên bộ IHR-NomDB và NomNaOCR (A6–A7), fine-tune PaddleOCR (A11), thêm mô hình
   ngôn ngữ Nôm (A8), mở sang ảnh cảnh (A10) và dịch thẳng ảnh sang nghĩa bằng mô hình đa phương thức (A12). Các bộ dữ liệu Nôm
   công khai chỉ có nhãn **mức dòng/cột** (IHR-NomDB, NomNaOCR). Độ đúng mức chuỗi còn thấp (29–43 %), độ đúng mức ký tự
   khoảng 82–85 %.
2. **Tài liệu chữ Hán/Nhật cổ.** Kiến trúc chủ đạo là *phát hiện chữ, rồi phân loại* (CenterNet, Faster R-CNN, YOLO; B1, B7)
   hoặc mô hình cả trang (KuroNet B6, MTHv2 B3, PageNet B9). Các bộ dữ liệu cỡ lớn (HisDoc1B, HUST-OBC) được dựng bằng **pipeline
   bán tự động**: mô hình sinh nhãn giả, người gán hạt giống hoặc kiểm chứng (B11, B13). Tri thức chuyên ngành được đưa vào
   qua logic (KESAR, B12) hoặc bộ thủ/IDS (B15, D4).
3. **Căn chỉnh ảnh với bản phiên âm có sẵn.** Hướng này đi từ DTW/HMM trên đặc trưng từ (C1, C2), qua tối ưu Bayes kết hợp bộ
   nhận dạng và hình học cho chữ Hán (C3), tới huấn luyện giám sát yếu chỉ cần bản phiên âm (C4, B9), căn bằng attention/CTC
   (C5, C6). Riêng với chữ Nôm, nhóm Fribourg (A15–A17) đã làm **căn chỉnh và phát hiện không cần nhãn người**: bộ phát hiện học
   trên trang tổng hợp từ phông, rồi tự huấn luyện trên chính kết quả căn chỉnh. Mọi công trình nói trên đều căn ảnh với văn
   bản **cùng hệ chữ** (Unicode chữ Nôm/chữ Hán, hoặc chữ Latinh với ảnh chữ Latinh).
4. **Kiểm chứng ảnh–chữ.** Họ hàm mất mát lề góc (ArcFace, sub-center ArcFace cho nhãn nhiễu; D1–D2) là chuẩn cho embedding
   phân biệt. Nhận dạng zero-shot chữ Hán dùng **nguyên mẫu từ chữ in** (CMPL D3) hoặc ghép ảnh–IDS kiểu CLIP (D4). Quyết định
   từ chối dựa trên max-logit/năng lượng (D5–D6).
5. **Sinh glyph.** Diffusion sinh phông one-shot (FontDiffuser, DiffCJK; E1–E2), biến chữ in thành chữ "viết tay" để huấn luyện
   (E3), tăng dữ liệu chữ cổ (Diff-Oracle E4, UniCalli E5), GAN phong cách văn bia cho chữ Nôm (A18).
6. **VLM/LMM.** Mô hình OCR tổng quát (GOT, Qwen2.5-VL, PaddleOCR-VL; F1–F3) mạnh trên tài liệu hiện đại nhưng **yếu rõ rệt với
   chữ cổ** (Chronicles-OCR F5, AncientDoc F4). MLLM chuyên dụng (TongGuOCR F6) hoặc VLM fine-tune trên dữ liệu tổng hợp (F7) thì
   tốt hơn, nhưng cần dữ liệu có nhãn. Dùng hai MLLM đồng thuận để tự gán nhãn (F9) dựa trên giả định lỗi độc lập, trong khi G3–G4
   cho thấy giả định này sai với các mô hình tương quan, và VLM còn có xu hướng thuận theo nhãn được gợi ý (F8).

---

## 4. (c) Định vị tính mới của đề tài (đối chiếu với những gì pipeline THỰC SỰ làm)

### 4.1 Pipeline thực tế (theo mã/tài liệu trong repo)

| Thành phần | Pipeline làm gì | Nguồn trong repo |
|---|---|---|
| OCR | Gọi API **Kim Hán Nôm** (HCMUS) như một hộp đen; `lang_type` 1 (Hán) cho STT, 2 (Nôm) cho sách in và Borg | `docs/DIEU_HUONG_PIPELINE_2026-09-28.md` §0; `README.md:109` |
| Đơn vị chữ | **CenterNet** (ResNet34 + FPN, tâm chữ), tiền huấn luyện trên **MTH/TKH**, fine-tune trên 445 trang Nôm bằng nhãn tự động; ép đúng N hộp (N = số âm Quốc ngữ của cột) bằng prune + **seam carving**; VAL box-F1 ≈ 0,84 | `train_crop/README.md` |
| Căn chỉnh | DP có dải / Levenshtein giữa chuỗi chữ kim và âm tiết Quốc ngữ theo cột; tập ứng viên R(âm) lấy từ từ điển Nôm↔QN (`Dict/QuocNgu_SinoNom.csv`) và Unihan `kVietnamese` | `pipeline/align_engine/align_production.py`, `anchor_align.py`; `Dict/_sources/README.md` |
| Hộp theo thị giác (TN6) | DP đơn điệu (Viterbi + forward-backward) trên đơn vị detector. Phát xạ = max cos(crop, glyph c), c ∈ R(âm), với glyph là **render phông NomNaTong/Plangothic + ảnh FontDiffuser**, cộng một vòng nguyên mẫu tự học lấy từ khối trang khác. Bật theo bộ (visual_dp / hybrid / pitch / legacy) | `pipeline/align_engine/visual_dp.py`; `docs/HOP_ANH_TN6_2026-09-28.md` §3 |
| Embedding | nom-embed v1 (ResNet18 + ArcFace, 1.591 lớp) và ArcFace v2 (**sub-center** K = 3, SAM, chia tập theo trang/LOBO), cả hai học trên nhãn tự động | `ArcFace/README.md`; `pipeline/gold_exact/export_assets.py:419` |
| Bộ kiểm ảnh↔chữ | CNN riêng (thân kiểu ResNet, hai đầu chiếu crop/glyph), mất mát softmax lề cosin trên lớp + tập ứng viên cứng; học trên crop thật + glyph tổng hợp có tăng cường (affine, nét, mực láng giềng); ngưỡng τ = 0,995 chọn theo LOBO | `lab/thu_nghiem_anh_chu/TN4_crop_chuan/h01_train_tn4.py`; `pipeline/gold_exact/signals_img.py`; `docs/GOLD_CHINH_XAC_2026-09-27.md` |
| Chọn chữ | Hai tầng: logit có điều kiện trên ứng viên R(âm) ∪ {kim} ∪ {lt2}, rồi hồi quy logistic mức ô. Với Borg, học LOBO theo sách (nguyên mẫu người lấy từ sách Borg còn lại) | `pipeline/chon_chu/model.py`; `CLAUDE.md` |
| Glyph sinh | FontDiffuser **chỉ suy luận** bằng checkpoint cộng đồng, ảnh phong cách lấy từ một trang STT2, sinh kho khoảng 21.837 glyph một lần trên Kaggle. **Không huấn luyện diffusion** | `lab/kaggle_diffusion/README.md`; HF `dzungpham/font-architect` |
| Chính sách nhãn | Tầng GOLD/SYLLABLE/REVIEW/QUARANTINE; cổng `gold_exact` **chỉ hạ, không sửa nhãn**; crop chuẩn v2 chỉ cho ô ok | `docs/DIEU_HUONG_PIPELINE_2026-09-28.md` §1 |
| Đánh giá | Đo trên nhãn người IHR-NomDB (mộc bản L16, TK) và Borg.tonch.18/34 (viết tay Công giáo); thạch bản chỉ ước lượng qua dị bản; STT là suy đoán | `docs/BAO_CAO_TONG_HOP_2026-10-01.md` §2–3 |

Số đo chính (`docs/BAO_CAO_TONG_HOP_2026-10-01.md`; tệp này đang sửa dở, chưa commit):
- Nhãn GOLD trùng chữ người: L16 **97,95 %** [97,69–98,18] (n 13.117), TK **98,60 %** (n 20.850), Borg Kinh **94,40 %**
  (n 72.693), Borg DungLy **96,70 %** (n 12.345).
- Ô `gold_exact = ok` (đúng cả nhãn lẫn crop): L16 99,21 %, TK 99,88 %.
- Riêng kim ở ô: mộc bản 95,2–97,5 %, nhưng viết tay Borg chỉ 38,0–50,5 % (khớp chặt).

Như vậy trên chữ viết tay, phần căn chỉnh + từ điển + kiểm thị giác nâng độ đúng của tầng GOLD lên rất xa so với kim
(chỉ tính trên tập con được giữ lại làm GOLD).

### 4.2 Thứ KHÔNG được tuyên bố là mới (hoặc không có trong pipeline)

- **Không** huấn luyện mô hình OCR Transformer/CRNN nào. OCR là kim (hộp đen bên ngoài).
- **Không** dùng LMM/VLM trong đường chạy. `pipeline/consensus_fusion/qwen_verifier.py` chỉ là khung mã, không được
  `run_pipeline.sh` gọi. PaddleOCR/GLM-OCR/NomNaOCR đã được thử rồi **bác** (`DIEU_HUONG` mục 7).
- **Không** huấn luyện diffusion. FontDiffuser chỉ dùng để sinh ảnh tham chiếu.
- CenterNet, ArcFace, sub-center ArcFace, SAM, Viterbi/forward-backward, DP căn chuỗi, seam carving đều là **kỹ thuật có sẵn**.
- **Không** "không dùng nhãn người hoàn toàn" theo nghĩa tuyệt đối. Nhãn người IHR/Borg được dùng để chọn ngưỡng (τ của TN1,
  LOBO) và làm nguyên mẫu viết tay cho Borg/STT (LOBO theo sách). Có thể nói chính xác là: *không dùng nhãn người của chính cuốn
  sách đang được gán/đo*.
- Embedding và detector học trên nhãn tự động, nên vẫn còn **vòng tròn** (chia theo trang/LOBO chỉ giảm, không loại bỏ). Từ điển
  Nôm↔QN vừa sinh ứng viên vừa từng được dùng làm "bằng chứng", nên không được coi là kênh kiểm độc lập (Unihan thì độc lập).

### 4.3 Điểm có thể khẳng định là mới (có điều kiện, dùng chữ "theo khảo sát của chúng tôi")

1. **Căn chỉnh khác hệ chữ: ảnh chữ Nôm (biểu ý) với bản Quốc ngữ (biểu âm), ở mức từng chữ.** Các công trình căn chỉnh trước
   (C1–C6, A15, B9) đều căn với bản phiên âm cùng hệ chữ. Ở đây một âm ứng với nhiều chữ, nên phải giải quyết bằng kết hợp OCR
   ngoài + từ điển âm→chữ + độ giống thị giác. Chúng tôi **không tìm thấy** công trình đã công bố nào làm việc này cho chữ Nôm
   (đã tìm theo các từ khoá ở §2). Điều này không chứng minh rằng chưa ai làm.
2. **Phát xạ thị giác ràng buộc bởi từ điển trong DP đơn điệu** (visual_dp): mỗi đơn vị detector được chấm với glyph phông +
   glyph FontDiffuser *chỉ trong tập R(âm)*, kèm nguyên mẫu tự học lấy từ khối trang khác. Về khái niệm, đây là biến thể khác
   hệ chữ của "transcript mapping dựa trên bộ nhận dạng" (C3) và "nguyên mẫu từ chữ in" (D3). Phần mới nằm ở cách ghép, không ở
   từng khối. Số đo: đúng vị trí so với hộp người Borg từ 61,5 lên 99,4 % (Kinh), từ 58,6 lên 99,8 % (DungLy).
3. **Bộ dữ liệu crop mức chữ có tầng tin cậy, chính sách "chỉ hạ".** GOLD còn tách thêm `gold_exact` (đúng cả ảnh lẫn chữ). So
   với các pipeline bán tự động (B11, B13) vốn có người tinh chỉnh, đề tài đặt bài toán là tự động hoàn toàn trên sách sản xuất
   và chỉ dùng nhãn người để đo.
4. **Đánh giá trên hai thể loại có nhãn người khác nhau** (mộc bản IHR, viết tay Công giáo Borg), mỗi số ghi rõ mức chắc
   ĐO / ƯỚC LƯỢNG / SUY ĐOÁN, đi kèm bộ đo tái lập có bất biến (`scripts/measure/`). Đây là đóng góp về phương pháp luận đo,
   không phải về mô hình.
5. **Bằng chứng thực nghiệm phủ định** có giá trị trích dẫn: VLM/engine OCR khác không bổ trợ được kim trên Nôm (precision phủ
   quyết 47–59 %); DINOv2 không phân biệt được chữ Nôm; đọc kim nhiều lượt có lỗi trùng 71–96 %. Các kết quả này thống nhất với
   F5 và G3.

**Bắt buộc so sánh trong luận văn:** A15 (Scius-Bertrand et al., 2021) là đối chứng gần nhất. Cần nói rõ khác biệt: nguồn văn bản
(Unicode Nôm so với Quốc ngữ), OCR ngoài, tầng nhãn và cách đo. Nếu được, nên chạy lại ý tưởng của họ (detector học từ phông +
self-training) trên IHR làm baseline.

---

## 5. (d) Khoảng trống còn lại

1. **Chưa có chuẩn công khai mức chữ cho Nôm.** IHR-NomDB chỉ có hộp cột, NomNaOCR chỉ có dòng. "Khe người" mà dự án dùng trên IHR
   là một mô hình vị trí, không phải hộp chữ do người vẽ. Hộp chữ người chỉ có ở Borg. Cần một tập con có hộp chữ đánh tay để
   đo detector/crop.
2. **Chữ viết tay vẫn yếu.** Kim chỉ đúng 38–54 % ở ô trên Borg; độ phủ GOLD của Borg DungLy là 68,7 %. STT (bộ chính) **không có
   sự thật người**, nên độ đúng chỉ là suy đoán. Thạch bản (L83, KVK) chỉ có cận dưới qua dị bản.
3. **Rò dữ liệu và tính độc lập của kênh.** Kim là hộp đen, không rõ đã học trên Nôm Foundation/IHR hay chưa. Nếu có, số 95–97 %
   trên IHR là lạc quan. NomNaOCR trùng nguồn với IHR. Các kênh tương quan làm số phiếu hiệu dụng nhỏ hơn số phiếu danh nghĩa (G3).
4. **Vòng tròn khi học embedding/detector từ nhãn tự động.** Chưa có thí nghiệm định lượng tách bạch mức vòng tròn (ví dụ huấn
   luyện chỉ trên sách khác rồi đo trên IHR/Borg).
5. **Dị thể, chữ PUA, âm một-nhiều.** Thước "V1+" và việc loại PUA khỏi phép đo là quy ước riêng của dự án; tài liệu ngoài chưa
   có chuẩn chung cho dị thể Nôm.
6. **Chưa có thí nghiệm downstream.** Trong `docs/` không thấy thí nghiệm huấn luyện một OCR/bộ phân loại trên bộ GOLD rồi đo trên
   tập ngoài (IHR, NomNaOCR test). Đây là bằng chứng giá trị sử dụng mà phản biện thường đòi.
7. **LMM trên Nôm chưa được đánh giá công khai.** Chúng tôi không tìm thấy chuẩn VLM nào cho chữ Nôm (các chuẩn F4–F5 chỉ có chữ
   Hán). Đây là cơ hội, nhưng nằm ngoài pipeline hiện tại.
8. **Sinh glyph chưa được đo riêng.** Chưa có ablation cho thấy ảnh FontDiffuser góp bao nhiêu vào phát xạ visual_dp / bộ kiểm,
   so với chỉ render phông. Cũng chưa có so sánh với GAN văn bia (A18).
9. **Bảo đảm thống kê cho GOLD.** Số GOLD hiện là ước lượng điểm + CI trên sách có nhãn. Khung PPI/PAC-labels (G6–G7) có thể cho
   bảo đảm trên sách không nhãn nhưng chưa áp dụng.

---

## 6. Chưa xác minh / loại khỏi danh sách chính

| Mục | Lý do |
|---|---|
| "Transliteration of Vietnamese National Scripts into Sino-Nom Scripts Using Transformer-based Model" (Springer, 10.1007/978-3-032-21625-0_10) | Chỉ thấy đoạn trích tìm kiếm (T5, 7 triệu cặp Việt–Hán, BLEU 69,73); chưa mở được trang, chưa rõ năm/hội nghị |
| "Parallel Corpus Construction for Chinese and Vietnamese in Historical Texts" (Springer 10.1007/978-3-031-98164-7_15) | Chỉ có tiêu đề |
| "Transliterating Nom Script into Vietnamese National Script Using Multilingual NMT" (Springer 978-981-99-7666-9_11); "Collecting automatically Sino-Nom variants…" (jol.vn) | Chỉ có tiêu đề |
| Phan T.V. (2011) "Development of Nom character segmentation for collecting patterns from historical document pages", HIP 2011, tr. 133–139 (ACM 10.1145/2037342.2037365) | Chỉ có trang liệt kê, chưa xác nhận đủ danh sách tác giả |
| Kaggle "New-SinoNom Dataset" và notebook "SinoNom Similarity Retrieval" (`README.md:405–406`) | Chỉ xác minh được tiêu đề dataset; nội dung, giấy phép, tác giả chưa đọc được |
| Việc arXiv 2508.07904 dùng kraken `ForcedAlignmentTaskModel` (`consensus_fusion/README.md:64`) | Không kiểm được trong tóm tắt |
| Kish (1965), hiệu ứng thiết kế (design effect) | Chưa tra nguồn gốc; nếu trích thì cần xác minh |
| Các số "Qwen3-VL-235B/flash" trong ghi chú nội bộ của repo | Là đo nội bộ, không phải tài liệu ngoài; không tìm thấy bài công bố về Qwen trên Nôm |
| "Survey on Vietnamese Document Analysis and Recognition" (arXiv 2506.05061) | Có thật (Le, Lam, Nguyen 2025), nhưng phần tóm tắt không nói về chữ Nôm, nên không đưa vào bảng |

---

## 7. Danh sách nên trích tối thiểu trong chương "Tổng quan" (gợi ý)

A6 IHR-NomDB, A7 NomNaOCR, A5 Nguyen et al. 2020, A3 Phan et al. 2016, A15 Scius-Bertrand et al. 2021 (bắt buộc),
A16–A18, A8, A13 (phiên âm Nôm↔QN), B1 CenterNet, B2/B3 TKH-MTH/MTHv2 (dữ liệu tiền huấn luyện detector), B9 PageNet,
B11 HisDoc1B, C3 Yin et al. 2013, C6 Peer et al. 2025, D1–D3, E1 FontDiffuser, F4–F5 (VLM yếu trên chữ cổ), G3, G6, G8, G10.
