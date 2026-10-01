"""stt_hai_luot — ĐƯỜNG STT HAI LƯỢT CHẶT (TN9, 01/10/2026; lab/thu_nghiem_kim/TN9_stt_in, measure_out/_tn9/KET_QUA.md).

  doc.py    cache kim l1skel_l2 (khung lt1 + chữ lt2 gióng chuỗi) -> prepared/<Sách>/kim_l1skel_l2/   (0 API)
  route.py  hợp theo ô hai bản dựng (lt1 ∪ l1skel_l2) + cổng chặt R4 + cổng hộp visual_dp lệch syl_index
  __main__  CLI: cache | info | aux-config | union | gate | selftest
Cấu hình: config/pipeline.yaml mục `stt_hai_luot` + books[].kim_read / box_decoder.
"""
