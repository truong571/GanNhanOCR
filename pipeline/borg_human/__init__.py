"""pipeline.borg_human — BỘ CROP NHÃN NGƯỜI từ 2 bản chép tay Vatican Borgiano Tonchinese
(Borg.tonch.18 = SachKinhThayCaBinh, Borg.tonch.34 = SachDungLyHoThan): GIÓNG chữ Nôm do người phiên gõ với hộp chữ trên ảnh.

Mã tái lập của phương pháp vòng 4 (r4/borg_align: detector v1, nhóm cột RTL, DP gióng với phát xạ thị giác font/nguyên mẫu,
luật keep) + làm sạch vòng 5 (r5/borg_quality: lọc Paddle độc lập trên hồ sơ đóng băng -> keep_v5). 0 API, không tải gì.
Chạy: `.venv/bin/python -m pipeline.borg_human --stage all` -> dataset/_BORG_NHAN_NGUOI/ (TÁCH khỏi GOLD tự động, dataset/_ALL).
Tham số đóng băng + nguồn: params.py · tài liệu: docs/BORG_NHAN_NGUOI_2026-09-27.md · nghiệm thu: scripts/measure/borg_human_eval.py
"""
