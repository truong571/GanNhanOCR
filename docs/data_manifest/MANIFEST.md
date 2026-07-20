# DỮ LIỆU NGOÀI ĐÃ CHUYỂN KHỎI REPO LUẬN VĂN
# Ngày: 2026-07-20 (Giai đoạn 2 — Dọn rác an toàn)
# Repo: /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
#
# Tất cả đều là dữ liệu BÊN THỨ BA hoặc artifact tái sinh được,
# KHÔNG có trong git (gitignored), 0 tham chiếu mã hoặc có cờ ghi đè.

## MTH_TKHMTH2200
   dung lượng : 4.7G
   số file    : 12796
## kkanji2
   dung lượng : 549M
   số file    : 140424
## font_diffusion_ckpt_failed
   dung lượng : 1.3G
   số file    : 16

## Nguồn gốc và cách khôi phục

### MTH_TKHMTH2200  (nguyên là <repo>/MTH/TKHMTH2200)
- Nguồn: HCIILAB (SCUT) — TKH/MTH Datasets, https://github.com/HCIILAB/TKH_MTH_Datasets_Release
- Vai trò: dữ liệu PRETRAIN cho detector CenterNet (~1,08M box, cùng miền ván khắc CJK).
  Đã "kết tinh" vào checkpoint detector đang dùng -> chỉ cần khi PRETRAIN LẠI.
- Còn tham chiếu trong mã: train_crop/build_mth_pretrain.py (đường dẫn MẶC ĐỊNH).
  KHÔI PHỤC KHI CẦN — không phải chép về, chỉ cần trỏ đường dẫn:
      python train_crop/build_mth_pretrain.py --mth-root ~/ThS_archive/external_data/MTH_TKHMTH2200
- GIỮ LẠI TRONG REPO: MTH/MTHv2_Datasets_Release/ (2 MB, readme + train/test split)
  vì cần trích dẫn trong luận văn.

### kkanji2  (nguyên là <repo>/kkanji2)
- Nguồn: Kuzushiji-Kanji (CODH / Kaggle Kuzushiji Recognition), 3.832 lớp ký tự.
- Vai trò: KHÔNG dùng. Đã grep toàn repo: 0 tham chiếu trong *.py *.sh *.ipynb *.yaml.
  Tải về để cân nhắc pretrain nhưng cuối cùng dùng MTH/TKH (cùng miền hơn).
- Khôi phục: chép về <repo>/kkanji2 nếu sau này muốn thử pretrain Kuzushiji.

### font_diffusion_ckpt_failed  (nguyên là <repo>/font_diffusion/ckpt/{FST,DRO-20260227-19P2})
- Vai trò: các lần train FontDiffuser KHÔNG thành công (FST mất các step cuối 9k/15k).
- GIỮ LẠI TRONG REPO: font_diffusion/ckpt/PROD/ (383 MB) — đây là checkpoint ĐANG DÙNG
  để sinh kho glyph gannhanocr-fd.
- Khôi phục: chỉ cần nếu muốn phân tích lại quá trình train thất bại.

## Lưu ý
Kho glyph đã sinh (gannhanocr-fd, 89.898 file) KHÔNG nằm ở đây — nó là submodule
HuggingFace, xem .gitmodules. Nghĩa là xoá thư mục này KHÔNG ảnh hưởng tín hiệu S3.
