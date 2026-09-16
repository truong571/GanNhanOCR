# HƯỚNG DẪN CHẠY KHỐI B (CNN ÂM TIẾT 5-FOLD OUT-OF-FOLD) TRÊN KAGGLE GPU

Thư mục `KhoiB/` này chứa toàn bộ công cụ để bạn chạy song song tác vụ huấn luyện CNN trên Kaggle GPU (T4 hoặc P100), giúp tiết kiệm thời gian cho máy Mac.

---

### 📦 1. CÁC TỆP ĐÃ ĐƯỢC CHUẨN BỊ SẴN TRONG `KhoiB/`

| Tệp | Mô tả |
|---|---|
| **`KhoiB_kaggle_dataset.zip`** *(74.9 MB)* | File nén chứa sẵn `crops.npz` (82.780 ảnh), `labels_final.csv` và `train_oof_cnn.py`. Dùng để tải lên Kaggle Dataset. |
| **`kaggle_run_khoib.ipynb`** | Notebook Jupyter đã viết sẵn các bước chạy, kiểm tra và xuất kết quả trên Kaggle. |
| **`train_oof_cnn.py`** | Script Python huấn luyện 5-fold CNN với mixed-precision fp16, tự động nhận diện GPU CUDA. |
| **`pack_for_kaggle.py`** | Script đóng gói lại dữ liệu trên Mac nếu có cập nhật mới (`.venv/bin/python KhoiB/pack_for_kaggle.py`). |

---

### 🚀 2. HƯỚNG DẪN 4 BƯỚC CHẠY TRÊN KAGGLE (MẤT ~12–15 PHÚT)

#### BƯỚC 1: Tải dữ liệu lên Kaggle Dataset
1. Mở trình duyệt, đăng nhập vào **[kaggle.com](https://www.kaggle.com)**.
2. Ở thanh menu bên trái, chọn **Datasets** $\rightarrow$ bấm nút **+ New Dataset** (góc trên bên phải).
3. Kéo thả tệp **`KhoiB/KhoiB_kaggle_dataset.zip`** vào khung upload.
4. Đặt tiêu đề Dataset: `khoib-data` (hoặc tên tùy ý) $\rightarrow$ bấm **Create**.

#### BƯỚC 2: Tạo Notebook và Chọn GPU
1. Bấm **+ New Notebook** trên Kaggle.
2. Ở panel bên phải phần **Notebook options**:
   - **Accelerator:** Chọn **GPU T4 x2** (hoặc **GPU P100**).
   - **Internet:** Bật **On** (hoặc Off đều được, code không cần mạng).
3. Bấm nút **+ Add Input** (ở panel bên phải) $\rightarrow$ tìm chọn dataset `khoib-data` vừa tạo ở Bước 1.

#### BƯỚC 3: Chạy Notebook
1. Chọn menu **File** $\rightarrow$ **Upload Notebook** $\rightarrow$ chọn tệp **`KhoiB/kaggle_run_khoib.ipynb`** từ máy tính của bạn.
2. Bấm nút **Run All** (hoặc nhấn `Shift + Enter` từng cell).
3. Theo dõi quá trình huấn luyện 5 Folds:
   - Mỗi Fold mất khoảng $\approx 2.5$ phút.
   - Tổng cộng 5 Folds hoàn thành trong khoảng **10 – 12 phút**.

#### BƯỚC 4: Tải kết quả về máy Mac
1. Sau khi chạy xong, ở cell cuối cùng, hệ thống tự động nén kết quả vào:  
   `/kaggle/working/p_visual_oof_results.zip`
2. Ở panel bên phải mục **Output**, tìm tệp `p_visual_oof_results.zip` $\rightarrow$ bấm dấu ba chấm $\rightarrow$ chọn **Download**.
3. Giải nén trên Mac và thả tệp **`p_visual_oof.csv`** vào thư mục dự án:
   - Vị trí: `lab/gan_nhan_2026-09-13/p_visual_oof.csv` (hoặc `dataset_out_v3/p_visual_oof.csv`).

---

### 🎯 3. KẾT QUẢ TỆP `p_visual_oof.csv` ĐEM LẠI LỢI ÍCH GÌ?
Khi có tệp này gửi về:
- **Mở khóa Hạng mục B-2:** Đưa phát xạ ảnh $P(\text{âm } j \mid \text{crop } i)$ vào DP ($\lambda = 0.25$), nâng độ định vị khe đúng từ $86.6\% \rightarrow 95.4\%$.
- **Mở khóa Hạng mục B-3:** Tự động thăng hạng $\approx 960$ ô từ REVIEW lên SYLLABLE với độ tin cậy $P \ge 0.90$.
