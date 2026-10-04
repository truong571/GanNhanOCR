# Kaggle: sinh ảnh glyph theo phong cách TỪNG cuốn cho đủ 10 bộ — việc cần làm (04/10/2026, bản v3: notebook dựng sẵn + đẩy HF)

> **CẢNH BÁO**: bộ chạy này **chưa từng chạy trên GPU Kaggle thật và HuggingFace thật**. Đã kiểm bằng: bộ sinh giả (34/34), mô hình thật trên Mac (MPS/CPU, 6/6 + 5/5 + 1 việc thật qua dòng lệnh), 22 phép kiểm vá lỗi (HF giả, chữ ký hàm khớp thư viện thật),
> **notebook chạy thật trong IPython kernel với Kaggle/HF giả (11/11)**, Python 3.10 và 3.14, cùng 3 người phản biện độc lập. **Ca thử (bước 4) là phép kiểm đầu tiên** — đừng bỏ qua.

## 0. Tóm tắt

- **Dữ liệu đã là mới nhất** (lượt dựng pipeline gần nhất 01/10, không có tệp nào dưới `dataset/` mới hơn 01/10 20:40; làm mới 04/10 cho cùng `plan_sha` 3f0dcc53… và `assets_sha` 03e119f4… ⇒ dữ liệu không đổi).
  **Chỉ dùng bản v3**: các gói `kaggle_pack_20261004` (v1), `_v2` và `kaggle_pack_20261004-1834` (do lần chạy `p11a` đầu của bạn tạo) cùng dữ liệu nhưng mang bộ chạy cũ (chưa có preflight, log gọn, đẩy HF 5 phút).
- **Hai tệp đưa lên Kaggle** (cùng tiền tố trong `measure_out/_tn11/full/`):
  1. `kaggle_pack_20261004_v3.zip` (≈ 472 MB) → làm **Dataset** (gói: kế hoạch, bộ chạy, ckpt FontDiffuser, phông, 10 ảnh phong cách);
  2. `kaggle_pack_20261004_v3.ipynb` → **Import Notebook** (3 ô mã: cấu hình + preflight · chạy · tiến độ). Sinh bởi `p13_tao_notebook_kaggle.py`.
- **10 cuốn**: stt2, stt4, stt11 (tách theo cột `book`), Chr, L83, KVK, L16, TK, B18, B34. Mỗi cuốn một **ảnh phong cách riêng** (crop chuẩn của một ô neo tự động sạch, chưa bị `chon_chu` chạm, nhị phân Otsu; L16/TK/B18/B34 nguồn crop 35–55 px bị phóng 2,3–3,3 lần)
  và **danh sách chữ ưu tiên** (S1, k = 5: chữ pipeline đã đọc/gán + 5 ứng viên đầu của MỖI ÂM theo tiên nghiệm không dùng nhãn người; lọc theo phông có glyph).
- **Quy mô**: 410 việc (lô 128 chữ), **51.730 ảnh** ≈ **38,8 giờ GPU = 19,4 giờ phiên** (T4 x2 chỉ trừ 1 giờ quota mỗi giờ chạy, ~30 giờ/tuần) theo 2,7 giây/ảnh/GPU — số đo ở lần Kaggle trước với wrapper lỗi `use_fst`, nên là **cận trên**.
  `BUDGET_HOURS = 7,5` ⇒ **3 ca**; 6,5 ⇒ 4 ca; 11 ⇒ 2 ca. Hợp ứng viên R(âm) đầy đủ (240 nghìn ảnh ≈ 90 giờ) KHÔNG làm hết một lượt nên sinh theo vòng ưu tiên.
- **Không tinh chỉnh FontDiffuser**: đo trước cho thấy hại, và B18/B34/L16/TK là `evaluation_only` (`dataset/<Bộ>/TAP_DANH_GIA.md`). Gói chỉ dùng mỗi cuốn **một crop làm ảnh phong cách** (điều kiện lúc sinh, không huấn luyện); danh sách chữ lấy từ nhãn **trước** `chon_chu` (không dùng nhãn người).
- **Kỳ vọng (trung thực)**: B34, ảnh sinh theo phong cách sách ngang phông (n = 111: Δ so với phông 0,0 [−8,1; +8,1]). Bốn thí nghiệm phía glyph trước (Ridge, đồng bộ quang học, kho FD đầy đủ, ablation) đổi Top-1 của bộ chọn chữ ≤ 0,12 điểm ⇒ **ngưỡng "có ích" +0,3 điểm (mục 4) nhiều khả năng không đạt**.
  Gói cho phép đo lại đủ 10 cuốn (4 bộ có nhãn người để chấm) và là bằng chứng đầy đủ cho luận văn dù kết quả thế nào.

## 1. (Chỉ khi chạy lại pipeline) làm mới dữ liệu — 0 API, CPU, ≈ 10–15 phút

```bash
bash lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p11a_lam_moi_du_lieu.sh [tiền tố gói mới]   # mặc định kaggle_pack_<ngày-giờ>
```
Màn hình chỉ có một dòng ✓/✗ mỗi chặng, số invariant PASS/FAIL, dòng tóm tắt gói và 2 tệp cần đưa lên Kaggle; log đầy đủ ở `measure_out/_tn11/full/lam_moi_logs/<giờ>/`.
Gói có mã băm kế hoạch/tài sản khác ⇒ **dùng kho HF mới** (bộ chạy từ chối trộn: thoát mã 2 nếu `plan_sha` hoặc `assets_sha` khác kho).

## 2. Kết quả không mất khi reset phiên — cơ chế HF

| Lớp bảo vệ | Hành vi |
| --- | --- |
| Đẩy HF định kỳ | mỗi **5 phút** (`--sync-every 300`) một commit gồm các lô mới xong; mất tối đa ≈ 5 phút việc nếu phiên bị giết đột ngột |
| Đẩy lần cuối | khi kết thúc / hết ngân sách / Cancel (SIGTERM) / dừng cứng: chờ tối đa 15 phút cho tới khi HF nhận hết |
| Kiểm trước khi chạy | `preflight` **ghi thử + đọc thử** repo HF, in tên tài khoản và số việc đã có; sai token/quyền/kế hoạch ⇒ ô đầu dừng (assert), chưa tốn giờ GPU |
| Chạy tiếp | mọi ca hỏi HF việc nào đã xong và bỏ qua (kiểm bằng notebook thật: ca hai làm lại 0 việc) |
| Chống trộn nhầm | `plan_sha.txt` + `assets_sha.txt` (ảnh phong cách, phông, ckpt) ghi lên repo; lệch ⇒ thoát mã 2 |
| Bản sao thứ hai | `/kaggle/working/tn11_bundles`: vài tar lớn + chỉ mục, giữ lại khi *Save Version* (dùng cho phương án B nếu HF hỏng) |
| Mỗi lô nguyên tử | tar sinh tất định, ghi `.part` rồi đổi tên; lô dở không bao giờ lên HF |

Repo mặc định `mdnt571/gannhanocr-tn11-full` (dataset **private**, tự tạo nếu chưa có; `mdnt571` suy từ repo HF sẵn có của bạn — **đổi `HF_REPO` ở ô đầu nếu tài khoản khác**; preflight sẽ báo tên tài khoản của token).

## 3. Việc bạn làm, theo thứ tự

1. **Dataset Kaggle (Private)**: kaggle.com → Datasets → New Dataset → kéo `kaggle_pack_20261004_v3.zip` vào (Kaggle tự giải nén; nếu chưa giải nén, notebook tự giải từ zip) → Create.
   Tài khoản phải **xác minh điện thoại** (chưa thì không bật được GPU/Internet).
2. **HuggingFace**: tạo token quyền **Write** (huggingface.co/settings/tokens). Không dán token vào notebook.
3. **Import notebook**: kaggle.com → Code → New Notebook → File → **Import Notebook** → chọn `kaggle_pack_20261004_v3.ipynb`. Settings: Accelerator = **GPU T4 x2** · Internet = **On** · Environment = **Pin to original**;
   File → Add-ons → **Secrets** → Add Secret tên đúng `HF_TOKEN` → bấm **Attach**; **Add Input** = Dataset ở bước 1.
4. **Ca THỬ** (`TEST = True`; ≈ 25 phút, 14 việc thật, kết quả giữ lại) — **ĐÃ CHẠY THẬT 04/10: đạt** (14/410 việc lên HF, `worker_exit_codes` [0, 0], `sync_ok` true, `use_fst=False`, 181 và 201 giây/việc). Notebook v3.1 mặc định `TEST = False`: **Save Version → Save & Run All** (KHÔNG chạy tương tác). Mở phiên đã lưu → xem đầu ra:
   `PREFLIGHT ĐẠT` (có dòng `kho hf:… ghi/đọc được · tài khoản …`) · `[env]` 2 × Tesla T4 · `[w0] [fd] sẵn sàng (cuda): … use_fst=False` ·
   `[run] {…}` có `worker_exit_codes` = [0, 0], `stop_reason` = `max_tasks`, `sync_ok` = true · `[status] 14/410`. Trên HF phải có `plan_sha.txt`, `assets_sha.txt` và 14 tệp `shards/…/NNNNN.tar`. `ema_s` ≈ giây/việc (kỳ vọng ≤ 340).
   Mã thoát/ô lỗi: preflight ✗ (token/quyền/kế hoạch), "KHÔNG CÓ GPU CUDA", "CỔNG use_fst", "TÀI SẢN KHÁC" → đọc dòng ✗ rồi sửa; **không** chạy tiếp ca thật khi ca thử lỗi.
5. **Ca THẬT**: mở Settings xem quota ("You have N hours remaining"); chỉ chạy khi N ≥ 2,5. Ở ô đầu giữ `TEST = False`, `BUDGET_HOURS = min(11, N − 1,5)` (thường 7,5) → Save & Run All **một lần** (bấm hai lần tạo hai ca song song; **không chạy hai ca cùng lúc**).
   Sau khi bấm, mở **Active Events** (góc dưới trái) và dừng phiên tương tác GPU của editor (vẫn tính quota tới khi nhàn rỗi). Màn hình ít dòng: sau `[fd] sẵn sàng` im lặng ≤ 10 phút rồi mỗi 10 phút một dòng
   `[tiến độ] n/410 việc … HF k lô …`; log đầy đủ ở `/kaggle/working/tn11_bundles/logs`. Lặp lại ca (cùng notebook, cùng `HF_REPO`) tới khi ô cuối báo `[status] ĐÃ XONG` (đo thật ≈ 38 việc/giờ ⇒ 410 việc ≈ 10,5 giờ phiên ≈ 2 ca ở 7,5 giờ; `BUDGET_HOURS = 11` gần xong trong 1 ca nhưng cần quota ≥ 12,5).
6. **Dừng sớm**: Active Events → **Cancel** (bộ chạy bắt SIGTERM, hoàn tất việc đang làm, đẩy nốt lên HF). Quá hạn nhận việc + 30 phút còn worker treo thì bị dừng cứng.
7. **Tải về Mac & kiểm** (đăng nhập HF một lần; lệnh `hf` nằm trong `.venv`):
   ```bash
   .venv/bin/hf auth login            # dán token (Read hoặc Write)
   .venv/bin/hf download mdnt571/gannhanocr-tn11-full --repo-type dataset --local-dir measure_out/_tn11/full/kaggle_ket_qua
   .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_runner_khung_chay.py validate \
       --pack measure_out/_tn11/full/kaggle_pack_20261004_v3 --out measure_out/_tn11/full/kaggle_ket_qua
   ```
   `n_errors` = 0; `missing` = số việc chưa làm; `images` = 51.730 trừ chữ bỏ qua; `uncovered_no_font` = chữ không phông nào vẽ được (hiện 1 chữ ở stt11); xem `skipped_by_reason`.
8. **Phương án B (không HF)**: bỏ `--remote` trong ô chạy; mỗi ca là **một notebook mới** có Input = Dataset **và đầu ra TẤT CẢ các ca trước**, `--prev` liệt kê ĐỦ chúng (vd `--prev /kaggle/input/nb-ca1 /kaggle/input/nb-ca2`);
   sau mỗi ca tải `tn11_bundles` về Mac rồi `w_runner_khung_chay.py unbundle --bundles <các thư mục> --dest measure_out/_tn11/full/kaggle_ket_qua`. Preflight sẽ báo ✗ vì thiếu `--remote` (cố ý) — bỏ ô preflight.

## 4. Chấm "có giúp ích không" (sau khi có ảnh)

```bash
.venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p12_cham_toan_bo.py --ket-qua measure_out/_tn11/full/kaggle_ket_qua \
    --pack measure_out/_tn11/full/kaggle_pack_20261004_v3 --stage all
```
Chấm trên 4 cuốn có nhãn người (nhãn người chỉ để chấm): (a) Top-1 một tín hiệu của ảnh sinh theo sách so với glyph phông **trên cùng tập ứng viên có ảnh** (không tính độ phủ — bài học của FD), CI cụm trang, tách nhóm chữ hiếm và nhóm phông NomNaTong / dự phòng,
loại các ô mang chữ dùng làm ảnh phong cách; (b) thêm vào bộ chọn chữ TN8 (LOBO), hai biến thể: có NaN (mang cả "có ảnh" = tiên nghiệm tần suất) và đã bù (chỉ tín hiệu ảnh).
Tiêu chí đăng ký trước: "có ích" nếu Δ Top-1 bộ chọn ≥ +0,30 điểm với CI cận dưới > 0 ở ≥ 2/4 bộ; ngược lại ghi kết quả âm.

## 5. Gắn vào pipeline — chỉ khi mục 4 đạt

Điểm gắn: thêm cột đặc trưng `f_gen` (cos crop ↔ ảnh sinh theo sách) vào `pipeline/chon_chu/features.py` và thư mục ảnh theo sách vào lớp `Glyphs` (`pipeline/gold_exact/signals_img.py:285`; bản chép `pipeline/borg_human/encoders.py:81`);
kéo theo học lại logit `chon_chu` (vài phút), hiệu chuẩn lại ngưỡng, rồi chạy lại 10 bộ từ cache (0 API, khoảng 1h40). Không có bước huấn luyện mạng nào.

## 6. Điều cần biết

- Mỗi chữ có **một ảnh cho mỗi cuốn** (51,7 nghìn ảnh cho 10,2 nghìn chữ khác nhau). Chữ ngoài phông NomNaTong (12,4 % chữ-cuốn) dùng chuỗi phông dự phòng (HAN NOM A 10 %, HanaMinB 2 %, …; chữ PUA chỉ vẽ bằng họ NomNaTong); ảnh nội dung của chúng khác kiểu NomNaTong
  (IoU ≈ 0,31 so với 0,84 cùng họ). **Phông dùng cho từng chữ ghi ở `plan.json`** (`books.<cuốn>.font_idx` theo `font_names`); `_meta.json` của tar KHÔNG ghi phông.
- Ảnh sinh chỉ phủ khoảng 41–45 % R(âm) mỗi ô ⇒ khi so ảnh phải so **cùng tập ứng viên có ảnh**.
- Lỗi `use_fst` của wrapper `core/ranking/fontdiffusion_gen.py:101` (gọi `use_fst=True` dù dòng 76 đặt `False` ⇒ FST khởi tạo ngẫu nhiên) **được vá ngoài `core/`** bằng `fd_wrapper_fix.py` (nằm trong gói); bộ chạy dừng mã 2 nếu mô hình nạp là bản FST hoặc thiếu checkpoint.
  Chưa sửa `core/` (bạn quyết định).
- Bộ kiểm (đều PASS lúc đóng gói): `w_runner_kiem_khung.py` 34/34 · `p11b_kiem_ban_va.py` 22/22 · `w_runner_kiem_that_cpu.py` 6/6 · `w_runner_phat_hien_fst.py` 5/5 · `p13b_kiem_notebook.py` 11/11 (cần venv riêng có nbclient+ipykernel; xem đầu tệp) ·
  một việc thật 128 ảnh qua dòng lệnh trên MPS (2,98 giây/ảnh trên Mac, không phải số của T4) · Python 3.10 biên dịch + chạy giả có đẩy kho · `w_quy_mo_va_ngan_sach_05_kiem.py` 39/39 invariant.
- Chưa chạy trên GPU Kaggle thật và HF thật; thư viện `huggingface_hub` của Kaggle có thể khác bản 1.17.0 dùng để kiểm chữ ký (các hàm dùng: `create_repo`, `whoami`, `list_repo_files`, `upload_folder`, `upload_file`, `hf_hub_download`).
- **Notebook v3.1 (04/10, sau ca thử thật)**: ca thử cho thấy `huggingface_hub` 1.29 (bản Kaggle) vẽ bảng tiến độ 3 dòng khi stderr là TTY (Kaggle cấp pty cho `!`), lặp ~9 dòng mỗi lần đẩy. Ô đầu nay đặt `HF_HUB_DISABLE_PROGRESS_BARS=1`
  (đọc lúc import ⇒ phải đặt trước mọi tiến trình con; đã chứng minh trong pty: 10 dòng → 0), `PYGAME_HIDE_SUPPORT_PROMPT=1` và `PYTHONWARNINGS=ignore:pkg_resources is deprecated:UserWarning` (chỉ che đúng cảnh báo đó). Không đổi bộ chạy ⇒ Dataset không phải tải lại;
  bộ chạy trong Dataset vẫn là bản đã chạy thật (sha256 c3cab025…). `p13b_kiem_notebook.py` nay 12/12. Ước lượng `[status]`/`[tiến độ]` dùng 2,7 giây/ảnh nên là **cận trên** (thật ≈ 1,5 giây/ảnh/GPU: 181 và 201 giây cho 128 ảnh).
- Ghi chú ngoài gói: `dataset/_ALL/TAP_DANH_GIA.md` gọi cả 4 bộ có nhãn người là "bản in, nhãn IHR-NomDB", còn `dataset/SachDungLyHoThan/TAP_DANH_GIA.md` nói B34 là bản chép tay với nhãn người từ xlsx; B18/B34 là chép tay Borg, L16/TK mới là IHR-NomDB.
