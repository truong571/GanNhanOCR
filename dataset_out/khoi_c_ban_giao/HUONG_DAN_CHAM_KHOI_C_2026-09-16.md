# Hướng dẫn chấm Khối C (mẻ mù hai câu, 16/09/2026)

Tài liệu này dành cho **người chấm**. Đọc hết một lần trước khi mở tệp HTML đầu tiên.
Mọi thứ về cách rút mẫu, tầng, trọng số nằm ở `docs/BAO_CAO_KHOI_C_C1_2026-09-16.md` — **không cần**
và **không nên** đọc trong lúc chấm.

## 0. Ba điều cấm (làm hỏng số liệu ngay lập tức)

1. **Không mở** `dataset_out/human_audit/khoi_c_2026-09-16/_khoa/` (khoá ô → nguồn), **không mở**
   `dataset_out/labels_final.csv`, `re-dataset/`, `BANG_SO_LIEU`, hay bất kỳ tệp nào cho biết ô đang
   chấm thuộc tier / rule / p / box_source nào. Màn hình chấm cố ý chỉ hiện ảnh + ÂM + MÃ; nếu bạn biết
   thêm thứ gì khác về ô thì verdict không còn mù.
2. **Không so ô này với ô khác, không đi tìm ô lặp.** Mẻ có ô lặp ẩn và ô mồi có đáp án; chúng đo độ ổn định
   của chính bạn. Gặp ô "hình như đã thấy" thì cứ chấm như ô mới.
3. **Không bỏ ô khó.** Bỏ chọn lọc làm khoảng tin cậy mất hiệu lực. Không đủ căn cứ thì bấm **KHÔNG RÕ** —
   đó là một câu trả lời hợp lệ và được đếm riêng.

## 1. Thứ tự làm

| Bước | Tệp | Số ô | Thời gian | Sau khi xong |
|---|---|---:|---|---|
| **Pilot** (C-1) | `pilot_50.html` | 50 | ≈ 5 phút | Xuất `verdicts_KC20260916-PILOT.jsonl` → chạy báo cáo pilot (§6) → **chỉ khi pilot đạt** mới sang phiên 1 |
| Phiên 1 | `phien_1.html` | 256 | ≈ 25–35 phút | Xuất `verdicts_KC20260916-S1.jsonl` |
| Phiên 2 | `phien_2.html` | 256 | ≈ 25–35 phút | Xuất `verdicts_KC20260916-S2.jsonl` |
| Phiên 3 | `phien_3.html` | 256 | ≈ 25–35 phút | Xuất `verdicts_KC20260916-S3.jsonl` |
| Phiên 4 | `phien_4.html` | 256 | ≈ 25–35 phút | Xuất `verdicts_KC20260916-S4.jsonl` |
| Phiên 5 | `phien_5.html` | 255 | ≈ 25–35 phút | Xuất `verdicts_KC20260916-S5.jsonl` |

Tất cả nằm trong `dataset_out/human_audit/khoi_c_2026-09-16/` (1.279 lượt / 5 phiên, cả mẻ ≈ 2,5 giờ).
**Một buổi ≤ 45 phút, không chấm hai phiên liền** (bài học 04/08: độ lệch trôi 4 % → 16 % → 35 % qua ba buổi liền). Tiến độ tự lưu trong trình duyệt
(localStorage) — đóng tab rồi mở lại vẫn ở đúng ô, nhưng phải **cùng trình duyệt, cùng máy**.

Ô pilot **không trùng** ô nào của 4 phiên chính; pilot chỉ để đo dwell / κ / mồi, không lấy số precision.

## 2. Màn hình một ô — HAI PHA

Mỗi màn là **một ô**. **Pha 1** (trước khi bạn trả lời Q1) mọi ô trông y hệt nhau:

- **crop** phóng ≈ 3× — ảnh cắt ra từ trang scan theo hộp của ô;
- **vị trí trên trang** — cột ±2 ô quanh nó, khung đỏ là hộp của ô (dùng để phân xử khi crop khó nhìn);
- **ÂM hiển thị** — âm Quốc ngữ mà bộ nhãn gán cho ô này;
- ô bên phải ghi *"MÃ (nếu có) hiện sau khi trả lời Q1"* — bạn **không** biết ô có mã hay không, và đó là cố ý.

**Pha 2** (ngay sau khi bấm Q1): nếu ô có mã Unicode thì **MÃ + glyph tham chiếu** (font) và câu **Q2** hiện ra;
nếu không có mã thì màn báo *"ô này không có mã (Q2 không áp dụng)"* và tự sang ô sau.
Sau khi thấy mã bạn **vẫn được đổi Q1** nếu thấy cần — câu trả lời mù được giữ riêng và lần đổi được ghi lại;
số liệu chính dùng câu trả lời **trước** khi thấy mã, nên hãy trả lời Q1 cho chắc ngay ở pha 1.

Phím: **1·2·3·4** cho Q1, **5·6·7** cho Q2 (chỉ ăn ở pha 2), **←/→** chuyển ô, nút **Ô chưa chấm** nhảy tới ô còn trống.
Trả lời đủ câu thì tự sang ô sau. Đổi ý được; mọi lần đổi đều được ghi lại (đó là dữ liệu, không phải lỗi).

## 3. Q1 — *Ô này cắt đúng MỘT chữ, và chữ đó đọc là ÂM hiển thị?*

Q1 hỏi **hai chuyện cùng lúc** — crop và âm. Chỉ khi cả hai đều ổn mới là ĐÚNG. Q1 được trả lời **khi chưa thấy
mã**: bạn đọc chữ Nôm trên crop bằng mắt mình rồi so với ÂM — không có glyph nào gợi ý.

| Phím | Lựa chọn | Định nghĩa (có ngưỡng) |
|:---:|---|---|
| **1** | **ĐÚNG** | crop chứa **một chữ trọn vẹn** (mất < 1/3, mực kề nhỏ vẫn tính là trọn), **và** chữ đó đọc đúng là ÂM hiển thị |
| **2** | **SAI CROP** | **cắt cụt > 1/3 chữ**, **hoặc** dính **≥ 2 chữ** (thấy hai chữ đủ hình), **hoặc** không phải chữ (trắng, vệt mực, hoa văn) |
| **3** | **SAI ÂM** | crop là **một chữ hoàn chỉnh**, nhưng chữ đó **không** đọc là ÂM hiển thị → ghi âm đúng vào ô "âm đúng là…" nếu biết |
| **4** | **KHÔNG RÕ** | đã nhìn cả khung đỏ trên trang mà vẫn không đủ căn cứ (mờ, nhoè, dị thể không nhận ra) |

Thứ tự xét: **crop trước, âm sau.** Crop hỏng → SAI CROP, không cần xét âm. Crop ổn → mới hỏi âm.

### Sáu ví dụ (crop lấy ngoài mẻ chấm, phóng 3×; ảnh nhúng base64 vì `*.png` bị gitignore — bản PNG ở `docs/img/khoi_c_2026-09-16/`)

| # | Crop | ÂM hiển thị | Trả lời | Vì sao |
|:-:|---|---|---|---|
| 1 | <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQkAAAFJCAAAAACtW4fUAAAEbElEQVR42u3d2VIbQQxA0VEq///LygNFKDCexdOLZnzmCUiBnesrtXqPXI4/8fnFK79c9fmzeD6ev6d+O/ZoEdcQiBNItI2O3an145vkxLs7EY8/SE7ImHeOjuAEJxqly9+qzChUeXICiUHREZuBcb5TFm27dZzo6US0/ewHNeacQKJ/dMSA0Ph6kXZpkxMdnIj2DWXL4VROINE7Oor0yfPnGzk/HMaJl5yI0+myXTf8hxUy5hQn4tAntnRvU/P/W1JZaUUL1pg5qajs8VqcaN3vyOajafIEEjfqlScnRAcSSMiYN02RnEACCSSQQAIJJJBAAgkkkCjeF417d0Q5gQQSSPQbvUtOiA4kkEACCQ8Sp+qJ5AQnbi8EJ5BAAgkkkEDixBOJASeQQAIJJJBAAgkkkEACCSSWG5233e3gQk4gcZuRmlifQCoROpyY7sTyfKpxzoUfnEBi1th2lF+mwYkJ8x1Re5kGJ5CYOhsYMqbo+PkhJxLyxMmckZwQHcv9xzFXTsHlBCcek2KorESHeuJAgaFXzgl5AgkkkHibGnPnZS1RdxEFJ5DoekdcV/Hj+dILNx5VrTEbrYyJJ12P3y69CrdgyZg3HbOKDbVX/r1DduZEhxrzyIcVTyrPmDecyYmBNzL3aYKbZwpOING7xlwGXxrovvL6Y9txzYskOYHEABKZSMiYuxNnvphTc8xIISeQqDBXHm0LhVBjXn/0LlfdSL1yGdPYdqv+V3LiPtFRq4/CCSSKZ8zkhOhAQp6ouwWKE0gggcSlKqvkhOhAAgkkkEACCSSQQAIJJAY9EUhwAgkkkEBiMY7Z9xgLTiCBBBL3WLd95EyIrVW6MfBWD04M2xuYrfcPckLGvEPGXGZOqB/eZs2JCXsDBzrxyulOnEBiUK88ZxxIwYn6IzXzxHhl7S8nkLjSLVinqszDJQUnpoxt57PRl8NXe8gTSLzHXQ2xkRnje42Qx/9qcuLqO+xz6buXnhNIIIHEZUkUOYaCE7cncdg0TiBRssYsceQAJy5/slfztY2cQOKNouNbHt4+RJITb5UxH1Y0ckJ0nIiOWEaemBIVagtO7M6Y0d2KWMalxa//0sPcCSeQ2BUdsSPHdNyxtfKn1181zgQkJ47VmL80qjklX6Y8IWPW7YHF57Kg7BoY0SQxcmJQrzxOfzBZsofCiedORI0pifPOBSdEx5iMGfc4ZGKlsebEMyeizfLBKHZT7472nBNIrK7bjpN7urKK/7nnTSQnBs8G5qu3qkbPrTXyhIxZvMbcMjp67YrgxFEnovUezrjU1R+cQGIlOnJ6vyg39wlyYtZ+0Tja9c6+LW/KEzLmJXdTn5gXjFI3g3Gixg77cPuuddvuDdSKio6rhQYnkEACiSuc4hQ18iUnCrWiWWU1BSeQKHjuHSeQQAIJJJBQY156BTcnkFBjcgIJJJBAAgkkkLhnvyOqTA5zAolK0RGFFk5wouJsICdkzMVIzZm7ozlh9I4TSCCBBBJIIIEEEld9/gH2idHF5ampuQAAAABJRU5ErkJggg==" alt="vd1_dung" height="110"> | chúa | **1 · ĐÚNG** | một chữ 主 trọn vẹn, đọc "chúa" |
| 2 | <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAANsAAAEOCAAAAAAlbD20AAADJElEQVR42u3d0W7kIAxAUbza//9l92lXrTrDJMEkUTh5a9VWc8WtHQyYyPbY50/Dhg0bNmzYsGHDNvv5e5+PEq2lccPWIm8j5P8njZs4ebf/kyItOSlODmoWP7/Mf791OIRykpOHU3b+ErLzpHGTu+e/Q4Zxw3bv98nmfZKTpVm4Ew/zvXhh3LDNjJOx39ucoCUnOVkV7tQnOTlJxfwUG40btklO9uuHeV0VnZNt1fpk7NcvTsnvnMSGDdsdcndUrFOnccNWkbvjBtZxUpy8z95IToqTxyJkvtrtY9yw1Tp5YOfYVVpycoXcHQNTmNiZ1neVmF7+MCe9T47oHUV7LL8bzsmn5u4omk1HO3urGyef5GTsKez03zBjwOctfznfJ3FOPiZ3R3cSnRXBsGSak5++yclnOBndge5vfcxpaTqN26q5O7bJEAOqjM+mjZt596TVmY9FyzBunDylpn3gjEMat4WdPGeFJTck5aj4SJxccG1x0pw6unUnPQc4Wbi8+LHAWJvWOfmYWtCMDB5XrO9w8hlObhcyt83Hx+Pqy1QenJS7S2cfJzSf5CQ2bNie3cOqpGjPSXt6R9pbtQnr4JwUJ1++Ge6aceuxj63WyWj32t/LyaWczKuF3DL95+RSTsb73RT9rbZR3XayM83h5Gq5e+MJ2Zxc25S7OTm+7yIrQuX3CdS7UMnJFZwc6WsRMzcJx6sWrJxsi/VmydLVw8NpvXOijZPubBqZI0f30NmxztWcVJ8cPF1bfoUTJxdx8sDNiYOHHF+e1c0N9nJyESdzrI3VyPHb2LZkyUm5u8rh7bLltpYvnDTHqaqNx9FDOilOcrKuWDRS85G7OWncsGHDhg0bNmzYsGHDhg0bNmzYsGHDhg0bNmzYsGHDhg0bNmzYsGHDhg0bNmzYsGFrD+rTG9NaUHKy2Wd+My05iQ0btuf12Nc7GluVk3FWS0njxsmrzs+6l5aT51xYc2C+E+7v5uRZN8WfcJc9J8XJGW0A5e6mPnlK4o75QZKTnJzR8Me4yd3t3FLk7/5p6pPYroqT7sfBpq8aNmzYsGHDhg0btjnPF+qJsj9vSxRCAAAAAElFTkSuQmCC" alt="vd2_dung_muc_ke" height="110"> | nhiều | **1 · ĐÚNG** | chữ 饒 trọn; vệt mực nhỏ phía dưới là của ô kề, **không** làm mất chữ → vẫn ĐÚNG |
| 3 | <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAP8AAACHCAAAAAD8EtC6AAACCklEQVR42u2dwXLDMAhEtZ3+/y9vTplxmtiVbYFw/HTqJW52eCAMSJHbrddPQz/60Y9+9KMf/ehHP/rvs35z/o2efxj7o7/QkhPhXy5jf/TfJv5/dAoX2FngH/6jl1e2gAoOBf/kP5kx/5j77HqysT/6S/J/zAW8/8nG/uifkf9oHM85WRb8E/8z+XexQiv8E//HBuENX3C9OhL8w39LKQElP7nTL+Af/q8L//m3APiH/1sNPGB/9E/p/6rDHfQM4KNaw8b+6E+v/78Hf3Wk6wMTp/7dB/6J/xWavEFvAf9uAfAP/2nvvMb+6Gf+c901hP3R/+3zD23GxJGxP/oLx//oTGntRQD+4T80+KulNgXE/DP8z6h/KriSs+wL7IJ8ey4a/uH/opm/D30xY3/0z8j/vWjvDpneH9Imhn/if/S0jw4dY3z/iAKOhsE/8b9m0cYp7TP4h/+CmX9agQj+0Y9+9BP/v3jO3+tbFfzDf+jUgWq3leEf/gdm/mozm7/YH/1p9U/FBGTl+gj8w3/oDT/quBelbfbFsD/6Z/GvTXTd0bdSXydrVmkU/uF/1B2G6ub/z2n3ib4A//AfcYGnXo90+VBm1XMozNgf/SP4V8xt5G3crHLEzYrwD/89wf/ka7JjGmo67QLwD/9rObmv0Bfm94/gv+bvX6j85YfYH/6N/dGPfvSj/47rAVECeRD79891AAAAAElFTkSuQmCC" alt="vd3_sai_crop_cut" height="110"> | mà | **2 · SAI CROP** | chữ 麻 mất **nửa trên** (> 1/3) → cụt, dù phần còn lại vẫn đoán ra được |
| 4 | <img src="data:image/png;base64,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" alt="vd4_sai_crop_dinh" height="110"> | nương | **2 · SAI CROP** | **hai chữ** đủ hình trong một crop (娘 ở dưới + chữ khác ở trên) → dính |
| 5 | <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMMAAAAeCAAAAAC0tPWHAAAAW0lEQVR42u2XOQoAMAgE15D/f9n0SSmSFWY7y8HBI1LjswQDDDDAAAMMMMDQl33VISV9gKHsUlGkXx7ikkeCPw6Gvh2naWsOl0bMJX+1Apdc59IrkvmxlLjkkgO7RQpAOtyODQAAAABJRU5ErkJggg==" alt="vd5_sai_crop_khong_chu" height="110"> | suy | **2 · SAI CROP** | gần như trắng, chỉ vài chấm mực → không phải chữ |
| 6 | <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAFKCAAAAAAcJG1nAAAEM0lEQVR42u3dW5LcMAhGYZPK/rdMnlK5dM+4baOb/WkBXXPmR4ARQpHbc9ePDTx48ODBgwcPHnz7FUH5Pv/oiXL7P5on5R8K32njUx48+GetnxMGOsqDf4zZ/71keODBgwcP/r7FjKT885S358GDBw8ePHjw4MGDBw8ePHjw4MGDBw8ePHjw4MGDBw8ePHjw4MFvj208jnP9wTGowZjy4Cta0eKM3cYkXfaUn+6mRfQWn8Nj9ms0SlN+dNd1nIlSsc0V7ez5reddwbDnwT89w0vKr3fBKM5JGIoZHB548MvBZ1IevAyvILmLrhkA5SeZARVb50ou5ef8nIvmW1+oY/ZTDrBr6/cov000CC5exU/Kg1/K7OPweUV2PNVQva2u38YHziwn6FmjfPM0Jw8PsO+ivVAn1M2V2+f/v98i0aN8n6JF2PPgFzf76N91UV/YoHzn2mza8/dUPst+JygPfgmzT8o/4UZl+++CoDz4bYIaXlw8cT/sFxtW8zi8Prrb87dLclwqBH9bs4+rFq96u5rycT9PR3lmf9ZBnTb63Cb5rqX8+NMYex78QLNv5ls+rmNEx51F+Tmz73jXmkP5BsrHkZaaeO0szOIRcu1tUahj9ntJ3EuOHU2GhOm3H6t8FrZMxbXO03y9ZpuUB19l9lmbmVVtpnj1vJQfVrfPvVwsrvnMbwMgh9dT+TidnMTElS+hjtlPubKpy9ONdcLd5WeXa49XMb6NqMUDlDg8Dm9grf7452/NZSPKlx0y96xBFcQ7ys850Tn3LOvyxhfqmH2zlpqWW+Sa15PbF04zj/atO1HWHsThcXj1B+SNinZZNjOP8ivp/sUfeSrbo/xqq2pYplAHfjnbT8qD/+3ygvLgG18wGn5pMr9ohPw4waT8uuvrg4JPbIDDY/aLTonKXVf7dVMk5e8/HuP9XQDKd3iX6nyfz+mEZ7+jQahj9rN6pdPOOHYPNilf8YFW9cpyXZd27pJweJvz+Wtx+Li5xvCkksPrnd19npTZ81MrHzNOP6I8+Bqzf5N7R/sqFOUHJjn/ypOtJjFH77IZh8fs90N39EnB+109FurmenEt7fmFvuqStwd/G7PPeT4/KT/2KZdY2ds5qAR//RXyWMvs3/y5Qt2TsjpH1Nf3PG8PHjx48ODBgwcPHjx48ODBgwcPHjx48ODBgwcPHjx48ODBgwcPHjz4bdy0lJish43y2+DuZ8rfuwkxpmteFerAgwc/5URyyoMHD/7+U84bzlCh/Db4itfRCXNVdQEOj9lPYfSH0r64avuKGeOLGWNeBeTwmP2odO2YXUel7VN+XvGz6cQWDk+GN/6trX7P1lN+TG6fxRHLnl/qq67+4Q7Kg19gSNCpBo3S8doc3oT1W3sePHjwj3+OufTZI2a/mPhJefCPHAAaJVki5RcVX2cGeBOPKX9o/QIlv9reQRRpXAAAAABJRU5ErkJggg==" alt="vd6_sai_am" height="110"> | *chúa* | **3 · SAI ÂM** | crop là một chữ 於 hoàn chỉnh (đọc "ở"), nhưng ÂM hiển thị lại là "chúa" → ghi "ở" vào ô âm đúng |

Ví dụ 6 là ví dụ **dựng**: crop 於 được ghép với âm sai để minh hoạ. Trong mẻ thật, ô kiểu này tồn tại và
bạn không phân biệt được với ô thường — cứ chấm theo cái nhìn thấy.

Ranh giới hay gặp:

- mất **dưới 1/3** chữ, hoặc dính **một mẩu nét** của chữ kề mà vẫn thấy trọn chữ → **không** phải SAI CROP;
- **dị thể** (viết khác nét nhưng cùng chữ, cùng âm) → **ĐÚNG**;
- ở pha 1 bạn chưa biết ô có mã hay không: xét crop và âm y như nhau cho mọi ô;
- crop mờ nhưng khung đỏ trên trang cho thấy rõ chữ → dùng khung đỏ để quyết; vẫn không được → **KHÔNG RÕ**.

## 4. Q2 — *MÃ Unicode hiển thị có đúng chữ trên crop?* (chỉ khi ô có mã)

| Phím | Lựa chọn | Định nghĩa |
|:---:|---|---|
| **5** | **MÃ ĐÚNG** | glyph tham chiếu đúng là chữ trên crop; **dị thể chấp nhận** (体/體, 𠊚/𠊛… nếu cùng chữ, cùng âm) |
| **6** | **MÃ SAI** | chữ trên crop là một chữ **khác** → ghi mã/chữ đúng vào ô "mã đúng là…" nếu biết |
| **7** | **KHÔNG RÕ** | không đủ căn cứ |

Q2 độc lập với Q1: crop **SAI CROP** ở Q1 thì Q2 vẫn trả lời được nếu vẫn nhìn ra chữ (ví dụ 3: nửa dưới 麻
vẫn nhận ra → Q2 có thể là MÃ ĐÚNG); crop trắng thì Q2 = KHÔNG RÕ. Gặp mã **𠊚** với âm "người": đó là ô đã qua
phán quyết QĐ-01 — Q2 của các ô này chỉ được đếm là "nhất quán với phán quyết", không dùng làm số đo; Q1 (crop +
âm, trả lời khi chưa thấy mã) mới là phép đo. Cứ chấm như mọi ô khác.

## 5. Xuất JSONL và đặt tệp ở đâu

1. Chấm **hết** phiên (thanh tiến độ đầy, nút *Ô chưa chấm* không còn nhảy).
2. Bấm **Xuất JSONL** (góc phải trên). Trình duyệt tải về tệp có sẵn tên, ví dụ `verdicts_KC20260916-S1.jsonl`.
3. **Chuyển tệp đó vào chính thư mục mẻ**: `dataset_out/human_audit/khoi_c_2026-09-16/` — giữ nguyên tên.
   Nếu trình duyệt không tự tải, sao chép toàn bộ nội dung ô văn bản hiện ra dưới màn hình và lưu thành tệp
   cùng tên (UTF-8) vào thư mục đó.
4. Sau khi xuất, phiên **khoá**. Muốn sửa thì bấm *Mở khoá* — mọi thay đổi sau đó đều được ghi vào lịch sử và
   phải **xuất lại** (tệp mới ghi đè tệp cũ; bộ ước lượng giữ bản xuất mới nhất theo `exported_at`).

Mỗi dòng JSONL là một ô: `item_id` (mã ngẫu nhiên, không suy ra nguồn), `q1` (cuối), **`q1_blind`** (Q1 lúc chưa
thấy mã — số liệu chính), `q1_am`, `q2` (null nếu ô không mã), `q2_code`, `dwell_q1_ms` (thời gian tới lúc trả lời
Q1), `dwell_ms` (tới lúc trả lời đủ), `n_q1_change_after_reveal`, `visits`, `hist` (mọi lần đổi ý, có cờ
`after_reveal`), `session_id`, `exported_at`, `source: "human"`. **Không sửa tay tệp này.**

## 6. Chạy ước lượng (người quản lý mẻ chạy; người chấm không cần)

Từ gốc repo, sau khi tệp verdict đã nằm trong thư mục mẻ:

```bash
# sau pilot
.venv/bin/python -m pipeline.ground_truth.estimate_khoi_c --pilot
#   -> dataset_out/human_audit/khoi_c_2026-09-16/pilot_report.md (+ .json)

# sau đủ 5 phiên (chạy được với ít phiên hơn — sẽ báo số ô chưa chấm)
.venv/bin/python -m pipeline.ground_truth.estimate_khoi_c
#   -> docs/KET_QUA_KHOI_C_<ngày>.md (+ .json)
```

Pilot **đạt** khi cả bốn điều sau đúng (in ở đầu `pilot_report.md`, mục BÁO ĐỘNG trống):

| Chỉ số | Ngưỡng | Nếu không đạt |
|---|---|---|
| ô mồi (5 ô: 2 dương xét Q1, 3 âm xét Q1 ∧ Q2) | ≥ 90 % (tức 5/5) | đọc lại §3–§4; chấm lại pilot |
| κ Q1 trên 5 ô lặp | ≥ 0,4 (mục tiêu ≥ 0,8; n = 5 nên chỉ là tín hiệu sớm) | viết lại rubric rồi chấm lại pilot |
| dwell p50 | 3–12 s / ô | < 3 s: quá nhanh; > 12 s: xem lại cách dùng khung đỏ |
| ô < 1,5 s | < 10 % | chấm lại pilot chậm hơn |

Bộ ước lượng tự kiểm: sha256 khoá và sha256 `labels_final.csv` phải trùng lúc dựng mẻ; mọi `item_id` phải
có trong khoá; verdict `source ≠ "human"` bị loại. Kết quả mẻ chính **không** tự ghi vào `BANG_SO_LIEU` —
việc đó làm sau khi người duyệt (`evidence()`, A-13).

## 7. Nếu có sự cố

- Mở lại tệp HTML mà mất tiến độ → bạn đang dùng trình duyệt khác / chế độ riêng tư; quay lại trình duyệt cũ.
- Phím số không ăn → đang đứng trong ô nhập "âm đúng là…"; bấm ra ngoài rồi bấm phím.
- Ảnh trang không hiện → tệp HTML bị cắt khi sao chép (mỗi phiên 6–7 MB); chép lại nguyên tệp.
- Lỡ xoá localStorage → chấm lại phiên đó từ đầu (không ghép hai lần chấm dở vào một tệp).
