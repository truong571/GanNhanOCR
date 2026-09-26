#!/usr/bin/env python
"""t03_report.py — TN3: sinh KET_QUA.md + tong_hop.html CHỈ từ tệp đã lưu (matrix.csv, t02_summary.json, t01_summary.json).
Không tính lại gì ngoài định dạng và vài phép cộng/chia hiển thị. 0 API.
Chạy: .venv/bin/python lab/thu_nghiem_anh_chu/TN3_tat_ca_bo/t03_report.py
"""
import json, math
from pathlib import Path
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
MX = pd.read_csv(HERE / 'matrix.csv')
S = json.load(open(HERE / 't02_summary.json'))
S1 = json.load(open(HERE / 't01_summary.json'))
SETS8 = ['stt2', 'stt4', 'stt11', 'Chr', 'L83', 'KVK', 'L16', 'TK']
DIRS = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'Rv3']
FULL = {'stt2': 'SachThanhTruyen stt2', 'stt4': 'SachThanhTruyen stt4', 'stt11': 'SachThanhTruyen stt11',
        'Chr': 'Chrestomathie1872', 'L83': 'LucVanTien1883', 'KVK': 'KimVanKieu1884', 'L16': 'LucVanTien1916',
        'TK': 'TruyenKieu1872'}
MUC = {'ĐO': 'CHẮC CHẮN THEO MÁY', 'ƯỚC LƯỢNG': 'ƯỚC LƯỢNG', 'SUY ĐOÁN': 'SUY ĐOÁN'}
INV = S['invariants']
assert INV['all_pass'] and S1['invariants']['all_pass'], 'invariant FAIL — không sinh báo cáo'


def I(x):
    return f'{int(round(x)):,}'.replace(',', '.')


def P(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '—'
    return f'{100 * x:.{d}f}'.replace('.', ',')


def F(x, d=1):
    return f'{x:,.{d}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def g(s, d, c='both_pt'):
    v = MX[(MX.set == s) & (MX.dir == d)][c].iloc[0]
    return None if (isinstance(v, float) and math.isnan(v)) else v


def acc_txt(s, d, bold=True):
    pt, lo, hi = g(s, d, 'both_pt'), g(s, d, 'both_lo'), g(s, d, 'both_hi')
    if pt is None:
        return '—'
    core = f'{P(pt)}'
    return (f'**{core}**' if bold else core) + f' [{P(lo)}–{P(hi)}]'


MEETS = {'dat_chac': 'ĐẠT (chắc: cận dưới CI ≥ 99,5 %)', 'dat_diem': 'đạt ở điểm', 'dat': 'ĐẠT (cận bi quan ≥ 99 %)', 'khong': 'không'}
DN = {'H0': 'H0 giữ nguyên', 'H1': 'H1 hình học', 'H2': 'H2 = H1 + kiểm ảnh', 'H3': 'H3 = H1 + dị bản người',
      'H4': 'H4 = H1+H2+H3', 'H5': 'H5 = H1 + sửa hộp', 'H6': 'H6 = H4 + sửa hộp', 'Rv3': 'Rv3 (tham chiếu)'}
R = S['rank']
_k = lambda s_, d_: int(MX[(MX.set == s_) & (MX.dir == d_)].keep.iloc[0])
ADD5 = sum(_k(s_, 'H5') - _k(s_, 'H1') for s_ in SETS8); ADD6 = sum(_k(s_, 'H6') - _k(s_, 'H4') for s_ in SETS8)
ADD5_TXT = ', '.join(f'{s_} +{_k(s_, "H5") - _k(s_, "H1")}' for s_ in SETS8 if _k(s_, 'H5') != _k(s_, 'H1')) or 'không bộ nào'
ADD6_TXT = ', '.join(f'{s_} +{_k(s_, "H6") - _k(s_, "H4")}' for s_ in SETS8 if _k(s_, 'H6') != _k(s_, 'H4')) or 'không bộ nào'
# thứ hạng của H4 theo điểm và theo cận dưới/bi quan (trong H0..H6, giữ ≥ 50 ô)
RK4 = {}
for s_ in SETS8:
    sub_ = MX[(MX.set == s_) & (MX.dir.isin(DIRS[:-1])) & (MX.keep >= 50)]
    h4 = sub_[sub_.dir == 'H4'].iloc[0]
    RK4[s_] = dict(pt_top=bool(h4.both_pt >= sub_.both_pt.max() - 1e-12), pess_top=bool(h4.acc_pess >= sub_.acc_pess.max() - 1e-12),
                   best_pt_dir=sub_.sort_values(['both_pt', 'keep'], ascending=[False, False]).dir.iloc[0],
                   best_pess_dir=sub_.sort_values(['acc_pess', 'keep'], ascending=[False, False]).dir.iloc[0])
N_PT_TOP = sum(v['pt_top'] for v in RK4.values()); N_PESS_TOP = sum(v['pess_top'] for v in RK4.values())
EXC = [s_ for s_, v in RK4.items() if not v['pess_top']]
FP, FH, FX = S['far_print_ihr_H1_survivors'], S['far_hand_borg'], S['fix_img_err_L16']
NOD = S['nod_bound_TK_check']
VAL = S['img_model_backtest_ihr']
T1I = S1['invariants']


def n_for(k, p=0.01, a=0.05):
    """cỡ mẫu ngẫu nhiên nhỏ nhất để cận trên Clopper–Pearson 1 phía (1−a) của tỉ lệ lỗi ≤ p khi thấy ≤ k lỗi."""
    n = k + 1
    while beta.ppf(1 - a, k + 1, n - k) > p:
        n += 1
    return n


N0_1, N1_1, N0_05 = n_for(0, 0.01), n_for(1, 0.01), n_for(0, 0.005)
ECD = pd.read_csv(HERE.parent / 'TN1_cong_kiem/err_class_catch.csv')
EC = lambda s_: float(ECD[(ECD.sach == s_) & (ECD.loai == 'anh_sai_khe')].ty_le_bi_ha_c.iloc[0])

L = []
w = L.append
w('# THỬ NGHIỆM 3 — Hướng nào để ảnh và chữ trong GOLD phải đúng, trên cả 8 bộ đang có trên đĩa (26/09/2026)')
w('')
w('Sinh tự động bởi `t03_report.py` từ `matrix.csv`, `t02_summary.json`, `t01_summary.json` '
  f'(invariant t01: **{"ALL PASS" if S1["invariants"]["all_pass"] else "FAIL"}**, t02: **{"ALL PASS" if INV["all_pass"] else "FAIL"}**). '
  '0 lần gọi API, không tải model, không mở ảnh bằng LLM, repo chỉ đọc ngoài `lab/thu_nghiem_anh_chu/TN3_tat_ca_bo/` và '
  '`measure_out/_thu_nghiem_anh_chu/TN3/`. MPS chỉ dùng cho bước chấm ứng viên sửa hộp của L83/KVK (TN2, 3 giây).')
w('')
w('Mục tiêu GOLD: (1) crop là đúng **một** chữ, (2) chữ đó đúng là nhãn. Ưu tiên độ chính xác hơn số lượng. '
  'Mọi hướng chỉ **hạ** ô (không bao giờ sửa nhãn); ô bị hạ không mất, chỉ không được chứng nhận "ảnh + chữ".')
w('')
w('Mức chắc: **CHẮC CHẮN THEO MÁY** = đo trên nhãn người IHR · **ƯỚC LƯỢNG** = suy từ văn bản người của dị bản + mô hình '
  'hiệu chuẩn trên IHR · **SUY ĐOÁN** = chuyển tỉ lệ từ sách khác, **cần người kiểm**.')
w('')
# ------------------------------------------------------------------ §0
tk3, tk4 = MX[(MX.set == 'TK') & (MX.dir == 'H3')].iloc[0], MX[(MX.set == 'TK') & (MX.dir == 'H4')].iloc[0]
l2 = MX[(MX.set == 'L16') & (MX.dir == 'H2')].iloc[0]
kv4, l84 = MX[(MX.set == 'KVK') & (MX.dir == 'H4')].iloc[0], MX[(MX.set == 'L83') & (MX.dir == 'H4')].iloc[0]
kvr = MX[(MX.set == 'KVK') & (MX.dir == 'Rv3')].iloc[0]
ch2 = MX[(MX.set == 'Chr') & (MX.dir == 'H2')].iloc[0]
stt = MX[MX.set.isin(['stt2', 'stt4', 'stt11'])]
w('## 0. Trả lời ngắn')
w('')
w(f'1. **Chỉ TruyenKieu1872 đạt mục tiêu** theo tiêu chí đăng ký trước (CHẮC CHẮN THEO MÁY). Hướng giữ nhiều ô nhất mà vẫn đạt là '
  f'**H3** (hình học + dị bản người): {I(tk3.keep)} ô ({P(tk3.keep_pct / 100, 1)} %), hai vế {acc_txt("TK", "H3")}. '
  f'H4 chặt hơn: {I(tk4.keep)} ô, {acc_txt("TK", "H4")}.')
w(f'2. **LucVanTien1916 không hướng nào đạt 99,5 %.** Tốt nhất là H2 = H4 (sách không có văn bản người của dị bản nên H3 ≡ H1): '
  f'{I(l2.keep)} ô ({P(l2.keep_pct / 100, 1)} %), {acc_txt("L16", "H2")} — hụt 99,5 % ở điểm, cận dưới {P(l2.both_lo)} %. CHẮC CHẮN THEO MÁY.')
w(f'3. **Thạch bản (KVK, L83): H4 chính xác nhất** (điểm KVK {P(kv4.both_pt)} %, L83 {P(l84.both_pt)} %), nhưng **cận bi quan chưa tới 99 %** '
  f'(KVK {P(kv4.acc_pess)} %, L83 {P(l84.acc_pess)} %; bỏ cận "không cần d" — độ nhạy thêm sau — còn {P(kv4.acc_pess_no_nod)} % / '
  f'{P(l84.acc_pess_no_nod)} %). ƯỚC LƯỢNG. Không chứng nhận được nếu không có người kiểm.')
w(f'4. **Chrestomathie1872 và 3 cuốn STT: không hướng nào đạt.** Chr H2 = H4: điểm {P(ch2.both_pt)} %, cận bi quan {P(ch2.acc_pess)} %. '
  f'STT H2 = H4: điểm {P(stt[stt.dir == "H2"].both_pt.min())}–{P(stt[stt.dir == "H2"].both_pt.max())} %, cận bi quan ≤ '
  f'{P(stt[stt.dir == "H2"].acc_pess.max())} % (tham chiếu Rv3: điểm ≤ {P(stt[stt.dir == "Rv3"].both_pt.max())} %, bi quan ≤ '
  f'{P(stt[stt.dir == "Rv3"].acc_pess.max())} %). SUY ĐOÁN — cần người kiểm.')
w(f'5. **H1 (hình học) là tầng bắt buộc đầu tiên ở mọi bộ.** Trên IHR nó cắt lỗi ảnh L16 {I(g("L16", "H0", "err_img_cells"))} → '
  f'{I(g("L16", "H1", "err_img_cells"))} ô (phần còn lại chủ yếu là "khe chưa xác định"), TK {I(g("TK", "H0", "err_img_cells"))} → '
  f'{I(g("TK", "H1", "err_img_cells"))}, chỉ hạ khoảng 5 % ô. Nhưng lỗi **chữ** gần như nguyên vẹn, nên một mình nó dừng ở '
  f'{P(g("L16", "H1"))} % (L16) / {P(g("TK", "H1"))} % (TK).')
w(f'6. **Vòng sửa hộp (H5, H6) gần như không đổi gì**: H5 thêm tổng {I(ADD5)} ô trên 8 bộ so với H1 ({ADD5_TXT}); '
  f'H6 thêm {I(ADD6)} ô so với H4 ({ADD6_TXT}) và làm L16 tụt {P(g("L16", "H4"))} → {P(g("L16", "H6"))} %. → để **TẮT** mặc định.')
w('7. **Hướng bền qua mọi loại sách: H4** (hình học → bộ kiểm ảnh → dị bản người nếu có). Theo điểm, H4 cao nhất (hoặc ngang) ở '
  f'{N_PT_TOP}/8 bộ, trải cả 4 loại sách (mộc bản, thạch bản, in văn xuôi, chép tay); theo cận dưới/bi quan cao nhất ở {N_PESS_TOP}/8 bộ'
  + (f' (ngoại lệ: {", ".join(EXC)} — xem §4)' if EXC else '') + '. Nhưng vế **chữ** chỉ được đẩy qua 99,5 % khi có văn bản người của '
  f'dị bản: trên TK, bỏ dị bản (H2) chỉ còn {P(g("TK", "H2"))} %, thêm dị bản (H4) lên {P(g("TK", "H4"))} %.')
w('8. **Bắt buộc người kiểm (hoặc đọc lại) trước khi gọi là "GOLD chính xác"**: L16 (sát ngưỡng), KVK, L83, Chr, stt2, stt4, stt11. '
  f'Đọc lại STT bằng kim lt2 (H7) **không đủ**: kể cả khi xoá hết lỗi riêng của STT, cận bi quan của STT vẫn ≤ '
  f'{P(stt.acc_pess_H7.max())} % (xem §6).')
w('')
# ------------------------------------------------------------------ §1
w('## 1. Cách làm (viết trước khi tính; không đổi sau khi thấy kết quả)')
w('')
w('### 1.1 Tám bộ')
w('')
w('| Bộ | Loại sách | GOLD | Sự thật có trên máy | Mức chắc của số độ chính xác |')
w('|---|---|---:|---|---|')
tru = {'stt2': 'không (Borg chỉ gián tiếp)', 'stt4': 'không (Borg chỉ gián tiếp)', 'stt11': 'không (Borg chỉ gián tiếp)',
       'Chr': 'không', 'L83': 'văn bản người của dị bản: LucVanTien1916 IHR', 'KVK': 'văn bản người của dị bản: Kiều 1871 LVD, TK1872 IHR',
       'L16': 'chữ người IHR + hộp cột người vẽ', 'TK': 'chữ người IHR + hộp cột người vẽ; dị bản Kiều 1871 LVD'}
for s in SETS8:
    r0 = MX[(MX.set == s) & (MX.dir == 'H0')].iloc[0]
    w(f'| {s} ({FULL[s]}) | {r0.type} | {I(r0.gold)} | {tru[s]} | {MUC[r0.truth]} |')
w('')
w('### 1.2 Bảy hướng + một tham chiếu')
w('')
w('| Mã | Nội dung | Ngưỡng / nguồn |')
w('|---|---|---|')
w('| H0 | giữ nguyên GOLD | — |')
w('| H1 | cổng hình học: hạ nếu A0 (crop rỗng/cắt nét, ảnh ô khác, 1 hộp 2 cột) ∪ B0 (cờ "một chữ": bleed, truncated, blank, tall, dup, '
  'ov_heavy, f_tight_nb) ∪ cột có số chữ OCR ≠ số âm QN ∪ cờ trượt vòng 2 | cờ trượt chọn NGOÀI sách: L16 dùng `bc_k06v1` (chọn trên TK '
  '= G6 của TN1); mọi bộ khác `bc_k05dxv05` (chọn trên L16 = G7 của TN1) |')
w('| H2 | H1 ∧ bộ kiểm ảnh ở τ = 0,995 | TN1/v3: L16 p_wood_T ∧ viss_T; TK p_wood_L ∧ viss_L; L83/KVK p_wood_T ∧ p_wood_L ∧ viss_X; '
  'Chr p_wood_T ∧ p_wood_L (C2); STT bộ kiểm viết tay LOBO r5 (q = 0,00015, ≥ 3 nguyên mẫu người) |')
w('| H3 | H1 ∧ nhãn được văn bản người của dị bản chứng (`attested`) | chỉ TK, KVK, L83 có văn bản người của dị bản; L16/Chr/STT: H3 ≡ H1 |')
w('| H4 | H1 ∧ H2 ∧ H3 | = cấu hình (e) của TN1 |')
w(f'| H5 | H1 ∪ ô được vòng sửa hộp TN2 nhận (luật trùng hộp NGHIÊM), không dính A0/B0/cột lệch; ô được sửa dùng crop mới | '
  f'θ TN2: L16 mô hình T θ = {F(S1["theta_used"]["LOBO"]["LucVanTien1916"][1], 2)}, TK mô hình L θ = {F(S1["theta_used"]["LOBO"]["TruyenKieu1872"][1], 2)} '
  f'(LOBO); bộ khác: T (θ {F(S1["theta_used"]["T_on_L16"], 2)}) và L (θ {F(S1["theta_used"]["L_on_TK"], 2)}) cùng chọn một ứng viên |')
w('| H6 | H4 ∪ ô được sửa (như H5) có `attested` ở bộ có dị bản | ô được sửa chỉ qua bộ kiểm của chính vòng sửa (m ≥ θ), không qua p_wood |')
w('| Rv3 | **tham chiếu, không xếp hạng**: H4 ∧ ¬luật A (ảnh ô khác, rescue, cầu tự dạng, văn bản yếu) ∧ ¬M-OCR t50 | = luật cấp ô của chính sách v3, BỎ các cổng cấp bộ |')
w('| H7 | chỉ mô tả, không chạy: đọc lại STT bằng kim lang_type = 2 (cần API) | §6 |')
w('')
w('### 1.3 Cách ra con số độ chính xác hai vế (đúng hai vế = crop ở đúng khe của chữ **và** nhãn đúng chữ đó)')
w('')
w('- **L16, TK — CHẮC CHẮN THEO MÁY.** Nhãn đúng = V1+(nhãn, chữ người IHR); ảnh đúng = `slot_ok = 1` (khe chưa xác định tính SAI). '
  'Ngưỡng của mỗi sách chọn trên sách kia (LOBO). CI 95 % bootstrap cụm theo trang, B = 2.000. Ô được sửa hộp: khe mới do TN2 chấm bằng hộp cột người vẽ.')
w('- **KVK, L83 — ƯỚC LƯỢNG.**')
w('  - Vế chữ: mỗi ô giữ thuộc một lớp dị bản (attested / contradicted / unattestable). Nhân số ô từng lớp với tỉ lệ nhãn sai **của cùng lớp, '
  'cùng hướng, đo trên TK** (hiệu chuẩn ngược; bootstrap trang TK, B = 2.000). **Giả định:** P(nhãn sai | lớp, hướng) chuyển nguyên từ TK '
  '(mộc bản, dị bản Kiều 1871, một tham chiếu) sang thạch bản (KVK: hai tham chiếu 1871/1872; L83: tham chiếu L16).')
w(f'  - Cận bi quan vế chữ = lớn nhất của: CI trên; cận "không cần d" (coi mọi chênh với dị bản là lỗi, q cận trên CI = {F(S["q_TK"]["hi"], 3)}, '
  f'KVK hai tham chiếu q = 1−(1−q)² = {F(S["q_TK"]["used"]["KVK"], 3)}); CI trên của tổng p M-OCR vòng 2.')
w(f'  - Vế ảnh: tổng p trượt từng ô của mô hình vòng 2 (`gate_v2`, logistic học trên IHR, bootstrap có sai số chuyển bộ σ = {F(S["gate_v2_sigma"], 3)}, '
  f'B = {I(S["B"]["img"])}). Hướng có bộ kiểm ảnh: (tổng p của phần giữ khi bỏ bộ kiểm) × tỉ lệ nhận trượt của bộ kiểm, đo trên IHR ở ô trượt/khe '
  f'chưa xđ **sống sót H1**: {FP["k"]}/{FP["n"]} = {P(FP["p"], 1)} %. Cận bi quan = không cho bộ kiểm điểm nào. Ô được sửa hộp: tỉ lệ khe mới sai '
  f'đo trên L16 = {FX["k_wrong"]}/{FX["n"]}.')
w('- **Chr, stt2/4/11 — SUY ĐOÁN.**')
w(f'  - Vế ảnh như trên. STT: bể trượt thật {I(S["stt_slip_pool"][0])}–{I(S["stt_slip_pool"][1])} ô (vòng 4–5) phân bổ theo p trượt vòng 2; '
  f'bộ kiểm viết tay nhận {P(FH["p"], 1)} % [{P(FH["ci"][0], 1)}–{P(FH["ci"][1], 1)}] crop trượt thật (đo trên Borg, LOBO theo sách).')
w('  - Vế chữ: không có sự thật cục bộ. Chuyển tỉ lệ nhãn sai đo trên L16/TK ở **cùng hướng** (bỏ điều kiện dị bản). Điểm = trung bình '
  'L16/TK; cận bi quan = lớn nhất của L16, TK, L83, KVK. Riêng STT cộng thêm lỗi chỉ STT có: cầu tự dạng + nhãn rescue (p vòng 2) '
  f'và bể "Hán hoá" do OCR lt1 {S["hanhoa_lt1"]["point"]} [{S["hanhoa_lt1"]["lo"]}–{S["hanhoa_lt1"]["hi"]}] ô, rải đều trên '
  f'{I(S["hanhoa_lt1"]["n_pop"])} ô có âm Borg viết bằng chữ Nôm riêng.')
w('- **Hai vế ngoài IHR**: số lỗi = lỗi ảnh + lỗi chữ (cộng, tức cận hợp). Khoảng = cộng hai đầu tương ứng (rộng hơn khoảng đồng thời).')
w('')
w('### 1.4 Tiêu chí đạt (định trước)')
w('')
w('- Đo được (L16, TK): điểm ≥ 99,5 %; ghi "chắc" khi cận dưới CI ≥ 99,5 %.')
w('- Ước lượng / suy đoán: **cận bi quan ≥ 99,0 %**.')
w('- Phải giữ ≥ 50 ô. Hướng tốt nhất của một bộ = hướng đạt mà giữ nhiều ô nhất. Rv3 là tham chiếu, không xếp hạng.')
w('')
# ------------------------------------------------------------------ §2
w('## 2. Ma trận bộ × hướng')
w('')
w('### 2.1 Bảng gọn: `% ô giữ · độ chính xác hai vế % [khoảng]`')
w('')
w('Khoảng: L16/TK = CI 95 % cụm trang; bộ khác = [cận bi quan – cận lạc quan]. ✔ = đạt tiêu chí §1.4.')
w('')
w('| Bộ (mức chắc) | ' + ' | '.join(DIRS) + ' |')
w('|---|' + '---|' * len(DIRS))
for s in SETS8:
    cells = []
    for d in DIRS:
        r = MX[(MX.set == s) & (MX.dir == d)].iloc[0]
        mk = ' ✔' if r.meets != 'khong' else ''
        cells.append(f'{P(r.keep_pct / 100, 1)} · {acc_txt(s, d, bold=False)}{mk}' if r.keep > 0 else '0 · —')
    w(f'| **{s}** ({MUC[MX[MX.set == s].truth.iloc[0]]}) | ' + ' | '.join(cells) + ' |')
w('')
w('### 2.2 Bảng đủ theo từng bộ')
w('')
for s in SETS8:
    sub = MX[MX.set == s]
    r0 = sub.iloc[0]
    w(f'#### {s} — {FULL[s]} ({r0.type}; GOLD {I(r0.gold)}; {I(r0.classes_before)} lớp nhãn) — {MUC[r0.truth]}')
    w('')
    w('| Hướng | Ô giữ (%) | Ô hạ | Lớp mất hết ô (ô trong đó) | Hai vế [khoảng] | Vế chữ [khoảng] | Vế ảnh [khoảng] | Lỗi ước/đo (ô) | Đạt |')
    w('|---|---:|---:|---:|---|---|---|---:|---|')
    for d in DIRS:
        r = sub[sub.dir == d].iloc[0]
        if r.keep == 0:
            w(f'| {DN[d]} | 0 | {I(r.demote)} | {I(r.classes_lost)} ({I(r.cells_in_lost_classes)}) | — | — | — | — | — |')
            continue
        lab = f'{P(r.lab_pt)} [{P(r.lab_lo)}–{P(r.lab_hi)}]'
        img = f'{P(r.img_pt)} [{P(r.img_lo)}–{P(r.img_hi)}]'
        if r.truth == 'ĐO':
            err = f'chữ {I(r.err_lab_cells)} · ảnh {I(r.err_img_cells)} · hai vế {I(r.err_both_cells)} / {I(r.n_eval)} ô có GT'
        else:
            err = f'{F(r.err_both_cells)} (bi quan {F(r.err_both_cells_hi)})'
        fx = f' ({I(r.n_fix_kept)} ô crop sửa)' if r.n_fix_kept else ''
        w(f'| {DN[d]} | {I(r.keep)}{fx} ({P(r.keep_pct / 100, 1)} %) | {I(r.demote)} | {I(r.classes_lost)} ({I(r.cells_in_lost_classes)}) | '
          f'{acc_txt(s, d)} | {lab} | {img} | {err} | {MEETS[r.meets]} |')
    w('')
    if s in ('L83', 'KVK'):
        DRV = {'can_khong_can_d': 'cận không-d', 'T_ci_tren': 'CI trên của phép chuyển TK', 'M_OCR_v2_ci_tren': 'M-OCR vòng 2'}
        drv = {d: DRV[sub[sub.dir == d].lab_pess_driver.iloc[0]] for d in DIRS if sub[sub.dir == d].keep.iloc[0] > 0}
        w('Cận bi quan vế chữ do thành phần nào quyết định: ' + ('mọi hướng: ' + next(iter(drv.values())) if len(set(drv.values())) == 1
                                                              else '; '.join(f'{d}: {v}' for d, v in drv.items()))
          + '. Độ nhạy (thêm sau): bỏ cận "không cần d" → cận bi quan hai vế ' + ', '.join(
            f'{d} {P(sub[sub.dir == d].acc_pess_no_nod.iloc[0])} %' for d in ('H2', 'H3', 'H4', 'Rv3')) + '.')
        w('')
    if s in ('stt2', 'stt4', 'stt11'):
        w('Độ nhạy (thêm sau, không xếp hạng): nếu bể trượt STT **không** được H1 giảm (rải đều theo số ô, như quy tắc `stt_ok_allowed` '
          'của v3) thì cận bi quan hai vế: ' + ', '.join(f'{d} {P(sub[sub.dir == d].acc_pess_stt_uniform.iloc[0])} %' for d in ('H1', 'H2', 'Rv3')) + '.')
        w('')
# ------------------------------------------------------------------ §3
w('## 3. Xếp hạng theo bộ')
w('')
w('| Bộ | Loại sách | Mức chắc | Hướng tốt nhất đạt tiêu chí (ô giữ) | Các hướng đạt | Hướng chính xác nhất theo cận dưới/bi quan | Rv3 (tham chiếu) |')
w('|---|---|---|---|---|---|---|')
for s in SETS8:
    rk = R[s]
    r0 = MX[(MX.set == s)].iloc[0]
    best = f'**{rk["best"]}** ({I(rk["best_keep"])} ô, {acc_txt(s, rk["best"], bold=False)})' if rk['best'] else '**không hướng nào**'
    mp = rk['max_pess_dir']
    w(f'| {s} | {r0.type} | {MUC[r0.truth]} | {best} | {", ".join(rk["dat"]) or "—"} | {mp}: {I(g(s, mp, "keep"))} ô, '
      f'{acc_txt(s, mp, bold=False)} | {MEETS[rk["rv3_meets"]]}: {acc_txt(s, "Rv3", bold=False)} |')
w('')
w('Ghi chú xếp hạng:')
w(f'- TK: H3 giữ nhiều hơn H4 {I(tk3.keep - tk4.keep)} ô, đổi lại {I(tk3.classes_lost)} lớp nhãn mất hết ô (H4: {I(tk4.classes_lost)}). '
  'Quy tắc "cần attested" không tham số, nhưng quyết định dùng nó dựa trên số đo TK và TK cũng là sách kiểm định ngược của text_attested '
  '→ số của H3/H4 trên TK là **trong mẫu** đối với luật này (như TN1 §5).')
L16g7 = S['L16_H1_G7_literal_in_sample']
w(f'- L16: G7 nguyên văn (ngưỡng trượt chọn trên chính L16) cho {P(L16g7["both_pt"])} % — không dùng vì không LOBO. '
  f'H2 = H4 = Rv3 ở L16 vì L16 không có văn bản người của dị bản và không có ô luật A/M-OCR nào lọt qua H2.')
w('')
# ------------------------------------------------------------------ §4
w('## 4. Hướng bền qua loại sách và khuyến nghị cho pipeline')
w('')
w('| Loại sách | Bộ | Hướng điểm cao nhất | H4: điểm | H4 cao nhất theo điểm? | Hướng cận dưới/bi quan cao nhất (giá trị) | H4: cận dưới/bi quan | H4 cao nhất theo cận? |')
w('|---|---|---|---|---|---|---|---|')
for s in SETS8:
    v = RK4[s]
    w(f'| {MX[MX.set == s].type.iloc[0]} | {s} | {v["best_pt_dir"]} ({P(g(s, v["best_pt_dir"]))} %) | {P(g(s, "H4"))} % | '
      f'{"có" if v["pt_top"] else "không"} | {v["best_pess_dir"]} ({P(g(s, v["best_pess_dir"], "acc_pess"))} %) | {P(g(s, "H4", "acc_pess"))} % | '
      f'{"có" if v["pess_top"] else "không"} |')
w('')
w(f'**Hướng bền = H4** (H1 → bộ kiểm ảnh → dị bản người nếu có): điểm cao nhất (hoặc ngang — hướng trùng tập ô với H4 như H2/H6 '
  f'ở bộ không có dị bản hay không có ô sửa) ở {N_PT_TOP}/8 bộ; theo cận dưới/bi quan cao nhất ở {N_PESS_TOP}/8 bộ'
  + (f'. Ngoại lệ {", ".join(EXC)}: cận bi quan vế chữ của H2/H4 lấy tỉ lệ từ L83 ở cùng hướng (chủ yếu là tỉ lệ lỗi của lớp "không tham chiếu" trên TK), '
     'cao hơn chút so với H1 — chênh nằm trong độ bất định của phép chuyển, không phải H1 tốt hơn' if EXC else '')
  + '. Không hướng nào thay được người ở vế chữ khi thiếu văn bản người.')
w('')
w('**Khuyến nghị cho pipeline (tệp phụ, không phá `labels.csv`):**')
w('')
w(f'1. **Luôn chạy H1** cho mọi sách mới (rẻ, 0 tham số chọn trên sách đó; IHR hạ khoảng 5 %). Ô bị H1 hạ → `text_only`.')
w('2. **Nếu có văn bản người của dị bản**: thêm H3 (chỉ nhận `attested`). Với mộc bản kiểu TK, H3 đã đủ (đo 99,72 %). '
  'Với thạch bản dùng **H4** và vẫn phải lấy mẫu người (§7), vì chuyển tỉ lệ TK → thạch bản chưa kiểm định được.')
w('3. **Nếu không có văn bản người**: H4 (= H2) là trần của máy — đo được 99,49 % trên L16; ở Chr/STT chỉ là SUY ĐOÁN. Gắn nhãn '
  '`chưa chứng nhận` cho tới khi có mẫu người.')
w('4. **STT (chép tay)**: thêm luật A + M-OCR (Rv3) trước khi cho người xem. Rv3 nâng điểm STT lên '
  f'{P(stt[stt.dir == "Rv3"].both_pt.min())}–{P(stt[stt.dir == "Rv3"].both_pt.max())} % (từ {P(stt[stt.dir == "H4"].both_pt.min())}–'
  f'{P(stt[stt.dir == "H4"].both_pt.max())} % ở H4), vẫn chưa đủ.')
w(f'5. **Vòng sửa hộp: TẮT mặc định** (H5 thêm {I(ADD5)} ô, H6 thêm {I(ADD6)} ô trên cả 8 bộ; L16 H6 giảm độ chính xác).')
w('')
# ------------------------------------------------------------------ §5
w('## 5. Kiểm định và chẩn đoán (đọc để biết ước lượng tin được tới đâu)')
w('')
w('**Vòng sửa hộp TN2 trên cả 6 book_set** (`t01_summary.json`; chạy thêm L83/KVK bằng bản sao s1a/s1b/s2 của TN2, chỉ đổi thư mục ra):')
w('')
w('| book_set | ô bị cờ kim_off | T nhận thô / L nhận thô / cùng ứng viên | huỷ trùng hộp | được sửa | khe mới đúng (IHR) |')
w('|---|---:|---|---:|---:|---|')
nm = {'LucVanTien1916': 'L16', 'TruyenKieu1872': 'TK', 'Chrestomathie1872': 'Chr', 'SachThanhTruyen': 'STT', 'LucVanTien1883': 'L83', 'KimVanKieu1884': 'KVK'}
for bs, a in nm.items():
    raw = T1I.get(f'{bs}_raw_T_L_both')
    raw_t = f'{raw[0]} / {raw[1]} / {raw[2]}' if raw else 'LOBO một mô hình'
    ns = S1['ihr_fix_new_slot'].get(bs)
    nst = f'{ns.get("1", 0)}/{ns.get("1", 0) + ns.get("0", 0)}' if ns else '—'
    w(f'| {a} | {I(T1I[f"{bs}_n_flag"])} | {raw_t} | {T1I[f"{bs}_huy_trung_hop_don_dieu"][0]} | {I(T1I[f"{bs}_n_fix"])} | {nst} |')
w('')
w(f'Bất biến L83/KVK: crop cũ cắt lại trùng md5 tệp giao nộp ({T1I["L83_KVK_old_crop_md5_reproduce"]}), chữ kim tại nom_idx = ocr_char ở '
  f'100 % ô bị cờ ({T1I["L83_KVK_kim_char_ok_all"]}); quyết định L16/TK/Chr/STT trùng từng ô với `TN2/decisions.csv`.')
w('')
w('**Kiểm định ngược mô hình ảnh trên IHR** (học trượt trên một sách, dự tổng p trong phần giữ của sách kia; "thật U" = lệch khe xác nhận '
  'bởi IHR-2 ∪ CG3, "thật slot≠1" = tính cả khe chưa xác định):')
w('')
w('| Sách · hướng | dự | thật U | thật slot≠1 |')
w('|---|---:|---:|---:|')
for k, v in VAL.items():
    w(f'| {k} | {F(v.get("du_sum_p", v.get("du")))} | {v["that_U"]} | {v["that_slot_ne_1"]} |')
w('')
vt, vl = VAL['TK|H1'], VAL['L16|H1']
w(f'→ Sau H1, mô hình dự **thừa** ở TK ({F(vt["du_sum_p"])} so với thật {vt["that_U"]}–{vt["that_slot_ne_1"]}) và **đúng cỡ** ở L16 với lệch khe thật '
  f'({F(vl["du_sum_p"])} so với {vl["that_U"]}), nhưng **không thấy** "khe chưa xác định" (L16 còn {vl["that_slot_ne_1"]} ô slot≠1). Vì số IHR tính '
  'khe chưa xác định là sai còn số ước lượng chỉ đếm trượt thật, **số IHR khắt khe hơn số ước lượng** của các bộ khác ở cùng hướng.')
w('')
w('**Cận "không cần d" trên TK** (vế chữ, so với lỗi nhãn đo được; hợp lệ khi cận ≥ đo):')
w('')
w('| Hướng | c (tỉ lệ bị chống) | cận | đo | hợp lệ |')
w('|---|---:|---:|---:|---|')
for d, v in NOD.items():
    w(f'| {d} | {F(v["c"], 3)} | {F(v["bound"])} | {v["measured"]} | {v["valid"]} |')
w('')
RAT = [v['bound'] / max(v['measured'], 1) for v in NOD.values()]
w(f'→ Cận luôn hợp lệ nhưng **lỏng {F(min(RAT), 0)}–{F(max(RAT), 0)} lần** trên TK (chữ thật của TK khác Kiều 1871 ở khoảng '
  f'{P(S["q_TK"]["d_point"], 1)} % vị trí dù nhãn đúng, tham số d của text_attested). Ở KVK (hai tham chiếu, '
  f'c ở H2 = {F(kv4.nod_c, 3)}) cận này chặt hơn nhiều nhưng vẫn là thành phần quyết định cận bi quan. Đây là lý do thạch bản không đạt '
  '99 % bi quan: **không có cặp dị bản thứ hai có nhãn người để kiểm định phép chuyển TK → thạch bản**.')
w('')
w(f'**Bộ kiểm ảnh với ô trượt sống sót H1**: nhận {FP["k"]}/{FP["n"]} ({P(FP["p"], 1)} %; L16 {FP["by_book"]["L16"][0]}/{FP["by_book"]["L16"][1]}, '
  f'TK {FP["by_book"]["TK"][0]}/{FP["by_book"]["TK"][1]}) — cao hơn hẳn tỉ lệ nhận ô sai khe nói chung của bộ kiểm p_wood (TN1 `err_class_catch.csv`: '
  f'L16 {P(1 - EC("L16"), 1)} %, TK {P(1 - EC("TK"), 1)} %): trượt nhỏ qua được hình học thì cũng dễ lừa bộ kiểm.')
w('')
# ------------------------------------------------------------------ §6
w('## 6. H7 — đọc lại STT bằng kim lang_type = 2 (chỉ mô tả; cần API, không chạy)')
w('')
w('- Số đã có (báo cáo v2 §3.1, §6.1): STT được OCR bằng lt1; lt1 gần như mù chữ Nôm riêng (đúng 2,45 % so với lt2 96,4 % trên chữ Nôm riêng, '
  'CHẮC CHẮN THEO MÁY); bể "Hán hoá" khoảng 750 [481–954] ô (CÓ THỂ); lợi ích ước sửa khoảng 340–1.100 nhãn; chi phí 448 lượt API '
  '(khoảng 18 phút), đề xuất pilot 41 lượt. Không giải được vế "một chữ".')
w('- **Trần lợi ích** (SUY ĐOÁN, tính từ ma trận: giả sử lt2 xoá **hết** lỗi riêng STT — Hán hoá, cầu tự dạng, rescue):')
w('')
w('| Bộ · hướng | hiện tại: điểm / bi quan | H7 trần: điểm / bi quan | H7 trần, bể trượt không được H1 giảm: bi quan |')
w('|---|---|---|---|')
for s in ('stt2', 'stt4', 'stt11'):
    for d in ('H1', 'H2', 'Rv3'):
        r = MX[(MX.set == s) & (MX.dir == d)].iloc[0]
        w(f'| {s} · {d} | {P(r.both_pt)} / {P(r.acc_pess)} % | {P(r.acc_pt_H7)} / {P(r.acc_pess_H7)} % | {P(r.acc_pess_H7_uniform)} % |')
w('')
w('→ H7 nâng điểm STT ở H1 lên khoảng 98 %, nhưng ở H2/Rv3 (tập đã lọc) chỉ thêm vài phần mười điểm, và **cận bi quan vẫn dưới 99 %** '
  f'vì hai phần H7 không chạm tới: (i) tỉ lệ nhãn sai chuyển từ sách khác (ở H2 bi quan tới {P(json.loads(g("stt2", "H2", "lab_rate_transfer"))[2])} %, '
  'lấy từ L83 cùng hướng, chủ yếu là lớp "không có tham chiếu" của TK); '
  '(ii) crop trượt một phần mà bộ kiểm viết tay vẫn nhận. H7 đáng làm để **giảm lỗi thật**, không để **chứng nhận**.')
w('')
# ------------------------------------------------------------------ §7
w('## 7. Cái gì chỉ người kiểm mới chốt được')
w('')
w('1. **Vế "đúng MỘT chữ" ở mọi bộ.** `slot_ok` chỉ xét tâm crop nằm đúng khe; không có sự thật nào về mực chữ kề lọt vào crop. '
  'Cờ B0 (bleed, tight_nb, tall…) được áp mà không đo được lỗi còn sót. CHẮC CHẮN (thiết kế đo).')
w(f'2. **L16**: {I(l2.keep)} ô H2/H4 ở {P(l2.both_pt)} % — {I(l2.err_img_cells)} ô lỗi vế ảnh đều là "khe chưa xác định" (TN1: sai khe 0), '
  f'cộng {I(l2.err_lab_cells)} ô sai chữ; người xem các ô này là chốt được sách này ở tập H4.')
w('3. **KVK, L83**: phép chuyển tỉ lệ lỗi nhãn TK → thạch bản chưa kiểm định được (một cặp dị bản duy nhất có nhãn người). '
  'Mẫu người trên tập H4 là cách duy nhất đưa cận bi quan lên ≥ 99 %.')
w('4. **Chr**: không có văn bản người nào trên máy → vế chữ hoàn toàn chuyển từ sách khác.')
w('5. **STT**: vế chữ (lt1, luật cầu tự dạng/rescue, quy ước người phiên Borg khác nhãn) và vế ảnh (trượt một phần) đều cần người.')
w('')
w(f'**Cỡ mẫu người để chứng nhận một tập đã lọc** (lấy ngẫu nhiên từ phần giữ của hướng; cận trên Clopper–Pearson một phía 95 %): '
  f'0 lỗi trong **{N0_1}** ô → sai ≤ 1 %; ≤ 1 lỗi trong **{N1_1}** ô → sai ≤ 1 %; 0 lỗi trong **{N0_05}** ô → sai ≤ 0,5 %. '
  'Cần làm riêng cho mỗi bộ (hoặc mỗi tầng trong bộ) muốn chứng nhận; ô bị hạ vẫn giữ trong bộ ở tầng thấp. Giao thức người v3 '
  '(`$SP/r6/policy_v3/human_protocol.md`) đã có sẵn công cụ và ước 13–16 giờ cho lượt tối thiểu.')
w('')
# ------------------------------------------------------------------ §8
w('## 8. Giới hạn')
w('')
w('- Chỉ hai sách có nhãn người (L16, TK); TK "trong miền" của kim nên tỉ lệ lỗi nền thấp, có thể không đại diện cho thạch bản (báo cáo v2 §3.1).')
w('- Số ước lượng cộng lỗi ảnh và lỗi chữ (cận hợp) và cộng hai đầu khoảng → khoảng rộng hơn khoảng đồng thời thật.')
w('- Mô hình ảnh vòng 2 chỉ ước "trượt thật", không ước "khe chưa xác định" — số IHR khắt khe hơn.')
w(f'- Cận bi quan của thạch bản dùng cận "không cần d" (định trước); đã biết nó lỏng {F(min(RAT), 0)}–{F(max(RAT), 0)} lần trên TK. Cột "độ nhạy" bỏ cận này được thêm **sau** '
  'khi thấy kết quả và **không** dùng để xếp hạng.')
w('- STT: bể trượt 1.630–2.500 và bể Hán hoá 750 [481–954] là SUY ĐOÁN/CÓ THỂ từ vòng 4–5; cách phân bổ vào từng ô (theo p trượt vòng 2, '
  'rải đều trên ô âm Borg-"nom") là giả định.')
w('- H6: ô được sửa không qua p_wood ở τ = 0,995 (điểm cho crop mới chưa tính); chỉ qua bộ kiểm của vòng sửa.')
w('- Luật "cột lệch số chữ" và "tall" (trong H1) được thêm vào v3 sau khi nhìn phần dư IHR (TN1 §5).')
w('')
# ------------------------------------------------------------------ §9
w('## 9. Tệp và tái lập')
w('')
w('- `tn2_l83kvk/{s1a_kimdet,s1b_candidates,s2_score}.py`: bản sao TN2, **chỉ** đổi thư mục ra (`measure_out/_thu_nghiem_anh_chu/TN3/tn2_l83kvk/`), '
  'danh sách bộ (L83, KVK) và cấu hình (KVK = `pipeline_KimVanKieu1884_b1.yaml` chính thức B1\').')
w('- `t01_fix_decisions.py` → `measure_out/_thu_nghiem_anh_chu/TN3/fix_decisions.csv`, `t01_summary.json` (dùng nguyên `propose`/`resolve` của TN2).')
w('- `t02_matrix.py` → `matrix.csv` (8 bộ × 8 hướng, mọi cột số), `t02_summary.json` (invariant, tham số, kiểm định), '
  '`measure_out/_thu_nghiem_anh_chu/TN3/cells_masks.pkl` (ô nào giữ ở hướng nào — cho người kiểm/pipeline), `boot_img_sums.npy`.')
w('- `t03_report.py` → tệp này + `tong_hop.html`.')
w('')
w('```bash')
w('cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR; D=lab/thu_nghiem_anh_chu/TN3_tat_ca_bo')
w('.venv/bin/python $D/tn2_l83kvk/s1a_kimdet.py               # hộp kim + detector thô L83/KVK (~50 s)')
w('.venv/bin/python $D/tn2_l83kvk/s1b_candidates.py --workers 4   # ứng viên + crop (~5 s)')
w('.venv/bin/python $D/tn2_l83kvk/s2_score.py                 # điểm bộ kiểm T/L (MPS, ~3 s)')
w('.venv/bin/python $D/t01_fix_decisions.py                   # quyết định sửa hộp 6 book_set (~6 s)')
w('.venv/bin/python $D/t02_matrix.py                          # ma trận + bootstrap (~35 s CPU)')
w('.venv/bin/python $D/t03_report.py                          # KET_QUA.md + tong_hop.html')
w('```')
w('')
w('Đầu vào chỉ đọc: `$SP/r6/policy_v3/out/{base_v2.pkl, v02_cells_0.995.pkl}`, `$SP/r6/policy_v3/summary.json`, `$SP/gold_img_audit/gate_v2/` '
  '(mô hình trượt vòng 2), `$SP/r5/text_attested/out/summary.json`, `lab/.../TN1_cong_kiem/{t01_summary.json, projection.csv, ladder_b.csv, main_table.csv}`, '
  '`lab/.../TN2_vong_sua_hop/{s3_eval.py, s3_summary.json}`, `measure_out/_thu_nghiem_anh_chu/TN2/{scores.csv, decisions.csv}`, `dataset/_ALL/labels.csv`.')
w('')
w('Invariant chính (`t02_summary.json`): H2/H4/V3 trùng TN1 (d)/(e)/v3 ở cả 6 book_set; H1 trùng G7 (TK và 4 bộ ngoài IHR) và G6 (L16); '
  'H0/H2/H4/Rv3 trên L16/TK tái lập đúng độ chính xác TN1 (a)/(d)/(e)/(f); p trượt vòng 2 tái lập `gate_v2.json`; σ chuyển bộ trùng; '
  'bể STT = [1.630, 2.500]; 4.750 ô âm Borg-"nom"; cận không-d hợp lệ trên TK ở mọi hướng.')
(HERE / 'KET_QUA.md').write_text('\n'.join(L) + '\n', encoding='utf-8')

# ================================================================== tong_hop.html
def esc(x):
    return str(x).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


CLS = {'ĐO': 'do', 'ƯỚC LƯỢNG': 'uoc', 'SUY ĐOÁN': 'suy'}
th = ''.join(f'<th scope="col">{esc(DN[d])}</th>' for d in DIRS)
body = []
for s in SETS8:
    sub = MX[MX.set == s]
    tr = sub.truth.iloc[0]
    tds = []
    for d in DIRS:
        r = sub[sub.dir == d].iloc[0]
        ok = r.meets != 'khong'
        best = R[s]['best'] == d
        if r.keep == 0:
            tds.append(f'<td class="{CLS[tr]}"><span class="k">0 ô</span></td>')
            continue
        tds.append(
            f'<td class="{CLS[tr]}{" ok" if ok else ""}{" best" if best else ""}">'
            f'<div class="a">{P(r.both_pt)}<small> %</small></div>'
            f'<div class="ci">[{P(r.both_lo)}–{P(r.both_hi)}]</div>'
            f'<div class="k">{I(r.keep)} ô · {P(r.keep_pct / 100, 1)} %</div>'
            f'<div class="k">mất {I(r.classes_lost)} lớp</div>'
            + ('<div class="tag">ĐẠT</div>' if ok else '') + '</td>')
    body.append(f'<tr><th scope="row"><b>{esc(s)}</b><br><span class="sub">{esc(FULL[s])}</span><br>'
                f'<span class="sub">{esc(sub.type.iloc[0])} · GOLD {I(sub.gold.iloc[0])}</span><br>'
                f'<span class="pill {CLS[tr]}">{esc(MUC[tr])}</span></th>' + ''.join(tds) + '</tr>')
rank_rows = []
for s in SETS8:
    rk = R[s]
    mp = rk['max_pess_dir']
    tr = MX[MX.set == s].truth.iloc[0]
    rank_rows.append(
        f'<tr><td class="{CLS[tr]}"><b>{esc(s)}</b></td><td>{esc(MX[MX.set == s].type.iloc[0])}</td>'
        f'<td>{("<b>" + rk["best"] + "</b> · " + I(rk["best_keep"]) + " ô") if rk["best"] else "<b>không hướng nào</b>"}</td>'
        f'<td>{esc(mp)} · {P(g(s, mp))} % (dưới/bi quan {P(g(s, mp, "acc_pess"))} %) · {I(g(s, mp, "keep"))} ô</td>'
        f'<td>{"cần" if not rk["best"] or s == "L16" else "không bắt buộc (vẫn nên lấy mẫu)"}</td></tr>')
html = f'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TN3 tổng hợp hướng GOLD</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1d1b; --mut:#6b6a66; --line:#dcd9d2; --do:#dff1e3; --do-b:#2f7d45; --uoc:#fbf0cc; --uoc-b:#9a7410;
  --suy:#ecebe8; --suy-b:#6b6a66; --ok:#1f6f3a; --card:#ffffff; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#161614; --fg:#ecebe6; --mut:#a3a19b; --line:#3a3935;
  --do:#1d3a25; --do-b:#7fcf95; --uoc:#3d3314; --uoc-b:#e6c35a; --suy:#2a2927; --suy-b:#a3a19b; --ok:#8fe0a8; --card:#1f1f1c; }} }}
:root[data-theme="dark"] {{ --bg:#161614; --fg:#ecebe6; --mut:#a3a19b; --line:#3a3935; --do:#1d3a25; --do-b:#7fcf95; --uoc:#3d3314;
  --uoc-b:#e6c35a; --suy:#2a2927; --suy-b:#a3a19b; --ok:#8fe0a8; --card:#1f1f1c; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
main {{ max-width:1400px; margin:0 auto; padding:20px 16px 48px; }}
h1 {{ font-size:1.35rem; margin:0 0 6px; }} h2 {{ font-size:1.1rem; margin:28px 0 8px; }}
p, li {{ color:var(--fg); }} .mut {{ color:var(--mut); font-size:.9rem; }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 4px; }}
.pill {{ display:inline-block; border-radius:999px; padding:1px 9px; font-size:.78rem; border:1px solid; white-space:nowrap; }}
.pill.do {{ background:var(--do); border-color:var(--do-b); }} .pill.uoc {{ background:var(--uoc); border-color:var(--uoc-b); }}
.pill.suy {{ background:var(--suy); border-color:var(--suy-b); }}
.wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; vertical-align:top; text-align:left; }}
thead th {{ position:sticky; top:0; background:var(--card); font-size:.82rem; font-weight:600; }}
tbody th {{ min-width:150px; font-weight:400; }}
td.do {{ background:var(--do); }} td.uoc {{ background:var(--uoc); }} td.suy {{ background:var(--suy); }}
td.ok {{ outline:2px solid var(--ok); outline-offset:-2px; }} td.best {{ outline-width:3px; }}
.a {{ font-size:1.05rem; font-weight:600; }} .a small {{ font-weight:400; font-size:.75rem; }}
.ci, .k, .sub {{ font-size:.78rem; color:var(--mut); }} .tag {{ font-size:.72rem; font-weight:700; color:var(--ok); }}
.grid td, .grid th {{ font-size:.88rem; }}
</style></head><body><main>
<h1>TN3 — Hướng nào để ảnh và chữ trong GOLD đúng, trên 8 bộ</h1>
<p class="mut">26/09/2026 · sinh bởi <code>t03_report.py</code> từ <code>matrix.csv</code> · 0 API · invariant {"ALL PASS" if INV["all_pass"] else "FAIL"}.
Mỗi ô: độ chính xác hai vế (crop đúng khe <b>và</b> nhãn đúng chữ), khoảng, số ô giữ, số lớp nhãn mất hết ô.
Khoảng: bộ đo được = CI 95 % cụm trang; bộ khác = [cận bi quan – cận lạc quan]. Viền xanh = đạt tiêu chí định trước
(đo: ≥ 99,5 % ở điểm; ước/suy: cận bi quan ≥ 99 %); viền đậm = hướng tốt nhất của bộ.</p>
<div class="legend"><span class="pill do">xanh: đo trên nhãn người (CHẮC CHẮN THEO MÁY)</span>
<span class="pill uoc">vàng: ước lượng (dị bản người + mô hình hiệu chuẩn IHR)</span>
<span class="pill suy">xám: suy đoán — cần người kiểm</span></div>
<h2>Ma trận bộ × hướng</h2>
<div class="wrap"><table><thead><tr><th scope="col">Bộ</th>{th}</tr></thead><tbody>
{''.join(body)}
</tbody></table></div>
<h2>Xếp hạng theo bộ</h2>
<div class="wrap"><table class="grid"><thead><tr><th>Bộ</th><th>Loại sách</th><th>Hướng tốt nhất đạt tiêu chí</th>
<th>Hướng chính xác nhất (cận dưới/bi quan)</th><th>Người kiểm</th></tr></thead><tbody>{''.join(rank_rows)}</tbody></table></div>
<h2>Khuyến nghị cho pipeline</h2>
<ol>
<li><b>Luôn chạy H1</b> (hình học) ở mọi sách: bắt gần hết lỗi sai khe, hạ khoảng 5 % trên IHR.</li>
<li><b>Có văn bản người của dị bản</b>: thêm H3; mộc bản kiểu TK đạt ngay (H3 {P(tk3.both_pt)} % [{P(tk3.both_lo)}–{P(tk3.both_hi)}], {I(tk3.keep)} ô).
Thạch bản: dùng H4 và vẫn phải lấy mẫu người.</li>
<li><b>Không có văn bản người</b>: H4 (= H2) là trần của máy — L16 đo {P(l2.both_pt)} % (hụt 99,5 %). Gắn "chưa chứng nhận" tới khi có mẫu người.</li>
<li><b>Hướng bền qua mọi loại sách: H4</b> (điểm cao nhất hoặc ngang ở {N_PT_TOP}/8 bộ; theo cận dưới/bi quan {N_PESS_TOP}/8).</li>
<li><b>Vòng sửa hộp (H5/H6): tắt mặc định</b> — H5 thêm {I(ADD5)} ô trên 8 bộ, H6 thêm {I(ADD6)}, và làm L16 tụt {P(g("L16", "H4"))} → {P(g("L16", "H6"))} %.</li>
<li><b>STT</b>: thêm luật A + M-OCR (Rv3); đọc lại lt2 (H7) giảm lỗi thật nhưng không đủ để chứng nhận.</li>
<li>Chứng nhận một tập đã lọc: mẫu ngẫu nhiên {N0_1} ô không lỗi → sai ≤ 1 % (95 % một phía); {N0_05} ô không lỗi → ≤ 0,5 %.</li>
</ol>
<p class="mut">Chi tiết cách ước lượng, giả định, kiểm định và giới hạn: <code>KET_QUA.md</code> cùng thư mục.</p>
</main></body></html>
'''
(HERE / 'tong_hop.html').write_text(html, encoding='utf-8')
print('OK', len(L), 'dòng MD;', len(html), 'byte HTML;', 'n0_1', N0_1, 'n1_1', N1_1, 'n0_05', N0_05)
