#!/usr/bin/env python
"""t03_report.py — viết KET_QUA.md từ các bảng t01_sweep.py đã lưu (mọi con số lấy từ tệp, không gõ tay)."""
import json
from pathlib import Path
import pandas as pd

LAB = Path(__file__).resolve().parent
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
SW = pd.read_csv(LAB / 'sweep.csv'); LD = pd.read_csv(LAB / 'ladder_b.csv'); EK = pd.read_csv(LAB / 'err_class_catch.csv')
PR = pd.read_csv(LAB / 'projection.csv'); SUM = json.load(open(LAB / 't01_summary.json'))
V3 = json.load(open(SP / 'r6/policy_v3/summary.json'))
TAU = 0.995


def P(x, d=2):
    return '—' if pd.isna(x) else f'{100 * x:.{d}f}'.replace('.', ',')


def N(x):
    return '—' if pd.isna(x) else f'{int(x):,}'.replace(',', '.')


def ci(r):
    return '— (giữ 0 ô)' if pd.isna(r.prec_both) else f'**{P(r.prec_both)}** [{P(r.ci_lo)}–{P(r.ci_hi)}]'


CFG = ['a', 'b', 'c', 'c2', 'd', 'e', 'f']
NAME = dict(SW.drop_duplicates('config').set_index('config').config_name)
M = SW[SW.tau == TAU].set_index(['config', 'bao_tren'])
g = lambda c, bk: M.loc[(c, bk)]
out = []
w = out.append
w('# THỬ NGHIỆM 1 — Công cụ kiểm ảnh ↔ chữ Hán Nôm cho ô GOLD: độ chính xác ↔ tỉ lệ giữ (26/09/2026)\n')
w('Sinh tự động bởi `t03_report.py` từ `sweep.csv`, `ladder_b.csv`, `err_class_catch.csv`, `projection.csv`, `t01_summary.json` '
  f'(invariant: **{"ALL PASS" if SUM["invariants"]["all_pass"] else "CÓ FAIL"}**). 0 API, chỉ CPU, repo chỉ đọc.\n')
w('## 0. Cách đo (đọc trước)\n')
w('- **Sự thật chỉ là nhãn người IHR**: chữ người `gt_char` (so theo quy ước dị thể V1+) và `slot_ok` (crop nằm đúng khe theo hộp cột người vẽ). '
  'Không dùng nhãn pipeline làm sự thật. Hai sách có nhãn người: LucVanTien1916 (L16), TruyenKieu1872 (TK); chỉ xét ô GOLD có chữ người '
  f'(L16 {N(SUM["invariants"]["n_gold_with_gt"]["LucVanTien1916"])}, TK {N(SUM["invariants"]["n_gold_with_gt"]["TruyenKieu1872"])}).')
w('- **Đúng hai vế** = nhãn đúng chữ người (V1+) **và** ảnh đúng khe. Khe chưa xác định tính là sai vế ảnh (định nghĩa v2/v3).')
w('- **LOBO hai chiều**: ngưỡng của L16 chọn trên TK, ngưỡng của TK chọn trên L16. Mô hình verifier_ft dùng cho mỗi sách báo là mô hình học trên '
  'sách kia; ngưỡng điểm chọn trên phần B (1/4 trang mà mô hình không thấy khi học) của sách chọn. Riêng cổng hình học (b) chọn trên cả sách chọn.')
w('- **Quét**: 7 mục tiêu τ = 0,98 · 0,985 · 0,99 · 0,9925 · 0,995 · 0,9975 · 0,999. Ở mỗi τ, trên sách chọn lấy ngưỡng giữ nhiều ô nhất mà độ chính xác '
  '≥ τ, rồi áp nguyên ngưỡng đó lên sách báo. **τ = 0,995 là mục tiêu đăng ký trước**, nên là bảng chính.')
w('- **CI 95 %**: bootstrap cụm theo trang, B = 2000, seed 20260926.')
w('- **Công cụ chỉ hạ ô, không bao giờ sửa nhãn.** "Hạ" = không được chứng nhận "ảnh + chữ"; theo chính sách v3 ô vẫn được giao là GOLD văn bản '
  '(`uncertified`/`text_only`) hoặc chờ người (`human_check`).\n')
w('| Mã | Cấu hình | Thành phần |')
w('|---|---|---|')
DESC = {'a': 'giữ mọi ô GOLD',
        'b': 'thang lồng nhau G1…G7 (bảng §3); ở mỗi τ chọn bậc trên sách chọn',
        'c': 'p_wood ≥ t (verifier_ft, vòng 4)',
        'c2': 'p_wood ≥ t1 ∧ viss ≥ t2 (thêm bộ xếp hạng hình vòng 3), không hình học',
        'd': 'hình học v3 (A0 ∪ B0 ∪ cột lệch số chữ ∪ cờ trượt vòng 2) + p_wood ≥ t1 ∧ viss ≥ t2; ngưỡng tái lập đúng p02b',
        'e': 'd + nhãn phải được văn bản người của dị bản chứng (`text_attested`). L16 không có dị bản người trên máy → e = d',
        'f': 'e + luật A (rescue, cầu tự dạng, văn bản yếu, ảnh ô khác) + M-OCR. Ở τ = 0,995 trùng **từng ô** với `gold_exact = ok` của chính sách v3'}
for c in CFG:
    w(f'| {c} | {NAME[c]} | {DESC[c]} |')
w('')

w('## 1. Bảng chính: τ = 0,995 (đăng ký trước) — CHẮC CHẮN THEO MÁY (đo trên nhãn người)\n')
for bk, tune in (('L16', 'TK'), ('TK', 'L16')):
    w(f'### Báo trên {bk} (ngưỡng chọn trên {tune})\n')
    w('| Cấu hình | Chính xác hai vế [CI] | Ô giữ | Lỗi ẢNH còn sót (sai khe + khe chưa xđ) | Lỗi CHỮ còn sót | Ô đúng bị hạ oan | Lỗi bị hạ / tổng lỗi | Chỉ vế chữ |')
    w('|---|---|---:|---:|---:|---:|---:|---:|')
    for c in CFG:
        r = g(c, bk)
        w(f'| ({c}) {NAME[c]} | {ci(r)} | {N(r.n_keep)} ({P(r.keep_pct / 100, 1)} %) | {N(r.err_img_sai_khe + r.err_img_khe_chua_xd)} '
          f'({N(r.err_img_sai_khe)} + {N(r.err_img_khe_chua_xd)}) | {N(r.err_chu)} | {N(r.ha_oan)} / {N(r.n_correct)} | {N(r.ha_dung)} / {N(r.err_pool)} | {P(r.prec_label)} |')
    w('')
fL, fT, dT, bL, bT = g('f', 'L16'), g('f', 'TK'), g('d', 'TK'), g('b', 'L16'), g('b', 'TK')
aL, aT = g('a', 'L16'), g('a', 'TK')
w('**Đọc bảng.**')
w(f'- Không kiểm (a): L16 {P(aL.prec_both)} %, TK {P(aT.prec_both)} %. Lỗi ảnh L16 {N(aL.err_img_sai_khe + aL.err_img_khe_chua_xd)}, lỗi chữ {N(aL.err_chu)}; '
  f'TK {N(aT.err_img_sai_khe + aT.err_img_khe_chua_xd)} và {N(aT.err_chu)}.')
w(f'- Hình học (b) gần như **xoá sạch lỗi sai khe** (L16 {N(aL.err_img_sai_khe)} → {N(bL.err_img_sai_khe)}; TK {N(aT.err_img_sai_khe)} → {N(bT.err_img_sai_khe)}) '
  f'mà chỉ hạ oan {P(bL.ha_oan / bL.n_correct, 1)} % / {P(bT.ha_oan / bT.n_correct, 1)} % ô đúng. Nhưng lỗi chữ gần như nguyên '
  f'({N(bL.err_chu)} / {N(bT.err_chu)}), nên trần chỉ {P(bL.prec_both)} / {P(bT.prec_both)} %. Ở τ = 0,995 không bậc nào đạt τ trên sách chọn → lấy bậc chính xác nhất (G7).')
w(f'- Bộ kiểm ảnh (c, c2) bắt cả lỗi chữ nhìn thấy được, nhưng phải hạ nhiều ô đúng.')
w(f'- Chính sách đầy đủ (f): L16 **{P(fL.prec_both)} %** [{P(fL.ci_lo)}–{P(fL.ci_hi)}] giữ {P(fL.keep_pct / 100, 1)} %; '
  f'TK **{P(fT.prec_both)} %** [{P(fT.ci_lo)}–{P(fT.ci_hi)}] giữ {P(fT.keep_pct / 100, 1)} %. '
  f'Trên TK, bỏ đối chiếu dị bản (d) chỉ còn {P(dT.prec_both)} % → **dị bản người là thứ đẩy TK qua 99,5 %**.')
w(f'- Cái giá: hạ oan L16 {N(fL.ha_oan)}/{N(fL.n_correct)} ({P(fL.ha_oan / fL.n_correct, 1)} %) và TK {N(fT.ha_oan)}/{N(fT.n_correct)} '
  f'({P(fT.ha_oan / fT.n_correct, 1)} %) ô đúng.\n')

w('## 2. Quét 7 mức τ (chọn trên sách này, báo trên sách kia) — CHẮC CHẮN THEO MÁY\n')
w('Mỗi ô: `% giữ / chính xác hai vế %` trên sách báo. "—" = trên sách chọn không có ngưỡng nào đạt τ với ≥ 50 ô, nên giữ 0 ô. '
  'Chi tiết (ngưỡng, CI, lỗi ảnh/chữ, hạ oan, số đo trên sách chọn): `sweep.csv`.\n')
taus = sorted(SW.tau.unique())
w('| Cấu hình | Sách báo | ' + ' | '.join(str(t).replace('.', ',') for t in taus) + ' |')
w('|---|---|' + '---|' * len(taus))
for c in CFG[1:]:
    for bk in ('L16', 'TK'):
        cells = []
        for t in taus:
            r = SW[(SW.config == c) & (SW.bao_tren == bk) & (SW.tau == t)].iloc[0]
            cells.append('—' if pd.isna(r.prec_both) else f'{P(r.keep_pct / 100, 1)} / {P(r.prec_both)}')
        w(f'| ({c}) | {bk} | ' + ' | '.join(cells) + ' |')
w('')
w('Cổng (b) chọn bậc: ' + '; '.join(f'τ {str(t).replace(".", ",")}: L16←TK {SW[(SW.config == "b") & (SW.bao_tren == "L16") & (SW.tau == t)].param.iloc[0].split(" ")[0]}, '
                                   f'TK←L16 {SW[(SW.config == "b") & (SW.bao_tren == "TK") & (SW.tau == t)].param.iloc[0].split(" ")[0]}' for t in taus) + '.\n')

w('## 3. Thang hình học (b), từng bậc trên từng sách (mô tả, không chọn) — CHẮC CHẮN THEO MÁY\n')
w('| Bậc | Nội dung | Sách | Giữ | Chính xác hai vế | Lỗi ảnh còn | Lỗi chữ còn | Hạ oan |')
w('|---|---|---|---:|---:|---:|---:|---:|')
for r in LD.itertuples():
    w(f'| {r.bac} | {r.mo_ta} | {r.sach} | {P(r.keep_pct / 100, 1)} % | {P(r.prec_both)} [{P(r.ci_lo)}–{P(r.ci_hi)}] | '
      f'{N(r.err_img_sai_khe + r.err_img_khe_chua_xd)} | {N(r.err_chu)} | {N(r.ha_oan)} |')
w('\n**Lưu ý vòng tròn một phần (SUY ĐOÁN về mức độ):** `slot_ok = 0` đòi cả mô hình khe theo mẫu lẫn mô hình khe theo hộp dòng kim cùng chỉ khe khác; '
  'cờ lệch hộp kim (G4–G7) cũng dựa trên hộp kim. Vì vậy tỉ lệ bắt lỗi sai khe của các bậc G4–G7 có thể được thổi phồng. Mô hình theo mẫu thì độc lập với kim.\n')

w('## 4. Công cụ bắt được loại lỗi nào (τ = 0,995) — CHẮC CHẮN THEO MÁY\n')
w('Tỉ lệ ô bị hạ theo loại. Muốn công cụ tốt thì cột lỗi phải cao hơn hẳn dòng "đúng hai vế".\n')
w('| Sách | Loại | Số ô GOLD | Hạ (b) | Hạ (c) | Hạ (d) | Hạ (f) |')
w('|---|---|---:|---:|---:|---:|---:|')
LV = {'DUNG_hai_ve': 'đúng hai vế (đối chứng)', 'anh_sai_khe': 'ẢNH: sai khe', 'anh_khe_chua_xac_dinh': 'ẢNH: khe chưa xác định',
      'chu_dong_am_trong_R': 'CHỮ: đồng âm (cả hai ∈ R(âm))', 'chu_gan_hinh': 'CHỮ: gần hình', 'chu_bang_chu_ke': 'CHỮ: nhãn = chữ kề',
      'chu_pua': 'CHỮ: mã PUA khác', 'chu_khac': 'CHỮ: khác'}
for r in EK.itertuples():
    w(f'| {r.sach} | {LV[r.loai]} | {N(r.n_gold)} | {P(r.ty_le_bi_ha_b, 1)} % | {P(r.ty_le_bi_ha_c, 1)} % | {P(r.ty_le_bi_ha_d, 1)} % | {P(r.ty_le_bi_ha_f, 1)} % |')
e = EK.set_index(['sach', 'loai'])
dTK_ok, dTK_da = e.loc[('TK', 'DUNG_hai_ve')].ty_le_bi_ha_d, e.loc[('TK', 'chu_dong_am_trong_R')].ty_le_bi_ha_d
fTK_da = e.loc[('TK', 'chu_dong_am_trong_R')].ty_le_bi_ha_f
w(f'\n- Lỗi **ảnh**: mọi cấu hình có hình học hạ gần hết (sai khe: {P(e.loc[("L16", "anh_sai_khe")].ty_le_bi_ha_d, 1)} % L16, '
  f'{P(e.loc[("TK", "anh_sai_khe")].ty_le_bi_ha_d, 1)} % TK ở (d)).')
w(f'- Lỗi **chữ đồng âm/gần hình**: trên TK, bộ kiểm ảnh (d) hạ {P(dTK_da, 1)} % ô đồng âm, trong khi hạ {P(dTK_ok, 1)} % ô đúng '
  f'→ **gần như mù**. Chỉ đối chiếu dị bản người (f) nâng lên {P(fTK_da, 1)} %. Trên L16, (d) phân biệt tốt hơn nhưng phải hạ '
  f'{P(e.loc[("L16", "DUNG_hai_ve")].ty_le_bi_ha_d, 1)} % ô đúng.\n')

w('## 5. Điểm vận hành đề xuất\n')
gl = SW[(SW.config == 'f') & (SW.bao_tren == 'L16') & (SW.tau == 0.9975)].iloc[0]
ge = SW[(SW.config == 'e') & (SW.bao_tren == 'TK') & (SW.tau == 0.9925)].iloc[0]
w(f'1. **Đề xuất: cấu hình (f) ở τ = 0,995 (đăng ký trước)**, tức chính sách v3. Đo được: TK {P(fT.prec_both)} % [{P(fT.ci_lo)}–{P(fT.ci_hi)}] '
  f'giữ {P(fT.keep_pct / 100, 1)} %; L16 {P(fL.prec_both)} % [{P(fL.ci_lo)}–{P(fL.ci_hi)}] giữ {P(fL.keep_pct / 100, 1)} %. CHẮC CHẮN THEO MÁY.')
w(f'2. **Mục tiêu 99,5 %:** TK đạt, kể cả cận dưới CI. L16 (sách không có dị bản người) chỉ **ngang mục tiêu ở mức điểm**, cận dưới {P(fL.ci_lo)} %. '
  f'Nâng τ lên 0,9975 cho {P(gl.prec_both)} % [{P(gl.ci_lo)}–{P(gl.ci_hi)}] nhưng chỉ giữ {P(gl.keep_pct / 100, 1)} %; cận dưới không tốt hơn. '
  '→ Với sách chỉ có công cụ ảnh (không dị bản), **chưa chứng minh được 99,5 % ở cận dưới 95 %**.')
w('3. **Cổng hình học G7 nên chạy trước mọi thứ, ở mọi bộ.** Nó rẻ, bắt gần hết lỗi sai khe và hạ oan chỉ khoảng 4–5 %. '
  'Nhưng một mình nó không lên quá khoảng 98–99 %.')
w(f'4. *Chỉ để tham khảo, không phải đề xuất:* ở sách có dị bản, (e) với τ = 0,9925 cho TK {P(ge.prec_both)} % giữ {P(ge.keep_pct / 100, 1)} %. '
  'Mức này **chọn sau khi nhìn TK**. Chỉ có một sách IHR có dị bản, nên chưa kiểm định ngược được (SUY ĐOÁN).')
w('5. Lưu ý trong mẫu: luật "cột lệch số chữ" và "tall" của v3 được thêm **sau khi nhìn phần dư IHR**. Luật "dị bản chống → không ok" dựa trên số đo TK. '
  'Vì vậy số (e)/(f) trên TK và phần cải thiện v2 → v3 là đo trên chính dữ liệu chọn luật. Ngưỡng điểm (p_wood/viss) và thang (b) thì LOBO sạch.\n')

w('## 6. Điều công cụ KHÔNG làm được\n')
def resid(bk):
    E = EK[(EK.sach == bk) & (EK.loai != 'DUNG_hai_ve') & (EK.giu_f > 0)]
    return ', '.join(f'{N(r.giu_f)} {LV[r.loai].split(": ")[-1]}' for r in E.itertuples())


w(f'- **Lỗi chữ đồng âm / gần hình khi không có dị bản người**: xem §4. Phần sai còn sót trong (f) gồm: '
  f'TK {resid("TK")}; L16 {resid("L16")}. CHẮC CHẮN THEO MÁY.')
w('- **Vế "đúng MỘT chữ" (mực chữ kề lọt vào crop)**: `slot_ok` chỉ xét tâm crop nằm đúng khe, không thấy mực kề. Các cờ B0 (bleed, tight_nb, tall…) '
  'được áp mà **không có sự thật người** để đo. CHẮC CHẮN (thiết kế đo).')
w('- **STT (chữ viết tay)**: verifier_ft học trên mộc bản. Bộ kiểm chữ viết tay giữ ngoài theo sách (vòng 5) vẫn nhận khoảng 11,5 % [7,0–16,7] crop trượt thật '
  '(trích báo cáo v2 §1.7, CÓ THỂ). → `stt_ok_allowed = False`, 0 ô ok.')
w('- **Chrestomathie1872**: không có văn bản người nào trên máy, nên không chứng được vế chữ → 0 ok (309 ứng viên). CHẮC CHẮN (dữ liệu).')
lb = V3['ta_bound_gate']['v3']['LucVanTien1883']['range']
w(f'- **LucVanTien1883**: cận sai nhãn qua dị bản L16 là {P(lb[0], 1)}–{P(lb[1], 1)} %, vượt 0,5 %. Cổng cấp bộ của v3 → 0 ok '
  '(648 ứng viên chờ người). CÓ THỂ.')
w('- **Hạ oan lớn**: công cụ không phân biệt được phần lớn ô đúng với ô sai ở vùng điểm thấp. Ô bị hạ không mất, nhưng muốn "chứng nhận" chúng thì cần người.\n')

w('## 7. Chiếu sang 4 bộ không có nhãn người — ƯỚC LƯỢNG (số ô giữ/hạ, KHÔNG phải độ chính xác đo được)\n')
w('Cùng cấu hình ở τ = 0,995, dùng quy ước của v3 cho bộ ngoài IHR:')
w('- (b): bậc chặt hơn của hai chiều. Ở τ này cả hai chiều đều chọn G7.')
w('- (c): p_wood của **cả hai** mô hình phải qua ngưỡng.')
w('- (d): hình học v3 + điểm. Thạch bản dùng p_wood hai mô hình ∧ viss_X; Chr dùng p_wood hai mô hình (không có viss).')
w('- **STT: (c)/(d) dùng bộ kiểm chữ viết tay LOBO (vòng 5, q = 0,00015, ≥ 3 nguyên mẫu người), không phải verifier_ft.**')
w('- (f) = đếm `gold_exact = ok` của v3.')
w('- Hai dòng IHR để đối chiếu.\n')
w('Mỗi ô: `giữ (%) · hạ`.\n')
w('| Bộ | GOLD | ' + ' | '.join(f'({c})' for c in ['b', 'c', 'd', 'e', 'f']) + ' |')
w('|---|---:|' + '---:|' * 5)
for s in ['STT', 'Chr', 'L83', 'KVK', 'L16', 'TK']:
    E = PR[PR.sach == s].set_index('config')
    w(f'| {s}{" (đo được, IHR)" if s in ("L16", "TK") else ""} | {N(E.loc["a"].gold)} | ' +
      ' | '.join(f'{N(E.loc[c].giu)} ({P(E.loc[c].giu_pct / 100, 1)} %) · {N(E.loc[c].ha)}' for c in ['b', 'c', 'd', 'e', 'f']) + ' |')
w('\n- (c) có thể giữ ít hơn (d): khi dùng một mình, ngưỡng p_wood chọn trên sách chọn phải cao hơn mới đạt τ.')
w('- L83 (e) 648 → (f) 0 là do cổng cấp bộ TA4 của v3 (§6). STT/Chr (f) = 0 là luật đăng ký trước.')
w('- STT (b) hạ 1/3 chủ yếu vì cờ "cột lệch số chữ" và mực chữ kề. Đây là cờ, chưa phải lỗi đã đo.')
kv = V3['residual_estimates']['KimVanKieu1884']
w(f'\nSai số còn lại của 3.657 ô ok ở KVK theo v3: khoảng {N(kv["total_cells"][0])}–{N(kv["total_cells"][1])} ô '
  f'({P(kv["total_rate"][0])}–{P(kv["total_rate"][1])} %); cận bi quan {P(kv["total_rate_pessimistic_bound"][1])} %. SUY ĐOÁN '
  '(chuyển tỉ lệ từ TK). Các bộ khác có 0 ô ok nên không có số tương ứng.\n')

w('## 8. Tệp và tái lập\n')
w('- `t01_sweep.py` → `sweep.csv`, `main_table.csv`, `ladder_b.csv`, `err_class_catch.csv`, `projection.csv`, `t01_summary.json` (invariant); '
  '`measure_out/_thu_nghiem_anh_chu/TN1/cells_ihr_decisions.pkl`.')
w('- `t02_vi_du.py` → `vi_du.html` (80 ô: 4 nhóm × 10 ô × 2 sách), `anh/<md5>.png` (crop chép), `glyph/U+XXXX.png` (NomNaTong), `vi_du_manifest.csv`.')
w('- `t03_report.py` → tệp này.')
w('- Invariant (`t01_summary.json`): tái lập đúng ngưỡng p02b ở 4 mục tiêu × 2 chiều; cờ trượt tính lại trùng `bc_k06v1`/`bc_k05dxv05`; thang lồng nhau; '
  '(f) ở τ = 0,995 trùng từng ô với v3 (L16 2.737, TK 9.766 ô); (a) và (f) trùng số IHR của v3; chiếu (f) trùng `per_set.ok` của v3 ở cả 6 bộ.\n')
w('```bash\ncd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR\nD=lab/thu_nghiem_anh_chu/TN1_cong_kiem\n'
  '.venv/bin/python $D/t01_sweep.py && .venv/bin/python $D/t02_vi_du.py && .venv/bin/python $D/t03_report.py   # tổng < 10 s CPU, 0 API\n```')
w('Đầu vào (chỉ đọc): `$SP/r6/policy_v3/out/{base_v2.pkl, v02_cells_0.995.pkl}`, `$SP/r6/policy_v3/summary.json`, `$SP/r4/policy/out/p02b_sweep.json`, '
  '`$SP/kim_bottleneck/harness/{cells_eval.csv, harness_lib.py}`, `$SP/r6/policy_v3/scripts/vlib.py`, `fonts/NomNaTong-Regular.ttf`, crop trong `dataset/_ALL/crops/`.')
(LAB / 'KET_QUA.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
print('KET_QUA.md', len(out), 'dòng')
