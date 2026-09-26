#!/usr/bin/env python
"""s09_report.py — sinh KET_QUA.md của THỬ NGHIỆM 4 từ các tệp số đã lưu (out_v1/, out_v2/, ../TN3_tat_ca_bo/matrix.csv).
Chỉ cộng/trừ/tỉ lệ trên bảng đã lưu; không mở ảnh. Chạy: .venv/bin/python lab/thu_nghiem_anh_chu/TN4_crop_chuan/s09_report.py"""
import json
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
TN3 = HERE.parent / 'TN3_tat_ca_bo'
SETS8 = ['stt2', 'stt4', 'stt11', 'Chr', 'L83', 'KVK', 'L16', 'TK']
TRUTH = {'stt2': 'SUY ĐOÁN', 'stt4': 'SUY ĐOÁN', 'stt11': 'SUY ĐOÁN', 'Chr': 'SUY ĐOÁN', 'L83': 'ƯỚC LƯỢNG', 'KVK': 'ƯỚC LƯỢNG',
         'L16': 'CHẮC CHẮN THEO MÁY', 'TK': 'CHẮC CHẮN THEO MÁY'}
TYPE = {'stt2': 'chép tay', 'stt4': 'chép tay', 'stt11': 'chép tay', 'Chr': 'in văn xuôi', 'L83': 'thạch bản', 'KVK': 'thạch bản',
        'L16': 'mộc bản IHR', 'TK': 'mộc bản IHR'}
J = lambda p: json.load(open(p))


def vn(x, d=0):
    if x is None or (isinstance(x, float) and x != x):
        return '–'
    s = f'{x:,.{d}f}'
    return s.replace(',', '§').replace('.', ',').replace('§', '.')


def sg(x, d=0):
    return ('+' if x > 0 else '') + vn(x, d)


def pc(x, d=2):
    return vn(100 * x, d) if x == x else '–'


def load(ver):
    O = HERE / f'out_{ver}'
    r = dict(geom=pd.read_csv(O / 'geom_table.csv').set_index('bo'), ihr=pd.read_csv(O / 'geom_ihr.csv').set_index('bo'),
             core=J(O / 'cov_core.json'), borg_geom=J(O / 'geom_borg.json'), s02=J(O / 's02_geom.json'),
             s03=J(O / 's03_vft.json'), s04b=J(O / 's04b.json'), s05=J(O / 's05_hand.json'), s06a=J(O / 's06a.json'),
             mx=pd.read_csv(O / 'tn3m_matrix.csv').set_index(['set', 'dir']), auc=pd.read_csv(O / 'auc_ihr.csv'),
             freeze=J(O / 't00_freeze.json'))
    for f in ('s05_hand_tight', 's10b_hand_retrain', 'boot_sel'):
        if (O / f'{f}.json').exists():
            r[f] = J(O / f'{f}.json')
    if (O / 'auc_ihr_tight.csv').exists():
        r['auc_tight'] = pd.read_csv(O / 'auc_ihr_tight.csv')
    if (O / 'tn3m_matrix_lai.csv').exists():
        r['mx_lai'] = pd.read_csv(O / 'tn3m_matrix_lai.csv').set_index(['set', 'dir'])
    return r


def main():
    V1, V2 = load('v1'), load('v2')
    MO = pd.read_csv(TN3 / 'matrix.csv').set_index(['set', 'dir'])
    M1, M2, ML = V1['mx'], V2['mx'], V2.get('mx_lai')
    g1, g2, ih1, ih2, c1, c2 = V1['geom'], V2['geom'], V1['ihr'], V2['ihr'], V1['core'], V2['core']
    tot = lambda g, c: int(g[c].sum())
    RT = V2.get('s10b_hand_retrain')
    BS = V2.get('boot_sel')
    L = []
    w = L.append
    w('# THỬ NGHIỆM 4 — Crop chuẩn tự động cho ô GOLD (26/09/2026)\n')
    w('Sinh tự động bởi `s09_report.py` từ các tệp số trong `out_v1/`, `out_v2/` và `../TN3_tat_ca_bo/matrix.csv`. 0 lần gọi API, không tải '
      'model, không mở ảnh bằng LLM (ảnh chỉ do chương trình đọc). Repo chỉ đọc ngoài `lab/thu_nghiem_anh_chu/TN4_crop_chuan/` và '
      '`measure_out/_thu_nghiem_anh_chu/TN4/`. Ví dụ ảnh: `vi_du.html` (mở bằng file://, ảnh chép trong `anh/`).\n')
    w('Mục tiêu GOLD: (1) crop là đúng MỘT chữ, căn chuẩn theo chữ; (2) chữ đó đúng là nhãn. TN4 chỉ đụng vế (1): thay crop cũ (bbox + pad 0,12 '
      '+ carve + tighten) bằng **crop chuẩn** tính lại từ ảnh trang, rồi đo (a) crop có sạch/đúng chữ hơn không, (b) bộ kiểm ảnh có giữ được '
      'nhiều ô hơn ở cùng độ chính xác không. Không sửa nhãn nào.\n')
    w('Mức chắc: **CHẮC CHẮN THEO MÁY** = đo trên nhãn/khe người IHR (L16, TK) hoặc nhãn người Borg · **ƯỚC LƯỢNG** = bộ ước lượng TN3 cho thạch '
      'bản · **SUY ĐOÁN** = Chr, STT (chuyển tỉ lệ từ sách khác, cần người kiểm). v2 là **hậu kiểm** (xem §1).\n')
    # ================================================================== 0
    w('## 0. Trả lời ngắn\n')
    cont = lambda ih, s, k: ih.loc[s, k]
    w(f"1. **Hai phiên bản.** v1 = thiết kế đăng ký trước (md5 `{V1['freeze']['cclib_md5'][:10]}`). v1 có ba lỗi chẩn đoán được bằng số (§2.2) nên "
      f"viết **v2 hậu kiểm** (md5 `{V2['freeze']['cclib_md5'][:10]}`): đưa thuật toán về đúng đặc tả (ranh giới = trung điểm tâm; chỉ thành phần vắt qua "
      "mới cắt seam; bóc nét kẻ), **không dò tham số**. Số của v2 phải đọc là hậu kiểm (có thể lạc quan).")
    w(f"2. **(a) Crop sạch hơn thật, không trượt khe — CHẮC CHẮN THEO MÁY (khe người IHR).** Mực nằm ngoài khe người: L16 "
      f"{pc(cont(ih2,'L16','muc_ngoai_khe_cu'))} % → {pc(cont(ih2,'L16','muc_ngoai_khe_moi'))} %, TK {pc(cont(ih2,'TK','muc_ngoai_khe_cu'))} % → "
      f"{pc(cont(ih2,'TK','muc_ngoai_khe_moi'))} %; ô có > 25 % mực ngoài khe: L16 {vn(cont(ih2,'L16','ngoai_khe_gt25_cu'))} → "
      f"{vn(cont(ih2,'L16','ngoai_khe_gt25_moi'))}, TK {vn(cont(ih2,'TK','ngoai_khe_gt25_cu'))} → {vn(cont(ih2,'TK','ngoai_khe_gt25_moi'))}. "
      f"Khe đổi 1 → ≠1: L16 {vn(ih2.loc['L16','hong'])} ô ({pc(ih2.loc['L16','hong_pct']/100)} %), TK {vn(ih2.loc['TK','hong'])}; sửa được khe "
      f"{vn(ih2.loc['L16','sua_khe'])}/{vn(ih2.loc['TK','sua_khe'])}. Cái giá: L16 {vn(c2['L16']['core']['giam_gt25'])} ô "
      f"({pc(c2['L16']['core']['giam_gt25']/c2['L16']['core']['n'])} %) mất > 25 % mực LÕI khe người (nguyên nhân SUY ĐOÁN: chữ nhỏ dính chữ kề, luật đầu nét lạ bỏ nhầm), TK "
      f"{vn(c2['TK']['core']['giam_gt25'])}.")
    w(f"3. **Cờ \"một chữ\" trên 8 bộ (118.954 ô).** Ô sạch một chữ: cũ {vn(tot(g2,'sach_cu'))} → v1 {vn(tot(g1,'sach_moi'))} → v2 "
      f"{vn(tot(g2,'sach_moi'))}. v2 **cứu** {vn(tot(g2,'cuu'))} ô B0 cũ thành crop sạch (chủ yếu bleed chép tay: stt4 "
      f"{vn(g2.loc['stt4','cuu'])}, stt11 {vn(g2.loc['stt11','cuu'])}) và **gắn cờ mới** {vn(tot(g2,'co_moi'))} ô mà crop cũ không cờ "
      f"(tổng cờ của crop chuẩn trên 8 bộ: hai chữ {vn(tot(g2,'two_moi'))}, cắt nét {vn(tot(g2,'cut_moi'))}, tall {vn(tot(g2,'tall_moi'))}, "
      f"stray {vn(tot(g2,'bleed_moi'))}). "
      f"Thạch bản mất ròng (L83 {sg(g2.loc['L83','sach_moi']-g2.loc['L83','sach_cu'])}, KVK {sg(g2.loc['KVK','sach_moi']-g2.loc['KVK','sach_cu'])}).")
    h = lambda M, s, d='H4': M.loc[(s, d)]
    bs_txt = ''
    if BS:
        bs_txt = (f" Nhưng chỉ riêng việc CHỌN NGƯỠNG (bootstrap trang tune, B = {BS['B']}) đã làm số ô H4 dao động L16 "
                  f"{vn(BS['LucVanTien1916']['cu']['giu_p5'])}–{vn(BS['LucVanTien1916']['cu']['giu_p95'])}, TK "
                  f"{vn(BS['TruyenKieu1872']['cu']['giu_p5'])}–{vn(BS['TruyenKieu1872']['cu']['giu_p95'])}; hiệu v2 − cũ ghép cặp: L16 trung vị "
                  f"{sg(BS['LucVanTien1916']['v2_tru_cu']['trung_vi'])} [{sg(BS['LucVanTien1916']['v2_tru_cu']['p5'])}; "
                  f"{sg(BS['LucVanTien1916']['v2_tru_cu']['p95'])}], TK {sg(BS['TruyenKieu1872']['v2_tru_cu']['trung_vi'])} "
                  f"[{sg(BS['TruyenKieu1872']['v2_tru_cu']['p5'])}; {sg(BS['TruyenKieu1872']['v2_tru_cu']['p95'])}] → **không phân biệt được với 0**.")
    w(f"4. **(b) Bộ kiểm ảnh KHÔNG hưởng lợi khi chỉ thay ảnh đầu vào** (bộ kiểm học trên crop cũ). AUC hai vế trên IHR giảm nhẹ (L16 p_wood "
      f"{vn(V2['auc'].set_index(['bo','tap','diem','viec']).loc[('L16','GOLD','pw','hai_ve')].hieu,4)}). H4 đo được (τ = 0,995, LOBO): "
      f"L16 cũ {vn(h(MO,'L16').keep)} ô {pc(h(MO,'L16').both_pt)} % → v2 {vn(h(M2,'L16').keep)} ô {pc(h(M2,'L16').both_pt)} % "
      f"[{pc(h(M2,'L16').both_lo)}–{pc(h(M2,'L16').both_hi)}]; TK cũ {vn(h(MO,'TK').keep)} ô {pc(h(MO,'TK').both_pt)} % → v2 "
      f"{vn(h(M2,'TK').keep)} ô {pc(h(M2,'TK').both_pt)} % [{pc(h(M2,'TK').both_lo)}–{pc(h(M2,'TK').both_hi)}] (v1: {vn(h(M1,'L16').keep)} / "
      f"{vn(h(M1,'TK').keep)})." + bs_txt + f" Thạch bản (ƯỚC LƯỢNG): KVK {vn(h(MO,'KVK').keep)} → {vn(h(M2,'KVK').keep)}, L83 "
      f"{vn(h(MO,'L83').keep)} → {vn(h(M2,'L83').keep)}. Chép tay/Chr (SUY ĐOÁN) **giảm mạnh**: STT "
      f"{vn(sum(h(MO,s).keep for s in ('stt2','stt4','stt11')))} → {vn(sum(h(M2,s).keep for s in ('stt2','stt4','stt11')))}, Chr "
      f"{vn(h(MO,'Chr').keep)} → {vn(h(M2,'Chr').keep)} — bộ kiểm viết tay nhận ít dương hơn hẳn (Borg DungLy, cùng FAR: "
      f"{pc(V2['s05']['borg_joint']['keep_v5|cu']['accept_pos'])} % → {pc(V2['s05']['borg_joint']['keep_v5|moi']['accept_pos'])} %).")
    if RT:
        w(f"5. **Học lại bộ kiểm viết tay trên crop chuẩn (hậu kiểm)**: Borg DungLy nhận dương {pc(RT['borg_joint']['hoc_lai']['keep_v5']['accept_pos'])} % "
          f"(FAR đồng âm {pc(RT['borg_joint']['hoc_lai']['keep_v5']['far_hom'],3)} %, trượt {pc(RT['borg_joint']['hoc_lai']['keep_v5']['far_slip'],3)} %) "
          f"so với cũ/cũ {pc(RT['borg_joint']['cu']['keep_v5']['accept_pos'])} %; STT được chứng nhận {pc(RT['stt_cert_rate']['cu'])} % (cũ) · "
          f"{pc(RT['stt_cert_rate']['dropin'])} % (thay ảnh) · {pc(RT['stt_cert_rate']['hoc_lai'])} % (học lại). "
          + ("Học lại **không phục hồi** tỉ lệ nhận ở điểm làm việc nghiêm (AUC dương/đồng âm "
             f"{vn(RT['borg_joint']['hoc_lai']['keep_v5']['auc_min_pos_vs_hom'],4)} so với cũ {vn(RT['borg_joint']['cu']['keep_v5']['auc_min_pos_vs_hom'],4)}, "
             f"dương/trượt {vn(RT['borg_joint']['hoc_lai']['keep_v5']['auc_min_pos_vs_slip'],4)} so với {vn(RT['borg_joint']['cu']['keep_v5']['auc_min_pos_vs_slip'],4)}): "
             "với chữ viết tay, crop chuẩn không giúp bộ kiểm. (Một lần học, một hạt giống; ngưỡng q = 0,00015 do vài âm cực trị quyết định — số nhận dương rất nhạy.)"
             if RT['borg_joint']['hoc_lai']['keep_v5']['accept_pos'] < RT['borg_joint']['cu']['keep_v5']['accept_pos'] - 0.02 else
             "Học lại phục hồi tỉ lệ nhận.") + " Xem §4.3.")
    if ML is not None:
        w(f"6. **Cấu hình \"lai\" (hậu kiểm, dùng được ngay)**: giao crop chuẩn v2 + cổng H1 dùng cờ một chữ của crop chuẩn + bộ kiểm hiện tại chấm "
          f"crop CŨ với ngưỡng TN1. H4: STT {vn(sum(h(MO,s).keep for s in ('stt2','stt4','stt11')))} → "
          f"{vn(sum(h(ML,s).keep for s in ('stt2','stt4','stt11')))}, Chr {vn(h(MO,'Chr').keep)} → {vn(h(ML,'Chr').keep)}, L16 "
          f"{vn(h(MO,'L16').keep)} → {vn(h(ML,'L16').keep)} ({pc(h(ML,'L16').both_pt)} %, đo với khe crop mới), TK {vn(h(MO,'TK').keep)} → "
          f"{vn(h(ML,'TK').keep)} ({pc(h(ML,'TK').both_pt)} %), KVK {vn(h(MO,'KVK').keep)} → {vn(h(ML,'KVK').keep)}, L83 {vn(h(MO,'L83').keep)} → "
          f"{vn(h(ML,'L83').keep)}; độ chính xác không đổi (trong khoảng).")
    w("7. **Khuyến nghị** (§7): đưa crop chuẩn v2 vào pipeline dưới dạng **tệp phụ + cờ**, dùng cờ một chữ của nó trong cổng hình học (cấu hình "
      "\"lai\"); **không** cho bộ kiểm hiện tại chấm crop chuẩn (thay ảnh không lợi; học lại bộ viết tay cũng không phục hồi); người xem mẫu ô L16 bị cắt "
      "nét trước khi thay crop giao nộp; không đổi kết luận TN3 (chỉ TK đạt mục tiêu chính xác).\n")
    # ================================================================== 1
    w('## 1. Thuật toán crop chuẩn\n')
    w('Mã: `cclib.py` (v1, đăng ký trước, đóng băng trước lượt đo) và `cclib_v2.py` (v2, viết SAU khi thấy kết quả hình học v1; đóng băng trước lượt '
      'đo v2; không dò tham số). Đơn vị: bước chữ `p` = trung vị khoảng cách tâm các hộp trong cột; bề ngang cột `w` = trung vị bề ngang 5 hộp quanh ô. '
      'Không tham số nào chọn trên IHR hay bằng bộ kiểm.\n')
    w('| Bước | v1 (đăng ký trước) | v2 (hậu kiểm) |\n|---|---|---|')
    w('| Láng giềng | hộp cùng cột (bản ghi build mọi tầng), bỏ hộp trùng khe (\\|Δcy\\| < 0,25·h) | như v1 |')
    w('| Dải khe dọc | ranh giới = SEAM ít mực trong trung điểm ± 0,25·d (hoà → hàng trên cùng); thiếu ô kề: cy ∓ 0,8·max(p,h) | ranh giới = ĐƯỜNG THẲNG qua trung điểm hai tâm; thiếu ô kề: trung điểm ảo 0,55·max(p,h); trung điểm xa hơn 0,8·max(p,h) bị kẹp |')
    w('| Dải khe ngang | xc ± 0,75·w, cắt ở trung điểm với tâm cột kề | như v1 |')
    w('| Nhị phân cục bộ | Otsu trên cửa sổ (dải + 0,30·p / 0,30·w), kẹp [64, 200] | như v1 (nới 0,35) |')
    w('| Nét kẻ | thành phần dài ≥ 1,2·p và mảnh ≤ 0,2·w (hoặc ngược lại) | BÓC trước: mở hình thái nhân dọc 1,2·p và ngang 1,2·w |')
    w('| Gán thành phần | trọn trong seam → của ô; vắt qua → giữ phần giữa hai seam; tâm x ngoài dải → cột khác | trọn một phía TRUNG ĐIỂM → gán trọn; chỉ thành phần vắt qua trung điểm mới cắt seam (trung điểm ± 0,25·d, phạt khoảng cách ≤ 20/255) |')
    w('| Đầu nét lạ / đốm | phần giữ < 20 % thành phần và < 10 % mực ô → bỏ; đốm < max(3 px, 0,0015·p·w) | như v1 |')
    w('| Kiểm "một chữ" (cờ, không ép) | blank < 0,02·p·w; cut = ranh giới cắt qua mực > 30 % số cột hoặc mực chạm mép cửa sổ; two = cao > 1,35·p (hoặc rộng > 1,35·w) có khe trắng ≥ 0,10·p chia 20/20; ink = mực/(p·w) ngoài [0,4; 2,5] × trung vị cùng (bộ, nhãn) (lớp < 5 ô: [0,25; 4] × trung vị bộ); cộng stray/border của pipeline đo trên crop chặt MỚI (bleed/truncated) | như v1 |')
    w('| Xuất | khung VUÔNG cạnh L + 2·ceil(0,10·L) quanh hộp mực, điểm ảnh từ trang GỐC (5 sách crop_source original; STT: trang đã xử lý như pipeline); mực ngoại lai (nở 1 px) và ngoài trang tô màu giấy (trung vị điểm không-mực); + bản 128×128 | như v1 |')
    w('\nĐịnh nghĩa phân tích (đăng ký trước trong `out_v1/t00_freeze.json`, dùng NGUYÊN cho v2):\n')
    for k, v in V1['freeze']['prereg'].items():
        w(f'- `{k}`: {v}')
    w(f"\nChạy thử trước khi đóng băng v1 (3 trang/bộ, chỉ để bắt lỗi chương trình): {V1['freeze']['smoke_test']}\n")
    w('Crop mới ghi ra `measure_out/_thu_nghiem_anh_chu/TN4/<v1|v2>/crops/<cell_uid>.png` (khung vuông, độ phân giải trang) và `crops128/` (128×128); '
      'ô không phải GOLD của 4 sách harness ở `aux/`, Borg ở `aux_borg/`. Không ghi gì vào `dataset/`.\n')
    # ================================================================== 2
    w('## 2. Crop cũ ↔ crop chuẩn: cờ "một chữ" trên 8 bộ\n')
    w('Crop cũ đo lại bằng chính luật `save_crop` dựng lại trên trang (md5 byte trùng tệp giao ở mẫu 150 ô/bộ — 24 ô lệch đều là ô rescue 64×64 của STT; '
      'cờ pipeline tính lại trùng 100 % `crop_quality_flag`/`seg_flag`). Sạch cũ = không cờ B0 cấp crop (bleed/truncated/blank/tight_nb/tall); sạch mới = '
      '`one_char_ok` ∧ ¬tall. Hỏng: chỉ đo được ở L16/TK (§3).\n')
    w('| Bộ | Loại | GOLD | sạch cũ | sạch v1 | sạch v2 | B0 cũ | cứu v2 | cờ mới v2 | bleed cũ→v2 | tall cũ→v2 | hai chữ v2 | cắt nét v2 | stray>0,08 cũ→v2 | hỏng v2 (đo) |')
    w('|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|')
    for s in SETS8:
        a, b = g1.loc[s], g2.loc[s]
        hong = (f"khe {vn(ih2.loc[s,'hong'])} · lõi {vn(c2[s]['core']['giam_gt25'])}" if s in ('L16', 'TK') else '–')
        w(f"| {s} | {TYPE[s]} | {vn(b.n)} | {vn(b.sach_cu)} | {vn(a.sach_moi)} | {vn(b.sach_moi)} | {vn(b.B0_cu)} | {vn(b.cuu)} | {vn(b.co_moi)} | "
          f"{vn(b.bleed_cu)}→{vn(b.bleed_moi)} | {vn(b.tall_cu)}→{vn(b.tall_moi)} | {vn(b.two_moi)} | {vn(b.cut_moi)} | "
          f"{vn(b.stray_cu_gt008)}→{vn(b.stray_moi_gt008)} | {hong} |")
    w(f"| **Tổng** | | {vn(tot(g2,'n'))} | {vn(tot(g2,'sach_cu'))} | {vn(tot(g1,'sach_moi'))} | {vn(tot(g2,'sach_moi'))} | {vn(tot(g2,'B0_cu'))} | "
      f"{vn(tot(g2,'cuu'))} | {vn(tot(g2,'co_moi'))} | {vn(tot(g2,'bleed_cu'))}→{vn(tot(g2,'bleed_moi'))} | {vn(tot(g2,'tall_cu'))}→{vn(tot(g2,'tall_moi'))} | "
      f"{vn(tot(g2,'two_moi'))} | {vn(tot(g2,'cut_moi'))} | {vn(tot(g2,'stray_cu_gt008'))}→{vn(tot(g2,'stray_moi_gt008'))} | |")
    w('\n"cứu" = B0 cũ ∧ crop mới sạch; "cờ mới" = cũ sạch ∧ crop mới bị cờ — cờ mới KHÔNG phải lỗi chắc chắn (kiểm "hai chữ" bắt hộp gộp hai chữ mà '
      'pipeline không cờ; "cắt nét" bắt chữ chạm chữ kề). bleed mới = stray_ink của pipeline đo trên crop chặt mới (mảnh mực tách rời sát mép).\n')
    w('### 2.1 Borg (chữ viết tay, nhãn người) — crop chuẩn trên ô keep ∨ keep_high\n')
    w('| Tập | n | sạch một chữ v1 | sạch v2 | cắt nét v1→v2 | hai chữ v1→v2 | tall v1→v2 | stray v1→v2 |\n|---|---:|---:|---:|---|---|---|---|')
    for k in ('keep_v5=1', 'keep_v5=0'):
        a, b = V1['borg_geom'][k], V2['borg_geom'][k]
        w(f"| {k} | {vn(b['n'])} | {vn(a['clean_new'])} ({pc(a['clean_new']/a['n'],1)} %) | {vn(b['clean_new'])} ({pc(b['clean_new']/b['n'],1)} %) | "
          f"{vn(a['cut'])}→{vn(b['cut'])} | {vn(a['two'])}→{vn(b['two'])} | {vn(a['tall'])}→{vn(b['tall'])} | {vn(a['bleed_new'])}→{vn(b['bleed_new'])} |")
    w('\n### 2.2 Vì sao có v2 (chẩn đoán v1 bằng số, không mở ảnh)\n')
    w(f"- **Nét kẻ dính chữ**: L16 v1 gắn cờ cắt nét {vn(g1.loc['L16','cut_moi'])} ô, TK {vn(g1.loc['TK','cut_moi'])}; chẩn đoán 56 ô ở 6 trang L16: 100 % là "
      "\"mực ô chạm mép TRÁI/PHẢI cửa sổ\" — nét kẻ cột/khung dính chữ thành một thành phần lưới mà luật \"thành phần mảnh\" của v1 không nhận ra. "
      f"v2 bóc nét kẻ bằng mở hình thái → cắt nét L16 {vn(g2.loc['L16','cut_moi'])}, TK {vn(g2.loc['TK','cut_moi'])}.")
    w(f"- **Ranh giới bám chữ kề**: KVK v1 làm stray_ink > 0,08 tăng {vn(g1.loc['KVK','stray_cu_gt008'])} → {vn(g1.loc['KVK','stray_moi_gt008'])} "
      "(mảnh chữ kề lọt vào): v1 lấy chính seam làm ranh giới, khi hoà năng lượng chọn hàng TRÊN CÙNG → ranh giới trên bám sát chữ kề; chữ đầu cột lấy "
      f"tới 0,8·max(p,h). v2 (ranh giới thẳng ở trung điểm, trung điểm ảo 0,55) → {vn(g2.loc['KVK','stray_moi_gt008'])}.")
    w(f"- **Ranh giới dưới cắt chữ ⿱**: cùng quy tắc hoà, ranh giới DƯỚI bám sát tâm ô (khe trắng bên trong chữ trên-dưới) → L16 v1 "
      f"{vn(c1['L16']['core']['giam_gt25'])} ô mất > 25 % mực lõi khe; v2 {vn(c2['L16']['core']['giam_gt25'])}.\n")
    # ================================================================== 3
    w('## 3. Còn đúng khe không, sạch hơn không — khe người IHR (CHẮC CHẮN THEO MÁY)\n')
    w('Khe của một crop = hai mô hình khe của harness (mẫu t + dòng kim, chép nguyên `h01_build_cells.py`) áp cho tâm crop (cũ: tâm bbox — tái lập 100 % '
      '`slot_ok` harness; mới: tâm hộp mực). Khe người để đo mực = hộp cột NGƯỜI vẽ × vị trí mẫu của khe; mực = xám < ngưỡng Otsu cục bộ của ô (cùng '
      'ngưỡng cho crop cũ và mới). "Lõi" = bỏ 15 % trên/dưới khe và 10 % hai bên cột (độ nhạy thêm sau khi thấy mép khe mẫu lệch và có nét kẻ). Đo mực '
      'trên ô GOLD có khe cũ = 1. Cột "cũ" lấy từ lượt đo v2 (ngưỡng Otsu cục bộ tính theo cửa sổ của v2; lượt v1 cho số "cũ" lệch ≤ 0,1 điểm %).\n')
    w('| Chỉ số | L16 cũ | L16 v1 | L16 v2 | TK cũ | TK v1 | TK v2 |\n|---|---:|---:|---:|---:|---:|---:|')
    for nm, a_, b_ in (('khe = 1', 'khe_cu_1', 'khe_moi_1'), ('khe = 0 (sai khe)', 'khe_cu_0', 'khe_moi_0'), ('khe chưa xác định', 'khe_cu_chua', 'khe_moi_chua')):
        w(f"| {nm} | {vn(ih2.loc['L16',a_])} | {vn(ih1.loc['L16',b_])} | {vn(ih2.loc['L16',b_])} | {vn(ih2.loc['TK',a_])} | {vn(ih1.loc['TK',b_])} | {vn(ih2.loc['TK',b_])} |")
    w(f"| **làm hỏng khe** (1 → ≠1) | | {vn(ih1.loc['L16','hong'])} | {vn(ih2.loc['L16','hong'])} | | {vn(ih1.loc['TK','hong'])} | {vn(ih2.loc['TK','hong'])} |")
    w(f"| sửa được khe (≠1 → 1) | | {vn(ih1.loc['L16','sua_khe'])} | {vn(ih2.loc['L16','sua_khe'])} | | {vn(ih1.loc['TK','sua_khe'])} | {vn(ih2.loc['TK','sua_khe'])} |")
    for nm, a_, b_, d in (('mực ngoài khe người (TB, %)', 'muc_ngoai_khe_cu', 'muc_ngoai_khe_moi', 'p'),
                          ('ô > 10 % mực ngoài khe', 'ngoai_khe_gt10_cu', 'ngoai_khe_gt10_moi', 'n'),
                          ('ô > 25 % mực ngoài khe', 'ngoai_khe_gt25_cu', 'ngoai_khe_gt25_moi', 'n'),
                          ('ô B0 cũ: mực ngoài khe (TB, %)', 'B0cu_ngoai_khe_cu', 'B0cu_ngoai_khe_moi', 'p')):
        f = pc if d == 'p' else vn
        w(f"| {nm} | {f(ih2.loc['L16',a_])} | {f(ih1.loc['L16',b_])} | {f(ih2.loc['L16',b_])} | {f(ih2.loc['TK',a_])} | {f(ih1.loc['TK',b_])} | {f(ih2.loc['TK',b_])} |")
    for nm, k in (('độ phủ mực khe — đầy đủ (TB, %)', 'full'), ('độ phủ mực khe — lõi (TB, %)', 'core')):
        w(f"| {nm} | {pc(c2['L16'][k]['phu_cu'])} | {pc(c1['L16'][k]['phu_moi'])} | {pc(c2['L16'][k]['phu_moi'])} | {pc(c2['TK'][k]['phu_cu'])} | {pc(c1['TK'][k]['phu_moi'])} | {pc(c2['TK'][k]['phu_moi'])} |")
    for nm, k in (('ô phủ < 80 % — đầy đủ', 'full'), ('ô phủ < 80 % — lõi', 'core')):
        w(f"| {nm} | {vn(c2['L16'][k]['lt80_cu'])} | {vn(c1['L16'][k]['lt80_moi'])} | {vn(c2['L16'][k]['lt80_moi'])} | {vn(c2['TK'][k]['lt80_cu'])} | {vn(c1['TK'][k]['lt80_moi'])} | {vn(c2['TK'][k]['lt80_moi'])} |")
    for nm, k in (('**mất > 25 % mực khe** — đầy đủ', 'full'), ('**mất > 25 % mực khe — lõi** (= cắt vào chữ)', 'core')):
        w(f"| {nm} | | {vn(c1['L16'][k]['giam_gt25'])} | {vn(c2['L16'][k]['giam_gt25'])} | | {vn(c1['TK'][k]['giam_gt25'])} | {vn(c2['TK'][k]['giam_gt25'])} |")
    r16 = ih2.loc['L16', 'ngoai_khe_gt25_moi'] / ih2.loc['L16', 'ngoai_khe_gt25_cu']; rtk = ih2.loc['TK', 'ngoai_khe_gt25_moi'] / ih2.loc['TK', 'ngoai_khe_gt25_cu']
    w(f"\nĐọc: v2 giảm số ô có > 25 % mực lạ còn {pc(r16,0)} % (L16) và {pc(rtk,0)} % (TK) so với crop cũ, ô B0 cũ giảm mực lạ từ "
      f"{pc(ih2.loc['L16','B0cu_ngoai_khe_cu'])} % xuống {pc(ih2.loc['L16','B0cu_ngoai_khe_moi'])} % (L16), mà khe gần như không đổi. Mất mực ở mép khe mẫu "
      "phần lớn là mực chữ kề/nét kẻ rơi vào khe mẫu (độ phủ LÕI gần như giữ). Phần còn lại ở L16 (chữ ~33 px, nét mảnh dính chữ kề, luật \"đầu nét lạ\" "
      "bỏ nhầm nét của chính chữ) là **ô bị làm hỏng thật mà không cờ nào bắt**. TK (chữ tách rời rõ) gần như không hỏng.\n")
    # ================================================================== 4
    w('## 4. Bộ kiểm ảnh trước ↔ sau\n')
    w('Bộ kiểm verifier_ft (CNN + nguyên mẫu + logistic "wood") và kênh viss (xếp hạng lại r01–r03, CHẠY LẠI với view A = crop chuẩn cho mọi ô 4 sách, '
      'nguyên mẫu "oth"/"self" từ crop chuẩn GOLD) được **học/hiệu chuẩn trên crop CŨ** và giữ nguyên ("thay ảnh", drop-in). Invariant: logistic tái lập '
      f"đúng p_wood cũ (sai tối đa {vn(V2['s03']['p_wood_T_old_reproduce_maxabs'],1)}), nhúng lại ảnh cũ trùng (cos ≥ {vn(V2['s03']['reembed_ihr_T_cos_min'],4)}; "
      f"encoder v1+v2 cos ≥ {vn(V2['s04b']['reembed_old_cos_min'],6)}); cấu trúc ứng viên r02 trùng lượt gốc (1.919.412 hàng).\n")
    w('### 4.1 AUC trên IHR (LOBO: L16 ← mô hình T, TK ← mô hình L), ô GOLD — CHẮC CHẮN THEO MÁY\n')
    w('"khe" = tách crop đúng/sai khe (khe của chính crop được chấm); "nhãn" = tách nhãn đúng/sai (ô khe = 1); "hai vế" = cả hai. Hiệu = v2 − cũ, '
      'CI 95 % bootstrap cụm trang (B = 500, ghép cặp).\n')
    w('| Bộ | Điểm | Việc | AUC cũ | AUC v1 | AUC v2 | hiệu v2 [CI] | v2 khung chặt |\n|---|---|---|---:|---:|---:|---|---:|')
    a1 = V1['auc'][V1['auc'].tap == 'GOLD'].set_index(['bo', 'diem', 'viec'])
    a2 = V2['auc'][V2['auc'].tap == 'GOLD'].set_index(['bo', 'diem', 'viec'])
    at = V2['auc_tight'][V2['auc_tight'].tap == 'GOLD'].set_index(['bo', 'diem', 'viec']) if 'auc_tight' in V2 else None
    for (bo, dm, vc), r in a2.iterrows():
        tt = vn(at.loc[(bo, dm, vc)].auc_moi, 4) if (at is not None and dm == 'pw') else '–'
        w(f"| {bo} | {'p_wood' if dm == 'pw' else 'viss'} | {vc} | {vn(r.auc_cu,4)} | {vn(a1.loc[(bo,dm,vc)].auc_moi,4)} | {vn(r.auc_moi,4)} | "
          f"{sg(r.hieu,4)} [{sg(r.hieu_lo,4)}; {sg(r.hieu_hi,4)}] | {tt} |")
    w('\n"v2 khung chặt" = độ nhạy hậu kiểm: cùng crop v2 nhưng đưa vào verifier_ft theo khung hộp mực + 4 px (giống khung crop cũ). Không khác khung vuông '
      '→ AUC giảm KHÔNG do khung/tỉ lệ; nhiều khả năng do nội dung crop khác kiểu ảnh bộ kiểm đã học (mực chữ kề bị xoá, nét mảnh bị cắt ở L16) — SUY ĐOÁN.\n')
    w('### 4.2 Bộ kiểm chữ viết tay LOBO-sách trên Borg DungLy giữ ngoài (nhãn người) — CHẮC CHẮN THEO MÁY\n')
    w('Mô hình/nguyên mẫu/hiệu chuẩn chỉ từ Borg Kinh; ngưỡng q = 0,00015 (chính sách v3); cặp dương (crop, chữ người) / âm đồng âm / âm trượt '
      '(chữ kề); chỉ ô DungLy có crop chuẩn v2.\n')
    w('| Tập | Crop · bộ kiểm | dương | nhận dương | FAR đồng âm | FAR trượt | AUC dương/trượt | AUC dương/đồng âm |\n|---|---|---:|---:|---:|---:|---:|---:|')
    for sub in ('keep_v5', 'keep_or_high'):
        rows = [('cũ · cũ', V2['s05']['borg_joint'][f'{sub}|cu']), ('v1 · cũ (thay ảnh)', V1['s05']['borg_joint'][f'{sub}|moi']),
                ('v2 · cũ (thay ảnh)', V2['s05']['borg_joint'][f'{sub}|moi'])]
        if 's05_hand_tight' in V2:
            rows.append(('v2 khung chặt · cũ', V2['s05_hand_tight']['borg_joint'][f'{sub}|moi']))
        if RT:
            rows.append(('**v2 · HỌC LẠI trên v2**', RT['borg_joint']['hoc_lai'][sub]))
        for nm, jj in rows:
            w(f"| {sub} | {nm} | {vn(jj['n_pos'])} | {pc(jj['accept_pos'])} % | {pc(jj['far_hom'],3)} % | {pc(jj['far_slip'],3)} % | "
              f"{vn(jj['auc_min_pos_vs_slip'],4)} | {vn(jj['auc_min_pos_vs_hom'],4)} |")
    w(f"\nSTT (52.707 ô GOLD) được chứng nhận ở q = 0,00015: cũ {pc(V1['s05']['stt_cert_rate']['old'])} % → v1 thay ảnh {pc(V1['s05']['stt_cert_rate']['new'])} % "
      f"→ v2 thay ảnh {pc(V2['s05']['stt_cert_rate']['new'])} %" + (f" → v2 học lại {pc(RT['stt_cert_rate']['hoc_lai'])} %" if RT else '') + '.\n')
    if RT:
        w('### 4.3 Học lại bộ kiểm viết tay trên crop chuẩn v2 (hậu kiểm)\n')
        w('`h01_train_tn4.py` = bản chép `r5/hand_lobo/h01_train.py` (chỉ đổi thư mục), cùng hạt giống/bước (2.800)/tập học (Borg Kinh keep fold 0–3 + '
          'IHR sách tune phần A), ảnh = crop chuẩn v2 (ô Borg không có crop chuẩn giữ ảnh cũ, không vào tập học). `s10b_hand_retrain_eval.py` chép đặc '
          'trưng (s01/h02) và hiệu chuẩn (h03: logistic đầy đủ Kinh fold 4; ngưỡng = lượng tử cross-fit hai nửa trang trên âm).\n')
        w(f"Ngưỡng q = 0,00015: học lại T {vn(RT['thr_hoc_lai']['T'],5)}, L {vn(RT['thr_hoc_lai']['L'],5)} (cũ T {RT['thr_cu']['T']['0.00015']}, "
          f"L {RT['thr_cu']['L']['0.00015']}).\n")
        w('| STT | H4 cũ | H4 v2 thay ảnh | H4 v2 học lại (H1 v2 ∧ chứng nhận ∧ n_hum ≥ 3) |\n|---|---:|---:|---:|')
        for s, v in RT['stt_H4'].items():
            w(f"| {s} | {vn(v['H4_cu'])} | {vn(v['H4_v2_dropin'])} | {vn(v['H4_v2_hoc_lai'])} |")
        w('\nĐộ chính xác của H4 học lại ở STT chưa ước lượng lại (cần tỉ lệ nhận trượt của bộ kiểm học lại trên Borg — cột FAR trượt ở §4.2 là số tương '
          'ứng); SUY ĐOÁN.\n')
    # ================================================================== 5
    w('## 5. H4 (và H2, H1) của TN3 với crop cũ ↔ crop chuẩn\n')
    w('H1 dùng cờ của CROP MỚI (A0_new, B0_new; CNT và cờ trượt BC giữ nguyên); điểm bộ kiểm tính trên crop mới; ngưỡng τ = 0,995 CHỌN LẠI trên sách kia '
      '(LOBO) đúng thủ tục TN1 (invariant: thủ tục tái lập đúng ngưỡng TN1 và số ô H2/H4 cũ ở cả 6 book_set). L16/TK đo với KHE CỦA CROP MỚI. Bộ khác: '
      'bộ ước lượng TN3 chạy lại (`s06c_tn3m.py` = bản chép `t02_matrix.py`, chỉ đổi mặt nạ và khe) — ƯỚC LƯỢNG/SUY ĐOÁN như TN3. Cột "lai" (hậu '
      'kiểm) = crop chuẩn v2 + H1 v2 + bộ kiểm chấm crop CŨ với ngưỡng TN1.\n')
    thr = V2['s06a']['thresholds']
    w('Ngưỡng chọn lại (C5 = (p_wood, viss); C2 = p_wood): ' + '; '.join(f"{k}: C5 {v['C5']}, C2 {v['C2']}" for k, v in thr.items())
      + ' · v1: ' + '; '.join(f"{k}: C5 {v['C5']}, C2 {v['C2']}" for k, v in V1['s06a']['thresholds'].items() if k.startswith('new')) + '.\n')
    for d in ('H4', 'H2', 'H1'):
        w(f'### {d}\n')
        hdr = '| Bộ | Mức chắc | cũ: giữ (%) · chính xác [khoảng] | v1 | v2 | Δ ô v2 − cũ |' + (' lai (v2 + bộ kiểm cũ) | Δ lai − cũ |' if (ML is not None and d != 'H1') else '')
        w(hdr); w('|---|---|---|---|---|---:|' + ('---|---:|' if (ML is not None and d != 'H1') else ''))
        for s in SETS8:
            def cell(M):
                r = M.loc[(s, d)]
                if r.keep == 0:
                    return '0'
                return f"{vn(r.keep)} ({vn(r.keep_pct,1)} %) · **{pc(r.both_pt)}** [{pc(r.both_lo)}–{pc(r.both_hi)}]" + (' ✔' if r.meets != 'khong' else '')
            line = f"| {s} | {TRUTH[s]} | {cell(MO)} | {cell(M1)} | {cell(M2)} | {sg(M2.loc[(s,d)].keep - MO.loc[(s,d)].keep)} |"
            if ML is not None and d != 'H1':
                line += f" {cell(ML)} | {sg(ML.loc[(s,d)].keep - MO.loc[(s,d)].keep)} |"
            w(line)
        w('')
    w('Khoảng: L16/TK = CI 95 % bootstrap cụm trang (B = 2.000); bộ khác = [cận bi quan – cận lạc quan] của bộ ước lượng TN3. ✔ = đạt tiêu chí TN3 '
      '(đo: điểm ≥ 99,5 %; ước lượng/suy đoán: cận bi quan ≥ 99 %; ≥ 50 ô). STT/Chr H2 = H4 (không có văn bản người của dị bản).\n')
    if BS:
        w('### 5.1 Độ nhạy: dao động do chọn ngưỡng (hậu kiểm)\n')
        w(f"Bootstrap B = {BS['B']} lần các TRANG phần B của sách tune → chọn lại C5 → áp lên sách thử (cùng mẫu trang cho cũ/v1/v2). Đếm ô H4 có GT "
          "(lệch ≤ 2 ô so với bảng trên vì bỏ ô không có chữ người).\n")
        w('| Sách thử | Crop | ô giữ (lượt gốc) | ô giữ trung vị [5 %–95 %] | chính xác trung vị (5 %) | hiệu với cũ: trung vị [5 %; 95 %] | P(mới > cũ) |')
        w('|---|---|---:|---|---|---|---:|')
        for t, bk in (('LucVanTien1916', 'L16'), ('TruyenKieu1872', 'TK')):
            for k, nm in (('cu', 'cũ'), ('v1', 'v1'), ('v2', 'v2')):
                o = BS[t][k]
                dd = BS[t].get(f'{k}_tru_cu')
                w(f"| {bk} | {nm} | {vn(o['goc_giu'])} | {vn(o['giu_trung_vi'])} [{vn(o['giu_p5'])}–{vn(o['giu_p95'])}] | {pc(o['cx_trung_vi'])} % "
                  f"({pc(o['cx_p5'])} %) | " + (f"{sg(dd['trung_vi'])} [{sg(dd['p5'])}; {sg(dd['p95'])}] | {vn(dd['ti_le_nhieu_hon'],2)} |" if dd else '– | – |'))
        w('\n→ Chênh số ô H4 giữa crop cũ và crop chuẩn ở L16/TK nằm gọn trong dao động do chọn ngưỡng; kết luận "giữ nhiều/ít hơn" **không được hỗ '
          'trợ** bởi dữ liệu. (Thạch bản dùng ngưỡng chọn trên IHR nên cũng chịu dao động này.)\n')
    # ================================================================== 6 invariant
    w('## 6. Invariant (tự kiểm; mọi mục True trừ ghi chú)\n')
    inv = {'v1 s02': V1['s02'], 'v2 s02': V2['s02'], 'v1 s03': V1['s03'], 'v2 s03': V2['s03'], 'v2 s04b': V2['s04b'],
           'v1 s05': V1['s05']['invariants'], 'v2 s05': V2['s05']['invariants'], 'v1 s06a': V1['s06a']['invariants'], 'v2 s06a': V2['s06a']['invariants']}
    for k, v in inv.items():
        w(f"- **{k}**: " + ', '.join(f"`{a}`={b}" for a, b in v.items() if a not in ('sec',)))
    w('- Ghi chú: `cert_old_eq_saved` = 0,99985 (8/52.707 ô STT nằm đúng ngưỡng; ngưỡng trong h03_dir_Kinh.json làm tròn 5 chữ số). `md5_mismatch_n` = 24 '
      'đều là ô rescue 64×64 (không qua save_crop). Invariant TN3 bên trong `s06c_tn3m.py` (H2 = TN1 (d)…) báo False là ĐÚNG KỲ VỌNG vì mặt nạ đã thay.\n')
    # ================================================================== 7 khuyến nghị
    w('## 7. Khuyến nghị cho pipeline\n')
    w(f"1. **Đưa crop chuẩn v2 vào pipeline dưới dạng TỆP PHỤ + CỜ, chưa thay crop giao nộp.** Lý do (CHẮC CHẮN THEO MÁY trên IHR): mực lạ so với khe người "
      f"giảm mạnh (ô > 25 % mực lạ còn {pc(r16,0)} %/{pc(rtk,0)} % ở L16/TK), khe gần như không đổi (hỏng ≤ {pc(ih2.loc['L16','hong_pct']/100)} %), "
      f"{vn(tot(g2,'cuu'))} ô B0 được cứu (8 bộ). Rủi ro: ~{pc(c2['L16']['core']['giam_gt25']/c2['L16']['core']['n'],1)} % ô L16 bị cắt vào nét (chữ nhỏ dính "
      "chữ kề) mà không cờ nào bắt → **cần người xem mẫu trước khi thay crop giao nộp** (dùng `vi_du.html` nhóm iv; cỡ mẫu chứng nhận ≤ 1 % lỗi theo TN3 §7: 0 lỗi/299 ô).")
    w("2. **Dùng kiểm \"một chữ\" của crop chuẩn (one_char_ok, two, cut, ink) thay phần cấp crop của B0 trong cổng H1** — H1 v2 giữ thêm ô ở chép tay "
      f"(stt2 {sg(M2.loc[('stt2','H1')].keep-MO.loc[('stt2','H1')].keep)}, stt4 {sg(M2.loc[('stt4','H1')].keep-MO.loc[('stt4','H1')].keep)}, "
      f"stt11 {sg(M2.loc[('stt11','H1')].keep-MO.loc[('stt11','H1')].keep)}), bớt ở thạch bản (KVK {sg(M2.loc[('KVK','H1')].keep-MO.loc[('KVK','H1')].keep)}, "
      f"L83 {sg(M2.loc[('L83','H1')].keep-MO.loc[('L83','H1')].keep)}) mà độ chính xác (đo/ước lượng) không đổi.")
    w("3. **KHÔNG cho bộ kiểm ảnh hiện tại chấm crop chuẩn** (\"thay ảnh\"): không lợi ở mộc bản (AUC giảm nhẹ, H4 trong nhiễu chọn ngưỡng) và hại rõ ở "
      "chép tay (Borg keep_v5: nhận dương giảm " + vn(100 * (V2['s05']['borg_joint']['keep_v5|cu']['accept_pos'] - V2['s05']['borg_joint']['keep_v5|moi']['accept_pos']), 1) + " điểm % ở cùng FAR). Cấu hình dùng được ngay là **\"lai\"**: giao crop chuẩn + H1 v2 + bộ kiểm chấm crop CŨ "
      "với ngưỡng TN1 (số ở §5, cột lai). Muốn bộ kiểm chấm crop chuẩn thì phải **học lại** bộ kiểm trên crop chuẩn"
      + (f"; nhưng với chữ viết tay, kể cả HỌC LẠI trên crop chuẩn v2 cũng chỉ nhận {pc(RT['borg_joint']['hoc_lai']['keep_v5']['accept_pos'])} % dương Borg "
         f"(cũ {pc(RT['borg_joint']['cu']['keep_v5']['accept_pos'])} %, thay ảnh {pc(RT['borg_joint']['dropin']['keep_v5']['accept_pos'])} %) ở cùng mức FAR "
         "→ với STT giữ bộ kiểm trên crop cũ" if RT else '') + '.')
    w("4. **Không đổi kết luận TN3**: crop chuẩn không đưa L16, thạch bản, Chr, STT qua ngưỡng chính xác; chỉ TK đạt. Vế \"một chữ\" vẫn chưa có sự thật "
      "người — khe người IHR chỉ là thước đo gián tiếp.")
    w('5. **Tham số đề xuất (v2, `cclib_v2.PARAMS`)**: ranh giới dọc = trung điểm tâm (virt 0,55·max(p,h) khi thiếu ô kề, kẹp 0,8·max(p,h)); seam chỉ cho '
      'thành phần vắt qua, dải ± 0,25·d, phạt 20/255; dải ngang xc ± 0,75·w; Otsu cục bộ kẹp [64, 200]; bóc nét kẻ nhân 1,2·p / 1,2·w; đầu nét lạ '
      '20 %/10 %; đốm max(3, 0,0015·p·w); cờ cut 30 %/chạm mép, two 1,35·p + khe 0,10·p + 20/20, ink [0,4; 2,5]; lề 10 % cạnh dài, 128×128. '
      'Việc nên thử tiếp (chưa đo): cờ thêm cho "đầu nét lạ bị bỏ ở chữ nhỏ" (L16), học lại verifier_ft in (T/L) trên crop chuẩn.\n')
    # ================================================================== 8 giới hạn
    w('## 8. Giới hạn\n')
    w('- **v2 là hậu kiểm**: cấu trúc v2 chọn sau khi thấy v1 trên toàn dữ liệu (kể cả IHR); không dò tham số nhưng vẫn có thể lạc quan. v1 (đăng ký '
      'trước) là số "sạch" về thủ tục: v1 kém v2 về số ô sạch ở cả 8 bộ và về mực lạ ở L16/TK (§2, §3), nhưng hỏng khe L16 v1 ít hơn '
      f"(v1 {vn(ih1.loc['L16','hong'])}, v2 {vn(ih2.loc['L16','hong'])}).")
    w('- Bộ kiểm ảnh học trên crop cũ; chỉ bộ kiểm viết tay được học lại (hậu kiểm). verifier_ft in và kênh viss chưa học lại.')
    w('- Khe người là khe MẪU (vị trí trung vị theo sách trong hộp cột người vẽ), không phải hộp chữ người; mép khe lệch vài px → dùng thêm "lõi". Mực đo '
      'bằng ngưỡng Otsu cục bộ của chính thuật toán mới (cùng ngưỡng cho cũ/mới).')
    w('- "Một chữ" không có sự thật người ở bộ nào; cờ bleed/tall/hai chữ là luật máy. Ô "cờ mới" không chắc là lỗi.')
    w('- STT: trang đã xử lý là ảnh nhị phân (không có nền giấy gốc); crop chuẩn STT giữ nền trắng như pipeline.')
    w('- Số ô H4 phụ thuộc mạnh vào chọn ngưỡng trên ~2.800–4.700 ô tune (§5.1); ước lượng ở bộ không có nhãn người thừa hưởng mọi giả định của TN3.')
    w('- Borg: crop chuẩn chỉ dựng cho ô keep ∨ keep_high (71.951), không cho ô không keep.\n')
    # ================================================================== 9 tệp
    w('## 9. Tệp và tái lập\n')
    w('| Tệp | Việc |\n|---|---|')
    for f, d in (('cclib.py / cclib_v2.py', 'thuật toán crop chuẩn v1 (đăng ký trước) / v2 (hậu kiểm) + dựng lại crop cũ'),
                 ('ver.py', 'chọn phiên bản qua biến môi trường TN4_VER = v1 hoặc v2'), ('hslot.py', 'mô hình khe người của harness cho tâm bất kỳ'),
                 ('s01_crop.py', 'cắt crop chuẩn (gold/aux/borg) + đo crop cũ, md5 byte, khe người'),
                 ('s02_geom.py, s02b_cov_core.py', 'bảng hình học 8 bộ, IHR, Borg; độ phủ lõi'),
                 ('s03_vft.py', 'p_wood trên crop chuẩn (bộ kiểm giữ nguyên); `TN4_FRAME=tight` = khung chặt'),
                 ('s04a_rerank_copies.py, s04b_embed_sv.py, rerank_v1/, rerank_v2/', 'chạy lại kênh viss (bản chép r02/r02b/r03 chỉ đổi thư mục)'),
                 ('s05_hand.py', 'bộ kiểm viết tay LOBO trên STT + Borg DungLy'), ('s06a_masks.py', 'H1/H2/H4 mới + chọn lại ngưỡng LOBO'),
                 ('s06b_make_tn3m.py → s06c_tn3m.py', 'bộ ước lượng TN3 với mặt nạ mới (`TN4_SIMG=SIMGo` = cấu hình lai)'),
                 ('s06d_boot_sel.py', 'độ nhạy chọn ngưỡng'), ('s07_auc.py', 'AUC IHR trước/sau'),
                 ('s10a_hand_retrain_prep.py, h01_train_tn4.py, s10b_hand_retrain_eval.py', 'học lại bộ kiểm viết tay trên crop v2'),
                 ('s08_vi_du.py → vi_du.html, anh/, vi_du_manifest.csv', 'ví dụ 4 nhóm × 24 ô, phân tầng theo bộ'),
                 ('s09_report.py → KET_QUA.md', 'báo cáo này'), ('run_ver.sh', 'chuỗi bước 2–9 cho một phiên bản')):
        w(f'| `{f}` | {d} |')
    w('\n```bash\ncd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR; L=lab/thu_nghiem_anh_chu/TN4_crop_chuan\n'
      'for V in v1 v2; do\n'
      '  for s in stt2 stt4 stt11 Chr L83 KVK L16 TK; do TN4_VER=$V .venv/bin/python $L/s01_crop.py --part gold --sets $s --workers 4 --tag $s; done  # ≈ 5 phút\n'
      '  TN4_VER=$V .venv/bin/python $L/s01_crop.py --part aux --workers 4; TN4_VER=$V .venv/bin/python $L/s01_crop.py --part borg --workers 4\n'
      '  TN4_VER=$V bash $L/run_ver.sh 1        # s02 → s07 (≈ 20 phút, MPS cho CNN)\n'
      'done\n'
      'TN4_VER=v2 TN4_FRAME=tight .venv/bin/python $L/s03_vft.py; TN4_VER=v2 TN4_FRAME=tight .venv/bin/python $L/s05_hand.py; TN4_VER=v2 TN4_FRAME=tight .venv/bin/python $L/s07_auc.py\n'
      'TN4_VER=v2 TN4_SIMG=SIMGo .venv/bin/python $L/s06c_tn3m.py      # cấu hình lai\n'
      '.venv/bin/python $L/s06d_boot_sel.py                              # độ nhạy chọn ngưỡng (~1 phút)\n'
      '.venv/bin/python $L/s10a_hand_retrain_prep.py && H=measure_out/_thu_nghiem_anh_chu/TN4/v2/hand_retrain\n'
      '.venv/bin/python $L/h01_train_tn4.py Kinh TruyenKieu1872 2800 T; .venv/bin/python $L/h01_train_tn4.py Kinh LucVanTien1916 2800 L   # 2 × ~10 phút MPS\n'
      '.venv/bin/python $L/s10b_hand_retrain_eval.py\n'
      '.venv/bin/python $L/s08_vi_du.py && .venv/bin/python $L/s09_report.py\n```\n')
    w('Đầu vào chỉ đọc: `dataset/_ALL/labels.csv`, `prepared/<Book>/{pages,manifest.json,dataset_out/labels.csv,kim_raw}`, `dataset_out/labels.csv` (STT), '
      '`data/<Book>/` (ảnh gốc, manifest.tsv IHR), scratchpad `$SP/{r6/policy_v3/out/base_v2.pkl, r4/verifier_ft, r5/hand_lobo, kim_bottleneck/{harness,rerank}, '
      'gold_img_audit/m_ocr, r4/borg_align, r5/borg_quality}`, `lab/.../TN1_cong_kiem`, `TN3_tat_ca_bo`.\n')
    (HERE / 'KET_QUA.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('ok', len(L))


if __name__ == '__main__':
    main()
