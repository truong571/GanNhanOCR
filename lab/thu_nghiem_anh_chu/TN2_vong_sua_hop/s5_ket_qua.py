#!/usr/bin/env python3
"""s5_ket_qua.py — sinh KET_QUA.md (mọi con số đọc từ tệp do s1b/s3/t01/t02 sinh ra) + de_xuat_chr_stt.csv (mẫu cho người xem)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
REPO = Path('/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR')
OUT = REPO / 'measure_out/_thu_nghiem_anh_chu/TN2'
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
sys.path.insert(0, str(SP / 'r5/policy_v2/scripts'))
from vlib import var_eq_plus  # noqa

S3 = json.loads((HERE / 's3_summary.json').read_text())
S3L = json.loads((HERE / 's3_summary_lenient.json').read_text())
T4 = json.loads((HERE / 't04_share_demoted_decisions_lenient.json').read_text())
CVL = pd.read_csv(HERE / 's3_theta_curve_lenient.csv')
assert S3['share_rule'] == 'strict' and S3L['share_rule'] == 'lenient'
S1 = json.loads((OUT / 's1b_summary.json').read_text())
T1 = json.loads((HERE / 't01_slot_invariant.json').read_text())
T2 = json.loads((HERE / 't02_diag.json').read_text())
CV = pd.read_csv(HERE / 's3_theta_curve.csv')
DEC = pd.read_csv(OUT / 'decisions.csv', dtype=str, keep_default_na=False)
SC = pd.read_csv(OUT / 'scores.csv', dtype=str, keep_default_na=False)


def cp(k, n):
    if n == 0:
        return (float('nan'), float('nan'))
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def pc(x, d=2):
    return '—' if x is None or x != x else f'{100 * x:.{d}f}'.replace('.', ',')


def nf(n):
    return f'{int(n):,}'.replace(',', '.')


def th_(t):
    return '—' if t is None else f'{t:.2f}'.replace('.', ',')


def ci(c, d=2):
    return f'[{pc(c[0], d)}–{pc(c[1], d)}]'


short = {'LucVanTien1916': 'L16', 'TruyenKieu1872': 'TK', 'Chrestomathie1872': 'Chr', 'SachThanhTruyen': 'STT'}
IH = ['LucVanTien1916', 'TruyenKieu1872']
L = {r['test']: r for r in S3['lobo']}
LL = {r['test']: r for r in S3L['lobo']}
# phần dư: lỗi ảnh=nhãn KHÔNG bị cờ (trần của mọi vòng lặp khởi từ cờ kim_off)
ce = pd.read_csv(SP / 'kim_bottleneck/harness/cells_eval.csv', dtype=str, keep_default_na=False)
gs = pd.read_csv(SP / 'gold_img_audit/measure/gold_suspicion.csv', dtype=str, keep_default_na=False, usecols=['cell_uid', 'is_gold', 'geo_f_kim_off'])
flag = set(gs[(gs.is_gold == '1') & (gs.geo_f_kim_off == '1')].cell_uid)
resid = {}
for b in IH:
    G = ce[(ce.book == b) & (ce.delivered_tier == 'GOLD') & (ce.geo_status == 'ok')]
    uf = G[~G.cell_uid.isin(flag)]
    resid[b] = dict(unflag=len(uf), unflag_img_wrong=int(sum(not var_eq_plus(a, c) for a, c in zip(uf.gt_img, uf.label))),
                    unflag_slot_wrong=int((uf.slot_ok != '1').sum()),
                    unflag_slot0=int((uf.slot_ok == '0').sum()))

lines = []
w = lines.append
w('# TN2 — Vòng lặp sửa hộp ảnh GOLD: độ chính xác trước khi đưa vào pipeline (26/09/2026)')
w('')
w('Thư mục: `lab/thu_nghiem_anh_chu/TN2_vong_sua_hop/` (script, bảng nhỏ, `vi_du.html` + `img/`); tệp lớn: '
  '`measure_out/_thu_nghiem_anh_chu/TN2/` (crop mới, `candidates.csv`, `scores.csv`, `decisions.csv`). '
  '0 lần gọi API, không tải model, repo chỉ đọc ngoài hai thư mục trên, không mở ảnh bằng LLM. Mọi số dưới đây do `s5_ket_qua.py` '
  'đọc từ tệp đã lưu (`s3_summary.json`, `s3_lobo_table.csv`, `s3_theta_curve.csv`, `t0*_*.json`, `s1b_summary.json`).')
w('')
w('Nhãn độ chắc: **CHẮC CHẮN THEO MÁY** = đo trên sự thật người IHR (hộp cột người vẽ + chữ GT người) · **ƯỚC LƯỢNG** = đếm của máy, chưa có sự thật · **SUY ĐOÁN** = suy luận chưa đo.')
w('')
w('## 1. Vòng lặp đã định (viết trước khi chấm)')
w('')
w('- **Ô vào vòng**: ô GOLD có cờ `geo_f_kim_off = 1` (tâm hộp ảnh cách tâm hộp kim của CHÍNH chữ nhãn > 0,5 bước cột theo y hoặc > 0,5 bề rộng theo x; `gold_suspicion.csv`).')
w('- **Ứng viên** (`s1b_candidates.py`): (a) `prev`/`next` = hộp kề trên/dưới trong cột (dịch chỉ số ±1; hộp phân biệt kề theo y trong bản ghi build mọi tầng); '
  '(b) `kim` = hộp cũ tịnh tiến cho tâm trùng tâm hộp kim `chars[nom_idx]` (dựng lại bằng `align_production._detect`, kim đọc ra đúng nhãn ở 100 % ô); '
  '(c) `det` = hộp detector THÔ (ckpt/ngưỡng/biên x theo sách) gần tâm kim nhất, chỉ khi cách ≤ 0,5 bước.')
w('- **Cắt**: đúng `build_dataset.save_crop` (pad 0,12, carve láng giềng = hộp kề theo y của chính hộp ứng viên, tighten, ảnh gốc khi sách khai `crop_source: original`).')
w('- **Chấm để CHỌN** (`s2_score.py`): bộ kiểm học trên nhãn người `r4/verifier_ft` (T học TK1872 phần A + Borg; L học LVT1916 phần A + Borg). '
  '`m = cos(crop, nguyên mẫu nhãn) − max cos(crop, đối thủ)`, đối thủ = chữ kim kề ±1/±2 ∪ đồng âm R(âm) ∪ top-5 gần hình, bỏ dị thể của nhãn.')
w('- **Nhận** ứng viên m cao nhất khi `m ≥ θ` VÀ `m > m(crop đang giao)`; rồi trên trạng thái cuối của cột: hộp mới không trùng (IoU < 0,5) hộp của bất kỳ ô giao nộp nào còn giữ hộp, '
  'và tâm y nằm giữa ô còn giữ có nom_idx liền trước/liền sau; vi phạm → huỷ (hai ô cùng dời vào trùng nhau → hạ cả hai). **Một lượt duy nhất. Không nhận → HẠ. Nhãn không bao giờ đổi.**')
w('- **Luật trùng hộp — hai bản**: **NGHIÊM (chính)** = ô bị hạ VẪN giữ hộp cũ (nó còn trong bộ dữ liệu ở tầng thấp), đúng nghĩa "không hai ô chung một hộp"; '
  '**NỚI (phụ)** = chỉ ô không bị hạ mới giữ hộp. Ở bản nới, ' + ', '.join(f'{T4[b]["trung_hop_voi_o_bi_ha"]}/{T4[b]["sua"]} ô sửa ở {short.get(b, b[:3])}' for b in ['LucVanTien1916', 'TruyenKieu1872', 'Chrestomathie1872', 'SachThanhTruyen']) +
  ' lấy đúng hộp của một ô bị hạ (hai ô chung một ảnh, khác tầng).')
w('- **LOBO**: sách L16 chọn bằng T, TK chọn bằng L (không mô hình nào thấy sách nó chấm). θ chọn trên sách KIA theo tiêu chí đăng ký trước: '
  'θ nhỏ nhất trên lưới −0,20…0,80 (bước 0,05) mà độ đúng của ô được sửa ≥ độ đúng của GOLD không bị cờ ở sách tune, và giữ được ở mọi θ lớn hơn.')
w('- **Chấm kết quả** (`s3_eval.py`, `slotlib.py`): chỉ mô hình khe người (chép nguyên logic `h01_build_cells.py`; bất biến: tái lập slot_ok/gt_img/gt_char của `cells_eval.csv` '
  f'{T1["LucVanTien1916"]["slot_ok_eq"]}/{T1["LucVanTien1916"]["n"]} và {T1["TruyenKieu1872"]["slot_ok_eq"]}/{T1["TruyenKieu1872"]["n"]} ô). Bộ kiểm dùng để chọn KHÔNG dùng để chấm.')
w('  - **Chỉ số chính "ảnh = nhãn"**: chữ GT người ở khe mà crop nằm (hai mô hình khe cùng chỉ) tương đương V1+ với nhãn. Đây đúng là yêu cầu "crop là chữ của nhãn". '
  'Ghi chú minh bạch: lượt chạy đầu dùng "khe == syl_idx" làm chỉ số chính; đổi sang "ảnh = nhãn" vì vài ô sửa đưa crop tới đúng chỗ chữ nhãn nhưng nhãn lệch âm (nom_idx ≠ khe); cả hai đều báo.')
w('  - Phụ: "khe" = slot_ok (crop ở khe syl_idx); "hai vế" = nhãn ~V1+ chữ GT tại syl_idx và slot_ok (định nghĩa báo cáo v2).')
w(f'  - Mẫu số: GOLD trên trang có GT người (L16 {nf(L["LucVanTien1916"]["n_P0"])} ô; TK {nf(L["TruyenKieu1872"]["n_P0"])} ô, bỏ '
  f'{nf(S3["ngoai_mau_so"]["TruyenKieu1872"]["gold_trang_khong_gt"])} ô trên trang không có GT); khe chưa xác định tính SAI. CI = bootstrap cụm theo trang, B = 2000.')
w('')
w('**Bất biến tái lập** (CHẮC CHẮN): crop cũ cắt lại trùng md5 tệp giao nộp: ' +
  ', '.join(f'{short.get(b, b[:3])} {S1[b]["old_exact"]}/{S1[b]["old_n"]}' for b in ['LucVanTien1916', 'TruyenKieu1872', 'Chrestomathie1872', 'SachThanhTruyen']) +
  ' (63 ô STT lệch đều là luật self_training_rescue — crop 64×64 không cắt bằng save_crop); chữ kim tại nom_idx == ocr_char ở 100 % ô bị cờ; invariants s3: ' +
  ', '.join(f'{k}={v}' for k, v in S3['invariants'].items()) + '.')
w('')
w('## 2. Kết quả LOBO trên hai sách IHR (luật trùng hộp NGHIÊM) — CHẮC CHẮN THEO MÁY')
w('')
w('| Sách chấm (θ chọn trên) | θ | bị cờ | nhận sửa | sửa đúng | sửa sai | sửa, khe mới chưa xác định | làm hỏng ô đúng | đổi hộp vẫn đúng | hạ (crop cũ sai) | hạ (crop cũ vốn đúng) |')
w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for b, r in L.items():
    w(f'| {short[b]} (θ từ {short[r["tune"]]}, bộ kiểm {r["model"]}) | {th_(r["theta"])} | {r["n_flag"]} | {r["n_accept"]} | {r["sua_dung"]} | {r["sua_sai"]} | '
      f'{r["sua_chua_xd"]} | {r["lam_hong_o_dung"]} | {r["doi_hop_van_dung"]} | {r["ha_o_sai"]} | {r["ha_o_dung"]} |')
w('')
w('Độ chính xác "ảnh = nhãn" của tập GOLD (P0 không làm gì · P1 chỉ hạ mọi ô bị cờ · P2 vòng sửa):')
w('')
w('| Sách | P0 [CI] (n) | P1 chỉ hạ [CI] (n) | P2 vòng sửa [CI] (n) | P2 − P1 [CI] | độ đúng của ô được sửa [CI cụm trang] · Clopper–Pearson |')
w('|---|---|---|---|---|---|')
for b, r in L.items():
    k = r['sua_dung'] + r['doi_hop_van_dung']; n = r['n_accept']; lo, hi = cp(k, n)
    w(f'| {short[b]} | {pc(r["anh_nhan_P0"])} {ci(r["anh_nhan_P0_ci"])} ({nf(r["n_P0"])}) | {pc(r["anh_nhan_P1"])} {ci(r["anh_nhan_P1_ci"])} ({nf(r["n_P1"])}) | '
      f'{pc(r["anh_nhan_P2"])} {ci(r["anh_nhan_P2_ci"])} ({nf(r["n_P2"])}) | {pc(r["anh_nhan_P2_minus_P1"])} {ci(r["anh_nhan_P2_minus_P1_ci"])} điểm % | '
      f'{k}/{n} = {pc(r["prec_fix"], 1)} % {ci(r.get("prec_fix_ci", [np.nan, np.nan]), 1)} · CP [{pc(lo, 1)}–{pc(hi, 1)}] |')
w('')
w('Chỉ số phụ (cùng ô, cùng quyết định):')
w('')
w('| Sách | khe P0 → P1 → P2 | hai vế P0 → P1 → P2 | ô sửa đúng theo chỉ-mô-hình-mẫu (độc lập hình học kim) |')
w('|---|---|---|---|')
for b, r in L.items():
    w(f'| {short[b]} | {pc(r["khe_P0"])} → {pc(r["khe_P1"])} → {pc(r["khe_P2"])} % (sửa khe đúng {r["khe_sua_dung"]}, sai {r["khe_sua_sai"]}, làm hỏng {r["khe_lam_hong"]}) | '
      f'{pc(r["hai_ve_P0"])} → {pc(r["hai_ve_P1"])} → {pc(r["hai_ve_P2"])} % | {pc(r["prec_fix_chi_mau"], 1)} % |')
w('')
nfix = {b: r['n_accept'] for b, r in L.items()}
nerr = {b: r['sua_sai'] + r['sua_chua_xd'] + r['lam_hong_o_dung'] for b, r in L.items()}
w('**So với "chỉ hạ"**: ở điểm vận hành LOBO, vòng sửa nhận ' + ', '.join(f'{nfix[b]} ô ở {short[b]}' for b in IH) + '; trong đó sửa sai rõ ' +
  ', '.join(f'{L[b]["sua_sai"]}' for b in IH) + ', làm hỏng ô đúng ' + ', '.join(f'{L[b]["lam_hong_o_dung"]}' for b in IH) +
  ', khe mới chưa xác định (tính sai) ' + ', '.join(f'{L[b]["sua_chua_xd"]}' for b in IH) + ' (L16, TK). Độ chính xác GOLD P2 − P1 = ' +
  ', '.join(f'{pc(L[b]["anh_nhan_P2_minus_P1"])} {ci(L[b]["anh_nhan_P2_minus_P1_ci"])} điểm % ({short[b]})' for b in IH) +
  ' — **không khác "chỉ hạ"**, chỉ **giữ lại thêm** ' + ', '.join(f'{L[b]["n_P2"] - L[b]["n_P1"]} ô ở {short[b]}' for b in IH) +
  '. Toàn bộ mức tăng so với P0 đến từ **việc hạ**.')
w('')
w('**Bản luật NỚI** (ô bị hạ không giữ hộp; `s3_summary_lenient.json`) — cùng tiêu chí chọn θ:')
w('')
w('| Sách | θ | nhận sửa | sửa đúng | sửa sai | chưa xác định | làm hỏng | đổi vẫn đúng | P1 → P2 [CI] (n P2) | P2 − P1 [CI] |')
w('|---|---:|---:|---:|---:|---:|---:|---:|---|---|')
for b in IH:
    r = LL[b]
    w(f'| {short[b]} | {th_(r["theta"])} | {r["n_accept"]} | {r["sua_dung"]} | {r["sua_sai"]} | {r["sua_chua_xd"]} | {r["lam_hong_o_dung"]} | {r["doi_hop_van_dung"]} | '
      f'{pc(r["anh_nhan_P1"])} → {pc(r["anh_nhan_P2"])} {ci(r["anh_nhan_P2_ci"])} ({nf(r["n_P2"])}) | {pc(r["anh_nhan_P2_minus_P1"])} {ci(r["anh_nhan_P2_minus_P1_ci"])} |')
w('')
w('**Tham chiếu "chỉ dùng hộp kim, không bộ kiểm"** (luật nghiêm; mọi ô bị cờ nhận ứng viên `kim`, vẫn qua kiểm trùng/đơn điệu — không có tham số nên không cần LOBO):')
w('')
w('| Sách | nhận | sửa đúng | sửa sai | chưa xác định | làm hỏng ô đúng | độ đúng ô sửa | GOLD P2 [CI] (n) |')
w('|---|---:|---:|---:|---:|---:|---:|---|')
for b in IH:
    k = L[b]['kimonly']
    w(f'| {short[b]} | {k["n_accept"]} | {k["sua_dung"]} | {k["sua_sai"]} | {k["sua_chua_xd"]} | {k["lam_hong"]} | {pc(k["prec_fix"], 1)} % | '
      f'{pc(k["anh_nhan_P2"])} {ci(k["anh_nhan_P2_ci"])} ({nf(k["n_P2"])}) |')
w('')
w('→ Hộp kim đặt đúng chỗ gần như mọi ô trượt thật (trần "có ít nhất một ứng viên đúng khe" = ' +
  ', '.join(f'{short[b]} {T2[b]["oracle_any_cand_slot1_of_before_wrong"][0]}/{T2[b]["oracle_any_cand_slot1_of_before_wrong"][1]}' for b in IH) +
  ' ô bị cờ có khe sai/chưa xác định), nhưng cờ kim_off cũng bắn vào ô mà **hộp kim mới là hộp lệch** (' +
  ', '.join(f'{short[b]} {T2[b]["before"]["1"]}' for b in IH) +
  ' ô bị cờ vốn đúng khe). Không có bộ kiểm, các ô này bị làm hỏng và P2 **thấp hơn** "chỉ hạ". Bộ kiểm là điều kiện cần.')
w('')
w('**Đường cong θ** (luật nghiêm, `s3_theta_curve.csv`; mỗi sách chấm bằng mô hình ngoài-sách của nó; KHÔNG phải LOBO vì θ nhìn chính sách đó):')
w('')
w('| θ | L16: nhận / sửa đúng / sai+chưa xđ / làm hỏng · độ đúng | TK: nhận / sửa đúng / sai+chưa xđ / làm hỏng · độ đúng |')
w('|---:|---|---|')
for th in [-0.2, 0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5]:
    cells = []
    for b in IH:
        x = CV[(CV.book == b) & (np.isclose(CV.theta, th))].iloc[0]
        cells.append(f'{x.n_accept} / {x.sua_dung} / {x.sua_sai + x.sua_chua_xd} / {x.lam_hong_o_dung} · {pc(x.prec_fix, 1)} %')
    w(f'| {th_(th)} | {cells[0]} | {cells[1]} |')
w('')
tT, tL = S3['theta_chosen_on']['TruyenKieu1872'], S3['theta_chosen_on']['LucVanTien1916']
xTK = CV[(CV.book == 'TruyenKieu1872') & np.isclose(CV.theta, tT)].iloc[0]
w(f'Tiêu chí chọn θ cho ra θ = {th_(tT)} khi tune trên TK và θ = {th_(tL)} khi tune trên L16: ngưỡng **không chuyển tốt giữa hai sách/hai mô hình** '
  f'(TK nhận θ {th_(tL)} của L16 nên sửa được {L["TruyenKieu1872"]["n_accept"]} ô, dù ở θ {th_(tT)} TK có {xTK.n_accept} ô sửa, đúng '
  f'{xTK.sua_dung + xTK.doi_hop_van_dung} — số không-LOBO). Lỗi ở θ thấp chủ yếu là **làm hỏng ô vốn đúng** (hộp kim lệch, bộ kiểm thích crop mới hơn), không phải sửa sai ô trượt.')
w('')
w('**Phần không với tới**: lỗi "ảnh = nhãn" còn lại trong GOLD KHÔNG bị cờ: ' +
  ', '.join(f'{short[b]} {resid[b]["unflag_img_wrong"]}/{nf(resid[b]["unflag"])} (trong đó khe sai/chưa xác định {resid[b]["unflag_slot_wrong"]})' for b in IH) +
  ' — phần lớn là nhãn đọc sai chữ (kim), vòng sửa HỘP không chữa được; ô trượt thật mà cờ kim_off bỏ sót ít (khe = 0 không cờ: ' +
  ', '.join(f'{short[b]} {resid[b]["unflag_slot0"]}' for b in IH) + ').')
w('')
w('## 3. Chrestomathie1872 và SachThanhTruyen — ƯỚC LƯỢNG, chưa kiểm được')
w('')
w('Không có sự thật người. Luật: hai bộ kiểm T (θ chọn trên L16, nơi T ngoài-sách) và L (θ chọn trên TK) phải **cùng chọn một ứng viên và cùng qua ngưỡng**, rồi kiểm trùng/đơn điệu.')
w('')
w('| Bộ · luật | GOLD | bị cờ | T nhận thô | L nhận thô | hai bộ cùng ứng viên | **đề xuất sửa** | **đề xuất hạ** | huỷ vì trùng hộp | loại ứng viên |')
w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
for tagr, SS in (('nghiêm', S3), ('nới', S3L)):
    for b, u in SS['unlabelled'].items():
        w(f'| {short[b]} · {tagr} | {nf(u["n_gold"])} | {nf(u["n_flag"])} | {u["raw_T"]} | {u["raw_L"]} | {u["raw_both_same_cand"]} | **{u["de_xuat_sua"]}** | {nf(u["de_xuat_ha"])} | {u["huy_trung_hop"]} | '
          + ', '.join(f'{k} {v}' for k, v in u['cand_counts'].items()) + ' |')
w('')
w('Mức chắc: **ƯỚC LƯỢNG** (số đếm máy). Độ đúng của đề xuất trên Chr/STT **chưa đo được**. Báo cáo v2 (§1 mục 7) ghi rằng ở STT một bộ kiểm chữ viết tay giữ ngoài theo sách '
  'vẫn nhận ≈ 11,5 % [7,0–16,7] crop trượt một phần; bộ kiểm dùng ở đây chưa được đo riêng trên STT (SUY ĐOÁN: rủi ro cao hơn IHR). '
  'Mẫu cho người xem: `vi_du.html` (nhóm 4–5) và danh sách đủ (luật nghiêm) `de_xuat_chr_stt.csv`.')
w('')
w('## 4. Kết luận')
w('')
ntot = sum(nfix.values()); etot = sum(nerr.values())
w(f'1. **An toàn ở điểm vận hành, mẫu nhỏ** (CHẮC CHẮN THEO MÁY): trên hai sách IHR (luật nghiêm) nhận sửa {ntot} ô; sửa sai rõ '
  f'{sum(L[b]["sua_sai"] for b in IH)}, làm hỏng ô đúng {sum(L[b]["lam_hong_o_dung"] for b in IH)}, khe mới chưa xác định {sum(L[b]["sua_chua_xd"] for b in IH)}. '
  'CP 95 % cho tỉ lệ lỗi của ô sửa: ' + ', '.join(f'{short[b]} ≤ {pc(cp(nerr[b], nfix[b])[1], 1)} % ({nerr[b]}/{nfix[b]})' for b in IH if nfix[b]) +
  ('' if all(nfix[b] for b in IH) else '; ' + ', '.join(f'{short[b]}: 0 ô được sửa' for b in IH if not nfix[b])) + '.')
w('2. **Lợi ích chỉ là độ phủ, không phải độ chính xác**: P2 = P1 trong sai số ở cả hai sách; vòng sửa giữ lại ' +
  ' và '.join(f'{L[b]["n_P2"] - L[b]["n_P1"]} ô ({short[b]})' for b in IH) + ' (bản nới: ' +
  ' và '.join(f'{LL[b]["n_P2"] - LL[b]["n_P1"]} ô' for b in IH) + ') trong số ' + ' và '.join(f'{L[b]["n_flag"]}' for b in IH) +
  ' ô bị cờ. Việc **HẠ** ô bị cờ mới là đòn bẩy độ chính xác (' +
  ', '.join(f'{short[b]} {pc(L[b]["anh_nhan_P0"])} → {pc(L[b]["anh_nhan_P1"])} %' for b in IH) + ').')
w(f'3. **Ngưỡng không ổn định giữa sách** (θ {th_(tT)} vs {th_(tL)}); ở θ thấp vòng sửa làm hỏng ô vốn đúng; bỏ bộ kiểm thì hại (P2 < P1).')
w(f'4. **Khuyến nghị**: (a) đưa **"hạ ô bị cờ kim_off"** vào pipeline trước (tệp phụ, không phá dữ liệu; cái giá: hạ cả ô vốn đúng — '
  + ', '.join(f'{short[b]} {L[b]["ha_o_dung"] + L[b]["doi_hop_van_dung"]}/{L[b]["n_flag"]}' for b in IH) + ' ô bị cờ); (b) vòng sửa chỉ nên vào như **bước TUỲ CHỌN, mặc định TẮT**, '
  f'cho sách khắc gỗ/thạch bản, luật trùng hộp nghiêm, ngưỡng bảo thủ θ = {th_(max(tT, tL))} và đòi hai bộ kiểm cùng chọn (như luật Chr/STT); (c) **không** bật cho STT/Chr '
  'cho tới khi người xem mẫu `vi_du.html` (nhóm 4–5). Lợi ích kỳ vọng nhỏ (SUY ĐOÁN: vài chục ô mỗi sách).')
w('')
w('## 5. Giới hạn')
w('')
w('- Chỉ hai sách IHR có sự thật; ' + ', '.join(f'{short[b]} nhận {nfix[b]} ô' for b in IH) + ' ở điểm LOBO nên kết luận "an toàn" dựa chủ yếu vào L16.')
w('- Mô hình khe người có một thành phần dùng hộp DÒNG của kim, còn ứng viên `kim` dùng hộp CHỮ của kim: có chung nguồn hình học. Độ nhạy bằng chỉ-mô-hình-mẫu '
  '(độc lập kim): ô được sửa đúng ' + ', '.join(f'{short[b]} {pc(L[b]["prec_fix_chi_mau"], 1)} %' for b in IH if L[b]['prec_fix_chi_mau'] is not None) + '.')
w('- "Ảnh = nhãn" đo bằng tâm hộp (như báo cáo v2), không đo mực chữ kề lọt vào crop mới (vế "một chữ").')
w('- Cờ vào vòng chỉ là kim_off; ô trượt mà kim cũng lệch theo (kim đọc chữ kề) không vào vòng.')
w('- Luật nghiêm chặn mọi ô muốn lấy hộp của ô bị hạ, nên một chuỗi trượt chỉ sửa được khi cả chuỗi cùng được nhận.')
w('')
w('## 6. Tái lập (≈ 5 phút, 0 API; MPS cho s2)')
w('')
w('```bash')
w('cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR; D=lab/thu_nghiem_anh_chu/TN2_vong_sua_hop')
w('.venv/bin/python $D/s1a_kimdet.py          # hộp kim + detector thô của trang có ô bị cờ (~4 phút)')
w('.venv/bin/python $D/s1b_candidates.py --workers 4   # ứng viên + crop bằng save_crop (~15 s)')
w('.venv/bin/python $D/s2_score.py            # điểm bộ kiểm T/L (~15 s, MPS)')
w('.venv/bin/python $D/t01_slot_invariant.py  # bất biến mô hình khe')
w('.venv/bin/python $D/s3_eval.py --share strict  # vòng lặp + LOBO + Chr/STT (~20 s)')
w('.venv/bin/python $D/s3_eval.py --share lenient # bản luật nới')
w('.venv/bin/python $D/t04_share_demoted.py decisions_lenient.csv  # ô sửa lấy hộp của ô bị hạ')
w('.venv/bin/python $D/t02_diag.py            # chẩn đoán ứng viên/trần')
w('.venv/bin/python $D/s4_html.py             # vi_du.html + img/')
w('.venv/bin/python $D/s5_ket_qua.py          # KET_QUA.md + de_xuat_chr_stt.csv')
w('```')
(HERE / 'KET_QUA.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
# danh sách đề xuất Chr/STT cho người xem
d = DEC[DEC.book_set.isin(['Chrestomathie1872', 'SachThanhTruyen']) & (DEC.decision == 'sua')]
sc = SC[SC.cand == 'deliv'].set_index('cell_uid')
d = d.assign(label=d.cell_uid.map(sc.label), syllable=d.cell_uid.map(sc.syllable), old_bbox=d.cell_uid.map(sc.old_bbox),
             image=d.cell_uid.map(lambda u: sc.at[u, 'crop'].split('dataset/_ALL/')[-1]))
d[['cell_uid', 'book_set', 'label', 'syllable', 'image', 'old_bbox', 'cand', 'new_bbox', 'm_new']].to_csv(HERE / 'de_xuat_chr_stt.csv', index=False)
print('KET_QUA.md', len(lines), 'dòng; de_xuat_chr_stt.csv', len(d))
