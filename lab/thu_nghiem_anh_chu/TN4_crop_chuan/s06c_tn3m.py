#!/usr/bin/env python
"""t02_matrix.py — THỬ NGHIỆM 3: áp 7 hướng định trước (H0..H6) + 1 tham chiếu lên 8 bộ; độ đúng HAI VẾ của phần giữ.

0 API, CPU, repo CHỈ ĐỌC (ghi: lab/thu_nghiem_anh_chu/TN3_tat_ca_bo/ và measure_out/_thu_nghiem_anh_chu/TN3/).
Không dùng nhãn pipeline làm sự thật. Không chọn ngưỡng nào ở đây: mọi ngưỡng lấy từ TN1 (τ = 0,995 đăng ký trước, LOBO
trên IHR) và TN2 (θ LOBO). Không mở ảnh.

BỘ (8): stt2, stt4, stt11 (SachThanhTruyen tách theo cell_uid), Chr, L83, KVK, L16, TK.
HƯỚNG (định trước, viết trước khi tính):
  H0 giữ nguyên GOLD.
  H1 cổng hình học = ¬(A0 ∪ B0 ∪ CNT ∪ BC): A0/B0/CNT như TN1 (B0 = cờ "một chữ": blank/truncated/dup/ov_heavy/tight_nb/
     bleed/tall; CNT = cột có số chữ OCR ≠ số âm QN); BC = cờ trượt vòng 2 chọn NGOÀI sách: L16 dùng bc_k06v1 (chọn trên TK,
     = G6 của TN1), mọi bộ khác dùng bc_k05dxv05 (chọn trên L16, = G7 của TN1). (G7 nguyên văn trên L16 là chọn trên chính
     L16 → chỉ báo kèm làm độ nhạy.)
  H2 H1 ∧ bộ kiểm ảnh ở τ = 0,995 (ngưỡng TN1/v3): L16 p_wood_T∧viss_T (C5 chọn trên TK); TK p_wood_L∧viss_L (C5 chọn
     trên L16); L83/KVK p_wood_T∧p_wood_L∧viss_X; Chr p_wood_T∧p_wood_L (C2); STT bộ kiểm viết tay LOBO r5
     (lobo_cert_00015 ∧ ≥3 nguyên mẫu người).
  H3 H1 ∧ nhãn được văn bản NGƯỜI của dị bản chứng (ta = attested) — chỉ ở TK (Kiều 1871 LVD), KVK (1871/1872), L83 (L16
     IHR); bộ không có văn bản người của dị bản: H3 ≡ H1.
  H4 H1 ∧ H2 ∧ H3 (= cấu hình (e) của TN1).
  H5 H1 ∪ ô được vòng sửa hộp TN2 nhận (luật trùng hộp NGHIÊM) và không dính A0/B0/CNT; ô được sửa dùng crop mới.
  H6 H4 ∪ ô được sửa (như H5) ∧ ta = attested ở bộ có dị bản; kiểm ảnh của ô được sửa = bộ kiểm của chính vòng sửa (m ≥ θ).
  Rv3 (THAM CHIẾU, không xếp hạng): H4 ∧ ¬luật A (ảnh ô khác/rescue/cầu tự dạng/văn bản yếu) ∧ ¬M-OCR t50 = luật cấp ô
     của chính sách v3 khi BỎ các cổng cấp bộ (stt_ok_allowed, chr_no_ok, TA4 của L83).
SỰ THẬT / ƯỚC LƯỢNG (xem KET_QUA.md §1):
  L16, TK  — ĐO trên nhãn người IHR: đúng hai vế = V1+(nhãn, chữ người) ∧ slot_ok = 1 (khe chưa xác định = sai);
             CI bootstrap cụm theo trang B = 2000.                                             CHẮC CHẮN THEO MÁY
  L83, KVK — vế chữ: chuyển tỉ lệ lỗi nhãn theo lớp (attested/contradicted/unattestable) đo trên TK ở CÙNG hướng
             (bootstrap trang TK); cận bi quan = max(CI trên, cận "không cần d" với q cận trên CI (KVK hai tham chiếu:
             1−(1−q)²), tổng p M-OCR vòng 2). Vế ảnh: p trượt từng ô của mô hình vòng 2 (gate_v2, hiệu chuẩn trên IHR,
             bootstrap có sai số chuyển bộ); hướng có bộ kiểm ảnh: × tỉ lệ nhận trượt của bộ kiểm đo trên IHR (ô trượt
             sống sót H1), cận bi quan = không cho bộ kiểm điểm nào.                          ƯỚC LƯỢNG
  Chr, stt* — vế ảnh như trên (STT: bể trượt 1.630–2.500 của vòng 4–5 phân bổ theo p trượt vòng 2; bộ kiểm viết tay:
             tỉ lệ nhận trượt thật 11,3 % [6,8–15,7] đo trên Borg); vế chữ: tỉ lệ lỗi nhãn đo trên IHR ở cùng hướng
             (bi quan = max của L16/TK/L83/KVK), STT cộng thêm p cầu tự dạng + nhãn rescue (vòng 2) và bể "Hán hoá" lt1
             750 [481–954] phân bổ đều trên 4.750 ô âm Borg-'nom'.                            SUY ĐOÁN
  Hai vế ngoài IHR: số lỗi = lỗi ảnh + lỗi chữ (cộng, cận hợp); khoảng = cộng hai đầu tương ứng.
TIÊU CHÍ ĐẠT (định trước): đo được → điểm ≥ 99,5 % (ghi "chắc" nếu cận dưới CI ≥ 99,5 %); ước lượng/suy đoán → cận bi
  quan ≥ 99,0 %; và giữ ≥ 50 ô. Hướng tốt nhất của bộ = hướng đạt giữ nhiều ô nhất.
Ra: measure_out/_thu_nghiem_anh_chu/TN3/{cells_masks.pkl}; lab/.../TN3_tat_ca_bo/{matrix.csv, t02_summary.json}
Chạy: .venv/bin/python lab/thu_nghiem_anh_chu/TN3_tat_ca_bo/t02_matrix.py [--B-img 1000]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

T0 = time.time()
ap = argparse.ArgumentParser()
ap.add_argument('--B-img', type=int, default=1000)
ap.add_argument('--B-ihr', type=int, default=2000)
ap.add_argument('--B-tk', type=int, default=2000)
ARGS = ap.parse_args()
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
BIG = REPO / 'measure_out/_thu_nghiem_anh_chu/TN3'
sys.path.insert(0, str(HERE))
from ver import BIG as TN4BIG, LABOUT, VER  # noqa
BIGO = TN4BIG / 'tn3m'; BIGO.mkdir(parents=True, exist_ok=True)
TN1 = REPO / 'lab/thu_nghiem_anh_chu/TN1_cong_kiem'
sys.path.insert(0, str(SP / 'r6/policy_v3/scripts'))
from vlib import var_eq_plus  # noqa
sys.path.insert(0, str(SP / 'gold_img_audit/gate_v2'))
import gate_v2 as G2  # noqa  (chỉ dùng hàm; không chạy main)

SEED = 20260926
INV = {}
SETS8 = ['stt2', 'stt4', 'stt11', 'Chr', 'L83', 'KVK', 'L16', 'TK']
STT3 = ['stt2', 'stt4', 'stt11']
TYPE = {'stt2': 'chép tay (lt1)', 'stt4': 'chép tay (lt1)', 'stt11': 'chép tay (lt1)', 'Chr': 'in văn xuôi',
        'L83': 'thạch bản', 'KVK': 'thạch bản', 'L16': 'mộc bản IHR', 'TK': 'mộc bản IHR'}
TRUTH = {s: 'SUY ĐOÁN' for s in STT3 + ['Chr']}; TRUTH.update(L83='ƯỚC LƯỢNG', KVK='ƯỚC LƯỢNG', L16='ĐO', TK='ĐO')
AB = {'Chrestomathie1872': 'Chr', 'LucVanTien1883': 'L83', 'KimVanKieu1884': 'KVK', 'LucVanTien1916': 'L16', 'TruyenKieu1872': 'TK'}
DIRS = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'Rv3']
DIR_NAME = {'H0': 'giữ nguyên GOLD', 'H1': 'cổng hình học', 'H2': 'H1 + bộ kiểm ảnh τ=0,995', 'H3': 'H1 + dị bản người',
            'H4': 'H1+H2+H3', 'H5': 'H1 + vòng sửa hộp', 'H6': 'H4 + vòng sửa hộp',
            'Rv3': 'tham chiếu: H4 + luật A + M-OCR (luật ô v3, bỏ cổng bộ)'}
NOTA = {'H0': 'H0', 'H1': 'H1', 'H2': 'H2', 'H3': 'H1', 'H4': 'H2', 'H5': 'H5', 'H6': 'H6nt', 'Rv3': 'Rv3nt'}   # bản bỏ điều kiện dị bản
CHECK = {'H2': 'H1', 'H4': 'H3', 'H6': 'H3', 'Rv3': 'Rv3nc'}          # hướng có bộ kiểm ảnh -> bản bỏ bộ kiểm (cho điểm bộ kiểm)

# ============================================================ dữ liệu + cờ (y hệt TN1)
X = pd.read_pickle(SP / 'r6/policy_v3/out/base_v2.pkl')
V3 = pd.read_pickle(SP / 'r6/policy_v3/out/v02_cells_0.995.pkl')[['cell_uid', 'gold_exact']]
X = X.merge(V3, on='cell_uid', how='left', validate='1:1')
n = len(X)
bk = X.cell_uid.str.split('/').str[1].values
S8 = np.where(X.book_set.values == 'SachThanhTruyen', bk, X.book_set.map(AB).fillna('').values).astype(object)
INV['sets8_cover_all'] = bool(set(S8) == set(SETS8))
b = lambda c: X[c].fillna(False).astype(bool).values
cqf = X.crop_quality_flag.fillna('').values
A0 = b('int_foreign') | b('blank') | b('truncated') | b('geo_f_dup_bbox')
B0 = ((cqf == 'blank') | b('blank') | (cqf == 'truncated') | b('truncated') | b('geo_f_dup_bbox') | b('geo_f_ov_heavy')
      | b('f_tight_nb') | (cqf == 'bleed') | (X.seg_flag.fillna('').astype(str).values == 'tall'))
CNT = b('geo_f_cnt_ocr_ne_qn')
AINT = b('int_foreign') | b('rescue') | b('similar') | b('weak_text')
MOCR = X.r2_reason.fillna('').values == 'M_ocr_t50'
ta = X.ta.fillna('na').values
TA_SETS = np.isin(S8, ['TK', 'KVK', 'L83'])
TA_OK = ~TA_SETS | (ta == 'attested')
ady, adx, visz = X.ady.values.astype(float), X.adx.values.astype(float), X.vis_z.values.astype(float)
visf = lambda v: np.nan_to_num(visz, nan=np.inf) <= v + 1e-12
BC_T = (np.nan_to_num(ady, nan=-1) > 0.6) | visf(-1.0)
BC_L = (np.nan_to_num(ady, nan=-1) > 0.5) | (np.nan_to_num(adx, nan=-1) > 0.5) | visf(-0.5)
INV['bc_T_eq_k06v1'] = bool((BC_T == b('bc_k06v1')).all()); INV['bc_L_eq_k05dxv05'] = bool((BC_L == b('bc_k05dxv05')).all())
isL16, isTK = S8 == 'L16', S8 == 'TK'
LITHO, CHR, STT = np.isin(S8, ['L83', 'KVK']), S8 == 'Chr', np.isin(S8, STT3)
# ngưỡng τ = 0,995 (TN1, tái lập p02b/v3)
T1S = json.load(open(TN1 / 't01_summary.json'))
V3S = json.load(open(SP / 'r6/policy_v3/summary.json'))
thT, thL = T1S['thresholds']['T_0.995'], T1S['thresholds']['L_0.995']
C5T, C5L, C2T, C2L = thT['C5'], thL['C5'], thT['C2'], thL['C2']
pre = V3S['thresholds']['prereg_0995']
INV['thresholds_eq_v3_prereg'] = bool(np.allclose(C5T, pre['C5_T']) and np.allclose(C5L, pre['C5_L'])
                                      and abs(C2T - pre['C2_T']) < 1e-9 and abs(C2L - pre['C2_L']) < 1e-9)
pwT, pwL = X.p_wood_T.values.astype(float), X.p_wood_L.values.astype(float)
vsT, vsL, vX = (X[c].fillna(0).values.astype(float) for c in ('viss_T', 'viss_L', 'viss_X'))
hand = (X.lobo_cert_00015.fillna(0).astype(int).values == 1) & (X.lobo_nh.fillna(0).values >= 3)
SIMG = np.zeros(n, bool)
SIMG[isL16] = ((pwT >= C5T[0]) & (vsT >= C5T[1]))[isL16]
SIMG[isTK] = ((pwL >= C5L[0]) & (vsL >= C5L[1]))[isTK]
SIMG[LITHO] = ((pwT >= C5T[0]) & (pwL >= C5L[0]) & (vX >= max(C5T[1], C5L[1])))[LITHO]
SIMG[CHR] = ((pwT >= C2T) & (pwL >= C2L))[CHR]
SIMG[STT] = hand[STT]
BCd = np.where(isL16, BC_T, BC_L)
H1 = ~(A0 | B0 | CNT | BCd)
FD = pd.read_csv(BIG / 'fix_decisions.csv', dtype=str, keep_default_na=False)
fixm = dict(zip(FD.cell_uid, FD.fix.astype(int)))
nsl = dict(zip(FD.cell_uid, FD.new_slot_ok))
FIX = np.array([fixm.get(u, 0) == 1 for u in X.cell_uid])
FIXOK = FIX & ~(A0 | B0 | CNT)
M = {'H0': np.ones(n, bool), 'H1': H1, 'H2': H1 & SIMG, 'H3': H1 & TA_OK, 'H4': H1 & SIMG & TA_OK,
     'H5': H1 | FIXOK}
M['H6'] = M['H4'] | (FIXOK & TA_OK)
M['Rv3'] = M['H4'] & ~(AINT | MOCR)
M['H6nt'] = M['H2'] | FIXOK
M['Rv3nt'] = M['H2'] & ~(AINT | MOCR)
M['Rv3nc'] = M['H3'] & ~(AINT | MOCR)
M['V3'] = X.gold_exact.values == 'ok'
M['H1g7'] = ~(A0 | B0 | CNT | BC_L)          # độ nhạy: G7 nguyên văn (trên L16 là chọn trong mẫu)
# === TN4: mặt nạ crop chuẩn (s06a)
_MN = pd.read_pickle(TN4BIG / 'masks_new.pkl'); assert (_MN.cell_uid.values == X.cell_uid.values).all()
import os as _os
_SIMGC = _os.environ.get('TN4_SIMG', 'SIMGn'); _SUFM = '' if _SIMGC == 'SIMGn' else '_lai'
H1 = _MN.H1n.values; SIMG = _MN[_SIMGC].values
M['H1'] = H1; M['H2'] = H1 & SIMG; M['H3'] = H1 & TA_OK; M['H4'] = H1 & SIMG & TA_OK; M['H5'] = H1 | FIXOK
M['H6'] = M['H4'] | (FIXOK & TA_OK); M['Rv3'] = M['H4'] & ~(AINT | MOCR); M['H6nt'] = M['H2'] | FIXOK
M['Rv3nt'] = M['H2'] & ~(AINT | MOCR); M['Rv3nc'] = M['H3'] & ~(AINT | MOCR); M['H1g7'] = H1
USEFIX = {'H5', 'H6', 'H6nt'}                # hướng dùng crop mới cho ô được sửa
# ---- invariants: khớp TN1 / v3
PJ = pd.read_csv(TN1 / 'projection.csv')
pj = lambda s, c: int(PJ[(PJ.sach == s) & (PJ.config == c)].giu.iloc[0])
for bs, a in [('SachThanhTruyen', 'STT'), ('Chrestomathie1872', 'Chr'), ('LucVanTien1883', 'L83'), ('KimVanKieu1884', 'KVK'),
              ('LucVanTien1916', 'L16'), ('TruyenKieu1872', 'TK')]:
    mm = X.book_set.values == bs
    INV[f'H2_eq_TN1_d_{a}'] = int((M['H2'] & mm).sum()) == pj(a, 'd')
    INV[f'H4_eq_TN1_e_{a}'] = int((M['H4'] & mm).sum()) == pj(a, 'e')
    INV[f'V3_eq_v3_ok_{a}'] = int((M['V3'] & mm).sum()) == V3S['per_set'][bs]['ok']
    if a != 'L16':
        INV[f'H1_eq_TN1_b_G7_{a}'] = int((M['H1'] & mm).sum()) == pj(a, 'b')
INV['H1_L16_eq_ladder_G6'] = True   # kiểm dưới (đo)
INV['fix_cells_in_gold'] = int(FIX.sum()) == int(FD.fix.astype(int).sum())

# ============================================================ (A) ĐO trên IHR (L16, TK)
gt = X.gt_char.fillna('').values
HAS = np.isin(S8, ['L16', 'TK']) & (gt != '')
lab = X.label.values
y_lab = np.zeros(n, bool)
y_lab[HAS] = [var_eq_plus(a, c) for a, c in zip(lab[HAS], gt[HAS])]
slot = _MN.slot_new.astype(str).values   # TN4: khe của crop mới
new_slot = np.array([nsl.get(u, '') for u in X.cell_uid], dtype=object)
INV['ihr_fix_all_have_new_slot'] = bool(((FIX & HAS) <= (new_slot != '')).all())
slot_fix = np.where(FIX & (new_slot != ''), new_slot, slot)
page = X.page.fillna('').values
rng = np.random.default_rng(SEED)
BIDX = {}
for bkn in ('L16', 'TK'):
    te = HAS & (S8 == bkn)
    up, inv_ = np.unique(page[te], return_inverse=True)
    BIDX[bkn] = (te, inv_, len(up), rng.integers(0, len(up), (ARGS.B_ihr, len(up))))


def measure_ihr(bkn, keep, use_fix):
    te, inv_, npg, idx = BIDX[bkn]
    sl = slot_fix if use_fix else slot
    k = keep[te]; okimg = (sl[te] == '1'); oklab = y_lab[te]
    arrs = [k, k & ~(oklab & okimg), k & ~oklab, k & ~okimg]
    P = np.stack([np.bincount(inv_, weights=a.astype(float), minlength=npg) for a in arrs])   # 4 × trang
    tot = P.sum(1)
    r = dict(n_eval=int(tot[0]))
    if tot[0] == 0:
        return r
    BS = P[:, idx].sum(2)                       # 4 × B
    for j, nm in ((1, 'both'), (2, 'lab'), (3, 'img')):
        r[f'{nm}_pt'] = 1 - tot[j] / tot[0]
        pb = 1 - BS[j] / np.maximum(BS[0], 1)
        r[f'{nm}_lo'], r[f'{nm}_hi'] = float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))
        r[f'{nm}_err'] = int(tot[j])
    return r


IHRM = {}
for bkn in ('L16', 'TK'):
    for d in DIRS + ['H6nt', 'Rv3nt', 'H1g7', 'V3']:
        IHRM[(bkn, d)] = measure_ihr(bkn, M[d], d in USEFIX)
# invariants: tái lập TN1 (L16 H1 = G6 98,06 %; H2 = (d); H4 = (e); TK H1 = G7 98,85 %)
LB = pd.read_csv(TN1 / 'ladder_b.csv'); MT = pd.read_csv(TN1 / 'main_table.csv')
lb = lambda s, g: float(LB[(LB.sach == s) & (LB.bac == g)].prec_both.iloc[0])
mt = lambda s, c: float(MT[(MT.bao_tren == s) & (MT.config == c)].prec_both.iloc[0])
INV['H1_L16_eq_ladder_G6'] = abs(IHRM[('L16', 'H1')]['both_pt'] - lb('L16', 'G6')) < 1e-4
INV['H1g7_L16_eq_ladder_G7'] = abs(IHRM[('L16', 'H1g7')]['both_pt'] - lb('L16', 'G7')) < 1e-4
INV['H1_TK_eq_ladder_G7'] = abs(IHRM[('TK', 'H1')]['both_pt'] - lb('TK', 'G7')) < 1e-4
for s in ('L16', 'TK'):
    INV[f'H2_{s}_eq_TN1_d'] = abs(IHRM[(s, 'H2')]['both_pt'] - mt(s, 'd')) < 1e-4
    INV[f'H4_{s}_eq_TN1_e'] = abs(IHRM[(s, 'H4')]['both_pt'] - mt(s, 'e')) < 1e-4
    INV[f'H0_{s}_eq_TN1_a'] = abs(IHRM[(s, 'H0')]['both_pt'] - mt(s, 'a')) < 1e-4
    INV[f'Rv3_{s}_eq_TN1_f'] = abs(IHRM[(s, 'Rv3')]['both_pt'] - mt(s, 'f')) < 1e-4

# ---- tỉ lệ nhận trượt của bộ kiểm ảnh sách in (IHR, ô trượt/khe chưa xđ SỐNG SÓT H1, ngưỡng LOBO của từng sách)
wr = HAS & M['H1'] & (slot != '1')
FAR_k, FAR_n = int((wr & SIMG).sum()), int(wr.sum())
FAR_print = dict(k=FAR_k, n=FAR_n, p=FAR_k / FAR_n,
                 by_book={s: [int((wr & SIMG & (S8 == s)).sum()), int((wr & (S8 == s)).sum())] for s in ('L16', 'TK')})
# ---- ô được sửa: tỉ lệ khe mới sai đo trên L16 (TK: 0 ô sửa)
fx_e = FIX & HAS
FIXR = dict(k_wrong=int((fx_e & (new_slot != '1')).sum()), n=int(fx_e.sum()))
bb = V3S['borg_B0_test']['realslip']['joint_notB0_raw']
FAR_hand = dict(p=bb['realslip_accept'], ci=bb['ci'], nguon='v3 summary borg_B0_test.realslip.joint_notB0_raw (Borg, LOBO)')
POOL_STT = V3S['borg_B0_test']['stt_residual']['pool']          # [1630, 2500]
INV['stt_pool_is_1630_2500'] = list(POOL_STT) == [1630, 2500]
HH_POP = np.isin(S8, STT3) & (X.borg_class.fillna('').values == 'nom')
INV['hanhoa_pop_4750'] = int(HH_POP.sum()) == 4750
HH = dict(point=750, lo=481, hi=954, n_pop=4750, nguon='r4 BAO_CAO_ANH_CHU §1 (mô hình trộn, CÓ THỂ); r5 v2 §1.3')

# ============================================================ (B) mô hình ảnh vòng 2 (gate_v2), p từng ô + bootstrap
XG = G2.load()
ix = pd.Index(XG.cell_uid).get_indexer(X.cell_uid)
INV['gate_v2_align'] = bool((ix >= 0).all())
Z, _ = G2.design(XG)
pooled = (XG.D2 & XG.book_set.isin(G2.IHR)).values
mo, br = G2.MOCR(XG), G2.Bridge()
oos = []
for t in ('U', 'I'):
    for src, tgt in (('LucVanTien1916', 'TruyenKieu1872'), ('TruyenKieu1872', 'LucVanTien1916')):
        ms = (XG.D2 & (XG.book_set == src)).values; mt_ = (XG.D2 & (XG.book_set == tgt)).values
        bta = G2.fit_logit(Z[ms], XG[t].values[ms].astype(float))
        oos.append(np.log(float(G2.pred(Z[mt_], bta).sum()) / int(XG[t].values[mt_].sum())))
SIGMA = float(np.sqrt(np.mean(np.square(oos))))
beta_U = G2.fit_logit(Z[pooled], XG.U.values[pooled].astype(float))
pm_pt, _ = G2.cell_probs(XG, Z, beta_U, 'U', mo, br)
IMGK = ['truot', 'crop_rong', 'rescue_anh_o_khac', 'khac']
p_img_pt = sum(pm_pt[k] for k in IMGK)[ix]
p_tr_pt = pm_pt['truot'][ix]
p_lx_pt = (pm_pt['cau_tu_dang'] + pm_pt['nhan_rescue'])[ix]
p_mo_pt = pm_pt['m_ocr'][ix]
GJ = json.load(open(SP / 'gold_img_audit/gate_v2/out/gate_v2.json'))
for bs, a in [('SachThanhTruyen', 'STT'), ('Chrestomathie1872', 'Chr'), ('LucVanTien1883', 'L83'), ('KimVanKieu1884', 'KVK')]:
    mm = X.book_set.values == bs
    INV[f'gate_v2_truot_U_reproduce_{a}'] = abs(p_tr_pt[mm].sum() - GJ['dedup']['truot'][a]['U']) < 0.2
INV['gate_v2_sigma'] = round(SIGMA, 3)
INV['gate_v2_sigma_eq'] = abs(SIGMA - GJ['slip_model']['transfer_sigma_log']) < 1e-3

# ---- ma trận mặt nạ cho bộ không có sự thật (tổng Σp qua phép nhân ma trận)
NONIHR = ['stt2', 'stt4', 'stt11', 'Chr', 'L83', 'KVK']
ROWS = []          # (set, dir, kind) ; kind: keep_nf = giữ & ¬sửa ; nc_nf = bản bỏ bộ kiểm & ¬sửa
MAT = []
for s in NONIHR:
    sm = S8 == s
    for d in DIRS:
        fixpart = FIX if d in ('H5', 'H6') else np.zeros(n, bool)
        ROWS.append((s, d, 'keep_nf')); MAT.append(M[d] & sm & ~fixpart)
        if d in CHECK:
            ROWS.append((s, d, 'nc_nf')); MAT.append(M[CHECK[d]] & sm & ~fixpart)
MAT = np.stack(MAT).astype(np.float32)
RI = {r: i for i, r in enumerate(ROWS)}
stt_mask = STT.astype(np.float32)


def img_sums(p_img, p_tr, pool=None):
    """Σ p ảnh theo hàng; STT: thay p trượt bằng bể trượt × tỉ trọng p trượt."""
    p = p_img.copy()
    if pool is not None:
        tot = float(p_tr[STT].sum())
        p[STT] = (p_img[STT] - p_tr[STT]) + p_tr[STT] * (pool / tot)
    return MAT @ p.astype(np.float32)


pool_pt = float(np.mean(POOL_STT))
IMG_PT = img_sums(p_img_pt, p_tr_pt, pool_pt)
IMG_PT_raw = img_sums(p_img_pt, p_tr_pt, None)          # STT theo mô hình vòng 2 (không dùng bể vòng 4) — độ nhạy
p_hh_pt = np.where(HH_POP, HH['point'] / HH['n_pop'], 0.0)
LX_PT = MAT @ (1 - (1 - p_lx_pt) * (1 - p_hh_pt)).astype(np.float32)
MO_PT = MAT @ p_mo_pt.astype(np.float32)
# bootstrap (như gate_v2: U/I luân phiên, trọng số Poisson IHR, sai số chuyển bộ log-chuẩn σ, bậc cầu, nhị thức M-OCR)
rngb = np.random.default_rng(SEED + 1)
BI, BLX, BMO, BFAR, BFH, BFIX = [], [], [], [], [], []
tb = time.time()
for bi in range(ARGS.B_img):
    t = 'U' if bi % 2 == 0 else 'I'
    w = rngb.poisson(1.0, pooled.sum()).astype(float)
    bta = G2.fit_logit(Z[pooled], XG[t].values[pooled].astype(float), w=w)
    tf = float(np.exp(rngb.normal(0, SIGMA)))
    bm = [0.0, 0.02, 0.05][rngb.integers(3)]
    pm, _ = G2.cell_probs(XG, Z, bta, t, mo, br, rng=rngb, transfer=tf, bridge_m=bm)
    pi_ = sum(pm[k] for k in IMGK)[ix]; ptr = pm['truot'][ix]
    BI.append(img_sums(pi_, ptr, float(rngb.uniform(*POOL_STT))))
    phh = np.where(HH_POP, rngb.uniform(HH['lo'], HH['hi']) / HH['n_pop'], 0.0)
    BLX.append(MAT @ (1 - (1 - (pm['cau_tu_dang'] + pm['nhan_rescue'])[ix]) * (1 - phh)).astype(np.float32))
    BMO.append(MAT @ pm['m_ocr'][ix].astype(np.float32))
    BFAR.append(rngb.beta(FAR_k + 0.5, FAR_n - FAR_k + 0.5))
    BFH.append(max(0.0, rngb.normal(FAR_hand['p'], (FAR_hand['ci'][1] - FAR_hand['ci'][0]) / 3.92)))
    BFIX.append(rngb.beta(FIXR['k_wrong'] + 0.5, FIXR['n'] - FIXR['k_wrong'] + 0.5))
BI, BLX, BMO = np.stack(BI), np.stack(BLX), np.stack(BMO)
BFAR, BFH, BFIX = np.array(BFAR), np.array(BFH), np.array(BFIX)
T_BOOT = round(time.time() - tb, 1)

# ============================================================ (C) vế chữ thạch bản: chuyển tỉ lệ theo lớp từ TK (cùng hướng)
TKm = HAS & isTK
CLS = ['attested', 'contradicted', 'unattestable', 'na']
tk_up, tk_inv = np.unique(page[TKm], return_inverse=True)
tk_idx = np.random.default_rng(SEED + 2).integers(0, len(tk_up), (ARGS.B_tk, len(tk_up)))
wrong_tk = ~y_lab[TKm]; cls_tk = ta[TKm]
TKR = {}      # d -> class -> (pt, boot array)
for d in DIRS + ['H6nt', 'Rv3nt']:
    k = M[d][TKm]
    TKR[d] = {}
    for c in CLS:
        mk = k & (cls_tk == c)
        W = np.bincount(tk_inv, weights=(mk & wrong_tk).astype(float), minlength=len(tk_up))
        N = np.bincount(tk_inv, weights=mk.astype(float), minlength=len(tk_up))
        TKR[d][c] = (W.sum(), N.sum(), W[tk_idx].sum(1), N[tk_idx].sum(1))


def tk_rate(d, c):
    """(điểm, mảng bootstrap, nguồn) — lùi về H1 rồi H0 nếu TK giữ < 30 ô của lớp."""
    for dd in (d, 'H1', 'H0'):
        w, nn, wb, nb = TKR[dd][c]
        if nn >= 30:
            return w / nn, wb / np.maximum(nb, 1), dd
    return 0.0, np.zeros(ARGS.B_tk), 'none'


TAP = V3S  # noqa
TA_SUM = json.load(open(SP / 'r5/text_attested/out/summary.json'))
q_pt = TA_SUM['reverse_TK_params']['point']['GOLD']['q']; q_hi = TA_SUM['reverse_TK_params']['ci']['GOLD']['q'][1]
QREF = {'L83': q_hi, 'KVK': 1 - (1 - q_hi) ** 2, 'TK': q_hi}      # KVK: hai tham chiếu (cận trên hợp khi tương quan dương)


def nod_bound(s_mask, d, qq):
    """cận 'không cần d' (d = 0) cho số nhãn sai trong phần giữ của hướng d."""
    P = M[NOTA[d]] & s_mask
    na_, nc_ = int((P & (ta == 'attested')).sum()), int((P & (ta == 'contradicted')).sum())
    K = M[d] & s_mask
    ka, kc, ku = (int((K & (ta == c)).sum()) for c in ('attested', 'contradicted', 'unattestable'))
    if na_ + nc_ == 0:
        return dict(c=None, bound=float(ka + kc + ku))
    c = nc_ / (na_ + nc_)
    emax = min(1.0, c / (1 - qq)); patt = min(1.0, emax * qq / (1 - c)) if c < 1 else 1.0
    return dict(c=round(c, 5), e_max=round(emax, 5), p_att_max=round(patt, 5),
                bound=float(ka * patt + min(kc, emax * (ka + kc)) + ku * emax))


# kiểm định cận trên TK: cận ≥ lỗi nhãn đo được ở mọi hướng
NODV = {}
for d in DIRS:
    nb_ = nod_bound(isTK & HAS, d, QREF['TK'])
    meas = int((M[d] & HAS & isTK & ~y_lab).sum())
    NODV[d] = dict(bound=round(nb_['bound'], 1), measured=meas, c=nb_.get('c'), valid=bool(nb_['bound'] >= meas))
INV['nod_bound_valid_on_TK_all_dirs'] = all(v['valid'] for v in NODV.values())

# ============================================================ (D) ghép ma trận 8 bộ × hướng
classes_before = {s: set(lab[S8 == s]) for s in SETS8}


def lost_classes(s, keep):
    sm = S8 == s
    kept = set(lab[sm & keep])
    lost = classes_before[s] - kept
    return len(lost), int(np.isin(lab[sm], list(lost)).sum())


def q(a, p):
    return float(np.percentile(a, p))


rows = []
ORDER = ['L16', 'TK', 'L83', 'KVK', 'Chr', 'stt2', 'stt4', 'stt11']     # thạch bản trước Chr/STT (cận bi quan dùng số thạch bản)
LITHO_LOOKUP = {'H6nt': 'H2', 'Rv3nt': 'H2'}                          # hướng không có hàng riêng ở thạch bản -> dùng H2
for s in ORDER:
    sm = S8 == s
    gold = int(sm.sum())
    for d in DIRS:
        keep = M[d] & sm
        nk = int(keep.sum())
        cl, cl_cells = lost_classes(s, M[d])
        r = dict(set=s, type=TYPE[s], truth=TRUTH[s], dir=d, dir_name=DIR_NAME[d], gold=gold, keep=nk,
                 keep_pct=round(100 * nk / gold, 2), demote=gold - nk, classes_before=len(classes_before[s]),
                 classes_lost=cl, cells_in_lost_classes=cl_cells, n_fix_kept=int((keep & FIX).sum()) if d in USEFIX else 0)
        if s in ('L16', 'TK'):
            m_ = IHRM[(s, d)]
            r.update(n_eval=m_['n_eval'])
            for nm in ('both', 'lab', 'img'):
                for e in ('pt', 'lo', 'hi'):
                    r[f'{nm}_{e}'] = m_.get(f'{nm}_{e}')
            r['err_lab_cells'] = m_.get('lab_err'); r['err_img_cells'] = m_.get('img_err'); r['err_both_cells'] = m_.get('both_err')
            r['acc_pess'] = m_.get('both_lo')
        elif nk > 0:
            # ---- vế ảnh
            i_k = RI[(s, d, 'keep_nf')]
            nfix = r['n_fix_kept']
            fix_pt = nfix * FIXR['k_wrong'] / max(FIXR['n'], 1)
            fix_b = nfix * BFIX
            if d in CHECK:
                i_nc = RI[(s, d, 'nc_nf')]
                far_b = BFH if s in STT3 else BFAR
                far_pt = FAR_hand['p'] if s in STT3 else FAR_print['p']
                img_b = BI[:, i_nc] * far_b + fix_b
                img_pt = float(IMG_PT[i_nc] * far_pt + fix_pt)
                nocredit_b = BI[:, i_k] + fix_b
                img_hi = max(q(img_b, 97.5), q(nocredit_b, 97.5))
                r['img_method'] = f"Σp(bản bỏ bộ kiểm) × tỉ lệ nhận trượt {'viết tay Borg' if s in STT3 else 'IHR'}; bi quan = không tính bộ kiểm"
            else:
                img_b = BI[:, i_k] + fix_b
                img_pt = float(IMG_PT[i_k] + fix_pt)
                img_hi = q(img_b, 97.5)
                r['img_method'] = 'Σp trượt vòng 2' + (' (STT: bể 1.630–2.500)' if s in STT3 else '')
            img_lo = q(img_b, 2.5)
            r['img_err_pt'], r['img_err_lo'], r['img_err_hi'] = img_pt, img_lo, img_hi
            r['img_err_stt_model_only'] = float(IMG_PT_raw[i_k]) if s in STT3 else None
            # ---- vế chữ
            mo_b = BMO[:, i_k]
            if s in ('L83', 'KVK'):
                Kc = {c: int((keep & (ta == c)).sum()) for c in CLS}
                pt_, arr = 0.0, np.zeros(ARGS.B_tk)
                src = {}
                for c, nc in Kc.items():
                    if nc == 0:
                        continue
                    rp, rb, sd = tk_rate(d, c)
                    pt_ += nc * rp; arr = arr + nc * rb; src[c] = sd
                nb_ = nod_bound(sm, d, QREF[s])
                lab_pt, lab_lo = pt_, q(arr, 2.5)
                cand = {'T_ci_tren': q(arr, 97.5), 'can_khong_can_d': nb_['bound'], 'M_OCR_v2_ci_tren': q(mo_b, 97.5)}
                lab_hi = max(cand.values())
                r['lab_hi_cands'] = json.dumps({k: round(v, 1) for k, v in cand.items()})
                r['lab_err_hi_no_nod'] = max(cand['T_ci_tren'], cand['M_OCR_v2_ci_tren'])   # ĐỘ NHẠY thêm SAU khi thấy kết quả
                r['lab_method'] = 'lớp dị bản × tỉ lệ lỗi TK cùng hướng; bi quan = max(CI trên, cận không-d, M-OCR)'
                r['lab_pess_driver'] = max(cand, key=cand.get)
                r['lab_classes'] = json.dumps(Kc); r['lab_tk_src'] = json.dumps(src)
                r['nod_c'] = nb_.get('c'); r['nod_bound'] = round(nb_['bound'], 1)
            else:        # Chr, STT: chuyển tỉ lệ đo trên IHR (hướng bỏ điều kiện dị bản)
                dn = NOTA[d]
                rL, rT = IHRM[('L16', dn)], IHRM[('TK', dn)]
                r_pt = np.mean([1 - rL['lab_pt'], 1 - rT['lab_pt']])
                r_lo = min(1 - rL['lab_hi'], 1 - rT['lab_hi'])
                litho_hi = []
                for s2 in ('L83', 'KVK'):
                    rr = [x for x in rows if x['set'] == s2 and x['dir'] == LITHO_LOOKUP.get(dn, dn)]
                    if rr and rr[0]['keep'] > 0:
                        litho_hi.append(rr[0]['lab_err_hi_T'] / rr[0]['keep'])
                r_hi = max([1 - rL['lab_lo'], 1 - rT['lab_lo']] + litho_hi)
                ex_pt = float(LX_PT[i_k]) if s in STT3 else 0.0
                ex_b = BLX[:, i_k] if s in STT3 else np.zeros(ARGS.B_img)
                lab_pt = r_pt * nk + (1 - r_pt) * ex_pt
                lab_lo = r_lo * nk + (1 - r_lo) * q(ex_b, 2.5)
                lab_hi = max(r_hi * nk + (1 - r_hi) * q(ex_b, 97.5), q(mo_b, 97.5))
                r['lab_method'] = ('tỉ lệ lỗi nhãn IHR cùng hướng (bi quan = max L16/TK/L83/KVK)'
                                   + (' + cầu tự dạng/rescue vòng 2 + bể Hán hoá lt1' if s in STT3 else ''))
                r['lab_rate_transfer'] = json.dumps([round(float(r_lo), 5), round(float(r_pt), 5), round(float(r_hi), 5)])
                r['lab_extra_stt'] = json.dumps([round(q(ex_b, 2.5), 1), round(float(ex_pt), 1), round(q(ex_b, 97.5), 1)]) if s in STT3 else None
                if s in STT3:   # H7 (chỉ mô tả): trần lợi ích của đọc lại lt2 = bỏ HẾT phần lỗi riêng STT (Hán hoá + cầu tự dạng + rescue)
                    r['lab_err_pt_H7'] = r_pt * nk; r['lab_err_hi_H7'] = max(r_hi * nk, q(mo_b, 97.5))
            r['lab_err_pt'], r['lab_err_lo'], r['lab_err_hi'] = float(lab_pt), float(lab_lo), float(lab_hi)
            if s in ('L83', 'KVK'):
                r['lab_err_hi_T'] = q(arr, 97.5)
            # ---- ghép hai vế (cộng số lỗi)
            for e in ('pt', 'lo', 'hi'):
                r[f'img_{e}'] = 1 - r[f'img_err_{e}'] / nk
                r[f'lab_{e}'] = 1 - r[f'lab_err_{e}'] / nk
            eb_pt = r['img_err_pt'] + r['lab_err_pt']; eb_lo = r['img_err_lo'] + r['lab_err_lo']; eb_hi = r['img_err_hi'] + r['lab_err_hi']
            r['both_pt'], r['both_lo'], r['both_hi'] = 1 - eb_pt / nk, max(0.0, 1 - eb_hi / nk), 1 - eb_lo / nk
            r['acc_pess'] = r['both_lo']
            # ---- ĐỘ NHẠY (thêm SAU khi thấy kết quả, KHÔNG dùng để xếp hạng):
            #  (i) thạch bản: bỏ cận "không cần d" khỏi cận bi quan (trên TK cận này lỏng ~60 lần so với lỗi đo, xem nod_bound_TK_check)
            #  (ii) STT: bể trượt KHÔNG được H1 giảm (phân bổ đều theo số ô, như quy tắc stt_ok_allowed của v3/r5)
            if s in ('L83', 'KVK'):
                r['acc_pess_no_nod'] = max(0.0, 1 - (r['img_err_hi'] + r['lab_err_hi_no_nod']) / nk)
            if s in STT3:
                n_stt_all = int(STT.sum())
                if d in CHECK:
                    n_nc = int(MAT[RI[(s, d, 'nc_nf')]].sum())
                    u_hi = POOL_STT[1] * n_nc / n_stt_all * FAR_hand['ci'][1]
                else:
                    u_hi = POOL_STT[1] * int(MAT[i_k].sum()) / n_stt_all
                r['img_err_hi_uniform'] = u_hi
                r['acc_pess_stt_uniform'] = max(0.0, 1 - (u_hi + r['lab_err_hi']) / nk)
                r['acc_pt_H7'] = 1 - (r['img_err_pt'] + r['lab_err_pt_H7']) / nk
                r['acc_pess_H7'] = max(0.0, 1 - (r['img_err_hi'] + r['lab_err_hi_H7']) / nk)
                r['acc_pess_H7_uniform'] = max(0.0, 1 - (u_hi + r['lab_err_hi_H7']) / nk)
            r['err_both_cells'] = round(eb_pt, 1); r['err_both_cells_hi'] = round(eb_hi, 1)
            # vế chữ/ảnh: lo/hi đảo cho độ đúng
            r['img_lo'], r['img_hi'] = 1 - r['img_err_hi'] / nk, 1 - r['img_err_lo'] / nk
            r['lab_lo'], r['lab_hi'] = 1 - r['lab_err_hi'] / nk, 1 - r['lab_err_lo'] / nk
        rows.append(r)
MX = pd.DataFrame(rows)
MX['_o'] = MX.set.map({x: i for i, x in enumerate(SETS8)}) * 100 + MX.dir.map({x: i for i, x in enumerate(DIRS)})
MX = MX.sort_values('_o').drop(columns='_o').reset_index(drop=True)

# ---- tiêu chí đạt + xếp hạng (định trước)
def meets(r):
    if r['keep'] < 50 or pd.isna(r.get('both_pt')):
        return 'khong'
    if r['truth'] == 'ĐO':
        if r['both_pt'] >= 0.995:
            return 'dat_chac' if r['both_lo'] >= 0.995 else 'dat_diem'
        return 'khong'
    return 'dat' if r['acc_pess'] >= 0.99 else 'khong'


MX['meets'] = [meets(r) for _, r in MX.iterrows()]
RANK = {}
for s in SETS8:
    sub = MX[(MX.set == s) & (MX.dir != 'Rv3')]
    ok = sub[sub.meets != 'khong'].sort_values(['keep', 'both_pt'], ascending=[False, False])
    best = ok.iloc[0] if len(ok) else None
    top_prec = sub.sort_values(['acc_pess', 'keep'], ascending=[False, False]).iloc[0]
    RANK[s] = dict(best=(best['dir'] if best is not None else None), best_keep=(int(best['keep']) if best is not None else 0),
                   dat=list(ok.dir), max_pess_dir=top_prec['dir'], max_pess=round(float(top_prec['acc_pess']), 5) if not pd.isna(top_prec['acc_pess']) else None,
                   rv3_meets=MX[(MX.set == s) & (MX.dir == 'Rv3')].meets.iloc[0])
MX.to_csv(LABOUT / f'tn3m_matrix{_SUFM}.csv', index=False)
np.save(BIGO / 'boot_img_sums.npy', BI)

# ---- kiểm định ngược mô hình ảnh trên IHR: học trượt trên 1 sách, dự Σp trong phần giữ của sách kia (H1, H2)
VAL = {}
for src, tgt, sb in (('LucVanTien1916', 'TruyenKieu1872', 'TK'), ('TruyenKieu1872', 'LucVanTien1916', 'L16')):
    ms = (XG.D2 & (XG.book_set == src)).values
    bta = G2.fit_logit(Z[ms], XG.U.values[ms].astype(float))
    pp = G2.pred(Z, bta)[ix]
    for d in ('H0', 'H1', 'H3'):
        kk = M[d] & HAS & (S8 == sb)
        VAL[f'{sb}|{d}'] = dict(du_sum_p=round(float(pp[kk].sum()), 1), that_U=int((kk & XG.U.values[ix]).sum()),
                                that_slot_ne_1=int((kk & (slot != '1')).sum()))
    kk = M['H1'] & HAS & (S8 == sb)
    VAL[f'{sb}|H2(Σp_H1×tỉ lệ nhận)'] = dict(du=round(float(pp[kk].sum()) * FAR_print['p'], 1),
                                            that_U=int((M['H2'] & HAS & (S8 == sb) & XG.U.values[ix]).sum()),
                                            that_slot_ne_1=int((M['H2'] & HAS & (S8 == sb) & (slot != '1')).sum()))

# ---- lưu mặt nạ từng ô (cho người kiểm / pipeline)
CM = pd.DataFrame({'cell_uid': X.cell_uid, 'set8': S8, 'label': lab, 'fix': FIX.astype(int)})
for d in DIRS:
    CM[f'keep_{d}'] = M[d].astype(np.int8)
CM.to_pickle(BIGO / 'cells_masks.pkl')

INV = {k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in INV.items()}
INV['all_pass'] = all(v for k, v in INV.items() if isinstance(v, bool))
SUM = dict(key='TN3_tat_ca_bo', date='2026-09-26', api_calls=0, seed=SEED, B=dict(img=ARGS.B_img, ihr=ARGS.B_ihr, tk=ARGS.B_tk),
           thresholds=dict(C5_T=C5T, C5_L=C5L, C2_T=C2T, C2_L=C2L, stt_q=0.00015, tn2_theta=json.load(open(HERE.parent / 'TN3_tat_ca_bo/t01_summary.json'))['theta_used']),
           far_print_ihr_H1_survivors=FAR_print, far_hand_borg=FAR_hand, fix_img_err_L16=FIXR, stt_slip_pool=POOL_STT,
           hanhoa_lt1=HH, q_TK=dict(point=q_pt, hi=q_hi, used=QREF, d_point=TA_SUM['reverse_TK_params']['point']['GOLD']['d']), gate_v2_sigma=SIGMA, nod_bound_TK_check=NODV,
           img_model_backtest_ihr=VAL, rank=RANK,
           L16_H1_G7_literal_in_sample={k: IHRM[('L16', 'H1g7')].get(k) for k in ('n_eval', 'both_pt', 'both_lo', 'both_hi')},
           ihr_H6nt_Rv3nt={f'{s_}|{d_}': {k: IHRM[(s_, d_)].get(k) for k in ('n_eval', 'both_pt', 'lab_pt', 'lab_lo', 'lab_hi')}
                           for s_ in ('L16', 'TK') for d_ in ('H6nt', 'Rv3nt')}, invariants=INV, boot_sec=T_BOOT, runtime_s=round(time.time() - T0, 1))
json.dump(SUM, open(LABOUT / f'tn3m_summary{_SUFM}.json', 'w'), ensure_ascii=False, indent=1, default=float)
pd.set_option('display.width', 250)
print(MX[['set', 'dir', 'keep', 'keep_pct', 'both_pt', 'both_lo', 'both_hi', 'meets']].to_string(index=False))
print(json.dumps(RANK, ensure_ascii=False))
print('INV', json.dumps({k: v for k, v in INV.items() if v is not True}, ensure_ascii=False, default=float))
print('runtime', SUM['runtime_s'], 'boot', T_BOOT)
