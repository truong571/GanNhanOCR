#!/usr/bin/env python
"""t01_sweep.py — THỬ NGHIỆM 1: công cụ kiểm ảnh ↔ chữ cho ô GOLD. Độ chính xác HAI VẾ ↔ tỉ lệ ô giữ lại, 6 cấu hình,
quét 7 mức, chọn ngưỡng trên sách này rồi báo trên sách kia (LOBO hai chiều), CI bootstrap cụm theo trang.

0 API, chỉ CPU, repo CHỈ ĐỌC (ghi: lab/thu_nghiem_anh_chu/TN1_cong_kiem/ và measure_out/_thu_nghiem_anh_chu/TN1/).
Không dùng nhãn pipeline làm sự thật: sự thật = chữ NGƯỜI IHR (gt_char) + slot_ok (crop nằm đúng khe theo hộp cột người vẽ).

Đúng hai vế (định nghĩa v2/v3): V1+(nhãn, gt_char) ∧ slot_ok == '1'; khe chưa xác định ('') tính là KHÔNG đạt vế ảnh.
Chọn ngưỡng (tune) dùng đúng quy ước r4/policy/p02b: y_var ∧ slot1 trên ô slot_det (để tái lập đúng ngưỡng chính sách v3).

Hai chiều LOBO:  T: chọn trên TruyenKieu1872 (TK) -> báo trên LucVanTien1916 (L16) ; điểm p_wood_T (mô hình học trên TK phần A),
                     viss_T, cờ trượt bc_k06v1 (cấu hình vòng 2 chọn trên TK)
                 L: chọn trên L16 -> báo trên TK ; p_wood_L, viss_L, bc_k05dxv05 (chọn trên L16)
Tập chọn cho cấu hình có điểm mô hình: phần B (1/4 trang, mô hình KHÔNG thấy khi học) của sách chọn; cổng hình học (b): cả sách chọn.

Cấu hình (ô bị hạ = không giữ; công cụ KHÔNG BAO GIỜ sửa nhãn):
  a  pipeline hiện tại (giữ mọi GOLD)
  b  chỉ cổng hình học/trượt: thang G1..G7 lồng nhau; ở mỗi mục tiêu τ chọn bậc có độ phủ lớn nhất đạt τ trên sách chọn
  c  chỉ bộ kiểm ảnh verifier_ft: p_wood ≥ t (C6 của p02)
  c2 hai bộ kiểm ảnh (verifier_ft + viss vòng 3), không hình học: p_wood ≥ t1 ∧ viss ≥ t2
  d  hình học + ảnh: ¬(A0 ∪ B0 ∪ CNT ∪ bc) ∧ p_wood ≥ t1 ∧ viss ≥ t2 (ngưỡng = C5 p02b, tái lập)
  e  d + đối chiếu dị bản: ô ở bộ có văn bản người của dị bản (TK, KVK, L83) phải 'attested' (L16 không có -> e = d)
  f  chính sách v3 đầy đủ trên IHR = e + luật A (rescue/cầu tự dạng/văn bản yếu/ảnh ô khác) + M-OCR; τ = 0,995 phải
     trùng TỪNG Ô với gold_exact == 'ok' của r6/policy_v3 (invariant)
Mục tiêu τ quét: 0,98 0,985 0,99 0,9925 0,995 0,9975 0,999 (0,995 = đăng ký trước).
Ra: lab/.../TN1_cong_kiem/{sweep.csv, ladder_b.csv, main_table.csv, projection.csv, t01_summary.json}
    measure_out/_thu_nghiem_anh_chu/TN1/cells_ihr_decisions.pkl (quyết định từng ô IHR ở τ=0,995, cho t02_vi_du.py)
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

T0 = time.time()
LAB = Path(__file__).resolve().parent
REPO = LAB.parents[2]
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
BIG = REPO / 'measure_out/_thu_nghiem_anh_chu/TN1'; BIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SP / 'r6/policy_v3/scripts'))
from vlib import var_eq_plus  # noqa  (V1+ = var_eq ∪ Unihan kJapanese(Old)Variant ∪ NFC)

X = pd.read_pickle(SP / 'r6/policy_v3/out/base_v2.pkl')
V3 = pd.read_pickle(SP / 'r6/policy_v3/out/v02_cells_0.995.pkl')[['cell_uid', 'gold_exact', 'gold_exact_reason']]
X = X.merge(V3, on='cell_uid', how='left', validate='1:1')
SW = json.load(open(SP / 'r4/policy/out/p02b_sweep.json'))
V3S = json.load(open(SP / 'r6/policy_v3/summary.json'))
INV = {}
n = len(X)
S = X.book_set.values
IHR = ['LucVanTien1916', 'TruyenKieu1872']
AB = {'SachThanhTruyen': 'STT', 'Chrestomathie1872': 'Chr', 'LucVanTien1883': 'L83', 'KimVanKieu1884': 'KVK',
      'LucVanTien1916': 'L16', 'TruyenKieu1872': 'TK'}
TA_SETS = np.isin(S, ['TruyenKieu1872', 'KimVanKieu1884', 'LucVanTien1883'])
LITHO = np.isin(S, ['LucVanTien1883', 'KimVanKieu1884'])
TAUS = [0.98, 0.985, 0.99, 0.9925, 0.995, 0.9975, 0.999]
PREREG = 0.995
MIN_ACC = 50
b = lambda c: X[c].fillna(False).astype(bool).values

# ------------------------------------------------------------------ tín hiệu (không học gì mới)
cqf = X.crop_quality_flag.fillna('').values
A0 = b('int_foreign') | b('blank') | b('truncated') | b('geo_f_dup_bbox')                  # crop hỏng / ảnh ô khác / 1 hộp 2 cột
B0 = ((cqf == 'blank') | b('blank') | (cqf == 'truncated') | b('truncated') | b('geo_f_dup_bbox') | b('geo_f_ov_heavy')
      | b('f_tight_nb') | (cqf == 'bleed') | (X.seg_flag.fillna('').astype(str).values == 'tall'))   # 'không đúng một chữ'
CNT = b('geo_f_cnt_ocr_ne_qn')                                                              # cột: số chữ OCR ≠ số âm QN
AINT = b('int_foreign') | b('rescue') | b('similar') | b('weak_text')                      # luật A của chính sách
MOCR = X.r2_reason.fillna('').values == 'M_ocr_t50'
ta = X.ta.fillna('na').values
TA_OK = ~TA_SETS | (ta == 'attested')
ady, adx, visz = X.ady.values.astype(float), X.adx.values.astype(float), X.vis_z.values.astype(float)


def kimf_dx():
    return (np.nan_to_num(ady, nan=-1) > 0.5) | (np.nan_to_num(adx, nan=-1) > 0.5)


def visf(v):
    return np.nan_to_num(visz, nan=np.inf) <= v + 1e-12


BC = {'T': (np.nan_to_num(ady, nan=-1) > 0.6) | visf(-1.0), 'L': kimf_dx() | visf(-0.5)}
INV['bc_T_equals_bc_k06v1'] = bool((BC['T'] == b('bc_k06v1')).all())
INV['bc_L_equals_bc_k05dxv05'] = bool((BC['L'] == b('bc_k05dxv05')).all())
G3 = A0 | B0 | CNT
LADDER = [('G0', 'không cổng', np.zeros(n, bool)),
          ('G1', 'A0: crop rỗng/cắt nét, 1 hộp 2 cột, ảnh ô khác', A0),
          ('G2', 'G1 + B0: mực chữ kề (bleed/tight_nb), hộp chồng nặng, hộp cao "tall"', A0 | B0),
          ('G3', 'G2 + cột có số chữ OCR ≠ số âm QN', G3),
          ('G4', 'G3 + lệch dọc hộp kim > 1,0 bước', G3 | (np.nan_to_num(ady, nan=-1) > 1.0)),
          ('G5', 'G3 + lệch dọc > 0,75 ∨ vis_z ≤ −1', G3 | (np.nan_to_num(ady, nan=-1) > 0.75) | visf(-1.0)),
          ('G6', 'G3 + lệch dọc > 0,6 ∨ vis_z ≤ −1 (cờ vòng 2 chọn trên TK)', G3 | BC['T']),
          ('G7', 'G3 + lệch dọc/ngang > 0,5 ∨ vis_z ≤ −0,5 (cờ vòng 2 chọn trên L16)', G3 | BC['L'])]
INV['ladder_nested'] = bool(all((LADDER[i][2] <= LADDER[i + 1][2]).all() for i in range(len(LADDER) - 1)))

# ------------------------------------------------------------------ sự thật NGƯỜI (chỉ IHR)
gt = X.gt_char.fillna('').values
HAS = np.isin(S, IHR) & (gt != '')
lab = X.label.values
y_lab = np.zeros(n, bool)
y_lab[HAS] = [var_eq_plus(a, c) for a, c in zip(lab[HAS], gt[HAS])]
slot = X.slot_ok.fillna('').astype(str).values
y_both = y_lab & (slot == '1')
yT_all = X.y_var.fillna(False).astype(bool).values & b('slot1')        # quy ước chọn ngưỡng của p02b
SDET = b('slot_det'); PARTB = b('partB')
page = X.page.fillna('').values
INV['n_gold_with_gt'] = {bk: int((HAS & (S == bk)).sum()) for bk in IHR}
DIRS = {'T': dict(tune='TruyenKieu1872', test='LucVanTien1916', pw='p_wood_T', vs='viss_T'),
        'L': dict(tune='LucVanTien1916', test='TruyenKieu1872', pw='p_wood_L', vs='viss_L')}
PW = {d: X[c['pw']].values.astype(float) for d, c in DIRS.items()}
VS = {d: X[c['vs']].fillna(0).values.astype(float) for d, c in DIRS.items()}


# ------------------------------------------------------------------ chọn ngưỡng trên sách chọn (tái lập p02/p02b)
def pick_t(s, y, tgt):
    o = np.argsort(-s, kind='stable'); cp = np.cumsum(y[o]) / np.arange(1, len(y) + 1)
    ok = np.nonzero(cp >= tgt)[0]; ok = ok[ok >= MIN_ACC - 1]
    return float(s[o][ok.max()]) if len(ok) else np.inf


def tune_mask(d, partb=True):
    m = (S == DIRS[d]['tune']) & HAS & SDET
    return m & PARTB if partb else m


def sel_C5(d, tgt, use_r2=True):
    tu = tune_mask(d); pw, vs = PW[d][tu], VS[d][tu]; y = yT_all[tu].astype(float)
    r2 = ~(A0 | BC[d])[tu] if use_r2 else np.ones(tu.sum(), bool)
    qs = np.unique(np.quantile(pw, np.linspace(0, 0.8, 41))); qv = np.unique(np.quantile(vs, np.linspace(0, 0.8, 41)))
    best = None
    for t1 in qs:
        a1 = r2 & (pw >= t1)
        for t2 in qv:
            a = a1 & (vs >= t2)
            if a.sum() < MIN_ACC:
                continue
            if y[a].mean() >= tgt and (best is None or a.mean() > best[2]):
                best = (float(t1), float(t2), a.mean())
    return (best[0], best[1]) if best else (np.inf, np.inf)


def sel_C2(d, tgt):
    tu = tune_mask(d); s = np.where(~(A0 | BC[d])[tu], PW[d][tu], -np.inf)
    return pick_t(s, yT_all[tu].astype(float), tgt)


def sel_C6(d, tgt):
    tu = tune_mask(d)
    return pick_t(PW[d][tu], yT_all[tu].astype(float), tgt)


def sel_G(d, tgt):
    tu = tune_mask(d, partb=False); y = yT_all[tu]
    best, bestp = None, None
    for i, (gid, _, m) in enumerate(LADDER):
        acc = ~m[tu]
        if acc.sum() < MIN_ACC:
            continue
        p = y[acc].mean()
        if p >= tgt and (best is None or acc.mean() > best[1]):
            best = (i, acc.mean())
        if bestp is None or p > bestp[1] + 1e-12:
            bestp = (i, p)
    return (best[0], True) if best else (bestp[0], False)


TH = {}
for d in DIRS:
    for tgt in TAUS:
        g, reached = sel_G(d, tgt)
        # ngưỡng làm tròn 5 chữ số như p02b/chính sách v3 (ô nằm đúng ngưỡng: quy ước của v3)
        r5 = lambda v: tuple(round(x, 5) for x in v) if isinstance(v, tuple) else round(v, 5)
        TH[(d, tgt)] = dict(G=g, G_reached=reached, C6=r5(sel_C6(d, tgt)), C5nr=r5(sel_C5(d, tgt, use_r2=False)),
                            C5=r5(sel_C5(d, tgt)), C2=r5(sel_C2(d, tgt)))
# invariant: tái lập đúng ngưỡng p02b (C5, C2) ở 4 mục tiêu chung
rep = {}
for d in DIRS:
    for tk, v in SW[d]['sweep'].items():
        mine = TH[(d, float(tk))]
        rep[f'{d}_{tk}'] = bool(np.allclose([round(mine['C5'][0], 5), round(mine['C5'][1], 5)], v['C5']['t'])
                                and abs(round(mine['C2'], 5) - v['C2']['t']) < 1e-9)
INV['reproduce_p02b_thresholds'] = rep


def keep_masks(d, tgt):
    th = TH[(d, tgt)]; pw, vs = PW[d], VS[d]
    t1, t2 = th['C5']
    geo_v3 = A0 | B0 | CNT | BC[d]
    m = {'a': np.ones(n, bool),
         'b': ~LADDER[th['G']][2],
         'c': pw >= th['C6'],
         'c2': (pw >= th['C5nr'][0]) & (vs >= th['C5nr'][1]),
         'd': ~geo_v3 & (pw >= t1) & (vs >= t2)}
    m['e'] = m['d'] & TA_OK
    m['f'] = m['e'] & ~(AINT | MOCR)
    return m


# ------------------------------------------------------------------ đo trên NHÃN NGƯỜI của sách báo (test)
rng = np.random.default_rng(20260926)
BOOT_B = 2000
_boot_idx = {}


def boot_ci(keep, te):
    """precision hai vế trên phần giữ; CI 95 % bootstrap cụm theo trang (B=2000)."""
    k = keep & te
    if k.sum() == 0:
        return None, None
    df = pd.DataFrame(dict(g=page[te], a=k[te].astype(float), e=(k & ~y_both)[te].astype(float))).groupby('g')[['a', 'e']].sum()
    A, E = df.a.values, df.e.values
    key = (int(te.sum()), len(A))
    if key not in _boot_idx:
        _boot_idx[key] = rng.integers(0, len(A), (BOOT_B, len(A)))
    idx = _boot_idx[key]
    pb = 1 - E[idx].sum(1) / np.maximum(A[idx].sum(1), 1)
    return round(float(np.percentile(pb, 2.5)), 5), round(float(np.percentile(pb, 97.5)), 5)


def measure(keep, te):
    k = keep & te
    nk = int(k.sum()); nt = int(te.sum())
    r = dict(n_gold_gt=nt, n_keep=nk, keep_pct=round(100 * nk / nt, 2))
    if nk:
        prec = float(y_both[k].mean()); lo, hi = boot_ci(keep, te)
        r.update(prec_both=round(prec, 5), ci_lo=lo, ci_hi=hi, prec_label=round(float(y_lab[k].mean()), 5),
                 prec_img=round(float((slot[k] == '1').mean()), 5))
    else:
        r.update(prec_both=None, ci_lo=None, ci_hi=None, prec_label=None, prec_img=None)
    r.update(err_img_sai_khe=int((k & (slot == '0')).sum()), err_img_khe_chua_xd=int((k & (slot == '')).sum()),
             err_chu=int((k & (slot == '1') & ~y_lab).sum()),
             err_total=int((k & ~y_both).sum()),
             ha_oan=int((te & ~keep & y_both).sum()), ha_dung=int((te & ~keep & ~y_both).sum()),
             err_pool=int((te & ~y_both).sum()), n_correct=int((te & y_both).sum()))
    return r


def tune_stats(d, tgt, cfg, keep):
    tu = tune_mask(d, partb=(cfg != 'b'))
    acc = keep[tu]
    return dict(tune_cov=round(float(acc.mean()), 4), tune_prec=round(float(yT_all[tu][acc].mean()), 5) if acc.any() else None)


CFG_NAMES = {'a': 'pipeline hiện tại (không kiểm)', 'b': 'chỉ hình học/trượt', 'c': 'chỉ bộ kiểm ảnh verifier_ft (p_wood)',
             'c2': 'hai bộ kiểm ảnh (p_wood ∧ viss), không hình học', 'd': 'hình học + ảnh', 'e': 'hình học + ảnh + dị bản (text_attested)',
             'f': 'chính sách v3 đầy đủ'}
rows = []
DEC = {}
for d, cfg in DIRS.items():
    te = (S == cfg['test']) & HAS
    for tgt in TAUS:
        km = keep_masks(d, tgt); th = TH[(d, tgt)]
        for c, keep in km.items():
            r = dict(config=c, config_name=CFG_NAMES[c], direction=d, chon_tren=AB[cfg['tune']], bao_tren=AB[cfg['test']], tau=tgt)
            if c == 'b':
                r['param'] = f"{LADDER[th['G']][0]}{'' if th['G_reached'] else ' (không bậc nào đạt τ trên sách chọn -> bậc chính xác nhất)'}"
            elif c == 'c':
                r['param'] = f"p_wood≥{th['C6']:.5f}"
            elif c == 'c2':
                r['param'] = f"p_wood≥{th['C5nr'][0]:.5f} ∧ viss≥{th['C5nr'][1]:.5f}"
            elif c in ('d', 'e', 'f'):
                r['param'] = f"p_wood≥{th['C5'][0]:.5f} ∧ viss≥{th['C5'][1]:.5f}"
            else:
                r['param'] = ''
            r.update(tune_stats(d, tgt, c, keep) if c != 'a' else dict(tune_cov=1.0, tune_prec=None))
            r.update(measure(keep, te))
            rows.append(r)
            if tgt == PREREG:
                DEC[(d, c)] = keep
SWP = pd.DataFrame(rows)
SWP.to_csv(LAB / 'sweep.csv', index=False)

# ------------------------------------------------------------------ thang (b) mô tả, không chọn: từng bậc trên từng sách
lad = []
for bk in IHR:
    te = (S == bk) & HAS
    for gid, desc, m in LADDER:
        r = dict(bac=gid, mo_ta=desc, sach=AB[bk]); r.update(measure(~m, te)); lad.append(r)
pd.DataFrame(lad).to_csv(LAB / 'ladder_b.csv', index=False)

# ------------------------------------------------------------------ invariants đối chiếu chính sách v3
ih = V3S['ihr_prereg_0995']
for d, cfg in DIRS.items():
    bk = cfg['test']; te = (S == bk) & HAS
    f = DEC[(d, 'f')]
    allg = (S == bk)
    INV[f'f_equals_v3_ok_all_gold_{AB[bk]}'] = bool(((f & allg) == ((X.gold_exact.values == 'ok') & allg)).all())
    INV[f'f_n_ok_all_gold_{AB[bk]}'] = int((f & allg).sum())
    mf = measure(f, te)
    INV[f'f_prec_matches_v3_{AB[bk]}'] = bool(abs(mf['prec_both'] - ih[bk]['both']) < 1e-4 and mf['n_keep'] == ih[bk]['ok'])
    ma = measure(np.ones(n, bool), te)
    INV[f'a_prec_matches_v3_pipeline_{AB[bk]}'] = bool(abs(ma['prec_both'] - ih[bk]['pipeline_all_gold']['both']) < 1e-4)
INV['all_pass'] = all(v for k, v in INV.items() if isinstance(v, bool)) and all(INV['reproduce_p02b_thresholds'].values())

# ------------------------------------------------------------------ bảng chính: τ = 0,995 (đăng ký trước)
MAIN = SWP[SWP.tau == PREREG].copy()
MAIN.to_csv(LAB / 'main_table.csv', index=False)

# ------------------------------------------------------------------ quyết định từng ô IHR ở τ = 0,995 (cho trang ví dụ)
ihr_idx = np.nonzero(np.isin(S, IHR))[0]
D = X.iloc[ihr_idx][['cell_uid', 'book_set', 'page', 'label', 'syl', 'gt_char', 'gt_img', 'slot_ok', 'p_wood_T', 'p_wood_L',
                     'viss_T', 'viss_L', 'ady', 'adx', 'vis_z', 'ta', 'gold_exact', 'gold_exact_reason', 'crop_quality_flag',
                     'seg_flag', 'stray_ink']].copy()
D['has_gt'] = HAS[ihr_idx]; D['y_lab'] = y_lab[ihr_idx]; D['y_both'] = y_both[ihr_idx]
for c in CFG_NAMES:
    D[f'keep_{c}'] = np.where(S[ihr_idx] == 'LucVanTien1916', DEC[('T', c)][ihr_idx], DEC[('L', c)][ihr_idx])
# lý do hạ theo cấu hình d/e/f (để giải thích ví dụ)
why = np.array([''] * n, dtype=object)
for d_, bk in (('T', 'LucVanTien1916'), ('L', 'TruyenKieu1872')):
    mm = S == bk; th = TH[(d_, PREREG)]
    reasons = [('A_luat_toan_ven', AINT), ('M_OCR', MOCR), ('B0_khong_mot_chu', A0 | B0), ('CNT_cot_lech_so_chu', CNT),
               ('truot_bc', BC[d_]), ('p_wood_thap', PW[d_] < th['C5'][0]), ('viss_thap', VS[d_] < th['C5'][1]),
               ('di_ban_khong_chung', ~TA_OK)]
    for nm, mk in reasons:
        sel = mm & mk & (why == '')
        why[sel] = nm
D['ly_do_ha_f'] = why[ihr_idx]
D.to_pickle(BIG / 'cells_ihr_decisions.pkl')

# ------------------------------------------------------------------ chiếu sang 4 bộ không có nhãn người (ƯỚC LƯỢNG, không phải độ chính xác)
th_T, th_L = TH[('T', PREREG)], TH[('L', PREREG)]
g_strict = max(th_T['G'], th_L['G'])     # thang lồng nhau -> AND hai chiều = bậc chặt hơn
GEO = LADDER[g_strict][2]
pwT, pwL = PW['T'], PW['L']
vX = X.viss_X.fillna(0).values
STT = S == 'SachThanhTruyen'; CHR = S == 'Chrestomathie1872'
hand = X.lobo_cert_00015.fillna(0).astype(int).values == 1
hand_nh = X.lobo_nh.fillna(0).values >= 3
geo_v3 = A0 | B0 | CNT | BC['L']          # chính sách v3 dùng bc_k05dxv05 cho mọi bộ ngoài L16
score_print = np.zeros(n, bool)
score_print[LITHO] = ((pwT >= th_T['C5'][0]) & (pwL >= th_L['C5'][0]) & (vX >= max(th_T['C5'][1], th_L['C5'][1])))[LITHO]
score_print[CHR] = ((pwT >= th_T['C2']) & (pwL >= th_L['C2']))[CHR]
c_print = (pwT >= th_T['C6']) & (pwL >= th_L['C6'])
PROJ = {'a': np.ones(n, bool), 'b': ~GEO,
        'c': np.where(STT, hand & hand_nh, c_print),
        'd': np.where(STT, ~GEO & hand & hand_nh, ~geo_v3 & score_print)}
PROJ['e'] = PROJ['d'] & TA_OK
PROJ['f'] = X.gold_exact.values == 'ok'
# hai sách IHR: dùng đúng quyết định LOBO của chiều tương ứng (L16 <- chọn trên TK ; TK <- chọn trên L16)
for d_, bk in (('T', 'LucVanTien1916'), ('L', 'TruyenKieu1872')):
    mm = S == bk
    for c in PROJ:
        PROJ[c] = np.where(mm, DEC[(d_, c)], PROJ[c])
prow = []
for bk in ['SachThanhTruyen', 'Chrestomathie1872', 'LucVanTien1883', 'KimVanKieu1884'] + IHR:
    mm = S == bk
    for c, keep in PROJ.items():
        prow.append(dict(sach=AB[bk], config=c, config_name=CFG_NAMES[c], gold=int(mm.sum()), giu=int((keep & mm).sum()),
                         ha=int((~keep & mm).sum()), giu_pct=round(100 * float(keep[mm].mean()), 2),
                         muc=('ĐO ĐƯỢC trên IHR (xem sweep)' if bk in IHR else 'ƯỚC LƯỢNG (không có nhãn người)')))
PR = pd.DataFrame(prow)
PR.to_csv(LAB / 'projection.csv', index=False)
INV['proj_f_matches_v3_per_set'] = {AB[bk]: int(PR[(PR.sach == AB[bk]) & (PR.config == 'f')].giu.iloc[0]) == V3S['per_set'][bk]['ok']
                                    for bk in V3S['per_set']}

# ------------------------------------------------------------------ công cụ bắt được loại lỗi nào? (IHR, τ = 0,995)
sys.path.insert(0, str(SP / 'kim_bottleneck/harness'))
from harness_lib import is_sim, is_pua  # noqa
CE = pd.read_csv(SP / 'kim_bottleneck/harness/cells_eval.csv', dtype=str, keep_default_na=False, usecols=['cell_uid', 'R', 'gt_prev', 'gt_next'])
XE = X[['cell_uid']].merge(CE, on='cell_uid', how='left', validate='1:1').fillna('')
Rv, gpv, gnv = XE.R.values, XE.gt_prev.values, XE.gt_next.values


def err_kind(i):
    if slot[i] == '0':
        return 'anh_sai_khe'
    if slot[i] == '':
        return 'anh_khe_chua_xac_dinh'
    a, g = lab[i], gt[i]
    if is_pua(a) or is_pua(g):
        return 'chu_pua'
    if a in (gpv[i], gnv[i]):
        return 'chu_bang_chu_ke'
    if is_sim(a, g):
        return 'chu_gan_hinh'
    if a in Rv[i] and g in Rv[i]:
        return 'chu_dong_am_trong_R'
    return 'chu_khac'


EK = np.array([''] * n, dtype=object)
for i in np.nonzero(HAS & ~y_both)[0]:
    EK[i] = err_kind(i)
EK[HAS & y_both] = 'DUNG_hai_ve'
ek_rows = []
for d_, bk in (('T', 'LucVanTien1916'), ('L', 'TruyenKieu1872')):
    te = (S == bk) & HAS
    for kind in ['DUNG_hai_ve', 'anh_sai_khe', 'anh_khe_chua_xac_dinh', 'chu_dong_am_trong_R', 'chu_gan_hinh', 'chu_bang_chu_ke', 'chu_pua', 'chu_khac']:
        mk = te & (EK == kind)
        r = dict(sach=AB[bk], loai=kind, n_gold=int(mk.sum()))
        for c in ('b', 'c', 'd', 'f'):
            kp = DEC[(d_, c)]
            r[f'giu_{c}'] = int((mk & kp).sum()); r[f'ty_le_bi_ha_{c}'] = round(float((~kp[mk]).mean()), 4) if mk.any() else None
        ek_rows.append(r)
EKD = pd.DataFrame(ek_rows); EKD.to_csv(LAB / 'err_class_catch.csv', index=False)
D['err_kind'] = EK[ihr_idx]; D['R'] = Rv[ihr_idx]; D['gt_prev'] = gpv[ihr_idx]; D['gt_next'] = gnv[ihr_idx]
D.to_pickle(BIG / 'cells_ihr_decisions.pkl')

SUM = dict(key='TN1_cong_kiem', date='2026-09-26', api_calls=0, repo_writes_outside_lab_measure_out=0,
           truth='V1+(nhãn, chữ người IHR) ∧ slot_ok==1; khe chưa xác định = sai vế ảnh',
           taus=TAUS, prereg=PREREG, boot=dict(B=BOOT_B, cluster='page', seed=20260926),
           ladder=[(g, dsc) for g, dsc, _ in LADDER], g_strict_for_projection=LADDER[g_strict][0],
           thresholds={f'{d}_{t}': {k: (list(v) if isinstance(v, tuple) else v) for k, v in TH[(d, t)].items()} for d in DIRS for t in TAUS},
           invariants=INV, runtime_s=round(time.time() - T0, 1))
json.dump(SUM, open(LAB / 't01_summary.json', 'w'), ensure_ascii=False, indent=1, default=float)
pd.set_option('display.width', 250)
print(MAIN[['config', 'bao_tren', 'param', 'n_keep', 'keep_pct', 'prec_both', 'ci_lo', 'ci_hi', 'err_img_sai_khe', 'err_img_khe_chua_xd',
            'err_chu', 'ha_oan']].to_string(index=False))
print(json.dumps(INV, ensure_ascii=False))
print('runtime', SUM['runtime_s'])
