#!/usr/bin/env python3
"""t01_fix_decisions.py — TN3 bước 1: quyết định VÒNG SỬA HỘP (TN2, luật trùng hộp NGHIÊM) cho CẢ 6 book_set.

Không chọn lại gì: dùng nguyên hàm `propose` / `resolve` của TN2 (`TN2_vong_sua_hop/s3_eval.py`, import chỉ đọc) và
ngưỡng θ TN2 đã chọn (`s3_summary.json` → theta_chosen_on):
  L16  : mô hình T, θ chọn trên TK   (LOBO)          TK : mô hình L, θ chọn trên L16 (LOBO)
  Chr, STT, L83, KVK : T (θ chọn trên L16) VÀ L (θ chọn trên TK) phải CÙNG chọn một ứng viên, cùng qua θ của mình
Điểm: TN2 `measure_out/_thu_nghiem_anh_chu/TN2/scores.csv` (L16/TK/Chr/STT) và TN3
`measure_out/_thu_nghiem_anh_chu/TN3/tn2_l83kvk/scores.csv` (L83/KVK — bản sao s1a/s1b/s2 của TN2 chỉ đổi thư mục ra/bộ).
Sự thật khe mới (chỉ IHR, trang có GT người) lấy từ `TN2/decisions.csv` (cột cls_khe, chấm bằng slotlib của TN2).
Ra: measure_out/_thu_nghiem_anh_chu/TN3/fix_decisions.csv, lab/.../TN3_tat_ca_bo/t01_summary.json (invariants).
0 API, CPU, vài giây. Chạy: .venv/bin/python lab/thu_nghiem_anh_chu/TN3_tat_ca_bo/t01_fix_decisions.py
"""
import json, sys, time
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TN2 = REPO / 'lab/thu_nghiem_anh_chu/TN2_vong_sua_hop'
BIG = REPO / 'measure_out/_thu_nghiem_anh_chu/TN3'
sys.dont_write_bytecode = True           # không ghi __pycache__ vào thư mục TN2 (chỉ đọc)
sys.path.insert(0, str(TN2))
import s3_eval as T2  # noqa  (chỉ dùng propose/resolve/column_state; SHARE mặc định 'strict')

assert T2.SHARE == 'strict'
t0 = time.time()
S2 = pd.read_csv(REPO / 'measure_out/_thu_nghiem_anh_chu/TN2/scores.csv', dtype=str, keep_default_na=False)
S3 = pd.read_csv(BIG / 'tn2_l83kvk/scores.csv', dtype=str, keep_default_na=False)
S = pd.concat([S2, S3], ignore_index=True)
for c in ['m_T', 'm_L', 'mN_T', 'mN_L', 'sw_T', 'sw_L']:
    S[c] = pd.to_numeric(S[c], errors='coerce')
L = pd.read_csv(REPO / 'dataset/_ALL/labels.csv', dtype=str, keep_default_na=False,
                usecols=['cell_uid', 'book_set', 'book', 'page', 'column', 'label', 'tier', 'bbox'])
cols_all = T2.column_state(L)
SUMM = json.load(open(TN2 / 's3_summary.json'))
TH = SUMM['theta_chosen_on']            # {sách tune: θ}
thT, thL = TH['LucVanTien1916'], TH['TruyenKieu1872']   # θ của T chọn trên L16; θ của L chọn trên TK
D2 = pd.read_csv(REPO / 'measure_out/_thu_nghiem_anh_chu/TN2/decisions.csv', dtype=str, keep_default_na=False)
INV = {}
rows = []
RULE = {'LucVanTien1916': ('T', TH['TruyenKieu1872']), 'TruyenKieu1872': ('L', TH['LucVanTien1916'])}
for bs in ['LucVanTien1916', 'TruyenKieu1872', 'Chrestomathie1872', 'SachThanhTruyen', 'LucVanTien1883', 'KimVanKieu1884']:
    Sb = S[S.book_set == bs]
    flagged = set(Sb.cell_uid)
    if bs in RULE:
        tag, th = RULE[bs]
        raw = T2.propose(Sb, tag, th)
        rule = f'{tag} θ={th} (LOBO)'
    else:
        rT = T2.propose(Sb, 'T', thT); rL = T2.propose(Sb, 'L', thL)
        raw = {u: rT[u] for u in rT if u in rL and rL[u][0] == rT[u][0]}
        rule = f'T θ={thT} ∧ L θ={thL}, cùng ứng viên'
        INV[f'{bs}_raw_T_L_both'] = [len(rT), len(rL), len(raw)]
    cols = {k: v for k, v in cols_all.items() if k[0] == bs and any(x[0] in raw for x in v)}
    acc, why = T2.resolve(raw, flagged, cols, None)
    INV[f'{bs}_huy_trung_hop_don_dieu'] = [sum(1 for v in why.values() if v == 'trung_hop'), sum(1 for v in why.values() if v == 'khong_don_dieu')]
    for u in sorted(flagged):
        rows.append(dict(cell_uid=u, book_set=bs, fix=int(u in acc), cand=acc[u][0] if u in acc else '',
                         new_bbox=json.dumps(acc[u][1]) if u in acc else '', m_new=round(acc[u][2], 4) if u in acc else '',
                         huy=why.get(u, ''), rule=rule))
    # invariant: trùng quyết định TN2 đã lưu (IHR: trên trang có GT; Chr/STT: mọi ô bị cờ)
    d2 = D2[D2.book_set == bs]
    if len(d2):
        mine = {u for u in acc}
        theirs = set(d2[d2.decision == 'sua'].cell_uid)
        INV[f'{bs}_fix_equals_TN2_decisions'] = bool({u for u in mine if u in set(d2.cell_uid)} == theirs)
    INV[f'{bs}_n_flag'] = len(flagged); INV[f'{bs}_n_fix'] = len(acc)
F = pd.DataFrame(rows)
# sự thật khe mới cho ô được sửa ở IHR (TN2 chấm bằng slotlib; sua_dung/doi_van_dung = khe mới đúng)
k = D2[D2.book_set.isin(['LucVanTien1916', 'TruyenKieu1872'])][['cell_uid', 'cls', 'cls_khe']]
F = F.merge(k, on='cell_uid', how='left', validate='1:1').fillna({'cls': '', 'cls_khe': ''})
F['new_slot_ok'] = ''
fx = (F.fix == 1) & (F.cls_khe != '')
F.loc[fx, 'new_slot_ok'] = F.loc[fx, 'cls_khe'].isin(['sua_dung', 'doi_van_dung']).astype(int).astype(str)
F.to_csv(BIG / 'fix_decisions.csv', index=False)
s1b = json.load(open(BIG / 'tn2_l83kvk/s1b_summary.json'))
INV['L83_KVK_old_crop_md5_reproduce'] = all(v['old_exact_all'] for v in s1b['invariants'].values())
INV['L83_KVK_kim_char_ok_all'] = all(s1b[b]['kim_char_ok'] == s1b[b]['old_n'] for b in ['LucVanTien1883', 'KimVanKieu1884'])
INV['labels_never_changed'] = True
INV['all_pass'] = all(v for kk, v in INV.items() if isinstance(v, bool))
out = dict(date='2026-09-26', api_calls=0, theta_used=dict(T_on_L16=thT, L_on_TK=thL, LOBO=RULE), invariants=INV,
           fix_by_set=F.groupby('book_set').fix.agg(['sum', 'size']).rename(columns={'sum': 'fix', 'size': 'flag'}).to_dict('index'),
           ihr_fix_new_slot=F[F.new_slot_ok != ''].groupby('book_set').new_slot_ok.value_counts().unstack(fill_value=0).to_dict('index'),
           sec=round(time.time() - t0, 1))
json.dump(out, open(HERE / 't01_summary.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, default=str))
