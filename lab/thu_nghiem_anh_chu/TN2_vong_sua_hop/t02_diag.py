"""t02_diag.py — chẩn đoán: mỗi loại ứng viên đúng khe bao nhiêu (theo mô hình khe người), trần 'oracle', lỗi của ô nhận."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from slotlib import SlotModel
REPO = Path('/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR'); OUT = REPO/'measure_out/_thu_nghiem_anh_chu/TN2'
S = pd.read_csv(OUT/'scores.csv', dtype=str, keep_default_na=False)
for c in ['m_T','m_L']: S[c] = pd.to_numeric(S[c], errors='coerce')
res = {}
for book, tag in [('LucVanTien1916','T'),('TruyenKieu1872','L')]:
    M = SlotModel(book); Sb = S[S.book_set==book].copy()
    o = [M.slot(r.page, r.column, r.syl_idx, json.loads(r.bbox)) for r in Sb.itertuples()]
    Sb['so'] = [str(x['slot_ok']) for x in o]
    Sb['tmpl_ok'] = [int(x.get('col_ok')==1 and x['slot_tmpl']==int(s)) for x, s in zip(o, Sb.syl_idx)]
    before = Sb[Sb.cand=='deliv'].set_index('cell_uid').so
    Sb['before'] = Sb.cell_uid.map(before)
    r = {}
    r['before'] = dict(Counter(before))
    r['cand_slot'] = {c: dict(Counter(g.so)) for c, g in Sb.groupby('cand')}
    r['cand_slot_given_before_wrong'] = {c: dict(Counter(g.so)) for c, g in Sb[Sb.before!='1'].groupby('cand')}
    wrong = Sb[(Sb.before!='1') & Sb.cand.isin(['prev','next','kim','det'])]
    r['oracle_any_cand_slot1_of_before_wrong'] = [int(wrong.groupby('cell_uid').so.apply(lambda s: (s=='1').any()).sum()), int(wrong.cell_uid.nunique())]
    r['kim_only_slot1_of_before_wrong'] = int(((wrong.cand=='kim')&(wrong.so=='1')).sum())
    # chọn theo verifier (argmax m) không ngưỡng
    f = Sb[Sb.cand.isin(['prev','next','kim','det'])]
    best = f.loc[f.groupby('cell_uid')[f'm_{tag}'].idxmax()]
    r['argmax_pick_counts'] = dict(Counter(best.cand))
    r['argmax_slot_by_before'] = {b: dict(Counter(g.so)) for b, g in best.groupby('before')}
    r['argmax_tmplonly_ok_by_before'] = {b: int(g.tmpl_ok.sum()) for b, g in best.groupby('before')}
    # hộp kim thuần hình học (không bộ kiểm)
    k = Sb[Sb.cand=='kim']
    r['kim_geom_slot_by_before'] = {b: dict(Counter(g.so)) for b, g in k.groupby('before')}
    r['kim_geom_tmplonly_ok_by_before'] = {b: [int(g.tmpl_ok.sum()), len(g)] for b, g in k.groupby('before')}
    res[book] = r
print(json.dumps(res, ensure_ascii=False, indent=0)[:4000])
(HERE/'t02_diag.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
