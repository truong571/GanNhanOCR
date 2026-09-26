#!/usr/bin/env python3
"""t00_explore.py — thăm dò: cờ trượt sẵn có so với sự thật khe người (L16/TK GOLD). Chỉ in tóm tắt."""
import sys, json
from pathlib import Path
import pandas as pd, numpy as np
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
ce = pd.read_csv(SP/'kim_bottleneck/harness/cells_eval.csv', dtype=str, keep_default_na=False)
ce = ce[(ce.eval_set=='ihr_human') & (ce.delivered_tier=='GOLD')]
gs = pd.read_csv(SP/'gold_img_audit/measure/gold_suspicion.csv', dtype=str, keep_default_na=False)
vf = pd.read_csv(SP/'r4/verifier_ft/gold_scores.csv', dtype=str, keep_default_na=False, usecols=['cell_uid','p_match','p_wood_T','p_wood_L','mRSN_T','mRSN_L'])
ce = ce.drop(columns=[c for c in ['geo_f_kim_slip','geo_strong_any'] if c in ce.columns])
d = ce.merge(gs[['cell_uid','geo_dy_kim','geo_f_kim_slip','geo_f_kim_off','geo_f_dup_bbox','vis_slip_flag','vis_mod','vis_cons','geo_strong_any']], on='cell_uid', how='left').merge(vf, on='cell_uid', how='left')
for b, x in d.groupby('book'):
    so = x.slot_ok
    print(b, 'n', len(x), 'slot_ok', dict(so.value_counts()))
    for f in ['geo_f_kim_slip','geo_f_kim_off','geo_f_dup_bbox','vis_slip_flag','vis_mod','geo_strong_any']:
        m = x[f]=='1'
        print(f'  {f:16s} n={m.sum():5d}  slot0={((so=="0")&m).sum():4d} slotU={((so=="")&m).sum():4d} slot1={((so=="1")&m).sum():5d}')
    pw = pd.to_numeric(x.p_wood_T if b=='LucVanTien1916' else x.p_wood_L, errors='coerce')
    for t in [0.5,0.8,0.9]:
        m = pw < t
        print(f'  p_wood_other<{t} n={m.sum():5d} slot0={((so=="0")&m).sum():4d} slotU={((so=="")&m).sum():4d} slot1={((so=="1")&m).sum():5d}')
    u = (x.geo_f_kim_off=='1')|(x.vis_slip_flag=='1')
    print('  union kim_off|vis_slip n', u.sum(), 'slot0', ((so=='0')&u).sum(), 'slotU', ((so=='')&u).sum(), 'of slot0 total', (so=='0').sum())
    dy = pd.to_numeric(x.geo_dy_kim, errors='coerce').abs()
    print('  |dy_kim| quantiles slot0', np.nanpercentile(dy[so=='0'], [10,25,50,75]).round(2), ' slot1', np.nanpercentile(dy[so=='1'], [50,90,99]).round(2))
