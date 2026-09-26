"""t04_share_demoted.py — chẩn đoán: ô được sửa mà hộp mới trùng (IoU≥0,5) hộp CŨ của một ô bị HẠ (ô hạ vẫn nằm ở tầng thấp với crop cũ)."""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import s3_eval as E
dec = pd.read_csv(E.OUT/(sys.argv[1] if len(sys.argv) > 1 else 'decisions_lenient.csv'), dtype=str, keep_default_na=False)
S = pd.read_csv(E.OUT/'scores.csv', dtype=str, keep_default_na=False, usecols=['cell_uid','cand','book_set','book','page','column','old_bbox'])
old = S[S.cand=='deliv'].set_index('cell_uid')
res = {}
for bs, d in dec.groupby('book_set'):
    ha = d[d.decision=='ha']; su = d[d.decision=='sua']
    key = lambda u: tuple(old.loc[u, ['book','page','column']])
    hb = {}
    for u in ha.cell_uid:
        hb.setdefault(key(u), []).append(json.loads(old.at[u,'old_bbox']))
    n = sum(any(E.iou(json.loads(b), h) >= 0.5 for h in hb.get(key(u), [])) for u, b in zip(su.cell_uid, su.new_bbox))
    res[bs] = dict(sua=len(su), trung_hop_voi_o_bi_ha=int(n))
print(json.dumps(res)); (HERE/('t04_share_demoted_' + (sys.argv[1] if len(sys.argv) > 1 else 'decisions_lenient.csv').replace('.csv', '') + '.json')).write_text(json.dumps(res, indent=1))
