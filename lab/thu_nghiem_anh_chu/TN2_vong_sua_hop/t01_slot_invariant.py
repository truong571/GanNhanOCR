"""t01_slot_invariant.py — bất biến: slotlib trên bbox GỐC tái lập slot_ok/gt_img/slot_tmpl/slot_kim của cells_eval.csv."""
import json, sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slotlib import SlotModel
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
res = {}
rows = [r for r in csv.DictReader(open(SP/'kim_bottleneck/harness/cells_eval.csv', encoding='utf-8')) if r['eval_set']=='ihr_human']
for book in ('LucVanTien1916','TruyenKieu1872'):
    M = SlotModel(book)
    n = eq = eqi = eqg = 0
    for r in rows:
        if r['book'] != book or not r['bbox']: continue
        o = M.slot(r['page'], r['column'], r['syl_idx'], json.loads(r['bbox']))
        n += 1; eq += str(o['slot_ok']) == r['slot_ok']; eqi += o['gt_img'] == r['gt_img']; eqg += o['gt_char'] == r['gt_char']
    res[book] = dict(n=n, slot_ok_eq=eq, gt_img_eq=eqi, gt_char_eq=eqg, pass_=(eq==n and eqi==n and eqg==n))
print(json.dumps(res))
Path(__file__).with_name('t01_slot_invariant.json').write_text(json.dumps(res, indent=1))
