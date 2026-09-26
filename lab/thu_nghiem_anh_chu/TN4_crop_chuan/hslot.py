"""hslot.py — mô hình khe NGƯỜI của harness (kim_bottleneck/harness/h01_build_cells.py), chép nguyên logic, áp cho
một tâm (cx, cy) BẤT KỲ (tâm bbox = như harness; tâm hộp mực crop mới = TN4). Chỉ đọc.
slot_ok = 1 nếu cột đúng và CẢ HAI mô hình (mẫu t + dòng kim; hoặc chỉ mẫu khi thiếu dòng kim) cho khe == syl_idx;
          0 nếu sai cột hoặc cả hai cùng lệch; '' nếu bất đồng / không có GT."""
import collections as C
import csv, json, math
from pathlib import Path

REPO = Path('/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR')
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
N1 = 6


def load_gt(book):
    gt = {}
    for r in csv.DictReader(open(REPO / 'data' / book / 'manifest.tsv', encoding='utf-8'), delimiter='\t'):
        if r['col_index'] == '' or r['part'] == '':
            continue
        x, y, w, h = map(float, r['col_bbox_xywh'].split(','))
        d = gt.setdefault((r['page_id'], int(r['col_index'])), dict(box=(x, y, w, h), t={}))
        d['t'][int(r['part'])] = r['nom_text']
    for d in gt.values():
        d['t1'] = d['t'].get(1, ''); d['t2'] = d['t'].get(2, '')
    return gt


class Slot:
    def __init__(self, book):
        self.book = book
        self.gt = load_gt(book)
        self.pm = {p['page_name']: p['page_id'] for p in json.loads((REPO / 'prepared' / book / 'manifest.json').read_text())['pages']}
        self.pages_gt = {k[0] for k in self.gt}
        self.mau = json.load(open(SP / 'kim_bottleneck/harness/build_summary.json'))[book]['mau_khe_t']
        self.klines = {}
        for pn, pid in self.pm.items():
            f = REPO / 'prepared' / book / 'kim_raw' / f'{pn}_lt2.json'
            if not f.exists():
                continue
            bx = json.loads(f.read_text())['boxes']
            cols = [(k[1], v['box']) for k, v in self.gt.items() if k[0] == pid]
            per = C.defaultdict(list)
            for b in bx:
                xs = [p[0] for p in b['points']]; ys = [p[1] for p in b['points']]
                xc = (min(xs) + max(xs)) / 2
                for ci, (x, y, w, h) in cols:
                    if x <= xc <= x + w:
                        per[ci].append((min(ys), max(ys), b.get('transcription', ''))); break
            for ci, L in per.items():
                L.sort()
                if len(L) == 2:
                    self.klines[(pid, ci)] = [(L[0][0], L[0][1], 1, L[0][2]), (L[1][0], L[1][1], 2, L[1][2])]

    def slot(self, page, column, syl_idx, cx, cy):
        pid = self.pm.get(page, '')
        if pid not in self.pages_gt or cx is None or cy is None or cx != cx or cy != cy:
            return ''
        cand = [(k[1], v) for k, v in self.gt.items() if k[0] == pid]
        inside = [(ci, v) for ci, v in cand if v['box'][0] <= cx <= v['box'][0] + v['box'][2]]
        if inside:
            gci, gv = inside[0]
        else:
            gci, gv = min(cand, key=lambda kv: min(abs(cx - kv[1]['box'][0]), abs(cx - kv[1]['box'][0] - kv[1]['box'][2])))
        x, y, w, h = gv['box']
        t = (cy - y) / h
        si = int(syl_idx); ec = int(column) - 1
        k_t = min(range(14), key=lambda j: abs(t - self.mau[j]))
        k_k = None
        if (pid, gci) in self.klines:
            L = self.klines[(pid, gci)]
            top, bot, prt, _ = min(L, key=lambda Q: 0 if Q[0] <= cy <= Q[1] else min(abs(cy - Q[0]), abs(cy - Q[1])))
            txt = gv['t'].get(prt, '')
            if txt:
                u = (cy - top) / max(1e-6, bot - top) * len(txt)
                sl = min(len(txt) - 1, max(0, math.floor(u)))
                k_k = sl if prt == 1 else N1 + sl
        col_ok = gci == ec
        on_t = col_ok and k_t == si
        on_k = None if k_k is None else (col_ok and k_k == si)
        if not col_ok:
            return '0'
        if on_t and (on_k is None or on_k):
            return '1'
        if (not on_t) and on_k is False:
            return '0'
        return ''
