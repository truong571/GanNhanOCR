"""slotlib.py — MÔ HÌNH KHE NGƯỜI (IHR) cho một hộp BẤT KỲ, chép nguyên logic của
kim_bottleneck/harness/h01_build_cells.py (hai mô hình khe: mẫu trung vị + dòng kim; hộp CỘT do người vẽ, chữ GT người).

SlotModel(book).slot(page, column, syl_idx, bbox) -> dict(slot_ok, gt_img, gt_char, slot_tmpl, slot_kim, geo_col, col_ok)
  slot_ok = 1  cột GT chứa tâm x == column−1 và khe mẫu == syl_idx và (khe dòng kim == syl_idx hoặc thiếu dòng kim)
          = 0  sai cột, hoặc cả hai mô hình cùng lệch
          = '' hai mô hình bất đồng / không có GT (báo cáo v2: tính là SAI)
  gt_img  = chữ GT ở khe ảnh đang chỉ (slot_ok=1: gt_char; 2 mô hình cùng chỉ khe khác: chữ ở khe đó; còn lại '')
Bất biến (kiểm ở s3): áp cho bbox GỐC phải tái lập đúng slot_ok/gt_img của cells_eval.csv.
"""
import collections as C
import csv
import json
import math
import statistics as ST
from pathlib import Path

REPO = Path('/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR')
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
        d['t1'] = d['t'].get(1, '')
        d['t2'] = d['t'].get(2, '')
    return gt


class SlotModel:
    def __init__(self, book):
        self.book = book
        self.gt = load_gt(book)
        self.pm = {p['page_name']: p['page_id'] for p in
                   json.loads((REPO / 'prepared' / book / 'manifest.json').read_text())['pages']}
        self.pages_gt = {k[0] for k in self.gt}
        self.cols_by_page = C.defaultdict(list)
        for k, v in self.gt.items():
            self.cols_by_page[k[0]].append((k[1], v))
        # dòng kim thô
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
                        per[ci].append((min(ys), max(ys), b.get('transcription', '')))
                        break
            for ci, L in per.items():
                L.sort()
                if len(L) == 2:
                    self.klines[(pid, ci)] = [(L[0][0], L[0][1], 1, L[0][2]), (L[1][0], L[1][1], 2, L[1][2])]
        # mẫu khe: như h01 — mọi ô labels_gated có bbox, cột GT 6+8, cột hình học == cột eval
        tt = C.defaultdict(list)
        for r in csv.DictReader(open(REPO / 'prepared' / book / 'dataset_out/labels_gated.csv', encoding='utf-8')):
            try:
                x1, y1, x2, y2 = json.loads(r['bbox'])
            except Exception:
                continue
            pid = self.pm.get(r['page'], '')
            if pid not in self.pages_gt:
                continue
            g = self._geo(pid, (x1 + x2) / 2, (y1 + y2) / 2)
            if g is None:
                continue
            gci, gv, t = g
            if len(gv['t1']) == 6 and len(gv['t2']) == 8 and gci == int(r['column']) - 1:
                tt[int(r['syl_idx'])].append(t)
        self.mau = [ST.median(tt[k]) for k in range(14)]

    def _geo(self, pid, cx, cy):
        cand = self.cols_by_page.get(pid)
        if not cand:
            return None
        inside = [(ci, v) for ci, v in cand if v['box'][0] <= cx <= v['box'][0] + v['box'][2]]
        if inside:
            gci, gv = inside[0]
        else:
            gci, gv = min(cand, key=lambda kv: min(abs(cx - kv[1]['box'][0]), abs(cx - kv[1]['box'][0] - kv[1]['box'][2])))
        x, y, w, h = gv['box']
        return gci, gv, (cy - y) / h

    def slot(self, page, column, syl_idx, bbox):
        column = int(column); si = int(syl_idx); ec = column - 1
        pid = self.pm.get(page, '')
        part = 1 if si < N1 else 2
        pos = si if part == 1 else si - N1
        ev = self.gt.get((pid, ec))
        s = (ev['t1'] if part == 1 else ev['t2']) if ev else ''
        gchar = s[pos] if 0 <= pos < len(s) else ''
        out = dict(slot_ok='', gt_img='', gt_char=gchar, slot_tmpl='', slot_kim='', geo_col='')
        if pid not in self.pages_gt or not bbox:
            return out
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        g = self._geo(pid, cx, cy)
        if g is None:
            return out
        gci, gv, t = g
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
            so = 0
        elif on_t and (on_k is None or on_k):
            so = 1
        elif (not on_t) and on_k is False:
            so = 0
        else:
            so = ''
        gimg = ''
        if so == 1:
            gimg = gchar
        else:
            k_img = k_t if (k_k is None or k_k == k_t) else None
            if k_img is not None and so == 0:
                s2 = gv['t1'] if k_img < N1 else gv['t2']; p2 = k_img if k_img < N1 else k_img - N1
                gimg = s2[p2] if 0 <= p2 < len(s2) else ''
        s3_ = gv['t1'] if k_t < N1 else gv['t2']; p3_ = k_t if k_t < N1 else k_t - N1
        gt_tmpl = s3_[p3_] if 0 <= p3_ < len(s3_) else ''      # chữ GT ở khe MẪU (chỉ mô hình mẫu, độc lập hình học kim)
        out.update(slot_ok=so, gt_img=gimg, slot_tmpl=k_t, slot_kim='' if k_k is None else k_k, geo_col=gci,
                   col_ok=int(col_ok), gt_tmpl=gt_tmpl)
        return out
