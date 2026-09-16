"""Cải tiến 4 — proxy 'hộp gộp 2 chữ' theo nhánh đếm (thr 0,2 ±0,25w): tỉ lệ hộp cao > 1,6× trung vị cột."""
import sys, glob, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"); sys.path.insert(0, str(REPO))
SCR = Path(__file__).parent
from core.text.dictionary import load_qn_to_nom
from pipeline.align_engine import align_production as ap
from pipeline.align_engine.syllable_normalize import build_readings, normalize_column
sys.path.insert(0, str(REPO/"train_crop"))
from infer_centernet import _nms_vertical
qn = load_qn_to_nom(str(REPO/"dict/QuocNgu_SinoNom.csv")); qs = set(qn); readings = build_readings(qn)
BOOKS = {"SachThanhTruyen2": "stt2", "SachThanhTruyen4": "stt4", "SachThanhTruyen11": "stt11"}
pbc = pickle.load(open(SCR/"pb_cache.pkl","rb"))
def colboxes(pb, thr, mg, x1, x2):
    w = x2-x1
    col = [b for b in pb if b[4] >= thr and x1-mg*w <= (b[0]+b[2])/2 <= x2+mg*w]
    col = _nms_vertical(col, 0.45); col.sort(key=lambda b: (b[1]+b[3])/2); return col
rows=[]; t0=time.time()
for bdir, bcode in BOOKS.items():
    dd = REPO/"prepared"/bdir
    for tf in sorted(glob.glob(str(dd/"transcriptions"/"page_*.json"))):
        if tf.endswith("_qn_ocr_cache.json"): continue
        page = Path(tf).stem
        d = ap._detect(page, dd, qs)
        if d is None: continue
        cols, qn_lines, iter_pairs, binary, page_ok = d
        pb = pbc.get((bcode,page))
        if pb is None: continue
        for nom_idx, line_id in iter_pairs:
            cl = cols[nom_idx]; syl0 = qn_lines[line_id]
            if not syl0 or not cl.get("x_range"): continue
            syl,_ = normalize_column(cl.get("chars"), syl0, qs, readings)
            n_ocr, n_qn = len(cl["chars"]), len(syl); x1,x2 = cl["x_range"]
            G = colboxes(pb, 0.2, 0.25, x1, x2)
            hs = np.array([b[3]-b[1] for b in G]) if G else np.array([])
            med = np.median(hs) if len(hs) else 0
            tall = int((hs > 1.6*med).sum()) if len(hs) else 0
            short = int((hs < 0.5*med).sum()) if len(hs) else 0
            rows.append(dict(book=bcode,page=page,column=line_id,n_ocr=n_ocr,n_qn=n_qn,n_det=len(G),tall=tall,short=short))
df=pd.DataFrame(rows); df.to_csv(SCR/"c4_merge_proxy.csv", index=False)
G=df.n_det; nq=df.n_qn; no=df.n_ocr
br = np.where(G==nq, np.where(nq==no,'eqQ=OCR', np.where(nq>no,'eqQ>OCR','eqQ<OCR')),
     np.where(G==no, np.where(no>nq,'eqOCR>Q','eqOCR<Q'), 'conflict'))
df['branch']=br
g=df.groupby('branch').agg(cot=('n_det','size'), hop=('n_det','sum'), tall=('tall','sum'), short=('short','sum'), cot_tall=('tall', lambda s:(s>0).sum()))
g['tall_pct']=(g.tall/g.hop*100).round(2); g['cot_tall_pct']=(g.cot_tall/g.cot*100).round(1)
print(g.to_string()); print(f"{time.time()-t0:.0f}s, cột {len(df)}")
