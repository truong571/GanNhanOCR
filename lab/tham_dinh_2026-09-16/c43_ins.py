"""C4-3: (A) kịch bản INS tổng hợp (|G|=|OCR|=n, |Q|=n+1) 3 biến thể + đối chiếu hopj_ok kịch bản 1; (B) kiểm hộp thật equal_ocr theo text-box thô boxes_raw."""
import sys, glob, json, pickle, random, time
from pathlib import Path
from collections import Counter
import numpy as np
REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"lab/gan_nhan_2026-09-13"))
SCR = Path(__file__).parent
import thuc_nghiem as T
from pipeline.align_engine.anchor_align import realign_column
from pipeline.align_engine.syllable_normalize import normalize_column, build_readings
sys.path.insert(0, str(REPO/"train_crop")); from infer_centernet import _nms_vertical
from core.align.run_full import nom_cols_hybrid
from pipeline.step2_align import _get_qn_lines
pbc = pickle.load(open(SCR/"pb_cache.pkl","rb"))
readings = build_readings(T.qn)
QS = sorted(T.qs)
def sample_pages():
    rng = random.Random(11); pages=[]
    for b in T.BOOKS:
        ts=[t for t in sorted(glob.glob(str(REPO/"prepared"/b/"transcriptions"/"page_*.json"))) if not t.endswith("_qn_ocr_cache.json")]
        pages += [(b, Path(t).stem) for t in rng.sample(ts,14)]
    return pages
def lab_acc(ops, truth_syl_of_nom):
    """nhãn đúng: cặp match có syl_idx == truth[nom_idx]; ô OCR không match (del) tính sai."""
    n=len(truth_syl_of_nom); ok=0
    m={o["nom_idx"]:o["syl_idx"] for o in ops if o["op"]=="match"}
    for t in range(n): ok += m.get(t,-1)==truth_syl_of_nom[t]
    return ok, n
def geo(thr, mg, matrix, mname):
    T.set_matrix(matrix); rng=random.Random(11); rng2=random.Random(23); pages=sample_pages(); tot=Counter()
    for b,page in pages:
        dd=REPO/"prepared"/b; od=json.load(open(dd/"detected"/f"{page}_ocr_cache.json",encoding="utf-8"))
        lines,_=_get_qn_lines(dd,page,T.qs); keys=sorted(lines)
        cs=nom_cols_hybrid(od["columns"],4)
        if len(cs)!=9: cs=nom_cols_hybrid(od["columns"],1)
        pb=pbc[(T.BOOKS[b],page)]
        for i in range(min(len(cs),len(keys))):
            cl=cs[i]; chars=[c["char"] for c in cl["chars"]]
            syl,_=normalize_column(cl["chars"],lines[keys[i]],T.qs,readings)
            if len(chars)!=len(syl) or len(chars)<10: continue
            x1,x2=cl["x_range"]; m=(x2-x1)*mg
            col=_nms_vertical([bb for bb in pb if bb[4]>=thr and x1-m<=(bb[0]+bb[2])/2<=x2+m],0.45)
            if len(col)!=len(chars): continue
            tot["col_used"]+=1
            n=len(chars); k=rng.randrange(2,n-2)
            # kịch bản 1 (a3_geo): OCR rụng chữ k, Q đủ  ==  INS-c (G và OCR cùng rụng k, Q đủ) — ops không phụ thuộc hộp
            c2=chars[:k]+chars[k+1:]; truth={t:(t if t<k else t+1) for t in range(n-1)}
            ok,nn=lab_acc(realign_column(c2,syl,T.qn,T.sim),truth); tot["insc_ok"]+=ok; tot["insc_n"]+=nn
            # DEL (a3_geo): Q rụng âm k
            s2=syl[:k]+syl[k+1:]; truth_d={t:(t if t<k else t-1) for t in range(n)}; truth_d[k]=-1
            ops=realign_column(chars,s2,T.qn,T.sim); mm={o["nom_idx"]:o["syl_idx"] for o in ops if o["op"]=="match"}
            for t in range(n):
                if t==k: tot["del_k_del"]+= (t not in mm); continue
                tot["del_n"]+=1; tot["del_ok"]+= mm.get(t,-2)==truth_d[t]
            # INS-rand: VietOCR bịa 1 âm ngẫu nhiên (trong từ điển) chèn trước vị trí k; |G|=|OCR|=n, |Q|=n+1
            s3=syl[:k]+[rng2.choice(QS)]+syl[k:]; truth_i={t:(t if t<k else t+1) for t in range(n)}
            ok,nn=lab_acc(realign_column(chars,s3,T.qn,T.sim),truth_i); tot["insr_ok"]+=ok; tot["insr_n"]+=nn
            # INS-dup: VietOCR đọc trùng âm k (từ dính/tách đôi) -> nhãn đúng nếu text bằng nhau
            s4=syl[:k+1]+syl[k:]; ops4=realign_column(chars,s4,T.qn,T.sim); m4={o["nom_idx"]:o["syl_idx"] for o in ops4 if o["op"]=="match"}
            for t in range(n):
                tot["insd_n"]+=1; j=m4.get(t,-1); tot["insd_ok"]+= (j>=0 and s4[j]==syl[t])
            # INS-frag: VietOCR tách âm k thành mảnh ngoài từ điển + phần còn lại (mảnh = 2 ký tự đầu)
            frag=syl[k][:2] if len(syl[k])>2 else syl[k]+"x"
            s5=syl[:k]+[frag]+syl[k:]; ok,nn=lab_acc(realign_column(chars,s5,T.qn,T.sim),truth_i); tot["insf_ok"]+=ok; tot["insf_n"]+=nn
    print(f"[{mname} thr={thr} ±{mg}w] cột {tot['col_used']} · INS-c(=kb1 hộp[j]) nhãn {tot['insc_ok']/tot['insc_n']:.1%} ({tot['insc_n']}) · DEL nhãn {tot['del_ok']/tot['del_n']:.1%} ({tot['del_n']}, chữ k ra del {tot['del_k_del']/tot['col_used']:.0%}) · INS-rand {tot['insr_ok']/tot['insr_n']:.1%} · INS-dup {tot['insd_ok']/tot['insd_n']:.1%} · INS-frag {tot['insf_ok']/tot['insf_n']:.1%}")
if __name__=="__main__":
    t0=time.time()
    geo(0.2,0.25,T.PROD,"PROD"); geo(0.2,0.25,T.CALIB,"CALIB"); geo(0.3,0.5,T.PROD,"PROD-legacy")
    print(f"{time.time()-t0:.0f}s")
