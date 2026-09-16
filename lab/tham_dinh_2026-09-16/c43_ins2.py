"""C4-3 (A'): như c43_ins nhưng đo theo VĂN BẢN nhãn (đúng nếu âm gán == âm thật), tách precision (trong cặp match) và recall (trên ô có âm thật)."""
import sys, random, time
from collections import Counter
from c43_ins import *
from c43_ins import T, pbc, readings, QS, REPO, sample_pages, json
from pipeline.step2_align import _get_qn_lines
from core.align.run_full import nom_cols_hybrid
from pipeline.align_engine.anchor_align import realign_column
from pipeline.align_engine.syllable_normalize import normalize_column
from infer_centernet import _nms_vertical
def run(thr, mg, matrix, mname):
    T.set_matrix(matrix); rng=random.Random(11); rng2=random.Random(23); pages=sample_pages(); tot=Counter()
    def score(tag, ops, truth_text, n_nom):
        mm={o["nom_idx"]:o["syl_idx"] for o in ops if o["op"]=="match"}
        for t in range(n_nom):
            has=truth_text[t] is not None
            if t in mm:
                lab=ops_syl[mm[t]]
                tot[tag+"_m"]+=1; tot[tag+"_m_ok"]+= (has and lab==truth_text[t])
                if not (has and lab==truth_text[t]): tot[tag+"_wrong"]+=1
            if has: tot[tag+"_n"]+=1; tot[tag+"_rec"]+= (t in mm and ops_syl[mm[t]]==truth_text[t])
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
            tot["col"]+=1; n=len(chars); k=rng.randrange(2,n-2)
            # baseline: cột nguyên vẹn (G=OCR=Q) — DP có tự sai không?
            ops_syl=syl; score("base", realign_column(chars,syl,T.qn,T.sim), syl, n)
            # INS-c: G & OCR cùng rụng chữ k, Q đủ (|G|=|OCR|=n-1 < |Q|=n) -> hộp[nom_idx] đúng theo định nghĩa; nhãn?
            c2=chars[:k]+chars[k+1:]; ops_syl=syl; score("insc", realign_column(c2,syl,T.qn,T.sim), [syl[t] for t in range(n) if t!=k], n-1)
            # DEL: Q rụng âm k (|G|=|OCR|=n > |Q|=n-1)
            s2=syl[:k]+syl[k+1:]; ops_syl=s2; score("del", realign_column(chars,s2,T.qn,T.sim), [syl[t] if t!=k else None for t in range(n)], n)
            # INS-rand: Q thừa 1 âm bịa
            s3=syl[:k]+[rng2.choice(QS)]+syl[k:]; ops_syl=s3; score("insr", realign_column(chars,s3,T.qn,T.sim), syl, n)
            # INS-frag: Q thừa mảnh ngoài từ điển
            frag=syl[k][:2] if len(syl[k])>2 else syl[k]+"x"; s5=syl[:k]+[frag]+syl[k:]; ops_syl=s5; score("insf", realign_column(chars,s5,T.qn,T.sim), syl, n)
    out=[f"[{mname} thr={thr} ±{mg}w] cột {tot['col']}"]
    for tag in ("base","insc","del","insr","insf"):
        out.append(f"{tag}: prec {tot[tag+'_m_ok']/tot[tag+'_m']:.1%} (sai {tot[tag+'_wrong']}/{tot[tag+'_m']}) · recall {tot[tag+'_rec']/tot[tag+'_n']:.1%}")
    print(" · ".join(out))
if __name__=="__main__":
    t0=time.time(); run(0.2,0.25,T.PROD,"PROD"); run(0.2,0.25,T.CALIB,"CALIB"); print(f"{time.time()-t0:.0f}s")
