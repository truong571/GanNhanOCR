"""cmd_geo tham số hoá (thr, margin, ma trận) trên cache hộp trang; + kịch bản DEL (VietOCR rụng âm) + kiểm mạch lạc |G|=|Q|>|OCR|."""
import sys, glob, json, pickle, random
from pathlib import Path
from collections import Counter
import numpy as np
REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"lab/gan_nhan_2026-09-13"))
SCR = Path(__file__).parent
import thuc_nghiem as T
from pipeline.align_engine.align_production import _monotone_assign, _reseg_column, _detect
from pipeline.align_engine.anchor_align import realign_column
from pipeline.align_engine.syllable_normalize import normalize_column, build_readings
sys.path.insert(0, str(REPO/"train_crop")); from infer_centernet import _nms_vertical
from core.align.run_full import nom_cols_hybrid
from pipeline.step2_align import _get_qn_lines
pbc = pickle.load(open(SCR/"pb_cache.pkl","rb"))
readings = build_readings(T.qn)
def sample_pages():
    rng = random.Random(11); pages=[]
    for b in T.BOOKS:
        ts=[t for t in sorted(glob.glob(str(REPO/"prepared"/b/"transcriptions"/"page_*.json"))) if not t.endswith("_qn_ocr_cache.json")]
        pages += [(b, Path(t).stem) for t in rng.sample(ts,14)]
    return pages
def geo(thr, mg, matrix, mname):
    T.set_matrix(matrix); rng=random.Random(11); pages=sample_pages(); tot=Counter()
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
            tot["col_eq"]+=1
            x1,x2=cl["x_range"]; m=(x2-x1)*mg
            col=_nms_vertical([bb for bb in pb if bb[4]>=thr and x1-m<=(bb[0]+bb[2])/2<=x2+m],0.45); col.sort(key=lambda bb:(bb[1]+bb[3])/2)
            if len(col)!=len(chars): tot["col_skip"]+=1; continue
            tot["col_used"]+=1
            boxes=[[int(v) for v in bb[:4]] for bb in col]
            y_top=min(c["bbox"][1] for c in cl["chars"]); y_bot=max(c["bbox"][3] for c in cl["chars"]); H=y_bot-y_top
            k=rng.randrange(2,len(chars)-2)
            # --- kịch bản 1: OCR rụng 1 chữ (|G|=|Q|=n, |OCR|=n-1)
            c2=chars[:k]+chars[k+1:]; n2=len(c2); truth={t:(t if t<k else t+1) for t in range(n2)}
            cys=[y_top+(t+0.5)*H/n2 for t in range(n2)]
            cl2={"chars":[{"char":c2[t],"bbox":[x1,int(y_top+t*H/n2),x2,int(y_top+(t+1)*H/n2)]} for t in range(n2)],"x_range":cl["x_range"]}
            assigned=_monotone_assign(cys,boxes,_reseg_column(cl2))
            ops=realign_column(c2,syl,T.qn,T.sim); ins=[o["syl_idx"] for o in ops if o["op"]=="ins"]
            g=ins[0] if len(ins)==1 else None
            tot["one_gap"]+= g is not None
            for t in range(n2):
                tot["n"]+=1
                q=next((qi for qi,bb in enumerate(boxes) if assigned and assigned[t]==bb),None)
                tot["prod_ok"]+= q==truth[t]
                # hộp[j] ↔ âm j: j = syl_idx của cặp match chứa nom_idx t
                j=next((o["syl_idx"] for o in ops if o["op"]=="match" and o["nom_idx"]==t),None)
                tot["hopj_ok"]+= (j==truth[t]); tot["hopj_none"]+= j is None
            # --- kịch bản 2 (DEL): VietOCR rụng 1 âm (|G|=|OCR|=n, |Q|=n-1) -> nhánh |G|=|OCR|: hộp[nom_idx]
            s2=syl[:k]+syl[k+1:]
            ops2=realign_column(chars,s2,T.qn,T.sim)
            for o in ops2:
                if o["op"]!="match": continue
                tot["n_del"]+=1
                # hộp theo chữ OCR i luôn đúng theo định nghĩa (G theo thứ tự OCR); câu hỏi là nhãn: âm o.syl_idx có đúng chữ o.nom_idx?
                truth_syl = o["nom_idx"] if o["nom_idx"]<k else o["nom_idx"]-1
                tot["del_pair_ok"]+= (o["syl_idx"]==truth_syl)
            # hộp[j] mù quáng ở kịch bản DEL (|G|=n ≠ |Q|=n-1) -> nếu cứ ép hộp[j]: sai từ k trở đi
            for o in ops2:
                if o["op"]=="match": tot["del_hopj_ok"]+= (o["syl_idx"]==o["nom_idx"])
    n=tot["n"]
    print(f"[{mname} thr={thr} ±{mg}w] cột khớp {tot['col_eq']} · dùng {tot['col_used']} ({tot['col_used']/tot['col_eq']:.0%}) · rụng-chữ {n} ô: prod {tot['prod_ok']/n:.1%} · hộp[j] {tot['hopj_ok']/n:.1%} (1 khe: {tot['one_gap']/tot['col_used']:.0%}) · DEL {tot['n_del']} cặp: nhãn đúng khi hộp[nom_idx] {tot['del_pair_ok']/tot['n_del']:.1%} · nếu ép hộp[j] {tot['del_hopj_ok']/tot['n_del']:.1%}")
for thr,mg in ((0.3,0.5),(0.2,0.5),(0.2,0.10),(0.2,0.25)):
    geo(thr,mg,T.PROD,"PROD")
geo(0.2,0.25,T.CALIB,"CALIB")
