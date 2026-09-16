import numpy as np, pandas as pd, sys
SP="/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/fb7207dc-964e-4d78-93c8-473f625e88a0/scratchpad"
REPO="/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"
sys.path.insert(0,f"{REPO}/lab/gan_nhan_2026-09-13")
RV=pd.read_pickle(f"{SP}/rv_review_vis.pkl")
R=pd.read_pickle(f"{SP}/a4_recs.pkl")
df=pd.read_csv(f"{REPO}/dataset_out/labels_final.csv",dtype=str,keep_default_na=False)
z=np.load(f"{REPO}/lab/gan_nhan_2026-09-13/emb.npz",allow_pickle=True)
LP=z["LP"].astype(np.float32); classes=list(z["classes"]); cid={s:i for i,s in enumerate(classes)}
top1=LP.argmax(1)
import thuc_nghiem as T
cols=T.load_cols()
colmap={(c["book"],c["page"],c["column"]):c for c in cols}
# FAR: P(âm hàng xóm | crop) trên ô REVIEW test và trên MỌI ô test
def far(sub,name):
    n=0; f9=0; f8=0; f5=0; a=0
    for b,p,c,j,row,s in zip(sub.book,sub.page,sub.column,sub.j,sub.row,sub.s):
        col=colmap[(b,p,c)]; syl=[x.lower() for x in col["syl"]]
        for jj in (j-1,j+1):
            if 0<=jj<len(syl) and syl[jj]!=s and syl[jj] in cid:
                pv=float(np.exp(LP[int(row),cid[syl[jj]]])); n+=1
                f9+=pv>=0.9; f8+=pv>=0.8; f5+=pv>=0.5; a+=top1[int(row)]==cid[syl[jj]]
    print(f"{name:40s} cặp(ô,âm-hàng-xóm)={n:5d} | argmax==hàng xóm {a} ({a/n:.2%}) | P≥0,5 {f5} ({f5/n:.2%}) | P≥0,8 {f8} ({f8/n:.2%}) | P≥0,9 {f9} ({f9/n:.2%})")
RT=RV[RV.split=="test"]
far(RT,"REVIEW v3 · test")
Rall=R[R.row.notna()].copy(); Rall["row"]=Rall.row.astype(int)
pg_split=df.drop_duplicates(["book","page"]).set_index(["book","page"]).split.to_dict()
Rall["split"]=[pg_split.get((b,p)) for b,p in zip(Rall.book,Rall.page)]
far(Rall[Rall.split=="test"],"MỌI ô · test")
far(Rall[(Rall.split=="test")&(Rall.v3=="CHAR_A")],"CHAR_A · test")
# precision proxy: CHAR_A test: agree & P>=0.9
def rep(d,name):
    k=d.s.isin(cid); n=len(d)
    pv=np.array([float(np.exp(LP[r,cid[s]])) if s in cid else np.nan for r,s in zip(d.row,d.s)])
    ag=np.array([(top1[r]==cid[s]) if s in cid else False for r,s in zip(d.row,d.s)])
    print(f"{name:40s} n={n} ∈706 {k.sum()} | argmax==âm {ag.sum()} ({ag.sum()/k.sum():.1%} của ∈706) | P≥0,9 {(ag&(pv>=0.9)).sum()} ({(ag&(pv>=0.9)).sum()/k.sum():.1%}) | P≥0,8 {(ag&(pv>=0.8)).sum()} ({(ag&(pv>=0.8)).sum()/k.sum():.1%})")
print("\n== chuẩn so sánh (test) ==")
for t in ["CHAR_A","CHAR_B","SYL"]:
    rep(Rall[(Rall.split=="test")&(Rall.v3==t)],f"{t} · test")
# REVIEW test: có corpus2 (LOO) nhưng không corpus4
print("\n== REVIEW test × cờ ngữ cảnh yếu ==")
for nm,m in [("corpus2∧¬corpus4",RT.corpus2&~RT.corpus4),("không cờ nào",~(RT.bigram|RT.corpus2|RT.corpus4|RT.tone)),("sim_unique",RT.sim_unique),("syl_changed",RT.syl_changed)]:
    d=RT[m]; 
    if len(d): print(f"{nm:22s} n={len(d):4d} | argmax==âm {d.agree.sum()} ({d.agree.mean():.1%}) | P≥0,9 {(d.agree&(d.p_vis>=0.9)).sum()} | P≥0,8 {(d.agree&(d.p_vis>=0.8)).sum()}")
# âm ngoài 706 lớp: có trong từ điển không?
qn=T.qn
out=RV[~RV.in_cls]
print("\nâm ∉706:",len(out),"| ∈ từ điển QN:",out.s.isin(set(qn)).sum(), "| ví dụ:",out.s.value_counts().head(12).to_dict())
