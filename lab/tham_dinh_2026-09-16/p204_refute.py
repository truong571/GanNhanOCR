import numpy as np, pandas as pd, pickle
SP="/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/fb7207dc-964e-4d78-93c8-473f625e88a0/scratchpad"
REPO="/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"
R=pd.read_pickle(f"{SP}/a4_recs.pkl")
df=pd.read_csv(f"{REPO}/dataset_out/labels_final.csv",dtype=str,keep_default_na=False)
z=np.load(f"{REPO}/lab/gan_nhan_2026-09-13/emb.npz",allow_pickle=True)
LP=z["LP"].astype(np.float32); Z=z["Z"].astype(np.float32); classes=list(z["classes"]); cid={s:i for i,s in enumerate(classes)}
Zn=Z/np.linalg.norm(Z,axis=1,keepdims=True).clip(1e-6)
top1=LP.argmax(1)
cols=pickle.load(open(f"{REPO}/lab/gan_nhan_2026-09-13/cols.pkl","rb"))
colmap={(c["book"],c["page"],c["column"]):c for c in cols}
Rall=R[R.row.notna()].copy(); Rall["row"]=Rall.row.astype(int)
pg_split=df.drop_duplicates(["book","page"]).set_index(["book","page"]).split.to_dict()
Rall["split"]=[pg_split.get((b,p)) for b,p in zip(Rall.book,Rall.page)]
rows_by_key={(b,p,c,j):r for b,p,c,j,r in zip(Rall.book,Rall.page,Rall.column,Rall.j,Rall.row)}
def analyse(sub,name,thr=0.9):
    ke=[];kk=[]
    for b,p,c,j,row,s in zip(sub.book,sub.page,sub.column,sub.j,sub.row,sub.s):
        col=colmap[(b,p,c)]; syl=[x.lower() for x in col["syl"]]
        own=float(np.exp(LP[row,cid[s]])) if s in cid else np.nan
        for jj in range(len(syl)):
            if jj==j or syl[jj]==s or syl[jj] not in cid: continue
            pv=float(np.exp(LP[row,cid[syl[jj]]]))
            nb=rows_by_key.get((b,p,c,jj))
            # cos giữa crop này và crop hàng xóm; cos giữa crop này và nguyên mẫu lớp hàng xóm là LP rồi
            cos_nb=float(Zn[row]@Zn[nb]) if nb is not None else np.nan
            rec=(row,jj,pv,own,cos_nb,abs(jj-j)==1)
            (ke if abs(jj-j)==1 else kk).append(rec)
    ke=pd.DataFrame(ke,columns=["row","jj","pv","own","cos_nb","adj"]); kk=pd.DataFrame(kk,columns=ke.columns)
    n_cells=len(sub)
    k9=ke[ke.pv>=thr]; kk9=kk[kk.pv>=thr]
    print(f"\n== {name}: ô={n_cells} cặp kề={len(ke)} cặp không kề={len(kk)}")
    print(f"  kề P≥{thr}: {len(k9)} cặp / {k9.row.nunique()} ô ({k9.row.nunique()/n_cells:.2%} ô) | trong đó P_own<0,1: {(k9.own<0.1).sum()} | P_own≥0,5: {(k9.own>=0.5).sum()} | NaN own: {k9.own.isna().sum()}")
    print(f"  không kề P≥{thr}: {len(kk9)} cặp / {kk9.row.nunique()} ô | P_own<0,1: {(kk9.own<0.1).sum()}")
    # FAR 'thật' của CNN = ô có P_own cao (crop đúng glyph) mà vẫn cho âm kề ≥thr
    both=k9[(k9.own>=0.5)]
    print(f"  FAR-CNN-thuần (P_own≥0,5 ∧ P_kề≥{thr}): {len(both)} / {len(ke)} = {len(both)/len(ke):.3%}")
    # tỉ lệ ô trượt hộp ước tính: P_kề≥0,9 ∧ P_own<0,1
    shift=k9[k9.own<0.1]
    print(f"  ô nghi TRƯỢT HỘP (P_kề≥{thr} ∧ P_own<0,1): {shift.row.nunique()} ô = {shift.row.nunique()/n_cells:.2%} ô")
    return ke,kk
A=Rall[(Rall.split=="test")&(Rall.v3=="CHAR_A")]
keA,kkA=analyse(A,"CHAR_A test")
for thr in (0.8,0.5):
    k=keA[keA.pv>=thr]; print(f"  [thr {thr}] kề: {len(k)} cặp, P_own<0,1: {(k.own<0.1).sum()}, P_own≥0,5: {(k.own>=0.5).sum()}")
# REVIEW v3 test
RV=pd.read_pickle(f"{SP}/rv_review_vis.pkl"); RT=RV[RV.split=="test"]
keR,kkR=analyse(RT,"REVIEW v3 test")
# cụm theo cột: các ô trượt hộp có nằm cùng cột (chuỗi trượt) không?
k9=keA[(keA.pv>=0.9)&(keA.own<0.1)]
key=[(b,p,c) for b,p,c in zip(df.book.values[k9.row],df.page.values[k9.row],df.column.values[k9.row])]
vc=pd.Series(key).value_counts()
print("\nô trượt hộp CHAR_A test theo cột: số cột",len(vc),"| cột có ≥2 ô:",(vc>=2).sum(),"| ô trong cột ≥2:",vc[vc>=2].sum())
print(vc.head(8).to_dict())
# tier/rule/page_cot_lech của các ô trượt
sub=df.iloc[k9.row.unique()]
print("page_cot_lech:",sub.page_cot_lech.value_counts().to_dict(),"| seg_flag:",sub.seg_flag.value_counts().head(5).to_dict(),"| crop_quality_flag:",sub.crop_quality_flag.value_counts().head(5).to_dict())
