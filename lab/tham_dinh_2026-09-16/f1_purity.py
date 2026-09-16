import numpy as np, pandas as pd
from collections import Counter
R='/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/'
df=pd.read_csv(R+'dataset_out/labels_final.csv',dtype=str,keep_default_na=False)
ok=np.load(R+'lab/gan_nhan_2026-09-13/crops.npz')['ok']
Z=np.load(R+'lab/gan_nhan_2026-09-13/emb.npz')['Z']
Z=Z/np.linalg.norm(Z,axis=1,keepdims=True)
e3=pd.read_csv(R+'lab/gan_nhan_2026-09-13/e3_all_classes.csv')
g=df[(df.tier=='GOLD')&ok]
def kmeans(E,k,seed=0,it=40):
    rng=np.random.default_rng(seed); C=E[rng.choice(len(E),k,replace=False)]
    for _ in range(it):
        a=np.argmax(E@C.T,1)
        C=np.array([E[a==j].mean(0) if (a==j).any() else C[j] for j in range(k)])
        C/=np.linalg.norm(C,axis=1,keepdims=True)
    return a
def ari(y,a):
    from scipy.special import comb
    ct=pd.crosstab(y,a).to_numpy()
    s_ij=comb(ct,2).sum(); s_i=comb(ct.sum(1),2).sum(); s_j=comb(ct.sum(0),2).sum(); n=comb(ct.sum(),2)
    exp=s_i*s_j/n; mx=(s_i+s_j)/2
    return (s_ij-exp)/(mx-exp) if mx!=exp else 0.0
rows=[]
for v in ['MOT','HAI','mo']:
  for _,r in e3[e3.v==v].iterrows():
    grp=g[(g.book==r.book)&(g.syllable==r.syl)&(g.label.isin([r.M,r.L]))]
    idx=grp.index.to_numpy(); y=(grp.label==r.L).to_numpy().astype(int)
    E=Z[idx]; a=kmeans(E,2)
    ct=pd.crosstab(y,a).to_numpy()
    purity=ct.max(0).sum()/ct.sum(); maj=max(1-y.mean(),y.mean())
    # cluster-level: does the bigger cluster hold >98% of cells? (shape-purity: one dense cluster)
    big=np.bincount(a).max()/len(a)
    # connectivity: fraction in largest CC at cos>=0.6
    S=E@E.T; A=S>=0.6
    seen=np.zeros(len(a),bool); best=0
    for s in range(len(a)):
        if seen[s]: continue
        st=[s]; seen[s]=True; c=0
        while st:
            u=st.pop(); c+=1
            for w in np.where(A[u]&~seen)[0]: seen[w]=True; st.append(w)
        best=max(best,c)
    # 5-NN LOO: fraction whose neighbour-majority label == own label
    np.fill_diagonal(S,-9); nn=np.argsort(-S,1)[:,:5]; p=(y[nn].mean(1)>0.5).astype(int); knn_acc=(p==y).mean()
    rows.append(dict(v=v,book=r.book,syl=r.syl,M=r.M,L=r.L,n=len(a),nA=int((y==0).sum()),nB=int(y.sum()),
        maj=round(maj,3),purity=round(purity,3),ari=round(ari(y,a),3),bigclus=round(big,3),cc06=round(best/len(a),3),knn_acc=round(knn_acc,3)))
T=pd.DataFrame(rows)
T.to_csv('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/fb7207dc-964e-4d78-93c8-473f625e88a0/scratchpad/f1_purity.csv',index=False)
for v in ['MOT','HAI','mo']:
    t=T[T.v==v]
    print(v,len(t),'lớp · ô',t.n.sum(),'· nA',t.nA.sum(),'· nB',t.nB.sum())
    print('  median maj %.3f purity %.3f ari %.3f bigclus %.3f cc06 %.3f knn_acc %.3f'%(t.maj.median(),t.purity.median(),t.ari.median(),t.bigclus.median(),t.cc06.median(),t.knn_acc.median()))
    print('  weighted maj %.3f purity %.3f'%((t.maj*t.n).sum()/t.n.sum(),(t.purity*t.n).sum()/t.n.sum()))
    print('  n lớp purity>=0.98:',(t.purity>=0.98).sum(),'· purity-maj>0.05:',((t.purity-t.maj)>0.05).sum(),'· bigclus>=0.98:',(t.bigclus>=0.98).sum(),'· cc06>=0.98:',(t.cc06>=0.98).sum())
t=T[T.v=='MOT']; print('MOT: tổng ô',t.n.sum(),'· tỉ lệ mã thứ hai',round(t.nB.sum()/t.n.sum(),3),'· min maj',t.maj.min(),'· max maj',t.maj.max())
print(t.sort_values('n',ascending=False).head(8).to_string(index=False))
