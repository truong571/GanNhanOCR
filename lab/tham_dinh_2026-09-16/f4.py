import numpy as np, pandas as pd, sys
from collections import Counter
from scipy.cluster.vq import kmeans2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
R='/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR'
df=pd.read_csv(f'{R}/dataset_out/labels_final.csv',dtype=str,keep_default_na=False)
z=np.load(f'{R}/lab/gan_nhan_2026-09-13/crops.npz'); ok=z['ok']
d=np.load(f'{R}/lab/gan_nhan_2026-09-13/emb.npz'); Z=d['Z'].astype(np.float32)
E=pd.read_csv(f'{R}/lab/gan_nhan_2026-09-13/e3_all_classes.csv')
g=df[(df.tier=='GOLD')&ok]
def ari(a,b):
    a=np.asarray(a);b=np.asarray(b)
    from math import comb
    ct=Counter(zip(a,b)); ra=Counter(a); rb=Counter(b)
    s=sum(comb(v,2) for v in ct.values()); sa=sum(comb(v,2) for v in ra.values()); sb=sum(comb(v,2) for v in rb.values())
    n=comb(len(a),2); exp=sa*sb/n if n else 0; mx=(sa+sb)/2
    return (s-exp)/(mx-exp) if mx!=exp else 0.0
def dbscan(S,eps,ms):
    # S cosine sim; dist=1-S
    n=len(S); D=1-S; nb=[np.where(D[i]<=eps)[0] for i in range(n)]
    core=np.array([len(x)>=ms for x in nb]); lab=-np.ones(n,int); c=0
    for i in range(n):
        if lab[i]!=-1 or not core[i]: continue
        lab[i]=c; st=[i]
        while st:
            j=st.pop()
            for k in nb[j]:
                if lab[k]==-1:
                    lab[k]=c
                    if core[k]: st.append(k)
        c+=1
    return lab
rows=[]
for _,r in E.iterrows():
    grp=g[(g.book==r.book)&(g.syllable==r.syl)]
    A=grp[grp.label==r.M].index.to_numpy(); B=grp[grp.label==r.L].index.to_numpy()
    if len(A)>200: A=np.random.default_rng(0).choice(A,200,replace=False)
    idx=np.concatenate([A,B]); y=np.r_[np.zeros(len(A)),np.ones(len(B))].astype(int)
    X=Z[idx]; S=X@X.T
    iu=np.triu_indices(len(idx),1)
    wa=S[:len(A),:len(A)][np.triu_indices(len(A),1)].mean(); wb=S[len(A):,len(A):][np.triu_indices(len(B),1)].mean(); ab=S[:len(A),len(A):].mean()
    out=dict(book=r.book,syl=r.syl,M=r.M,L=r.L,nA=len(A),nB=len(B),v=r.v,knn=r.knn,cosA=round(wa,3),cosB=round(wb,3),cosAB=round(ab,3))
    for eps in (0.15,0.25,0.35):
        lab=dbscan(S,eps,5); out[f'db{eps}_nc']=int(lab.max()+1); out[f'db{eps}_noise']=round((lab==-1).mean(),2)
        m=lab!=-1; out[f'db{eps}_ari']=round(ari(y[m],lab[m]),2) if m.sum()>1 and lab.max()>=1 else 0.0
    best=-1
    for seed in range(5):
        c,l=kmeans2(X,2,seed=seed,minit='++'); best=max(best,ari(y,l))
    out['km_ari']=round(best,2)
    Dm=np.clip(1-S,0,2); np.fill_diagonal(Dm,0); Lk=linkage(squareform(Dm,checks=False),'average'); l=fcluster(Lk,2,'maxclust')
    out['agg_ari']=round(ari(y,l),2); out['agg_min']=int(min(Counter(l).values()))
    rows.append(out)
T=pd.DataFrame(rows)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
T.to_csv(sys.argv[1] if len(sys.argv)>1 else '/dev/null',index=False)
for v in ('HAI','MOT','mo'):
    t=T[T.v==v]; print(f'== {v} n={len(t)}')
    print(' cos trong-lớp A trung vị',t.cosA.median().round(3),'B',t.cosB.median().round(3),'giữa',t.cosAB.median().round(3))
    for eps in (0.15,0.25,0.35): print(f' DBSCAN eps{eps}: nhiễu trung vị {t[f"db{eps}_noise"].median():.2f}, số cụm trung vị {t[f"db{eps}_nc"].median():.0f}, ARI trung vị {t[f"db{eps}_ari"].median():.2f}, ARI>=0.5: {(t[f"db{eps}_ari"]>=0.5).sum()}')
    print(f' kmeans ARI trung vị {t.km_ari.median():.3f}, >=0.5: {(t.km_ari>=0.5).sum()}, <0.3: {(t.km_ari<0.3).sum()}')
    print(f' agglo ARI trung vị {t.agg_ari.median():.3f}, >=0.5: {(t.agg_ari>=0.5).sum()}, cụm nhỏ nhất trung vị {t.agg_min.median():.0f}')
print(T[T.v=='HAI'][['book','syl','M','L','nA','nB','knn','cosA','cosB','cosAB','db0.35_nc','db0.35_noise','km_ari','agg_ari','agg_min']].sort_values('km_ari').to_string(index=False))
