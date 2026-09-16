import numpy as np, pandas as pd, sys
SP="/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/fb7207dc-964e-4d78-93c8-473f625e88a0/scratchpad"
REPO="/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"
R=pd.read_pickle(f"{SP}/a4_recs.pkl")
df=pd.read_csv(f"{REPO}/dataset_out/labels_final.csv",dtype=str,keep_default_na=False)
z=np.load(f"{REPO}/lab/gan_nhan_2026-09-13/emb.npz",allow_pickle=True)
LP=z["LP"].astype(np.float32); classes=list(z["classes"]); cid={s:i for i,s in enumerate(classes)}
ok=np.load(f"{REPO}/lab/gan_nhan_2026-09-13/crops.npz")["ok"]
print("cặp",len(R),"LP",LP.shape,"ok",ok.sum(),"/",len(ok))
# sanity: prod_syl vs df.syllable at row
rv=R[R.row.notna()].copy(); rv["row"]=rv.row.astype(int)
print("kiểm row↔labels_final: syllable khớp", (df.syllable.values[rv.row.values]==rv.prod_syl.values).mean())
RV=R[R.v3=="REVIEW"].copy()
print("\nREVIEW v3:",len(RV)); print(RV.prod_tier.value_counts().to_dict())
print("REVIEW có row:",RV.row.notna().sum(), "· có crop(ok):", int(ok[RV.row.dropna().astype(int).values].sum()))
RV["has_row"]=RV.row.notna()
RV["split"]=[df.split.values[int(r)] if pd.notna(r) else "(khe)" for r in RV.row]
print("split:",RV.split.value_counts().to_dict())
# split by page even for khe: use page of df
pg_split=df.drop_duplicates(["book","page"]).set_index(["book","page"]).split.to_dict()
RV["split"]=[pg_split.get((b,p),"?") for b,p in zip(RV.book,RV.page)]
print("split(theo trang):",RV.split.value_counts().to_dict())
print("cờ (¬direct∧¬sim_unique∧¬ctx):",((~RV.direct)&(~RV.sim_unique)&(~RV.bigram)&(~RV.corpus2)&(~RV.corpus4)&(~RV.tone)).sum())
print("p≥0,8:",(RV.p>=0.8).sum(),"p≥0,5:",(RV.p>=0.5).sum())
print("âm ∈ 706 lớp:",RV.s.isin(cid).sum(), "· syl_changed:",RV.syl_changed.sum())
RV.to_pickle(f"{SP}/rv_review.pkl")
