import numpy as np, pandas as pd
SP="/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/fb7207dc-964e-4d78-93c8-473f625e88a0/scratchpad"
REPO="/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"
RV=pd.read_pickle(f"{SP}/rv_review.pkl")
z=np.load(f"{REPO}/lab/gan_nhan_2026-09-13/emb.npz",allow_pickle=True)
LP=z["LP"].astype(np.float32); classes=list(z["classes"]); cid={s:i for i,s in enumerate(classes)}
top1=LP.argmax(1); pmax=np.exp(LP.max(1))
RV=RV[RV.row.notna()].copy(); RV["row"]=RV.row.astype(int)
RV["in_cls"]=RV.s.isin(cid)
RV["p_vis"]=[float(np.exp(LP[r,cid[s]])) if s in cid else np.nan for r,s in zip(RV.row,RV.s)]
RV["agree"]=[(top1[r]==cid[s]) if s in cid else False for r,s in zip(RV.row,RV.s)]
RV["pmax"]=pmax[RV.row.values]
RV["top1_syl"]=[classes[top1[r]] for r in RV.row]
def rep(m,name):
    d=RV[m]; n=len(d); k=d.in_cls.sum()
    a=d.agree.sum(); a9=(d.agree&(d.p_vis>=0.9)).sum(); a8=(d.agree&(d.p_vis>=0.8)).sum(); a5=(d.agree&(d.p_vis>=0.5)).sum()
    print(f"{name:34s} n={n:5d} âm∈706={k:5d} ({k/n:.0%}) | argmax==âm {a:5d} ({a/n:.1%} của n, {a/max(k,1):.1%} của ∈706) | P≥0,5 {a5:5d} ({a5/n:.1%}) | P≥0,8 {a8:5d} ({a8/n:.1%}) | P≥0,9 {a9:5d} ({a9/n:.1%})")
print("== REVIEW v3 có crop, theo split (test = out-of-fold) ==")
for sp in ["test","val","train"]:
    rep(RV.split==sp, f"split={sp}")
rep(RV.split.notna(),"TẤT CẢ (train in-sample cho GOLD cũ)")
print("\n== test, theo prod_tier ==")
for t in ["REVIEW","SILVER_uncalibrated","GOLD","SYLLABLE"]:
    rep((RV.split=="test")&(RV.prod_tier==t), f"test·{t}")
print("\n== train+val chưa từng học (REVIEW cũ + SILVER cũ; chỉ rò trang) ==")
rep((RV.split!="test")&RV.prod_tier.isin(["REVIEW","SILVER_uncalibrated"]),"train+val·REVIEW+SILVER cũ")
rep((RV.split!="test")&RV.prod_tier.isin(["GOLD","SYLLABLE"]),"train+val·GOLD+SYL cũ (in-sample)")
print("\n== test, theo prod_rule ==")
for r,g in RV[RV.split=="test"].groupby("prod_rule"):
    if len(g)>=20: rep((RV.split=="test")&(RV.prod_rule==r), f"test·{r}")
# khi không đồng thuận: top1 nói gì? có phải âm hàng xóm?
RV.to_pickle(f"{SP}/rv_review_vis.pkl")
