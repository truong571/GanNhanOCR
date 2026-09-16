import numpy as np, pandas as pd, sys, cv2
sys.path.insert(0,'/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/gan_nhan_2026-09-13')
from thi_giac_am_tiet import _hog
exec(open('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/fb7207dc-964e-4d78-93c8-473f625e88a0/scratchpad/f4_mine_x91.py').read().split('rows=[]')[0])
X=np.load(R+'lab/gan_nhan_2026-09-13/crops.npz')['X']
sel=e3[e3.v=='HAI'].sort_values('nB',ascending=False).head(5)
for _,r in sel.iterrows():
    idx,y=get(r.book,r.syl,r.M,r.L);H=_hog(X[idx])
    a=ari(y,kmeans(H,2))
    nB=int(y.sum());A=idx[y==0][:2*nB];B=idx[y==1];idx2=np.concatenate([A,B]);y2=np.r_[np.zeros(len(A)),np.ones(len(B))]
    a2=ari(y2,kmeans(_hog(X[idx2]),2))
    # kNN-with-labels bacc on HOG for reference (from e3)
    print(f"{r.book} {r.syl} {r.M}/{r.L} nA={int((y==0).sum())} nB={nB} hog_knn={r.hog} hog_km={a:.2f} hog_km_bal={a2:.2f}")
