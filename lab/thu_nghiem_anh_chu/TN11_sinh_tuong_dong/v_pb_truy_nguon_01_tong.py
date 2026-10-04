"""v_pb_truy_nguon_01 (04/10) — PHẢN BIỆN: tự tính lại dòng TỔNG của bảng K12 (độc lập với v_truy_nguon_01).
0 API, CPU, chỉ đọc. Ra: measure_out/_tn11/verify/pb_truy_nguon/tong.json
"""
import json, itertools
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "measure_out/_tn11/verify/pb_truy_nguon"; OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["stt2","stt4","stt11","Chr","L83","KVK","L16","TK","B18","B34"]
pages = dict(zip(ORDER,[160,145,143,65,105,163,99,161,529,112]))
cells = dict(zip(ORDER,[28589,27526,27427,8016,14474,22750,13760,22499,90747,19348]))
tr = dict(zip(ORDER,[60.4,58.1,61.0,74.2,75.4,88.9,49.8,48.2,43.0,34.6]))
sa = dict(zip(ORDER,[93.8,83.3,84.2,90.0,85.0,94.4,94.7,89.5,70.3,61.9]))
tg = dict(zip(ORDER,[33.4,25.2,23.2,15.8,9.6,5.5,44.9,41.3,27.3,27.3]))
out = {}
out["tong_trang"] = sum(pages.values()); out["tong_o"] = sum(cells.values())
out["moi_hang_sau_tru_truoc"] = {b: round(sa[b]-tr[b],1) for b in ORDER}
out["moi_hang_khop_cot_tang"] = {b: abs(round(sa[b]-tr[b],1)-tg[b])<1e-9 for b in ORDER}
def wm(d,w): 
    W=np.array([w[b] for b in ORDER],float); X=np.array([d[b] for b in ORDER],float); return float((W*X).sum()/W.sum())
one = {b:1 for b in ORDER}
sq = {b:cells[b]**0.5 for b in ORDER}; lg = {b:np.log(cells[b]) for b in ORDER}; lg1={b:np.log1p(cells[b]) for b in ORDER}
res = {}
for nm,w in (("don",one),("theo_o",cells),("theo_trang",pages),("sqrt_o",sq),("log_o",lg)):
    res[nm] = dict(truoc=round(wm(tr,w),2), sau=round(wm(sa,w),2), tang=round(wm(tg,w),2), hieu=round(wm(sa,w)-wm(tr,w),2))
out["cac_cach_tinh"] = res
# phép tính trung bình khác: trung vị, bỏ cực trị, hình học, điều hòa; và tập con
arr_tr = np.array([tr[b] for b in ORDER]); arr_sa = np.array([sa[b] for b in ORDER])
out["khac"] = dict(
  trung_vi=(float(np.median(arr_tr)), float(np.median(arr_sa))),
  hinh_hoc=(round(float(np.exp(np.log(arr_tr).mean())),2), round(float(np.exp(np.log(arr_sa).mean())),2)),
  dieu_hoa=(round(float(len(arr_tr)/np.sum(1/arr_tr)),2), round(float(len(arr_sa)/np.sum(1/arr_sa)),2)),
  tb_bo_B18_B34=(round(float(np.mean([tr[b] for b in ORDER[:8]])),2), round(float(np.mean([sa[b] for b in ORDER[:8]])),2)),
  tb_chi_4_bo_co_nhan_nguoi=(round(float(np.mean([tr[b] for b in ('L16','TK','B18','B34')])),2), round(float(np.mean([sa[b] for b in ('L16','TK','B18','B34')])),2)),
  tb_chi_6_bo_pipeline=(round(float(np.mean([tr[b] for b in ORDER[:6]])),2), round(float(np.mean([sa[b] for b in ORDER[:6]])),2)),
)
# Thử mọi tập con của các hàng (trọng số 1) xem có tập nào cho Trước = 54.3 ± 0.05 và Sau = 84.7 ± 0.05 CÙNG LÚC
hits = []
for k in range(1, 11):
    for sub in itertools.combinations(ORDER, k):
        t = np.mean([tr[b] for b in sub]); s = np.mean([sa[b] for b in sub])
        if abs(t-54.3) <= 0.05: hits.append(("truoc54.3", sub, round(float(t),2), round(float(s),2)))
both = [h for h in hits if abs(h[3]-84.7) <= 0.05]
out["tap_con_trung_binh_don_cho_truoc_54.3"] = len(hits); out["tap_con_cho_ca_truoc_54.3_va_sau_84.7"] = [dict(b=list(h[1]), truoc=h[2], sau=h[3]) for h in both][:10]
# Trọng số theo ô nhưng loại 1 bộ
for ex in ORDER:
    ss = [b for b in ORDER if b != ex]
    W = np.array([cells[b] for b in ss],float)
    t = (W*np.array([tr[b] for b in ss])).sum()/W.sum(); s = (W*np.array([sa[b] for b in ss])).sum()/W.sum()
    out.setdefault("theo_o_loai_1_bo", {})[ex] = (round(float(t),2), round(float(s),2))
# Mỗi cột 'Sau' có dạng làm tròn của phân số k/n (n = số ô p05)? kiểm khả năng k/n hợp lệ
npc = dict(zip(ORDER,[16,18,19,20,20,18,19,19,18,18]))
chk = {}
for b in ORDER:
    n = npc[b]; ok = [k for k in range(n+1) if round(100*k/n,1)==sa[b] or abs(100*k/n - sa[b])<0.051]
    chk[b] = dict(n=n, k_hop_le=ok)
out["sau_la_phan_so_k_tren_n_p05"] = chk
(OUT/"tong.json").write_text(json.dumps(out,ensure_ascii=False,indent=1),encoding="utf-8")
print(json.dumps({k:out[k] for k in ("tong_trang","tong_o","moi_hang_khop_cot_tang","cac_cach_tinh","khac","tap_con_trung_binh_don_cho_truoc_54.3","tap_con_cho_ca_truoc_54.3_va_sau_84.7","theo_o_loai_1_bo")},ensure_ascii=False,indent=1))
print(json.dumps(chk,ensure_ascii=False))
