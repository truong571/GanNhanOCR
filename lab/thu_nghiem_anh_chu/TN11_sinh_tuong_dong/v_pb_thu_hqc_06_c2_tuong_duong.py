"""v_pb_thu_hqc_06_c2_tuong_duong.py — PHẢN BIỆN độc lập (04/10): (a) hàm nhị phân HQC3 của người kiểm chứng (v_thu_hqc_lib.canon_bin) có TƯƠNG ĐƯƠNG từng điểm ảnh
với binarize_canonical GỐC của K09 (p05_full_corpus_he_quy_chieu.py:135-163) trên glyph 112x112 và 128x128 không; (b) tính LẠI C2-C0 / C2G-C0 từ nhúng đã lưu
(c2_crops_*.npy, c2_canon_font.npz; c2g_*) bằng mã của tôi. Chỉ đọc."""
import importlib.util, json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
HERE = Path(__file__).parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu")); sys.path.insert(0, str(HERE))
import tn8lib as T
import v_pb_thu_hqc_00_tai_tinh as A
import v_thu_hqc_lib as H
spec = importlib.util.spec_from_file_location("p05fc", HERE / "p05_full_corpus_he_quy_chieu.py")
K09 = importlib.util.module_from_spec(spec); spec.loader.exec_module(K09)
from pipeline.gold_exact import signals_img as SI
gl = SI.Glyphs()
rng = np.random.default_rng(3)
chars = [chr(x) for x in rng.choice(np.arange(0x4E00, 0x9FA0), 400, replace=False)]
res = {}
same112 = same128 = n112 = n128 = 0
for ch in chars:
    g = gl.font(ch)
    if g is None: continue
    a = H.canon_bin(g); b = K09.binarize_canonical(g)
    n112 += 1; same112 += int(np.array_equal(a, b))
    g2 = K09.render_font_raw(ch, 128)
    a = H.canon_bin(g2); b = K09.binarize_canonical(g2)
    n128 += 1; same128 += int(np.array_equal(a, b))
res["glyph_112_giong_het_K09"] = f"{same112}/{n112}"; res["glyph_128_giong_het_K09"] = f"{same128}/{n128}"
print(res)
# (b) tính lại C2-C0 từ nhúng đã lưu
SRC = REPO / "measure_out/_tn11/verify/thu_hqc"
def load(f):
    z = np.load(f, allow_pickle=False); return {str(k): v for k, v in zip(z["keys"], z["E"])}
out = {"tuong_duong": res}
for pre, nm in (("c2", "C2 (nhị phân)"), ("c2g", "C2G (xám, chuẩn hoá hộp)")):
    g = load(SRC / "emb" / f"{pre}_canon_font.npz")
    for b in ["B18", "B34", "L16", "TK"]:
        ids = np.load(SRC / "emb" / f"{pre}_crops_{b}_ids.npy"); E2 = np.load(SRC / "emb" / f"{pre}_crops_{b}.npy")
        F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "f_font"]]
        F = F[F.i.isin(set(int(x) for x in ids))].reset_index(drop=True)
        row = {int(i): k for k, i in enumerate(ids)}
        cats = pd.Categorical(F.c)
        G = np.zeros((len(cats.categories), 512), np.float32); has = np.zeros(len(cats.categories), bool)
        for j, ch in enumerate(cats.categories):
            v = g.get(str(ch))
            if v is not None and not np.isnan(v[0]): G[j] = v; has[j] = True
        codes = cats.codes.astype(np.int64)
        ei = np.array([row[int(i)] for i in F.i.values])
        s = np.full(len(F), np.nan, np.float32); k = np.nonzero(has[codes])[0]
        s[k] = np.einsum("ij,ij->i", E2[ei[k]], G[codes[k]])
        miss = ~np.isnan(F.f_font.values) & np.isnan(s)
        badc = pd.Series(miss).groupby(F.i.values).any()
        o0, cells = A.per_cell(F.i.values, F.f_font.values, F.y.values.astype(np.int8), "first")
        o1, _ = A.per_cell(F.i.values, s, F.y.values.astype(np.int8), "first")
        cov = ~badc.reindex(cells).values.astype(bool)
        page = T.load_base(b).page.values[cells]
        d = ((o1 - o0) * 100)[cov]
        m, lo, hi, _ = A.cluster_ci(d, page[cov], seed=9)
        out[f"{pre}_{b}"] = dict(n=int(cov.sum()), n_mau=int(len(cells)), top1_C0=round(float(o0[cov].mean() * 100), 2), top1_new=round(float(o1[cov].mean() * 100), 2),
                                 delta=round(m, 2), ci_trang=[round(lo, 2), round(hi, 2)])
        print(pre, b, out[f"{pre}_{b}"])
(REPO / "measure_out/_tn11/verify/pb_thu_hqc/c2_tuong_duong.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
