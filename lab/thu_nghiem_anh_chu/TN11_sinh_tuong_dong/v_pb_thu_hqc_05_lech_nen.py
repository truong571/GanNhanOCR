"""v_pb_thu_hqc_05_lech_nen.py — PHẢN BIỆN độc lập (04/10): tính LẠI cos(glyph nền trắng sản xuất, glyph HQC) và cos(trắng chữ A, trắng chữ B khác) từ nhúng đã lưu;
đối chiếu số trong khẳng định "lech_pha_nen" của người kiểm chứng (danhgia.json: diag_nen). Chỉ đọc."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "measure_out/_tn11/verify/thu_hqc"
def load(f):
    f = Path(f)
    if not f.exists(): return None
    z = np.load(f, allow_pickle=False); return {str(k): v for k, v in zip(z["keys"], z["E"])}
W = load(SRC / "emb/white_font.npz")
dg = json.loads((SRC / "danhgia.json").read_text(encoding="utf-8"))
out = {}
for b in ["B18", "B34", "L16", "TK"]:
    sets = {"C1": load(SRC / f"emb/primary_p0_{b}_font.npz"), "PAPERONLY": load(SRC / f"emb/paperonly_{b}_font.npz"),
            "LIT": load(SRC / f"emb/literal_{b}_font.npz"), "CTRLWHITE": load(SRC / f"emb/ctrlwhite_{b}_font.npz")}
    # tập chữ chung: chữ có đủ cả bốn nhúng (chars500)
    common = set(W)
    for v in sets.values():
        if v is not None: common &= set(v)
    common = sorted(c for c in common if not np.isnan(W[c][0]))
    r = dict(n_common=len(common))
    for nm, v in sets.items():
        if v is None: continue
        x = np.array([float(W[c] @ v[c] / (np.linalg.norm(W[c]) * np.linalg.norm(v[c]))) for c in common])
        r[nm] = dict(mean=round(float(x.mean()), 3), p05=round(float(np.quantile(x, .05)), 3), p50=round(float(np.median(x)), 3))
    # trắng vs trắng KHÁC chữ (cặp ngẫu nhiên)
    rng = np.random.default_rng(5)
    idx = rng.integers(0, len(common), size=(4000, 2)); idx = idx[idx[:, 0] != idx[:, 1]]
    xs = np.array([float(W[common[i]] @ W[common[j]]) for i, j in idx])
    r["white_vs_white_khac_chu_ngau_nhien"] = dict(mean=round(float(xs.mean()), 3), p95=round(float(np.quantile(xs, .95)), 3))
    r["ho_bao_cao"] = {k: v.get("mean") for k, v in (dg[b].get("diag_nen") or {}).items() if k.startswith("font_")}
    r["ho_white_vs_white_khac_chu_cung_o"] = (dg[b].get("diag_nen") or {}).get("font_white_vs_white_khac_chu_cung_o", {}).get("mean")
    out[b] = r
    print(b, json.dumps(r, ensure_ascii=False))
OUT = REPO / "measure_out/_tn11/verify/pb_thu_hqc"; OUT.mkdir(parents=True, exist_ok=True)
(OUT / "lech_nen.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
