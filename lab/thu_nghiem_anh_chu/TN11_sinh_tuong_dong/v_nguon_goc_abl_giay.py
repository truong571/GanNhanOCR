"""TN11 v_nguon_goc_abl_giay (03/10) — vì sao HQC2 (glyph trên 'mẫu giấy sách') không hơn HQC1 (K09)? Thử NHIỄU patch giấy giả lập so với nền phẳng. CPU, 0 API.

Trên đúng các ô duy nhất đã qua lọc của lượt 10 hạt giống (nhieu_mau_s0_9.json; mỗi ô cùng 3 ứng viên top-f_vW như p05), chấm lại với E_full (vector crop sản xuất)
bốn biến thể glyph, mọi biến thể nhúng norm=False trừ 2n:
  HQC1  = phông thô nền trắng (như p05)
  HQC2  = blend_on_paper với paper_patch của p05 (thật nếu tìm được vùng ≤200 điểm mực, nếu không thì nhiễu Gaussian i.i.d. N(bg_median, bg_std))
  HQC2f = như HQC2 nhưng patch PHẲNG = bg_median (cùng độ sáng nền, cùng mực ink_median, KHÔNG nhiễu kết cấu)
  HQC2n = HQC2 nhúng norm=True (kéo giãn phân vị như crop sản xuất)
Đo: Top-1 từng biến thể; chênh cặp HQC2f−HQC2 và HQC2f−HQC1 (McNemar chính xác, CI bootstrap theo ô).
Ra: measure_out/_tn11/verify/nguon_goc/abl_giay.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tn8lib as T  # noqa: E402
import v_nguon_goc_nhieu_mau as V  # noqa: E402

OUT = REPO / "measure_out/_tn11/verify/nguon_goc"


def main():
    torch.set_num_threads(3)
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    enc = SI.Scorers(Assets(), "cpu", T.OUT / "emb", False, lambda m: None).enc()
    runs_all = json.loads((OUT / "nhieu_mau_s0_9.json").read_text(encoding="utf-8"))
    res = {}
    for b in ("B34", "B18", "L16", "TK"):
        cells = sorted({rec["cell_id"] for r in runs_all[b] for rec in r["records"]})
        E = V.nrm(np.load(T.OUT / "emb" / f"{b}_enc.npy").astype(np.float32))
        F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")
        S = F[F.i.isin(cells)].copy()
        S["rk"] = S.f_vW.fillna(-9)
        S = S.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(3)
        paper = V.extract_book_paper_model(b)
        flat = dict(paper); flat["paper_patch"] = np.full((128, 128), paper["bg_median"], np.uint8)
        chars = sorted(set(S.c))
        raw = {c: V.render_font_raw(c, 128) for c in chars}
        sets = {"HQC1": [raw[c] for c in chars], "HQC2": [V.blend_on_paper(raw[c], paper) for c in chars], "HQC2f": [V.blend_on_paper(raw[c], flat) for c in chars]}
        emb = {k: dict(zip(chars, V.nrm(np.asarray(enc.embed(v, norm=False), np.float32)))) for k, v in sets.items()}
        emb["HQC2n"] = dict(zip(chars, V.nrm(np.asarray(enc.embed(sets["HQC2"], norm=True), np.float32))))
        acc = {k: [] for k in emb}
        for cell in cells:
            sub = S[S.i == cell]
            g = sub[sub.y == 1]
            if g.empty:
                continue
            gt = g.iloc[0].c
            for k in emb:
                sc = {c: float(E[cell] @ emb[k][c]) for c in sub.c}
                acc[k].append(int(max(sc, key=sc.get) == gt))
        A = {k: np.array(v) for k, v in acc.items()}
        n = len(A["HQC1"])
        rng = np.random.default_rng(0); idx = rng.integers(0, n, size=(10000, n))
        row = {"n_o": n, "ho_so_giay": {k: paper[k] for k in ("bg_median", "ink_median", "bg_std")},
               "top1": {k: round(float(a.mean() * 100), 1) for k, a in A.items()}}
        for a, c in (("HQC2f", "HQC2"), ("HQC2f", "HQC1"), ("HQC2", "HQC1"), ("HQC2n", "HQC2")):
            d = A[a] - A[c]; bs = d[idx].mean(1) * 100
            bb = int(((A[a] == 1) & (A[c] == 0)).sum()); cc = int(((A[a] == 0) & (A[c] == 1)).sum())
            from scipy.stats import binomtest
            row[f"{a}_tru_{c}"] = {"mean": round(float(d.mean() * 100), 1), "ci95": [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)], "a_dung_c_sai": bb, "a_sai_c_dung": cc,
                                   "mcnemar_p": round(float(binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0, 4)}
        res[b] = row
        print(f"[abl giấy] {b}: n={n} (giấy {paper['bg_median']}/{paper['ink_median']}, σ nền {paper['bg_std']}) Top-1 {row['top1']} | HQC2f−HQC2 {row['HQC2f_tru_HQC2']['mean']:+.1f} {row['HQC2f_tru_HQC2']['ci95']} p={row['HQC2f_tru_HQC2']['mcnemar_p']} "
              f"| HQC2f−HQC1 {row['HQC2f_tru_HQC1']['mean']:+.1f} {row['HQC2f_tru_HQC1']['ci95']} p={row['HQC2f_tru_HQC1']['mcnemar_p']}", flush=True)
    (OUT / "abl_giay.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {OUT / 'abl_giay.json'}")


if __name__ == "__main__":
    main()
