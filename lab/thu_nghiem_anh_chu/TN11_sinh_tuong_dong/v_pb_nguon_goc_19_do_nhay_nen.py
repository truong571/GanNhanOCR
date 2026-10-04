"""v_pb_nguon_goc_19 (04/10) — PHẢN BIỆN: lý do K09 "CNN rất nhạy độ sáng nền (179 so với 255) nên cần HQC2" — tự đo độc lập (không dùng mã người kiểm chứng).

300 chữ ngẫu nhiên (seed 0) từ ứng viên của B34. Với mỗi chữ: glyph nền trắng (p05 render_font_raw) so với
  (a) glyph nhúng lên NỀN PHẲNG 179 (mực 35)  — độ sáng giống hồ sơ B34 nhưng không có kết cấu,
  (b) glyph nhúng lên patch p05 THẬT của B34 (nhiễu Gaussian giả lập, p05:112-114).
cos giữa hai vector nhúng, với norm=True (kéo giãn phân vị như crop sản xuất) và norm=False (như p05 làm cho glyph).
Ra: measure_out/_tn11/verify/pb_nguon_goc/do_nhay_nen.json   (CPU, 0 API)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
spec = importlib.util.spec_from_file_location("p05_goc", str(HERE / "p05_full_corpus_he_quy_chieu.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
from pipeline.gold_exact import signals_img as SI
from pipeline.gold_exact.common import Assets

enc = SI.Scorers(Assets(), "cpu", M.EMB_DIR, False, lambda m: None).enc()
from fontTools.ttLib import TTFont
cmap = set(TTFont(M.FONT_PATH).getBestCmap())
F = pd.read_pickle(M.CAND_DIR / "B34.pkl")[["c"]]
all_chars = sorted(set(F.c))
chars = [c for c in all_chars if c and ord(c[0]) in cmap]      # CHỈ chữ có trong phông NomNaTong (chữ thiếu đều render giống nhau)
rng = np.random.default_rng(0)
pick = [chars[i] for i in rng.choice(len(chars), size=300, replace=False)]
paper = M.extract_book_paper_model("B34")
flat = dict(paper)
flat["paper_patch"] = np.full((128, 128), int(round(paper["bg_median"])), np.uint8)
raw = [M.render_font_raw(c, 128) for c in pick]
res = {"profil_B34": {k: paper[k] for k in ("bg_median", "ink_median", "bg_std")}, "n_chu": len(pick), "n_chu_ung_vien_B34": len(all_chars), "n_chu_co_trong_phong": len(chars)}
for nm in (True, False):
    def emb(imgs):
        return M.nrm(np.concatenate([np.asarray(enc.embed(imgs[i:i + 100], norm=nm), np.float32) for i in range(0, len(imgs), 100)]))
    Vw = emb(raw)
    Vf = emb([M.blend_on_paper(r, flat) for r in raw])
    Vn = emb([M.blend_on_paper(r, paper) for r in raw])
    cf = np.sum(Vw * Vf, 1); cn = np.sum(Vw * Vn, 1)
    # tham chiếu: cos giữa hai chữ khác nhau (nền trắng) và truy hồi Top-1 của chính nó
    S = Vw @ Vw.T
    other = S[~np.eye(len(S), dtype=bool)].mean()
    res[f"norm={nm}"] = {"cos_trang_vs_nen_phang_179_tb": round(float(cf.mean()), 4), "min": round(float(cf.min()), 4),
                         "cos_trang_vs_patch_p05_nhieu_tb": round(float(cn.mean()), 4), "min_nhieu": round(float(cn.min()), 4),
                         "cos_hai_chu_khac_nhau_tb": round(float(other), 4),
                         "truy_hoi_top1_nen_phang": round(float(np.mean(np.argmax(Vf @ Vw.T, 1) == np.arange(len(pick)))) * 100, 1),
                         "truy_hoi_top1_nhieu_p05": round(float(np.mean(np.argmax(Vn @ Vw.T, 1) == np.arange(len(pick)))) * 100, 1)}
print(json.dumps(res, ensure_ascii=False, indent=1))
(OUT / "do_nhay_nen.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
