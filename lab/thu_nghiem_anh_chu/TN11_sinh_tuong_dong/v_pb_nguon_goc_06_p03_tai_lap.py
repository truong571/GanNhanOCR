"""v_pb_nguon_goc_06 (04/10) — PHẢN BIỆN: ket_qua_p03.json có đúng là đầu ra của mã p03 trên ảnh sinh đã lưu không? (viết lại độc lập, CPU, không gọi FD)

Dựng lại từ đầu các bước p03:72-92 (chọn ô: rare n_self<2, default_rng(42), 12 ô, top-3 theo f_vW, giữ ô có chữ đúng) và p03:158-256
(ảnh phông / 'FD chung' (đường dẫn gannhanocr-fd như p03:168-175) / ảnh sinh đã lưu + dilate 3x3 x2; nhúng norm=False; cos & margin),
rồi so TỪNG Ô với ket_qua_p03.json. Thêm biến thể: (a) không dilate; (b) 'FD chung' = phông (đúng như p03 làm).
Ra: measure_out/_tn11/verify/pb_nguon_goc/p03_tai_lap_doc_lap.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

torch.set_num_threads(2)
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
J = json.loads((REPO / "measure_out/_tn11/p03_direct/ket_qua_p03.json").read_text(encoding="utf-8"))
EMB, CAND = T.OUT / "emb", T.OUT / "cand"
FONT = str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf")


def nrm(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-9)


def render(ch, size=128):
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, int(size * 0.75))
    bb = d.textbbox((0, 0), ch, font=f)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((size - w) // 2 - bb[0], (size - h) // 2 - bb[1]), ch, fill=0, font=f)
    return np.array(img)


book = "B34"
D = T.load_base(book)
E = nrm(np.load(EMB / f"{book}_enc.npy").astype(np.float32))
F = pd.read_pickle(CAND / f"{book}.pkl")
tru = F[F.y == 1].drop_duplicates("i").set_index("i")
rare = tru.index[tru.n_self.fillna(0) < 2]
sel = np.sort(np.random.default_rng(42).choice(rare, size=12, replace=False))
S = F[F.i.isin(sel)].copy()
S["rk"] = S.f_vW.fillna(-9)
S = S.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(3)
ht = S.groupby("i").y.max()
valid = sorted(ht[ht == 1].index)
S = S[S.i.isin(valid)].copy()
cells = sorted(set(S.i))
json_cells = [r["cell_id"] for r in J["details"]]
res = {"chon_o": {"n_rare": int(len(rare)), "chon_12": [int(x) for x in sel], "hop_le": [int(x) for x in cells], "json_cells": json_cells, "khop_json": [int(x) for x in cells] == sorted(json_cells)}}

from pipeline.gold_exact import signals_img as SI
from pipeline.gold_exact.common import Assets
enc = SI.Scorers(Assets(), "cpu", EMB, False, lambda m: None).enc()
chars = sorted(set(S.c))
gen_dir = REPO / "measure_out/_tn11/p03_direct/gen_book_style/b34_style"
k3 = np.ones((3, 3), np.uint8)
imgs = {"font": [], "gen": [], "gen_nodil": []}
n_gen = 0
for c in chars:
    f_im = render(c)
    imgs["font"].append(f_im)
    gp = gen_dir / f"U+{ord(c):04X}.png"
    if gp.exists():
        g = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE)
        imgs["gen"].append(cv2.dilate(g, k3, iterations=2)); imgs["gen_nodil"].append(g); n_gen += 1
    else:
        imgs["gen"].append(f_im); imgs["gen_nodil"].append(f_im)   # p03:182-183 (rơi về fd_im = phông)
V = {k: dict(zip(chars, nrm(np.asarray(enc.embed(v, norm=False), np.float32)))) for k, v in imgs.items()}
rows = []
for cid in cells:
    sub = S[S.i == cid]
    gt = sub[sub.y == 1].iloc[0].c
    out = {"cell_id": int(cid), "gt": gt}
    for k in ("font", "gen", "gen_nodil"):
        s = {c: float(E[cid] @ V[k][c]) for c in sub.c}
        top = max(s, key=s.get)
        wrong = [c for c in sub.c if c != gt]
        out[f"acc_{k}"] = int(top == gt); out[f"cos_{k}"] = round(s[gt], 4)
        out[f"margin_{k}"] = round(s[gt] - max(s[w] for w in wrong), 4)
    rows.append(out)
jd = {r["cell_id"]: r for r in J["details"]}
maxd = 0.0
ok_acc = True
for r in rows:
    j = jd.get(r["cell_id"])
    if j is None:
        ok_acc = False
        continue
    for mine, theirs in (("font", "font"), ("gen", "gen")):
        maxd = max(maxd, abs(r[f"cos_{mine}"] - j[f"cos_{theirs}"]), abs(r[f"margin_{mine}"] - j[f"margin_{theirs}"]))
        ok_acc &= (r[f"acc_{mine}"] == j[f"acc_{theirs}"])
res["so_voi_json"] = {"n_o": len(rows), "max_lech_cos_margin": round(maxd, 5), "acc_giong_het": bool(ok_acc), "n_chu_co_anh_sinh": n_gen, "n_chu": len(chars)}
res["top1"] = {k: f"{sum(r[f'acc_{k}'] for r in rows)}/{len(rows)}" for k in ("font", "gen", "gen_nodil")}
res["json_top1"] = J["metrics"]["top1_accuracy"]
res["chi_tiet"] = rows
(OUT / "p03_tai_lap_doc_lap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "chi_tiet"}, ensure_ascii=False))
