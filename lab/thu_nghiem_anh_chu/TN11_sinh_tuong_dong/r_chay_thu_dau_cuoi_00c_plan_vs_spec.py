"""Rà soát GÓI: bản VÁ make_plan(font_of)/cmd_pack(font_of_file) có đúng ý? So plan.json trong gói với spec nguồn (kaggle_spec/chars_<cuốn>.txt, font_of_<cuốn>.tsv, spec.json):
  - chars của kế hoạch == chars_<cuốn>.txt trừ chữ không phông (giữ thứ tự ưu tiên);
  - phông của từng chữ trong kế hoạch == phông ghi ở font_of_<cuốn>.tsv; chữ có tên phông rỗng == uncovered;
  - ảnh phong cách trong gói == styles/<cuốn>_s<j>.png của spec (từng byte); số phong cách == len(style_pngs);
  - dựng lại make_plan() từ spec (CPU) cho ra ĐÚNG sha của gói (tái lập kế hoạch).
Ra: .../review/chay_thu_dau_cuoi/00c_plan_vs_spec.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import json
from pathlib import Path

import r_chay_thu_dau_cuoi_lib as L

K = L.load_runner()
rep = L.Rep("00c_plan_vs_spec")
plan = K.load_plan(L.PACK)
SPEC = L.FULL / "kaggle_spec"
spec = json.loads((SPEC / "spec.json").read_text(encoding="utf-8"))
names = plan["font_names"]
bad_chars, bad_font, bad_unc, bad_style, order_bad = [], [], [], [], []
for b, v in plan["books"].items():
    chars = [c for c in (SPEC / f"chars_{b}.txt").read_text(encoding="utf-8").split("\n") if c]
    fo = {}
    for ln in (SPEC / f"font_of_{b}.tsv").read_text(encoding="utf-8").split("\n"):
        if "\t" in ln:
            c, n = ln.split("\t")
            fo[c] = n
    exp_cov = [c for c in dict.fromkeys(chars) if fo.get(c)]
    exp_unc = [c for c in dict.fromkeys(chars) if not fo.get(c)]
    if list(v["chars"]) != exp_cov:
        bad_chars.append(b)
    if v["uncovered"] != "".join(exp_unc):
        bad_unc.append(b)
    mism = [(c, names[i], fo[c]) for c, i in zip(v["chars"], v["font_idx"]) if names[i] != fo[c]]
    if mism:
        bad_font.append((b, mism[:2]))
    for j in range(v["styles"]):
        if L.sha256_file(L.PACK / "data" / b / f"style_{j}.png") != L.sha256_file(SPEC / "styles" / f"{b}_s{j}.png"):
            bad_style.append((b, j))
    if v["styles"] != len(spec["books"][b]["style_pngs"]):
        bad_style.append((b, "so_luong"))
rep.inv("Q1.chars_ke_hoach_dung_chars_spec_tru_chu_khong_phong_giu_thu_tu", not bad_chars and not bad_unc, dict(sai_chars=bad_chars, sai_uncovered=bad_unc))
rep.inv("Q2.phong_tung_chu_dung_font_of_tsv", not bad_font, dict(sai=bad_font[:2]))
rep.inv("Q3.anh_phong_cach_trong_goi_giong_spec_tung_byte", not bad_style, dict(sai=bad_style))
# dựng lại kế hoạch từ spec bằng make_plan thật
from fontTools.ttLib import TTFont  # noqa: E402

fonts = {}
for n, p in spec["fonts"].items():
    f = TTFont(str(L.REPO / p), lazy=True); s = set()
    for st in f["cmap"].tables:
        s |= set(st.cmap)
    fonts[n] = s
# cmd_pack: thứ tự phông = thứ tự khoá spec["fonts"]
books = {}
for b, v in spec["books"].items():
    books[b] = dict(chars=[c for c in Path(v["chars_file"]).read_text(encoding="utf-8").split("\n") if c], styles=len(v["style_pngs"]))
    books[b]["font_of"] = dict(ln.split("\t") for ln in Path(v["font_of_file"]).read_text(encoding="utf-8").split("\n") if "\t" in ln)
rebuilt = K.make_plan(books, [("base", 0)], 128, fonts)
rep.inv("Q4.dung_lai_make_plan_tu_spec_cho_dung_sha_cua_goi", rebuilt["sha"] == plan["sha"], dict(dung_lai=rebuilt["sha"][:12], goi=plan["sha"][:12], thu_tu_phong_spec=list(spec["fonts"])))
rep.save()
