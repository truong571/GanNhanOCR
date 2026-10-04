"""Rà soát GÓI (tĩnh, không GPU): bản chạy trong gói == bản ở lab/; kế hoạch toàn vẹn; mỗi (cuốn, chữ) có glyph trong phông được gán (fontTools cmap);
ckpt đúng băm; kiểm gói (không tệp nhãn); ZIP == thư mục từng byte; ảnh phong cách hợp lệ.
Ra: measure_out/_tn11/full/review/chay_thu_dau_cuoi/00_kiem_goi.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import hashlib
import json
import zipfile
from collections import Counter

import r_chay_thu_dau_cuoi_lib as L

K = L.load_runner()
rep = L.Rep("00_kiem_goi")
PACK, HERE = L.PACK, L.HERE

# P1 bản chạy trong gói == bản ở lab/
same_runner = L.sha256_file(PACK / "w_runner_khung_chay.py") == L.sha256_file(HERE / "w_runner_khung_chay.py")
same_fix = L.sha256_file(PACK / "code" / "fd_wrapper_fix.py") == L.sha256_file(HERE / "fd_wrapper_fix.py")
rep.inv("P1.ban_chay_trong_goi_giong_lab", same_runner and same_fix, dict(runner=same_runner, wrapper_fix=same_fix))

# P2 kế hoạch
plan = K.load_plan(PACK)                         # ném SystemExit nếu sha sai
man = json.loads((PACK / "manifest.json").read_text(encoding="utf-8"))
bc = json.loads((L.FULL / "kaggle_pack_20261004_bao_cao.json").read_text(encoding="utf-8"))
n_img = sum(t["hi"] - t["lo"] for t in plan["tasks"])
rep.inv("P2.sha_ke_hoach_khop_plan_manifest_baocao", plan["sha"] == man["plan_sha"] == bc["plan_sha"], dict(sha=plan["sha"][:12]))
rep.inv("P2.410_viec_51730_anh", len(plan["tasks"]) == 410 and n_img == 51730 == bc["n_anh"] == sum(len(v["chars"]) for v in plan["books"].values()),
        dict(viec=len(plan["tasks"]), anh=n_img))
ids = [t["id"] for t in plan["tasks"]]
bad_range = [t["id"] for t in plan["tasks"] if not (0 <= t["lo"] < t["hi"] <= len(plan["books"][t["book"]]["chars"]) and t["hi"] - t["lo"] <= plan["shard"])]
cover = {b: sorted((t["lo"], t["hi"]) for t in plan["tasks"] if t["book"] == b) for b in plan["books"]}
exact = all(c[0][0] == 0 and c[-1][1] == len(plan["books"][b]["chars"]) and all(c[i][1] == c[i + 1][0] for i in range(len(c) - 1)) for b, c in cover.items())
rep.inv("P2.id_duy_nhat_lo_khong_chong_phu_kin_chu", len(set(ids)) == len(ids) and not bad_range and exact, dict(bad_range=bad_range[:3], phu_kin=exact))
missing_style = [(t["book"], t["style"]) for t in plan["tasks"] if not (PACK / "data" / t["book"] / f"style_{t['style']}.png").exists()]
rep.inv("P2.moi_viec_co_anh_phong_cach", not missing_style, dict(thieu=missing_style[:3], passes=plan["passes"]))
len_ok = all(len(v["chars"]) == len(v["font_idx"]) and all(0 <= i < len(plan["font_names"]) for i in v["font_idx"]) for v in plan["books"].values())
dups = {b: len(v["chars"]) - len(set(v["chars"])) for b, v in plan["books"].items() if len(v["chars"]) != len(set(v["chars"]))}
rep.inv("P2.font_idx_cung_do_dai_chars_khong_trung_chu", len_ok and not dups, dict(trung=dups))
rep.data("font_names_thu_tu", plan["font_names"])
rep.data("passes", plan["passes"])

# P3 mỗi (cuốn, chữ) có glyph trong phông được gán (cmap thật của tệp phông trong gói)
from fontTools.ttLib import TTFont  # noqa: E402

fmap = json.loads((PACK / "fonts.json").read_text(encoding="utf-8"))
cm = {}
for n, fn in fmap.items():
    f = TTFont(str(PACK / "code" / "font_diffusion" / "fonts" / fn), lazy=True)
    s = set()
    for st in f["cmap"].tables:
        s |= set(st.cmap)
    cm[n] = s
rep.inv("P3.moi_phong_trong_fonts_json_co_tep", set(fmap) == set(plan["font_names"]) and all((PACK / "code" / "font_diffusion" / "fonts" / fn).exists() for fn in fmap.values()),
        dict(fonts=fmap))
miss, nchar = [], 0
for b, v in plan["books"].items():
    for c, i in zip(v["chars"], v["font_idx"]):
        nchar += 1
        if ord(c) not in cm[plan["font_names"][i]]:
            miss.append((b, f"U+{ord(c):04X}", plan["font_names"][i]))
rep.inv("P3.moi_chu_co_glyph_trong_phong_duoc_gan", not miss and nchar == 51730, dict(kiem=nchar, thieu=miss[:5]))
unc = {b: v["uncovered"] for b, v in plan["books"].items() if v["uncovered"]}
rep.data("uncovered", {b: [f"U+{ord(c):04X}" for c in s] for b, s in unc.items()})
cnt_font = {b: dict(Counter(plan["font_names"][i] for i in v["font_idx"])) for b, v in plan["books"].items()}
rep.data("so_chu_theo_phong_moi_cuon", cnt_font)
allc = [c for v in plan["books"].values() for c in v["chars"]]
rep.data("chu_khac_nhau_toan_goi", len(set(allc)))
rep.inv("P3.moi_chu_la_mot_diem_ma", all(len(c) == 1 for c in allc), dict(n_astral=sum(ord(c) > 0xFFFF for c in allc), n_PUA=sum(0xE000 <= ord(c) <= 0xF8FF or ord(c) >= 0xF0000 for c in allc)))

# P4 ckpt
bad_ck = K.verify_ckpt(man, PACK / "ckpt")
rep.inv("P4.ckpt_dung_bam_manifest", not bad_ck and len(man["ckpt_sha256"]) == 3, dict(loi=bad_ck))

# P5 kiểm gói + không tệp nhãn
au = K.audit_pack(PACK)
files = [p for p in PACK.rglob("*") if p.is_file()]
junk = [p.relative_to(PACK).as_posix() for p in files if p.name in (".DS_Store",) or "__pycache__" in p.parts or p.suffix in (".log", ".pyc", ".csv", ".tsv", ".pkl", ".npy", ".parquet")]
rep.inv("P5.audit_pack_sach_va_it_muc_cap_mot", au["sach"] and au["top_level_ok"] and not junk, dict(audit=au, rac=junk[:5]))
name_hits = [p.relative_to(PACK).as_posix() for p in files if any(x in p.name.lower() for x in ("label", "nhan", "human", "truth", "gt_", "gold"))]
rep.inv("P5.ten_tep_khong_goi_y_nhan", not name_hits, dict(hits=name_hits[:5]))
js = sorted(p.relative_to(PACK).as_posix() for p in files if p.suffix == ".json")
rep.inv("P5.chi_3_json", js == ["fonts.json", "manifest.json", "plan.json"], dict(json=js))
# plan.json chỉ có khoá cấu trúc (không cột nhãn người/vị trí ô/đường dẫn ảnh gốc)
pk = set(plan) | {k for v in plan["books"].values() for k in v} | {k for t in plan["tasks"] for k in t}
rep.inv("P5.khoa_plan_khong_chua_nhan_hay_duong_dan", pk <= {"tool", "shard", "passes", "font_names", "books", "tasks", "sha", "chars", "font_idx", "uncovered", "styles", "id", "book", "model", "style", "lo", "hi"}, dict(khoa=sorted(pk)))

# P6 ZIP == thư mục
zp = L.FULL / "kaggle_pack_20261004.zip"
with zipfile.ZipFile(zp) as z:
    zn = {i.filename: i for i in z.infolist() if not i.is_dir()}
    bad_crc = z.testzip()
    dir_files = {p.relative_to(PACK).as_posix(): p for p in files}
    diff = []
    for n, i in zn.items():
        if n not in dir_files:
            diff.append(("chi_o_zip", n)); continue
        if hashlib.sha256(z.read(n)).hexdigest() != L.sha256_file(dir_files[n]):
            diff.append(("khac_noi_dung", n))
    diff += [("chi_o_thu_muc", n) for n in dir_files if n not in zn]
    top_zip = {n.split("/")[0] for n in zn}
rep.inv("P6.zip_giong_thu_muc_tung_byte", not diff and bad_crc is None and len(zn) == len(dir_files) == 112, dict(zip_files=len(zn), dir_files=len(dir_files), khac=diff[:3], testzip=bad_crc))
rep.inv("P6.plan_json_o_goc_zip_va_it_muc_cap_mot", "plan.json" in zn and len(top_zip) <= 50 and "w_runner_khung_chay.py" in zn and sum(n.endswith("w_runner_khung_chay.py") for n in zn) == 1,
        dict(top=sorted(top_zip), n_runner=sum(n.endswith("w_runner_khung_chay.py") for n in zn)))

# P7 ảnh phong cách
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sinfo = {}
okst = True
for b, v in plan["books"].items():
    for j in range(v["styles"]):
        im = Image.open(PACK / "data" / b / f"style_{j}.png")
        a = np.array(im.convert("L"))
        ink = float((a < 128).mean() * 100)
        binary = bool(np.isin(a, (0, 255)).mean() > 0.98)
        sinfo[f"{b}/s{j}"] = dict(size=im.size, mode=im.mode, ink_pct=round(ink, 1), binary=binary)
        okst &= im.size == (128, 128) and 2 < ink < 40
rep.inv("P7.20_anh_phong_cach_128x128_muc_hop_ly", okst and len(sinfo) == 20, dict(mau=dict(list(sinfo.items())[:3])))
rep.data("anh_phong_cach", sinfo)

# P8 mọi .py trong code/ biên dịch được (cú pháp)
err_py = []
for p in (PACK / "code").rglob("*.py"):
    try:
        compile(p.read_text(encoding="utf-8"), str(p), "exec")
    except Exception as e:  # noqa: BLE001
        err_py.append((p.name, type(e).__name__))
rep.inv("P8.moi_py_trong_code_bien_dich_duoc", not err_py, dict(loi=err_py[:3]))
rep.save()
