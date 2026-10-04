"""r_huong_dan_01_so_lieu.py — REVIEW hướng "huong_dan" (1/5): đối chiếu MỌI con số đếm được trong
docs/KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md với gói thật (plan.json, manifest.json, báo cáo, zip, thư mục gói, bộ chạy).
0 GPU, 0 API, 0 mạng; chỉ ĐỌC repo. Mỗi phép đo có invariant PASS/FAIL (quy ước CLAUDE.md), không dùng LLM nhìn dữ liệu.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_huong_dan_01_so_lieu.py
Ra: measure_out/_tn11/full/review/huong_dan/01_so_lieu.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections
import hashlib
import importlib.util
import json
import os
import re
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FULL = REPO / "measure_out" / "_tn11" / "full"
OUT = FULL / "review" / "huong_dan"
GUIDE = REPO / "docs" / "KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md"
PACK = FULL / "kaggle_pack_20261004"
ZIP = FULL / "kaggle_pack_20261004.zip"
REP = FULL / "kaggle_pack_20261004_bao_cao.json"
LAB_RUNNER = HERE / "w_runner_khung_chay.py"
SEC_PER_IMG = 2.7                                  # số trong hướng dẫn (mục 0)

spec = importlib.util.spec_from_file_location("K", LAB_RUNNER)
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)                          # chỉ để dùng plan_sha/canon/rel_path của CHÍNH bộ chạy

R = {"tao": time.strftime("%Y-%m-%d %H:%M:%S"), "invariants": {}, "so_do": {}}


def inv(name: str, ok: bool, detail=None):
    R["invariants"][name] = dict(ok=bool(ok), detail=detail)
    print(("PASS" if ok else "FAIL"), name, "" if detail is None else json.dumps(detail, ensure_ascii=False, default=str)[:260], flush=True)


def vn_float(s: str) -> float:
    return float(s.replace(".", "").replace(",", ".")) if "," in s and "." in s else float(s.replace(",", "."))


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    g = GUIDE.read_text(encoding="utf-8")
    plan = json.loads((PACK / "plan.json").read_text(encoding="utf-8"))
    man = json.loads((PACK / "manifest.json").read_text(encoding="utf-8"))
    rep = json.loads(REP.read_text(encoding="utf-8"))
    books = plan["books"]

    # ------------------------------------------------------------------ A. mã băm, bộ chạy, kế hoạch
    sha_re = K.plan_sha(plan)
    inv("A1.sha_ke_hoach_tinh_lai_khop_plan_manifest_baocao", sha_re == plan["sha"] == man["plan_sha"] == rep["plan_sha"], dict(tinh_lai=sha_re[:12], plan=plan["sha"][:12], manifest=man["plan_sha"][:12], bao_cao=rep["plan_sha"][:12]))
    inv("A2.sha_bat_dau_3f0dcc5348ae", plan["sha"].startswith("3f0dcc5348ae"), plan["sha"][:12])
    inv("A3.bo_chay_trong_goi_giong_bo_chay_lab", sha_file(PACK / "w_runner_khung_chay.py") == sha_file(LAB_RUNNER), sha_file(LAB_RUNNER)[:12])
    n_tasks = len(plan["tasks"])
    n_img = sum(t["hi"] - t["lo"] for t in plan["tasks"])
    n_chars_cov = sum(len(v["chars"]) for v in books.values())
    inv("A4.410_viec", n_tasks == 410 == rep["n_viec"], n_tasks)
    inv("A5.51730_anh", n_img == 51730 == n_chars_cov == rep["n_anh"], dict(tong_hi_lo=n_img, chu_phu=n_chars_cov))
    inv("A6.mot_phong_cach_mot_luot", plan["passes"] == [["base", 0]] and all(v["styles"] == 2 for v in books.values()),
        dict(passes=plan["passes"], styles_trong_goi={b: v["styles"] for b, v in books.items()}))
    inv("A7.lo_128", plan["shard"] == 128 and max(t["hi"] - t["lo"] for t in plan["tasks"]) == 128, dict(shard=plan["shard"]))
    sizes = collections.Counter(t["hi"] - t["lo"] for t in plan["tasks"])
    R["so_do"]["viec_theo_co"] = dict(sorted(sizes.items(), key=lambda kv: -kv[0]))
    per_book = {b: dict(chu=len(v["chars"]), khong_phong=len(v["uncovered"]), viec=sum(1 for t in plan["tasks"] if t["book"] == b)) for b, v in books.items()}
    R["so_do"]["theo_cuon"] = per_book
    inv("A8.10_cuon_dung_ten", list(books) == ["stt2", "stt4", "stt11", "Chr", "L83", "KVK", "L16", "TK", "B18", "B34"], list(books))
    unc = {b: v["uncovered"] for b, v in books.items() if v["uncovered"]}
    inv("A9.dung_1_chu_khong_phong_o_stt11", list(unc) == ["stt11"] and len(unc["stt11"]) == 1 and rep["khong_phong_tong"] == 1, {b: [hex(ord(c)) for c in s] for b, s in unc.items()})
    # xen kẽ giữa các cuốn: 20 % việc đầu phải có đủ 10 cuốn
    first = {t["book"] for t in plan["tasks"][: n_tasks // 5]}
    inv("A10.xen_ke_10_cuon_trong_20pct_dau", len(first) == 10, len(first))

    # ------------------------------------------------------------------ B. giờ GPU / giờ phiên / ca
    gpu_h = n_img * SEC_PER_IMG / 3600
    inv("B1.38_8_gio_GPU", abs(gpu_h - 38.8) < 0.05 and rep["gio_gpu_uoc"] == 38.8, round(gpu_h, 3))
    inv("B2.19_4_gio_phien_2GPU", abs(gpu_h / 2 - 19.4) < 0.05 and rep["gio_phien_2gpu_uoc"] == 19.4, round(gpu_h / 2, 3))
    R["so_do"]["gio_gpu"] = round(gpu_h, 3)
    R["so_do"]["gio_phien_2gpu"] = round(gpu_h / 2, 3)
    R["so_do"]["gio_chia_3_ca"] = round(gpu_h / 2 / 3, 3)
    R["so_do"]["est_s_mac_dinh_bo_chay"] = round(2.66 * 128, 2)
    R["so_do"]["giay_moi_viec_128_theo_2_7"] = round(128 * SEC_PER_IMG, 1)

    # ------------------------------------------------------------------ C. kích thước, zip, thư mục gói
    zsize = ZIP.stat().st_size
    inv("C1.zip_472_MB", round(zsize / 1e6) == 472 and rep["zip"]["MB"] == 472.1, dict(bytes=zsize, MB=round(zsize / 1e6, 1), MiB=round(zsize / 2**20, 1)))
    dir_files = [p for p in PACK.rglob("*") if p.is_file()]
    dir_bytes = sum(p.stat().st_size for p in dir_files)
    R["so_do"]["thu_muc_goi"] = dict(tep=len(dir_files), bytes=dir_bytes, MB=round(dir_bytes / 1e6, 1))
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        bad = z.testzip()
        top = collections.Counter(n.split("/")[0] for n in names)
        rels = {p.relative_to(PACK).as_posix(): p for p in dir_files}
        inv("C2.zip_nguyen_ven_CRC", bad is None, bad)
        inv("C3.zip_cung_tap_tep_voi_thu_muc", set(names) == set(rels), dict(zip=len(names), thu_muc=len(rels)))
        inv("C4.plan_json_nam_o_goc_zip_va_7_muc_cap_mot", "plan.json" in names and len(top) == 7 and len(top) <= 50, dict(top))
        diff = []
        for info in z.infolist():
            if info.filename in rels:
                if info.file_size != rels[info.filename].stat().st_size:
                    diff.append(info.filename)
        inv("C5.kich_thuoc_tung_tep_zip_khop_thu_muc", not diff, diff[:3])
        sha_diff = []
        for info in z.infolist():
            if info.filename in rels and info.file_size < 5_000_000:     # tệp nhỏ: so băm đầy đủ (ckpt/phông lớn đã so kích thước + CRC ở trên)
                h = hashlib.sha256(z.read(info.filename)).hexdigest()
                if h != sha_file(rels[info.filename]):
                    sha_diff.append(info.filename)
        inv("C6.bam_tep_nho_trong_zip_khop_thu_muc", not sha_diff, sha_diff[:3])
        r_unz = sum(i.file_size for i in z.infolist())
    R["so_do"]["zip"] = dict(bytes=zsize, giai_nen_bytes=r_unz, giai_nen_MB=round(r_unz / 1e6, 1))
    ck = {}
    for fn, h in man["ckpt_sha256"].items():
        ck[fn] = sha_file(PACK / "ckpt" / fn) == h
    inv("C7.bam_3_ckpt_trong_goi_khop_manifest", all(ck.values()) and len(ck) == 3, ck)
    inv("C8.ckpt_manifest_khop_baocao", man["ckpt_sha256"] == rep["ckpt_sha256"], None)

    # ------------------------------------------------------------------ D. chữ khác nhau, phông
    allc = set()
    for v in books.values():
        allc |= set(v["chars"])
    inv("D1.10_2_nghin_chu_khac_nhau", 10150 <= len(allc) <= 10250, len(allc))
    R["so_do"]["chu_khac_nhau"] = len(allc)
    fn = plan["font_names"]
    use = collections.Counter()
    pua_fonts = collections.Counter()
    n_pua = 0
    for v in books.values():
        for c, i in zip(v["chars"], v["font_idx"]):
            use[fn[i]] += 1
            o = ord(c)
            if 0xE000 <= o <= 0xF8FF or 0xF0000 <= o <= 0xFFFFD or 0x100000 <= o <= 0x10FFFD:
                n_pua += 1
                pua_fonts[fn[i]] += 1
    R["so_do"]["anh_theo_phong"] = dict(use)
    R["so_do"]["anh_phong_du_phong_pct"] = round(100 * (n_img - use["NomNaTong"]) / n_img, 2)
    inv("D2.chu_PUA_chi_ve_bang_ho_NomNaTong", set(pua_fonts) <= {"NomNaTong", "NomNaTongLight2"} and n_pua > 0, dict(n_pua=n_pua, phong=dict(pua_fonts)))
    inv("D3.bao_cao_phong_theo_cuon_khop_ke_hoach", all(
        sum(rep["sach"][b]["phong"].values()) == len(books[b]["chars"]) for b in books), None)
    # phông nào trong kế hoạch có thật trong fonts.json + trong gói
    fj = json.loads((PACK / "fonts.json").read_text(encoding="utf-8"))
    inv("D4.moi_phong_ke_hoach_co_tep_trong_goi", all((PACK / "code" / "font_diffusion" / "fonts" / fj[n]).exists() for n in fn) and set(fj) == set(fn), fj)

    # D5/D6: bản vá font_of (make_plan/cmd_pack, mới trong phiên này, KHÔNG có phép thử trong w_runner_kiem_khung.py) cho ra đúng phông đã gán?
    kt = (HERE / "w_runner_kiem_khung.py").read_text(encoding="utf-8")
    inv("D5a.w_runner_kiem_khung.py_khong_co_phep_thu_nao_cho_font_of_(ban_va_cua_phien_nay)", "font_of" not in kt, dict(so_lan_font_of=kt.count("font_of"), mtime_test=time.strftime("%H:%M:%S", time.localtime((HERE / "w_runner_kiem_khung.py").stat().st_mtime))))
    spec_dir = FULL / "kaggle_spec"
    mism = {}
    for b, v in books.items():
        tsv = {}
        for ln in (spec_dir / f"font_of_{b}.tsv").read_text(encoding="utf-8").split("\n"):
            if "\t" in ln:
                c, nm = ln.split("\t")
                tsv[c] = nm
        pm = {c: fn[i] for c, i in zip(v["chars"], v["font_idx"])}
        mism[b] = dict(sai_phong=sum(1 for c, nm in pm.items() if tsv.get(c) != nm), khong_phong_plan=len(v["uncovered"]), khong_phong_tsv=sum(1 for nm in tsv.values() if nm == ""),
                       chu_trong_chars_txt=len([c for c in (spec_dir / f"chars_{b}.txt").read_text(encoding="utf-8").split("\n") if c]), chu_ke_hoach=len(v["chars"]) + len(v["uncovered"]))
    inv("D5.phong_trong_plan_json_khop_font_of_<cuon>.tsv_cho_ca_51730_chu_va_chu_khong_phong_khop", all(m["sai_phong"] == 0 and m["khong_phong_plan"] == m["khong_phong_tsv"] and m["chu_trong_chars_txt"] == m["chu_ke_hoach"] for m in mism.values()), mism)
    from fontTools.ttLib import TTFont
    cms = {}
    for nm in fn:
        f = TTFont(str(PACK / "code" / "font_diffusion" / "fonts" / fj[nm]), lazy=True)
        cm = set()
        for tb in f["cmap"].tables:
            cm |= set(tb.cmap)
        cms[nm] = cm
    bad_cmap = sum(1 for v in books.values() for c, i in zip(v["chars"], v["font_idx"]) if ord(c) not in cms[fn[i]])
    inv("D6.moi_chu_nam_trong_cmap_cua_phong_duoc_gan_(51730_chu)", bad_cmap == 0, dict(chu_ngoai_cmap=bad_cmap))

    # ------------------------------------------------------------------ E. từng câu số trong hướng dẫn
    claims = []

    def claim(key, rx, fn_ok, measured):
        m = re.search(rx, g)
        if not m:
            claims.append(dict(khoan=key, tim_thay_trong_huong_dan=False, do=measured))
            inv("E." + key, False, "không tìm thấy câu trong hướng dẫn")
            return
        line = g[: m.start()].count("\n") + 1
        ok = bool(fn_ok(m.groups()))
        claims.append(dict(khoan=key, dong=line, trich=m.group(0)[:90], huong_dan=list(m.groups()), do=measured, ok=ok))
        inv("E." + key, ok, dict(dong=line, huong_dan=list(m.groups()), do=measured))

    claim("viec_lo", r"(\d+) việc \(lô (\d+) chữ\)", lambda x: [int(i) for i in x] == [410, 128], [n_tasks, 128])
    claim("anh", r"\*\*([\d\.]+) ảnh\*\*", lambda x: int(x[0].replace(".", "")) == n_img, n_img)
    claim("gio", r"≈ \*\*([\d,]+) giờ GPU = ([\d,]+) giờ phiên\*\*", lambda x: (vn_float(x[0]), vn_float(x[1])) == (38.8, 19.4), [round(gpu_h, 1), round(gpu_h / 2, 1)])
    claim("giay_anh", r"\(([\d,]+) giây/ảnh", lambda x: vn_float(x[0]) == SEC_PER_IMG, SEC_PER_IMG)
    claim("zip_MB", r"\((\d+) MB; thư mục", lambda x: int(x[0]) == round(zsize / 1e6), round(zsize / 1e6))
    claim("anh_chu_khac_nhau", r"\(([\d,]+) nghìn ảnh cho ([\d,]+) nghìn chữ khác nhau\)", lambda x: (round(n_img / 1000, 1), round(len(allc) / 1000, 1)) == (vn_float(x[0]), vn_float(x[1])), [round(n_img / 1000, 1), round(len(allc) / 1000, 1)])
    claim("mot_chu_khong_phong", r"hiện (\d+) chữ ở (stt11)", lambda x: (int(x[0]), x[1]) == (1, "stt11"), unc and {b: len(s) for b, s in unc.items()})
    claim("so_kiem_khung", r"`w_runner_kiem_khung\.py` (\d+)/(\d+)", lambda x: x[0] == x[1] == "34", None)
    claim("ca_6_5", r"≈ \*\*(\d+) ca ~([\d,]+) giờ\*\*", lambda x: x[0] == "3" and abs(gpu_h / 2 / 3 - vn_float(x[1])) < 0.1, round(gpu_h / 2 / 3, 2))
    claim("R_am_day_du", r"\((\d+) nghìn ảnh ≈ (\d+) giờ, ~(\d+) tuần quota\)", lambda x: True, "xem so_do.R_am_day_du")
    # 240 nghìn ảnh ≈ 89 giờ: lấy từ ngan_sach.json (kịch bản F) và ket_qua.json (S2) — hai nguồn, hai hằng số giây/ảnh
    ns = json.loads((FULL / "runner" / "ngan_sach.json").read_text(encoding="utf-8"))
    F = next(x for x in ns["F_ngan_sach"] if x["kich_ban"].startswith("F."))
    kq = json.loads((FULL / "quy_mo_va_ngan_sach" / "ket_qua.json").read_text(encoding="utf-8"))
    S2 = kq["cascade_so_voi_NomNaTong_don"]["S2_hop_S0"]
    phu = {b: v["_phu_ung_vien_cascade"]["chien_luoc"]["S1_k5"] for b, v in kq["duong_cong_phu"].items()}
    R["so_do"]["phu_R_am_moi_o_S1_k5"] = dict(min=min(phu.values()), max=max(phu.values()), tb=round(sum(phu.values()) / len(phu), 4), theo_cuon=phu)
    claim("phu_41_45", r"khoảng (\d+)–(\d+) % R\(âm\) mỗi ô", lambda x: (round(min(phu.values()) * 100), round(max(phu.values()) * 100)) == (int(x[0]), int(x[1])), [round(min(phu.values()) * 100, 1), round(max(phu.values()) * 100, 1)])
    R["so_do"]["R_am_day_du"] = dict(ngan_sach_json=dict(anh=F["anh_can_sinh"], gio_GPU=F["gio_GPU"], gio_dong_ho=F["gio_dong_ho_T4x2_dong"], tuan=F["so_tuan_han_muc"]),
                                      ket_qua_json_S2=dict(anh=S2["anh_cascade"], gio_GPU=S2["gio_gpu_cascade"], gio_phien_theo_2_7=round(S2["anh_cascade"] * 2.7 / 3600 / 2, 1)))
    inv("E.R_am_day_du_89h_tu_ngan_sach_json_va_90h_neu_2_7", F["gio_dong_ho_T4x2_dong"] == 89.4 and abs(S2["anh_cascade"] * 2.7 / 3600 / 2 - 90.5) < 0.1, R["so_do"]["R_am_day_du"])
    R["yeu_cau"] = claims

    # ------------------------------------------------------------------ F. nguồn tốc độ 2,7 giây/ảnh
    A = ns["A_log_chay_that"]
    fst = {b: dict(s_per_img=A[b]["s_per_img"], lo_32="lô 32" in ns["luu_y"], tc_wrapper_goc=A[b]["fst_evidence"]["loading_line_original_wrapper"], tc_wrapper_sua=A[b]["fst_evidence"]["loading_line_fixed_wrapper"],
                    dong_model_with_fst=A[b]["fst_evidence"]["model_class_with_fst_lines"]) for b in ("B34", "B18")}
    R["so_do"]["nguon_2_7_s_anh"] = fst
    inv("F1.2_7_s_anh_do_tren_Kaggle_nhung_voi_wrapper_GOC_use_fst_True", all(v["tc_wrapper_goc"] == 1 and v["tc_wrapper_sua"] == 0 and v["dong_model_with_fst"] >= 1 for v in fst.values()) and all(2.6 <= v["s_per_img"] <= 2.7 for v in fst.values()), fst)
    doc = (REPO / "docs" / "SINH_ANH_SO_SANH_2026-10-03.md").read_text(encoding="utf-8")
    m = re.search(r"p02 chuẩn \| Như p02 nhưng `use_fst=False`[^|]*?(\d+) ảnh, ([\d,]+) giây/ảnh \(([^)]*)\)", doc)
    R["so_do"]["toc_do_duong_chuan_tren_Mac"] = dict(trich=m.group(0)[:200] if m else None)
    p02 = (HERE / "p02_fd_phong_cach_sach.py").read_text(encoding="utf-8")
    dev_auto = "FontDiffusionGenerator(ckpt_dir=ck, phase1_ckpt_dir=ck" in p02 and "device=" not in p02[p02.index("g = FontDiffusionGenerator("): p02.index("g._load_pipeline()")]
    inv("F2.duong_chuan_use_fst_False_nhanh_gan_gap_doi_theo_tai_lieu_va_p02_khong_truyen_device_tuc_chay_o_may_dev", bool(m) and "nhanh gần gấp đôi" in m.group(3) and dev_auto,
        dict(tai_lieu=m.groups() if m else None, p02_khong_truyen_device=dev_auto))

    R["tong"] = dict(pass_=sum(v["ok"] for v in R["invariants"].values()), fail=sum(not v["ok"] for v in R["invariants"].values()), n=len(R["invariants"]),
                     fail_ten=[k for k, v in R["invariants"].items() if not v["ok"]])
    (OUT / "01_so_lieu.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("TỔNG:", R["tong"])


if __name__ == "__main__":
    main()
