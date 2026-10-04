"""r_goi_du_lieu_04_toan_ven_goi.py — HƯỚNG 4: TOÀN VẸN GÓI (thư mục + zip) của gói Kaggle sinh ảnh theo sách.

  4a  ckpt: sha256 trong manifest.json == tệp trong gói == tệp gốc font_diffusion/ckpt/PROD == bao_cao.json; đầu safetensors hợp lệ (tổng byte tensor == kích thước tệp).
  4b  plan.json: sha tính lại == plan['sha'] == manifest == báo cáo; dựng lại kế hoạch bằng make_plan của bộ chạy từ spec + cmap phông của gói == plan.json (từng trường);
      số việc/số ảnh == báo cáo; việc liền nhau, id duy nhất, thứ tự xen kẽ.
  4c  code/: so từng tệp .py với font_diffusion/src, font_diffusion/inference, core/ranking/fontdiffusion_gen.py, fd_wrapper_fix.py, w_runner_khung_chay.py; đóng kín import nội bộ; mô-đun bên thứ ba cần pip.
  4d  zip: mở được, testzip (CRC), cấu trúc gốc (plan.json ở gốc), số mục cấp một, tên lạ, trùng tên, từng thành viên == tệp trong thư mục (CRC32, kích thước), kiểu nén.
  4e  tên phông có dấu cách: fonts.json ↔ tệp; FontManager (trích nguyên văn lớp từ mã trong gói) và RealGen._font tra phông theo stem an toàn.
  4f  báo cáo/hướng dẫn: số trong bao_cao.json và docs/KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md khớp gói; kích thước.

    cd measure_out/_tn11/full/review/goi_du_lieu && PYTHONDONTWRITEBYTECODE=1 ../../../../../.venv/bin/python -B ../../../../../lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_goi_du_lieu_04_toan_ven_goi.py
Ra: 04_toan_ven_goi.json.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast
import hashlib
import json
import os
import re
import time
import zipfile
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r_goi_du_lieu_00_lib as Lb  # noqa: E402
import w_runner_khung_chay as R  # noqa: E402  (chỉ import hàm; không chạy main)


def crc_file(p: Path) -> int:
    c = 0
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def main():
    t0 = time.time()
    inv = Lb.Inv()
    res: dict = dict(tao_luc=time.strftime("%Y-%m-%d %H:%M:%S"))
    plan = json.loads((Lb.PACK / "plan.json").read_text(encoding="utf-8"))
    man = json.loads((Lb.PACK / "manifest.json").read_text(encoding="utf-8"))
    rep = json.loads(Lb.REPORT.read_text(encoding="utf-8"))
    # ------------------------------------------------------------------------------ 4a ckpt
    prod = Lb.REPO / "font_diffusion" / "ckpt" / "PROD"
    ck = {}
    for fn, h in man["ckpt_sha256"].items():
        pk = Lb.PACK / "ckpt" / fn
        src = prod / fn
        d = dict(manifest=h[:16], goi=Lb.sha256_file(pk)[:16] if pk.exists() else None, goc=Lb.sha256_file(src)[:16] if src.exists() else None,
                 bao_cao=rep["ckpt_sha256"].get(fn, "")[:16], bytes_goi=pk.stat().st_size if pk.exists() else None, bytes_goc=src.stat().st_size if src.exists() else None)
        d["tat_ca_bang_nhau"] = len({d["manifest"], d["goi"], d["goc"], d["bao_cao"]}) == 1
        with open(pk, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            head = json.loads(f.read(n).decode("utf-8"))
        tens = {k: v for k, v in head.items() if k != "__metadata__"}
        end = max((v["data_offsets"][1] for v in tens.values()), default=0)
        d["safetensors"] = dict(n_tensor=len(tens), kich_thuoc_header=n, byte_tensor=end, dung_kich_thuoc_tep=(8 + n + end == d["bytes_goi"]), dtype=dict(Counter(v["dtype"] for v in tens.values())))
        ck[fn] = d
    res["4a_ckpt"] = ck
    res["4a_ckpt_la_con_tro_LFS"] = {fn: (d["bytes_goi"] < 2000) for fn, d in ck.items()}
    inv.check("4a_3_ckpt_manifest==goi==goc==bao_cao (sha256)", len(ck) == 3 and all(d["tat_ca_bang_nhau"] for d in ck.values()), {fn: d for fn, d in ck.items() if not d["tat_ca_bang_nhau"]})
    inv.check("4a_dau_safetensors_hop_le_va_khop_kich_thuoc", all(d["safetensors"]["dung_kich_thuoc_tep"] for d in ck.values()))
    inv.check("4a_ckpt_khong_phai_con_tro_git_LFS (kích thước > 1 MB)", all(d["bytes_goi"] > 1_000_000 for d in ck.values()), {fn: d["bytes_goi"] for fn, d in ck.items()})
    # ------------------------------------------------------------------------------ 4b plan
    sha_re = R.plan_sha(plan)
    res["4b_plan_sha"] = dict(trong_plan=plan["sha"], tinh_lai=sha_re, manifest=man["plan_sha"], bao_cao=rep["plan_sha"], load_plan_cua_bo_chay_ok=None)
    inv.check("4b_plan_sha_tinh_lai==plan['sha']==manifest==bao_cao", len({plan["sha"], sha_re, man["plan_sha"], rep["plan_sha"]}) == 1)
    try:
        R.load_plan(Lb.PACK)
        res["4b_plan_sha"]["load_plan_cua_bo_chay_ok"] = True
    except SystemExit as e:
        res["4b_plan_sha"]["load_plan_cua_bo_chay_ok"] = str(e)
    inv.check("4b_load_plan_cua_bo_chay_chap_nhan", res["4b_plan_sha"]["load_plan_cua_bo_chay_ok"] is True)
    # dựng lại từ spec + cmap phông của GÓI
    spec = json.loads((Lb.SPEC / "spec.json").read_text(encoding="utf-8"))
    fonts_json = json.loads((Lb.PACK / "fonts.json").read_text(encoding="utf-8"))
    fonts = {}
    for name in spec["fonts"]:
        fonts[name] = Lb.cmap_union(Lb.PACK / "code" / "font_diffusion" / "fonts" / fonts_json[name])
    books = {}
    for b, v in spec["books"].items():
        books[b] = dict(chars=[c for c in Path(v["chars_file"]).read_text(encoding="utf-8").split("\n") if c], styles=len(v["style_pngs"]))
        books[b]["font_of"] = dict(ln.split("\t") for ln in Path(v["font_of_file"]).read_text(encoding="utf-8").split("\n") if "\t" in ln)
    plan2 = R.make_plan(books, [("base", 0)], 128, fonts)
    res["4b_dung_lai_ke_hoach"] = dict(bang_nhau_tung_truong=(plan2 == plan), sha_dung_lai=plan2["sha"], khac_o_truong=[k for k in plan if plan.get(k) != plan2.get(k)])
    inv.check("4b_dung_lai_make_plan_tu_spec+cmap_phong_goi == plan.json (từng trường)", plan2 == plan, res["4b_dung_lai_ke_hoach"]["khac_o_truong"])
    spec_fonts_src = {}
    for name, rel in spec["fonts"].items():
        a, b2 = Lb.REPO / rel, Lb.PACK / "code" / "font_diffusion" / "fonts" / fonts_json[name]
        spec_fonts_src[name] = dict(nguon=rel, bang_nhau_byte=(Lb.sha256_file(a) == Lb.sha256_file(b2)), sha=Lb.sha256_file(b2)[:16])
    res["4b_phong_trong_goi_bang_phong_nguon"] = spec_fonts_src
    inv.check("4b_6_phong_trong_goi_trung_byte_voi_tep_nguon", all(v["bang_nhau_byte"] for v in spec_fonts_src.values()), [k for k, v in spec_fonts_src.items() if not v["bang_nhau_byte"]])
    tasks = plan["tasks"]
    n_img = sum(len(v["chars"]) for v in plan["books"].values()) * len(plan["passes"])
    ids = [t["id"] for t in tasks]
    cont = True
    for b, v in plan["books"].items():
        ts = [t for t in tasks if t["book"] == b]
        pos = 0
        for t in sorted(ts, key=lambda x: x["lo"]):
            cont &= (t["lo"] == pos and t["hi"] > t["lo"] and t["hi"] - t["lo"] <= plan["shard"])
            pos = t["hi"]
        cont &= (pos == len(v["chars"]))
    order_ok = all(tasks[i]["id"].split("/")[2] <= tasks[i + 1]["id"].split("/")[2] for i in range(len(tasks) - 1))
    res["4b_so_viec_va_anh"] = dict(n_viec=len(tasks), n_viec_bao_cao=rep["n_viec"], n_anh=int(n_img), n_anh_bao_cao=rep["n_anh"], id_duy_nhat=len(set(ids)) == len(ids), viec_lien_nhau_phu_het_chu=cont,
                                    xen_ke_theo_lo_tang_dan=order_ok, passes=plan["passes"], style_dung=sorted({t["style"] for t in tasks}), mo_hinh=sorted({t["model"] for t in tasks}),
                                    gio_gpu_tinh_lai=round(n_img * 2.7 / 3600, 1), gio_gpu_bao_cao=rep["gio_gpu_uoc"], gio_phien_tinh_lai=round(n_img * 2.7 / 3600 / 2, 1), gio_phien_bao_cao=rep["gio_phien_2gpu_uoc"],
                                    so_ca_6_5h=round(n_img * 2.7 / 3600 / 2 / 6.5, 2))
    inv.check("4b_410_viec_51730_anh_khop_bao_cao", len(tasks) == rep["n_viec"] == 410 and n_img == rep["n_anh"] == 51730, (len(tasks), n_img))
    inv.check("4b_id_viec_duy_nhat_va_lo_lien_nhau_phu_het_chu_moi_cuon", len(set(ids)) == len(ids) and cont)
    inv.check("4b_gio_GPU_38,8_va_19,4_tinh_lai_khop_bao_cao", res["4b_so_viec_va_anh"]["gio_gpu_tinh_lai"] == rep["gio_gpu_uoc"] and res["4b_so_viec_va_anh"]["gio_phien_tinh_lai"] == rep["gio_phien_2gpu_uoc"])
    pb = {b: dict(chars=len(v["chars"]), uncovered=len(v["uncovered"])) for b, v in plan["books"].items()}
    inv.check("4b_manifest.books_khop_plan_va_bao_cao.theo_sach_ke_hoach", pb == man["books"] and all(rep["theo_sach_ke_hoach"][b] == dict(chu=v["chars"], khong_phong=v["uncovered"]) for b, v in pb.items()))
    # style_1 không dùng
    res["4b_style_1_png_khong_duoc_ke_hoach_dung"] = dict(plan_styles_moi_cuon=sorted({v["styles"] for v in plan["books"].values()}), style_trong_cac_viec=sorted({t["style"] for t in tasks}),
                                                         ghi_chu="kế hoạch chỉ dùng style_0 (passes base:0); 10 tệp style_1.png đi kèm nhưng KHÔNG được dùng")
    # plan_sha không phủ ảnh phong cách / phông / ckpt?
    res["4b_plan_sha_pham_vi"] = dict(truong_trong_plan=sorted(plan.keys()), khong_chua_bam_anh_phong_cach=True, khong_chua_bam_phong=True,
                                      manifest_co_bam=sorted(k for k in man if "sha" in k), ghi_chu="đổi ảnh phong cách/phông không đổi plan_sha ⇒ bộ chạy không phát hiện trộn hai phiên bản ảnh phong cách")
    # ------------------------------------------------------------------------------ 4c code
    pairs = []
    for sub in ("src", "inference"):
        for dp, dn, fn in os.walk(Lb.REPO / "font_diffusion" / sub):
            for f in fn:
                if f.endswith(".py"):
                    rel = os.path.relpath(os.path.join(dp, f), Lb.REPO / "font_diffusion")
                    pairs.append((Lb.REPO / "font_diffusion" / rel, Lb.PACK / "code" / "font_diffusion" / rel, f"font_diffusion/{rel}"))
    pairs += [(Lb.REPO / "core/ranking/fontdiffusion_gen.py", Lb.PACK / "code/core/ranking/fontdiffusion_gen.py", "core/ranking/fontdiffusion_gen.py"),
              (Lb.HERE / "fd_wrapper_fix.py", Lb.PACK / "code/fd_wrapper_fix.py", "fd_wrapper_fix.py"),
              (Lb.HERE / "w_runner_khung_chay.py", Lb.PACK / "w_runner_khung_chay.py", "w_runner_khung_chay.py")]
    diff, miss = [], []
    for a, b2, tag in pairs:
        if not b2.exists():
            miss.append(tag); continue
        if Lb.sha256_file(a) != Lb.sha256_file(b2):
            diff.append(tag)
    pack_py = {p.relative_to(Lb.PACK / "code" / "font_diffusion").as_posix() for p in (Lb.PACK / "code" / "font_diffusion").rglob("*.py")}
    src_py = {f"{sub}/{p.relative_to(Lb.REPO / 'font_diffusion' / sub).as_posix()}" for sub in ("src", "inference") for p in (Lb.REPO / "font_diffusion" / sub).rglob("*.py")}
    non_py = sorted(p.relative_to(Lb.REPO / "font_diffusion").as_posix() for sub in ("src", "inference") for p in (Lb.REPO / "font_diffusion" / sub).rglob("*") if p.is_file() and p.suffix not in (".py", ".pyc"))
    res["4c_code"] = dict(so_tep_so_sanh=len(pairs), khac_noi_dung=diff, thieu_trong_goi=miss, tep_py_thua_trong_goi=sorted(pack_py - src_py - {"fonts"}), tep_khong_phai_py_ben_nguon_khong_dong_goi=non_py,
                          git_trang_thai_font_diffusion="submodule (git ls-files chỉ liệt kê gitlink); mã hiện hành = tệp trong thư mục font_diffusion/")
    inv.check("4c_78_tep_ma_trong_goi_trung_byte_voi_ma_hien_hanh", not diff and not miss and len(pairs) == 78, (len(pairs), diff, miss))
    # đóng kín import nội bộ + bên thứ ba
    mods_internal, third = set(), Counter()
    py_files = list((Lb.PACK / "code").rglob("*.py")) + [Lb.PACK / "w_runner_khung_chay.py"]
    stdlib = set(sys.stdlib_module_names)
    internal_tops = {"src", "inference", "core", "fd_wrapper_fix"}
    for p in py_files:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for nm in names:
                top = nm.split(".")[0]
                if top in internal_tops:
                    mods_internal.add(nm)
                elif top not in stdlib:
                    third[top] += 1
    unresolved = []
    for m in sorted(mods_internal):
        base = Lb.PACK / "code" / ("font_diffusion" if m.split(".")[0] in ("src", "inference") else "")
        pth = base / (m.replace(".", "/") + ".py")
        pkg = base / m.replace(".", "/") / "__init__.py"
        if not (pth.exists() or pkg.exists()):
            # có thể là tên hàm: from src.tools.utils import x  -> m = src.tools.utils đã được kiểm; nếu m là gói thì pkg
            unresolved.append(m)
    pip_map = R.PIP
    cov = {t: (t in pip_map) for t in third}
    res["4c_import"] = dict(noi_bo_so_mo_dun=len(mods_internal), noi_bo_khong_tim_thay=unresolved, ben_thu_ba=dict(sorted(third.items())),
                            duoc_ensure_deps_cai_neu_thieu=sorted(t for t, v in cov.items() if v),
                            khong_do_ensure_deps_cai=sorted(t for t, v in cov.items() if not v),
                            ghi_chu="mô-đun 'khong_do_ensure_deps_cai' phải có sẵn trên ảnh Docker Kaggle (torch, torchvision, numpy, PIL, yaml, cv2 ...): CHƯA xác minh trên Kaggle thật")
    inv.check("4c_moi_import_noi_bo_(src, inference, core, fd_wrapper_fix)_co_tep_trong_goi", not unresolved, unresolved)
    # ---- 4c2: bao đóng import Ở MỨC MODULE từ các điểm vào của RealGen (không tính import trong hàm/lớp), mô-đun bên thứ ba nào bắt buộc
    code_root, fd_root = Lb.PACK / "code", Lb.PACK / "code" / "font_diffusion"

    def resolve(mod: str, pkg_ctx: str | None = None, level: int = 0):
        """tên mô-đun nội bộ -> đường tệp .py (trong gói) hoặc None."""
        if level:
            parts = pkg_ctx.split(".")
            parts = parts[:len(parts) - level + 1] if level > 0 else parts
            mod = ".".join(parts + ([mod] if mod else []))
        for base in (fd_root, code_root):
            f, g = base / (mod.replace(".", "/") + ".py"), base / mod.replace(".", "/") / "__init__.py"
            if f.exists():
                return f, mod
            if g.exists():
                return g, mod
        return None, mod

    def module_level_imports(tree, mod_name, is_pkg):
        out = []   # (kiểu 'must'|'optional', tên đầy đủ, level, các tên nhập)

        def visit(body, optional):
            for node in body:
                if isinstance(node, ast.Import):
                    for a in node.names:
                        out.append((optional, a.name, 0, []))
                elif isinstance(node, ast.ImportFrom):
                    out.append((optional, node.module or "", node.level, [a.name for a in node.names]))
                elif isinstance(node, ast.Try):
                    visit(node.body, True)
                    for h in node.handlers:
                        visit(h.body, True)
                    visit(node.orelse, True); visit(node.finalbody, True)
                elif isinstance(node, (ast.If, ast.With)):
                    visit(node.body, optional)
                    if isinstance(node, ast.If):
                        visit(node.orelse, optional)
        visit(tree.body, False)
        return out
    entry = ["core.ranking.fontdiffusion_gen", "fd_wrapper_fix", "inference.sample_optimized", "src.tools.utils", "src.configs.fontdiffuser"]
    seen, queue, must3, opt3, lazy_note = set(), list(entry), Counter(), Counter(), []
    unresolved_mod = []
    while queue:
        m = queue.pop()
        if m in seen:
            continue
        seen.add(m)
        f, m = resolve(m)
        if f is None:
            unresolved_mod.append(m); continue
        is_pkg = f.name == "__init__.py"
        tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        ctx = m if is_pkg else ".".join(m.split(".")[:-1])
        for optional, name, level, names in module_level_imports(tree, m, is_pkg):
            full = name
            if level:
                parts = (m if is_pkg else ".".join(m.split(".")[:-1])).split(".")
                parts = parts[:len(parts) - level + 1]
                full = ".".join(parts + ([name] if name else []))
            top = full.split(".")[0]
            if top in internal_tops or resolve(full)[0] is not None:
                queue.append(full)
                for n in names:
                    queue.append(full + "." + n)        # có thể là mô-đun con
            elif top in stdlib:
                continue
            else:
                (opt3 if optional else must3)[top] += 1
    unresolved_mod = [u for u in unresolved_mod if not any(u.endswith("." + n) or u == n for n in ("",))]
    pre_assumed = {"torch", "torchvision", "numpy", "PIL", "yaml", "tqdm", "scipy", "matplotlib", "pandas", "datasets"}
    klog = Lb.FULL.parent / "kaggle_out" / "log_B34.txt"
    kl = klog.read_text(encoding="utf-8", errors="replace") if klog.exists() else ""
    import re as _re
    env_k = dict(log=str(klog.relative_to(Lb.REPO)), python=(_re.search(r"Python (3\.\d+\.\d+)\)", kl) or [None, None])[1], pygame=(_re.search(r"pygame (\d[\d.]+)", kl) or [None, None])[1],
                 sdl=(_re.search(r"SDL (\d[\d.]+)", kl) or [None, None])[1], da_nap_pipeline_thanh_cong=("U-Net built successfully" in kl and "FontDiffusion loaded" in kl),
                 cac_goi_cai_thu_cong_trong_lan_do=_re.findall(r"pip install[^\n]*", kl)[:3])
    must_list = sorted(must3)
    res["4c2_bao_dong_import_muc_module"] = dict(moi_truong_Kaggle_lan_chay_truoc=env_k, ghi_chu_datasets="src/dataset/hffont_dataset_fst.py nhập `datasets` ở mức module; lần Kaggle trước (cùng mã, cùng PIP) nạp được => `datasets` có sẵn trên ảnh Kaggle (suy ra, không kiểm trực tiếp)", diem_vao=entry, so_mo_dun_noi_bo_duoc_nap=len(seen) - len(unresolved_mod), ben_thu_ba_bat_buoc=dict(sorted(must3.items())), ben_thu_ba_tuy_chon_trong_try=dict(sorted(opt3.items())),
                                                  ensure_deps_cai_neu_thieu=[t for t in must_list if t in pip_map], gia_dinh_san_co_tren_Kaggle=[t for t in must_list if t in pre_assumed],
                                                  KHONG_duoc_cai_cung_khong_gia_dinh_san_co=[t for t in must_list if t not in pip_map and t not in pre_assumed],
                                                  khong_tim_thay_tep_co_the_la_thuoc_tinh=sorted(set(unresolved_mod))[:20])
    inv.check("4c2_moi_mo_dun_ben_thu_ba_BAT_BUOC_o_muc_module_deu_duoc_ensure_deps_cai_hoac_gia_dinh_san_co_tren_Kaggle", not res["4c2_bao_dong_import_muc_module"]["KHONG_duoc_cai_cung_khong_gia_dinh_san_co"],
              res["4c2_bao_dong_import_muc_module"]["KHONG_duoc_cai_cung_khong_gia_dinh_san_co"])
    # ------------------------------------------------------------------------------ 4d zip
    z = zipfile.ZipFile(Lb.ZIP)
    infos = z.infolist()
    names = [i.filename for i in infos]
    top = sorted({n.split("/")[0] for n in names})
    t1 = time.time()
    bad = z.testzip()
    dirfiles = {p.relative_to(Lb.PACK).as_posix(): p for p in Lb.PACK.rglob("*") if p.is_file()}
    miss_in_zip = sorted(set(dirfiles) - set(names))
    extra_in_zip = sorted(set(names) - set(dirfiles))
    crc_bad, size_bad = [], []
    for i in infos:
        if i.filename in dirfiles:
            p = dirfiles[i.filename]
            if i.file_size != p.stat().st_size:
                size_bad.append(i.filename)
            elif crc_file(p) != i.CRC:
                crc_bad.append(i.filename)
    comp = Counter((Path(i.filename).suffix.lower(), "STORED" if i.compress_type == zipfile.ZIP_STORED else "DEFLATED" if i.compress_type == zipfile.ZIP_DEFLATED else str(i.compress_type)) for i in infos)
    res["4d_zip"] = dict(kich_thuoc_MB=round(Lb.ZIP.stat().st_size / 1e6, 1), kich_thuoc_bao_cao_MB=rep["zip"]["MB"], so_thanh_vien=len(infos), muc_cap_mot=top, so_muc_cap_mot=len(top), plan_json_o_goc="plan.json" in names,
                         testzip_thanh_vien_hong=bad, thieu_so_voi_thu_muc=miss_in_zip, thua_so_voi_thu_muc=extra_in_zip, khac_kich_thuoc=size_bad, khac_CRC32=crc_bad,
                         ten_la=[n for n in names if "__MACOSX" in n or ".DS_Store" in n or n.startswith("/") or ".." in n.split("/") or "\\" in n or n.endswith("/")],
                         trung_ten=[n for n, c in Counter(names).items() if c > 1], ten_co_dau_cach=[n for n in names if " " in n], ten_phi_ASCII=[n for n in names if not n.isascii()],
                         do_dai_duong_dan_lon_nhat=max(len(n) for n in names), kieu_nen={f"{k[0]}:{k[1]}": v for k, v in sorted(comp.items())}, giay_testzip=round(time.time() - t1, 1))
    inv.check("4d_zip_testzip_khong_loi_va_mo_duoc", bad is None)
    inv.check("4d_plan_json_o_goc_zip", "plan.json" in names)
    inv.check("4d_so_muc_cap_mot_<=_50", len(top) <= 50, len(top))
    inv.check("4d_zip_va_thu_muc_cung_tap_tep_va_CRC32_kich_thuoc_tung_tep", not miss_in_zip and not extra_in_zip and not crc_bad and not size_bad, (miss_in_zip[:3], extra_in_zip[:3], crc_bad[:3], size_bad[:3]))
    inv.check("4d_khong_ten_la (__MACOSX/.DS_Store/tuyet_doi/..)/trung_ten", not res["4d_zip"]["ten_la"] and not res["4d_zip"]["trung_ten"], res["4d_zip"]["ten_la"])
    inv.check("4d_kich_thuoc_zip_khop_bao_cao (±0,1 MB)", abs(res["4d_zip"]["kich_thuoc_MB"] - rep["zip"]["MB"]) <= 0.1, (res["4d_zip"]["kich_thuoc_MB"], rep["zip"]["MB"]))
    # ------------------------------------------------------------------------------ 4e phông có dấu cách
    fm_src = (Lb.PACK / "code/font_diffusion/inference/sample_optimized.py").read_text(encoding="utf-8")
    tree = ast.parse(fm_src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FontManager")
    code = ast.get_source_segment(fm_src, cls)
    import logging
    from functools import lru_cache
    from typing import Any
    loaded = []

    def load_ttf(path, fsize=128):
        loaded.append(path)
        return ("FONT", path)

    def is_char_in_font(font_path, char):
        return True
    ns = dict(os=os, logger=logging.getLogger("r_goi_du_lieu"), lru_cache=lru_cache, Any=Any, load_ttf=load_ttf, is_char_in_font=is_char_in_font)
    exec(compile(ast.Module(body=[ast.parse(code).body[0]], type_ignores=[]), "FontManager(trich_tu_goi)", "exec"), ns)
    FM = ns["FontManager"]
    fm_rows = {}
    for name, fn in fonts_json.items():
        p = Lb.PACK / "code" / "font_diffusion" / "fonts" / fn
        fm = FM(str(p))
        stem = Path(str(p)).stem
        got = fm.get_font(stem)
        fm_rows[name] = dict(tep=fn, co_dau_cach=" " in fn, ten_trong_FontManager=fm.get_font_names(), stem_RealGen=stem, tra_theo_stem_ok=(got == ("FONT", str(p))), stem_bang_ten_FontManager=(fm.get_font_names() == [stem]))
    res["4e_phong_dau_cach"] = fm_rows
    inv.check("4e_FontManager(tu_goi)_va_RealGen_tra_phong_theo_stem_ok_6_phong (kể cả 'HAN NOM A.ttf')", all(v["tra_theo_stem_ok"] and v["stem_bang_ten_FontManager"] for v in fm_rows.values()), [k for k, v in fm_rows.items() if not v["tra_theo_stem_ok"]])
    inv.check("4e_fonts.json_gia_tri_la_ten_tep_that_trong_code/font_diffusion/fonts", all((Lb.PACK / "code/font_diffusion/fonts" / fn).exists() for fn in fonts_json.values()))
    # RealGen._font dùng p.stem; worker ghép đường dẫn từ fonts.json: kiểm văn bản mã
    rg = (Lb.PACK / "w_runner_khung_chay.py").read_text(encoding="utf-8")
    res["4e_dong_ma_lien_quan"] = dict(RealGen_font_dung_stem=("fm.get_font(p.stem)" in rg), worker_ghep_duong_dan_phong=('Path(a.tmp) / "font_diffusion" / "fonts" / f' in rg),
                                      popen_dung_danh_sach_khong_shell=("subprocess.Popen(cmd, env=env)" in rg and "shell=True" not in rg))
    inv.check("4e_ma_bo_chay_dung_stem_va_Popen_danh_sach (không shell) nên dấu cách an toàn", all(res["4e_dong_ma_lien_quan"].values()), res["4e_dong_ma_lien_quan"])
    # ------------------------------------------------------------------------------ 4f báo cáo + hướng dẫn + kích thước
    parts = {}
    for sub in ("ckpt", "code", "data"):
        parts[sub] = round(sum(p.stat().st_size for p in (Lb.PACK / sub).rglob("*") if p.is_file()) / 1e6, 2)
    parts["code/font_diffusion/fonts"] = round(sum(p.stat().st_size for p in (Lb.PACK / "code/font_diffusion/fonts").rglob("*") if p.is_file()) / 1e6, 2)
    parts["goc(plan,manifest,fonts,runner)"] = round(sum((Lb.PACK / f).stat().st_size for f in ("plan.json", "manifest.json", "fonts.json", "w_runner_khung_chay.py")) / 1e6, 3)
    parts["tong_thu_muc"] = round(sum(p.stat().st_size for p in Lb.PACK.rglob("*") if p.is_file()) / 1e6, 1)
    res["4f_kich_thuoc_MB"] = parts
    # so với bao_cao
    txt_n = {b: len([c for c in (Lb.SPEC / f"chars_{b}.txt").read_text(encoding="utf-8").split("\n") if c]) for b in Lb.ORDER}
    font_cnt = {b: dict(Counter(plan["font_names"][i] for i in plan["books"][b]["font_idx"])) for b in Lb.ORDER}
    res["4f_bao_cao_khop_goi"] = dict(so_chu_danh_sach_khop=all(rep["sach"][b]["so_chu_danh_sach"] == txt_n[b] for b in Lb.ORDER),
                                     phong_theo_cuon_khop=all(dict(sorted(rep["sach"][b]["phong"].items())) == dict(sorted(font_cnt[b].items())) for b in Lb.ORDER),
                                     khong_phong_tong=rep["khong_phong_tong"], khong_phong_tinh_lai=sum(len(v["uncovered"]) for v in plan["books"].values()))
    inv.check("4f_bao_cao.so_chu_va_phong_theo_cuon_khop_plan", res["4f_bao_cao_khop_goi"]["so_chu_danh_sach_khop"] and res["4f_bao_cao_khop_goi"]["phong_theo_cuon_khop"])
    doc = (Lb.REPO / "docs/KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md").read_text(encoding="utf-8")
    claims = {"472 MB": "472 MB" in doc, "410 việc": "410 việc" in doc, "51.730 ảnh": "51.730" in doc, "38,8 giờ GPU": "38,8 giờ GPU" in doc, "19,4 giờ phiên": "19,4 giờ phiên" in doc, "plan.json nằm ngay gốc": "plan.json` nằm ngay gốc" in doc,
              "1 chữ ở stt11": "hiện 1 chữ ở stt11" in doc}
    res["4f_so_trong_huong_dan"] = dict(co_trong_doc=claims, zip_MB_thuc=res["4d_zip"]["kich_thuoc_MB"], viec=len(tasks), anh=int(n_img))
    inv.check("4f_so_lieu_huong_dan_(472MB,410,51.730,38,8,19,4,1_chữ_stt11)_khop_goi", all(claims.values()) and abs(res["4d_zip"]["kich_thuoc_MB"] - 472) < 1, claims)
    # kiểm gói bằng chính audit_pack của bộ chạy
    ap = R.audit_pack(Lb.PACK)
    res["4f_audit_pack_cua_bo_chay"] = ap
    inv.check("4f_audit_pack_cua_bo_chay_sach_va_top_level_ok", ap["sach"] and ap["top_level_ok"], ap)
    res["bat_bien"] = inv.summary(); res["bat_bien_chi_tiet"] = inv.rows; res["giay"] = round(time.time() - t0, 1)
    Lb.jdump(res, "04_toan_ven_goi.json")
    print(f"[04] xong {res['giay']}s; bất biến {res['bat_bien']['dat']}/{res['bat_bien']['tong']} rớt={res['bat_bien']['rot']}")


if __name__ == "__main__":
    main()
