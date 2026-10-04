"""r_huong_dan_02_lenh_va_duong_dan.py — REVIEW hướng "huong_dan" (2/5): từng LỆNH và đường dẫn trong
docs/KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md có đúng với bộ chạy/CLI hiện có không. Trích lệnh thẳng từ tệp hướng dẫn, đưa qua argparse THẬT của
w_runner_khung_chay.py (các hàm cmd_* bị thay bằng bộ bắt tham số), qua bộ phân tích tham số THẬT của `hf download` (huggingface_hub trong .venv),
rồi chạy thật `validate` trên cấu trúc thư mục giống kết quả `hf download`. 0 GPU, 0 API, 0 mạng.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_huong_dan_02_lenh_va_duong_dan.py
Ra: measure_out/_tn11/full/review/huong_dan/02_lenh_va_duong_dan.json (+ val_sim/, prev_sim/ tạm, xoá khi xong)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import contextlib
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FULL = REPO / "measure_out" / "_tn11" / "full"
OUT = FULL / "review" / "huong_dan"
GUIDE = REPO / "docs" / "KAGGLE_SINH_ANH_THEO_SACH_2026-10-04.md"
PACK = FULL / "kaggle_pack_20261004"
RUNNER = HERE / "w_runner_khung_chay.py"

spec = importlib.util.spec_from_file_location("K", RUNNER)
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)

R = {"tao": time.strftime("%Y-%m-%d %H:%M:%S"), "invariants": {}, "lenh": []}


def inv(name, ok, detail=None):
    R["invariants"][name] = dict(ok=bool(ok), detail=detail)
    print(("PASS" if ok else "FAIL"), name, "" if detail is None else json.dumps(detail, ensure_ascii=False, default=str)[:300], flush=True)


# ------------------------------------------------------------------------------------------------ trích lệnh từ tệp hướng dẫn
def code_blocks(text: str) -> list[list[str]]:
    blocks, cur, inb = [], [], False
    for ln in text.split("\n"):
        if ln.strip().startswith("```"):
            if inb:
                blocks.append(cur); cur = []
            inb = not inb
            continue
        if inb:
            cur.append(ln)
    return blocks


def join_continuations(lines: list[str]) -> list[str]:
    out, buf = [], ""
    for ln in lines:
        s = ln.rstrip()
        if s.endswith("\\"):
            buf += s[:-1].strip() + " "
        else:
            out.append((buf + s.strip()).strip()); buf = ""
    return [o for o in out if o]


PLACE = [(r"\$\(find /kaggle/input -name w_runner_khung_chay\.py \| head -1\)", "w_runner_khung_chay.py"), (r"<hf_user>", "nguoidung"), (r"<user>", "nguoidung"),
         (r"<các thư mục>", "/tmp/b1 /tmp/b2"), (r"<tên-notebook>", "nb-ca-truoc")]


def fill(cmd: str) -> str:
    for a, b in PLACE:
        cmd = re.sub(a, b, cmd)
    return cmd


def parse_runner(argv: list[str]) -> dict:
    """Đưa argv qua argparse THẬT của bộ chạy; trả về tham số đã parse hoặc lỗi."""
    cap = {}
    saved = {n: getattr(K, n) for n in ("cmd_run", "cmd_worker", "cmd_validate", "cmd_unbundle", "cmd_pack", "cmd_make_plan")}
    for n in saved:
        setattr(K, n, (lambda nm: (lambda a: cap.setdefault("args", {**vars(a), "fn": nm})))(n))
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            K.main(argv)
        return dict(ok=True, args=cap.get("args"))
    except SystemExit as e:
        return dict(ok=False, exit=e.code, stderr=err.getvalue().strip().splitlines()[-1:] if err.getvalue() else None)
    finally:
        for n, f in saved.items():
            setattr(K, n, f)


def main():
    g = GUIDE.read_text(encoding="utf-8")
    blocks = [join_continuations(b) for b in code_blocks(g)]
    cmds = [c for b in blocks for c in b]
    R["so_khoi_ma"] = len(blocks)
    R["cac_lenh_trich_tu_huong_dan"] = cmds

    # ---- 1. lệnh Kaggle (ô code) qua argparse thật
    kcmds = [c for c in cmds if c.startswith("!python")]
    inv("L0.tim_thay_2_lenh_Kaggle_ca_thu_va_ca_that", len(kcmds) == 2, len(kcmds))
    for c in kcmds:
        argv = shlex.split(fill(c)[len("!python"):])
        assert argv[0] == "w_runner_khung_chay.py"
        pr = parse_runner(argv[1:])
        a = pr["args"] or {}
        R["lenh"].append(dict(lenh=c[:140], dung_argparse=pr["ok"], bat_tham_so={k: a.get(k) for k in ("fn", "out", "tmp", "bundle_dir", "remote", "max_tasks", "budget_hours", "safety_s", "workers", "device", "gen_bs", "est_s", "pack", "prev", "bundle_every")}))
        inv("L1.cu_phap_dung_voi_argparse:" + ("ca_thu" if "--max-tasks" in c else "ca_that"), pr["ok"], pr.get("stderr") or {k: a.get(k) for k in ("max_tasks", "budget_hours", "safety_s", "device", "workers")})
    ca_thu = next(c for c in kcmds if "--max-tasks" in c)
    a_thu = parse_runner(shlex.split(fill(ca_thu)[len("!python"):])[1:])["args"]
    deadline_offset = a_thu["budget_hours"] * 3600 - a_thu["safety_s"]
    R["ca_thu_deadline_tu_dau_run_s"] = deadline_offset
    inv("L2.ca_thu_deadline_chi_600s_ke_tu_dau_run", abs(deadline_offset - 600) < 1e-6, dict(budget_h=a_thu["budget_hours"], safety_s=a_thu["safety_s"], offset_s=deadline_offset))
    # không truyền --pack / --device / --workers => dò gói ở /kaggle/input, tự chọn thiết bị
    inv("L3.lenh_Kaggle_khong_truyen_device_nen_tu_roi_ve_CPU_neu_phien_khong_co_GPU", a_thu["device"] is None and a_thu["workers"] == 0, dict(device=a_thu["device"], workers=a_thu["workers"]))
    src = RUNNER.read_text(encoding="utf-8")
    inv("L3b.ma_nguon_co_roi_CPU_im_lang", 'dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")' in src and "ngpu = max(1, torch.cuda.device_count())" in src,
        "w_runner_khung_chay.py: cmd_worker dòng 'dev = a.device or ...' và cmd_run 'ngpu = max(1, ...)'; không có raise khi không có CUDA")
    inv("L3c.khong_co_cong_chan_neu_khong_co_GPU", "torch.cuda.is_available()" in src and not re.search(r"raise SystemExit\([^)]*(GPU|CUDA|cuda)", src), None)

    # ---- 2. lệnh `validate` + `hf download` ở mục 2.6
    vcmd = next(c for c in cmds if "validate" in c)
    pv = parse_runner(shlex.split(fill(vcmd))[2:])      # bỏ '.venv/bin/python' và đường dẫn tệp
    inv("L4.validate_cu_phap_dung_argparse", pv["ok"] and pv["args"]["fn"] == "cmd_validate", pv.get("stderr") or {k: pv["args"].get(k) for k in ("pack", "out")})
    inv("L4b.validate_pack_va_out_ton_tai_hoac_se_duoc_tao", (REPO / pv["args"]["pack"]).exists() and (REPO / pv["args"]["out"]).parent.exists(), pv["args"])
    hcmd = next(c for c in cmds if c.startswith("hf download"))
    R["lenh_hf"] = hcmd
    inv("L5.hf_nam_ngoai_PATH_neu_khong_kich_hoat_venv_(lenh_ke_ben_dung_.venv/bin/python)", (REPO / ".venv/bin/hf").exists() and not hcmd.startswith(".venv/bin/hf"), dict(co_trong_venv=(REPO / ".venv/bin/hf").exists(), lenh=hcmd[:40]))
    try:
        import typer
        from huggingface_hub.cli.hf import app
        import huggingface_hub
        cmd = typer.main.get_command(app).commands["download"]
        toks = shlex.split(fill(hcmd))[2:]
        ctx = cmd.make_context("download", toks, resilient_parsing=False)
        inv("L6.hf_download_cu_phap_dung_voi_huggingface_hub_" + huggingface_hub.__version__, ctx.params["repo_type"] == "dataset" and ctx.params["local_dir"].endswith("kaggle_ket_qua") and ctx.params["repo_id"] == "nguoidung/gannhanocr-tn11-full",
            {k: v for k, v in ctx.params.items() if v not in (None, (), [], False) and k != "max_workers"})
    except Exception as e:  # noqa: BLE001
        inv("L6.hf_download_cu_phap_dung", False, f"{type(e).__name__}: {e}")
    inv("L7.huong_dan_khong_noi_dang_nhap_HF_tren_Mac_cho_repo_private", not re.search(r"hf auth login|huggingface-cli login|HF_TOKEN=|--token", g[g.index("Tải kết quả về Mac"): g.index("Phương án B (không HF)")]), "đoạn mục 2.6 không có bước đăng nhập")
    tokf = [Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"]
    R["may_nay_da_dang_nhap_HF"] = dict(tep_token=[p.exists() for p in tokf], HF_TOKEN_env=bool(os.environ.get("HF_TOKEN")))

    # ---- 3. Phương án B: --prev / unbundle
    thatc = next(c for c in kcmds if "--max-tasks" not in c)
    b_cmd = re.sub(r"--remote \S+ ", "", fill(thatc)) + " --prev /kaggle/input/nb-ca-truoc --bundle-every 1800"
    pb = parse_runner(shlex.split(b_cmd[len("!python"):])[1:])
    inv("B1.phuong_an_B_run_cu_phap_dung", pb["ok"] and pb["args"]["remote"] is None and pb["args"]["prev"] == ["/kaggle/input/nb-ca-truoc"] and pb["args"]["bundle_every"] == 1800.0, pb.get("stderr") or {k: pb["args"].get(k) for k in ("remote", "prev", "bundle_every", "bundle_dir")})
    ub = [m for m in re.findall(r"`(w_runner_khung_chay\.py unbundle[^`]*)`", g)]
    pu = parse_runner(shlex.split(fill(ub[0]))[1:]) if ub else dict(ok=False)
    inv("B2.unbundle_cu_phap_dung_argparse", pu["ok"], pu.get("stderr") or {k: pu["args"].get(k) for k in ("bundles", "dest")})
    inv("B3.lenh_unbundle_trong_huong_dan_thieu_duong_dan_python_va_tep", ub and not ub[0].startswith((".venv", "python")), ub[0][:60] if ub else None)

    # mô phỏng: ca 3 chỉ nạp đầu ra ca 2 thì làm lại việc của ca 1
    plan = K.load_plan(PACK)
    sim = OUT / "prev_sim"
    shutil.rmtree(sim, ignore_errors=True)
    ids = [t["id"] for t in plan["tasks"]]
    ca1, ca2 = ids[:6], ids[6:12]

    def make_bundle(root: Path, done_ids_: list, session: str, already: set):
        o = root / "work"
        for tid in done_ids_:
            p = o / K.rel_path(tid); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x")
        K.bundle_new(o, root / "nb" / "tn11_bundles", session, 0, already, plan["sha"])

    make_bundle(sim / "ca1", ca1, "s1", set())
    pd1 = K.prev_done([sim / "ca1" / "nb"], plan["sha"])
    make_bundle(sim / "ca2", ca2, "s2", pd1)                  # ca 2 chạy với --prev = đầu ra ca 1 => bundle chỉ chứa việc mới
    only2 = K.prev_done([sim / "ca2" / "nb"], plan["sha"])
    both = K.prev_done([sim / "ca1" / "nb", sim / "ca2" / "nb"], plan["sha"])
    inv("B4.dau_ra_moi_ca_chi_chua_viec_moi_cua_ca_do", only2 == set(ca2) and pd1 == set(ca1), dict(ca1=len(pd1), ca2=len(only2)))
    inv("B5.ca_3_chi_nap_dau_ra_ca_2_thi_bo_sot_viec_ca_1_va_lam_lai", set(ca1).isdisjoint(only2) and both == set(ca1) | set(ca2), dict(viec_lam_lai_neu_chi_nap_ca2=len(set(ca1) - only2), neu_nap_ca1_va_ca2=len(set(ca1) - both)))
    shutil.rmtree(sim, ignore_errors=True)

    # ---- 4. dò gói `find_pack` ở các bố cục /kaggle/input khác nhau (khớp 'plan.json nằm ngay gốc')
    import glob as _glob
    real_glob, real_isfile = _glob.glob, os.path.isfile
    layouts = {
        "dataset_gốc /kaggle/input/tn11-full-pack/plan.json": ["/kaggle/input/tn11-full-pack/plan.json"],
        "bố cục sâu 3 cấp /kaggle/input/datasets/<chủ>/tn11-full-pack/plan.json": ["/kaggle/input/datasets/chu/tn11-full-pack/plan.json"],
        "dataset + đầu ra notebook trước (không có plan.json)": ["/kaggle/input/tn11-full-pack/plan.json"],
    }
    found = {}
    for name, planfiles in layouts.items():
        dirs1 = sorted({"/".join(p.split("/")[:4]) for p in planfiles} | ({"/kaggle/input/nb-ca-truoc"} if "notebook" in name else set()))
        dirs2 = sorted({"/".join(p.split("/")[:5]) for p in planfiles})
        dirs3 = sorted({"/".join(p.split("/")[:6]) for p in planfiles})
        table = {"/kaggle/input/*": dirs1, "/kaggle/input/*/*": dirs2, "/kaggle/input/*/*/*": dirs3}
        _glob.glob = lambda pat, _t=table: list(_t.get(pat, []))
        os.path.isfile = lambda p, _pf=planfiles: p in {str(Path(x)) for x in _pf} or real_isfile(p)
        try:
            found[name] = str(K.find_pack(None))
        except SystemExit as e:
            found[name] = f"SystemExit: {e}"
        finally:
            _glob.glob, os.path.isfile = real_glob, real_isfile
    R["find_pack_theo_bo_cuc"] = found
    inv("F1.find_pack_tim_duoc_plan_json_o_ca_3_bo_cuc", all(v.endswith("/tn11-full-pack") for v in found.values()), found)

    # ---- 5. p11a_lam_moi_du_lieu.sh
    sh = HERE / "p11a_lam_moi_du_lieu.sh"
    txt = sh.read_text(encoding="utf-8")
    steps = re.search(r"for s in ([^;]+); do", txt).group(1).split()
    missing = [f"w_hien_trang_du_lieu_{s}.py" for s in steps if not (HERE / f"w_hien_trang_du_lieu_{s}.py").exists()]
    extra = [x for x in ("w_quy_mo_va_ngan_sach_00_chay_het.py", "p11_dung_goi_toan_bo.py") if not (HERE / x).exists()]
    inv("S1.p11a_ton_tai_chay_duoc_moi_buoc_ton_tai", os.access(sh, os.X_OK) and not missing and not extra and (REPO / ".venv/bin/python").exists(), dict(thieu=missing + extra, thi_hanh=bool(os.stat(sh).st_mode & stat.S_IXUSR), buoc=len(steps)))
    inv("S2.p11a_ra_thu_muc_moi_theo_ngay_gio_khong_phai_20261004", 'kaggle_pack_$(date +%Y%m%d-%H%M)' in txt, "OUT mặc định kaggle_pack_<YYYYMMDD-HHMM>: các lệnh mục 2.1/2.6 vẫn ghi tên kaggle_pack_20261004")
    p11 = (HERE / "p11_dung_goi_toan_bo.py").read_text(encoding="utf-8")
    inv("S3.p11_ghi_de_kaggle_spec_tai_cho_khi_lam_moi", 'spec_dir = FULL / "kaggle_spec"' in p11 and 'write_text(json.dumps(spec' in p11, "p11 ghi đè kaggle_spec/ (chars_*.txt, font_of_*.tsv, spec.json, styles/)")

    # ---- 6. `validate` thật trên cấu trúc như `hf download` (shards/<cuốn>/<mô hình>_s<j>/<NNNNN>.tar + tệp phụ của HF)
    vs = OUT / "val_sim"
    shutil.rmtree(vs, ignore_errors=True)
    png = b"\x89PNG\r\n\x1a\n"
    pick = [t for t in plan["tasks"]][:12]
    for t in pick:
        chars = K.task_chars(plan, t)
        K.write_shard(vs, t, plan, {c: png + (str(ord(c)) * 4).encode() for c in chars}, {})
    (vs / "plan_sha.txt").write_text(plan["sha"], encoding="utf-8")
    (vs / ".gitattributes").write_text("*.tar filter=lfs\n", encoding="utf-8")
    (vs / ".cache" / "huggingface").mkdir(parents=True)
    (vs / ".cache" / "huggingface" / "x").write_text("x")

    def run_validate():
        p = subprocess.run([sys.executable, str(RUNNER), "validate", "--pack", str(PACK), "--out", str(vs)], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        try:
            return p.returncode, json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            return p.returncode, dict(raw=(p.stdout + p.stderr)[-300:])

    rc, v = run_validate()
    R["validate_thu_muc_gia_lap_12_viec"] = dict(exit=rc, ket_qua={k: v.get(k) for k in ("tasks", "present", "missing", "images", "n_errors", "uncovered_no_font", "skipped_by_reason")})
    inv("V1.validate_chay_tren_cau_truc_tai_ve_n_errors_0_exit_0", rc == 0 and v.get("n_errors") == 0 and v.get("present") == 12 and v.get("missing") == 398 and v.get("uncovered_no_font") == {"stt11": 1}, R["validate_thu_muc_gia_lap_12_viec"])
    # làm hỏng một tar: cắt cụt
    victim = vs / K.rel_path(pick[3]["id"])
    victim.write_bytes(victim.read_bytes()[:3000])
    rc2, v2 = run_validate()
    inv("V2.validate_bat_tar_bi_cat_cut_n_errors_lon_hon_0_exit_1", rc2 == 1 and v2.get("n_errors", 0) >= 1, dict(exit=rc2, n_errors=v2.get("n_errors"), errors=(v2.get("errors") or [])[:1]))
    inv("V3.exit_0_ke_ca_khi_con_thieu_398_viec_nen_phai_kiem_missing_bang_0_de_biet_da_xong", True, "exit 0 với missing=398 (V1)")
    shutil.rmtree(vs, ignore_errors=True)

    # ---- 7. đường dẫn nêu trong hướng dẫn
    toks = sorted(set(re.findall(r"`([^`\n]+)`", g)))
    pathlike = [t for t in toks if re.search(r"\.(py|sh|json|md|zip|csv|txt|tsv)\b", t) and " " not in t.strip() and not t.startswith("!")]
    bases = [REPO, HERE, FULL]
    res = {}
    for t in pathlike:
        if "<" in t:
            res[t] = "mẫu (có <…>)"
            continue
        if t.startswith("/kaggle") or t.startswith("shards/") or t in ("status.json", "env.json", "plan_sha.txt", "labels.csv", "tasks-w*.jsonl", "plan.json"):
            res[t] = "đường dẫn chạy-thời-gian/ngoài Mac"
            continue
        hit = next((str(b / t) for b in bases if (b / t).exists()), None)
        res[t] = "CÓ" if hit else "THIẾU"
    R["duong_dan_trong_huong_dan"] = res
    miss = {k: v for k, v in res.items() if v == "THIẾU"}
    inv("D1.duong_dan_trong_huong_dan_deu_co_tru_cac_muc_neu_ben_duoi", set(miss) <= {"p12_cham_toan_bo.py", "measure_out/_tn11/full/kaggle_ket_qua"}, miss)
    inv("D2.p12_cham_toan_bo.py_ton_tai", any(p.name == "p12_cham_toan_bo.py" for p in HERE.parent.rglob("p12_*.py")), "tìm p12_*.py dưới lab/thu_nghiem_anh_chu/")
    tap = {d.name: (d / "TAP_DANH_GIA.md").exists() for d in sorted((REPO / "dataset").iterdir()) if d.is_dir() and d.name in ("LucVanTien1916", "TruyenKieu1872", "SachKinhThayCaBinh", "SachDungLyHoThan")}
    inv("D3.TAP_DANH_GIA_ton_tai_cho_ca_4_bo_nhan_nguoi", len(tap) == 4 and all(tap.values()), tap)
    inv("D4.Glyphs_khong_o_pipeline/chon_chu/features.py", "class Glyphs" not in (REPO / "pipeline/chon_chu/features.py").read_text(encoding="utf-8")
        and "class Glyphs" in (REPO / "pipeline/gold_exact/signals_img.py").read_text(encoding="utf-8"), "Glyphs ở pipeline/gold_exact/signals_img.py:285 và pipeline/borg_human/encoders.py:81")

    R["tong"] = dict(pass_=sum(v["ok"] for v in R["invariants"].values()), fail=sum(not v["ok"] for v in R["invariants"].values()), n=len(R["invariants"]),
                     fail_ten=[k for k, v in R["invariants"].items() if not v["ok"]])
    (OUT / "02_lenh_va_duong_dan.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("TỔNG:", R["tong"])


if __name__ == "__main__":
    main()
