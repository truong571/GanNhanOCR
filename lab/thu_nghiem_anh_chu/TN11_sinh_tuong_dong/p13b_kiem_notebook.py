"""TN11 p13b (04/10) — CHẠY THẬT notebook Kaggle (p13) trong IPython kernel với Kaggle/HF GIẢ (stub sinh ảnh, không GPU, không mạng).
Chứng minh phần keo của notebook: `!python {RUNNER} … {extra}` nở đúng, preflight chặn đúng, token không lộ ra màn hình, màn hình gọn, HF nhận đủ tar.
Cần nbclient+ipykernel → chạy bằng một venv RIÊNG (không đụng .venv của dự án):
    python3 -m venv <dir> && <dir>/bin/pip install -q ipykernel nbclient nbformat
    <dir>/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p13b_kiem_notebook.py --nb measure_out/_tn11/full/kaggle_notebook/tn11_full_kaggle.ipynb
Ghi vào measure_out/_tn11/full/kiem_ban_va/nb_<giờ>/ (thư mục mới mỗi lần). Thoát 1 nếu có FAIL.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out/_tn11/full/kiem_ban_va"
TOKEN = "hf_FAKE_khong_duoc_lo_ra_man_hinh_123"
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


def load_runner():
    spec = importlib.util.spec_from_file_location("K", HERE / "w_runner_khung_chay.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["K"] = m
    spec.loader.exec_module(m)
    return m


def mini_pack(K, d: Path, chars, shard=4):
    import shutil
    d.mkdir(parents=True, exist_ok=True)
    plan = K.make_plan({"X": dict(chars=list(chars), styles=1)}, [("base", 0)], shard, fonts=None)
    (d / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    (d / "ckpt").mkdir(exist_ok=True)
    (d / "fonts.json").write_text("{}", encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps(dict(plan_sha=plan["sha"], ckpt_sha256={}, assets_sha=K.compute_assets_sha(d, {})), ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(HERE / "w_runner_khung_chay.py", d / "w_runner_khung_chay.py")
    return plan


FAKE_HF = r'''import os, shutil
from pathlib import Path
from .errors import EntryNotFoundError
ROOT = Path(os.environ["FAKE_HF_ROOT"])
def _r(repo_id): return ROOT / repo_id.replace("/", "__")
def get_token(): return None
class HfApi:
    def __init__(self, token=None): self.token = token
    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False): _r(repo_id).mkdir(parents=True, exist_ok=True)
    def whoami(self): return {"name": "fakeuser"}
    def list_repo_files(self, repo_id, repo_type=None): return [p.relative_to(_r(repo_id)).as_posix() for p in _r(repo_id).rglob("*") if p.is_file()]
    def upload_folder(self, folder_path, repo_id, repo_type=None, allow_patterns=None, commit_message=None):
        n = 0
        for rel in allow_patterns or []:
            src = Path(folder_path) / rel
            if src.is_file():
                dst = _r(repo_id) / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst); n += 1
        with open(ROOT / "commits.log", "a") as f: f.write(f"{repo_id}\t{n}\t{commit_message}\n")
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None):
        dst = _r(repo_id) / path_in_repo; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path_or_fileobj if isinstance(path_or_fileobj, bytes) else path_or_fileobj.read())
def hf_hub_download(repo_id, filename, repo_type=None, token=None):
    f = _r(repo_id) / filename
    if not f.exists(): raise EntryNotFoundError(filename)
    return str(f)
'''


def scenario(nb_path: Path, W: Path, name: str, *, secret_ok=True, pre_repo=None, test_flag=True, n_chars=16, copy_repo=None, zip_pack=False):
    """Chạy notebook (đã thay đường dẫn /kaggle/* và cờ stub) trong kernel; trả (ô_lỗi, văn bản các đầu ra, repo_dir)."""
    import nbclient
    import nbformat
    K = load_runner()
    S = W / name
    (S / "input").mkdir(parents=True)
    for d in ("tmp", "working", "hf_root"):
        (S / d).mkdir()
    plan = mini_pack(K, S / "input" / "gói", chars="".join(chr(0x4E00 + i) for i in range(n_chars)))
    if zip_pack:                                                   # Dataset chưa giải nén: chỉ còn tệp zip trong input
        import shutil
        import zipfile
        pk = S / "input" / "gói"
        with zipfile.ZipFile(S / "input" / "gói.zip", "w") as z:
            for f in sorted(pk.rglob("*")):                       # cả thư mục (ckpt/ rỗng của gói mini) lẫn tệp
                z.write(f, f.relative_to(pk).as_posix())
        shutil.rmtree(pk)
    fake = S / "fake"
    (fake / "huggingface_hub").mkdir(parents=True)
    (fake / "huggingface_hub" / "errors.py").write_text("class EntryNotFoundError(Exception):\n    pass\n", encoding="utf-8")
    (fake / "huggingface_hub" / "__init__.py").write_text(FAKE_HF, encoding="utf-8")
    ks = (f"class UserSecretsClient:\n    def get_secret(self, k):\n        assert k == 'HF_TOKEN'\n        return {TOKEN!r}\n" if secret_ok
          else "class UserSecretsClient:\n    def get_secret(self, k):\n        raise RuntimeError('secret chưa đính')\n")
    (fake / "kaggle_secrets.py").write_text(ks, encoding="utf-8")
    repo_dir = S / "hf_root" / "mdnt571__gannhanocr-tn11-full"
    if copy_repo:
        import shutil
        shutil.copytree(copy_repo, repo_dir)
    if pre_repo:
        repo_dir.mkdir(parents=True, exist_ok=True)
        for k, v in pre_repo.items():
            (repo_dir / k).write_text(v, encoding="utf-8")
    nb = nbformat.read(str(nb_path), as_version=4)
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        s = c.source
        s = s.replace("/kaggle/input/**/", f"{S}/input/**/").replace("/kaggle/tmp", f"{S}/tmp").replace("/kaggle/working", f"{S}/working")
        s = s.replace('"--device", "cuda"]', '"--stub", "--tmp", ' + repr(str(S / "tmp")) + "]").replace("--device cuda", "--stub --stub-ms 40")
        s = re.sub(r"^TEST = (True|False)", f"TEST = {test_flag}", s, flags=re.M)
        c.source = s
    nb.cells.append(nbformat.v4.new_code_cell("!python -c \"import os; print('ENV', os.environ.get('HF_HUB_DISABLE_PROGRESS_BARS'), os.environ.get('PYGAME_HIDE_SUPPORT_PROMPT'), os.environ.get('PYTHONWARNINGS'))\""))
    old = {k: os.environ.get(k) for k in ("PYTHONPATH", "FAKE_HF_ROOT", "PYTHONDONTWRITEBYTECODE")}
    os.environ.update(PYTHONPATH=str(fake) + os.pathsep + os.environ.get("PYTHONPATH", ""), FAKE_HF_ROOT=str(S / "hf_root"), PYTHONDONTWRITEBYTECODE="1")
    try:
        cl = nbclient.NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=True, resources={"metadata": {"path": str(S)}})
        cl.execute()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    text, errs = [], []
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []):
            if o.output_type == "stream":
                text.append(o.text)
            elif o.output_type == "error":
                errs.append(i)
                text.append(f"[ERROR cell {i}] {o.ename}: {o.evalue}")
            elif o.output_type in ("execute_result", "display_data"):
                text.append(str(o.get("data", {}).get("text/plain", "")))
    return errs, "\n".join(text), repo_dir, plan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", default=str(REPO / "measure_out/_tn11/full/kaggle_notebook/tn11_full_kaggle.ipynb"))
    a = ap.parse_args(argv)
    W = OUT / f"nb_{time.strftime('%H%M%S')}"
    W.mkdir(parents=True)
    nbp = Path(a.nb)

    # S1: ca thử đủ luồng
    errs, text, repo, plan = scenario(nbp, W, "s1_ca_thu")
    n_tar = len(list((repo / "shards").rglob("*.tar"))) if repo.exists() else 0
    check("N1.chay_het_khong_loi_o", not errs, f"ô lỗi: {errs}" + ("" if not errs else " | " + text[-400:].replace("\n", " ⏎ ")))
    check("N2.preflight_DAT_va_status_DA_XONG", "PREFLIGHT ĐẠT" in text and "[status] ĐÃ XONG" in text, "")
    check("N3.HF_nhan_du_tar_va_dau_sha", n_tar == len(plan["tasks"]) and (repo / "plan_sha.txt").read_text().strip() == plan["sha"] and (repo / "assets_sha.txt").exists(), f"tar={n_tar}/{len(plan['tasks'])}")
    check("N4.token_khong_lo_ra_man_hinh", TOKEN not in text, "")
    n_lines = len([ln for ln in text.splitlines() if ln.strip()])
    check("N5.man_hinh_gon", n_lines <= 40, f"{n_lines} dòng")
    check("N5b.moi_truong_gon_duoc_ke_thua_boi_tien_trinh_con", "ENV 1 1 ignore:pkg_resources is deprecated:UserWarning" in text, "")
    check("N6.co_dong_env_run_va_worker", "[env]" in text and "[run] {" in text and "[w0] [worker w0]" in text, "")      # [tiến độ] chỉ in mỗi 600 s: ca stub xong sớm hơn nên không có
    print("---- màn hình ca thử (rút gọn) ----\n" + "\n".join(ln[:160] for ln in text.splitlines()[:14]), flush=True)

    # S2: phiên Kaggle sau (cùng kho HF) -> không làm lại việc nào
    errs2, text2, repo2, _ = scenario(nbp, W, "s2_chay_tiep", copy_repo=repo)
    check("N6b.chay_tiep_bo_viec_da_co_o_HF", not errs2 and f"xong ở kho ngoài {len(plan['tasks'])} | còn 0" in text2 and "[status] ĐÃ XONG" in text2, f"errs={errs2}")
    # S3: kho HF chứa kế hoạch KHÁC -> preflight phải chặn trước khi chạy
    errs3, text3, repo3, _ = scenario(nbp, W, "s3_kho_khac_ke_hoach", pre_repo={"plan_sha.txt": "0" * 64})
    check("N7.kho_khac_ke_hoach_bi_chan_o_preflight", errs3 == [1] and "PREFLIGHT KHÔNG ĐẠT" in text3 + "\n" and "KHO CHỨA KẾ HOẠCH/TÀI SẢN KHÁC" in text3 and not list((repo3 / "shards").rglob("*.tar")), f"errs={errs3}")
    # S4: quên đính secret
    errs4, text4, _, _ = scenario(nbp, W, "s4_thieu_secret", secret_ok=False)
    check("N8.thieu_secret_bao_ro_va_dung", errs4 == [1] and "Chưa đính secret HF_TOKEN" in text4, f"errs={errs4}")
    # S5: ca thật (TEST=False) với 40 chữ → 10 việc
    errs5, text5, repo5, plan5 = scenario(nbp, W, "s5_ca_that", test_flag=False, n_chars=40)
    n5 = len(list((repo5 / "shards").rglob("*.tar"))) if repo5.exists() else 0
    check("N9.ca_that_TEST_False_xong_het", not errs5 and n5 == len(plan5["tasks"]) and "[status] ĐÃ XONG" in text5, f"errs={errs5} tar={n5}/{len(plan5['tasks'])}")

    # S6: Dataset còn là tệp zip -> notebook tự giải nén rồi chạy
    errs6, text6, repo6, plan6 = scenario(nbp, W, "s6_zip_chua_giai_nen", zip_pack=True)
    n6 = len(list((repo6 / "shards").rglob("*.tar"))) if repo6.exists() else 0
    check("N10.dataset_la_zip_tu_giai_nen", not errs6 and n6 == len(plan6["tasks"]) and "[status] ĐÃ XONG" in text6, f"errs={errs6} tar={n6}/{len(plan6['tasks'])}")

    nf = [n for n, ok in RES if not ok]
    print(f"TỔNG: pass={len(RES) - len(nf)} fail={len(nf)} n={len(RES)} · {W}")
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
