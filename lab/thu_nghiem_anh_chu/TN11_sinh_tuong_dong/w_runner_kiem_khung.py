"""TN11 "full" — hướng RUNNER/KAGGLE: KIỂM THỬ CPU cho w_runner_khung_chay.py bằng bộ sinh GIẢ (0 GPU, 0 API, 0 mạng).

Mỗi thử nghiệm kiểm một tính chất thiết kế có `invariants` PASS/FAIL (quy ước của repo: không dùng LLM nhìn kết quả):
  T1 mỗi việc đúng một lần + cân bằng tải 2 "GPU" (hàng đợi động) so với chia tĩnh theo cuốn + thứ tự xen kẽ giữa các cuốn
  T2 ca chết sau khi nhận việc (khoá + .part còn lại) -> chạy lại làm nốt đúng phần thiếu, byte giống bản tham chiếu
  T2b ca chết giữa lô (os._exit trong bộ sinh) -> như T2
  T3 ngân sách giờ: dừng SẠCH (mã 0), không nhận việc mới khi không kịp; chạy lại hoàn tất, không việc nào làm hai lần
  T4 kho ngoài (thư mục): mất sạch đĩa cục bộ (ca mới) -> chỉ làm phần chưa có; xoá 25 % kho -> làm lại đúng phần mất, byte giống
  T5 kế hoạch khác với kho ngoài -> từ chối (mã 2), không ghi gì
  T6 chữ bị bỏ qua được ghi rõ (yêu cầu = sinh + bỏ qua), validate sạch
  T7 kế hoạch thật B34/B18 với chuỗi phông: số chữ phủ/không phủ, sha ổn định
  T8 HFRemote: chữ ký hàm khớp huggingface_hub cài sẵn + chạy đủ đường với HfApi giả (không mạng)
  T9 cổng use_fst: lớp giả đúng tên FontDiffuserModelDPM / ...WithFST + logger bị hạ mức ERROR vẫn bắt được cảnh báo

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_runner_kiem_khung.py
Ra: measure_out/_tn11/full/runner/kiem_khung.json (+ sim/ là thư mục làm việc, ghi đè mỗi lần)
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
import w_runner_khung_chay as K  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "full" / "runner"
SIM = OUT / "sim"
RUNNER = HERE / "w_runner_khung_chay.py"
NCHARS = {"stt2": 900, "stt4": 1300, "stt11": 700, "Chr": 2100, "L83": 500, "KVK": 2700, "L16": 800, "TK": 1500, "B18": 2300, "B34": 600}
R = {"tao": time.strftime("%Y-%m-%d %H:%M:%S"), "invariants": {}}


def inv(name: str, ok: bool, detail=None):
    R["invariants"][name] = dict(ok=bool(ok), detail=detail)
    print(("PASS" if ok else "FAIL"), name, "" if detail is None else json.dumps(detail, ensure_ascii=False, default=str)[:300], flush=True)


def mk_pack(d: Path, nchars: dict, passes=(("base", 0), ("base", 1)), shard=128, mutate=False) -> dict:
    books = {}
    for i, (b, n) in enumerate(nchars.items()):
        base = 0x4E00 + i * 3000
        ch = [chr(base + k) for k in range(n)]
        if mutate and i == 0:
            ch = ch[::-1]
        books[b] = dict(chars=ch, styles=2)
    plan = K.make_plan(books, list(passes), shard, fonts=None)
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    for b in books:
        for j in range(2):
            (d / "data" / b).mkdir(parents=True, exist_ok=True)
            (d / "data" / b / f"style_{j}.png").write_bytes(b"fake")
    return plan


def run(pack: Path, out: Path, *extra, remote=None, workers=2, ttl="600", budget="1", safety="0", ms="0.4", est="0.2"):
    cmd = [sys.executable, str(RUNNER), "run", "--pack", str(pack), "--out", str(out), "--stub", "--workers", str(workers), "--ttl", ttl,
           "--budget-hours", budget, "--safety-s", safety, "--stub-ms", ms, "--est-s", est, "--sync-every", "0.5"]
    if remote:
        cmd += ["--remote", remote]
    cmd += list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def shard_hashes(out: Path) -> dict:
    return {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "shards").rglob("*.tar")}


def jsonl_ids(out: Path) -> list:
    ids = []
    for f in (out / "status").glob("tasks-*.jsonl"):
        ids += [json.loads(l)["id"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return ids


def status(out: Path) -> dict:
    return json.loads((out / "status" / "status.json").read_text(encoding="utf-8"))


def leftover(out: Path) -> dict:
    return dict(locks=len(list((out / "locks").glob("*.lock"))) if (out / "locks").exists() else 0,
                parts=len(list((out / "shards").rglob("*.part"))) if (out / "shards").exists() else 0)


def validate(pack: Path, out: Path) -> tuple:
    p = subprocess.run([sys.executable, str(RUNNER), "validate", "--pack", str(pack), "--out", str(out)], capture_output=True, text=True)
    try:
        return p.returncode, json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return p.returncode, p.stdout + p.stderr


def main():
    if SIM.exists():
        shutil.rmtree(SIM)
    SIM.mkdir(parents=True)
    pack = SIM / "pack"
    plan = mk_pack(pack, NCHARS)
    tasks = plan["tasks"]; ids = [t["id"] for t in tasks]
    R["ke_hoach"] = dict(books=len(NCHARS), chars=sum(NCHARS.values()), tasks=len(tasks), shard=plan["shard"], sha=plan["sha"][:12])

    # ---------------- T1
    ref = SIM / "ref"
    t0 = time.time()
    code, log = run(pack, ref)
    st = status(ref)
    inv("T1.exit0_va_complete", code == 0 and st["complete"], dict(code=code, tasks=st["tasks_done"], total=st["tasks_total"], giay=round(time.time() - t0, 1)))
    jl = jsonl_ids(ref)
    inv("T1.moi_viec_dung_mot_lan", sorted(jl) == sorted(ids) and max(Counter(jl).values()) == 1, dict(logged=len(jl), unique=len(set(jl))))
    vc, vr = validate(pack, ref)
    inv("T1.validate_sach", vc == 0 and vr["present"] == len(tasks) and vr["n_errors"] == 0 and vr["images"] == sum(NCHARS.values()) * 2 and sum(b["images"] for b in vr["per_book"].values()) == vr["images"], vr)
    inv("T1.khong_con_khoa_hay_part", leftover(ref) == dict(locks=0, parts=0), leftover(ref))
    per = {}
    for f in (ref / "status").glob("tasks-w*.jsonl"):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        per[f.stem.split("-")[1]] = dict(tasks=len(rows), images=sum(r["n_gen"] for r in rows), sec=round(sum(r["sec"] for r in rows), 2))
    imgs = [v["images"] for v in per.values()]
    total_imgs = sum(imgs)
    # chia tĩnh theo cuốn như tn11_kaggle.py:245 (cuốn thứ k -> GPU k % 2), tải = số ảnh của cuốn (2 lượt phong cách)
    stat_load = [0, 0]
    for k, (b, n) in enumerate(NCHARS.items()):
        stat_load[k % 2] += n * 2
    inv("T1.can_bang_dong_tot_hon_tinh", abs(imgs[0] - imgs[1]) < abs(stat_load[0] - stat_load[1]) and abs(imgs[0] - imgs[1]) <= 2 * plan["shard"],
        dict(dong=per, chenh_dong=abs(imgs[0] - imgs[1]), chenh_tinh=abs(stat_load[0] - stat_load[1]), tai_tinh=stat_load,
             makespan_tinh_tren_ly_tuong=round(max(stat_load) / (sum(stat_load) / 2), 3), makespan_dong_tren_ly_tuong=round(max(imgs) / (total_imgs / 2), 3)))
    # xen kẽ: sau 20 % số việc đầu, mọi cuốn đều đã có ít nhất một lô được xếp
    k20 = int(len(tasks) * 0.2)
    books_seen = {t["book"] for t in tasks[:k20]}
    inv("T1.xen_ke_moi_cuon_trong_20pct_dau", len(books_seen) == len(NCHARS), dict(books_seen=len(books_seen), tasks_first=k20))
    ref_h = shard_hashes(ref)

    # ---------------- T2 (chết sau khi nhận việc, để lại khoá + .part)
    o2 = SIM / "o2"
    code, log = run(pack, o2, "--crash-after", "5")
    st1 = status(o2); lf = leftover(o2)
    inv("T2.ca_1_chua_xong_va_con_dau_vet", code == 0 and not st1["complete"] and lf["locks"] >= 1 and lf["parts"] >= 1,
        dict(code=code, done=st1["tasks_done"], total=st1["tasks_total"], left=lf, codes=st1["worker_exit_codes"]))
    n1 = len(jsonl_ids(o2))
    time.sleep(1.2)
    code, log = run(pack, o2, ttl="0.5")
    st2 = status(o2)
    ids2 = jsonl_ids(o2)
    inv("T2.ca_2_hoan_tat_dung_phan_thieu", code == 0 and st2["complete"] and len(ids2) - n1 == st1["tasks_total"] - st1["tasks_done"] and max(Counter(ids2).values()) == 1,
        dict(thieu_truoc=st1["tasks_total"] - st1["tasks_done"], lam_them=len(ids2) - n1))
    inv("T2.sach_dau_vet_va_byte_giong_tham_chieu", leftover(o2) == dict(locks=0, parts=0) and shard_hashes(o2) == ref_h, dict(left=leftover(o2)))

    # ---------------- T2b (chết giữa lô)
    o2b = SIM / "o2b"
    mid_char = chr(0x4E00 + 5 * 3000 + 1000)       # một chữ của cuốn KVK nằm sâu bên trong
    code, log = run(pack, o2b, "--stub-crash", mid_char)
    s1 = status(o2b)
    inv("T2b.ca_1_mot_worker_chet_giua_lo", not s1["complete"] and 23 in s1["worker_exit_codes"],
        dict(codes=s1["worker_exit_codes"], thieu=s1["tasks_total"] - s1["tasks_done"], left=leftover(o2b)))
    time.sleep(1.2)
    code, log = run(pack, o2b, "--stub-crash", mid_char, ttl="0.5")      # cờ crash_once đã có => không chết nữa
    inv("T2b.ca_2_hoan_tat_byte_giong", status(o2b)["complete"] and shard_hashes(o2b) == ref_h, dict(left=leftover(o2b)))

    # ---------------- T3 ngân sách giờ
    o3 = SIM / "o3"
    code, log = run(pack, o3, budget="0.0009", ms="1.5", est="0.15")           # 3,24 s, mỗi việc ≈ 0,19 s
    s3 = status(o3)
    rows3 = [json.loads(l) for f in (o3 / "status").glob("tasks-w*.jsonl") for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    ends = [r["t"] for r in rows3]
    max_task = max(r["sec"] for r in rows3)
    inv("T3.dung_sach_ma0_khong_vuot_han", code == 0 and not s3["complete"] and s3["stop_reason"] == "budget" and max(ends) <= s3["deadline_epoch"] + max_task + 0.05,
        dict(code=code, done=s3["tasks_done"], total=s3["tasks_total"], stop=s3["worker_stop"], tre_nhat_giay=round(max(ends) - s3["deadline_epoch"], 3), max_task_s=round(max_task, 3)))
    n3 = len(jsonl_ids(o3))
    code, log = run(pack, o3)
    ids3 = jsonl_ids(o3)
    inv("T3.chay_lai_hoan_tat_khong_lam_hai_lan", status(o3)["complete"] and max(Counter(ids3).values()) == 1 and len(ids3) == len(tasks) and shard_hashes(o3) == ref_h,
        dict(truoc=n3, sau=len(ids3)))

    # ---------------- T4 kho ngoài + mất đĩa
    rem = SIM / "remote"
    a1 = SIM / "a1"
    code, log = run(pack, a1, remote=f"dir:{rem}", budget="0.0009", ms="1.5", est="0.15")
    done_a1 = K.DirRemote(str(rem)).list_done()
    inv("T4.ca_A_day_len_kho_ngoai", len(done_a1) > 0 and len(done_a1) == len(K.done_ids(a1)) and (rem / "plan_sha.txt").exists(), dict(remote=len(done_a1), total=len(tasks)))
    a2 = SIM / "a2"                                                                  # ca MỚI, đĩa trống
    code, log = run(pack, a2, remote=f"dir:{rem}", ms="0.4")
    ids_a2 = set(jsonl_ids(a2))
    inv("T4.ca_B_chi_lam_phan_chua_co", status(a2)["complete"] and not (ids_a2 & done_a1) and len(ids_a2) + len(done_a1) == len(tasks), dict(da_co=len(done_a1), lam_them=len(ids_a2)))
    inv("T4.kho_ngoai_day_du_va_byte_giong", K.DirRemote(str(rem)).list_done() == set(ids) and shard_hashes(rem) == ref_h, dict(files=len(shard_hashes(rem))))
    lost = sorted(K.DirRemote(str(rem)).list_done())[::4]
    for tid in lost:
        (rem / K.rel_path(tid)).unlink()
    a3 = SIM / "a3"
    code, log = run(pack, a3, remote=f"dir:{rem}")
    ids_a3 = set(jsonl_ids(a3))
    inv("T4.mat_25pct_kho_lam_lai_dung_phan_mat", ids_a3 == set(lost) and shard_hashes(rem) == ref_h, dict(mat=len(lost), lam_lai=len(ids_a3)))

    # ---------------- T5 kế hoạch khác
    pack5 = SIM / "pack5"
    mk_pack(pack5, NCHARS, mutate=True)
    a5 = SIM / "a5"
    code, log = run(pack5, a5, remote=f"dir:{rem}")
    inv("T5.ke_hoach_khac_bi_tu_choi_ma2_khong_ghi", code == 2 and not (a5 / "shards").exists() and "KẾ HOẠCH KHÁC" in log, dict(code=code))

    # ---------------- T6 chữ bị bỏ qua
    o6 = SIM / "o6"
    bad = "".join(chr(0x4E00 + 3 * 3000 + k) for k in (3, 4, 200))               # 3 chữ của cuốn Chr
    code, log = run(pack, o6, "--stub-fail", bad)
    vc, vr = validate(pack, o6)
    sk = 0
    import tarfile
    for p in (o6 / "shards" / "Chr").rglob("*.tar"):
        with tarfile.open(p) as tf:
            sk += len(json.loads(tf.extractfile("_meta.json").read())["skipped"])
    inv("T6.bo_qua_duoc_ghi_va_validate_sach", vc == 0 and sk == 6, dict(skipped_total_Chr=sk, ky_vong="3 chữ × 2 lượt phong cách", validate=vr))

    # ---------------- T7 kế hoạch thật với chuỗi phông
    import pandas as pd
    chain = [("NomNaTong", "font_diffusion/fonts/NomNaTong-Regular.ttf"), ("HanNomA", "font_diffusion/fonts/HAN NOM A.ttf"), ("HanNomB", "font_diffusion/fonts/HAN NOM B.ttf"),
             ("HanaMinA", "font_diffusion/fonts/HanaMinA.ttf"), ("HanaMinB", "font_diffusion/fonts/HanaMinB.ttf"), ("HanaMinC", "font_diffusion/fonts/HanaMinC.otf")]
    from fontTools.ttLib import TTFont
    fonts = {}
    for n, p in chain:
        f = TTFont(str(REPO / p), lazy=True); s = set()
        for st_ in f["cmap"].tables:
            s |= set(st_.cmap)
        fonts[n] = s
    books = {}
    for b in ("B34", "B18"):
        P = pd.read_pickle(REPO / "measure_out" / "_tn11" / f"p02_{b}" / "plan.pkl")
        books[b] = dict(chars=list(P["chars"]), styles=2)
    p1 = K.make_plan(books, [("base", 0), ("base", 1)], 128, fonts)
    p2 = K.make_plan(books, [("base", 0), ("base", 1)], 128, fonts)
    cov = {b: dict(yeu_cau=len(books[b]["chars"]), phu=len(p1["books"][b]["chars"]), khong_phong_nao=len(p1["books"][b]["uncovered"]),
                   dung_NomNaTong=sum(1 for i in p1["books"][b]["font_idx"] if i == 0)) for b in books}
    inv("T7.ke_hoach_that_B34_B18", p1["sha"] == p2["sha"] and cov["B34"]["yeu_cau"] == 1019 and cov["B34"]["dung_NomNaTong"] == 793 and cov["B18"]["dung_NomNaTong"] == 685
        and cov["B34"]["khong_phong_nao"] == 2 and cov["B18"]["khong_phong_nao"] == 2,
        dict(cov=cov, tasks=len(p1["tasks"]), plan_json_KB=round(len(json.dumps(p1, ensure_ascii=False).encode()) / 1024, 1), sha=p1["sha"][:12]))

    # ---------------- T8 HFRemote
    import inspect
    import huggingface_hub as hh
    sig_ok, detail = True, {}
    need = {"upload_folder": ["folder_path", "repo_id", "repo_type", "allow_patterns", "commit_message"], "list_repo_files": ["repo_id", "repo_type"],
            "create_repo": ["repo_id", "repo_type", "private", "exist_ok"], "upload_file": ["path_or_fileobj", "path_in_repo", "repo_id", "repo_type"]}
    for fn, args in need.items():
        ps = inspect.signature(getattr(hh.HfApi, fn)).parameters
        miss = [a for a in args if a not in ps]
        detail[fn] = miss
        sig_ok &= not miss
    ps = inspect.signature(hh.hf_hub_download).parameters
    miss = [a for a in ("repo_id", "filename", "repo_type", "token") if a not in ps]
    detail["hf_hub_download"] = miss; sig_ok &= not miss
    try:
        from huggingface_hub.errors import EntryNotFoundError  # noqa: F401
    except Exception as e:  # noqa: BLE001
        sig_ok = False; detail["EntryNotFoundError"] = str(e)
    detail["huggingface_hub"] = hh.__version__
    inv("T8.chu_ky_ham_khop_huggingface_hub_cai_san", sig_ok, detail)

    fake_root = SIM / "fakehf"
    fake_root.mkdir()

    class FakeApi:
        def __init__(self, token=None):
            self.calls = []

        def create_repo(self, repo_id, repo_type, private, exist_ok):
            self.calls.append(("create_repo", repo_id, repo_type, private, exist_ok))

        def list_repo_files(self, repo_id, repo_type):
            return [p.relative_to(fake_root).as_posix() for p in fake_root.rglob("*") if p.is_file()]

        def upload_folder(self, folder_path, repo_id, repo_type, allow_patterns, commit_message):
            for r in allow_patterns:
                d = fake_root / r; d.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(Path(folder_path) / r, d)

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
            d = fake_root / path_in_repo; d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(path_or_fileobj)

    real_api, real_dl = hh.HfApi, hh.hf_hub_download

    def fake_dl(repo_id, filename, repo_type, token):
        from huggingface_hub.errors import EntryNotFoundError
        p = fake_root / filename
        if not p.exists():
            raise EntryNotFoundError("không có")
        return str(p)
    hh.HfApi, hh.hf_hub_download = FakeApi, fake_dl
    try:
        import os
        os.environ["HF_TOKEN"] = "x"
        hf_out = SIM / "hf_out"
        t_hf0 = time.time()
        try:
            K.main(["run", "--pack", str(pack), "--out", str(hf_out), "--stub", "--workers", "2", "--ttl", "600", "--budget-hours", "1", "--safety-s", "0",
                    "--stub-ms", "0.4", "--est-s", "0.2", "--sync-every", "0.5", "--remote", "hf:fake/tn11"])
            hf_code = None
        except SystemExit as e:
            hf_code = e.code
    finally:
        hh.HfApi, hh.hf_hub_download = real_api, real_dl
    remote_files = {p.relative_to(fake_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in fake_root.rglob("*.tar")}
    inv("T8.hfremote_day_du_voi_HfApi_gia", hf_code in (0, None) and remote_files == ref_h and (fake_root / "plan_sha.txt").read_text() == plan["sha"],
        dict(code=hf_code, files=len(remote_files), giay=round(time.time() - t_hf0, 1)))

    # ---------------- T9 cổng use_fst bằng lớp giả
    class FontDiffuserModelDPM:         # đúng tên lớp chuẩn
        pass

    class FontDiffuserModelDPMWithFST:  # đúng tên lớp lỗi
        fst_module = object(); mss_encoder = object()

    class G:
        def __init__(self, m):
            self.pipe = type("P", (), {"model": m})()
    ok1 = K.assert_standard_path(G(FontDiffuserModelDPM()), [], raise_error=False)["ok"]
    r2 = K.assert_standard_path(G(FontDiffuserModelDPMWithFST()), [], raise_error=False)
    try:
        K.assert_standard_path(G(FontDiffuserModelDPMWithFST()), [])
        raised = False
    except K.FstGuardError:
        raised = True
    lg = logging.getLogger("src.builders.build")
    lg.setLevel(logging.ERROR)          # như notebook cũ: cảnh báo bị nuốt
    with K.FstWarningCapture() as cap:
        lg.warning("⚠ Checkpoint for 'fst_module' not found in /x")
    lvl_back = lg.level == logging.ERROR
    r3 = K.assert_standard_path(G(FontDiffuserModelDPM()), cap.msgs, raise_error=False)
    inv("T9.cong_use_fst_logic", ok1 and not r2["ok"] and raised and lvl_back and not r3["ok"] and len(cap.msgs) == 1,
        dict(chuan=ok1, loi_lop=r2["model_class"], raised=raised, bat_duoc_canh_bao_du_logger_ERROR=len(cap.msgs), muc_logger_khoi_phuc=lvl_back))

    # ---------------- T10 chạy thử nhanh (--max-tasks) cho 15 phút đầu trên Kaggle
    o10 = SIM / "o10"
    code, log = run(pack, o10, "--max-tasks", "3")
    s10 = status(o10)
    inv("T10.max_tasks_thoat_sach", code == 0 and s10["stop_reason"] == "max_tasks" and s10["tasks_done_local"] == 6 and set(s10["worker_stop"].values()) == {"max_tasks"},
        dict(code=code, done=s10["tasks_done_local"], stop=s10["worker_stop"]))

    # ---------------- T11 dựng gói thật (không ckpt) + kiểm gói + chạy bộ sinh giả trên gói
    spec_dir = SIM / "spec"
    spec_dir.mkdir()
    spec = {"fonts": {n: p for n, p in chain}, "books": {}}
    for b in ("B34", "B18"):
        P = pd.read_pickle(REPO / "measure_out" / "_tn11" / f"p02_{b}" / "plan.pkl")
        (spec_dir / f"chars_{b}.txt").write_text("\n".join(P["chars"]), encoding="utf-8")
        spec["books"][b] = dict(chars_file=str(spec_dir / f"chars_{b}.txt"), style_pngs=[str(REPO / "measure_out" / "_tn11" / f"p02_{b}" / f"style_{j}.png") for j in (0, 1)])
    (spec_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    pd_dir = SIM / "pack_demo"
    pp = subprocess.run([sys.executable, str(RUNNER), "pack", "--spec", str(spec_dir / "spec.json"), "--out", str(pd_dir), "--no-ckpt", "--repo", str(REPO)], capture_output=True, text=True)
    au = K.audit_pack(pd_dir) if pd_dir.exists() else {}
    inv("T11.dong_goi_va_kiem_goi", pp.returncode == 0 and au.get("sach") and au.get("top_level_ok") and (pd_dir / "plan.json").exists() and (pd_dir / "code/fd_wrapper_fix.py").exists(),
        dict(audit=au, out=pp.stdout.strip()[-300:], err=pp.stderr.strip()[-300:]))
    plan_demo = K.load_plan(pd_dir)
    o11 = SIM / "o11"
    code, log = run(pd_dir, o11)
    vc, vr = validate(pd_dir, o11)
    inv("T11b.chay_tren_goi_that_va_validate", code == 0 and status(o11)["complete"] and vc == 0 and vr["present"] == len(plan_demo["tasks"]),
        dict(tasks=len(plan_demo["tasks"]), validate=vr))
    bad_pack = SIM / "pack_bad"
    shutil.copytree(pd_dir / "data", bad_pack / "data")
    (bad_pack / "data" / "B34" / "labels_human.csv").write_text("x,y\n", encoding="utf-8")
    (bad_pack / "plan.json").write_text("{}")
    au_bad = K.audit_pack(bad_pack)
    inv("T11c.kiem_goi_bat_tep_nhan_nguoi", (not au_bad["sach"]) and any("labels_human.csv" in x for x in au_bad["tep_nghi_ngo"]), au_bad)

    # ---------------- T12 kiểm băm ckpt
    ck_dir = SIM / "ck"
    ck_dir.mkdir()
    man = {"ckpt_sha256": {}}
    for fn, data in (("unet.safetensors", b"u" * 1000), ("style_encoder.safetensors", b"s" * 800)):
        (ck_dir / fn).write_bytes(data)
        man["ckpt_sha256"][fn] = hashlib.sha256(data).hexdigest()
    ok_ = K.verify_ckpt(man, ck_dir)
    (ck_dir / "unet.safetensors").write_bytes(b"u" * 999 + b"X")
    tamper = K.verify_ckpt(man, ck_dir)
    (ck_dir / "style_encoder.safetensors").unlink()
    miss = K.verify_ckpt(man, ck_dir)
    inv("T12.kiem_bam_ckpt", ok_ == [] and any("sai băm" in x for x in tamper) and any("thiếu" in x for x in miss), dict(ok=ok_, sua=tamper, thieu=miss))

    # ---------------- T14 validator bắt tar hỏng
    import tarfile
    some = sorted((ref / "shards").rglob("*.tar"))[0]
    task0 = next(t for t in tasks if K.rel_path(t["id"]) == some.relative_to(ref).as_posix())
    good = K.validate_shard(some, task0, plan)
    # (a) thay một ảnh bằng byte khác, giữ nguyên meta
    bad1 = SIM / "bad1.tar"
    with tarfile.open(some) as src, tarfile.open(bad1, "w") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read()
            if m.name.startswith("U+") and not getattr(dst, "_done_one", False):
                data = data[:-1] + b"Z"; dst._done_one = True
            ti = tarfile.TarInfo(m.name); ti.size = len(data); ti.mtime = 0
            dst.addfile(ti, io.BytesIO(data))
    e1 = K.validate_shard(bad1, task0, plan)
    # (b) cắt cụt tệp (ghi dở)
    bad2 = SIM / "bad2.tar"
    bad2.write_bytes(some.read_bytes()[: some.stat().st_size // 2])
    e2 = K.validate_shard(bad2, task0, plan)
    inv("T14.validator_bat_tar_hong", good == [] and len(e1) >= 1 and len(e2) >= 1, dict(sua_anh=e1[:2], cat_cut=e2[:2]))

    # ---------------- T15 đầu ra Kaggle ≤ 500 tệp: bundle + chỉ mục; phiên sau gắn đầu ra phiên trước; bung ra byte giống tham chiếu
    b1, b2 = SIM / "bundles1", SIM / "bundles2"
    s15a, s15b = SIM / "s15a", SIM / "s15b"
    code, log = run(pack, s15a, "--bundle-dir", str(b1), "--bundle-every", "0.5", budget="0.0009", ms="1.5", est="0.15")
    sa = status(s15a)
    idx_a = [json.loads(p_.read_text(encoding="utf-8")) for p_ in sorted(b1.glob("bundle_*.index.json"))]
    ids_in_a = [i for ix in idx_a for i in ix["ids"]]
    inv("T15.ca_A_bundle_day_du_va_it_tep", code == 0 and len(idx_a) >= 1 and set(ids_in_a) == K.done_ids(s15a) and len(ids_in_a) == len(set(ids_in_a)) and len(list(b1.rglob("*"))) <= 34,   # 04/10: + thư mục logs/ (log worker) và status_<phiên>/ (bản sao trạng thái) cạnh các bundle
        dict(bundles=len(idx_a), tasks=len(ids_in_a), tep_trong_thu_muc_dau_ra=len(list(b1.rglob("*")))))
    code, log = run(pack, s15b, "--bundle-dir", str(b2), "--bundle-every", "0.5", "--prev", str(b1))                  # ca mới: scratch trống, đầu ra phiên trước làm đầu vào
    sb = status(s15b)
    ids_b = set(jsonl_ids(s15b))
    idx_b = [json.loads(p_.read_text(encoding="utf-8")) for p_ in sorted(b2.glob("bundle_*.index.json"))]
    ids_in_b = [i for ix in idx_b for i in ix["ids"]]
    inv("T15.ca_B_chi_lam_phan_con_lai_tu_chi_muc", sb["complete"] and not (ids_b & set(ids_in_a)) and set(ids_in_a) | set(ids_in_b) == set(ids) and not (set(ids_in_a) & set(ids_in_b)),
        dict(lam_ca_B=len(ids_b), trong_bundle_A=len(ids_in_a), trong_bundle_B=len(ids_in_b)))
    up = subprocess.run([sys.executable, str(RUNNER), "unbundle", "--bundles", str(b1), str(b2), "--dest", str(SIM / "unb")], capture_output=True, text=True)
    inv("T15.bung_bundle_byte_giong_tham_chieu", up.returncode == 0 and shard_hashes(SIM / "unb") == ref_h, dict(out=up.stdout.strip()[-200:]))
    # chỉ mục không đáng tin khi tar cụt hoặc khác kế hoạch
    victim = sorted(b1.glob("bundle_*.tar"))[0]
    data0 = victim.read_bytes()
    victim.write_bytes(data0[: len(data0) // 2])
    trusted = K.prev_done([str(b1)], plan["sha"])
    victim.write_bytes(data0)
    wrong = K.prev_done([str(b1)], "0" * 64)
    inv("T15.chi_muc_khong_tin_khi_tar_cut_hoac_khac_ke_hoach", len(trusted) < len(set(ids_in_a)) and wrong == set(), dict(tin_khi_cut=len(trusted), tong=len(set(ids_in_a)), khac_ke_hoach=len(wrong)))

    # ---------------- T17 Syncer: lỗi mạng tạm thời -> thử lại; lỗi kéo dài -> không đánh dấu đã đẩy, ca vẫn thoát sạch
    sy_out = SIM / "sy_out"
    for tid in ids[:5]:
        pth = sy_out / K.rel_path(tid); pth.parent.mkdir(parents=True, exist_ok=True); pth.write_bytes(b"x" * 10)

    class Flaky:
        def __init__(self, fail_n):
            self.fail_n, self.calls, self.got = fail_n, 0, []

        def push(self, root, rels, message=""):
            self.calls += 1
            if self.calls <= self.fail_n:
                raise ConnectionError("429 Too Many Requests")
            self.got += list(rels)
    f1 = Flaky(2)
    sy1 = K.Syncer(sy_out, f1, 999, set(), log=lambda *_: None, retries=4, backoff=0.01)
    ok1 = sy1.flush()
    f2 = Flaky(10 ** 9)
    sy2 = K.Syncer(sy_out, f2, 999, set(), log=lambda *_: None, retries=3, backoff=0.01)
    ok2 = sy2.flush()
    pushed_after_fail = len(sy2.pushed)
    f2.fail_n = 0
    ok3 = sy2.flush()                                                                  # mạng hồi lại ở lượt sau: đẩy bù đủ
    inv("T17.syncer_thu_lai_va_khong_danh_dau_sai", ok1 and sy1.fail == 2 and len(f1.got) == 5 and (not ok2) and pushed_after_fail == 0 and ok3 and len(f2.got) == 5,
        dict(thu_lai=sy1.fail, sau_loi_keo_dai_da_day=pushed_after_fail, sau_hoi_phuc=len(f2.got)))

    R["tong"] = dict(pass_=sum(v["ok"] for v in R["invariants"].values()), fail=sum(not v["ok"] for v in R["invariants"].values()), n=len(R["invariants"]))
    (OUT / "kiem_khung.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("TỔNG:", R["tong"])


if __name__ == "__main__":
    main()
