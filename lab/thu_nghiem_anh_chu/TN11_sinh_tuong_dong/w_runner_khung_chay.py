"""TN11 "full" — hướng RUNNER/KAGGLE: KHUNG chạy bền (đề xuất, mã mới; KHÔNG thay tn11_kaggle.py) để sinh ảnh glyph theo phong cách
TỪNG cuốn cho cả 10 cuốn trên Kaggle (T4 x2) qua nhiều ca.

Tính chất thiết kế (mỗi ý đều có phép thử CPU trong w_runner_kiem_khung.py):
  1. KẾ HOẠCH BẤT BIẾN (plan.json, dựng trên máy Mac, có sha256): danh sách việc = (cuốn, mô hình, phong cách, LÔ chữ). Lô nhỏ (mặc định 128 chữ
     ≈ 5,7 phút/GPU T4). Thứ tự XEN KẼ giữa các cuốn theo lô ưu tiên -> dừng bất kỳ lúc nào, mọi cuốn đều đã có phần chữ ưu tiên cao.
  2. MỘT VIỆC = MỘT TỆP tar nguyên tử (ghi .part rồi os.replace; byte tất định: mtime=0, hạt giống theo mã việc). Việc "xong" <=> tệp tồn tại.
     Tiếp tục theo từng lô (không phải theo cả cuốn); chạy lại việc cũ cho ra đúng byte cũ (băm trùng).
  3. HÀNG ĐỢI ĐỘNG giữa các GPU (khoá tệp O_EXCL, hết hạn TTL): không chia tĩnh theo cuốn; 1 hay 2 GPU đều chạy.
  4. NGÂN SÁCH GIỜ (--budget-hours): không nhận việc mới nếu không kịp -> thoát SẠCH (mã 0) để phiên Save & Run All được tính là thành công.
  5. LƯU BỀN NGOÀI PHIÊN: đẩy tar lên kho ngoài (HF dataset repo riêng, hoặc thư mục) bằng luồng nền có thử lại; đầu phiên liệt kê kho để biết việc đã xong.
  6. CỔNG use_fst: assert_standard_path() dừng ngay (mã 2) nếu mô hình nạp là FontDiffuserModelDPMWithFST hoặc có cảnh báo thiếu checkpoint.
  7. CHỮ KHÔNG VẼ ĐƯỢC không biến mất âm thầm: kế hoạch ghi rõ chữ nào dùng phông nào và chữ nào không phông nào có (uncovered).
  8. Không có nhãn người nào trong gói: make-plan chỉ nhận danh sách chữ + ảnh phong cách; validate kiểm gói.

Lệnh (xem w_runner_kiem_khung.py để chạy thử bằng bộ sinh giả, 0 GPU):
    python w_runner_khung_chay.py make-plan --books books.json --fonts fonts.json --out plan.json
    python w_runner_khung_chay.py run --pack <thư mục có plan.json> --out /kaggle/working/tn11_full --tmp /kaggle/tmp/tn11 \\
           --remote hf:<user>/<repo> --budget-hours 11
    python w_runner_khung_chay.py validate --pack <thư mục> --out <thư mục shards>

Phần GPU thật (RealGen) đã kiểm tương đương với wrapper trên CPU (w_runner_phat_hien_fst.py); CHƯA chạy trên GPU Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

TOOL = "w_runner_khung_chay/1"
SHARD_RE = re.compile(r"shards/([^/]+)/([^/]+)/(\d{5})\.tar$")


# ================================================================================================ tiện ích
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def seed_of(task_id: str) -> int:
    """Hạt giống tất định theo mã việc (không phụ thuộc thứ tự chạy/lần tiếp tục)."""
    return int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:4], "big")


def canon(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def now() -> float:
    return time.time()


# ================================================================================================ kế hoạch
def make_plan(books: dict, passes: list, shard: int = 128, fonts: dict | None = None) -> dict:
    """books: {cuốn: {"chars": [chữ theo thứ tự ƯU TIÊN GIẢM DẦN], "styles": số ảnh phong cách}}
    fonts: {tên phông: tập mã điểm} theo thứ tự ưu tiên (None = bỏ qua kiểm phông). passes: [(mô_hình, phong_cách), ...]."""
    font_names = list(fonts) if fonts else ["default"]
    pb = {}
    for b, spec in books.items():
        covered, fidx, unc = [], [], []
        fo = spec.get("font_of")                          # {chữ: tên phông}: gán CỤ THỂ (vd. quy tắc PUA, kiểm vẽ thật) thay cho "phông đầu có chữ"
        for c in dict.fromkeys(spec["chars"]):          # bỏ trùng, giữ thứ tự
            if fonts is None:
                covered.append(c); fidx.append(0); continue
            if fo is not None:
                k = font_names.index(fo[c]) if (c in fo and fo[c] in fonts and ord(c) in fonts[fo[c]]) else None
            else:
                k = next((i for i, n in enumerate(font_names) if ord(c) in fonts[n]), None)
            if k is None:
                unc.append(c)
            else:
                covered.append(c); fidx.append(k)
        pb[b] = dict(chars="".join(covered), font_idx=fidx, uncovered="".join(unc), styles=int(spec["styles"]))
    nsh = {b: math.ceil(len(v["chars"]) / shard) for b, v in pb.items()}
    tasks = []
    for r in range(max(nsh.values()) if nsh else 0):
        for model, style in passes:
            for b, v in pb.items():
                if r < nsh[b] and style < v["styles"]:
                    tasks.append(dict(id=f"{b}/{model}_s{style}/{r:05d}", book=b, model=model, style=style,
                                      lo=r * shard, hi=min((r + 1) * shard, len(v["chars"]))))
    plan = dict(tool=TOOL, shard=shard, passes=[list(p) for p in passes], font_names=font_names, books=pb, tasks=tasks)
    plan["sha"] = plan_sha(plan)
    return plan


def plan_sha(plan: dict) -> str:
    p = {k: v for k, v in plan.items() if k != "sha"}
    return sha256_bytes(canon(p))


def load_plan(pack: Path) -> dict:
    plan = json.loads((pack / "plan.json").read_text(encoding="utf-8"))
    if plan_sha(plan) != plan.get("sha"):
        raise SystemExit("plan.json hỏng/bị sửa: sha không khớp")
    return plan


def task_chars(plan: dict, task: dict) -> list[str]:
    v = plan["books"][task["book"]]
    return list(v["chars"][task["lo"]:task["hi"]])


def task_fonts(plan: dict, task: dict) -> list[str]:
    v = plan["books"][task["book"]]
    return [plan["font_names"][i] for i in v["font_idx"][task["lo"]:task["hi"]]]


def rel_path(task_id: str) -> str:
    b, mp, r = task_id.split("/")
    return f"shards/{b}/{mp}/{r}.tar"


# ================================================================================================ tar nguyên tử, tất định
def write_shard(out: Path, task: dict, plan: dict, images: dict, skipped: dict) -> Path:
    path = out / rel_path(task["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp-{os.getpid()}-{path.name}.part")
    chars = task_chars(plan, task)
    order = sorted(images, key=ord)
    meta = dict(task=task["id"], plan_sha=plan["sha"], seed=seed_of(task["id"]), requested="".join(chars), generated="".join(order),
                skipped={c: skipped[c] for c in sorted(skipped, key=ord)},
                images_sha256=sha256_bytes(b"".join(hashlib.sha256(images[c]).digest() for c in order)))
    with tarfile.open(tmp, "w") as tf:
        for c in order:
            ti = tarfile.TarInfo(f"U+{ord(c):04X}.png"); ti.size = len(images[c]); ti.mtime = 0; ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(images[c]))
        mb = canon(meta)
        ti = tarfile.TarInfo("_meta.json"); ti.size = len(mb); ti.mtime = 0; ti.mode = 0o644
        tf.addfile(ti, io.BytesIO(mb))
    os.replace(tmp, path)
    return path


def validate_shard(path: Path, task: dict, plan: dict) -> list[str]:
    """Trả về danh sách lỗi (rỗng = tốt)."""
    errs = []
    try:
        with tarfile.open(path) as tf:
            names = tf.getnames()
            meta = json.loads(tf.extractfile("_meta.json").read())
            imgs = {n: tf.extractfile(n).read() for n in names if n.startswith("U+")}
    except Exception as e:  # noqa: BLE001
        return [f"{task['id']}: không đọc được tar ({type(e).__name__})"]
    if names[-1] != "_meta.json":
        errs.append(f"{task['id']}: _meta.json không phải thành viên cuối (ghi dở?)")
    if meta.get("plan_sha") != plan["sha"]:
        errs.append(f"{task['id']}: plan_sha khác kế hoạch")
    req = task_chars(plan, task)
    gen, sk = list(meta["generated"]), meta["skipped"]
    if sorted(req) != sorted(gen + list(sk)):
        errs.append(f"{task['id']}: sinh+bỏ qua ≠ yêu cầu ({len(gen)}+{len(sk)} vs {len(req)})")
    if set(imgs) != {f"U+{ord(c):04X}.png" for c in gen}:
        errs.append(f"{task['id']}: tên ảnh trong tar ≠ meta.generated")
    h = sha256_bytes(b"".join(hashlib.sha256(imgs[f"U+{ord(c):04X}.png"]).digest() for c in sorted(gen, key=ord)))
    if h != meta.get("images_sha256"):
        errs.append(f"{task['id']}: băm ảnh sai")
    return errs


# ================================================================================================ khoá việc (hàng đợi động)
def try_claim(lockdir: Path, task_id: str, worker: str, ttl: float) -> bool:
    lockdir.mkdir(parents=True, exist_ok=True)
    p = lockdir / (task_id.replace("/", "__") + ".lock")
    body = json.dumps(dict(worker=worker, pid=os.getpid(), t=now())).encode()
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, body); os.close(fd)
        return True
    except FileExistsError:
        try:
            age = now() - p.stat().st_mtime
        except FileNotFoundError:
            return False
        if age <= ttl:
            return False
        # khoá chết (tiến trình đã mất): cướp lại. Việc là tất định + ghi nguyên tử nên nếu hai bên cùng cướp chỉ tốn công, không sai.
        try:
            os.replace(p, p.with_name(p.name + f".stale-{os.getpid()}-{int(now())}"))
        except FileNotFoundError:
            return False
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, body); os.close(fd)
            return True
        except FileExistsError:
            return False


def release(lockdir: Path, task_id: str):
    try:
        (lockdir / (task_id.replace("/", "__") + ".lock")).unlink()
    except FileNotFoundError:
        pass


def cleanup_stale(out: Path, ttl: float) -> dict:
    """Đầu phiên: xoá .part cũ (ghi dở) và khoá hết hạn."""
    n_part = n_lock = 0
    for p in list((out / "shards").rglob("*.part")) if (out / "shards").exists() else []:
        if now() - p.stat().st_mtime > ttl:
            p.unlink(); n_part += 1
    for p in list((out / "locks").glob("*")) if (out / "locks").exists() else []:
        if now() - p.stat().st_mtime > ttl:
            p.unlink(); n_lock += 1
    return dict(part_removed=n_part, locks_removed=n_lock)


# ================================================================================================ kho ngoài
class DirRemote:
    """Kho ngoài = thư mục (thử nghiệm; cũng dùng được cho Kaggle Dataset gắn vào /kaggle/input hoặc ổ đồng bộ)."""
    kind = "dir"

    def __init__(self, root: str):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def list_done(self) -> set:
        return {f"{m.group(1)}/{m.group(2)}/{m.group(3)}" for p in self.root.rglob("*.tar")
                if (m := SHARD_RE.search(p.relative_to(self.root).as_posix()))}

    def push(self, local_root: Path, rels: list, message: str = ""):
        for r in rels:
            dst = self.root / r
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(local_root / r, tmp); os.replace(tmp, dst)

    def get_text(self, rel: str):
        p = self.root / rel
        return p.read_text(encoding="utf-8") if p.exists() else None

    def put_text(self, rel: str, text: str):
        p = self.root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text, encoding="utf-8")


class HFRemote:
    """HF dataset repo RIÊNG. Cần Internet + HF_TOKEN (Kaggle Secrets). CHƯA chạy thật trong phiên này (chỉ kiểm chữ ký hàm với huggingface_hub cài sẵn)."""
    kind = "hf"

    def __init__(self, repo_id: str, token: str | None = None, repo_type: str = "dataset", private: bool = True):
        from huggingface_hub import HfApi
        self.api = HfApi(token=token)
        self.repo_id, self.repo_type, self.private, self.token = repo_id, repo_type, private, token
        self.api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

    def list_done(self) -> set:
        files = self.api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)
        return {f"{m.group(1)}/{m.group(2)}/{m.group(3)}" for f in files if (m := SHARD_RE.search(f))}

    def push(self, local_root: Path, rels: list, message: str = ""):
        # upload_folder tự chia nhiều commit nếu nhiều tệp; allow_patterns là danh sách đường dẫn đúng (không có ký tự đại diện)
        self.api.upload_folder(folder_path=str(local_root), repo_id=self.repo_id, repo_type=self.repo_type, allow_patterns=list(rels),
                               commit_message=message or f"{len(rels)} lô")

    def get_text(self, rel: str):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            return Path(hf_hub_download(repo_id=self.repo_id, filename=rel, repo_type=self.repo_type, token=self.token)).read_text(encoding="utf-8")
        except EntryNotFoundError:
            return None

    def put_text(self, rel: str, text: str):
        self.api.upload_file(path_or_fileobj=text.encode("utf-8"), path_in_repo=rel, repo_id=self.repo_id, repo_type=self.repo_type)


def open_remote(spec: str | None):
    if not spec:
        return None
    if spec.startswith("dir:"):
        return DirRemote(spec[4:])
    if spec.startswith("hf:"):
        tok = os.environ.get("HF_TOKEN")
        if not tok:
            try:
                from kaggle_secrets import UserSecretsClient
                tok = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:  # noqa: BLE001
                tok = None
        if not tok:
            try:
                from huggingface_hub import get_token                       # token đã `hf auth login` (chạy trên Mac)
                tok = get_token()
            except Exception:  # noqa: BLE001
                tok = None
        if not tok:
            raise SystemExit("Thiếu HF_TOKEN (Kaggle: Add-ons → Secrets, gắn vào notebook)")
        return HFRemote(spec[3:], tok)
    raise SystemExit(f"--remote không hiểu: {spec}")


class Syncer(threading.Thread):
    """Luồng nền đẩy tar mới lên kho ngoài; thử lại có lùi; chỉ coi là đã đẩy khi thành công."""

    def __init__(self, out: Path, remote, every_s: float, pushed_rel: set, log=print, retries=4, backoff=1.0):
        super().__init__(daemon=True)
        self.out, self.remote, self.every_s, self.log, self.retries, self.backoff = out, remote, every_s, log, retries, backoff
        self.pushed = set(pushed_rel); self.ev = threading.Event(); self.fail = 0; self.n_push = 0
        self.lock = threading.Lock()

    def run(self):
        while not self.ev.wait(self.every_s):
            self.flush()

    def flush(self) -> bool:
        with self.lock:                                   # luồng nền và lần đẩy cuối không đẩy trùng cùng tệp
            return self._flush()

    def _flush(self) -> bool:
        base = self.out / "shards"
        rels = sorted(p.relative_to(self.out).as_posix() for p in base.rglob("*.tar")) if base.exists() else []
        new = [r for r in rels if r not in self.pushed]
        if not new:
            return True
        for k in range(1, self.retries + 1):
            try:
                self.remote.push(self.out, new, f"{len(new)} lô ({time.strftime('%Y-%m-%d %H:%M:%S')})")
                self.pushed |= set(new); self.n_push += len(new)
                return True
            except Exception as e:  # noqa: BLE001
                self.fail += 1
                self.log(f"[sync] lần {k}/{self.retries}: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(self.backoff * k)
        return False

    def stop(self, timeout: float = 900.0) -> bool:
        """Lần đẩy cuối có hạn chờ (huggingface_hub không đặt timeout mạng): quá hạn thì bỏ, trả False; tar đã ghi vẫn nằm ở đầu ra/kho phụ."""
        self.ev.set()
        res = {}
        t = threading.Thread(target=lambda: res.setdefault("ok", self.flush()), daemon=True)
        t.start(); t.join(timeout)
        return bool(res.get("ok", False))


# ================================================================================================ gói đầu ra Kaggle (≤ 500 tệp)
# Thực tế Kaggle: đầu ra phiên (và dataset tạo từ đầu ra) hay bị cắt ở 500 tệp (nhân viên Kaggle: "không có giới hạn cố ý" nhưng người dùng gặp; xem báo cáo).
# Nên các lô nhỏ nằm ở /kaggle/tmp (scratch ~60 GiB, KHÔNG phải đầu ra), còn /kaggle/working chỉ nhận vài tệp tar LỚN + chỉ mục JSON.
def bundle_new(out: Path, bdir: Path, session: str, k: int, already: set, plan_sha_: str):
    base = out / "shards"
    rels = sorted(p.relative_to(out).as_posix() for p in base.rglob("*.tar")) if base.exists() else []
    new = [r for r in rels if (m := SHARD_RE.search(r)) and f"{m.group(1)}/{m.group(2)}/{m.group(3)}" not in already]
    if not new:
        return None, []
    bdir.mkdir(parents=True, exist_ok=True)
    name = f"bundle_{session}_{k:03d}"
    tmp = bdir / f".{name}.tar.part"
    with tarfile.open(tmp, "w") as tf:
        for r in new:
            tf.add(out / r, arcname=r)
    ids = [f"{m.group(1)}/{m.group(2)}/{m.group(3)}" for r in new if (m := SHARD_RE.search(r))]
    os.replace(tmp, bdir / f"{name}.tar")                                   # tar trước, chỉ mục sau: có chỉ mục <=> có dữ liệu
    idx = dict(session=session, k=k, plan_sha=plan_sha_, ids=ids, tar=f"{name}.tar", tar_bytes=(bdir / f"{name}.tar").stat().st_size)
    itmp = bdir / f".{name}.index.json.part"
    itmp.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    os.replace(itmp, bdir / f"{name}.index.json")
    return bdir / f"{name}.tar", ids


def prev_done(dirs: list, plan_sha_: str) -> set:
    """Việc đã xong ở các phiên trước theo chỉ mục bundle_*.index.json (đầu ra phiên trước gắn làm đầu vào: /kaggle/input/<...>)."""
    done = set()
    for d in dirs or []:
        for ip in Path(d).rglob("bundle_*.index.json"):
            try:
                idx = json.loads(ip.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            tar = ip.with_name(idx.get("tar", ""))
            if idx.get("plan_sha") != plan_sha_ or not tar.exists() or tar.stat().st_size != idx.get("tar_bytes"):
                continue                                                      # khác kế hoạch / thiếu / cụt -> không tin
            done |= set(idx["ids"])
    return done


class Bundler(threading.Thread):
    def __init__(self, out: Path, bdir: Path, session: str, every_s: float, already: set, plan_sha_: str, log=print):
        super().__init__(daemon=True)
        self.out, self.bdir, self.session, self.every_s, self.already, self.plan_sha, self.log = out, bdir, session, every_s, set(already), plan_sha_, log
        self.ev = threading.Event(); self.k = 0; self.made = []; self.lock = threading.Lock()

    def run(self):
        while not self.ev.wait(self.every_s):
            self.flush()

    def flush(self):
        with self.lock:
            return self._flush()

    def _flush(self):
        try:
            tar, ids = bundle_new(self.out, self.bdir, self.session, self.k, self.already, self.plan_sha)
        except Exception as e:  # noqa: BLE001
            self.log(f"[bundle] lỗi {type(e).__name__}: {str(e)[:120]}")
            return None
        if tar:
            self.already |= set(ids); self.k += 1; self.made.append(tar.name)
        return tar

    def stop(self):
        self.ev.set()
        return self.flush()


def unbundle(bundle_dirs: list, dest: Path) -> dict:
    """Phía Mac: bung mọi bundle_*.tar thành dest/shards/...; trùng tên phải trùng byte (tất định) nếu không báo lỗi."""
    n_new = n_same = 0
    conflicts = []
    for d in bundle_dirs:
        for tp in sorted(Path(d).rglob("bundle_*.tar")):
            with tarfile.open(tp) as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    data = tf.extractfile(m).read()
                    dst = dest / m.name
                    if dst.exists():
                        if dst.read_bytes() == data:
                            n_same += 1
                        else:
                            conflicts.append(m.name)
                    else:
                        dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data); n_new += 1
    return dict(new=n_new, duplicates_identical=n_same, conflicts=conflicts)


# ================================================================================================ cổng use_fst
class FstWarningCapture:
    """Bắt cảnh báo logger 'src.builders.build' kiểu "⚠ Checkpoint for 'fst_module' not found" (chỉ có khi dựng nhánh FST).
    Chú ý: notebook cũ (lab/kaggle_diffusion/diffusion_run.ipynb cell 15/19) đặt logger này ở mức ERROR -> cảnh báo bị nuốt; ở đây tạm hạ mức xuống INFO."""

    LOGGERS = ("src.builders.build",)

    def __init__(self):
        import logging
        self.msgs: list = []
        outer = self

        class H(logging.Handler):
            def emit(self, rec):
                if rec.levelno >= logging.WARNING:
                    outer.msgs.append(f"{rec.name}: {rec.getMessage()}")
        self.h = H(level=logging.WARNING)
        self._prev = {}

    def __enter__(self):
        import logging
        logging.getLogger("").addHandler(self.h)
        for n in self.LOGGERS:
            lg = logging.getLogger(n)
            self._prev[n] = lg.level
            if lg.level == 0 or lg.level > logging.WARNING:
                lg.setLevel(logging.INFO)
        return self

    def __exit__(self, *a):
        import logging
        logging.getLogger("").removeHandler(self.h)
        for n, lv in self._prev.items():
            logging.getLogger(n).setLevel(lv)


class FstGuardError(RuntimeError):
    pass


def assert_standard_path(gen, captured: list | None = None, raise_error: bool = True) -> dict:
    """Đường chuẩn = pipe.model là FontDiffuserModelDPM (unet+style_encoder+content_encoder), không có mô-đun FST, không có cảnh báo
    thiếu checkpoint. Bản wrapper lỗi (use_fst=True) trả FontDiffuserModelDPMWithFST với mô-đun FST khởi tạo ngẫu nhiên."""
    m = gen.pipe.model
    m = getattr(m, "_orig_mod", m)                       # torch.compile bọc
    cls = type(m).__name__
    fst_attrs = [a for a in ("fst_module", "mss_encoder", "fst_projection", "original_style_projection") if hasattr(m, a)]
    warn = [w for w in (captured or []) if "not found" in w or "Checkpoint for" in w]
    ok = cls == "FontDiffuserModelDPM" and not fst_attrs and not warn
    rep = dict(ok=ok, model_class=cls, fst_attributes=fst_attrs, warnings=warn)
    if not ok and raise_error:
        raise FstGuardError(f"Mô hình KHÔNG ở đường chuẩn use_fst=False: {json.dumps(rep, ensure_ascii=False)}")
    return rep


# ================================================================================================ bộ sinh
class StubGen:
    """Bộ sinh GIẢ cho thử nghiệm điều khiển (không GPU): ảnh = byte tất định theo (cuốn, mô hình, phong cách, chữ, hạt giống).
    crash_chars: gặp chữ này thì tiến trình CHẾT (os._exit) đúng một lần (cờ trên đĩa) — mô phỏng ca bị cắt giữa một lô."""
    name = "stub"

    def __init__(self, ms_per_img: float = 0.0, fail_chars: str = "", crash_chars: str = "", flag: Path | None = None):
        self.ms, self.fail, self.crash, self.flag = ms_per_img, set(fail_chars), set(crash_chars), flag

    def generate(self, task: dict, chars: list, fonts: list, style_path: Path | None, seed: int):
        for i in range(int(os.environ.get("TN11_STUB_NOISE", "0") or 0)):       # chỉ để thử bộ lọc log: mô phỏng worker "ồn"
            print(f"UserWarning: ruido {i} FutureWarning blah blah", flush=True)
        if os.environ.get("TN11_STUB_ERRLINE"):
            print("Error: lỗi giả để thử bộ lọc", flush=True)
        imgs, sk = {}, {}
        for c in chars:
            if c in self.crash and self.flag is not None and not self.flag.exists():
                self.flag.parent.mkdir(parents=True, exist_ok=True); self.flag.write_text("crashed", encoding="utf-8")
                os._exit(23)
            if c in self.fail:
                sk[c] = "gen_error"; continue
            h = hashlib.sha256(f"{task['book']}|{task['model']}|{task['style']}|{ord(c)}|{seed}".encode()).digest()
            imgs[c] = b"\x89PNG\r\n\x1a\n" + h * 4
        if self.ms:
            time.sleep(len(chars) * self.ms / 1000)
        return imgs, sk


class RealGen:
    """Bộ sinh THẬT (FontDiffuser use_fst=False) trong bộ nhớ: không đệm đĩa theo ảnh, không lọc cmap từng chữ, hạt giống theo lô (generator CPU),
    phông theo chữ (một pipeline, đổi FontManager), lùi lô khi OOM. Kiểm cổng use_fst ngay khi nạp."""
    name = "real"

    def __init__(self, repo_code: Path, ckpt_dir: Path, font_paths: dict, device: str, batch: int = 32, guidance: float = 2.0, cache_dir: Path | None = None):
        sys.path.insert(0, str(repo_code)); sys.path.insert(0, str(repo_code / "font_diffusion"))
        import torch
        self.torch = torch
        from fd_wrapper_fix import FixedFontDiffusionGenerator
        self.font_paths = font_paths
        first = next(iter(font_paths.values()))
        self.g = FixedFontDiffusionGenerator(ckpt_dir=str(ckpt_dir), phase1_ckpt_dir=str(ckpt_dir), font_path=str(first),
                                             cache_dir=str(cache_dir or (Path(os.environ.get("TMPDIR", "/tmp")) / "tn11_unused_cache")), device=device, batch_size=batch)
        with FstWarningCapture() as cap:
            self.g._load_pipeline()
        self.guard = assert_standard_path(self.g, cap.msgs)          # ném FstGuardError nếu sai
        print(f"[fd] sẵn sàng ({device}): {self.guard.get('model_class')}, đường chuẩn use_fst=False", flush=True)
        self.g.pipe.guidance_scale = guidance
        self.batch, self.device = batch, device
        self._fm = {}
        self._style = {}

    def _font(self, name):
        if name not in self._fm:
            from inference.sample_optimized import FontManager
            p = Path(self.font_paths[name])
            fm = FontManager(str(p))
            self._fm[name] = (fm, fm.get_font(p.stem))
        return self._fm[name][1]

    def _style_tensor(self, path: Path):
        k = str(path)
        if k not in self._style:
            from inference.sample_optimized import get_style_transform
            from PIL import Image
            self._style[k] = get_style_transform(self.g.args.style_image_size)(Image.open(path).convert("RGB"))
        return self._style[k]

    def generate(self, task: dict, chars: list, fonts: list, style_path: Path, seed: int):
        torch = self.torch
        from inference.sample_optimized import get_content_transform
        from src.tools.utils import ttf2im
        a = self.g.args
        ct = get_content_transform(a.content_image_size)
        st = self._style_tensor(style_path)
        items, sk = [], {}
        for c, fn in zip(chars, fonts):
            im = ttf2im(font=self._font(fn), char=c)
            if im is None:
                sk[c] = "render_none"; continue
            items.append((c, ct(im)))
        imgs, i, bs, bi = {}, 0, self.batch, 0
        while i < len(items):
            cur = items[i:i + bs]
            try:
                bc = torch.stack([t for _, t in cur]).to(self.device, dtype=torch.float32)
                bst = st[None].repeat(len(cur), 1, 1, 1).to(self.device, dtype=torch.float32)
                gen = torch.Generator().manual_seed((seed + bi) & 0x7FFFFFFF)
                with torch.inference_mode():
                    out = self.g.pipe.generate(content_images=bc, style_images=bst, batch_size=len(cur), order=a.order,
                                               num_inference_step=a.num_inference_steps, content_encoder_downsample_size=a.content_encoder_downsample_size,
                                               t_start=a.t_start, t_end=a.t_end, dm_size=a.content_image_size, algorithm_type=a.algorithm_type,
                                               skip_type=a.skip_type, method=a.method, correcting_x0_fn=a.correcting_x0_fn, generator=gen)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    sk[cur[0][0]] = "oom"; i += 1; continue
                bs = max(1, bs // 2); continue
            for (c, _), im in zip(cur, out):
                buf = io.BytesIO(); im.save(buf, format="PNG"); imgs[c] = buf.getvalue()
            i += len(cur); bi += 1
        return imgs, sk


# ================================================================================================ vòng lặp worker
def worker_loop(plan: dict, pack: Path, out: Path, wid: str, gen, deadline: float, est_s: float, ttl: float,
                crash_after: int = 0, log=print, max_tasks: int = 0) -> dict:
    lockdir, stat = out / "locks", out / "status"
    stat.mkdir(parents=True, exist_ok=True)
    ema, n_done, n_img, stop, consec = est_s, 0, 0, "complete", 0
    jl = stat / f"tasks-{wid}.jsonl"
    stopfile = out / "STOP"
    rd = stat / "remote_done.txt"
    remote_done = set(rd.read_text(encoding="utf-8").split()) if rd.exists() else set()
    failed = set()
    for task in plan["tasks"]:
        path = out / rel_path(task["id"])
        if path.exists() or task["id"] in remote_done or task["id"] in failed:
            continue
        if stopfile.exists():
            stop = "stop_file"; break
        if now() + ema * 1.15 > deadline:
            stop = "budget"; break
        if max_tasks and n_done >= max_tasks:
            stop = "max_tasks"; break
        if not try_claim(lockdir, task["id"], wid, ttl):
            continue
        if path.exists():                                  # đã xong giữa lúc kiểm và khoá
            release(lockdir, task["id"]); continue
        t0 = now()
        try:
            chars, fonts = task_chars(plan, task), task_fonts(plan, task)
            style = pack / "data" / task["book"] / f"style_{task['style']}.png"
            imgs, sk = gen.generate(task, chars, fonts, style, seed_of(task["id"]))
            if not imgs or len(sk) * 2 >= max(1, len(chars)):   # lỗi hệ thống (OOM lô 1, vẽ hỏng...) KHÔNG được ghi thành "xong" bằng tar rỗng
                raise RuntimeError(f"task_rong: {len(imgs)} ảnh, {len(sk)} bỏ qua / {len(chars)} chữ")
            write_shard(out, task, plan, imgs, sk)
            consec = 0
        except FstGuardError:
            raise
        except Exception as e:  # noqa: BLE001 — lỗi một việc không giết cả ca; 3 lỗi liên tiếp -> dừng (lỗi hệ thống kiểu CUDA hỏng)
            release(lockdir, task["id"]); failed.add(task["id"]); consec += 1
            with open(stat / f"errors-{wid}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(id=task["id"], err=f"{type(e).__name__}: {str(e)[:200]}", t=round(now(), 3)), ensure_ascii=False) + "\n")
            if consec >= 3:
                stop = "errors"; break
            continue
        release(lockdir, task["id"])
        dt = now() - t0
        ema = 0.7 * ema + 0.3 * dt if n_done else dt
        n_done += 1; n_img += len(imgs)
        with open(jl, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(id=task["id"], worker=wid, n_gen=len(imgs), n_skip=len(sk), sec=round(dt, 3), t=round(now(), 3)), ensure_ascii=False) + "\n")
        if crash_after and n_done >= crash_after:           # chỉ thử nghiệm: mô phỏng phiên chết ngay sau khi NHẬN việc kế (để lại khoá + .part)
            for nxt in plan["tasks"]:
                if (out / rel_path(nxt["id"])).exists() or nxt["id"] in remote_done:
                    continue
                if try_claim(lockdir, nxt["id"], wid, ttl):       # nhận một việc chưa ai nhận rồi chết: để lại khoá + .part
                    pth = out / rel_path(nxt["id"])
                    pth.parent.mkdir(parents=True, exist_ok=True)
                    pth.with_name(f".tmp-{os.getpid()}-{pth.name}.part").write_bytes(b"ghi do")
                    break
            os._exit(17)
    return dict(worker=wid, tasks=n_done, images=n_img, stop=stop, ema_s=round(ema, 3), failed=sorted(failed))


# ================================================================================================ môi trường, gói, nạp mã
PIP = {"diffusers": "diffusers", "pygame": "pygame", "info_nce": "info-nce-pytorch", "kornia": "kornia", "einops": "einops", "fontTools": "fonttools",
       "safetensors": "safetensors", "skimage": "scikit-image", "cv2": "opencv-python-headless", "accelerate": "accelerate", "huggingface_hub": "huggingface_hub"}


def ensure_deps(pins: dict | None = None) -> list:
    """Cài phần còn thiếu (cần Internet). Mô-đun->tên pip lấy từ tn11_kaggle.py:31. Ghim phiên bản: manifest["pip_pins"] = {tên_pip: "tên==x.y.z"}."""
    import importlib
    miss = []
    for mod, pkg in PIP.items():
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001
            miss.append((pins or {}).get(pkg, pkg))
    if miss:
        print("[deps] pip install", " ".join(miss), flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-input", *miss], capture_output=True, text=True)
        if r.returncode != 0:                                   # chỉ in khi lỗi (ca thật không bị tràn log bởi pip)
            print((r.stdout + r.stderr)[-1500:], flush=True)
            raise SystemExit(f"pip install lỗi: {' '.join(miss)}")
    return miss


def env_snapshot(light: bool = False) -> dict:
    """Ghi phiên bản môi trường vào status/env.json (để ghim lại sau lần chạy thành công; ảnh Docker Kaggle cập nhật ~2 tuần/lần)."""
    import platform
    e = dict(python=sys.version.split()[0], platform=platform.platform())
    if light:
        return e
    for m in ("torch", "diffusers", "safetensors", "huggingface_hub", "numpy", "PIL", "fontTools", "pygame", "cv2"):
        try:
            mod = __import__(m)
            e[m] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001
            e[m] = None
    try:
        import torch
        e["cuda"] = torch.version.cuda
        e["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        e["gpus"] = []
    return e


def verify_ckpt(manifest: dict, ckpt_dir: Path) -> list:
    bad = []
    for fn, h in manifest.get("ckpt_sha256", {}).items():
        p = ckpt_dir / fn
        if not p.exists():
            bad.append(f"thiếu {fn}")
            continue
        hh = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                hh.update(b)
        if hh.hexdigest() != h:
            bad.append(f"sai băm {fn}")
    return bad


def setup_real(pack: Path, tmp: Path) -> dict:
    """Chép code/ -> tmp (ổ /kaggle/tmp ~60 GiB, không tính vào 20 GiB của /kaggle/working) và ckpt -> tmp/font_diffusion/ckpt/PROD, kiểm băm."""
    man = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if not (tmp / "font_diffusion" / "src").exists():
        shutil.copytree(pack / "code", tmp, dirs_exist_ok=True)
    ck = tmp / "font_diffusion" / "ckpt" / "PROD"
    ck.mkdir(parents=True, exist_ok=True)
    for fn in man.get("ckpt_sha256", {}):
        if not (ck / fn).exists():
            shutil.copy2(pack / "ckpt" / fn, ck / fn)
    bad = verify_ckpt(man, ck)
    if bad:
        raise SystemExit(f"checkpoint lỗi: {bad} — gắn lại dataset")
    return man


FORBIDDEN_RE = re.compile(r"(nhan_nguoi|human|truth|ground|gt_|\.csv$|\.tsv$|\.pkl$|labels)", re.I)
ALLOWED_JSON = {"plan.json", "fonts.json", "manifest.json"}


def audit_pack(out: Path) -> dict:
    """Kiểm gói TRƯỚC khi tải lên: không có tệp nhãn/nhãn người, số mục cấp một ≤ 50 (giới hạn Kaggle Datasets), tổng kích thước."""
    files = [p for p in out.rglob("*") if p.is_file()]
    bad = [p.relative_to(out).as_posix() for p in files if FORBIDDEN_RE.search(p.name) and p.name not in ALLOWED_JSON]
    other_json = [p.relative_to(out).as_posix() for p in files if p.suffix == ".json" and p.name not in ALLOWED_JSON]
    top = {p.relative_to(out).parts[0] for p in files}
    return dict(files=len(files), MB=round(sum(p.stat().st_size for p in files) / 1e6, 1), top_level=len(top), top_level_ok=len(top) <= 50,
                tep_nghi_ngo=bad + other_json, sach=not (bad or other_json))


def cmd_pack(a):
    """Dựng THƯ MỤC gói cho Kaggle Dataset (không nén; Kaggle tự giải nén zip nếu bạn upload zip). KHÔNG chứa nhãn."""
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    repo = Path(a.repo)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} không trống")
    out.mkdir(parents=True, exist_ok=True)
    from fontTools.ttLib import TTFont
    fonts, fpaths = {}, {}
    for name, p in spec["fonts"].items():
        f = TTFont(str(repo / p), lazy=True); s = set()
        for st in f["cmap"].tables:
            s |= set(st.cmap)
        fonts[name] = s; fpaths[name] = repo / p
    books = {}
    for b, v in spec["books"].items():
        books[b] = dict(chars=[c for c in Path(v["chars_file"]).read_text(encoding="utf-8").split("\n") if c], styles=len(v["style_pngs"]))
        if v.get("font_of_file"):                       # TSV "chữ<TAB>tên phông"
            books[b]["font_of"] = dict(ln.split("\t") for ln in Path(v["font_of_file"]).read_text(encoding="utf-8").split("\n") if "\t" in ln)
    passes = [(m, int(j)) for m, j in (x.split(":") for x in a.passes)]
    plan = make_plan(books, passes, a.shard, fonts)
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    (out / "fonts.json").write_text(json.dumps({n: p.name for n, p in fpaths.items()}, ensure_ascii=False), encoding="utf-8")
    for b, v in spec["books"].items():
        (out / "data" / b).mkdir(parents=True, exist_ok=True)
        for j, sp in enumerate(v["style_pngs"]):
            shutil.copyfile(sp, out / "data" / b / f"style_{j}.png")
    shutil.copyfile(os.path.abspath(__file__), out / "w_runner_khung_chay.py")
    code = out / "code"
    (code / "core" / "ranking").mkdir(parents=True, exist_ok=True)
    (code / "core" / "__init__.py").write_text(""); (code / "core" / "ranking" / "__init__.py").write_text("")
    shutil.copyfile(repo / "core/ranking/fontdiffusion_gen.py", code / "core/ranking/fontdiffusion_gen.py")
    shutil.copyfile(Path(a.wrapper_fix), code / "fd_wrapper_fix.py")
    fd = repo / "font_diffusion"
    for sub in ("src", "inference"):
        for f in sorted((fd / sub).rglob("*.py")):
            d = code / "font_diffusion" / f.relative_to(fd); d.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(f, d)
    (code / "font_diffusion" / "fonts").mkdir(parents=True, exist_ok=True)
    for p in fpaths.values():
        shutil.copyfile(p, code / "font_diffusion" / "fonts" / p.name)
    man = dict(tool=TOOL, plan_sha=plan["sha"], ckpt_sha256={}, books={b: dict(chars=len(v["chars"]), uncovered=len(v["uncovered"])) for b, v in plan["books"].items()})
    if not a.no_ckpt:
        (out / "ckpt").mkdir(exist_ok=True)
        for f in sorted((fd / "ckpt" / "PROD").glob("*.safetensors")):
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
            man["ckpt_sha256"][f.name] = h.hexdigest()
            shutil.copyfile(f, out / "ckpt" / f.name)
    man["assets_sha"] = compute_assets_sha(out, man["ckpt_sha256"])
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    rep = audit_pack(out)
    print(f"[pack] {out}: {json.dumps(rep, ensure_ascii=False)} | {len(plan['tasks'])} việc | sha {plan['sha'][:12]}")
    if not rep["sach"] or not rep["top_level_ok"]:
        raise SystemExit("gói không đạt kiểm (xem tep_nghi_ngo / top_level)")


# ================================================================================================ lệnh
def cmd_make_plan(a):
    spec = json.loads(Path(a.books).read_text(encoding="utf-8"))
    books = {b: dict(chars=[c for c in Path(v["chars_file"]).read_text(encoding="utf-8").split("\n") if c], styles=v["styles"]) for b, v in spec.items()}
    fonts = None
    if a.fonts:
        from fontTools.ttLib import TTFont
        fonts = {}
        for name, p in json.loads(Path(a.fonts).read_text(encoding="utf-8")).items():
            f = TTFont(p, lazy=True); s = set()
            for st in f["cmap"].tables:
                s |= set(st.cmap)
            fonts[name] = s
    passes = [tuple(p.split(":")) for p in a.passes]
    passes = [(m, int(j)) for m, j in passes]
    plan = make_plan(books, passes, a.shard, fonts)
    Path(a.out).write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    print(f"[make-plan] {len(plan['tasks'])} việc, {sum(len(v['chars']) for v in plan['books'].values())} chữ phủ được, "
          f"{sum(len(v['uncovered']) for v in plan['books'].values())} chữ không phông nào có; sha {plan['sha'][:12]}")


def find_pack(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    if (here / "plan.json").exists():
        return here
    import glob
    for d in sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*")) + sorted(glob.glob("/kaggle/input/*/*/*")):
        if os.path.isfile(os.path.join(d, "plan.json")):
            return Path(d)
    raise SystemExit("Không thấy plan.json trong /kaggle/input")


def done_ids(out: Path) -> set:
    base = out / "shards"
    return {f"{m.group(1)}/{m.group(2)}/{m.group(3)}" for p in base.rglob("*.tar") if (m := SHARD_RE.search(p.relative_to(out).as_posix()))} if base.exists() else set()


def cmd_worker(a):
    pack, out = find_pack(a.pack), Path(a.out)
    plan = load_plan(pack)
    try:
        if a.stub:
            gen = StubGen(a.stub_ms, a.stub_fail, a.stub_crash, out / "status" / "crash_once.flag")
        else:
            import torch
            dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
            fonts = {n: str(Path(a.tmp) / "font_diffusion" / "fonts" / f) for n, f in json.loads((pack / "fonts.json").read_text(encoding="utf-8")).items()}
            gen = RealGen(Path(a.tmp), Path(a.tmp) / "font_diffusion" / "ckpt" / "PROD", fonts, dev, a.gen_bs, cache_dir=Path(a.tmp) / "_unused_cache")
        r = worker_loop(plan, pack, out, a.worker, gen, a.deadline, a.est_s, a.ttl, a.crash_after, max_tasks=a.max_tasks)
    except FstGuardError as e:
        print(f"[worker {a.worker}] CỔNG use_fst: {e}", flush=True)
        out.mkdir(parents=True, exist_ok=True); (out / "STOP").write_text(str(e), encoding="utf-8")
        sys.exit(2)
    (out / "status").mkdir(parents=True, exist_ok=True)
    (out / "status" / f"worker-{a.worker}.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    print(f"[worker {a.worker}] {json.dumps(r, ensure_ascii=False)}", flush=True)


def safe_dir(p: str) -> Path:
    """Tạo thư mục; nếu không ghi được (vd. /kaggle/tmp không tồn tại/không cho tạo) thì lùi về /tmp/<tên>."""
    q = Path(p)
    try:
        q.mkdir(parents=True, exist_ok=True)
        return q
    except OSError:
        alt = Path("/tmp") / q.name
        alt.mkdir(parents=True, exist_ok=True)
        print(f"[run] không tạo được {q} -> dùng {alt}", flush=True)
        return alt


QUIET_ENV = dict(PYTHONWARNINGS="ignore", TQDM_DISABLE="1", HF_HUB_DISABLE_PROGRESS_BARS="1", DIFFUSERS_VERBOSITY="error", TRANSFORMERS_VERBOSITY="error",
                 TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
IMPORTANT_RE = re.compile(r"(\[worker|\[fd\]|Traceback|Error\b|CỔNG|KHÔNG CÓ GPU|OutOfMemory|task_rong|Killed|Segmentation)")


def _pump(pr, logpath: Path, tag: str, verbose: bool):
    """Đọc hết đầu ra worker (không để đầy ống): GHI MỌI DÒNG vào tệp log, chỉ IN dòng quan trọng ra màn hình."""
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with open(logpath, "a", encoding="utf-8", errors="replace") as lf:
        for line in pr.stdout:
            lf.write(line)
            if verbose or IMPORTANT_RE.search(line):
                print(f"[{tag}] {line.rstrip()[:300]}", flush=True)


def fmt_h(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}"


def progress_line(plan: dict, out: Path, rdone: set, pdone: set, t_start: float, deadline: float, syncer, n0_local: int) -> str:
    done = done_ids(out) | set(rdone) | set(pdone)
    tasks = plan["tasks"]
    n_done = sum(1 for t in tasks if t["id"] in done)
    imgs = sum(t["hi"] - t["lo"] for t in tasks if t["id"] in done)
    tot = sum(t["hi"] - t["lo"] for t in tasks)
    el = now() - t_start
    new = len(done_ids(out)) - n0_local
    eta = ""
    if new >= 2:
        per = el / new
        rem = len(tasks) - n_done
        eta = f" · còn {rem} việc ≈ {fmt_h(rem * per)}"
        if now() + rem * per > deadline:
            eta += " (ca này không đủ: sẽ cần ca sau)"
    hf = ""
    if syncer is not None:
        hf = f" · HF {len(syncer.pushed)} lô" + (f" (lỗi đẩy {syncer.fail})" if syncer.fail else "")
    return f"[tiến độ] {n_done}/{len(tasks)} việc ({100 * n_done / max(1, len(tasks)):.0f} %) · {imgs:,}/{tot:,} ảnh{hf} · chạy {fmt_h(el)}{eta}"


def compute_assets_sha(root: Path, ckpt_sha: dict) -> str:
    ah = hashlib.sha256()
    for sp in sorted(root.glob("data/*/style_*.png")):
        ah.update(sp.relative_to(root).as_posix().encode()); ah.update(hashlib.sha256(sp.read_bytes()).digest())
    fdir = root / "code" / "font_diffusion" / "fonts"
    for fp in sorted(fdir.glob("*")) if fdir.exists() else []:
        ah.update(fp.name.encode()); ah.update(hashlib.sha256(fp.read_bytes()).digest())
    ah.update(json.dumps(ckpt_sha, sort_keys=True).encode()); ah.update((root / "fonts.json").read_bytes())
    return ah.hexdigest()


def cmd_preflight(a):
    """Kiểm 1 phút TRƯỚC khi chạy ca dài (in <= 8 dòng): gói nguyên vẹn, GPU, chỗ trống, HF ghi/đọc được và cùng kế hoạch. Thoát 2 nếu có ✗."""
    ok_all = True

    def rep(ok, msg):
        nonlocal ok_all
        ok_all = ok_all and bool(ok)
        print(("✓ " if ok else "✗ ") + msg, flush=True)
    pack = find_pack(a.pack)
    plan = load_plan(pack)
    man = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    n_img = sum(len(v["chars"]) for v in plan["books"].values())
    bad = verify_ckpt(man, pack / "ckpt") if (pack / "ckpt").exists() else ["thiếu thư mục ckpt"]
    ash = compute_assets_sha(pack, man.get("ckpt_sha256", {})) if (pack / "fonts.json").exists() else None
    rep(not bad and ash == man.get("assets_sha"), f"gói: {len(plan['tasks'])} việc, {n_img:,} ảnh, kế hoạch {plan['sha'][:8]}" + ("" if not bad and ash == man.get("assets_sha") else f" — GÓI HỎNG ({bad or 'mã băm tài sản lệch'}): gắn lại dataset"))
    if not a.stub and a.device in (None, "cuda"):
        try:
            import torch
            n = torch.cuda.device_count()
            names = sorted({torch.cuda.get_device_name(i) for i in range(n)})
        except Exception:  # noqa: BLE001
            n, names = 0, []
        rep(n >= 1, f"GPU: {n} × {', '.join(names)}" if n else "GPU: KHÔNG CÓ — Settings → Accelerator = GPU T4 x2")
    for pth in (a.tmp, str(Path(a.tmp).parent)):
        try:
            free = shutil.disk_usage(pth if Path(pth).exists() else "/").free / 1e9
            rep(free >= 3.0, f"chỗ trống quanh {pth}: {free:.0f} GB")
            break
        except OSError:
            continue
    if a.remote:
        try:
            remote = open_remote(a.remote)
        except SystemExit as e:
            rep(False, f"kho ngoài: {e}")
            remote = None
        except Exception as e:  # noqa: BLE001
            rep(False, f"kho ngoài: {type(e).__name__}: {str(e)[:160]}")
            remote = None
        if remote is not None:
            try:
                who = ""
                if getattr(remote, "kind", "") == "hf":
                    try:
                        who = " · tài khoản " + str(remote.api.whoami().get("name", "?"))
                    except Exception:  # noqa: BLE001
                        who = ""
                remote.put_text(f"_preflight/{time.strftime('%Y%m%d-%H%M%S')}.txt", "ok")        # ghi thử
                rsha, rash = remote.get_text("plan_sha.txt"), remote.get_text("assets_sha.txt")
                rdone = remote.list_done()
                same = (not rsha or rsha.strip() == plan["sha"]) and (not rash or not man.get("assets_sha") or rash.strip() == man["assets_sha"])
                rep(same, f"kho {a.remote}: ghi/đọc được{who} · đã có {len(rdone)}/{len(plan['tasks'])} việc" + ("" if same else " — KHO CHỨA KẾ HOẠCH/TÀI SẢN KHÁC: dùng repo mới"))
            except Exception as e:  # noqa: BLE001
                rep(False, f"kho {a.remote}: {type(e).__name__}: {str(e)[:160]}")
    else:
        rep(False, "KHÔNG có --remote: kết quả chỉ nằm ở phiên Kaggle (dễ mất khi reset) — thêm --remote hf:<user>/<repo>")
    print("PREFLIGHT " + ("ĐẠT" if ok_all else "KHÔNG ĐẠT"), flush=True)
    sys.exit(0 if ok_all else 2)


def cmd_status(a):
    """Tiến độ gọn (<= 5 dòng) từ kho ngoài (--remote) hoặc thư mục kết quả cục bộ (--out); dùng được trên Mac không cần tải ảnh."""
    pack = find_pack(a.pack)
    plan = load_plan(pack)
    if a.remote:
        done = open_remote(a.remote).list_done()
    else:
        done = done_ids(Path(a.out))
    tasks = plan["tasks"]
    n_done = sum(1 for t in tasks if t["id"] in done)
    imgs = sum(t["hi"] - t["lo"] for t in tasks if t["id"] in done)
    tot = sum(t["hi"] - t["lo"] for t in tasks)
    rem_img = tot - imgs
    per_book = {}
    for t in tasks:
        d = per_book.setdefault(t["book"], [0, 0])
        d[1] += 1
        d[0] += 1 if t["id"] in done else 0
    gpu_h = rem_img * 2.7 / 3600
    print(f"[status] {n_done}/{len(tasks)} việc ({100 * n_done / max(1, len(tasks)):.0f} %) · {imgs:,}/{tot:,} ảnh · còn ≈ {gpu_h:.1f} giờ GPU = {gpu_h / 2:.1f} giờ phiên (2 GPU, cận trên)", flush=True)
    print("  " + " · ".join(f"{b} {d[0]}/{d[1]}" for b, d in per_book.items()), flush=True)
    print("[status] ĐÃ XONG — chạy validate + p12 để chấm" if n_done == len(tasks) else "[status] chưa xong — chạy lại ô 'run' (tự bỏ qua việc đã có ở kho)", flush=True)


def cmd_run(a):
    t_start = now()
    pack = find_pack(a.pack)
    out = safe_dir(a.out)
    a.tmp = str(safe_dir(a.tmp)) if not a.stub else a.tmp
    plan = load_plan(pack)
    if not a.stub and a.device in (None, "cuda"):                  # cổng GPU: quên bật Accelerator thì dừng NGAY (trước đây chạy CPU im lặng và báo xong)
        try:
            import torch
            n_cuda = torch.cuda.device_count()
        except Exception:  # noqa: BLE001
            n_cuda = 0
        if n_cuda == 0:
            print("[run] KHÔNG CÓ GPU CUDA — bật Settings → Accelerator = GPU T4 x2 rồi chạy lại (hoặc thêm --device cpu/mps để thử trên máy không GPU)", flush=True)
            sys.exit(2)
    remote = open_remote(a.remote)
    man = None
    if not a.stub:
        man = setup_real(pack, Path(a.tmp))
        ensure_deps(man.get("pip_pins"))
    else:
        mp = pack / "manifest.json"
        man = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    (out / "status").mkdir(parents=True, exist_ok=True)
    env = env_snapshot(light=a.stub)
    (out / "status" / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[env] {json.dumps(env, ensure_ascii=False)}", flush=True)                 # /kaggle/tmp không được lưu sau phiên: để lại dấu vết ở log
    rep = cleanup_stale(out, a.ttl)
    if (out / "STOP").exists():
        (out / "STOP").unlink()
    sig = {"v": None}

    def _on_sig(signum, frame):                                                      # Cancel/SIGTERM: không chết ngay; dừng nhận việc mới, flush, ghi status
        sig["v"] = signum
        try:
            (out / "STOP").write_text(f"signal {signum}", encoding="utf-8")
        except OSError:
            pass
    signal.signal(signal.SIGTERM, _on_sig); signal.signal(signal.SIGINT, _on_sig)
    if remote:
        rsha = remote.get_text("plan_sha.txt")
        if rsha and rsha.strip() != plan["sha"]:
            print(f"[run] KẾ HOẠCH KHÁC kho ngoài ({rsha.strip()[:12]} ≠ {plan['sha'][:12]}) — dừng, không trộn", flush=True)
            sys.exit(2)
        if not rsha:
            remote.put_text("plan_sha.txt", plan["sha"])
        ash = man.get("assets_sha")                                                 # ảnh phong cách/phông/ckpt: plan_sha không bao phủ
        if ash:
            rash = remote.get_text("assets_sha.txt")
            if rash and rash.strip() != ash:
                print(f"[run] TÀI SẢN KHÁC kho ngoài (ảnh phong cách/phông/ckpt: {rash.strip()[:12]} ≠ {ash[:12]}) — dừng, không trộn; dùng kho HF mới", flush=True)
                sys.exit(2)
            if not rash:
                remote.put_text("assets_sha.txt", ash)
        rdone = remote.list_done()
    else:
        rdone = set()
    pdone = prev_done(a.prev, plan["sha"])
    remote_ids = set(rdone)                                                                   # chỉ phần THỰC SỰ có ở kho ngoài (cho Syncer)
    ldone = done_ids(out)
    (out / "status").mkdir(parents=True, exist_ok=True)
    rdone = rdone | pdone                                                                     # kho ngoài ∪ đầu ra các phiên trước
    (out / "status" / "remote_done.txt").write_text("\n".join(sorted(rdone)), encoding="utf-8")   # worker bỏ qua việc đã xong ở nơi khác
    todo = [t for t in plan["tasks"] if t["id"] not in ldone and t["id"] not in rdone]
    print(f"[run] kế hoạch {len(plan['tasks'])} việc | xong cục bộ {len(ldone)} | xong ở kho ngoài {len(rdone)} | còn {len(todo)} | dọn {rep}", flush=True)
    deadline = t_start + a.budget_hours * 3600 - a.safety_s
    syncer = None
    if remote:
        pushed = {rel_path(t) for t in remote_ids}
        syncer = Syncer(out, remote, a.sync_every, pushed, backoff=a.sync_backoff)
        syncer.start()
    bundler = None
    if a.bundle_dir:
        bundler = Bundler(out, Path(a.bundle_dir), a.session, a.bundle_every, pdone, plan["sha"])
        bundler.start()
    ngpu = a.workers
    if ngpu == 0:
        try:
            import torch
            ngpu = max(1, torch.cuda.device_count())
        except Exception:  # noqa: BLE001
            ngpu = 1
    procs, pumps = [], []
    logdir = Path(a.log_dir) if a.log_dir else (Path(a.bundle_dir) / "logs" if a.bundle_dir else out / "logs")
    n0_local = len(ldone)
    for k in range(ngpu):
        cmd = [sys.executable, os.path.abspath(__file__), "worker", "--pack", str(pack), "--out", str(out), "--worker", f"w{k}",
               "--deadline", str(deadline), "--est-s", str(a.est_s), "--ttl", str(a.ttl), "--tmp", str(a.tmp), "--gen-bs", str(a.gen_bs)]
        if a.stub:
            cmd += ["--stub", "--stub-ms", str(a.stub_ms), "--stub-fail", a.stub_fail, "--stub-crash", a.stub_crash]
        if a.crash_after and k == 0:
            cmd += ["--crash-after", str(a.crash_after)]
        if a.max_tasks:
            cmd += ["--max-tasks", str(a.max_tasks)]
        if a.device:
            cmd += ["--device", a.device]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k)) if not a.stub else dict(os.environ)
        env.update(QUIET_ENV)
        pr = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
        th = threading.Thread(target=_pump, args=(pr, logdir / f"worker-w{k}.log", f"w{k}", a.verbose), daemon=True)
        th.start()
        procs.append(pr); pumps.append(th)
    hard_at = deadline + a.hard_extra_s
    codes, hard = [None] * len(procs), False
    last_prog = now()
    while any(c is None for c in codes):
        for i, pr in enumerate(procs):
            if codes[i] is None:
                codes[i] = pr.poll()
        if any(c is None for c in codes) and now() > hard_at and not hard:
            hard = True
            print(f"[run] QUÁ HẠN DỪNG CỨNG ({a.hard_extra_s:.0f} s sau hạn nhận việc): dừng worker còn chạy", flush=True)
            for pr in procs:
                if pr.poll() is None:
                    pr.terminate()
            t_kill = now() + 60
            while any(pr.poll() is None for pr in procs) and now() < t_kill:
                time.sleep(1)
            for pr in procs:
                if pr.poll() is None:
                    pr.kill()
        time.sleep(1 if a.stub else 5)
        if a.progress_every and now() - last_prog >= a.progress_every:
            last_prog = now()
            print(progress_line(plan, out, rdone, pdone, t_start, deadline, syncer, n0_local), flush=True)
    for th in pumps:
        th.join(10)
    fin = syncer.stop() if syncer else True
    if bundler:
        bundler.stop()
    ldone = done_ids(out)
    all_done = {t["id"] for t in plan["tasks"]} <= (ldone | rdone)
    stops = {}
    for k in range(ngpu):
        wf = out / "status" / f"worker-w{k}.json"
        stops[f"w{k}"] = json.loads(wf.read_text(encoding="utf-8")).get("stop") if wf.exists() else "no_report"
    status = dict(signal=sig["v"], hard_stop=hard, plan_sha=plan["sha"], tasks_total=len(plan["tasks"]), tasks_done=len(ldone | rdone), tasks_done_local=len(ldone), bundles=(bundler.made if bundler else []),
                  complete=all_done, worker_exit_codes=codes, worker_stop=stops, sync_ok=fin, seconds=round(now() - t_start, 1),
                  budget_hours=a.budget_hours, deadline_epoch=round(deadline, 3),
                  stop_reason="complete" if all_done else ("guard" if 2 in codes else ("signal" if sig["v"] else ("hard_stop" if hard else ("max_tasks" if "max_tasks" in stops.values() else ("budget" if "budget" in stops.values() else "crash_or_error"))))))
    (out / "status").mkdir(parents=True, exist_ok=True)
    (out / "status" / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[run] {json.dumps(status, ensure_ascii=False)}", flush=True)
    if a.bundle_dir:                                                                 # /kaggle/tmp không được lưu: chép status (nhỏ) cạnh bundle
        try:
            shutil.copytree(out / "status", Path(a.bundle_dir) / f"status_{a.session}", dirs_exist_ok=True)
        except OSError as e:  # noqa: BLE001
            print(f"[run] không chép được status: {e}", flush=True)
    # mã thoát: 2 = lỗi hệ thống (cổng use_fst/kế hoạch/không GPU/tài sản lệch); 3 = KHÔNG làm được việc nào và chưa xong (hỏng thật: tránh "thành công giả");
    # còn lại 0 để phiên Kaggle được tính là THÀNH CÔNG (đầu ra gắn được vào phiên sau)
    if 2 in codes:
        sys.exit(2)
    if status["stop_reason"] == "crash_or_error" and not all_done and len(ldone) == 0:
        sys.exit(3)
    sys.exit(0)


def cmd_unbundle(a):
    r = unbundle(a.bundles, Path(a.dest))
    print(json.dumps(r, ensure_ascii=False))
    sys.exit(1 if r["conflicts"] else 0)


def cmd_validate(a):
    """Kiểm toàn bộ tar đã có theo kế hoạch + tổng hợp: số ảnh, chữ bỏ qua theo lý do, chữ không phông nào có (uncovered), còn thiếu bao nhiêu việc."""
    pack, out = Path(a.pack), Path(a.out)
    plan = load_plan(pack)
    errs, n = [], 0
    imgs, skipped, per_book = 0, {}, {}
    for t in plan["tasks"]:
        p = out / rel_path(t["id"])
        if p.exists():
            n += 1
            e = validate_shard(p, t, plan)
            errs += e
            if not e:
                with tarfile.open(p) as tf:
                    meta = json.loads(tf.extractfile("_meta.json").read())
                imgs += len(meta["generated"])
                for c, why in meta["skipped"].items():
                    skipped[why] = skipped.get(why, 0) + 1
                pb = per_book.setdefault(t["book"], dict(tasks=0, images=0))
                pb["tasks"] += 1; pb["images"] += len(meta["generated"])
    miss = [t["id"] for t in plan["tasks"] if not (out / rel_path(t["id"])).exists()]
    unc = {b: len(v["uncovered"]) for b, v in plan["books"].items() if v["uncovered"]}
    print(json.dumps(dict(tasks=len(plan["tasks"]), present=n, missing=len(miss), images=imgs, skipped_by_reason=skipped, uncovered_no_font=unc,
                          per_book=per_book, errors=errs[:20], n_errors=len(errs)), ensure_ascii=False))
    sys.exit(1 if errs else 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-plan"); p.add_argument("--books", required=True); p.add_argument("--fonts"); p.add_argument("--out", required=True)
    p.add_argument("--passes", nargs="+", default=["base:0", "base:1"]); p.add_argument("--shard", type=int, default=128)
    for name in ("run", "worker"):
        p = sub.add_parser(name)
        p.add_argument("--pack"); p.add_argument("--out", required=True); p.add_argument("--tmp", default="/kaggle/tmp/tn11")
        p.add_argument("--ttl", type=float, default=1800.0); p.add_argument("--est-s", type=float, default=2.66 * 128)
        p.add_argument("--gen-bs", type=int, default=32); p.add_argument("--stub", action="store_true"); p.add_argument("--stub-ms", type=float, default=0.0)
        p.add_argument("--stub-fail", default=""); p.add_argument("--stub-crash", default="")
        p.add_argument("--crash-after", type=int, default=0); p.add_argument("--device", default=None)
        p.add_argument("--max-tasks", type=int, default=0, help="mỗi worker chỉ làm N việc rồi thoát (chạy thử 15 phút đầu trên Kaggle)")
        if name == "run":
            p.add_argument("--remote"); p.add_argument("--budget-hours", type=float, default=11.0); p.add_argument("--safety-s", type=float, default=1200.0)
            p.add_argument("--workers", type=int, default=0); p.add_argument("--hard-extra-s", type=float, default=1800.0, help="quá hạn nhận việc + chừng này giây thì dừng cứng worker"); p.add_argument("--sync-every", type=float, default=300.0, help="đẩy tar mới lên kho ngoài mỗi chừng này giây (mặc định 5 phút: mất tối đa ~5 phút việc khi phiên chết)"); p.add_argument("--sync-backoff", type=float, default=30.0)
            p.add_argument("--bundle-dir", default=None, help="thư mục đầu ra Kaggle (/kaggle/working/...): chỉ vài tar lớn + chỉ mục")
            p.add_argument("--bundle-every", type=float, default=1800.0); p.add_argument("--prev", nargs="*", default=[], help="đầu ra các phiên trước (/kaggle/input/...)")
            p.add_argument("--progress-every", type=float, default=600.0, help="in một dòng tiến độ mỗi chừng này giây (0 = tắt)")
            p.add_argument("--log-dir", default=None, help="log đầy đủ của worker (mặc định <bundle-dir>/logs); màn hình chỉ hiện dòng quan trọng")
            p.add_argument("--verbose", action="store_true", help="in MỌI dòng worker ra màn hình")
            p.add_argument("--session", default=time.strftime("%Y%m%d-%H%M%S"))
        else:
            p.add_argument("--worker", required=True); p.add_argument("--deadline", type=float, required=True)
    p = sub.add_parser("pack"); p.add_argument("--spec", required=True); p.add_argument("--out", required=True)
    p.add_argument("--repo", default=str(Path(__file__).resolve().parents[3])); p.add_argument("--wrapper-fix", default=str(Path(__file__).with_name("fd_wrapper_fix.py")))
    p.add_argument("--passes", nargs="+", default=["base:0", "base:1"]); p.add_argument("--shard", type=int, default=128); p.add_argument("--no-ckpt", action="store_true")
    p = sub.add_parser("preflight"); p.add_argument("--pack"); p.add_argument("--remote"); p.add_argument("--device", default=None); p.add_argument("--tmp", default="/kaggle/tmp/tn11")
    p.add_argument("--stub", action="store_true")
    p = sub.add_parser("status"); p.add_argument("--pack"); p.add_argument("--remote"); p.add_argument("--out", default="/kaggle/tmp/tn11_full")
    p = sub.add_parser("unbundle"); p.add_argument("--bundles", nargs="+", required=True); p.add_argument("--dest", required=True)
    p = sub.add_parser("validate"); p.add_argument("--pack", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"make-plan": cmd_make_plan, "run": cmd_run, "worker": cmd_worker, "validate": cmd_validate, "pack": cmd_pack, "unbundle": cmd_unbundle,
     "preflight": cmd_preflight, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
