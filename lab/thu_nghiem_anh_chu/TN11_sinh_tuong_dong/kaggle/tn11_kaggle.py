"""TN11 trên Kaggle — tinh chỉnh FontDiffuser theo nét từng sách Borg rồi sinh glyph ứng viên (GPU T4, 0 API kim).

Chạy trong notebook Kaggle (GPU T4 x2, Internet ON), một ô:
    !python /kaggle/input/<tên-dataset>/tn11_kaggle.py
Tuỳ chọn: --steps 3000 --save-at 1000 3000 --bs 8 --accum 2 --books B34 B18

Việc làm, mỗi sách (B34, B18; 2 GPU thì chạy song song mỗi sách một GPU):
  1. Tinh chỉnh UNet + bộ mã phong cách của FontDiffuser (bộ mã nội dung đóng băng) trên cặp (glyph phông NomNaTong, crop ô
     neo TỰ ĐỘNG của sách) — đúng công thức p04 bản máy: MSE dự đoán nhiễu, bỏ điều kiện 10 %, AdamW lr 1e-5, clip 1.0.
     Lưu mốc ở --save-at; có last.pt để chạy tiếp nếu phiên Kaggle bị ngắt.
  2. Sinh glyph cho mọi chữ ứng viên của 200 ô đo (chars.txt) với: base (checkpoint gốc, one-shot), ft<bước> (đã tinh chỉnh);
     ảnh phong cách style_0 / style_1 (ô neo điển hình của chính sách).
  3. Nén /kaggle/working/tn11_ket_qua.zip (ảnh PNG + nhật ký loss) -> tải về, đưa lại cho Claude chấm bằng p07 trên máy.
Không có nhãn người nào trong gói dữ liệu (chỉ crop ô neo tự động + danh sách chữ cần sinh).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PIP = {"diffusers": "diffusers", "pygame": "pygame", "info_nce": "info-nce-pytorch", "kornia": "kornia", "einops": "einops",
       "fontTools": "fonttools", "safetensors": "safetensors", "skimage": "scikit-image", "cv2": "opencv-python-headless", "accelerate": "accelerate"}
WORK = Path(os.environ.get("TN11_WORK", "/kaggle/working/tn11"))
OUTZIP = WORK.parent / "tn11_ket_qua.zip"
DEV = os.environ.get("TN11_DEVICE", "cuda")          # chỉ để thử trên máy (mps/cpu)


def find_input() -> Path:
    if os.environ.get("TN11_INPUT"):
        return Path(os.environ["TN11_INPUT"])
    for d in sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*")) + sorted(glob.glob("/kaggle/input/*/*/*")):
        if os.path.isfile(os.path.join(d, "tn11_manifest.json")):
            return Path(d)
    raise SystemExit("Không thấy tn11_manifest.json trong /kaggle/input — đã gắn dataset TN11 vào notebook chưa?")


def ensure_pip():
    miss = []
    for mod, pkg in PIP.items():
        try:
            importlib.import_module(mod)
        except Exception:
            miss.append(pkg)
    if miss:
        print("pip install", " ".join(miss), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *miss])


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def setup(IN: Path) -> dict:
    man = json.loads((IN / "tn11_manifest.json").read_text(encoding="utf-8"))
    if not (WORK / "font_diffusion" / "src").exists():
        shutil.copytree(IN / "code", WORK, dirs_exist_ok=True)
    ck = WORK / "font_diffusion" / "ckpt" / "PROD"
    ck.mkdir(parents=True, exist_ok=True)
    for fn, h in man["ckpt_sha256"].items():
        dst = ck / fn
        if not dst.exists():
            shutil.copy2(IN / "ckpt" / fn, dst)
        if sha256(dst) != h:
            raise SystemExit(f"checkpoint {fn} sai mã băm — tải lại dataset")
    sys.path.insert(0, str(WORK)); sys.path.insert(0, str(WORK / "font_diffusion"))
    return man


def load_gen(cache_dir: Path, bs: int):
    from fd_wrapper_fix import FixedFontDiffusionGenerator as FontDiffusionGenerator   # use_fst=False (đường chuẩn)
    ck = str(WORK / "font_diffusion" / "ckpt" / "PROD")
    g = FontDiffusionGenerator(ckpt_dir=ck, phase1_ckpt_dir=ck, font_path=str(WORK / "font_diffusion/fonts/NomNaTong-Regular.ttf"),
                               cache_dir=str(cache_dir), device=DEV, batch_size=bs)
    g._load_pipeline()
    g.pipe.guidance_scale = 2.0
    return g


def train_book(book: str, IN: Path, a, g, base_sd) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from inference.sample_optimized import get_content_transform, get_style_transform
    from src.builders.build import build_ddpm_scheduler
    from src.tools.utils import ttf2im
    ftd = WORK / "ft" / book
    ftd.mkdir(parents=True, exist_ok=True)
    final = ftd / f"ft{a.steps}.pt"
    log_f = WORK / "out" / book / "train_log.json"
    log_f.parent.mkdir(parents=True, exist_ok=True)
    if final.exists() and all((ftd / f"ft{s}.pt").exists() for s in a.save_at):
        print(f"[{book}] đã có mô hình tinh chỉnh — bỏ qua học", flush=True)
        return json.loads(log_f.read_text()) if log_f.exists() else {}
    m = g.pipe.model; args = g.args; dev = DEV
    m.unet.load_state_dict(base_sd["unet"]); m.style_encoder.load_state_dict(base_sd["style_encoder"])
    Z = np.load(IN / "data" / book / "pairs.npz"); X, L = Z["X"], Z["L"]
    acp = build_ddpm_scheduler(args).alphas_cumprod.to(torch.float32)
    font = g.font_manager.get_font(g.font_manager.get_font_names()[0])
    ct = get_content_transform(args.content_image_size); stt = get_style_transform(args.style_image_size)
    cont = {}
    for c in set(L.tolist()):
        try:
            im = ttf2im(font=font, char=c)
        except Exception:
            im = None
        if im is not None and np.asarray(im).min() < 200:
            cont[c] = ct(im)
    ok = np.array([l in cont for l in L]); X, L = X[ok], L[ok]
    tgt = torch.stack([stt(Image.fromarray(x).convert("RGB")) for x in X])
    by = {}
    for k, l in enumerate(L):
        by.setdefault(l, []).append(k)
    others = np.arange(len(L))
    print(f"[{book}] học {len(L)} cặp, {len(cont)} chữ; {a.steps} bước × lô {a.bs} × tích luỹ {a.accum}", flush=True)
    for p in m.content_encoder.parameters():
        p.requires_grad_(False)
    params = [p for p in list(m.unet.parameters()) + list(m.style_encoder.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    step0, hist = 0, []
    last = ftd / "last.pt"
    if last.exists():
        st = torch.load(last, map_location="cpu")
        m.unet.load_state_dict(st["unet"]); m.style_encoder.load_state_dict(st["style_encoder"]); opt.load_state_dict(st["opt"])
        step0, hist = st["step"], st["hist"]
        print(f"[{book}] chạy tiếp từ bước {step0}", flush=True)
    m.train()
    rng = np.random.default_rng(step0); torch.manual_seed(step0)
    t0 = time.time(); run = []
    for step in range(step0 + 1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            idx = rng.integers(0, len(L), a.bs)
            sidx = []
            for i in idx:
                while True:
                    j = int(rng.choice(others))
                    if L[j] != L[i]:
                        break
                sidx.append(j)
            x0 = tgt[idx]; st_im = tgt[sidx].clone(); ci = torch.stack([cont[L[i]] for i in idx])
            drop = torch.rand(a.bs) < 0.1
            ci[drop] = 1; st_im[drop] = 1
            ts = torch.randint(0, 1000, (a.bs,))
            eps = torch.randn_like(x0)
            ab = acp[ts].view(-1, 1, 1, 1)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            pred = m(xt.to(dev), ts.to(dev), [ci.to(dev), st_im.to(dev)],
                     content_encoder_downsample_size=args.content_encoder_downsample_size, version="V3")
            loss = torch.nn.functional.mse_loss(pred.float(), eps.to(dev)) / a.accum
            loss.backward()
            run.append(float(loss) * a.accum)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 100 == 0:
            hist.append([step, float(np.mean(run[-100 * a.accum:]))])
            print(f"[{book}] bước {step}: loss {hist[-1][1]:.4f} [{time.time() - t0:.0f}s]", flush=True)
        if step in a.save_at:
            torch.save({"unet": m.unet.state_dict(), "style_encoder": m.style_encoder.state_dict()}, ftd / f"ft{step}.pt")
        if step % 500 == 0 or step == a.steps:
            torch.save({"unet": m.unet.state_dict(), "style_encoder": m.style_encoder.state_dict(), "opt": opt.state_dict(),
                        "step": step, "hist": hist}, last)
            log_f.write_text(json.dumps(dict(book=book, n_pairs=int(len(L)), n_chars=len(cont), hist=hist), ensure_ascii=False))
    m.eval()
    return json.loads(log_f.read_text())


def gen_book(book: str, IN: Path, a, g, base_sd):
    import torch
    chars = [c for c in (IN / "data" / book / "chars.txt").read_text(encoding="utf-8").split("\n") if c]
    if a.max_chars:
        chars = chars[: a.max_chars]
    out = WORK / "out" / book
    plan = [("base", 0), ("base", 1)] + [(f"ft{s}", j) for s in a.save_at for j in ((0, 1) if s == max(a.save_at) else (0,))]
    for model, j in plan:
        sd = base_sd if model == "base" else torch.load(WORK / "ft" / book / f"{model}.pt", map_location="cpu")
        g.pipe.model.unet.load_state_dict(sd["unet"]); g.pipe.model.style_encoder.load_state_dict(sd["style_encoder"])
        g.pipe.model.eval()
        g.cache_dir = out
        torch.manual_seed(0)
        t0 = time.time()
        g.generate(chars, str(IN / "data" / book / f"style_{j}.png"), style_name=f"{model}_s{j}")
        n = len(list((out / f"{model}_s{j}").glob("U+*.png")))
        print(f"[{book}] sinh {model} phong cách {j}: {n} ảnh [{time.time() - t0:.0f}s]", flush=True)


def run_book(book: str, a):
    import torch
    IN = find_input()
    setup(IN)
    g = load_gen(WORK / "out" / book, a.gen_bs)
    base_sd = {"unet": {k: v.detach().cpu().clone() for k, v in g.pipe.model.unet.state_dict().items()},
               "style_encoder": {k: v.detach().cpu().clone() for k, v in g.pipe.model.style_encoder.state_dict().items()}}
    try:
        train_book(book, IN, a, g, base_sd)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(f"[{book}] hết bộ nhớ GPU — chạy lại với --bs 4 --accum 4 (mốc last.pt được giữ)")
    gen_book(book, IN, a, g, base_sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", default=["B34", "B18"])
    ap.add_argument("--book", default=None, help="(nội bộ) chạy một sách trong tiến trình con")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--save-at", type=int, nargs="+", default=[1000, 3000])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gen-bs", type=int, default=32)
    ap.add_argument("--max-chars", type=int, default=0, help="(thử) chỉ sinh N chữ đầu")
    a = ap.parse_args()
    a.save_at = sorted(set(a.save_at) | {a.steps})
    if a.book:
        run_book(a.book, a)
        return
    ensure_pip()
    IN = find_input()
    man = setup(IN)
    print("Gói dữ liệu:", json.dumps(man["books"], ensure_ascii=False), flush=True)
    import torch
    ngpu = torch.cuda.device_count()
    print(f"GPU: {ngpu} × {torch.cuda.get_device_name(0) if ngpu else '—'}", flush=True)
    if ngpu == 0 and DEV == "cuda":
        raise SystemExit("Cần bật GPU (Settings → Accelerator → GPU T4 x2)")
    common = ["--steps", str(a.steps), "--bs", str(a.bs), "--accum", str(a.accum), "--lr", str(a.lr), "--gen-bs", str(a.gen_bs), "--max-chars", str(a.max_chars),
              "--save-at", *map(str, a.save_at)]
    t0 = time.time()
    if ngpu >= 2 and len(a.books) >= 2:
        procs = []
        for k, b in enumerate(a.books):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k % ngpu))
            logf = open(WORK / f"log_{b}.txt", "w")
            procs.append((b, subprocess.Popen([sys.executable, __file__, "--book", b, *common], env=env,
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), logf))
        import threading

        def pump(b, p, f):
            for line in p.stdout:
                f.write(line); f.flush()
                if line.startswith(f"[{b}]") or "Error" in line or "Traceback" in line:
                    print(line, end="", flush=True)
        th = [threading.Thread(target=pump, args=x) for x in procs]
        [t.start() for t in th]; [t.join() for t in th]
        bad = [b for b, p, _ in procs if p.wait() != 0]
    else:
        bad = []
        for b in a.books:
            if subprocess.call([sys.executable, __file__, "--book", b, *common]) != 0:
                bad.append(b)
    with zipfile.ZipFile(OUTZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((WORK / "out").rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(WORK / "out"))
        for f in sorted(WORK.glob("log_*.txt")):
            z.write(f, f.name)
    print(f"\nXONG sau {(time.time() - t0) / 60:.0f} phút. Lỗi: {bad or 'không'}. Tải về: {OUTZIP} "
          f"({OUTZIP.stat().st_size / 1e6:.0f} MB) — bảng Output bên phải.", flush=True)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
