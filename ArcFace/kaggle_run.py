"""One-cell Kaggle entrypoint: train -> export -> evaluate.

On Kaggle, upload the bundled ArcFace/data as a Dataset; it mounts at
/kaggle/input/<slug>/. Point --data at it (auto-detected if it contains
manifest.csv). Checkpoints + logs go to /kaggle/working (persisted as output).

    !python ArcFace/kaggle_run.py                        # sensible defaults (SAM, page-disjoint)
    !python ArcFace/kaggle_run.py --split lobo --holdout stt4   # cross-book generalisation
    !python ArcFace/kaggle_run.py --epochs 40 --k 3 --sampler confusion
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def autodetect_data(given: str) -> str:
    if given and (Path(given) / "manifest.csv").exists():
        return given
    for base in ("/kaggle/input", str(HERE)):
        b = Path(base)
        if not b.exists():
            continue
        hit = list(b.glob("**/manifest.csv"))
        if hit:
            return str(hit[0].parent)
    return str(HERE / "data")


def run(mod_args):
    print("\n$ python", " ".join(mod_args), flush=True)
    subprocess.run([sys.executable, *mod_args], cwd=str(HERE), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--out", default="/kaggle/working/checkpoints"
                    if Path("/kaggle/working").exists() else str(HERE / "checkpoints"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sampler", default="balanced", choices=["balanced", "confusion"])
    ap.add_argument("--split", default="page_disjoint", choices=["page_disjoint", "lobo"])
    ap.add_argument("--holdout", default="")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--swa", action="store_true")
    ap.add_argument("--hf-repo", default="", help="push+resume via this HF repo (Kaggle-reset safe)")
    ap.add_argument("--hf-token", default="")
    args = ap.parse_args()

    data = autodetect_data(args.data)
    print(f"[kaggle] data={data}\n[kaggle] out={args.out}"
          + (f"\n[kaggle] hf-repo={args.hf_repo} (resume+push ON)" if args.hf_repo else ""))
    common = ["--data", data, "--split", args.split, "--holdout", args.holdout]
    hf = (["--hf-repo", args.hf_repo] + (["--hf-token", args.hf_token] if args.hf_token else [])) \
        if args.hf_repo else []

    train = ["train.py", *common, *hf, "--out", args.out, "--epochs", str(args.epochs),
             "--batch", str(args.batch), "--k", str(args.k), "--sampler", args.sampler]
    if not args.no_sam:
        train.append("--sam")
    if args.swa:
        train.append("--swa")
    run(train)
    run(["export_checkpoint.py", "--data", data, *hf,
         "--ckpt", str(Path(args.out) / "train_best.pt"),
         "--out", str(Path(args.out) / "best.pt")])
    run(["evaluate.py", *common, "--ckpt", str(Path(args.out) / "best.pt")])
    print(f"\n[kaggle] DONE. Deploy: {Path(args.out) / 'best.pt'} -> repo nom-embed/best.pt")


if __name__ == "__main__":
    main()
