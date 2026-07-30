"""Đánh giá AUC bắt-lỗi THẬT của checkpoint ArcFace mới bằng 846 verdict NGƯỜI
đã có sẵn (dataset_out/ground_truth/verdicts_reanchored.csv) — KHÔNG cần audit
lại. Đối chiếu trực tiếp với con số CŨ đã đo bằng người (bank_cos AUC = 0.566-
0.572, docs/BANG_SO_LIEU_CHINH_THUC.md) — không dùng proxy auto-label.

Verdicts cũ dùng tên sách trước khi đổi ("yen2/yen4/yen11"); dữ liệu hiện tại
dùng "stt2/stt4/stt11". remap() đổi tiền tố; đã kiểm chứng 828/828 crop còn tồn
tại và 825/828 (99.6%) label khớp label_old tại đường dẫn ánh xạ trước khi tin
dùng cách ghép này.

Usage:
    python3 ArcFace/eval_human_verdicts.py [--ckpt ArcFace/checkpoints/best.pt]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ArcFace"))
from model import NomEmbedder      # noqa: E402
from dataset import NomCropDataset  # noqa: E402

BOOK_MAP = {"yen2": "stt2", "yen4": "stt4", "yen11": "stt11"}


def remap(fname: str) -> str:
    m = re.match(r"^(yen\d+)_(.*)$", fname)
    if not m:
        return fname
    return BOOK_MAP.get(m.group(1), m.group(1)) + "_" + m.group(2)


def _auc(scores: list[float], is_pos: list[bool]) -> float:
    """AUC: xác suất 1 mẫu positive ('correct') có điểm CAO HƠN 1 mẫu negative."""
    s = torch.tensor(scores, dtype=torch.float64)
    y = torch.tensor(is_pos, dtype=torch.bool)
    p, n = s[y], s[~y]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    return float((ranks[y].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(REPO / "ArcFace/checkpoints/best.pt"))
    ap.add_argument("--verdicts", default=str(REPO / "dataset_out/ground_truth/verdicts_reanchored.csv"))
    ap.add_argument("--src-root", default=str(REPO / "dataset_out"))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available()
                       else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ck = torch.load(args.ckpt, map_location=dev)
    lab2idx = ck["classes"]
    emb = NomEmbedder(ck["embed_dim"], pretrained=False, arch=ck["arch"]).to(dev)
    emb.load_state_dict(ck["backbone"]); emb.eval()
    Wn = F.normalize(ck["head"]["W"].to(dev).float(), dim=1)

    rows = [r for r in csv.DictReader(open(args.verdicts, encoding="utf-8"))
            if r["image_new"] and r["verdict"] in ("correct", "wrong_label", "wrong_image")]

    src_root = Path(args.src_root)
    paths, label_idx, is_correct = [], [], []
    skipped_no_file = skipped_no_class = 0
    for r in rows:
        tier_dir, fname = r["image_new"].split("/", 1)
        new_img = f"{tier_dir}/{remap(fname)}"
        p = src_root / new_img
        if not p.exists():
            skipped_no_file += 1
            continue
        lab = r["label_old"]
        if lab not in lab2idx:
            skipped_no_class += 1
            continue
        paths.append(p)
        label_idx.append(lab2idx[lab])
        is_correct.append(r["verdict"] == "correct")

    n_wrong = len(is_correct) - sum(is_correct)
    print(f"[human-verdict eval] dùng {len(paths)}/{len(rows)} verdict "
          f"(bỏ {skipped_no_file} thiếu file, {skipped_no_class} nhãn ngoài từ điển {len(lab2idx)} lớp)")
    print(f"  correct={sum(is_correct)}  wrong={n_wrong}")
    if n_wrong < 5:
        print("  [CẢNH BÁO] quá ít mẫu 'wrong' -> AUC không ổn định, chỉ mang tính tham khảo.")

    ds = NomCropDataset(paths, label_idx, img=ck["img"], train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=2)

    scores = []
    with torch.no_grad():
        for x, yy, _ in dl:
            x, yy = x.to(dev), yy.to(dev)
            e = emb(x)
            cos = (e * Wn[yy]).sum(1)          # cosine(embedding crop, prototype của label_old)
            scores += cos.cpu().tolist()

    a = _auc(scores, is_correct)
    print(f"\n[human-verdict eval] AUC bắt-lỗi THẬT (checkpoint mới, verdict NGƯỜI) = {a:.3f}")
    print("  So với CŨ: bank_cos AUC bắt-lỗi = 0.566-0.572 (đo trên cùng loại verdict NGƯỜI, "
          "docs/BANG_SO_LIEU_CHINH_THUC.md)")


if __name__ == "__main__":
    main()
