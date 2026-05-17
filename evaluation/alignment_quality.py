"""Muc 4.4 — danh gia he thong gan nhan: AMR (Alignment Match Rate) + CER.

Yeu cau:
  evaluation/gt/<book>/<page>.json   # ground truth do nguoi gan tay
  {
    "syllables": ["nhi", "nguyet", "nhi", "thap", ...],
    "chars":     ["二",  "月",     "二",  "十",   ...]   # chu Nom dung
  }

Script doc labels.csv predicted cho trang do, so voi GT, in AMR + CER.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "dataset" / "all" / "labels.csv"
GT_DIR = REPO / "evaluation" / "gt"
OUT = REPO / "evaluation" / "reports" / "alignment_quality.md"


def edit_distance(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    return dp[n][m]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", default=str(GT_DIR))
    args = p.parse_args()
    gt_dir = Path(args.gt_dir)

    if not gt_dir.exists() or not any(gt_dir.rglob("*.json")):
        print(f"!! Khong tim thay ground truth o {gt_dir}")
        print("Tao file dang:")
        print('  evaluation/gt/SachThanhTruyen2/page_0012.json')
        print('  { "chars": ["二","月","二","十", ...] }')
        return

    df = pd.read_csv(LABELS)
    results = []
    for gt_file in sorted(gt_dir.rglob("*.json")):
        book = gt_file.parent.name
        page = gt_file.stem
        with open(gt_file, encoding="utf-8") as f:
            gt = json.load(f)
        gt_chars = list(gt.get("chars", []))

        pred = df[(df["source"] == book) & (df["page"] == page)].sort_values(
            ["bbox", "crop_file"]
        )
        pred_chars = pred["nom_char"].astype(str).tolist()

        if not pred_chars:
            print(f"!! Khong co prediction cho {book}/{page}")
            continue

        ed = edit_distance(pred_chars, gt_chars)
        cer = ed / max(len(gt_chars), 1)
        amr = (
            sum(a == b for a, b in zip(pred_chars, gt_chars))
            / max(len(gt_chars), 1)
        )
        results.append({
            "book": book,
            "page": page,
            "pred_len": len(pred_chars),
            "gt_len": len(gt_chars),
            "edit_dist": ed,
            "AMR": round(amr * 100, 2),
            "CER": round(cer * 100, 2),
        })

    if not results:
        print("Khong co cap (pred, GT) nao tinh duoc.")
        return

    rep = pd.DataFrame(results)
    overall = {
        "pages": len(rep),
        "AMR_mean": round(rep["AMR"].mean(), 2),
        "CER_mean": round(rep["CER"].mean(), 2),
        "AMR_weighted": round(
            (rep["AMR"] * rep["gt_len"]).sum() / rep["gt_len"].sum(), 2
        ),
        "CER_weighted": round(
            (rep["CER"] * rep["gt_len"]).sum() / rep["gt_len"].sum(), 2
        ),
    }

    lines = [
        "# Alignment quality (muc 4.4)",
        "",
        f"Danh gia tren {overall['pages']} trang co ground truth.",
        "",
        "## Per-page",
        "",
        rep.to_markdown(index=False),
        "",
        "## Overall",
        "",
        f"- AMR trung binh (theo trang): **{overall['AMR_mean']}%**",
        f"- AMR weighted (theo so ky tu): **{overall['AMR_weighted']}%**",
        f"- CER trung binh: **{overall['CER_mean']}%**",
        f"- CER weighted: **{overall['CER_weighted']}%**",
        "",
        "## Ghi chu",
        "",
        "- AMR (Alignment Match Rate) = ti le ky tu pred dung vi tri va dung nhan.",
        "- CER (Character Error Rate) = edit-distance(pred, gt) / len(gt). Cang thap cang tot.",
        "- Can chuan bi them GT cho moi sach (>=1 trang) de so sanh lien sach.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
