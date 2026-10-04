"""v_truy_nguon_08 (04/10) — "HỒ SƠ MẪU GIẤY" của p04/p05 (K02, công nghệ 1): hồ sơ lấy từ MỘT trang/sách có đại diện không? 0 API, CPU, chỉ ĐỌC.

p04/p05 extract_book_paper_model: trang thứ 10 (sorted(pages)[min(10, len-1)]), ngưỡng Otsu, bg = trung vị điểm >= ngưỡng, ink = trung vị điểm < ngưỡng.
Script dùng ĐÚNG phép tính đó (sao nguyên văn phần thống kê) cho trang p05 đã dùng và cho 8 trang ngẫu nhiên khác của cùng sách (seed cố định), rồi báo
trung bình, độ lệch chuẩn giữa các trang, và có phải "nền phẳng 255" (HQC2 ≈ HQC1 theo cấu trúc) hay không.
Ra: measure_out/_tn11/verify/truy_nguon/ho_so_giay.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "truy_nguon"


def stats(png: Path):
    g = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = g[g >= t]; ink = g[g < t]
    return dict(bg=float(np.median(paper)) if len(paper) else 240.0, ink=float(np.median(ink)) if len(ink) else 60.0,
                bg_std=float(np.std(paper)) if len(paper) else 10.0, otsu=float(t), ink_frac=float((g < t).mean()))


def main():
    res = {}
    rng = np.random.default_rng(20261004)
    for b in T.ORDER:
        cfg = T.BOOKS[b]
        pages = sorted((T.REPO / cfg["prep"] / "pages").glob("*.png"))
        p05 = pages[min(10, len(pages) - 1)]
        s05 = stats(p05)
        pick = rng.choice(len(pages), size=min(8, len(pages)), replace=False)
        S = [stats(pages[i]) for i in pick]
        r = dict(trang_p05=p05.name, p05=s05, so_trang_khac=len(S),
                 bg_tb=round(float(np.mean([s["bg"] for s in S])), 1), bg_sd=round(float(np.std([s["bg"] for s in S])), 1),
                 ink_tb=round(float(np.mean([s["ink"] for s in S])), 1), ink_sd=round(float(np.std([s["ink"] for s in S])), 1),
                 bg_min=round(float(min(s["bg"] for s in S)), 1), bg_max=round(float(max(s["bg"] for s in S)), 1),
                 ink_min=round(float(min(s["ink"] for s in S)), 1), ink_max=round(float(max(s["ink"] for s in S)), 1),
                 nen_phang_255=bool(all(s["bg"] >= 250 for s in S) and s05["bg"] >= 250))
        res[b] = r
        print(f"{b:6s} p05: bg {s05['bg']:5.1f} ink {s05['ink']:5.1f} | 8 trang: bg {r['bg_tb']:5.1f}±{r['bg_sd']:4.1f} [{r['bg_min']},{r['bg_max']}] "
              f"ink {r['ink_tb']:5.1f}±{r['ink_sd']:4.1f} [{r['ink_min']},{r['ink_max']}] | nền phẳng 255: {r['nen_phang_255']}")
    (OUT / "ho_so_giay.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
