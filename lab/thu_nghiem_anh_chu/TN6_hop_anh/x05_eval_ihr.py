"""TN6 x05 — đo một bản dựng IHR (LucVanTien1916 / TruyenKieu1872) trên NHÃN NGƯỜI IHR-NomDB: nhãn V1+ (chữ người cùng vị trí),
KHE theo mô hình khe người của harness (pipeline.gold_exact.eval_ihr.Slot: mẫu vị trí trong hộp CỘT người vẽ + dòng kim thô; áp
cho TÂM bbox), đúng hai vế, số GOLD. Chỉ ĐỌC mã gold_exact. 0 API, không mở ảnh.

  .venv/bin/python lab/thu_nghiem_anh_chu/TN6_hop_anh/x05_eval_ihr.py --book LucVanTien1916 --variant visual_dp
  (--variant current = prepared/<Bộ>/dataset_out/labels_gated.csv hiện hành)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pipeline.gold_exact.eval_ihr import Slot            # noqa: E402 (chỉ đọc)
from pipeline.gold_exact.common import Assets, set_lexicon, var_eq_plus  # noqa: E402
set_lexicon(Assets())


def run(book, variant):
    lab = (REPO / f"prepared/{book}/dataset_out/labels_gated.csv" if variant == "current"
           else REPO / f"measure_out/_tn6/{book}/{variant}/dataset_out/labels_gated.csv")
    od = REPO / f"measure_out/_tn6/{book}/{variant}/eval"; od.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(REPO / "config/gold_exact.yaml", encoding="utf-8"))
    S = Slot(book, cfg["ihr_slot_template"][book])
    L = pd.read_csv(lab, dtype=str, keep_default_na=False)
    gt, sl = [], []
    for pg, col, si, bb in zip(L.page, L.column, L.syl_idx, L.bbox):
        try:
            b = json.loads(bb); cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        except Exception:  # noqa: BLE001
            cx = cy = None
        try:
            g = S.gt_char(pg, int(float(col)), int(float(si)))
            s = S.slot(pg, int(float(col)), int(float(si)), cx, cy)
        except Exception:  # noqa: BLE001
            g, s = "", ""
        gt.append(g); sl.append(s)
    L["gt_char"] = gt; L["slot"] = sl
    L["y_lab"] = [bool(g) and var_eq_plus(a, g) for a, g in zip(L.label, L.gt_char)]
    out = dict(book=book, variant=variant, labels=str(lab.relative_to(REPO)), tiers=L.tier.value_counts().to_dict(),
               box_source=L.box_source.value_counts().to_dict())
    for name in ("GOLD", "SYLLABLE", "REVIEW", "GOLD_text_only", "ALL"):
        t = L if name == "ALL" else L[L.tier == name]
        t = t[t.gt_char != ""]
        if not len(t):
            continue
        out[name] = dict(n=int(len(t)), lab_v1p=round(float(t.y_lab.mean()), 4),
                         slot1=round(float((t.slot == "1").mean()), 4), slot0=round(float((t.slot == "0").mean()), 4),
                         slot_undet=round(float((t.slot == "").mean()), 4),
                         both=round(float((t.y_lab & (t.slot == "1")).mean()), 4))
    json.dump(out, open(od / "tn6_ihr_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
    L[["page", "column", "nom_idx", "syl_idx", "tier", "label", "gt_char", "slot", "y_lab", "box_source", "bbox"]].to_pickle(od / "ihr_join.pkl")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--variant", required=True)
    a = ap.parse_args()
    print(json.dumps(run(a.book, a.variant), ensure_ascii=False, default=str))
