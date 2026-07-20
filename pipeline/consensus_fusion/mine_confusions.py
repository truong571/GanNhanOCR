"""Truy vết CONFUSION HỆ THỐNG toàn corpus — nơi kim+dict "đồng thuận" nhưng cùng sai.

Bối cảnh (đo ở [[s3-nondiscriminative-systematic-confusion]]): lỗi nhãn KHÔNG ngẫu nhiên
mà là confusion hệ thống + tương quan (㝵/người: 8/24=33% sai, 765 crop). Không model nào
"độc lập" bắt được vì tất cả cùng học/chứa lỗi. Cách bắt duy nhất đáng tin = neo vào nhãn
người chấm rồi PHÓNG ra toàn corpus theo chữ ký (syllable → label).

Pass này gom mọi cặp (syllable, label) mức char (GOLD/SILVER) và với mỗi cặp báo:
  population    số crop toàn corpus mang cặp này  (= tác động nếu cặp sai)
  audit_n/wrong số crop của cặp này đã được người chấm + số bị chấm SAI
  audit_rate    tỉ lệ sai đo được (chỉ khi audit_n>0)
  exp_wrong     audit_rate × population  = số nhãn sai kỳ vọng của cặp (khi có audit)
  ocr_chars     các ocr_char thô nuôi cặp này (confusion nguồn)
  dict_ok       label có nằm trong readings từ điển của syllable? (True = S1∩S2 "đồng thuận")

Xếp hạng: cặp ĐÃ audit & có sai lên đầu theo exp_wrong (rủi ro đã chứng minh, tác động lớn),
rồi tới cặp CHƯA audit theo population (rủi ro chưa biết, tác động lớn → audit tiếp theo).

Chạy:  .venv/bin/python -m pipeline.consensus_fusion.mine_confusions [--top 40]
Xuất:  dataset_out/fusion/confusions.csv
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.text.dictionary import load_qn_to_nom

REPO = Path(__file__).resolve().parents[2]
_WRONG = {"wrong_label", "wrong_image"}


def _audit_verdict_by_image() -> dict[str, str]:
    gt = REPO / "dataset_out" / "ground_truth"
    id2img = {}
    with open(gt / "audit_gold" / "manifest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                m = json.loads(line)
                id2img[str(m["item_id"])] = str(m["image"])
    out = {}
    for f in sorted(glob.glob(str(gt / "verdicts_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    img = id2img.get(str(r["item_id"]))
                    if img is not None:
                        out[img] = str(r["verdict"])   # image -> verdict
    return out


def run(top: int) -> pd.DataFrame:
    root = REPO / "dataset_out"
    # Base = labels_remediated (chứa đủ 846 crop audited, gồm yen4/yen11 mà labels.csv
    # thiếu). Dedup theo image (bug duplicate-export). Chỉ nhãn char còn tin (GOLD/SILVER).
    df = pd.read_csv(root / "labels_remediated.csv", dtype=str)
    df = df.drop_duplicates("image", keep="first")
    df = df[df["label"].astype(str).str.len().eq(1)
            & df["tier"].isin(["GOLD", "SILVER"])]

    cfg = yaml.safe_load((REPO / "config" / "pipeline.yaml").read_text())
    qn = load_qn_to_nom(str(REPO / cfg["paths"]["qn_to_nom_dict"]))

    verd = _audit_verdict_by_image()          # image -> verdict (846)
    df = df.copy()
    df["verdict"] = df["image"].map(verd)      # NaN nếu crop này chưa được audit
    df["audited"] = df["verdict"].notna()
    df["is_wrong"] = df["verdict"].isin(_WRONG)

    rows = []
    for (syl, lab), g in df.groupby(["syllable", "label"]):
        cands = qn.get(str(syl).lower(), [])
        au = g["audited"].sum()
        wr = g["is_wrong"].sum()
        rate = (wr / au) if au else np.nan
        pop = len(g)
        rows.append({
            "syllable": syl, "label": lab,
            "unicode": f"U+{ord(lab):04X}" if len(str(lab)) == 1 else "",
            "population": pop,
            "audit_n": int(au), "audit_wrong": int(wr),
            "audit_rate": round(rate, 3) if au else "",
            "exp_wrong": round(rate * pop, 1) if au and wr else ("" if au else ""),
            "ocr_chars": "/".join(sorted(set(g["ocr_char"].dropna()))[:5]),
            "dict_ok": str(lab) in cands,
            "tiers": "/".join(sorted(set(g["tier"]))),
            "rules": "/".join(sorted(set(g["rule"]))),
        })
    conf = pd.DataFrame(rows)

    # xếp: (đã audit & có sai) theo exp_wrong ↓, rồi (chưa audit) theo population ↓
    conf["_exp"] = pd.to_numeric(conf["exp_wrong"], errors="coerce")
    conf["_grp"] = np.where(conf["audit_wrong"] > 0, 0, np.where(conf["audit_n"] > 0, 2, 1))
    conf = conf.sort_values(
        ["_grp", "_exp", "population"], ascending=[True, False, False]).drop(columns=["_exp", "_grp"])

    outp = root / "fusion" / "confusions.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    conf.to_csv(outp, index=False)

    # tóm tắt
    audited_pairs = conf[conf["audit_n"] > 0]
    bad = conf[conf["audit_wrong"] > 0]
    exp_total = pd.to_numeric(bad["exp_wrong"], errors="coerce").sum()
    n_char = int(df.shape[0])
    print("=" * 68)
    print(f" CONFUSION CENSUS  ({conf.shape[0]} cặp syllable→label | {n_char} crop char)")
    print("=" * 68)
    print(f" cặp đã có crop được audit : {len(audited_pairs)} "
          f"(phủ {int(audited_pairs['audit_n'].sum())} crop)")
    print(f" cặp audit ra SAI          : {len(bad)}  "
          f"→ nhãn sai kỳ vọng (rate×pop, chỉ các cặp này) ≈ {exp_total:.0f}")
    print(f"\n TOP {top} confusion đã chứng minh (audit ra sai), theo exp_wrong:")
    print(f"   {'syl':10s} {'lab':>3s} {'pop':>5s} {'a_n':>4s} {'a_w':>4s} {'rate':>5s} {'exp':>6s}  ocr_chars")
    for _, r in bad.head(top).iterrows():
        print(f"   {str(r['syllable'])[:10]:10s} {r['label']:>3s} {r['population']:5d} "
              f"{r['audit_n']:4d} {r['audit_wrong']:4d} {str(r['audit_rate']):>5s} "
              f"{str(r['exp_wrong']):>6s}  {r['ocr_chars']}")
    # cặp chưa audit, tác động lớn nhất → hàng đợi audit tiếp theo
    un = conf[conf["audit_n"] == 0].head(10)
    print(f"\n Cặp CHƯA audit, population lớn nhất (nên audit tiếp — rủi ro chưa biết):")
    for _, r in un.iterrows():
        print(f"   {str(r['syllable'])[:10]:10s} {r['label']:>3s} pop={r['population']:5d} "
              f"dict_ok={r['dict_ok']} ocr={r['ocr_chars']}")
    print(f"\n -> {outp}")
    return conf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.consensus_fusion.mine_confusions")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args(argv)
    run(args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
