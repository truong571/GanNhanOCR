#!/usr/bin/env python3
"""Phân tích thí nghiệm crop THEO ĐÚNG prereg_crop_test.json (đọc vision/ket_qua/crop_test/*.csv.gz; OFFLINE).

  .venv/bin/python vision/crop_test/analyze_crop_test.py
Ghi: vision/ket_qua/crop_test/ket_qua_crop_test.json và BAO_CAO_CROP_TEST.md."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "ket_qua" / "crop_test"
B = 5000


def boot(vals, clusters, seed=17):
    """→ (trung bình, cận dưới, cận trên, n, p hai phía bootstrap). NaN bị bỏ; bootstrap theo cụm."""
    v = np.asarray(vals, float)
    c = np.asarray(clusters)
    ok = ~np.isnan(v)
    v, c = v[ok], c[ok]
    if len(v) == 0:
        return (float("nan"),) * 3 + (0, float("nan"))
    u, inv = np.unique(c, return_inverse=True)
    S = np.bincount(inv, weights=v)
    N = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    means = S[idx].sum(1) / N[idx].sum(1)
    p = 2 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), int(len(v)), min(1.0, p)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def delta(df, v, mask=None, base="base", cl="cl"):
    d = df if mask is None else df[mask]
    dt = (d[f"t_{v}"] - d[f"t_{base}"]) * 100.0                    # điểm %
    dm = d[f"m_{v}"] - d[f"m_{base}"]
    ds = d[f"s_{v}"] - d[f"s_{base}"]
    r = dict(n=int(dt.notna().sum()))
    for k, x in (("top1_pp", dt), ("m_hom", dm), ("s_label", ds)):
        mean, lo, hi, n, p = boot(x.values, d[cl].values)
        r[k] = dict(mean=mean, lo=lo, hi=hi, p=p)
    r["chg"] = float(np.nanmean(d[f"chg_{v}"])) if f"chg_{v}" in d else float("nan")
    r["top1_base"] = float(np.nanmean(d[f"t_{base}"]) * 100)
    r["top1_var"] = float(np.nanmean(d[f"t_{v}"]) * 100)
    return r


def fmt(r):
    t, m = r["top1_pp"], r["m_hom"]
    return (f"n={r['n']:6d} đổi crop {100 * r['chg']:5.1f} % | top1 {r['top1_base']:5.2f}→{r['top1_var']:5.2f} Δ{t['mean']:+6.2f} pp [{t['lo']:+6.2f}; {t['hi']:+6.2f}] | "
            f"Δm_hom {m['mean']:+.4f} [{m['lo']:+.4f}; {m['hi']:+.4f}] | Δs {r['s_label']['mean']:+.4f}")


def main():
    lines, res = [], {}

    def log(s=""):
        print(s)
        lines.append(s)
    reg = open(HERE.parent / "prereg" / "prereg_crop_test.sha256").read().split()[0][:16]
    log(f"# Thí nghiệm crop bằng so ảnh — phân tích theo prereg (sha256 {reg}…)\n")

    # ───────────── E1: nới cửa sổ L16
    f = OUT / "E1_L16.csv.gz"
    if f.exists():
        d = pd.read_csv(f, low_memory=False)
        d["cl"] = d["page"]
        cut = (d["status"] == "agree") & (d["harm"] == 1)
        strata = {"TẤT CẢ ô GOLD L16 có nhãn người": None, "bị cắt (Vision agree & hại=1)": cut, "KHÔNG bị cắt (còn lại)": ~cut,
                  "Vision agree & hại=0": (d["status"] == "agree") & (d["harm"] == 0), "Vision không đọc": ~d["status"].isin(["agree", "variant", "disagree"])}
        log(f"## E1 — nới cửa sổ L16 (n={len(d)} ô; cơ sở: top1 {100 * d['t_base'].mean():.2f} %, m_hom trung vị {d['m_base'].median():.3f})\n")
        e1 = {}
        for name, mk in strata.items():
            log(f"### {name}")
            e1[name] = {}
            for v in ("adapt", "py20", "py30", "py45"):
                r = delta(d, v, mk)
                e1[name][v] = r
                log(f"- {v:6s} {fmt(r)}")
            log()
        ps = [e1["TẤT CẢ ô GOLD L16 có nhãn người"][v]["top1_pp"]["p"] for v in ("py20", "py30", "py45")]
        adj = holm(np.array(ps))
        log("Holm (3 pad đồng loạt, điểm cuối Δtop1 trên TẤT CẢ ô): " + ", ".join(f"{v} p_hiệu_chỉnh={a:.3f}" for v, a in zip(("py20", "py30", "py45"), adj)) + "\n")
        res["E1"] = e1
        pr = e1["TẤT CẢ ô GOLD L16 có nhãn người"]["adapt"]["top1_pp"]
        guard = e1["KHÔNG bị cắt (còn lại)"]["adapt"]["top1_pp"]
        ok_primary = pr["lo"] > 0 and pr["mean"] >= 0.3
        ok_guard = guard["lo"] >= -0.3
        log(f"**Điểm cuối chính E1** (Δtop1 `adapt` trên mọi ô): {pr['mean']:+.2f} pp [{pr['lo']:+.2f}; {pr['hi']:+.2f}] → "
            f"{'CẢI THIỆN (cận dưới > 0 và ≥ +0,3)' if ok_primary else 'KHÔNG đạt tiêu chí cải thiện'}; bảo vệ 'không hại' trên ô không bị cắt: {guard['lo']:+.2f} ≥ −0,3 → {'đạt' if ok_guard else 'VI PHẠM'}\n")
        res["E1_verdict"] = dict(primary_ok=bool(ok_primary), guard_ok=bool(ok_guard), primary=pr, guard=guard)

    # ───────────── E2: dời tâm
    parts = []
    for key in ("Chr", "stt2", "stt4", "stt11"):
        f = OUT / f"E2_{key}.csv.gz"
        if f.exists():
            x = pd.read_csv(f, low_memory=False)
            x["book"] = key
            x["cl"] = key + "|" + x["page"].astype(str)
            parts.append(x)
    if parts:
        log("## E2 — dời tâm ô lệch (Vision khớp chữ, |dy| > 0,10 bước); rc_*/vis_union dùng Vision ⇒ chỉ là CẬN TRÊN (oracle)\n")
        e2 = {}
        for x in parts + [pd.concat(parts, ignore_index=True)]:
            name = x["book"].iloc[0] if x["book"].nunique() == 1 else "GỘP 4 sách"
            harm = x["harm"] == 1
            log(f"### {name} (n={len(x)}; bị cắt hại=1: {int(harm.sum())}; cơ sở top1 {100 * x['t_base'].mean():.2f} %)")
            e2[name] = {}
            for lab, mk in (("mọi ô lệch tâm", None), ("bị cắt (hại=1)", harm), ("lệch nhưng không bị cắt", ~harm)):
                e2[name][lab] = {}
                for v in ("rc_full", "rc_half", "rc_rand", "vis_union", "adapt"):
                    e2[name][lab][v] = delta(x, v, mk)
                log(f"- {lab}:")
                for v in ("rc_full", "rc_half", "rc_rand", "vis_union", "adapt"):
                    log(f"    {v:9s} {fmt(e2[name][lab][v])}")
            log()
        allx = pd.concat(parts, ignore_index=True)
        dd = ((allx["t_rc_full"] - allx["t_base"]) - (allx["t_rc_rand"] - allx["t_base"])) * 100.0
        mean, lo, hi, n, p = boot(dd.values, allx["cl"].values)
        full = e2["GỘP 4 sách"]["mọi ô lệch tâm"]["rc_full"]["top1_pp"]
        ok = lo > 0 and full["mean"] >= 0.5
        log(f"**Điểm cuối chính E2** (Δtop1 rc_full − Δtop1 rc_rand, gộp): {mean:+.2f} pp [{lo:+.2f}; {hi:+.2f}] (n={n}); Δtop1 rc_full {full['mean']:+.2f} pp [{full['lo']:+.2f}; {full['hi']:+.2f}] → "
            f"{'CẢI THIỆN (cận dưới > 0 và rc_full ≥ +0,5)' if ok else 'KHÔNG đạt tiêu chí cải thiện'}\n")
        res["E2"] = e2
        res["E2_verdict"] = dict(primary_ok=bool(ok), diff_vs_rand=dict(mean=mean, lo=lo, hi=hi, n=n), rc_full=full)

    # ───────────── kiểm soát phụ
    ctrl = {}
    for key in ("TK", "KVK", "L83"):
        f = OUT / f"ctrl_{key}.csv.gz"
        if f.exists():
            x = pd.read_csv(f, low_memory=False)
            x["cl"] = x["page"]
            ctrl[key] = delta(x, "adapt")
    if ctrl:
        log("## Kiểm soát phụ — `adapt` trên sách khác (mẫu ngẫu nhiên ≤ 3.000 ô GOLD)\n")
        for k, r in ctrl.items():
            guard = r["top1_pp"]["lo"] >= -0.3
            log(f"- {k:4s} {fmt(r)} | bảo vệ (cận dưới ≥ −0,3): {'đạt' if guard else 'VI PHẠM'}")
        res["ctrl"] = ctrl
        log()
    (OUT / "ket_qua_crop_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (OUT / "BAO_CAO_CROP_TEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT / 'BAO_CAO_CROP_TEST.md'}")


if __name__ == "__main__":
    main()
