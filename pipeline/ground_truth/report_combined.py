"""Đọc verdict của mẻ GỘP -> bảng precision/CI theo TỪNG tier + kappa nội tại.

`estimate` chuẩn trả về MỘT con số precision cho cả mẻ. Mẻ gộp cần khác: mỗi tier là một
tuyên bố riêng (GOLD, SILVER, SYLLABLE có dân số riêng, ngưỡng p0 riêng, và đi vào những
dòng khác nhau của bảng trong luận văn). Module này:

  * tách verdict theo tier, tính Wilson + Clopper-Pearson + cận dưới một phía cho từng tier
    (SRS trong từng ô sách -> ước lượng Horvitz-Thompson có FPC cho tier)
  * gộp ba tier thành precision toàn tập có nhãn, cũng bằng Horvitz-Thompson
  * ghép các ô LẶP với bản gốc -> Cohen's kappa nội tại + ma trận đảo verdict
  * xuất report.json + BANG_KET_QUA.md dán thẳng được vào luận văn

Ô lặp KHÔNG bao giờ vào precision: chúng mang design_weight rỗng và bị loại tường minh.

CHẠY
----
    .venv/bin/python -m pipeline.ground_truth.report_combined --dir dataset_out/ground_truth/audit_combined
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import estimate as est
from . import stats

REPO = Path(__file__).resolve().parents[2]

# Ngưỡng tuyên bố mặc định mỗi tier (dùng cho quyết định chấp nhận một phía).
DEFAULT_P0 = {"GOLD": 0.97, "SILVER": 0.85, "SYLLABLE": 0.85}
VERDICT_ORDER = ("correct", "wrong_label", "unsure")


def _tier_of(stratum: str) -> str:
    return str(stratum).split("|", 1)[0]


def _strata_triples(scored: pd.DataFrame) -> list[tuple[int, int, int]]:
    """(N_h, n_h, k_h) cho từng tầng có mặt trong phần đã chấm được.

    N_h lấy từ cột `stratum_N` do bộ dựng mẻ ghi sẵn. Chỉ khi thiếu cột đó mới suy
    N_h = design_weight * n_h — cách suy này CO LẠI theo số ô "không đọc được" bị loại,
    nên trọng số giữa các tầng lệch đi; dùng làm phương án dự phòng thôi.
    """
    triples = []
    for _, gg in scored.groupby("stratum"):
        n_h = len(gg)
        k_h = int((gg["verdict"] == "correct").sum())
        N_h = None
        if "stratum_N" in gg.columns:
            v = pd.to_numeric(gg["stratum_N"], errors="coerce").dropna()
            if not v.empty:
                N_h = int(v.iloc[0])
        if N_h is None:
            w = pd.to_numeric(gg["design_weight"], errors="coerce").dropna()
            if w.empty:
                continue
            N_h = int(round(float(w.iloc[0]) * n_h))
        triples.append((max(N_h, n_h), n_h, k_h))
    return triples


def cohens_kappa(pairs: list[tuple[str, str]]) -> dict:
    """Cohen's kappa cho hai lần chấm CÙNG người trên cùng ô (test-retest nội tại).

    Trả về kappa, tỷ lệ đồng thuận thô, và ma trận nhầm lẫn đầy đủ. kappa nhạy với tỷ lệ
    nền: khi gần như mọi ô đều 'correct' thì kappa thấp dù đồng thuận thô rất cao — nên
    LUÔN báo cả hai con số cạnh nhau, đừng trích riêng kappa.
    """
    if not pairs:
        return {"n": 0, "kappa": None, "observed_agreement": None, "matrix": {}}
    cats = list(VERDICT_ORDER)
    idx = {c: i for i, c in enumerate(cats)}
    m = [[0] * len(cats) for _ in cats]
    for a, b in pairs:
        if a in idx and b in idx:
            m[idx[a]][idx[b]] += 1
    n = sum(sum(r) for r in m)
    if n == 0:
        return {"n": 0, "kappa": None, "observed_agreement": None, "matrix": {}}
    po = sum(m[i][i] for i in range(len(cats))) / n
    row = [sum(r) / n for r in m]
    col = [sum(m[i][j] for i in range(len(cats))) / n for j in range(len(cats))]
    pe = sum(row[i] * col[i] for i in range(len(cats)))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {
        "n": n, "kappa": kappa, "observed_agreement": po, "expected_agreement": pe,
        "matrix": {f"{a}->{b}": m[idx[a]][idx[b]] for a in cats for b in cats
                   if m[idx[a]][idx[b]]},
        "note": ("kappa thấp KHÔNG tự động nghĩa là chấm ẩu khi tỷ lệ nền lệch nặng về "
                 "'correct' — đọc kèm observed_agreement và ma trận."),
    }


def tier_report(g: pd.DataFrame, tier: str, conf: float, p0: float | None) -> dict:
    """Precision + CI cho một tier. Trong tier, mỗi (tier|sách) là một tầng SRS."""
    scored = g[g["verdict"] != "unsure"]
    n_scored, k = len(scored), int((scored["verdict"] == "correct").sum())
    out: dict = {
        "tier": tier,
        "n_audited": int(len(g)),
        "n_scored": n_scored,
        "n_correct": k,
        "n_wrong_label": int((g["verdict"] == "wrong_label").sum()),
        "n_unsure": int((g["verdict"] == "unsure").sum()),
    }
    if n_scored == 0:
        out["note"] = "không có ô chấm được (toàn 'không đọc được')"
        return out

    out["precision"] = k / n_scored
    out["wilson_ci"] = stats.wilson_ci(k, n_scored, conf)
    out["cp_ci"] = stats.clopper_pearson_ci(k, n_scored, conf)
    out["cp_lower_one_sided"] = stats.cp_lower_bound(k, n_scored, conf)

    strata = _strata_triples(scored)
    if strata:
        pt, lo, hi = stats.stratified_mean_ci(strata, conf)
        out["weighted_precision"] = pt
        out["weighted_ci"] = (lo, hi)
        out["population_N"] = sum(N for N, _, _ in strata)

    if p0 is not None:
        lcb = out["cp_lower_one_sided"]
        out["acceptance"] = {
            "p0": p0, "defects": n_scored - k, "n": n_scored,
            "one_sided_lower_bound": lcb, "accept": bool(lcb >= p0),
            "note": ("Phân bổ theo tỷ lệ dân số nên trọng số gần như bằng nhau giữa các "
                     "tầng; cận Clopper–Pearson dùng được như một tuyên bố thận trọng."),
        }
    out["per_book"] = [
        {"stratum": s,
         "n": int((gg["verdict"] != "unsure").sum()),
         "correct": int((gg["verdict"] == "correct").sum()),
         "unsure": int((gg["verdict"] == "unsure").sum())}
        for s, gg in g.groupby("stratum")
    ]
    return out


def build(dir_: Path, conf: float = 0.95, p0: dict | None = None,
          drop_unknown: bool = False) -> dict:
    p0 = {**DEFAULT_P0, **(p0 or {})}
    verdicts = est.load_verdicts(dir_)
    manifest = est.load_manifest(dir_ / "manifest.jsonl")
    j = verdicts.merge(manifest, on="item_id", how="left", validate="one_to_one")

    miss = j["stratum"].isna()
    if miss.any():
        if not drop_unknown:
            raise ValueError(
                f"{int(miss.sum())} verdict không có hàng manifest tương ứng — gần như "
                f"chắc chắn là verdict của MẺ KHÁC lọt vào. Dùng --drop-unknown nếu chắc.")
        print(f"[report] BỎ {int(miss.sum())} verdict lạc")
        j = j[~miss]

    n_missing = len(manifest) - len(j)
    if n_missing > 0:
        print(f"[report] CẢNH BÁO: {n_missing}/{len(manifest)} ô CHƯA chấm — bị loại. "
              f"Nếu việc bỏ sót liên quan tới độ khó của ô thì ước lượng sẽ chệch.")

    is_repeat = j["stratum"].astype(str) == "__repeat__"
    prob, rep = j[~is_repeat].copy(), j[is_repeat].copy()
    prob["tier_"] = prob["stratum"].map(_tier_of)

    tiers = [tier_report(g, t, conf, p0.get(t))
             for t, g in prob.groupby("tier_")]

    # tổng hợp toàn tập có nhãn — Horvitz–Thompson trên MỌI tầng của MỌI tier
    scored = prob[prob["verdict"] != "unsure"]
    all_strata = _strata_triples(scored)
    overall = None
    if all_strata:
        pt, lo, hi = stats.stratified_mean_ci(all_strata, conf)
        overall = {"weighted_precision": pt, "weighted_ci": (lo, hi),
                   "population_N": sum(N for N, _, _ in all_strata),
                   "n_scored": len(scored)}

    # kappa nội tại: ghép ô lặp với bản gốc
    vmap = dict(zip(j["item_id"], j["verdict"]))
    pairs, per_tier_pairs = [], {}
    for r in rep.itertuples():
        orig = getattr(r, "repeat_of", None)
        if not orig or orig not in vmap:
            continue
        first, second = vmap[orig], r.verdict
        pairs.append((first, second))
        t = _tier_of(manifest.set_index("item_id").loc[orig, "stratum"]) \
            if orig in set(manifest["item_id"]) else "?"
        per_tier_pairs.setdefault(t, []).append((first, second))

    reliability = cohens_kappa(pairs)
    reliability["per_tier"] = {t: cohens_kappa(p) for t, p in sorted(per_tier_pairs.items())}
    reliability["n_repeat_items"] = int(len(rep))
    reliability["n_repeat_unmatched"] = int(len(rep) - len(pairs))

    # Khung rút mẫu là dân số CHƯA từng chấm, không phải toàn tier. Ghi lại phần bị loại
    # để bài báo nói đúng phạm vi suy rộng thay vì lặng lẽ báo N của cả tier.
    frame = None
    plan_p = dir_ / "plan.json"
    if plan_p.exists():
        try:
            pl = json.loads(plan_p.read_text(encoding="utf-8"))
            frame = [{"tier": t["tier"], "N_tier": t["N"], "N_frame": t["N_unaudited"],
                      "excluded_pct": (t["N"] - t["N_unaudited"]) / t["N"] if t["N"] else 0.0}
                     for t in pl.get("tiers", [])]
        except (json.JSONDecodeError, KeyError):
            frame = None

    return {
        "dir": str(dir_.relative_to(REPO)) if dir_.is_relative_to(REPO) else str(dir_),
        "conf": conf,
        "sampling_frame": frame,
        "n_items_in_batch": int(len(manifest)),
        "n_verdicts": int(len(j)),
        "n_ungraded": int(max(0, n_missing)),
        "tiers": sorted(tiers, key=lambda d: d["tier"]),
        "overall_usable": overall,
        "reliability": reliability,
    }


def interrater(dir_: Path, conf: float = 0.95) -> dict:
    """κ giữa NGƯỜI THỨ NHẤT và NGƯỜI THỨ HAI trên mẻ `make_interrater_batch`.

    Ghép `orig_verdict` (đã nằm sẵn trong manifest, không lộ ra HTML) với verdict người
    thứ hai. Mẫu là SRS phân tầng nên κ đọc thẳng được, không phải hiệu chỉnh.
    """
    man = est.load_manifest(dir_ / "manifest.jsonl")
    if "orig_verdict" not in man.columns:
        raise ValueError(
            f"{dir_}/manifest.jsonl không có cột orig_verdict — thư mục này không phải "
            f"mẻ liên người (dựng bằng pipeline.ground_truth.make_interrater_batch)")
    v2 = est.load_verdicts(dir_).rename(columns={"verdict": "verdict2"})
    j = v2.merge(man, on="item_id", how="inner", validate="one_to_one")

    n_ungraded = len(man) - len(j)
    pairs = list(zip(j["orig_verdict"].astype(str), j["verdict2"].astype(str)))
    out = cohens_kappa(pairs)
    out["n_ungraded"] = int(max(0, n_ungraded))

    # Đồng thuận CÓ ĐIỀU KIỆN — bền với thiết kế mẫu, đọc được ngay cả khi κ khó diễn giải
    cond = {}
    for v1 in VERDICT_ORDER:
        sub = j[j["orig_verdict"] == v1]
        if len(sub):
            cond[v1] = {"n": int(len(sub)),
                        "trùng khớp": int((sub["verdict2"] == v1).sum()),
                        "tỷ lệ": float((sub["verdict2"] == v1).mean())}
    out["conditional_agreement"] = cond

    # cả hai coi là lỗi / chỉ một người coi là lỗi -> mô tả bất đồng theo hướng
    err1 = j["orig_verdict"] == "wrong_label"
    err2 = j["verdict2"] == "wrong_label"
    out["defect_overlap"] = {
        "cả hai gọi là lỗi": int((err1 & err2).sum()),
        "chỉ người 1 gọi là lỗi": int((err1 & ~err2).sum()),
        "chỉ người 2 gọi là lỗi": int((~err1 & err2).sum()),
        "cả hai gọi là đúng/không đọc được": int((~err1 & ~err2).sum()),
    }
    if out["observed_agreement"] is not None and out["n"]:
        k = int(round(out["observed_agreement"] * out["n"]))
        out["agreement_ci"] = stats.wilson_ci(k, out["n"], conf)

    per_tier = {}
    if "stratum" in j.columns:
        j = j.assign(tier_=j["stratum"].map(_tier_of))
        for t, g in j.groupby("tier_"):
            per_tier[t] = cohens_kappa(list(zip(g["orig_verdict"].astype(str),
                                                g["verdict2"].astype(str))))
    out["per_tier"] = per_tier
    out["conf"] = conf
    out["dir"] = str(dir_.relative_to(REPO)) if dir_.is_relative_to(REPO) else str(dir_)
    return out


def _md_interrater(r: dict) -> str:
    lo, hi = r.get("agreement_ci") or (None, None)
    cond = "\n".join(
        f"| người 1 gọi **{k}** | {v['n']} | {v['trùng khớp']} | {v['tỷ lệ']:.1%} |"
        for k, v in r.get("conditional_agreement", {}).items())
    mat = "\n".join(f"| {k.replace('->', ' → ')} | {v} |"
                    for k, v in sorted(r.get("matrix", {}).items()))
    ov = "\n".join(f"| {k} | {v} |" for k, v in r.get("defect_overlap", {}).items())
    tier_rows = []
    for t, d in sorted(r.get("per_tier", {}).items()):
        kap = "—" if d["kappa"] is None else f"{d['kappa']:.3f}"
        tier_rows.append(f"| {t} | {d['n']} | {_pct(d['observed_agreement'])} | {kap} |")
    tiers = "\n".join(tier_rows)
    warn = ("" if not r.get("n_ungraded") else
            f"\n⚠️ còn {r['n_ungraded']} ô người thứ hai CHƯA chấm — số dưới đây là tạm thời.\n")
    return f"""# Đồng thuận LIÊN NGƯỜI — hai người chấm độc lập

Nguồn: `{r['dir']}` · độ tin cậy {r['conf']:.0%} · **{r['n']} cặp** ghép được
{warn}
Mẫu rút **ngẫu nhiên phân tầng theo tier** từ mẻ người thứ nhất đã chấm, nên κ dưới đây
**đọc thẳng được**, không phải hiệu chỉnh theo tỷ trọng dân số.

## Kết quả

| Chỉ số | Giá trị |
|---|---:|
| **Cohen's κ** | **{'—' if r['kappa'] is None else f"{r['kappa']:.3f}"}** |
| **Đồng thuận thô** | **{_pct(r['observed_agreement'])}** {'' if lo is None else f"[{lo:.1%} · {hi:.1%}]"} |
| Đồng thuận kỳ vọng ngẫu nhiên | {_pct(r.get('expected_agreement'))} |

**Thang Landis & Koch (1977)**: κ < 0,00 kém · 0,00–0,20 rất yếu · 0,21–0,40 yếu ·
0,41–0,60 trung bình · 0,61–0,80 **tốt** · 0,81–1,00 **rất tốt**.

## Đồng thuận có điều kiện — bền với thiết kế mẫu

| Nhóm | n | Người 2 nói giống | Tỷ lệ |
|---|---:|---:|---:|
{cond}

Đây là đại lượng cần đọc khi κ bị kéo xuống bởi tỷ lệ nền lệch: nếu
*"người 1 gọi wrong_label"* mà người 2 đồng ý ở tỷ lệ cao, thì các ô bị tính là lỗi là
lỗi **thật**, không phải người 1 gọi quá tay.

## Hai người có bắt cùng những lỗi không

| | Số ô |
|---|---:|
{ov}

## Ma trận bất đồng (người 1 → người 2)

| | Số ô |
|---|---:|
{mat}

## Theo tier

| Tier | n | Đồng thuận thô | κ |
|---|---:|---:|---:|
{tiers}

---

**Cách viết vào luận văn.** Nếu κ ≥ 0,61 (mức "tốt"): *"Độ đồng thuận liên người trên mẫu
ngẫu nhiên n = {r['n']} đạt κ = … (đồng thuận thô …%), tức tiêu chí đánh giá tái lập được
qua người chấm khác."* Nếu κ thấp hơn, **vẫn phải báo** — kèm bảng đồng thuận có điều kiện
để chỉ ra bất đồng nằm ở nhóm nào, và ghi nó vào phần *Giới hạn*.
"""


def _pct(x) -> str:
    return "—" if x is None else f"{x:.1%}"


def _md(rep: dict) -> str:
    rows = []
    for t in rep["tiers"]:
        if "precision" not in t:
            rows.append(f"| {t['tier']} | — | {t['n_audited']} | — | — | — | — | — |")
            continue
        w = t.get("weighted_ci")
        rows.append(
            f"| {t['tier']} | {t.get('population_N', '—'):,} | {t['n_scored']} | "
            f"{t['n_correct']} | {t['n_wrong_label']} | {t['n_unsure']} | "
            f"**{_pct(t['precision'])}** | [{_pct(t['cp_ci'][0])} · {_pct(t['cp_ci'][1])}] | "
            f"{_pct(t['cp_lower_one_sided'])} |")
    ov = rep["overall_usable"]
    ov_row = ("| **Toàn tập có nhãn** | "
              f"{ov['population_N']:,} | {ov['n_scored']} | — | — | — | "
              f"**{_pct(ov['weighted_precision'])}** | "
              f"[{_pct(ov['weighted_ci'][0])} · {_pct(ov['weighted_ci'][1])}] | — |"
              ) if ov else ""
    fr = rep.get("sampling_frame") or []
    frame_note = ""
    if fr:
        bits = ", ".join(
            f"{f['tier']} {f['N_frame']:,}/{f['N_tier']:,} ({f['excluded_pct']:.1%} đã chấm "
            f"ở mẻ trước, bị loại)" for f in fr)
        frame_note = (
            "> **Phạm vi suy rộng.** Cột N là **khung rút mẫu** = phần dân số chưa từng "
            f"được chấm, không phải toàn tier: {bits}. Loại các ô đã chấm là chủ ý — để "
            "trí nhớ buổi trước không nhiễm vào verdict lần này. Mọi tuyên bố precision "
            "vì vậy áp cho phần dân số này.")
    rel = rep["reliability"]
    acc = "\n".join(
        f"| {t['tier']} | {t['acceptance']['p0']:.0%} | {t['acceptance']['defects']} | "
        f"{_pct(t['acceptance']['one_sided_lower_bound'])} | "
        f"{'**ĐẠT**' if t['acceptance']['accept'] else 'chưa đạt'} |"
        for t in rep["tiers"] if t.get("acceptance"))
    mat = "\n".join(f"| {k.replace('->', ' → ')} | {v} |"
                    for k, v in sorted(rel.get("matrix", {}).items()))
    return f"""# Kết quả kiểm định nhãn bằng người — mẻ gộp

Nguồn: `{rep['dir']}` · độ tin cậy {rep['conf']:.0%} · {rep['n_verdicts']}/{rep['n_items_in_batch']} ô đã chấm
{"" if not rep['n_ungraded'] else f"⚠️ còn {rep['n_ungraded']} ô CHƯA chấm — số dưới đây chỉ tạm thời."}

## Bảng chính — precision theo tier

| Tier | N (dân số) | n (chấm được) | Đúng | Sai nhãn | Không đọc được | Precision | CI 95 % (CP) | Cận dưới một phía |
|---|---:|---:|---:|---:|---:|---:|---|---:|
{chr(10).join(rows)}
{ov_row}

*Ô "không đọc được" bị loại khỏi mẫu số, báo riêng — mập mờ không bao giờ âm thầm được
tính là đúng hay sai. "Toàn tập có nhãn" gộp bằng Horvitz–Thompson có hiệu chỉnh tổng thể
hữu hạn (FPC) trên mọi tầng (tier × sách).*

{frame_note}

## Quyết định chấp nhận (một phía, {rep['conf']:.0%})

| Tier | Ngưỡng p₀ | Số lỗi | Cận dưới đạt được | Kết luận |
|---|---:|---:|---:|---|
{acc}

## Độ tin cậy người chấm — kiểm tra lặp NỘI TẠI

{rel['n_repeat_items']} ô được đưa vào mẻ hai lần, cách nhau tối thiểu 200 vị trí, người
chấm không biết ô nào là ô lặp.

| Chỉ số | Giá trị |
|---|---:|
| Số cặp ghép được | {rel['n']} |
| **Đồng thuận thô** | **{_pct(rel['observed_agreement'])}** |
| **Cohen's κ** | **{'—' if rel['kappa'] is None else f"{rel['kappa']:.3f}"}** |
| Đồng thuận kỳ vọng ngẫu nhiên | {_pct(rel.get('expected_agreement'))} |

| Đảo verdict (lần 1 → lần 2) | Số ô |
|---|---:|
{mat}

**Diễn giải κ** — κ ≥ 0,8: tiêu chí ổn định, precision ở trên công bố được · 0,4–0,8: công
bố được nhưng **phải nêu κ kèm theo** như một giới hạn · < 0,4: thiết kế nhiệm vụ vẫn
chưa ổn định, phải viết rubric chi tiết hơn rồi chấm lại.

⚠️ κ nhạy với tỷ lệ nền: khi gần như mọi ô đều "đúng" thì κ thấp **dù** đồng thuận thô rất
cao. Luôn trích κ **cạnh** đồng thuận thô và ma trận đảo verdict, đừng trích κ một mình.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.ground_truth.report_combined")
    ap.add_argument("--dir", default="dataset_out/ground_truth/audit_combined")
    ap.add_argument("--interrater", metavar="DIR",
                    help="thư mục mẻ liên người -> κ giữa hai người + KAPPA_LIEN_NGUOI.md "
                         "(chạy riêng, không cần --dir)")
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--p0-gold", type=float, default=DEFAULT_P0["GOLD"])
    ap.add_argument("--p0-silver", type=float, default=DEFAULT_P0["SILVER"])
    ap.add_argument("--p0-syllable", type=float, default=DEFAULT_P0["SYLLABLE"])
    ap.add_argument("--drop-unknown", action="store_true")
    args = ap.parse_args(argv)

    if args.interrater:
        ir_dir = Path(args.interrater)
        if not ir_dir.is_absolute():
            ir_dir = REPO / ir_dir
        r = interrater(ir_dir, args.conf)
        (ir_dir / "interrater.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        (ir_dir / "KAPPA_LIEN_NGUOI.md").write_text(_md_interrater(r), encoding="utf-8")
        kap = "—" if r["kappa"] is None else f"{r['kappa']:.3f}"
        print(f"[liên người] n={r['n']} κ={kap} "
              f"(đồng thuận thô {r['observed_agreement']:.1%})")
        print(f"[liên người] -> {ir_dir / 'KAPPA_LIEN_NGUOI.md'}")
        return 0

    d = Path(args.dir)
    if not d.is_absolute():
        d = REPO / d
    rep = build(d, args.conf,
                {"GOLD": args.p0_gold, "SILVER": args.p0_silver,
                 "SYLLABLE": args.p0_syllable},
                drop_unknown=args.drop_unknown)
    (d / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    (d / "BANG_KET_QUA.md").write_text(_md(rep), encoding="utf-8")
    for t in rep["tiers"]:
        if "precision" in t:
            print(f"[report] {t['tier']:9s} n={t['n_scored']:4d} "
                  f"precision={t['precision']:.1%} "
                  f"CI95=[{t['cp_ci'][0]:.1%}, {t['cp_ci'][1]:.1%}]")
    r = rep["reliability"]
    if r["kappa"] is not None:
        print(f"[report] κ nội tại = {r['kappa']:.3f} "
              f"(đồng thuận thô {r['observed_agreement']:.1%}, n={r['n']})")
    print(f"[report] -> {d / 'report.json'} · {d / 'BANG_KET_QUA.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
