"""T3 — quét ma trận chi phí Needleman-Wunsch trên ĐẦU VÀO THẬT.

⚠️ BẢN NÀY LÀ BẢN THỨ HAI. Bản đầu (2026-08-24) đã bị đợt phản biện độc lập bác với kết
luận THIẾT KẾ HỎNG, và nó cho ra kết quả SAI đã lỡ báo cáo (xem docs/VIEC_CAN_LAM.md,
mục "T3 — thiết kế hỏng"). Năm chỗ đã sửa nằm ở docstring `perturb.sweep_config`.

LƯỚI (78 cấu hình)
    COST_SIMILAR         0,10 · 0,20 · 0,30 · 0,45 · 0,60
    COST_NODICT          0,80 · 0,90 · 1,00
    COST_DEL = COST_INS  0,45 · 0,60 · 0,70 · 0,85 · 1,00      (nới để BAO cực trị)
    BAND_SLACK           tách riêng 3 cấu hình phụ — đo được là KHÔNG nhận dạng được

MỘT TRỤC + RÀNG BUỘC (không phải Pareto)
----------------------------------------
Hai trục cũ tương quan Pearson +0,905 nên KHÔNG phải đánh đổi; khung Pareto chỉ là trang
trí, và trục `yield_confirmed` còn NGƯỢC DẤU với sản lượng thật (cấu hình "thắng" phát ra
ÍT hơn 352 cặp ghép, mà op 'del' biến mất hoàn toàn khỏi labels.csv). Luật đúng:

    RÀNG BUỘC  yield_match >= của MỐC        MỤC TIÊU  cực đại anchor_retention
    NGƯỠNG     chỉ đổi khi hơn MỐC nhiều hơn nhiễu seed

    python -m pipeline.lab.sweep_t3 --round a
    python -m pipeline.lab.sweep_t3 --round b --top 8
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "lab" / "t3_sweep.csv"

# BAND tách riêng: đo được là KHÔNG NHẬN DẠNG ĐƯỢC bằng chuẩn này (neo sinh bởi band4
# trùng 100,0% neo của band2; ở mức nhiễu 1-3 thao tác thì |m-n| hiếm khi vượt băng cơ
# sở). Để trong lưới chính thì 3/4 số cấu hình tiêu tài nguyên mà không sinh thông tin.
BANDS_SIDE = [(2, False), (3, False), (4, False), (2, True)]
# Lưới NỚI RỘNG để BAO được cực trị: bản trước ghim 7/8 điểm Pareto vào cạnh dưới
# (di = 0,60 = giá trị nhỏ nhất được phép) -> hướng tối ưu chỉ RA NGOÀI lưới, nên
# không thể kết luận 0,60 là tốt; phải thấy đường cong quay đầu.
SIMILAR = [0.10, 0.20, 0.30, 0.45, 0.60]
NODICT = [0.80, 0.90, 1.00]
DELINS = [0.45, 0.60, 0.70, 0.85, 1.00]

BASELINE = {"band": 2, "adaptive": False, "similar": 0.30, "nodict": 0.90, "delins": 0.70}


def grid():
    for sim, nod, di in itertools.product(SIMILAR, NODICT, DELINS):
        yield {"band": 2, "adaptive": False, "similar": sim, "nodict": nod, "delins": di}
    for b, adp in BANDS_SIDE:                       # phụ: chỉ đổi băng, chi phí giữ mốc
        if (b, adp) == (2, False):
            continue
        yield {"band": b, "adaptive": adp, **{k: BASELINE[k] for k in ("similar", "nodict", "delins")}}


def label(c: dict) -> str:
    return (f"band{c['band']}{'a' if c['adaptive'] else ''}"
            f"_sim{c['similar']:.2f}_nod{c['nodict']:.2f}_di{c['delins']:.2f}")


def run_one(cfg: dict, cols, qn, sim, seeds: int) -> dict:
    from pipeline.lab.perturb import sweep_config
    r = sweep_config(cols, qn, sim, band_slack=cfg["band"], seeds=seeds,
                     adaptive_band=cfg["adaptive"],
                     cost={"COST_SIMILAR": cfg["similar"],
                           "COST_NODICT": cfg["nodict"],
                           "COST_DEL": cfg["delins"], "COST_INS": cfg["delins"]})
    r.update({"config": label(cfg), **{f"p_{k}": v for k, v in cfg.items()},
              "n_columns": len(cols), "seeds": seeds,
              "is_baseline": all(cfg[k] == v for k, v in BASELINE.items())})
    return r


def decide(rows: list[dict], noise: float = 0.0) -> dict:
    """LUẬT CHỌN VIẾT TRƯỚC — một trục + ràng buộc, KHÔNG phải Pareto.

    Đợt phản biện đo được hai trục tương quan Pearson +0,905 (Spearman 0,82-0,99): chúng
    KHÔNG phải đánh đổi, nên khung Pareto chỉ là trang trí. Luật đúng (chính
    docs/CHUONG_TRINH_THI_NGHIEM §T3.3 đã đề, mã cũ không cài):

      RÀNG BUỘC  yield_match >= yield_match của MỐC   (không cấu hình nào được phép phát
                 ra ÍT nhãn hơn — op 'del' biến mất hoàn toàn khỏi labels.csv, kể cả REVIEW)
      MỤC TIÊU   cực đại `anchor_retention`
      NGƯỠNG     chỉ đổi khỏi MỐC khi hơn MỐC nhiều hơn `noise` (độ lệch giữa các seed).
                 Đổi ma trận chi phí là đổi TOÀN BỘ bộ nhãn — cải thiện dưới mức nhiễu
                 không đáng cái giá đó.
      ĐỘ BỀN     (thêm 2026-08-24) mức tăng phải CÒN vượt nhiễu sau khi BỎ lớp hỏng suy
                 biến `swap_syl`. Lỗ hổng của bản luật đầu: nó chỉ so mức tăng GỘP với
                 nhiễu. Đo được ở vòng B: cấu hình "thắng" hơn mốc +0,00070 gộp, nhưng
                 87% mức đó đến từ RIÊNG `swap_syl` — lớp đổi chỗ hai âm kề, tạo phép
                 ghép CHÉO mà căn chỉnh đơn điệu về nguyên tắc không thể sinh ra. Bỏ lớp
                 đó thì mức tăng còn +0,00009, DƯỚI nhiễu 0,00023. Một cải thiện chỉ tồn
                 tại trên một phép thử bất khả thi thì không phải cải thiện.
    """
    DEGENERATE = ("swap_syl",)
    base = next((r for r in rows if r["is_baseline"]), None)
    if base is None:
        return {"winner": None, "why": "không có cấu hình MỐC trong lô"}
    floor = base["yield_match"]
    ok = [r for r in rows if r["yield_match"] >= floor]
    best = max(ok, key=lambda r: (r["anchor_retention"] or 0)) if ok else base
    gain = (best["anchor_retention"] or 0) - (base["anchor_retention"] or 0)

    def robust(r):
        ks = [k for k in r if k.startswith("ret_") and k not in ("ret_noise0", "ret_drop")
              and k[4:] not in DEGENERATE]
        vals = [float(r[k]) for k in ks if r.get(k) not in (None, "")]
        return sum(vals) / len(vals) if vals else None
    rb, rw = robust(base), robust(best)
    gain_robust = (rw - rb) if (rb is not None and rw is not None) else None

    if best is base or gain <= noise:
        return {"winner": base["config"], "gain": gain, "gain_robust": gain_robust,
                "n_pass_floor": len(ok), "why": (
                    f"GIỮ MỐC — cấu hình tốt nhất qua ràng buộc ({best['config']}) chỉ hơn "
                    f"{gain:+.5f}, không vượt nhiễu seed {noise:.5f}")}
    if gain_robust is not None and gain_robust <= noise:
        return {"winner": base["config"], "gain": gain, "gain_robust": gain_robust,
                "n_pass_floor": len(ok), "why": (
                    f"GIỮ MỐC — {best['config']} hơn {gain:+.5f} gộp, NHƯNG bỏ lớp suy biến "
                    f"{'/'.join(DEGENERATE)} thì chỉ còn {gain_robust:+.5f} < nhiễu "
                    f"{noise:.5f}. Cải thiện chỉ tồn tại trên một phép thử bất khả thi "
                    f"(ghép chéo, căn chỉnh đơn điệu không sinh ra được) thì không phải "
                    f"cải thiện. Kèm cái giá: yield_confirmed "
                    f"{best['yield_confirmed'] - base['yield_confirmed']:+} cặp.")}
    return {"winner": best["config"], "gain": gain, "gain_robust": gain_robust,
            "n_pass_floor": len(ok), "why": (
                f"ĐỔI — hơn mốc {gain:+.5f} gộp và {gain_robust:+.5f} sau khi bỏ lớp suy "
                f"biến, cả hai > nhiễu {noise:.5f}; yield_match {best['yield_match']} >= sàn {floor}")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.lab.sweep_t3")
    ap.add_argument("--round", choices=["a", "b"], default="a")
    ap.add_argument("--sample", type=int, default=1200, help="số cột cho vòng A")
    ap.add_argument("--seeds", type=int, default=0, help="0 = tự chọn theo vòng")
    ap.add_argument("--top", type=int, default=10, help="số cấu hình vào vòng B")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="nhiễu seed — chỉ đổi khỏi MỐC khi hơn nhiều hơn mức này")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO))
    from core.text.dictionary import dict_dir, load_qn_to_nom, load_similarity_dict
    from pipeline.lab.extract_columns import load

    cols_all = load()
    qn = load_qn_to_nom(str(dict_dir() / "QuocNgu_SinoNom.csv"))
    sim = load_similarity_dict(str(dict_dir() / "SinoNom_Similar.csv"))
    out = Path(args.out)

    if args.round == "a":
        # mẫu ĐỀU trên danh sách đã sắp -> tất định, phủ cả ba sách
        step = max(1, len(cols_all) // args.sample)
        cols = cols_all[::step][: args.sample]
        seeds = args.seeds or 3
        configs = list(grid())
    else:
        prev = [r for r in csv.DictReader(open(out, encoding="utf-8"))
                if r.get("round") == "a"]
        for r in prev:
            r["yield_confirmed"] = int(r["yield_confirmed"])
            r["anchor_retention"] = float(r["anchor_retention"] or 0)
        for r in prev:
            r["yield_match"] = int(r["yield_match"]); r["is_baseline"] = r["is_baseline"] == "True"
        base = next(r for r in prev if r["is_baseline"])
        ok = sorted((r for r in prev if r["yield_match"] >= base["yield_match"]),
                    key=lambda r: -(r["anchor_retention"] or 0))
        keep = {r["config"] for r in ok[: args.top]} | {base["config"]}
        configs = [c for c in grid() if label(c) in keep]
        cols = cols_all
        seeds = args.seeds or 5
        print(f"[t3] vòng B: {len(configs)} cấu hình (biên Pareto vòng A + mốc)")

    print(f"[t3] vòng {args.round.upper()}: {len(configs)} cấu hình × {len(cols):,} cột "
          f"× seeds={seeds}", flush=True)
    rows, t0 = [], time.time()
    for i, cfg in enumerate(configs, 1):
        r = run_one(cfg, cols, qn, sim, seeds)
        r["round"] = args.round
        rows.append(r)
        el = time.time() - t0
        print(f"  [{i:3}/{len(configs)}] {r['config']:34} "
              f"match={r['yield_match']:6} ret={r['anchor_retention']:.5f} "
              f"ret0={r['ret_noise0']:.5f} drop={r['ret_drop']:.5f}"
              f"{'  <-- MỐC' if r['is_baseline'] else ''}"
              f"   [{el/60:.1f}m, còn ~{el/i*(len(configs)-i)/60:.0f}m]", flush=True)

    old = []
    if out.exists():
        old = [r for r in csv.DictReader(open(out, encoding="utf-8"))
               if r.get("round") != args.round]
    allr = old + rows
    fields: list[str] = []
    for r in allr:
        for k in r:
            if k not in fields:
                fields.append(k)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in allr:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[t3] -> {out}  ({len(rows)} dòng vòng {args.round.upper()})")

    d = decide(rows, noise=args.noise)
    print(f"\n[t3] QUYẾT ĐỊNH: {d['winner']}")
    print(f"     {d['why']}")
    print(f"     qua sàn yield_match: {d.get('n_pass_floor')}/{len(rows)} cấu hình")
    base = next((r for r in rows if r["is_baseline"]), None)
    top = sorted((r for r in rows if base and r["yield_match"] >= base["yield_match"]),
                 key=lambda r: -(r["anchor_retention"] or 0))[:8]
    print("\n[t3] TOP 8 qua ràng buộc:")
    for r in top:
        print(f"   {r['config']:34} match={r['yield_match']:6} ret={r['anchor_retention']:.5f}"
              f"{'  <-- MỐC' if r['is_baseline'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
