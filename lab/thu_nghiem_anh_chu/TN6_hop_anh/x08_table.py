"""TN6 x08 — bảng so sánh các biến thể box_decoder theo bộ (đọc eval/*.json do x04/x05/x06 sinh) -> measure_out/_tn6/table.{md,json}."""
import json
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
D = REPO / "measure_out/_tn6"
NAME = {"current": "pitch (trước TN6)", "legacy": "legacy", "visual_dp_v0": "visual_dp v0 (không agree)", "visual_dp": "visual_dp",
        "visual_dp_hybrid": "visual_dp_hybrid"}
VARS = ("current", "legacy", "visual_dp_v0", "visual_dp", "visual_dp_hybrid")


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}"


rows, js = [], {}
for book in ("SachKinhThayCaBinh", "SachDungLyHoThan"):
    for v in VARS:
        f = D / book / v / "eval/tn6_summary.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        g, a = d["GOLD"], d["all"]
        ng = d["tiers"].get("GOLD", 0)
        est = round(ng * (g["both"] or 0))
        js[f"{book}/{v}"] = dict(gold=ng, slot_all=a["slot"], n_all=a["n"], slot_gold=g["slot"], n_gold_truth=g["n"], v1p_gold=g["v1p"],
                                 both_gold=g["both"], est_both=est, quarantine=d["tiers"].get("QUARANTINE", 0),
                                 text_only=d["tiers"].get("GOLD_text_only", 0))
        rows.append(f"| {book} | {NAME[v]} | {ng:,} | {pct(a['slot'])} (n {a['n']:,}) | {pct(g['slot'])} (n {g['n']:,}) | {pct(g['v1p'])} | "
                    f"{pct(g['both'])} | {est:,} | {d['tiers'].get('QUARANTINE', 0):,} |")
for book in ("LucVanTien1916", "TruyenKieu1872"):
    for v in VARS:
        f = D / book / v / "eval/tn6_ihr_summary.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        g, a = d["GOLD"], d["ALL"]
        ng = d["tiers"].get("GOLD", 0)
        est = round(ng * g["both"])
        js[f"{book}/{v}"] = dict(gold=ng, slot_all=a["slot1"], n_all=a["n"], slot_gold=g["slot1"], v1p_gold=g["lab_v1p"], both_gold=g["both"],
                                 est_both=est, quarantine=d["tiers"].get("QUARANTINE", 0), text_only=d["tiers"].get("GOLD_text_only", 0))
        rows.append(f"| {book} | {NAME[v]} | {ng:,} | {pct(a['slot1'])} (n {a['n']:,}) | {pct(g['slot1'])} (n {g['n']:,}) | "
                    f"{pct(g['lab_v1p'])} | {pct(g['both'])} | {est:,} | {d['tiers'].get('QUARANTINE', 0):,} |")
hdr = ("| Bộ | box_decoder | GOLD có ảnh | đúng vị trí MỌI ô (%) | đúng vị trí GOLD (%) | nhãn V1+ GOLD (%) | GOLD đúng hai vế (%) | "
       "GOLD đúng hai vế ước lượng | QUARANTINE |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")
md = hdr + "\n" + "\n".join(rows)
# ---- bảng 2: thạch bản/mộc bản — proxy ô tham chiếu box_ref (x09) + dị bản (auto_precision cross) [ƯỚC LƯỢNG]
rows2 = []
for book in ("LucVanTien1883", "KimVanKieu1884", "LucVanTien1916", "TruyenKieu1872"):
    for v in ("current", "legacy", "visual_dp", "visual_dp_hybrid"):
        f = D / book / v / "eval/tn6_boxref.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        cr = (REPO / f"prepared/{book}/dataset_out/auto_precision_gated/cross/summary.json" if v == "current"
              else D / book / v / "dataset_out/auto_precision_gated/cross/summary.json")
        eq = eqd = ""
        if cr.exists():
            try:
                rb = json.load(open(cr))["books"][book]["refs"]
                r0 = next(iter(rb.values()))["all_tiers"]
                eq, eqd = f"{r0['GOLD_eq_pct']:.1f} (n {r0['n_GOLD']:,})", f"{r0['GOLD_eq_or_di_the_pct']:.1f}"
            except (KeyError, StopIteration, ValueError):
                pass
        js[f"{book}/{v}/boxref"] = dict(d, cross_eq=eq)
        rows2.append(f"| {book} | {NAME[v]} | {d['tiers'].get('GOLD', 0):,} | {pct(d['ALL']['ok'])} (n {d['ALL']['n']:,}) | "
                     f"{pct(d['ALL']['ok_honest'])} | {pct(d['GOLD']['ok'])} | {eq or '—'} | {eqd or '—'} |")
hdr2 = ("| Bộ | box_decoder | GOLD có ảnh | trong ô tham chiếu MỌI ô (%) | … bỏ ô ink_cut (%) | trong ô tham chiếu GOLD (%) | "
        "GOLD khớp dị bản (%) | khớp ∨ dị thể cùng âm (%) |\n|---|---|---:|---:|---:|---:|---:|---:|")
md += "\n\n" + hdr2 + "\n" + "\n".join(rows2)
(D / "table.md").write_text(md + "\n", encoding="utf-8")
json.dump(js, open(D / "table.json", "w"), ensure_ascii=False, indent=1)
print(md)
