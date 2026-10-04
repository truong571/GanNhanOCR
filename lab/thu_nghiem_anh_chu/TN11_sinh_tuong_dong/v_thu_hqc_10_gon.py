"""v_thu_hqc_10_gon.py — bảng GỌN (ket_qua / chenh_lech) từ danhgia.json (F1: mọi ô) + danhgia_k3.json (F2: mẫu 500 ô, K=3, giấy-chỉ, nguyên văn)
để chép vào báo cáo; ghi ket_qua_gon.json. Mọi số là số script đo (v_thu_hqc_05_danhgia.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v_thu_hqc_lib as H  # noqa: E402

OUT = H.OUT


def jl(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    dg = jl(OUT / "danhgia.json") or {}
    k3 = jl(OUT / "danhgia_k3.json") or {}
    kq, cl = [], []
    for b in H.BOOKS:
        sets = (dg.get(b) or {}).get("sets", {})
        s500 = ((k3.get(b) or {}).get("sets", {}) or {}).get("s500", {})
        for kind, grp in (("font", "tat_ca"), ("font", "chu_hiem"), ("fd", "tat_ca")):
            r = (sets.get("all") or {}).get(kind)
            if not r:
                continue
            for cd in ("C0", "C1"):
                x = r[cd][grp]
                lab = cd + (" (glyph nền trắng sản xuất)" if cd == "C0" else f" (HQC2 giấy+quang học, K={'3' if kind == 'fd' else '1'})")
                kq.append(dict(bo=b, tin_hieu=f"f_{kind}", dieu_kien=lab + ", mọi ô", nhom=grp, n=x["n"], top1=x["top1"], ci_thap=x["ci_thap"], ci_cao=x["ci_cao"]))
            d = r["d_C1_C0"][grp]
            if grp != "chu_hiem" or True:
                pass
        for kind in ("font", "fd"):
            r = (sets.get("all") or {}).get(kind)
            if not r:
                continue
            if kind == "font" or True:
                d = r["d_C1_C0"]["tat_ca"]
                cl.append(dict(bo=b, tin_hieu=f"f_{kind}", so_sanh=f"f_{kind}: C1-C0, mọi ô (n={d['n']})", nhom="tat_ca", delta=d["delta"], ci_thap=d["ci_thap"], ci_cao=d["ci_cao"]))
        s1000 = ((sets.get("s1000") or {}).get("font") or {})
        for nm, lab in (("C2-C0", "f_font: C2 (nhị phân) - C0, mẫu 1000"), ("CTRLWHITE-C0", "f_font: chỉ quang học (nền trắng) - C0, mẫu 1000")):
            d = (s1000.get("delta", {}).get(nm) or {}).get("tat_ca")
            if d:
                cl.append(dict(bo=b, tin_hieu="f_font", so_sanh=lab, nhom="tat_ca", delta=d["delta"], ci_thap=d["ci_thap"], ci_cao=d["ci_cao"]))
        for nm, lab in (("PAPERONLY-C0", "f_font: chỉ giấy+mực (norm như crop) - C0, mẫu 500"), ("LIT-C0", "f_font: HQC2 nguyên văn K09 (norm=False) - C0, mẫu 500")):
            d = ((s500.get("font") or {}).get("delta", {}).get(nm) or {}).get("tat_ca")
            if d:
                cl.append(dict(bo=b, tin_hieu="f_font", so_sanh=lab, nhom="tat_ca", delta=d["delta"], ci_thap=d["ci_thap"], ci_cao=d["ci_cao"]))
    H.jdump(dict(ket_qua=kq, chenh_lech=cl), OUT / "ket_qua_gon.json")
    print(len(kq), "dòng ket_qua;", len(cl), "dòng chenh_lech")
    for x in kq:
        print(f"{x['bo']:4s} {x['tin_hieu']:7s} {x['dieu_kien'][:46]:46s} {x['nhom']:9s} n={x['n']:6d} top1={x['top1']:6.2f} [{x['ci_thap']:6.2f},{x['ci_cao']:6.2f}]")
    for x in cl:
        print(f"{x['bo']:4s} {x['so_sanh'][:70]:70s} Δ={x['delta']:+6.2f} [{x['ci_thap']:+6.2f},{x['ci_cao']:+6.2f}]")


if __name__ == "__main__":
    main()
