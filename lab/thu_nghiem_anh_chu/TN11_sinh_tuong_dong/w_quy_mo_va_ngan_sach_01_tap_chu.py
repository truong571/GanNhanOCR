"""w_quy_mo_va_ngan_sach_01_tap_chu.py — BƯỚC 1: S0 / S1(k) / S2 mỗi cuốn, hợp nhất toàn kho, số cặp (cuốn, chữ). 0 API, CPU, không nhãn người.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_quy_mo_va_ngan_sach_01_tap_chu.py

Ra: measure_out/_tn11/full/quy_mo_va_ngan_sach/buoc1_tap_chu.json (kích thước + invariants) và work/plans.pkl (danh sách chữ xếp hạng cho các bước sau).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import w_quy_mo_va_ngan_sach_lib as L  # noqa: E402


def main():
    t0 = time.time()
    L.OUT.mkdir(parents=True, exist_ok=True)
    (L.OUT / "work").mkdir(exist_ok=True)
    W = L.build_world()
    C = W["C"]
    plans = {}
    for b in L.ORDER:
        plans[b] = {m: L.plan_book(W["cells"][b], W["Rm"], W["n_self"][b], W["n_oth"][b], W["g_oth"][b], m) for m in ("literal", "sach")}
        print(f"[b1] {b}: S0 literal {len(plans[b]['literal']['S0'])} sạch {len(plans[b]['sach']['S0'])}; S1(3/5/8) sạch "
              f"{[len(plans[b]['sach']['S1'][k]) for k in L.KS]}; S2 {len(plans[b]['sach']['S2'])} [{time.time() - t0:.0f}s]", flush=True)

    res = dict(nguon={}, theo_sach={}, hop_nhat={}, invariants=[])
    inv = res["invariants"]

    def check(name, ok, detail=""):
        inv.append(dict(ten=name, ok=bool(ok), chi_tiet=str(detail)[:300]))
        print(f"[inv] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)

    for b in L.ORDER:
        p = L.labels_path(b)
        res["nguon"][b] = dict(tep=str(p.relative_to(L.REPO)), mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
                               bytes=p.stat().st_size, sha256=L.sha256_file(p))
        sch = plans[b]["sach"]; lit = plans[b]["literal"]
        blk = Counter(L.block_of(c) for c in lit["S0"])
        res["theo_sach"][b] = dict(
            n_o=sch["n_cells"], n_am=sch["n_am"], n_am_co_R=sch["n_am_co_R"], n_o_am_co_R=sch["n_o_am_co_R"],
            S0_literal=len(lit["S0"]), S0_sach=len(sch["S0"]),
            S0_chi_literal=len(lit["S0"] - sch["S0"]), S0_chi_sach=len(sch["S0"] - lit["S0"]),
            S1_literal={str(k): len(lit["S1"][k]) for k in L.KS_ALL}, S1_sach={str(k): len(sch["S1"][k]) for k in L.KS_ALL},
            S2=len(sch["S2"]), S2_hop_S0_literal=len(lit["S2"] | lit["S0"]), S2_hop_S0_sach=len(sch["S2"] | sch["S0"]),
            S0_ngoai_S2_sach=len(sch["S0"] - sch["S2"]), S0_theo_khoi_unicode=dict(blk),
            S0_PUA=int(sum(v for k, v in blk.items() if k.startswith("PUA"))))

    # ---- hợp nhất toàn kho + số cặp (cuốn, chữ)
    for mode in ("literal", "sach"):
        hn = {}
        for name, getter in [("S0", lambda p: p["S0"])] + [(f"S1_k{k}", (lambda kk: lambda p: p["S1"][kk])(k)) for k in L.KS_ALL] + \
                            [("S2", lambda p: p["S2"]), ("S2_hop_S0", lambda p: p["S2"] | p["S0"])]:
            sets = {b: getter(plans[b][mode]) for b in L.ORDER}
            u = set().union(*sets.values())
            cnt = Counter(c for s in sets.values() for c in s)
            hn[name] = dict(so_cap=sum(len(s) for s in sets.values()), hop=len(u), chu_o_ge2_cuon=sum(1 for v in cnt.values() if v >= 2),
                            chu_o_10_cuon=sum(1 for v in cnt.values() if v == len(L.ORDER)))
        res["hop_nhat"][mode] = hn

    # ---- invariants
    tot = sum(res["theo_sach"][b]["n_o"] for b in L.ORDER)
    check("tong_o_10_bo = 275136 (docs/SINH_ANH_SO_SANH: 10 bộ/1.682 trang/275.136 ô)", tot == 275136, tot)
    ok_nest = all(plans[b][m]["S0"] <= plans[b][m]["S1"][3] <= plans[b][m]["S1"][5] <= plans[b][m]["S1"][8] <= (plans[b][m]["S2"] | plans[b][m]["S0"])
                  for b in L.ORDER for m in ("literal", "sach"))
    check("S0 ⊆ S1(3) ⊆ S1(5) ⊆ S1(8) ⊆ S2∪S0 (10 bộ x 2 bản)", ok_nest)
    ok_mono = all(all(plans[b][m]["S1"][k] <= plans[b][m]["S1"][k + 1] for k in L.KS_ALL[:-1]) for b in L.ORDER for m in ("literal", "sach"))
    check("S1(k) đơn điệu theo k = 1..12", ok_mono)
    ok_rank = all(len(set(plans[b][m]["ranked"])) == len(plans[b][m]["ranked"]) and set(plans[b][m]["ranked"]) == plans[b][m]["S0"] | plans[b][m]["S2"]
                  for b in L.ORDER for m in ("literal", "sach"))
    check("danh sách xếp hạng không trùng và = S0 ∪ S2", ok_rank)
    ok_pref = all(set(plans[b][m]["ranked"][:len(plans[b][m]["S1"][k])]) == plans[b][m]["S1"][k] for b in L.ORDER for m in ("literal", "sach") for k in L.KS_ALL)
    check("tiền tố danh sách xếp hạng độ dài |S1(k)| = S1(k) (k = 1..12)", ok_pref)
    same8 = [b for b in L.ORDER if plans[b]["literal"]["S0"] == plans[b]["sach"]["S0"]]
    check("bản literal = bản sạch ở 8/10 bộ (khác chỉ ở B18, B34)", sorted(set(L.ORDER) - set(same8)) == ["B18", "B34"], f"giống: {same8}")
    pre_len_ok = all(max((len(x) for x in W["cells"][b].lab_pre), default=0) <= 1 for b in L.ORDER)
    check("nhãn trước chon_chu đều ≤ 1 ký tự (tách 'tầng|nhãn|âm:..' đúng)", pre_len_ok)
    idem = all(C.R_key(rk) == rk for rk in W["Rm"])
    check("R_key idempotent trên mọi khoá âm", idem)
    non_cjk = sum(1 for b in L.ORDER for c in plans[b]["literal"]["S0"] | plans[b]["literal"]["S2"] if L.block_of(c) == "other")
    check("không có chữ ngoài các khối CJK/PUA trong S0 ∪ S2", non_cjk == 0, non_cjk)
    anchors = {b: int(sum(W["n_self"][b].values())) for b in L.ORDER}
    nhan_nguoi = [m for m in ("t00_base", "tn6lib", "borg_endtoend_eval", "pipeline.gold_exact.eval_ihr", "auto_precision") if m in sys.modules]
    check("danh sách chữ dựng khi CHƯA nạp mô-đun nhãn người (t00_base, tn6lib, borg_endtoend_eval, eval_ihr, auto_precision)", not nhan_nguoi, nhan_nguoi)
    res["sha256_ke_hoach"] = {m: {b: L.sha256_of(plans[b][m]["ranked"]) for b in L.ORDER} for m in ("literal", "sach")}
    res["cot_dau_vao_ke_hoach"] = L.PLAN_COLS + ["lab_pre (suy từ chon_chu_truoc)", "rk (suy từ syllable)"]
    res["o_neo_tu_dong"] = anchors
    res["thoi_gian_giay"] = round(time.time() - t0, 1)
    (L.OUT / "buoc1_tap_chu.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(L.OUT / "work" / "plans.pkl", "wb") as f:
        pickle.dump(dict(plans=plans, Rm=W["Rm"], n_self=W["n_self"], n_oth=W["n_oth"], g_oth=W["g_oth"]), f, protocol=4)
    print(f"[b1] xong [{time.time() - t0:.0f}s]; FAIL: {[i['ten'] for i in inv if not i['ok']] or 'không'}")


if __name__ == "__main__":
    main()
