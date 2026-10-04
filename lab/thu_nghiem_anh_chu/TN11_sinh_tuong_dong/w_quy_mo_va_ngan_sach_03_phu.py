"""w_quy_mo_va_ngan_sach_03_phu.py — BƯỚC 3: ĐƯỜNG CONG PHỦ = tỉ lệ ô của cuốn mà chữ đúng thuộc N chữ đầu của danh sách ưu tiên. 0 API, CPU.

Danh sách ưu tiên mỗi cuốn (bản "sach", không nhãn người; xem lib): S0 xếp theo bằng chứng, rồi ứng viên R(âm) xếp theo hạng tiên nghiệm tốt nhất.
Sự thật CHỈ để CHẤM (nạp SAU khi danh sách đã dựng và băm ở bước 1):
  B18, B34, L16, TK : chữ người (t00_base.base_of: gt_char) — "nguoi" = mọi ô có chữ người; "nguoi_hai_ve" = chữ người ∧ vị trí đúng (pos_ok);
                      "nguoi_kim_sai" = ô có chữ người ≠ chữ kim (phần khó, nơi S1/S2 mới có tác dụng)  [ĐO]
  L83, KVK          : "diban" = chữ dị bản cùng vị trí âm (ref; ƯỚC LƯỢNG, không phải sự thật); "diban_kim_sai" = ô mà chữ kim không trùng dị bản nào
  10 cuốn           : "gold" = nhãn GOLD (cột label của ô tier == GOLD) làm XẤP XỈ — LẠC QUAN: nhãn GOLD sinh từ chính chữ kim nên ≥ |S0| gần như tự nhiên 100 %
  giữ-ra theo khối trang: danh sách dựng CHỈ từ bằng chứng (S0, ô neo) của 4/5 khối trang, đo trên khối còn lại (5 khối liền, tn8lib.page_blocks) — ước lượng
                      phủ chữ "chưa thấy" trong sách; bi quan hơn thực tế vì bản thật luôn có chữ kim của chính ô.
Ba biến thể danh sách: "cascade" (chỉ chữ có phông trong chuỗi đề xuất — khả thi), "nomna" (chỉ chữ có trong NomNaTong-Regular.ttf — không đổi mã), "ly_tuong" (không lọc phông).
Ngoài ra: phủ ỨNG VIÊN (không cần sự thật) = tỉ lệ R(âm) của ô có ảnh sinh.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/w_quy_mo_va_ngan_sach_03_phu.py
Ra: phu.json.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w_quy_mo_va_ngan_sach_lib as L  # noqa: E402

MAIN = "sach"
QS = (0.90, 0.95, 0.97, 0.99)
STRAT = ("S0",) + tuple(f"S1_k{k}" for k in L.KS_ALL) + ("S2_hop_S0",)
FINE_NS = list(range(0, 12001, 100))        # lưới mịn (bước 100 chữ) cho quy hoạch động ở bước 4


def strat_sets(pf: dict) -> dict:
    d = {"S0": pf["S0"]}
    for k in L.KS_ALL:
        d[f"S1_k{k}"] = pf["S1"][k]
    d["S2_hop_S0"] = pf["S2"] | pf["S0"]
    return d


def n_for(p: np.ndarray, q: float):
    """N nhỏ nhất để phủ ≥ q (None nếu trần < q)."""
    p = np.sort(p)
    k = math.ceil(q * len(p)) - 1
    return int(p[k]) if len(p) and p[k] < L.INF else None


def min_pos(cands: list, pos: dict) -> np.ndarray:
    return np.array([min(pos.get(x, L.INF) for x in cs) for cs in cands], dtype=np.int64)


def member(cands: list, s: set) -> float:
    return float(np.mean([any(x in s for x in cs) for cs in cands])) if cands else float("nan")


def summarize(cands: list, pos: dict, pf: dict) -> dict:
    p = min_pos(cands, pos)
    ss = strat_sets(pf)
    return dict(NS=[round(x, 5) for x in L.curve(p, L.NS)], fine=[round(x, 5) for x in L.curve(p, FINE_NS)],
                chien_luoc={k: member(cands, s) for k, s in ss.items()},
                chien_luoc_theo_vi_tri={k: L.curve(p, [len(s)])[0] for k, s in ss.items()}, tran=float(np.mean(p < L.INF)),
                do_dai_ds=len(pf["ranked"]), N_dat={str(q): n_for(p, q) for q in QS})


def main():
    t0 = time.time()
    pl = pickle.load(open(L.OUT / "work" / "plans.pkl", "rb"))
    ph = pickle.load(open(L.OUT / "work" / "phong.pkl", "rb"))
    s1 = json.load(open(L.OUT / "buoc1_tap_chu.json", encoding="utf-8"))
    P = pl["plans"]
    inv = []

    def check(name, ok, detail=""):
        inv.append(dict(ten=name, ok=bool(ok), chi_tiet=str(detail)[:300]))
        if not ok:
            print(f"[inv] FAIL {name} {detail}", flush=True)

    same = all(L.sha256_of(P[b][m]["ranked"]) == s1["sha256_ke_hoach"][m][b] for b in L.ORDER for m in ("literal", "sach"))
    check("danh sách chữ trong plans.pkl trùng mã băm ghi ở bước 1 (chưa nạp nhãn người)", same)
    variants = {"cascade": lambda c: c in ph["avail_cas"], "nomna": lambda c: c in ph["avail_base"], "ly_tuong": lambda c: True}
    W = L.build_world(log=lambda *_: None)
    C = W["C"]
    cells, Rm = W["cells"], W["Rm"]
    # chỉ nạp mô-đun nhãn người SAU khi mọi danh sách đã cố định
    import t00_base as B0  # noqa: E402
    import tn8lib as T  # noqa: E402
    res = dict(NS=L.NS, FINE_NS=FINE_NS, QS=list(QS), sach={}, invariants=inv)
    old_gt = []
    for b in L.ORDER:
        Lb = cells[b]
        n = len(Lb)
        plan = P[b][MAIN]
        pf = {v: L.filtered(plan, fn) for v, fn in variants.items()}
        pos = {v: L.positions(pf[v]["ranked"]) for v in variants}
        R = dict(kich_thuoc={v: {k: len(s) for k, s in strat_sets(pf[v]).items()} for v in variants}, n_o=n)
        kim = Lb.ocr_char.to_numpy(dtype=object)
        # ---- sự thật: tên -> (nhãn mức tin, danh sách tập chữ đúng mỗi ô, mặt nạ ô)
        truth = {}
        gold_m = ((Lb.tier == "GOLD") & (Lb.label != "")).to_numpy()
        lab = Lb.label.to_numpy(dtype=object)
        truth["gold"] = ("GOLD-xap-xi", [(x,) for x in lab], gold_m)
        if b in L.HUMAN or b in ("L83", "KVK"):
            D = B0.base_of(b)
            ok_align = (len(D) == n and (D.page.to_numpy() == Lb.page.to_numpy()).all() and (D.syllable.to_numpy() == Lb.syllable.to_numpy()).all()
                        and (D.ocr_char.to_numpy() == Lb.ocr_char.to_numpy()).all() and (D.tier.to_numpy() == Lb.tier.to_numpy()).all())
            check(f"{b}: bảng nền t00_base.base_of khớp từng hàng với nhãn dùng dựng danh sách (page, âm, kim, tầng)", ok_align)
            if b in L.HUMAN:
                gt = D.gt_char.to_numpy(dtype=object)
                posok = D.pos_ok.to_numpy(bool)
                gts = [(x,) for x in gt]
                truth["nguoi"] = ("nguoi-DO", gts, gt != "")
                truth["nguoi_hai_ve"] = ("nguoi+vi-tri-DO", gts, (gt != "") & posok)
                truth["nguoi_kim_sai"] = ("nguoi-o-kim-sai-DO", gts, (gt != "") & (gt != kim))
                old = T.load_base(b)
                mg = D[["cell_uid", "gt_char"]].merge(old[["cell_uid", "gt_char"]].rename(columns={"gt_char": "g0"}), on="cell_uid", how="inner")
                old_gt.append((b, len(mg), float((mg.gt_char == mg.g0).mean())))
            if b in ("L83", "KVK"):
                ref = D.ref.to_numpy(dtype=object)
                rl = [tuple(r.split("|")) if r else () for r in ref]
                truth["diban"] = ("di-ban-UOC-LUONG", rl, ref != "")
                truth["diban_kim_sai"] = ("di-ban-kim-khac-UOC-LUONG", rl, (ref != "") & np.array([k not in r for k, r in zip(kim, rl)]))
        R["duong_cong"] = {}
        for tname, (nhan, cl, mask) in truth.items():
            idx = np.nonzero(mask)[0]
            cands = [cl[i] for i in idx]
            row = dict(n_o=int(len(idx)), nhan=nhan,
                       ty_le_chu_dung_bang_kim=float(np.mean([kim[i] in cl[i] for i in idx])) if len(idx) else None)
            for v in variants:
                row[v] = summarize(cands, pos[v], pf[v])
            if tname in ("nguoi", "diban", "gold", "nguoi_kim_sai") and len(idx):
                pg = Lb.page.to_numpy()[idx]
                ss = strat_sets(pf["cascade"])
                row["ci95_cum_trang_cascade"] = {k: [round(x, 5) for x in T.boot_ci_pages(np.array([any(x in ss[k] for x in cs) for cs in cands], float), pg)]
                                                 for k in ("S0", "S1_k3", "S1_k5", "S1_k8", "S2_hop_S0")}
            if len(idx) and tname in ("gold", "nguoi", "nguoi_kim_sai", "nguoi_hai_ve"):
                nb = L.variant_index(C, pf["cascade"]["ranked"])
                pv = L.variant_positions([c[0] for c in cands], pos["cascade"], nb)
                row["bien_the_V1plus_cascade"] = dict(NS=[round(x, 5) for x in L.curve(pv, L.NS)],
                                                      chien_luoc={k: L.curve(pv, [len(s)])[0] for k, s in strat_sets(pf["cascade"]).items()},
                                                      tran=float(np.mean(pv < L.INF)))
            R["duong_cong"][tname] = row
        # ---- phủ ứng viên (không cần sự thật)
        bounds = {k: len(s) for k, s in strat_sets(pf["cascade"]).items()}
        for v in ("cascade", "nomna"):
            bd = {k: len(s) for k, s in strat_sets(pf[v]).items()}
            a, f, tot = L.cand_coverage(Rm, plan["rk_counts"], pos[v], L.NS + list(bd.values()))
            R[f"phu_ung_vien_{v}"] = dict(NS=[round(float(x), 5) for x in a[:len(L.NS)]], ca_tap_NS=[round(float(x), 5) for x in f[:len(L.NS)]],
                                          chien_luoc={k: round(float(a[len(L.NS) + i]), 5) for i, k in enumerate(bd)},
                                          ca_tap_chien_luoc={k: round(float(f[len(L.NS) + i]), 5) for i, k in enumerate(bd)}, n_o_co_R=int(tot))
        # ---- kiểm giữ-ra theo khối trang
        blocks = T.page_blocks(Lb.page.values, 5)
        ho = {t: dict(p=np.full(n, L.INF, dtype=np.int64), flags={k: np.zeros(n, bool) for k in STRAT}) for t in truth
              if t in ("gold", "nguoi", "nguoi_kim_sai", "diban")}
        for k in range(5):
            ev = blocks != k
            ns = L.anchors_counter(Lb, Rm, mask=ev)
            fp = L.filtered(L.plan_book(Lb, Rm, ns, W["n_oth"][b], W["g_oth"][b], MAIN, mask=ev), variants["cascade"])
            fpos = L.positions(fp["ranked"])
            ss = strat_sets(fp)
            for tname, h in ho.items():
                nhan, cl, mask = truth[tname]
                for i in np.nonzero((blocks == k) & mask)[0]:
                    cs = cl[i]
                    h["p"][i] = min(fpos.get(x, L.INF) for x in cs)
                    for kk, s in ss.items():
                        h["flags"][kk][i] = any(x in s for x in cs)
        R["giu_ra"] = {}
        for tname, h in ho.items():
            m = truth[tname][2]
            p = h["p"][m]
            R["giu_ra"][tname] = dict(n_o=int(m.sum()), NS=[round(x, 5) for x in L.curve(p, L.NS)], fine=[round(x, 5) for x in L.curve(p, FINE_NS)],
                                      chien_luoc={kk: float(f[m].mean()) for kk, f in h["flags"].items()}, tran=float(np.mean(p < L.INF)),
                                      N_dat={str(q): n_for(p, q) for q in QS})
        res["sach"][b] = R
        # ---- invariants theo cuốn
        for tname, row in R["duong_cong"].items():
            for v in variants:
                cvv = row[v]["NS"]
                check(f"{b}/{tname}/{v}: đường cong đơn điệu không giảm theo N", all(cvv[i] <= cvv[i + 1] + 1e-12 for i in range(len(cvv) - 1)))
                check(f"{b}/{tname}/{v}: phủ theo thành viên tập = phủ theo vị trí ở biên S0/S1(k)/S2 (tính chất tiền tố)",
                      all(abs(row[v]["chien_luoc"][k] - row[v]["chien_luoc_theo_vi_tri"][k]) < 1e-9 for k in STRAT))
                check(f"{b}/{tname}/{v}: trần (N = cả danh sách) = phủ của S2∪S0 lọc phông", abs(row[v]["tran"] - row[v]["chien_luoc"]["S2_hop_S0"]) < 1e-9)
            check(f"{b}/{tname}: trần phủ: lý tưởng ≥ cascade ≥ NomNaTong một mình",
                  row["ly_tuong"]["tran"] >= row["cascade"]["tran"] - 1e-12 and row["cascade"]["tran"] >= row["nomna"]["tran"] - 1e-12)
        print(f"[b3] {b}: {len(truth)} bộ sự thật [{time.time() - t0:.0f}s]", flush=True)
    check("gt_char của base_of trùng bảng nền TN8 cũ (30/09) trên mọi cell_uid khớp (4 bộ nhãn người)", all(x[2] == 1.0 and x[1] > 0 for x in old_gt), old_gt)
    res["gt_so_voi_tn8_cu"] = old_gt
    res["thoi_gian_giay"] = round(time.time() - t0, 1)
    (L.OUT / "phu.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    nfail = [i["ten"] for i in inv if not i["ok"]]
    print(f"[b3] xong [{time.time() - t0:.0f}s]; {len(inv)} invariant, FAIL: {nfail or 'không'}")


if __name__ == "__main__":
    main()
