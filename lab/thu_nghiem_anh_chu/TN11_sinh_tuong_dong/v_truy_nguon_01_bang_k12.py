"""v_truy_nguon_01 (03/10) — TRUY NGUỒN từng số của bảng "Top-1 trước -> sau" (K12) + kiểm số học dòng TỔNG + kiểm K01.
0 API, CPU, chỉ ĐỌC (không sửa pipeline/core/config/data/dataset/prepared/models, không sửa p01-p07).

Cách làm (không do LLM đọc số): dựng một "kho số có nguồn" từ chính tệp đầu ra của dự án, rồi với mỗi số trong K12 liệt kê MỌI mục
cùng bộ có giá trị trùng (|Δ| <= 0,051) kèm loại + n + khoá nguồn. Loại:
  (i)   Top-1 MỘT tín hiệu trên mọi ô có sự thật, ~26 ứng viên (p01_<bộ>.json)
  (ii)  Top-1 trong top-3 ứng viên, n ~ 20 ô (p05_full_corpus/ket_qua_full_10_bo.json)
  (iii) tỉ lệ ô thuộc tầng GOLD (độ PHỦ của pipeline, KHÔNG phải độ đúng)  (dataset/_ALL/gold_exact.csv / labels_final.csv)
  (iv)  khác: các tỉ lệ khác tính được từ bảng nhãn (kim ∈ R, ô neo, ...) để thử mọi cách giải thích số STT.
Ra: measure_out/_tn11/verify/truy_nguon/bang_k12.json

    PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_truy_nguon_01_bang_k12.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "truy_nguon"
OUT.mkdir(parents=True, exist_ok=True)
TN11 = REPO / "measure_out" / "_tn11"

ORDER = ["stt2", "stt4", "stt11", "Chr", "L83", "KVK", "L16", "TK", "B18", "B34"]

# ---- đầu vào = đúng các số trong báo cáo cần kiểm (K01, K12) ------------------------------------------------------------------
K01 = {"stt2": (160, 28589), "stt4": (145, 27526), "stt11": (143, 27427), "Chr": (65, 8016), "L83": (105, 14474),
       "KVK": (163, 22750), "L16": (99, 13760), "TK": (161, 22499), "B18": (529, 90747), "B34": (112, 19348)}
K01_TOTAL = (1682, 275136)
K12 = {  # trước, sau, mức tăng
    "stt2": (60.4, 93.8, 33.4), "stt4": (58.1, 83.3, 25.2), "stt11": (61.0, 84.2, 23.2), "Chr": (74.2, 90.0, 15.8),
    "L83": (75.4, 85.0, 9.6), "KVK": (88.9, 94.4, 5.5), "L16": (49.8, 94.7, 44.9), "TK": (48.2, 89.5, 41.3),
    "B18": (43.0, 70.3, 27.3), "B34": (34.6, 61.9, 27.3)}
K12_TOTAL = (54.3, 84.7, 30.4)
TOL = 0.051


def wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def pool_for(b: str, n_all: dict, gold_cnt: dict, lab_final: pd.DataFrame, C) -> list[dict]:
    """Mọi số có nguồn của bộ b (value tính theo %)."""
    P = []
    # (i) p01 — 4 bộ có nhãn người
    f = TN11 / f"p01_{b}.json"
    if f.exists():
        J = json.loads(f.read_text(encoding="utf-8"))
        n_ts = J["o_co_su_that"]; n_hiem = J["o_chu_dung_khong_co_mau_cung_sach"]
        for m, v in J["top1"].items():
            for g, n in (("tat_ca", n_ts), ("chu_hiem", n_hiem), ("chu_co_mau", n_ts - n_hiem)):
                P.append(dict(dk=f"p01.{m}.{g}", value=v[g], loai="(i) Top-1 một tín hiệu, ~26 ứng viên, mọi ô có sự thật", n=n,
                              nguon=f"measure_out/_tn11/p01_{b}.json:top1.{m}.{g}"))
    # (ii) p05 full corpus — 10 bộ
    J5 = json.loads((TN11 / "p05_full_corpus" / "ket_qua_full_10_bo.json").read_text(encoding="utf-8"))
    for r in J5:
        if r["book"] != b:
            continue
        for k in ("hqc1_white_canvas", "hqc2_book_paper_canvas", "hqc3_canonical_normalized"):
            P.append(dict(dk=f"p05.{k}", value=r[k]["top1"], loai="(ii) Top-1 trong top-3 ứng viên, n ~ 20 ô", n=r["n_cells"],
                          nguon=f"measure_out/_tn11/p05_full_corpus/ket_qua_full_10_bo.json[{b}].{k}.top1"))
    # (iii) GOLD% bản cuối (gold_exact.csv = các ô tầng GOLD của bộ giao nộp)  / mọi ô của bộ
    P.append(dict(dk="gold_final", value=round(100 * gold_cnt[b] / n_all[b], 2), loai="(iii) % ô thuộc tầng GOLD (độ phủ, KHÔNG phải độ đúng)",
                  n=n_all[b], nguon=f"dataset/_ALL/gold_exact.csv (set8=={b}) {gold_cnt[b]} / {n_all[b]} ô"))
    # (iii') GOLD% trong bảng TN8 base (dựng từ bản nhãn cũ hơn với STT)
    D = T.load_base(b)
    P.append(dict(dk="gold_tn8_base", value=round(100 * float((D.tier == "GOLD").mean()), 2), loai="(iii) % ô GOLD trong measure_out/_tn8/base (bản nhãn lúc dựng TN8)",
                  n=len(D), nguon=f"measure_out/_tn8/base/{b}.pkl tier==GOLD"))
    # (iv) các tỉ lệ khác (thử mọi cách giải thích)
    Rm = {s: C.R_of(s) for s in set(D.syllable)}
    kim_in_R = np.array([bool(k) and k in Rm[s] for k, s in zip(D.ocr_char, D.syllable)])
    cells = pd.read_pickle(T.OUT / "cand" / f"{b}_cells.pkl")
    alts = {
        "kim ∈ R(âm) / mọi ô": kim_in_R.mean(),
        "ô neo (direct ∧ nhãn=kim ∈ R ∧ crop ok) / mọi ô": cells.anchor.mean(),
        "nhãn khác rỗng / mọi ô": (D.label != "").mean(),
        "GOLD ∪ SYLLABLE / mọi ô (base)": D.tier.isin(["GOLD", "SYLLABLE", "GOLD_text_only"]).mean(),
        "rule direct* / mọi ô": D.rule.str.startswith("s1_inter_s2_direct").mean(),
        "crop ok / mọi ô": (cells.crop_ok > 0).mean(),
        "n_ocr == n_qn / mọi ô": (D.n_ocr == D.n_qn).mean(),
    }
    for k, v in alts.items():
        P.append(dict(dk=f"alt.{k}", value=round(100 * float(v), 2), loai="(iv) tỉ lệ khác tính từ bảng nhãn", n=len(D), nguon=f"{k} (measure_out/_tn8/base/{b}.pkl)"))
    if b.startswith("stt"):
        R9 = json.loads((REPO / "measure_out/_tn9/stt_recipes.json").read_text(encoding="utf-8"))[b]
        for rc, v in R9["recipes"].items():
            P.append(dict(dk=f"tn9.{rc}", value=round(100 * v["share"], 2), loai="(iii) % GOLD / âm QN theo công thức TN9 (mẫu số khác: số âm QN)",
                          n=R9["N"], nguon=f"measure_out/_tn9/stt_recipes.json:{b}.recipes.{rc}.share"))
        x = lab_final[lab_final.book == b]
        for t in ("GOLD",):
            P.append(dict(dk="gold_labels_final", value=round(100 * float((x.tier == t).mean()), 2), loai="(iii) % ô GOLD trong dataset_out/labels_final.csv (bản hiện hành)",
                          n=len(x), nguon=f"dataset_out/_labels_final.csv book=={b} tier=={t}"))
        P.append(dict(dk="gold_syl_labels_final", value=round(100 * float(x.tier.isin(["GOLD", "SYLLABLE"]).mean()), 2), loai="(iv) GOLD ∪ SYLLABLE trong labels_final",
                      n=len(x), nguon=f"dataset_out/labels_final.csv book=={b}"))
    return P


def main():
    # --- nền: số ô, số trang, GOLD ---------------------------------------------------------------------------------------------
    C = T.lex()
    lab_final = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False, usecols=["book", "page", "tier"])
    G = pd.read_csv(REPO / "dataset/_ALL/gold_exact.csv", dtype=str, keep_default_na=False, usecols=["set8", "gold_exact"])
    gold_cnt = G.groupby("set8").size().to_dict()
    n_all, pages_all, base_cells, base_pages = {}, {}, {}, {}
    for b in ORDER:
        D = T.load_base(b)
        base_cells[b], base_pages[b] = len(D), int(D.page.nunique())
        if b.startswith("stt"):
            x = lab_final[lab_final.book == b]
            n_all[b], pages_all[b] = len(x), int(x.page.nunique())
        else:
            n_all[b], pages_all[b] = len(D), int(D.page.nunique())

    # --- K01 -------------------------------------------------------------------------------------------------------------------
    k01 = []
    for b in ORDER:
        kp, kc = K01[b]
        k01.append(dict(bo=b, bao_cao_trang=kp, bao_cao_o=kc, o_hien_hanh=n_all[b], trang_hien_hanh=pages_all[b],
                        o_trong_tn8_base=base_cells[b], lech_o_so_voi_hien_hanh=kc - n_all[b], lech_o_so_voi_tn8_base=kc - base_cells[b]))
    k01_tot = dict(bao_cao=K01_TOTAL, tong_trang=sum(pages_all.values()), tong_o_hien_hanh=sum(n_all.values()),
                   tong_o_tn8_base=sum(base_cells.values()),
                   tong_bao_cao_tinh_lai=(sum(v[0] for v in K01.values()), sum(v[1] for v in K01.values())))

    # --- K12: truy nguồn ----------------------------------------------------------------------------------------------------------
    pools = {b: pool_for(b, n_all, gold_cnt, lab_final, C) for b in ORDER}
    trace = []
    for b in ORDER:
        for cot, v in (("truoc", K12[b][0]), ("sau", K12[b][1])):
            same = [p for p in pools[b] if abs(p["value"] - v) <= TOL]
            other = []
            for b2 in ORDER:
                if b2 == b:
                    continue
                other += [dict(bo=b2, **p) for p in pools[b2] if abs(p["value"] - v) <= TOL and p["loai"].startswith(("(i)", "(ii)", "(iii) %"))]
            trace.append(dict(bo=b, cot=cot, gia_tri=v, khop_cung_bo=same, khop_bo_khac_so_luong=len(other),
                              khop_bo_khac_mau=[f"{o['bo']}: {o['nguon']}" for o in other[:3]]))
    # --- có định nghĩa nào (công thức TN9, GOLD cũ/mới, kim∈R, ...) cho CẢ BA bộ STT cùng lúc (60,4; 58,1; 61,0)? ------------------
    stt3 = ("stt2", "stt4", "stt11")
    dks = sorted({p["dk"] for bk in stt3 for p in pools[bk]})
    stt_khop = {}
    for d in dks:
        m3 = [any(p["dk"] == d and abs(p["value"] - K12[bk][0]) <= TOL for p in pools[bk]) for bk in stt3]
        if any(m3):
            stt_khop[d] = dict(stt2=m3[0], stt4=m3[1], stt11=m3[2])
    stt_dinh_nghia_khop_ca_ba = [d for d, v in stt_khop.items() if all(v.values())]
    stt_so_dinh_nghia_da_thu = len(dks)
    # --- "Sau" có phải max(HQC1,2,3)? ---------------------------------------------------------------------------------------------
    J5 = {r["book"]: r for r in json.loads((TN11 / "p05_full_corpus" / "ket_qua_full_10_bo.json").read_text(encoding="utf-8"))}
    sau_vs_hqc = []
    for b in ORDER:
        r = J5[b]
        h = [r["hqc1_white_canvas"]["top1"], r["hqc2_book_paper_canvas"]["top1"], r["hqc3_canonical_normalized"]["top1"]]
        sau_vs_hqc.append(dict(bo=b, hqc1=h[0], hqc2=h[1], hqc3=h[2], max_hqc=max(h), sau_bao_cao=K12[b][1], n_p05=r["n_cells"],
                               sau_bang_max_hqc=abs(max(h) - K12[b][1]) <= TOL,
                               cos_hqc=[r["hqc1_white_canvas"]["mean_cos"], r["hqc2_book_paper_canvas"]["mean_cos"], r["hqc3_canonical_normalized"]["mean_cos"]],
                               wilson95_cua_sau=[round(x, 1) for x in wilson(round(K12[b][1] * r["n_cells"] / 100), r["n_cells"])]))
    n_p05 = sum(r["n_cells"] for r in J5.values())
    # tổng ô đúng gộp 10 bộ theo từng HQC (k = round(top1 * n / 100)) — để xem "đồng bộ HQC" có hơn nền trắng không
    gop = {}
    for key in ("hqc1_white_canvas", "hqc2_book_paper_canvas", "hqc3_canonical_normalized"):
        kk = sum(round(r[key]["top1"] * r["n_cells"] / 100) for r in J5.values())
        lo_, hi_ = wilson(kk, n_p05)
        gop[key] = dict(dung=int(kk), tren=int(n_p05), pct=round(100 * kk / n_p05, 1), wilson95=[round(lo_, 1), round(hi_, 1)])
    # số bộ mà "Sau" của báo cáo == HQC1 (nền trắng, baseline cũ của chính p04/p05)
    so_sau_bang_hqc1 = [b for b in ORDER if abs(J5[b]["hqc1_white_canvas"]["top1"] - K12[b][1]) <= TOL]
    so_sau_hon_hqc1 = {b: round(max(J5[b]["hqc2_book_paper_canvas"]["top1"], J5[b]["hqc3_canonical_normalized"]["top1"]) - J5[b]["hqc1_white_canvas"]["top1"], 1) for b in ORDER}

    # --- kiểm số học dòng TỔNG ------------------------------------------------------------------------------------------------------
    tr = np.array([K12[b][0] for b in ORDER]); sa = np.array([K12[b][1] for b in ORDER]); tg = np.array([K12[b][2] for b in ORDER])
    w = np.array([K01[b][1] for b in ORDER], float); wp = np.array([K01[b][0] for b in ORDER], float)
    gain_row_ok = [abs((K12[b][1] - K12[b][0]) - K12[b][2]) <= 0.051 for b in ORDER]

    def wm(x, ww):
        return float((x * ww).sum() / ww.sum())

    arith = dict(
        trung_binh_don=dict(truoc=round(float(tr.mean()), 2), sau=round(float(sa.mean()), 2), tang_trung_binh_cot_tang=round(float(tg.mean()), 2),
                            tang_hieu_hai_tb=round(float(sa.mean() - tr.mean()), 2)),
        trung_binh_theo_so_o_K01=dict(truoc=round(wm(tr, w), 2), sau=round(wm(sa, w), 2), tang=round(wm(sa, w) - wm(tr, w), 2)),
        trung_binh_theo_so_trang=dict(truoc=round(wm(tr, wp), 2), sau=round(wm(sa, wp), 2), tang=round(wm(sa, wp) - wm(tr, wp), 2)),
        trung_vi=dict(truoc=float(np.median(tr)), sau=float(np.median(sa)), tang=float(np.median(tg))),
        bao_cao=dict(truoc=K12_TOTAL[0], sau=K12_TOTAL[1], tang=K12_TOTAL[2]),
        lech_voi_bao_cao=dict(
            truoc_tb_don=round(float(tr.mean()) - K12_TOTAL[0], 2), truoc_tb_theo_o=round(wm(tr, w) - K12_TOTAL[0], 2),
            sau_tb_don=round(float(sa.mean()) - K12_TOTAL[1], 2), sau_tb_theo_o=round(wm(sa, w) - K12_TOTAL[1], 2),
            tang_tb_cot_tang=round(float(tg.mean()) - K12_TOTAL[2], 2), tang_tb_theo_o=round(wm(sa - tr, w) - K12_TOTAL[2], 2)),
        moi_hang_sau_tru_truoc_bang_tang=dict(zip(ORDER, gain_row_ok)),
        hieu_cua_hai_so_TONG_trong_bao_cao=round(K12_TOTAL[1] - K12_TOTAL[0], 2),
    )
    # nếu cột "Trước" của 4 bộ có nhãn người nhất quán = f_font (như B18/B34) thay vì f_fd (L16/TK)
    J = {b: json.loads((TN11 / f"p01_{b}.json").read_text(encoding="utf-8")) for b in ("B18", "B34", "L16", "TK")}
    arith["neu_L16_TK_truoc_la_f_font"] = {b: dict(truoc=J[b]["top1"]["f_font"]["tat_ca"], f_fd=J[b]["top1"]["f_fd"]["tat_ca"],
                                                   tang=round(K12[b][1] - J[b]["top1"]["f_font"]["tat_ca"], 1)) for b in ("L16", "TK")}
    # nếu cột "Trước" của MỌI bộ là GOLD% bản cuối (iii) -> mức tăng
    g_pct = {b: round(100 * gold_cnt[b] / n_all[b], 2) for b in ORDER}
    arith["neu_truoc_la_GOLD_pct_ban_cuoi"] = {b: dict(gold_pct=g_pct[b], sau_bao_cao=K12[b][1], hieu=round(K12[b][1] - g_pct[b], 1)) for b in ORDER}
    arith["gold_pct_ban_cuoi_tb_don"] = round(float(np.mean(list(g_pct.values()))), 2)

    out = dict(k01=k01, k01_tong=k01_tot, k12_truy_nguon=trace, sau_vs_hqc=sau_vs_hqc, tong_o_do_trong_bang_p05=n_p05,
               ty_le_p05_tren_k01=round(100 * n_p05 / K01_TOTAL[1], 4), so_hoc=arith, gop_hqc_10_bo=gop,
               bo_co_sau_bang_hqc1=so_sau_bang_hqc1, max_hqc23_tru_hqc1_diem=so_sau_hon_hqc1,
               stt_truoc_dinh_nghia_da_thu=stt_so_dinh_nghia_da_thu, stt_truoc_khop_tung_bo=stt_khop,
               stt_truoc_dinh_nghia_khop_ca_ba=stt_dinh_nghia_khop_ca_ba)
    (OUT / "bang_k12.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # ---- in gọn ----------------------------------------------------------------------------------------------------------------
    print("== K01 (trang, ô): báo cáo vs hiện hành vs TN8 base")
    for r in k01:
        print(f"  {r['bo']:6s} báo cáo {r['bao_cao_o']:6d} | hiện hành {r['o_hien_hanh']:6d} (Δ {r['lech_o_so_voi_hien_hanh']:+5d}) | TN8 base {r['o_trong_tn8_base']:6d} (Δ {r['lech_o_so_voi_tn8_base']:+5d}) | trang {r['bao_cao_trang']}/{r['trang_hien_hanh']}")
    print("  tổng:", k01_tot)
    print("\n== K12 truy nguồn: khớp CÙNG BỘ (loại · n · nguồn)")
    for t in trace:
        print(f" {t['bo']:6s} {t['cot']:5s} {t['gia_tri']:5.1f}  khớp cùng bộ {len(t['khop_cung_bo'])}; bộ khác {t['khop_bo_khac_so_luong']}")
        for m in t["khop_cung_bo"][:6]:
            print(f"        {m['value']:6.2f}  n={m['n']:6d}  {m['loai'][:60]} | {m['nguon'][:110]}")
    print("\n== Sau == max(HQC1..3)?")
    for r in sau_vs_hqc:
        print(f"  {r['bo']:6s} HQC {r['hqc1']:5.1f}/{r['hqc2']:5.1f}/{r['hqc3']:5.1f} max {r['max_hqc']:5.1f} | Sau báo cáo {r['sau_bao_cao']:5.1f} | n={r['n_p05']:2d} | Sau=max? {r['sau_bang_max_hqc']} | Wilson {r['wilson95_cua_sau']}")
    print(f"\n  Σ n(p05) = {n_p05} ô = {out['ty_le_p05_tren_k01']} % của {K01_TOTAL[1]}")
    print("  gộp 185 ô theo HQC:", gop)
    print("  bộ có 'Sau' == HQC1 (nền trắng):", so_sau_bang_hqc1, "| max(HQC2,HQC3) − HQC1 (điểm):", so_sau_hon_hqc1)
    print(f"  STT 'Trước': đã thử {stt_so_dinh_nghia_da_thu} định nghĩa; khớp từng bộ: {stt_khop}; khớp CẢ BA: {stt_dinh_nghia_khop_ca_ba}")
    print("\n== SỐ HỌC dòng TỔNG")
    print(json.dumps(arith, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
