"""TN11 v_nguon_goc_wilson (03/10) — Khoảng tin cậy Wilson 95 % cho từng ô của BANG_TONG_HOP_10_BO.md (p05) + đối chiếu Bảng IV (K12).
0 mô hình, 0 API, chỉ ĐỌC.

  1. Với mỗi (bộ, HQC) trong bảng: n = n_cells ở ket_qua_full_10_bo.json, k = round(Top-1 x n / 100) (kiểm k/n làm tròn lại đúng số hiển thị),
     Wilson 95 % (z = 1,96), nửa độ rộng = (hi-lo)/2 và độ lệch một phía lớn nhất = max(p-lo, hi-p). Đếm số ô có CI rộng hơn ±15 điểm.
  2. Với mỗi bộ: ba CI (HQC1/2/3) có đôi một chồng lấn không (bằng chứng yếu cho "không phân biệt được"; kiểm cặp đúng cần dữ liệu từng ô — xem v_nguon_goc_nhieu_mau.py).
  3. Bảng IV (K12): mỗi số 'sau' lấy từ cột nào của p05/p01, mỗi số 'trước' lấy từ đâu; tổng ~54,3 -> 84,7 (+30,4) có khớp trung bình các dòng không;
     'sau' có phải max(HQC1,2,3) theo từng bộ không; 'trước' của 4 bộ nhãn người có phải min(f_font, f_fd) của p01 không.
Ra: measure_out/_tn11/verify/nguon_goc/wilson_va_k12.json
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
M11 = REPO / "measure_out/_tn11"
OUT_DIR = M11 / "verify" / "nguon_goc"
OUT_DIR.mkdir(parents=True, exist_ok=True)
Z = 1.959963984540054


def wilson(k, n, z=Z):
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - hw) * 100, min(1.0, ctr + hw) * 100


def main():
    md = (M11 / "p05_full_corpus/BANG_TONG_HOP_10_BO.md").read_text(encoding="utf-8").splitlines()
    J = {r["book"]: r for r in json.loads((M11 / "p05_full_corpus/ket_qua_full_10_bo.json").read_text(encoding="utf-8"))}
    rows = []
    for ln in md:
        m = re.match(r"\|\s*(\d+)\s*\|\s*\*\*(\w+)\*\*\s*\|(.*)\|\s*([\d.]+)%\s*\(cos\s*([\d.]+)\)\s*\|\s*([\d.]+)%\s*\(cos\s*([\d.]+)\)\s*\|\s*([\d.]+)%\s*\(cos\s*([\d.]+)\)\s*\|", ln)
        if m:
            rows.append({"book": m.group(2), "HQC1": float(m.group(4)), "HQC2": float(m.group(6)), "HQC3": float(m.group(8)),
                         "cos": [float(m.group(5)), float(m.group(7)), float(m.group(9))]})
    assert len(rows) == 10, len(rows)
    cells, wide15, wide15_one, wide30 = [], 0, 0, 0
    for r in rows:
        b = r["book"]; n = J[b]["n_cells"]
        r["n"] = n
        r["human_label"] = J[b]["is_human_label"]
        for h in ("HQC1", "HQC2", "HQC3"):
            p = r[h]
            k = round(p * n / 100)
            consistent = round(k / n * 100, 1) == p
            lo, hi = wilson(k, n)
            hw = (hi - lo) / 2
            dev = max(p - lo, hi - p)
            cells.append({"book": b, "hqc": h, "n": n, "k": k, "top1_pct": p, "k_over_n_khop_so_hien_thi": bool(consistent), "wilson_lo": round(lo, 1), "wilson_hi": round(hi, 1),
                          "nua_do_rong": round(hw, 1), "lech_mot_phia_max": round(dev, 1), "rong_hon_15_nua_do_rong": bool(hw > 15), "rong_hon_15_lech_mot_phia": bool(dev > 15)})
    # bất biến: Wilson tự viết == scipy.stats.binomtest(...).proportion_ci(method="wilson") trên cả 30 ô
    from scipy.stats import binomtest
    mx = 0.0
    for c in cells:
        ci = binomtest(c["k"], c["n"]).proportion_ci(confidence_level=0.95, method="wilson")
        mx = max(mx, abs(ci.low * 100 - c["wilson_lo"]), abs(ci.high * 100 - c["wilson_hi"]))
    print(f"[W] bất biến: lệch lớn nhất giữa Wilson tự viết và scipy (làm tròn 0,1) = {mx:.3f} điểm %")
    n_cells_tbl = len(cells)
    c15 = sum(c["rong_hon_15_nua_do_rong"] for c in cells)
    c15o = sum(c["rong_hon_15_lech_mot_phia"] for c in cells)
    hws = np.array([c["nua_do_rong"] for c in cells])
    # chồng lấn CI trong mỗi bộ
    ov = {}
    for r in rows:
        b = r["book"]
        iv = {c["hqc"]: (c["wilson_lo"], c["wilson_hi"]) for c in cells if c["book"] == b}
        pairs = [("HQC1", "HQC2"), ("HQC1", "HQC3"), ("HQC2", "HQC3")]
        ov[b] = {f"{a}-{c}": bool(max(iv[a][0], iv[c][0]) <= min(iv[a][1], iv[c][1])) for a, c in pairs}
    all_overlap = sum(all(v.values()) for v in ov.values())
    summary = {"n_o_bang": n_cells_tbl, "n_o_khop_k_over_n": int(sum(c["k_over_n_khop_so_hien_thi"] for c in cells)),
               "so_o_CI_rong_hon_pm15_theo_nua_do_rong": int(c15), "so_o_CI_rong_hon_pm15_theo_lech_mot_phia_max": int(c15o),
               "nua_do_rong_min_tb_max": [round(float(hws.min()), 1), round(float(hws.mean()), 1), round(float(hws.max()), 1)],
               "so_bo_ca_3_cap_CI_chong_lan": int(all_overlap), "lech_max_so_voi_scipy_diem_pct": round(mx, 3), "tong_so_o_danh_gia": int(sum(r["n"] for r in rows)),
               "n_moi_bo": {r["book"]: r["n"] for r in rows}}
    print(f"[W] {n_cells_tbl} ô: CI rộng hơn ±15 (nửa độ rộng) = {c15}; (lệch một phía lớn nhất) = {c15o}; nửa độ rộng min/tb/max = {summary['nua_do_rong_min_tb_max']}; "
          f"tổng n = {summary['tong_so_o_danh_gia']} ô; {all_overlap}/10 bộ có cả 3 CI HQC đôi một chồng lấn", flush=True)
    for c in cells:
        print(f"    {c['book']:5s} {c['hqc']} n={c['n']:2d} k={c['k']:2d} {c['top1_pct']:5.1f}% Wilson [{c['wilson_lo']:5.1f}, {c['wilson_hi']:5.1f}] ±{c['nua_do_rong']:4.1f} (lệch max {c['lech_mot_phia_max']:4.1f})"
              f"{'  >15' if c['rong_hon_15_nua_do_rong'] else ''}")

    # --------------------------------------------------------------- K12
    k12 = {"stt2": (60.4, 93.8), "stt4": (58.1, 83.3), "stt11": (61.0, 84.2), "Chr": (74.2, 90.0), "L83": (75.4, 85.0), "KVK": (88.9, 94.4),
           "L16": (49.8, 94.7), "TK": (48.2, 89.5), "B18": (43.0, 70.3), "B34": (34.6, 61.9)}
    p01 = {b: json.loads((M11 / f"p01_{b}.json").read_text(encoding="utf-8"))["top1"] for b in ("B18", "B34", "L16", "TK")}
    out_rows = {}
    for b, (bf, af) in k12.items():
        r = next(x for x in rows if x["book"] == b)
        src_after = [h for h in ("HQC1", "HQC2", "HQC3") if r[h] == af]
        if b in p01:
            src_after += [f"p01.{c}" for c, v in p01[b].items() if v["tat_ca"] == af]
        src_before = []
        if b in p01:
            src_before += [f"p01.{c}" for c, v in p01[b].items() if v["tat_ca"] == bf]
        src_before += [h for h in ("HQC1", "HQC2", "HQC3") if r[h] == bf]
        mx = max(r["HQC1"], r["HQC2"], r["HQC3"])
        out_rows[b] = {"truoc": bf, "sau": af, "gain_neu": round(af - bf, 1), "nguon_sau": src_after, "nguon_truoc": src_before,
                       "sau_la_max_HQC123_cua_p05": bool(af == mx), "p05": {"HQC1": r["HQC1"], "HQC2": r["HQC2"], "HQC3": r["HQC3"], "n": r["n"]},
                       "p05_HQC1_tru_HQC_tot_nhat": round(mx - r["HQC1"], 1)}
        if b in p01:
            out_rows[b]["p01_f_font"] = p01[b]["f_font"]["tat_ca"]; out_rows[b]["p01_f_fd"] = p01[b]["f_fd"]["tat_ca"]
            out_rows[b]["truoc_la_min_f_font_f_fd"] = bool(bf == min(p01[b]["f_font"]["tat_ca"], p01[b]["f_fd"]["tat_ca"]))
    bf_all = np.array([v[0] for v in k12.values()]); af_all = np.array([v[1] for v in k12.values()])
    gains = af_all - bf_all
    p05_means = {h: round(float(np.mean([r[h] for r in rows])), 2) for h in ("HQC1", "HQC2", "HQC3")}
    p05_best = round(float(np.mean([max(r["HQC1"], r["HQC2"], r["HQC3"]) for r in rows])), 2)
    w = np.array([r["n"] for r in rows], float)
    p05_w = {h: round(float(np.sum([r[h] * r["n"] for r in rows]) / w.sum()), 2) for h in ("HQC1", "HQC2", "HQC3")}
    n8 = [b for b in ("stt2", "stt4", "stt11", "Chr", "L83", "KVK", "L16", "TK")]
    rec = {"trung_binh_truoc_10_dong": round(float(bf_all.mean()), 2), "trung_binh_sau_10_dong": round(float(af_all.mean()), 2), "trung_binh_gain_10_dong": round(float(gains.mean()), 2),
           "bao_cao_noi": {"truoc": 54.3, "sau": 84.7, "gain": 30.4}, "khop_sau": bool(abs(af_all.mean() - 84.7) < 0.06), "khop_truoc": bool(abs(bf_all.mean() - 54.3) < 0.06),
           "khop_gain": bool(abs(gains.mean() - 30.4) < 0.06),
           "so_dong_sau_la_max_HQC123_trong_8_bo_p05": int(sum(out_rows[b]["sau_la_max_HQC123_cua_p05"] for b in n8)),
           "so_bo_nhan_nguoi_truoc_la_min_p01": int(sum(out_rows[b].get("truoc_la_min_f_font_f_fd", False) for b in ("B18", "B34", "L16", "TK"))),
           "p05_trung_binh_theo_bo": p05_means, "p05_trung_binh_max_3_HQC": p05_best, "p05_trung_binh_theo_n_o": p05_w,
           "p05_so_bo_HQC2_hon_HQC1": int(sum(r["HQC2"] > r["HQC1"] for r in rows)), "p05_so_bo_HQC2_bang_HQC1": int(sum(r["HQC2"] == r["HQC1"] for r in rows)), "p05_so_bo_HQC2_kem_HQC1": int(sum(r["HQC2"] < r["HQC1"] for r in rows)),
           "p05_so_bo_HQC3_hon_HQC1": int(sum(r["HQC3"] > r["HQC1"] for r in rows)), "p05_so_bo_HQC3_bang_HQC1": int(sum(r["HQC3"] == r["HQC1"] for r in rows)), "p05_so_bo_HQC3_kem_HQC1": int(sum(r["HQC3"] < r["HQC1"] for r in rows)),
           "tong_o_p05": int(w.sum()), "tong_o_ngu_lieu_BAO_CAO_TONG_HOP_2026_10_01": 275136, "ti_le_o_da_do": round(float(w.sum() / 275136 * 100), 3)}
    # trung bình có trọng số theo số ô của mỗi bộ trong bảng gốc (D rows, từ giao_thuc.json) — thử xem 54,3 / 84,7 có sinh ra từ cách tính nào không
    try:
        gt = json.loads((OUT_DIR / "giao_thuc.json").read_text(encoding="utf-8"))["A_giao_thuc"]
        wD = np.array([gt[b]["n_D"] for b in k12], float)
        rec["trung_binh_co_trong_so_n_D"] = {"truoc": round(float((bf_all * wD).sum() / wD.sum()), 2), "sau": round(float((af_all * wD).sum() / wD.sum()), 2), "tong_n_D": int(wD.sum())}
    except Exception as e:  # noqa: BLE001
        rec["trung_binh_co_trong_so_n_D"] = f"không tính được: {e}"
    print(f"[K12] trọng số n_D: {rec['trung_binh_co_trong_so_n_D']}")
    print(f"[K12] trung bình 10 dòng: trước {rec['trung_binh_truoc_10_dong']} -> sau {rec['trung_binh_sau_10_dong']} (gain {rec['trung_binh_gain_10_dong']}); báo cáo nêu 54,3 -> 84,7 (+30,4): "
          f"khớp trước={rec['khop_truoc']} sau={rec['khop_sau']} gain={rec['khop_gain']}")
    print(f"      'sau' = max(HQC1,2,3) ở {rec['so_dong_sau_la_max_HQC123_trong_8_bo_p05']}/8 bộ p05; 'trước' của 4 bộ nhãn người = min(f_font,f_fd) của p01 ở {rec['so_bo_nhan_nguoi_truoc_la_min_p01']}/4")
    print(f"      p05 trung bình theo bộ: HQC1 {p05_means['HQC1']} HQC2 {p05_means['HQC2']} HQC3 {p05_means['HQC3']} | max-3 {p05_best} | theo n ô: {p05_w}")
    print(f"      HQC2 so HQC1: hơn {rec['p05_so_bo_HQC2_hon_HQC1']} / bằng {rec['p05_so_bo_HQC2_bang_HQC1']} / kém {rec['p05_so_bo_HQC2_kem_HQC1']}; HQC3 so HQC1: hơn {rec['p05_so_bo_HQC3_hon_HQC1']} / bằng {rec['p05_so_bo_HQC3_bang_HQC1']} / kém {rec['p05_so_bo_HQC3_kem_HQC1']}; "
          f"tổng ô p05 {rec['tong_o_p05']} / 275.136 = {rec['ti_le_o_da_do']} %")
    for b, v in out_rows.items():
        print(f"      {b:5s} trước {v['truoc']:5.1f} <- {v['nguon_truoc']} | sau {v['sau']:5.1f} <- {v['nguon_sau']} | p05 HQC1/2/3 {v['p05']['HQC1']}/{v['p05']['HQC2']}/{v['p05']['HQC3']} (n={v['p05']['n']}) | HQC tốt nhất − HQC1 = {v['p05_HQC1_tru_HQC_tot_nhat']}")
    (OUT_DIR / "wilson_va_k12.json").write_text(json.dumps({"tom_tat_wilson": summary, "o_wilson": cells, "chong_lan_CI_trong_bo": ov, "k12_doi_chieu": out_rows, "k12_tong": rec}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {OUT_DIR / 'wilson_va_k12.json'}")


if __name__ == "__main__":
    main()
