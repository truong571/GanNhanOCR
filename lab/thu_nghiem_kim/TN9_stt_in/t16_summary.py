"""t16_summary.py — gom số TN9 thành measure_out/_tn9/summary.json (chỉ đọc các JSON do t01…t15 sinh; không tính lại). 0 API."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402


def J(name):
    f = T.OUT / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def wilson(k, n):
    lo, hi = T.lex().wilson(int(k), int(n))
    return [round(lo, 4), round(hi, 4)]


def main():
    rec, cnt, lat = J("stt_recipes.json"), J("count_stats.json"), J("latent2.json")
    cmpf = J("stt_builds/compare__lt1__legacy.json") if (T.OUT / "stt_builds/compare__lt1__legacy.json").exists() else {}
    bm, box, strata, base = J("build_margin.json"), J("box_arbiter.json"), J("box_strata.json"), J("box_baseline.json")
    pr, pv = J("print_recipes.json"), J("print_variant.json")
    S = {"note": "TN9 30/09/2026 — 0 API; mọi số từ script lab/thu_nghiem_kim/TN9_stt_in/t01…t16 (đầu ra measure_out/_tn9/). "
                 "Mức tin: DO = đo trên nhãn người; UL = ước lượng (dị bản / tương tự IHR/Borg); SD = suy đoán (mô hình, không sự thật).",
         "sandbox_repro_stt": cmpf.get("repro_labels_csv"), "stt": {}, "print": {}}
    for bk in T.STT:
        r = rec.get(bk, {}).get("recipes", {})
        c = {k: v.get(bk, {}) for k, v in cnt.items()}
        S["stt"][bk] = dict(
            current=dict(gold_share=r.get("R0_prod", {}).get("share"), prec_text_SD=r.get("R0_prod", {}).get("prec_text"),
                         geo_ok=r.get("R0_prod", {}).get("geo_ok")),
            reads={k: dict(gold=v.get("gold"), gold_over_qn=v.get("gold_over_qn"), cells=v.get("cells"),
                           frac_cols_ocr_eq_qn=v.get("frac_ocr_eq_qn"), ratio_ocr_qn=v.get("ratio_ocr_qn"),
                           syl_without_cell=v.get("syl_without_cell"),
                           cnn_margin_pos=bm.get(k, {}).get(bk, {}).get("margin_pos")) for k, v in c.items()},
            recipes={k: dict(share=v["share"], gold=v["gold"], prec_text_SD=v["prec_text"], geo_ok=v["geo_ok"],
                             comp=v["comp"]) for k, v in r.items()},
            latent_prec_by_class_SD=lat.get(bk, {}).get("range"),
            box_arbiter=box.get(f"stt_l1skel_l2_{bk}"),
            reached_90_90=False)
    S["stt_box"] = dict(borg_calibration=box.get("borg_B18_legacy_vs_vdp"), strata=strata, baseline_same_box=base)
    for b in ("KVK", "L83", "Chr"):
        p = pr.get(b, {})
        v = p.get("variant", {})
        ci = {}
        for k in ("GOLD", "direct_qn_geo", "notplaus_geo"):
            if v.get(k) and v[k][0] is not None:
                ci[k] = dict(rate=v[k][0], n=v[k][1], wilson=wilson(round(v[k][0] * v[k][1]), v[k][1]))
        S["print"][b] = dict(steps=p.get("steps"), variant_agreement_UL=ci)
    S["print"]["ihr_analog_DO"] = pr.get("ihr_analog")
    S["print"]["KVK"]["reached_90_90"] = (pr.get("KVK", {}).get("steps", {}).get("P3", {}).get("share", 0) >= 0.90)
    S["print"]["L83"]["reached_90_90"] = False
    S["print"]["Chr"]["reached_90_90"] = False
    # ---- dòng tiêu đề theo bộ (số lấy từ các khối trên)
    H = {}
    for bk in T.STT:
        rr = S["stt"][bk]["recipes"]
        H[bk] = dict(current=rr["R0_prod"]["share"], current_prec_text_SD=rr["R0_prod"]["prec_text_SD"],
                     best_union=rr["R2_union"]["share"], best_union_prec_text_SD=rr["R2_union"]["prec_text_SD"],
                     best_strict=rr["R4_prec"]["share"], best_strict_prec_text_SD=rr["R4_prec"]["prec_text_SD"],
                     evidence="SD", reached_90_90=False,
                     blockers=["không có sự thật STT", "24-30% vị trí âm không có lần đọc kim ∈ R",
                               "mô hình ảnh STT nhiễm (encoder học nhãn lt1) hoặc yếu (CNN Borg)"])
    for b in ("KVK", "L83", "Chr"):
        st = S["print"][b]["steps"]
        best = "P3" if b == "KVK" else "P2"
        H[b] = dict(current=st["P0"]["share"], best=st[best]["share"], best_recipe=best,
                    evidence="UL" if b != "Chr" else "SD", reached_90_90=bool(S["print"][b]["reached_90_90"]))
    S["headline"] = H
    tp = {bk: dict(gold_over_qn={k: S["stt"][bk]["reads"].get(k, {}).get("gold_over_qn") for k in
                                 ("lt1__legacy", "l1skel_l2__legacy", "twopass__legacy", "lt2__legacy")},
                   kim_geo_gold=J("kimgeo.json").get("twopass__legacy", {}).get(bk, {}).get("geo_own_eq_ocr"),
                   kim_geo_gold_l1skel=J("kimgeo.json").get("l1skel_l2__legacy", {}).get(bk, {}).get("geo_own_eq_ocr"))
          for bk in T.STT}
    S["twopass_verdict"] = dict(adopt=False, why="khung lt1 đã đủ số chữ (cột khớp đếm như lt1); ghép hình học đặt sai chữ lt2 "
                                "(đổi 54-58% vị trí vs 31-34% gióng chuỗi) -> nhiều khe DP, ít GOLD, lệch crop↔nhãn", per_book=tp)
    T.jdump(S, T.OUT / "summary.json")
    print(json.dumps({b: S["print"][b].get("steps") for b in ("KVK", "L83", "Chr")}, ensure_ascii=False))
    print(json.dumps({b: {k: v["share"] for k, v in S["stt"][b]["recipes"].items()} for b in T.STT}, ensure_ascii=False))


if __name__ == "__main__":
    main()
