"""t18_test_route.py — kiểm bản CÀI pipeline.stt_hai_luot (union + gate) trên bản dựng hộp cát, so đích mô phỏng t17. 0 API.
Thư mục tạm measure_out/_tn9/route_test/{P,A,work}: P = l1skel_l2__visual_dp, A = lt1__visual_dp (chỉ CSV; crop không chép),
boxes_syl.csv dựng từ bản legacy cùng đọc (hộp legacy = hộp tham chiếu syl_index). Không rescue (t17 không có).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402
from pipeline.stt_hai_luot import route as RT  # noqa: E402


def main():
    root = T.OUT / "route_test"
    if root.exists():
        shutil.rmtree(root)
    for nm, src, leg in (("P", "l1skel_l2__visual_dp", "l1skel_l2__legacy"), ("A", "lt1__visual_dp", "lt1__legacy")):
        d = root / nm; d.mkdir(parents=True)
        shutil.copy2(T.OUT / "stt_builds" / src / "dataset_out/labels_final.csv", d / "labels_final.csv")
        V = T.rd(T.OUT / "stt_builds" / src / "dataset_out/labels_final.csv")
        Lg = T.rd(T.OUT / "stt_builds" / leg / "dataset_out/labels_final.csv")
        Lg["k"] = Lg.book + "/" + Lg.page + "/" + Lg.column + "/" + Lg.syl_idx
        m = dict(zip(Lg.k, Lg.bbox))
        V["k"] = V.book + "/" + V.page + "/" + V.column + "/" + V.syl_idx
        B = pd.DataFrame(dict(book=V.book, page=V.page, column=V.column, nom_idx=V.nom_idx, syl_idx=V.syl_idx, bbox=V.bbox,
                              bbox_syl=[m.get(k, "") for k in V.k], box_source=V.box_source))
        B.to_csv(d / "boxes_syl.csv", index=False)
    rep_u = RT.union(root / "P", root / "A", root / "work", workers=6)
    rep_g = RT.gate(root / "P/labels_final.csv", root / "work", root / "P", root / "A", "strict", True)
    sim = json.loads((T.OUT / "sim_route.json").read_text())["visual_dp"]
    out = {}
    for s8, book in (("stt2", "stt2"), ("stt4", "stt4"), ("stt11", "stt11")):
        g = rep_g["per_book"].get(book, {})
        out[s8] = dict(cai=dict(gold=g.get("gold_sau"), share=g.get("ty_le_gold_sau"), ha_vdp=g.get("ha_vdp"), o=g.get("o")),
                       mo_phong=dict(gold=sim[s8]["gold_final"], share=sim[s8]["share_final"], ha_vdp=sim[s8]["vdp_demoted"],
                                     o=sim[s8]["keys"]))
        print(f"[t18] {s8}: cài {out[s8]['cai']} | mô phỏng {out[s8]['mo_phong']}", flush=True)
    T.jdump(dict(union=rep_u, gate=rep_g, so_sanh=out), T.OUT / "route_test.json")


if __name__ == "__main__":
    main()
