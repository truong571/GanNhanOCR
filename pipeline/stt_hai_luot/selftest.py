"""selftest — đường STT hai lượt (0 API, dữ liệu tổng hợp + từ điển thật).

  .venv/bin/python -m pipeline.stt_hai_luot selftest
Kiểm: cấu hình (mặc định/lỗi), khoá books[].kim_read, phép ghép l1skel_l2 (hình học giữ nguyên, chữ lt2 vào vị trí gióng
được, khoảng trống giữ lt1, lt2 khác ảnh -> không ghép), lớp bằng chứng, config bản phụ, hợp theo ô + cổng strict/union/off
+ cổng vdp + trả lại hàng rescue bị hạ (đầu-cuối trên hai bảng nhãn nhỏ).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_res = [0, []]


def chk(name, cond, detail=""):
    if cond:
        _res[0] += 1
        print(f"  ok   {name}")
    else:
        _res[1].append(name)
        print(f"  FAIL {name}  {detail}")


def _line(x, y, h, s):
    return {"transcription": s, "points": [[x, y], [x + 50, y], [x + 50, y + h], [x, y + h]], "difficult": False}


def t_cfg():
    from . import route as RT
    d = RT.cfg_of({})
    chk("cfg vắng mục -> enabled false (đường cũ)", d["enabled"] is False and d["declared"] is False)
    d = RT.cfg_of({"stt_hai_luot": {"enabled": True}})
    chk("cfg mặc định gate strict, aux lt1", d["gate"] == "strict" and d["aux_read"] == "lt1" and d["vdp_gate"] is False)
    for bad in ({"gate": "xx"}, {"aux_read": "lt3"}, {"enabled": "yes"}, {"la": 1}):
        try:
            RT.cfg_of({"stt_hai_luot": bad}); ok = False
        except ValueError:
            ok = True
        chk(f"cfg sai {list(bad)} -> ValueError", ok)


def t_layout():
    from pipeline.align_engine.book_layout import DEFAULT_LAYOUT, book_layout
    chk("kim_read vắng / lt1 -> DEFAULT_LAYOUT (STT cũ không đổi byte)",
        book_layout({"name": "s"}) is DEFAULT_LAYOUT and book_layout({"name": "s", "kim_read": "lt1"}) is DEFAULT_LAYOUT)
    L = book_layout({"name": "s", "kim_read": "l1skel_l2"})
    chk("kim_read l1skel_l2 -> thư mục kim_l1skel_l2", L.kim_cache_dir == "kim_l1skel_l2" and DEFAULT_LAYOUT.kim_cache_dir == "detected")
    try:
        book_layout({"name": "s", "kim_read": "lt9"}); ok = False
    except ValueError:
        ok = True
    chk("kim_read sai -> ValueError", ok)


def t_merge():
    from core.ocr.ocr_api import boxes_to_columns
    from .doc import merged_cache
    raw1 = [_line(100, 100, 300, "羅天甲"), _line(200, 100, 200, "乙丙")]
    c1 = {"image_hash": "h", "coords_space": "fullpage", "boxes_raw": raw1, "columns": boxes_to_columns(raw1)}
    raw2 = [_line(100, 100, 300, "𪜀天甲"), _line(200, 100, 200, "丙")]
    c2 = {"image_hash": "h", "coords_space": "fullpage", "boxes_raw": raw2}
    m, st = merged_cache(c1, c2)
    geo = [[c["bbox"] for c in col] for col in m["columns"]] == [[c["bbox"] for c in col] for col in c1["columns"]]
    chk("ghép: hình học cột/hộp giữ nguyên lt1", geo and st["fallback"] == 0)
    flat = {c["lt1"]: (c["char"], c["lt2"]) for col in m["columns"] for c in col}
    chk("ghép: vị trí gióng lấy chữ lt2 (羅 -> 𪜀), trùng giữ", flat["羅"] == ("𪜀", "𪜀") and flat["天"] == ("天", "天"))
    chk("ghép: vị trí khoảng trống (乙, dòng lt2 ngắn) giữ chữ lt1, lt2 rỗng", flat["乙"] == ("乙", "") and flat["丙"] == ("丙", "丙"))
    chk("ghép: thống kê vị trí", st["pos"] == 5 and st["pos_l2"] == 4 and st["pos_changed"] == 1, str(st))
    m2, st2 = merged_cache(c1, dict(c2, image_hash="khac"))
    chk("lt2 khác ảnh -> không ghép, chữ = lt1", st2["no_lt2"] == 1
        and all(c["char"] == c["lt1"] and c["lt2"] == "" for col in m2["columns"] for c in col))


def t_classify():
    from . import route as RT
    R = {"là": {"羅", "𪜀"}}
    eq = lambda a, b: bool(a) and bool(b) and a == b  # noqa: E731
    cases = [("羅", "羅", "羅", "s1_inter_s2_direct", "agree"), ("羅", "𪜀", "𪜀", "s1_inter_s2_direct", "conflict"),
             ("天", "𪜀", "𪜀", "s1_inter_s2_direct", "lt2only"), ("羅", "", "羅", "s1_inter_s2_direct", "lt1_nolt2"),
             ("羅", "天", "羅", "s1_inter_s2_direct", "lt1_l2out"), ("天", "", "羅", "self_training_rescue", "rescue"),
             ("天", "", "羅", "s1_inter_s2_similar", "similar"), ("天", "", "天", "x", "other")]
    got = RT.classify([c[0] for c in cases], [c[1] for c in cases], ["là"] * len(cases), [c[2] for c in cases],
                      [c[3] for c in cases], lambda s: R.get(s, set()), eq)
    chk("lớp bằng chứng 8 trường hợp", list(got) == [c[4] for c in cases], str(list(got)))


COLS = ["image", "book", "page", "column", "ocr_char", "syllable", "label", "unicode", "label_level", "tier", "rule",
        "image_md5", "bbox", "nom_idx", "syl_idx", "box_source", "label_canonical", "tier_goc", "rule_goc"]


def _row(syl_idx, oc, lab, tier, rule, img="", bbox=None, bs="vdp_det"):
    bbox = bbox or f"[0, {20 * syl_idx}, 10, {20 * syl_idx + 10}]"     # hộp riêng mỗi ô (trừ khi cố ý trùng)
    return dict(image=img, book="stt2", page="page_0001", column="1", ocr_char=oc, syllable="là", label=lab,
                unicode="", label_level="char" if lab else "", tier=tier, rule=rule, image_md5="m", bbox=bbox,
                nom_idx=str(syl_idx), syl_idx=str(syl_idx), box_source=bs, label_canonical=lab, tier_goc="", rule_goc="")


def t_route(mode, vdp):
    from . import route as RT
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); P, A, W = td / "P", td / "A", td / "W"
        for d in (P, A):
            (d / "gold").mkdir(parents=True)
        (A / "gold" / "a2.png").write_bytes(b"png-a2")
        rowsP = [_row(1, "𪜀", "𪜀", "GOLD", "s1_inter_s2_direct"),                       # k1 lt2only
                 _row(2, "羅", "", "REVIEW", "no_context"),                                # k2 P REVIEW, A GOLD
                 _row(3, "羅", "羅", "GOLD", "s1_inter_s2_direct"),                        # k3 lt1_l2out -> hạ
                 _row(4, "𪜀", "𪜀", "GOLD", "s1_inter_s2_direct"),                        # k4 xung đột -> hạ
                 _row(5, "羅", "羅", "GOLD", "s1_inter_s2_direct", bs="vdp_low"),  # k5 vdp lệch
                 _row(6, "天", "", "REVIEW", "no_context"),                                # k6 sẽ bị rescue
                 _row(7, "羅", "羅", "GOLD", "s1_inter_s2_direct", bs="vdp_low"),  # k7 vdp trùng
                 _row(8, "羅", "羅", "GOLD", "s1_inter_s2_direct", img="gold/p8.png", bbox="[5, 5, 20, 20]"),   # k8+k9 chung
                 _row(9, "羅", "羅", "GOLD", "s1_inter_s2_direct", img="gold/p9.png", bbox="[5,5,20,20]")]      # hộp -> AE-1
        rowsA = [_row(1, "天", "", "SYLLABLE", "syl_ctx:x"), _row(2, "羅", "羅", "GOLD", "s1_inter_s2_direct", img="gold/a2.png"),
                 _row(3, "羅", "羅", "GOLD", "s1_inter_s2_direct"), _row(4, "羅", "羅", "GOLD", "s1_inter_s2_direct"),
                 _row(5, "羅", "羅", "GOLD", "s1_inter_s2_direct"), _row(6, "天", "", "REVIEW", "no_context"),
                 _row(7, "羅", "羅", "GOLD", "s1_inter_s2_direct"), _row(8, "羅", "羅", "GOLD", "s1_inter_s2_direct"),
                 _row(9, "羅", "羅", "GOLD", "s1_inter_s2_direct")]
        pd.DataFrame(rowsP, columns=COLS).to_csv(P / "labels_final.csv", index=False)
        pd.DataFrame(rowsA, columns=COLS).to_csv(A / "labels_final.csv", index=False)
        L2 = {"1": "𪜀", "2": "", "3": "天", "4": "𪜀", "5": "羅", "6": "", "7": "羅", "8": "羅", "9": "羅"}
        fake = lambda sub, bk, wd, w, log: np.array([L2[s] for s in sub.syl_idx], dtype=object)  # noqa: E731
        RT.union(P, A, W, l2_fn=fake, log=lambda m: None)
        U = pd.read_csv(P / "labels_final.csv", dtype=str, keep_default_na=False).set_index("syl_idx")
        chk(f"[{mode}] hợp: ô 2 lấy hàng bản phụ (GOLD) + crop chép hl_lt1__", U.at["2", "tier"] == "GOLD"
            and U.at["2", "image"] == "gold/hl_lt1__a2.png" and (P / "gold/hl_lt1__a2.png").read_bytes() == b"png-a2")
        chk(f"[{mode}] hợp: ô 1 lấy hàng bản chính (GOLD lt2)", U.at["1", "label"] == "𪜀" and U.at["1", "hai_luot"] == "P:lt2only")
        # rescue giả lập trên ô 6
        L = pd.read_csv(P / "labels_final.csv", dtype=str, keep_default_na=False)
        i6 = L.index[L.syl_idx == "6"][0]
        L.loc[i6, ["tier", "rule", "label", "image"]] = ["GOLD", "self_training_rescue", "羅", "gold/rescue_6.png"]
        L.to_csv(P / "labels_final.csv", index=False)
        for d in (P, A):
            pd.DataFrame([dict(book="stt2", page="page_0001", column="1", nom_idx=k, syl_idx=k, bbox="",
                               bbox_syl=("[0, 300, 10, 310]" if k == "5" else "[0, 140, 10, 150]"), box_source="")
                          for k in ("5", "7")]).to_csv(d / "boxes_syl.csv", index=False)
        rep = RT.gate(P / "labels_final.csv", W, P, A, mode, vdp, log=lambda m: None)
        G = pd.read_csv(P / "labels_final.csv", dtype=str, keep_default_na=False).set_index("syl_idx")
        gold = set(G.index[G.tier == "GOLD"])
        if mode == "strict":
            exp = {"1", "2"} | ({"7"} if vdp else {"5", "7"})
            chk(f"[strict vdp={vdp}] GOLD còn lại", gold == exp, str(sorted(gold)))
            chk("[strict] hạ lt1_l2out / xung đột -> REVIEW + luật đánh dấu + tier_goc",
                G.at["3", "tier"] == "REVIEW" and G.at["3", "rule"].endswith("|hai_luot:ha_lt1_l2out")
                and G.at["3", "tier_goc"] == "GOLD" and G.at["4", "rule"].endswith("|hai_luot:ha_conflict")
                and G.at["3", "hai_luot_truoc"] == "GOLD|羅")
            chk("[strict] rescue không đồng thuận -> trả lại hàng TRƯỚC rescue", G.at["6", "tier"] == "REVIEW"
                and G.at["6", "label"] == "" and G.at["6", "image"] == "" and G.at["6", "rule"] == "no_context|hai_luot:ha_rescue")
            chk("[strict] nguồn gốc: nhãn lt2 / lấy bản lt1", G.at["1", "rule"].endswith("|hai_luot:lt2")
                and G.at["2", "rule"].endswith("|hai_luot:lt1"))
            if vdp:
                chk("[strict] vdp_low lệch syl_index -> REVIEW; trùng -> giữ",
                    G.at["5", "rule"].endswith("|hai_luot:vdp_vdp_low") and G.at["7", "tier"] == "GOLD"
                    and rep["per_book"]["stt2"]["ha_vdp"] == 1)
        else:
            chk(f"[{mode}] không hạ ô nào theo lớp", gold == {"1", "2", "3", "4", "5", "6", "7"} - ({"5"} if vdp else set()),
                str(sorted(gold)))
        chk(f"[{mode}] hai ô dùng được chung một hộp cùng cột (AE-1) -> cả hai REVIEW |hai_luot:trung_hop",
            G.at["8", "tier"] == "REVIEW" and G.at["9", "tier"] == "REVIEW" and G.at["8", "rule"].endswith("|hai_luot:trung_hop")
            and rep["per_book"]["stt2"]["ha_trung_hop"] == 2)


def t_aux_config():
    import yaml
    from . import route as RT
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "c.yaml"
        src.write_text(yaml.safe_dump({"books": [{"name": "SachThanhTruyen2", "kim_read": "l1skel_l2", "box_decoder": "visual_dp"}],
                                       "paths": {"data_dir": "prepared"},
                                       "stt_hai_luot": {"enabled": True, "aux_dir": "x/aux", "vdp_gate": True}}), encoding="utf-8")
        RT.aux_config(src, Path(td) / "a.yaml")
        a = yaml.safe_load((Path(td) / "a.yaml").read_text(encoding="utf-8"))
        chk("config bản phụ: bỏ kim_read (lt1), giữ box_decoder + mục stt_hai_luot, dataset_out = aux_dir",
            "kim_read" not in a["books"][0] and a["books"][0]["box_decoder"] == "visual_dp"
            and a["run"]["dataset_out"] == "x/aux" and a["stt_hai_luot"]["vdp_gate"] is True)


def run() -> tuple[int, list]:
    t_cfg(); t_layout(); t_merge(); t_classify(); t_aux_config()
    for mode, vdp in (("strict", True), ("strict", False), ("union", False), ("off", True)):
        t_route(mode, vdp)
    return _res[0], _res[1]


def main() -> int:
    ok, fail = run()
    print(f"RESULT: {ok} passed, {len(fail)} failed" + (f" {fail}" if fail else ""))
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
