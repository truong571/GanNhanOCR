"""t00_base.py — bảng ô ĐẦY ĐỦ (mọi tầng) của 10 bộ + sự thật nối sẵn. 0 API, CPU, không mở ảnh.

Sự thật (chỉ để ĐO, không vào quyết định):
  B18/B34  chữ người = borg_endtoend_eval (ô -> chỉ số âm trang -> câu "đếm bằng" -> chữ Nôm người);
           vị trí = tn6lib.join (tâm bbox ∈ hộp người keep_high+; pos_known = có hộp người tin được).
  L16/TK   chữ người + khe = pipeline.gold_exact.eval_ihr.Slot (tâm bbox; khe '' = sai, như TN6/eval_ihr).
  L83/KVK  chữ DỊ BẢN cùng vị trí âm (auto_precision.match_verses, âm giống hệt) — ƯỚC LƯỢNG, không phải sự thật.
Ra: measure_out/_tn8/base/<bộ>.pkl
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import tn8lib as T  # noqa: E402

COLS = ["page", "column", "nom_idx", "syl_idx", "syllable", "ocr_char", "label", "tier", "rule", "gate_reason", "box_source",
        "count_source", "bbox", "crop_quality_flag", "p_register", "n_ocr", "n_qn", "n_det", "qn_count_unfixed", "image",
        "tier_v3", "context_evidence", "book"]


def to_int(s):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return -1


def base_of(b: str) -> pd.DataFrame:
    cfg = T.BOOKS[b]
    L = T.rd(T.REPO / cfg["labels"])
    if cfg.get("book_filter"):
        L = L[L.book == cfg["book_filter"]]
    L = L.reset_index(drop=True)
    for c in COLS:
        if c not in L.columns:
            L[c] = ""
    D = L[COLS].copy()
    D.rename(columns={"book": "book_col"}, inplace=True)
    D["set"] = b
    D["column"] = D.column.map(to_int)
    D["nom_i"] = D.nom_idx.map(to_int)
    D["syl_i"] = D.syl_idx.map(to_int)
    D["cell_uid"] = [f"{cfg['full']}/{bc}/{p}/c{c}/n{n}/s{s}" for bc, p, c, n, s in
                     zip(D.book_col, D.page, D.column, D.nom_idx, D.syl_idx)]
    D["gt_char"] = ""
    D["pos_known"] = False
    D["pos_ok"] = False
    D["ref"] = ""           # chữ dị bản (L83/KVK), ghép bằng '|'
    tr = cfg["truth"]
    if tr == "borg":
        import borg_endtoend_eval as BE
        import tn6lib as TL
        book = cfg["full"]
        H = BE.human_pages(book)
        spans = BE.load_spans(book)
        s2c = {pg: BE.syl_to_char(h) for pg, h in H.items()}
        gts = []
        for pg, col, si in zip(D.page, D.column, D.syl_i):
            j0 = spans.get((pg, col))
            g = ""
            if j0 is not None and si >= 0:
                ch = s2c.get(pg, [])
                J = j0 + si
                if 0 <= J < len(ch) and ch[J]:
                    g = ch[J]
            gts.append(g)
        D["gt_char"] = gts
        Lj = L.copy()
        J = TL.join(book, Lj)
        D["pos_known"] = (J.has_h & J.keep_level.isin(TL.KEEP_OK)).values
        D["pos_ok"] = (D.pos_known & J.slot_ok.values).values
        D["keep_level"] = J.keep_level.values
    elif tr == "ihr":
        from pipeline.gold_exact.eval_ihr import Slot
        gcfg = yaml.safe_load(open(T.REPO / "config/gold_exact.yaml", encoding="utf-8"))
        S = Slot(cfg["full"], gcfg["ihr_slot_template"][cfg["full"]])
        gt, sl = [], []
        for pg, col, si, bb in zip(D.page, D.column, D.syl_i, D.bbox):
            b4 = T.bbox_of(bb)
            cx, cy = ((b4[0] + b4[2]) / 2, (b4[1] + b4[3]) / 2) if b4 else (None, None)
            try:
                g = S.gt_char(pg, col, si) if si >= 0 else ""
                s = S.slot(pg, col, si, cx, cy) if si >= 0 else ""
            except Exception:  # noqa: BLE001
                g, s = "", ""
            gt.append(g); sl.append(s)
        D["gt_char"] = gt
        D["slot"] = sl
        D["pos_known"] = D.gt_char != ""
        D["pos_ok"] = np.array(sl) == "1"
    elif tr == "diban":
        import auto_precision as AP
        acfg = dict(AP.CROSS_BOOKS[cfg["full"]])
        acfg["refs"] = list(acfg["refs"])
        verses, _ = AP.match_verses(cfg["full"], acfg, 0.75)
        refs = defaultdict(list)
        for v in verses:
            for i in range(v["n"]):
                if v["same_pos"][i] == "1":
                    refs[(v["page"], int(v["column"]), v["syl_offset"] + i)].append(v["ref_nom"][i])
        D["ref"] = ["|".join(refs.get((p, c, s), [])) for p, c, s in zip(D.page, D.column, D.syl_i)]
    return D


def main():
    t0 = time.time()
    for b in (sys.argv[1:] or T.ORDER):
        D = base_of(b)
        T.base_path(b).parent.mkdir(parents=True, exist_ok=True)
        D.to_pickle(T.base_path(b))
        print(f"[t00] {b}: {len(D)} ô, GOLD {int((D.tier == 'GOLD').sum())} ({(D.tier == 'GOLD').mean():.3f}), "
              f"gt {int((D.gt_char != "").sum())}, pos_known {int(D.pos_known.sum())}, ref {int((D.ref != '').sum())} "
              f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
