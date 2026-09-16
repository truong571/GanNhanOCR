"""B-5 (DANH_MUC 1B) — so sánh bản build --lock-scope col vs cell / cell_fallback (16/09).

Cách chạy (mọi số trong báo cáo B-5 tái lập từ đây):
  SP=<scratch>; .venv/bin/python -m pipeline.align_engine.build_dataset --reseg detector \
      --no-crops --pages KhoiB/v3/b5/pages60.csv --lock-scope col  --out $SP/p60_col
  ... --lock-scope cell --out $SP/p60_cell ; ... --lock-scope cell_fallback --out $SP/p60_fb
  full (có crop, ≈7 phút/lần): bỏ --no-crops/--pages, --out $SP/full_cell | $SP/full_fb
  .venv/bin/python lab/tham_dinh_2026-09-16/b5_khoa_o.py dataset_out_v3/labels.csv $SP/full_cell/labels.csv
Census thật (AE-1/F1/MD5_DUP) cần crop: pipeline.remediation.census.run_census(labels.csv).
Hộp 3 nhánh của chính ô khoá TRƯỚC khi ghi đè bbox_cu: $out/qd01_lock_boxes.csv (build_dataset ghi).
"""
import sys, json, itertools
import pandas as pd

USABLE = ("GOLD", "SILVER", "SYLLABLE")

def load(p):
    d = pd.read_csv(p, dtype={"page": str}, keep_default_na=False)
    d["column"] = d["column"].astype(int)
    d["nom_idx"] = d["nom_idx"].astype(str)
    d["bb"] = d["bbox"].map(lambda s: tuple(json.loads(s)) if s and s != "null" else None)
    return d

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    if i <= 0: return 0.0
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0

def census(d, name):
    u = d[d.tier.isin(USABLE) & d.bb.notna()].copy()
    u["bbs"] = u.bb.map(str)
    dup = u.groupby(["book", "page", "column", "bbs"])["bbs"].transform("size") > 1
    n_ae1_rows = int(dup.sum()); n_ae1_grp = int(u[dup].groupby(["book","page","column","bbs"]).ngroups)
    # IoU >= 0,5 cùng cột và cột kề
    n_same = n_adj = 0; n_same_lock = n_adj_lock = 0
    pairs_lock = []
    for (b, p), g in u.groupby(["book", "page"]):
        cols = {c: list(zip(gg.bb, gg.box_source, gg.nom_idx)) for c, gg in g.groupby("column")}
        for c, items in cols.items():
            for (b1, s1, n1), (b2, s2, n2) in itertools.combinations(items, 2):
                if iou(b1, b2) >= 0.5:
                    n_same += 1
                    if "qd01_locked" in (s1, s2) or "legacy_locked_col" in (s1, s2):
                        n_same_lock += 1; pairs_lock.append((b, p, c, n1, s1, c, n2, s2))
            if c + 1 in cols:
                for (b1, s1, n1) in items:
                    for (b2, s2, n2) in cols[c + 1]:
                        if iou(b1, b2) >= 0.5:
                            n_adj += 1
                            if "qd01_locked" in (s1, s2):
                                n_adj_lock += 1; pairs_lock.append((b, p, c, n1, s1, c+1, n2, s2))
    print(f"[{name}] usable {len(u):,} | AE-1 trùng bbox đúng byte: {n_ae1_rows} ô / {n_ae1_grp} nhóm | "
          f"IoU>=0,5 cùng cột {n_same} cặp (dính ô khoá {n_same_lock}) | cột kề {n_adj} cặp (dính ô khoá {n_adj_lock})")
    for x in pairs_lock[:20]:
        print("   ", x)
    return u

if __name__ == "__main__":
    pc, pl = sys.argv[1], sys.argv[2]
    col, cell = load(pc), load(pl)
    print("rows", len(col), len(cell), "| tiers col", col.tier.value_counts().to_dict(), "| cell", cell.tier.value_counts().to_dict())
    k = ["book", "page", "column", "nom_idx"]
    m = col.merge(cell, on=k, suffixes=("_col", "_cell"), how="outer", indicator=True)
    print("join", m._merge.value_counts().to_dict())
    both = m[m._merge == "both"]
    same_tier = (both.tier_col == both.tier_cell).mean()
    chg = both[both.bb_col != both.bb_cell]
    print(f"tier giống {same_tier:.4%} | ô đổi bbox: {len(chg):,}/{len(both):,} | usable đổi bbox: "
          f"{int(chg.tier_cell.isin(USABLE).sum()):,}/{int(both.tier_cell.isin(USABLE).sum()):,}")
    # ô đổi bbox theo box_source cell / count_source col
    print("  đổi bbox theo box_source_cell:", chg.box_source_cell.value_counts().to_dict())
    print("  đổi bbox theo count_source_col:", chg.count_source_col.value_counts().to_dict())
    # QĐ-01
    for nm, d in (("col", col), ("cell", cell)):
        q = d[d.qd01_locked.astype(str) == "1"]
        print(f"[{nm}] QĐ-01 locked {len(q)} | box_source {q.box_source.value_counts().to_dict()}")
    # phân bố box_source ô usable
    for nm, d in (("col", col), ("cell", cell)):
        u = d[d.tier.isin(USABLE)]
        vc = u.box_source.value_counts()
        det = vc.get("detector", 0) + vc.get("qd01_locked", 0)
        print(f"[{nm}] usable {len(u):,} box_source {vc.to_dict()} | detector+qd01_locked = {det:,} ({det/len(u):.1%})")
    # cột từng khoá (col: legacy_locked_col) -> ở cell thành gì
    lk = both[both.count_source_col == "legacy_locked_col"]
    lku = lk[lk.tier_cell.isin(USABLE)]
    print(f"cột khoá (col=legacy_locked_col): {len(lk):,} ô, usable {len(lku):,} | cell count_source "
          f"{lku.count_source_cell.value_counts().to_dict()} | cell box_source {lku.box_source_cell.value_counts().to_dict()}")
    uc = census(col, "col"); ue = census(cell, "cell")
