"""v_pb_truy_nguon_06 (04/10) — PHẢN BIỆN: p05 dùng rng = default_rng(hash(book) % 100000). Người kiểm chứng kết luận 'chạy lại sẽ không ra
BANG_TONG_HOP_10_BO.md' vì hash(str) bị xáo mỗi tiến trình. Kiểm: với PYTHONHASHSEED cố định (0..255) thì dãy n_cells (số ô giữ lại sau top-3) của 10 bộ có
trùng JSON gốc (16,18,19,20,20,18,19,19,18,18) không? Dùng đúng logic chọn ô của p05:166-201 (chỉ bảng ứng viên, 0 mô hình). 0 API, CPU.
Ra: measure_out/_tn11/verify/pb_truy_nguon/tai_lap_hash.json
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T
OUT = REPO / "measure_out/_tn11/verify/pb_truy_nguon"
J = {r["book"]: r["n_cells"] for r in json.loads((REPO / "measure_out/_tn11/p05_full_corpus/ket_qua_full_10_bo.json").read_text(encoding="utf-8"))}
print("n_cells goc:", J, flush=True)
pre = {}
for b in T.ORDER:
    D = T.load_base(b)
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "f_vW"]]
    if "y" in F.columns and F.y.sum() > 0:
        tru = F[F.y == 1].drop_duplicates("i").set_index("i"); target = np.array(list(tru.index))
    else:
        lbl = D.label.values; tgt = lbl[F.i.values]; vm = (F.c.values == tgt) & (tgt != "")
        target = np.array(sorted(set(F.i.values[vm]))); F["y"] = np.where(vm, 1, 0)
    # top-3 theo f_vW cho mọi ô mục tiêu (đúng p05:196-197) -> ô có chữ đúng trong top-3?
    S = F[F.i.isin(target)].copy(); S["rk"] = S.f_vW.fillna(-9)
    S = S.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(3)
    has = S.groupby("i").y.max()
    valid3 = set(has[has == 1].index)
    pre[b] = (target, valid3)
    print(b, "target", len(target), "valid3", len(valid3), flush=True)
    del F, S
res = {"khop": [], "thu": {}}
for hs in list(range(0, 257)):
    env = {**os.environ, "PYTHONHASHSEED": str(hs)}
    code = "print(' '.join(str(hash(b) % 100000) for b in {}))".format(list(T.ORDER))
    o = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).stdout.split()
    seeds = dict(zip(T.ORDER, map(int, o)))
    n = {}
    for b in T.ORDER:
        target, valid3 = pre[b]
        rng = np.random.default_rng(seeds[b])
        sel = np.sort(rng.choice(target, size=min(20, len(target)), replace=False))
        n[b] = int(sum(1 for i in sel if i in valid3))
    ok = sum(n[b] == J[b] for b in T.ORDER)
    res["thu"][hs] = dict(seeds=seeds, n=n, so_bo_khop=ok)
    if ok == 10: res["khop"].append(hs)
best = sorted(res["thu"].items(), key=lambda kv: -kv[1]["so_bo_khop"])[:5]
print("PYTHONHASHSEED khop 10/10:", res["khop"]); print("top 5 so bo khop:", [(k, v["so_bo_khop"]) for k, v in best])
print("PYTHONHASHSEED=0 :", res["thu"][0]["n"], "so khop", res["thu"][0]["so_bo_khop"])
(OUT / "tai_lap_hash.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
