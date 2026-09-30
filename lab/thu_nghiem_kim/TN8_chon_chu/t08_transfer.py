"""t08_transfer.py — áp bộ chọn chữ cho bộ KHÔNG có sự thật: mô hình học trên CẢ HAI bộ có sự thật cùng họ
(viết tay: B18+B34 -> STT; in: L16+TK -> Chr, L83, KVK), ngưỡng chọn trên dự đoán ngoài-khối của hai bộ học (tiêu chí
biên). STT chạy 3 chế độ kim: lt1 (đọc hiện hành), lt2 (Nôm), union (lt1 ∪ lt2, phá hoà bằng ảnh). 0 API.
Ra: measure_out/_tn8/pred/<bộ>__from_<họ>__<chế độ>.pkl, measure_out/_tn8/models/<họ>.pkl
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
import t06_eval as E  # noqa: E402

FAM = {"hand": ["B18", "B34"], "print": ["L16", "TK"]}
MOD = T.OUT / "models"


def train_family(fam: str, force=False):
    f = MOD / f"{fam}.pkl"
    if f.exists() and not force:
        return pickle.load(open(f, "rb"))
    t0 = time.time()
    train = [E.load_book(b) for b in FAM[fam]]
    models, oof = {}, {b: [] for b in FAM[fam]}
    for part in ("in", "out"):
        models[part] = E.fit_models(train, part)
        for k in range(5):
            m1k, m2k = E.fit_models(train, part, blocks_excl={b: [k] for b in FAM[fam]})
            for Bk in train:
                rows = np.nonzero(Bk["cells"].blk.values == k)[0]
                oof[Bk["b"]].append(E.predict(Bk, m1k, m2k, part, rows))
    O = pd.concat([E.assemble(Bk, oof[Bk["b"]]).assign(book=Bk["b"]) for Bk in train])
    MOD.mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(models=models, oof=O), open(f, "wb"))
    print(f"[t08] học họ {fam} xong [{time.time() - t0:.0f}s]", flush=True)
    return dict(models=models, oof=O)


def apply(fam: str, b: str, mode="kim"):
    M = train_family(fam)
    Bk = E.load_book(b, mode)
    preds = [E.predict(Bk, *M["models"][part], part) for part in ("in", "out")]
    out = E.assemble(Bk, preds)
    out.to_pickle(T.OUT / "pred" / f"{b}__from_{fam}__{mode}.pkl")
    return out


if __name__ == "__main__":
    jobs = sys.argv[1:] or ["stt2:kim", "stt2:lt2", "stt2:union", "stt4:kim", "stt4:lt2", "stt4:union", "stt11:kim", "stt11:lt2",
                            "stt11:union", "Chr:kim", "L83:kim", "KVK:kim"]
    for j in jobs:
        b, mode = j.split(":")
        fam = "hand" if b.startswith("stt") else "print"
        o = apply(fam, b, mode)
        print(f"[t08] {b} ({mode}): {len(o)} ô, phần {o.part.value_counts().to_dict()}", flush=True)
