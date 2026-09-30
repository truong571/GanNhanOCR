"""t07_pair_arbiter.py — ai đúng khi hai lần đọc kim STT bất đồng (lt1 Hán vs lt2 Nôm, cả hai ∈ R(âm))? 0 API.

Trọng tài = CNN kiểm ảnh↔chữ SẠCH với STT (hand_*_Kinh, học Borg Kinh + IHR; KHÔNG học STT) — điểm f_vW (TN8 cand).
Encoder MultiEnc v1/v2 học trên crop STT nhãn lt1 (trong mẫu) nên KHÔNG dùng làm trọng tài.
1) Borg (có người, CNN LOBO như TN8: B18 <- hand_*_DungLy, B34 <- hand_*_Kinh): trên ô kim SAI mà kim ∈ R (tình huống
   "hai chữ ∈ R, một đúng"), độ đúng cặp acc = P(f_vW(chữ người) > f_vW(chữ kim)); tách theo khối Unicode chữ người.
2) STT: ô xung đột (lt1 ∈ R, lt2 ∈ R, khác nhau) — tỉ lệ f = P(f_vW(lt2) > f_vW(lt1)); nếu một trong hai đúng và trọng tài
   đúng cặp với xác suất acc (đối xứng): π2 = P(lt2 đúng) = (f − (1 − acc)) / (2·acc − 1). acc STT ≤ acc Borg (CNN yếu hơn
   trên STT: top-1 ô chắc 81,5 % vs 90–94 % Borg) -> báo π2 theo acc Borg và theo acc_STT ước từ ô chắc (cặp nhãn vs
   chữ lt2-thay-thế hệ thống) — cả hai đều là SUY ĐOÁN [SĐ].
3) Theo LỚP hệ thống (âm, chữ lt1, chữ lt2) ≥ 10 ô: f theo lớp.
Ra measure_out/_tn9/pair_arbiter.json
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402
from t05_inR_model import blk  # noqa: E402


def score_map(F: pd.DataFrame, col="f_vW"):
    return dict(zip(zip(F.i.values, F.c.values), F[col].values))


def main():
    C = T.lex()
    out = {"borg": {}, "stt": {}}
    for b in ("B18", "B34"):
        D = pd.read_pickle(T.T8 / "base" / f"{b}.pkl")
        F = pd.read_pickle(T.T8 / "cand" / f"{b}.pkl")
        S = score_map(F)
        R = {s: C.R_of(s) for s in set(D.syllable)}
        wins = defaultdict(list)
        for i, (k, g, s) in enumerate(zip(D.ocr_char.values, D.gt_char.values, D.syllable.values)):
            if not g or not k or k not in R[s] or C.var_eq_plus(k, g):
                continue
            a, c = S.get((i, g)), S.get((i, k))
            if a is None or c is None or np.isnan(a) or np.isnan(c):
                continue
            wins["all"].append(a > c); wins[blk(g)].append(a > c)
        out["borg"][b] = {k: [round(float(np.mean(v)), 4), len(v)] for k, v in wins.items()}
        print(f"[t07] Borg {b}: acc cặp (người > kim-sai-∈R) {out['borg'][b]}", flush=True)
    acc_b = float(np.mean([out["borg"][b]["all"][0] for b in out["borg"]]))
    for b in T.STT:
        D = T.load_stt(b)
        F = pd.read_pickle(T.T8 / "cand" / f"{b}.pkl")
        S = score_map(F)
        R = {s: C.R_of(s) for s in set(D.syllable)}
        lab, l1, l2, syl = D.label.values, D.ocr_char.values, D.lt2.values, D.syllable.values
        conf, cls = [], defaultdict(list)
        pairs = defaultdict(int)
        for i in range(len(D)):
            x, y, s = l1[i], l2[i], syl[i]
            if not x or not y or x not in R[s] or y not in R[s] or C.var_eq_plus(x, y):
                continue
            a, c = S.get((i, x)), S.get((i, y))
            if a is None or c is None or np.isnan(a) or np.isnan(c):
                continue
            conf.append(c > a)
            cls[(s, x, y)].append(c > a)
            pairs[(s, x, y)] += 1
        f = float(np.mean(conf)) if conf else np.nan
        # acc_STT: trên ô CHẮC (GOLD, lt1 == lt2 == nhãn ∈ R) mà nhãn là chữ lt1 của một lớp xung đột (s, x, y) -> f_vW(x) > f_vW(y)?
        alt = defaultdict(set)
        for (s, x, y) in pairs:
            alt[(s, x)].add(y); alt[(s, y)].add(x)
        sure_w = []
        for i in range(len(D)):
            if D.tier.values[i] != "GOLD" or not lab[i] or lab[i] != l1[i] or not C.var_eq_plus(lab[i], l2[i]):
                continue
            for y in alt.get((syl[i], lab[i]), ()):
                a, c = S.get((i, lab[i])), S.get((i, y))
                if a is None or c is None or np.isnan(a) or np.isnan(c):
                    continue
                sure_w.append(a > c)
        acc_s = float(np.mean(sure_w)) if sure_w else np.nan
        pi = {}
        for nm, acc in (("acc_borg", acc_b), ("acc_stt_sure", acc_s)):
            pi[nm] = dict(acc=round(acc, 4), pi_lt2=round(float((f - (1 - acc)) / (2 * acc - 1)), 4) if acc > 0.5 else None)
        top = sorted(cls.items(), key=lambda kv: -len(kv[1]))[:15]
        out["stt"][b] = dict(n_conflict=len(conf), f_lt2_beats_lt1=round(f, 4), n_sure_pairs=len(sure_w), pi=pi,
                             top_classes=[[f"{s} {x}/{y}", len(v), round(float(np.mean(v)), 3)] for (s, x, y), v in top])
        print(f"[t07] {b}: xung đột {len(conf)}, f(lt2 thắng) {f:.3f}; acc_STT(ô chắc, cặp hệ thống) {acc_s:.3f} (n {len(sure_w)}); "
              f"π_lt2 {pi}", flush=True)
        print(f"      lớp: {out['stt'][b]['top_classes'][:10]}", flush=True)
    T.jdump(out, T.OUT / "pair_arbiter.json")


if __name__ == "__main__":
    main()
