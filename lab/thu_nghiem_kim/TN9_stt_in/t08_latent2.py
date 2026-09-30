"""t08_latent2.py — mô hình lớp ẩn HAI lần đọc kim (lt1, lt2) + âm QN để SUY ĐOÁN độ đúng nhãn STT theo loại ô. 0 API. [SĐ]

Mỗi ô có cả hai chữ: quan sát (E = lt1 ≡ lt2 [V1+], X1 = lt1 ∈ R(âm), X2 = lt2 ∈ R(âm)) -> 6 loại.
Ẩn: (c1, c2) = (lt1 đúng, lt2 đúng); π11, π10, π01, π00. Tham số: s = P(chữ đúng ∈ R) (Borg: 0,984–0,996),
q1, q2 = P(∈ R | sai), e = P(hai lần đọc sai TRÙNG nhau | cả hai sai), q12 = P(∈ R | sai trùng).
  P(E=1,X=1) = π11 s + π00 e q12            P(E=1,X=0) = π11 (1−s) + π00 e (1−q12)
  P(E=0,x1,x2) = π10 [s|1−s]_{x1} [q2|1−q2]_{x2} + π01 [q1|1−q1]_{x1} [s|1−s]_{x2} + π00 (1−e) [q1]_{x1} [q2]_{x2}
6 loại = 5 bậc tự do; ẩn π11, π10, π01, e, q1 (5) với RÀNG BUỘC ĐẶT TRƯỚC: q2 = q của Borg (kim lt2 cùng chế độ Nôm, dải
0,090–0,145), q12 ∈ {q1, 2·q1 (lỗi hệ thống hay rơi vào R hơn)}, s ∈ {0,984; 0,996}. Giải bình phương tối thiểu có biên.
Độ đúng theo loại (vd "cả hai ∈ R và trùng" = π11 s / P(E=1,X=1)); báo dải theo mọi tổ hợp ràng buộc.
Ra measure_out/_tn9/latent2.json
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402


def probs(th, s, q2, q12mult):
    p11, p10, p01, e, q1 = th
    p00 = max(0.0, 1 - p11 - p10 - p01)
    q12 = min(1.0, q1 * q12mult)
    P = {}
    P["E1X1"] = p11 * s + p00 * e * q12
    P["E1X0"] = p11 * (1 - s) + p00 * e * (1 - q12)
    for x1, x2 in itertools.product((1, 0), (1, 0)):
        a = (s if x1 else 1 - s) * (q2 if x2 else 1 - q2)
        b = (q1 if x1 else 1 - q1) * (s if x2 else 1 - s)
        c = (q1 if x1 else 1 - q1) * (q2 if x2 else 1 - q2)
        P[f"E0{x1}{x2}"] = p10 * a + p01 * b + p00 * (1 - e) * c
    return P, p00, q12


KEYS = ["E1X1", "E1X0", "E011", "E010", "E001", "E000"]


def fit(obs, s, q2, q12mult):
    f = np.array([obs[k] for k in KEYS], float)
    f = f / f.sum()

    def res(th):
        P, p00, _ = probs(th, s, q2, q12mult)
        pen = 10 * min(0.0, 1 - th[0] - th[1] - th[2])
        return np.array([P[k] for k in KEYS]) - f + pen

    best = None
    for x0 in ([0.6, 0.1, 0.15, 0.3, 0.15], [0.5, 0.05, 0.2, 0.5, 0.2], [0.55, 0.15, 0.1, 0.2, 0.1]):
        r = least_squares(res, x0, bounds=([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]))
        if best is None or r.cost < best.cost:
            best = r
    th = best.x
    P, p00, q12 = probs(th, s, q2, q12mult)
    p11, p10, p01, e, q1 = th
    prec = {
        "agree_inR (lt1≡lt2 ∈ R)": p11 * s / max(P["E1X1"], 1e-12),
        "conflict: lt1 đúng": p10 * s * q2 / max(P["E011"], 1e-12),
        "conflict: lt2 đúng": p01 * q1 * s / max(P["E011"], 1e-12),
        "lt1∈R, lt2∉R: lt1 đúng": p10 * s * (1 - q2) / max(P["E010"], 1e-12),
        "lt2∈R, lt1∉R: lt2 đúng": p01 * (1 - q1) * s / max(P["E001"], 1e-12),
    }
    return dict(theta=dict(p11=round(p11, 4), p10=round(p10, 4), p01=round(p01, 4), p00=round(p00, 4), e=round(e, 4),
                           q1=round(q1, 4), q12=round(q12, 4)),
                rmse=round(float(np.sqrt(np.mean(best.fun ** 2))), 5),
                acc_lt1=round(p11 + p10, 4), acc_lt2=round(p11 + p01, 4),
                prec={k: round(float(v), 4) for k, v in prec.items()})


def main():
    C = T.lex()
    out = {}
    for b in T.STT:
        D = T.load_stt(b)
        R = {s: C.R_of(s) for s in set(D.syllable)}
        obs = dict.fromkeys(KEYS, 0)
        for x, y, s in zip(D.ocr_char.values, D.lt2.values, D.syllable.values):
            if not x or not y:
                continue
            E = C.var_eq_plus(x, y)
            x1, x2 = x in R[s], y in R[s]
            if E:
                obs["E1X1" if (x1 or x2) else "E1X0"] += 1
            else:
                obs[f"E0{int(x1)}{int(x2)}"] += 1
        res = dict(obs=obs, fits=[])
        for s, q2, m in itertools.product((0.984, 0.996), (0.090, 0.117, 0.145), (1.0, 2.0)):
            r = fit(obs, s, q2, m)
            r.update(s=s, q2=q2, q12mult=m)
            res["fits"].append(r)
        keys = list(res["fits"][0]["prec"])
        res["range"] = {k: [round(min(f["prec"][k] for f in res["fits"]), 4), round(max(f["prec"][k] for f in res["fits"]), 4)]
                        for k in keys}
        res["acc_range"] = {"lt1": [min(f["acc_lt1"] for f in res["fits"]), max(f["acc_lt1"] for f in res["fits"])],
                            "lt2": [min(f["acc_lt2"] for f in res["fits"]), max(f["acc_lt2"] for f in res["fits"])]}
        res["rmse_max"] = max(f["rmse"] for f in res["fits"])
        out[b] = res
        print(f"[t08] {b}: obs {obs}", flush=True)
        print(f"      dải độ đúng theo loại: {res['range']} | acc lt1 {res['acc_range']['lt1']} lt2 {res['acc_range']['lt2']} "
              f"| rmse ≤ {res['rmse_max']}", flush=True)
    T.jdump(out, T.OUT / "latent2.json")


if __name__ == "__main__":
    main()
