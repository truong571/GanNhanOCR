"""v_pb_truy_nguon_02 (04/10) — PHẢN BIỆN: tự tính lại ĐỘ CHÍNH XÁC Ô NEO (độc lập với v_truy_nguon_02), nhiều định nghĩa.
Ô neo = cells.anchor (t05_feats.py:74-76). Sự thật = D.gt_char (nhãn người). Đúng = nhãn==gt hoặc var_eq_plus (dị thể).
0 API, CPU. Ra: measure_out/_tn11/verify/pb_truy_nguon/o_neo.json
"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T
OUT = REPO / "measure_out/_tn11/verify/pb_truy_nguon"; OUT.mkdir(parents=True, exist_ok=True)
C = T.lex(); veq = C.var_eq_plus
res = {}
for b in ("B18", "B34", "L16", "TK"):
    D = T.load_base(b); cells = pd.read_pickle(T.OUT / "cand" / f"{b}_cells.pkl")
    assert len(D) == len(cells)
    anc = cells.anchor.values.astype(bool)
    gt = D.gt_char.values; lab = D.label.values; pg = D.page.values
    has = (gt != "")
    exact = np.array([bool(g) and l == g for l, g in zip(lab, gt)])
    var = np.array([bool(g) and (l == g or veq(l, g)) for l, g in zip(lab, gt)])
    pos_known = D.pos_known.values.astype(bool); pos_ok = D.pos_ok.values.astype(bool)
    tier = D.tier.values
    ink = cells.ink.values; crop_ok = cells.crop_ok.values
    def row(mask, ok, name):
        m = mask & has
        n = int(m.sum())
        if n == 0: return dict(ten=name, n=0)
        p, lo, hi = T.boot_ci_pages(ok[m].astype(float), pg[m])
        return dict(ten=name, n=n, pct=round(100*p, 2), ci=[round(100*lo, 2), round(100*hi, 2)])
    R = []
    R.append(row(anc, var, "neo tap TN8 (crop_ok>0), di the, moi o co gt"))
    R.append(row(anc, exact, "neo tap TN8, GIONG HET (khong di the)"))
    R.append(row(anc & pos_known, var, "neo & pos_known, di the"))
    R.append(row(anc & pos_ok, var, "neo & pos_ok (vi tri khop), di the"))
    R.append(row(anc & (tier == "GOLD"), var, "neo & tier GOLD, di the"))
    R.append(row(anc & (tier != "GOLD"), var, "neo & tier != GOLD, di the"))
    # theo luat 'khoi muc>0' = ink>0 thay cho crop_ok
    anc_ink = (D.rule.str.startswith("s1_inter_s2_direct").values & (lab != "") & (lab == D.ocr_char.values))
    Rm = {s: C.R_of(s) for s in set(D.syllable)}
    anc_ink &= np.array([l in Rm[s] for l, s in zip(lab, D.syllable.values)])
    R.append(row(anc_ink & (ink > 0), var, "luat neo + ink>0 (thay crop_ok), di the"))
    R.append(row(anc_ink, var, "luat neo KHONG loc crop/ink, di the"))
    # loai PUA (chu Nom private-use) khoi sach
    pua = np.array([any(0xE000 <= ord(ch) <= 0xF8FF or 0xF0000 <= ord(ch) for ch in l) for l in lab])
    R.append(row(anc & ~pua, var, "neo bo PUA, di the"))
    R.append(row(anc & ~pua & pos_ok, var, "neo bo PUA & pos_ok, di the"))
    # sai gt: so luong o neo co gt != '' / tong
    res[b] = dict(so_o=len(D), so_neo=int(anc.sum()), so_neo_co_gt=int((anc & has).sum()), neo_theo_tier={t: int(((tier == t) & anc).sum()) for t in sorted(set(tier))},
                  so_neo_ink0=int((anc & (ink <= 0)).sum()), so_neo_crop_bad=int((anc & (crop_ok <= 0)).sum()), ket_qua=R)
    print(b, json.dumps(res[b]["ket_qua"], ensure_ascii=False)[:1400], flush=True)
(OUT / "o_neo.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
