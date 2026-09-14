"""Gom số liệu đầu mối vào tom_tat.json (chạy bằng .venv của kho, sau tong_hop.py / phan_tich_them.py / score_m4.py)."""
import json, glob, sys
import pandas as pd
sys.path.insert(0, "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/paddle")
from score import load_vocab, corpus_char_freq, score_set
from score_m4 import summarize
P = "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/paddle"
q2n, vocab = load_vocab(); freq = corpus_char_freq()
th = json.load(open(f"{P}/tong_hop.json"))
pt = json.load(open(f"{P}/phan_tich_them.json"))
cs = json.load(open(f"{P}/charset_coverage.json"))
m4 = {v: summarize(f"{P}/ket_qua/M4_{v}.csv") for v in ("PP-OCRv6", "PP-OCRv5")}
vl = {}
for s in ("M1", "M2", "M3"):
    f = f"{P}/ket_qua/{s}_PaddleOCR-VL-1.6.csv"
    try:
        res = pd.read_csv(f, dtype=str, keep_default_na=False)
        out, d = score_set(s, res, vocab, freq)
        d = d[d.sample_id.isin(res.sample_id)]
        vl[s] = dict(n_done=len(res), in_R=f"{int(d.first_in_R.sum())}/{len(d)}", any_in_R=int(d.any_in_R.sum()), one_cjk=round(float(d.one_cjk.mean()), 3),
                     eq_kinh=int(d.eq_kinh.sum()), sec=round(float(pd.to_numeric(res.sec, errors="coerce").mean()), 2))
        if s == "M3":
            import collections; vl[s]["top"] = collections.Counter(d["first"]).most_common(8)
    except FileNotFoundError:
        pass
tt = dict(
    moi_truong=dict(thiet_bi="Apple M4 16GB, CPU (không MPS)", python="3.10.20 (Homebrew) trong /tmp/venv_paddle",
                    paddlepaddle="3.3.1", paddleocr="3.7.0", paddlex="3.7.2",
                    models=["PP-OCRv6_medium_rec (76MB)", "PP-OCRv6_medium_det", "PP-OCRv5_server_rec (84MB)", "PP-OCRv5_server_det", "PaddleOCR-VL-1.6-0.9B (1.8GB)"]),
    charset=cs["PP-OCRv6_medium_rec"],
    M1={k: dict(in_R=v["first_in_R"], eq_kinh=v["eq_kinhhannom"], one_cjk=v["one_cjk"], sec=v["sec_per_cell"]) for k, v in th.items() if k.startswith("M1")},
    M2={k: dict(rescue=f'{v["rescue_count"]}/{v["rescue_n"]}', rate=v["rescue_rate"], lift_freq=v["lift_freq"], lift_uniform=v["lift_uniform"],
               SILVER=v["by_tier"]["SILVER"]["first_in_R"], SYLLABLE=v["by_tier"]["SYLLABLE"]["first_in_R"], sec=v["sec_per_cell"]) for k, v in th.items() if k.startswith("M2")},
    M3={k: dict(pct_𠊚=v["pct_𠊚"], pct_㝵=v["pct_㝵"], in_R_nguoi=v["pct_in_R_nguoi"], top=v["dist_top"][:8]) for k, v in th.items() if k.startswith("M3")},
    M4=m4, phan_tich_them=pt, PaddleOCR_VL=vl)
json.dump(tt, open(f"{P}/tom_tat.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(tt["PaddleOCR_VL"], ensure_ascii=False))
