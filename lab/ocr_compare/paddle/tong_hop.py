"""Tổng hợp mọi kết quả trong ket_qua/ thành bảng + JSON (chạy bằng .venv của kho)."""
import sys, json, glob, collections
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/paddle")
from score import load_vocab, corpus_char_freq, score_set, cjk_chars

K = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/paddle/ket_qua")
q2n, vocab = load_vocab(); freq = corpus_char_freq()
rows, full = [], {}
for p in sorted(K.glob("M[123]_*.csv")):
    name = p.stem.split("_")[0]; cfg = p.stem[len(name) + 1:]
    res = pd.read_csv(p, dtype=str, keep_default_na=False)
    variants = [("", res)]
    if "pred_seq" in res:  # chế độ cột: thêm biến thể gióng theo chuỗi
        r2 = res.copy(); r2["pred"] = r2.pred_seq; variants.append(("+seq", r2))
    for suf, rr in variants:
        out, d = score_set(name, rr, vocab, freq)
        full[f"{p.stem}{suf}"] = out
        row = dict(set=name, cfg=cfg + suf, n=out["n"], pred_nonempty=out["n_pred_nonempty"], one_cjk=round(out["one_cjk"], 3),
                   in_R=round(out["first_in_R"], 3), eq_kinh=round(out["eq_kinhhannom"], 3), sec=round(out["sec_per_cell"], 3))
        if name == "M2":
            row.update(rescue=f'{out["rescue_count"]}/{out["rescue_n"]}', lift_freq=round(out["lift_freq"], 1), lift_unif=round(out["lift_uniform"], 0),
                       rescue_eq_label=round(out["rescue_eq_label"], 2) if out["rescue_eq_label"] is not None else None,
                       silver_in_R=round(out["by_tier"]["SILVER"]["first_in_R"], 3), syl_in_R=round(out["by_tier"]["SYLLABLE"]["first_in_R"], 3))
        if name == "M3":
            row.update(pct_nguoi_dung=out["pct_𠊚"], pct_nguoi_dac=out["pct_㝵"], top=" ".join(f"{c}:{k}" for c, k, _ in out["dist_top"][:6]))
        rows.append(row)
t = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for s in ("M1", "M2", "M3"):
    print(f"\n=== {s} ===")
    print(t[t.set == s].drop(columns=["set"]).to_string(index=False))
json.dump(full, open(K.parent / "tong_hop.json", "w"), ensure_ascii=False, indent=1)
t.to_csv(K.parent / "tong_hop.csv", index=False)
