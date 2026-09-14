"""Chấm kết quả OCR theo giao thức chung (M1-M3) — độc lập engine.

Đầu vào: CSV kết quả có cột sample_id, pred (chuỗi engine đọc ra, đã strip), sec (giây/ô).
Trọng tài: R(âm) = cột R_candidates của mau/M*.csv (từ Dict/QuocNgu_SinoNom.csv).
Đối chứng ngẫu nhiên: P(chữ ngẫu nhiên ∈ R(âm)) = |R(âm)| / |từ vựng Nôm| (đều), và
phiên bản trọng số theo tần suất ocr_char trong labels.csv (chữ hay gặp).
"""
import json, sys, unicodedata, collections
from pathlib import Path
import pandas as pd

REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
MAU = REPO / "lab/ocr_compare/mau"


def is_cjk(c: str) -> bool:
    o = ord(c)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2A6DF
            or 0x2A700 <= o <= 0x2EBEF or 0x30000 <= o <= 0x323AF or 0xF900 <= o <= 0xFAFF
            or 0x2F00 <= o <= 0x2FDF or 0x2E80 <= o <= 0x2EFF)


def cjk_chars(s: str) -> str:
    return "".join(c for c in s if is_cjk(c))


def load_vocab():
    sys.path.insert(0, str(REPO))
    from core.text.dictionary import load_qn_to_nom
    q2n = load_qn_to_nom(str(REPO / "Dict/QuocNgu_SinoNom.csv"))
    vocab = {c for v in q2n.values() for c in v}
    return q2n, vocab


def corpus_char_freq():
    df = pd.read_csv(REPO / "dataset_out/labels.csv", dtype=str, keep_default_na=False)
    return collections.Counter(df.ocr_char)


def score_set(name: str, res: pd.DataFrame, vocab: set, freq: collections.Counter) -> dict:
    m = pd.read_csv(MAU / f"{name}.csv", dtype=str, keep_default_na=False)
    d = m.merge(res[["sample_id", "pred", "sec"]], on="sample_id", how="left")
    d["pred"] = d.pred.fillna("")
    d["R"] = d.R_candidates.map(lambda s: set(x for x in s.split("|") if x))
    d["pred_cjk"] = d.pred.map(cjk_chars)
    d["n_cjk"] = d.pred_cjk.map(len)
    d["one_cjk"] = d.n_cjk == 1
    d["first"] = d.pred_cjk.map(lambda s: s[:1])
    # chữ đọc ra ∈ R: xét ký tự CJK đầu tiên (đơn ký tự); phiên bản "bất kỳ" xét mọi ký tự đọc ra
    d["first_in_R"] = [f != "" and f in r for f, r in zip(d["first"], d.R)]
    d["any_in_R"] = [any(c in r for c in p) for p, r in zip(d.pred_cjk, d.R)]
    d["eq_kinh"] = d["first"] == d.ocr_char
    d["first_in_vocab"] = [f != "" and f in vocab for f in d["first"]]
    d["kinh_in_R"] = d.ocr_in_R == "True"
    n = len(d)
    tot_freq = sum(freq.values())
    p_rand_uniform = float(sum(len(r) / len(vocab) for r in d.R) / n)
    p_rand_freq = float(sum(sum(freq[c] for c in r) / tot_freq for r in d.R) / n)
    out = dict(
        set=name, n=n, n_pred_nonempty=int((d.pred != "").sum()),
        one_cjk=float(d.one_cjk.mean()), n_cjk_mean=float(d.n_cjk.mean()),
        first_in_R=float(d.first_in_R.mean()), any_in_R=float(d.any_in_R.mean()),
        eq_kinhhannom=float(d.eq_kinh.mean()),
        first_in_vocab_given_pred=float(d.first_in_vocab[d["first"] != ""].mean()) if (d["first"] != "").any() else None,
        n_first_pred=int((d["first"] != "").sum()),
        sec_per_cell=float(pd.to_numeric(d.sec, errors="coerce").mean()),
        p_random_in_R_uniform=p_rand_uniform, p_random_in_R_freqweighted=p_rand_freq,
    )
    if name == "M2":
        rescued = d[~d.kinh_in_R]
        out["rescue_rate"] = float(rescued.first_in_R.mean())
        out["rescue_n"] = int(len(rescued))
        out["rescue_count"] = int(rescued.first_in_R.sum())
        out["lift_uniform"] = out["rescue_rate"] / p_rand_uniform if p_rand_uniform else None
        out["lift_freq"] = out["rescue_rate"] / p_rand_freq if p_rand_freq else None
        out["rescue_eq_label"] = float((rescued["first"] == rescued.label)[rescued.first_in_R].mean()) if rescued.first_in_R.any() else None
        out["by_tier"] = {t: dict(n=int(len(g)), first_in_R=float(g.first_in_R.mean()), one_cjk=float(g.one_cjk.mean()))
                          for t, g in rescued.groupby("tier")}
    if name == "M3":
        cnt = collections.Counter(d["first"])
        out["dist_top"] = [(c or "<rỗng>", k, round(k / n, 4)) for c, k in cnt.most_common(15)]
        out["pct_𠊚"] = cnt["𠊚"] / n
        out["pct_㝵"] = cnt["㝵"] / n
        out["pct_in_R_nguoi"] = float(d.first_in_R.mean())
    out["by_book"] = {b: dict(n=int(len(g)), first_in_R=float(g.first_in_R.mean()), one_cjk=float(g.one_cjk.mean()))
                      for b, g in d.groupby("book")}
    return out, d


if __name__ == "__main__":
    q2n, vocab = load_vocab()
    freq = corpus_char_freq()
    for path in sys.argv[1:]:
        res = pd.read_csv(path, dtype=str, keep_default_na=False)
        name = Path(path).stem.split("_")[0]
        out, d = score_set(name, res, vocab, freq)
        print(json.dumps(out, ensure_ascii=False, indent=1))
