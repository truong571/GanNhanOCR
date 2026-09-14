"""Chấm kết quả GLM-OCR theo giao thức chung. Logic chỉ số M1-M3 giống lab/ocr_compare/paddle/score.py
(sao chép để kết quả GLM không phụ thuộc chỉnh sửa về sau của tệp đó), thêm:
  - chi phí "bịa" của VLM: >1 ký tự CJK, có chữ Latin, rỗng/từ chối, chữ ngoài từ vựng Nôm
  - M4: so số ký tự đọc ra với kinhhannom, tỷ lệ trùng theo Levenshtein
Dùng: PYTHONPATH=<repo> <repo>/.venv/bin/python score_glm.py out/M1_a.csv out/M2_a.csv ...
"""
import json, sys, re, collections, unicodedata
from pathlib import Path
import pandas as pd

REPO = Path('/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR')
MAU = REPO / 'lab/ocr_compare/mau'


def is_cjk(c):
    o = ord(c)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2A6DF
            or 0x2A700 <= o <= 0x2EBEF or 0x30000 <= o <= 0x323AF or 0xF900 <= o <= 0xFAFF
            or 0x2F00 <= o <= 0x2FDF or 0x2E80 <= o <= 0x2EFF or 0xF0000 <= o <= 0x10FFFD)


def cjk_chars(s):
    return ''.join(c for c in s if is_cjk(c))


def has_latin(s):
    return bool(re.search(r'[A-Za-z]', s))


def load_vocab():
    sys.path.insert(0, str(REPO))
    from core.text.dictionary import load_qn_to_nom
    q2n = load_qn_to_nom(str(REPO / 'Dict/QuocNgu_SinoNom.csv'))
    return q2n, {c for v in q2n.values() for c in v}


def corpus_char_freq():
    df = pd.read_csv(REPO / 'dataset_out/labels.csv', dtype=str, keep_default_na=False)
    return collections.Counter(df.ocr_char)


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_set(name, res, vocab, freq):
    m = pd.read_csv(MAU / f'{name}.csv', dtype=str, keep_default_na=False)
    d = m.merge(res[['sample_id', 'pred', 'sec']], on='sample_id', how='inner')
    d['R'] = d.R_candidates.map(lambda s: set(x for x in s.split('|') if x))
    d['pred_cjk'] = d.pred.map(cjk_chars)
    d['n_cjk'] = d.pred_cjk.map(len)
    d['one_cjk'] = d.n_cjk == 1
    d['first'] = d.pred_cjk.map(lambda s: s[:1])
    d['first_in_R'] = [f != '' and f in r for f, r in zip(d['first'], d.R)]
    d['any_in_R'] = [any(c in r for c in p) for p, r in zip(d.pred_cjk, d.R)]
    d['eq_kinh'] = d['first'] == d.ocr_char
    d['first_in_vocab'] = [f != '' and f in vocab for f in d['first']]
    d['kinh_in_R'] = d.ocr_in_R == 'True'
    d['is_empty'] = d.pred.str.strip() == ''
    d['latin'] = d.pred.map(has_latin)
    d['multi'] = d.n_cjk > 1
    d['no_cjk'] = d.n_cjk == 0
    n = len(d)
    tot = sum(freq.values())
    p_u = float(sum(len(r) / len(vocab) for r in d.R) / n)
    p_f = float(sum(sum(freq[c] for c in r) / tot for r in d.R) / n)
    out = dict(set=name, n=n,
               one_cjk=round(d.one_cjk.mean(), 4), n_cjk_mean=round(d.n_cjk.mean(), 2),
               first_in_R=round(d.first_in_R.mean(), 4), any_in_R=round(d.any_in_R.mean(), 4),
               eq_kinhhannom=round(d.eq_kinh.mean(), 4),
               first_in_vocab_given_pred=round(d.first_in_vocab[d['first'] != ''].mean(), 4) if (d['first'] != '').any() else None,
               bia=dict(empty=round(d.is_empty.mean(), 4), no_cjk=round(d.no_cjk.mean(), 4), latin=round(d.latin.mean(), 4),
                        multi_cjk=round(d.multi.mean(), 4), first_not_in_vocab=round((~d.first_in_vocab & (d['first'] != '')).mean(), 4)),
               sec_per_cell=round(pd.to_numeric(d.sec, errors='coerce').mean(), 3),
               p_random_in_R_uniform=round(p_u, 4), p_random_in_R_freqweighted=round(p_f, 4))
    if name == 'M2':
        r_ = d[~d.kinh_in_R]
        out['rescue_n'] = int(len(r_)); out['rescue_count'] = int(r_.first_in_R.sum())
        out['rescue_rate'] = round(r_.first_in_R.mean(), 4)
        out['rescue_any_count'] = int(r_.any_in_R.sum())
        out['lift_uniform'] = round(out['rescue_rate'] / p_u, 2) if p_u else None
        out['lift_freq'] = round(out['rescue_rate'] / p_f, 2) if p_f else None
        # trong số cứu được: trùng nhãn pipeline (label) — chỉ tham khảo, label không phải chân lý
        out['rescue_eq_label'] = round(float((r_['first'] == r_.label)[r_.first_in_R].mean()), 4) if r_.first_in_R.any() else None
        out['by_tier'] = {t: dict(n=int(len(g)), first_in_R=round(g.first_in_R.mean(), 4), one_cjk=round(g.one_cjk.mean(), 4))
                          for t, g in r_.groupby('tier')}
    if name == 'M3':
        cnt = collections.Counter(d['first'])
        out['dist_top'] = [(c or '<rỗng>', k, round(k / n, 4)) for c, k in cnt.most_common(12)]
        out['pct_𠊚'] = round(cnt['𠊚'] / n, 4); out['pct_㝵'] = round(cnt['㝵'] / n, 4)
        out['pct_𠊛'] = round(cnt['𠊛'] / n, 4)
        out['pct_in_R_nguoi'] = round(d.first_in_R.mean(), 4)
    out['by_book'] = {b: dict(n=int(len(g)), first_in_R=round(g.first_in_R.mean(), 4), one_cjk=round(g.one_cjk.mean(), 4))
                      for b, g in d.groupby('book')}
    return out, d


def score_m4(res):
    m = pd.read_csv(MAU / 'M4.csv', dtype=str, keep_default_na=False)
    d = m.merge(res[['sample_id', 'pred', 'sec']], on='sample_id', how='inner')
    d['pred_cjk'] = d.pred.map(cjk_chars)
    d['n_pred'] = d.pred_cjk.map(len)
    d['n_kinh'] = pd.to_numeric(d.kinhhannom_n)
    d['lev'] = [lev(p, k) for p, k in zip(d.pred_cjk, d.kinhhannom_text)]
    d['cer_vs_kinh'] = d.lev / d.n_kinh
    d['pos_match'] = [sum(a == b for a, b in zip(p, k)) / len(k) for p, k in zip(d.pred_cjk, d.kinhhannom_text)]
    out = dict(set='M4', n=len(d), n_kinh_mean=round(d.n_kinh.mean(), 2), n_pred_mean=round(d.n_pred.mean(), 2),
               n_pred_eq_kinh=int((d.n_pred == d.n_kinh).sum()),
               n_pred_within_2=int(((d.n_pred - d.n_kinh).abs() <= 2).sum()),
               ratio_pred_over_kinh=round((d.n_pred / d.n_kinh).mean(), 3),
               cer_vs_kinhhannom_mean=round(d.cer_vs_kinh.mean(), 3),
               chars_equal_1_minus_cer=round(1 - d.cer_vs_kinh.mean(), 3),
               sec_per_col=round(pd.to_numeric(d.sec).mean(), 2),
               rows=[dict(id=r.sample_id, n_kinh=int(r.n_kinh), n_pred=int(r.n_pred), lev=int(r.lev), pred=r.pred_cjk[:40]) for r in d.itertuples()])
    return out, d


if __name__ == '__main__':
    q2n, vocab = load_vocab(); freq = corpus_char_freq()
    allout = {}
    for path in sys.argv[1:]:
        res = pd.read_csv(path, dtype=str, keep_default_na=False)
        stem = Path(path).stem; name = stem.split('_')[0]
        if name == 'M4':
            out, d = score_m4(res)
        else:
            out, d = score_set(name, res, vocab, freq)
        allout[stem] = out
        print('=====', stem); print(json.dumps(out, ensure_ascii=False, indent=1))
        d.drop(columns=['R'], errors='ignore').to_csv(Path(path).with_name(stem + '_scored.csv'), index=False)
    json.dump(allout, open(REPO / 'lab/ocr_compare/glm/scores.json', 'w'), ensure_ascii=False, indent=1)
