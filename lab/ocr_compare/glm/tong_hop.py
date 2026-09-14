"""Tổng hợp mọi cấu hình đã chạy thành bảng Markdown + scores.json.
Dùng: PYTHONPATH=<repo> <repo>/.venv/bin/python tong_hop.py  > ket_qua.md
"""
import glob, json, sys, collections
from pathlib import Path
import pandas as pd
sys.path.insert(0, '/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/glm')
from score_glm import score_set, score_m4, load_vocab, corpus_char_freq, cjk_chars, REPO, MAU

G = REPO / 'lab/ocr_compare/glm'


def md(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join('' if (v is None or (isinstance(v, float) and v != v)) else (f'{v:.3f}' if isinstance(v, float) else str(v)) for v in r) + ' |')
    return '\n'.join(out)
q2n, vocab = load_vocab(); freq = corpus_char_freq()
rows = []; m3 = {}; m4 = {}; col = {}
allout = {}
for path in sorted(glob.glob(str(G / 'out/*.csv'))):
    stem = Path(path).stem
    if stem.endswith('_scored'):
        continue
    res = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(res) == 0:
        continue
    name = stem.split('_')[0]; cfg = stem[len(name) + 1:]
    if name == 'M4':
        out, d = score_m4(res); m4[cfg] = out; allout[stem] = out; continue
    if cfg == 'col-ctx':
        # pred = chữ căn được cho ô; sec = giây/cột
        res = res.copy()
        out, d = score_set(name, res, vocab, freq)
        out['n_col_pred_over_kinh'] = round((pd.to_numeric(res.n_col_pred) / res.col_kinh.str.len()).mean(), 3)
        out['col_cer_vs_kinh'] = round((pd.to_numeric(res.col_lev) / res.col_kinh.str.len()).mean(), 3)
        out['align_op'] = res.align_op.value_counts(normalize=True).round(3).to_dict()
        col[name] = out; allout[stem] = out
        d.drop(columns=['R'], errors='ignore').to_csv(Path(path).with_name(stem + '_scored.csv'), index=False)
        continue
    out, d = score_set(name, res, vocab, freq); allout[stem] = out
    d.drop(columns=['R'], errors='ignore').to_csv(Path(path).with_name(stem + '_scored.csv'), index=False)
    rows.append(dict(set=name, cfg=cfg, n=out['n'], one_cjk=out['one_cjk'], in_R=out['first_in_R'], eq_kinh=out['eq_kinhhannom'],
                     empty=out['bia']['empty'], no_cjk=out['bia']['no_cjk'], latin=out['bia']['latin'], multi=out['bia']['multi_cjk'],
                     sec=out['sec_per_cell'], rescue=out.get('rescue_rate'), lift_freq=out.get('lift_freq'), lift_uni=out.get('lift_uniform')))
    if name == 'M3':
        m3[cfg] = out
json.dump(allout, open(G / 'scores.json', 'w'), ensure_ascii=False, indent=1)
t = pd.DataFrame(rows)
print('# Kết quả GLM-OCR theo giao thức chung\n')
print('Cấu hình: `<prompt>-<mode>-s<scale>`; prompt a = "Text Recognition:" (gốc SDK), b = nói rõ 1 chữ Hán/Nôm viết tay, c = ĐÓNG (cho danh sách R(âm)); mode nat = tự nhiên, force = cấm EOS/dấu ở token đầu; col-ctx = đọc nguyên cột rồi căn với kinhhannom.\n')
print('Chỉ số: one_cjk = đúng 1 ký tự CJK; in_R = ký tự CJK đầu ∈ R(âm) [CHỈ SỐ CHÍNH]; eq_kinh = trùng kinhhannom; empty/no_cjk/latin/multi = chi phí bịa/từ chối; sec = giây/ô.\n')
for s in ['M1', 'M2', 'M3']:
    tt = t[t.set == s]
    if s in col:
        c = col[s]; tt = pd.concat([tt, pd.DataFrame([dict(set=s, cfg='col-ctx (bổ sung)', n=c['n'], one_cjk=c['one_cjk'], in_R=c['first_in_R'], eq_kinh=c['eq_kinhhannom'],
                     empty=c['bia']['empty'], no_cjk=c['bia']['no_cjk'], latin=c['bia']['latin'], multi=c['bia']['multi_cjk'], sec=c['sec_per_cell'],
                     rescue=c.get('rescue_rate'), lift_freq=c.get('lift_freq'), lift_uni=c.get('lift_uniform'))])])
    print(f'\n## {s}\n'); print(md(tt.drop(columns=['set'])))
if m3:
    print('\n## M3 — phân bố ký tự đọc ra (top 12) theo cấu hình\n')
    for cfg, o in m3.items():
        print(f'- **{cfg}**: 𠊚 {o["pct_𠊚"]:.1%}, 㝵 {o["pct_㝵"]:.1%}, 𠊛 {o["pct_𠊛"]:.1%}, ∈R(người) {o["pct_in_R_nguoi"]:.1%}; top: ' + ', '.join(f'{c} {k}' for c, k, _ in o['dist_top']))
    if 'M3' in col:
        o = col['M3']; print(f'- **col-ctx**: 𠊚 {o["pct_𠊚"]:.1%}, 㝵 {o["pct_㝵"]:.1%}, 𠊛 {o["pct_𠊛"]:.1%}, ∈R(người) {o["pct_in_R_nguoi"]:.1%}; top: ' + ', '.join(f'{c} {k}' for c, k, _ in o['dist_top']))
if m4:
    print('\n## M4 — 20 cột nguyên vẹn\n')
    for cfg, o in m4.items():
        print(f'- **{cfg}**: n_kinh TB {o["n_kinh_mean"]}, n_GLM TB {o["n_pred_mean"]}, số cột đếm đúng {o["n_pred_eq_kinh"]}/20, ±2 chữ {o["n_pred_within_2"]}/20, tỷ số n_GLM/n_kinh {o["ratio_pred_over_kinh"]}, CER so với kinhhannom {o["cer_vs_kinhhannom_mean"]}, {o["sec_per_col"]} s/cột')
        for r in o['rows'][:20]:
            print(f'    - {r["id"]}: kinh {r["n_kinh"]} / GLM {r["n_pred"]} lev={r["lev"]} `{r["pred"]}`')
if col:
    print('\n## col-ctx — chi tiết căn cột\n')
    for s, o in col.items():
        print(f'- {s}: n={o["n"]}, tỷ số n_GLM/n_kinh {o["n_col_pred_over_kinh"]}, CER cột so kinhhannom {o["col_cer_vs_kinh"]}, phép căn {o["align_op"]}, in_R {o["first_in_R"]}, eq_kinh {o["eq_kinhhannom"]}' + (f', rescue {o["rescue_rate"]} (lift_freq {o["lift_freq"]})' if s == 'M2' else ''))

# ---------- Phân tích bổ sung ----------
print('\n## Bổ sung\n')
ctrl = sorted(glob.glob(str(G / 'out/*CTRL*.csv')))
ctrl = [c for c in ctrl if not c.endswith('_scored.csv')]
if ctrl:
    for path in ctrl:
        res = pd.read_csv(path, dtype=str, keep_default_na=False)
        cnt = collections.Counter(cjk_chars(p)[:1] or '<rỗng>' for p in res.pred)
        n = len(res)
        print(f'- ĐỐI CHỨNG {Path(path).stem}: n={n}; phân bố chữ đọc ra khi cho danh sách R(người) nhưng ảnh KHÔNG phải "người": ' + ', '.join(f'{c} {k} ({k/n:.0%})' for c, k in cnt.most_common(8)))
p = G / 'out/M2_col-ctx_scored.csv'
if p.exists():
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    raw = pd.read_csv(G / 'out/M2_col-ctx.csv', dtype=str, keep_default_na=False)
    d = d.merge(raw[['sample_id', 'align_op']], on='sample_id')
    r = d[d.first_in_R == 'True']; lab = r[r.label != '']
    print(f'- col-ctx M2 cứu {len(r)}/{len(d)}: SILVER {int((r.tier=="SILVER").sum())}/{int((d.tier=="SILVER").sum())}, SYLLABLE {int((r.tier=="SYLLABLE").sum())}/{int((d.tier=="SYLLABLE").sum())}; '
          f'trong {len(lab)} ô có nhãn pipeline, {int((lab["first"]==lab.label).sum())} trùng nhãn; theo sách ' + str(r.book.value_counts().to_dict()))
    print('  - ví dụ (kinhhannom → GLM / âm): ' + ', '.join(f'{a}→{b}/{s}' for a, b, s in zip(r.ocr_char.head(16), r['first'].head(16), r.syllable.head(16))))
