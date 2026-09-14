"""Gióng đầu ra GLM-OCR CẢ TRANG về từng ô kinhhannom rồi chấm M1/M2/M3 theo giao thức chung.
GLM không trả toạ độ → gióng theo THỨ TỰ:
  (1) lọc dòng: bỏ dòng không có CJK (số trang "252", dãy số cột "9 8 7 …", markdown, Latin) — đếm riêng làm chi phí bịa;
  (2) dòng thứ k ↔ cột k kinhhannom (đọc phải→trái, cả hai cùng thứ tự). Nếu số dòng ≠ số cột: DP thứ tự cho phép bỏ
      dòng/cột, chi phí = Levenshtein chuẩn hoá (dùng nội dung kinhhannom làm MỐC VỊ TRÍ, không dùng để chấm);
  (3) trong cột, 2 cách gióng ký tự: 'lev' = căn Levenshtein với chuỗi kinhhannom (như col-ctx lượt trước; thiên vị về phía
      trùng kinhhannom) và 'prop' = chia đều theo thứ tự (mù nội dung: pred[round(pos*(n_pred)/n_kinh)]), báo cả hai.
Dùng: PYTHONPATH=<repo> <repo>/.venv/bin/python align_score.py --mode full
Đầu ra: out/<mode>_pages.csv (thống kê trang), out/<mode>_M{1,2,3}_aligned.csv, ket_qua_<mode>.json
"""
import argparse, json, re, sys, collections, random
from pathlib import Path
import pandas as pd
REPO = Path('/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR'); HERE = REPO / 'lab/ocr_compare/glm_trang'
MAU = REPO / 'lab/ocr_compare/mau'
BOOK_DIR = {'stt2': 'SachThanhTruyen2', 'stt4': 'SachThanhTruyen4', 'stt11': 'SachThanhTruyen11'}

def is_cjk(c):
    o = ord(c)
    return (0x3400 <= o <= 0x9FFF or 0x20000 <= o <= 0x323AF or 0xF900 <= o <= 0xFAFF or 0x2E80 <= o <= 0x2FDF or 0xF0000 <= o <= 0x10FFFD)
def cjk(s): return ''.join(c for c in s if is_cjk(c))

def lev_align(pred, kinh):
    n, m = len(kinh), len(pred)
    D = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): D[i][0] = i
    for j in range(m+1): D[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+(kinh[i-1] != pred[j-1]))
    i, j = n, m; out = {}
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i-1][j-1]+(kinh[i-1] != pred[j-1]):
            out[i-1] = pred[j-1]; i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i-1][j]+1: out[i-1] = ''; i -= 1
        else: j -= 1
    return out, D[n][m]

def nlev(a, b):
    if not a and not b: return 0.0
    return lev_align(a, b)[1] / max(len(a), len(b))

def match_lines_cols(lines, cols):
    """DP thứ tự: gán dòng→cột, cho phép bỏ. Trả về dict col_idx -> line_idx."""
    L, C = len(lines), len(cols)
    if L == C: return {i: i for i in range(C)}
    INF = 1e9; D = [[INF]*(C+1) for _ in range(L+1)]; D[0][0] = 0; bt = {}
    skip = 0.9  # chi phí bỏ 1 dòng/cột (< 1 = chi phí thay thế tối đa)
    for i in range(L+1):
        for j in range(C+1):
            if i == 0 and j == 0: continue
            best = (INF, None)
            if i > 0 and j > 0:
                c = D[i-1][j-1] + nlev(lines[i-1], cols[j-1]);
                if c < best[0]: best = (c, 'm')
            if i > 0 and D[i-1][j] + skip < best[0]: best = (D[i-1][j] + skip, 'l')
            if j > 0 and D[i][j-1] + skip < best[0]: best = (D[i][j-1] + skip, 'c')
            D[i][j] = best[0]; bt[(i, j)] = best[1]
    i, j = L, C; out = {}
    while i > 0 or j > 0:
        op = bt[(i, j)]
        if op == 'm': out[j-1] = i-1; i -= 1; j -= 1
        elif op == 'l': i -= 1
        else: j -= 1
    return out

def parse_page(rec):
    raw = '\n'.join(rec['texts'])
    lines_raw = [l.strip() for l in raw.split('\n')]
    lines_raw = [l for l in lines_raw if l]
    stats = dict(n_lines_raw=len(lines_raw), n_num_lines=0, n_latin_lines=0, n_md_lines=0, n_noncjk_chars=0, n_repeat_lines=0,
                 latin_chars=0, gen_tokens=rec['gen_tokens'], sec=rec['sec'])
    lines = []
    for l in lines_raw:
        c = cjk(l)
        if re.fullmatch(r'[\d\s.,:;|—\-]+', l): stats['n_num_lines'] += 1; continue
        if not c:
            if re.search(r'[A-Za-z]', l): stats['n_latin_lines'] += 1
            if re.search(r'[`#*|$]', l): stats['n_md_lines'] += 1
            continue
        stats['latin_chars'] += len(re.findall(r'[A-Za-z]', l))
        stats['n_noncjk_chars'] += sum(1 for ch in l if not is_cjk(ch) and not ch.isspace())
        # lặp: chuỗi có 1 đoạn ≥3 ký tự lặp ≥3 lần liên tiếp
        if re.search(r'(.{2,6})\1\1', c): stats['n_repeat_lines'] += 1
        # tách dòng nếu engine ghi số cột lẫn nội dung: chỉ giữ CJK
        lines.append(c)
    stats['n_lines_cjk'] = len(lines); stats['n_chars'] = sum(map(len, lines))
    return lines, stats

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', default='full'); args = ap.parse_args()
    sys.path.insert(0, str(REPO))
    from core.text.dictionary import load_qn_to_nom
    q2n = load_qn_to_nom(str(REPO / 'Dict/QuocNgu_SinoNom.csv')); vocab = {c for v in q2n.values() for c in v}
    labels = pd.read_csv(REPO / 'dataset_out/labels.csv', dtype=str, keep_default_na=False)
    freq = collections.Counter(labels.ocr_char); tot = sum(freq.values())
    outdir = HERE / 'out' / args.mode
    pages = {}; prow = []
    for fn in sorted(outdir.glob('*.json')):
        rec = json.load(open(fn)); key = (rec['book'], rec['page'])
        cache = json.load(open(REPO / f'prepared/{BOOK_DIR[key[0]]}/detected/{key[1]}_ocr_cache.json'))
        cols = [''.join(c['char'] for c in col) for col in cache['columns']]
        lines, st = parse_page(rec)
        m = match_lines_cols(lines, cols)
        st.update(book=key[0], page=key[1], n_cols_kinh=len(cols), n_chars_kinh=sum(map(len, cols)),
                  n_cols_matched=len(m), same_count=len(lines) == len(cols), positional=all(m.get(j) == j for j in range(len(cols))) if len(lines) == len(cols) else False)
        # CER cột (trên cột đã gióng) — so với kinhhannom, chỉ tham khảo
        levs = [(lev_align(lines[li], cols[j])[1], len(cols[j])) for j, li in m.items()]
        st['cer_vs_kinh'] = round(sum(a for a, _ in levs) / max(1, sum(b for _, b in levs)), 3)
        st['n_pred_over_kinh'] = round(st['n_chars'] / max(1, st['n_chars_kinh']), 3)
        pages[key] = (cache, cols, lines, m); prow.append(st)
    P = pd.DataFrame(prow); P.to_csv(HERE / f'out/{args.mode}_pages.csv', index=False)
    res = {'mode': args.mode, 'n_pages': len(P)}
    res['trang'] = dict(
        n_lines_cjk_mean=round(P.n_lines_cjk.mean(), 2), pct_pages_9_lines=round((P.n_lines_cjk == P.n_cols_kinh).mean(), 4),
        dist_lines_minus_cols=P.eval('n_lines_cjk - n_cols_kinh').value_counts().sort_index().to_dict(),
        n_chars_mean=round(P.n_chars.mean(), 1), n_chars_kinh_mean=round(P.n_chars_kinh.mean(), 1), ratio_mean=round(P.n_pred_over_kinh.mean(), 3),
        cer_vs_kinh_mean=round(P.cer_vs_kinh.mean(), 3), sec_mean=round(P.sec.mean(), 1), gen_tokens_mean=round(P.gen_tokens.mean(), 1),
        bia=dict(pages_with_num_line=round((P.n_num_lines > 0).mean(), 4), num_lines_mean=round(P.n_num_lines.mean(), 2),
                 pages_with_latin_line=round((P.n_latin_lines > 0).mean(), 4), pages_with_latin_chars=round((P.latin_chars > 0).mean(), 4),
                 pages_with_md=round((P.n_md_lines > 0).mean(), 4), pages_with_repeat=round((P.n_repeat_lines > 0).mean(), 4),
                 noncjk_chars_per_page=round(P.n_noncjk_chars.mean(), 2),
                 pages_missing_cols=round((P.n_lines_cjk < P.n_cols_kinh).mean(), 4), pages_extra_lines=round((P.n_lines_cjk > P.n_cols_kinh).mean(), 4),
                 cols_unmatched_total=int((P.n_cols_kinh - P.n_cols_matched).sum())))
    for name in ['M1', 'M2', 'M3']:
        mm = pd.read_csv(MAU / f'{name}.csv', dtype=str, keep_default_na=False)
        rows = []
        for r in mm.itertuples():
            key = (r.book, r.page); d = dict(sample_id=r.sample_id, has_page=key in pages, col_matched=False, pred_lev='', pred_prop='', n_pred=0, n_kinh=0)
            if key in pages:
                cache, cols, lines, m = pages[key]; ci = int(r.column) - 1
                if ci < len(cols) and ci in m:
                    col = cache['columns'][ci]; kinh = cols[ci]; pred = lines[m[ci]]
                    bb = json.loads(r.bbox); yc = (bb[1]+bb[3])/2
                    pos = min(range(len(col)), key=lambda k: abs((col[k]['bbox'][1]+col[k]['bbox'][3])/2 - yc))
                    amap, _ = lev_align(pred, kinh)
                    d.update(col_matched=True, pred_lev=amap.get(pos, ''), n_pred=len(pred), n_kinh=len(kinh),
                             pred_prop=pred[min(len(pred)-1, round(pos*len(pred)/len(kinh)))] if pred else '')
            rows.append(d)
        A = mm.merge(pd.DataFrame(rows), on='sample_id'); A['R'] = A.R_candidates.map(lambda s: set(x for x in s.split('|') if x))
        out = dict(n=len(A), pct_page_read=round(A.has_page.mean(), 4), pct_col_matched=round(A.col_matched.mean(), 4))
        p_f = float(sum(sum(freq[c] for c in r) / tot for r in A.R) / len(A)); p_u = float(sum(len(r)/len(vocab) for r in A.R) / len(A))
        out['p_random_in_R_freq'] = round(p_f, 4); out['p_random_in_R_uniform'] = round(p_u, 4)
        for al in ['lev', 'prop']:
            p = A[f'pred_{al}']; inR = [x != '' and x in r for x, r in zip(p, A.R)]; A[f'inR_{al}'] = inR; A[f'eq_{al}'] = (p == A.ocr_char)
            o = dict(got_char=round((p != '').mean(), 4), in_R=round(float(pd.Series(inR).mean()), 4), eq_kinh=round(A[f'eq_{al}'].mean(), 4),
                     in_vocab_given_char=round(float(pd.Series([x in vocab for x in p if x]).mean()), 4) if (p != '').any() else None)
            if name == 'M1':
                o['inR_but_ne_kinh'] = int(sum(1 for x, r, k in zip(p, A.R, A.ocr_char) if x and x in r and x != k))
            if name == 'M2':
                o['rescue_count'] = int(sum(inR)); o['rescue_rate'] = o['in_R']
                o['lift_freq'] = round(o['in_R']/p_f, 1); o['lift_uniform'] = round(o['in_R']/p_u, 1)
                # đối chứng hoán vị: tráo pred giữa các ô M2 (giữ R) 200 lần
                rng = random.Random(20260911); pl = list(p); perm = []
                for _ in range(200):
                    rng.shuffle(pl); perm.append(sum(1 for x, r in zip(pl, A.R) if x and x in r) / len(A))
                o['perm_in_R_mean'] = round(sum(perm)/len(perm), 4); o['perm_in_R_max'] = round(max(perm), 4)
                o['lift_perm'] = round(o['in_R']/max(1e-9, sum(perm)/len(perm)), 1)
                o['rescue_eq_label'] = int(sum(1 for x, r, l in zip(p, A.R, A.label) if x and x in r and x == l))
                o['by_tier'] = {t: dict(n=int(len(g)), in_R=round(g[f'inR_{al}'].mean(), 4), rescue=int(g[f'inR_{al}'].sum())) for t, g in A.groupby('tier')}
            if name == 'M3':
                cnt = collections.Counter(p); o['dist_top'] = [(c or '<rỗng>', k) for c, k in cnt.most_common(12)]
                o['pct_𠊚'] = round(cnt['𠊚']/len(A), 4); o['pct_㝵'] = round(cnt['㝵']/len(A), 4); o['pct_𠊛'] = round(cnt['𠊛']/len(A), 4)
            o['by_book'] = {b: round(g[f'inR_{al}'].mean(), 4) for b, g in A.groupby('book')}
            out[al] = o
        out['n_pred_minus_n_kinh_mean'] = round(float((A.n_pred - A.n_kinh)[A.col_matched].mean()), 2)
        out['pct_col_len_within2'] = round(float(((A.n_pred - A.n_kinh).abs() <= 2)[A.col_matched].mean()), 4)
        res[name] = out
        A.drop(columns=['R']).to_csv(HERE / f'out/{args.mode}_{name}_aligned.csv', index=False)
    json.dump(res, open(HERE / f'ket_qua_{args.mode}.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))

if __name__ == '__main__': main()
