import argparse
import pandas as pd
from collections import defaultdict
import json
import re
import os
import os.path as osp
import glob
from xlsxwriter.workbook import Workbook

dir_path = osp.dirname(osp.realpath(__file__))

print('Loading test set')
test_set = pd.read_csv(osp.join(dir_path, 'Dictionary Evaluation.csv'))
test_set['qn'] = test_set['qn']\
    .apply(lambda x: re.sub(r'\d+', '', x))\
    .apply(lambda x: re.sub(r'–', ' ', x))\
    .apply(lambda x: re.sub(r'\s+', ' ', x))\
    .apply(lambda x: x.strip())
# re-saved the csv file
test_set.to_csv(osp.join(dir_path, 'Dictionary Evaluation.csv'), index=False)
print(len(test_set), 'entries')
print('Loading qn to sn')
qn_to_sn_df = pd.read_csv(osp.join(dir_path, 'QuocNgu_SinoNom_Dic.csv'))
qn_to_sn = defaultdict(set)
for _, row in qn_to_sn_df.iterrows():
    qn_to_sn[row['QuocNgu']].add(row['SinoNom'])
print(len(qn_to_sn), 'entries')

def process_dict(dict_path, is_saved=False, is_log=False, results_dir=None):
    if is_log:
        print('Evaluating dictionary:', dict_path)
    sim_dict = pd.read_csv(dict_path)
    sim_dict = {
        row['char']: set(eval(row['sim']))
        for _, row in sim_dict.iterrows()
    }
    def is_similar(o, s, q):
        if o == s: return ('exact', o)
        S1 = sim_dict.get(o, set())
        S2 = qn_to_sn.get(q, set())
        S = S1.intersection(S2)
        if len(S) == 0: return ('no_sim', None)
        return ('replaced', S.pop())

    debug_similar = []
    def align(ocr, sn, qn_ori):
        qn = qn_ori.split()
        similar = [[
            is_similar(o, s, q)
            for o in ocr
        ] for s, q in zip(sn, qn)]
        debug_similar.append(similar)
        dp = [[None for _ in range(len(ocr) + 1)] for _ in range(len(sn) + 1)]
        n = len(sn)
        m = len(ocr)
        costs = {
            'insert': 1,
            'delete': 1,
            'true_replace': 1,
            'false_replace': 2,
            'exact': 0,
        }
        def memoi(i, j):
            if i == n and j == m: return 0
            if dp[i][j] is not None: return dp[i][j]
            res = 1e9
            if i < n and j < m and similar[i][j][0] == 'exact': res = min(res, memoi(i + 1, j + 1) + costs['exact']) # exact
            if i < n and j < m and similar[i][j][0] == 'replaced': res = min(res, memoi(i + 1, j + 1) + costs['true_replace']) # true replace
            if i < n and j < m and similar[i][j][0] == 'no_sim': res = min(res, memoi(i + 1, j + 1) + costs['false_replace']) # false replace
            if i < n: res = min(res, memoi(i + 1, j) + costs['insert']) # insert
            if j < m: res = min(res, memoi(i, j + 1) + costs['delete']) # delete
            dp[i][j] = res
            return res
        traceback = []
        counts = defaultdict(int)
        def backtrack(i, j):
            if i == n and j == m: return
            res = memoi(i, j)
            if i < n and j < m and similar[i][j][0] == 'exact' and res == memoi(i + 1, j + 1) + costs['exact']:
                traceback.append(ocr[j])
                backtrack(i + 1, j + 1)
            elif i < n and j < m and similar[i][j][0] == 'replaced' and res == memoi(i + 1, j + 1) + costs['true_replace']:
                counts['true_replace'] += 1
                traceback.append(f'<truerep char=\'{similar[i][j][1]}\'>{ocr[j]}</truerep>')
                backtrack(i + 1, j + 1)
            elif i < n and j < m and similar[i][j][0] == 'no_sim' and res == memoi(i + 1, j + 1) + costs['false_replace']:
                counts['false_replace'] += 1
                traceback.append(f'<falserep char=\'{sn[i]}\'>{ocr[j]}</falserep>')
                backtrack(i + 1, j + 1)
            elif i < n and res == memoi(i + 1, j) + costs['insert']:
                counts['insert'] += 1
                traceback.append(f'<ins>{sn[i]}</ins>')
                backtrack(i + 1, j)
            elif j < m and res == memoi(i, j + 1) + costs['delete']:
                counts['delete'] += 1
                traceback.append(f'<del>{ocr[j]}</del>')
                backtrack(i, j + 1)
            return res
        error = backtrack(0, 0)
        return {
            'correction': traceback,
            'error': error,
            'counts': counts,
            'ocr': ocr,
            'sn': sn,
            'qn': qn_ori,
        }

    total_error = defaultdict(int)
    reports = []
    for i, row in test_set.iterrows():
        report = align(row['ocr'], row['nom'], row['qn'])
        for k, v in report['counts'].items():
            total_error[k] += v
        report['page_id'] = row['page_id']
        report['row number'] = row['row number']
        report['location'] = row['location']
        reports.append(report)
        # print(''.join(report['correction']), json.dumps(report['counts']))

    # sort keys
    total_key_order = {
        'true_replace': 0,
        'false_replace': 1,
        'insert': 2,
        'delete': 3,
    }
    total_error = dict(sorted(total_error.items(), key=lambda x: total_key_order[x[0]]))
    if is_log:
        print('Total error:', json.dumps(total_error))

    if is_saved:
        with Workbook(f"{results_dir}/dict_eval_{osp.basename(dict_path).split('.')[0]}.xlsx") as workbook:
            worksheet   = workbook.add_worksheet(f"Result")
            font_format = workbook.add_format({'font_name': 'Nom Na Tong'})
            red         = workbook.add_format({'color': 'red', 'font_name': 'Nom Na Tong'})
            yellow      = workbook.add_format({'color': 'yellow', 'font_name': 'Nom Na Tong'})
            blue        = workbook.add_format({'color': 'blue', 'font_name': 'Nom Na Tong'})
            green       = workbook.add_format({'color': 'green', 'font_name': 'Nom Na Tong'})
            black       = workbook.add_format({'color': 'black', 'font_name': 'Nom Na Tong'})
            worksheet.write(0, 0, 'page_id', font_format)
            worksheet.write(0, 1, 'row number', font_format)
            worksheet.write(0, 2, 'location', font_format)
            worksheet.write(0, 3, 'ocr', font_format)
            worksheet.write(0, 4, 'correction', font_format)
            worksheet.write(0, 5, 'nom', font_format)
            worksheet.write(0, 6, 'qn', font_format)
            worksheet.write(0, 7, 'string', font_format)

            for i, report in enumerate(reports, start = 1):
                correction = report['correction']
                ocrs = []
                corrs = []
                for s in correction:
                    if match := re.match(r"^<falserep char='(.)'>(.)<\/falserep>$", s):
                        ocrs.extend((red, match.group(2)))
                        corrs.extend((red, match.group(1)))
                    elif match := re.match(r"^<truerep char='(.)'>(.)<\/truerep>$", s):
                        ocrs.extend((green, match.group(2)))
                        corrs.extend((green, match.group(1)))
                    elif match := re.match(r"^<ins>(.)<\/ins>$", s):
                        corrs.extend((blue, match.group(1)))
                    elif match := re.match(r"^<del>(.)<\/del>$", s):
                        ocrs.extend((yellow, match.group(1)))
                    else:
                        ocrs.extend((black, s))
                        corrs.extend((black, s))
                worksheet.write(i, 0, report['page_id'], font_format)
                worksheet.write(i, 1, report['row number'], font_format)
                worksheet.write(i, 2, report['location'], font_format)
                worksheet.write(i, 3, '', font_format)
                worksheet.write_rich_string(i, 3, *ocrs)
                worksheet.write_rich_string(i, 4, *corrs)
                worksheet.write(i, 5, report['sn'], font_format)
                worksheet.write(i, 6, report['qn'], font_format)
                worksheet.write(i, 7, ''.join(report['correction']), font_format)
    return total_error

def main(input_dir, output_dir):
    dict_paths = sorted(glob.glob(osp.join(input_dir, '*.csv')))
    print(f"Found {len(dict_paths)} dictionaries")
    os.makedirs(output_dir, exist_ok=True)
    log = open(osp.join(output_dir, 'log.txt'), 'w')
    df = pd.DataFrame(columns=['dict_path', 'true_replace', 'false_replace', 'insert', 'delete'])

    for dict_path in dict_paths:
        total_error = process_dict(dict_path, is_saved=True, is_log=True, results_dir=output_dir)
        print('='*40)
        log.write(f'{dict_path}\n')
        log.write(json.dumps(total_error) + '\n')
        log.write('='*40 + '\n')
        log.flush()
        df = pd.concat([df, pd.DataFrame([{
            'dict_path': dict_path,
            'true_replace': total_error.get('true_replace', 0),
            'false_replace': total_error.get('false_replace', 0),
            'insert': total_error.get('insert', 0),
            'delete': total_error.get('delete', 0),
        }])], ignore_index=True)

    df.to_csv(osp.join(output_dir, 'summary.csv'), index=False)

    print(f"Done evaluating {len(dict_paths)} dictionaries")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./simdicts')
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
