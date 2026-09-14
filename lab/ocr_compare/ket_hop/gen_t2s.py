"""Sinh t2s_map.json (phồn→giản, OpenCC) cho mọi chữ xuất hiện trong chi_tiet/*_luong.csv. Chạy bằng /tmp/venv_paddle/bin/python (có opencc)."""
import pandas as pd, json, opencc
B = '/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/ocr_compare/ket_hop'
t2s = opencc.OpenCC('t2s'); chars = set()
for s in ('M1', 'M2', 'M3'):
    d = pd.read_csv(f'{B}/chi_tiet/{s}_luong.csv', dtype=str, keep_default_na=False)
    for c in ('K', 'P', 'P_y', 'P_all', 'G', 'P6c', 'P5c', 'PVL', 'Gf', 'label'):
        for v in d[c]: chars.update(v)
mp = {c: t2s.convert(c) for c in chars if c.strip()}
json.dump({k: v for k, v in mp.items() if v != k}, open(f'{B}/t2s_map.json', 'w'), ensure_ascii=False, indent=0)
