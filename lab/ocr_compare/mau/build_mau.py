"""Dựng bộ mẫu chung M1-M4 cho so sánh OCR (seed 20260911).

Chạy:  PYTHONPATH=<repo> <repo>/.venv/bin/python lab/ocr_compare/mau/build_mau.py
Đầu ra: lab/ocr_compare/mau/M1.csv M2.csv M3.csv M4.csv + cot/*.png + README.md

Trọng tài R(âm) = qn_to_nom[syllable] từ Dict/QuocNgu_SinoNom.csv (load_qn_to_nom của pipeline,
đã chuẩn hoá dấu). Cột R_candidates = danh sách '|' các chữ trong R(âm).
"""
import json, os, random, sys
import pandas as pd
from PIL import Image

REPO = '/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR'
sys.path.insert(0, REPO)
from core.text.dictionary import load_qn_to_nom
from core.text.text_utils import normalize_tone_marks

SEED = 20260911
OUT = f'{REPO}/lab/ocr_compare/mau'
BOOK_DIR = {'stt2': 'SachThanhTruyen2', 'stt4': 'SachThanhTruyen4', 'stt11': 'SachThanhTruyen11'}

R = load_qn_to_nom(f'{REPO}/Dict/QuocNgu_SinoNom.csv')
def R_of(s):
    return R.get(normalize_tone_marks(s.strip().lower()), [])

df = pd.read_csv(f'{REPO}/dataset_out/labels.csv', dtype=str, keep_default_na=False)
df['R_candidates'] = ['|'.join(R_of(s)) for s in df.syllable]
df['ocr_in_R'] = [c in R_of(s) for c, s in zip(df.ocr_char, df.syllable)]
df['has_crop'] = [os.path.exists(f'{REPO}/dataset_out/{i}') for i in df.image]
df['R_size'] = [len(R_of(s)) for s in df.syllable]

KEEP = ['image', 'book', 'page', 'column', 'ocr_char', 'syllable', 'label', 'tier', 'rule',
        'bbox', 'crop_w', 'crop_h', 'R_candidates', 'R_size', 'ocr_in_R']

def spread_sample(pool, n, rng):
    """Lấy n ô, rải đều qua các trang (round-robin trên trang xáo trộn)."""
    by_page = {p: g.index.tolist() for p, g in pool.groupby('page')}
    for v in by_page.values():
        rng.shuffle(v)
    pages = list(by_page)
    rng.shuffle(pages)
    out = []
    while len(out) < n:
        progressed = False
        for p in pages:
            if by_page[p]:
                out.append(by_page[p].pop()); progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return pool.loc[out]

rng = random.Random(SEED)
# M1: đối chứng — GOLD/s1_inter_s2_direct, ocr_char ∈ R, 200 ô/sách, rải đều trang
m1_pool = df[(df.tier == 'GOLD') & (df.rule == 's1_inter_s2_direct') & df.has_crop & df.ocr_in_R]
m1 = pd.concat([spread_sample(m1_pool[m1_pool.book == b], 200, rng) for b in ['stt2', 'stt4', 'stt11']])

# M2: khối trượt — ocr_char ∉ R(âm), tier SILVER/SYLLABLE, R(âm) khác rỗng (nếu R rỗng thì không OCR
# nào "cứu" được theo định nghĩa → loại, ghi số lượng vào README)
m2_pool_all = df[(~df.ocr_in_R) & df.tier.isin(['SILVER', 'SYLLABLE']) & df.has_crop]
n_R_empty = int((m2_pool_all.R_size == 0).sum())
m2_pool = m2_pool_all[m2_pool_all.R_size > 0]
m2 = pd.concat([spread_sample(m2_pool[m2_pool.book == b], 200, rng) for b in ['stt2', 'stt4', 'stt11']])

# M3: lớp 㝵/người — 300 ô, tỷ lệ theo sách
m3_pool = df[(df.ocr_char == '㝵') & (df.syllable == 'người') & df.has_crop]
idx = m3_pool.index.tolist(); rng.shuffle(idx)
m3 = m3_pool.loc[idx[:300]]

for name, m in [('M1', m1), ('M2', m2), ('M3', m3)]:
    m = m[KEEP].copy()
    m.insert(0, 'sample_id', [f'{name}_{i:04d}' for i in range(len(m))])
    m.to_csv(f'{OUT}/{name}.csv', index=False)
    print(name, len(m), m.book.value_counts().to_dict(), m.tier.value_counts().to_dict())

# M4: 20 cột nguyên vẹn — ảnh cột cắt từ trang theo hộp cột (boxes_raw) trong cache kinhhannom
os.makedirs(f'{OUT}/cot', exist_ok=True)
pages = df[['book', 'page']].drop_duplicates()
rows = []
per_book = {'stt2': 7, 'stt4': 7, 'stt11': 6}
for b, k in per_book.items():
    pg = pages[pages.book == b].page.tolist(); rng.shuffle(pg)
    for p in pg[:k]:
        cache = f'{REPO}/prepared/{BOOK_DIR[b]}/detected/{p}_ocr_cache.json'
        d = json.load(open(cache))
        cols = d['columns']; raw = d['boxes_raw']
        ci = rng.randrange(len(cols))
        col = cols[ci]
        # hộp cột = union bbox của các chữ trong cột (coords_space fullpage)
        xs0 = min(c['bbox'][0] for c in col); ys0 = min(c['bbox'][1] for c in col)
        xs1 = max(c['bbox'][2] for c in col); ys1 = max(c['bbox'][3] for c in col)
        img = Image.open(f'{REPO}/{d["image"]}')
        pad = 8
        crop = img.crop((max(0, xs0 - pad), max(0, ys0 - pad), min(img.width, xs1 + pad), min(img.height, ys1 + pad)))
        fn = f'cot/{b}_{p}_c{ci+1:02d}.png'
        crop.save(f'{OUT}/{fn}')
        s = ''.join(c['char'] for c in col)
        rows.append(dict(sample_id=f'M4_{len(rows):02d}', image=fn, book=b, page=p, column=ci + 1,
                         bbox=json.dumps([xs0, ys0, xs1, ys1]), col_w=crop.width, col_h=crop.height,
                         kinhhannom_text=s, kinhhannom_n=len(col)))
m4 = pd.DataFrame(rows); m4.to_csv(f'{OUT}/M4.csv', index=False)
print('M4', len(m4), m4.kinhhannom_n.describe().to_dict())

with open(f'{OUT}/README.md', 'w') as f:
    f.write(f"""# Bộ mẫu chung so sánh OCR (seed {SEED})

Nguồn: dataset_out/labels.csv ({len(df)} ô). Trọng tài R(âm) = qn_to_nom[syllable] từ Dict/QuocNgu_SinoNom.csv
(nạp bằng core.text.dictionary.load_qn_to_nom, chuẩn hoá dấu như pipeline; {len(R)} âm).

| Tập | n | Định nghĩa |
|---|---|---|
| M1 | {len(m1)} | GOLD / s1_inter_s2_direct, ocr_char ∈ R(âm) — đối chứng; 200 ô/sách, rải đều trang (round-robin) |
| M2 | {len(m2)} | ocr_char ∉ R(âm), tier SILVER/SYLLABLE, R(âm) ≠ ∅ — PHÉP ĐO CHÍNH; 200 ô/sách, rải đều trang. (Pool {len(m2_pool_all)}, trong đó {n_R_empty} ô có âm ngoài từ điển → R rỗng, đã loại) |
| M3 | {len(m3)} | ocr_char = 㝵, syllable = người — phá sản; ngẫu nhiên từ {len(m3_pool)} ô. LƯU Ý R('người') chứa cả 㝵 lẫn 𠊚 nên R không phân xử được, phải đo phân bố |
| M4 | {len(m4)} | cột nguyên vẹn, cắt từ prepared/<sách>/pages/<trang>.png theo union bbox chữ trong cache kinhhannom (+8px); cột chọn ngẫu nhiên trong trang |

Cột `image` của M1-M3 là đường dẫn tương đối trong dataset_out/. `R_candidates` = các chữ trong R(âm), phân cách '|'.
M4: `image` tương đối trong thư mục này (cot/), `kinhhannom_text` = chuỗi kinhhannom đọc ra cho cột, `kinhhannom_n` = số chữ.
""")
print('done')
