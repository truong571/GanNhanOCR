#!/usr/bin/env python
"""s08_vi_du.py — vi_du.html tự đủ (mở file://): crop CŨ ↔ crop CHUẨN (v2 và v1) + chữ nhãn vẽ bằng fonts/NomNaTong-Regular.ttf,
4 nhóm × ~24 ô, lấy mẫu PHÂN TẦNG theo bộ (hạt giống cố định; không chọn tay, không mở ảnh):
  (i)  ô bleed (cờ cũ) được CỨU thành crop sạch một chữ (v2), khe không đổi;
  (ii) ô bị crop chuẩn gắn cờ "hai chữ" / tall / cắt nét (v2);
  (iii) ô bình thường (cũ sạch, mới sạch);
  (iv) ô bị LÀM HỎNG: L16/TK = đo được (khe 1 -> ≠1, hoặc mất > 25 % mực LÕI khe người — s02b);
       bộ khác = NGHI hỏng (crop cũ sạch nhưng crop mới bị cờ cắt nét 'cut').
Ảnh chép vào lab/.../TN4_crop_chuan/anh/<md5>.png. Chỉ đọc bảng + tệp ảnh (máy đọc byte, LLM không xem).
Chạy: .venv/bin/python lab/thu_nghiem_anh_chu/TN4_crop_chuan/s08_vi_du.py"""
import hashlib, html, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BIGR = REPO / 'measure_out/_thu_nghiem_anh_chu/TN4'
ANH = HERE / 'anh'; ANH.mkdir(exist_ok=True)
FONT = ImageFont.truetype(str(REPO / 'fonts/NomNaTong-Regular.ttf'), 100)
SETS8 = ['stt2', 'stt4', 'stt11', 'Chr', 'L83', 'KVK', 'L16', 'TK']
NPER = 24
rng = np.random.default_rng(20260926)


def uid_path(uid):
    a = uid.split('/')
    return '/'.join(a[:3]) + '/' + '_'.join(a[3:]) + '.png'


def cp(p):
    p = Path(p)
    if not p.exists():
        return ''
    b = p.read_bytes(); h = hashlib.md5(b).hexdigest()
    q = ANH / f'{h}.png'
    if not q.exists():
        q.write_bytes(b)
    return f'anh/{h}.png'


def glyph(ch):
    im = Image.new('L', (128, 128), 255)
    d = ImageDraw.Draw(im)
    try:
        bb = d.textbbox((0, 0), ch, font=FONT)
        d.text(((128 - (bb[2] - bb[0])) / 2 - bb[0], (128 - (bb[3] - bb[1])) / 2 - bb[1]), ch, font=FONT, fill=0)
    except Exception:
        pass
    b = np.asarray(im).tobytes(); h = hashlib.md5(b + ch.encode()).hexdigest()
    q = ANH / f'{h}.png'
    if not q.exists():
        im.save(q)
    return f'anh/{h}.png'


def strat(df, n=NPER):
    if not len(df):
        return df
    sets = [s for s in SETS8 if (df.set8 == s).any()]
    per = max(1, int(np.ceil(n / len(sets))))
    parts = []
    for s in sets:
        g = df[df.set8 == s]
        parts.append(g.iloc[rng.permutation(len(g))[:per]])
    out = pd.concat(parts)
    return out.iloc[:max(n, len(sets))]


def main():
    V2 = pd.read_pickle(BIGR / 'v2/gold_tn4.pkl').set_index('cell_uid')
    V1 = pd.read_pickle(BIGR / 'v1/gold_tn4.pkl').set_index('cell_uid')[['flags', 'clean_new', 'slot_new', 'cov_new', 'cont_new']]
    D = V2.join(V1, rsuffix='_v1').reset_index()
    D['file_old'] = [str(REPO / 'dataset/_ALL' / p) for p in D.image]
    CC = pd.read_pickle(BIGR / 'v2/cov_core.pkl').set_index('key')
    D['core_drop'] = D.cell_uid.map(CC.core_drop).fillna(False).astype(bool)
    ihr = D.set8.isin(['L16', 'TK'])
    G = {
        '(i) Ô bleed được CỨU thành crop sạch một chữ (v2), khe không đổi':
            D[D.bleed_old & D.rescued & ~D.damaged & ~D.core_drop],
        '(ii) Ô bị crop chuẩn GẮN CỜ hai chữ / tall / cắt nét (v2) — không ép':
            D[D.f_two | D.tall_new | D.f_cut],
        '(iii) Ô bình thường (cũ sạch, mới sạch một chữ)':
            D[~D.B0_crop_old & D.clean_new & ~D.damaged & ~D.core_drop],
        '(iv) Ô bị LÀM HỎNG — L16/TK: ĐO được (khe 1→≠1, hoặc mất > 25 % mực LÕI khe người); bộ khác: NGHI hỏng (cũ sạch, mới bị cờ cắt nét)':
            pd.concat([D[ihr & (D.damaged | D.core_drop)], D[~ihr & ~D.B0_crop_old & D.f_cut]]),
    }
    css = """:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6960;--card:#fff;--line:#e3e0d6;--acc:#9b3d1f}
@media (prefers-color-scheme:dark){:root{--bg:#1b1a18;--fg:#eceae4;--mut:#a29f95;--card:#252421;--line:#3a3833;--acc:#e0875f}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,Segoe UI,sans-serif;margin:0;padding:16px}
h1{font-size:20px;margin:0 0 6px}h2{font-size:16px;margin:28px 0 8px;color:var(--acc)}p{max-width:980px;color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
.c{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.row{display:flex;gap:6px;align-items:flex-end}.row figure{margin:0;text-align:center;font-size:11px;color:var(--mut)}
.row img{height:88px;width:auto;max-width:96px;object-fit:contain;background:#fff;border:1px solid var(--line);image-rendering:auto}
.m{font-size:11px;color:var(--mut);word-break:break-all;margin-top:4px}.m b{color:var(--fg)}"""
    parts = [f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>Crop chuẩn TN4</title><style>{css}</style></head><body>',
             '<h1>THỬ NGHIỆM 4 — crop cũ ↔ crop chuẩn (v2, v1)</h1>',
             '<p>Mỗi thẻ: crop GIAO NỘP hiện tại (cũ) · crop chuẩn v2 (128×128, khuyến nghị thử) · crop chuẩn v1 (đăng ký trước) · chữ NHÃN vẽ bằng '
             'NomNaTong. Mẫu lấy ngẫu nhiên phân tầng theo bộ (hạt giống 20260926), không chọn tay. Cờ: cut = nét bị cắt/chạm mép cửa sổ, '
             'two = hai khối mực (hai chữ), tall = cao > 1,8 × rộng, ink = lượng mực lệch lớp. Khe: 1 đúng / 0 sai / – chưa xác định (chỉ L16/TK, '
             'mô hình khe người). mực ngoài khe / độ phủ: so với khe người (L16/TK).</p>']
    manifest = []
    for title, df in G.items():
        smp = strat(df)
        parts.append(f'<h2>{html.escape(title)} — {len(df):,} ô trong nhóm, hiện {len(smp)}</h2><div class="grid">'.replace(',', '.'))
        for r in smp.itertuples():
            o = cp(r.file_old)
            n2 = cp(BIGR / 'v2/crops128' / uid_path(r.cell_uid))
            n1 = cp(BIGR / 'v1/crops128' / uid_path(r.cell_uid))
            gl = glyph(r.label)
            fo = ','.join(k for k, v in (('bleed', r.bleed_old), ('tall', r.tall_old), ('trunc', r.trunc_old), ('tight_nb', r.tightnb_old)) if v) or 'sạch'
            fn = r.flags or ('sạch' if r.clean_new else 'cờ pipeline')
            fv1 = r.flags_v1 if isinstance(r.flags_v1, str) and r.flags_v1 else ('sạch' if r.clean_new_v1 else 'cờ pipeline')
            ex = ''
            if r.set8 in ('L16', 'TK'):
                ex = (f' · khe cũ/v2/v1: {r.slot_old_bbox or "–"}/{r.slot_new or "–"}/{r.slot_new_v1 or "–"}'
                      f' · ngoài khe cũ/v2: {r.cont_old:.2f}/{r.cont_new:.2f} · phủ cũ/v2: {r.cov_old:.2f}/{r.cov_new:.2f}'
                      if r.cov_old == r.cov_old else f' · khe cũ/v2: {r.slot_old_bbox or "–"}/{r.slot_new or "–"}')
            parts.append(
                f'<div class="c"><div class="row">'
                f'<figure><img src="{o}" alt="cũ"><figcaption>cũ</figcaption></figure>'
                f'<figure><img src="{n2}" alt="v2"><figcaption>chuẩn v2</figcaption></figure>'
                f'<figure><img src="{n1}" alt="v1"><figcaption>v1</figcaption></figure>'
                f'<figure><img src="{gl}" alt="nhãn"><figcaption>nhãn {html.escape(r.label)}</figcaption></figure></div>'
                f'<div class="m"><b>{r.set8}</b> {html.escape(r.cell_uid)}<br>cờ cũ: <b>{fo}</b> · v2: <b>{html.escape(fn)}</b> · v1: {html.escape(fv1)}{ex}</div></div>')
            manifest.append(dict(nhom=title[:4], set8=r.set8, cell_uid=r.cell_uid, old=o, v2=n2, v1=n1, glyph=gl))
        parts.append('</div>')
    parts.append('</body></html>')
    (HERE / 'vi_du.html').write_text('\n'.join(parts), encoding='utf-8')
    pd.DataFrame(manifest).to_csv(HERE / 'vi_du_manifest.csv', index=False)
    M = pd.DataFrame(manifest)
    print(M.groupby(['nhom', 'set8']).size().unstack(fill_value=0).to_string())
    print('missing imgs', int((M[['old', 'v2', 'v1']] == '').sum().sum()), 'files', len(list(ANH.iterdir())))


if __name__ == '__main__':
    main()
