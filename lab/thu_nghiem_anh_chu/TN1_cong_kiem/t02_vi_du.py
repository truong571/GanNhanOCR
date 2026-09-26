#!/usr/bin/env python
"""t02_vi_du.py — trang HTML cục bộ để người dùng tự xem bằng mắt 4 nhóm ô của công cụ kiểm (cấu hình f = chính sách v3, τ = 0,995)
trên 2 sách có nhãn người IHR (L16 chọn ngưỡng trên TK, TK chọn ngưỡng trên L16):
  (i) giữ lại và đúng hai vế · (ii) bị hạ và thật sự sai (bắt đúng) · (iii) bị hạ oan (đúng hai vế) · (iv) lọt lưới (giữ mà sai)
Mỗi nhóm lấy 10 ô / sách (seed cố định). Crop được CHÉP vào anh/<md5>.png; chữ vẽ bằng fonts/NomNaTong-Regular.ttf thành glyph/<U+XXXX>.png.
Không mở ảnh bằng LLM. Tự đủ, mở bằng file://. Đầu vào: measure_out/_thu_nghiem_anh_chu/TN1/cells_ihr_decisions.pkl (t01_sweep.py).
Chạy: .venv/bin/python lab/thu_nghiem_anh_chu/TN1_cong_kiem/t02_vi_du.py [--cfg f] [--per 10]
"""
import argparse, hashlib, html, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[2]
SP = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
BIG = REPO / 'measure_out/_thu_nghiem_anh_chu/TN1'
ap = argparse.ArgumentParser(); ap.add_argument('--cfg', default='f'); ap.add_argument('--per', type=int, default=10)
A = ap.parse_args()
FONT = REPO / 'fonts/NomNaTong-Regular.ttf'
CMAP = set(TTFont(str(FONT), lazy=True).getBestCmap().keys())
PIL_FONT = ImageFont.truetype(str(FONT), 88)
ANH = LAB / 'anh'; GLY = LAB / 'glyph'
for d in (ANH, GLY):
    d.mkdir(exist_ok=True)

D = pd.read_pickle(BIG / 'cells_ihr_decisions.pkl')
D = D[D.has_gt].copy()
CE = pd.read_csv(SP / 'kim_bottleneck/harness/cells_eval.csv', dtype=str, keep_default_na=False,
                 usecols=['cell_uid', 'image', 'column', 'syl_idx', 'ocr_char', 'bbox'])
D = D.merge(CE, on='cell_uid', how='left', validate='1:1')
# nhãn pipeline của ô kề (cùng trang, cùng cột, syl_idx ± 1) — chỉ để tham khảo, KHÔNG phải sự thật
CE_all = pd.read_csv(SP / 'kim_bottleneck/harness/cells_eval.csv', dtype=str, keep_default_na=False,
                     usecols=['book', 'page', 'column', 'syl_idx', 'label'])
key = {(r.book, r.page, r.column, r.syl_idx): r.label for r in CE_all.itertuples()}
BK_OF = {'LucVanTien1916': 'LucVanTien1916', 'TruyenKieu1872': 'TruyenKieu1872'}
AB = {'LucVanTien1916': 'L16', 'TruyenKieu1872': 'TK'}
kp = D[f'keep_{A.cfg}'].values.astype(bool); yb = D.y_both.values.astype(bool)
D['grp'] = np.select([kp & yb, ~kp & ~yb, ~kp & yb, kp & ~yb], ['i', 'ii', 'iii', 'iv'], default='?')
GROUPS = [('i', 'Giữ lại và đúng', 'Công cụ giữ; ảnh đúng khe và nhãn đúng chữ người.'),
          ('ii', 'Bị hạ và thật sự sai (bắt đúng)', 'Công cụ hạ; ô sai ảnh (khe) hoặc sai chữ theo nhãn người.'),
          ('iii', 'Bị hạ oan', 'Công cụ hạ nhưng ô đúng cả hai vế. Ô vẫn được giao là GOLD văn bản, chỉ không được chứng nhận "ảnh + chữ".'),
          ('iv', 'Lọt lưới (giữ mà sai)', 'Công cụ giữ nhưng ô sai. Đây là phần sai số còn lại của tập được chứng nhận.')]
rng = np.random.default_rng(20260926)
pick = []
for g, _, _ in GROUPS:
    for bk in AB:
        E = D[(D.grp == g) & (D.book_set == bk)]
        if g == 'ii':   # nửa lỗi ảnh, nửa lỗi chữ nếu đủ
            im = E[E.err_kind.str.startswith('anh')]; ch = E[E.err_kind.str.startswith('chu')]
            k1 = min(len(im), A.per // 2); k2 = min(len(ch), A.per - k1)
            sel = pd.concat([im.iloc[rng.permutation(len(im))[:k1]], ch.iloc[rng.permutation(len(ch))[:k2]]])
        else:
            sel = E.iloc[rng.permutation(len(E))[:A.per]]
        pick.append(sel)
P = pd.concat(pick)


def glyph(ch):
    """vẽ 1 chữ bằng NomNaTong -> glyph/U+XXXX.png ; trả (tên tệp, có trong font?)"""
    ch = ch if isinstance(ch, str) else ''
    if not ch:
        return None, False
    name = 'U+' + '_'.join(f'{ord(c):04X}' for c in ch) + '.png'
    has = all(ord(c) in CMAP for c in ch)
    f = GLY / name
    if not f.exists():
        im = Image.new('L', (104, 104), 255); dr = ImageDraw.Draw(im)
        dr.text((52, 52), ch, font=PIL_FONT, fill=0, anchor='mm')
        im.save(f)
    return name, has


def crop_copy(rel):
    src = REPO / rel
    if not rel or not src.exists():
        return None
    h = hashlib.md5(src.read_bytes()).hexdigest()
    dst = ANH / f'{h}{src.suffix.lower()}'
    if not dst.exists():
        shutil.copyfile(src, dst)
    return dst.name


ERR_VI = {'DUNG_hai_ve': 'đúng hai vế', 'anh_sai_khe': 'LỖI ẢNH: crop nằm sai khe', 'anh_khe_chua_xac_dinh': 'ẢNH: khe chưa xác định (tính là sai)',
          'chu_dong_am_trong_R': 'LỖI CHỮ: đồng âm (cả hai chữ ∈ R(âm))', 'chu_gan_hinh': 'LỖI CHỮ: gần hình', 'chu_bang_chu_ke': 'LỖI CHỮ: nhãn = chữ kề',
          'chu_pua': 'LỖI CHỮ: mã PUA khác', 'chu_khac': 'LỖI CHỮ: khác'}


def fmt(v, p=3):
    try:
        return '' if pd.isna(v) else f'{float(v):.{p}f}'
    except (TypeError, ValueError):
        return html.escape(str(v))


def gcell(title, ch):
    ch = ch if isinstance(ch, str) else ''
    name, has = glyph(ch)
    if not name:
        return f'<div class="g"><div class="gl empty">—</div><div class="gt">{title}</div></div>'
    miss = '' if has else '<span class="miss" title="font NomNaTong không có chữ này">∅font</span>'
    return (f'<div class="g"><img class="gl" src="glyph/{name}" alt="{html.escape(ch)}"><div class="gt">{title}<br>'
            f'<code>{html.escape(ch)}</code> U+{ord(ch[0]):04X}{miss}</div></div>')


cards = {g: [] for g, _, _ in GROUPS}
man = []
for r in P.itertuples():
    img = crop_copy(r.image)
    d_ = 'T' if r.book_set == 'LucVanTien1916' else 'L'
    pw = r.p_wood_T if d_ == 'T' else r.p_wood_L; vs = r.viss_T if d_ == 'T' else r.viss_L
    si = int(r.syl_idx)
    prevL = key.get((BK_OF[r.book_set], r.page, r.column, str(si - 1)), ''); nextL = key.get((BK_OF[r.book_set], r.page, r.column, str(si + 1)), '')
    lab_ok = '✔' if r.y_lab else '✘'
    slot_txt = {'1': 'đúng khe', '0': 'SAI khe', '': 'chưa xác định'}[str(r.slot_ok) if isinstance(r.slot_ok, str) else '']
    reason = r.gold_exact_reason or '(ok)'
    crop_html = f'<img class="crop" src="anh/{img}" alt="crop">' if img else '<div class="crop empty">không có tệp crop</div>'
    cards[r.grp].append(f'''<div class="card">
 <div class="hd"><b>{AB[r.book_set]}</b> · {html.escape(r.page)} c{r.column} âm #{si} «{html.escape(r.syl if isinstance(r.syl, str) else '')}» · <span class="ek {r.err_kind[:3]}">{ERR_VI.get(r.err_kind, r.err_kind)}</span></div>
 <div class="row">{crop_html}
  {gcell('nhãn pipeline ' + lab_ok, r.label)}{gcell('chữ người (GT)', r.gt_char)}{gcell('chữ người ở khe crop nằm', r.gt_img)}</div>
 <div class="row small">{gcell('GT kề trước', r.gt_prev)}{gcell('GT kề sau', r.gt_next)}{gcell('nhãn ô trước', prevL)}{gcell('nhãn ô sau', nextL)}</div>
 <table class="sc"><tr><td>vế ảnh (slot_ok)</td><td>{slot_txt}</td><td>vế chữ (V1+)</td><td>{lab_ok}</td></tr>
 <tr><td>p_wood ({'mô hình TK' if d_ == 'T' else 'mô hình L16'})</td><td>{fmt(pw, 4)}</td><td>viss</td><td>{fmt(vs)}</td></tr>
 <tr><td>lệch dọc/ngang hộp kim</td><td>{fmt(r.ady, 2)} / {fmt(r.adx, 2)}</td><td>vis_z</td><td>{fmt(r.vis_z, 2)}</td></tr>
 <tr><td>dị bản người</td><td>{html.escape(str(r.ta))}</td><td>cờ crop / seg</td><td>{html.escape(str(r.crop_quality_flag))} / {html.escape(str(r.seg_flag))}</td></tr>
 <tr><td colspan="4">quyết định v3: <b>{html.escape(str(r.gold_exact))}</b> — {html.escape(reason)}</td></tr></table>
 <div class="uid">{html.escape(r.cell_uid)}</div></div>''')
    man.append(dict(grp=r.grp, cell_uid=r.cell_uid, book=AB[r.book_set], crop=img, label=r.label, gt=r.gt_char, err_kind=r.err_kind,
                    p_wood=pw, viss=vs, reason=reason))

cnt = D.groupby(['grp', 'book_set']).size().unstack(fill_value=0)
tabs = ''.join(f'<a href="#{g}">({g}) {t} <span class="n">L16 {cnt.loc[g, "LucVanTien1916"] if g in cnt.index else 0:,} · TK {cnt.loc[g, "TruyenKieu1872"] if g in cnt.index else 0:,}</span></a>'
               for g, t, _ in GROUPS).replace(',', '.')
sections = ''.join(f'<section id="{g}"><h2>({g}) {t}</h2><p class="lead">{dsc} Hiển thị {len(cards[g])} ô ngẫu nhiên (seed 20260926).</p>'
                   f'<div class="grid">{"".join(cards[g])}</div></section>' for g, t, dsc in GROUPS)
HTML = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ví dụ công cụ kiểm</title>
<style>
:root{{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--bd:#dedbd2;--ok:#1f7a3a;--bad:#b3261e;--warn:#9a6700;--gl:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171715;--fg:#ecebe6;--mut:#a3a29c;--card:#22221f;--bd:#3a3935;--ok:#6fcf8a;--bad:#ff8a80;--warn:#e3b341;--gl:#f3f2ee}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
header,section{{max-width:1400px;margin:0 auto;padding:12px 16px}}
h1{{font-size:20px;margin:8px 0}} h2{{font-size:17px;margin:18px 0 4px}} .lead{{color:var(--mut);margin:0 0 10px}}
nav{{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}} nav a{{padding:6px 10px;border:1px solid var(--bd);border-radius:6px;color:var(--fg);text-decoration:none;background:var(--card)}}
.n{{color:var(--mut);font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:8px}}
.hd{{font-size:12.5px;margin-bottom:6px}} .row{{display:flex;gap:6px;align-items:flex-start;flex-wrap:wrap}}
.crop{{height:104px;max-width:110px;object-fit:contain;border:1px solid var(--bd);background:#fff}}
.g{{width:72px;text-align:center}} .gl{{width:64px;height:64px;border:1px solid var(--bd);background:var(--gl);display:block;margin:0 auto}}
.small .gl{{width:40px;height:40px}} .small .g{{width:64px}}
.gt{{font-size:10.5px;color:var(--mut);line-height:1.2;margin-top:2px}} .empty{{display:flex;align-items:center;justify-content:center;color:var(--mut)}}
.miss{{color:var(--warn);margin-left:2px}}
table.sc{{width:100%;border-collapse:collapse;font-size:11.5px;margin-top:6px}} table.sc td{{border-top:1px solid var(--bd);padding:2px 4px}}
.ek{{font-weight:600}} .ek.DUN{{color:var(--ok)}} .ek.anh,.ek.chu{{color:var(--bad)}}
.uid{{font-size:10px;color:var(--mut);word-break:break-all;margin-top:4px}}
.note{{border-left:3px solid var(--warn);padding:6px 10px;background:var(--card);margin:8px 0}}
</style></head><body><header>
<h1>Công cụ kiểm ảnh ↔ chữ: ví dụ trên 2 sách có nhãn người (THỬ NGHIỆM 1)</h1>
<p>Cấu hình <b>{A.cfg}</b> (chính sách v3 đầy đủ), mục tiêu đăng ký trước τ = 0,995. Ngưỡng của L16 chọn trên TK, ngưỡng của TK chọn trên L16 (LOBO).
Sự thật = <b>chữ người IHR</b> (đúng chữ theo quy ước dị thể V1+) và <b>slot_ok</b> (crop nằm đúng khe theo hộp cột người vẽ). Khe chưa xác định tính là sai vế ảnh.</p>
<div class="note">Công cụ chỉ <b>hạ</b> ô hoặc đưa ô cho người xem, <b>không bao giờ sửa nhãn</b>. Ô "bị hạ" vẫn nằm trong bộ giao nộp là GOLD văn bản; chỉ không được chứng nhận "ảnh + chữ".
"nhãn ô trước/sau" là nhãn pipeline của ô kề, chỉ để tham khảo, không phải sự thật. Chữ được vẽ bằng NomNaTong; "∅font" = font không có chữ đó.</div>
<nav>{tabs}</nav></header>{sections}
<header><p class="lead">Sinh bởi t02_vi_du.py (0 API). Số ô mỗi nhóm trên nhãn đầu trang tính trên toàn bộ ô GOLD có nhãn người của từng sách.</p></header>
</body></html>'''
(LAB / 'vi_du.html').write_text(HTML, encoding='utf-8')
pd.DataFrame(man).to_csv(LAB / 'vi_du_manifest.csv', index=False)
print(json.dumps(dict(cards={g: len(v) for g, v in cards.items()}, crops=len(list(ANH.iterdir())), glyphs=len(list(GLY.iterdir())),
                      missing_crop=int(sum(m['crop'] is None for m in man)), group_counts=cnt.to_dict()), default=int))
