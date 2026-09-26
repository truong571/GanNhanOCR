#!/usr/bin/env python3
"""s4_html.py — TN2 bước 4: trang xem ví dụ vi_du.html (tự đủ, mở bằng file://).

Nhóm: (1) IHR sửa đúng [điểm vận hành LOBO], (2) IHR sửa sai / làm hỏng / chưa xác định [θ = 0, THẤP hơn điểm vận hành,
để lộ kiểu lỗi — ở điểm vận hành gần như không có], (3) IHR bị hạ [điểm vận hành; crop 'mới' = ứng viên tốt nhất bị từ chối],
(4) Chr đề xuất sửa, (5) STT đề xuất sửa [ƯỚC LƯỢNG, chưa kiểm]. ~20 ô/nhóm, mẫu ngẫu nhiên hạt 20260926.
Mỗi ô: crop CŨ | crop MỚI, chữ nhãn vẽ bằng fonts/NomNaTong-Regular.ttf (PNG), chữ kim kề trên/dưới, điểm m trước/sau,
IHR: chữ GT người ở khe crop cũ / khe crop mới. Ảnh chép vào img/<md5>.png cạnh trang.
Chạy: .venv/bin/python s4_html.py
"""
import hashlib, html, io, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s3_eval as E  # noqa
from slotlib import SlotModel  # noqa
REPO = E.REPO; OUT = E.OUT
IMG = HERE / 'img'
FONT = ImageFont.truetype(str(REPO / 'fonts/NomNaTong-Regular.ttf'), 52)
RNG = np.random.default_rng(20260926)
N = 20


def put_file(src):
    b = Path(src).read_bytes()
    h = hashlib.md5(b).hexdigest()
    dst = IMG / f'{h}.png'
    if not dst.exists():
        dst.write_bytes(b)
    return f'img/{h}.png'


_gcache = {}


def glyph(ch):
    if not ch:
        return ''
    if ch in _gcache:
        return _gcache[ch]
    im = Image.new('L', (64, 64), 255)
    d = ImageDraw.Draw(im)
    bb = d.textbbox((0, 0), ch, font=FONT)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((64 - w) / 2 - bb[0], (64 - h) / 2 - bb[1]), ch, font=FONT, fill=0)
    if np.asarray(im).min() > 200:          # font không có chữ này (PUA...) -> chỉ hiện mã chữ
        _gcache[ch] = ''
        return ''
    buf = io.BytesIO(); im.save(buf, format='PNG')
    h5 = hashlib.md5(buf.getvalue()).hexdigest()
    (IMG / f'{h5}.png').write_bytes(buf.getvalue())
    _gcache[ch] = f'img/{h5}.png'
    return _gcache[ch]


def gtag(ch, cls=''):
    if not ch:
        return '<span class="na">—</span>'
    src = glyph(ch)
    if not src:
        return f'<span class="g {cls} nofont" title="NomNaTong không có chữ này">{html.escape(ch)} U+{ord(ch[0]):04X}</span>'
    return f'<span class="g {cls}"><img src="{src}" alt="{html.escape(ch)}"><i>{html.escape(ch)}</i></span>'


def card(r, verdict, vcls, mcols, gt_old='', gt_new='', note=''):
    old = put_file(OUT / r['crop_old']) if r.get('crop_old') else ''
    new = put_file(OUT / r['crop_new']) if r.get('crop_new') else ''
    uid = r['cell_uid'].split('/', 2)[2]
    ms = ' · '.join(f'm<sub>{t}</sub> {r.get("m_old_" + t, float("nan")):+.2f} → {r.get("m_new_" + t, float("nan")):+.2f}' for t in mcols)
    gt = ''
    if gt_old or gt_new:
        gt = f'<div class="row small">GT người ở khe crop cũ {gtag(gt_old)} · khe crop mới {gtag(gt_new)}</div>'
    return f'''<div class="card">
  <div class="hd"><span class="tag {vcls}">{verdict}</span><span class="uid">{html.escape(r["book_set"][:4])} {html.escape(uid)}</span></div>
  <div class="imgs">
    <figure><img class="crop" src="{old}"><figcaption>cũ</figcaption></figure>
    <figure><img class="crop" src="{new}"><figcaption>mới ({html.escape(r.get("cand", ""))})</figcaption></figure>
    <figure>{gtag(r["label"], "big")}<figcaption>nhãn · {html.escape(r.get("syllable", ""))}</figcaption></figure>
  </div>
  <div class="row small">kim kề trên {gtag(r.get("nb_m1", ""))} dưới {gtag(r.get("nb_p1", ""))}</div>
  {gt}<div class="row small mono">{ms}</div>{f'<div class="row small">{html.escape(note)}</div>' if note else ''}
</div>'''


def main():
    if IMG.exists():
        shutil.rmtree(IMG)
    IMG.mkdir()
    S, L = E.load_all()
    cols_all = E.column_state(L)
    dec = pd.read_csv(OUT / 'decisions.csv', dtype=str, keep_default_na=False)
    Si = S.set_index(['cell_uid', 'cand'])

    def rowinfo(u, cand, tags):
        d = Si.loc[(u, 'deliv')]
        r = dict(cell_uid=u, book_set=d.book_set, label=d.label, syllable=d.syllable, nb_m1=d.nb_m1, nb_p1=d.nb_p1,
                 crop_old=d.crop, cand=cand)
        for t in tags:
            r['m_old_' + t] = float(d[f'm_{t}'])
        if cand:
            c = Si.loc[(u, cand)]
            r['crop_new'] = c.crop
            for t in tags:
                r['m_new_' + t] = float(c[f'm_{t}'])
        return r

    def best_cand(u, tag):
        g = S[(S.cell_uid == u) & S.cand.isin(E.FIXC) & S[f'm_{tag}'].notna()]
        return g.loc[g[f'm_{tag}'].idxmax(), 'cand'] if len(g) else ''

    M = {b: SlotModel(b) for b in E.IHR}
    ce = pd.read_csv(E.SP / 'kim_bottleneck/harness/cells_eval.csv', dtype=str, keep_default_na=False, usecols=['cell_uid', 'geo_status'])
    in_univ = set(ce[ce.geo_status == 'ok'].cell_uid)      # mẫu số IHR: trang có GT người

    def gt_at(u, cand):
        d = Si.loc[(u, cand)]
        return M[d.book_set].slot(d.page, d.column, d.syl_idx, json.loads(d.bbox))['gt_img']

    def samp(df, n):
        return df.iloc[RNG.permutation(len(df))[:n]] if len(df) > n else df

    groups = []
    # (1) sửa đúng — điểm vận hành LOBO
    d1 = dec[dec.book_set.isin(E.IHR) & (dec.cls == 'sua_dung')]
    d1 = samp(d1, N)
    cards = []
    for r in d1.itertuples():
        t = E.MODEL_FOR[r.book_set]
        cards.append(card(rowinfo(r.cell_uid, r.cand, [t]), 'sửa đúng', 'ok', [t], gt_at(r.cell_uid, 'deliv'), gt_at(r.cell_uid, r.cand)))
    groups.append(('IHR — sửa đúng (điểm vận hành LOBO)', f'{len(dec[dec.book_set.isin(E.IHR) & (dec.cls=="sua_dung")])} ô; hiện {len(cards)}. '
                   'Sự thật = chữ GT người ở khe mà crop nằm (mô hình khe người). CHẮC CHẮN THEO MÁY.', cards))
    # (2) lỗi: gom theo thứ tự θ=0 nghiêm -> θ=−0,2 nghiêm -> θ=0 nới -> "chỉ hộp kim" (không bộ kiểm), tới ~N ô
    cards, seen = [], set()
    settings = [('θ = 0, luật nghiêm', 'strict', 0.0), ('θ = −0,2, luật nghiêm', 'strict', -0.2),
                ('θ = 0, luật nới', 'lenient', 0.0), ('chỉ hộp kim, không bộ kiểm', 'strict', None)]
    for lab_s, share, th in settings:
        E.SHARE = share
        for book in E.IHR:
            t = E.MODEL_FOR[book]
            Sb = S[S.book_set == book]
            raw = E.propose(Sb, t, th) if th is not None else E.propose_kim(Sb)
            cols = {k: v for k, v in cols_all.items() if k[0] == book and any(x[0] in raw for x in v)}
            acc, _ = E.resolve(raw, set(Sb.cell_uid), cols, None)
            errs = []
            for u, (cand, bb, m) in acc.items():
                if u not in in_univ or u in seen:
                    continue
                d = Si.loc[(u, 'deliv')]
                g0, g1 = gt_at(u, 'deliv'), gt_at(u, cand)
                ok0 = E.var_eq_plus(g0, d.label); ok1 = E.var_eq_plus(g1, d.label)
                if ok1:
                    continue
                v = 'làm hỏng ô đúng' if ok0 else ('sửa sai' if g1 else 'sửa, khe mới chưa xác định')
                op = dec[(dec.cell_uid == u) & (dec.decision == 'sua')]
                note = f'{lab_s}' + (' — CŨNG bị nhận ở điểm vận hành' if len(op) else ' — KHÔNG bị nhận ở điểm vận hành')
                errs.append((u, cand, t, v, g0, g1, note))
            for (u, cand, t, v, g0, g1, note) in errs:
                if len(cards) >= N:
                    break
                seen.add(u)
                cards.append(card(rowinfo(u, cand, [t]), v, 'bad', [t], g0, g1, note))
    E.SHARE = 'strict'
    groups.append(('IHR — sửa sai / làm hỏng / chưa xác định (ngoài điểm vận hành, để lộ kiểu lỗi)',
                   f'{len(cards)} ô, gom lần lượt từ: θ = 0 nghiêm → θ = −0,2 nghiêm → θ = 0 nới → "chỉ hộp kim" (không bộ kiểm). '
                   'Ở điểm vận hành LOBO chỉ các ô ghi "CŨNG bị nhận" mới xảy ra.', cards))
    # (3) hạ — điểm vận hành
    d3 = dec[dec.book_set.isin(E.IHR) & (dec.decision == 'ha')]
    d3 = pd.concat([samp(d3[d3.cls == 'ha_o_sai'], 12), samp(d3[d3.cls == 'ha_o_dung'], 8)])
    cards = []
    for r in d3.itertuples():
        t = E.MODEL_FOR[r.book_set]
        bc = best_cand(r.cell_uid, t)
        v = 'hạ (crop cũ sai)' if r.cls == 'ha_o_sai' else 'hạ (crop cũ vốn đúng)'
        cards.append(card(rowinfo(r.cell_uid, bc, [t]), v, 'warn' if r.cls == 'ha_o_sai' else 'bad', [t],
                          gt_at(r.cell_uid, 'deliv'), gt_at(r.cell_uid, bc) if bc else '',
                          'crop "mới" = ứng viên điểm cao nhất nhưng KHÔNG được nhận' + (f' (huỷ: {r.huy})' if r.huy else '')))
    groups.append(('IHR — bị hạ (điểm vận hành LOBO)', f'{len(dec[dec.book_set.isin(E.IHR) & (dec.decision=="ha")])} ô; hiện 12 ô crop cũ sai + 8 ô crop cũ vốn đúng.', cards))
    # (4)(5) Chr / STT
    decl = pd.read_csv(OUT / 'decisions_lenient.csv', dtype=str, keep_default_na=False)
    for bs, nm in (('Chrestomathie1872', 'Chr'), ('SachThanhTruyen', 'STT')):
        dd = dec[(dec.book_set == bs) & (dec.decision == 'sua')]
        cards = [card(rowinfo(r.cell_uid, r.cand, ['T', 'L']), 'đề xuất sửa (nghiêm)', 'warn', ['T', 'L'], note='ƯỚC LƯỢNG — chưa kiểm được, cần người xem')
                 for r in samp(dd, N).itertuples()]
        dl = decl[(decl.book_set == bs) & (decl.decision == 'sua') & ~decl.cell_uid.isin(dd.cell_uid)]
        cards += [card(rowinfo(r.cell_uid, r.cand, ['T', 'L']), 'chỉ luật nới đề xuất', 'warn', ['T', 'L'],
                       note='ƯỚC LƯỢNG — luật nghiêm chặn vì trùng hộp với ô bị hạ')
                  for r in samp(dl, max(0, N - len(cards))).itertuples()]
        groups.append((f'{nm} — đề xuất sửa (ƯỚC LƯỢNG, chưa kiểm)', f'{len(dd)} ô đề xuất sửa (luật nghiêm) / {len(dec[dec.book_set == bs])} ô bị cờ; hiện {len(cards)} (thêm ô chỉ luật nới đề xuất nếu thiếu). '
                       'Hai bộ kiểm T và L phải cùng chọn một ứng viên và cùng qua ngưỡng của mình.', cards))
    sec = '\n'.join(f'<section><h2>{html.escape(t)}</h2><p class="sub">{html.escape(s)}</p><div class="grid">{"".join(c)}</div></section>'
                    for t, s, c in groups)
    page = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vòng sửa hộp TN2</title>
<style>
:root{{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e1dc;--ok:#1f7a4a;--bad:#b3261e;--warn:#9a6700}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#161614;--fg:#ecebe6;--mut:#a3a29c;--card:#201f1d;--line:#34332f;--ok:#5fc48d;--bad:#ff8a80;--warn:#e0b04a}}}}
:root[data-theme="dark"]{{--bg:#161614;--fg:#ecebe6;--mut:#a3a29c;--card:#201f1d;--line:#34332f;--ok:#5fc48d;--bad:#ff8a80;--warn:#e0b04a}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,sans-serif;margin:0;padding:16px;max-width:1400px;margin:auto}}
h1{{font-size:20px;margin:4px 0}} h2{{font-size:16px;margin:28px 0 4px}} .sub{{color:var(--mut);margin:0 0 10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}}
.hd{{display:flex;justify-content:space-between;gap:6px;align-items:center;margin-bottom:6px}}
.uid{{font:11px ui-monospace,monospace;color:var(--mut);overflow-wrap:anywhere;text-align:right}}
.tag{{font-size:12px;font-weight:600}} .tag.ok{{color:var(--ok)}} .tag.bad{{color:var(--bad)}} .tag.warn{{color:var(--warn)}}
.imgs{{display:flex;gap:8px;align-items:flex-end}} figure{{margin:0;text-align:center}} figcaption{{font-size:11px;color:var(--mut)}}
img.crop{{height:72px;width:auto;max-width:72px;object-fit:contain;background:#fff;border:1px solid var(--line)}}
.g{{display:inline-flex;align-items:center;gap:2px;vertical-align:middle}} .g img{{height:26px;background:#fff;border-radius:3px}}
.g.big img{{height:72px}} .nofont{{font-size:11px;border:1px dashed var(--line);padding:1px 3px}} .g i{{font-style:normal;font-size:11px;color:var(--mut)}} .g.big i{{display:none}}
.row{{margin-top:4px}} .small{{font-size:12px}} .mono{{font-family:ui-monospace,monospace}} .na{{color:var(--mut)}}
</style></head><body>
<h1>Vòng lặp sửa hộp ảnh GOLD — thử nghiệm 2 (26/09/2026)</h1>
<p class="sub">Luật trùng hộp NGHIÊM (ô bị hạ vẫn giữ hộp cũ). Vòng lặp chỉ đổi HỘP/crop, không bao giờ đổi nhãn. m = cos(crop, nguyên mẫu nhãn) − max cos(crop, đối thủ: chữ kim kề ±1/±2, đồng âm, gần hình);
bộ kiểm T học TK1872, L học LVT1916 (IHR chấm bằng mô hình KHÔNG thấy sách đó). Sự thật IHR = mô hình khe người (hộp cột người vẽ + chữ GT người) — không phải bộ kiểm.
Số liệu: KET_QUA.md.</p>
{sec}
</body></html>'''
    (HERE / 'vi_du.html').write_text(page, encoding='utf-8')
    print('vi_du.html', sum(len(c) for _, _, c in groups), 'ô;', len(list(IMG.iterdir())), 'ảnh')


if __name__ == '__main__':
    main()
