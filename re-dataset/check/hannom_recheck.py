# -*- coding: utf-8 -*-
"""
CHẠY LẠI BỘ KIỂM TRA HÁN NÔM VÀ SINH LẠI labels.xlsx (bản rút gọn + cột phán quyết)

  Đầu vào : re-dataset/labels.xlsx      (bộ nhãn giao nộp, 30 cột)
  Đầu ra  : re-dataset/check/labels.xlsx
              - sheet "Nhan"     : bộ nhãn đã lọc cột + 4 cột mới
              - sheet "Thong_ke" : thống kê kết quả sau khi chạy
              - sheet "Cap_sai"  : danh sách cặp (chữ, âm) bị đánh False, kèm đề xuất sửa

Cột mới trên sheet "Nhan":
  dung_sai   : TRUE  nếu chữ Hán Nôm giải thích được (phán quyết A-E) và không lỗi hình thức
               FALSE nếu bị lỗi hình thức, hoặc phán quyết F/G/H (không giải thích được)
               ''    nếu phán quyết I (ngoài mọi từ điển - máy không kết luận được)
  sua_thanh  : chữ đề xuất thay thế khi dung_sai = FALSE ('' nếu không tìm được ứng viên)
  ung_vien_sua : tối đa 5 ứng viên, kèm số lần cặp đó xuất hiện trong chính corpus
  ket_luan   : nhánh phán quyết A-I (giữ nguyên logic cây quyết định gốc)

Logic kiểm tra tái sử dụng nguyên vẹn từ 'hannom_check (1).py'.
"""
import os, re, sys, collections, unicodedata, importlib.util
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, 'hannom_check (1).py')
IN   = os.path.join(HERE, '..', 'labels.xlsx')
OUT  = os.path.join(HERE, 'labels.xlsx')

_spec = importlib.util.spec_from_file_location('hannom_check', SRC)
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)

# cột giữ lại: định danh ảnh + nội dung nhãn. Bỏ toàn bộ cột kỹ thuật nội bộ
# (s3_cosine, ink_pct, crop_w/h, image_md5, seg_flag, split*, bbox, seg_backend,
#  crop_quality_flag, stray_ink, border_ink, readmitted_*, rule_goc, tier_goc,
#  usable_image, page_cot_lech, label_in_train) vì không phục vụ việc rà nhãn.
KEEP = ['image', 'book', 'page', 'column', 'ocr_char', 'syllable',
        'label', 'unicode', 'label_level', 'tier']
NEED = ['ocr_char', 'syllable', 'label', 'unicode', 'label_level']


CACHE = os.path.join(HERE, '_cache')


def _get_cached(url, timeout=60):
    """Đọc từ điển ở _cache/ nếu đã tải trước đó, chỉ ra mạng khi thiếu.
    Ba tệp ~24MB, tải lại mỗi lần chạy rất chậm nên cache lại là cần thiết."""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, os.path.basename(url))
    if os.path.exists(f) and os.path.getsize(f) > 0:
        return open(f, 'rb').read()
    data = _get_net(url, timeout)
    open(f, 'wb').write(data)
    return data


_get_net = hc._get
hc._get = _get_cached


def load_dicts(use_internet=True):
    hv, nom, ng = {}, {}, hc.NghiaDict()
    if use_internet:
        print('  tải âm Hán Việt ...');       hv = hc.load_hanviet()
        print('  tải âm Nôm ...');            nom = hc.load_nom()
        print('  tải nghĩa tiếng Việt ...');  ng = hc.load_nghia()
    return hv, nom, ng


def verdict_of(lab, syl, hv, nom, ng):
    """Cây quyết định A-I, giống hệt bản gốc."""
    refs_nom = set(nom.get(lab, []))
    refs_hv = set(hv.get(lab, [])) | set(ng.readings(lab))
    sem_score, sem_ev = ng.check(lab, syl)
    ph_score, ph_ref = (hc.best_phonetic(syl, list(refs_hv | refs_nom))
                        if (refs_hv or refs_nom) else (None, None))
    if syl in refs_nom:                                   v = 'A. KHỚP TỪ ĐIỂN NÔM'
    elif syl in refs_hv:                                  v = 'B. ĐỌC ÂM HÁN VIỆT'
    elif ph_score is not None and ph_score >= 0.9:        v = 'D. MƯỢN ÂM, KHÁC THANH ĐIỆU'
    elif sem_score is not None and sem_score >= 0.6:      v = 'C. ĐỌC THEO NGHĨA (huấn độc)'
    elif ph_score is not None and ph_score >= 0.5:        v = 'E. MƯỢN ÂM HỢP LÝ'
    elif sem_score is not None and 0.2 < sem_score < 0.6: v = 'F. NGHĨA GẦN ĐÚNG – NÊN XEM'
    elif ph_score is not None and ph_score >= 0.3:        v = 'G. ÂM LỆCH – NÊN XEM LẠI'
    elif refs_hv or refs_nom or sem_score is not None:    v = 'H. KHÔNG GIẢI THÍCH ĐƯỢC – XEM LẠI'
    else:                                                 v = 'I. KHÔNG CÓ TRONG TỪ ĐIỂN'
    return {'ket_luan': v,
            'diem_giong_am': None if ph_score is None else round(ph_score, 2),
            'am_tham_chieu': ph_ref,
            'diem_khop_nghia': sem_score,
            'dan_chung_nghia': sem_ev,
            'cac_am_tu_dien': ', '.join(sorted(refs_nom | refs_hv)[:12])}


def formal_errors(df):
    """Lỗi hình thức chắc chắn -> ép FALSE. Trả về Series[bool] và Series[str] mô tả."""
    bad = pd.Series('', index=df.index)

    def add(mask, msg):
        bad.loc[mask & (bad == '')] = msg

    lab, uni, lv = df['label'], df['unicode'], df['label_level'].fillna('')
    has = lab.notna() & (lab.astype(str) != '')
    add(has & (lab.astype(str).str.len() != 1), 'label không phải 1 ký tự')

    def cp_bad(r):
        l, u = r['label'], r['unicode']
        if not isinstance(l, str) or not isinstance(u, str) or len(l) != 1:
            return ''
        uu = u.strip().upper().replace('U+', '').replace('0X', '')
        if not re.fullmatch(r'[0-9A-F]{4,6}', uu):
            return 'mã unicode sai định dạng'
        return '' if chr(int(uu, 16)) == l else 'unicode không khớp label'
    add(has, '')  # no-op giữ thứ tự
    cp = df.loc[has].apply(cp_bad, axis=1)
    for i, m in cp[cp != ''].items():
        if bad.at[i] == '':
            bad.at[i] = m

    blk = has & lab.map(lambda c: isinstance(c, str) and len(c) == 1
                        and hc.block_of(c) == 'NGOÀI KHỐI HÁN (!)')
    add(blk, 'label không nằm trong khối chữ Hán')

    hid = lab.map(lambda v: isinstance(v, str) and any(
        unicodedata.category(ch) in ('Cf', 'Cc') or ch in '️︎' for ch in v))
    add(hid, 'label chứa ký tự ẩn/điều khiển')

    nfc = lab.map(lambda v: isinstance(v, str) and unicodedata.normalize('NFC', v) != v)
    add(nfc, 'label chưa chuẩn NFC')

    add((lv == 'char') & (~has), 'label_level=char nhưng thiếu label')
    return bad


def main(use_internet=True):
    print('ĐỌC:', os.path.abspath(IN))
    raw = pd.read_excel(IN, dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    df = raw[[c for c in KEEP if c in raw.columns]].copy()
    df = df.apply(lambda s: s.str.strip() if s.dtype == object else s)
    n = len(df)
    print('  %d dòng, giữ %d/%d cột' % (n, df.shape[1], raw.shape[1]))

    print('TẢI TỪ ĐIỂN')
    hv, nom, ng = load_dicts(use_internet)
    print('  âm Hán Việt %d | âm Nôm %d | nghĩa %d chữ' % (len(hv), len(nom), len(ng.detail)))

    # --- phán quyết theo CẶP (chữ, âm), rồi ánh xạ ngược về dòng ---
    lab_s = df['label'].fillna('')
    syl_s = df['syllable'].fillna('')
    labeled = df[(lab_s != '') & (syl_s != '')]
    pair_n = labeled.groupby(['label', 'syllable']).size()
    print('ĐỐI CHIẾU %d cặp (chữ, âm) khác nhau' % len(pair_n))
    verd = {}
    for (lab, syl) in pair_n.index:
        verd[(lab, syl)] = verdict_of(lab, syl, hv, nom, ng)

    # --- chỉ mục ngược âm -> chữ, để đề xuất sửa ---
    by_syl = collections.defaultdict(set)
    for ch, rs in nom.items():
        for r in rs:
            by_syl[r].add(ch)
    for ch, rs in hv.items():
        for r in rs:
            by_syl[r].add(ch)
    for ch, rs in ng.mean.items():
        for r in rs:
            by_syl[r].add(ch)
    corpus_n = {k: int(v) for k, v in pair_n.items()}     # tần suất cặp trong chính corpus

    def suggest(lab, syl, ocr, formal):
        """Đề xuất chữ thay thế. Ưu tiên: (1) chữ suy ra từ mã unicode nếu lệch codepoint,
        (2) chữ trong từ điển đọc đúng âm này, xếp theo tần suất trong corpus."""
        pool = by_syl.get(syl, set())
        if not pool:
            return '', ''
        scored = []
        for c in pool:
            if c == lab:
                continue
            scored.append((corpus_n.get((c, syl), 0), c == ocr, c))
        if not scored:
            return '', ''
        scored.sort(key=lambda t: (-t[0], not t[1], t[2]))
        top = scored[:5]
        return top[0][2], ', '.join('%s(%d)' % (c, k) for k, _, c in top)

    formal = formal_errors(df)

    ket, dung, sua, ung = [], [], [], []
    cache_sg = {}
    for lab, syl, ocr, lv, f in zip(lab_s, syl_s, df['ocr_char'].fillna(''),
                                    df['label_level'].fillna(''), formal):
        if lab == '' and lv == 'syllable':
            ket.append('S. GÁN Ở MỨC ÂM TIẾT (không có chữ)')
            dung.append(''); sua.append(''); ung.append('')
            continue
        v = verd.get((lab, syl))
        k = v['ket_luan'] if v else 'I. KHÔNG CÓ TRONG TỪ ĐIỂN'
        ket.append(k)
        if f:
            ok = False
        elif k[0] in 'ABCDE':
            ok = True
        elif k[0] in 'FGH':
            ok = False
        else:
            ok = None                       # nhánh I: máy không kết luận được
        dung.append('' if ok is None else bool(ok))
        if ok is False:
            key = (lab, syl, ocr)
            if key not in cache_sg:
                cache_sg[key] = suggest(lab, syl, ocr, f)
            s1, s5 = cache_sg[key]
            # lệch codepoint: chữ đúng chính là chữ ứng với mã unicode
            sua.append(s1); ung.append(s5)
        else:
            sua.append(''); ung.append('')

    df['ket_luan'] = ket
    df['dung_sai'] = dung
    df['sua_thanh'] = sua
    df['ung_vien_sua'] = ung
    df['loi_hinh_thuc'] = formal.values

    # sửa riêng cho lệch codepoint: đề xuất = chữ của mã unicode
    m = df['loi_hinh_thuc'] == 'unicode không khớp label'
    if m.any():
        def from_uni(u):
            uu = str(u).strip().upper().replace('U+', '').replace('0X', '')
            return chr(int(uu, 16)) if re.fullmatch(r'[0-9A-F]{4,6}', uu) else ''
        df.loc[m, 'sua_thanh'] = df.loc[m, 'unicode'].map(from_uni)

    # ---------------- thống kê ----------------
    vc = df['ket_luan'].value_counts()
    n_true = int((df['dung_sai'] == True).sum())
    n_false = int((df['dung_sai'] == False).sum())
    n_na = int((df['dung_sai'] == '').sum())
    stats = [
        ('Tổng số dòng', n),
        ('Số cột đầu ra / đầu vào', '%d / %d' % (df.shape[1], raw.shape[1])),
        ('Số chữ Hán Nôm khác nhau', int(lab_s[lab_s != ''].nunique())),
        ('Số âm tiết khác nhau', int(syl_s[syl_s != ''].nunique())),
        ('Số cặp (chữ, âm) khác nhau', len(pair_n)),
        ('', ''),
        ('dung_sai = TRUE  (giải thích được)', '%d  (%.2f%%)' % (n_true, 100 * n_true / n)),
        ('dung_sai = FALSE (nghi sai)', '%d  (%.2f%%)' % (n_false, 100 * n_false / n)),
        ('dung_sai = rỗng  (ngoài từ điển / mức âm tiết)', '%d  (%.2f%%)' % (n_na, 100 * n_na / n)),
        ('Trong đó có đề xuất sửa', int((df['sua_thanh'] != '').sum())),
        ('FALSE do lỗi hình thức', int((df['loi_hinh_thuc'] != '').sum())),
        ('', ''),
    ]
    for k, v in vc.items():
        stats.append(('Phán quyết  ' + k, '%d  (%.2f%%)' % (v, 100 * v / n)))
    stats.append(('', ''))
    for t, c in df['tier'].fillna('(trống)').value_counts().items():
        sub = df[df['tier'].fillna('(trống)') == t]
        stats.append(('Tier %s: TRUE / FALSE / rỗng' % t,
                      '%d / %d / %d  (tổng %d)' % ((sub['dung_sai'] == True).sum(),
                                                   (sub['dung_sai'] == False).sum(),
                                                   (sub['dung_sai'] == '').sum(), c)))
    st = pd.DataFrame(stats, columns=['chi_tieu', 'gia_tri'])

    # sheet cặp sai, gộp theo cặp để rà tay cho nhanh
    bad = df[df['dung_sai'] == False]
    cap_sai = (bad.groupby(['label', 'syllable', 'ket_luan', 'sua_thanh',
                            'ung_vien_sua', 'loi_hinh_thuc'], dropna=False)
                  .size().reset_index(name='so_dong')
                  .sort_values('so_dong', ascending=False))

    print('GHI:', os.path.abspath(OUT))
    with pd.ExcelWriter(OUT, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Nhan', index=False)
        st.to_excel(w, sheet_name='Thong_ke', index=False)
        cap_sai.to_excel(w, sheet_name='Cap_sai', index=False)
    print(st.to_string(index=False))
    return df, st, cap_sai


if __name__ == '__main__':
    main(use_internet='--offline' not in sys.argv)
