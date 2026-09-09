# -*- coding: utf-8 -*-
"""
KIỂM TRA CHẤT LƯỢNG BỘ NHÃN HÁN NÔM  (labels.xlsx)
Cột kỳ vọng: ocr_char | syllable | label | unicode | label_level

Chạy trên Google Colab:
    !pip -q install pandas openpyxl xlsxwriter
    # upload labels.xlsx rồi chạy file này
"""

import io, os, re, json, sys, zipfile, unicodedata, collections
import urllib.request
import pandas as pd

# ============================================================
# 0. CẤU HÌNH
# ============================================================
INPUT_FILE  = "labels.xlsx"          # đổi path nếu cần
OUTPUT_FILE = "bao_cao_kiem_tra.xlsx"
USE_INTERNET = True                  # tắt nếu Colab không có mạng
RARE_MIN_RATIO = 0.02                # cách đọc chiếm <2% tổng số lần của chữ đó -> nghi ngờ
RARE_DOMINANT_N = 20                 # và cách đọc chính xuất hiện >= 20 lần


# ============================================================
# 1. TIỆN ÍCH XỬ LÝ ÂM TIẾT TIẾNG VIỆT
# ============================================================
TONE_TABLE = str.maketrans(
    'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ',
    'aaaaaăăăăăăââââââeeeeeêêêêêêiiiiioooooôôôôôôơơơơơơuuuuuưưưưưưyyyyy')

TONE_GROUPS = ['àằầèềìòồờùừỳ', 'áắấéếíóốớúứý', 'ảẳẩẻểỉỏổởủửỷ',
               'ãẵẫẽễĩõỗỡũữỹ', 'ạặậẹệịọộợụựỵ']
TONE_OF = {c: i + 1 for i, g in enumerate(TONE_GROUPS) for c in g}

ONSETS = ['ngh', 'ng', 'nh', 'ch', 'tr', 'th', 'ph', 'kh', 'gh', 'gi', 'qu',
          'b', 'c', 'd', 'đ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's',
          't', 'v', 'x']
CODAS = ['nh', 'ng', 'ch', 'c', 'm', 'n', 'p', 't', 'i', 'y', 'o', 'u']
VN_CHARS = set('aăâbcdđeêghiklmnoôơpqrstuưvxy' +
               'àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ')

def nfc(s):
    return unicodedata.normalize('NFC', str(s))

def detone(s):
    return nfc(s).lower().translate(TONE_TABLE)

def get_tone(s):
    for c in nfc(s).lower():
        if c in TONE_OF:
            return TONE_OF[c]
    return 0

VOWELS = set('aăâeêioôơuưy')

def split_syllable(s):
    """Tách âm tiết -> (âm đầu, vần không dấu). Xử lý riêng gi/qu (gìn, gì, quốc)."""
    base = detone(s)
    for o in ONSETS:
        if base.startswith(o) and len(base) > len(o):
            rest = base[len(o):]
            # 'gi'/'qu' mà phần còn lại không có nguyên âm -> chữ i/u chính là vần
            if o in ('gi', 'qu') and not (set(rest) & VOWELS):
                return o[0], o[1] + rest
            return o, rest
    return '', base

# nhóm âm đầu có thể thay thế nhau khi mượn âm trong chữ Nôm
ONSET_CLASSES = [('c', 'k', 'q', 'qu', 'kh', 'g', 'gh'),
                 ('d', 'gi', 'r', 'v', 'nh'),
                 ('tr', 'ch', 't', 'th'),
                 ('s', 'x', 'ch'),
                 ('ng', 'ngh', 'nh'),
                 ('t', 'th', 'đ', 'd'),
                 ('l', 'n', 'nh', 'r'),
                 ('b', 'm', 'ph', 'v'),
                 ('h', '', 'kh'),
                 ('tr', 'l', 's', 'gi'),   # tổ hợp *tl, *bl thời tiền Nôm
                 ('nh', 'l', 'd'),
                 ('ph', 'b', 'v')]
ONSET_MAP = collections.defaultdict(set)
for g in ONSET_CLASSES:
    for x in g:
        ONSET_MAP[x].update(g)

def onset_sim(a, b):
    if a == b:
        return 1.0
    return 0.6 if b in ONSET_MAP.get(a, ()) else 0.0

DIPHTHONGS = [('yê', 'i'), ('iê', 'i'), ('ia', 'i'), ('ya', 'i'),
              ('uô', 'u'), ('ua', 'u'), ('ươ', 'ư'), ('ưa', 'ư')]

def _norm_rhyme(r):
    for a, b in DIPHTHONGS:          # nguyên âm đôi trước
        r = r.replace(a, b)
    r = re.sub(r'(ng|nh|m)$', 'n', r)   # phụ âm cuối cùng nhóm
    r = re.sub(r'(c|p|ch)$', 't', r)
    r = re.sub(r'y$', 'i', r)
    for a, b in [('ă', 'a'), ('â', 'a'), ('ê', 'e'), ('ô', 'o'), ('ơ', 'o'),
                 ('ư', 'u'), ('e', 'i'), ('u', 'o')]:   # nguyên âm đơn gần nhau
        r = r.replace(a, b)
    return r

def rhyme_sim(a, b):
    if a == b:
        return 1.0
    return 0.6 if _norm_rhyme(a) == _norm_rhyme(b) else 0.0

def phonetic_score(syll, ref):
    """0..1 : mức hợp lý khi chữ có âm Hán Việt `ref` được đọc Nôm là `syll`."""
    if syll == ref:
        return 1.0
    if detone(syll) == detone(ref):
        return 0.95
    o1, r1 = split_syllable(syll)
    o2, r2 = split_syllable(ref)
    return 0.5 * onset_sim(o1, o2) + 0.5 * rhyme_sim(r1, r2)

def best_phonetic(syll, refs):
    best, arg = 0.0, None
    for r in refs:
        s = phonetic_score(syll, r)
        if s > best:
            best, arg = s, r
    return best, arg

SYLLABLE_RE = re.compile(r'^[' + ''.join(sorted(VN_CHARS)) + r']+$')

def syllable_shape_ok(s):
    """Kiểm tra cấu trúc âm tiết quốc ngữ: âm đầu + vần + (phụ âm cuối)."""
    if not SYLLABLE_RE.match(detone(s).lower()):
        return False, 'ký tự lạ / không phải quốc ngữ'
    o, r = split_syllable(s)
    if r == '':
        return False, 'thiếu phần vần'
    coda = ''
    for c in CODAS:
        if r.endswith(c) and len(r) > len(c):
            coda = c
            break
    nucleus = r[:len(r) - len(coda)] if coda else r
    if not nucleus:
        return False, 'thiếu nguyên âm'
    if not set(detone(nucleus)) <= set('aăâeêioôơuưy'):
        return False, 'nguyên âm không hợp lệ: ' + nucleus
    if len(nucleus) > 3:
        return False, 'nguyên âm quá dài: ' + nucleus
    return True, ''


# ============================================================
# 2. TẢI TỪ ĐIỂN THAM CHIẾU (tuỳ chọn, cần internet)
# ============================================================
HV_URL  = "https://raw.githubusercontent.com/omnilingual/nomhan-for-yomitan/main/res/han_raw.txt"
NOM_URL = "https://raw.githubusercontent.com/pearapple123/rime-chunom/master/chu_nom.dict.yaml"
# Từ điển Hán Nôm có LỜI GIẢI NGHĨA BẰNG TIẾNG VIỆT (kiểu Thiều Chửu), ~12.300 chữ, ~25MB
NGHIA_URL = "https://raw.githubusercontent.com/thu-tram/tu-dien-han-nom/main/kanji.json"
UNIHAN_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"

def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def load_hanviet():
    """chữ -> danh sách âm Hán Việt (khối CJK cơ bản U+4E00..)."""
    try:
        txt = _get(HV_URL).decode('utf-8')
    except Exception as e:
        print('  ! Không tải được từ điển Hán Việt:', e)
        return {}
    d, code = {}, 0x4E00
    for line in txt.split('\n'):
        line = line.strip()
        if line:
            d[chr(code)] = [x.strip().lower() for x in line.split(',') if x.strip()]
        code += 1
    return d

def load_nom():
    """chữ -> danh sách âm Nôm (gồm cả cách đọc theo nghĩa)."""
    try:
        txt = _get(NOM_URL).decode('utf-8')
    except Exception as e:
        print('  ! Không tải được từ điển Nôm:', e)
        return {}
    d = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        p = line.split('\t')
        if len(p) >= 2 and len(p[0]) == 1:
            d.setdefault(p[0], set()).add(p[1].strip().lower())
    return {k: sorted(v) for k, v in d.items()}

def load_unihan_vietnamese():
    """kVietnamese trong Unihan (tuỳ chọn, file ~8MB)."""
    try:
        raw = _get(UNIHAN_URL, timeout=180)
        d = {}
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            with z.open('Unihan_Readings.txt') as f:
                for line in io.TextIOWrapper(f, encoding='utf-8'):
                    if line.startswith('#') or '\t' not in line:
                        continue
                    cp, field, val = line.rstrip('\n').split('\t', 2)
                    if field == 'kVietnamese':
                        ch = chr(int(cp[2:], 16))
                        d[ch] = [x.strip().lower() for x in val.split()]
        return d
    except Exception as e:
        print('  ! Bỏ qua Unihan:', e)
        return {}


# ============================================================
# 2b. KIỂM TRA THEO NGHĨA TIẾNG VIỆT
# ============================================================
# Từ ngữ pháp / quá thông dụng: xuất hiện trong lời giải nghĩa của hàng nghìn chữ
# nên không thể coi là bằng chứng "đọc theo nghĩa".
STOP_WORDS = set('''là mà và thì cho của được một những các có không đã sẽ cũng nên
khi như này kia ấy ra vào lại rồi ở cùng hay nếu vì bởi đến với theo gọi nói dùng
chỉ cái sự việc ta ai gì nào đó trong ngoài rất đều tức nghĩa chữ'''.split())

WORD_RE = re.compile(r'[a-zàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗ'
                     r'ộớờởỡợụủứừửữựỳỵỷỹ]+')

class NghiaDict:
    """Từ điển Hán Nôm có lời giải nghĩa tiếng Việt.

    Dùng để phát hiện chữ được ĐỌC THEO NGHĨA (huấn độc) chứ không mượn âm:
    黄 (hoàng) đọc là "vàng", 君 (quân) đọc là "vua", 種 (chủng) đọc là "giống"...
    Cách làm: xem âm quốc ngữ có xuất hiện trong lời giải nghĩa của chữ đó không,
    đồng thời phạt những từ quá phổ biến (là, một, cho, người...) vì chúng có mặt
    trong hàng nghìn mục từ nên không nói lên điều gì.
    """

    def __init__(self, entries=None):
        self.detail, self.mean, self.df = {}, {}, collections.Counter()
        self.n = 0
        if entries:
            self._build(entries)

    def _build(self, entries):
        for e in entries:
            ch = e.get('kanji')
            if not ch or len(ch) != 1:
                continue
            self.detail[ch] = e.get('detail') or ''
            hv = (e.get('mean') or '').lower()
            self.mean[ch] = [x.strip() for x in re.split(r'[,;/]', hv) if x.strip()]
        self.n = max(len(self.detail), 1)
        for t in self.detail.values():
            self.df.update(set(WORD_RE.findall(t.lower())))

    def readings(self, ch):
        """Âm Hán Việt lấy từ từ điển này."""
        return self.mean.get(ch, [])

    def check(self, ch, syllable):
        """-> (điểm 0..1, dẫn chứng). None nếu chữ không có trong từ điển."""
        text = self.detail.get(ch)
        if text is None:
            return None, ''
        low = text.lower()
        if syllable not in WORD_RE.findall(low):
            return 0.0, ''
        rate = self.df[syllable] / self.n            # độ phổ biến của từ
        if syllable in STOP_WORDS or rate > 0.05:
            return 0.2, '(từ quá thông dụng, không đủ làm bằng chứng)'
        # nghĩa gốc nằm ở đoạn đầu, trước dấu ## ngăn các nghĩa phụ
        first = low.split('##')[0]
        score = 0.9 if syllable in WORD_RE.findall(first) else 0.7
        if rate > 0.03:
            score -= 0.2
        i = low.find(syllable)
        j = low.rfind('.', 0, i) + 1
        k = low.find('.', i)
        snippet = text[j:(k if k > 0 else i + 60)].replace('##', ' | ').strip()
        return score, snippet[:160]


def load_nghia():
    """Tải từ điển giải nghĩa tiếng Việt (~25MB, mất vài chục giây)."""
    try:
        data = json.loads(_get(NGHIA_URL, timeout=300).decode('utf-8'))
        return NghiaDict(data)
    except Exception as e:
        print('  ! Không tải được từ điển nghĩa tiếng Việt:', e)
        return NghiaDict()


# ============================================================
# 3. THÔNG TIN UNICODE
# ============================================================
BLOCKS = [(0x2E80, 0x2EFF, 'CJK Radicals Supplement'),
          (0x2F00, 0x2FDF, 'Kangxi Radicals'),
          (0x3400, 0x4DBF, 'CJK Ext A'),
          (0x4E00, 0x9FFF, 'CJK URO'),
          (0xF900, 0xFAFF, 'CJK Compatibility Ideographs'),
          (0x20000, 0x2A6DF, 'CJK Ext B'),
          (0x2A700, 0x2B73F, 'CJK Ext C'),
          (0x2B740, 0x2B81F, 'CJK Ext D'),
          (0x2B820, 0x2CEAF, 'CJK Ext E'),
          (0x2CEB0, 0x2EBEF, 'CJK Ext F'),
          (0x2EBF0, 0x2EE5F, 'CJK Ext I'),
          (0x2F800, 0x2FA1F, 'CJK Compatibility Supplement'),
          (0x30000, 0x3134F, 'CJK Ext G'),
          (0x31350, 0x323AF, 'CJK Ext H')]

def block_of(ch):
    cp = ord(ch)
    for lo, hi, name in BLOCKS:
        if lo <= cp <= hi:
            return name
    return 'NGOÀI KHỐI HÁN (!)'


# ============================================================
# 4. CHẠY KIỂM TRA
# ============================================================
def main(input_file=INPUT_FILE, output_file=OUTPUT_FILE, use_internet=USE_INTERNET):
    print('=' * 70)
    print('ĐỌC DỮ LIỆU:', input_file)
    df = pd.read_excel(input_file, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    need = ['ocr_char', 'syllable', 'label', 'unicode', 'label_level']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit('Thiếu cột: %s | Đang có: %s' % (missing, list(df.columns)))
    df['row'] = df.index + 2          # số dòng thật trong Excel (có header)
    n = len(df)
    print('Số dòng: %d | Số cột: %d' % (n, df.shape[1]))

    issues = []                        # (row, mã lỗi, mức, mô tả)
    def flag(row, code, level, desc):
        issues.append({'row': row, 'ma_loi': code, 'muc_do': level, 'mo_ta': desc})

    # ---------- 4.1 Ô trống ----------
    print('\n[1] Kiểm tra ô trống & khoảng trắng thừa')
    for c in need:
        blank = df[df[c].isna() | (df[c].astype(str).str.strip() == '')]
        if c in ('label', 'unicode'):
            blank = blank[blank['label_level'] != 'syllable']   # hợp lệ khi gán mức âm tiết
        for r in blank['row']:
            flag(r, 'E01_TRONG', 'LỖI', 'Cột "%s" bị trống' % c)
        raw = df[c].dropna().astype(str)
        for r, v in zip(df.loc[raw.index, 'row'], raw):
            if v != v.strip():
                flag(r, 'E02_KHOANG_TRANG', 'CẢNH BÁO', 'Cột "%s" có khoảng trắng đầu/cuối' % c)
    print('    -> %d vấn đề' % len(issues))

    df = df.apply(lambda s: s.str.strip() if s.dtype == object else s)

    # ---------- 4.2 label_level ----------
    print('[2] Kiểm tra logic label_level')
    lv = df['label_level'].fillna('')
    print('    Phân bố:', dict(lv.value_counts()))
    bad_lv = df[~lv.isin(['char', 'syllable'])]
    for r, v in zip(bad_lv['row'], bad_lv['label_level']):
        flag(r, 'E03_LEVEL_LA', 'LỖI', 'label_level lạ: %r' % v)
    m = (lv == 'char') & (df['label'].isna() | df['unicode'].isna())
    for r in df.loc[m, 'row']:
        flag(r, 'E04_CHAR_THIEU_LABEL', 'LỖI', 'label_level=char nhưng thiếu label/unicode')
    m = (lv == 'syllable') & df['label'].notna()
    for r in df.loc[m, 'row']:
        flag(r, 'W05_SYLLABLE_CO_LABEL', 'CẢNH BÁO', 'label_level=syllable nhưng vẫn có label')

    # ---------- 4.3 unicode <-> label ----------
    print('[3] Kiểm tra codepoint khớp với ký tự label')
    sub = df[df['label'].notna() & df['unicode'].notna()]
    n_mismatch = n_badfmt = n_multi = 0
    for r, lab, uni in zip(sub['row'], sub['label'], sub['unicode']):
        u = str(uni).strip().upper().replace('U+', '').replace('0X', '')
        if not re.fullmatch(r'[0-9A-F]{4,6}', u):
            flag(r, 'E06_UNICODE_SAI_DINH_DANG', 'LỖI', 'Mã unicode sai định dạng: %r' % uni)
            n_badfmt += 1
            continue
        if len(lab) != 1:
            flag(r, 'E07_LABEL_NHIEU_KY_TU', 'LỖI',
                 'label có %d ký tự (%r) nhưng chỉ có 1 codepoint' % (len(lab), lab))
            n_multi += 1
            continue
        if chr(int(u, 16)) != lab:
            flag(r, 'E08_UNICODE_KHONG_KHOP', 'LỖI',
                 'U+%s = %r nhưng label = %r' % (u, chr(int(u, 16)), lab))
            n_mismatch += 1
    print('    Sai định dạng: %d | label nhiều ký tự: %d | lệch codepoint: %d'
          % (n_badfmt, n_multi, n_mismatch))

    # ---------- 4.4 khối Unicode ----------
    print('[4] Kiểm tra khối Unicode của label')
    labs = df['label'].dropna()
    blocks = collections.Counter(block_of(c) for c in labs if len(c) == 1)
    for b, c in blocks.most_common():
        print('    %-32s %d' % (b, c))
    for r, lab in zip(df.loc[labs.index, 'row'], labs):
        if len(lab) == 1 and block_of(lab) == 'NGOÀI KHỐI HÁN (!)':
            flag(r, 'E09_KHONG_PHAI_CHU_HAN', 'LỖI',
                 'label %r (U+%04X) không nằm trong khối chữ Hán' % (lab, ord(lab)))

    # ---------- 4.5 chuẩn hoá NFC / ký tự ẩn ----------
    print('[5] Kiểm tra chuẩn hoá NFC & ký tự ẩn')
    cnt = 0
    for c in ['ocr_char', 'syllable', 'label']:
        s = df[c].dropna()
        for r, v in zip(df.loc[s.index, 'row'], s):
            if unicodedata.normalize('NFC', v) != v:
                flag(r, 'W10_KHONG_NFC', 'CẢNH BÁO', 'Cột "%s" chưa chuẩn NFC: %r' % (c, v))
                cnt += 1
            if any(unicodedata.category(ch) in ('Cf', 'Cc') or ch in '\ufe0f\ufe0e' for ch in v):
                flag(r, 'E11_KY_TU_AN', 'LỖI', 'Cột "%s" chứa ký tự ẩn/điều khiển' % c)
                cnt += 1
    print('    -> %d vấn đề' % cnt)

    # ---------- 4.6 âm tiết quốc ngữ ----------
    print('[6] Kiểm tra âm tiết quốc ngữ')
    cnt = 0
    for r, v in zip(df['row'], df['syllable']):
        if pd.isna(v):
            continue
        ok, why = syllable_shape_ok(v)
        if not ok:
            flag(r, 'E12_AM_TIET_KHONG_HOP_LE', 'LỖI', '"%s": %s' % (v, why))
            cnt += 1
    print('    -> %d âm tiết bất thường' % cnt)

    # ---------- 4.7 mâu thuẫn nội bộ ----------
    print('[7] Kiểm tra mâu thuẫn nội bộ')
    d2 = df[df['label'].notna()]
    conflict = (d2.groupby(['ocr_char', 'syllable'])['label']
                  .nunique().reset_index(name='so_label'))
    conflict = conflict[conflict['so_label'] > 1]
    print('    Cặp (ocr_char, syllable) gán ra nhiều label khác nhau: %d' % len(conflict))
    if len(conflict):
        key = set(zip(conflict['ocr_char'], conflict['syllable']))
        for r, o, s, l in zip(d2['row'], d2['ocr_char'], d2['syllable'], d2['label']):
            if (o, s) in key:
                flag(r, 'E13_MAU_THUAN_LABEL', 'LỖI',
                     'Cùng (%s, %s) nhưng label khác nhau, ở đây là %s' % (o, s, l))

    uni_conf = d2.groupby('label')['unicode'].nunique()
    for lab in uni_conf[uni_conf > 1].index:
        for r in d2.loc[d2['label'] == lab, 'row']:
            flag(r, 'E14_UNICODE_MAU_THUAN', 'LỖI', 'Chữ %s được gán nhiều mã unicode' % lab)

    # ---------- 4.8 cách đọc hiếm ----------
    print('[8] Phát hiện cách đọc hiếm (có thể gõ nhầm)')
    pair = d2.groupby(['label', 'syllable']).size().reset_index(name='n')
    tot = pair.groupby('label')['n'].transform('sum')
    top = pair.groupby('label')['n'].transform('max')
    pair['tong_cua_chu'] = tot
    pair['ty_le'] = pair['n'] / tot
    rare = pair[(pair['ty_le'] < RARE_MIN_RATIO) & (top >= RARE_DOMINANT_N)].copy()
    rare = rare.sort_values(['tong_cua_chu', 'n'], ascending=[False, True])
    print('    -> %d cặp (chữ, âm) hiếm bất thường' % len(rare))

    # ---------- 4.9 đối chiếu từ điển: ÂM và NGHĨA ----------
    print('[9] Đối chiếu chữ ↔ âm với từ điển (âm Hán Việt, âm Nôm, nghĩa tiếng Việt)')
    hv, nom, uh, ng = {}, {}, {}, NghiaDict()
    if use_internet:
        print('    Tải từ điển âm Hán Việt ...'); hv = load_hanviet()
        print('    Tải từ điển âm Nôm ...');      nom = load_nom()
        print('    Tải từ điển nghĩa tiếng Việt (~25MB, hơi lâu) ...'); ng = load_nghia()
        # bỏ chú thích nếu muốn dùng thêm Unihan kVietnamese (~8MB)
        # print('    Tải Unihan ...');            uh = load_unihan_vietnamese()
    print('    Âm Hán Việt: %d chữ | Âm Nôm: %d chữ | Nghĩa tiếng Việt: %d chữ'
          % (len(hv), len(nom), len(ng.detail)))

    rows = []
    for lab, syl, cnt_ in zip(pair['label'], pair['syllable'], pair['n']):
        refs_nom = set(nom.get(lab, [])) | set(uh.get(lab, []))
        refs_hv = set(hv.get(lab, [])) | set(ng.readings(lab))
        sem_score, sem_ev = ng.check(lab, syl)
        ph_score, ph_ref = best_phonetic(syl, list(refs_hv | refs_nom)) \
            if (refs_hv or refs_nom) else (None, None)

        if syl in refs_nom:
            verdict, src = 'A. KHỚP TỪ ĐIỂN NÔM', 'âm Nôm'
        elif syl in refs_hv:
            verdict, src = 'B. ĐỌC ÂM HÁN VIỆT', 'âm Hán Việt'
        elif ph_score is not None and ph_score >= 0.9:
            verdict, src = 'D. MƯỢN ÂM, KHÁC THANH ĐIỆU', 'âm'
        elif sem_score is not None and sem_score >= 0.6:
            verdict, src = 'C. ĐỌC THEO NGHĨA (huấn độc)', 'nghĩa tiếng Việt'
        elif ph_score is not None and ph_score >= 0.5:
            verdict, src = 'E. MƯỢN ÂM HỢP LÝ', 'âm'
        elif sem_score is not None and 0.2 < sem_score < 0.6:
            verdict, src = 'F. NGHĨA GẦN ĐÚNG – NÊN XEM', 'nghĩa tiếng Việt'
        elif ph_score is not None and ph_score >= 0.3:
            verdict, src = 'G. ÂM LỆCH – NÊN XEM LẠI', 'âm'
        elif refs_hv or refs_nom or sem_score is not None:
            verdict, src = 'H. KHÔNG GIẢI THÍCH ĐƯỢC – XEM LẠI', ''
        else:
            verdict, src = 'I. KHÔNG CÓ TRONG TỪ ĐIỂN', ''

        rows.append({'label': lab, 'syllable': syl, 'so_lan': cnt_,
                     'unicode': 'U+%04X' % ord(lab) if len(lab) == 1 else '',
                     'ket_luan': verdict, 'can_cu': src,
                     'diem_giong_am': None if ph_score is None else round(ph_score, 2),
                     'am_tham_chieu': ph_ref,
                     'diem_khop_nghia': sem_score,
                     'dan_chung_nghia': sem_ev,
                     'cac_am_tu_dien': ', '.join(sorted(refs_nom | refs_hv)[:12])})
    dic = pd.DataFrame(rows).sort_values(['ket_luan', 'so_lan'], ascending=[True, False])
    for k, g in dic.groupby('ket_luan'):
        print('    %-38s %5d cặp  (%d dòng)' % (k, len(g), g['so_lan'].sum()))

    review = dic[dic['ket_luan'].str.startswith(('F.', 'G.', 'H.'))]

    # ---------- 4.10 thống kê chung ----------
    print('[10] Thống kê chung')
    stats = {
        'Tổng số dòng': n,
        'Số dòng trùng lặp hoàn toàn': int(df.duplicated(subset=need).sum()),
        'Số chữ Hán Nôm khác nhau (label)': int(df['label'].nunique()),
        'Số âm tiết khác nhau': int(df['syllable'].nunique()),
        'Số cặp (chữ, âm) khác nhau': len(pair),
        'Dòng có ocr_char khác label (OCR sai/đã sửa)':
            int((df['label'].notna() & (df['ocr_char'] != df['label'])).sum()),
        'Dòng gán ở mức âm tiết (thiếu chữ)': int((lv == 'syllable').sum()),
        'Số LỖI phát hiện': sum(1 for i in issues if i['muc_do'] == 'LỖI'),
        'Số CẢNH BÁO': sum(1 for i in issues if i['muc_do'] == 'CẢNH BÁO'),
        'Cặp cần rà lại theo từ điển': len(review),
        'Dòng cần rà lại': int(review['so_lan'].sum()),
    }
    for k, v in stats.items():
        print('    %-40s %s' % (k, v))

    # ---------- 4.11 xuất báo cáo ----------
    iss = pd.DataFrame(issues, columns=['row', 'ma_loi', 'muc_do', 'mo_ta'])
    if len(iss):
        iss = iss.merge(df[['row'] + need], on='row', how='left') \
                 .sort_values(['muc_do', 'ma_loi', 'row'])
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as w:
        pd.DataFrame([{'chi_tieu': k, 'gia_tri': v} for k, v in stats.items()]) \
            .to_excel(w, sheet_name='Tong_quan', index=False)
        iss.to_excel(w, sheet_name='Loi_va_canh_bao', index=False)
        review.to_excel(w, sheet_name='Can_ra_lai_tu_dien', index=False)
        dic[dic['ket_luan'].str.startswith('C.')].to_excel(
            w, sheet_name='Doc_theo_nghia', index=False)
        dic[dic['ket_luan'].str.startswith('I.')].to_excel(
            w, sheet_name='Ngoai_tu_dien', index=False)
        rare.to_excel(w, sheet_name='Cach_doc_hiem', index=False)
        dic.to_excel(w, sheet_name='Doi_chieu_tu_dien', index=False)
        pd.DataFrame(sorted(blocks.items(), key=lambda x: -x[1]),
                     columns=['khoi_unicode', 'so_dong']).to_excel(
            w, sheet_name='Khoi_Unicode', index=False)
        pair.sort_values('n', ascending=False).to_excel(
            w, sheet_name='Tan_suat_cap', index=False)
    print('\nĐã ghi báo cáo:', output_file)
    return df, iss, dic, rare


if __name__ == '__main__':
    main()
