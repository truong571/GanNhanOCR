"""Thí nghiệm bổ sung: GLM-OCR đọc NGUYÊN CỘT chứa ô (ảnh cột cắt từ trang theo cache kinhhannom), rồi căn
chuỗi đọc ra với chuỗi kinhhannom của cột (Levenshtein alignment) để lấy chữ GLM ứng với từng ô.
Lý do: mô hình từ chối crop chữ đơn (EOS ngay) nhưng đọc được chữ viết tay trong ngữ cảnh cột.
Lưu ý: việc căn dùng chuỗi kinhhannom làm mốc vị trí (cùng phân đoạn cột), KHÔNG dùng nội dung chữ để chấm.

Dùng: /tmp/venv_glm/bin/python run_glm_col.py --set M2
Đầu ra: out/<set>_col-ctx.csv (sample_id, pred [chữ căn được cho ô], col_pred, col_kinh, pos, align_op, sec)
"""
import argparse, csv, os, sys, time, json, contextlib, importlib
import pandas as pd
from PIL import Image

REPO = '/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR'
MAU = f'{REPO}/lab/ocr_compare/mau'
OUT = f'{REPO}/lab/ocr_compare/glm/out'
MODEL = 'mlx-community/GLM-OCR-bf16'
BOOK_DIR = {'stt2': 'SachThanhTruyen2', 'stt4': 'SachThanhTruyen4', 'stt11': 'SachThanhTruyen11'}
PROMPT = 'Text Recognition:'


def is_cjk(c):
    o = ord(c)
    return (0x3400 <= o <= 0x9FFF or 0x20000 <= o <= 0x323AF or 0xF900 <= o <= 0xFAFF or 0xF0000 <= o <= 0x10FFFD)


def align(pred, kinh):
    """Levenshtein alignment; trả về map: chỉ số trong kinh -> (op, ký tự pred hoặc '')"""
    n, m = len(kinh), len(pred)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (kinh[i - 1] != pred[j - 1]))
    i, j = n, m; out = {}
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (kinh[i - 1] != pred[j - 1]):
            out[i - 1] = ('eq' if kinh[i - 1] == pred[j - 1] else 'sub', pred[j - 1]); i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            out[i - 1] = ('del', ''); i -= 1
        else:
            j -= 1  # insertion in pred
    return out, D[n][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    import mlx.core as mx
    common = importlib.import_module('mlx_vlm.generate.common'); dispatch = importlib.import_module('mlx_vlm.generate.dispatch')
    @contextlib.contextmanager
    def _noop(*a, **k):
        yield
    common.wired_limit = _noop; dispatch.wired_limit = _noop
    mx.set_default_device(mx.cpu if args.device == 'cpu' else mx.gpu)
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    model, processor = load(MODEL); config = load_config(MODEL)
    formatted = apply_chat_template(processor, config, PROMPT, num_images=1)

    m = pd.read_csv(f'{MAU}/{args.set}.csv', dtype=str, keep_default_na=False)
    os.makedirs(OUT, exist_ok=True)
    out_path = f'{OUT}/{args.set}_col-ctx.csv'
    done = set()
    if os.path.exists(out_path):
        done = set(pd.read_csv(out_path, dtype=str, keep_default_na=False).sample_id)
    new = not os.path.exists(out_path)
    f = open(out_path, 'a', newline=''); w = csv.writer(f)
    if new:
        w.writerow(['sample_id', 'pred', 'col_pred', 'col_kinh', 'pos', 'align_op', 'col_lev', 'n_col_pred', 'sec'])
    col_cache = {}
    for _, r in m.iterrows():
        if r.sample_id in done:
            continue
        key = (r.book, r.page, int(r.column))
        if key not in col_cache:
            cache = json.load(open(f'{REPO}/prepared/{BOOK_DIR[r.book]}/detected/{r.page}_ocr_cache.json'))
            if key[2] - 1 >= len(cache['columns']):  # ô có chỉ số cột ngoài cache (hiếm) → không đọc được
                w.writerow([r.sample_id, '', '', '', '', 'nocol', '', 0, '0']); f.flush(); done.add(r.sample_id); continue
            col = cache['columns'][key[2] - 1]
            xs0 = min(c['bbox'][0] for c in col); ys0 = min(c['bbox'][1] for c in col)
            xs1 = max(c['bbox'][2] for c in col); ys1 = max(c['bbox'][3] for c in col)
            img = Image.open(f'{REPO}/{cache["image"]}').convert('RGB'); pad = 8
            crop = img.crop((max(0, xs0 - pad), max(0, ys0 - pad), min(img.width, xs1 + pad), min(img.height, ys1 + pad)))
            t0 = time.time()
            for attempt in range(4):
                try:
                    res = generate(model, processor, formatted, [crop], max_tokens=96, temperature=0.0, repetition_penalty=1.1, verbose=False); break
                except RuntimeError as e:
                    print(f'  retry {attempt+1}: {str(e)[:80]}', flush=True); time.sleep(5 * (attempt + 1))
            else:
                raise SystemExit('GPU lỗi liên tục')
            dt = time.time() - t0
            pred_cjk = ''.join(c for c in res.text if is_cjk(c))
            kinh = ''.join(c['char'] for c in col)
            amap, lev = align(pred_cjk, kinh)
            col_cache[key] = (col, pred_cjk, kinh, amap, lev, dt)
        col, pred_cjk, kinh, amap, lev, dt = col_cache[key]
        # vị trí ô trong cột: khớp bbox y-center
        bb = json.loads(r.bbox); yc = (bb[1] + bb[3]) / 2
        pos = min(range(len(col)), key=lambda k: abs((col[k]['bbox'][1] + col[k]['bbox'][3]) / 2 - yc))
        op, ch = amap.get(pos, ('del', ''))
        w.writerow([r.sample_id, ch, pred_cjk, kinh, pos, op, lev, len(pred_cjk), f'{dt:.3f}'])
        f.flush(); done.add(r.sample_id)
        if len(done) % 50 == 0:
            print(f'[{args.set} col-ctx] {len(done)}/{len(m)} kinh={r.ocr_char} -> {ch!r} ({op}) col_lev={lev} n={len(pred_cjk)}/{len(kinh)} {dt:.1f}s', flush=True)
    f.close(); print('done', out_path, 'cols', len(col_cache), flush=True)


if __name__ == '__main__':
    main()
