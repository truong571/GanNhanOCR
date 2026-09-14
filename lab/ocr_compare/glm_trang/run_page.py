"""GLM-OCR đọc CẢ TRANG (mlx-vlm, GPU Metal), prompt gốc SDK "Text Recognition:", ảnh nguyên tỷ lệ (1663×2727 ≈ 5.7k token ảnh).
Tập trang = hợp trang trong mau/M1..M3.csv. Đầu ra thô: out/<mode>/<book>_<page>.json (text, lines, sec, token). Resume được.
Dùng: /tmp/venv_glm/bin/python run_page.py --mode full        # cả trang
      /tmp/venv_glm/bin/python run_page.py --mode strip2 --limit 60   # 2 dải dọc (phải→trái) rồi ghép
"""
import argparse, os, time, contextlib, importlib, json
import pandas as pd
from PIL import Image
REPO = '/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR'
HERE = f'{REPO}/lab/ocr_compare/glm_trang'
MODEL = 'mlx-community/GLM-OCR-bf16'
BOOK_DIR = {'stt2': 'SachThanhTruyen2', 'stt4': 'SachThanhTruyen4', 'stt11': 'SachThanhTruyen11'}
PROMPT = 'Text Recognition:'

def pages():
    s = set()
    for m in ['M1', 'M2', 'M3']:
        d = pd.read_csv(f'{REPO}/lab/ocr_compare/mau/{m}.csv', dtype=str, keep_default_na=False)
        s |= set(zip(d.book, d.page))
    return sorted(s)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', default='full'); ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    import mlx.core as mx
    common = importlib.import_module('mlx_vlm.generate.common'); dispatch = importlib.import_module('mlx_vlm.generate.dispatch')
    @contextlib.contextmanager
    def _noop(*a, **k): yield
    common.wired_limit = _noop; dispatch.wired_limit = _noop
    mx.set_default_device(mx.gpu)
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    model, processor = load(MODEL); config = load_config(MODEL)
    fmt = apply_chat_template(processor, config, PROMPT, num_images=1)
    outdir = f'{HERE}/out/{args.mode}'; os.makedirs(outdir, exist_ok=True)
    P = pages()
    if args.limit: P = P[::max(1, len(P)//args.limit)][:args.limit]
    for i, (book, page) in enumerate(P):
        fn = f'{outdir}/{book}_{page}.json'
        if os.path.exists(fn): continue
        im = Image.open(f'{REPO}/prepared/{BOOK_DIR[book]}/pages/{page}.png').convert('RGB')
        strips = 1 if args.mode == 'full' else int(args.mode[-1])
        parts = []; t0 = time.time(); ptok = gtok = 0
        for s in range(strips):
            x0 = int(im.width*(strips-1-s)/strips); x1 = int(im.width*(strips-s)/strips)
            sub = im.crop((x0, 0, x1, im.height)) if strips > 1 else im
            for attempt in range(4):
                try:
                    res = generate(model, processor, fmt, [sub], max_tokens=700, temperature=0.0, repetition_penalty=1.1, verbose=False); break
                except RuntimeError as e:
                    print('retry', e, flush=True); time.sleep(5*(attempt+1))
            else: raise SystemExit('GPU lỗi')
            parts.append(res.text); ptok += res.prompt_tokens; gtok += res.generation_tokens
        dt = time.time() - t0
        json.dump({'book': book, 'page': page, 'mode': args.mode, 'image_size': im.size, 'texts': parts,
                   'sec': round(dt, 2), 'prompt_tokens': ptok, 'gen_tokens': gtok}, open(fn, 'w'), ensure_ascii=False, indent=0)
        print(f'[{i+1}/{len(P)}] {book} {page} {dt:.1f}s gen={gtok}', flush=True)
    print('DONE', flush=True)

if __name__ == '__main__': main()
