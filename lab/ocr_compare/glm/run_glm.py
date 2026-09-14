"""Chạy GLM-OCR (mlx-community/GLM-OCR-bf16 qua mlx-vlm 0.7.0, Apple Silicon, CPU) trên bộ mẫu chung M1-M4.

Dùng: /tmp/venv_glm/bin/python run_glm.py --set M2 --prompt a --mode nat --scale 3
Đầu ra: out/<set>_<prompt>-<mode>-s<scale>.csv  (sample_id, pred, pred_raw, sec, n_tokens, prompt_tokens) — resume được.

Cách nhắc (prompt):
  a   = trung tính, prompt gốc SDK glmocr gửi cho mô hình: "Text Recognition:"
  b   = nói rõ là MỘT chữ Hán/Nôm viết tay, chỉ xuất 1 chữ (tiếng Trung, ngắn — bản dài CN+EN gây lảm nhảm tiếng Anh khi ép)
  c   = ĐÓNG: cho danh sách ứng viên R(âm) và âm Quốc ngữ, hỏi chọn 1 — dùng thông tin từ điển,
        KHÔNG so trực tiếp với a/b
  col = cột dọc (M4): đọc từ trên xuống, chỉ xuất chuỗi ký tự
Chế độ (mode):
  nat   = giải mã tham lam tự nhiên (temperature 0, repetition_penalty 1.1 như config SDK)
  force = như nat nhưng CẤM EOS + dấu/ngoặc/markdown ở TOKEN ĐẦU TIÊN (logits_processor) → ép mô hình đọc,
          đo "mô hình sẽ nói gì nếu buộc phải nói"; tỷ lệ từ chối ở nat là chi phí thật.
Ảnh: crop ×scale (LANCZOS) + viền trắng 8%. Processor GLM-OCR: patch 14, merge 2 → 28 px/token, min 112×112.
"""
import argparse, csv, os, sys, time, json, contextlib, importlib
import pandas as pd
from PIL import Image, ImageOps

REPO = '/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR'
MAU = f'{REPO}/lab/ocr_compare/mau'
OUT = f'{REPO}/lab/ocr_compare/glm/out'
MODEL = 'mlx-community/GLM-OCR-bf16'

PROMPTS = {
    'a': 'Text Recognition:',
    'b': '识别图中这一个手写汉字（越南喃字），只输出该字。',
    'c': '图中是越南喃字古籍里手写的一个字，读音为越南语“{syl}”。它是下列候选字之一：\n{cands}\n只输出其中一个字。',
    'col': '这是越南喃字古籍的一列竖排手写文字，从上到下阅读。请从上到下逐字识别，只输出识别出的汉字/喃字序列（不要空格、不要标点、不要解释）。',
}
BAN_STR = ['`', '``', '```', '\n', '\n\n', ' ', '答', '：', ':', '。', '-', '—', '#', '*', '(', ')', '[', ']', '.', 'Wait', 'The', ',', '<', '</', '$', '$$', '^', '+', '=', '|', '>', 'I', '1', '2', 'x']


def prep(img_path, scale):
    im = Image.open(img_path).convert('RGB')
    if scale and scale != 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    return ImageOps.expand(im, border=int(0.08 * max(im.size)), fill='white')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True)
    ap.add_argument('--prompt', required=True, choices=list(PROMPTS))
    ap.add_argument('--mode', default='nat', choices=['nat', 'force'])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['gpu', 'cpu'])
    ap.add_argument('--scale', type=float, default=3.0)
    ap.add_argument('--max-tokens', type=int, default=12)
    ap.add_argument('--fixed-syl', default='', help='ĐỐI CHỨNG prompt c: dùng R(âm này) cho MỌI ô bất kể ảnh (đo thiên vị vị trí)')
    args = ap.parse_args()

    import mlx.core as mx
    # mlx-vlm 0.7.0: wired_limit đọc device_info()['max_recommended_working_set_size'] → KeyError trên CPU; vá no-op
    common = importlib.import_module('mlx_vlm.generate.common'); dispatch = importlib.import_module('mlx_vlm.generate.dispatch')
    @contextlib.contextmanager
    def _noop(*a, **k):
        yield
    common.wired_limit = _noop; dispatch.wired_limit = _noop
    mx.set_default_device(mx.cpu if args.device == 'cpu' else mx.gpu)
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    model, processor = load(MODEL)
    config = load_config(MODEL)
    tok = processor.tokenizer
    ban = {tok.convert_tokens_to_ids(s) for s in ['<|endoftext|>', '<|user|>', '<|observation|>']}
    for s in BAN_STR:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            ban.add(ids[0])
    ban_arr = mx.array(sorted(ban))

    class Force:
        def __init__(self): self.n = 0
        def __call__(self, tokens, logits):
            self.n += 1
            if self.n == 1:
                logits[..., ban_arr] = -1e9
            return logits

    m = pd.read_csv(f'{MAU}/{args.set}.csv', dtype=str, keep_default_na=False)
    if args.limit:
        m = m.head(args.limit)
    os.makedirs(OUT, exist_ok=True)
    tag = f'{args.set}_{args.prompt}-{args.mode}-s{args.scale:g}' + (f'-CTRL-{args.fixed_syl}' if args.fixed_syl else '')
    FIXED_R = ''
    if args.fixed_syl:
        sys.path.insert(0, REPO)
        from core.text.dictionary import load_qn_to_nom
        FIXED_R = '|'.join(load_qn_to_nom(f'{REPO}/Dict/QuocNgu_SinoNom.csv')[args.fixed_syl])
    out_path = f'{OUT}/{tag}.csv'
    done = set()
    if os.path.exists(out_path):
        done = set(pd.read_csv(out_path, dtype=str, keep_default_na=False).sample_id)
    new = not os.path.exists(out_path)
    f = open(out_path, 'a', newline='')
    w = csv.writer(f)
    if new:
        w.writerow(['sample_id', 'pred', 'pred_raw', 'sec', 'n_tokens', 'prompt_tokens'])
    is_col = args.set == 'M4'
    root = f'{MAU}/' if is_col else f'{REPO}/dataset_out/'
    max_tokens = 96 if is_col else args.max_tokens
    t_start = time.time(); n_run = 0
    for i, r in m.iterrows():
        if r.sample_id in done:
            continue
        if args.prompt == 'c':
            syl, cands_s = (r.syllable, r.R_candidates)
            if args.fixed_syl:
                syl, cands_s = args.fixed_syl, FIXED_R
            cands = ' '.join(x for x in cands_s.split('|') if x)
            prompt = PROMPTS['c'].format(syl=syl, cands=cands)
        else:
            prompt = PROMPTS[args.prompt]
        im = prep(root + r.image, args.scale)
        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        kw = dict(max_tokens=max_tokens, temperature=0.0, repetition_penalty=1.1, verbose=False)
        if args.mode == 'force':
            kw['logits_processors'] = [Force()]
        t0 = time.time()
        for attempt in range(4):  # Metal GPU Timeout khi tranh chấp GPU → thử lại
            try:
                res = generate(model, processor, formatted, [im], **kw); break
            except RuntimeError as e:
                print(f'  retry {attempt+1} {r.sample_id}: {str(e)[:80]}', flush=True); time.sleep(5 * (attempt + 1))
                if args.mode == 'force':
                    kw['logits_processors'] = [Force()]
        else:
            raise SystemExit('GPU lỗi liên tục, dừng để chạy lại (resume)')
        dt = time.time() - t0
        pred = res.text.strip().replace('\n', ' ')
        w.writerow([r.sample_id, pred, json.dumps(res.text, ensure_ascii=False), f'{dt:.3f}', res.generation_tokens, res.prompt_tokens])
        f.flush(); done.add(r.sample_id); n_run += 1
        if n_run % 50 == 0 or n_run <= 3:
            print(f'[{tag}] {len(done)}/{len(m)} {r.sample_id} kinh={r.get("ocr_char", "")} pred={pred!r} {dt:.2f}s', flush=True)
    f.close()
    print(f'done {tag} n_run={n_run} {time.time()-t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
