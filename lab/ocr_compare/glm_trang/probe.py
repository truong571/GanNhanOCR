"""Thăm dò: GLM-OCR đọc CẢ TRANG (mlx-vlm), nhiều prompt / tỷ lệ / dải. In đầu ra thô + thời gian + số token ảnh."""
import sys, time, contextlib, importlib, json
from PIL import Image
MODEL = 'mlx-community/GLM-OCR-bf16'
REPO = '/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR'
PROMPTS = {
 'a': 'Text Recognition:',
 'p': '这是一页越南喃字古籍，手写汉字/喃字，共9列竖排文字，从右到左阅读，每列从上到下。页顶的印刷阿拉伯数字是列号，不是内容。请逐列识别，每列输出一行（第一行为最右列），只输出汉字/喃字，不要空格、标点、解释。',
}
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
page = sys.argv[1]; prompt = sys.argv[2]; scale = float(sys.argv[3]); strips = int(sys.argv[4]) if len(sys.argv) > 4 else 1
im = Image.open(page).convert('RGB')
if scale != 1: im = im.resize((int(im.width*scale), int(im.height*scale)), Image.LANCZOS)
fmt = apply_chat_template(processor, config, PROMPTS[prompt], num_images=1)
W = im.width
for s in range(strips):
    x0 = int(W*(strips-1-s)/strips); x1 = int(W*(strips-s)/strips)   # phải → trái
    sub = im.crop((x0, 0, x1, im.height)) if strips > 1 else im
    t0 = time.time()
    res = generate(model, processor, fmt, [sub], max_tokens=600, temperature=0.0, repetition_penalty=1.1, verbose=False)
    print(f'--- strip {s} size={sub.size} {time.time()-t0:.1f}s prompt_tok={res.prompt_tokens} gen_tok={res.generation_tokens}')
    print(res.text)
