"""v_kythuat (03/10): CHẨN ĐOÁN đường điều kiện phong cách của FontDiffuser trong TN11 — CPU, KHÔNG sửa tệp gốc.

Phát hiện khi đọc mã/log: core/ranking/fontdiffusion_gen.py::_load_pipeline gọi load_fontdiffuser_pipeline(use_fst=True) trong khi
font_diffusion/ckpt/PROD chỉ có unet/style_encoder/content_encoder (bản gốc, không FST). Log mọi lượt TN11 (p02/p03/p04) in
"Checkpoint for 'mss_encoder' / 'fst_module' / 'fst_projection' / 'original_style_projection' not found" -> bốn mô-đun này mang
trọng số NGẪU NHIÊN và FontDiffuserModelDPMWithFST.forward đưa kết quả của chúng vào ngữ cảnh chú ý-chéo ở các khối UP (encoder_hidden_states[2]).

Phép đo (cùng nhiễu khởi đầu, cùng ảnh nội dung; so hai đường):
  FST  = đúng như TN11 (torch.manual_seed(0) ngay trước _load_pipeline, như p02:115)
  STD  = load_fontdiffuser_pipeline(use_fst=False): FontDiffuserModelDPM — cách dùng đúng của trọng số PROD (cách của bài báo)
Phong cách: style_0, style_1 (ô neo B34 do p02 dựng, chỉ ĐỌC) + biến thể "đậm" (erode x3) và "mảnh" (dilate x3) của style_0
(nét mực tối trên nền trắng: erode làm ĐẬM nét, dilate làm MẢNH nét — xem signals_img.augs).
Đo (mọi số do script sinh):
  d_phong_cach  = |ảnh(style_0) − ảnh(style_1)| trung bình (thang 0-255), cùng chữ, cùng nhiễu
  d_nhieu       = |ảnh(style_0, seed A) − ảnh(style_0, seed B)| (đường chuẩn: độ biến thiên do nhiễu)
  d_mo_hinh     = |ảnh FST − ảnh STD| cùng (chữ, phong cách, nhiễu)
  muc_ti_le     = tỉ lệ điểm ảnh tối (<128); dmuc = mực(đậm) − mực(mảnh): đường phong cách THẬT phải cho dmuc > 0 và lớn
Ra: measure_out/_tn11/verify/kythuat/fst_random/{ket_qua.json, *.png}
    PYTORCH_ENABLE_MPS_FALLBACK=0 CUDA_VISIBLE_DEVICES= .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_kythuat_fst_random.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "font_diffusion"))
OUT = REPO / "measure_out" / "_tn11" / "verify" / "kythuat" / "fst_random"
OUT.mkdir(parents=True, exist_ok=True)
CK = REPO / "font_diffusion" / "ckpt" / "PROD"
P02 = REPO / "measure_out" / "_tn11" / "p02_B34"
CHARS = list("孟恒魂拪仁")        # 5 chữ: 3 chữ từ ví dụ báo cáo + 拪 + 仁 (ít nét)
CFG = 2.0                       # đúng tham số TN11 (p02:118)

torch.set_num_threads(4)
DEV = "cpu"


def save(im, name):
    im.save(OUT / name)


def gen_batch(pipe, args, content, style_pil_list, seed):
    """sinh len(content) ảnh với MỘT ảnh phong cách, nhiễu khởi đầu cố định theo seed (cùng seed -> cùng x_T)."""
    from inference.sample_optimized import get_style_transform
    stt = get_style_transform(args.style_image_size)
    st = stt(style_pil_list.convert("RGB"))
    B = len(content)
    bs = st[None, :].repeat(B, 1, 1, 1)
    gen = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        ims = pipe.generate(content_images=content.to(DEV), style_images=bs.to(DEV), batch_size=B, order=args.order,
                            num_inference_step=args.num_inference_steps,
                            content_encoder_downsample_size=args.content_encoder_downsample_size,
                            t_start=args.t_start, t_end=args.t_end, dm_size=args.content_image_size,
                            algorithm_type=args.algorithm_type, skip_type=args.skip_type, method=args.method,
                            correcting_x0_fn=args.correcting_x0_fn, generator=gen)
    return [np.asarray(i.convert("L"), np.float32) for i in ims]


def main():
    t00 = time.time()
    from core.ranking.fontdiffusion_gen import FontDiffusionGenerator
    from inference.sample_optimized import get_content_transform, load_fontdiffuser_pipeline
    from src.tools.utils import ttf2im

    # ---- đường FST (đúng như TN11 p02: seed 0 ngay trước _load_pipeline)
    torch.manual_seed(0)
    g = FontDiffusionGenerator(ckpt_dir=str(CK), phase1_ckpt_dir=str(CK), font_path=str(REPO / "font_diffusion/fonts/NomNaTong-Regular.ttf"),
                               cache_dir=str(OUT / "_tmp"), batch_size=len(CHARS), device=DEV)
    g._load_pipeline()
    pipe_fst = g.pipe
    args = g.args
    assert args.device == DEV and type(pipe_fst.model).__name__ == "FontDiffuserModelDPMWithFST", type(pipe_fst.model).__name__
    # chứng minh trọng số FST là ngẫu nhiên: không có tệp ckpt tương ứng
    miss = [n for n in ("mss_encoder", "fst_module", "fst_projection", "original_style_projection")
            if not any((CK / f"{n}{e}").exists() for e in (".safetensors", ".pth"))]
    # ---- đường STD: mô hình không FST, cùng trọng số PROD
    pipe_std = load_fontdiffuser_pipeline(args=args, use_fst=False)
    assert type(pipe_std.model).__name__ == "FontDiffuserModel" + "DPM", type(pipe_std.model).__name__
    for p in (pipe_fst, pipe_std):
        p.guidance_scale = CFG

    # ---- đầu vào
    font = g.font_manager.get_font(g.font_manager.get_font_names()[0])
    ct = get_content_transform(args.content_image_size)
    content = torch.stack([ct(ttf2im(font=font, char=c)) for c in CHARS])
    s0 = cv2.imread(str(P02 / "style_0.png"), cv2.IMREAD_GRAYSCALE)
    s1 = cv2.imread(str(P02 / "style_1.png"), cv2.IMREAD_GRAYSCALE)
    k3 = np.ones((3, 3), np.uint8)
    styles = {"s0": s0, "s1": s1, "dam": cv2.erode(s0, k3, iterations=3), "manh": cv2.dilate(s0, k3, iterations=3)}
    ink_style = {k: float((v < 128).mean()) for k, v in styles.items()}
    for k, v in styles.items():
        cv2.imwrite(str(OUT / f"style_{k}.png"), v)
    pil = {k: Image.fromarray(v).convert("RGB") for k, v in styles.items()}

    R = {}
    for tag, pipe in (("FST", pipe_fst), ("STD", pipe_std)):
        for k in ("s0", "s1", "dam", "manh"):
            t0 = time.time()
            R[(tag, k, 1)] = gen_batch(pipe, args, content, pil[k], seed=1)
            print(f"[{tag}] phong cách {k} seed 1: {len(CHARS)} ảnh [{time.time() - t0:.0f}s]", flush=True)
        t0 = time.time()
        R[(tag, "s0", 2)] = gen_batch(pipe, args, content, pil["s0"], seed=2)
        print(f"[{tag}] phong cách s0 seed 2 (đường chuẩn nhiễu) [{time.time() - t0:.0f}s]", flush=True)

    def l1(a, b):
        return float(np.mean([np.abs(x - y).mean() for x, y in zip(a, b)]))

    def ink(a):
        return float(np.mean([(x < 128).mean() for x in a]))

    res = dict(thiet_lap=dict(chars="".join(CHARS), guidance=CFG, steps=args.num_inference_steps, device=DEV,
                              torch=torch.__version__, thieu_ckpt_FST=miss, model_fst=type(pipe_fst.model).__name__,
                              model_std=type(pipe_std.model).__name__, ink_anh_phong_cach=ink_style),
               seconds=round(time.time() - t00))
    for tag in ("FST", "STD"):
        res[tag] = dict(
            d_phong_cach_s0_s1=l1(R[(tag, "s0", 1)], R[(tag, "s1", 1)]),
            d_nhieu_seed1_seed2_s0=l1(R[(tag, "s0", 1)], R[(tag, "s0", 2)]),
            d_dam_manh=l1(R[(tag, "dam", 1)], R[(tag, "manh", 1)]),
            muc={k: ink(R[(tag, k, 1)]) for k in ("s0", "s1", "dam", "manh")},
        )
        res[tag]["dmuc_dam_tru_manh"] = res[tag]["muc"]["dam"] - res[tag]["muc"]["manh"]
        res[tag]["ti_so_d_phong_cach_tren_d_nhieu"] = res[tag]["d_phong_cach_s0_s1"] / max(res[tag]["d_nhieu_seed1_seed2_s0"], 1e-9)
    res["d_mo_hinh_FST_vs_STD"] = {k: l1(R[("FST", k, 1)], R[("STD", k, 1)]) for k in ("s0", "s1", "dam", "manh")}
    # montage: hàng = (mô hình, phong cách), cột = chữ
    rows = []
    for tag in ("FST", "STD"):
        for k in ("s0", "s1", "dam", "manh"):
            rows.append(np.concatenate([x.astype(np.uint8) for x in R[(tag, k, 1)]], axis=1))
    cv2.imwrite(str(OUT / "montage_FST(4 hang tren)_STD(4 hang duoi)_s0_s1_dam_manh.png"), np.concatenate(rows, axis=0))
    (OUT / "ket_qua.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
