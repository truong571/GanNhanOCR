"""v_pb_kythuat (04/10) — PHẢN BIỆN độc lập: hai khởi tạo FST ngẫu nhiên (hạt 11 và 22) có THỰC SỰ khác nhau không, và đầu ra của chúng so với ngữ cảnh phong cách chuẩn ra sao? (CPU, 0 API)

v_pb_kythuat_fst_lap.py cho |ảnh(FST hạt 11) − ảnh(FST hạt 22)| chỉ 1,35/255 (so với 20,1 giữa FST và đường chuẩn, 32,2 do đổi nhiễu). Cần loại trừ khả năng hai FST thực ra
giống hệt nhau (lỗi hạt giống) và hiểu cơ chế: đo trực tiếp, KHÔNG chạy U-Net:
  (a) độ lệch tham số lớn nhất giữa hai mô-đun FST dựng với hạt 11 và 22 (cùng cấu hình mã TN11: build_mss_encoder/build_fst/build_fst_projection/build_original_style_projection);
  (b) cùng một ảnh phong cách (p02_B34/style_0.png): cosine giữa hai tập token ngữ cảnh combined_style_condition (A vs B) theo từng token và trung bình tập;
  (c) so với token phong cách CHUẨN style_hidden_states (style_encoder PROD đã huấn luyện): chuẩn L2 trung bình mỗi token (FST ngẫu nhiên vs chuẩn) và số token (221 vs 9?).
Ra: measure_out/_tn11/verify/pb_kythuat/fst_khac_nhau.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 CUDA_VISIBLE_DEVICES= nice -n 10 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_pb_kythuat_fst_khac_nhau.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "font_diffusion"))
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_kythuat"
torch.set_num_threads(3)


def main():
    from src.configs.fontdiffuser import get_parser
    from src.builders.build import (build_fst, build_fst_projection, build_mss_encoder, build_original_style_projection,
                                    build_style_encoder, get_unet_cross_attention_dim, build_unet, load_state_dict_auto)
    from inference.sample_optimized import get_style_transform
    args = get_parser().parse_args([])
    args.style_image_size = (96, 96)
    unet = build_unet(args=args)
    cad = get_unet_cross_attention_dim(unet)
    ch = args.fst_feature_channels
    if isinstance(ch, str):
        ch = [int(x.strip()) for x in ch.split(",")]

    def build(seed):
        torch.manual_seed(seed)
        m = build_mss_encoder(args=args); f = build_fst(args=args)
        p = build_fst_projection(ch[-1], cad); o = build_original_style_projection(1024, cad)
        return m.eval(), f.eval(), p.eval(), o.eval()

    A = build(11); B = build(22)
    se = build_style_encoder(args=args)
    se.load_state_dict(load_state_dict_auto(str(REPO / "font_diffusion/ckpt/PROD/style_encoder.safetensors")))
    se.eval()
    s0 = Image.fromarray(cv2.imread(str(REPO / "measure_out/_tn11/p02_B34/style_0.png"), cv2.IMREAD_GRAYSCALE)).convert("RGB")
    x = get_style_transform(args.style_image_size)(s0)[None]
    res = dict(cross_attn_dim=int(cad), feature_channels=ch)
    # (a) tham số
    maxd = 0.0
    for ma, mb in zip(A, B):
        for (na, pa), (nb, pb) in zip(ma.named_parameters(), mb.named_parameters()):
            maxd = max(maxd, float((pa - pb).abs().max()))
    res["a_do_lech_tham_so_toi_da_A_vs_B"] = maxd

    def ctx(M, style_vec):
        m, f, p, o = M
        with torch.no_grad():
            feats = m(x)
            tr = f(feats, feats)
            fc = p(tr)
            oc = o(style_vec).unsqueeze(1)
            return torch.cat([fc, oc], dim=1)[0]            # (N+1, D)

    with torch.no_grad():
        sif, sv, _ = se(x)
        std_tokens = sif.permute(0, 2, 3, 1).reshape(1, -1, sif.shape[1])[0]      # (h*w, C) = style_hidden_states chuẩn
    cA, cB = ctx(A, sv), ctx(B, sv)
    res["b_so_token_FST"] = int(cA.shape[0]); res["c_so_token_chuan"] = int(std_tokens.shape[0])
    cos_tok = torch.nn.functional.cosine_similarity(cA, cB, dim=1)
    res["b_cos_tung_token_A_vs_B"] = dict(tb=float(cos_tok.mean()), min=float(cos_tok.min()), max=float(cos_tok.max()))
    mA, mB = cA.mean(0), cB.mean(0)
    res["b_cos_token_trung_binh_A_vs_B"] = float(torch.nn.functional.cosine_similarity(mA, mB, dim=0))
    res["b_chuan_L2_token_FST_A"] = float(cA.norm(dim=1).mean()); res["b_chuan_L2_token_FST_B"] = float(cB.norm(dim=1).mean())
    res["c_chuan_L2_token_phong_cach_chuan"] = float(std_tokens.norm(dim=1).mean())
    res["c_ti_so_do_lech_giua_cac_token_FST"] = float(cA.std(0).mean() / cA.abs().mean())
    res["c_ti_so_do_lech_giua_cac_token_chuan"] = float(std_tokens.std(0).mean() / std_tokens.abs().mean())
    (OUT / "fst_khac_nhau.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
