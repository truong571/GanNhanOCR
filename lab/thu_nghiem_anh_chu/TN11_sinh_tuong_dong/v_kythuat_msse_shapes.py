"""v_kythuat (03/10): kích thước thật của MSSE trong font_diffusion/src/modules/msse.py (CPU, vài giây, không sửa mã gốc).
Báo cáo nói "MSSE 5 tầng phân giải 128 đến 8". Script đưa ảnh phong cách 96x96 (kích thước mặc định của FontDiffuser) qua
MultiScaleStyleEncoder(3, 64, 5) và in hình dạng đầu ra từng tầng + số tham số; kiểm tra thêm khối deformable (DeformConv2d)
và các lớp RSI/MCA có trong U-Net dựng từ build_unet (cùng cấu hình mặc định).
Ra: measure_out/_tn11/verify/kythuat/msse_shapes.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "font_diffusion"))
OUT = REPO / "measure_out" / "_tn11" / "verify" / "kythuat"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)

from src.modules.msse import MultiScaleStyleEncoder  # noqa: E402

res = {}
for size in (96, 128):
    m = MultiScaleStyleEncoder(in_channels=3, base_channels=64, num_scales=5).eval()
    with torch.no_grad():
        feats = m(torch.zeros(1, 3, size, size))
    res[f"vao_{size}x{size}"] = [list(f.shape[1:]) for f in feats]   # [C,H,W]
res["so_tham_so_MSSE"] = sum(p.numel() for p in MultiScaleStyleEncoder(3, 64, 5).parameters())

# U-Net mặc định của font_diffusion: liệt kê khối
from src.configs.fontdiffuser import get_parser  # noqa: E402
from src.builders.build import build_unet  # noqa: E402
import io, contextlib  # noqa: E402

args = get_parser().parse_args([])
with contextlib.redirect_stdout(io.StringIO()):
    unet = build_unet(args)
names = {}
for n, mod in unet.named_modules():
    t = type(mod).__name__
    if t in ("MCADownBlock2D", "StyleRSIUpBlock2D", "DeformConv2d", "OffsetRefStrucInter", "ChannelAttnBlock", "SpatialTransformer",
             "DownBlock2D", "UpBlock2D"):
        names[t] = names.get(t, 0) + 1
res["unet_khoi"] = names
res["unet_tham_so"] = sum(p.numel() for p in unet.parameters())
res["tham_so_mac_dinh"] = dict(resolution=args.resolution, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps,
                               algorithm_type=args.algorithm_type, order=args.order, method=args.method, skip_type=args.skip_type,
                               content_encoder_downsample_size=args.content_encoder_downsample_size, mss_num_scales=args.mss_num_scales,
                               mss_base_channels=args.mss_base_channels, fst_num_queries=args.fst_num_queries)
import torchvision  # noqa: E402
res["torchvision"] = torchvision.__version__
res["torch"] = torch.__version__
(OUT / "msse_shapes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
