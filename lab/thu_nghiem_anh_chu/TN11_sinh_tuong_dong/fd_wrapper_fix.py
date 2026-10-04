"""Bản sửa wrapper FontDiffusionGenerator cho các thí nghiệm TN11 (core/ranking/fontdiffusion_gen.py KHÔNG được sửa — quy ước CLAUDE.md).

Lỗi gốc (phát hiện 04/10 khi thẩm định): _build_args đặt args.use_fst = False (chú thích "production weights at HF repo root are
non-FST") nhưng _load_pipeline gọi load_fontdiffuser_pipeline(args, use_fst=True); tham số hàm thắng args ⇒ dựng FontDiffuserWithFST với
mss_encoder + fst_module khởi tạo NGẪU NHIÊN (log: "Checkpoint for 'fst_module' not found") ⇒ truyền phong cách bị yếu (chênh mực do ảnh
phong cách chỉ +5,0 so với +17,9 ở đường chuẩn; nội dung chữ vẫn giữ). Lớp này gọi đúng use_fst=False (mô hình FontDiffuser gốc, đúng 3 tệp
unet/style_encoder/content_encoder trong font_diffusion/ckpt/PROD).
"""
from __future__ import annotations

import time

from core.ranking.fontdiffusion_gen import FontDiffusionGenerator


class FixedFontDiffusionGenerator(FontDiffusionGenerator):
    def _load_pipeline(self):
        if self._loaded:
            return
        print(f"  Loading FontDiffusion pipeline (use_fst=False, đường chuẩn) on {self.device}...", flush=True)
        t0 = time.time()
        self.args = self._build_args()
        from inference.sample_optimized import FontManager, load_fontdiffuser_pipeline
        self.pipe = load_fontdiffuser_pipeline(args=self.args, use_fst=False)
        self.font_manager = FontManager(self.font_path)
        self._loaded = True
        print(f"  FontDiffusion loaded in {time.time() - t0:.1f}s", flush=True)
