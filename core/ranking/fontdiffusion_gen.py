"""FontDiffusion batch generator with disk cache.

Loads model once, generates characters in batches, caches to disk.
"""

import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image

# Add FontDiffusion to path
_fd_root = str(Path(__file__).resolve().parents[2] / "font_diffusion")
if _fd_root not in sys.path:
    sys.path.insert(0, _fd_root)


class FontDiffusionGenerator:
    """Wraps FontDiffusion pipeline for batch generation with caching."""

    def __init__(
        self,
        ckpt_dir: str,
        phase1_ckpt_dir: str | None = None,
        font_path: str = "font_diffusion/fonts/NomNaTong-Regular.ttf",
        device: str | None = None,
        cache_dir: str = "prepared/fd_cache",
        batch_size: int = 4,
    ):
        self.ckpt_dir = ckpt_dir
        self.phase1_ckpt_dir = phase1_ckpt_dir
        self.font_path = font_path
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.pipe = None
        self.font_manager = None
        self.args = None
        self._loaded = False

    def _build_args(self) -> Namespace:
        """Build args namespace from FontDiffusion's own parser defaults."""
        from src.configs.fontdiffuser import get_parser

        parser = get_parser()
        args = parser.parse_args([])

        # Override with our paths
        args.ckpt_dir = self.ckpt_dir
        args.phase_1_ckpt_dir = self.phase1_ckpt_dir
        args.fst_ckpt_path = self.ckpt_dir
        args.ttf_path = self.font_path
        args.device = self.device
        args.use_fst = True
        args.batch_size = self.batch_size
        args.character_input = True
        args.save_image = False
        args.style_image_size = (args.style_image_size, args.style_image_size)
        args.content_image_size = (args.content_image_size, args.content_image_size)

        return args

    def _load_pipeline(self):
        """Load FontDiffusion pipeline (once)."""
        if self._loaded:
            return

        print(f"  Loading FontDiffusion pipeline on {self.device}...", flush=True)
        start = time.time()

        self.args = self._build_args()

        from inference.sample_optimized import (
            load_fontdiffuser_pipeline,
            FontManager,
        )

        self.pipe = load_fontdiffuser_pipeline(
            args=self.args, use_fst=True,
        )
        self.font_manager = FontManager(self.font_path)
        self._loaded = True

        elapsed = time.time() - start
        print(f"  FontDiffusion loaded in {elapsed:.1f}s")

    def _get_cache_path(self, char: str, style_name: str) -> Path:
        """Get cache file path for a character + style combination."""
        return self.cache_dir / style_name / f"U+{ord(char):04X}.png"

    def generate(
        self,
        characters: list[str],
        style_image_path: str,
        style_name: str = "default",
    ) -> dict[str, str]:
        """Generate handwritten-style images for characters.

        Args:
            characters: List of Unicode characters to generate.
            style_image_path: Path to style reference image (crop from book).
            style_name: Name for cache subdirectory.

        Returns:
            {char: image_path} for successfully generated characters.
        """
        results: dict[str, str] = {}

        # Check cache first (before loading model)
        to_generate = []
        for char in characters:
            cache_path = self._get_cache_path(char, style_name)
            if cache_path.exists():
                results[char] = str(cache_path)
            else:
                to_generate.append(char)

        if not to_generate:
            return results

        # Only load model when we actually need to generate
        self._load_pipeline()

        # Filter to chars available in font
        font_name = self.font_manager.get_font_names()[0]
        available = self.font_manager.get_available_chars_for_font(
            font_name, to_generate,
        )

        if not available:
            return results

        # Import processing functions
        from inference.sample_optimized import (
            image_process_batch,
            get_style_transform,
            get_content_transform,
        )
        from src.tools.utils import ttf2im

        # Prepare transforms
        style_transform = get_style_transform(self.args.style_image_size)
        content_transform = get_content_transform(self.args.content_image_size)

        # Load style image once
        style_image = Image.open(style_image_path).convert("RGB")
        style_tensor = style_transform(style_image)

        # Get font
        font = self.font_manager.get_font(font_name)

        # Render content images
        content_tensors = []
        valid_chars = []
        for char in available:
            try:
                content_img = ttf2im(font=font, char=char)
                if content_img is not None:
                    content_tensors.append(content_transform(content_img))
                    valid_chars.append(char)
            except Exception:
                continue

        if not content_tensors:
            return results

        # Process in small batches: generate → save → free memory
        cache_style_dir = self.cache_dir / style_name
        cache_style_dir.mkdir(parents=True, exist_ok=True)

        total = len(valid_chars)
        n_batches = (total + self.batch_size - 1) // self.batch_size
        generated = 0
        start = time.time()

        print(f"    FontDiffusion: generating {total} images "
              f"({n_batches} batches of {self.batch_size})...", flush=True)

        dtype = torch.float32

        for i in range(0, total, self.batch_size):
            batch_chars = valid_chars[i:i + self.batch_size]
            batch_content = torch.stack(content_tensors[i:i + self.batch_size])
            batch_style = style_tensor[None, :].repeat(len(batch_chars), 1, 1, 1)

            batch_content = batch_content.to(self.device, dtype=dtype)
            batch_style = batch_style.to(self.device, dtype=dtype)

            with torch.inference_mode():
                images = self.pipe.generate(
                    content_images=batch_content,
                    style_images=batch_style,
                    batch_size=len(batch_content),
                    order=self.args.order,
                    num_inference_step=self.args.num_inference_steps,
                    content_encoder_downsample_size=self.args.content_encoder_downsample_size,
                    t_start=self.args.t_start,
                    t_end=self.args.t_end,
                    dm_size=self.args.content_image_size,
                    algorithm_type=self.args.algorithm_type,
                    skip_type=self.args.skip_type,
                    method=self.args.method,
                    correcting_x0_fn=self.args.correcting_x0_fn,
                )

            # Save immediately and free memory
            for char, img in zip(batch_chars, images):
                cache_path = self._get_cache_path(char, style_name)
                img.save(str(cache_path))
                results[char] = str(cache_path)

            generated += len(images)
            batch_num = i // self.batch_size + 1
            elapsed = time.time() - start
            rate = elapsed / generated
            remaining = rate * (total - generated)
            print(f"      [{batch_num}/{n_batches}] {generated}/{total} done "
                  f"({rate:.1f}s/img, ~{remaining/60:.0f}min left)", flush=True)

            # Free GPU memory
            del batch_content, batch_style, images
            if self.device != "cpu":
                torch.mps.empty_cache() if "mps" in self.device else torch.cuda.empty_cache()

        elapsed = time.time() - start
        print(f"    FontDiffusion: {generated} images in {elapsed:.1f}s "
              f"({elapsed/max(1,generated):.1f}s/img)")

        return results

    def clear_cache(self):
        """Clear disk cache."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
