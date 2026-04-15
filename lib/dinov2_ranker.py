"""DINOv2-based character similarity ranker."""

import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as T
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class DINOv2Ranker:
    """Rank candidates using DINOv2 cosine similarity.

    1. Load DINOv2 model (pretrained, no training needed)
    2. Crop image -> embedding
    3. Rendered candidate -> embedding
    4. Cosine similarity -> ranking
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        font_path: str | None = None,
        font_size: int = 180,
        device: str | None = None,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required: pip install torch torchvision")

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.font_size = font_size

        print(f"  Loading DINOv2 ({model_name}) on {device}...", end=" ", flush=True)
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
        self.model.to(self.device)
        self.model.eval()
        print("OK")

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(244, antialias=True),
            T.CenterCrop(224),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.font_path = font_path
        self._pygame_font = None
        self._font_cache = {}
        self._crop_cache = {}
        self._char_cache = {}

    def _init_pygame_font(self):
        if self._pygame_font is not None:
            return
        if not HAS_PYGAME:
            return
        if not pygame.get_init():
            pygame.init()
        if self.font_path and Path(self.font_path).exists():
            self._pygame_font = pygame.font.Font(self.font_path, self.font_size)

    def _render_char(self, char: str) -> Image.Image | None:
        if char in self._font_cache:
            return self._font_cache[char]
        self._init_pygame_font()
        if self._pygame_font is None:
            return None
        try:
            surface = self._pygame_font.render(char, True, (0, 0, 0), (255, 255, 255))
            raw = pygame.image.tobytes(surface, "RGB")
            w, h = surface.get_size()
            img = Image.frombytes("RGB", (w, h), raw)
            max_dim = max(w, h)
            padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            padded.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
            self._font_cache[char] = padded
            return padded
        except Exception:
            self._font_cache[char] = None
            return None

    @torch.no_grad()
    def _embed(self, img: Image.Image) -> torch.Tensor:
        tensor = self.transform(img)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        tensor = tensor.unsqueeze(0).to(self.device)
        emb = self.model(tensor)
        if isinstance(emb, tuple):
            emb = emb[0]
        return F.normalize(emb.squeeze(0), dim=0)

    def _embed_crop(self, crop_path: str) -> torch.Tensor | None:
        if crop_path in self._crop_cache:
            return self._crop_cache[crop_path]
        try:
            img = Image.open(crop_path).convert("RGB")
        except Exception:
            return None
        emb = self._embed(img)
        self._crop_cache[crop_path] = emb
        return emb

    def _embed_char(self, char: str) -> torch.Tensor | None:
        if char in self._char_cache:
            return self._char_cache[char]
        rendered = self._render_char(char)
        if rendered is None:
            return None
        emb = self._embed(rendered)
        self._char_cache[char] = emb
        return emb

    def rank_candidates(
        self, crop_path: str, candidates: list[str],
    ) -> list[tuple[str, float]]:
        """Rank candidates by DINOv2 cosine similarity.

        Returns: [(char, score)] sorted by score descending
        """
        crop_emb = self._embed_crop(crop_path)
        if crop_emb is None:
            return [(c, 0.0) for c in candidates]

        results = []
        for char in candidates:
            char_emb = self._embed_char(char)
            if char_emb is None:
                results.append((char, 0.0))
                continue
            sim = float(F.cosine_similarity(crop_emb.unsqueeze(0), char_emb.unsqueeze(0)))
            sim = max(0.0, (sim + 1.0) / 2.0)
            results.append((char, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def rank_candidates_from_paths(
        self, crop_path: str, candidate_images: dict[str, str],
    ) -> list[tuple[str, float]]:
        """Rank candidates using pre-generated images (e.g. FontDiffusion output).

        Args:
            crop_path: Path to the handwritten crop image.
            candidate_images: {char: image_path} — generated handwritten-style images.

        Returns: [(char, score)] sorted by score descending.
        """
        crop_emb = self._embed_crop(crop_path)
        if crop_emb is None:
            return [(c, 0.0) for c in candidate_images]

        results = []
        for char, img_path in candidate_images.items():
            # Embed the FontDiffusion-generated image (not font-rendered)
            fd_emb = self._embed_crop(img_path)
            if fd_emb is None:
                results.append((char, 0.0))
                continue
            sim = float(F.cosine_similarity(crop_emb.unsqueeze(0), fd_emb.unsqueeze(0)))
            sim = max(0.0, (sim + 1.0) / 2.0)
            results.append((char, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def clear_cache(self):
        self._crop_cache.clear()
