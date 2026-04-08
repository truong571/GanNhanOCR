#!/usr/bin/env python3
"""
dinov2_ranker.py - DINOv2-based character similarity ranker

Sử dụng DINOv2 (Foundation Vision Model, pretrained trên 142M ảnh)
để so sánh hình dạng giữa crop viết tay và font-rendered candidates.

Thay thế template matching + IoU bằng semantic embedding → chính xác hơn
với chữ viết tay biến thể.

Tích hợp vào label_characters.py qua tham số embed_ranker.
"""

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
    """Rank candidates bằng DINOv2 cosine similarity.

    Quy trình:
    1. Load DINOv2 model (pretrained, KHÔNG cần training)
    2. Mỗi crop viết tay → embedding 384-d (hoặc 768-d)
    3. Mỗi candidate rendered → embedding 384-d
    4. Cosine similarity → ranking

    Integration:
        ranker = DINOv2Ranker()
        results = ranker.rank_candidates(crop_path, ["候", "俟", "侯"])
        # → [("候", 0.92), ("侯", 0.85), ("俟", 0.78)]
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        font_path: str | None = None,
        font_size: int = 180,
        device: str | None = None,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required. Install: pip install torch torchvision")

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model_name = model_name
        self.font_size = font_size

        # Load DINOv2 model
        print(f"  Loading DINOv2 ({model_name}) on {device}...", end=" ", flush=True)
        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, verbose=False
        )
        self.model.to(self.device)
        self.model.eval()
        print("OK")

        # Transform: DINOv2 expects 224x224, normalized
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(244, antialias=True),
            T.CenterCrop(224),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Font for rendering candidates
        self.font_path = font_path
        self._pygame_font = None
        self._font_cache = {}  # char → rendered PIL Image

        # Embedding cache
        self._crop_cache = {}   # crop_path → embedding tensor
        self._char_cache = {}   # char → embedding tensor

    def _init_pygame_font(self):
        """Lazy init pygame font."""
        if self._pygame_font is not None:
            return
        if not HAS_PYGAME:
            return
        if not pygame.get_init():
            pygame.init()
        if self.font_path and Path(self.font_path).exists():
            self._pygame_font = pygame.font.Font(self.font_path, self.font_size)

    def _render_char(self, char: str) -> Image.Image | None:
        """Render 1 ký tự Nôm thành ảnh PIL."""
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

            # Pad to square
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
        """Compute DINOv2 embedding for a PIL Image."""
        tensor = self.transform(img)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        tensor = tensor.unsqueeze(0).to(self.device)
        emb = self.model(tensor)
        if isinstance(emb, tuple):
            emb = emb[0]
        return F.normalize(emb.squeeze(0), dim=0)

    def _embed_crop(self, crop_path: str) -> torch.Tensor | None:
        """Embed a crop image file."""
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
        """Embed a rendered character."""
        if char in self._char_cache:
            return self._char_cache[char]

        rendered = self._render_char(char)
        if rendered is None:
            return None

        emb = self._embed(rendered)
        self._char_cache[char] = emb
        return emb

    def rank_candidates(
        self, crop_path: str, candidates: list[str]
    ) -> list[tuple[str, float]]:
        """Rank candidates bằng DINOv2 cosine similarity.

        Args:
            crop_path: Đường dẫn ảnh crop viết tay
            candidates: Danh sách ký tự Nôm candidates

        Returns:
            [(char, similarity_score)] sorted by score descending
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

            # Cosine similarity
            sim = float(F.cosine_similarity(
                crop_emb.unsqueeze(0), char_emb.unsqueeze(0)
            ))
            # Normalize to 0-1 range (cosine can be -1 to 1)
            sim = max(0.0, (sim + 1.0) / 2.0)
            results.append((char, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def clear_cache(self):
        """Clear embedding caches (useful between pages)."""
        self._crop_cache.clear()
        # Keep char_cache — font renderings don't change
