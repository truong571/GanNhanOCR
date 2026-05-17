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
        model_name: str = "dinov2_vitb14_reg",
        font_path: str | None = None,
        font_size: int = 180,
        device: str | None = None,
        embedding_cache_dir: str | None = None,
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
        self.model_name = model_name

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
        self._font_cache: dict = {}
        self._crop_cache: dict = {}
        self._char_cache: dict = {}

        # Optional persistent on-disk embedding cache. Re-runs of step 3 load
        # embeddings from disk instead of re-running the model on every crop +
        # every fd_cache image (~50k + 22k embeddings, very slow on CPU).
        self._embedding_cache_dir: Path | None = (
            Path(embedding_cache_dir) if embedding_cache_dir else None
        )
        self._crop_cache_dirty = False
        if self._embedding_cache_dir is not None:
            self._load_persistent_cache()

    def _persistent_cache_path(self) -> Path:
        # Single .npz keyed by file path (str). Per-model isolation.
        return self._embedding_cache_dir / f"crop_emb_{self.model_name}.npz"

    def _load_persistent_cache(self) -> None:
        f = self._persistent_cache_path()
        if not f.exists():
            return
        try:
            data = np.load(str(f), allow_pickle=False)
            for k in data.files:
                arr = data[k]
                self._crop_cache[k] = torch.from_numpy(arr).to(self.device)
            print(f"  [DINOv2 cache] loaded {len(self._crop_cache):,} embeddings "
                  f"from {f.name}")
        except Exception as e:
            print(f"  [DINOv2 cache] failed to load {f.name}: {e}")

    def save_cache(self) -> None:
        """Persist accumulated embeddings to disk. Caller invokes after step 3."""
        if self._embedding_cache_dir is None or not self._crop_cache_dirty:
            return
        f = self._persistent_cache_path()
        f.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v.detach().cpu().numpy() for k, v in self._crop_cache.items()}
        np.savez(str(f), **arrays)
        self._crop_cache_dirty = False
        print(f"  [DINOv2 cache] saved {len(arrays):,} embeddings to {f.name}")

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

        # Primary path: pygame (faster) when available
        self._init_pygame_font()
        if self._pygame_font is not None:
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
                pass

        # Fallback: PIL (no system SDL needed)
        if not self.font_path or not Path(self.font_path).exists():
            self._font_cache[char] = None
            return None
        try:
            from PIL import ImageDraw, ImageFont
            font = ImageFont.truetype(self.font_path, self.font_size)
            tmp = Image.new("RGB", (self.font_size * 2, self.font_size * 2), (255, 255, 255))
            draw = ImageDraw.Draw(tmp)
            bbox = draw.textbbox((0, 0), char, font=font)
            w = max(1, bbox[2] - bbox[0])
            h = max(1, bbox[3] - bbox[1])
            max_dim = max(w, h)
            padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            d = ImageDraw.Draw(padded)
            d.text(((max_dim - w) // 2 - bbox[0], (max_dim - h) // 2 - bbox[1]),
                   char, fill=(0, 0, 0), font=font)
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
        self._crop_cache_dirty = True
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
