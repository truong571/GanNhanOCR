#!/usr/bin/env python3
"""
embed_ranker.py - Ranking ứng viên Nôm bằng deep embedding + FAISS

Module này:
  1. Load model embedding đã train (từ train_embedding.py)
  2. Build FAISS index cho gallery ký tự font
  3. Cung cấp hàm query: crop viết tay → top-K ký tự + similarity score

Sử dụng trong label_characters.py để thay thế visual_similarity thô.

Usage (standalone test):
  python embedding/embed_ranker.py --checkpoint embedding/checkpoints/best.pt \\
      --gallery embedding/data/gallery --test-image some_crop.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Lazy imports (FAISS có thể chưa cài)
# ---------------------------------------------------------------------------

def _import_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# EmbedRanker: load model + build index + query
# ---------------------------------------------------------------------------

class EmbedRanker:
    """Ranking ký tự Nôm bằng contrastive embedding.

    Workflow:
      1. __init__: load model, build gallery index
      2. rank_candidates(): crop image + list ứng viên → sorted by similarity
      3. query_topk(): crop image → top-K ký tự gần nhất trong toàn gallery
    """

    def __init__(
        self,
        checkpoint_path: str,
        gallery_dir: str,
        device: str = "cpu",
        img_size: int = 96,
    ):
        """
        Args:
            checkpoint_path: path tới best.pt từ train_embedding.py
            gallery_dir: thư mục chứa ảnh font render ({UNICODE_HEX}.png)
            device: "cpu" hoặc "cuda"
            img_size: kích thước ảnh input (phải khớp với model)
        """
        self.device = torch.device(device)
        self.img_size = img_size

        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Transform (không augment)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        # Build gallery index
        self.gallery_dir = Path(gallery_dir)
        self.gallery_chars = {}  # {unicode_hex: char}
        self.gallery_embeddings = None  # (N, embed_dim)
        self.gallery_codes = []  # [unicode_hex, ...] theo thứ tự index
        self.faiss_index = None

        self._build_gallery_index()

    def _load_model(self, checkpoint_path: str):
        """Load model từ checkpoint."""
        from embedding.train_embedding import EmbeddingModel

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        embed_dim = ckpt.get("embed_dim", 256)

        model = EmbeddingModel(embed_dim=embed_dim, pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        return model

    @torch.no_grad()
    def _encode_image(self, img: Image.Image) -> np.ndarray:
        """Encode 1 ảnh thành embedding vector."""
        tensor = self.transform(img).unsqueeze(0).to(self.device)  # (1, 1, H, W)
        embedding = self.model(tensor)  # (1, embed_dim)
        return embedding.cpu().numpy()[0]

    @torch.no_grad()
    def _encode_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Encode batch ảnh thành embeddings."""
        tensors = torch.stack([self.transform(img) for img in images])
        tensors = tensors.to(self.device)
        embeddings = self.model(tensors)
        return embeddings.cpu().numpy()

    def _build_gallery_index(self):
        """Build FAISS index cho tất cả ảnh trong gallery."""
        faiss = _import_faiss()

        gallery_path = self.gallery_dir
        if not gallery_path.exists():
            print(f"[EmbedRanker] Gallery không tồn tại: {gallery_path}")
            return

        # Load tất cả ảnh gallery
        png_files = sorted(gallery_path.glob("*.png"))
        if not png_files:
            print(f"[EmbedRanker] Gallery trống: {gallery_path}")
            return

        images = []
        codes = []
        for f in png_files:
            code = f.stem.upper()
            try:
                char = chr(int(code, 16))
            except ValueError:
                continue

            img = Image.open(f).convert("L")
            images.append(img)
            codes.append(code)
            self.gallery_chars[code] = char

        if not images:
            return

        # Encode tất cả gallery images
        print(f"[EmbedRanker] Encoding {len(images)} gallery images...")
        batch_size = 256
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            embs = self._encode_batch(batch)
            all_embeddings.append(embs)

        self.gallery_embeddings = np.vstack(all_embeddings).astype(np.float32)
        self.gallery_codes = codes

        # Build FAISS index (cosine similarity = inner product on L2-normalized vectors)
        if faiss is not None:
            dim = self.gallery_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)  # Inner Product
            self.faiss_index.add(self.gallery_embeddings)
            print(f"[EmbedRanker] FAISS index: {self.faiss_index.ntotal} vectors, dim={dim}")
        else:
            print("[EmbedRanker] FAISS không có, dùng numpy cosine similarity (chậm hơn)")

    def _char_to_code(self, char: str) -> str:
        """Chuyển ký tự → unicode hex code."""
        return f"{ord(char):04X}"

    def get_embedding(self, image_path: str) -> np.ndarray | None:
        """Lấy embedding cho 1 ảnh."""
        try:
            img = Image.open(image_path).convert("L")
            return self._encode_image(img)
        except Exception:
            return None

    def get_embedding_from_cv2(self, cv2_img: np.ndarray) -> np.ndarray | None:
        """Lấy embedding từ ảnh OpenCV (grayscale numpy array)."""
        try:
            img = Image.fromarray(cv2_img).convert("L")
            return self._encode_image(img)
        except Exception:
            return None

    def similarity(self, embed_a: np.ndarray, embed_b: np.ndarray) -> float:
        """Cosine similarity giữa 2 embeddings."""
        return float(np.dot(embed_a, embed_b))

    def rank_candidates(
        self,
        crop_path: str,
        candidates: list[str],
    ) -> list[tuple[str, float]]:
        """Xếp hạng ứng viên bằng embedding similarity.

        Args:
            crop_path: đường dẫn ảnh crop viết tay
            candidates: danh sách ký tự Nôm ứng viên (từ từ điển)

        Returns:
            [(char, similarity_score)] sorted by score descending
            score trong [0, 1], 1 = giống nhất
        """
        if self.gallery_embeddings is None:
            return [(c, 0.0) for c in candidates]

        # Encode crop
        crop_embed = self.get_embedding(crop_path)
        if crop_embed is None:
            return [(c, 0.0) for c in candidates]

        # Tính similarity với từng candidate
        scored = []
        for char in candidates:
            code = self._char_to_code(char)

            if code in self.gallery_codes:
                idx = self.gallery_codes.index(code)
                gallery_embed = self.gallery_embeddings[idx]
                sim = self.similarity(crop_embed, gallery_embed)
                # Clamp to [0, 1]
                sim = max(0.0, min(1.0, (sim + 1) / 2))
            else:
                sim = 0.0

            scored.append((char, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def query_topk(self, crop_path: str, k: int = 10) -> list[tuple[str, float]]:
        """Tìm top-K ký tự giống nhất trong toàn bộ gallery.

        Args:
            crop_path: đường dẫn ảnh crop viết tay
            k: số kết quả trả về

        Returns:
            [(char, similarity_score)] top-K, sorted by score descending
        """
        crop_embed = self.get_embedding(crop_path)
        if crop_embed is None:
            return []

        crop_embed = crop_embed.reshape(1, -1).astype(np.float32)

        if self.faiss_index is not None:
            # FAISS search (nhanh)
            scores, indices = self.faiss_index.search(crop_embed, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                code = self.gallery_codes[idx]
                char = self.gallery_chars.get(code, "?")
                sim = max(0.0, min(1.0, (float(score) + 1) / 2))
                results.append((char, sim))
            return results
        else:
            # Numpy fallback (chậm hơn)
            sims = np.dot(self.gallery_embeddings, crop_embed.T).flatten()
            top_idx = np.argsort(sims)[::-1][:k]
            results = []
            for idx in top_idx:
                code = self.gallery_codes[idx]
                char = self.gallery_chars.get(code, "?")
                sim = max(0.0, min(1.0, (float(sims[idx]) + 1) / 2))
                results.append((char, sim))
            return results

    @property
    def is_ready(self) -> bool:
        """Kiểm tra model và gallery đã sẵn sàng."""
        return self.gallery_embeddings is not None and len(self.gallery_codes) > 0


# ---------------------------------------------------------------------------
# Singleton instance (để label_characters.py import)
# ---------------------------------------------------------------------------

_ranker_instance: EmbedRanker | None = None


def get_ranker(
    checkpoint_path: str | None = None,
    gallery_dir: str | None = None,
    device: str = "cpu",
) -> EmbedRanker | None:
    """Lấy hoặc tạo EmbedRanker singleton.

    Returns None nếu checkpoint không tồn tại hoặc model lỗi.
    """
    global _ranker_instance

    if _ranker_instance is not None:
        return _ranker_instance

    if checkpoint_path is None or gallery_dir is None:
        return None

    if not Path(checkpoint_path).exists():
        print(f"[EmbedRanker] Checkpoint không tồn tại: {checkpoint_path}")
        return None

    try:
        _ranker_instance = EmbedRanker(
            checkpoint_path=checkpoint_path,
            gallery_dir=gallery_dir,
            device=device,
        )
        if not _ranker_instance.is_ready:
            print("[EmbedRanker] Gallery trống, tắt embedding ranking")
            _ranker_instance = None
        return _ranker_instance
    except Exception as e:
        print(f"[EmbedRanker] Lỗi khởi tạo: {e}")
        return None


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test embed_ranker")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--gallery", required=True, help="Gallery directory")
    parser.add_argument("--test-image", required=True, help="Test crop image")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    ranker = EmbedRanker(
        checkpoint_path=args.checkpoint,
        gallery_dir=args.gallery,
        device=args.device,
    )

    print(f"\nQuery: {args.test_image}")
    results = ranker.query_topk(args.test_image, k=args.topk)
    print(f"Top-{args.topk} kết quả:")
    for i, (char, score) in enumerate(results):
        code = f"U+{ord(char):04X}"
        print(f"  {i+1}. {char} ({code})  score={score:.4f}")


if __name__ == "__main__":
    main()
