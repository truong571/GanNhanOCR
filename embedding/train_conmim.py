#!/usr/bin/env python3
"""
train_conmim.py - ConMIM self-supervised pre-training cho ký tự Hán Nôm

Kế thừa từ:
  - TencentARC/ConMIM (Masked Image Modeling with Denoising Contrast)
  - notebook version-3 (ViT-S + ConMIM cho SinoNom)

Giai đoạn:
  Phase 1: ConMIM pre-training (self-supervised, không cần nhãn)
           → Học biểu diễn hình dạng ký tự từ toàn bộ ảnh Nôm
  Phase 2: Fine-tune với SupConLoss (supervised, cần nhãn)
           → Phân biệt giữa các class ký tự

Usage:
  # Phase 1: ConMIM pre-training
  python embedding/train_conmim.py --phase pretrain \\
      --data-dir embedding/data/sinonom_dataset \\
      --output-dir embedding/checkpoints/conmim

  # Phase 2: Fine-tune với SupConLoss
  python embedding/train_conmim.py --phase finetune \\
      --data-dir embedding/data/sinonom_dataset \\
      --pretrained embedding/checkpoints/conmim/pretrain_best.pt \\
      --manifest embedding/data/manifest.csv \\
      --output-dir embedding/checkpoints/conmim

  # Lưu ý: Cần clone ConMIM trước:
  # git clone https://github.com/TencentARC/ConMIM.git embedding/ConMIM
"""

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# ViT-Small cho ảnh Nôm (tùy chỉnh từ notebook version-3)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Chia ảnh thành patches và embed."""
    def __init__(self, img_size=96, patch_size=8, in_channels=1, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSmall(nn.Module):
    """Vision Transformer Small cho ký tự Nôm.

    Config (từ notebook version-3):
      patch_size=8, img_size=96, embed_dim=384, depth=12, num_heads=6
      → 144 patches (96/8 = 12, 12×12 = 144)
    """

    def __init__(
        self,
        img_size=96,
        patch_size=8,
        in_channels=1,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        drop_rate=0.,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, return_all_tokens=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if return_all_tokens:
            return x
        return x[:, 0]  # CLS token


# ---------------------------------------------------------------------------
# ConMIM Pre-training Head
# ---------------------------------------------------------------------------

class ConMIMHead(nn.Module):
    """Head cho ConMIM pre-training: dự đoán patch bị mask."""

    def __init__(self, embed_dim=384, patch_dim=192):
        super().__init__()
        # patch_dim = patch_size * patch_size * in_channels = 8*8*1 = 64
        # Nhưng target là feature từ momentum encoder, nên dùng embed_dim
        self.head = nn.Linear(embed_dim, patch_dim)

    def forward(self, x):
        return self.head(x)


class ConMIMModel(nn.Module):
    """ConMIM: Masked Image Modeling with Denoising Contrast.

    Simplified version cho ký tự Nôm:
    1. Mask ngẫu nhiên 60% patches
    2. Encode ảnh đã mask
    3. Dự đoán pixel patches bị mask
    """

    def __init__(self, img_size=96, patch_size=8, embed_dim=384, depth=12, mask_ratio=0.6):
        super().__init__()
        self.encoder = ViTSmall(
            img_size=img_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth,
        )
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * 1  # grayscale
        self.decoder_head = nn.Linear(embed_dim, patch_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, imgs):
        """Chia ảnh thành patches."""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = H // p
        w = W // p
        patches = imgs.reshape(B, C, h, p, w, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return patches

    def random_masking(self, x, mask_ratio):
        """Mask ngẫu nhiên patches."""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep unmasked patches
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_keep, 0)  # 0 = keep, 1 = mask

        return mask, ids_restore

    def forward(self, x):
        B = x.shape[0]
        target = self.patchify(x)  # (B, N, patch_dim)

        # Encode patches
        patch_embeds = self.encoder.patch_embed(x)  # (B, N, D)

        # Random masking
        mask, ids_restore = self.random_masking(patch_embeds, self.mask_ratio)

        # Replace masked patches with mask token
        mask_tokens = self.mask_token.expand(B, self.num_patches, -1)
        w = mask.unsqueeze(-1)
        patch_embeds = patch_embeds * (1 - w) + mask_tokens * w

        # Add CLS + position embedding
        cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
        x_full = torch.cat([cls_tokens, patch_embeds], dim=1)
        x_full = x_full + self.encoder.pos_embed
        x_full = self.encoder.pos_drop(x_full)

        # Transformer blocks
        for blk in self.encoder.blocks:
            x_full = blk(x_full)
        x_full = self.encoder.norm(x_full)

        # Decode masked patches
        patch_preds = self.decoder_head(x_full[:, 1:])  # Skip CLS, (B, N, patch_dim)

        # Loss: MSE chỉ trên masked patches
        loss = (patch_preds - target) ** 2
        loss = (loss.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-8)

        return loss


# ---------------------------------------------------------------------------
# Fine-tune model (Phase 2): ViT encoder + Projection head
# ---------------------------------------------------------------------------

class ViTEmbeddingModel(nn.Module):
    """ViT backbone (từ ConMIM) + Projection head cho contrastive learning."""

    def __init__(self, encoder: ViTSmall, embed_dim=256):
        super().__init__()
        self.encoder = encoder
        vit_dim = encoder.embed_dim
        self.projection = nn.Sequential(
            nn.Linear(vit_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        cls_feat = self.encoder(x)  # (B, vit_dim)
        embed = self.projection(cls_feat)  # (B, embed_dim)
        embed = F.normalize(embed, p=2, dim=1)
        return embed


# ---------------------------------------------------------------------------
# Dataset cho pre-training (không cần nhãn)
# ---------------------------------------------------------------------------

class UnlabeledImageDataset(Dataset):
    """Dataset ảnh không nhãn cho ConMIM pre-training."""

    def __init__(self, data_dir: str, transform=None, max_images=500000):
        self.transform = transform
        self.image_paths = []

        data_dir = Path(data_dir)

        # Tìm tất cả ảnh trong cấu trúc ImageFolder
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            self.image_paths.extend(sorted(data_dir.rglob(ext)))

        if len(self.image_paths) > max_images:
            random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:max_images]

        print(f"UnlabeledDataset: {len(self.image_paths)} images từ {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def pretrain(args):
    """Phase 1: ConMIM pre-training."""
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
        transforms.ToTensor(),
    ])

    dataset = UnlabeledImageDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    model = ConMIMModel(
        img_size=args.img_size, patch_size=8,
        embed_dim=384, depth=12, mask_ratio=0.6,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nConMIM Pre-training")
    print(f"  Device: {device}")
    print(f"  Dataset: {len(dataset)} images")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Model: ViT-S (384-d, 12 layers, patch=8)")
    print(f"{'='*60}\n")

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            loss = model(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                avg = total_loss / n_batches
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] loss={avg:.4f}")

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}  lr={lr_now:.6f}")

        # Save
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
        }
        torch.save(ckpt, output_dir / "pretrain_latest.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, output_dir / "pretrain_best.pt")
            print(f"  → Best model saved (loss={avg_loss:.4f})")

    print(f"\nPre-training complete. Best loss={best_loss:.4f}")


def finetune(args):
    """Phase 2: Fine-tune với SupConLoss."""
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-trained encoder
    encoder = ViTSmall(img_size=args.img_size, patch_size=8, embed_dim=384, depth=12)

    if args.pretrained and Path(args.pretrained).exists():
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        print(f"Loaded pre-trained encoder: {args.pretrained}")
    else:
        print("[WARN] Không có pre-trained weights, fine-tune từ đầu")

    # Build fine-tune model
    model = ViTEmbeddingModel(encoder, embed_dim=args.embed_dim).to(device)

    # Import SupConLoss và Dataset từ train_embedding.py
    from embedding.train_embedding import (
        SupConLoss, SinoNomPairDataset, get_train_transform, train_one_epoch, validate,
    )

    criterion = SupConLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Dataset
    transform = get_train_transform(args.img_size)
    dataset = SinoNomPairDataset(args.manifest, transform=transform)

    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"\nConMIM Fine-tuning (SupConLoss)")
    print(f"  Device: {device}")
    print(f"  Dataset: {n_train} train, {n_val} val")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.6f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "embed_dim": args.embed_dim,
            "img_size": args.img_size,
            "best_val_loss": min(best_val_loss, val_loss),
        }
        torch.save(ckpt, output_dir / "finetune_latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, output_dir / "finetune_best.pt")
            # Lưu thêm dạng tương thích với embed_ranker.py
            compat_ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "embed_dim": args.embed_dim,
                "img_size": args.img_size,
                "backbone": "vit_small_conmim",
            }
            torch.save(compat_ckpt, output_dir / "best.pt")
            print(f"  → Best model saved (val_loss={val_loss:.4f})")

    print(f"\nFine-tuning complete. Best val_loss={best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ConMIM pre-training + fine-tuning")

    parser.add_argument("--phase", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--data-dir", type=str, default="embedding/data/sinonom_dataset",
                        help="Thư mục ảnh cho pre-training")
    parser.add_argument("--manifest", type=str, default="embedding/data/manifest.csv",
                        help="Manifest CSV cho fine-tuning")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Pre-trained checkpoint cho fine-tuning")
    parser.add_argument("--output-dir", type=str, default="embedding/checkpoints/conmim")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.phase == "pretrain":
        pretrain(args)
    else:
        finetune(args)


if __name__ == "__main__":
    main()
