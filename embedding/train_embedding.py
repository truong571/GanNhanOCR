#!/usr/bin/env python3
"""
train_embedding.py - Train contrastive embedding model cho ký tự Hán Nôm

Kế thừa từ:
  - notebook similarity-retrieval (ResNet50 + SupConLoss)
  - notebook version-3 (ConMIM + ViT-S)

Kiến trúc:
  ResNet50 (freeze layer 1-2) → Projection Head (256-d) → SupConLoss
  Positive pairs: ảnh scan + ảnh font render cùng ký tự
  Augmentation: stroke-safe (affine nhẹ, Gaussian blur, color jitter)

Usage:
  # Train từ đầu:
  python embedding/train_embedding.py --manifest embedding/data/manifest.csv

  # Resume training:
  python embedding/train_embedding.py --manifest embedding/data/manifest.csv --resume embedding/checkpoints/latest.pt

  # Train trên Kaggle/Colab (xem notebook):
  # Upload thư mục embedding/ lên, chạy train_embedding.py
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# SupConLoss (Supervised Contrastive Loss)
# Từ: https://arxiv.org/abs/2004.11362
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Kéo embeddings cùng class lại gần, đẩy embeddings khác class ra xa.
    Temperature thấp (0.1) → phân biệt mạnh hơn.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (batch_size, embed_dim) - L2-normalized embeddings
            labels: (batch_size,) - class labels

        Returns:
            scalar loss
        """
        device = features.device
        batch_size = features.shape[0]

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Mask: 1 nếu cùng class, 0 nếu khác class
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)

        # Loại bỏ diagonal (so sánh với chính mình)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Log-softmax (numerical stability)
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Trung bình log-prob cho positive pairs
        pos_count = mask.sum(dim=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

        # Chỉ tính loss cho samples có ít nhất 1 positive pair
        valid = pos_count > 0
        loss = -mean_log_prob[valid].mean()

        return loss


# ---------------------------------------------------------------------------
# Model: ResNet50 + Projection Head
# ---------------------------------------------------------------------------

class EmbeddingModel(nn.Module):
    """ResNet50 backbone + Projection head cho contrastive learning.

    - Input: grayscale 96×96
    - Backbone: ResNet50 (freeze layer 1-2, train layer 3-4)
    - Projection: 2048 → 512 → 256 (MLP + ReLU + L2-norm)
    """

    def __init__(self, embed_dim=256, pretrained=True):
        super().__init__()

        # Backbone: ResNet50
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )

        # Thay conv1: 3 channels → 1 channel (grayscale)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Lấy tất cả layers trừ FC cuối
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,  # Freeze
            resnet.layer2,  # Freeze
            resnet.layer3,  # Train
            resnet.layer4,  # Train
            resnet.avgpool,
        )

        # Freeze layer 1-2 (conv1 + layer1 + layer2)
        for name, param in self.backbone.named_parameters():
            # Freeze layers 0-5 (conv1, bn1, relu, maxpool, layer1, layer2)
            layer_idx = name.split(".")[0]
            if layer_idx.isdigit() and int(layer_idx) <= 5:
                param.requires_grad = False

        # Projection head: 2048 → 512 → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, 96, 96) grayscale images

        Returns:
            (B, embed_dim) L2-normalized embeddings
        """
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)  # (B, 2048)
        embed = self.projection(feat)  # (B, embed_dim)
        embed = F.normalize(embed, p=2, dim=1)  # L2 normalize
        return embed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SinoNomPairDataset(Dataset):
    """Dataset cho contrastive learning.

    Mỗi sample: load ảnh scan HOẶC ảnh font, trả về (image, class_id).
    DataLoader sẽ tạo batch với nhiều class → SupConLoss tự tìm positive/negative.
    """

    def __init__(self, manifest_path: str, transform=None, max_per_class=50):
        """
        Args:
            manifest_path: CSV với columns: char, unicode_hex, class_id, scan_path, font_path
            transform: augmentation pipeline
            max_per_class: giới hạn số samples/class (tránh class imbalance)
        """
        self.transform = transform
        self.samples = []  # [(image_path, class_id, is_font)]

        # Load manifest
        class_counts = {}
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_id = int(row["class_id"])
                scan_path = row["scan_path"]
                font_path = row["font_path"]

                count = class_counts.get(class_id, 0)
                if count >= max_per_class:
                    continue

                # Thêm scan image
                if os.path.exists(scan_path):
                    self.samples.append((scan_path, class_id, False))
                    class_counts[class_id] = count + 1

        # Thêm font images (1 per class)
        font_added = set()
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_id = int(row["class_id"])
                font_path = row["font_path"]
                if class_id not in font_added and os.path.exists(font_path):
                    self.samples.append((font_path, class_id, True))
                    font_added.add(class_id)

        self.n_classes = len(font_added)
        print(f"Dataset: {len(self.samples)} samples, {self.n_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_id, is_font = self.samples[idx]

        img = Image.open(path).convert("L")  # Grayscale

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, class_id


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def get_train_transform(img_size=96):
    """Stroke-safe augmentation cho chữ viết tay.

    Không dùng flip/rotation mạnh (làm biến dạng chữ).
    Chỉ dùng: affine nhẹ, blur, color jitter, noise.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(
            degrees=5,           # Xoay nhẹ ±5°
            translate=(0.05, 0.05),  # Dịch nhẹ 5%
            scale=(0.9, 1.1),    # Scale nhẹ ±10%
            shear=3,             # Nghiêng nhẹ ±3°
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ], p=0.3),
        transforms.ToTensor(),
        # Random noise
        transforms.RandomApply([
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02),
        ], p=0.2),
        transforms.Lambda(lambda x: x.clamp(0, 1)),
    ])


def get_val_transform(img_size=96):
    """Transform cho validation (không augment)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        loss = criterion(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] loss={avg_loss:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        loss = criterion(embeddings, labels)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    manifest_path: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-4,
    embed_dim: int = 256,
    img_size: int = 96,
    temperature: float = 0.1,
    resume: str | None = None,
    device: str = "cuda",
):
    """Main training function."""
    set_seed(42)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    train_transform = get_train_transform(img_size)
    dataset = SinoNomPairDataset(manifest_path, transform=train_transform)

    # Split train/val (90/10)
    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"Train: {n_train}, Val: {n_val}")

    # Model
    model = EmbeddingModel(embed_dim=embed_dim).to(device)
    criterion = SupConLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # Training
    print(f"\n{'='*60}")
    print(f"Training: {epochs} epochs, batch={batch_size}, lr={lr}")
    print(f"Model: ResNet50 + ProjectionHead({embed_dim}-d)")
    print(f"Loss: SupConLoss(temperature={temperature})")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr_now:.6f}")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": min(best_val_loss, val_loss),
            "embed_dim": embed_dim,
            "img_size": img_size,
        }

        torch.save(ckpt, output_dir / "latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, output_dir / "best.pt")
            print(f"  → Best model saved (val_loss={val_loss:.4f})")

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, output_dir / f"epoch_{epoch:03d}.pt")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoints: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train contrastive embedding model cho ký tự Hán Nôm",
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path tới manifest.csv (từ prepare_data.py)")
    parser.add_argument("--output-dir", type=str, default="embedding/checkpoints",
                        help="Thư mục lưu checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path checkpoint để resume training")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    train(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        img_size=args.img_size,
        temperature=args.temperature,
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
