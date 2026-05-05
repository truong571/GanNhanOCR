"""Train Local CNN OCR from labeled dataset (tier 1 matched only).

Usage:
  python scripts/train_local_ocr.py

Input:  dataset/all/labels.csv + crops
Output: models/local_ocr.pth + models/class_map.json

Only uses tier=1 + matched=True samples (dictionary confirmed = reliable labels).
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


class NomCropDataset(Dataset):
    def __init__(self, rows: list[dict], base_dir: Path, class_map: dict, transform=None):
        self.rows = rows
        self.base_dir = base_dir
        self.class_map = class_map
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        crop_path = self.base_dir / row["crop_file"]
        img = Image.open(crop_path).convert("L")
        if self.transform:
            img = self.transform(img)
        label = self.class_map[row["nom_char"]]["class_id"]
        return img, label


def main():
    dataset_dir = Path("dataset/all")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Load labels — only tier 1 matched (reliable)
    labels_csv = dataset_dir / "labels.csv"
    if not labels_csv.exists():
        print("ERROR: Run pipeline first (need dataset/all/labels.csv)")
        sys.exit(1)

    with open(labels_csv) as f:
        all_rows = list(csv.DictReader(f))

    reliable = [r for r in all_rows
                if r.get("tier") == "1"
                and r.get("matched") in ("True", True)
                and r.get("nom_char")]

    print(f"Total rows: {len(all_rows)}")
    print(f"Tier 1 matched (training data): {len(reliable)}")

    if len(reliable) < 50:
        print("Not enough training data. Need more labeled books.")
        sys.exit(1)

    # Build class map from reliable samples
    chars = sorted(set(r["nom_char"] for r in reliable))
    class_map = {
        c: {"class_id": i, "unicode": f"U+{ord(c):04X}"}
        for i, c in enumerate(chars)
    }
    print(f"Classes: {len(class_map)}")

    # Filter rows with valid crops
    valid_rows = []
    for r in reliable:
        if r["nom_char"] in class_map:
            crop_path = dataset_dir / r["crop_file"]
            if crop_path.exists():
                valid_rows.append(r)
    print(f"Valid samples: {len(valid_rows)}")

    # Split train/val
    import random
    random.seed(42)
    random.shuffle(valid_rows)
    split = int(len(valid_rows) * 0.9)
    train_rows = valid_rows[:split]
    val_rows = valid_rows[split:]
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_ds = NomCropDataset(train_rows, dataset_dir, class_map, transform)
    val_ds = NomCropDataset(val_rows, dataset_dir, class_map, val_transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # Model: ResNet18 (1-channel input)
    num_classes = len(class_map)
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train
    epochs = 30
    best_acc = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(imgs)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(imgs)

        train_acc = correct / max(1, total) * 100
        val_acc = val_correct / max(1, val_total) * 100

        print(f"  Epoch {epoch:2d}/{epochs}: "
              f"loss={total_loss/max(1,total):.3f} "
              f"train_acc={train_acc:.1f}% "
              f"val_acc={val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), str(models_dir / "local_ocr.pth"))

    # Save class map
    with open(models_dir / "local_ocr_class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    print(f"\nBest val accuracy: {best_acc:.1f}%")
    print(f"Saved: models/local_ocr.pth + models/local_ocr_class_map.json")
    print(f"\nTo use in pipeline, set in config/pipeline.yaml:")
    print(f"  step3:")
    print(f"    use_local_ocr: true")


if __name__ == "__main__":
    main()
