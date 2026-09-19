"""HUẤN LUYỆN OCR NÔM NÂNG CAO (KẾT HỢP PHƯƠNG ÁN 1 & PHƯƠNG ÁN 2).
Enhanced Self-Training v3:
- Phương án 1: Two-Stage Model với SE-ResNet trên 54.094 nhãn sạch mở rộng.
- Phương án 2: StrokeEnhancer (tăng cường nét bút lông, mô phỏng mực khô/loang, khử nhiễu nền giấy cổ).

Chạy trên Kaggle GPU (Tesla T4/P100) hoặc Local (MPS/CPU).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# =============================================================================
# PHƯƠNG ÁN 2: MODULE TĂNG CƯỜNG NÉT BÚT LÔNG & KHỬ NHIỄU NỀN GIẤY CỔ
# =============================================================================

class StrokeEnhancer:
    """Module tăng cường nét mực bút lông chép tay thế kỷ 19.

    Mô phỏng các hiệu ứng vật lý của bút lông và giấy cổ:
    1. Chuẩn hoá tương phản nền giấy (giấy điệp/giấy sắc ố vàng).
    2. Nét bút tỳ mạnh / mực đẫm loang (Dilation).
    3. Ngọn bút lướt nhanh / mực khô xước nét (Erosion / Khát bút).
    4. Rung ngọn bút & góc viết tay thể hành-thảo.
    """

    @staticmethod
    def normalize_contrast(t: torch.Tensor) -> torch.Tensor:
        """Cân bằng tương phản thích nghi cho ảnh crop xám."""
        # t shape: (1, H, W), giá trị [0, 1]
        mi = t.min()
        ma = t.max()
        if ma - mi > 1e-4:
            t = (t - mi) / (ma - mi)
        return t

    @staticmethod
    def morph_dilation(t: torch.Tensor) -> torch.Tensor:
        """Mô phỏng mực đẫm / nét đậm bằng Max Pooling."""
        # Chữ đen trên nền trắng -> mực là giá trị thấp, nên đảo ngược để xử lý nét mực
        inv = 1.0 - t
        dilated = F.max_pool2d(inv.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        return 1.0 - dilated

    @staticmethod
    def morph_erosion(t: torch.Tensor) -> torch.Tensor:
        """Mô phỏng ngọn bút lướt nhanh / mực khô (khát bút) bằng Min Pooling."""
        inv = 1.0 - t
        eroded = -F.max_pool2d(-inv.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        return 1.0 - eroded


class EnhancedNomDataset(Dataset):
    """Dataset nạp ảnh crop và áp dụng Calligraphy Data Augmentation."""

    def __init__(
        self,
        images: np.ndarray,
        labels: list[int] | None = None,
        is_train: bool = False,
    ):
        self.images = images  # shape (N, 64, 64), uint8
        self.labels = labels
        self.is_train = is_train

        # Biến dạng hình học ngẫu nhiên thể hiện lối viết hành-thảo
        self.affine_aug = transforms.Compose([
            transforms.RandomRotation(degrees=7, fill=255),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.94, 1.06),
                shear=(-5, 5),
                fill=255,
            ),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        arr = self.images[idx].astype(np.float32) / 255.0  # [0, 1]
        t = torch.from_numpy(arr).unsqueeze(0)  # (1, 64, 64)

        if self.is_train:
            # 1. Cân bằng tương phản thích nghi
            t = StrokeEnhancer.normalize_contrast(t)

            # 2. Chuyển sang PIL-like tensor để xoay/co giãn nét chữ thảo
            t = self.affine_aug(t)

            # 3. Mô phỏng mực loang hoặc mực khô ngẫu nhiên
            p = np.random.rand()
            if p < 0.22:
                t = StrokeEnhancer.morph_dilation(t)
            elif p < 0.44:
                t = StrokeEnhancer.morph_erosion(t)

            # 4. Thêm nhiễu ngẫu nhiên mô phỏng thớ giấy cổ
            if np.random.rand() < 0.30:
                noise = torch.randn_like(t) * 0.02
                t = torch.clamp(t + noise, 0.0, 1.0)
        else:
            t = StrokeEnhancer.normalize_contrast(t)

        # Chuẩn hoá về zero-mean, unit-variance
        t = (t - 0.5) / 0.5

        if self.labels is not None:
            return t, self.labels[idx]
        return t, idx


# =============================================================================
# PHƯƠNG ÁN 1: MÔ HÌNH ENHANCED NOM OCR NET (RESNET + SE-ATTENTION)
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Channel Attention.

    Giúp mô hình tập trung vào các đặc trưng nét chữ Nôm tinh vi
    (bộ thủ, nét chấm, phẩy, móc, tránh nhầm lẫn các dị thể gần hình).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.SiLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        scale = self.fc(x).view(b, c, 1, 1)
        return x * scale


class ResidualBlock(nn.Module):
    """Residual Block với GroupNorm và SE-Attention."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.se = SEBlock(out_c)

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act2(out + res)
        return out


class EnhancedNomOCRNet(nn.Module):
    """Mô hình nhận diện chữ Nôm chép tay chuyên sâu (ResNet-SE Backbone)."""

    def __init__(self, num_classes: int):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
        )

        # 4 Stage Residual Blocks với Downsampling
        self.stage1 = nn.Sequential(
            ResidualBlock(48, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32

        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16

        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        self.stage4 = nn.Sequential(
            ResidualBlock(256, 384, stride=1),
            ResidualBlock(384, 384, stride=1),
        )

        # Global Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(384, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.pool1(self.stage1(out))
        out = self.pool2(self.stage2(out))
        out = self.pool3(self.stage3(out))
        out = self.stage4(out)
        out = self.gap(out).flatten(1)
        out = self.dropout(out)
        return self.fc(out)


# =============================================================================
# HUẤN LUYỆN & SUY DIỄN GIẢI CỨU TẬP DỮ LIỆU
# =============================================================================

def train_and_evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print("=" * 72)
    print("HUẤN LUYỆN ENHANCED NOM OCR NET (PHƯƠNG ÁN 1 + PHƯƠNG ÁN 2)")
    print(f"[Device] Sử dụng: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Local'})")
    print("=" * 72)

    # 1. Nạp dữ liệu
    print("\n1. Nạp dữ liệu...")
    npz = np.load(args.crops)
    X_all = npz["X"]
    df_labels = pd.read_csv(args.labels)
    print(f"  Tổng số ảnh crops: {len(X_all):,}")
    print(f"  Tổng số dòng nhãn: {len(df_labels):,}")

    if len(X_all) != len(df_labels):
        min_len = min(len(X_all), len(df_labels))
        print(f"  [Điều chỉnh] Cắt kích thước khớp nhau: {min_len:,}")
        X_all = X_all[:min_len]
        df_labels = df_labels.iloc[:min_len].copy()

    # Nạp từ điển
    qn_to_nom = {}
    if os.path.exists(args.dict):
        df_dict = pd.read_csv(args.dict, keep_default_na=False, na_values=[""])
        syl_col = df_dict.columns[0]
        nom_col = df_dict.columns[1] if len(df_dict.columns) > 1 else df_dict.columns[0]
        for _, row in df_dict.iterrows():
            s = str(row[syl_col]).strip().lower()
            c = str(row[nom_col]).strip()
            if s and c:
                qn_to_nom.setdefault(s, set()).add(c)
        print(f"  Từ điển nạp thành công: {len(qn_to_nom):,} âm Quốc ngữ.")

    # 2. Lọc tập huấn luyện (is_train_v2 == True)
    train_mask = df_labels["is_train_v2"] == True
    df_train = df_labels[train_mask].copy()
    indices_train = df_train.index.to_numpy()

    # Tạo bảng ánh xạ nhãn
    unique_chars = sorted(df_train["label_v2"].dropna().unique())
    char_to_id = {c: i for i, c in enumerate(unique_chars)}
    id_to_char = {i: c for i, c in enumerate(unique_chars)}
    num_classes = len(unique_chars)
    print(f"\n2. Tập huấn luyện: {len(df_train):,} ảnh | {num_classes:,} lớp chữ Nôm.")

    # Phân chia Train / Val (90% / 10%)
    np.random.seed(42)
    perm = np.random.permutation(len(indices_train))
    val_size = int(0.10 * len(indices_train))
    val_idxs = indices_train[perm[:val_size]]
    tr_idxs = indices_train[perm[val_size:]]

    X_tr = X_all[tr_idxs]
    y_tr = [char_to_id[df_labels.at[i, "label_v2"]] for i in tr_idxs]

    X_val = X_all[val_idxs]
    y_val = [char_to_id[df_labels.at[i, "label_v2"]] for i in val_idxs]

    print(f"  - Train: {len(X_tr):,} ảnh (kèm Stroke Augmentation & SE-Attention)")
    print(f"  - Val:   {len(X_val):,} ảnh đối chuẩn")

    train_ds = EnhancedNomDataset(X_tr, y_tr, is_train=True)
    val_ds = EnhancedNomDataset(X_val, y_val, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # 3. Khởi tạo mô hình
    model = EnhancedNomOCRNet(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Tổng tham số mô hình EnhancedNomOCRNet: {total_params:,}")

    # Loss với Label Smoothing (0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 4. Vòng lặp huấn luyện
    print("\n3. Bắt đầu huấn luyện...")
    metrics_history = []
    best_val_top1 = 0.0
    best_weights_path = Path(args.out_dir) / "best_enhanced_nom_ocr.pt"
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(targets)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # Đánh giá Val Top-1 và Top-5
        model.eval()
        val_loss, val_top1_correct, val_top5_correct, val_total = 0.0, 0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * len(targets)
                val_total += len(targets)

                # Top-1
                preds = outputs.argmax(dim=1)
                val_top1_correct += (preds == targets).sum().item()

                # Top-5
                top5_preds = outputs.topk(min(5, num_classes), dim=1).indices
                val_top5_correct += top5_preds.eq(targets.view(-1, 1)).sum().item()

        val_loss /= val_total
        val_top1 = val_top1_correct / val_total
        val_top5 = val_top5_correct / val_total
        dt = time.time() - t0

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"Val: loss={val_loss:.4f} Top-1={val_top1:.3f} Top-5={val_top5:.3f} | "
            f"{dt:.1f}s"
        )

        metrics_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_top1": round(val_top1, 4),
            "val_top5": round(val_top5, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
        })

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "char_to_id": char_to_id,
                "id_to_char": id_to_char,
                "val_top1": val_top1,
                "val_top5": val_top5,
            }, best_weights_path)

    print(f"\n✓ Đã lưu trọng số tốt nhất (Val Top-1: {best_val_top1:.1%}) tại: {best_weights_path}")

    # Lưu metrics
    metrics_path = Path(args.out_dir) / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)

    # 5. Suy diễn giải cứu các ô chưa được xác nhận (REVIEW & SILVER)
    print("\n4. Suy diễn giải cứu trên các ô REVIEW và SILVER còn lại...")
    # Nạp lại checkpoint tốt nhất
    ckpt = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Tập cần giải cứu: các ô không nằm trong train_mask
    unlabeled_mask = ~train_mask
    df_unlabeled = df_labels[unlabeled_mask].copy()
    unlabeled_indices = df_unlabeled.index.to_numpy()
    print(f"  Tổng số ô suy diễn: {len(df_unlabeled):,} ô.")

    X_infer = X_all[unlabeled_indices]
    infer_ds = EnhancedNomDataset(X_infer, is_train=False)
    infer_loader = DataLoader(infer_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, _ in infer_loader:
            images = images.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)

            top1_probs, top1_idx = probs.max(dim=1)
            all_preds.extend([id_to_char[i.item()] for i in top1_idx])
            all_probs.extend(top1_probs.cpu().numpy().tolist())

    df_unlabeled["pred_char"] = all_preds
    df_unlabeled["pred_prob"] = all_probs

    # Giao thoa an toàn với từ điển
    results = []
    rescue_high, rescue_med, not_dict, low_conf = 0, 0, 0, 0

    for _, r in df_unlabeled.iterrows():
        syl = str(r.get("syllable", "")).strip().lower()
        cand_dict = qn_to_nom.get(syl, set())
        pred_c = r["pred_char"]
        prob = r["pred_prob"]
        in_d = pred_c in cand_dict

        if in_d:
            if prob >= 0.85:
                decision = "RESCUE_V3_HIGH"
                rescue_high += 1
            elif prob >= 0.70:
                decision = "RESCUE_V3_MED"
                rescue_med += 1
            else:
                decision = "RESCUE_V3_LOW_CONF"
                low_conf += 1
        else:
            decision = "NOT_IN_DICT"
            not_dict += 1

        u_code = f"U+{ord(pred_c):04X}" if pred_c else ""
        results.append({
            "book": r.get("book", ""),
            "page": r.get("page", ""),
            "column": r.get("column", ""),
            "nom_idx": r.get("nom_idx", ""),
            "syllable": syl,
            "tier_old": r.get("tier", ""),
            "ocr_char_old": r.get("ocr_char", ""),
            "predicted_nom": pred_c,
            "unicode": u_code,
            "prob": round(prob, 4),
            "in_dict": in_d,
            "decision": decision,
        })

    df_res = pd.DataFrame(results)
    res_csv_path = Path(args.out_dir) / "enhanced_pseudo_labels.csv"
    df_res.to_csv(res_csv_path, index=False)

    print("\n" + "=" * 72)
    print("KẾT QUẢ SUY DIỄN GIẢI CỨU VÒNG 3 (ENHANCED OCR):")
    print(f"  - RESCUE_V3_HIGH (Prob >= 0.85 + Dict): {rescue_high:,} ô")
    print(f"  - RESCUE_V3_MED  (Prob >= 0.70 + Dict): {rescue_med:,} ô")
    print(f"  => TỔNG CỘNG CỨU THÊM: {rescue_high + rescue_med:,} ô!")
    print(f"  - RESCUE_V3_LOW_CONF: {low_conf:,} ô")
    print(f"  - NOT_IN_DICT: {not_dict:,} ô")
    print(f"\n✓ Đã lưu bảng giải cứu: {res_csv_path}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Enhanced Nom OCR Net")
    parser.add_argument("--crops", default="crops.npz", help="Đường dẫn crops.npz")
    parser.add_argument("--labels", default="labels_expanded.csv", help="Đường dẫn labels_expanded.csv")
    parser.add_argument("--dict", default="QuocNgu_SinoNom.csv", help="Đường dẫn từ điển")
    parser.add_argument("--out-dir", default="output", help="Thư mục xuất kết quả")
    parser.add_argument("--epochs", type=int, default=20, help="Số epochs (mặc định 20)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_and_evaluate(args)


if __name__ == "__main__":
    main()
