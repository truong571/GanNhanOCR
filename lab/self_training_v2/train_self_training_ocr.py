"""HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN KÝ TỰ NÔM TRÊN TẬP GOLD & SUY DIỄN GIẢI CỨU TẬP REVIEW.
(Vòng lặp tự gán nhãn thứ hai - Self-training / Pseudo-labeling)

Chạy trên Kaggle GPU (T4/P100) hoặc máy Local (Apple Silicon MPS / CPU).

Đầu vào:
  - crops.npz: Mảng ảnh crop (82.780, 64, 64)
  - labels_final.csv: Bảng nhãn hiện hành
  - QuocNgu_SinoNom.csv: Từ điển đối chiếu ứng viên âm-chữ

Đầu ra:
  - review_pseudo_labels.csv: Bảng kết quả dự đoán và giải cứu cho 12.871 ô REVIEW
  - training_metrics.json: Báo cáo Loss, Top-1, Top-5 accuracy qua từng epoch
  - best_nom_ocr.pt: Trọng số mô hình OCR Nôm tốt nhất
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SZ = 64
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3


# --------------------------------------------------------------------------- Model Architecture
def conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class NomOCRNet(nn.Module):
    """Mô hình CNN nhận diện ký tự chữ Nôm viết tay bút lông."""
    def __init__(self, n_classes: int, emb_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),    # 64 -> 32
            conv_block(32, 64),   # 32 -> 16
            conv_block(64, 128),  # 16 -> 8
            conv_block(128, 256), # 8 -> 4
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.fc(feat)
        return logits


# --------------------------------------------------------------------------- Dataset & Augmentation
class NomDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: list[int] | None = None, augment: bool = False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx].astype(np.float32) / 255.0  # (64, 64)
        
        # Simple fast augmentation on numpy
        if self.augment and random.random() < 0.5:
            # Random slight shift
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            img = np.roll(img, dx, axis=1)
            img = np.roll(img, dy, axis=0)

        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 64, 64)

        if self.labels is not None:
            return tensor, self.labels[idx]
        return tensor


# --------------------------------------------------------------------------- Helper Loader
def load_dictionary(dict_path: str) -> dict[str, set[str]]:
    qn_to_nom = defaultdict(set)
    with open(dict_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            qn = r.get("QuocNgu", "").strip().lower()
            nom = r.get("SinoNom", "").strip()
            if qn and nom:
                qn_to_nom[qn].add(nom)
    return qn_to_nom


# --------------------------------------------------------------------------- Main Training Routine
def train_and_evaluate(args):
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Sử dụng GPU CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Sử dụng Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[Device] Sử dụng CPU")

    # 1. Load dữ liệu
    print("\n1. Nạp dữ liệu...")
    crops_file = np.load(args.crops)
    X_all = crops_file["X"]  # (82780, 64, 64)
    df_labels = pd.read_csv(args.labels)
    qn_to_nom = load_dictionary(args.dict)

    print(f"  Tổng số ảnh: {len(X_all):,}")
    print(f"  Tổng số dòng nhãn: {len(df_labels):,}")
    if len(X_all) != len(df_labels):
        print(f"  ⚠️ Cảnh báo: Số ảnh ({len(X_all):,}) khác số nhãn ({len(df_labels):,}) do lệch thế hệ.")
        min_len = min(len(X_all), len(df_labels))
        print(f"  -> Tự động khớp {min_len:,} dòng đầu tiên để chạy liên tục không bị dừng!")
        X_all = X_all[:min_len]
        df_labels = df_labels.iloc[:min_len].reset_index(drop=True)

    # 2. Tách tập GOLD làm Train/Val
    gold_mask = (df_labels["tier"] == "GOLD") & (df_labels["label"].fillna("") != "")
    gold_indices = np.where(gold_mask)[0]
    gold_chars = df_labels.loc[gold_indices, "label"].values

    # Xây dựng từ điển lớp ký tự Nôm (Vocabulary)
    unique_chars = sorted(list(set(gold_chars)))
    n_classes = len(unique_chars)
    char_to_id = {c: i for i, c in enumerate(unique_chars)}
    id_to_char = {i: c for i, c in enumerate(unique_chars)}
    gold_labels = [char_to_id[c] for c in gold_chars]

    print(f"\n2. Tập GOLD huấn luyện:")
    print(f"  Số mẫu GOLD: {len(gold_indices):,}")
    print(f"  Số lớp ký tự Nôm: {n_classes:,}")

    # Chia Train (90%) và Val (10%)
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(gold_indices))
    val_size = int(len(gold_indices) * 0.10)
    train_split = shuffled_idx[val_size:]
    val_split = shuffled_idx[:val_size]

    train_imgs = X_all[gold_indices[train_split]]
    train_lbls = [gold_labels[i] for i in train_split]
    val_imgs = X_all[gold_indices[val_split]]
    val_lbls = [gold_labels[i] for i in val_split]

    train_ds = NomDataset(train_imgs, train_lbls, augment=True)
    val_ds = NomDataset(val_imgs, val_lbls, augment=False)

    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # 3. Khởi tạo mô hình
    model = NomOCRNet(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\n3. Bắt đầu huấn luyện {EPOCHS} epochs...")
    metrics_log = []
    best_val_acc = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_nom_ocr.pt"

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct_top1 = 0
        total_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(y)
            pred = logits.argmax(dim=1)
            correct_top1 += (pred == y).sum().item()
            total_samples += len(y)

        scheduler.step()
        train_loss = total_loss / total_samples
        train_acc = correct_top1 / total_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_top1 = 0
        val_top5 = 0
        val_samples = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss += loss.item() * len(y)
                # Top 1 & Top 5
                _, top5 = logits.topk(min(5, n_classes), dim=1)
                val_top1 += (top5[:, 0] == y).sum().item()
                val_top5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
                val_samples += len(y)

        val_loss /= val_samples
        val_acc1 = val_top1 / val_samples
        val_acc5 = val_top5 / val_samples

        epoch_log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_top1": round(val_acc1, 4),
            "val_top5": round(val_acc5, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6)
        }
        metrics_log.append(epoch_log)

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.1%} | "
              f"Val Loss: {val_loss:.4f}, Top-1: {val_acc1:.1%}, Top-5: {val_acc5:.1%}")

        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc1": val_acc1,
                "val_acc5": val_acc5,
                "id_to_char": id_to_char,
            }, best_ckpt_path)

    total_time = time.time() - t0
    print(f"\n✅ Hoàn thành huấn luyện trong {total_time/60:.1f} phút. Best Val Top-1: {best_val_acc:.1%}")

    # Lưu metrics log
    with open(out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)

    # 4. Suy diễn trên tập REVIEW
    print("\n4. Suy diễn và Giải cứu tập REVIEW...")
    # Load best checkpoint
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rv_mask = df_labels["tier"] == "REVIEW"
    rv_indices = np.where(rv_mask)[0]
    rv_imgs = X_all[rv_indices]
    rv_ds = NomDataset(rv_imgs, augment=False)
    rv_loader = DataLoader(rv_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    all_preds = []
    with torch.no_grad():
        for x in rv_loader:
            x = x.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                topk_prob, topk_id = probs.topk(min(5, n_classes), dim=1)
            
            topk_prob = topk_prob.cpu().numpy()
            topk_id = topk_id.cpu().numpy()

            for p_arr, id_arr in zip(topk_prob, topk_id):
                all_preds.append([(id_to_char[idx], float(p)) for p, idx in zip(p_arr, id_arr)])

    # 5. Đối chiếu ứng viên từ điển để lọc giải cứu
    rescue_rows = []
    n_rescue_high = 0
    n_rescue_med = 0

    for idx, (orig_idx, cand_list) in enumerate(zip(rv_indices, all_preds)):
        row = df_labels.iloc[orig_idx]
        syl = str(row.get("syllable", "")).strip().lower()
        dict_cands = qn_to_nom.get(syl, set())

        # Tìm ứng viên xác suất cao nhất nằm trong từ điển
        matched_char = ""
        matched_prob = 0.0
        for c, p in cand_list:
            if c in dict_cands:
                matched_char = c
                matched_prob = p
                break

        # Nếu không có ứng viên nào trong từ điển, lấy top 1 thuần túy
        top1_char, top1_prob = cand_list[0]

        if matched_char and matched_prob >= 0.85:
            decision = "RESCUE_GOLD_HIGH"
            n_rescue_high += 1
        elif matched_char and matched_prob >= 0.70:
            decision = "RESCUE_GOLD_MED"
            n_rescue_med += 1
        elif matched_char:
            decision = "RESCUE_LOW_CONF"
        else:
            decision = "NOT_IN_DICT"

        u_hex = f"U+{ord(matched_char):04X}" if matched_char else (f"U+{ord(top1_char):04X}" if top1_char else "")

        rescue_rows.append({
            "book": row.get("book", ""),
            "page": row.get("page", ""),
            "column": row.get("column", ""),
            "nom_idx": row.get("nom_idx", ""),
            "syllable": syl,
            "ocr_char_old": row.get("ocr_char", ""),
            "predicted_nom": matched_char if matched_char else top1_char,
            "unicode": u_hex,
            "prob": round(matched_prob if matched_char else top1_prob, 4),
            "in_dict": bool(matched_char),
            "decision": decision,
            "image": row.get("image", "")
        })

    res_df = pd.DataFrame(rescue_rows)
    out_csv = out_dir / "review_pseudo_labels.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("KẾT QUẢ GIẢI CỨU TẬP REVIEW BẰNG SELF-TRAINING VÒNG 2:")
    print("=" * 60)
    print(f"Tổng số ô REVIEW được phân tích: {len(res_df):,}")
    print(f"  - RESCUE_GOLD_HIGH (P >= 0.85 & Trong Từ Điển):  {n_rescue_high:,} ô ({n_rescue_high/len(res_df):.1%})")
    print(f"  - RESCUE_GOLD_MED  (0.70 <= P < 0.85 & Trong Từ Điển): {n_rescue_med:,} ô ({n_rescue_med/len(res_df):.1%})")
    print(f"  - Tổng số ô có thể giải cứu lên GOLD v2:       {n_rescue_high + n_rescue_med:,} ô ({(n_rescue_high + n_rescue_med)/len(res_df):.1%})")
    print(f"Kết quả chi tiết đã xuất vào: {out_csv}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Nom OCR Self-Training & REVIEW Rescue")
    parser.add_argument("--crops", default="crops.npz", help="Đường dẫn crops.npz")
    parser.add_argument("--labels", default="labels_final.csv", help="Đường dẫn labels_final.csv")
    parser.add_argument("--dict", default="QuocNgu_SinoNom.csv", help="Đường dẫn QuocNgu_SinoNom.csv")
    parser.add_argument("--out-dir", default="self_training_results", help="Thư mục xuất kết quả")
    args = parser.parse_args()

    train_and_evaluate(args)


if __name__ == "__main__":
    main()
