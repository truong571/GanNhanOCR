#!/usr/bin/env python3
"""
iterative_refine.py - Iterative refinement: lặp lại train → label → retrain

Quy trình:
  Vòng 1: Pipeline gốc → labels.csv (có confidence: high/medium/low)
  Vòng 2: Lọc ký tự HIGH → thêm vào training data → retrain embedding → label lại
  Vòng 3: Lặp lại → converge khi không thêm được HIGH nữa

Usage:
  python embedding/iterative_refine.py \\
      --prepared-dir data/prepared/SachThanhTruyen4 \\
      --manifest embedding/data/manifest.csv \\
      --checkpoint embedding/checkpoints/best.pt \\
      --gallery embedding/data/gallery \\
      --max-rounds 3
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def extract_high_confidence_pairs(labels_csv: Path) -> list[dict]:
    """Lọc các cặp (crop_image, ký_tự_Nôm) có confidence = 'high' từ labels.csv."""
    pairs = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("confidence") == "high" and row.get("nom_char"):
                pairs.append({
                    "image_path": row.get("image", ""),
                    "nom_char": row["nom_char"],
                    "unicode_hex": f"{ord(row['nom_char']):04X}",
                })
    return pairs


def augment_manifest(
    original_manifest: Path,
    high_pairs: list[dict],
    gallery_dir: Path,
    output_manifest: Path,
):
    """Thêm ký tự HIGH confidence từ pipeline vào manifest training.

    Mỗi crop ảnh viết tay HIGH → thêm 1 row mới vào manifest,
    với class_id tương ứng.
    """
    # Load manifest gốc
    rows = []
    code_to_id = {}
    max_id = 0

    with open(original_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            code = row["unicode_hex"]
            cid = int(row["class_id"])
            code_to_id[code] = cid
            max_id = max(max_id, cid)

    # Thêm high confidence pairs
    added = 0
    for pair in high_pairs:
        code = pair["unicode_hex"]
        img_path = pair["image_path"]

        if not Path(img_path).exists():
            continue

        # Tìm hoặc tạo class_id
        if code in code_to_id:
            cid = code_to_id[code]
        else:
            max_id += 1
            cid = max_id
            code_to_id[code] = cid

        font_path = gallery_dir / f"{code}.png"
        if not font_path.exists():
            continue

        rows.append({
            "char": pair["nom_char"],
            "unicode_hex": code,
            "class_id": str(cid),
            "scan_path": img_path,
            "font_path": str(font_path),
        })
        added += 1

    # Save augmented manifest
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["char", "unicode_hex", "class_id", "scan_path", "font_path"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Augmented manifest: {len(rows)} rows (+{added} từ pipeline)")
    return added


def run_command(cmd: list[str], desc: str):
    """Chạy command và in output."""
    print(f"\n{'─'*60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─'*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {desc}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Iterative refinement")
    parser.add_argument("--prepared-dir", type=str, required=True,
                        help="Thư mục prepared data (có labeled/)")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Manifest CSV gốc")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint embedding hiện tại")
    parser.add_argument("--gallery", type=str, required=True,
                        help="Gallery directory")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Số vòng lặp tối đa")
    parser.add_argument("--font", type=str, default=None)
    parser.add_argument("--epochs-per-round", type=int, default=10,
                        help="Số epochs training mỗi vòng")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    gallery_dir = Path(args.gallery)
    base_manifest = Path(args.manifest)

    prev_high_count = 0

    for round_num in range(1, args.max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  VÒNG {round_num}/{args.max_rounds}")
        print(f"{'='*60}")

        # --- Step 1: Chạy labeling với checkpoint hiện tại ---
        label_cmd = [
            sys.executable, "label_characters.py", str(prepared_dir),
            "--embedding", args.checkpoint,
            "--gallery", args.gallery,
        ]
        if args.font:
            label_cmd.extend(["--font", args.font])

        print(f"\n  Step 1: Gán nhãn với embedding checkpoint")
        run_command(label_cmd, f"Label round {round_num}")

        # --- Step 2: Đếm HIGH confidence ---
        labels_csv = prepared_dir / "labeled" / "labels.csv"
        if not labels_csv.exists():
            print(f"[ERROR] labels.csv không tồn tại: {labels_csv}")
            break

        high_pairs = extract_high_confidence_pairs(labels_csv)
        high_count = len(high_pairs)

        print(f"\n  Step 2: HIGH confidence = {high_count} ký tự")

        if high_count <= prev_high_count:
            print(f"  Không thêm được HIGH mới ({high_count} ≤ {prev_high_count}), dừng.")
            break

        improvement = high_count - prev_high_count
        print(f"  Cải thiện: +{improvement} HIGH mới")
        prev_high_count = high_count

        if round_num >= args.max_rounds:
            print(f"  Đã đạt max rounds ({args.max_rounds}), dừng.")
            break

        # --- Step 3: Augment manifest với HIGH pairs ---
        round_dir = Path(f"embedding/checkpoints/round_{round_num}")
        round_dir.mkdir(parents=True, exist_ok=True)

        augmented_manifest = round_dir / "manifest_augmented.csv"
        added = augment_manifest(
            base_manifest, high_pairs, gallery_dir, augmented_manifest
        )

        if added == 0:
            print("  Không có pair mới để augment, dừng.")
            break

        # --- Step 4: Retrain embedding ---
        train_cmd = [
            sys.executable, "embedding/train_embedding.py",
            "--manifest", str(augmented_manifest),
            "--output-dir", str(round_dir),
            "--epochs", str(args.epochs_per_round),
            "--resume", args.checkpoint,
            "--device", args.device,
        ]

        print(f"\n  Step 4: Retrain embedding ({args.epochs_per_round} epochs)")
        success = run_command(train_cmd, f"Train round {round_num}")

        if success:
            new_checkpoint = round_dir / "best.pt"
            if new_checkpoint.exists():
                args.checkpoint = str(new_checkpoint)
                print(f"  Checkpoint mới: {new_checkpoint}")
            else:
                print(f"  [WARN] best.pt không tồn tại, dùng latest.pt")
                latest = round_dir / "latest.pt"
                if latest.exists():
                    args.checkpoint = str(latest)

    # --- Kết quả cuối ---
    print(f"\n{'='*60}")
    print(f"  ITERATIVE REFINEMENT HOÀN TẤT")
    print(f"  Vòng lặp: {min(round_num, args.max_rounds)}")
    print(f"  HIGH confidence cuối: {prev_high_count}")
    print(f"  Checkpoint cuối: {args.checkpoint}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
