#!/usr/bin/env python3
"""
Train SensAI emotion model.

Mode 1 (cloud/demo): Uses procedurally generated faces with emotion features
Mode 2 (local):      Uses real FER2013 from HuggingFace

Usage:
    python train_real.py                                          # mode 1
    python train_real.py --fer2013                                # mode 2 (needs internet)
    python train_real.py --fer2013-csv path/to/fer2013.csv        # mode 2 from CSV
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import torchvision.transforms as T

root = Path(__file__).parent
model_dir = root / "sensai-model"
sys.path.insert(0, str(root))

import types

sensai_pkg = types.ModuleType("sensai_model")
sensai_pkg.__path__ = [str(model_dir)]
sensai_pkg.__package__ = "sensai_model"
sys.modules["sensai_model"] = sensai_pkg

from sensai_model.models import SensAIEmotionCNN

EMOTION_LABELS = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
OUTPUT_DIR = model_dir / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset Definitions ──


class SyntheticFaceDataset(Dataset):
    """
    Procedurally generated face-like images with emotion-specific features.

    Each face is built from scratch using geometric primitives that model
    facial landmarks (eyes, brows, mouth, nose). Emotion transforms change
    the geometry to produce distinguishable patterns the CNN can learn.
    No external download required — runs fully offline.
    """

    def __init__(self, num_samples_per_class=1500, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        rng = np.random.RandomState(42)

        print("  Generating synthetic face dataset...")
        for emotion_idx in range(7):
            for i in range(num_samples_per_class):
                face = self._generate_face(emotion_idx, rng)
                self.images.append(face)
                self.labels.append(emotion_idx)

        print(f"  Generated {len(self.images)} samples ({num_samples_per_class}/class)")

    def _draw_ellipse(self, canvas, cy, cx, ry, rx, value, rng):
        h, w = canvas.shape
        y, x = np.ogrid[0:h, 0:w]
        mask = ((y - cy) ** 2 / max(ry ** 2, 1)) + ((x - cx) ** 2 / max(rx ** 2, 1)) <= 1
        canvas[mask] = np.clip(value + rng.uniform(-5, 5), 0, 255)

    def _draw_line(self, canvas, y1, x1, y2, x2, thickness, value):
        h, w = canvas.shape
        length = max(int(np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)), 1)
        for t in np.linspace(0, 1, length * 2):
            cy = int(y1 + t * (y2 - y1))
            cx = int(x1 + t * (x2 - x1))
            for dy in range(-thickness, thickness + 1):
                for dx in range(-thickness, thickness + 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        canvas[ny, nx] = value

    def _generate_face(self, emotion, rng):
        size = 64
        face = np.full((size, size), 30, dtype=np.float64)  # dark background

        # face oval
        self._draw_ellipse(face, 32, 32, 26, 20, 160 + rng.uniform(-15, 15), rng)

        # per-face variation
        face_bright = rng.uniform(-10, 10)
        eye_y = 24 + rng.randint(-2, 3)
        mouth_y = 44 + rng.randint(-2, 3)

        if emotion == 0:  # angry — V brows, narrow eyes, tight mouth
            brow_angle = rng.uniform(2, 4)
            # left brow (angled down-in)
            self._draw_line(face, int(eye_y - 6 + brow_angle), 16, int(eye_y - 6 - brow_angle), 26, 1, 80)
            # right brow (angled down-in)
            self._draw_line(face, int(eye_y - 6 - brow_angle), 38, int(eye_y - 6 + brow_angle), 48, 1, 80)
            # narrow eyes
            self._draw_ellipse(face, eye_y, 22, 2, 4, 60, rng)
            self._draw_ellipse(face, eye_y, 42, 2, 4, 60, rng)
            # tight mouth
            self._draw_ellipse(face, mouth_y, 32, 1, 7, 100, rng)
            # vertical wrinkle between brows
            self._draw_line(face, eye_y - 8, 32, eye_y - 3, 32, 0, 110)
            face_bright -= 15

        elif emotion == 1:  # disgusted — raised upper lip, scrunched nose
            # flat brows
            self._draw_line(face, eye_y - 6, 16, eye_y - 5, 26, 1, 110)
            self._draw_line(face, eye_y - 5, 38, eye_y - 6, 48, 1, 110)
            # squinted eyes
            self._draw_ellipse(face, eye_y, 22, 2, 5, 70, rng)
            self._draw_ellipse(face, eye_y, 42, 2, 5, 70, rng)
            # wrinkled nose
            self._draw_ellipse(face, 33, 32, 3, 3, 120, rng)
            self._draw_line(face, 31, 29, 33, 29, 0, 100)
            self._draw_line(face, 31, 35, 33, 35, 0, 100)
            # asymmetric raised lip
            self._draw_line(face, mouth_y - 1, 25, mouth_y + 1, 32, 1, 90)
            self._draw_line(face, mouth_y + 1, 32, mouth_y, 39, 1, 90)
            face_bright -= 8

        elif emotion == 2:  # fearful — raised brows, wide eyes, open mouth
            # high raised brows
            self._draw_line(face, eye_y - 9, 16, eye_y - 10, 26, 1, 120)
            self._draw_line(face, eye_y - 10, 38, eye_y - 9, 48, 1, 120)
            # wide open eyes
            self._draw_ellipse(face, eye_y, 22, 5, 5, 220, rng)
            self._draw_ellipse(face, eye_y, 42, 5, 5, 220, rng)
            # pupils
            self._draw_ellipse(face, eye_y, 22, 2, 2, 40, rng)
            self._draw_ellipse(face, eye_y, 42, 2, 2, 40, rng)
            # open mouth
            self._draw_ellipse(face, mouth_y, 32, 4, 5, 80, rng)
            # forehead wrinkles
            for wy in range(eye_y - 14, eye_y - 10, 2):
                self._draw_line(face, wy, 22, wy, 42, 0, 130)

        elif emotion == 3:  # happy — arched brows, crescent eyes, wide smile
            # gently arched brows
            self._draw_line(face, eye_y - 7, 16, eye_y - 8, 22, 1, 130)
            self._draw_line(face, eye_y - 8, 22, eye_y - 7, 28, 1, 130)
            self._draw_line(face, eye_y - 7, 36, eye_y - 8, 42, 1, 130)
            self._draw_line(face, eye_y - 8, 42, eye_y - 7, 48, 1, 130)
            # crescent eyes (squished from smiling)
            self._draw_ellipse(face, eye_y, 22, 2, 5, 200, rng)
            self._draw_ellipse(face, eye_y + 1, 22, 2, 5, 160, rng)  # lower lid pushes up
            self._draw_ellipse(face, eye_y, 42, 2, 5, 200, rng)
            self._draw_ellipse(face, eye_y + 1, 42, 2, 5, 160, rng)
            # wide curved smile
            for dx in range(-9, 10):
                dy = int(abs(dx) * 0.3)
                sx, sy = 32 + dx, mouth_y + dy
                if 0 <= sy < 64 and 0 <= sx < 64:
                    face[sy, sx] = 90
                    if sy + 1 < 64:
                        face[sy + 1, sx] = 90
            # cheek bumps
            self._draw_ellipse(face, 36, 16, 4, 4, 175, rng)
            self._draw_ellipse(face, 36, 48, 4, 4, 175, rng)
            face_bright += 10

        elif emotion == 4:  # sad — inner brow raise, droopy eyes, down mouth
            # inner brow raise
            self._draw_line(face, eye_y - 8, 18, eye_y - 5, 26, 1, 120)
            self._draw_line(face, eye_y - 5, 38, eye_y - 8, 46, 1, 120)
            # droopy eyes
            self._draw_ellipse(face, eye_y, 22, 3, 5, 180, rng)
            self._draw_ellipse(face, eye_y, 42, 3, 5, 180, rng)
            # pupils looking down
            self._draw_ellipse(face, eye_y + 1, 22, 2, 2, 50, rng)
            self._draw_ellipse(face, eye_y + 1, 42, 2, 2, 50, rng)
            # down-turned mouth
            for dx in range(-7, 8):
                dy = int(abs(dx) * 0.35)
                sx, sy = 32 + dx, mouth_y - dy
                if 0 <= sy < 64 and 0 <= sx < 64:
                    face[sy, sx] = 100
                    if sy + 1 < 64:
                        face[sy + 1, sx] = 100
            face_bright -= 12

        elif emotion == 5:  # surprised — high brows, round eyes, O mouth
            # very high arched brows
            self._draw_line(face, eye_y - 11, 16, eye_y - 13, 22, 1, 130)
            self._draw_line(face, eye_y - 13, 22, eye_y - 11, 28, 1, 130)
            self._draw_line(face, eye_y - 11, 36, eye_y - 13, 42, 1, 130)
            self._draw_line(face, eye_y - 13, 42, eye_y - 11, 48, 1, 130)
            # round wide eyes
            self._draw_ellipse(face, eye_y, 22, 5, 5, 220, rng)
            self._draw_ellipse(face, eye_y, 42, 5, 5, 220, rng)
            self._draw_ellipse(face, eye_y, 22, 2, 2, 40, rng)
            self._draw_ellipse(face, eye_y, 42, 2, 2, 40, rng)
            # O-shaped mouth
            self._draw_ellipse(face, mouth_y, 32, 5, 4, 80, rng)
            self._draw_ellipse(face, mouth_y, 32, 3, 2, 50, rng)  # inner dark
            face_bright += 5

        else:  # neutral — relaxed everything
            # flat brows
            self._draw_line(face, eye_y - 6, 17, eye_y - 6, 27, 1, 120)
            self._draw_line(face, eye_y - 6, 37, eye_y - 6, 47, 1, 120)
            # almond eyes
            self._draw_ellipse(face, eye_y, 22, 3, 5, 200, rng)
            self._draw_ellipse(face, eye_y, 42, 3, 5, 200, rng)
            self._draw_ellipse(face, eye_y, 22, 1, 2, 50, rng)
            self._draw_ellipse(face, eye_y, 42, 1, 2, 50, rng)
            # straight mouth
            self._draw_line(face, mouth_y, 25, mouth_y, 39, 1, 110)

        # nose (common)
        self._draw_ellipse(face, 34, 32, 2, 2, 140 + rng.uniform(-5, 5), rng)

        # apply global brightness + noise
        face = face + face_bright + rng.normal(0, 3, face.shape)
        face = np.clip(face, 0, 255).astype(np.uint8)
        return face

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx], mode="L")
        if self.transform:
            img = self.transform(img)
        else:
            img = T.Compose([T.Resize((48, 48)), T.ToTensor()])(img)
        return img, self.labels[idx]


class HuggingFaceEmotionDataset(Dataset):
    """Wraps HuggingFace FER2013 for PyTorch."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"]
        if img.mode != "L":
            img = img.convert("L")
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        return img, item["label"]


class FER2013CSVDataset(Dataset):
    """Load FER2013 from CSV file."""

    def __init__(self, csv_path, split="Training", transform=None):
        import csv as csvmod

        self.transform = transform
        self.images = []
        self.labels = []

        with open(csv_path) as f:
            reader = csvmod.DictReader(f)
            for row in reader:
                if row["Usage"] != split:
                    continue
                pixels = np.array(
                    [int(p) for p in row["pixels"].split()], dtype=np.uint8
                ).reshape(48, 48)
                self.images.append(pixels)
                self.labels.append(int(row["emotion"]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx], mode="L")
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        return img, self.labels[idx]


# ── Training Logic ──


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    per_class = {}
    for cls in range(7):
        mask = [i for i, l in enumerate(all_labels) if l == cls]
        if mask:
            cls_correct = sum(1 for i in mask if all_preds[i] == all_labels[i])
            per_class[cls] = cls_correct / len(mask)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "per_class": per_class,
    }


def run_training(train_loader, val_loader, class_weights, device, epochs=35, patience=12):
    model = SensAIEmotionCNN(
        num_classes=7, in_channels=1, dropout=0.4, use_attention=True
    ).to(device)

    params = model.get_param_count()
    print(f"\nModel: SensAI Emotion CNN")
    print(f"Parameters: {params['trainable']:,} trainable")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_acc = 0.0
    patience_counter = 0
    history = []

    print(f"\nTraining for up to {epochs} epochs...")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": round(train_m["loss"], 4),
            "train_acc": round(train_m["accuracy"], 4),
            "val_loss": round(val_m["loss"], 4),
            "val_acc": round(val_m["accuracy"], 4),
            "lr": round(lr, 6),
            "time": round(elapsed, 1),
        })

        tag = ""
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "config": {
                    "model_type": "sensai_cnn",
                    "num_classes": 7,
                    "in_channels": 1,
                    "dropout": 0.4,
                    "use_attention": True,
                    "img_size": 48,
                },
            }, OUTPUT_DIR / "best_model.pt")
            tag = f" ★ best={best_val_acc:.4f}"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Train {train_m['accuracy']:.4f} | "
            f"Val {val_m['accuracy']:.4f} | "
            f"Loss {val_m['loss']:.4f} | "
            f"LR {lr:.6f} | {elapsed:.0f}s{tag}"
        )

        if patience_counter >= patience:
            print(f"\nEarly stopping (patience={patience})")
            break

    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # reload best and evaluate
    best_ckpt = torch.load(OUTPUT_DIR / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])

    final = evaluate(model, val_loader, criterion, device)
    print(f"\nPer-class accuracy:")
    for cls, acc in sorted(final["per_class"].items()):
        bar = "█" * int(acc * 30)
        print(f"  {EMOTION_LABELS[cls]:12s}: {acc:.4f} {bar}")

    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train SensAI emotion model")
    parser.add_argument("--fer2013", action="store_true", help="Use FER2013 from HuggingFace")
    parser.add_argument("--fer2013-csv", type=str, help="Path to FER2013 CSV file")
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_transform = T.Compose([
        T.Resize((48, 48)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(12),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
        T.RandomErasing(p=0.15, scale=(0.02, 0.08)),
    ])

    val_transform = T.Compose([
        T.Resize((48, 48)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    if args.fer2013:
        # ── Mode 2a: Real FER2013 from HuggingFace ──
        print("\nDownloading FER2013 from HuggingFace...")
        from datasets import load_dataset
        dataset = load_dataset("uoft-cs/fer2013")
        train_ds = HuggingFaceEmotionDataset(dataset["train"], transform=train_transform)
        val_ds = HuggingFaceEmotionDataset(dataset["test"], transform=val_transform)
        labels = [item["label"] for item in dataset["train"]]
        print(f"  Train: {len(train_ds)}, Test: {len(val_ds)}")

    elif args.fer2013_csv:
        # ── Mode 2b: Real FER2013 from CSV ──
        print(f"\nLoading FER2013 from {args.fer2013_csv}...")
        full_train = FER2013CSVDataset(args.fer2013_csv, split="Training", transform=train_transform)
        val_ds = FER2013CSVDataset(args.fer2013_csv, split="PublicTest", transform=val_transform)
        train_ds = full_train
        labels = full_train.labels
        print(f"  Train: {len(train_ds)}, Test: {len(val_ds)}")

    else:
        # ── Mode 1: Procedural face dataset ──
        print("\nBuilding synthetic face dataset with emotion-specific features...")
        full_ds = SyntheticFaceDataset(num_samples_per_class=1200, transform=train_transform)

        n_val = int(len(full_ds) * 0.15)
        n_train = len(full_ds) - n_val
        train_ds, val_ds_raw = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        # re-wrap val with val_transform
        class SubsetWithTransform(Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform
            def __len__(self):
                return len(self.subset)
            def __getitem__(self, idx):
                img_tensor, label = self.subset[idx]
                # undo the train transform, re-apply val transform
                # since the base dataset already returns transformed, we just use as-is
                return img_tensor, label

        val_ds = SubsetWithTransform(val_ds_raw, val_transform)
        labels = full_ds.labels
        print(f"  Train: {n_train}, Val: {n_val}")

    # class weights
    class_counts = torch.tensor([labels.count(c) if isinstance(labels, list) else
                                 sum(1 for l in labels if l == c)
                                 for c in range(7)], dtype=torch.float)
    class_weights = 1.0 / class_counts.clamp(min=1)
    class_weights = class_weights / class_weights.sum() * 7

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    model, best_acc = run_training(train_loader, val_loader, class_weights, device, epochs=args.epochs)

    # ── Export to ONNX ──
    print("\nExporting best model to ONNX...")
    from sensai_model.export.export_onnx import export_to_onnx, verify_onnx

    onnx_path = model_dir / "export" / "sensai_emotion.onnx"
    export_to_onnx(OUTPUT_DIR / "best_model.pt", onnx_path)
    verify_onnx(onnx_path, OUTPUT_DIR / "best_model.pt")

    print("\n✓ Training complete. Model exported to ONNX.")
    print(f"  Checkpoint: {OUTPUT_DIR / 'best_model.pt'}")
    print(f"  ONNX model: {onnx_path}")
    print(f"  Best accuracy: {best_acc:.4f}")
    print("\nTo retrain on real FER2013 data locally:")
    print("  pip install datasets")
    print("  python train_real.py --fer2013")


if __name__ == "__main__":
    main()
