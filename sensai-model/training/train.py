"""
SensAI training pipeline.

Usage:
    python -m sensai-model.training.train              # synthetic data (validation)
    python -m sensai-model.training.train --data fer2013 --csv path/to/fer2013.csv
    python -m sensai-model.training.train --data folder --dir path/to/images/
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from sensai_model.models import SensAIEmotionCNN, MobileEmotionNet
from sensai_model.training.config import TrainConfig
from sensai_model.training.dataset import (
    FER2013Dataset,
    ImageFolderEmotionDataset,
    SyntheticEmotionDataset,
    get_transforms,
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict:
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
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
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

    per_class_acc = {}
    for cls in range(max(all_labels) + 1):
        cls_mask = [i for i, l in enumerate(all_labels) if l == cls]
        if cls_mask:
            cls_correct = sum(1 for i in cls_mask if all_preds[i] == all_labels[i])
            per_class_acc[cls] = cls_correct / len(cls_mask)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "per_class_accuracy": per_class_acc,
    }


def train(config: TrainConfig, data_source: str = "synthetic", **kwargs):
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transforms = get_transforms(
        img_size=config.img_size,
        augment=config.augment,
        rotation=config.rotation_range,
        brightness=config.brightness_range,
        contrast=config.contrast_range,
    )

    # load dataset
    if data_source == "synthetic":
        print("Using synthetic dataset for pipeline validation...")
        full_dataset = SyntheticEmotionDataset(
            num_samples=5000,
            img_size=config.img_size,
            num_classes=config.num_classes,
        )
    elif data_source == "fer2013":
        csv_path = kwargs.get("csv_path")
        if not csv_path:
            raise ValueError("--csv required for fer2013 data source")
        print(f"Loading FER2013 from {csv_path}...")
        full_dataset = FER2013Dataset(
            csv_path, split="Training", transform=transforms["train"]
        )
    elif data_source == "folder":
        data_dir = kwargs.get("data_dir")
        if not data_dir:
            raise ValueError("--dir required for folder data source")
        print(f"Loading images from {data_dir}...")
        full_dataset = ImageFolderEmotionDataset(
            data_dir, transform=transforms["train"]
        )
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    # split
    n = len(full_dataset)
    n_val = int(n * 0.15)
    n_train = n - n_val
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Train: {n_train} samples, Val: {n_val} samples")

    # model
    if config.model_type == "mobile_emotion":
        model = MobileEmotionNet(
            num_classes=config.num_classes,
            in_channels=config.in_channels,
            dropout=config.dropout,
        )
    else:
        model = SensAIEmotionCNN(
            num_classes=config.num_classes,
            in_channels=config.in_channels,
            dropout=config.dropout,
            use_attention=config.use_attention,
        )

    model = model.to(device)
    params = model.get_param_count()
    print(f"Model: {config.model_type}")
    print(f"Parameters: {params['trainable']:,} trainable / {params['total']:,} total")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # training loop
    best_val_acc = 0.0
    patience_counter = 0
    history = []

    print(f"\nStarting training for {config.epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, config.epochs + 1):
        start = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "lr": lr,
            "time": elapsed,
        }
        history.append(record)

        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s"
        )

        # save best
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "config": {
                    "model_type": config.model_type,
                    "num_classes": config.num_classes,
                    "in_channels": config.in_channels,
                    "dropout": config.dropout,
                    "use_attention": config.use_attention,
                    "img_size": config.img_size,
                },
            }
            save_path = config.output_dir / "best_model.pt"
            torch.save(checkpoint, save_path)
            print(f"  → Saved best model (val_acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={config.patience})")
                break

    print("-" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # save final + history
    torch.save(model.state_dict(), config.output_dir / "final_model.pt")

    with open(config.output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # per-class evaluation
    print("\nPer-class accuracy:")
    final_eval = evaluate(model, val_loader, criterion, device)
    for cls, acc in sorted(final_eval["per_class_accuracy"].items()):
        label = config.emotion_labels[cls] if cls < len(config.emotion_labels) else str(cls)
        print(f"  {label:12s}: {acc:.4f}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train SensAI emotion model")
    parser.add_argument(
        "--data",
        choices=["synthetic", "fer2013", "folder"],
        default="synthetic",
    )
    parser.add_argument("--csv", type=str, help="Path to FER2013 CSV")
    parser.add_argument("--dir", type=str, help="Path to image folder dataset")
    parser.add_argument(
        "--model",
        choices=["sensai_cnn", "mobile_emotion"],
        default="sensai_cnn",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    config = TrainConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    train(
        config,
        data_source=args.data,
        csv_path=args.csv,
        data_dir=args.dir,
    )


if __name__ == "__main__":
    main()
