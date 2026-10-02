"""
Dataset loaders for emotion training.

Supports:
  1. FER2013 CSV format (Kaggle)
  2. Image folder format (custom datasets)
  3. Synthetic data for pipeline validation
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class FER2013Dataset(Dataset):
    """
    FER2013 facial expression dataset.

    Download from: https://www.kaggle.com/datasets/msambare/fer2013
    Expected CSV format: emotion,pixels,Usage
    Where pixels is space-separated 48x48 = 2304 values.
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "Training",
        transform: T.Compose | None = None,
    ):
        self.transform = transform
        self.images: list[np.ndarray] = []
        self.labels: list[int] = []

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Usage"] != split:
                    continue
                pixels = np.array(
                    [int(p) for p in row["pixels"].split()], dtype=np.uint8
                ).reshape(48, 48)
                self.images.append(pixels)
                self.labels.append(int(row["emotion"]))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.fromarray(self.images[idx], mode="L")

        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img, self.labels[idx]


class ImageFolderEmotionDataset(Dataset):
    """
    Custom dataset from folder structure:
        data/
        ├── angry/
        │   ├── img001.jpg
        │   └── ...
        ├── happy/
        └── ...
    """

    EMOTION_MAP = {
        "angry": 0,
        "disgusted": 1,
        "fearful": 2,
        "happy": 3,
        "sad": 4,
        "surprised": 5,
        "neutral": 6,
    }

    def __init__(
        self,
        root_dir: str | Path,
        transform: T.Compose | None = None,
    ):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for emotion_name, label in self.EMOTION_MAP.items():
            emotion_dir = self.root / emotion_name
            if not emotion_dir.exists():
                continue
            for img_path in sorted(emotion_dir.glob("*")):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")

        if self.transform:
            img = self.transform(img)
        else:
            img = T.Compose([T.Resize((48, 48)), T.ToTensor()])(img)

        return img, label


class SyntheticEmotionDataset(Dataset):
    """
    Synthetic dataset for pipeline validation.
    Generates random face-like patterns with known emotion labels.
    """

    def __init__(
        self,
        num_samples: int = 5000,
        img_size: int = 48,
        num_classes: int = 7,
        seed: int = 42,
    ):
        self.img_size = img_size
        self.num_classes = num_classes

        rng = np.random.RandomState(seed)

        self.images = []
        self.labels = []

        for i in range(num_samples):
            label = i % num_classes
            img = self._generate_face(label, rng)
            self.images.append(img)
            self.labels.append(label)

    def _generate_face(self, emotion: int, rng: np.random.RandomState) -> np.ndarray:
        """Generate a synthetic face-like pattern encoding the emotion class."""
        s = self.img_size
        img = np.zeros((s, s), dtype=np.float32)

        # base face shape — ellipse
        y, x = np.ogrid[-s // 2 : s // 2, -s // 2 : s // 2]
        face_mask = (x * x) / (s * 0.35) ** 2 + (y * y) / (s * 0.45) ** 2 < 1
        img[face_mask] = 0.6 + rng.uniform(-0.1, 0.1)

        # eyes — position varies by emotion
        eye_y = s // 3
        eye_spread = s // 4
        eye_size = 2 + emotion % 3  # varies per emotion
        for ex in [s // 2 - eye_spread, s // 2 + eye_spread]:
            img[
                eye_y - eye_size : eye_y + eye_size,
                ex - eye_size : ex + eye_size,
            ] = 0.2

        # mouth — shape encodes emotion
        mouth_y = int(s * 0.65)
        mouth_w = s // 4 + emotion * 2
        mouth_h = 1 + emotion % 4
        curve = emotion - 3  # negative = frown, positive = smile
        for mx in range(-mouth_w, mouth_w):
            my = mouth_y + int(curve * (mx / mouth_w) ** 2)
            my = max(0, min(s - 1, my))
            mx_abs = s // 2 + mx
            if 0 <= mx_abs < s:
                for dy in range(mouth_h):
                    row = min(s - 1, my + dy)
                    img[row, mx_abs] = 0.3

        # eyebrows — angle encodes emotion
        brow_y = s // 4
        brow_angle = (emotion - 3) * 0.5
        for bx in range(-8, 8):
            for ex_center in [s // 2 - eye_spread, s // 2 + eye_spread]:
                by = int(brow_y + brow_angle * abs(bx))
                bx_abs = ex_center + bx
                if 0 <= bx_abs < s and 0 <= by < s:
                    img[by, bx_abs] = 0.3

        # add noise
        noise = rng.normal(0, 0.05, (s, s)).astype(np.float32)
        img = np.clip(img + noise, 0, 1)

        return img

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, H, W)
        return img, self.labels[idx]


def get_transforms(
    img_size: int = 48,
    augment: bool = True,
    rotation: int = 15,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> dict[str, T.Compose]:
    train_transforms = [T.Resize((img_size, img_size))]

    if augment:
        train_transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(rotation),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ColorJitter(brightness=brightness, contrast=contrast),
            T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])

    train_transforms.extend([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])

    val_transforms = [
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ]

    return {
        "train": T.Compose(train_transforms),
        "val": T.Compose(val_transforms),
    }
