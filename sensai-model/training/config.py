from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: Path = Path("sensai-model/data")
    output_dir: Path = Path("sensai-model/checkpoints")
    model_type: str = "sensai_cnn"  # "sensai_cnn" or "mobile_emotion"

    num_classes: int = 7
    emotion_labels: list[str] = field(
        default_factory=lambda: [
            "angry",
            "disgusted",
            "fearful",
            "happy",
            "sad",
            "surprised",
            "neutral",
        ]
    )

    img_size: int = 48
    in_channels: int = 1  # grayscale

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10  # early stopping

    # augmentation
    augment: bool = True
    rotation_range: int = 15
    horizontal_flip: bool = True
    brightness_range: float = 0.2
    contrast_range: float = 0.2

    # model architecture
    dropout: float = 0.4
    use_attention: bool = True

    seed: int = 42
    num_workers: int = 2

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
