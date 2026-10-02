"""
MobileNet-based emotion model — transfer learning for higher accuracy.

Uses MobileNetV2 backbone pretrained on ImageNet, with custom emotion head.
Higher accuracy than the custom CNN, slightly larger (~2.3M params).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MobileEmotionNet(nn.Module):
    """
    MobileNetV2 adapted for grayscale facial emotion recognition.

    Modifications from standard MobileNetV2:
        - First conv accepts 1-channel (grayscale) input
        - Classification head replaced with emotion-specific layers
        - Backbone can be frozen for faster fine-tuning
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        dropout: float = 0.4,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        backbone = models.mobilenet_v2(weights=None)

        # adapt first conv for grayscale
        original_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            # unfreeze last 3 blocks for fine-tuning
            for param in self.features[-3:].parameters():
                param.requires_grad = True

        feature_dim = backbone.last_channel  # 1280

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def get_param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
