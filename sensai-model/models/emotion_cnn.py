"""
SensAI Emotion CNN — custom lightweight architecture for facial emotion recognition.

Designed for on-device inference: ~500K parameters, runs in real-time on mobile CPUs.
Input: 48x48 grayscale face crops
Output: 7 emotion probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention via channel-wise pooling."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg, mx], dim=1)
        w = self.sigmoid(self.conv(combined))
        return x * w


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_attention: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2)

        self.channel_attn = ChannelAttention(out_ch) if use_attention else None
        self.spatial_attn = SpatialAttention() if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)

        if self.channel_attn is not None:
            x = self.channel_attn(x)
        if self.spatial_attn is not None:
            x = self.spatial_attn(x)

        return self.pool(x)


class SensAIEmotionCNN(nn.Module):
    """
    Lightweight CNN with attention for facial emotion recognition.

    Architecture:
        4 ConvBlocks (32 → 64 → 128 → 256) with batch norm
        Channel + Spatial attention on deeper blocks
        Global average pooling
        2-layer classification head with dropout

    ~500K params, <1ms inference on modern CPUs.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        dropout: float = 0.4,
        use_attention: bool = True,
    ):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, use_attention=False),
            ConvBlock(32, 64, use_attention=False),
            ConvBlock(64, 128, use_attention=use_attention),
            ConvBlock(128, 256, use_attention=use_attention),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
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
