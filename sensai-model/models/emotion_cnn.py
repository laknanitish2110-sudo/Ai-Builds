"""
SensAI Emotion CNN v2 — residual architecture with CBAM attention.

Input: 48x48 grayscale face crops
Output: 7 emotion probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(x.view(b, c, -1).max(dim=2)[0])
        w = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        w = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_attention: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.channel_attn = ChannelAttention(out_ch) if use_attention else None
        self.spatial_attn = SpatialAttention() if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.channel_attn is not None:
            out = self.channel_attn(out)
        if self.spatial_attn is not None:
            out = self.spatial_attn(out)

        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class SensAIEmotionCNN(nn.Module):
    """
    v2: Residual blocks with CBAM attention for facial emotion recognition.

    Architecture:
        Stem conv (1 -> 64)
        Stage 1: 2x ResBlock 64  (48x48 -> 24x24)
        Stage 2: 2x ResBlock 128 (24x24 -> 12x12) + CBAM
        Stage 3: 2x ResBlock 256 (12x12 -> 6x6)   + CBAM
        Global average pooling
        FC(256, 128) -> FC(128, 7)

    ~2.8M params, ~11MB ONNX. 2.3x bigger than v1.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        dropout: float = 0.5,
        use_attention: bool = True,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, stride=2, use_attention=False),
            ResidualBlock(64, 64, use_attention=False),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, use_attention=use_attention),
            ResidualBlock(128, 128, use_attention=use_attention),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_attention=use_attention),
            ResidualBlock(256, 256, use_attention=use_attention),
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
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
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
