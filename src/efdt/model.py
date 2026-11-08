"""Compact 1D CNN specified in the EFDT manuscript resource-analysis section."""
from __future__ import annotations

import torch
from torch import nn


class SamePadConv1d(nn.Module):
    """Conv1d with TensorFlow-style SAME padding for arbitrary stride."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = int(x.shape[-1])
        out_length = (length + self.stride - 1) // self.stride
        total_pad = max((out_length - 1) * self.stride + self.kernel_size - length, 0)
        left = total_pad // 2
        right = total_pad - left
        if total_pad:
            x = nn.functional.pad(x, (left, right))
        return self.conv(x)


class CompactECGCNN(nn.Module):
    """14,581-parameter CNN; input is (batch, 12 leads, 1,000 samples)."""

    def __init__(self, num_leads: int = 12, num_classes: int = 5):
        super().__init__()
        self.conv1 = SamePadConv1d(num_leads, 16, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = SamePadConv1d(16, 32, kernel_size=5, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = SamePadConv1d(32, 64, kernel_size=5, stride=2)
        self.activation = nn.ReLU(inplace=False)
        self.classifier = nn.Linear(64, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, leads, samples), received {tuple(x.shape)}")
        if x.shape[1] != 12 or x.shape[2] != 1000:
            raise ValueError(f"Expected 12x1000 ECG input, received {tuple(x.shape[1:])}")
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        return x.mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
