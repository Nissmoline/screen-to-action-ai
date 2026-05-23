"""CNN policy network for supervised desktop imitation learning."""

from __future__ import annotations

import torch
from torch import nn


class PolicyCNN(nn.Module):
    """Predict profile action logits from a captured screen image."""

    def __init__(self, num_actions: int, input_channels: int = 3) -> None:
        super().__init__()
        if num_actions <= 0:
            raise ValueError("num_actions must be greater than 0")

        self.num_actions = num_actions
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_actions),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return action logits for a batch of images."""
        return self.classifier(self.features(images))
