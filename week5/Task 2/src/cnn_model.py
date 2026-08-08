"""
cnn_model.py

Defines a Convolutional Neural Network for MNIST handwritten-digit image
classification, with alternating convolutional and max-pooling layers
followed by dense (fully connected) layers, using Dropout and BatchNorm
for regularization.
"""

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """
    CNN architecture:

        [Conv(32) -> BN -> ReLU] x2 -> MaxPool -> Dropout
        [Conv(64) -> BN -> ReLU] x2 -> MaxPool -> Dropout
        Flatten -> Dense(128) -> BN -> ReLU -> Dropout -> Dense(num_classes)

    Input: 1x28x28 grayscale images (MNIST size).
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # Block 1: 1 -> 32 channels, 28x28 -> 14x14
        self.block1 = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),        # 28x28 -> 14x14
            nn.Dropout(dropout / 2),
        )

        # Block 2: 32 -> 64 channels, 14x14 -> 7x7
        self.block2 = nn.Sequential(
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),        # 14x14 -> 7x7
            nn.Dropout(dropout / 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),  # raw logits (softmax applied via CrossEntropyLoss)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


def build_model(num_classes: int = 10) -> CNNClassifier:
    """Convenience factory with default settings."""
    return CNNClassifier(num_classes=num_classes, dropout=0.3)
