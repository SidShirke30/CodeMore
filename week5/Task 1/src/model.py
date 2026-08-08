"""
model.py

Defines a configurable Multi-Layer Perceptron (MLP) for binary/multi-class
tabular classification using PyTorch.
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    A feedforward multi-layer perceptron.

    Architecture:
        Input layer -> [Linear -> BatchNorm -> ReLU -> Dropout] x N hidden layers -> Output layer

    Args:
        input_dim (int): Number of input features.
        hidden_dims (list[int]): Sizes of each hidden layer, in order.
        output_dim (int): Number of output units (1 for binary classification
            with BCEWithLogitsLoss, >1 for multi-class with CrossEntropyLoss).
        dropout (float): Dropout probability applied after each hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        # Output layer: raw logits. Sigmoid/Softmax applied outside the model
        # (inside the loss function) for numerical stability.
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) initialization, appropriate for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model(input_dim: int, output_dim: int = 1) -> MLPClassifier:
    """Convenience factory with a sensible default architecture."""
    return MLPClassifier(
        input_dim=input_dim,
        hidden_dims=[64, 32, 16],
        output_dim=output_dim,
        dropout=0.2,
    )
