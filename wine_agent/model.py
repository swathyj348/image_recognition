from __future__ import annotations

import torch
from torch import nn


class WineClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def predict_logits(model: nn.Module, features):
    model.eval()
    with torch.no_grad():
        return model(features)
