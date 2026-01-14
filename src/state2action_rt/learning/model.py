from __future__ import annotations

import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(
        self,
        num_actions: int,
        num_grids: int,
        embedding_dim: int = 256,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if num_actions <= 0 or num_grids <= 0:
            raise ValueError("num_actions and num_grids must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.num_actions = num_actions
        self.num_grids = num_grids
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(256, embedding_dim)
        self.card_head = nn.Linear(embedding_dim, num_actions)
        self.grid_head = nn.Linear(embedding_dim, num_grids)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        pooled = self.pool(feats).flatten(1)
        emb = self.embedding(pooled)
        card_logits = self.card_head(emb)
        grid_logits = self.grid_head(emb)
        return card_logits, grid_logits
