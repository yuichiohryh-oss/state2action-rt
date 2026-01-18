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
        aux_dim: int = 0,
    ) -> None:
        super().__init__()
        if num_actions <= 0 or num_grids <= 0:
            raise ValueError("num_actions and num_grids must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if aux_dim < 0:
            raise ValueError("aux_dim must be non-negative")
        self.num_actions = num_actions
        self.num_grids = num_grids
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.aux_dim = aux_dim

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
        head_in_dim = embedding_dim + self.aux_dim
        self.card_head = nn.Linear(head_in_dim, num_actions)
        self.grid_head = nn.Linear(head_in_dim, num_grids)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        pooled = self.pool(feats).flatten(1)
        return self.embedding(pooled)

    def forward(
        self, x_img: torch.Tensor, x_aux: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encode_image(x_img)
        if self.aux_dim > 0:
            if x_aux is None:
                raise ValueError("x_aux is required when aux_dim > 0")
            if x_aux.dim() == 1:
                x_aux = x_aux.unsqueeze(0)
            if x_aux.size(1) != self.aux_dim:
                raise ValueError(f"x_aux must have shape [B, {self.aux_dim}]")
            if x_aux.size(0) != emb.size(0):
                raise ValueError("x_aux batch size mismatch")
            x_aux = x_aux.to(dtype=emb.dtype, device=emb.device)
            emb = torch.cat([emb, x_aux], dim=1)
        card_logits = self.card_head(emb)
        grid_logits = self.grid_head(emb)
        return card_logits, grid_logits
