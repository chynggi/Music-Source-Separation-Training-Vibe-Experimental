from __future__ import annotations

from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


class _TFCBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: Callable[[int], nn.Module],
        act: Callable[[], nn.Module],
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            norm(in_channels),
            act(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _TDFBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        y = x.permute(0, 3, 1, 2).reshape(b * t, c, f)
        y = self.net(y)
        y = y.reshape(b, t, c, f).permute(0, 2, 3, 1)
        return y


class TFCTDFBlock(nn.Module):
    """Stacked Time-Frequency Convolution (TFC) and Frequency Transform (TDF) blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        *,
        norm_layer: Callable[[int], nn.Module] | None = None,
        act_layer: Callable[[], nn.Module] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm = norm_layer or (lambda c: nn.GroupNorm(1, c))
        act = act_layer or (lambda: nn.GELU())

        blocks = []
        current_in = in_channels
        for _ in range(num_layers):
            tfc = _TFCBlock(current_in, out_channels, norm, act, dropout)
            tdf = _TDFBlock(out_channels, dropout)
            tfc_out = _TFCBlock(out_channels, out_channels, norm, act, dropout)
            blocks.append(nn.ModuleDict({"tfc1": tfc, "tdf": tdf, "tfc2": tfc_out}))
            current_in = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.use_projection = in_channels != out_channels
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if self.use_projection
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, block in enumerate(self.blocks):
            residual = x
            y = block["tfc1"](x)
            y = y + block["tdf"](y)
            y = block["tfc2"](y)
            if idx == 0 and self.use_projection:
                residual = self.shortcut(residual)
            x = y + residual
        return x
