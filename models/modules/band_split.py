from __future__ import annotations

from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


class EfficientBandSplit(nn.Module):
    """Split spectrograms into independent sub-bands and process with grouped convs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        n_bands: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_bands <= 0:
            raise ValueError("n_bands must be positive")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_bands = n_bands
        self.kernel_size = kernel_size
        self.dropout = dropout

        padding = kernel_size // 2
        self.band_processor = nn.Conv2d(
            in_channels * n_bands,
            out_channels * n_bands,
            kernel_size=kernel_size,
            padding=padding,
            groups=n_bands,
            bias=bias,
        )
        self.post_norm = nn.BatchNorm2d(out_channels) if out_channels > 1 else nn.Identity()
        self.post_act = nn.GELU()
        self.post_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, c, f, t = x.shape
        self._original_freq = f
        if f % self.n_bands != 0:
            pad = (self.n_bands - (f % self.n_bands)) % self.n_bands
            x = F.pad(x, (0, 0, 0, pad))
            f = f + pad
        band_size = f // self.n_bands
        bands = torch.chunk(x, chunks=self.n_bands, dim=2)
        return torch.cat(bands, dim=1), band_size

    def _merge(self, x: torch.Tensor, band_size: int, original_freq: int) -> torch.Tensor:
        b, _, _, t = x.shape
        x = x.view(b, self.n_bands, self.out_channels, band_size, t)
        x = (
            x.permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(b, self.out_channels, band_size * self.n_bands, t)
        )
        return x[:, :, :original_freq, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stacked, band_size = self._split(x)
        processed = self.band_processor(stacked)
        merged = self._merge(processed, band_size, self._original_freq)
        merged = self.post_norm(merged)
        merged = self.post_act(merged)
        return self.post_dropout(merged)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"n_bands={self.n_bands}, kernel_size={self.kernel_size}, dropout={self.dropout}"
        )


class MultiBandMaskEstimator(nn.Module):
    """Estimate spectrogram masks by aggregating per-band refinements."""

    def __init__(
        self,
        in_channels: int,
        mask_channels: int,
        *,
        n_bands: int = 4,
        dropout: float = 0.0,
    activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or mask_channels <= 0:
            raise ValueError("in_channels and mask_channels must be positive")
        if n_bands <= 0:
            raise ValueError("n_bands must be positive")

        self.n_bands = n_bands
        self.activation = activation

        self.band_split = EfficientBandSplit(
            in_channels,
            in_channels,
            n_bands=n_bands,
            dropout=dropout,
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.mask_proj = nn.Conv2d(in_channels, mask_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.band_split(x)
        feats = self.refine(feats)
        mask = self.mask_proj(feats)
        if self.activation is not None:
            mask = self.activation(mask)
        return mask
