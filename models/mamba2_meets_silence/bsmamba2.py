"""Main BSMamba2 module adapted for the Music-Source-Separation-Training repo."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .components import BandSplitModule, DualPathModule, MaskEstimationModule


class BSMamba2(nn.Module):
    """Band-split Mamba2 core that predicts complex masks."""

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 441,
        num_subbands: int = 62,
        hidden_dim: int = 256,
        num_layers: int = 6,
        d_state: int = 64,
        d_conv: int = 4,
        dropout: float = 0.0,
        num_sources: int = 1,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_sources = num_sources

        self.band_split = BandSplitModule(
            n_fft=n_fft,
            num_subbands=num_subbands,
            hidden_dim=hidden_dim,
        )

        self.dual_path = DualPathModule(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        self.mask_estimation = MaskEstimationModule(
            band_boundaries=self.band_split.band_boundaries,
            hidden_dim=hidden_dim,
            num_subbands=num_subbands,
            num_sources=num_sources,
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Return complex masks with shape (batch, num_sources, time, freq, 2)."""

        band_features = self.band_split(spec)
        band_features = self.dual_path(band_features)
        masks = self.mask_estimation(band_features)
        return masks
