"""Core components for the Mamba2 Meets Silence architecture."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .mamba2 import Mamba2Block, RMSNorm


def _build_band_boundaries(freq_bins: int, num_subbands: int) -> List[int]:
    """Create approximately equal-width sub-band boundaries."""

    linspace = torch.linspace(0, freq_bins, steps=num_subbands + 1)
    boundaries = linspace.round().to(torch.int64).tolist()
    boundaries[0] = 0
    boundaries[-1] = freq_bins

    # Ensure strictly increasing boundaries (at least one bin per band).
    for idx in range(1, len(boundaries)):
        if boundaries[idx] <= boundaries[idx - 1]:
            boundaries[idx] = min(boundaries[idx - 1] + 1, freq_bins)

    boundaries[-1] = freq_bins
    return boundaries


class BandSplitModule(nn.Module):
    """Split the frequency axis and apply lightweight feature extractors."""

    def __init__(
        self,
        n_fft: int = 2048,
        num_subbands: int = 62,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_subbands = num_subbands
        self.freq_dim = n_fft // 2 + 1
        self.hidden_dim = hidden_dim

        self.band_boundaries = _build_band_boundaries(self.freq_dim, num_subbands)
        self.band_widths = [self.band_boundaries[i + 1] - self.band_boundaries[i] for i in range(num_subbands)]

        self.band_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for width in self.band_widths
        ])

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        batch, time, freq, _ = spec.shape
        band_features = []

        for idx, mlp in enumerate(self.band_mlps):
            start = self.band_boundaries[idx]
            end = self.band_boundaries[idx + 1]

            band_real = spec[:, :, start:end, 0]
            band_imag = spec[:, :, start:end, 1]
            band = torch.cat([band_real, band_imag], dim=-1)

            band_feat = mlp(band)
            band_features.append(band_feat)

        band_features = torch.stack(band_features, dim=2)
        return band_features


class DualPathModule(nn.Module):
    """Dual-path Mamba2 processing along time and band dimensions."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        d_state: int = 64,
        d_conv: int = 4,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.time_blocks = nn.ModuleList([
            nn.ModuleDict({
                "mamba": Mamba2Block(hidden_dim, d_state, d_conv, expand=2, dropout=dropout),
                "norm": RMSNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])

        self.band_blocks = nn.ModuleList([
            nn.ModuleDict({
                "mamba": Mamba2Block(hidden_dim, d_state, d_conv, expand=2, dropout=dropout),
                "norm": RMSNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])

    @staticmethod
    def _process_time_axis(x: torch.Tensor, block: nn.ModuleDict) -> torch.Tensor:
        x_norm = block["norm"](x)
        x_out, _ = block["mamba"](x_norm)
        return x + x_out

    @staticmethod
    def _process_band_axis(x: torch.Tensor, block: nn.ModuleDict) -> torch.Tensor:
        x_norm = block["norm"](x)
        x_out, _ = block["mamba"](x_norm)
        return x + x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, num_bands, hidden_dim = x.shape

        for layer_idx in range(self.num_layers):
            time_block = self.time_blocks[layer_idx]
            band_block = self.band_blocks[layer_idx]

            x_time = x.reshape(batch * num_bands, time, hidden_dim)
            if self.use_gradient_checkpointing and self.training:
                x_time = torch.utils.checkpoint.checkpoint(  # type: ignore[attr-defined]
                    self._process_time_axis,
                    x_time,
                    time_block,
                    use_reentrant=False,
                )
            else:
                x_time = self._process_time_axis(x_time, time_block)

            x = x_time.reshape(batch, num_bands, time, hidden_dim)
            x = x.transpose(1, 2).contiguous()

            x_band = x.reshape(batch * time, num_bands, hidden_dim)
            if self.use_gradient_checkpointing and self.training:
                x_band = torch.utils.checkpoint.checkpoint(  # type: ignore[attr-defined]
                    self._process_band_axis,
                    x_band,
                    band_block,
                    use_reentrant=False,
                )
            else:
                x_band = self._process_band_axis(x_band, band_block)

            x = x_band.reshape(batch, time, num_bands, hidden_dim)

        return x


class MaskEstimationModule(nn.Module):
    """Estimate complex-valued separation masks for each sub-band."""

    def __init__(
        self,
        band_boundaries: List[int],
        hidden_dim: int,
        num_subbands: int,
        num_sources: int,
    ) -> None:
        super().__init__()
        self.band_boundaries = band_boundaries
        self.band_widths = [band_boundaries[i + 1] - band_boundaries[i] for i in range(num_subbands)]
        self.num_sources = num_sources
        self.freq_dim = band_boundaries[-1]

        self.mask_mlps = nn.ModuleList([
            nn.Sequential(
                RMSNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GLU(dim=-1),
                nn.Linear(hidden_dim, width * 2 * num_sources),
                nn.Tanh(),
            )
            for width in self.band_widths
        ])

    def forward(self, band_features: torch.Tensor) -> torch.Tensor:
        batch, time, num_bands, hidden_dim = band_features.shape
        masks = []

        for idx, mlp in enumerate(self.mask_mlps):
            band_feat = band_features[:, :, idx, :]
            mask = mlp(band_feat)
            width = self.band_widths[idx]
            mask = mask.view(batch, time, width, 2, self.num_sources)
            masks.append(mask)

        mask = torch.cat(masks, dim=2)
        mask = mask.permute(0, 4, 1, 2, 3).contiguous()
        return mask
