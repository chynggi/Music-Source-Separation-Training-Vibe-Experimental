import torch
import torch.nn as nn
import typing as tp

from .utils import freq2bands


class BandSplitModule(nn.Module):
    """Band-split encoder that groups STFT bins into learnable sub-bands."""

    def __init__(
        self,
        sr: int,
        n_fft: int,
        bandsplits: tp.List[tp.Tuple[int, int]],
        t_timesteps: int,
        fc_dim: int,
        complex_as_channel: bool,
        is_mono: bool,
    ) -> None:
        super().__init__()

        frequency_mul = 1
        if complex_as_channel:
            frequency_mul *= 2
        if not is_mono:
            frequency_mul *= 2

        self.cac = complex_as_channel
        self.is_mono = is_mono
        self.bandwidth_indices = freq2bands(bandsplits, sr, n_fft)
        self.layernorms = nn.ModuleList(
            [nn.LayerNorm([(end - start) * frequency_mul, t_timesteps]) for start, end in self.bandwidth_indices]
        )
        self.fcs = nn.ModuleList(
            [nn.Linear((end - start) * frequency_mul, fc_dim) for start, end in self.bandwidth_indices]
        )

    def generate_subband(self, x: torch.Tensor) -> tp.Iterator[torch.Tensor]:
        for start_index, end_index in self.bandwidth_indices:
            yield x[:, :, start_index:end_index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split complex spectrogram ``x`` into sub-band embeddings."""
        subbands = []
        for idx, band in enumerate(self.generate_subband(x)):
            batch, channels, freqs, frames = band.shape
            if band.dtype == torch.cfloat:
                band = torch.view_as_real(band).permute(0, 1, 4, 2, 3)
            band = band.reshape(batch, -1, frames)
            band = self.layernorms[idx](band)
            band = band.transpose(-1, -2)
            band = self.fcs[idx](band)
            subbands.append(band)
        return torch.stack(subbands, dim=1)
