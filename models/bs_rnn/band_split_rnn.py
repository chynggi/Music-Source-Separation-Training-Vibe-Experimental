from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .modules import BandSplitModule, BandSequenceModelModule, BandTransformerModelModule, MaskEstimationModule


class MultiMaskBandSplitRNN(nn.Module):
    """BandSplitRNN core that predicts a mask per instrument."""

    def __init__(
        self,
        sr: int,
        n_fft: int,
        bandsplits: List[Tuple[int, int]],
        t_timesteps: int,
        fc_dim: int,
        mlp_dim: int,
        num_sources: int,
        complex_as_channel: bool,
        is_mono: bool,
        bottleneck_layer: str,
        rnn_dim: int,
        rnn_type: str,
        bidirectional: bool,
        num_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
        transformer_ff_dim: Optional[int] = None,
        return_mask: bool = False,
        activation: str = "tanh",
    ) -> None:
        super().__init__()

        if num_sources < 1:
            raise ValueError("num_sources must be >= 1")

        self.cac = complex_as_channel
        self.return_mask = return_mask

        self.band_split = BandSplitModule(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim,
            complex_as_channel=complex_as_channel,
            is_mono=is_mono,
        )

        bottleneck = bottleneck_layer.lower()
        if bottleneck == "rnn":
            self.bottleneck = BandSequenceModelModule(
                input_dim_size=fc_dim,
                hidden_dim_size=rnn_dim,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                num_layers=num_layers,
            )
        elif bottleneck in {"att", "transformer"}:
            ff_dim = transformer_ff_dim if transformer_ff_dim is not None else mlp_dim
            self.bottleneck = BandTransformerModelModule(
                input_dim_size=fc_dim,
                hidden_dim_size=ff_dim,
                num_layers=num_layers,
                n_heads=transformer_heads,
                dropout=transformer_dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unsupported bottleneck layer: {bottleneck_layer}")

        mask_modules = []
        for _ in range(num_sources):
            mask_modules.append(
                MaskEstimationModule(
                    sr=sr,
                    n_fft=n_fft,
                    bandsplits=bandsplits,
                    t_timesteps=t_timesteps,
                    fc_dim=fc_dim,
                    mlp_dim=mlp_dim,
                    complex_as_channel=complex_as_channel,
                    is_mono=is_mono,
                    activation=activation,
                )
            )
        self.mask_estimators = nn.ModuleList(mask_modules)

    @staticmethod
    def wiener(x_hat: Tensor, x_complex: Tensor) -> Tensor:
        # Placeholder for potential Wiener filtering improvements
        return x_hat

    def compute_masks(self, x: Tensor) -> Tensor:
        features = self.band_split(x)
        features = self.bottleneck(features)
        masks = [module(features) for module in self.mask_estimators]
        mask = torch.stack(masks, dim=1)
        return mask

    def forward(self, x: Tensor) -> Tensor:
        x_complex = None
        if not self.cac:
            x_complex = x
            x = x.abs()

        eps = 1e-5
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        std = std.clamp(min=eps)
        x_norm = (x - mean) / std

        masks = self.compute_masks(x_norm)
        if self.return_mask:
            return masks

        separated = masks * x_norm.unsqueeze(1)
        separated = separated * std.unsqueeze(1) + mean.unsqueeze(1)

        if not self.cac and x_complex is not None:
            separated = self.wiener(separated, x_complex.unsqueeze(1))

        return separated


class BandSplitRNNSeparator(nn.Module):
    """Waveform-to-waveform separator built on top of MultiMaskBandSplitRNN."""

    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        num_instruments: int,
        bandsplits: List[Tuple[int, int]],
        fc_dim: int,
        mlp_dim: int,
        bottleneck_layer: str,
        rnn_dim: int,
        rnn_type: str,
        bidirectional: bool,
        num_layers: int,
        complex_as_channel: bool = True,
        is_mono: bool = False,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.0,
        transformer_ff_dim: Optional[int] = None,
        activation: str = "tanh",
        chunk_size: Optional[int] = None,
        t_timesteps: Optional[int] = None,
        return_mask: bool = False,
        window_fn: Optional[str] = None,
    ) -> None:
        super().__init__()

        if t_timesteps is None and chunk_size is not None:
            t_timesteps = (chunk_size - win_length) // hop_length + 1
            t_timesteps = max(1, t_timesteps)
        elif t_timesteps is None:
            raise ValueError("Either t_timesteps or chunk_size must be provided.")

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_instruments = num_instruments
        self.window_fn = window_fn or "hann"
        self.core = MultiMaskBandSplitRNN(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim,
            mlp_dim=mlp_dim,
            num_sources=num_instruments,
            complex_as_channel=complex_as_channel,
            is_mono=is_mono,
            bottleneck_layer=bottleneck_layer,
            rnn_dim=rnn_dim,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            num_layers=num_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            transformer_ff_dim=transformer_ff_dim,
            return_mask=return_mask,
            activation=activation,
        )

    def _build_window(self, device: torch.device) -> Tensor:
        if self.window_fn == "hann":
            return torch.hann_window(self.win_length, device=device)
        if self.window_fn == "hamming":
            return torch.hamming_window(self.win_length, device=device)
        raise ValueError(f"Unsupported window function: {self.window_fn}")

    def _reshape_audio(self, audio: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        batch, channels, _ = audio.shape
        return audio.view(batch * channels, -1), (batch, channels)

    def forward(self, mixture: Tensor) -> Tensor:
        mixture, shape = self._reshape_audio(mixture)
        batch, channels = shape
        device = mixture.device
        window = self._build_window(device)
        stft = torch.stft(
            mixture,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        freq_bins, frames = stft.shape[-2:]
        stft = stft.view(batch, channels, freq_bins, frames)

        separated_spec = self.core(stft)
        separated_spec = separated_spec.view(batch * self.num_instruments * channels, freq_bins, frames)

        separated_audio = torch.istft(
            separated_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=mixture.shape[-1],
        )
        separated_audio = separated_audio.view(batch, self.num_instruments, channels, -1)
        return separated_audio
