from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.modules import EfficientBandSplit, RoPETransformer, TFCTDFBlock


class ComplexSTFT(nn.Module):
    """Differentiable complex STFT/ISTFT pair with fixed window."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int | None = None,
        *,
        center: bool = True,
        window: str = "hann",
    ) -> None:
        super().__init__()
        win_length = win_length or n_fft
        if window == "hann":
            win = torch.hann_window(win_length)
        else:
            raise ValueError(f"Unsupported window type: {window}")
        self.register_buffer("window", win, persistent=False)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.freq_bins = n_fft // 2 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        window = self.window.to(x.device, x.dtype)
        spec = torch.stft(
            x.view(-1, length),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
            window=window,
            return_complex=True,
        )
        spec = spec[..., : self.freq_bins, :]
        spec = torch.view_as_real(spec)
        spec = spec.permute(0, 3, 1, 2)
        spec = spec.reshape(batch, channels, 2, self.freq_bins, -1)
        spec = spec.reshape(batch, channels * 2, self.freq_bins, -1)
        return spec

    def inverse(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        batch, channels2, freq, frames = spec.shape
        channels = channels2 // 2
        if freq < self.freq_bins:
            pad = self.freq_bins - freq
            spec = F.pad(spec, (0, 0, 0, pad))
            freq = self.freq_bins
        spec = spec.view(batch, channels, 2, freq, frames)
        spec = spec.permute(0, 1, 3, 4, 2)
        spec_complex = torch.view_as_complex(spec.contiguous())
        spec_complex = spec_complex.view(batch * channels, freq, frames)
        window = self.window.to(spec_complex.device, spec_complex.dtype)
        audio = torch.istft(
            spec_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
            window=window,
            length=length,
        )
        audio = audio.view(batch, channels, -1)
        return audio


@dataclass(frozen=True)
class MoisesLightConfig:
    n_bands: int = 4
    n_enc: int = 3
    n_dec: int = 1
    n_rope: int = 5
    n_split_enc: int = 3
    n_split_dec: int = 1
    g: int = 48
    dropout: float = 0.1
    num_stems: int = 1
    audio_channels: int = 2
    n_fft: int = 4096
    hop_length: int = 1024
    win_length: int | None = None
    mask_activation: str = "sigmoid"


def _norm_factory(num_groups: int = 8) -> callable:
    def factory(channels: int) -> nn.Module:
        groups = min(num_groups, channels)
        return nn.GroupNorm(num_groups=groups, num_channels=channels)

    return factory


def _act_factory() -> callable:
    return lambda: nn.GELU()


class DownsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MoisesLight(nn.Module):
    """Moises-Light: Resource-efficient band-split UNet with RoPE transformer bottleneck."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = MoisesLightConfig(**kwargs)
        cfg = self.config

        self.spec_channels = cfg.audio_channels * 2
        self.stft = ComplexSTFT(cfg.n_fft, cfg.hop_length, cfg.win_length)
        self.input_proj = nn.Conv2d(self.spec_channels, cfg.g, kernel_size=1)

        norm = _norm_factory()
        act = _act_factory()

        self.band_splits_enc = nn.ModuleList(
            [EfficientBandSplit(cfg.g, cfg.g, n_bands=cfg.n_bands, dropout=cfg.dropout) for _ in range(cfg.n_split_enc)]
        )

        self.encoder: nn.ModuleList[nn.ModuleDict] = nn.ModuleList()
        in_channels = cfg.g
        encoder_channels: List[int] = []
        for idx in range(cfg.n_enc):
            block = TFCTDFBlock(
                in_channels,
                in_channels,
                num_layers=max(1, cfg.n_dec),
                norm_layer=norm,
                act_layer=act,
                dropout=cfg.dropout,
            )
            stage = {"block": block}
            if idx < cfg.n_enc - 1:
                stage["down"] = DownsampleStage(in_channels, in_channels * 2, cfg.dropout)
                encoder_channels.append(in_channels)
                in_channels *= 2
            self.encoder.append(nn.ModuleDict(stage))

        self.bottleneck_channels = in_channels
        self.transformer = RoPETransformer(
            d_model=self.bottleneck_channels,
            n_heads=max(1, self.bottleneck_channels // 48),
            n_layers=cfg.n_rope,
            dropout=cfg.dropout,
            ff_mult=2.0,
        )

        decoder_channels: List[int] = list(reversed(encoder_channels))
        self.decoder: nn.ModuleList[nn.ModuleDict] = nn.ModuleList()
        current_channels = self.bottleneck_channels
        for skip_channels in decoder_channels:
            up = UpsampleStage(current_channels, skip_channels, cfg.dropout)
            block = TFCTDFBlock(
                skip_channels * 2,
                skip_channels,
                num_layers=max(1, cfg.n_dec),
                norm_layer=norm,
                act_layer=act,
                dropout=cfg.dropout,
            )
            self.decoder.append(nn.ModuleDict({"up": up, "block": block}))
            current_channels = skip_channels

        self.band_splits_dec = nn.ModuleList(
            [
                EfficientBandSplit(current_channels, current_channels, n_bands=cfg.n_bands, dropout=cfg.dropout)
                for _ in range(cfg.n_split_dec)
            ]
        )
        self.output_proj = nn.Conv2d(current_channels, cfg.num_stems * self.spec_channels, kernel_size=1)
        self.mask_activation = torch.sigmoid if cfg.mask_activation == "sigmoid" else torch.tanh

        total_scale = 2 ** (cfg.n_enc - 1) if cfg.n_enc > 1 else 1
        self.freq_multiple = total_scale
        self.time_multiple = total_scale

    def _pad_spec(self, spec: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, freq, frames = spec.shape
        pad_freq = (self.freq_multiple - freq % self.freq_multiple) % self.freq_multiple
        pad_time = (self.time_multiple - frames % self.time_multiple) % self.time_multiple
        if pad_freq != 0 or pad_time != 0:
            spec = F.pad(spec, (0, pad_time, 0, pad_freq))
        return spec, pad_freq, pad_time

    @staticmethod
    def _match_spatial(a: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        freq, frames = target_hw
        if a.shape[-2] == freq and a.shape[-1] == frames:
            return a
        return F.interpolate(a, size=(freq, frames), mode="bilinear", align_corners=False)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.dim() != 3:
            raise ValueError("Input must have shape (batch, channels, time)")
        batch, _, length = mixture.shape
        spec = self.stft(mixture)
        original_freq, original_time = spec.shape[-2:]
        spec_padded, _, _ = self._pad_spec(spec)

        feats = self.input_proj(spec_padded)
        for module in self.band_splits_enc:
            feats = module(feats)

        skips: List[torch.Tensor] = []
        for idx, stage in enumerate(self.encoder):
            feats = stage["block"](feats)
            if idx < len(self.encoder) - 1:
                skips.append(feats)
                feats = stage["down"](feats)

        feats = self.transformer(feats)

        for stage, skip in zip(self.decoder, reversed(skips)):
            feats = stage["up"](feats)
            if feats.shape[-2:] != skip.shape[-2:]:
                feats = self._match_spatial(feats, skip.shape[-2:])
            feats = torch.cat([feats, skip], dim=1)
            feats = stage["block"](feats)

        for module in self.band_splits_dec:
            feats = module(feats)

        mask = self.output_proj(feats)
        mask = self.mask_activation(mask)
        mask = mask[..., : original_freq, : original_time]

        spec_cropped = spec_padded[..., : original_freq, : original_time]
        mask = mask.view(batch, self.config.num_stems, self.spec_channels, original_freq, original_time)
        spec_cropped = spec_cropped.unsqueeze(1).expand(-1, self.config.num_stems, -1, -1, -1)
        estimated_spec = mask * spec_cropped
        estimated_spec = estimated_spec.view(batch * self.config.num_stems, self.spec_channels, original_freq, original_time)

        audio = self.stft.inverse(estimated_spec, length=length)
        audio = audio.view(batch, self.config.num_stems, self.config.audio_channels, -1)
        if self.config.num_stems == 1:
            audio = audio[:, 0]
        return audio
