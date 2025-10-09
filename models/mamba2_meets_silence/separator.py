"""High level wrapper that plugs BSMamba2 into the MSST training pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .bsmamba2 import BSMamba2


class Mamba2MeetsSilence(nn.Module):
    """Waveform-to-waveform separator built on top of the BSMamba2 core."""

    def __init__(
        self,
        sample_rate: int = 44100,
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
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_sources = num_sources

        self.core = BSMamba2(
            n_fft=n_fft,
            hop_length=hop_length,
            num_subbands=num_subbands,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            dropout=dropout,
            num_sources=num_sources,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        window = torch.hann_window(n_fft)
        self.register_buffer("window", window, persistent=False)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """Separate the provided mixture into ``num_sources`` waveforms.

        Args:
            mixture: Tensor shaped ``(batch, channels, samples)``.

        Returns:
            Tensor shaped ``(batch, num_sources, channels, samples)``.
        """

        batch, channels, samples = mixture.shape
        mixture = mixture.float()

        mono = mixture.mean(dim=1)
        mono_spec = self._stft(mono)
        mask = self.core(mono_spec)  # (batch, num_sources, time, freq, 2)

        channel_specs = [self._stft(mixture[:, ch, :]).unsqueeze(1) for ch in range(channels)]
        channel_specs = torch.cat(channel_specs, dim=1)  # (batch, channels, time, freq, 2)

        outputs = []
        for src_idx in range(self.num_sources):
            src_mask = mask[:, src_idx].unsqueeze(1)
            masked_spec = channel_specs * src_mask
            audio = self._istft_multi(masked_spec, samples)
            outputs.append(audio.unsqueeze(1))

        separated = torch.cat(outputs, dim=1)
        return separated

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(audio.device),
            center=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = spec.permute(0, 2, 1, 3).contiguous()
        return spec

    def _istft_multi(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        batch, channels, time, freq, _ = spec.shape
        spec = spec.permute(0, 1, 3, 2, 4).contiguous()
        spec = torch.view_as_complex(spec)
        spec = spec.view(batch * channels, freq, time)

        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(spec.device),
            center=True,
            length=length,
        )
        audio = audio.view(batch, channels, length)
        return audio
