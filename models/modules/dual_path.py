from __future__ import annotations

from torch import nn, Tensor


class DualPathBlock(nn.Module):
    """Apply time- and frequency-domain modules sequentially with residual links."""

    def __init__(self, time_module: nn.Module, freq_module: nn.Module) -> None:
        super().__init__()
        self.time_module = time_module
        self.freq_module = freq_module

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.time_module(x)
        x = x + self.freq_module(x)
        return x
