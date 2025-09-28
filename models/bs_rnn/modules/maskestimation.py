import torch
import torch.nn as nn
import typing as tp

from .utils import freq2bands


class GLU(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x[..., : self.input_dim] * self.sigmoid(x[..., self.input_dim :])


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation_type: str) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self._select_activation(activation_type)(),
            nn.Linear(hidden_dim, output_dim),
            GLU(output_dim),
        )

    @staticmethod
    def _select_activation(activation_type: str) -> tp.Type[nn.Module]:
        if activation_type == "tanh":
            return nn.Tanh
        if activation_type == "relu":
            return nn.ReLU
        if activation_type == "gelu":
            return nn.GELU
        raise ValueError("Unsupported activation type")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MaskEstimationModule(nn.Module):
    """Decode sub-band features into complex masks."""

    def __init__(
        self,
        sr: int,
        n_fft: int,
        bandsplits: tp.List[tp.Tuple[int, int]],
        t_timesteps: int,
        fc_dim: int,
        mlp_dim: int,
        complex_as_channel: bool,
        is_mono: bool,
        activation: str = "tanh",
    ) -> None:
        super().__init__()

        frequency_mul = 1
        if complex_as_channel:
            frequency_mul *= 2
        if not is_mono:
            frequency_mul *= 2

        self.cac = complex_as_channel
        self.is_mono = is_mono
        self.frequency_mul = frequency_mul

        self.bandwidths = [(end - start) for start, end in freq2bands(bandsplits, sr, n_fft)]
        self.layernorms = nn.ModuleList([nn.LayerNorm([t_timesteps, fc_dim]) for _ in self.bandwidths])
        self.mlps = nn.ModuleList(
            [MLP(fc_dim, mlp_dim, bandwidth * frequency_mul, activation) for bandwidth in self.bandwidths]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for idx in range(x.shape[1]):
            out = self.layernorms[idx](x[:, idx])
            out = self.mlps[idx](out)
            batch, steps, feats = out.shape
            if self.cac:
                out = out.view(batch, -1, 2, feats // self.frequency_mul, steps).permute(0, 1, 3, 4, 2)
                out = torch.view_as_complex(out.contiguous())
            else:
                out = out.view(batch, -1, feats // self.frequency_mul, steps).contiguous()
            outputs.append(out)
        return torch.cat(outputs, dim=-2)
