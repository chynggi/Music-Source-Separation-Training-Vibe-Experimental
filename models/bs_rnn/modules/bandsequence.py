import torch
import torch.nn as nn


class RNNModule(nn.Module):
    """Bi-directional RNN block applied along time or sub-band axes."""

    def __init__(
        self,
        input_dim_size: int,
        hidden_dim_size: int,
        rnn_type: str,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.groupnorm = nn.GroupNorm(input_dim_size, input_dim_size)
        rnn_cls = getattr(nn, rnn_type)
        self.rnn = rnn_cls(input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_dim_size * 2 if bidirectional else hidden_dim_size
        self.fc = nn.Linear(out_dim, input_dim_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, bands, steps, dims = x.shape
        out = x.view(batch * bands, steps, dims)
        out = self.groupnorm(out.transpose(-1, -2)).transpose(-1, -2)
        out = self.rnn(out)[0]
        out = self.fc(out)
        out = out.view(batch, bands, steps, dims)
        out = out + x
        out = out.permute(0, 2, 1, 3).contiguous()
        return out


class BandSequenceModelModule(nn.Module):
    """Dual-path RNN stack operating along time and band dimensions."""

    def __init__(
        self,
        input_dim_size: int,
        hidden_dim_size: int,
        rnn_type: str,
        bidirectional: bool,
        num_layers: int,
    ) -> None:
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    RNNModule(input_dim_size, hidden_dim_size, rnn_type, bidirectional),
                    RNNModule(input_dim_size, hidden_dim_size, rnn_type, bidirectional),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
