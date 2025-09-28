import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention


class TransformerModule(nn.Module):
    """Dual-path transformer block mixing information across time or bands."""

    def __init__(
        self,
        embed_dim: int,
        dim_ff: int,
        n_heads: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.groupnorm = nn.GroupNorm(embed_dim, embed_dim)
        self.mha = MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.recurrent = nn.LSTM(embed_dim, dim_ff, batch_first=True, bidirectional=bidirectional)
        out_dim = dim_ff * 2 if bidirectional else dim_ff
        self.linear = nn.Linear(out_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, bands, steps, dims = x.shape
        out = x.view(batch * bands, steps, dims)
        out = self.groupnorm(out.transpose(-1, -2)).transpose(-1, -2)

        mha_in = out.transpose(0, 1)
        mha_out, _ = self.mha(mha_in, mha_in, mha_in)
        out = mha_out.transpose(0, 1) + out

        rnn_out, _ = self.recurrent(out)
        out = self.linear(rnn_out) + out

        out = out.view(batch, bands, steps, dims)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out


class BandTransformerModelModule(nn.Module):
    """Dual-path transformer stack for BandSplitRNN bottleneck."""

    def __init__(
        self,
        input_dim_size: int,
        hidden_dim_size: int,
        num_layers: int,
        n_heads: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    TransformerModule(input_dim_size, hidden_dim_size, n_heads, dropout, bidirectional),
                    TransformerModule(input_dim_size, hidden_dim_size, n_heads, dropout, bidirectional),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
