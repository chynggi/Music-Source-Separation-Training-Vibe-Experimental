import math
import torch
from torch import nn


def _build_sinusoidal_embedding(length: int, dim: int) -> torch.Tensor:
    """Return standard sinusoidal positional embeddings."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class AbsolutePositionalEncoding(nn.Module):
    """Add absolute positional information to transformer inputs."""

    def __init__(self, length: int, dim: int, *, learnable: bool = False) -> None:
        super().__init__()
        self.length = length
        self.dim = dim
        self.learnable = learnable

        if learnable:
            self.embedding = nn.Parameter(torch.randn(length, dim))
        else:
            pe = _build_sinusoidal_embedding(length, dim)
            self.register_buffer("embedding", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (batch, seq, dim), got {tuple(x.shape)}")
        batch, seq_len, dim = x.shape
        if seq_len != self.length:
            raise ValueError(
                f"Sequence length {seq_len} does not match positional embedding length {self.length}."
            )
        if dim != self.dim:
            raise ValueError(
                f"Feature dim {dim} does not match positional embedding dim {self.dim}."
            )

        embedding = self.embedding
        if not self.learnable:
            embedding = embedding.to(x.dtype)
        return embedding.unsqueeze(0).expand(batch, -1, -1)

    def add_to(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience helper to add positional encoding to input."""
        return x + self.forward(x)
