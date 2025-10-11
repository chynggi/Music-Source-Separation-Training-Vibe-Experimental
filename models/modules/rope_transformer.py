from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .dual_path import DualPathBlock


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


class RotaryEmbedding(nn.Module):
    """Generate rotary position embeddings for a given sequence length."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(dtype=dtype, device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[..., : x.shape[-1]]
    sin = sin[..., : x.shape[-1]]
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with rotary position embedding support."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_head = d_model // n_heads
        self.scale = self.dim_head ** -0.5

        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.dim_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, seq, self.n_heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.view(b, seq, self.n_heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.view(b, seq, self.n_heads, self.dim_head).permute(0, 2, 1, 3)

        cos, sin = self.rotary(seq, x.device, x.dtype)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=False
        )
        out = out.permute(0, 2, 1, 3).contiguous().view(b, seq, self.d_model)
        out = self.to_out(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(d_model * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, ff_mult: float = 4.0) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, ff_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class SequenceTransformer(nn.Module):
    """Apply transformer blocks to 2D feature maps along one axis."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, ff_mult: float, axis: str) -> None:
        super().__init__()
        if axis not in {"time", "freq"}:
            raise ValueError("axis must be 'time' or 'freq'")
        self.axis = axis
        self.block = TransformerBlock(d_model, n_heads, dropout=dropout, ff_mult=ff_mult)

    def _reshape(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        b, c, f, t = x.shape
        if self.axis == "time":
            seq = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
            ctx = (b, c, f, t)
        else:
            seq = x.permute(0, 3, 2, 1).reshape(b * t, f, c)
            ctx = (b, c, f, t)
        return seq, ctx

    def _restore(self, seq: torch.Tensor, ctx: tuple[int, ...]) -> torch.Tensor:
        b, c, f, t = ctx
        if self.axis == "time":
            seq = seq.reshape(b, f, t, c).permute(0, 3, 1, 2)
        else:
            seq = seq.reshape(b, t, f, c).permute(0, 3, 2, 1)
        return seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, ctx = self._reshape(x)
        seq = self.block(seq)
        return self._restore(seq, ctx)


class RoPETransformer(nn.Module):
    """Dual-path transformer stack with rotary position embeddings."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 5,
        dropout: float = 0.0,
        ff_mult: float = 2.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DualPathBlock(
                    SequenceTransformer(d_model, n_heads, dropout, ff_mult, axis="time"),
                    SequenceTransformer(d_model, n_heads, dropout, ff_mult, axis="freq"),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
