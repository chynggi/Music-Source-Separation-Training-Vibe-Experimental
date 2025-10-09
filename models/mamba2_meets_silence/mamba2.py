"""Mamba2 block implementation used by the Mamba2 Meets Silence architecture.

This module provides a bidirectional state space model with optional CUDA
acceleration via the ``mamba-ssm`` and ``causal-conv1d`` packages. The code is
adapted from the standalone MMS reference implementation to integrate with the
Music-Source-Separation-Training project.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Try to import optimized selective scan kernels from mamba-ssm. When the
# dependency is missing the implementation transparently falls back to the pure
# PyTorch variant.
try:  # pragma: no cover - optional dependency
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # type: ignore

    MAMBA_SSM_AVAILABLE = True
except Exception:  # pragma: no cover - handled dynamically at runtime
    selective_scan_fn = None  # type: ignore[assignment]
    MAMBA_SSM_AVAILABLE = False

# ``causal-conv1d`` offers a fused causal depth-wise convolution that is
# substantially faster than the PyTorch alternative on CUDA. As above we silently
# fall back to ``nn.Conv1d`` when it is not available.
try:  # pragma: no cover - optional dependency
    from causal_conv1d import causal_conv1d_fn  # type: ignore

    CAUSAL_CONV1D_AVAILABLE = True
except Exception:  # pragma: no cover - handled dynamically at runtime
    causal_conv1d_fn = None  # type: ignore[assignment]
    CAUSAL_CONV1D_AVAILABLE = False


class Mamba2Block(nn.Module):
    """Bidirectional Mamba2 block with selective state updates."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_inner + d_state + d_state, bias=False)

        self.A_log = nn.Parameter(torch.randn(self.d_inner))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = nn.LayerNorm(self.d_inner)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seqlen, _ = x.shape

        out_fwd, state_fwd = self._forward_direction(x, state)

        x_rev = torch.flip(x, dims=[1])
        out_bwd, state_bwd = self._forward_direction(x_rev, state)
        out_bwd = torch.flip(out_bwd, dims=[1])

        out = 0.5 * (out_fwd + out_bwd)
        if state_fwd is not None and state_bwd is not None:
            state = 0.5 * (state_fwd + state_bwd)
        else:
            state = None

        return out, state

    def _forward_direction(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seqlen, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        if CAUSAL_CONV1D_AVAILABLE and x.is_cuda:
            x = rearrange(x, "b l d -> b d l").contiguous()
            weight = self.conv1d.weight
            bias = self.conv1d.bias
            try:
                x = causal_conv1d_fn(x, weight.squeeze(1), bias, activation="silu")
            except Exception:
                x = self.conv1d(x)[:, :, :seqlen]
            x = rearrange(x, "b d l -> b l d")
        else:
            x = rearrange(x, "b l d -> b d l")
            x = self.conv1d(x)[:, :, :seqlen]
            x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)

        x_proj = self.x_proj(x)
        delta, B, C = torch.split(x_proj, [self.d_inner, self.d_state, self.d_state], dim=-1)

        A = -torch.exp(self.A_log.float())
        delta = F.softplus(delta)

        if x.dtype in (torch.float16, torch.bfloat16):
            delta = delta.to(x.dtype)
            B = B.to(x.dtype)
            C = C.to(x.dtype)
            A = A.to(x.dtype)

        y = self._selective_scan(x, delta, A, B, C, self.D, state)
        y = y * F.silu(z)

        out = self.out_proj(y)
        out = self.dropout(out)

        return out, None

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if MAMBA_SSM_AVAILABLE and u.is_cuda:
            return self._selective_scan_cuda(u, delta, A, B, C, D, state)
        return self._selective_scan_pytorch(u, delta, A, B, C, D, state)

    def _selective_scan_cuda(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seqlen, d_inner = u.shape
        _, _, d_state = B.shape

        target_dtype = u.dtype
        delta = delta.to(target_dtype)
        A = A.to(target_dtype)
        B = B.to(target_dtype)
        C = C.to(target_dtype)
        D = D.to(target_dtype)

        u_t = rearrange(u, "b l d -> b d l").contiguous()
        delta_t = rearrange(delta, "b l d -> b d l").contiguous()
        B_t = rearrange(B, "b l n -> b n l").contiguous()
        C_t = rearrange(C, "b l n -> b n l").contiguous()

        A_expanded = A.unsqueeze(-1).expand(d_inner, d_state).contiguous()

        try:
            y = selective_scan_fn(  # type: ignore[misc]
                u_t,
                delta_t,
                A_expanded,
                B_t,
                C_t,
                D=D,
                z=None,
                delta_bias=None,
                delta_softplus=False,
                return_last_state=False,
            )
            y = rearrange(y, "b d l -> b l d")
            return y
        except Exception:
            return self._selective_scan_pytorch(u, delta, A, B, C, D, state)

    def _selective_scan_pytorch(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seqlen, d_inner = u.shape
        _, _, d_state = B.shape

        if state is None:
            state = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)

        deltaA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, -1, 1))
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        outputs = []
        x = state
        for i in range(seqlen):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.einsum("bn,bdn->bd", C[:, i], x) + D * u[:, i]
            outputs.append(y)

        y = torch.stack(outputs, dim=1)
        return y


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm variant used in the architecture."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm
