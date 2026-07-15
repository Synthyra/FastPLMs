"""Gated feed-forward transition used by Boltz2 pair representations."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import vb_layers_initialize as init


class Transition(nn.Module):
    """Apply a normalized SwiGLU-style transition.

    The checkpoint-facing module names remain ``norm`` and ``fc1`` through
    ``fc3``. For an input tensor X with shape ``(..., d)``, the module returns
    a tensor with shape ``(..., d_out)``.
    """

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        output_dim = dim if out_dim is None else out_dim
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, output_dim, bias=False)
        self.silu = nn.SiLU()
        self.hidden = hidden

        init.bias_init_one_(self.norm.weight)
        init.bias_init_zero_(self.norm.bias)
        init.lecun_normal_init_(self.fc1.weight)
        init.lecun_normal_init_(self.fc2.weight)
        init.final_init_(self.fc3.weight)

    def _project_hidden_slice(self, normalized: Tensor, start: int, stop: int) -> Tensor:
        """Return one hidden-width contribution to the output projection."""

        gate = self.silu(torch.matmul(normalized, self.fc1.weight[start:stop].T))
        value = torch.matmul(normalized, self.fc2.weight[start:stop].T)
        return torch.matmul(gate * value, self.fc3.weight[:, start:stop].T)

    def forward(self, x: Tensor, chunk_size: int | None = None) -> Tensor:
        """Transform X, optionally accumulating the hidden dimension in chunks."""

        # X is the normalized input tensor with shape (..., d).
        normalized = self.norm(x)
        if chunk_size is None or self.training:
            # H is the gated hidden tensor with shape (..., d_hidden).
            hidden_states = self.silu(self.fc1(normalized)) * self.fc2(normalized)
            return self.fc3(hidden_states)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        output: Tensor | None = None
        for start in range(0, self.hidden, chunk_size):
            contribution = self._project_hidden_slice(
                normalized,
                start,
                min(start + chunk_size, self.hidden),
            )
            output = contribution if output is None else output + contribution
        if output is None:  # hidden is normally positive; keep malformed configs explicit.
            raise ValueError("hidden must be positive")
        return output
