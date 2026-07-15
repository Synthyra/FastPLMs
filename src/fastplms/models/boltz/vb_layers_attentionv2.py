"""Cross-attention variant of Boltz2 pair-biased attention."""

from __future__ import annotations

from einops.layers.torch import Rearrange
from torch import Tensor, nn

from . import vb_layers_initialize as init
from ._pair_attention import pair_biased_attention, reshape_heads


class AttentionPairBias(nn.Module):
    """Attend from S to K while adding either learned or precomputed pair bias."""

    def __init__(
        self,
        c_s: int,
        c_z: int | None = None,
        num_heads: int | None = None,
        inf: float = 1e6,
        compute_pair_bias: bool = True,
    ) -> None:
        super().__init__()
        if num_heads is None or c_s % num_heads:
            raise ValueError("num_heads must divide c_s")
        if compute_pair_bias and c_z is None:
            raise ValueError("c_z is required when pair bias is learned")

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)
        self.compute_pair_bias = compute_pair_bias
        if compute_pair_bias:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(c_z),
                nn.Linear(c_z, num_heads, bias=False),
                Rearrange("b ... h -> b h ..."),
            )
        else:
            self.proj_z = Rearrange("b ... h -> b h ...")
        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        init.final_init_(self.proj_o.weight)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        k_in: Tensor,
        multiplicity: int = 1,
    ) -> Tensor:
        """Transform S with K and pair input Z, returning shape ``(b, l_q, d)``."""

        query = reshape_heads(self.proj_q(s), self.num_heads)
        key = reshape_heads(self.proj_k(k_in), self.num_heads)
        value = reshape_heads(self.proj_v(k_in), self.num_heads)
        pair_bias = self.proj_z(z).repeat_interleave(multiplicity, dim=0)
        attended = pair_biased_attention(
            query,
            key,
            value,
            pair_bias,
            mask,
            self.inf,
        )
        attended = attended.reshape(s.shape[0], -1, self.c_s)
        gate = self.proj_g(s).sigmoid()
        return self.proj_o(gate * attended)
