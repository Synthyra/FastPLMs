"""Residue-aware pooling implemented entirely with PyTorch."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

POOLING_NAMES = frozenset({"mean", "max", "norm", "median", "std", "var", "cls", "parti"})


def _validate_inputs(X: Tensor, M: Tensor) -> Tensor:
    if not isinstance(X, Tensor) or not isinstance(M, Tensor):
        raise TypeError("X and M must be tensors.")
    if X.ndim != 3:
        raise ValueError(f"X must have shape (b, l, d), got {tuple(X.shape)}.")
    if not X.is_floating_point():
        raise TypeError("X must use a floating-point embedding dtype.")
    if M.shape != X.shape[:2]:
        raise ValueError(f"M must have shape (b, l)={tuple(X.shape[:2])}, got {tuple(M.shape)}.")
    if M.is_complex():
        raise TypeError("M must be a boolean or binary numeric residue mask.")
    if not bool(torch.isfinite(M).all()) or not bool(((M == 0) | (M == 1)).all()):
        raise ValueError("M must contain only finite binary mask values.")
    M = M.to(device=X.device, dtype=torch.bool)
    if not bool(M.any(dim=1).all()):
        raise ValueError("Every sample must contain at least one biological residue.")
    if not bool((torch.isfinite(X) | ~M.unsqueeze(-1)).all()):
        raise ValueError("Biological residue embeddings produced non-finite output.")
    return M


def _pooled_attention(attentions: Tensor | Sequence[Tensor], *, batch_size: int) -> Tensor:
    """Max-pool layer/head attention A to shape ``(b, l, l)``.

    ``parti`` historically keeps the strongest directed edge across the
    available attention maps before PageRank. Replacing NetworkX with Torch
    must not change that reduction.
    """

    if isinstance(attentions, Sequence):
        if not attentions:
            raise ValueError("parti received an empty attention sequence.")
        # Each A_i has shape (b, h, l, l).
        A = torch.stack(tuple(attentions), dim=1)
    else:
        A = attentions

    if A.ndim == 5:
        if A.shape[0] != batch_size and A.shape[1] == batch_size:
            A = A.transpose(0, 1)
        if A.shape[0] != batch_size:
            raise ValueError("Five-dimensional attentions must use (b, n, h, l, l).")
        A = A.flatten(1, 2).amax(dim=1)
    elif A.ndim == 4:
        if A.shape[0] != batch_size:
            raise ValueError("Four-dimensional attentions must use (b, h, l, l).")
        A = A.amax(dim=1)
    elif A.ndim == 3:
        if A.shape[0] != batch_size:
            raise ValueError("Three-dimensional attentions must use (b, l, l).")
    else:
        raise ValueError("Attentions must have shape (b, l, l), (b, h, l, l), or (b, n, h, l, l).")
    return A


def pagerank_weights(
    A: Tensor,
    *,
    damping: float = 0.85,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> Tensor:
    """Compute PageRank weights for a non-negative attention matrix A.

    A has shape ``(l, l)``. Rows are normalized into transition
    probabilities; dangling rows transition uniformly.
    """

    if not isinstance(A, Tensor):
        raise TypeError("A must be a tensor.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {tuple(A.shape)}.")
    if not A.is_floating_point():
        raise TypeError("A must use a floating-point attention dtype.")
    if not isinstance(damping, (int, float)) or isinstance(damping, bool):
        raise TypeError("damping must be a finite float in [0, 1).")
    if not math.isfinite(float(damping)) or not 0 <= damping < 1:
        raise ValueError("damping must be a finite float in [0, 1).")
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("tolerance must be a positive finite float.")
    if not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise ValueError("tolerance must be a positive finite float.")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise TypeError("max_iterations must be a positive integer.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    length = A.shape[0]
    if length == 0:
        raise ValueError("PageRank requires at least one residue.")
    if not bool(torch.isfinite(A).all()):
        raise ValueError("A must contain only finite attention values.")
    work_dtype = torch.float64 if A.dtype == torch.float64 else torch.float32
    P = A.detach().to(dtype=work_dtype).clamp_min(0)
    row_sum = P.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(P, 1.0 / length)
    P = torch.where(row_sum > 0, P / row_sum.clamp_min(torch.finfo(work_dtype).tiny), uniform)
    p = torch.full((length,), 1.0 / length, device=P.device, dtype=work_dtype)
    teleport = (1.0 - damping) / length
    for _ in range(max_iterations):
        p_next = teleport + damping * (P.transpose(0, 1) @ p)
        if torch.linalg.vector_norm(p_next - p, ord=1) <= tolerance:
            p = p_next
            break
        p = p_next
    return p / p.sum()


class Pooler:
    """Apply one or more pooling operations to biological residue rows."""

    def __init__(self, pooling: str | Sequence[str] = ("mean",)) -> None:
        pooling_value: object = pooling
        if isinstance(pooling_value, (bytes, bytearray)) or not isinstance(
            pooling_value, (str, Sequence)
        ):
            raise TypeError("pooling must be a name or a sequence of names.")
        names = (pooling_value,) if isinstance(pooling_value, str) else tuple(pooling_value)
        if not all(isinstance(name, str) for name in names):
            raise TypeError("pooling names must be strings.")
        if not names:
            raise ValueError("At least one pooling operation is required.")
        unknown = set(names) - POOLING_NAMES
        if unknown:
            raise ValueError(f"Unknown pooling operations: {sorted(unknown)}.")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate pooling operations are not supported: {duplicates}.")
        self.names = names

    def output_slices(self, d: int) -> dict[str, tuple[int, int]]:
        """Return the output interval assigned to each pooler."""

        if not isinstance(d, int) or isinstance(d, bool):
            raise TypeError("d must be a positive integer.")
        if d <= 0:
            raise ValueError("d must be a positive integer.")
        return {name: (i * d, (i + 1) * d) for i, name in enumerate(self.names)}

    def __call__(
        self,
        X: Tensor,
        residue_mask: Tensor,
        *,
        attentions: Tensor | Sequence[Tensor] | None = None,
        attention_backend: str | None = None,
    ) -> Tensor:
        M = _validate_inputs(X, residue_mask)
        M_expanded = M.unsqueeze(-1)
        count = M_expanded.sum(dim=1).clamp_min(1)
        X_residues = X.masked_fill(~M_expanded, 0)
        outputs: list[Tensor] = []

        for name in self.names:
            if name == "mean":
                Y = X_residues.sum(dim=1) / count
            elif name == "max":
                Y = X.masked_fill(~M_expanded, -torch.inf).max(dim=1).values
            elif name == "norm":
                Y = torch.linalg.vector_norm(X_residues, ord=2, dim=1)
            elif name == "median":
                Y = X.masked_fill(~M_expanded, torch.nan).nanmedian(dim=1).values
            elif name in {"var", "std"}:
                mean = X_residues.sum(dim=1, keepdim=True) / count.unsqueeze(1)
                centered = (X - mean).masked_fill(~M_expanded, 0)
                variance = (centered**2).sum(dim=1) / count
                Y = variance.sqrt() if name == "std" else variance
            elif name == "cls":
                Y = X[:, 0]
            else:
                if attention_backend != "eager":
                    raise ValueError(
                        "parti requires attn_implementation='eager' so full "
                        "attention matrices are available."
                    )
                if attentions is None:
                    raise ValueError("parti requires model attention matrices.")
                if int(M.sum(dim=1).max().item()) > 2048:
                    raise ValueError("parti supports at most 2,048 biological residues.")
                A = _pooled_attention(attentions, batch_size=X.shape[0]).to(X.device)
                pooled: list[Tensor] = []
                for X_i, M_i, A_i in zip(X, M, A, strict=True):
                    indices = M_i.nonzero(as_tuple=True)[0]
                    A_residue = A_i.index_select(0, indices).index_select(1, indices)
                    w = pagerank_weights(A_residue).to(dtype=X.dtype)
                    pooled.append(w @ X_i.index_select(0, indices))
                Y = torch.stack(pooled)
            if not bool(torch.isfinite(Y).all()):
                raise ValueError(
                    f"Pooling operation {name!r} produced non-finite output from "
                    "biological residue embeddings."
                )
            outputs.append(Y)

        return torch.cat(outputs, dim=-1)


__all__ = ["POOLING_NAMES", "Pooler", "pagerank_weights"]
