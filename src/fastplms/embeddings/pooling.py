"""Residue-aware pooling implemented entirely with PyTorch."""

from __future__ import annotations

import math
import torch

from collections.abc import Sequence
from torch import Tensor


POOLING_NAMES = frozenset({"mean", "max", "norm", "median", "std", "var", "cls", "parti"})
POOLING_SEMANTICS = {
    "version": 2,
    "mask": "biological_residues_only",
    "accumulator": "float64_for_float64_input_else_float32",
    "output_dtype": "input_dtype_after_reduction",
    "variance_correction": 0,
    "empty_residues": "reject",
    "singleton_variance": 0,
}


# Pooling over every attended token, CLS and EOS included: the canonical policy of a store that
# keeps the special tokens. Version 3 supersedes the residue-only version 2 for those stores.
POOLING_SEMANTICS_TOKENS = {
    "version": 3,
    "mask": "all_attended_tokens_including_cls_and_eos",
    "accumulator": "float64_for_float64_input_else_float32",
    "output_dtype": "input_dtype_after_reduction",
    "variance_correction": 0,
    "empty_residues": "reject",
    "singleton_variance": 0,
}
TOKEN_POOLING_NAMES = ("mean", "var", "std", "max", "norm")


def pool_token_rows(X: Tensor, token_mask: Tensor, names: Sequence[str]) -> Tensor:
    """Pool every attended token of each sequence, CLS and EOS included, without any device read.

    The reductions equal ``Pooler`` over a mask that is true on every token: float32 accumulation
    (float64 for float64 input), population variance from the two-pass mean, and a cast to the input
    dtype after reduction. Unlike ``Pooler`` this skips its input validation, which reads device
    values and stalls the host; the caller checks finiteness on the device after the copy lands.
    """
    # X: (b, n, d) padded, n = the longest l + 2; token_mask: (b, n) true on CLS, residues and EOS
    unsupported = [name for name in names if name not in TOKEN_POOLING_NAMES]
    if unsupported or not names or len(set(names)) != len(names):
        raise ValueError(f"Token pooling supports {TOKEN_POOLING_NAMES}, each once; received {list(names)}.")
    work = torch.float64 if X.dtype == torch.float64 else torch.float32
    values = X.to(work)  # (b, n, d)
    kept = token_mask.unsqueeze(-1)  # (b, n, 1)
    count = kept.sum(dim=1).clamp_min(1).to(work)  # (b, 1), attended rows per sequence: l + 2
    masked = values.masked_fill(~kept, 0)  # (b, n, d), padding zeroed
    mean = masked.sum(dim=1) / count  # (b, d)
    outputs: list[Tensor] = []
    for name in names:
        if name == "mean":
            pooled = mean  # (b, d)
        elif name == "max":
            pooled = values.masked_fill(~kept, -torch.inf).amax(dim=1)  # (b, d)
        elif name == "norm":
            pooled = torch.linalg.vector_norm(masked, ord=2, dim=1)  # (b, d)
        else:
            centered = (values - mean.unsqueeze(1)).masked_fill(~kept, 0)  # (b, n, d)
            variance = (centered * centered).sum(dim=1) / count  # (b, d)
            pooled = variance.sqrt() if name == "std" else variance  # (b, d)
        outputs.append(pooled.to(X.dtype))
    return torch.cat(outputs, dim=-1)  # (b, len(names) * d)


def _validate_inputs(X: Tensor, M: Tensor) -> Tensor:
    # X: (b, l, d); M: (b, l)
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
    M = M.to(device=X.device, dtype=torch.bool)  # (b, l)
    if not bool(M.any(dim=1).all()):
        raise ValueError("Every sample must contain at least one biological residue.")
    if not bool((torch.isfinite(X) | ~M.unsqueeze(-1)).all()):
        raise ValueError("Biological residue embeddings produced non-finite output.")
    return M  # (b, l)


def _pooled_attention(attentions: Tensor | Sequence[Tensor], *, batch_size: int) -> Tensor:
    """Max-pool layer/head attention A to shape ``(b, l, l)``.

    ``parti`` historically keeps the strongest directed edge across the
    available attention maps before PageRank. Replacing NetworkX with Torch
    must not change that reduction.
    """

    # attentions: (b, ..., l, l), or a sequence of (b, h, l, l) layer maps
    if isinstance(attentions, Sequence):
        if not attentions:
            raise ValueError("parti received an empty attention sequence.")
        # Each A_i: (b, h, l, l).
        A = torch.stack(tuple(attentions), dim=1)  # (b, n, h, l, l)
    else:
        A = attentions  # (b, ..., l, l)

    if A.ndim == 5:
        if A.shape[0] != batch_size and A.shape[1] == batch_size:
            A = A.transpose(0, 1)  # (b, n, h, l, l)
        if A.shape[0] != batch_size:
            raise ValueError("Five-dimensional attentions must use (b, n, h, l, l).")
        A = A.flatten(1, 2).amax(dim=1)  # (b, l, l)
    elif A.ndim == 4:
        if A.shape[0] != batch_size:
            raise ValueError("Four-dimensional attentions must use (b, h, l, l).")
        A = A.amax(dim=1)  # (b, l, l)
    elif A.ndim == 3:
        if A.shape[0] != batch_size:
            raise ValueError("Three-dimensional attentions must use (b, l, l).")
    else:
        raise ValueError("Attentions must have shape (b, l, l), (b, h, l, l), or (b, n, h, l, l).")
    return A  # (b, l, l)


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

    # A: (l, l)
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
    P = A.detach().to(dtype=work_dtype).clamp_min(0)  # (l, l)
    row_sum = P.sum(dim=-1, keepdim=True)  # (l, 1)
    uniform = torch.full_like(P, 1.0 / length)  # (l, l)
    P = torch.where(  # (l, l)
        row_sum > 0,
        P / row_sum.clamp_min(torch.finfo(work_dtype).tiny),
        uniform,
    )
    p = torch.full((length,), 1.0 / length, device=P.device, dtype=work_dtype)  # (l,)
    teleport = (1.0 - damping) / length
    for _ in range(max_iterations):
        p_next = teleport + damping * (P.transpose(0, 1) @ p)  # (l,)
        if torch.linalg.vector_norm(p_next - p, ord=1) <= tolerance:
            p = p_next  # (l,)
            break
        p = p_next  # (l,)
    return p / p.sum()  # (l,)


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
        # X: (b, l, d); residue_mask: (b, l); attentions: (b, ..., l, l), or a sequence of (b, h, l, l) layer maps
        M = _validate_inputs(X, residue_mask)  # (b, l)
        output_dtype = X.dtype
        # Sum/variance in FP16 can overflow even when the final answer is representable.
        # Retain FP64 precision, otherwise accumulate in FP32 and cast only the result.
        X = X.to(dtype=torch.float64 if X.dtype == torch.float64 else torch.float32)  # (b, l, d)
        M_expanded = M.unsqueeze(-1)  # (b, l, 1)
        count = M_expanded.sum(dim=1).clamp_min(1)  # (b, 1)
        X_residues = X.masked_fill(~M_expanded, 0)  # (b, l, d)
        outputs: list[Tensor] = []

        for name in self.names:
            if name == "mean":
                Y = X_residues.sum(dim=1) / count  # (b, d)
            elif name == "max":
                Y = X.masked_fill(~M_expanded, -torch.inf).max(dim=1).values  # (b, d)
            elif name == "norm":
                Y = torch.linalg.vector_norm(X_residues, ord=2, dim=1)  # (b, d)
            elif name == "median":
                Y = X.masked_fill(~M_expanded, torch.nan).nanmedian(dim=1).values  # (b, d)
            elif name in {"var", "std"}:
                mean = X_residues.sum(dim=1, keepdim=True) / count.unsqueeze(1)  # (b, 1, d)
                centered = (X - mean).masked_fill(~M_expanded, 0)  # (b, l, d)
                variance = (centered**2).sum(dim=1) / count  # (b, d)
                Y = variance.sqrt() if name == "std" else variance  # (b, d)
            elif name == "cls":
                Y = X[:, 0]  # (b, d)
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
                A = _pooled_attention(attentions, batch_size=X.shape[0]).to(X.device)  # (b, l, l)
                pooled: list[Tensor] = []
                for X_i, M_i, A_i in zip(X, M, A, strict=True):
                    # X_i: (l, d); M_i: (l,); A_i: (l, l)
                    indices = M_i.nonzero(as_tuple=True)[0]  # (r,)
                    A_residue = A_i.index_select(0, indices).index_select(1, indices)  # (r, r)
                    w = pagerank_weights(A_residue).to(dtype=X.dtype)  # (r,)
                    pooled.append(w @ X_i.index_select(0, indices))  # (d,)
                Y = torch.stack(pooled)  # (b, d)
            Y = Y.to(dtype=output_dtype)  # (b, d)
            if not bool(torch.isfinite(Y).all()):
                raise ValueError(
                    f"Pooling operation {name!r} produced non-finite output from "
                    "biological residue embeddings."
                )
            outputs.append(Y)

        return torch.cat(outputs, dim=-1)  # (b, len(self.names) * d)


__all__ = [
    "POOLING_NAMES", "POOLING_SEMANTICS_TOKENS", "TOKEN_POOLING_NAMES", "Pooler", "pagerank_weights",
    "pool_token_rows",
]
