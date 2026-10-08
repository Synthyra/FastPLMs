"""Exact SHA-256 digests of tensor values, the identities that structure bundles, goldens, and validation reports record.

Three definitions are in use and each is bound to records already written, so each keeps its own bytes:

- ``raw_tensor_sha256`` hashes the value bytes alone, so equal bytes in a different dtype or shape match.
- ``typed_tensor_sha256`` hashes the dtype name, the shape's ``repr``, and then the value bytes.
- ``tensor_set_sha256`` hashes each tensor of a mapping in name order: the name, the dtype name, the shape's ``repr``,
  and the value bytes.
"""

from __future__ import annotations

import hashlib
import torch

from collections.abc import Mapping
from torch import Tensor


def tensor_bytes(tensor: Tensor) -> bytes:
    """Return the value bytes of ``tensor`` in memory order, from its contiguous CPU copy, for any shape."""

    # tensor: (...)
    flattened = tensor.detach().cpu().contiguous().reshape(-1)  # (n,)
    return flattened.view(torch.uint8).numpy().tobytes()


def raw_tensor_sha256(tensor: Tensor) -> str:
    """Hash the value bytes of ``tensor`` alone."""

    # tensor: (...)
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def typed_tensor_sha256(tensor: Tensor) -> str:
    """Hash the dtype name, the shape's ``repr``, and the value bytes of ``tensor``."""

    # tensor: (...)
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor_bytes(tensor))
    return digest.hexdigest()


def tensor_set_sha256(tensors: Mapping[str, Tensor]) -> str:
    """Hash every name, dtype, shape, and value of ``tensors`` in name order."""

    # tensors: (...) one tensor per name, any shape
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]  # (...)
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor_bytes(tensor))
    return digest.hexdigest()
