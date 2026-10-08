"""Dictionaries on disk as JSON, pickle, or PyTorch files, gzipped when the path ends in `.gz`.

Each reader and writer takes a plain or gzipped path, so a caller does not need to know how a
cache was stored, and asserts that the path carries its format's suffix and that the payload is a
dictionary. An atomic write goes to a temporary sibling first and replaces the target with
`os.replace`, so a reader never sees a half-written file. `foundry.serialization.atomic` holds the
atomic writers of bytes, text and JSON, and the one that hands any other writer a temporary path;
they are exported here too.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import numpy as np

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from foundry.compression import PathLike, is_gzip_path, open_binary_maybe_gzip, open_text_maybe_gzip
from foundry.serialization.atomic import atomic_replace, write_bytes_atomic, write_json_atomic, write_text_atomic


if TYPE_CHECKING:
    import torch


def read_json_dict(path: PathLike) -> dict[str, Any]:
    _assert_path_suffix(path, (".json", ".json.gz"))
    with open_text_maybe_gzip(path, mode="rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict), "JSON payload must be a dict."
    return payload


def write_json_dict(path: PathLike, payload: dict[str, Any], allow_nan: bool = True) -> None:
    _assert_path_suffix(path, (".json", ".json.gz"))
    assert isinstance(payload, dict), "JSON payload must be a dict."
    _ensure_parent_dir(path)
    with open_text_maybe_gzip(path, mode="wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=allow_nan)


def write_json_dict_atomic(path: PathLike, payload: dict[str, Any], allow_nan: bool = True) -> None:
    _assert_path_suffix(path, (".json", ".json.gz"))
    assert isinstance(payload, dict), "JSON payload must be a dict."
    _ensure_parent_dir(path)
    temp_path = _atomic_temp_path(path)
    with open_text_maybe_gzip(temp_path, mode="wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=allow_nan)
    os.replace(temp_path, str(path))


def write_json(path: PathLike, payload: dict[str, Any]) -> None:
    """Write `payload` as strict JSON: NumPy values become Python ones and non-finite floats null."""
    write_json_dict(path, normalize_json_payload(payload), allow_nan=False)


def normalize_json_payload(payload: Any) -> Any:
    """`payload` with NumPy arrays and scalars as Python values, tuples as lists, and NaN or infinity as None."""
    if isinstance(payload, dict):
        return {key: normalize_json_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [normalize_json_payload(value) for value in payload]
    if isinstance(payload, np.ndarray):
        return normalize_json_payload(payload.tolist())  # (...) -> lists nested to the array's rank
    if isinstance(payload, np.integer):
        return int(payload)
    if isinstance(payload, (float, np.floating)):
        value = float(payload)
        return value if math.isfinite(value) else None
    return payload


def read_pickle_dict(path: PathLike) -> dict[str, Any]:
    _assert_path_suffix(path, (".pkl", ".pkl.gz"))
    with open_binary_maybe_gzip(path, mode="rb") as handle:
        payload = pickle.load(handle)
    assert isinstance(payload, dict), "Pickle payload must be a dict."
    return payload


def write_pickle_dict(path: PathLike, payload: dict[str, Any], compresslevel: int | None = None) -> None:
    _assert_path_suffix(path, (".pkl", ".pkl.gz"))
    assert isinstance(payload, dict), "Pickle payload must be a dict."
    _ensure_parent_dir(path)
    with open_binary_maybe_gzip(path, mode="wb", compresslevel=compresslevel) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_pickle_dict_atomic(path: PathLike, payload: dict[str, Any], compresslevel: int | None = None) -> None:
    _assert_path_suffix(path, (".pkl", ".pkl.gz"))
    assert isinstance(payload, dict), "Pickle payload must be a dict."
    _ensure_parent_dir(path)
    temp_path = _atomic_temp_path(path)
    with open_binary_maybe_gzip(temp_path, mode="wb", compresslevel=compresslevel) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, str(path))


def read_torch_dict(
    path: PathLike,
    map_location: str | torch.device | None = None,
    weights_only: bool | None = None,
) -> dict[str, Any]:
    import torch  # Only the PyTorch readers and writers need it, so the rest works without it.

    _assert_path_suffix(path, (".pt", ".pth"))
    # An arbitrary mapping; each tensor leaf keeps its stored shape (...).
    payload = torch.load(path, map_location=map_location, weights_only=weights_only)
    assert isinstance(payload, dict), "Torch payload must be a dict."
    return payload


def write_torch_dict(path: PathLike, payload: dict[str, Any]) -> None:
    import torch

    _assert_path_suffix(path, (".pt", ".pth"))
    assert isinstance(payload, dict), "Torch payload must be a dict."
    _ensure_parent_dir(path)
    torch.save(payload, path)


def _assert_path_suffix(path: PathLike, allowed_suffixes: Sequence[str]) -> None:
    assert str(path).lower().endswith(tuple(allowed_suffixes)), (
        f"Expected file extension in {tuple(allowed_suffixes)}, got: {path}"
    )


def _ensure_parent_dir(path: PathLike) -> None:
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _atomic_temp_path(path: PathLike) -> str:
    """The sibling an atomic write goes to first. A gzipped target keeps `.gz` last."""
    path_str = str(path)
    if is_gzip_path(path_str):
        return f"{path_str[:-3]}.tmp.gz"
    return f"{path_str}.tmp"


__all__ = [
    "atomic_replace",
    "normalize_json_payload",
    "read_json_dict",
    "read_pickle_dict",
    "read_torch_dict",
    "write_bytes_atomic",
    "write_json",
    "write_json_atomic",
    "write_json_dict",
    "write_json_dict_atomic",
    "write_pickle_dict",
    "write_pickle_dict_atomic",
    "write_text_atomic",
    "write_torch_dict",
]
