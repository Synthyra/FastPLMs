"""The two JSON text forms FastPLMs writes: compact for hashing, indented for files people read.

This file exists twice, byte for byte: here and as ``features/json_files.py``. ``features`` loads as a
standalone package (``foundry.embedding.private_store``) and imports nothing outside itself, so it cannot
reach this module. ``tests/tier1_unit/test_features_package_is_self_contained.py`` fails when the two differ.
"""

from __future__ import annotations

import json

from typing import Any


def compact_json(value: Any, *, ensure_ascii: bool = True, allow_nan: bool = True) -> str:
    """Serialize with sorted keys and no whitespace, the form a digest or an identity is taken over."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
    )


def indented_json(
    value: Any,
    *,
    ensure_ascii: bool = True,
    allow_nan: bool = True,
    sort_keys: bool = True,
) -> str:
    """Serialize with two-space indentation and one trailing newline, the form of a stored JSON file."""

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            allow_nan=allow_nan,
        )
        + "\n"
    )
