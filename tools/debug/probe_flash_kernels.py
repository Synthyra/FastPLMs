"""Probe the two precompiled Hugging Face FlashAttention repositories."""

from __future__ import annotations

import inspect
import json
from importlib.metadata import version
from pathlib import Path
from kernels import get_kernel_variants, has_kernel

from fastplms.attention._kernel_lock import load_locked_kernel
from fastplms.registry import get_model_registry


API_NAMES = (
    "fwd",
    "varlen_fwd",
    "flash_attn_func",
    "flash_attn_varlen_func",
)


def _variant_summary(decisions: list[object]) -> dict[str, object]:
    """Keep the diagnostic concise while preserving every accepted variant."""
    return {
        "accepted": [
            repr(getattr(decision, "variant", decision))
            for decision in decisions
            if type(decision).__name__ == "VariantAccepted"
        ],
        "rejected_count": sum(
            type(decision).__name__ == "VariantRejected" for decision in decisions
        ),
    }


def main() -> None:
    result = {
        "kernels_version": version("kernels"),
        "api_signatures": {
            "get_kernel_variants": str(inspect.signature(get_kernel_variants)),
            "has_kernel": str(inspect.signature(has_kernel)),
        },
        "repositories": {},
    }
    for spec in get_model_registry().attention_kernels.values():
        repository = spec.repository
        available = has_kernel(
            repository,
            revision=spec.revision,
        )
        compatible_variants = get_kernel_variants(
            repository,
            revision=spec.revision,
        )
        try:
            kernel = load_locked_kernel(repository, spec.revision)
        except Exception as error:
            result["repositories"][repository] = {
                "error": f"{type(error).__name__}: {error}",
                "revision": spec.revision,
                "version": spec.version,
                "has_kernel": available,
                "variants": _variant_summary(compatible_variants),
            }
            continue
        result["repositories"][repository] = {
            "revision": spec.revision,
            "version": spec.version,
            "has_kernel": available,
            "variants": _variant_summary(compatible_variants),
            "module_file": str(Path(kernel.__file__).resolve()),
            "api": {name: callable(getattr(kernel, name, None)) for name in API_NAMES},
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
