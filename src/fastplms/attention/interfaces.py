"""Transformers-compatible attention selection for FastPLMs models."""

from __future__ import annotations

import torch
from collections.abc import Mapping
from functools import partial
from typing import Any
from transformers import AttentionInterface, AttentionMaskInterface

from ._core import (
    AttentionBackend,
    get_attn_implementation,
    kernels_flash_attention_func,
    resolve_attention_backend,
    set_config_attn_implementation,
)
from ._kernel_lock import require_kernels_package


def _kernels_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    implementation: str,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run one canonical FlashAttention backend through Hugging Face kernels.

    Transformers attention functions receive Q, K, and V with shape
    (b, h, l, d) and return an output with shape (b, l, h, d). The shared
    FastPLMs kernel adapter uses the latter layout internally.
    """

    # query, key, value: (b, h, l, d); attention_mask: (b, l) or None
    dropout = float(kwargs.get("dropout", 0.0) or 0.0)
    if module.training and dropout:
        raise RuntimeError(
            "Hugging Face kernels FlashAttention is inference-only when attention dropout "
            "is nonzero. Use SDPA for this training configuration."
        )
    causal = bool(kwargs.get("is_causal", getattr(module, "is_causal", False)))
    softmax_scale = kwargs.get("scaling")
    output = kernels_flash_attention_func(
        query_states=query.transpose(1, 2).contiguous(),  # (b, l, h, d)
        key_states=key.transpose(1, 2).contiguous(),  # (b, l, h, d)
        value_states=value.transpose(1, 2).contiguous(),  # (b, l, h, d)
        attention_mask_2d=attention_mask,
        causal=causal,
        softmax_scale=softmax_scale,
        implementation=implementation,
    )  # (b, l, h, d)
    return output, None  # (b, l, h, d), None


# Keep FastPLMs' kernels-only adapters local to this registry instance.
# ``GeneralInterface.register`` updates Transformers' class-wide mapping, so
# using it here would replace the canonical FlashAttention handlers for every
# model in the process, including models unrelated to FastPLMs.
FASTPLMS_ATTENTION_FUNCTIONS = AttentionInterface()
FASTPLMS_ATTENTION_MASKS = AttentionMaskInterface()
FASTPLMS_ATTENTION_FUNCTIONS["flash_attention_2"] = partial(
    _kernels_attention_forward,
    implementation="flash_attention_2",
)
FASTPLMS_ATTENTION_FUNCTIONS["flash_attention_3"] = partial(
    _kernels_attention_forward,
    implementation="flash_attention_3",
)
for _flash_name in ("flash_attention_2", "flash_attention_3"):
    FASTPLMS_ATTENTION_MASKS[_flash_name] = FASTPLMS_ATTENTION_MASKS[_flash_name]


class FastPLMsAttentionMixin:
    """Synchronize Transformers attention selection with custom model layers.

    Model families retain their checkpoint parameter names. Only runtime
    attributes are updated when ``set_attn_implementation`` is called.
    """

    _supports_sdpa = True
    _supports_flex_attn = True
    # Transformers 5.13 uses the singular flag during model construction. A
    # family opts in only when its manifest entry advertises at least one of
    # the two FastPLMs kernels-only FlashAttention implementations.
    _supports_flash_attn = False
    _supports_flash_attn_2 = False
    _supports_flash_attn_3 = False
    _fastplms_attention_implementations = (
        "eager",
        "sdpa",
        "flex_attention",
    )

    def _validate_attention_name(self, implementation: str) -> None:
        if implementation not in self._fastplms_attention_implementations:
            raise ValueError(
                f"{type(self).__name__} does not support {implementation!r}; expected one of "
                f"{self._fastplms_attention_implementations}."
            )

    def _check_and_adjust_attn_implementation(
        self,
        attn_implementation: str | None,
        is_init_check: bool = False,
        allow_all_kernels: bool = False,
    ) -> str:
        """Resolve attention without invoking Transformers' source-Flash probe.

        The standard ``flash_attention_2`` and ``flash_attention_3`` names are
        retained for the Transformers API, but FastPLMs resolves them only
        through the exact Hugging Face ``kernels`` artifacts pinned by
        ``models.toml``. Repository-qualified or otherwise external kernels
        are never accepted through this model hook.
        """

        if allow_all_kernels:
            raise ValueError("FastPLMs does not load external attention kernels.")
        if attn_implementation is None:
            return super()._check_and_adjust_attn_implementation(
                None,
                is_init_check=is_init_check,
                allow_all_kernels=False,
            )

        self._validate_attention_name(attn_implementation)
        if attn_implementation in {"flash_attention_2", "flash_attention_3"}:
            if not self._supports_flash_attn:
                raise ValueError(
                    f"{type(self).__name__} does not advertise kernels-only FlashAttention."
                )
            # Validate the lightweight Python dependency here, but defer binary
            # download and import until Q, K, and V have passed the CUDA gate.
            require_kernels_package()
            return attn_implementation

        return super()._check_and_adjust_attn_implementation(
            attn_implementation,
            is_init_check=is_init_check,
            allow_all_kernels=False,
        )

    def __init__(self, config, *args: Any, **kwargs: Any) -> None:
        sentinel = object()
        internal = getattr(config, "_attn_implementation_internal", sentinel)
        canonical = (
            getattr(config, "_attn_implementation", None) if internal is sentinel else internal
        )
        legacy = getattr(config, "attn_backend", None)
        requested = canonical if canonical is not None else legacy
        if requested is not None:
            if not isinstance(requested, str):
                raise TypeError(
                    "The configured attention implementation must be a string or None; "
                    f"received {type(requested).__name__}."
                )
            self._validate_attention_name(requested)
            # ``PreTrainedModel.__init__`` resolves a missing Transformers
            # implementation to the family default.  Legacy FastPLMs configs
            # persist their explicit choice in ``attn_backend``, so forward it
            # into the canonical Transformers field before the base class can
            # replace it with SDPA.  A non-None canonical value still wins,
            # including an explicit ``attn_implementation=...`` load override.
            if canonical is None and legacy is not None:
                set_config_attn_implementation(config, legacy)
        super().__init__(config, *args, **kwargs)
        # Transformers resolves an unspecified implementation during the base
        # model initialization. Synchronize that choice before family layers
        # are constructed.
        resolved = get_attn_implementation(config)
        self._validate_attention_name(resolved)
        set_config_attn_implementation(config, resolved)

    def set_attn_implementation(
        self,
        attn_implementation: str | Mapping[str, str],
        allow_all_kernels: bool = False,
    ) -> None:
        """Select an advertised backend and update every instantiated layer."""
        if isinstance(attn_implementation, Mapping):
            if set(attn_implementation) == {""}:
                attn_implementation = attn_implementation[""]
            else:
                raise ValueError(
                    "FastPLMs models have one attention backbone; pass a string or {'': name}."
                )
        resolved_name = self._check_and_adjust_attn_implementation(
            attn_implementation,
            is_init_check=False,
            allow_all_kernels=allow_all_kernels,
        )
        set_config_attn_implementation(self.config, resolved_name)
        resolved = resolve_attention_backend(resolved_name)
        for module in self.modules():
            if module is self:
                continue
            for attribute in ("attn_backend", "attention_backend", "_attn_backend"):
                if attribute not in module.__dict__:
                    continue
                current = module.__dict__[attribute]
                module.__dict__[attribute] = (
                    resolved if isinstance(current, AttentionBackend) else resolved_name
                )


def validate_transformers_attention_interfaces() -> None:
    """Verify that Transformers exposes functions and masks for every backend.

    Transformers 5.13 registers these canonical names. The FastPLMs function
    overrides remain instance-local and do not replace process-global handlers.
    """
    function_registry = FASTPLMS_ATTENTION_FUNCTIONS
    mask_registry = FASTPLMS_ATTENTION_MASKS
    missing_functions = [
        name
        for name in (
            "sdpa",
            "flex_attention",
            "flash_attention_2",
            "flash_attention_3",
        )
        if name not in function_registry
    ]
    missing_masks = [
        name
        for name in (
            "eager",
            "sdpa",
            "flex_attention",
            "flash_attention_2",
            "flash_attention_3",
        )
        if name not in mask_registry
    ]
    if missing_functions or missing_masks:
        raise RuntimeError(
            "Transformers attention registry is incomplete: "
            f"functions={missing_functions}, masks={missing_masks}."
        )
