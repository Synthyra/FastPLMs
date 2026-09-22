"""Transformers-compatible attention selection for FastPLMs models."""

from __future__ import annotations

import torch
from collections.abc import Mapping
from functools import partial
from typing import Any
from transformers import AttentionInterface, AttentionMaskInterface

from ._auto import (
    AUTO_ATTENTION,
    AttentionResolution,
    attention_execution_context,
    deferred_resolution,
    needs_execution_context,
    provisional_implementation,
    resolve_auto_attention,
)
from ._core import (
    AttentionBackend,
    canonical_checkpoint_attention_backend,
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
    # Preference order for ``attn_implementation="auto"``, mirrored from the
    # family's ``attention_auto_order`` in models.toml. Empty rejects the request.
    _fastplms_attention_auto_order: tuple[str, ...] = ()
    # Supplied by the Transformers ``PreTrainedModel`` that follows this mixin in every
    # MRO. Each family's configuration class adds the ``attn_backend`` field this mixin
    # reads, and they share no base that declares it, so the boundary is dynamic.
    config: Any

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
        stored = getattr(config, "_attn_implementation", None) if internal is sentinel else internal
        legacy = getattr(config, "attn_backend", None)
        requested = stored if stored is not None else legacy
        auto_requested = requested == AUTO_ATTENTION
        serialized_backend: str | None = None
        if auto_requested:
            # The configuration never holds ``auto``. Family layers are built on the
            # provisional implementation, and a saved copy keeps the backend that a
            # named load of the same checkpoint would have stored.
            requested = provisional_implementation(self._require_attention_auto_order())
            serialized_backend = legacy if legacy not in (None, AUTO_ATTENTION) else requested
            stored = None
        if requested is not None:
            if not isinstance(requested, str):
                raise TypeError(
                    "The configured attention implementation must be a string or None; "
                    f"received {type(requested).__name__}."
                )
            # A serialized configuration can name a backend with the historical
            # spelling used by the official source it was converted from. That
            # names the same implementation, so translate it here rather than
            # rejecting a checkpoint that asked for an implementation FastPLMs has.
            canonical = canonical_checkpoint_attention_backend(requested)
            self._validate_attention_name(canonical)
            # ``PreTrainedModel.__init__`` resolves a missing Transformers
            # implementation to the family default.  Legacy FastPLMs configs
            # persist their explicit choice in ``attn_backend``, so forward it
            # into the canonical Transformers field before the base class can
            # replace it with SDPA.  A stored canonical value already agrees and
            # is left untouched, including an explicit
            # ``attn_implementation=...`` load override.
            if canonical != stored:
                set_config_attn_implementation(config, canonical)
        super().__init__(config, *args, **kwargs)
        # Transformers resolves an unspecified implementation during the base
        # model initialization. Synchronize that choice before family layers
        # are constructed.
        resolved = get_attn_implementation(config)
        self._validate_attention_name(resolved)
        set_config_attn_implementation(config, resolved)
        if auto_requested:
            self.__dict__["_fastplms_serialized_attn_backend"] = serialized_backend
            self._begin_auto_attention()

    def _require_attention_auto_order(self) -> tuple[str, ...]:
        order = self._fastplms_attention_auto_order
        if not order:
            raise ValueError(
                f"{type(self).__name__} does not support attn_implementation='auto'; "
                f"request one of {self._fastplms_attention_implementations}."
            )
        return order

    @property
    def attention_resolution(self) -> AttentionResolution | None:
        """The record of an ``auto`` request, or None when a backend was named."""
        return self.__dict__.get("_fastplms_attention_resolution")

    def _begin_auto_attention(self) -> None:
        """Resolve now when no candidate needs a device, else at the first forward."""
        order = self._require_attention_auto_order()
        self._cancel_pending_auto_attention()
        if not needs_execution_context(order):
            resolution = resolve_auto_attention(order, None)
            self._apply_attn_implementation(resolution.resolved)
            self.__dict__["_fastplms_attention_resolution"] = resolution
            return
        resolution = deferred_resolution(order)
        self._apply_attn_implementation(resolution.resolved)
        self.__dict__["_fastplms_attention_resolution"] = resolution
        # The first forward runs inside the caller's autocast context, which is
        # what decides FlashAttention eligibility for FP32 parameters.
        self.__dict__["_fastplms_auto_attention_hook"] = (
            self._as_module().register_forward_pre_hook(_resolve_auto_attention_before_forward)
        )

    def _as_module(self) -> torch.nn.Module:
        if not isinstance(self, torch.nn.Module):
            raise TypeError(
                f"{type(self).__name__} must be a torch.nn.Module to defer attention selection."
            )
        return self

    def _cancel_pending_auto_attention(self) -> None:
        hook = self.__dict__.pop("_fastplms_auto_attention_hook", None)
        if hook is not None:
            hook.remove()

    def resolve_attn_implementation(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> AttentionResolution:
        """Settle a pending ``auto`` request for the device and dtype of the next forward.

        The first forward does this by itself. Call it earlier, for example before
        ``torch.compile`` or before fingerprinting an embedding run, and pass
        ``dtype`` when the forward will run under an autocast context that is not
        active yet. A settled request returns its record unchanged.
        """
        resolution = self.attention_resolution
        if resolution is None:
            raise RuntimeError(
                f"{type(self).__name__} was not configured with attn_implementation='auto'."
            )
        if not resolution.deferred:
            return resolution
        self._cancel_pending_auto_attention()
        resolution = resolve_auto_attention(
            self._require_attention_auto_order(),
            attention_execution_context(self._as_module(), device=device, dtype=dtype),
        )
        self._apply_attn_implementation(resolution.resolved)
        self.__dict__["_fastplms_attention_resolution"] = resolution
        return resolution

    def save_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        """Save without the machine-specific outcome of an ``auto`` request."""
        # ``save_pretrained`` comes from the ``PreTrainedModel`` later in the MRO.
        if self.attention_resolution is None:
            return super().save_pretrained(*args, **kwargs)  # type: ignore[misc]
        selected_backend = self.config.attn_backend
        self.config.attn_backend = self.__dict__["_fastplms_serialized_attn_backend"]
        try:
            return super().save_pretrained(*args, **kwargs)  # type: ignore[misc]
        finally:
            self.config.attn_backend = selected_backend

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
        if attn_implementation == AUTO_ATTENTION:
            if allow_all_kernels:
                raise ValueError("FastPLMs does not load external attention kernels.")
            self.__dict__.setdefault(
                "_fastplms_serialized_attn_backend", getattr(self.config, "attn_backend", None)
            )
            self._begin_auto_attention()
            return
        # A named request replaces any earlier automatic selection.
        self._cancel_pending_auto_attention()
        self.__dict__.pop("_fastplms_attention_resolution", None)
        self.__dict__.pop("_fastplms_serialized_attn_backend", None)
        self._apply_attn_implementation(attn_implementation, allow_all_kernels)

    def _apply_attn_implementation(
        self, attn_implementation: str, allow_all_kernels: bool = False
    ) -> None:
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


# Selection reads the manifest, can load a kernel, and rewrites layer attributes.
# It runs eagerly so that a compiled model never traces it.
@torch.compiler.disable  # type: ignore[untyped-decorator]
def _resolve_auto_attention_before_forward(
    module: torch.nn.Module, _arguments: tuple[Any, ...]
) -> None:
    if not isinstance(module, FastPLMsAttentionMixin):
        raise TypeError("The automatic attention hook belongs on a FastPLMs model.")
    module.resolve_attn_implementation()


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
