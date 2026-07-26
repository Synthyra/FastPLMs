"""Transformers-compatible attention selection for ESMFold2's ESMC backbone."""

from __future__ import annotations

from collections.abc import Mapping

from ...attention import FastPLMsAttentionMixin, get_attn_implementation


class ESMFold2AttentionMixin(FastPLMsAttentionMixin):
    """Route the outer Transformers attention API into the loaded ESMC model."""

    _supports_attention_backend = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_flash_attn_2 = False
    _supports_flash_attn_3 = False
    _fastplms_attention_implementations = (
        "eager",
        "sdpa",
        "flex_attention",
    )

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        config.esmc_attn_backend = get_attn_implementation(config)

    def set_attn_implementation(
        self,
        attn_implementation: str | Mapping[str, str],
        allow_all_kernels: bool = False,
    ) -> None:
        """Set one canonical backend on ESMFold2 and its loaded ESMC model."""

        if allow_all_kernels:
            raise ValueError(
                "ESMFold2 accepts only its declared built-in attention backends; "
                "external attention kernels are not supported."
            )
        super().set_attn_implementation(attn_implementation)
        resolved = get_attn_implementation(self.config)
        self.config.esmc_attn_backend = resolved
        esmc = getattr(self, "_esmc", None)
        if esmc is not None:
            esmc.set_attn_implementation(resolved)


__all__ = ["ESMFold2AttentionMixin"]
