"""Independent deterministic transforms for official checkpoint state.

The transform name comes exclusively from ``models.toml``. A missing transform
is a compliance failure, not an invitation to compare only intersecting keys.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

import torch

State = Mapping[str, torch.Tensor]
Transform = Callable[[State], dict[str, torch.Tensor]]


def _identity(state: State) -> dict[str, torch.Tensor]:
    return dict(state)


def _esm2_fair_to_fastplms(state: State) -> dict[str, torch.Tensor]:
    """Map the pinned Meta ESM2 module names to the Hugging Face ESM schema."""

    mapped: dict[str, torch.Tensor] = {}
    projection_names = {"q_proj": "query", "k_proj": "key", "v_proj": "value"}
    for key, value in state.items():
        target: str | None = None
        if key == "embed_tokens.weight":
            target = "esm.embeddings.word_embeddings.weight"
        elif key.startswith("layers."):
            match = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
            if match is None:
                raise AssertionError(f"Unrecognized official ESM2 layer key: {key}")
            layer, suffix = match.groups()
            prefix = f"esm.encoder.layer.{layer}."
            if suffix == "self_attn.rot_emb.inv_freq":
                target = f"{prefix}attention.self.rotary_embeddings.inv_freq"
            for source_name, target_name in projection_names.items():
                if suffix.startswith(f"self_attn.{source_name}."):
                    parameter = suffix.rsplit(".", 1)[-1]
                    target = f"{prefix}attention.self.{target_name}.{parameter}"
                    break
            if suffix.startswith("self_attn.out_proj."):
                parameter = suffix.rsplit(".", 1)[-1]
                target = f"{prefix}attention.output.dense.{parameter}"
            elif suffix.startswith("self_attn_layer_norm."):
                parameter = suffix.rsplit(".", 1)[-1]
                target = f"{prefix}attention.LayerNorm.{parameter}"
            elif suffix.startswith("fc1."):
                parameter = suffix.rsplit(".", 1)[-1]
                target = f"{prefix}intermediate.dense.{parameter}"
            elif suffix.startswith("fc2."):
                parameter = suffix.rsplit(".", 1)[-1]
                target = f"{prefix}output.dense.{parameter}"
            elif suffix.startswith("final_layer_norm."):
                parameter = suffix.rsplit(".", 1)[-1]
                target = f"{prefix}LayerNorm.{parameter}"
        elif key.startswith("emb_layer_norm_after."):
            target = f"esm.encoder.{key}"
        elif key.startswith("contact_head."):
            target = f"esm.{key}"
        elif key == "lm_head.weight":
            target = "lm_head.decoder.weight"
        elif key == "lm_head.bias":
            mapped["lm_head.bias"] = value
            mapped["lm_head.decoder.bias"] = value
            continue
        elif key.startswith("lm_head."):
            target = key

        if target is None:
            raise AssertionError(f"Unrecognized official ESM2 state key: {key}")
        mapped[target] = value
    return mapped


def _esmc_to_fastplms(state: State) -> dict[str, torch.Tensor]:
    """Apply the declared ESMC checkpoint-key normalization exactly once."""

    replacements = (
        (".attn.layernorm_qkv.layer_norm_bias", ".attn.layernorm_qkv.0.bias"),
        (".attn.layernorm_qkv.layer_norm_weight", ".attn.layernorm_qkv.0.weight"),
        (".attn.layernorm_qkv.weight", ".attn.layernorm_qkv.1.weight"),
        (".ffn.layer_norm_bias", ".ffn.0.bias"),
        (".ffn.layer_norm_weight", ".ffn.0.weight"),
        (".ffn.fc1_weight", ".ffn.1.weight"),
        (".ffn.fc2_weight", ".ffn.3.weight"),
    )
    mapped: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if raw_key.endswith("._extra_state"):
            continue
        key = raw_key.removeprefix("esmc.")
        if key.startswith("lm_head."):
            key = f"sequence_head.{key.removeprefix('lm_head.')}"
        for source, target in replacements:
            key = key.replace(source, target)
        mapped[key] = value
    return mapped


_ESMFOLD_DERIVED_BUFFERS = frozenset(
    {
        "positional_encoding._float_tensor",
        "trunk.structure_module.atom_mask",
        "trunk.structure_module.default_frames",
        "trunk.structure_module.group_idx",
        "trunk.structure_module.lit_positions",
    }
)


def _esmfold_meta_to_fastplms(state: State) -> dict[str, torch.Tensor]:
    """Map Meta ESMFold state and omit heads unused by structure inference."""

    if any(key.startswith("esm.encoder.") for key in state):
        return {
            key: value
            for key, value in state.items()
            if key not in _ESMFOLD_DERIVED_BUFFERS
            and not key.startswith(("mlm_head.", "esm.contact_head."))
        }

    folding: dict[str, torch.Tensor] = {}
    native_esm: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key in _ESMFOLD_DERIVED_BUFFERS:
            continue
        if key.startswith("esm."):
            inner = key.removeprefix("esm.")
            if inner.startswith(("lm_head.", "contact_head.")):
                continue
            native_esm[inner] = value
        else:
            folding[key] = value
    mapped_esm = _esm2_fair_to_fastplms(native_esm)
    overlap = set(folding).intersection(mapped_esm)
    if overlap:
        raise AssertionError(f"ESMFold state-key collision: {sorted(overlap)[:20]}")
    return {**folding, **mapped_esm}


TRANSFORMS: dict[str, Transform] = {
    "identity": _identity,
    "esm2_hf_to_fastplms_v1": _esm2_fair_to_fastplms,
    "esmc_to_fastplms_v1": _esmc_to_fastplms,
    "esm3_to_fastplms_v1": _identity,
    "e1_to_fastplms_v1": _identity,
    "dplm_to_fastplms_v1": _identity,
    "dplm2_to_fastplms_v1": _identity,
    "ankh_t5_to_fastplms_v1": _identity,
    "esmfold_meta_to_fastplms_v1": _esmfold_meta_to_fastplms,
}


def transform_state(name: str, state: State) -> dict[str, torch.Tensor]:
    """Return transformed state or fail if the manifest names no implementation."""

    try:
        transform = TRANSFORMS[name]
    except KeyError as error:
        raise AssertionError(f"No compliance state transform is registered for {name!r}") from error
    return transform(state)


def transform_parameter_names(name: str, parameter_name: str) -> tuple[str, ...]:
    """Map one parameter name, including intentionally duplicated alias names."""

    marker = torch.empty(0)
    transformed = transform_state(name, {parameter_name: marker})
    return tuple(transformed)
