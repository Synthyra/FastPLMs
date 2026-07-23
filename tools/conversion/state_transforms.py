"""Pure, deterministic checkpoint state transforms declared by ``models.toml``.

The functions in this module operate only on in-memory tensor mappings. They
cannot download a checkpoint, authenticate to a service, or mutate a Hub
repository. Artifact assembly and safetensors sharding remain centralized in
``tools.artifacts.build``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping

import torch

StateDict = dict[str, torch.Tensor]
Transform = Callable[[Mapping[str, torch.Tensor], frozenset[str] | None], StateDict]


class StateTransformError(RuntimeError):
    """Raised when a declared state transform cannot be applied exactly."""


def _clone_tensor(key: str, value: object) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise StateTransformError(f"State entry {key!r} is not a tensor.")
    return value.detach().cpu().clone()


def _validate_expected(state: StateDict, expected_keys: frozenset[str] | None) -> None:
    if expected_keys is None:
        return
    actual = frozenset(state)
    missing = sorted(expected_keys - actual)
    unexpected = sorted(actual - expected_keys)
    if missing or unexpected:
        raise StateTransformError(
            "Transformed state keys do not match the expected model state. "
            f"Missing: {missing[:20]}; unexpected: {unexpected[:20]}."
        )


def _map_state(
    state: Mapping[str, torch.Tensor],
    key_mapper: Callable[[str], str | None],
    expected_keys: frozenset[str] | None,
) -> StateDict:
    if not state:
        raise StateTransformError("A checkpoint state dictionary cannot be empty.")
    transformed: StateDict = {}
    for key in sorted(state):
        if not isinstance(key, str) or not key:
            raise StateTransformError(f"Invalid state key: {key!r}.")
        mapped = key_mapper(key)
        if mapped is None:
            continue
        if mapped in transformed:
            raise StateTransformError(f"State-key collision while mapping {key!r} to {mapped!r}.")
        transformed[mapped] = _clone_tensor(key, state[key])
    _validate_expected(transformed, expected_keys)
    return transformed


def _identity(
    state: Mapping[str, torch.Tensor],
    expected_keys: frozenset[str] | None,
) -> StateDict:
    return _map_state(state, lambda key: key, expected_keys)


def _cast_floating(
    state: Mapping[str, torch.Tensor],
    expected_keys: frozenset[str] | None,
    dtype: torch.dtype,
) -> StateDict:
    transformed = _identity(state, expected_keys)
    return {
        key: value.to(dtype=dtype) if value.is_floating_point() else value
        for key, value in transformed.items()
    }


def _drop_unused_rotary_position_table(
    state: Mapping[str, torch.Tensor],
    expected_keys: frozenset[str] | None,
) -> StateDict:
    return _map_state(
        state,
        lambda key: None if key == "esm.embeddings.position_embeddings.weight" else key,
        expected_keys,
    )


def _esm2(
    state: Mapping[str, torch.Tensor],
    expected_keys: frozenset[str] | None,
) -> StateDict:
    """Map pinned fair-esm ESM2 names to the canonical FastPLMs schema."""

    if not state:
        raise StateTransformError("A checkpoint state dictionary cannot be empty.")
    keys = frozenset(state)
    official_schema = "embed_tokens.weight" in keys or any(
        key.startswith(("layers.", "emb_layer_norm_after.", "contact_head.")) for key in keys
    )
    canonical_schema = any(key.startswith("esm.") for key in keys) or any(
        key.startswith("lm_head.decoder.") for key in keys
    )
    if official_schema and canonical_schema:
        raise StateTransformError("ESM2 checkpoint mixes official and canonical parameter schemas.")
    if canonical_schema:
        return _identity(state, expected_keys)

    transformed: StateDict = {}

    def store(source_key: str, target_key: str) -> None:
        if target_key in transformed:
            raise StateTransformError(
                f"State-key collision while mapping {source_key!r} to {target_key!r}."
            )
        transformed[target_key] = _clone_tensor(source_key, state[source_key])

    projection_names = {"q_proj": "query", "k_proj": "key", "v_proj": "value"}
    for key in sorted(state):
        target: str | None = None
        if key == "embed_tokens.weight":
            target = "esm.embeddings.word_embeddings.weight"
        elif key.startswith("layers."):
            match = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
            if match is None:
                raise StateTransformError(f"Unrecognized official ESM2 layer key: {key!r}.")
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
            store(key, "lm_head.bias")
            continue
        elif key.startswith("lm_head."):
            target = key

        if target is None:
            raise StateTransformError(f"Unrecognized official ESM2 state key: {key!r}.")
        store(key, target)

    _validate_expected(transformed, expected_keys)
    return transformed


def _esmc_key(key: str) -> str | None:
    if key.endswith("._extra_state"):
        return None
    if key.startswith("esmc."):
        key = key[len("esmc.") :]
    if key.startswith("lm_head."):
        key = f"sequence_head.{key[len('lm_head.') :]}"
    replacements = (
        (".attn.layernorm_qkv.layer_norm_bias", ".attn.layernorm_qkv.0.bias"),
        (".attn.layernorm_qkv.layer_norm_weight", ".attn.layernorm_qkv.0.weight"),
        (".attn.layernorm_qkv.weight", ".attn.layernorm_qkv.1.weight"),
        (".ffn.layer_norm_bias", ".ffn.0.bias"),
        (".ffn.layer_norm_weight", ".ffn.0.weight"),
        (".ffn.fc1_weight", ".ffn.1.weight"),
        (".ffn.fc2_weight", ".ffn.3.weight"),
    )
    for old, new in replacements:
        key = key.replace(old, new)
    return key


def _esmc(
    state: Mapping[str, torch.Tensor],
    expected_keys: frozenset[str] | None,
) -> StateDict:
    return _map_state(state, _esmc_key, expected_keys)


def _esm3(
    state: Mapping[str, torch.Tensor],
    expected_keys: frozenset[str] | None,
) -> StateDict:
    transformed = _map_state(
        state,
        lambda key: key if key.startswith("esm3.") else f"esm3.{key}",
        expected_keys,
    )
    return {
        key: value.to(dtype=torch.float32) if value.is_floating_point() else value
        for key, value in transformed.items()
    }


def _e1(
    state: Mapping[str, torch.Tensor],
    expected_keys: frozenset[str] | None,
) -> StateDict:
    return _cast_floating(state, expected_keys, torch.bfloat16)


def _ankh_t5(
    state: Mapping[str, torch.Tensor],
    expected_keys: frozenset[str] | None,
) -> StateDict:
    """Preserve ANKH's complete official encoder-decoder T5 state exactly."""

    keys = frozenset(state)
    required_exact = {
        "shared.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "lm_head.weight",
    }
    missing = sorted(required_exact - keys)
    has_encoder_block = any(key.startswith("encoder.block.") for key in keys)
    has_decoder_block = any(key.startswith("decoder.block.") for key in keys)
    has_cross_attention = any(".EncDecAttention." in key for key in keys)
    if missing or not has_encoder_block or not has_decoder_block or not has_cross_attention:
        raise StateTransformError(
            "ANKH publication requires the complete official T5 state, including shared, "
            "encoder, decoder, cross-attention, and language-model-head parameters. "
            f"Missing required keys: {missing}; encoder blocks: {has_encoder_block}; "
            f"decoder blocks: {has_decoder_block}; cross-attention: {has_cross_attention}."
        )
    return _identity(state, expected_keys)


def _boltz2(
    state: Mapping[str, torch.Tensor],
    expected_keys: frozenset[str] | None,
) -> StateDict:
    if expected_keys is None:
        raise StateTransformError(
            "boltz2_inference_core_v1 requires the expected FastPLMs core keys."
        )
    transformed: StateDict = {}
    unsupported: list[str] = []
    for source_key in sorted(state):
        if source_key.startswith("ema."):
            continue
        key = source_key
        if key.startswith("model."):
            key = key[len("model.") :]
        if key.startswith("module."):
            key = key[len("module.") :]
        canonical = key if key.startswith("core.") else f"core.{key}"
        if canonical not in expected_keys:
            bare = canonical[len("core.") :]
            if bare.startswith(("template_module.", "bfactor_module.")):
                continue
            unsupported.append(source_key)
            continue
        if canonical in transformed:
            raise StateTransformError(
                f"State-key collision while mapping {source_key!r} to {canonical!r}."
            )
        transformed[canonical] = _clone_tensor(source_key, state[source_key])
    if unsupported:
        raise StateTransformError(
            f"Boltz2 checkpoint contains undeclared non-inference parameters: {unsupported[:20]}."
        )
    _validate_expected(transformed, expected_keys)
    return transformed


_ESMFOLD_DERIVED_BUFFERS = frozenset(
    {
        "positional_encoding._float_tensor",
        "trunk.structure_module.atom_mask",
        "trunk.structure_module.default_frames",
        "trunk.structure_module.group_idx",
        "trunk.structure_module.lit_positions",
    }
)


def _esmfold(
    state: Mapping[str, torch.Tensor],
    expected_keys: frozenset[str] | None,
) -> StateDict:
    """Map native Meta ESMFold and prior canonical mirrors to package state."""

    if not state:
        raise StateTransformError("ESMFold checkpoint state cannot be empty.")
    canonical = any(key.startswith("esm.encoder.") for key in state)
    if canonical:
        if expected_keys is not None:
            expected_keys = frozenset(
                key
                for key in expected_keys
                if key not in _ESMFOLD_DERIVED_BUFFERS
                and not key.startswith(("mlm_head.", "esm.contact_head."))
            )
        transformed = _map_state(
            state,
            lambda key: (
                None
                if key in _ESMFOLD_DERIVED_BUFFERS
                or key.startswith(("mlm_head.", "esm.contact_head."))
                else key
            ),
            None,
        )
        _validate_expected(transformed, expected_keys)
        return transformed

    folding: StateDict = {}
    native_esm: StateDict = {}
    for key in sorted(state):
        if key in _ESMFOLD_DERIVED_BUFFERS:
            continue
        if key.startswith("esm."):
            inner = key.removeprefix("esm.")
            if inner.startswith(("lm_head.", "contact_head.")):
                continue
            native_esm[inner] = _clone_tensor(key, state[key])
            continue
        folding[key] = _clone_tensor(key, state[key])
    mapped_esm = _esm2(native_esm, None)
    overlap = sorted(set(folding).intersection(mapped_esm))
    if overlap:
        raise StateTransformError(f"ESMFold state-key collision: {overlap[:20]}.")
    transformed = {**folding, **mapped_esm}
    _validate_expected(transformed, expected_keys)
    return transformed


_TRANSFORMS: dict[str, Transform] = {
    "identity": _identity,
    "esm2_hf_to_fastplms_v1": _esm2,
    "esmc_to_fastplms_v1": _esmc,
    "esm3_to_fastplms_v1": _esm3,
    "e1_to_fastplms_v1": _e1,
    "dplm_to_fastplms_v1": _drop_unused_rotary_position_table,
    "dplm2_to_fastplms_v1": _drop_unused_rotary_position_table,
    "ankh_t5_to_fastplms_v1": _ankh_t5,
    "boltz2_inference_core_v1": _boltz2,
    "esmfold_meta_to_fastplms_v1": _esmfold,
}


def available_state_transforms() -> tuple[str, ...]:
    """Return stable transform identifiers accepted by the local converter."""

    return tuple(sorted(_TRANSFORMS))


def apply_state_transform(
    transform_id: str,
    state: Mapping[str, torch.Tensor],
    *,
    expected_keys: Iterable[str] | None = None,
) -> StateDict:
    """Apply one manifest-declared transform without mutating ``state``."""

    try:
        transform = _TRANSFORMS[transform_id]
    except KeyError as error:
        raise StateTransformError(f"Unknown state transform: {transform_id!r}.") from error
    expected = frozenset(expected_keys) if expected_keys is not None else None
    return transform(state, expected)


__all__ = [
    "StateDict",
    "StateTransformError",
    "apply_state_transform",
    "available_state_transforms",
]
