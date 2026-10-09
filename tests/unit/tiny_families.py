"""One tiny configuration of every manifest family, small enough to build and save on a CPU in a second.

Each configuration keeps every module kind of its family (embedding, attention, feed-forward, heads) at a
width of a few units, so a state dictionary has the same key structure as a published checkpoint.
"""

from __future__ import annotations

from typing import Any

from fastplms.models.ankh.modeling_ankh import FastAnkhConfig
from fastplms.models.boltz.modeling_boltz2 import Boltz2Config
from fastplms.models.dplm.modeling_dplm import DPLMConfig
from fastplms.models.dplm2.modeling_dplm2 import DPLM2Config
from fastplms.models.e1.modeling_e1 import E1Config
from fastplms.models.esm2.modeling_fastesm import FastEsmConfig
from fastplms.models.esm3.modeling_esm3 import FastESM3Config
from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig, ESMplusplusModel
from fastplms.models.esmfold.modeling_fast_esmfold import FastEsmFoldConfig
from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config
from fastplms.models.esmfold2.modeling_esmfold2 import _ESMFold2ESMplusplusAdapter
from fastplms.models.esmfold2.modeling_esmfold2_common import NUM_RES_TYPES


def transformer_values(vocab_size: int) -> dict[str, Any]:
    return {
        "vocab_size": vocab_size,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 16,
        "pad_token_id": 1,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "mask_token_id": min(7, vocab_size - 1),
        "position_embedding_type": "rotary",
        "attn_backend": "eager",
        "num_labels": 3,
    }


def tiny_esmfold_config() -> FastEsmFoldConfig:
    return FastEsmFoldConfig(
        vocab_size=33,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
        pad_token_id=1,
        mask_token_id=32,
        position_embedding_type="rotary",
        is_folding_model=True,
        attn_backend="eager",
        esmfold_config={
            "fp16_esm": False,
            "bypass_lm": True,
            "lddt_head_hid_dim": 4,
            "trunk": {
                "num_blocks": 1,
                "sequence_state_dim": 8,
                "pairwise_state_dim": 4,
                "sequence_head_width": 4,
                "pairwise_head_width": 2,
                "position_bins": 4,
                "max_recycles": 1,
                "chunk_size": None,
                "structure_module": {
                    "sequence_dim": 8,
                    "pairwise_dim": 4,
                    "ipa_dim": 2,
                    "resnet_dim": 4,
                    "num_heads_ipa": 2,
                    "num_qk_points": 1,
                    "num_v_points": 1,
                    "dropout_rate": 0.0,
                    "num_blocks": 1,
                    "num_transition_layers": 1,
                    "num_resnet_blocks": 1,
                    "num_angles": 7,
                },
            },
        },
    )


def tiny_esmfold2_config(model_type: str = "release", **overrides: Any) -> ESMFold2Config:
    """The no-block ESMFold2 configuration of ``model_type``, with ``overrides`` replacing any configuration field."""

    atom_token_width = 8
    input_feature_width = atom_token_width // 2 + 2 * NUM_RES_TYPES + 1
    values: dict[str, Any] = dict(
        type=model_type,
        d_single=8,
        d_pair=8,
        num_loops=0,
        num_diffusion_samples=1,
        lm_d_model=8,
        lm_num_layers=1,
        inputs={
            "d_inputs": input_feature_width,
            "atom_encoder": {
                "d_atom": 8,
                "d_token": atom_token_width,
                "n_blocks": 0,
                "n_heads": 2,
                "swa_window_size": 32,
                "expansion_ratio": 2,
                "n_spatial_rope_pairs_per_axis": 1,
                "n_uid_rope_pairs": 1,
            },
        },
        folding_trunk={"n_layers": 0, "n_heads": 2, "dropout": 0.0},
        structure_head={
            "diffusion_module": {
                "c_atom": 8,
                "c_token": 8,
                "c_z": 8,
                "c_s_inputs": input_feature_width,
                "fourier_dim": 8,
                "atom_num_blocks": 0,
                "atom_num_heads": 2,
                "token_num_blocks": 0,
                "token_num_heads": 2,
                "transition_multiplier": 2,
            },
            "distogram_bins": 8,
            "inference_num_steps": 1,
        },
        confidence_head={
            "enabled": False,
            "folding_trunk": {"n_layers": 0, "n_heads": 2, "dropout": 0.0},
            "num_plddt_bins": 4,
            "num_pde_bins": 4,
            "num_pae_bins": 4,
            "distogram_bins": 8,
        },
        msa_encoder={"enabled": False},
        lm_encoder={"enabled": False, "n_layers": 0},
        parcae={"enabled": True, "min_steps": 1, "max_steps": 1, "coda_n_layers": 0},
    )
    return ESMFold2Config(**{**values, **overrides})


TINY_CONFIDENCE_HEAD: dict[str, Any] = {
    "enabled": True,
    "folding_trunk": {"n_layers": 1, "n_heads": 2, "dropout": 0.0},
    "num_plddt_bins": 4,
    "num_pde_bins": 4,
    "num_pae_bins": 4,
    "distogram_bins": 8,
}  # ``confidence_head`` override that turns the toy ESMFold2 head on, with one trunk block of toy width

TINY_MSA_ENCODER: dict[str, Any] = {
    "enabled": True,
    "d_msa": 8,
    "d_hidden": 2,
    "n_layers": 2,
    "n_heads_msa": 2,
    "msa_head_width": 4,
}  # ``msa_encoder`` override (with ``msa_conditioning=True``) that turns the toy ESMFold2 alignment encoder on


def tiny_esmfold2_backbone() -> _ESMFold2ESMplusplusAdapter:
    """A one-layer ESM++ over a 64-token vocabulary, wrapped as the toy ESMFold2 models expect their ESMC backbone."""

    return _ESMFold2ESMplusplusAdapter(ESMplusplusModel(tiny_esmc_config(vocab_size=64)).eval())


def tiny_esm2_config(**overrides: Any) -> FastEsmConfig:
    """The 16-token ESM2 configuration of the sequence contract tests, with ``overrides`` replacing any field."""

    values: dict[str, Any] = dict(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=16,
        pad_token_id=1,
        mask_token_id=5,
        num_labels=3,
        position_embedding_type="absolute",
        attn_backend="eager",
    )
    return FastEsmConfig(**{**values, **overrides})


def tiny_esmc_config(**overrides: Any) -> ESMplusplusConfig:
    """The 16-token ESM++ configuration of the sequence contract tests, with ``overrides`` replacing any field."""

    values: dict[str, Any] = dict(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        dropout=0.0,
        pad_token_id=1,
        mask_token_id=5,
        attn_backend="eager",
    )
    return ESMplusplusConfig(**{**values, **overrides})


def tiny_esm3_config(**overrides: Any) -> FastESM3Config:
    """The one-layer ESM3 configuration of the CPU contract tests, with ``overrides`` replacing any field."""

    values: dict[str, Any] = dict(
        hidden_size=8,
        num_attention_heads=2,
        num_vector_heads=2,
        num_hidden_layers=1,
        attn_backend="eager",
    )
    return FastESM3Config(**{**values, **overrides})


def dplm_values(vocab_size: int) -> dict[str, Any]:
    """Constructor arguments of a one-layer DPLM or DPLM2 configuration of width 32 over ``vocab_size`` tokens."""

    return {
        "vocab_size": vocab_size,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 64,
        "pad_token_id": 1,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "mask_token_id": 32,
        "position_embedding_type": "rotary",
        "attn_backend": "sdpa",
    }


def tiny_config(family_id: str) -> Any:
    if family_id == "esm2":
        values = transformer_values(16)
        values["position_embedding_type"] = "absolute"
        return FastEsmConfig(**values)
    if family_id == "esm_plusplus":
        return ESMplusplusConfig(
            vocab_size=16,
            hidden_size=8,
            num_attention_heads=2,
            num_hidden_layers=1,
            dropout=0.0,
            pad_token_id=1,
            mask_token_id=7,
            num_labels=3,
            attn_backend="eager",
        )
    if family_id == "esm3":
        return tiny_esm3_config()
    if family_id == "e1":
        config = E1Config(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_num_sequences=4,
            max_num_positions_within_seq=16,
            max_num_positions_global=16,
            attn_backend="sdpa",
            dtype="float32",
            num_labels=3,
        )
        config.use_cache = False
        return config
    if family_id == "dplm":
        return DPLMConfig(**transformer_values(16))
    if family_id == "dplm2":
        values = transformer_values(64)
        values["attn_backend"] = "sdpa"
        return DPLM2Config(**values)
    if family_id == "ankh":
        return FastAnkhConfig(
            vocab_size=16,
            d_model=8,
            d_kv=4,
            d_ff=16,
            num_heads=2,
            num_layers=1,
            num_decoder_layers=1,
            dropout_rate=0.0,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
            attn_backend="eager",
            use_cache=False,
            num_labels=3,
        )
    if family_id == "boltz2":
        return Boltz2Config(core_kwargs={"width": 3})
    if family_id == "esmfold":
        return tiny_esmfold_config()
    if family_id == "esmfold2":
        return tiny_esmfold2_config()
    raise AssertionError(f"Missing tiny configuration for manifest family {family_id!r}.")
