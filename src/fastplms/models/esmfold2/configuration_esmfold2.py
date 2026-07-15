# Copyright 2026 Biohub. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration schema for release and experimental ESMFold2 checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar, cast

from transformers.configuration_utils import PretrainedConfig

_ESMC_ATTENTION_IMPLEMENTATIONS = frozenset({"eager", "flex_attention", "sdpa"})
_ESMC_PRECISIONS = frozenset({"auto", "bf16", "fp32", "fp8"})


def _esmc_backbone_checkpoint_ids() -> tuple[str, str]:
    """Return the manifest-pinned official and FastPLMs ESMC repositories."""

    from fastplms.registry import RegistryError, get_model_registry

    registry = get_model_registry()
    family = registry.families["esmfold2"]
    if family.backbone_model is None:
        raise RegistryError("families.esmfold2 must declare backbone_model.")
    backbone = registry[family.backbone_model]
    return backbone.official.repo_id, backbone.fast.repo_id


def normalize_esmc_id(esmc_id: str) -> str:
    """Resolve an official ESMC identifier to its FastPLMs checkpoint mirror."""

    official_repo, fast_repo = _esmc_backbone_checkpoint_ids()
    return fast_repo if esmc_id == official_repo else esmc_id


def normalize_esmc_attention_implementation(
    implementation: str | dict[str, str] | None,
) -> str | None:
    """Validate the ESMC backend and translate the historical ``flex`` name."""

    if isinstance(implementation, dict):
        if tuple(implementation) != ("",):
            raise ValueError(
                "ESMFold2 has one ESMC attention backbone; use a string or {'': implementation}."
            )
        implementation = implementation[""]
    canonical = "flex_attention" if implementation == "flex" else implementation
    if canonical is not None and canonical not in _ESMC_ATTENTION_IMPLEMENTATIONS:
        expected = sorted(_ESMC_ATTENTION_IMPLEMENTATIONS)
        raise ValueError(
            f"Unsupported ESMFold2 attention implementation {canonical!r}; "
            f"expected one of {expected}."
        )
    return canonical


NestedConfig = TypeVar("NestedConfig")


def _nested_config(value: Any, config_type: type[NestedConfig]) -> NestedConfig:
    if isinstance(value, config_type):
        return value
    return config_type(**value) if isinstance(value, dict) else config_type()


def _coerce_nested_field(
    value: NestedConfig | dict[str, Any], config_type: type[NestedConfig]
) -> NestedConfig:
    """Convert serialized nested dictionaries while retaining supplied objects."""

    return config_type(**value) if isinstance(value, dict) else value


@dataclass
class AtomAttentionConfig:
    """Sliding-window atom attention and three-dimensional RoPE settings."""

    d_atom: int = field(default=128)
    d_token: int = field(default=768)
    n_blocks: int = field(default=3)
    n_heads: int = field(default=4)
    swa_window_size: int = field(default=128)
    expansion_ratio: int = field(default=2)
    spatial_rope_base_frequency: float = field(default=20.0)
    n_spatial_rope_pairs_per_axis: int = field(default=2)
    n_uid_rope_pairs: int = field(default=10)
    uid_rope_base_frequency: float = field(default=10000.0)


@dataclass
class DiffusionModuleConfig:
    """Dimensions and depth of the coordinate diffusion network."""

    sigma_data: float = field(default=16.0)
    c_atom: int = field(default=128)
    c_token: int = field(default=768)
    c_z: int = field(default=256)
    c_s_inputs: int = field(default=451)
    fourier_dim: int = field(default=256)
    relpos_r_max: int = field(default=32)
    relpos_s_max: int = field(default=2)
    atom_num_blocks: int = field(default=3)
    atom_num_heads: int = field(default=4)
    token_num_blocks: int = field(default=12)
    token_num_heads: int = field(default=16)
    transition_multiplier: int = field(default=2)


@dataclass
class FoldingTrunkConfig:
    """Iterative pair/single trunk dimensions."""

    n_layers: int = field(default=24)
    n_heads: int = field(default=8)
    dropout: float = field(default=0.0)


@dataclass
class InputsEmbedderConfig:
    """Input feature width and atom encoder settings."""

    d_inputs: int = field(default=451)
    atom_encoder: AtomAttentionConfig = field(default_factory=AtomAttentionConfig)

    def __post_init__(self) -> None:
        self.atom_encoder = _coerce_nested_field(self.atom_encoder, AtomAttentionConfig)


@dataclass
class DiffusionStructureHeadConfig:
    """Training and inference schedules for coordinate denoising."""

    diffusion_module: DiffusionModuleConfig = field(default_factory=DiffusionModuleConfig)
    distogram_bins: int = field(default=128)
    train_noise_log_mean: float = field(default=-1.2)
    train_noise_log_std: float = field(default=1.5)
    gamma_0: float = field(default=0.605)
    gamma_min: float = field(default=1.107)
    noise_scale: float = field(default=0.0)
    step_scale: float = field(default=1.0)
    inference_s_max: float = field(default=160.0)
    inference_s_min: float = field(default=4e-4)
    inference_p: float = field(default=8.0)
    inference_num_steps: int = field(default=68)

    def __post_init__(self) -> None:
        self.diffusion_module = _coerce_nested_field(self.diffusion_module, DiffusionModuleConfig)


@dataclass
class ConfidenceHeadConfig:
    """Confidence-bin definitions and the compact confidence trunk."""

    enabled: bool = field(default=True)
    num_plddt_bins: int = field(default=50)
    num_pde_bins: int = field(default=64)
    num_pae_bins: int = field(default=64)
    min_dist: float = field(default=2.0)
    max_dist: float = field(default=52.0)
    distogram_bins: int = field(default=128)
    folding_trunk: FoldingTrunkConfig = field(
        default_factory=lambda: FoldingTrunkConfig(n_layers=4)
    )

    def __post_init__(self) -> None:
        self.folding_trunk = _coerce_nested_field(self.folding_trunk, FoldingTrunkConfig)


@dataclass
class MSAEncoderConfig:
    """Optional multiple-sequence-alignment encoder settings."""

    enabled: bool = field(default=False)
    d_msa: int = field(default=128)
    d_hidden: int = field(default=32)
    n_layers: int = field(default=4)
    n_heads_msa: int = field(default=8)
    msa_head_width: int = field(default=32)


@dataclass
class LMEncoderConfig:
    """Release-model pair encoder derived from language-model states."""

    enabled: bool = field(default=True)
    n_layers: int = field(default=4)
    lm_dropout: float = field(default=0.25)
    per_loop_lm_dropout: bool = field(default=True)


@dataclass
class ParcaeConfig:
    """Release-model diffusion-loop scheduler settings."""

    enabled: bool = field(default=True)
    poisson_mean: float = field(default=3.0)
    min_steps: int = field(default=1)
    max_steps: int | None = field(default=6)
    coda_n_layers: int = field(default=2)


_SCALAR_DEFAULTS: tuple[tuple[str, Any], ...] = (
    ("d_single", 384),
    ("d_pair", 256),
    ("n_relative_residx_bins", 32),
    ("n_relative_chain_bins", 2),
    ("num_loops", 10),
    ("num_diffusion_samples", 8),
    ("disable_msa_features", False),
    ("lm_dropout", 0.0),
    ("force_lm_dropout_during_inference", False),
    ("lm_mask_pct", 0.0),
    ("lm_d_model", 2560),
    ("lm_num_layers", 80),
)
_NESTED_CONFIGS = (
    ("inputs", InputsEmbedderConfig),
    ("folding_trunk", FoldingTrunkConfig),
    ("structure_head", DiffusionStructureHeadConfig),
    ("confidence_head", ConfidenceHeadConfig),
    ("msa_encoder", MSAEncoderConfig),
    ("parcae", ParcaeConfig),
    ("lm_encoder", LMEncoderConfig),
)


class ESMFold2Config(PretrainedConfig):
    """Serializable ESMFold2 architecture, runtime, and precision settings."""

    model_type = "esmfold2"
    has_no_defaults_at_init = True

    def __init__(self, **kwargs: Any) -> None:
        legacy_backend = normalize_esmc_attention_implementation(kwargs.get("esmc_attn_backend"))
        requested_backend = normalize_esmc_attention_implementation(
            kwargs.get("attn_implementation")
        )
        resolved_backend = requested_backend or legacy_backend
        kwargs["attn_implementation"] = resolved_backend
        super().__init__(**kwargs)

        self.type = kwargs.get("type", "release")
        if self.type not in {"experimental", "release"}:
            raise ValueError(
                f"ESMFold2Config.type must be 'release' or 'experimental', got {self.type!r}"
            )

        for name, default in _SCALAR_DEFAULTS:
            setattr(self, name, kwargs.get(name, default))

        _official_esmc_repo, default_esmc_repo = _esmc_backbone_checkpoint_ids()
        self.esmc_id = normalize_esmc_id(kwargs.get("esmc_id", default_esmc_repo))
        self.esmc_attn_backend = resolved_backend
        self.esmc_precision = str(kwargs.get("esmc_precision", "auto"))
        if self.esmc_precision not in _ESMC_PRECISIONS:
            raise ValueError(
                "esmc_precision must be 'auto', 'bf16', 'fp32', or 'fp8', "
                f"got {self.esmc_precision!r}."
            )

        for name, config_type in _NESTED_CONFIGS:
            setattr(self, name, _nested_config(kwargs.get(name), config_type))
        self.msa_encoder_overwrite = bool(kwargs.get("msa_encoder_overwrite", True))

    def to_dict(self) -> dict[str, Any]:
        output = cast(dict[str, Any], super().to_dict())
        for name, _config_type in _NESTED_CONFIGS:
            output[name] = asdict(getattr(self, name))
        return output


__all__ = [
    "ESMFold2Config",
    "LMEncoderConfig",
    "MSAEncoderConfig",
    "ParcaeConfig",
    "normalize_esmc_attention_implementation",
    "normalize_esmc_id",
]
