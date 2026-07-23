import copy
import inspect
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from . import vb_const as const
from . import vb_layers_initialize as init
from .cif_writer import write_cif
from .minimal_featurizer import build_boltz2_features
from .minimal_structures import ProteinStructureTemplate
from .vb_const import bond_types as _vb_const_bond_types  # noqa: F401
from .vb_layers_attention import AttentionPairBias as _vb_layers_attention_marker  # noqa: F401
from .vb_layers_attentionv2 import AttentionPairBias as _vb_layers_attentionv2_marker  # noqa: F401
from .vb_layers_confidence_utils import (
    compute_ptms as _vb_layers_confidence_utils_marker,  # noqa: F401
)
from .vb_layers_dropout import get_dropout_mask as _vb_layers_dropout_marker  # noqa: F401
from .vb_layers_initialize import gating_init_ as _vb_layers_initialize_marker  # noqa: F401
from .vb_layers_outer_product_mean import (
    OuterProductMean as _vb_layers_outer_product_mean_marker,  # noqa: F401
)
from .vb_layers_pair_averaging import (
    PairWeightedAveraging as _vb_layers_pair_averaging_marker,  # noqa: F401
)
from .vb_layers_pairformer import PairformerModule
from .vb_layers_transition import Transition as _vb_layers_transition_marker  # noqa: F401
from .vb_layers_triangular_mult import (
    TriangleMultiplicationIncoming as _vb_layers_triangular_mult_marker,  # noqa: F401
)
from .vb_loss_diffusionv2 import weighted_rigid_align as _vb_loss_diffusionv2_marker  # noqa: F401
from .vb_modules_confidencev2 import ConfidenceModule
from .vb_modules_diffusion_conditioning import DiffusionConditioning
from .vb_modules_diffusionv2 import AtomDiffusion, DiffusionModule
from .vb_modules_encodersv2 import RelativePositionEncoder
from .vb_modules_transformersv2 import (
    DiffusionTransformer as _vb_modules_transformersv2_marker,  # noqa: F401
)
from .vb_modules_trunkv2 import (
    ContactConditioning,
    DistogramModule,
    InputEmbedder,
    MSAModule,
)
from .vb_modules_utils import LinearNoBias as _vb_modules_utils_marker  # noqa: F401
from .vb_potentials_potentials import (
    get_potentials as _vb_potentials_potentials_marker,  # noqa: F401
)
from .vb_potentials_schedules import (
    ParameterSchedule as _vb_potentials_schedules_marker,  # noqa: F401
)
from .vb_tri_attn_attention import (
    TriangleAttentionStartingNode as _vb_tri_attn_attention_marker,  # noqa: F401
)
from .vb_tri_attn_primitives import Attention as _vb_tri_attn_primitives_marker  # noqa: F401
from .vb_tri_attn_utils import permute_final_dims as _vb_tri_attn_utils_marker  # noqa: F401


def _default_steering_args() -> dict[str, Any]:
    return {
        "fk_steering": False,
        "num_particles": 3,
        "fk_lambda": 4.0,
        "fk_resampling_interval": 3,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
        "num_gd_steps": 16,
    }


def _boltz2_reference_diffusion_overrides() -> dict[str, Any]:
    # Match Boltz2 CLI inference defaults from boltz.main/Boltz2DiffusionParams.
    return {
        "gamma_0": 0.8,
        "gamma_min": 1.0,
        "noise_scale": 1.003,
        "rho": 7,
        "step_scale": 1.5,
        "sigma_min": 0.0001,
        "sigma_max": 160.0,
        "sigma_data": 16.0,
        "P_mean": -1.2,
        "P_std": 1.5,
        "coordinate_augmentation": True,
        "alignment_reverse_diff": True,
        "synchronize_sigmas": True,
    }


def _enforce_pairformer_v2(pairformer_args: Mapping[str, Any], context: str) -> dict[str, Any]:
    if not isinstance(pairformer_args, Mapping):
        raise TypeError(f"Expected {context} pairformer_args to be a dictionary.")
    out = _to_plain_python(copy.deepcopy(pairformer_args))
    if "v2" in out and not out["v2"]:
        raise ValueError(f"{context} pairformer_args['v2'] must be True for Boltz2.")
    out["v2"] = True
    return out


def _require_key(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required key '{key}' in checkpoint hyperparameters.")
    return mapping[key]


def _state_dict_without_wrappers(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    cleaned: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("ema."):
            continue
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned[new_key] = value
    return cleaned


def _to_cpu_detached(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, nested_value in value.items():
            out[key] = _to_cpu_detached(nested_value)
        return out
    if isinstance(value, list):
        return [_to_cpu_detached(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_detached(item) for item in value)
    return value


@dataclass(frozen=True)
class _RandomState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Tensor
    torch_cuda: list[Tensor] | None


@contextmanager
def _seed_context(seed: int | None) -> Iterator[None]:
    """Temporarily seed every RNG used by Boltz2 and restore caller state."""

    if seed is None:
        yield
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(
            "Boltz2 seed must be an int or None; bool and coerced values are not accepted."
        )
    state = _RandomState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    normalized_seed = seed % (2**32)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)
    try:
        yield
    finally:
        random.setstate(state.python)
        np.random.set_state(state.numpy)
        torch.random.set_rng_state(state.torch_cpu)
        if state.torch_cuda is not None:
            torch.cuda.set_rng_state_all(state.torch_cuda)


def _to_plain_python(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: dict[Any, Any] = {}
        for key, nested_value in value.items():
            out[key] = _to_plain_python(nested_value)
        return out
    if isinstance(value, list):
        return [_to_plain_python(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain_python(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_plain_python(item) for item in value]
    return value


def _filtered_kwargs(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(target.__init__)
    allowed = set(signature.parameters.keys())
    allowed.discard("self")
    filtered: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in allowed:
            filtered[key] = value
    return filtered


@dataclass
class Boltz2ModelOutput(ModelOutput):
    """Raw Boltz2 inference output with standard AutoModel controls."""

    last_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    pdistogram: torch.Tensor | None = None
    s: torch.Tensor | None = None
    z: torch.Tensor | None = None
    sample_atom_coords: torch.Tensor | None = None
    diff_token_repr: torch.Tensor | None = None
    s_conf: torch.Tensor | None = None
    z_conf: torch.Tensor | None = None
    pde_logits: torch.Tensor | None = None
    plddt_logits: torch.Tensor | None = None
    resolved_logits: torch.Tensor | None = None
    pde: torch.Tensor | None = None
    plddt: torch.Tensor | None = None
    complex_plddt: torch.Tensor | None = None
    complex_iplddt: torch.Tensor | None = None
    complex_pde: torch.Tensor | None = None
    complex_ipde: torch.Tensor | None = None
    pae_logits: torch.Tensor | None = None
    pae: torch.Tensor | None = None
    ptm: torch.Tensor | None = None
    iptm: torch.Tensor | None = None
    ligand_iptm: torch.Tensor | None = None
    protein_iptm: torch.Tensor | None = None
    pair_chains_iptm: Any = None


@dataclass
class Boltz2StructureOutput(ModelOutput):
    sample_atom_coords: torch.Tensor | None = None
    atom_pad_mask: torch.Tensor | None = None
    plddt: torch.Tensor | None = None
    confidence_score: torch.Tensor | None = None
    complex_plddt: torch.Tensor | None = None
    iptm: torch.Tensor | None = None
    ptm: torch.Tensor | None = None
    sequence: str | None = None
    structure_template: ProteinStructureTemplate | None = None
    raw_output: dict[str, torch.Tensor] | None = None
    seed: int | None = None


class Boltz2Config(PretrainedConfig):
    model_type = "boltz2_automodel"

    def __init__(
        self,
        core_kwargs: dict[str, Any] | None = None,
        num_bins: int = 64,
        default_recycling_steps: int = 3,
        default_sampling_steps: int = 200,
        default_diffusion_samples: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if core_kwargs is None:
            core_kwargs = {}
        self.core_kwargs = core_kwargs
        self.num_bins = num_bins
        self.default_recycling_steps = default_recycling_steps
        self.default_sampling_steps = default_sampling_steps
        self.default_diffusion_samples = default_diffusion_samples

    @classmethod
    def from_hyperparameters(
        cls,
        hparams: dict[str, Any],
        use_kernels: bool = False,
        default_recycling_steps: int | None = None,
        default_sampling_steps: int | None = None,
        default_diffusion_samples: int | None = None,
    ) -> "Boltz2Config":
        if not isinstance(hparams, dict):
            raise TypeError("Expected checkpoint hyperparameters as a dictionary.")
        required = [
            "atom_s",
            "atom_z",
            "token_s",
            "token_z",
            "num_bins",
            "embedder_args",
            "msa_args",
            "pairformer_args",
            "score_model_args",
            "diffusion_process_args",
        ]
        for key in required:
            _require_key(hparams, key)

        pairformer_args = _enforce_pairformer_v2(
            hparams["pairformer_args"],
            context="checkpoint",
        )
        diffusion_process_args = _to_plain_python(copy.deepcopy(hparams["diffusion_process_args"]))
        diffusion_overrides = _boltz2_reference_diffusion_overrides()
        for key in diffusion_overrides:
            diffusion_process_args[key] = diffusion_overrides[key]

        core_kwargs: dict[str, Any] = {
            "atom_s": hparams["atom_s"],
            "atom_z": hparams["atom_z"],
            "token_s": hparams["token_s"],
            "token_z": hparams["token_z"],
            "num_bins": hparams["num_bins"],
            "embedder_args": _to_plain_python(copy.deepcopy(hparams["embedder_args"])),
            "msa_args": _to_plain_python(copy.deepcopy(hparams["msa_args"])),
            "pairformer_args": pairformer_args,
            "score_model_args": _to_plain_python(copy.deepcopy(hparams["score_model_args"])),
            "diffusion_process_args": diffusion_process_args,
            "use_kernels": use_kernels,
        }

        if "confidence_model_args" in hparams:
            confidence_model_args = _to_plain_python(
                copy.deepcopy(hparams["confidence_model_args"])
            )
            if "pairformer_args" in confidence_model_args:
                confidence_model_args["pairformer_args"] = _enforce_pairformer_v2(
                    confidence_model_args["pairformer_args"],
                    context="confidence",
                )
            core_kwargs["confidence_model_args"] = confidence_model_args
        else:
            core_kwargs["confidence_model_args"] = None

        core_kwargs["confidence_prediction"] = hparams.get("confidence_prediction", True)
        core_kwargs["token_level_confidence"] = hparams.get("token_level_confidence", True)
        core_kwargs["alpha_pae"] = hparams.get("alpha_pae", 0.0)
        core_kwargs["atoms_per_window_queries"] = hparams.get("atoms_per_window_queries", 32)
        core_kwargs["atoms_per_window_keys"] = hparams.get("atoms_per_window_keys", 128)
        core_kwargs["atom_feature_dim"] = hparams.get("atom_feature_dim", 128)
        core_kwargs["bond_type_feature"] = hparams.get("bond_type_feature", False)
        core_kwargs["run_trunk_and_structure"] = hparams.get("run_trunk_and_structure", True)
        core_kwargs["skip_run_structure"] = hparams.get("skip_run_structure", False)
        core_kwargs["fix_sym_check"] = hparams.get("fix_sym_check", False)
        core_kwargs["cyclic_pos_enc"] = hparams.get("cyclic_pos_enc", False)
        core_kwargs["use_no_atom_char"] = hparams.get("use_no_atom_char", False)
        core_kwargs["use_atom_backbone_feat"] = hparams.get("use_atom_backbone_feat", False)
        core_kwargs["use_residue_feats_atoms"] = hparams.get("use_residue_feats_atoms", False)
        core_kwargs["conditioning_cutoff_min"] = hparams.get("conditioning_cutoff_min", 4.0)
        core_kwargs["conditioning_cutoff_max"] = hparams.get("conditioning_cutoff_max", 20.0)

        if "steering_args" in hparams and hparams["steering_args"] is not None:
            core_kwargs["steering_args"] = _to_plain_python(copy.deepcopy(hparams["steering_args"]))
        else:
            core_kwargs["steering_args"] = _default_steering_args()

        if "validation_args" in hparams:
            validation_args = hparams["validation_args"]
            if not isinstance(validation_args, Mapping):
                raise TypeError(
                    "Expected 'validation_args' in checkpoint hyperparameters to be a mapping."
                )
            if default_recycling_steps is None and "recycling_steps" in validation_args:
                default_recycling_steps = validation_args["recycling_steps"]
            if default_sampling_steps is None and "sampling_steps" in validation_args:
                default_sampling_steps = validation_args["sampling_steps"]
            if default_diffusion_samples is None and "diffusion_samples" in validation_args:
                default_diffusion_samples = validation_args["diffusion_samples"]

        if default_recycling_steps is None:
            default_recycling_steps = 3
        if default_sampling_steps is None:
            default_sampling_steps = 200
        if default_diffusion_samples is None:
            default_diffusion_samples = 1

        return cls(
            core_kwargs=core_kwargs,
            num_bins=hparams["num_bins"],
            default_recycling_steps=default_recycling_steps,
            default_sampling_steps=default_sampling_steps,
            default_diffusion_samples=default_diffusion_samples,
        )


class Boltz2InferenceCore(nn.Module):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        embedder_args: dict[str, Any],
        msa_args: dict[str, Any],
        pairformer_args: dict[str, Any],
        score_model_args: dict[str, Any],
        diffusion_process_args: dict[str, Any],
        confidence_model_args: dict[str, Any] | None = None,
        atom_feature_dim: int = 128,
        confidence_prediction: bool = True,
        token_level_confidence: bool = True,
        alpha_pae: float = 0.0,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        run_trunk_and_structure: bool = True,
        skip_run_structure: bool = False,
        bond_type_feature: bool = False,
        fix_sym_check: bool = False,
        cyclic_pos_enc: bool = False,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        use_kernels: bool = False,
        steering_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.use_kernels = use_kernels
        self.confidence_prediction = confidence_prediction
        self.token_level_confidence = token_level_confidence
        self.alpha_pae = alpha_pae
        self.run_trunk_and_structure = run_trunk_and_structure
        self.skip_run_structure = skip_run_structure
        self.bond_type_feature = bond_type_feature
        self.steering_args = (
            steering_args if steering_args is not None else _default_steering_args()
        )
        if "v2" not in pairformer_args:
            raise ValueError("Boltz2 requires pairformer_args['v2'].")
        if not pairformer_args["v2"]:
            raise ValueError("Boltz2 requires pairformer_args['v2']=True.")

        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            "use_no_atom_char": use_no_atom_char,
            "use_atom_backbone_feat": use_atom_backbone_feat,
            "use_residue_feats_atoms": use_residue_feats_atoms,
            **embedder_args,
        }
        full_embedder_args = _filtered_kwargs(InputEmbedder, full_embedder_args)
        self.input_embedder = InputEmbedder(**full_embedder_args)

        self.s_init = nn.Linear(token_s, token_s, bias=False)
        self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
        self.z_init_2 = nn.Linear(token_s, token_z, bias=False)
        self.rel_pos = RelativePositionEncoder(
            token_z,
            fix_sym_check=fix_sym_check,
            cyclic_pos_enc=cyclic_pos_enc,
        )
        self.token_bonds = nn.Linear(1, token_z, bias=False)
        if self.bond_type_feature:
            self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

        self.contact_conditioning = ContactConditioning(
            token_z=token_z,
            cutoff_min=conditioning_cutoff_min,
            cutoff_max=conditioning_cutoff_max,
        )
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        msa_kwargs = _filtered_kwargs(
            MSAModule,
            {"token_z": token_z, "token_s": token_s, **msa_args},
        )
        self.msa_module = MSAModule(**msa_kwargs)

        pairformer_kwargs = _filtered_kwargs(
            PairformerModule,
            {"token_s": token_s, "token_z": token_z, **pairformer_args},
        )
        if "token_s" not in pairformer_kwargs or "token_z" not in pairformer_kwargs:
            raise RuntimeError("Boltz2 pairformer construction lost its token dimensions.")
        pairformer_token_s = pairformer_kwargs.pop("token_s")
        pairformer_token_z = pairformer_kwargs.pop("token_z")
        self.pairformer_module = PairformerModule(
            pairformer_token_s,
            pairformer_token_z,
            **pairformer_kwargs,
        )

        diffusion_conditioning_kwargs = {
            "token_s": token_s,
            "token_z": token_z,
            "atom_s": atom_s,
            "atom_z": atom_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_encoder_depth": score_model_args["atom_encoder_depth"],
            "atom_encoder_heads": score_model_args["atom_encoder_heads"],
            "token_transformer_depth": score_model_args["token_transformer_depth"],
            "token_transformer_heads": score_model_args["token_transformer_heads"],
            "atom_decoder_depth": score_model_args["atom_decoder_depth"],
            "atom_decoder_heads": score_model_args["atom_decoder_heads"],
            "atom_feature_dim": atom_feature_dim,
            "conditioning_transition_layers": score_model_args["conditioning_transition_layers"],
            "use_no_atom_char": use_no_atom_char,
            "use_atom_backbone_feat": use_atom_backbone_feat,
            "use_residue_feats_atoms": use_residue_feats_atoms,
        }
        diffusion_conditioning_kwargs = _filtered_kwargs(
            DiffusionConditioning,
            diffusion_conditioning_kwargs,
        )
        self.diffusion_conditioning = DiffusionConditioning(**diffusion_conditioning_kwargs)

        structure_score_model_args = {
            "token_s": token_s,
            "atom_s": atom_s,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            **score_model_args,
        }
        structure_score_model_args = _filtered_kwargs(
            DiffusionModule,
            structure_score_model_args,
        )
        structure_module_kwargs = {
            "score_model_args": structure_score_model_args,
            "compile_score": False,
            **diffusion_process_args,
        }
        structure_module_kwargs = _filtered_kwargs(AtomDiffusion, structure_module_kwargs)
        self.structure_module = AtomDiffusion(**structure_module_kwargs)
        self.distogram_module = DistogramModule(token_z, num_bins)

        if self.confidence_prediction:
            if confidence_model_args is None:
                raise ValueError(
                    "confidence_prediction=True requires confidence_model_args in config."
                )
            confidence_kwargs = {
                "token_s": token_s,
                "token_z": token_z,
                "token_level_confidence": token_level_confidence,
                "bond_type_feature": bond_type_feature,
                "fix_sym_check": fix_sym_check,
                "cyclic_pos_enc": cyclic_pos_enc,
                "conditioning_cutoff_min": conditioning_cutoff_min,
                "conditioning_cutoff_max": conditioning_cutoff_max,
                **confidence_model_args,
            }
            confidence_kwargs = _filtered_kwargs(ConfidenceModule, confidence_kwargs)
            self.confidence_module = ConfidenceModule(**confidence_kwargs)

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 3,
        num_sampling_steps: int | None = None,
        diffusion_samples: int = 1,
        max_parallel_samples: int | None = None,
        run_confidence_sequentially: bool = True,
        detach_confidence: bool = True,
    ) -> dict[str, Tensor]:
        s_inputs = self.input_embedder(feats)
        s_init = self.s_init(s_inputs)

        z_init = self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        relative_position_encoding = self.rel_pos(feats)
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feats["token_bonds"].float())
        if self.bond_type_feature:
            z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + self.contact_conditioning(feats)

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        if self.run_trunk_and_structure:
            for _ in range(recycling_steps + 1):
                s = s_init + self.s_recycle(self.s_norm(s))
                z = z_init + self.z_recycle(self.z_norm(z))
                z = z + self.msa_module(
                    z,
                    s_inputs,
                    feats,
                    use_kernels=self.use_kernels,
                )
                s, z = self.pairformer_module(
                    s,
                    z,
                    mask=mask,
                    pair_mask=pair_mask,
                    use_kernels=self.use_kernels,
                )

        pdistogram = self.distogram_module(z)
        output: dict[str, Tensor] = {
            "pdistogram": pdistogram,
            "s": s,
            "z": z,
        }

        if self.run_trunk_and_structure and (not self.skip_run_structure):
            q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                self.diffusion_conditioning(
                    s_trunk=s,
                    z_trunk=z,
                    relative_position_encoding=relative_position_encoding,
                    feats=feats,
                )
            )
            diffusion_conditioning = {
                "q": q,
                "c": c,
                "to_keys": to_keys,
                "atom_enc_bias": atom_enc_bias,
                "atom_dec_bias": atom_dec_bias,
                "token_trans_bias": token_trans_bias,
            }
            with torch.autocast("cuda", enabled=False):
                struct_out = self.structure_module.sample(
                    s_trunk=s.float(),
                    s_inputs=s_inputs.float(),
                    feats=feats,
                    num_sampling_steps=num_sampling_steps,
                    atom_mask=feats["atom_pad_mask"].float(),
                    multiplicity=diffusion_samples,
                    max_parallel_samples=max_parallel_samples,
                    steering_args=self.steering_args,
                    diffusion_conditioning=diffusion_conditioning,
                )
            output.update(struct_out)

        if self.confidence_prediction:
            if self.skip_run_structure:
                x_pred = feats["coords"].repeat_interleave(diffusion_samples, 0)
            else:
                if "sample_atom_coords" not in output:
                    raise RuntimeError("Structure sampling did not produce sample_atom_coords.")
                x_pred = output["sample_atom_coords"]

            if detach_confidence:
                s_inputs_c = s_inputs.detach()
                s_c = s.detach()
                z_c = z.detach()
                x_pred_c = x_pred.detach()
                pdist_c = output["pdistogram"][:, :, :, 0].detach()
            else:
                s_inputs_c = s_inputs
                s_c = s
                z_c = z
                x_pred_c = x_pred
                pdist_c = output["pdistogram"][:, :, :, 0]

            output.update(
                self.confidence_module(
                    s_inputs=s_inputs_c,
                    s=s_c,
                    z=z_c,
                    x_pred=x_pred_c,
                    feats=feats,
                    pred_distogram_logits=pdist_c,
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
            )

        return output


class Boltz2Model(PreTrainedModel):
    config_class = Boltz2Config
    base_model_prefix = "core"
    all_tied_weights_keys: ClassVar[dict[str, str]] = {}

    def __init__(self, config: Boltz2Config) -> None:
        super().__init__(config)
        if not isinstance(config.core_kwargs, dict):
            raise TypeError("config.core_kwargs must be a dictionary.")
        self.core = Boltz2InferenceCore(**config.core_kwargs)

    def _init_weights(self, module: nn.Module) -> None:
        return

    def _detied_state_dict(self) -> dict[str, Tensor]:
        raw_state = self.state_dict()
        seen_ptrs: dict[int, str] = {}
        out: dict[str, Tensor] = {}
        for key, tensor in raw_state.items():
            if torch.is_tensor(tensor):
                ptr = tensor.untyped_storage().data_ptr()
                if ptr in seen_ptrs:
                    out[key] = tensor.clone()
                else:
                    seen_ptrs[ptr] = key
                    out[key] = tensor
            else:
                out[key] = tensor
        return out

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        if "safe_serialization" not in kwargs:
            kwargs["safe_serialization"] = True
        if "state_dict" not in kwargs:
            kwargs["state_dict"] = self._detied_state_dict()
        super().save_pretrained(save_directory, **kwargs)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_boltz_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: str | torch.device = "cpu",
        use_kernels: bool = False,
        default_recycling_steps: int | None = None,
        default_sampling_steps: int | None = None,
        default_diffusion_samples: int | None = None,
        *,
        allow_unsafe_pickle: bool = False,
    ) -> "Boltz2Model":
        """Import an official Boltz Lightning checkpoint into the inference model.

        Lightning checkpoints require Python pickle deserialization. The caller
        must opt in with the literal value ``True`` after independently verifying
        that the checkpoint comes from a trusted source and matches its expected
        cryptographic hash.
        """

        if allow_unsafe_pickle is not True:
            raise ValueError(
                "Boltz Lightning checkpoints contain Python pickle and cannot be loaded "
                "safely. Pass allow_unsafe_pickle=True only for a trusted, hash-verified "
                "checkpoint."
            )

        # The official Lightning checkpoints contain OmegaConf objects, so this explicit
        # trusted-checkpoint boundary requires full Python unpickling.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint must deserialize to a dictionary.")
        _require_key(checkpoint, "hyper_parameters")
        _require_key(checkpoint, "state_dict")

        hparams = checkpoint["hyper_parameters"]
        if not isinstance(hparams, dict):
            raise TypeError("Checkpoint hyper_parameters must be a dictionary.")
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint state_dict must be a dictionary.")

        config = Boltz2Config.from_hyperparameters(
            hparams,
            use_kernels=use_kernels,
            default_recycling_steps=default_recycling_steps,
            default_sampling_steps=default_sampling_steps,
            default_diffusion_samples=default_diffusion_samples,
        )
        model = cls(config)
        cleaned = _state_dict_without_wrappers(state_dict)
        target_keys = set(model.core.state_dict().keys())
        for key in target_keys:
            if ".attention.norm_s." in key:
                raise RuntimeError(
                    "Boltz2 inference core unexpectedly uses v1 attention parameters. "
                    "Expected pairformer v2 architecture."
                )
        filtered: dict[str, Tensor] = {}
        for key, value in cleaned.items():
            if key in target_keys:
                filtered[key] = value

        missing = sorted(target_keys.difference(filtered.keys()))
        if missing:
            raise RuntimeError(
                "Checkpoint is missing required parameters for Boltz2 inference core. "
                f"Missing keys (first 20): {missing[:20]}"
            )

        load_result = model.core.load_state_dict(filtered, strict=False)
        loaded_missing = sorted(load_result.missing_keys)
        if loaded_missing:
            raise RuntimeError(
                "Model has unexpected missing keys after load_state_dict. "
                f"Missing keys (first 20): {loaded_missing[:20]}"
            )
        if load_result.unexpected_keys:
            raise RuntimeError(
                "Model has unexpected checkpoint keys after load_state_dict: "
                f"{sorted(load_result.unexpected_keys)[:20]}"
            )
        model.eval()
        return model

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        diffusion_samples: int | None = None,
        max_parallel_samples: int | None = None,
        run_confidence_sequentially: bool = True,
        detach_confidence: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> Boltz2ModelOutput | tuple[Any, ...]:
        output_attentions = (
            self.config.output_attentions if output_attentions is None else output_attentions
        )
        if output_attentions:
            raise NotImplementedError(
                "Boltz2 does not expose normalized attention tensors from its structure "
                "modules. output_attentions=True is unsupported."
            )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if recycling_steps is None:
            recycling_steps = self.config.default_recycling_steps
        if num_sampling_steps is None:
            num_sampling_steps = self.config.default_sampling_steps
        if diffusion_samples is None:
            diffusion_samples = self.config.default_diffusion_samples
        raw_output = self.core(
            feats=feats,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            diffusion_samples=diffusion_samples,
            max_parallel_samples=max_parallel_samples,
            run_confidence_sequentially=run_confidence_sequentially,
            detach_confidence=detach_confidence,
        )
        token_state = raw_output.get("s")
        pair_state = raw_output.get("z")
        model_output = Boltz2ModelOutput(
            last_hidden_state=token_state,
            hidden_states=(token_state, pair_state)
            if output_hidden_states and token_state is not None and pair_state is not None
            else None,
            **raw_output,
        )
        return model_output if return_dict else model_output.to_tuple()

    def _to_model_device(
        self,
        feats: dict[str, Tensor],
        float_dtype: torch.dtype = torch.float32,
    ) -> dict[str, Tensor]:
        moved: dict[str, Tensor] = {}
        for key, value in feats.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    moved[key] = value.to(device=self.device, dtype=float_dtype)
                else:
                    moved[key] = value.to(device=self.device)
            else:
                moved[key] = value
        return moved

    def predict_structure(
        self,
        amino_acid_sequence: str,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        diffusion_samples: int | None = None,
        max_parallel_samples: int | None = None,
        run_confidence_sequentially: bool = True,
        float_dtype: torch.dtype = torch.float32,
        seed: int | None = None,
    ) -> Boltz2StructureOutput:
        if float_dtype != torch.float32:
            raise ValueError(
                "Boltz2 predict_structure requires FP32 features and parameter storage; "
                "CUDA compute runs under the declared BF16 autocast policy."
            )
        parameter_dtypes = {
            parameter.dtype for parameter in self.parameters() if parameter.is_floating_point()
        }
        if parameter_dtypes and parameter_dtypes != {torch.float32}:
            raise ValueError(
                "Boltz2 predict_structure requires FP32 parameter storage before CUDA "
                f"BF16 autocast; found {sorted(map(str, parameter_dtypes))}."
            )

        with _seed_context(seed):
            feats, template = build_boltz2_features(
                amino_acid_sequence=amino_acid_sequence,
                num_bins=self.config.num_bins,
                atoms_per_window_queries=(
                    self.core.input_embedder.atom_encoder.atoms_per_window_queries
                ),
            )
            feats = self._to_model_device(feats, float_dtype=torch.float32)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with torch.no_grad(), autocast_context:
                output = self.forward(
                    feats=feats,
                    recycling_steps=recycling_steps,
                    num_sampling_steps=num_sampling_steps,
                    diffusion_samples=diffusion_samples,
                    max_parallel_samples=max_parallel_samples,
                    run_confidence_sequentially=run_confidence_sequentially,
                    return_dict=True,
                )

        sample_atom_coords_value = output.get("sample_atom_coords")
        if not isinstance(sample_atom_coords_value, torch.Tensor):
            raise RuntimeError("Boltz2 structure sampling did not return coordinate tensors.")
        sample_atom_coords = sample_atom_coords_value.detach().cpu()
        non_finite_mask = torch.logical_not(torch.isfinite(sample_atom_coords))
        if torch.any(non_finite_mask):
            raise RuntimeError(
                "sample_atom_coords contains non-finite values. "
                f"Non-finite count: {int(non_finite_mask.sum().item())}"
            )
        atom_pad_mask = feats["atom_pad_mask"][0].detach().cpu()
        plddt = output["plddt"].detach().cpu() if "plddt" in output else None
        complex_plddt = (
            output["complex_plddt"].detach().cpu() if "complex_plddt" in output else None
        )
        iptm = output["iptm"].detach().cpu() if "iptm" in output else None
        ptm = output["ptm"].detach().cpu() if "ptm" in output else None

        confidence_score = None
        if (complex_plddt is not None) and (iptm is not None) and (ptm is not None):
            if torch.allclose(iptm, torch.zeros_like(iptm)):
                confidence_score = (4 * complex_plddt + ptm) / 5
            else:
                confidence_score = (4 * complex_plddt + iptm) / 5

        return Boltz2StructureOutput(
            sample_atom_coords=sample_atom_coords,
            atom_pad_mask=atom_pad_mask,
            plddt=plddt,
            confidence_score=confidence_score,
            complex_plddt=complex_plddt,
            iptm=iptm,
            ptm=ptm,
            sequence=template.sequence,
            structure_template=template,
            raw_output={key: _to_cpu_detached(val) for key, val in output.items()},
            seed=seed,
        )

    def save_as_cif(
        self,
        structure_output: Boltz2StructureOutput,
        output_path: str,
        sample_index: int = 0,
    ) -> str:
        if structure_output.structure_template is None:
            raise ValueError("structure_output.structure_template is required for CIF export.")
        if structure_output.sample_atom_coords is None:
            raise ValueError("structure_output.sample_atom_coords is required for CIF export.")
        if structure_output.atom_pad_mask is None:
            raise ValueError("structure_output.atom_pad_mask is required for CIF export.")
        return write_cif(
            structure_template=structure_output.structure_template,
            atom_coords=structure_output.sample_atom_coords,
            atom_mask=structure_output.atom_pad_mask,
            output_path=output_path,
            plddt=structure_output.plddt,
            sample_index=sample_index,
        )
