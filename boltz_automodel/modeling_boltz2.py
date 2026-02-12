import copy
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch._dynamo
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from .cif_writer import write_cif
from .minimal_featurizer import build_boltz2_features
from .minimal_structures import ProteinStructureTemplate
from .runtime import ensure_boltz_importable


ensure_boltz_importable()
import boltz.model.layers.initialize as init  # noqa: E402
from boltz.data import const  # noqa: E402
from boltz.model.layers.pairformer import PairformerModule  # noqa: E402
from boltz.model.modules.confidencev2 import ConfidenceModule  # noqa: E402
from boltz.model.modules.diffusion_conditioning import DiffusionConditioning  # noqa: E402
from boltz.model.modules.diffusionv2 import AtomDiffusion, DiffusionModule  # noqa: E402
from boltz.model.modules.encodersv2 import RelativePositionEncoder  # noqa: E402
from boltz.model.modules.trunkv2 import (  # noqa: E402
    ContactConditioning,
    DistogramModule,
    InputEmbedder,
    MSAModule,
)


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


def _require_key(mapping: dict[str, Any], key: str) -> Any:
    assert key in mapping, f"Missing required key '{key}' in checkpoint hyperparameters."
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
class Boltz2StructureOutput(ModelOutput):
    sample_atom_coords: Optional[torch.Tensor] = None
    atom_pad_mask: Optional[torch.Tensor] = None
    plddt: Optional[torch.Tensor] = None
    confidence_score: Optional[torch.Tensor] = None
    complex_plddt: Optional[torch.Tensor] = None
    iptm: Optional[torch.Tensor] = None
    ptm: Optional[torch.Tensor] = None
    sequence: Optional[str] = None
    structure_template: Optional[ProteinStructureTemplate] = None
    raw_output: Optional[dict[str, torch.Tensor]] = None


class Boltz2Config(PretrainedConfig):
    model_type = "boltz2_automodel"

    def __init__(
        self,
        core_kwargs: Optional[dict[str, Any]] = None,
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
        default_recycling_steps: Optional[int] = None,
        default_sampling_steps: Optional[int] = None,
        default_diffusion_samples: Optional[int] = None,
    ) -> "Boltz2Config":
        assert isinstance(hparams, dict), "Expected checkpoint hyperparameters as a dictionary."
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

        core_kwargs: dict[str, Any] = {
            "atom_s": hparams["atom_s"],
            "atom_z": hparams["atom_z"],
            "token_s": hparams["token_s"],
            "token_z": hparams["token_z"],
            "num_bins": hparams["num_bins"],
            "embedder_args": _to_plain_python(copy.deepcopy(hparams["embedder_args"])),
            "msa_args": _to_plain_python(copy.deepcopy(hparams["msa_args"])),
            "pairformer_args": _to_plain_python(copy.deepcopy(hparams["pairformer_args"])),
            "score_model_args": _to_plain_python(copy.deepcopy(hparams["score_model_args"])),
            "diffusion_process_args": _to_plain_python(
                copy.deepcopy(hparams["diffusion_process_args"])
            ),
            "use_kernels": use_kernels,
        }

        if "confidence_model_args" in hparams:
            core_kwargs["confidence_model_args"] = _to_plain_python(
                copy.deepcopy(hparams["confidence_model_args"])
            )
        else:
            core_kwargs["confidence_model_args"] = None

        if "confidence_prediction" in hparams:
            core_kwargs["confidence_prediction"] = hparams["confidence_prediction"]
        else:
            core_kwargs["confidence_prediction"] = True

        if "token_level_confidence" in hparams:
            core_kwargs["token_level_confidence"] = hparams["token_level_confidence"]
        else:
            core_kwargs["token_level_confidence"] = True

        if "alpha_pae" in hparams:
            core_kwargs["alpha_pae"] = hparams["alpha_pae"]
        else:
            core_kwargs["alpha_pae"] = 0.0

        if "atoms_per_window_queries" in hparams:
            core_kwargs["atoms_per_window_queries"] = hparams["atoms_per_window_queries"]
        else:
            core_kwargs["atoms_per_window_queries"] = 32

        if "atoms_per_window_keys" in hparams:
            core_kwargs["atoms_per_window_keys"] = hparams["atoms_per_window_keys"]
        else:
            core_kwargs["atoms_per_window_keys"] = 128

        if "atom_feature_dim" in hparams:
            core_kwargs["atom_feature_dim"] = hparams["atom_feature_dim"]
        else:
            core_kwargs["atom_feature_dim"] = 128

        if "bond_type_feature" in hparams:
            core_kwargs["bond_type_feature"] = hparams["bond_type_feature"]
        else:
            core_kwargs["bond_type_feature"] = False

        if "run_trunk_and_structure" in hparams:
            core_kwargs["run_trunk_and_structure"] = hparams["run_trunk_and_structure"]
        else:
            core_kwargs["run_trunk_and_structure"] = True

        if "skip_run_structure" in hparams:
            core_kwargs["skip_run_structure"] = hparams["skip_run_structure"]
        else:
            core_kwargs["skip_run_structure"] = False

        if "fix_sym_check" in hparams:
            core_kwargs["fix_sym_check"] = hparams["fix_sym_check"]
        else:
            core_kwargs["fix_sym_check"] = False

        if "cyclic_pos_enc" in hparams:
            core_kwargs["cyclic_pos_enc"] = hparams["cyclic_pos_enc"]
        else:
            core_kwargs["cyclic_pos_enc"] = False

        if "use_no_atom_char" in hparams:
            core_kwargs["use_no_atom_char"] = hparams["use_no_atom_char"]
        else:
            core_kwargs["use_no_atom_char"] = False

        if "use_atom_backbone_feat" in hparams:
            core_kwargs["use_atom_backbone_feat"] = hparams["use_atom_backbone_feat"]
        else:
            core_kwargs["use_atom_backbone_feat"] = False

        if "use_residue_feats_atoms" in hparams:
            core_kwargs["use_residue_feats_atoms"] = hparams["use_residue_feats_atoms"]
        else:
            core_kwargs["use_residue_feats_atoms"] = False

        if "conditioning_cutoff_min" in hparams:
            core_kwargs["conditioning_cutoff_min"] = hparams["conditioning_cutoff_min"]
        else:
            core_kwargs["conditioning_cutoff_min"] = 4.0

        if "conditioning_cutoff_max" in hparams:
            core_kwargs["conditioning_cutoff_max"] = hparams["conditioning_cutoff_max"]
        else:
            core_kwargs["conditioning_cutoff_max"] = 20.0

        if "steering_args" in hparams and hparams["steering_args"] is not None:
            core_kwargs["steering_args"] = _to_plain_python(
                copy.deepcopy(hparams["steering_args"])
            )
        else:
            core_kwargs["steering_args"] = _default_steering_args()

        if "validation_args" in hparams:
            validation_args = hparams["validation_args"]
            assert isinstance(validation_args, Mapping), (
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
        confidence_model_args: Optional[dict[str, Any]] = None,
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
        steering_args: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.use_kernels = use_kernels
        self.confidence_prediction = confidence_prediction
        self.token_level_confidence = token_level_confidence
        self.alpha_pae = alpha_pae
        self.run_trunk_and_structure = run_trunk_and_structure
        self.skip_run_structure = skip_run_structure
        self.bond_type_feature = bond_type_feature
        self.steering_args = steering_args if steering_args is not None else _default_steering_args()

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

        torch._dynamo.config.cache_size_limit = 512  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 512  # noqa: SLF001

        msa_kwargs = _filtered_kwargs(MSAModule, {"token_z": token_z, "token_s": token_s, **msa_args})
        self.msa_module = MSAModule(**msa_kwargs)

        pairformer_kwargs = _filtered_kwargs(
            PairformerModule,
            {"token_s": token_s, "token_z": token_z, **pairformer_args},
        )
        assert "token_s" in pairformer_kwargs and "token_z" in pairformer_kwargs
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
            assert confidence_model_args is not None, (
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
        num_sampling_steps: Optional[int] = None,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        run_confidence_sequentially: bool = True,
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
                assert "sample_atom_coords" in output, (
                    "Structure sampling did not produce sample_atom_coords."
                )
                x_pred = output["sample_atom_coords"]

            output.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    x_pred=x_pred.detach(),
                    feats=feats,
                    pred_distogram_logits=output["pdistogram"][:, :, :, 0].detach(),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                    use_kernels=self.use_kernels,
                )
            )

        return output


class Boltz2Model(PreTrainedModel):
    config_class = Boltz2Config
    base_model_prefix = "core"

    def __init__(self, config: Boltz2Config) -> None:
        super().__init__(config)
        assert isinstance(config.core_kwargs, dict), "config.core_kwargs must be a dictionary."
        self.core = Boltz2InferenceCore(**config.core_kwargs)

    def _init_weights(self, module: nn.Module) -> None:  # noqa: ARG002
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
            kwargs["safe_serialization"] = False
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
        default_recycling_steps: Optional[int] = None,
        default_sampling_steps: Optional[int] = None,
        default_diffusion_samples: Optional[int] = None,
    ) -> "Boltz2Model":
        # Boltz Lightning checkpoints include OmegaConf objects and require full unpickling.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        assert isinstance(checkpoint, dict), "Checkpoint must deserialize to a dictionary."
        _require_key(checkpoint, "hyper_parameters")
        _require_key(checkpoint, "state_dict")

        hparams = checkpoint["hyper_parameters"]
        assert isinstance(hparams, dict), "Checkpoint hyper_parameters must be a dictionary."
        state_dict = checkpoint["state_dict"]
        assert isinstance(state_dict, dict), "Checkpoint state_dict must be a dictionary."

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
        filtered: dict[str, Tensor] = {}
        for key, value in cleaned.items():
            if key in target_keys:
                filtered[key] = value

        missing = sorted(target_keys.difference(filtered.keys()))
        allowed_missing_substrings = [
            ".attention.norm_s.weight",
            ".attention.norm_s.bias",
        ]
        disallowed_missing: list[str] = []
        for key in missing:
            is_allowed = False
            for token in allowed_missing_substrings:
                if token in key:
                    is_allowed = True
            if not is_allowed:
                disallowed_missing.append(key)
        assert len(disallowed_missing) == 0, (
            "Checkpoint is missing required parameters for Boltz2 inference core. "
            f"Missing keys (first 20): {disallowed_missing[:20]}"
        )

        load_result = model.core.load_state_dict(filtered, strict=False)
        loaded_missing = sorted(load_result.missing_keys)
        loaded_disallowed_missing: list[str] = []
        for key in loaded_missing:
            is_allowed = False
            for token in allowed_missing_substrings:
                if token in key:
                    is_allowed = True
            if not is_allowed:
                loaded_disallowed_missing.append(key)
        assert len(loaded_disallowed_missing) == 0, (
            "Model has unexpected missing keys after load_state_dict. "
            f"Missing keys (first 20): {loaded_disallowed_missing[:20]}"
        )
        assert len(load_result.unexpected_keys) == 0
        model.eval()
        return model

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: Optional[int] = None,
        num_sampling_steps: Optional[int] = None,
        diffusion_samples: Optional[int] = None,
        max_parallel_samples: Optional[int] = None,
        run_confidence_sequentially: bool = True,
    ) -> dict[str, Tensor]:
        if recycling_steps is None:
            recycling_steps = self.config.default_recycling_steps
        if num_sampling_steps is None:
            num_sampling_steps = self.config.default_sampling_steps
        if diffusion_samples is None:
            diffusion_samples = self.config.default_diffusion_samples
        return self.core(
            feats=feats,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            diffusion_samples=diffusion_samples,
            max_parallel_samples=max_parallel_samples,
            run_confidence_sequentially=run_confidence_sequentially,
        )

    def _to_model_device(
        self,
        feats: dict[str, Tensor],
        float_dtype: torch.dtype,
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
        recycling_steps: Optional[int] = None,
        num_sampling_steps: Optional[int] = None,
        diffusion_samples: Optional[int] = None,
        max_parallel_samples: Optional[int] = None,
        run_confidence_sequentially: bool = True,
        float_dtype: Optional[torch.dtype] = None,
    ) -> Boltz2StructureOutput:
        if float_dtype is None:
            float_dtype = torch.float32

        feats, template = build_boltz2_features(
            amino_acid_sequence=amino_acid_sequence,
            num_bins=self.config.num_bins,
            atoms_per_window_queries=self.core.input_embedder.atom_encoder.atoms_per_window_queries,
        )
        feats = self._to_model_device(feats, float_dtype=float_dtype)

        with torch.no_grad():
            output = self.forward(
                feats=feats,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                diffusion_samples=diffusion_samples,
                max_parallel_samples=max_parallel_samples,
                run_confidence_sequentially=run_confidence_sequentially,
            )

        sample_atom_coords = output["sample_atom_coords"].detach().cpu()
        atom_pad_mask = feats["atom_pad_mask"][0].detach().cpu()
        plddt = output["plddt"].detach().cpu() if "plddt" in output else None
        complex_plddt = output["complex_plddt"].detach().cpu() if "complex_plddt" in output else None
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
        )

    def save_as_cif(
        self,
        structure_output: Boltz2StructureOutput,
        output_path: str,
        sample_index: int = 0,
    ) -> str:
        assert structure_output.structure_template is not None, (
            "structure_output.structure_template is required for CIF export."
        )
        assert structure_output.sample_atom_coords is not None, (
            "structure_output.sample_atom_coords is required for CIF export."
        )
        assert structure_output.atom_pad_mask is not None, (
            "structure_output.atom_pad_mask is required for CIF export."
        )
        return write_cif(
            structure_template=structure_output.structure_template,
            atom_coords=structure_output.sample_atom_coords,
            atom_mask=structure_output.atom_pad_mask,
            output_path=output_path,
            plddt=structure_output.plddt,
            sample_index=sample_index,
        )
