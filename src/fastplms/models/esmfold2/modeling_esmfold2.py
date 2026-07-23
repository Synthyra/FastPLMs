"""PyTorch ESMFold2 model: the standard released architecture.

Quickstart::

    from transformers import ESMFold2Model

    model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
    open("ubq.pdb", "w").write(model.infer_protein_as_pdb("MQIFVKTLTGKT..."))

For multi-chain, ligand, and MSA inputs, use ``model.input_types`` together
with ``model.fold(...)`` or ``model.prepare_structure_input(...)``.
"""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from ...attention import get_attn_implementation, set_config_attn_implementation

try:
    from fastplms.models.ttt import FastPLMTestTimeTrainingMixin, TTTConfig
except ModuleNotFoundError as error:
    if error.name != "fastplms":
        raise
    from ..ttt import FastPLMTestTimeTrainingMixin, TTTConfig

from .attention import ESMFold2AttentionMixin
from .configuration_esmfold2 import ESMFold2Config, normalize_esmc_id
from .embedding import ESMFold2EmbeddingMixin
from .esmfold2_constants_esm3 import (
    SEQUENCE_BOS_TOKEN,
    SEQUENCE_EOS_TOKEN,
    SEQUENCE_MASK_TOKEN,
    SEQUENCE_PAD_TOKEN,
    SEQUENCE_STANDARD_AA_MAX_TOKEN,
    SEQUENCE_STANDARD_AA_MIN_TOKEN,
    SEQUENCE_VOCAB,
)
from .modeling_esmfold2_common import (
    CHAR_VOCAB_SIZE,
    MAX_ATOMIC_NUMBER,
    MSA_CONDITIONING_INPUT_NAMES,
    NUM_RES_TYPES,
    DiffusionStructureHead,
    FoldingTrunk,
    InputsEmbedder,
    LanguageModelShim,
    MSAPairWeightedAveraging,
    OuterProductMean,
    ResIdxAsymIdSymIdEntityIdEncoding,
    RowAttentionPooling,
    SwiGLUMLP,
    TriangleMultiplicativeUpdate,
    _categorical_mean,
    _compute_intra_token_idx,
    compute_lm_hidden_states,
    gather_rep_atom_coords,
    gather_token_to_atom,
    maybe_apply_msa_column_masking,
    maybe_subsample_msa,
    validate_kernel_backend,
    validate_msa_conditioning_inputs,
)

_ESMC_FP8_LINEAR_SUFFIX = ".attn.out_proj"
_ESMC_FP8_EXPECTED_PROJECTIONS = 80
_EPS = 1e-6
_NONPOLYMER_ID = 4

# Default for the triangle, OPM, and pair-transition l^2 operations. Caps peak
# memory so l around 2k folds on an 80 GB GPU (about 76 GB at chunk=128 for
# l=1438;
# chunk=64 leaves headroom for the largest foldbench targets). Override via
# ``model.set_chunk_size(...)``; pass None to disable chunking (faster for
# short l but OOM-prone past approximately 600).
_DEFAULT_CHUNK_SIZE = 64


@dataclass
class ESMFold2Output(ModelOutput):
    """Transformers-compatible output shared by released and experimental folds.

    ``last_hidden_state`` is the final pair representation. When requested,
    ``hidden_states`` contains the token-input representation followed by the
    final pair representation. The structure trunks do not expose normalized
    post-softmax attention tensors, so ``output_attentions=True`` fails
    explicitly instead of returning incomplete data.
    """

    last_hidden_state: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None
    distogram_logits: Tensor | None = None
    sample_atom_coords: Tensor | None = None
    representative_atom_coords: Tensor | None = None
    atom_pad_mask: Tensor | None = None
    residue_index: Tensor | None = None
    entity_id: Tensor | None = None
    plddt_logits: Tensor | None = None
    plddt: Tensor | None = None
    plddt_per_atom: Tensor | None = None
    plddt_ca: Tensor | None = None
    complex_plddt: Tensor | None = None
    complex_iplddt: Tensor | None = None
    pae_logits: Tensor | None = None
    pae: Tensor | None = None
    pde_logits: Tensor | None = None
    pde: Tensor | None = None
    resolved_logits: Tensor | None = None
    ptm: Tensor | None = None
    iptm: Tensor | None = None
    pair_chains_iptm: Tensor | None = None


def _resolve_structure_output_controls(
    config: ESMFold2Config,
    *,
    output_attentions: bool | None,
    output_hidden_states: bool | None,
    return_dict: bool | None,
) -> tuple[bool, bool]:
    resolved_attentions = (
        config.output_attentions if output_attentions is None else output_attentions
    )
    if resolved_attentions:
        raise NotImplementedError(
            "ESMFold2 does not expose normalized attention tensors from its structure "
            "trunk. output_attentions=True is unsupported."
        )
    resolved_hidden_states = (
        config.output_hidden_states
        if output_hidden_states is None
        else output_hidden_states
    )
    resolved_return_dict = config.use_return_dict if return_dict is None else return_dict
    return bool(resolved_hidden_states), bool(resolved_return_dict)


def _finalize_structure_output(
    output: dict[str, Tensor],
    *,
    token_input_state: Tensor,
    pair_state: Tensor,
    output_hidden_states: bool,
    return_dict: bool,
) -> ESMFold2Output | tuple[Any, ...]:
    model_output = ESMFold2Output(
        last_hidden_state=pair_state,
        hidden_states=(token_input_state, pair_state) if output_hidden_states else None,
        **output,
    )
    return model_output if return_dict else model_output.to_tuple()


class _ESMFold2ESMplusplusAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @property
    def config(self):
        return self.model.config

    def set_attn_implementation(self, attn_implementation: str) -> None:
        """Update ESMC through its Transformers-compatible attention API."""

        self.model.set_attn_implementation(attn_implementation)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        sequence_id: Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        compute_sae: bool = True,
        normalize_sae: bool = False,
    ):
        del return_dict, compute_sae, normalize_sae
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            esmfold2_hidden_states=True,
        )
        if output_hidden_states:
            hidden_states = output.hidden_states
            if hidden_states is None:
                raise RuntimeError("ESM++ did not return requested hidden states.")
            if isinstance(hidden_states, torch.Tensor):
                output.hidden_states = hidden_states
            else:
                output.hidden_states = torch.stack(tuple(hidden_states), dim=0)
        return output


def _load_fastplms_esmplusplus_for_esmfold2(
    esmc_model_path: str,
    attn_backend: str,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool = False,
) -> _ESMFold2ESMplusplusAdapter:
    from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
        ESMplusplusConfig,
        ESMplusplusModel,
    )

    normalized_path = normalize_esmc_id(esmc_model_path)
    source_revision, _ = _manifest_esmc_checkpoint_contract(normalized_path)
    revision_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
    }
    if source_revision is not None:
        revision_kwargs["revision"] = source_revision
    esmc_config = ESMplusplusConfig.from_pretrained(normalized_path, **revision_kwargs)
    set_config_attn_implementation(esmc_config, attn_backend)
    load_kwargs: dict[str, Any] = {
        "config": esmc_config,
        "torch_dtype": dtype,
        **revision_kwargs,
    }
    if device.type == "cuda":
        # Device mapping constructs parameters on the destination GPU instead
        # of materializing the 6B backbone in host memory first.
        load_kwargs["device_map"] = {"": str(device)}
    esmc = ESMplusplusModel.from_pretrained(normalized_path, **load_kwargs)
    if device.type != "cuda":
        esmc = esmc.to(device=device, dtype=dtype)
    else:
        loaded_device = next(esmc.parameters()).device
        if loaded_device != device:
            raise RuntimeError(
                f"ESMC loaded on {loaded_device}, expected direct loading on {device}."
            )
    return _ESMFold2ESMplusplusAdapter(esmc).eval()


def _manifest_esmc_checkpoint_contract(
    esmc_model_path: str,
) -> tuple[str | None, dict[str, str]]:
    """Return the immutable manifest identity for a registered ESMC source.

    Local checkpoint directories deliberately return no Hub revision. A known
    Hub repository is always loaded at the revision and file identities in
    ``models.toml`` instead of following a mutable branch.
    """

    normalized_path = normalize_esmc_id(esmc_model_path)
    try:
        if Path(normalized_path).exists():
            return None, {}
    except OSError:
        # A repository ID may be too long or otherwise invalid as a local path.
        pass

    from fastplms.registry import get_model_registry

    registry = get_model_registry()
    backbone_model = registry.families["esmfold2"].backbone_model
    if backbone_model is None:
        raise RuntimeError("families.esmfold2 must declare backbone_model.")
    spec = registry[backbone_model]
    for checkpoint in (spec.fast, spec.official):
        if checkpoint.repo_id == normalized_path:
            return checkpoint.revision, {
                item.path: item.encoded for item in checkpoint.files
            }
    if "/" in normalized_path:
        raise ValueError(
            f"Remote ESMC source {normalized_path!r} is not the manifest-declared "
            f"ESMFold2 backbone {spec.fast.repo_id!r}."
        )
    return None, {}


ESMCPrecision = Literal["auto", "bf16", "fp32", "fp8"]


@dataclass(frozen=True, slots=True)
class ESMCPrecisionStatus:
    """Resolved ESMC precision and the evidence used to choose it."""

    requested: str
    resolved: str
    reason: str
    device: str
    transformer_engine_version: str | None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _transformer_engine_version() -> str | None:
    try:
        return importlib.metadata.version("transformer-engine")
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_transformer_engine() -> tuple[Any, Any]:
    """Load Transformer Engine lazily so core imports stay dependency-free."""

    try:
        te = importlib.import_module("transformer_engine.pytorch")
        recipe = importlib.import_module("transformer_engine.common.recipe")
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            f"Transformer Engine could not be imported: {type(error).__name__}: {error}"
        ) from error
    if not hasattr(recipe, "Float8CurrentScaling"):
        raise RuntimeError(
            "Transformer Engine does not expose Float8CurrentScaling, which is "
            "required by the validated ESMC FP8 path."
        )
    return te, recipe


def _te_fp8_capability(device: torch.device) -> tuple[bool, str]:
    """Return whether the validated Transformer Engine FP8 path can run."""

    if device.type != "cuda":
        return False, "FP8 requires direct ESMC loading onto a CUDA device."
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable."
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except (AssertionError, RuntimeError, ValueError) as error:
        return False, f"CUDA capability query failed: {error}"
    if not (major >= 9 or (major == 8 and minor >= 9)):
        return False, f"CUDA capability {major}.{minor} does not support FP8."
    try:
        te, _ = _load_transformer_engine()
    except RuntimeError as error:
        return False, str(error)

    probe = getattr(te, "is_fp8_available", None)
    if probe is None:
        try:
            probe = importlib.import_module("transformer_engine.pytorch.fp8").is_fp8_available
        except (ImportError, AttributeError, OSError, RuntimeError) as error:
            return False, f"Transformer Engine has no usable FP8 probe: {error}"
    try:
        try:
            result = probe(return_reason=True)
        except TypeError:
            result = probe()
    except (OSError, RuntimeError) as error:
        return False, f"Transformer Engine FP8 probe failed: {error}"
    if isinstance(result, tuple):
        available = bool(result[0])
        detail = str(result[1]) if len(result) > 1 and result[1] else ""
    else:
        available = bool(result)
        detail = ""
    if not available:
        return False, detail or "Transformer Engine reports FP8 unavailable."
    return True, (
        "Transformer Engine reports FP8 availability; FastPLMs will convert "
        "the validated ESMC attention output projections."
    )


def _resolve_esmc_precision(requested: str, device: torch.device) -> ESMCPrecisionStatus:
    allowed = {"auto", "bf16", "fp32", "fp8"}
    if requested not in allowed:
        raise ValueError(f"precision must be one of {sorted(allowed)}, got {requested!r}.")
    if requested in {"auto", "bf16", "fp32"}:
        resolved = "bf16" if requested == "auto" else requested
        reason = (
            "Automatic precision defaults to BF16; select esmc_precision='fp8' "
            "explicitly to opt in to the validated Transformer Engine path."
            if requested == "auto"
            else "Precision was selected explicitly."
        )
        return ESMCPrecisionStatus(
            requested=requested,
            resolved=resolved,
            reason=reason,
            device=str(device),
            transformer_engine_version=_transformer_engine_version(),
        )
    available, reason = _te_fp8_capability(device)
    if not available:
        raise RuntimeError(f"esmc_precision='fp8' is unavailable: {reason}")
    return ESMCPrecisionStatus(
        requested=requested,
        resolved="fp8",
        reason=reason,
        device=str(device),
        transformer_engine_version=_transformer_engine_version(),
    )


def _install_esmc_backbone(
    model: Any,
    esmc_model_path: str,
    *,
    precision: str,
    device: str | torch.device | None = None,
    local_files_only: bool = False,
) -> None:
    target_device = torch.device(device) if device is not None else model.device
    if target_device.type == "cuda" and target_device.index is None and torch.cuda.is_available():
        target_device = torch.device("cuda", torch.cuda.current_device())
    model_device = torch.device(model.device)
    if model_device.type == "cuda" and model_device.index is None and torch.cuda.is_available():
        model_device = torch.device("cuda", torch.cuda.current_device())
    if target_device != model_device:
        raise ValueError(
            f"ESMC target device {target_device} must match the ESMFold2 device "
            f"{model_device}. Move ESMFold2 before loading or reloading ESMC."
        )
    status = _resolve_esmc_precision(precision, target_device)
    normalized_source = normalize_esmc_id(esmc_model_path)
    source_revision, source_files = _manifest_esmc_checkpoint_contract(normalized_source)
    attention_implementation = get_attn_implementation(model.config)
    model.config.esmc_attn_backend = attention_implementation
    dtype = torch.float32 if status.resolved == "fp32" else torch.bfloat16
    esmc = _load_fastplms_esmplusplus_for_esmfold2(
        esmc_model_path=esmc_model_path,
        attn_backend=attention_implementation,
        device=target_device,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    if esmc.config.hidden_size != model.config.lm_d_model:
        raise ValueError(
            f"ESMFold2 expected lm_d_model={model.config.lm_d_model}, "
            f"but loaded ESMC hidden_size={esmc.config.hidden_size}."
        )
    if esmc.config.num_hidden_layers != model.config.lm_num_layers:
        raise ValueError(
            f"ESMFold2 expected lm_num_layers={model.config.lm_num_layers}, "
            f"but loaded ESMC num_hidden_layers={esmc.config.num_hidden_layers}."
        )
    esmc.eval().requires_grad_(False)
    fp8_module_paths: tuple[str, ...] = ()
    if status.resolved == "fp8":
        fp8_module_paths = _convert_esmc_attention_outputs_to_te(esmc)
        status = ESMCPrecisionStatus(
            requested=status.requested,
            resolved=status.resolved,
            reason=(
                f"{status.reason} Converted {len(fp8_module_paths)} projections; "
                "canonical checkpoint weights remain BF16."
            ),
            device=status.device,
            transformer_engine_version=status.transformer_engine_version,
        )
    model._esmc_source = normalized_source
    model._esmc_source_revision = source_revision
    model._esmc_source_files = source_files
    model._esmc_local_files_only = local_files_only
    model._esmc_precision_policy = precision
    model._esmc_precision_status = status
    model._esmc_fp8 = status.resolved == "fp8"
    model._esmc_fp8_module_paths = fp8_module_paths
    model.config.esmc_precision = precision
    model._esmc = esmc
    model._ttt_lm_head = None


def _drop_transient_esmc_state(
    module: nn.Module,
    state_dict: dict[str, Tensor],
    prefix: str,
    local_metadata: dict[str, Any],
) -> None:
    """Exclude runtime ESMC/TTT modules from canonical folding checkpoints."""

    del module, local_metadata
    transient_prefixes = (f"{prefix}_esmc.", f"{prefix}_ttt_lm_head.")
    for key in tuple(state_dict):
        if key.startswith(transient_prefixes):
            del state_dict[key]


def _reload_esmc_bf16_for_gradients(model: Any, *, reason: str) -> None:
    """Use BF16 temporarily without overwriting the persisted serving policy."""

    policy = model._esmc_precision_policy
    model.reload_esmc(precision="bf16", device=model.device)
    model._esmc_precision_policy = policy
    model.config.esmc_precision = policy
    status = model._esmc_precision_status
    model._esmc_precision_status = ESMCPrecisionStatus(
        requested=policy,
        resolved="bf16",
        reason=reason,
        device=status.device,
        transformer_engine_version=status.transformer_engine_version,
    )


class PairTransition(nn.Module):
    """LayerNorm + SwiGLU feed-forward residual block on the pair representation."""

    def __init__(self, d_model: int, expansion_ratio: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUMLP(d_model, expansion_ratio=expansion_ratio, bias=False)
        self._chunk_size: int | None = _DEFAULT_CHUNK_SIZE

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def forward(self, x: Tensor) -> Tensor:
        if self._chunk_size is None or x.shape[1] <= self._chunk_size:
            return self.ffn(self.norm(x))
        out: list[Tensor] = []
        for s in range(0, x.shape[1], self._chunk_size):
            e = min(s + self._chunk_size, x.shape[1])
            sl = x[:, s:e]
            out.append(self.ffn(self.norm(sl)))
        return torch.cat(out, dim=1)


class ConfidenceHead(nn.Module):
    """Predicts pLDDT, PAE, PDE, resolved-atom probability and distogram bins."""

    boundaries: Tensor

    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__()
        ch = config.confidence_head
        d_single = config.d_single
        d_pair = config.d_pair
        d_inputs = config.inputs.d_inputs

        boundaries = torch.linspace(ch.min_dist, ch.max_dist, ch.distogram_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(ch.distogram_bins, d_pair)

        self.s_norm = nn.LayerNorm(d_single)
        self.s_inputs_to_single = nn.Linear(d_inputs, d_single, bias=False)
        self.s_to_z = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_transpose = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_in1 = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_in2 = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_out = nn.Linear(d_pair, d_pair, bias=False)
        self.s_input_to_s = nn.Linear(d_inputs, d_single, bias=False)
        self.s_inputs_norm = nn.LayerNorm(d_inputs)
        self.z_norm = nn.LayerNorm(d_pair)

        self.row_attention_pooling = RowAttentionPooling(d_pair=d_pair, d_single=d_single)

        pf = ch.folding_trunk
        self.folding_trunk = FoldingTrunk(n_layers=pf.n_layers, d_pair=d_pair, expansion_ratio=4)

        # Heads.
        self.plddt_ln = nn.LayerNorm(d_single)
        max_atoms_per_token = 23
        self.plddt_weight = nn.Parameter(
            torch.zeros(max_atoms_per_token, d_single, ch.num_plddt_bins)
        )

        self.pae_ln = nn.LayerNorm(d_pair)
        self.pae_head = nn.Linear(d_pair, ch.num_pae_bins, bias=False)

        self.pde_ln = nn.LayerNorm(d_pair)
        self.pde_head = nn.Linear(d_pair, ch.num_pde_bins, bias=False)

        self.resolved_ln = nn.LayerNorm(d_single)
        # 2 = resolved logits ([unresolved, resolved]).
        self.resolved_weight = nn.Parameter(torch.zeros(max_atoms_per_token, d_single, 2))

    def set_kernel_backend(self, backend: str | None) -> None:
        self.folding_trunk.set_kernel_backend(backend)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)

    @staticmethod
    def _repeat_batch(x: Tensor, num_diffusion_samples: int) -> Tensor:
        return x if num_diffusion_samples == 1 else x.repeat_interleave(num_diffusion_samples, 0)

    @staticmethod
    def _flatten_sample_axis(x: Tensor) -> Tensor:
        if x.ndim == 4:
            b, mult, n, c = x.shape
            return x.reshape(b * mult, n, c)
        return x

    def forward(
        self,
        s_inputs: Tensor,
        z: Tensor,
        x_pred: Tensor,
        distogram_atom_idx: Tensor,
        token_attention_mask: Tensor,
        atom_to_token: Tensor,
        atom_attention_mask: Tensor,
        asym_id: Tensor,
        mol_type: Tensor,
        num_diffusion_samples: int = 1,
        relative_position_encoding: Tensor | None = None,
        token_bonds_encoding: Tensor | None = None,
    ) -> dict[str, Tensor]:
        s_inputs_normed = self.s_inputs_norm(s_inputs)

        z_base = self.z_norm(z)
        if relative_position_encoding is not None:
            z_base = z_base + relative_position_encoding
        if token_bonds_encoding is not None:
            z_base = z_base + token_bonds_encoding
        z_base = z_base + self.s_to_z(s_inputs_normed).unsqueeze(2)
        z_base = z_base + self.s_to_z_transpose(s_inputs_normed).unsqueeze(1)
        z_base = z_base + self.s_to_z_prod_out(
            self.s_to_z_prod_in1(s_inputs_normed)[:, :, None, :]
            * self.s_to_z_prod_in2(s_inputs_normed)[:, None, :, :]
        )

        pair = self._repeat_batch(z_base, num_diffusion_samples)
        x_pred_flat = self._flatten_sample_axis(x_pred)
        atom_to_token_m = self._repeat_batch(atom_to_token, num_diffusion_samples)
        atom_mask_m = self._repeat_batch(atom_attention_mask, num_diffusion_samples)
        rep_idx_m = self._repeat_batch(distogram_atom_idx, num_diffusion_samples).long()
        mask = self._repeat_batch(token_attention_mask, num_diffusion_samples)
        expanded_batch_size = pair.shape[0]

        rep_coords = gather_rep_atom_coords(x_pred_flat, rep_idx_m)
        rep_distances = torch.cdist(
            rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist"
        )
        distogram_bins = (rep_distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.dist_bin_pairwise_embed(distogram_bins)

        pair_mask = mask[:, :, None].float() * mask[:, None, :].float()

        # FoldingTrunk handles the bf16 cast internally during inference so
        # each block's fused trimul engages. In-place residual avoids an
        # extra fp32 pair allocation.
        with torch.amp.autocast("cuda", enabled=pair.is_cuda, dtype=torch.bfloat16):
            pair_delta = self.folding_trunk(pair, pair_attention_mask=pair_mask)
        pair.add_(pair_delta.float())
        del pair_delta
        single = self.row_attention_pooling(pair, mask)

        atom_mask_f = atom_mask_m.float()
        s_at_atoms = gather_token_to_atom(single, atom_to_token_m)
        s_at_atoms_ln = self.plddt_ln(s_at_atoms)

        intra_idx = _compute_intra_token_idx(atom_to_token_m)
        intra_idx = intra_idx.clamp(max=self.plddt_weight.shape[0] - 1)
        w_plddt = self.plddt_weight[intra_idx]
        plddt_logits = torch.einsum("...c,...cb->...b", s_at_atoms_ln, w_plddt)
        plddt_per_atom = _categorical_mean(plddt_logits, start=0.0, end=1.0)

        sequence_length = single.shape[1]
        plddt_sum = torch.zeros(
            expanded_batch_size,
            sequence_length,
            device=single.device,
            dtype=plddt_per_atom.dtype,
        )
        atom_count = torch.zeros(
            expanded_batch_size,
            sequence_length,
            device=single.device,
            dtype=plddt_per_atom.dtype,
        )
        atom_mask_t = atom_mask_f.to(plddt_per_atom.dtype)
        plddt_sum.scatter_add_(1, atom_to_token_m, plddt_per_atom * atom_mask_t)
        atom_count.scatter_add_(1, atom_to_token_m, atom_mask_t)
        plddt = plddt_sum / atom_count.clamp(min=1e-6)

        complex_plddt = (plddt_per_atom * atom_mask_f).sum(dim=-1) / (
            atom_mask_f.sum(dim=-1) + _EPS
        )

        expanded_type = self._repeat_batch(mol_type, num_diffusion_samples)
        expanded_asym = self._repeat_batch(asym_id, num_diffusion_samples)
        is_ligand = (expanded_type == _NONPOLYMER_ID).float()
        inter_chain = (expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)).float()
        near_contact = (rep_distances < 8).float()
        interface_per_token = (near_contact * inter_chain * (1.0 - is_ligand).unsqueeze(-1)).amax(
            dim=-1
        )
        iplddt_weight = torch.where(
            is_ligand.bool(),
            torch.full_like(interface_per_token, 2.0),
            interface_per_token,
        )
        iplddt_weight_atoms = gather_token_to_atom(
            iplddt_weight.unsqueeze(-1), atom_to_token_m
        ).squeeze(-1)
        atom_iplddt_w = atom_mask_f * iplddt_weight_atoms
        complex_iplddt = (plddt_per_atom * atom_iplddt_w).sum(dim=-1) / (
            atom_iplddt_w.sum(dim=-1) + _EPS
        )

        plddt_ca = plddt_per_atom.gather(1, rep_idx_m)

        # PAE
        pae_logits = self.pae_head(self.pae_ln(pair))
        pae = _categorical_mean(pae_logits, start=0.0, end=32.0).detach()

        # PDE
        pde_logits = self.pde_head(self.pde_ln(pair))
        pde = _categorical_mean(pde_logits, start=0.0, end=32.0).detach()

        # Resolved (per-atom binary).
        s_at_atoms_res = self.resolved_ln(s_at_atoms)
        w_res = self.resolved_weight[intra_idx]
        resolved_logits = torch.einsum("...c,...cb->...b", s_at_atoms_res, w_res)

        # pTM / ipTM from pae_logits.
        n_bins = pae_logits.shape[-1]
        bin_width = 32.0 / n_bins
        bin_centers = torch.arange(0.5 * bin_width, 32.0, bin_width, device=pae_logits.device)
        mask_f = mask.float()
        n_residues = mask_f.sum(dim=-1, keepdim=True)
        d0 = 1.24 * (n_residues.clamp(min=19) - 15) ** (1 / 3) - 1.8
        tm_per_bin = 1 / (1 + (bin_centers / d0) ** 2)
        pae_probs = F.softmax(pae_logits, dim=-1)
        tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(dim=-1)

        pair_mask_2d = mask_f.unsqueeze(-1) * mask_f.unsqueeze(-2)
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (pair_mask_2d.sum(dim=-1) + _EPS)
        ptm = ptm_per_row.max(dim=-1).values

        inter_chain_mask = (
            expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)
        ).float() * pair_mask_2d
        iptm_per_row = (tm_expected * inter_chain_mask).sum(dim=-1) / (
            inter_chain_mask.sum(dim=-1) + _EPS
        )
        iptm = iptm_per_row.max(dim=-1).values

        max_chain_id = int(expanded_asym.max().item()) if expanded_batch_size > 0 else 0
        n_chains = max_chain_id + 1
        pair_chains_iptm = torch.zeros(
            expanded_batch_size,
            n_chains,
            n_chains,
            device=tm_expected.device,
            dtype=tm_expected.dtype,
        )
        for c1 in range(n_chains):
            chain_c1 = (expanded_asym == c1).float() * mask_f
            if chain_c1.sum() == 0:
                continue
            for c2 in range(n_chains):
                chain_c2 = (expanded_asym == c2).float() * mask_f
                pair_m = chain_c1.unsqueeze(-1) * chain_c2.unsqueeze(-2)
                denom = pair_m.sum(dim=(-1, -2)) + _EPS
                pair_chains_iptm[:, c1, c2] = (tm_expected * pair_m).sum(dim=(-1, -2)) / denom

        return {
            "plddt_logits": plddt_logits,
            "plddt": plddt.detach(),
            "plddt_per_atom": plddt_per_atom.detach(),
            "plddt_ca": plddt_ca.detach(),
            "complex_plddt": complex_plddt.detach(),
            "complex_iplddt": complex_iplddt.detach(),
            "pae_logits": pae_logits,
            "pae": pae,
            "pde_logits": pde_logits,
            "pde": pde,
            "resolved_logits": resolved_logits,
            "ptm": ptm.detach(),
            "iptm": iptm.detach(),
            "pair_chains_iptm": pair_chains_iptm.detach(),
        }


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


def _convert_esmc_attention_outputs_to_te(module: nn.Module) -> tuple[str, ...]:
    """Replace the 80 ESMC attention output projections with TE linears.

    Converting every ESMC linear compounds FP8 error across the 80-layer
    network. The validated inference path limits FP8 GEMMs to each layer's
    attention output projection. Transformer Engine retains canonical BF16
    parameters and creates runtime quantization workspaces during autocast.
    """

    te, _ = _load_transformer_engine()
    converted: list[str] = []

    def walk(owner: nn.Module, prefix: str = "") -> None:
        for name, child in tuple(owner.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and path.endswith(_ESMC_FP8_LINEAR_SUFFIX):
                replacement = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    params_dtype=child.weight.dtype,
                    device=child.weight.device,
                )
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    if child.bias is not None:
                        replacement.bias.copy_(child.bias)
                replacement.eval().requires_grad_(False)
                setattr(owner, name, replacement)
                converted.append(path)
            else:
                walk(child, path)

    walk(module)
    if len(converted) != _ESMC_FP8_EXPECTED_PROJECTIONS:
        raise RuntimeError(
            "ESMC FP8 conversion expected exactly "
            f"{_ESMC_FP8_EXPECTED_PROJECTIONS} attention output projections, "
            f"found {len(converted)}."
        )
    return tuple(converted)


@contextmanager
def _lm_precision_context(precision: str, device: torch.device):
    """Apply the resolved ESMC inference precision."""

    if device.type != "cuda" or precision == "fp32":
        yield
        return
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if precision == "fp8":
            te, recipe = _load_transformer_engine()
            fp8_recipe = recipe.Float8CurrentScaling(
                use_power_2_scales=False,
                fp8_format=recipe.Format.HYBRID,
            )
            with te.autocast(enabled=True, recipe=fp8_recipe):
                yield
        else:
            yield


class ESMFold2Model(
    FastPLMTestTimeTrainingMixin,
    ESMFold2EmbeddingMixin,
    ESMFold2AttentionMixin,
    PreTrainedModel,
):
    """ESMFold2: all-atom structure prediction with an ESMC PLM backbone.

    This is the standard released ESMFold2 architecture (uses a linear-
    recurrent trunk, internally referred to as "parcae").

    Forward kwargs that callers commonly override:

    * ``num_loops`` (default ``config.num_loops``): trunk refinement
      loops.
    * ``num_diffusion_samples`` (default ``config.num_diffusion_samples``):
      parallel structure samples; the confidence head re-runs once per
      sample, so memory scales linearly. Pass ``1`` for cheap inference.
    * ``num_sampling_steps`` (default ``config.structure_head.inference_num_steps``):
      diffusion ODE solver steps. Lower for speed, higher for quality.

    Memory / perf knobs:

    * ``model.set_chunk_size(int|None)``: caps l^2 ops (triangle / OPM /
      pair transition) at this token-axis chunk. Default 64: fits
      l approximately 2k on an 80 GB GPU. Pass ``None`` for faster inference
      when l is below 600.
    * ``model.set_kernel_backend(None | "fused" | "cuequivariance")``:
      select kernel backend (None = reference path).
    """

    config_class = ESMFold2Config
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [r"\._extra_state$"]

    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__(config)
        d_inputs = config.inputs.d_inputs
        d_pair = config.d_pair

        self.inputs_embedder = InputsEmbedder(config)
        self.z_init_1 = nn.Linear(d_inputs, d_pair, bias=False)
        self.z_init_2 = nn.Linear(d_inputs, d_pair, bias=False)
        self.rel_pos = ResIdxAsymIdSymIdEntityIdEncoding(
            n_relative_residx_bins=config.n_relative_residx_bins,
            n_relative_chain_bins=config.n_relative_chain_bins,
            d_pair=d_pair,
        )
        self.token_bonds = nn.Linear(1, d_pair, bias=False)
        self.language_model = LanguageModelShim(
            d_z=d_pair, d_model=config.lm_d_model, num_layers=config.lm_num_layers
        )
        self._esmc: nn.Module | None = None
        self._esmc_fp8: bool = False
        self._esmc_fp8_module_paths: tuple[str, ...] = ()
        self._esmc_source: str = config.esmc_id
        self._esmc_source_revision: str | None = None
        self._esmc_source_files: dict[str, str] = {}
        self._esmc_local_files_only = False
        self._esmc_precision_policy: str = str(getattr(config, "esmc_precision", "auto"))
        self._esmc_precision_status = ESMCPrecisionStatus(
            requested=self._esmc_precision_policy,
            resolved="unloaded",
            reason="ESMC has not been loaded.",
            device=str(self.device),
            transformer_engine_version=_transformer_engine_version(),
        )
        self._ttt_lm_head: nn.Module | None = None
        self._esmfold2_input_builder: Any | None = None
        self._kernel_backend: str | None = None

        pf = config.folding_trunk
        self.folding_trunk = FoldingTrunk(n_layers=pf.n_layers, d_pair=d_pair, expansion_ratio=4)
        if config.lm_encoder.enabled:
            self.lm_encoder: FoldingTrunk | None = FoldingTrunk(
                n_layers=config.lm_encoder.n_layers, d_pair=d_pair, expansion_ratio=4
            )
        else:
            self.lm_encoder = None

        self.parcae_input_norm = nn.LayerNorm(d_pair)
        self.parcae_log_a = nn.Parameter(torch.zeros(d_pair))
        parcae_decay_init = math.sqrt(1.0 / 5.0)
        parcae_delta_init = -math.log(parcae_decay_init)
        self.parcae_log_delta = nn.Parameter(
            torch.full((d_pair,), _inverse_softplus(parcae_delta_init), dtype=torch.float32)
        )
        self.parcae_b_cont = nn.Parameter(torch.eye(d_pair))
        self.parcae_readout = nn.Linear(d_pair, d_pair, bias=False)
        nn.init.eye_(self.parcae_readout.weight)
        self.parcae_coda = FoldingTrunk(
            n_layers=config.parcae.coda_n_layers, d_pair=d_pair, expansion_ratio=4
        )

        # Heads --------------------------------------------------------------
        self.structure_head = DiffusionStructureHead(config)
        self.distogram_head = nn.Linear(d_pair, config.structure_head.distogram_bins, bias=True)
        self.confidence_head = ConfidenceHead(config)

        msa_cfg = config.msa_encoder
        self.msa_encoder = None
        if msa_cfg.enabled:
            self.msa_encoder = MSAEncoder(
                d_msa=msa_cfg.d_msa,
                d_pair=d_pair,
                d_inputs=d_inputs,
                d_hidden=msa_cfg.d_hidden,
                n_layers=msa_cfg.n_layers,
                n_heads_msa=msa_cfg.n_heads_msa,
                msa_head_width=msa_cfg.msa_head_width,
            )

        self.post_init()
        self._register_state_dict_hook(_drop_transient_esmc_state)
        self.init_ttt({"lora_target_replace_module": "MultiHeadAttention"})

    @property
    def esmc_precision_status(self) -> ESMCPrecisionStatus:
        return self._esmc_precision_status

    def load_esmc(
        self,
        esmc_model_path: str,
        precision: ESMCPrecision = "auto",
        device: str | torch.device | None = None,
        local_files_only: bool = False,
    ) -> None:
        """Load canonical ESMC weights and resolve the inference precision."""

        _install_esmc_backbone(
            self,
            esmc_model_path,
            precision=precision,
            device=device,
            local_files_only=local_files_only,
        )

    def reload_esmc(
        self,
        precision: ESMCPrecision = "auto",
        device: str | torch.device | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        """Reload canonical weights with the requested precision policy."""

        source = self._esmc_source or self.config.esmc_id
        old_esmc = self._esmc
        old_head = self._ttt_lm_head
        self._esmc = None
        self._esmc_fp8 = False
        self._esmc_fp8_module_paths = ()
        self._ttt_lm_head = None
        del old_esmc, old_head
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.load_esmc(
            source,
            precision=precision,
            device=device,
            local_files_only=(
                self._esmc_local_files_only
                if local_files_only is None
                else local_files_only
            ),
        )

    def _ensure_ttt_bf16(self) -> None:
        if self._esmc_fp8:
            _reload_esmc_bf16_for_gradients(
                self,
                reason="TTT requires BF16; the persisted serving policy is unchanged.",
            )

    def _ensure_ttt_lm_head(self) -> None:
        self._ensure_ttt_bf16()
        if self._esmc is None:
            raise RuntimeError("ESMFold2 TTT requires load_esmc=True.")
        if self._ttt_lm_head is not None:
            return
        from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
            ESMplusplusConfig,
            ESMplusplusForMaskedLM,
        )

        source = self._esmc_source or self.config.esmc_id
        source_revision = self._esmc_source_revision
        if source_revision is None:
            source_revision, _ = _manifest_esmc_checkpoint_contract(source)
        revision_kwargs: dict[str, Any] = {
            "local_files_only": self._esmc_local_files_only,
        }
        if source_revision is not None:
            revision_kwargs["revision"] = source_revision
        esmc_config = ESMplusplusConfig.from_pretrained(
            source,
            **revision_kwargs,
        )
        set_config_attn_implementation(esmc_config, get_attn_implementation(self.config))
        mlm, loading_info = ESMplusplusForMaskedLM.from_pretrained(
            source,
            config=esmc_config,
            output_loading_info=True,
            **revision_kwargs,
        )
        missing_head_keys = [
            key for key in loading_info["missing_keys"] if key.startswith("sequence_head")
        ]
        if missing_head_keys:
            raise RuntimeError(
                "ESMFold2 TTT could not load a pretrained ESM++ MLM head from "
                f"{source}: missing {missing_head_keys}"
            )
        dtype = next(self._esmc.parameters()).dtype
        mlm = mlm.to(device=self.device, dtype=dtype).eval()
        self._ttt_lm_head = mlm.sequence_head
        self._ttt_lm_head.requires_grad_(False)
        del mlm

    def _ttt_get_trainable_modules(self) -> list[nn.Module]:
        self._ensure_ttt_bf16()
        if self._esmc is None:
            raise RuntimeError("ESMFold2 TTT requires load_esmc=True.")
        return [self._esmc]

    def _ttt_tokenize(
        self,
        seq: str | list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if input_ids is not None:
            return input_ids
        if seq is None:
            raise ValueError("Pass either seq or input_ids for ESMFold2 TTT.")
        sequences = [seq] if isinstance(seq, str) else seq
        if not sequences:
            raise ValueError("ESMFold2 TTT requires at least one protein sequence.")
        token_to_id = {token: idx for idx, token in enumerate(SEQUENCE_VOCAB)}
        encoded = []
        for sequence in sequences:
            token_ids = [SEQUENCE_BOS_TOKEN]
            for amino_acid in sequence:
                token_ids.append(token_to_id[amino_acid if amino_acid in token_to_id else "X"])
            token_ids.append(SEQUENCE_EOS_TOKEN)
            encoded.append(token_ids)
        max_len = max(len(token_ids) for token_ids in encoded)
        input_tensor = torch.full(
            (len(encoded), max_len),
            SEQUENCE_PAD_TOKEN,
            dtype=torch.long,
        )
        for row, token_ids in enumerate(encoded):
            input_tensor[row, : len(token_ids)] = torch.tensor(
                token_ids,
                dtype=torch.long,
            )
        return input_tensor

    def _ttt_mask_token(self) -> int:
        return SEQUENCE_MASK_TOKEN

    def _ttt_padding_token(self) -> int:
        return SEQUENCE_PAD_TOKEN

    def _ttt_replacement_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.arange(
            SEQUENCE_STANDARD_AA_MIN_TOKEN,
            SEQUENCE_STANDARD_AA_MAX_TOKEN,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

    def _ttt_non_special_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids >= SEQUENCE_STANDARD_AA_MIN_TOKEN) & (
            input_ids < SEQUENCE_STANDARD_AA_MAX_TOKEN
        )

    def _ttt_predict_logits(
        self,
        batch: torch.Tensor | dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if not isinstance(batch, torch.Tensor):
            raise TypeError("ESMFold2 TTT expects input_ids tensors.")
        self._ensure_ttt_bf16()
        if self._esmc is None:
            raise RuntimeError("ESMFold2 TTT requires load_esmc=True.")
        self._ensure_ttt_lm_head()
        if self._ttt_lm_head is None:
            raise RuntimeError("ESMFold2 TTT MLM head initialization failed.")
        attention_mask = batch.ne(SEQUENCE_PAD_TOKEN)
        output = self._esmc(
            input_ids=batch,
            attention_mask=attention_mask,
            return_dict=True,
            compute_sae=False,
        )
        return self._ttt_lm_head(output.last_hidden_state)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *args, load_esmc: bool = True, **kwargs
    ):
        if cls is ESMFold2Model and "config" not in kwargs:
            config = ESMFold2Config.from_pretrained(pretrained_model_name_or_path, **kwargs)
            if config.type == "experimental":
                raise ValueError(
                    "FastPLMs ESMFold2 supports the released ESMFold2 and "
                    "ESMFold2-Fast checkpoints. Experimental ESMFold2 configs "
                    "are not part of the self-contained AutoModel package."
                )
            kwargs["config"] = config
        # Pop the precision knob before forwarding to the HF loader.
        esmc_precision = kwargs.pop("esmc_precision", None)
        local_files_only = bool(kwargs.get("local_files_only", False))
        output_loading_info = bool(kwargs.get("output_loading_info", False))
        loaded = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if output_loading_info:
            model, loading_info = loaded
        else:
            model = loaded
        if load_esmc:
            model.load_esmc(
                model.config.esmc_id,
                precision=esmc_precision or model.config.esmc_precision,
                local_files_only=local_files_only,
            )
        return (model, loading_info) if output_loading_info else model

    def set_kernel_backend(self, backend: str | None) -> None:
        """Select kernel backend.

        Args:
            backend: ``None`` (reference path), ``"fused"`` (requires the
                unavailable source-built Triton bundle), or
                ``"cuequivariance"`` (requires the ``structure,cueq`` extras
                on a supported Linux CUDA 13 host).
        """
        validate_kernel_backend(backend)
        self.folding_trunk.set_kernel_backend(backend)
        if self.lm_encoder is not None:
            self.lm_encoder.set_kernel_backend(backend)
        self.parcae_coda.set_kernel_backend(backend)
        self.confidence_head.set_kernel_backend(backend)
        self.structure_head.set_kernel_backend(backend)
        self._kernel_backend = backend

    def apply_torch_compile(self, mode: str = "fixed_seqlen", dynamic: bool | None = None) -> None:
        """Compile l^2-heavy blocks.

        ``mode='fixed_seqlen'`` recompiles per l; ``'dynamic_seqlen'`` compiles
        once.

        Does NOT stack with our Triton kernels: call ``set_kernel_backend(None)``
        before compiling.
        """
        if dynamic is None:
            dynamic = mode == "dynamic_seqlen"
        kwargs: dict = {"dynamic": dynamic}

        from .modeling_esmfold2_common import (
            DiffusionModule,
            DiffusionTransformer,
            PairUpdateBlock,
        )

        compile_targets = (
            PairUpdateBlock,
            DiffusionTransformer,
            DiffusionModule,
            MSAEncoderBlock,
        )

        def _maybe_compile(module: nn.Module) -> None:
            if isinstance(module, compile_targets):
                module.forward = torch.compile(module.forward, **kwargs)  # type: ignore[assignment]

        self.apply(_maybe_compile)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)
        if self.lm_encoder is not None:
            self.lm_encoder.set_chunk_size(chunk_size)
        self.parcae_coda.set_chunk_size(chunk_size)
        self.confidence_head.set_chunk_size(chunk_size)
        if self.msa_encoder is not None:
            self.msa_encoder.set_chunk_size(chunk_size)

    def _compute_lm_hidden_states(
        self,
        input_ids: Tensor,
        asym_id: Tensor,
        residue_index: Tensor,
        mol_type: Tensor,
        tok_mask: Tensor,
        lm_mask_pct: float = 0.0,
    ) -> Tensor:
        if self._esmc_fp8 and torch.is_grad_enabled():
            _reload_esmc_bf16_for_gradients(
                self,
                reason=(
                    "Gradient-enabled ESMC execution requires BF16; the persisted "
                    "serving policy is unchanged."
                ),
            )
        if self._esmc is None:
            raise RuntimeError("ESMFold2 requires load_esmc=True for LM feature extraction.")
        # Transformer Engine FP8 kernels require l to be a multiple of 16.
        pad_to = 16 if self._esmc_fp8 else None
        with _lm_precision_context(self._esmc_precision_status.resolved, self.device):
            return compute_lm_hidden_states(
                self._esmc,
                input_ids,
                asym_id,
                residue_index,
                mol_type,
                tok_mask,
                pad_to_multiple=pad_to,
                lm_mask_pct=lm_mask_pct,
                mask_token_id=SEQUENCE_MASK_TOKEN,
            )

    def _discretized_dynamics(self) -> tuple[Tensor, Tensor]:
        delta = F.softplus(self.parcae_log_delta)
        a = torch.exp(-delta * torch.exp(self.parcae_log_a))
        b = delta[:, None] * self.parcae_b_cont
        return a, b

    def _init_pair_state(self, ref: Tensor) -> Tensor:
        std = math.sqrt(2.0 / (5.0 * ref.shape[-1]))
        state = torch.empty_like(ref, dtype=torch.float32)
        nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return state.to(dtype=ref.dtype)

    def _run_one_loop(
        self,
        z: Tensor,
        z_init: Tensor,
        lm_z: Tensor | None,
        _msa_inputs: dict | None,
        pair_mask: Tensor,
        a: Tensor,
        b_mat: Tensor,
        tok_mask: Tensor,
        total_steps: int,
    ) -> Tensor:
        # Helper method (not inline) so per-iter locals free on return:
        # otherwise leaks about 2 GB of l^2 * c_z data into distogram/sample scope.
        # training=True forces dropout under eval(), matching the per-loop
        # dropout strategy used at train time.
        lm_cfg = self.config.lm_encoder
        _per_loop_lm_dropout = (
            lm_z is not None
            and getattr(lm_cfg, "per_loop_lm_dropout", False)
            and getattr(lm_cfg, "lm_dropout", 0.0) > 0.0
        )
        _lm_dropout_p = getattr(lm_cfg, "lm_dropout", 0.0)

        for _ in range(total_steps):
            if _per_loop_lm_dropout:
                if lm_z is None:
                    raise RuntimeError("Per-loop LM dropout requires LM pair features.")
                lm_z_i: Tensor | None = F.dropout(lm_z, p=_lm_dropout_p, training=True)
            else:
                lm_z_i = lm_z

            refined_lm_z: Tensor | None = None
            if lm_z_i is not None and self.lm_encoder is not None:
                refined_lm_z = self.lm_encoder(
                    lm_z_i.to(z_init.dtype), pair_attention_mask=pair_mask
                )

            z_inject_pair = z_init
            if lm_z_i is not None and self.lm_encoder is None:
                z_inject_pair = z_inject_pair + lm_z_i.to(z_inject_pair.dtype)

            if self.msa_encoder is not None and _msa_inputs is not None:
                msa_i, mask_i, hd_i, dv_i = maybe_subsample_msa(
                    _msa_inputs["msa"],
                    _msa_inputs["msa_attention_mask"],
                    _msa_inputs["has_deletion"],
                    _msa_inputs["deletion_value"],
                    max_depth=_msa_inputs["max_depth"],
                    enabled=_msa_inputs["subsample_enabled"],
                )
                b_msa, m, l_msa = msa_i.shape
                msa_oh = F.one_hot(msa_i.permute(0, 2, 1).long(), num_classes=NUM_RES_TYPES).float()
                msa_attn = (
                    mask_i.permute(0, 2, 1).float()
                    if mask_i is not None
                    else tok_mask[:, :, None].expand(-1, -1, m).float()
                )
                # Bias-free MSAEncoder.embed requires zeroed padding.
                msa_oh = msa_oh * msa_attn.unsqueeze(-1)
                hd = (
                    hd_i.permute(0, 2, 1).float()
                    if hd_i is not None
                    else torch.zeros(b_msa, l_msa, m, device=msa_i.device)
                )
                dv = (
                    dv_i.permute(0, 2, 1).float()
                    if dv_i is not None
                    else torch.zeros(b_msa, l_msa, m, device=msa_i.device)
                )
                msa_pair = self.msa_encoder(
                    x_pair=z_inject_pair,
                    x_inputs=_msa_inputs["x_inputs"],
                    msa_oh=msa_oh,
                    has_deletion=hd,
                    deletion_value=dv,
                    msa_attention_mask=msa_attn,
                ).to(z_inject_pair.dtype)
                z_inject_pair = (
                    msa_pair if self.config.msa_encoder_overwrite else (z_inject_pair + msa_pair)
                )

            if refined_lm_z is not None:
                z_inject_pair = z_inject_pair + refined_lm_z.to(z_inject_pair.dtype)

            injected_pair = self.parcae_input_norm(z_inject_pair)
            z = a * z + F.linear(injected_pair.to(z.dtype), b_mat)
            z = self.folding_trunk(z, pair_attention_mask=pair_mask)

        return z

    def forward(
        self,
        token_index: Tensor,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        mol_type: Tensor,
        res_type: Tensor,
        token_bonds: Tensor,
        token_attention_mask: Tensor,
        ref_pos: Tensor,
        ref_element: Tensor,
        ref_charge: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        atom_attention_mask: Tensor,
        atom_to_token: Tensor,
        distogram_atom_idx: Tensor,
        deletion_mean: Tensor | None = None,
        msa: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        lm_hidden_states: Tensor | None = None,
        num_loops: int | None = None,
        num_diffusion_samples: int | None = None,
        num_sampling_steps: int | None = None,
        lm_mask_pct: float | None = None,
        msa_max_depth: int = 1024,
        msa_column_mask_rate: float = 0.1,
        msa_subsample_at_inference: bool = True,
        early_exit: bool = False,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        max_inference_sigma: float | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> ESMFold2Output | tuple[Any, ...]:
        output_hidden_states, return_dict = _resolve_structure_output_controls(
            self.config,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        validate_msa_conditioning_inputs(
            self.config,
            msa=msa,
            msa_attention_mask=msa_attention_mask,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            deletion_mean=deletion_mean,
        )
        tok_mask = token_attention_mask
        atm_mask = atom_attention_mask
        disto_idx = distogram_atom_idx

        n_loops: int = num_loops if num_loops is not None else self.config.num_loops
        n_samples: int = (
            num_diffusion_samples
            if num_diffusion_samples is not None
            else self.config.num_diffusion_samples
        )
        total_steps = max(1, n_loops + 1)

        if res_type.dim() == 2:
            res_type_oh = F.one_hot(res_type.long(), num_classes=NUM_RES_TYPES).float()
            res_type_oh = res_type_oh * tok_mask.unsqueeze(-1).float()
        else:
            res_type_oh = res_type.float()

        if msa is not None:
            msa_oh_profile = F.one_hot(msa.long(), num_classes=NUM_RES_TYPES).float()
            if msa_attention_mask is not None:
                mask_f = msa_attention_mask.float().unsqueeze(-1)
                msa_oh_profile = msa_oh_profile * mask_f
                valid_seq_count = msa_attention_mask.float().sum(dim=1).clamp(min=1)
                profile = msa_oh_profile.sum(dim=1) / valid_seq_count.unsqueeze(-1)
            else:
                profile = msa_oh_profile.mean(dim=1)
        else:
            profile = res_type_oh

        if deletion_mean is None:
            deletion_mean = torch.zeros(
                res_type.shape[0], res_type.shape[1], device=res_type.device
            )

        ref_element_oh = F.one_hot(ref_element.long(), num_classes=MAX_ATOMIC_NUMBER).float()
        ref_atom_name_chars_oh = F.one_hot(
            ref_atom_name_chars.long(), num_classes=CHAR_VOCAB_SIZE
        ).float()
        # Bias-free downstream Linears require zeroed padding.
        atm_mask_f = atm_mask.float()
        ref_element_oh = ref_element_oh * atm_mask_f.unsqueeze(-1)
        ref_atom_name_chars_oh = ref_atom_name_chars_oh * atm_mask_f.unsqueeze(-1).unsqueeze(-1)
        atom_to_token = atom_to_token * atm_mask.long()

        use_amp = ref_pos.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            x_inputs = self.inputs_embedder(
                aatype=res_type_oh,
                profile=profile.float(),
                deletion_mean=deletion_mean.float(),
                ref_pos=ref_pos,
                atom_attention_mask=atm_mask,
                ref_space_uid=ref_space_uid,
                ref_charge=ref_charge,
                ref_element=ref_element_oh,
                ref_atom_name_chars=ref_atom_name_chars_oh,
                atom_to_token=atom_to_token,
            )

            z_init = self.z_init_1(x_inputs).unsqueeze(2) + self.z_init_2(x_inputs).unsqueeze(1)

            relative_position_encoding = self.rel_pos(
                residue_index=residue_index,
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                token_index=token_index,
            )
            token_bonds_encoding = self.token_bonds(token_bonds.float())
            z_init = z_init + relative_position_encoding + token_bonds_encoding

            if lm_hidden_states is None and input_ids is not None and self._esmc is not None:
                lm_hidden_states = self._compute_lm_hidden_states(
                    input_ids,
                    asym_id,
                    residue_index,
                    mol_type,
                    tok_mask,
                    lm_mask_pct=(self.config.lm_mask_pct if lm_mask_pct is None else lm_mask_pct),
                )
            lm_z: Tensor | None = None
            if lm_hidden_states is not None:
                lm_z = self.language_model(lm_hidden_states.detach())
            del lm_hidden_states

            pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()

            z = self._init_pair_state(z_init)

            a, b = self._discretized_dynamics()
            a = a.view(1, 1, 1, -1).to(device=z.device, dtype=z.dtype)
            b_mat = b.to(device=z.device, dtype=z.dtype)

            _msa_inputs: dict | None = None
            if self.msa_encoder is not None and msa is not None:
                msa_attention_mask = maybe_apply_msa_column_masking(
                    msa_attention_mask,
                    msa_column_mask_rate,
                )
                _msa_inputs = dict(
                    x_inputs=x_inputs,
                    msa=msa,
                    msa_attention_mask=msa_attention_mask,
                    has_deletion=has_deletion,
                    deletion_value=deletion_value,
                    max_depth=msa_max_depth,
                    subsample_enabled=msa_subsample_at_inference,
                )

            # Method call (not inline loop) frees per-iteration l^2 * c_z locals.
            z = self._run_one_loop(
                z=z,
                z_init=z_init,
                lm_z=lm_z,
                _msa_inputs=_msa_inputs,
                pair_mask=pair_mask,
                a=a,
                b_mat=b_mat,
                tok_mask=tok_mask,
                total_steps=total_steps,
            )
            del z_init, lm_z, _msa_inputs, a, b_mat

            z = self.parcae_readout(z)
            z = self.parcae_coda(z, pair_attention_mask=pair_mask)

            z = z.float()
        distogram_logits = self.distogram_head(z + z.transpose(-2, -3))

        structure_output = self.structure_head.sample(
            z_trunk=z,
            s_inputs=x_inputs,
            s_trunk=None,
            relative_position_encoding=relative_position_encoding,
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_mask=atm_mask,
            ref_element=ref_element_oh,
            ref_atom_name_chars=ref_atom_name_chars_oh,
            ref_space_uid=ref_space_uid,
            tok_idx=atom_to_token,
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            token_index=token_index,
            sym_id=sym_id,
            token_attention_mask=tok_mask,
            num_diffusion_samples=n_samples,
            num_sampling_steps=num_sampling_steps,
            max_inference_sigma=max_inference_sigma,
            noise_scale=noise_scale,
            step_scale=step_scale,
            return_atom_repr=False,
            denoising_early_exit_rmsd=(0.10 if early_exit else None),
        )

        sample_coords = structure_output["sample_atom_coords"]
        if sample_coords is None:
            raise RuntimeError("ESMFold2 structure sampling did not return coordinates.")
        output: dict[str, Tensor] = {"distogram_logits": distogram_logits}
        output["sample_atom_coords"] = sample_coords

        confidence_output = self.confidence_head(
            s_inputs=x_inputs.detach(),
            z=z.detach().float(),
            x_pred=sample_coords.detach(),
            distogram_atom_idx=disto_idx,
            token_attention_mask=tok_mask,
            atom_to_token=atom_to_token,
            atom_attention_mask=atm_mask,
            asym_id=asym_id,
            mol_type=mol_type,
            num_diffusion_samples=n_samples,
            relative_position_encoding=relative_position_encoding.detach(),
            token_bonds_encoding=token_bonds_encoding.detach(),
        )
        output.update(confidence_output)
        output["atom_pad_mask"] = atm_mask.unsqueeze(0) if atm_mask.dim() == 1 else atm_mask
        output["residue_index"] = residue_index
        output["entity_id"] = entity_id
        return _finalize_structure_output(
            output,
            token_input_state=x_inputs,
            pair_state=z,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def infer_protein(self, seq: str, **forward_kwargs) -> ESMFold2Output:
        from .protein_utils import prepare_protein_features

        if forward_kwargs.pop("return_dict", True) is not True:
            raise ValueError(
                "infer_protein always returns a mapping; return_dict=False is invalid."
            )
        features = prepare_protein_features(seq)
        if not self.config.msa_conditioning:
            for name in MSA_CONDITIONING_INPUT_NAMES:
                features.pop(name, None)
        features = {k: v.to(self.device) for k, v in features.items()}
        return self(**features, **forward_kwargs, return_dict=True)

    @property
    def input_builder(self):
        if self._esmfold2_input_builder is None:
            from .esmfold2_processor import ESMFold2InputBuilder

            self._esmfold2_input_builder = ESMFold2InputBuilder()
        return self._esmfold2_input_builder

    @property
    def input_types(self):
        from . import esmfold2_types

        return esmfold2_types

    def prepare_structure_input(self, input, seed: int | None = None):
        return self.input_builder.prepare_model_input(
            self,
            input,
            seed=seed,
            device=self.device,
        )

    def fold(
        self,
        input,
        *,
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        num_diffusion_samples: int = 1,
        seed: int | None = None,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        max_inference_sigma: int | None = None,
        early_exit: bool = False,
        complex_id: str = "pred",
    ):
        return self.input_builder.fold(
            self,
            input,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            seed=seed,
            noise_scale=noise_scale,
            step_scale=step_scale,
            max_inference_sigma=max_inference_sigma,
            early_exit=early_exit,
            complex_id=complex_id,
        )

    def _fold_protein_no_ttt(
        self,
        sequence: str,
        *,
        chain_id: str = "A",
        msa: Any | None = None,
        msa_path: str | Path | None = None,
        msa_max_sequences: int | None = None,
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        num_diffusion_samples: int = 1,
        seed: int | None = None,
        complex_id: str = "pred",
    ):
        from .esmfold2_types import MSA, ProteinInput, StructurePredictionInput

        if msa is not None and msa_path is not None:
            raise ValueError("Pass at most one of msa or msa_path.")
        if msa_path is not None:
            msa = MSA.from_a3m(msa_path, max_sequences=msa_max_sequences)
        if msa is not None:
            query = str(msa.query).replace("-", "").upper()
            if query != sequence.upper():
                raise ValueError(
                    "MSA query does not match sequence: "
                    f"expected {sequence.upper()!r}, got {query!r}"
                )

        input = StructurePredictionInput(
            sequences=[ProteinInput(id=chain_id, sequence=sequence, msa=msa)]
        )
        return self.fold(
            input,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            seed=seed,
            complex_id=complex_id,
        )

    @staticmethod
    def _ttt_mean_plddt(result) -> float:
        if result.plddt is None:
            raise RuntimeError("ESMFold2 result has no pLDDT tensor.")
        return float(result.plddt.float().mean().item())

    def _ttt_select_result(self, result):
        if isinstance(result, list):
            if not result:
                raise RuntimeError("ESMFold2 fold returned an empty result list.")
            return max(result, key=self._ttt_mean_plddt)
        return result

    def _ttt_eval_step(
        self,
        step: int,
        loss: float,
        seq: str | list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], float | None]:
        del input_ids
        if not isinstance(seq, str):
            raise TypeError("ESMFold2 fold TTT is protein-only and sequence-string only.")
        fold_kwargs = kwargs["fold_kwargs"]
        was_training = self.training
        self.eval()
        try:
            result = self._fold_protein_no_ttt(seq, **fold_kwargs)
        finally:
            self.train(was_training)
        selected = self._ttt_select_result(result)
        plddt = self._ttt_mean_plddt(selected)
        return {
            "step": step,
            "loss": loss,
            "plddt": plddt,
            "result": selected,
        }, plddt

    def fold_protein(
        self,
        sequence: str,
        *,
        chain_id: str = "A",
        msa: Any | None = None,
        msa_path: str | Path | None = None,
        msa_max_sequences: int | None = None,
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        num_diffusion_samples: int = 1,
        seed: int | None = None,
        complex_id: str = "pred",
        ttt: bool = False,
        ttt_config: TTTConfig | dict[str, Any] | None = None,
    ):
        if ttt:
            return self.fold_protein_ttt(
                sequence=sequence,
                chain_id=chain_id,
                msa=msa,
                msa_path=msa_path,
                msa_max_sequences=msa_max_sequences,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=num_diffusion_samples,
                seed=seed,
                complex_id=complex_id,
                ttt_config=ttt_config,
            )
        return self._fold_protein_no_ttt(
            sequence=sequence,
            chain_id=chain_id,
            msa=msa,
            msa_path=msa_path,
            msa_max_sequences=msa_max_sequences,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            seed=seed,
            complex_id=complex_id,
        )

    def fold_protein_ttt(
        self,
        sequence: str,
        *,
        chain_id: str = "A",
        msa: Any | None = None,
        msa_path: str | Path | None = None,
        msa_max_sequences: int | None = None,
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        num_diffusion_samples: int = 1,
        seed: int | None = None,
        complex_id: str = "pred",
        ttt_config: TTTConfig | dict[str, Any] | None = None,
    ):
        self._ensure_ttt_bf16()
        if self._esmc is None:
            raise RuntimeError("ESMFold2 TTT requires load_esmc=True.")
        fold_kwargs = {
            "chain_id": chain_id,
            "msa": msa,
            "msa_path": msa_path,
            "msa_max_sequences": msa_max_sequences,
            "num_loops": num_loops,
            "num_sampling_steps": num_sampling_steps,
            "num_diffusion_samples": num_diffusion_samples,
            "seed": seed,
            "complex_id": complex_id,
        }
        baseline = self._ttt_select_result(self._fold_protein_no_ttt(sequence, **fold_kwargs))
        baseline_plddt = self._ttt_mean_plddt(baseline)
        best_result = baseline
        best_plddt = baseline_plddt
        best_step = 0
        step_plddts = [baseline_plddt]

        cfg = self.ttt_config.merged(ttt_config).merged(
            {"eval_each_step": True, "automatic_best_state_reset": False}
        )
        try:
            metrics = self.ttt(
                seq=sequence,
                ttt_config=cfg,
                fold_kwargs=fold_kwargs,
            )
            for step_metric in metrics["step_metrics"]:
                step_plddt = step_metric["plddt"]
                step_plddts.append(step_plddt)
                if step_plddt > best_plddt:
                    best_plddt = step_plddt
                    best_step = step_metric["step"]
                    best_result = step_metric["result"]
            best_result.ttt_metrics = {
                "losses": metrics["losses"],
                "step_plddts": step_plddts,
                "baseline_plddt": baseline_plddt,
                "best_plddt": best_plddt,
                "best_step": best_step,
            }
            return best_result
        finally:
            if "_ttt_initialized" in self.__dict__ and self._ttt_initialized:
                self.ttt_reset()

    @staticmethod
    def result_to_cif(result) -> str:
        if isinstance(result, list):
            raise TypeError("Pass one MolecularComplexResult at a time.")
        return result.complex.to_mmcif()

    @staticmethod
    def result_to_pdb(result) -> str:
        if isinstance(result, list):
            raise TypeError("Pass one MolecularComplexResult at a time.")
        return result.complex.to_protein_complex().to_pdb_string()

    def save_as_cif(self, result, output_path: str | Path) -> None:
        Path(output_path).write_text(self.result_to_cif(result))

    def save_as_pdb(self, result, output_path: str | Path) -> None:
        Path(output_path).write_text(self.result_to_pdb(result))

    def infer_protein_as_cif(self, seq: str, **forward_kwargs) -> str:
        return self.result_to_cif(self.fold_protein(seq, **forward_kwargs))

    def infer_protein_as_pdb(self, seq: str, **forward_kwargs) -> str:
        return self.result_to_pdb(self.fold_protein(seq, **forward_kwargs))


class MSAEncoderBlock(nn.Module):
    """One MSA encoder block: OPM into pair, MSA pair-weighted averaging, triangle update."""

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_hidden: int,
        n_heads_msa: int,
        msa_head_width: int,
        is_final_block: bool = False,
    ) -> None:
        super().__init__()
        self.is_final_block = is_final_block
        self.outer_product_mean = OuterProductMean(d_msa, d_hidden, d_pair)
        if not is_final_block:
            self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
                d_msa, d_pair, n_heads_msa, msa_head_width
            )
            self.msa_transition = PairTransition(d_msa, expansion_ratio=4)
        self.tri_mul_out = TriangleMultiplicativeUpdate(dim=d_pair, _outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(dim=d_pair, _outgoing=False)
        self.pair_transition = PairTransition(d_pair, expansion_ratio=4)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.outer_product_mean.set_chunk_size(chunk_size)
        self.tri_mul_out.set_chunk_size(chunk_size)
        self.tri_mul_in.set_chunk_size(chunk_size)
        if not self.is_final_block:
            self.msa_transition.set_chunk_size(chunk_size)
        self.pair_transition.set_chunk_size(chunk_size)

    def forward(
        self,
        m: Tensor,
        pair: Tensor,
        msa_attention_mask: Tensor,
        pair_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pair = pair + self.outer_product_mean(m, msa_attention_mask)
        if not self.is_final_block:
            m = m + self.msa_pair_weighted_averaging(m, pair, pair_attention_mask)
            m = m + self.msa_transition(m)
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
        pair = pair + self.pair_transition(pair)
        return m, pair


class MSAEncoder(nn.Module):
    """Stack of [`MSAEncoderBlock`] layers that conditions the pair on an MSA."""

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_inputs: int,
        d_hidden: int = 32,
        n_layers: int = 4,
        n_heads_msa: int = 8,
        msa_head_width: int = 16,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(35, d_msa, bias=False)
        self.project_inputs = nn.Linear(d_inputs, d_msa, bias=False)
        self.blocks = nn.ModuleList(
            [
                MSAEncoderBlock(
                    d_msa=d_msa,
                    d_pair=d_pair,
                    d_hidden=d_hidden,
                    n_heads_msa=n_heads_msa,
                    msa_head_width=msa_head_width,
                    is_final_block=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

    def set_chunk_size(self, chunk_size: int | None) -> None:
        for block in self.blocks:
            cast(MSAEncoderBlock, block).set_chunk_size(chunk_size)

    def forward(
        self,
        x_pair: Tensor,
        x_inputs: Tensor,
        msa_oh: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        msa_attention_mask: Tensor,
    ) -> Tensor:
        # Every input tensor is pre-transposed to shape (b, l, m, ...) before this call.
        m_feat = torch.cat(
            [msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1
        )
        m = self.embed(m_feat) + self.project_inputs(x_inputs).unsqueeze(2)
        tok_mask = msa_attention_mask[:, :, 0].bool()
        pair_attention_mask = tok_mask.unsqueeze(2) & tok_mask.unsqueeze(1)
        for block in self.blocks:
            m, x_pair = block(m, x_pair, msa_attention_mask, pair_attention_mask)
        return x_pair
