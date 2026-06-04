"""FastPLMs ESMFold2 experimental architecture.

This module supports Biohub's experimental binder-design checkpoints. The
released ESMFold2 architecture in ``modeling_esmfold2.py`` intentionally
rejects those configs because the experimental trunk uses explicit pair-loop
re-injection and a different confidence/MSA stack.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_utils import PreTrainedModel

from .configuration_esmfold2 import ESMFold2Config
from .modeling_esmfold2 import (
    _convert_te_modules_to_fp8_inplace,
    _lm_precision_context,
)
from .modeling_esmfold2_common import (
    CHAR_VOCAB_SIZE,
    MAX_ATOMIC_NUMBER,
    NUM_RES_TYPES,
    DiffusionModule,
    DiffusionStructureHead,
    DiffusionTransformer,
    FoldingTrunk,
    InputsEmbedder,
    LanguageModelShim,
    MSAPairWeightedAveraging,
    OuterProductMean,
    PairUpdateBlock,
    ResIdxAsymIdSymIdEntityIdEncoding,
    RowAttentionPooling,
    SwiGLUMLP,
    TriangleMultiplicativeUpdate,
    _categorical_mean,
    _compute_intra_token_idx,
    _seed_context,
    compute_lm_hidden_states,
    gather_rep_atom_coords,
    gather_token_to_atom,
)

_EPS = 1e-5
_NONPOLYMER_ID = 3


class ConfidenceHead(nn.Module):
    """Experimental confidence head predicting pLDDT, PAE, pTM, and ipTM."""

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
        self.row_attention_pooling = RowAttentionPooling(
            d_pair=d_pair, d_single=d_single
        )

        pf = ch.folding_trunk
        self.folding_trunk = FoldingTrunk(
            n_layers=pf.n_layers, d_pair=d_pair, expansion_ratio=4
        )

        self.plddt_ln = nn.LayerNorm(d_single)
        max_atoms_per_token = 23
        self.plddt_weight = nn.Parameter(
            torch.zeros(max_atoms_per_token, d_single, ch.num_plddt_bins)
        )
        self.pae_head = nn.Linear(d_pair, ch.num_pae_bins, bias=False)

    def set_kernel_backend(self, backend: str | None) -> None:
        self.folding_trunk.set_kernel_backend(backend)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)

    @staticmethod
    def _repeat_batch(x: Tensor, num_diffusion_samples: int) -> Tensor:
        if num_diffusion_samples == 1:
            return x
        return x.repeat_interleave(num_diffusion_samples, 0)

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
        batch_mult = pair.shape[0]

        rep_coords = gather_rep_atom_coords(x_pred_flat, rep_idx_m)
        rep_distances = torch.cdist(
            rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist"
        )
        distogram_bins = (
            (rep_distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        )
        pair = pair + self.dist_bin_pairwise_embed(distogram_bins)

        pair_mask = mask[:, :, None].float() * mask[:, None, :].float()
        pair = pair + self.folding_trunk(pair, pair_attention_mask=pair_mask)
        single = self.row_attention_pooling(pair, mask)

        atom_mask_f = atom_mask_m.float()
        s_at_atoms = gather_token_to_atom(single, atom_to_token_m)
        s_at_atoms = self.plddt_ln(s_at_atoms)
        intra_idx = _compute_intra_token_idx(atom_to_token_m)
        intra_idx = intra_idx.clamp(max=self.plddt_weight.shape[0] - 1)
        plddt_weight = self.plddt_weight[intra_idx]
        plddt_logits = torch.einsum("...c,...cb->...b", s_at_atoms, plddt_weight)
        plddt_per_atom = _categorical_mean(plddt_logits, start=0.0, end=1.0)

        length = single.shape[1]
        plddt_sum = torch.zeros(
            batch_mult, length, device=single.device, dtype=plddt_per_atom.dtype
        )
        atom_count = torch.zeros(
            batch_mult, length, device=single.device, dtype=plddt_per_atom.dtype
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
        inter_chain = (
            expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)
        ).float()
        near_contact = (rep_distances < 8).float()
        interface_per_token = (
            near_contact * inter_chain * (1.0 - is_ligand).unsqueeze(-1)
        ).amax(dim=-1)
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

        pae_logits = self.pae_head(pair)
        pae = _categorical_mean(pae_logits, start=0.0, end=32.0).detach()

        n_bins = pae_logits.shape[-1]
        bin_width = 32.0 / n_bins
        bin_centers = torch.arange(
            0.5 * bin_width, 32.0, bin_width, device=pae_logits.device
        )
        mask_f = mask.float()
        n_res = mask_f.sum(dim=-1, keepdim=True)
        d0 = 1.24 * (n_res.clamp(min=19) - 15) ** (1 / 3) - 1.8
        tm_per_bin = 1 / (1 + (bin_centers / d0) ** 2)
        pae_probs = F.softmax(pae_logits, dim=-1)
        tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(dim=-1)

        pair_mask_2d = mask_f.unsqueeze(-1) * mask_f.unsqueeze(-2)
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (
            pair_mask_2d.sum(dim=-1) + _EPS
        )
        ptm = ptm_per_row.max(dim=-1).values

        inter_chain_mask = (
            expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)
        ).float() * pair_mask_2d
        iptm_per_row = (tm_expected * inter_chain_mask).sum(dim=-1) / (
            inter_chain_mask.sum(dim=-1) + _EPS
        )
        iptm = iptm_per_row.max(dim=-1).values

        max_chain_id = int(expanded_asym.max().item()) if batch_mult > 0 else 0
        n_chains = max_chain_id + 1
        pair_chains_iptm = torch.zeros(
            batch_mult,
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
                pair_chains_iptm[:, c1, c2] = (tm_expected * pair_m).sum(
                    dim=(-1, -2)
                ) / denom

        return {
            "plddt_logits": plddt_logits,
            "plddt": plddt.detach(),
            "plddt_per_atom": plddt_per_atom.detach(),
            "plddt_ca": plddt_ca.detach(),
            "complex_plddt": complex_plddt.detach(),
            "complex_iplddt": complex_iplddt.detach(),
            "pae_logits": pae_logits,
            "pae": pae,
            "ptm": ptm.detach(),
            "iptm": iptm.detach(),
            "pair_chains_iptm": pair_chains_iptm.detach(),
        }


class _TransitionFFN(nn.Module):
    def __init__(self, d_model: int, expansion_ratio: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUMLP(d_model, expansion_ratio=expansion_ratio, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(self.norm(x))


class MSAEncoderBlock(nn.Module):
    """One experimental MSA update block."""

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_hidden: int = 32,
        n_heads_msa: int = 8,
        msa_head_width: int = 32,
    ) -> None:
        super().__init__()
        self.outer_product_mean = OuterProductMean(
            d_msa, d_hidden, d_pair, divide_outer_before_proj=True
        )
        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
            d_msa, d_pair, n_heads_msa, msa_head_width
        )
        self.msa_transition = _TransitionFFN(d_msa, expansion_ratio=4)
        self.tri_mul_out = TriangleMultiplicativeUpdate(dim=d_pair, _outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(dim=d_pair, _outgoing=False)
        self.pair_transition = _TransitionFFN(d_pair, expansion_ratio=4)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.outer_product_mean.set_chunk_size(chunk_size)
        self.tri_mul_out.set_chunk_size(chunk_size)
        self.tri_mul_in.set_chunk_size(chunk_size)

    def forward(
        self,
        msa_repr: Tensor,
        pair_repr: Tensor,
        msa_attention_mask: Tensor,
        pair_attention_mask: Tensor,
        msa_track_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        mask4d = (
            msa_track_mask[:, None, None, None].to(dtype=msa_repr.dtype)
            if msa_track_mask is not None
            else None
        )

        pair_mask4d = mask4d[:, :, :1] if mask4d is not None else None

        msa_update = self.msa_pair_weighted_averaging(
            msa_repr, pair_repr, pair_attention_mask
        )
        if mask4d is not None:
            msa_update = msa_update * mask4d
        msa_repr = msa_repr + msa_update

        msa_transition = self.msa_transition(msa_repr)
        if mask4d is not None:
            msa_transition = msa_transition * mask4d
        msa_repr = msa_repr + msa_transition

        pair_opm = self.outer_product_mean(msa_repr, msa_attention_mask)
        if pair_mask4d is not None:
            pair_opm = pair_opm * pair_mask4d
        pair_repr = pair_repr + pair_opm

        pair_out = self.tri_mul_out(pair_repr, mask=pair_attention_mask)
        if pair_mask4d is not None:
            pair_out = pair_out * pair_mask4d
        pair_repr = pair_repr + pair_out

        pair_in = self.tri_mul_in(pair_repr, mask=pair_attention_mask)
        if pair_mask4d is not None:
            pair_in = pair_in * pair_mask4d
        pair_repr = pair_repr + pair_in

        pair_transition = self.pair_transition(pair_repr)
        if pair_mask4d is not None:
            pair_transition = pair_transition * pair_mask4d
        pair_repr = pair_repr + pair_transition
        return msa_repr, pair_repr


class MSAEncoder(nn.Module):
    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_inputs: int,
        d_hidden: int = 32,
        n_layers: int = 4,
        n_heads_msa: int = 8,
        msa_head_width: int = 32,
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
                )
                for _ in range(n_layers)
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
        batch_size, _, depth = msa_attention_mask.shape
        m_feat = torch.cat(
            [msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)],
            dim=-1,
        )
        m = self.embed(m_feat) + self.project_inputs(x_inputs).unsqueeze(2)
        if depth > 1:
            msa_track_mask = msa_attention_mask[:, :, 1:].any(dim=(1, 2))
        else:
            msa_track_mask = torch.zeros(
                batch_size, dtype=torch.bool, device=x_pair.device
            )
        tok_mask = msa_attention_mask[:, :, 0]
        pair_attention_mask = tok_mask.unsqueeze(2) * tok_mask.unsqueeze(1)
        for block in self.blocks:
            m, x_pair = cast(MSAEncoderBlock, block)(
                m,
                x_pair,
                msa_attention_mask,
                pair_attention_mask,
                msa_track_mask,
            )
        return x_pair * msa_track_mask[:, None, None, None].to(dtype=x_pair.dtype)


class ESMFold2ExperimentalModel(PreTrainedModel):
    """Experimental ESMFold2 architecture used by binder-design checkpoints."""

    config_class = ESMFold2Config
    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$"]

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
        self._esmc_fp8 = False
        self._esmfold2_input_builder: Any | None = None

        pf = config.folding_trunk
        self.folding_trunk = FoldingTrunk(
            n_layers=pf.n_layers, d_pair=d_pair, expansion_ratio=4
        )
        self.pair_loop_proj = nn.Sequential(
            nn.LayerNorm(d_pair), nn.Linear(d_pair, d_pair, bias=False)
        )
        nn.init.zeros_(cast(nn.Linear, self.pair_loop_proj[1]).weight)

        self.structure_head = DiffusionStructureHead(config)
        self.distogram_head = nn.Linear(
            d_pair, config.structure_head.distogram_bins, bias=True
        )
        self.confidence_head: ConfidenceHead | None = (
            ConfidenceHead(config) if config.confidence_head.enabled else None
        )

        msa_cfg = config.msa_encoder
        self.msa_encoder: MSAEncoder | None = None
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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_kernel_backend(self, backend: str | None) -> None:
        self.folding_trunk.set_kernel_backend(backend)
        if self.confidence_head is not None:
            self.confidence_head.set_kernel_backend(backend)
        self.structure_head.set_kernel_backend(backend)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)
        if self.confidence_head is not None:
            self.confidence_head.set_chunk_size(chunk_size)
        if self.msa_encoder is not None:
            self.msa_encoder.set_chunk_size(chunk_size)

    def configure_lm_dropout(
        self,
        lm_dropout: float,
        *,
        force_lm_dropout_during_inference: bool = True,
    ) -> None:
        self.config.lm_dropout = lm_dropout
        self.config.force_lm_dropout_during_inference = (
            force_lm_dropout_during_inference
        )

    def load_esmc(self, esmc_model_path: str, precision: str = "bf16") -> None:
        from .modeling_esmc import ESMCModel

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp8": torch.bfloat16,
        }
        if precision not in dtype_map:
            raise ValueError(
                f"precision must be one of {list(dtype_map)}, got {precision!r}"
            )
        esmc = (
            ESMCModel.from_pretrained(esmc_model_path)
            .to(device=self.device, dtype=dtype_map[precision])
            .eval()
        )
        for parameter in esmc.parameters():
            parameter.requires_grad_(False)
        if precision == "fp8":
            with torch.no_grad():
                _convert_te_modules_to_fp8_inplace(esmc)
            self._esmc_fp8 = True
        else:
            self._esmc_fp8 = False
        self._esmc = esmc

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        load_esmc: bool = True,
        **kwargs,
    ):
        if "config" not in kwargs:
            kwargs["config"] = ESMFold2Config.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        esmc_precision = kwargs.pop("esmc_precision", "bf16")
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if load_esmc:
            model.load_esmc(model.config.esmc_id, precision=esmc_precision)
        return model

    def apply_torch_compile(
        self, mode: str = "fixed_seqlen", dynamic: bool | None = None
    ) -> None:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = 512
        torch._dynamo.config.accumulated_cache_size_limit = 512
        torch._dynamo.config.capture_scalar_outputs = True

        if dynamic is None:
            dynamic = mode == "dynamic_seqlen"
        compile_kwargs: dict[str, bool] = {"dynamic": dynamic}
        compile_targets = (
            PairUpdateBlock,
            DiffusionTransformer,
            DiffusionModule,
            MSAEncoderBlock,
        )

        def _maybe_compile(module: nn.Module) -> None:
            if isinstance(module, compile_targets):
                module.forward = torch.compile(module.forward, **compile_kwargs)

        self.apply(_maybe_compile)

    def _compute_lm_hidden_states(
        self,
        input_ids: Tensor,
        asym_id: Tensor,
        residue_index: Tensor,
        mol_type: Tensor,
        tok_mask: Tensor,
    ) -> Tensor:
        assert self._esmc is not None
        pad_to = 8 if self._esmc_fp8 else None
        with _lm_precision_context(self._esmc_fp8):
            return compute_lm_hidden_states(
                self._esmc,
                input_ids,
                asym_id,
                residue_index,
                mol_type,
                tok_mask,
                pad_to_multiple=pad_to,
            )

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
        res_type_soft: Tensor | None = None,
        num_loops: int | None = None,
        num_diffusion_samples: int | None = None,
        num_sampling_steps: int | None = None,
        early_exit: bool = False,
        seed: int | None = None,
        calculate_confidence: bool = True,
        provide_soft_sequence_to_msa_and_profile: bool = True,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        max_inference_sigma: int | None = None,
    ) -> dict[str, Tensor]:
        del noise_scale, step_scale, max_inference_sigma
        tok_mask = token_attention_mask
        atm_mask = atom_attention_mask
        n_loops = num_loops if num_loops is not None else self.config.num_loops
        n_samples = (
            num_diffusion_samples
            if num_diffusion_samples is not None
            else self.config.num_diffusion_samples
        )

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

        if res_type_soft is not None:
            res_type_oh = res_type_soft.float()
            if (
                not self.config.disable_msa_features
                and provide_soft_sequence_to_msa_and_profile
            ):
                profile = res_type_oh
                msa = res_type_oh.unsqueeze(1)
                msa_attention_mask = tok_mask.unsqueeze(1)

        if deletion_mean is None:
            deletion_mean = torch.zeros(
                res_type.shape[0], res_type.shape[1], device=res_type.device
            )
        if self.config.disable_msa_features:
            profile = torch.zeros_like(profile)
            deletion_mean = torch.zeros_like(deletion_mean)

        ref_element_oh = F.one_hot(
            ref_element.long(), num_classes=MAX_ATOMIC_NUMBER
        ).float()
        ref_atom_name_chars_oh = F.one_hot(
            ref_atom_name_chars.long(), num_classes=CHAR_VOCAB_SIZE
        ).float()
        atm_mask_f = atm_mask.float()
        ref_element_oh = ref_element_oh * atm_mask_f.unsqueeze(-1)
        ref_atom_name_chars_oh = ref_atom_name_chars_oh * atm_mask_f.unsqueeze(
            -1
        ).unsqueeze(-1)
        atom_to_token = atom_to_token * atm_mask.long()

        use_amp = ref_pos.device.type == "cuda"
        with (
            torch.set_grad_enabled(res_type_soft is not None),
            torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16),
        ):
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

            z_init = self.z_init_1(x_inputs).unsqueeze(2) + self.z_init_2(
                x_inputs
            ).unsqueeze(1)
            relative_position_encoding = self.rel_pos(
                residue_index=residue_index,
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                token_index=token_index,
            )
            token_bonds_encoding = self.token_bonds(token_bonds.float())
            z_init = z_init + relative_position_encoding + token_bonds_encoding

            if (
                lm_hidden_states is None
                and input_ids is not None
                and self._esmc is not None
            ):
                lm_hidden_states = self._compute_lm_hidden_states(
                    input_ids, asym_id, residue_index, mol_type, tok_mask
                )
            if lm_hidden_states is not None:
                lm_dropout = (
                    self.config.lm_dropout
                    if self.config.force_lm_dropout_during_inference or self.training
                    else 0.0
                )
                lm_z = self.language_model(
                    lm_hidden_states.detach(), lm_dropout=lm_dropout
                )
                z_init = z_init + lm_z.to(z_init.dtype)

            msa_kwargs: dict[str, Tensor] | None = None
            if self.msa_encoder is not None and msa is not None:
                if msa.dim() == 4:
                    batch_msa, depth, length_msa, _ = msa.shape
                    msa_oh = msa.permute(0, 2, 1, 3).float()
                else:
                    batch_msa, depth, length_msa = msa.shape
                    msa_oh = F.one_hot(
                        msa.permute(0, 2, 1).long(), num_classes=NUM_RES_TYPES
                    ).float()
                msa_attn = (
                    msa_attention_mask.permute(0, 2, 1).float()
                    if msa_attention_mask is not None
                    else tok_mask[:, :, None].expand(-1, -1, depth).float()
                )
                msa_oh = msa_oh * msa_attn.unsqueeze(-1)
                hd = (
                    has_deletion.permute(0, 2, 1).float()
                    if has_deletion is not None
                    else torch.zeros(batch_msa, length_msa, depth, device=msa.device)
                )
                dv = (
                    deletion_value.permute(0, 2, 1).float()
                    if deletion_value is not None
                    else torch.zeros(batch_msa, length_msa, depth, device=msa.device)
                )
                msa_kwargs = {
                    "x_inputs": x_inputs,
                    "msa_oh": msa_oh,
                    "has_deletion": hd,
                    "deletion_value": dv,
                    "msa_attention_mask": msa_attn,
                }

            pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()
            z = torch.zeros_like(z_init)
            prev_pair: Tensor | None = None
            prev_disto_probs: Tensor | None = None
            for loop_num in range(n_loops + 1):
                z = z_init + self.pair_loop_proj(z)
                if msa_kwargs is not None and self.msa_encoder is not None:
                    z = z + self.msa_encoder(x_pair=z, **msa_kwargs).to(z.dtype)
                z = self.folding_trunk(z, pair_attention_mask=pair_mask)

                if early_exit and loop_num < n_loops:
                    l2_converged = False
                    if prev_pair is not None and loop_num > 0:
                        rel_l2 = (z.float() - prev_pair.float()).norm() / prev_pair.float().norm().clamp(
                            min=1e-8
                        )
                        l2_converged = rel_l2.item() < 0.25
                    prev_pair = z.detach().clone()
                    sym_z = z.float() + z.float().transpose(-2, -3)
                    cur_probs = F.softmax(self.distogram_head(sym_z).float(), dim=-1)
                    if prev_disto_probs is not None and loop_num > 0:
                        kl_per_pair = (
                            cur_probs
                            * (
                                cur_probs.clamp(min=1e-8)
                                / prev_disto_probs.clamp(min=1e-8)
                            ).log()
                        ).sum(-1)
                        kl = (kl_per_pair + kl_per_pair.transpose(-1, -2)).mean() / 2
                        if l2_converged or kl.item() < 0.05:
                            break
                    prev_disto_probs = cur_probs.detach()

            distogram_logits = self.distogram_head(z + z.transpose(-2, -3))

        with torch.no_grad(), _seed_context(seed):
            structure_output = self.structure_head.sample(
                z_trunk=z.float(),
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
                return_atom_repr=False,
                denoising_early_exit_rmsd=(0.10 if early_exit else None),
            )
        sample_coords = structure_output["sample_atom_coords"]
        assert sample_coords is not None

        output: dict[str, Tensor] = {
            "distogram_logits": distogram_logits,
            "sample_atom_coords": sample_coords,
        }
        if calculate_confidence and self.confidence_head is not None:
            confidence_output = self.confidence_head(
                s_inputs=x_inputs.detach(),
                z=z.detach().float(),
                x_pred=sample_coords.detach(),
                distogram_atom_idx=distogram_atom_idx,
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
        output["atom_pad_mask"] = (
            atm_mask.unsqueeze(0) if atm_mask.dim() == 1 else atm_mask
        )
        output["residue_index"] = residue_index
        output["entity_id"] = entity_id
        return output

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
        return self.input_builder.prepare_input(input, seed=seed, device=self.device)

    @torch.no_grad()
    def infer_protein(self, seq: str, **forward_kwargs) -> dict[str, Tensor]:
        from .protein_utils import prepare_protein_features

        features = prepare_protein_features(seq)
        features = {name: tensor.to(self.device) for name, tensor in features.items()}
        output = self(**features, **forward_kwargs)
        for name in (
            "res_type",
            "atom_to_token",
            "ref_atom_name_chars",
            "atom_attention_mask",
            "token_attention_mask",
            "residue_index",
        ):
            output[name] = features[name]
        return output

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

    def fold_protein(
        self,
        sequence: str,
        *,
        chain_id: str = "A",
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        num_diffusion_samples: int = 1,
        seed: int | None = None,
        complex_id: str = "pred",
    ):
        from .esmfold2_types import ProteinInput, StructurePredictionInput

        input = StructurePredictionInput(
            sequences=[ProteinInput(id=chain_id, sequence=sequence)]
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
    def result_to_cif(result) -> str:
        assert not isinstance(result, list), "Pass one MolecularComplexResult at a time."
        return result.complex.to_mmcif()

    @staticmethod
    def result_to_pdb(result) -> str:
        assert not isinstance(result, list), "Pass one MolecularComplexResult at a time."
        return result.complex.to_protein_complex().to_pdb_string()

    def save_as_cif(self, result, output_path: str | Path) -> None:
        Path(output_path).write_text(self.result_to_cif(result))

    def save_as_pdb(self, result, output_path: str | Path) -> None:
        Path(output_path).write_text(self.result_to_pdb(result))

    def infer_protein_as_cif(self, seq: str, **forward_kwargs) -> str:
        return self.result_to_cif(self.fold_protein(seq, **forward_kwargs))

    def infer_protein_as_pdb(self, seq: str, **forward_kwargs) -> str:
        return self.result_to_pdb(self.fold_protein(seq, **forward_kwargs))


__all__ = [
    "ConfidenceHead",
    "MSAEncoder",
    "MSAEncoderBlock",
    "ESMFold2ExperimentalModel",
]
