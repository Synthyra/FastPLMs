"""Input preparation and output decoding for ESMFold2 inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .esmfold2_conformers import load_ccd
from .esmfold2_molecular_complex import MolecularComplexResult
from .esmfold2_output import build_molecular_complex_from_features
from .esmfold2_prepare_input import ChainInfo, prepare_esmfold2_input
from .esmfold2_types import MSA, Modification, ProteinInput, StructurePredictionInput
from .modeling_esmfold2_common import MSA_CONDITIONING_INPUT_NAMES
from .reproducibility import seed_context

# Backward-compatible private alias for the pinned parity helpers. New callers
# should import ``seed_context`` from the public ``fastplms.models.esmfold2``
# package instead of reaching into implementation modules.
_seed_context = seed_context


@dataclass(frozen=True)
class _SplitProteinState:
    ids: dict[str, list[str]]
    modifications: dict[str, list[Modification]]
    msas: dict[str, MSA | None]


@dataclass(frozen=True)
class _PendingProtein:
    source: ProteinInput
    sequence: str
    state: _SplitProteinState


def _chain_starts(chains: list[str]) -> list[int]:
    starts: list[int] = []
    position = 0
    for chain in chains:
        starts.append(position)
        position += len(chain) + 1
    return starts


def _split_modifications(
    item: ProteinInput,
    chains: list[str],
    starts: list[int],
) -> dict[str, list[Modification]]:
    grouped: dict[str, list[Modification]] = {}
    if item.modifications is None:
        return grouped
    for chain, start in zip(chains, starts, strict=True):
        end = start + len(chain)
        adjusted = [
            Modification(position=modification.position - start, ccd=modification.ccd)
            for modification in item.modifications
            if start <= modification.position < end
        ]
        grouped.setdefault(chain, []).extend(adjusted)
    return grouped


def _split_msas(
    item: ProteinInput,
    chains: list[str],
    starts: list[int],
) -> dict[str, MSA | None]:
    grouped: dict[str, MSA | None] = {}
    if item.msa is None:
        return grouped
    for chain, start in zip(chains, starts, strict=True):
        if chain not in grouped:
            grouped[chain] = item.msa.select_positions(np.arange(start, start + len(chain)))
    return grouped


def _split_protein(item: ProteinInput) -> tuple[list[_PendingProtein], _SplitProteinState]:
    chains = ":".join(item.sequence.split("|")).split(":")
    starts = _chain_starts(chains)
    base_id = item.id[0] if isinstance(item.id, list) else item.id
    ids: dict[str, list[str]] = {}
    for index, chain in enumerate(chains):
        chain_ids = ids.setdefault(chain, [])
        chain_ids.append(f"{base_id}_{index}")
    state = _SplitProteinState(
        ids=ids,
        modifications=_split_modifications(item, chains, starts),
        msas=_split_msas(item, chains, starts),
    )
    pending = [
        _PendingProtein(item, chain, state)
        for chain, chain_ids in ids.items()
        if chain_ids
    ]
    return pending, state


def _resolve_pending(pending: _PendingProtein) -> ProteinInput:
    item = pending.source
    sequence = pending.sequence
    state = pending.state
    return ProteinInput(
        id=state.ids[sequence],
        sequence=sequence,
        msa=state.msas.get(sequence) if item.msa else None,
        modifications=(state.modifications.get(sequence) if item.modifications else None),
    )


def clean_esmfold2_input(input: StructurePredictionInput) -> StructurePredictionInput:
    """Expand chain delimiters and group repeated protein sequences by entity."""

    if input.pocket is not None:
        raise NotImplementedError(
            "ESMFold2 pocket conditioning is present in the upstream input schema but "
            "the published ESMFold2 feature pipeline drops it. FastPLMs refuses this "
            "input instead of silently emitting an all-zero pocket feature."
        )
    if input.distogram_conditioning is not None:
        raise NotImplementedError(
            "ESMFold2 distogram conditioning is present in the upstream input schema but "
            "the published ESMFold2 forward does not consume it. FastPLMs refuses this "
            "input instead of silently ignoring the supplied distogram."
        )

    cleaned: list[Any] = []
    for item in input.sequences:
        if not isinstance(item, ProteinInput):
            cleaned.append(item)
            continue
        sequence = ":".join(item.sequence.split("|"))
        if ":" not in sequence:
            cleaned.append(item)
            continue
        if input.covalent_bonds is not None:
            raise ValueError(
                "Covalent bonds are not supported when using chainbreaks. "
                "Chains must be separated into multiple ProteinInput objects."
            )
        pending, _state = _split_protein(item)
        cleaned.extend(pending)

    resolved = [
        _resolve_pending(item) if isinstance(item, _PendingProtein) else item
        for item in cleaned
    ]
    return StructurePredictionInput(
        sequences=resolved,
        pocket=input.pocket,
        distogram_conditioning=input.distogram_conditioning,
        covalent_bonds=input.covalent_bonds,
    )


def _batch_features(
    features: dict[str, Any],
    device: torch.device | str | None,
) -> dict[str, Any]:
    return {
        name: (value[None].to(device) if device is not None else value[None])
        if isinstance(value, Tensor)
        else value
        for name, value in features.items()
    }


def _sampler_overrides(
    noise_scale: float | None,
    step_scale: float | None,
    max_inference_sigma: int | None,
) -> dict[str, Any]:
    values = {
        "noise_scale": noise_scale,
        "step_scale": step_scale,
        "max_inference_sigma": max_inference_sigma,
    }
    return {name: value for name, value in values.items() if value is not None}


class ESMFold2InputBuilder:
    """Prepare public input objects, run folding, and decode model tensors."""

    def __init__(self, ccd_cache: Path | None = None) -> None:
        load_ccd(ccd_cache)

    def prepare_input(
        self,
        input: StructurePredictionInput,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[dict[str, Any], list[ChainInfo]]:
        cleaned = clean_esmfold2_input(input)
        with seed_context(seed):
            features, chain_infos = prepare_esmfold2_input(cleaned, seed=seed)
            return _batch_features(features, device), chain_infos

    def prepare_model_input(
        self,
        model: Any,
        input: StructurePredictionInput,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[dict[str, Any], list[ChainInfo]]:
        """Prepare features while enforcing the checkpoint's MSA contract."""

        msa_conditioning = getattr(model.config, "msa_conditioning", None)
        if not isinstance(msa_conditioning, bool):
            raise RuntimeError("The ESMFold2 config has no Boolean msa_conditioning contract.")
        if not msa_conditioning:
            explicit_msa_ids = [
                item.id
                for item in input.sequences
                if isinstance(item, ProteinInput) and item.msa is not None
            ]
            if explicit_msa_ids:
                raise ValueError(
                    "This ESMFold2 checkpoint was trained without MSA conditioning and "
                    f"rejects explicit MSAs for protein inputs {explicit_msa_ids!r}."
                )
        features, chain_infos = self.prepare_input(input, seed=seed, device=device)
        if not msa_conditioning:
            for name in MSA_CONDITIONING_INPUT_NAMES:
                features.pop(name, None)
        return features, chain_infos

    def __call__(
        self,
        input: StructurePredictionInput,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[dict[str, Any], list[ChainInfo]]:
        return self.prepare_input(input, seed=seed, device=device)

    def _decode_sample(
        self,
        output: Mapping[str, Tensor],
        features: dict[str, Tensor],
        chain_infos: list[ChainInfo],
        sample: int,
        complex_id: str,
    ) -> MolecularComplexResult:
        plddt = output["plddt"][sample]
        molecular_complex = build_molecular_complex_from_features(
            coords=output["sample_atom_coords"][sample],
            plddt=plddt,
            atom_mask=features["atom_attention_mask"][0],
            ref_element=features["ref_element"][0],
            ref_atom_name_chars=features["ref_atom_name_chars"][0],
            chain_infos=chain_infos,
            complex_id=complex_id,
        )

        def sample_tensor(name: str) -> Tensor | None:
            value = output.get(name)
            return None if value is None else value[sample].detach().cpu()

        def shared_tensor(name: str) -> Tensor | None:
            value = output.get(name)
            return None if value is None else value[0].detach().cpu()

        ptm = output.get("ptm")
        iptm = output.get("iptm")
        return MolecularComplexResult(
            complex=molecular_complex,
            plddt=plddt.detach().cpu(),
            ptm=float(ptm[sample].item()) if ptm is not None else None,
            iptm=float(iptm[sample].item()) if iptm is not None else None,
            pae=sample_tensor("pae"),
            distogram=shared_tensor("distogram_logits"),
            pair_chains_iptm=sample_tensor("pair_chains_iptm"),
            residue_index=shared_tensor("residue_index"),
            entity_id=shared_tensor("entity_id"),
        )

    def decode(
        self,
        output: Mapping[str, Tensor],
        features: dict[str, Tensor],
        chain_infos: list[ChainInfo],
        *,
        num_diffusion_samples: int = 1,
        complex_id: str = "pred",
    ) -> MolecularComplexResult | list[MolecularComplexResult]:
        results = [
            self._decode_sample(output, features, chain_infos, sample, complex_id)
            for sample in range(output["sample_atom_coords"].shape[0])
        ]
        return results[0] if num_diffusion_samples == 1 and len(results) == 1 else results

    def fold(
        self,
        model: Any,
        input: StructurePredictionInput,
        *,
        num_loops: int = 3,
        num_sampling_steps: int = 200,
        num_diffusion_samples: int = 1,
        seed: int | None = None,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        max_inference_sigma: int | None = None,
        early_exit: bool = False,
        complex_id: str = "pred",
    ) -> MolecularComplexResult | list[MolecularComplexResult]:
        features, chain_infos = self.prepare_model_input(
            model,
            input,
            seed=seed,
            device=model.device,
        )
        overrides = _sampler_overrides(noise_scale, step_scale, max_inference_sigma)
        with torch.no_grad(), seed_context(seed):
            output = model(
                **features,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=num_diffusion_samples,
                early_exit=early_exit,
                return_dict=True,
                **overrides,
            )
        return self.decode(
            output,
            features,
            chain_infos,
            num_diffusion_samples=num_diffusion_samples,
            complex_id=complex_id,
        )


__all__ = ["ESMFold2InputBuilder", "clean_esmfold2_input", "seed_context"]
