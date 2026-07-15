"""Input preparation and output decoding for ESMFold2 inference."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
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


@dataclass(frozen=True)
class _RandomState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Tensor
    torch_cuda: list[Tensor] | None


def _capture_random_state() -> _RandomState:
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return _RandomState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=cuda_state,
    )


def _restore_random_state(state: _RandomState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(state.torch_cuda)


@contextmanager
def _seed_context(seed: int | None) -> Iterator[None]:
    """Seed Python, NumPy, and Torch temporarily, then restore every stream."""

    if seed is None:
        yield
        return
    state = _capture_random_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        _restore_random_state(state)


@dataclass(frozen=True)
class _PendingProtein:
    source: ProteinInput
    sequence: str


@dataclass(frozen=True)
class _SplitProteinState:
    ids: dict[str, list[str]]
    modifications: dict[str, list[Modification]]
    msas: dict[str, MSA | None]


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
    pending: list[_PendingProtein] = []
    for index, chain in enumerate(chains):
        chain_ids = ids.setdefault(chain, [])
        chain_ids.append(f"{base_id}_{index}")
        if len(chain_ids) == 1:
            pending.append(_PendingProtein(item, chain))
    state = _SplitProteinState(
        ids=ids,
        modifications=_split_modifications(item, chains, starts),
        msas=_split_msas(item, chains, starts),
    )
    return pending, state


def _resolve_pending(pending: _PendingProtein, state: _SplitProteinState) -> ProteinInput:
    item = pending.source
    sequence = pending.sequence
    return ProteinInput(
        id=state.ids[sequence],
        sequence=sequence,
        msa=state.msas.get(sequence) if item.msa else None,
        modifications=(state.modifications.get(sequence) if item.modifications else None),
    )


def clean_esmfold2_input(input: StructurePredictionInput) -> StructurePredictionInput:
    """Expand chain delimiters and group repeated protein sequences by entity."""

    cleaned: list[Any] = []
    latest_state = _SplitProteinState({}, {}, {})
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
        pending, latest_state = _split_protein(item)
        cleaned.extend(pending)

    resolved = [
        _resolve_pending(item, latest_state) if isinstance(item, _PendingProtein) else item
        for item in cleaned
    ]
    return StructurePredictionInput(
        sequences=resolved,
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
        with _seed_context(seed):
            features, chain_infos = prepare_esmfold2_input(cleaned, seed=seed)
            return _batch_features(features, device), chain_infos

    def __call__(
        self,
        input: StructurePredictionInput,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[dict[str, Any], list[ChainInfo]]:
        return self.prepare_input(input, seed=seed, device=device)

    def _decode_sample(
        self,
        output: dict[str, Tensor],
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
        output: dict[str, Tensor],
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
        features, chain_infos = self.prepare_input(input, seed=seed, device=model.device)
        overrides = _sampler_overrides(noise_scale, step_scale, max_inference_sigma)
        with torch.no_grad(), _seed_context(seed):
            output = model(
                **features,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=num_diffusion_samples,
                early_exit=early_exit,
                **overrides,
            )
        return self.decode(
            output,
            features,
            chain_infos,
            num_diffusion_samples=num_diffusion_samples,
            complex_id=complex_id,
        )


__all__ = ["ESMFold2InputBuilder", "clean_esmfold2_input"]
