"""Local FastPLMs binder-design research example.

This is a FastPLMs-only variant of the Biohub ESMFold2 binder design workflow.
It uses FastPLMs ESMFold2 experimental checkpoints for folding and FastPLMs
ESM++ checkpoints for the masked-LM regularizer.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import platform
import random
import re
import secrets
import sys
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForMaskedLM

from fastplms.models.esm_plusplus.modeling_esm_plusplus import EsmSequenceTokenizer
from fastplms.models.esmfold2 import seed_context
from fastplms.models.esmfold2.esmfold2_constants import (
    ELEMENT_NUMBER_TO_SYMBOL,
    PROTEIN_1TO3,
    PROTEIN_3TO1,
    RES_TYPE_TO_CCD,
)

logger = logging.getLogger(__name__)


TOKENS = ["<pad>", "-"] + [RES_TYPE_TO_CCD[i] for i in range(2, 33)]
ELEMENTS = ["X"] * (max(ELEMENT_NUMBER_TO_SYMBOL) + 1)
ELEMENTS[0] = "<pad>"
for _atomic_num, _symbol in ELEMENT_NUMBER_TO_SYMBOL.items():
    ELEMENTS[_atomic_num] = _symbol[:1] + _symbol[1:].lower()
TOKEN_IDS = {token: idx for idx, token in enumerate(TOKENS)}
AA_DIMS = 20
CYS_IDX = TOKEN_IDS[PROTEIN_1TO3["C"]] - 2
MUTABLE_TOKEN = "#"
BinderPromptStr = str

LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2}
DEFAULT_STEPS = 150
DEFAULT_LOG_INTERVAL = 5
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_TEMPERATURE_MIN = 1e-2
DEFAULT_ESMC_MASK_FRACTION = 0.15
DEFAULT_SELECTION_TOP_K = 84
MINIBINDER_PI_CUTOFF = 6.0
DEFAULT_CONSENSUS_IPTM_THRESHOLD = 0.9


@dataclass(frozen=True)
class PromptFactory:
    name: str
    template: str
    length_ranges: dict[str, tuple[int, int]]
    is_antibody: bool

    def sample(self, seed: int) -> BinderPromptStr:
        rng = random.Random(seed)
        sampled_lengths = {
            key: MUTABLE_TOKEN * rng.randint(low, high)
            for key, (low, high) in self.length_ranges.items()
        }
        return self.template.format(**sampled_lengths)


BINDER_PROMPT_FACTORIES = {
    "minibinder": PromptFactory(
        name="minibinder",
        template="{seq}",
        length_ranges={"seq": (60, 200)},
        is_antibody=False,
    ),
    "trastuzumab_framework_vhvl": PromptFactory(
        name="trastuzumab_framework_vhvl",
        template=(
            "EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}YIHWVRQAPGKGLEWVARI{hcdr2}"
            "TRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR{hcdr3}WGQGTLVTVSS"
            "GGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY"
            "{lcdr2}GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK"
        ),
        length_ranges={
            "hcdr1": (7, 9),
            "hcdr2": (5, 6),
            "hcdr3": (9, 15),
            "lcdr1": (11, 16),
            "lcdr2": (7, 7),
            "lcdr3": (9, 9),
        },
        is_antibody=True,
    ),
    "atezolizumab_framework_vhvl": PromptFactory(
        name="atezolizumab_framework_vhvl",
        template=(
            "EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}WIHWVRQAPGKGLEWVAWI{hcdr2}"
            "TYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCAR{hcdr3}WGQGTLVTVSS"
            "GGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY"
            "{lcdr2}GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK"
        ),
        length_ranges={
            "hcdr1": (7, 9),
            "hcdr2": (5, 6),
            "hcdr3": (9, 15),
            "lcdr1": (11, 16),
            "lcdr2": (7, 7),
            "lcdr3": (9, 9),
        },
        is_antibody=True,
    ),
    "ocankitug_framework_vhvl": PromptFactory(
        name="ocankitug_framework_vhvl",
        template=(
            "QVQLVQSGAEVKKPGSSVKVSCKAS{hcdr1}WMHWVRQAPGQGLEWMGII{hcdr2}"
            "TSLNQKFQGRVTITADTSTSTAYMELSSLRSEDTAVYYCAR{hcdr3}WGQGTLVTVSS"
            "GGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY"
            "{lcdr2}GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK"
        ),
        length_ranges={
            "hcdr1": (7, 9),
            "hcdr2": (5, 6),
            "hcdr3": (8, 14),
            "lcdr1": (11, 16),
            "lcdr2": (7, 7),
            "lcdr3": (9, 9),
        },
        is_antibody=True,
    ),
}

TARGET_SEQUENCES = {
    "cd45": (
        "GSPGEPQIIFCRSEAAHQGVITWNPPQRSFHNFTLCYIKETEKDCLNLDKNLIKYDLQNLKPYT"
        "KYVLSLHAYIIAKVQRNGSAAMCHFTTKSAPPSQVWNMTVSMTSDNSMHVKCRPPRDRNGPHE"
        "RYHLEVEAGNTLVRNESHKNCDFRVKDLQYSTDYTFKAYFHNGDYPGEPFILHHSTSY"
    ),
    "ctla4": (
        "MHVAQPAVVLASSRGIASFVCEYASPGKATEVRVTVLRQADSQVTEVCAATYMMGNELTFLDDSI"
        "CTGTSSGNQVNLTIQGLRAMDTGLYICKVELMYPPPYYLGIGNGTQIYVIDPE"
    ),
    "egfr": (
        "RKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTV"
        "KEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDV"
        "IISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCV"
    ),
    "pd-l1": (
        "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYR"
        "QRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"
    ),
    "pdgfr": (
        "GFLPNDAEELFIFLTEITEITIPCRVTDPQLVVTLHEKKGDVALPVPYDHQRGFSGIFEDRSYIC"
        "KTTIGDREVDSDAYYVYRLQVSSINVSVNAVQTVVRQGENITLMCIVIGNEVVNFEWTYPRKES"
        "GRLVEPVTDFLLDMPYHIRSILHIPSAELEDSGTYTCNVTESVNDHQDEKAINITVVE"
    ),
}


def _repo_name(name: str) -> str:
    if "/" in name:
        return name
    return f"Synthyra/{name}"


_IMMUTABLE_HUB_REVISION = re.compile(r"[0-9a-fA-F]{40}\Z")


@cache
def _registered_fast_revisions() -> dict[str, str]:
    from fastplms.registry import get_model_registry

    return {model.fast.repo_id: model.fast.revision for model in get_model_registry().values()}


def _normalize_model_revisions(
    revisions: Mapping[str, str] | None,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_name, raw_revision in (revisions or {}).items():
        repo_id = _repo_name(str(raw_name).strip())
        revision = str(raw_revision).strip().lower()
        namespace, separator, repository = repo_id.partition("/")
        if (
            not separator
            or not namespace
            or not repository
            or "/" in repository
            or _IMMUTABLE_HUB_REVISION.fullmatch(revision) is None
        ):
            raise ValueError(
                "Model revisions must map a repository to an immutable 40-character "
                f"Git commit; got {raw_name!r}={raw_revision!r}."
            )
        previous = normalized.get(repo_id)
        if previous is not None and previous != revision:
            raise ValueError(
                f"Conflicting revisions were supplied for {repo_id!r}: "
                f"{previous!r} and {revision!r}."
            )
        normalized[repo_id] = revision
    return normalized


def _parse_model_revision_args(values: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or ():
        repo_id, separator, revision = value.partition("=")
        if not separator or not repo_id.strip() or not revision.strip():
            raise ValueError(
                f"--model-revision must use REPO=40_CHARACTER_COMMIT syntax; got {value!r}."
            )
        normalized = _normalize_model_revisions({repo_id: revision})
        normalized_repo, normalized_revision = next(iter(normalized.items()))
        previous = parsed.get(normalized_repo)
        if previous is not None and previous != normalized_revision:
            raise ValueError(
                f"Conflicting revisions were supplied for {normalized_repo!r}: "
                f"{previous!r} and {normalized_revision!r}."
            )
        parsed[normalized_repo] = normalized_revision
    return parsed


def _resolve_model_source(
    model_name: str,
    revisions: Mapping[str, str],
) -> tuple[str, str]:
    repo_id = _repo_name(model_name)
    revision = revisions.get(repo_id) or _registered_fast_revisions().get(repo_id)
    if revision is None:
        raise ValueError(
            f"Custom model repository {repo_id!r} requires an immutable revision. "
            f"Pass --model-revision {repo_id}=<40-character-commit>."
        )
    if _IMMUTABLE_HUB_REVISION.fullmatch(revision) is None:
        raise ValueError(
            f"Resolved revision for {repo_id!r} is not an immutable Git commit: {revision!r}."
        )
    return repo_id, revision.lower()


def _configure_offline_mode(local_files_only: bool) -> None:
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _record_model_load_identity(
    model: Any,
    *,
    repo_id: str,
    revision: str,
    local_files_only: bool,
) -> None:
    model.__dict__["_fastplms_binder_load_identity"] = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "local_files_only": local_files_only,
    }


def build_initial_soft_sequence_logits(sequence: str, batch_size: int) -> torch.Tensor:
    if all(aa == MUTABLE_TOKEN for aa in sequence):
        logits = 0.01 * torch.randn([batch_size, len(sequence), AA_DIMS])
        logits[:, :, CYS_IDX] = -1e6
    else:
        logits = torch.zeros([batch_size, len(sequence), AA_DIMS])
        for i, aa in enumerate(sequence):
            if aa == MUTABLE_TOKEN:
                logits[:, i, :] = 0.01 * torch.randn(batch_size, AA_DIMS)
                logits[:, i, CYS_IDX] = -1e6
            else:
                if aa not in PROTEIN_1TO3:
                    raise ValueError(
                        f"Unsupported fixed binder residue {aa!r} at position {i}; "
                        "use an uppercase canonical amino acid or '#'."
                    )
                token_id = TOKEN_IDS[PROTEIN_1TO3[aa]]
                logits[:, i, token_id - 2] = 10.0
    return logits.requires_grad_(True)


def build_gradient_mask(sequence: str, batch_size: int) -> torch.Tensor:
    mask = torch.ones([batch_size, len(sequence), AA_DIMS])
    fixed_positions = [i for i, aa in enumerate(sequence) if aa != MUTABLE_TOKEN]
    mask[:, fixed_positions, :] = 0.0
    mask[:, :, CYS_IDX] = 0.0
    return mask


def sequence_to_one_hot(sequence: str, device: torch.device | str = "cuda") -> torch.Tensor:
    target_index = [TOKEN_IDS[PROTEIN_1TO3[letter]] for letter in sequence]
    one_hot = F.one_hot(torch.tensor(target_index), num_classes=len(TOKENS))
    return one_hot.to(device).unsqueeze(0).float()


def get_mid_points() -> torch.Tensor:
    boundaries = torch.linspace(2, 52.0, 127)
    lower = torch.tensor([1.0])
    upper = torch.tensor([57.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    return (exp_boundaries[:-1] + exp_boundaries[1:]) / 2


def binned_entropy(dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float) -> torch.Tensor:
    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


def compute_contact_loss(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: torch.Tensor,
    binder_mask: torch.Tensor,
) -> torch.Tensor:
    con_loss = binned_entropy(distogram_logits, bin_distance, cutoff)
    position = torch.arange(distogram_logits.shape[1])
    p_dist = position[:, None] - position[None, :]
    if min_sep > 0:
        separation_mask = (torch.abs(p_dist) >= min_sep).to(distogram_logits.device)
        binder_mask = torch.logical_and(separation_mask, binder_mask)
    per_residue = masked_min_k(con_loss, mask=binder_mask, k=num_contacts).to(
        distogram_logits.device
    )
    return masked_average(per_residue, mask=chain_mask).to(distogram_logits.device)


def compute_intra_contact_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=2,
        min_sep=9,
        cutoff=14.0,
        chain_mask=is_binder,
        binder_mask=is_binder,
    )


def compute_inter_contact_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=1,
        min_sep=0,
        cutoff=22.0,
        chain_mask=1 - is_binder,
        binder_mask=is_binder,
    )


def compute_globularity_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    binder_disto = distogram_logits[:, -binder_length:, -binder_length:, :]
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    bin_distance = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(bin_distance), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n**0.365)
    return F.elu(rg_term - rg_th)


def compute_structure_losses(
    distogram_logits: torch.Tensor, binder_length: int
) -> dict[str, torch.Tensor]:
    bin_distance = get_mid_points().to(distogram_logits.device)
    losses: dict[str, torch.Tensor] = {}
    losses["intra_contact_loss"] = compute_intra_contact_loss(
        distogram_logits, binder_length, bin_distance
    )
    losses["inter_contact_loss"] = compute_inter_contact_loss(
        distogram_logits, binder_length, bin_distance
    )
    losses["glob_loss"] = compute_globularity_loss(distogram_logits, binder_length, bin_distance)
    batch = distogram_logits.size(0)
    total = torch.tensor([0.0] * batch, device=distogram_logits.device)
    total = total + LOSS_WEIGHTS["intra_contact"] * losses["intra_contact_loss"]
    total = total + LOSS_WEIGHTS["inter_contact"] * losses["inter_contact_loss"]
    total = total + LOSS_WEIGHTS["glob"] * losses["glob_loss"]
    losses["total_loss"] = total
    return losses


def _binding_confidence_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    probs = torch.softmax(dgram, dim=-1)
    cutoff_mask = bin_distance < cutoff
    p_cut = probs[..., cutoff_mask]
    p_cut = p_cut / (p_cut.sum(-1, keepdim=True) + 1e-8)
    return -(p_cut * torch.log(p_cut + 1e-10)).sum(-1)


def _entropy_to_confidence(mean_entropy: float) -> float:
    return float(max(0.0, min(1.0, 1.0 - mean_entropy / math.log(51))))


def _cdr_indices(binder_sequence: str) -> list[int]:
    from abnumber import Chain

    chains = list(
        Chain.multiple_domains(
            binder_sequence,
            scheme="chothia",
            allowed_species=None,
            use_anarcii=True,
        )
    )
    if not chains:
        raise ValueError("AbNumber did not identify an antibody domain in the binder sequence.")

    indices: list[int] = []
    search_start = 0
    for chain in chains:
        domain_sequence = str(chain.seq)
        domain_start = binder_sequence.find(domain_sequence, search_start)
        if domain_start < 0:
            raise RuntimeError(
                "AbNumber returned an antibody domain that cannot be aligned back to "
                "the supplied binder sequence."
            )
        indices.extend(
            domain_start + offset
            for offset, (position, _residue) in enumerate(chain)
            if position.is_in_cdr()
        )
        search_start = domain_start + len(domain_sequence)
    if not indices:
        raise ValueError("AbNumber identified antibody domains but no Chothia CDR residues.")
    return indices


def compute_distogram_iptm_proxy(
    distogram_logits: torch.Tensor,
    target_length: int,
    binder_sequence: str,
    is_antibody: bool,
    cdr_indices: list[int] | None = None,
) -> dict[str, float]:
    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]
    binder_length = len(binder_sequence)
    expected_length = target_length + binder_length
    if distogram_logits.ndim != 3 or distogram_logits.shape[:2] != (
        expected_length,
        expected_length,
    ):
        raise ValueError(
            "Distogram logits must have shape "
            f"({expected_length}, {expected_length}, bins); got "
            f"{tuple(distogram_logits.shape)}."
        )

    bin_distance = get_mid_points().to(distogram_logits.device)
    binder_start = target_length

    def _mean_lowest_k(entropies: torch.Tensor, k: int) -> float:
        sorted_entropies, _ = torch.sort(entropies.reshape(-1))
        k = min(k, sorted_entropies.numel())
        return float(sorted_entropies[:k].mean())

    binder_to_target_entropy = _binding_confidence_entropy(
        distogram_logits[binder_start:, :target_length, :], bin_distance, cutoff=22.0
    )
    distogram_iptm_proxy = _entropy_to_confidence(
        _mean_lowest_k(binder_to_target_entropy, k=binder_length)
    )

    if not is_antibody:
        cdr_distogram_iptm_proxy = float("nan")
    else:
        if cdr_indices is None:
            cdr_indices = _cdr_indices(binder_sequence)
        cdr_rows = [binder_start + i for i in cdr_indices]
        cdr_to_target_entropy = _binding_confidence_entropy(
            distogram_logits[cdr_rows, :target_length, :], bin_distance, cutoff=22.0
        )
        cdr_distogram_iptm_proxy = _entropy_to_confidence(
            _mean_lowest_k(cdr_to_target_entropy, k=len(cdr_indices))
        )
    return {
        "distogram_iptm_proxy": distogram_iptm_proxy,
        "cdr_distogram_iptm_proxy": cdr_distogram_iptm_proxy,
    }


_ATOM_FEATURE_DIMS = {
    "ref_pos": 1,
    "ref_element": 1,
    "ref_charge": 1,
    "ref_atom_name_chars": 1,
    "ref_space_uid": 1,
    "atom_attention_mask": 1,
    "atom_to_token": 1,
    "is_resolved": 1,
    "gt_coords": 2,
}


def _resize_tensor(tensor: torch.Tensor, *, dim: int, size: int) -> torch.Tensor:
    current = tensor.shape[dim]
    if current > size:
        raise ValueError(
            f"Refusing to truncate atom features from {current} to {size}; "
            "batch padding must preserve every prepared atom."
        )
    if current == size:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = size - current
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=dim)


def _prepared_atom_count(features: dict[str, torch.Tensor]) -> int:
    sizes = {features[key].shape[dim] for key, dim in _ATOM_FEATURE_DIMS.items() if key in features}
    if not sizes:
        raise ValueError("Prepared ESMFold2 features contain no atom-axis tensors.")
    if len(sizes) != 1:
        raise ValueError(f"Prepared ESMFold2 atom axes disagree: {sorted(sizes)}")
    return sizes.pop()


def _pad_prepared_atom_features(
    prepared: list[tuple[dict[str, torch.Tensor], list[Any]]],
) -> list[tuple[dict[str, torch.Tensor], list[Any]]]:
    """Pad a prepared batch to its largest atom table without truncation."""

    if not prepared:
        raise ValueError("At least one prepared ESMFold2 input is required.")
    largest = max(_prepared_atom_count(features) for features, _ in prepared)
    max_atoms = ((largest + 31) // 32) * 32
    padded: list[tuple[dict[str, torch.Tensor], list[Any]]] = []
    for features, chain_infos in prepared:
        resized = dict(features)
        for key, dim in _ATOM_FEATURE_DIMS.items():
            if key in resized:
                resized[key] = _resize_tensor(resized[key], dim=dim, size=max_atoms)
        padded.append((resized, chain_infos))
    return padded


def prepare_esmfold2_tensors(
    model: Any,
    input_data: Any,
    max_atoms: int | None = None,
    seed: int | None = None,
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    features, chain_infos = model.prepare_structure_input(input_data, seed=seed)
    if max_atoms is not None:
        for key, dim in _ATOM_FEATURE_DIMS.items():
            if key in features:
                features[key] = _resize_tensor(features[key], dim=dim, size=max_atoms)
    return features, chain_infos


def _filter_model_forward_kwargs(
    model: Any, kwargs: dict[str, torch.Tensor | int | bool | None]
) -> dict[str, torch.Tensor | int | bool | None]:
    signature = inspect.signature(model.forward)
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if accepts_kwargs:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def fold_and_get_distogram(
    model: Any,
    target_seq: str,
    target_one_hot: torch.Tensor,
    design: torch.Tensor,
    num_loops: int = 0,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    padding = (2, 11)
    padded_design = F.pad(design, padding, mode="constant", value=0)

    token_lists = torch.argmax(padded_design, dim=-1)
    designed_seq = [
        [PROTEIN_3TO1[TOKENS[int(tkn.item())]] for tkn in token_list] for token_list in token_lists
    ]
    seq_list = [target_seq + "|" + "".join(seq) for seq in designed_seq]
    prepared_inputs: list[tuple[dict[str, torch.Tensor], list[Any]]] = []
    for seq in seq_list:
        target, binder = seq.split("|")
        input_types = model.input_types
        inputs_raw = input_types.StructurePredictionInput(
            sequences=[
                input_types.ProteinInput(id="A", sequence=target, msa=None),
                input_types.ProteinInput(id="B", sequence=binder, msa=None),
            ]
        )
        prepared_inputs.append(prepare_esmfold2_tensors(model, inputs_raw, seed=seed))

    prepared_inputs = _pad_prepared_atom_features(prepared_inputs)
    inputs_list = [features for features, _ in prepared_inputs]
    chain_info_list = [chain_infos for _, chain_infos in prepared_inputs]

    inputs = {
        key: torch.cat([inp[key] for inp in inputs_list], dim=0).to(design.device)
        for key in inputs_list[0]
    }
    inputs["res_type_soft"] = torch.cat(
        (target_one_hot.repeat(design.size(0), 1, 1), padded_design), dim=1
    )

    forward_kwargs: dict[str, torch.Tensor | int | bool | None] = dict(inputs)
    forward_kwargs.update(
        {
            "num_diffusion_samples": 1,
            "num_sampling_steps": num_sampling_steps,
            "num_loops": num_loops,
            "calculate_confidence": calculate_confidence,
            "seed": seed,
        }
    )

    with seed_context(seed):
        output = model(**_filter_model_forward_kwargs(model, forward_kwargs))

    result: dict[str, Any] = {
        "distogram_logits": output["distogram_logits"],
        "inputs": inputs,
        "chain_info_list": chain_info_list,
        "output": output,
        "seq_list": seq_list,
    }
    if calculate_confidence:
        for key in ("ptm", "iptm", "plddt"):
            if key in output:
                result[key] = output[key]
    return result


@cache
def _folding_trunk_to_lm_aa_vocab_matrix(device: torch.device) -> torch.Tensor:
    three_to_one_map = {v: k for k, v in PROTEIN_1TO3.items()}
    ft_aas = [three_to_one_map[tok_3letter] for tok_3letter in TOKENS[2:22]]
    tokenizer = EsmSequenceTokenizer()
    lm_vocab = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
    lm_aas = [lm_vocab[i][0] for i in range(4, 24)]
    ft_to_lm_aa_matrix = torch.zeros(20, 20)
    for ft_idx, ft_aa in enumerate(ft_aas):
        lm_idx = lm_aas.index(ft_aa)
        ft_to_lm_aa_matrix[ft_idx, lm_idx] = 1
    return ft_to_lm_aa_matrix.to(device=device)


def _one_hot_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return F.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.size(-1)).to(probs.dtype)


def _straight_through(discrete: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
    return continuous + (discrete - continuous).detach()


def compute_fastplms_pseudoperplexity_nll(
    lm_model: Any,
    binder_design: torch.Tensor,
    score_mask: torch.Tensor,
    batch_size: int = 4,
    n_passes: int = 4,
    mask_fraction: float = DEFAULT_ESMC_MASK_FRACTION,
) -> torch.Tensor:
    device = binder_design.device
    lm_vocab_size = lm_model.config.vocab_size
    model_dtype = lm_model.embed.weight.dtype

    target_esm = binder_design @ _folding_trunk_to_lm_aa_vocab_matrix(device)
    input_esm = _straight_through(_one_hot_from_probs(target_esm), target_esm)
    input_ids = torch.zeros(
        (binder_design.size(0), binder_design.size(1) + 2, lm_vocab_size),
        dtype=model_dtype,
        device=device,
    )
    tokenizer = lm_model.tokenizer
    input_ids[:, 0, tokenizer.cls_token_id] = 1
    input_ids[:, -1, tokenizer.eos_token_id] = 1
    input_ids[:, 1:-1, 4:24] = input_esm.to(model_dtype)

    if score_mask.ndim == 1:
        score_mask = score_mask.unsqueeze(0).expand(binder_design.size(0), -1)
    elif score_mask.shape != binder_design.shape[:2]:
        raise ValueError(
            f"Expected score_mask with shape {(binder_design.size(0), binder_design.size(1))}, "
            f"got {tuple(score_mask.shape)}"
        )
    score_mask = score_mask.to(device=device, dtype=torch.bool)

    mask_token = torch.zeros(lm_vocab_size, dtype=model_dtype, device=device)
    mask_token[tokenizer.mask_token_id] = 1
    losses = []
    for batch_idx in range(binder_design.size(0)):
        position_indices = score_mask[batch_idx].nonzero(as_tuple=False).flatten()
        num_positions = int(position_indices.numel())
        if num_positions == 0:
            raise ValueError("Pseudoperplexity score mask selected zero positions.")

        num_masked = max(1, math.ceil(mask_fraction * num_positions))
        random_scores = torch.rand((n_passes, num_positions), device=device)
        masked_offsets = random_scores.topk(num_masked, dim=-1, largest=False).indices
        pass_masks = torch.zeros((n_passes, binder_design.size(1)), dtype=torch.bool, device=device)
        pass_masks[
            torch.arange(n_passes, device=device)[:, None],
            position_indices[masked_offsets],
        ] = True

        masked_sequences = input_ids[batch_idx : batch_idx + 1].repeat(n_passes, 1, 1)
        mask_rows, mask_cols = pass_masks.nonzero(as_tuple=True)
        masked_sequences[mask_rows, mask_cols + 1] = mask_token

        target_weights = target_esm[batch_idx]
        masked_nlls = []
        for start in range(0, n_passes, batch_size):
            stop = min(start + batch_size, n_passes)
            chunk = masked_sequences[start:stop]
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                hidden = lm_model.transformer(
                    x=chunk @ lm_model.embed.weight.to(chunk.dtype),
                    attention_mask=None,
                    output_hidden_states=False,
                    output_attentions=False,
                ).last_hidden_state
                logits = lm_model.sequence_head(hidden)
            log_probs = logits.log_softmax(dim=-1)[:, 1:-1, 4:24]
            nlls = -(log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)).sum(dim=-1)
            masked_nlls.append(nlls[pass_masks[start:stop]])
        losses.append(torch.cat(masked_nlls, dim=0).mean())
    return torch.stack(losses, dim=0)


def normalized_gradient_tensor(grad: torch.Tensor, gradient_mask: torch.Tensor) -> torch.Tensor:
    masked_grad = grad * gradient_mask
    index_has_nonzero_grad = torch.square(masked_grad).sum(-1) > 0
    eff_l = index_has_nonzero_grad.sum(-1)
    grad_norm = torch.linalg.norm(masked_grad, axis=(-1, -2))
    normalized_grad = (masked_grad / (grad_norm[:, None, None] + 1e-7)) * torch.sqrt(
        eff_l[:, None, None]
    )
    return normalized_grad * gradient_mask


def _tensor_mean_float(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().mean().cpu().item())


def _metric_float(output: dict[str, Any], key: str) -> float | None:
    if key not in output:
        return None
    value = output[key]
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def _require_fresh_output_directory(output_dir: str | Path | None) -> Path | None:
    if output_dir is None:
        return None
    result_dir = Path(output_dir)
    if result_dir.exists() or result_dir.is_symlink():
        raise FileExistsError(
            f"Binder output directory {result_dir} already exists. Choose a new path; "
            "existing, partial, and empty run directories are never reused."
        )
    return result_dir


def _reserve_output_directory(output_dir: str | Path | None) -> Path | None:
    result_dir = _require_fresh_output_directory(output_dir)
    if result_dir is None:
        return None
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        result_dir.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"Binder output directory {result_dir} was created by another run. Choose a new path."
        ) from error
    return result_dir


def _validate_design_sequence(name: str, sequence: str, *, allow_mutable: bool) -> None:
    if not sequence:
        raise ValueError(f"{name} must not be empty.")
    allowed = set(PROTEIN_1TO3)
    if allow_mutable:
        allowed.add(MUTABLE_TOKEN)
    invalid = [(index, residue) for index, residue in enumerate(sequence) if residue not in allowed]
    if invalid:
        index, residue = invalid[0]
        mutable_note = " or '#'" if allow_mutable else ""
        raise ValueError(
            f"{name} contains unsupported residue {residue!r} at position {index}; "
            f"use uppercase canonical amino acids{mutable_note}."
        )


def design_binder(
    inversion_models: dict[str, Any],
    critic_models: dict[str, Any],
    lm_model: Any,
    target_name: str | None,
    target_sequence: str | None,
    binder_name: str | None,
    binder_sequence: str | None,
    is_antibody: bool | None,
    seed: int,
    batch_size: int = 1,
    steps: int = DEFAULT_STEPS,
    log_interval: int = DEFAULT_LOG_INTERVAL,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    temperature_min: float = DEFAULT_TEMPERATURE_MIN,
    output_dir: str | Path | None = None,
    device: torch.device | str = "cuda",
) -> tuple[list[str], dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    if (target_name is None) == (target_sequence is None):
        raise ValueError("Provide exactly one of target_name or target_sequence.")
    if (binder_name is None) == (binder_sequence is None):
        raise ValueError("Provide exactly one of binder_name or binder_sequence.")
    if not inversion_models:
        raise ValueError("At least one inversion model is required.")
    if not critic_models:
        raise ValueError("At least one critic model is required.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}.")
    if steps <= 0:
        raise ValueError(f"steps must be positive; got {steps}.")
    if log_interval <= 0:
        raise ValueError(f"log_interval must be positive; got {log_interval}.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(f"learning_rate must be finite and positive; got {learning_rate}.")
    if not math.isfinite(temperature_min) or not 0 < temperature_min <= 1:
        raise ValueError(
            f"temperature_min must be finite and in the interval (0, 1]; got {temperature_min}."
        )

    device = torch.device(device)
    if target_name is not None:
        if target_name not in TARGET_SEQUENCES:
            raise ValueError(
                f"Unknown target_name {target_name!r}; choose one of "
                f"{sorted(TARGET_SEQUENCES)} or pass target_sequence."
            )
        target_sequence = TARGET_SEQUENCES[target_name]
    if target_sequence is None:
        raise RuntimeError("Target sequence resolution failed.")

    if binder_name is None:
        if binder_sequence is None:
            raise RuntimeError("Binder sequence resolution failed.")
        if is_antibody is None:
            is_antibody = False
    else:
        if binder_name not in BINDER_PROMPT_FACTORIES:
            raise ValueError(
                f"Unknown binder_name {binder_name!r}; choose one of "
                f"{sorted(BINDER_PROMPT_FACTORIES)} or pass binder_sequence."
            )
        binder_prompt_factory = BINDER_PROMPT_FACTORIES[binder_name]
        if is_antibody is not None and binder_prompt_factory.is_antibody != is_antibody:
            raise ValueError(
                f"Binder prompt {binder_name!r} has is_antibody="
                f"{binder_prompt_factory.is_antibody}, not {is_antibody}."
            )
        is_antibody = binder_prompt_factory.is_antibody
        binder_sequence = binder_prompt_factory.sample(seed=seed)
    if binder_sequence is None or is_antibody is None:
        raise RuntimeError("Binder prompt resolution failed.")
    _validate_design_sequence("target_sequence", target_sequence, allow_mutable=False)
    _validate_design_sequence("binder_sequence", binder_sequence, allow_mutable=True)
    mutable_binder_indices = [i for i, aa in enumerate(binder_sequence) if aa == MUTABLE_TOKEN]
    binder_length = len(binder_sequence)
    result_dir = _reserve_output_directory(output_dir)
    target_one_hot = sequence_to_one_hot(target_sequence, device=device)

    with seed_context(seed), torch.device(device):
        logits = build_initial_soft_sequence_logits(binder_sequence, batch_size=batch_size)
        gradient_mask = build_gradient_mask(binder_sequence, batch_size=batch_size)
    logits = logits.to(device)
    gradient_mask = gradient_mask.to(device)

    trajectory: dict[int, dict[str, torch.Tensor]] = {}
    optimizer = optim.SGD([logits], lr=learning_rate)
    best_iptm: list[float] = [-1.0] * batch_size
    best_loss: list[float] = [float("inf")] * batch_size
    best_sequences: list[str] = [""] * batch_size
    best_logits: list[torch.Tensor | None] = [None] * batch_size
    best_steps: list[int | None] = [None] * batch_size
    model_names = list(inversion_models)

    progress = tqdm(range(steps), desc="design", dynamic_ncols=True)
    for step in progress:
        optimizer.zero_grad()
        t = (step + 1) / steps
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = temperature_min + (1 - temperature_min) * remaining

        replicate_choice = random.Random(seed + step).randint(0, len(model_names) - 1)
        inversion_model = inversion_models[model_names[replicate_choice]]
        design = F.softmax(logits / temperature, dim=-1)
        calculate_confidence = temperature < 0.05

        fold_result = fold_and_get_distogram(
            inversion_model,
            target_sequence,
            target_one_hot,
            design,
            num_loops=1,
            num_sampling_steps=50 if calculate_confidence else 1,
            calculate_confidence=calculate_confidence,
            seed=seed + step,
        )
        sequences: list[str] = fold_result["seq_list"]
        losses = compute_structure_losses(fold_result["distogram_logits"], binder_length)
        structure_loss = losses["total_loss"]
        structure_grad = torch.autograd.grad(structure_loss.mean(), logits)[0]

        design = F.softmax(logits / temperature, dim=-1)
        score_mask = gradient_mask.sum(dim=-1) > 0
        with seed_context(seed + step):
            plm_loss = compute_fastplms_pseudoperplexity_nll(
                lm_model=lm_model,
                binder_design=design,
                score_mask=score_mask,
                batch_size=4,
                n_passes=4,
            )
        plm_grad = torch.autograd.grad(plm_loss.mean(), logits)[0]
        candidate_logits = logits.detach().clone()

        logits.grad = normalized_gradient_tensor(structure_grad, gradient_mask) + (
            0.05 if is_antibody else 0.15
        ) * normalized_gradient_tensor(plm_grad, gradient_mask)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * temperature
        optimizer.step()

        step_losses = {key: value.detach().cpu() for key, value in losses.items()}
        step_losses["plm_loss"] = plm_loss.detach().cpu()
        step_losses["total_loss"] = (structure_loss + plm_loss).detach().cpu()
        trajectory[step] = step_losses

        iptm = fold_result.get("iptm")
        for batch_idx in range(batch_size):
            current_loss = float(step_losses["total_loss"][batch_idx].item())
            if iptm is not None and iptm[batch_idx] is not None:
                current_iptm = float(iptm[batch_idx].item())
                if current_iptm > best_iptm[batch_idx]:
                    best_iptm[batch_idx] = current_iptm
                    best_sequences[batch_idx] = sequences[batch_idx]
                    best_loss[batch_idx] = current_loss
                    best_logits[batch_idx] = candidate_logits[batch_idx].cpu()
                    best_steps[batch_idx] = step
            elif current_loss < best_loss[batch_idx]:
                best_sequences[batch_idx] = sequences[batch_idx]
                best_loss[batch_idx] = current_loss
                best_logits[batch_idx] = candidate_logits[batch_idx].cpu()
                best_steps[batch_idx] = step

        if step % log_interval == 0:
            loss_str = "  ".join(
                f"{key}={_tensor_mean_float(value):.4f}" for key, value in step_losses.items()
            )
            logger.info("step %3d | %s T=%.4f", step, loss_str, temperature)
        progress.set_postfix(
            loss=f"{_tensor_mean_float(step_losses['total_loss']):.3f}",
            temp=f"{temperature:.3f}",
        )

    if any(not sequence for sequence in best_sequences):
        raise RuntimeError("Optimization completed without selecting every binder sequence.")
    if any(value is None for value in best_logits):
        raise RuntimeError("Optimization completed without retaining every selected logit tensor.")
    if any(value is None for value in best_steps):
        raise RuntimeError("Optimization completed without retaining every selected step.")
    if result_dir is not None:
        _write_trajectory(result_dir / "trajectory.jsonl", trajectory)
        _write_fasta(result_dir / "best_sequences.fasta", best_sequences)

    critic_results: list[dict[str, Any]] = []
    target_length = len(target_sequence.replace("|", ""))
    for batch_idx, best_seq in enumerate(best_sequences):
        binder_seq = best_seq.split("|")[-1]
        binder_design = sequence_to_one_hot(binder_seq, device=device)[..., 2:22]
        for critic_name, critic_model in critic_models.items():
            final_fold = fold_and_get_distogram(
                critic_model,
                target_sequence,
                target_one_hot,
                binder_design,
                num_loops=3,
                num_sampling_steps=200,
                calculate_confidence=True,
                seed=seed,
            )
            final_output = final_fold["output"]
            final_inputs = final_fold["inputs"]
            chain_infos = final_fold["chain_info_list"][0]
            complex_result = critic_model.input_builder.decode(
                final_output,
                final_inputs,
                chain_infos,
                num_diffusion_samples=1,
                complex_id=f"{critic_name}-{batch_idx}",
            )
            cif_text = critic_model.result_to_cif(complex_result)
            pdb_text = critic_model.result_to_pdb(complex_result)
            iptm_proxy_scores = compute_distogram_iptm_proxy(
                final_fold["distogram_logits"],
                target_length,
                binder_seq,
                is_antibody,
                cdr_indices=mutable_binder_indices if is_antibody else None,
            )
            iptm_value = None
            if "iptm" in final_fold:
                iptm_value = float(final_fold["iptm"][0].item())
            ptm_value = _metric_float(final_fold, "ptm")
            mean_plddt = _metric_float(final_fold, "plddt")

            structure_stem = f"batch{batch_idx}_{critic_name.replace('/', '_')}"
            logits_path = None
            if result_dir is not None:
                cif_path = result_dir / f"{structure_stem}.cif"
                pdb_path = result_dir / f"{structure_stem}.pdb"
                logits_path_obj = result_dir / f"{structure_stem}_logits.pt"
                cif_path.write_text(cif_text, encoding="utf-8")
                pdb_path.write_text(pdb_text, encoding="utf-8")
                torch.save(best_logits[batch_idx], logits_path_obj)
                logits_path = str(logits_path_obj)

            row = {
                "is_antibody": is_antibody,
                "critic_name": critic_name,
                "batch_idx": batch_idx,
                "designed_sequence": best_seq,
                "binder_sequence": binder_seq,
                "target_length": target_length,
                "binder_length": len(binder_seq),
                "final_loss": best_loss[batch_idx],
                "selected_step": best_steps[batch_idx],
                "ptm": ptm_value,
                "iptm": iptm_value,
                "mean_plddt": mean_plddt,
                "pdb": pdb_text,
                "cif": cif_text,
                "logits_path": logits_path,
            }
            row.update(iptm_proxy_scores)
            critic_results.append(row)

    if result_dir is not None:
        _write_results_table(result_dir / "results.parquet", critic_results)
        _write_official_selection_table(
            result_dir / "selection.parquet",
            critic_results,
            required_hero_critics=tuple(critic_models),
        )
        _write_run_manifest(
            result_dir / "run_manifest.json",
            seed=seed,
            batch_size=batch_size,
            steps=steps,
            log_interval=log_interval,
            learning_rate=learning_rate,
            temperature_min=temperature_min,
            target_sequence=target_sequence,
            binder_sequence=binder_sequence,
            is_antibody=is_antibody,
            inversion_models=inversion_models,
            critic_models=critic_models,
            lm_model=lm_model,
            device=device,
        )
    return best_sequences, trajectory, critic_results


def _write_trajectory(path: Path, trajectory: dict[int, dict[str, torch.Tensor]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for step, losses in trajectory.items():
            row = {"step": step}
            for key, value in losses.items():
                row[key] = [float(x) for x in value.reshape(-1).tolist()]
            handle.write(json.dumps(row) + "\n")


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _configuration_digest(config: Any) -> str | None:
    if config is None or not hasattr(config, "to_dict"):
        return None
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_load_identity(model: Any) -> dict[str, Any]:
    identity = getattr(model, "__dict__", {}).get(
        "_fastplms_binder_load_identity",
        {},
    )
    return dict(identity) if isinstance(identity, dict) else {}


def _tokenizer_identity(
    tokenizer: Any,
    *,
    model: Any | None = None,
) -> dict[str, Any] | None:
    if tokenizer is None:
        return None
    config = getattr(model, "config", None)
    load_identity = _model_load_identity(model)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else getattr(tokenizer, "vocab", None)
    vocab_digest = None
    if isinstance(vocab, dict):
        payload = json.dumps(
            sorted((str(token), int(index)) for token, index in vocab.items()),
            separators=(",", ":"),
        )
        vocab_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    tokenizer_revision = (
        init_kwargs.get("revision") or getattr(tokenizer, "_commit_hash", None)
        if isinstance(init_kwargs, dict)
        else getattr(tokenizer, "_commit_hash", None)
    )
    return {
        "class": type(tokenizer).__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "repo_id": load_identity.get("repo_id") or getattr(config, "_name_or_path", None),
        "requested_revision": (
            load_identity.get("requested_revision") or getattr(config, "_commit_hash", None)
        ),
        "hub_revision": getattr(config, "_commit_hash", None),
        "weights_revision": getattr(config, "fastplms_weights_revision", None),
        "runtime_revision": getattr(config, "fastplms_runtime_revision", None),
        "revision": tokenizer_revision or load_identity.get("requested_revision"),
        "local_files_only": load_identity.get("local_files_only"),
        "vocab_size": len(vocab) if isinstance(vocab, dict) else None,
        "vocab_sha256": vocab_digest,
        "special_token_ids": {
            name: getattr(tokenizer, f"{name}_token_id", None)
            for name in ("bos", "cls", "eos", "mask", "pad", "sep", "unk")
        },
    }


def _parameter_dtype_identity(model: Any) -> tuple[str | None, list[str], dict[str, int]]:
    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        parameters = (parameter for _name, parameter in named_parameters())
    else:
        raw_parameters = getattr(model, "parameters", None)
        parameters = iter(raw_parameters()) if callable(raw_parameters) else iter(())

    dtype_numel: dict[str, int] = {}
    for parameter in parameters:
        dtype = str(parameter.dtype)
        dtype_numel[dtype] = dtype_numel.get(dtype, 0) + int(parameter.numel())
    dtypes = sorted(dtype_numel)
    if not dtypes:
        summary = None
    elif len(dtypes) == 1:
        summary = dtypes[0]
    else:
        summary = f"mixed[{','.join(dtypes)}]"
    return summary, dtypes, {dtype: dtype_numel[dtype] for dtype in dtypes}


def _effective_precision_identity(model: Any) -> dict[str, Any] | None:
    status = getattr(model, "esmc_precision_status", None)
    as_dict = getattr(status, "as_dict", None)
    if callable(as_dict):
        return dict(as_dict())
    return None


def _model_identity(name: str, model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    load_identity = _model_load_identity(model)
    parameter_dtype, parameter_dtypes, parameter_dtype_numel = _parameter_dtype_identity(model)
    attention_backend = next(
        (
            getattr(config, field)
            for field in (
                "esmc_attn_backend",
                "attn_backend",
                "attention_backend",
                "_attn_implementation",
            )
            if config is not None and getattr(config, field, None) is not None
        ),
        None,
    )
    return {
        "name": name,
        "requested": name,
        "resolved": getattr(config, "_name_or_path", None),
        "repo_id": load_identity.get("repo_id") or getattr(config, "_name_or_path", None),
        "requested_revision": (
            load_identity.get("requested_revision") or getattr(config, "_commit_hash", None)
        ),
        "hub_revision": getattr(config, "_commit_hash", None),
        "weights_revision": getattr(config, "fastplms_weights_revision", None),
        "runtime_revision": getattr(config, "fastplms_runtime_revision", None),
        "revision": (
            getattr(config, "_commit_hash", None) or load_identity.get("requested_revision")
        ),
        "local_files_only": load_identity.get("local_files_only"),
        "attention_backend": attention_backend,
        "kernel_backend": getattr(model, "_kernel_backend", None),
        "parameter_dtype": parameter_dtype,
        "parameter_dtypes": parameter_dtypes,
        "parameter_dtype_numel": parameter_dtype_numel,
        "effective_precision": _effective_precision_identity(model),
        "configuration_sha256": _configuration_digest(config),
    }


def _write_run_manifest(
    path: Path,
    *,
    seed: int,
    batch_size: int,
    steps: int,
    log_interval: int,
    learning_rate: float,
    temperature_min: float,
    target_sequence: str,
    binder_sequence: str,
    is_antibody: bool,
    inversion_models: dict[str, Any],
    critic_models: dict[str, Any],
    lm_model: Any,
    device: torch.device,
) -> None:
    def sequence_hash(sequence: str) -> str:
        return hashlib.sha256(sequence.encode("ascii")).hexdigest()

    manifest = {
        "schema_version": 2,
        "seed": seed,
        "batch_size": batch_size,
        "steps": steps,
        "learning_rate": learning_rate,
        "temperature_min": temperature_min,
        "target_sequence_sha256": sequence_hash(target_sequence),
        "binder_prompt_sha256": sequence_hash(binder_sequence),
        "device": str(device),
        "configuration": {
            "batch_size": batch_size,
            "steps": steps,
            "log_interval": log_interval,
            "optimizer": {
                "class": "torch.optim.SGD",
                "learning_rate": learning_rate,
            },
            "temperature_min": temperature_min,
            "is_antibody": is_antibody,
            "loss_weights": LOSS_WEIGHTS,
            "plm_batch_size": 4,
            "plm_passes": 4,
            "compute_dtype": ("torch.bfloat16" if device.type == "cuda" else "torch.float32"),
        },
        "command": list(sys.argv),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "fastplms": _package_version("fastplms"),
            "cuda": torch.version.cuda,
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        },
        "models": {
            "inversion": [_model_identity(name, model) for name, model in inversion_models.items()],
            "critics": [_model_identity(name, model) for name, model in critic_models.items()],
            "language_model": _model_identity(
                getattr(getattr(lm_model, "config", None), "_name_or_path", "language_model"),
                lm_model,
            ),
        },
        "tokenizer": _tokenizer_identity(
            getattr(lm_model, "tokenizer", None),
            model=lm_model,
        ),
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _write_fasta(path: Path, sequences: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx, sequence in enumerate(sequences):
            handle.write(f">design_{idx}\n{sequence}\n")


def _write_results_table(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_parquet(path, index=False)


def _binder_sequence_from_designed_sequence(designed_sequence: str) -> str:
    parts = designed_sequence.split("|")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "designed_sequence must contain nonempty target and binder chains "
            f"separated by one '|'; got {designed_sequence!r}."
        )
    return parts[1]


def _compute_isoelectric_points(sequences: list[str]) -> list[float]:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    return [float(ProteinAnalysis(sequence).isoelectric_point()) for sequence in sequences]


def annotate_official_selection_scores(result_df: Any) -> Any:
    """Add the official binder-design selection components to critic rows.

    Mirrors the paper Appendix A.3.1.2 and the official notebook selection cell:
    minibinders with pI >= 6 are filtered and the approved critics contribute
    mean iPTM.
    """
    import pandas as pd

    df = result_df.copy() if isinstance(result_df, pd.DataFrame) else pd.DataFrame(result_df)
    required_columns = [
        "critic_name",
        "designed_sequence",
        "is_antibody",
        "iptm",
        "distogram_iptm_proxy",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing selection columns: {missing_columns}")
    if df["distogram_iptm_proxy"].isna().any():
        raise ValueError("distogram_iptm_proxy must be present for every selection row")

    binder_sequences = [
        _binder_sequence_from_designed_sequence(sequence)
        for sequence in df["designed_sequence"].tolist()
    ]
    is_antibody = df["is_antibody"].astype(bool)
    df["binder_sequence"] = binder_sequences
    df["isoelectric_point"] = _compute_isoelectric_points(binder_sequences)
    df["passes_official_pi_filter"] = is_antibody | df["isoelectric_point"].lt(MINIBINDER_PI_CUTOFF)
    df["official_iptm_score_component"] = df["iptm"]
    return df


def select_official_designs(
    result_df: Any,
    top_k: int = DEFAULT_SELECTION_TOP_K,
    consensus_iptm_threshold: float = DEFAULT_CONSENSUS_IPTM_THRESHOLD,
    group_columns: tuple[str, ...] = ("target_name", "binder_name"),
    required_hero_critics: tuple[str, ...] | None = None,
) -> Any:
    """Rank candidates using the official ESM binder-design selection strategy."""
    df = annotate_official_selection_scores(result_df)
    available_group_columns = [column for column in group_columns if column in df.columns]
    selection_columns = [
        *available_group_columns,
        "designed_sequence",
        "iptm_score",
        "iptm_proxy_score",
        "hero_iptm_min",
        "hero_iptm_median",
        "hero_iptm_max",
        "critic_count",
        "hero_critic_count",
        "required_hero_critic_count",
        "batch_idx",
        "binder_sequence",
        "is_antibody",
        "isoelectric_point",
        "selection_score",
        "all_hero_critics_pass",
        "consensus_iptm_threshold",
    ]
    df = df[df["passes_official_pi_filter"]].copy()
    if df.empty:
        import pandas as pd

        return pd.DataFrame(columns=selection_columns)

    if required_hero_critics is None:
        required_hero_critics = tuple(dict.fromkeys(df["critic_name"].dropna().tolist()))
    else:
        required_hero_critics = tuple(dict.fromkeys(required_hero_critics))
    if not required_hero_critics:
        raise ValueError("At least one required hero critic must be specified")
    required_hero_critic_count = len(required_hero_critics)
    is_required_hero_critic = df["critic_name"].isin(required_hero_critics)
    df["hero_iptm_score_component"] = df["official_iptm_score_component"].where(
        is_required_hero_critic
    )
    df["scored_hero_critic_name"] = df["critic_name"].where(
        is_required_hero_critic & df["hero_iptm_score_component"].notna()
    )

    key_columns = [*available_group_columns, "designed_sequence"]
    summary_columns = [
        "batch_idx",
        "binder_sequence",
        "is_antibody",
        "isoelectric_point",
    ]
    summary_aggregations = {
        column: (column, "first") for column in summary_columns if column in df.columns
    }
    scores = df.groupby(key_columns, as_index=False).agg(
        iptm_score=("hero_iptm_score_component", "mean"),
        iptm_proxy_score=("distogram_iptm_proxy", "mean"),
        hero_iptm_min=("hero_iptm_score_component", "min"),
        hero_iptm_median=("hero_iptm_score_component", "median"),
        hero_iptm_max=("hero_iptm_score_component", "max"),
        critic_count=("critic_name", "nunique"),
        hero_critic_count=("scored_hero_critic_name", "nunique"),
        **summary_aggregations,
    )
    scores["required_hero_critic_count"] = required_hero_critic_count
    scores["selection_score"] = scores["iptm_score"].fillna(0.0)
    scores["all_hero_critics_pass"] = scores["hero_critic_count"].eq(
        required_hero_critic_count
    ) & scores["hero_iptm_min"].gt(consensus_iptm_threshold)
    scores["consensus_iptm_threshold"] = consensus_iptm_threshold

    if available_group_columns:
        sort_columns = [*available_group_columns, "selection_score"]
        ascending = [*[True] * len(available_group_columns), False]
        scores = scores.sort_values(sort_columns, ascending=ascending)
        return (
            scores.groupby(available_group_columns, group_keys=False, sort=False)
            .head(top_k)
            .reset_index(drop=True)
        )
    return scores.nlargest(min(len(scores), top_k), "selection_score").reset_index(drop=True)


def _write_official_selection_table(
    path: Path,
    rows: list[dict[str, Any]],
    required_hero_critics: tuple[str, ...] | None = None,
) -> None:
    selection_df = select_official_designs(
        rows,
        required_hero_critics=required_hero_critics,
    )
    selection_df.to_parquet(path, index=False)


def _log_official_selection_summary(rows: list[dict[str, Any]]) -> None:
    selection_df = select_official_designs(rows)
    if selection_df.empty:
        logger.info("Official selection table is empty after pI filtering")
        return
    top = selection_df.iloc[0]
    logger.info(
        "Top official selection | score=%.4f hero_mean=%.4f proxy_mean=%.4f "
        "hero_min=%.4f all_hero_pass=%s binder=%s",
        float(top["selection_score"]),
        float(top["iptm_score"]),
        float(top["iptm_proxy_score"]) if not math.isnan(top["iptm_proxy_score"]) else 0.0,
        float(top["hero_iptm_min"]),
        bool(top["all_hero_critics_pass"]),
        top["binder_sequence"],
    )


_ESMC_CACHE: Any | None = None
_ESMC_CACHE_KEY: tuple[str, str] | None = None
_ESMC_CACHE_CONTEXT: dict[str, Any] = {}


_ESMC_CONTEXT_FIELDS = (
    "_esmc_fp8",
    "_esmc_fp8_module_paths",
    "_esmc_source",
    "_esmc_source_revision",
    "_esmc_source_files",
    "_esmc_local_files_only",
    "_esmc_precision_policy",
    "_esmc_precision_status",
)


def _load_fold_model(
    model_name: str,
    revision: str,
    lm_dropout: float,
    cache_esmc: bool,
    device: torch.device | str,
    kernel_backend: str | None,
    compile_model: bool,
    local_files_only: bool,
) -> Any:
    global _ESMC_CACHE, _ESMC_CACHE_CONTEXT, _ESMC_CACHE_KEY
    repo_id = _repo_name(model_name)
    model = AutoModel.from_pretrained(
        repo_id,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=True,
        load_esmc=not cache_esmc,
        dtype=torch.float32,
    )
    _record_model_load_identity(
        model,
        repo_id=repo_id,
        revision=revision,
        local_files_only=local_files_only,
    )
    model = model.to(device=device)
    if cache_esmc:
        esmc_cache_key = (str(model.config.esmc_id), str(torch.device(device)))
        if _ESMC_CACHE is None or esmc_cache_key != _ESMC_CACHE_KEY:
            model.load_esmc(
                model.config.esmc_id,
                device=device,
                local_files_only=local_files_only,
            )
            _ESMC_CACHE = model._esmc
            _ESMC_CACHE_KEY = esmc_cache_key
            _ESMC_CACHE_CONTEXT = {field: getattr(model, field) for field in _ESMC_CONTEXT_FIELDS}
        else:
            model._esmc = _ESMC_CACHE
            for field, value in _ESMC_CACHE_CONTEXT.items():
                setattr(model, field, value.copy() if isinstance(value, dict) else value)
    model.configure_lm_dropout(lm_dropout, force_lm_dropout_during_inference=True)
    if kernel_backend is not None:
        model.set_kernel_backend(kernel_backend)
    if compile_model:
        model.apply_torch_compile()
    return model.eval().requires_grad_(False)


class FastPLMsBinderDesign:
    lm_name = "Synthyra/ESMplusplus_6B"
    inversion_model_names = ("ESMFold2-Experimental-Fast-Cutoff2025",)
    hero_critic_model_names = (
        "ESMFold2-Experimental-Fast-Cutoff2025",
        "ESMFold2-Experimental-Cutoff2025",
    )

    def load(
        self,
        device: str = "cuda",
        kernel_backend: str | None = None,
        compile_model: bool = False,
        inversion_model_names: tuple[str, ...] | None = None,
        critic_model_names: tuple[str, ...] | None = None,
        lm_name: str | None = None,
        model_revisions: Mapping[str, str] | None = None,
        local_files_only: bool = False,
    ) -> None:
        _configure_offline_mode(local_files_only)
        revisions = _normalize_model_revisions(model_revisions)
        selected_inversion_models = inversion_model_names or self.inversion_model_names
        selected_critic_models = critic_model_names or self.hero_critic_model_names
        selected_lm = lm_name or self.lm_name
        if len(set(selected_inversion_models)) != len(selected_inversion_models):
            raise ValueError("Inversion model names must be unique.")
        if len(set(selected_critic_models)) != len(selected_critic_models):
            raise ValueError("Critic model names must be unique.")
        inversion_sources = {
            model_name: _resolve_model_source(model_name, revisions)
            for model_name in selected_inversion_models
        }
        critic_sources = {
            model_name: _resolve_model_source(model_name, revisions)
            for model_name in selected_critic_models
        }
        lm_repo_id, lm_revision = _resolve_model_source(selected_lm, revisions)
        selected_repositories = {
            repo_id
            for repo_id, _revision in (
                *inversion_sources.values(),
                *critic_sources.values(),
                (lm_repo_id, lm_revision),
            )
        }
        unused_revisions = sorted(set(revisions) - selected_repositories)
        if unused_revisions:
            raise ValueError(
                "Model revisions were supplied for repositories that are not loaded: "
                f"{unused_revisions}."
            )

        self.device = torch.device(device)
        self.inversion_models = {
            model_name: _load_fold_model(
                model_name,
                revision=inversion_sources[model_name][1],
                lm_dropout=0.5,
                cache_esmc=True,
                device=device,
                kernel_backend=kernel_backend,
                compile_model=compile_model,
                local_files_only=local_files_only,
            )
            for model_name in selected_inversion_models
        }
        self.critic_models = {
            model_name: _load_fold_model(
                model_name,
                revision=critic_sources[model_name][1],
                lm_dropout=0.25,
                cache_esmc=True,
                device=device,
                kernel_backend=kernel_backend,
                compile_model=compile_model,
                local_files_only=local_files_only,
            )
            for model_name in selected_critic_models
        }
        self.lm_model = (
            AutoModelForMaskedLM.from_pretrained(
                lm_repo_id,
                revision=lm_revision,
                local_files_only=local_files_only,
                trust_remote_code=True,
                dtype=torch.float32,
            )
            .to(device=device)
            .eval()
            .requires_grad_(False)
        )
        _record_model_load_identity(
            self.lm_model,
            repo_id=lm_repo_id,
            revision=lm_revision,
            local_files_only=local_files_only,
        )
        self.inversion_model_names = tuple(selected_inversion_models)
        self.hero_critic_model_names = tuple(selected_critic_models)
        self.lm_name = selected_lm

    def design(
        self,
        target_name: str | None = None,
        target_sequence: str | None = None,
        binder_name: str | None = None,
        binder_sequence: str | None = None,
        is_antibody: bool | None = None,
        seed: int = 0,
        batch_size: int = 1,
        steps: int = DEFAULT_STEPS,
        output_dir: str | None = None,
    ) -> tuple[list[str], dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
        return design_binder(
            self.inversion_models,
            self.critic_models,
            self.lm_model,
            target_name=target_name,
            target_sequence=target_sequence,
            binder_name=binder_name,
            binder_sequence=binder_sequence,
            is_antibody=is_antibody,
            seed=seed,
            batch_size=batch_size,
            steps=steps,
            output_dir=output_dir,
            device=self.device,
        )


def _design_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target_name": args.target_name,
        "target_sequence": args.target_sequence,
        "binder_name": args.binder_name,
        "binder_sequence": args.binder_sequence,
        "is_antibody": args.is_antibody,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "output_dir": args.output_dir,
    }


def run_local(args: argparse.Namespace) -> None:
    _require_fresh_output_directory(args.output_dir)
    runner = FastPLMsBinderDesign()
    runner.load(
        kernel_backend=args.kernel_backend,
        compile_model=args.compile_model,
        inversion_model_names=(
            tuple(args.inversion_model_names) if args.inversion_model_names is not None else None
        ),
        critic_model_names=(
            tuple(args.critic_model_names) if args.critic_model_names is not None else None
        ),
        lm_name=args.lm_model,
        model_revisions=args.model_revisions,
        local_files_only=args.local_files_only,
    )
    best_sequences, _, results = runner.design(**_design_kwargs_from_args(args))
    logger.info("Designed sequences: %s", best_sequences)
    logger.info("Returned %d critic rows", len(results))
    _log_official_selection_summary(results)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-name", default="pd-l1")
    parser.add_argument("--target-sequence", default=None)
    parser.add_argument("--binder-name", default="minibinder")
    parser.add_argument("--binder-sequence", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output-dir", default="binder_design_out")
    parser.add_argument(
        "--inversion-model",
        dest="inversion_model_names",
        action="append",
        default=None,
        help="FastPLMs inversion checkpoint; repeat to use multiple checkpoints.",
    )
    parser.add_argument(
        "--critic-model",
        dest="critic_model_names",
        action="append",
        default=None,
        help="FastPLMs critic checkpoint; repeat to use multiple checkpoints.",
    )
    parser.add_argument(
        "--lm-model",
        default=None,
        help="FastPLMs masked-language-model checkpoint.",
    )
    parser.add_argument(
        "--model-revision",
        dest="model_revisions",
        action="append",
        default=[],
        metavar="REPO=COMMIT",
        help=(
            "Immutable 40-character Hub commit for a custom model repository; "
            "repeat once per custom repository. Registered defaults use models.toml."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help=(
            "Use cached snapshots only and set HF_HUB_OFFLINE=1 plus "
            "TRANSFORMERS_OFFLINE=1 before model loading."
        ),
    )
    parser.add_argument("--kernel-backend", default=None)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--is-antibody", dest="is_antibody", action="store_true")
    parser.add_argument("--not-antibody", dest="is_antibody", action="store_false")
    parser.set_defaults(is_antibody=None)
    args = parser.parse_args(argv)
    try:
        args.model_revisions = _parse_model_revision_args(args.model_revisions)
    except ValueError as error:
        parser.error(str(error))
    if args.target_sequence is not None:
        args.target_name = None
    if args.binder_sequence is not None:
        args.binder_name = None
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the local binder-design workflow from explicit CLI arguments."""

    run_local(parse_args(argv))
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    raise SystemExit(main())
