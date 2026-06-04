# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "abnumber",
#     "biopython",
#     "modal",
#     "pandas",
#     "pyarrow",
#     "tqdm",
# ]
# ///
"""FastPLMs binder design with local and Modal execution.

This is a FastPLMs-only variant of the Biohub ESMFold2 binder design tutorial.
It uses FastPLMs ESMFold2 experimental checkpoints for folding and FastPLMs
ESM++ checkpoints for the masked-LM regularizer.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForMaskedLM

from fastplms.esm_plusplus.modeling_esm_plusplus import EsmSequenceTokenizer
from fastplms.esmfold2.esmfold2_constants import (
    ELEMENT_NUMBER_TO_SYMBOL,
    PROTEIN_1TO3,
    PROTEIN_3TO1,
    RES_TYPE_TO_CCD,
)
from fastplms.esmfold2.modeling_esmfold2_common import _seed_context as seed_context

try:
    import modal
except ImportError:
    modal = None

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
SCALING_CHECKPOINT_SUBSTRING = "ESMFold2-Experimental-Fast-base"


def _is_scaling_critic_name(critic_name: str) -> bool:
    return SCALING_CHECKPOINT_SUBSTRING in critic_name


@dataclass(frozen=True)
class PromptFactory:
    name: str
    template: str
    length_ranges: dict[str, tuple[int, int]]
    is_antibody: bool

    def sample(self, seed: int) -> BinderPromptStr:
        random.seed(seed)
        sampled_lengths = {
            key: MUTABLE_TOKEN * random.randint(low, high)
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
                assert aa in PROTEIN_1TO3, aa
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


def binned_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
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
    losses["glob_loss"] = compute_globularity_loss(
        distogram_logits, binder_length, bin_distance
    )
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
    from abnumber.common import _anarci_align

    result = _anarci_align(
        sequences=[binder_sequence], scheme="chothia", allowed_species=None
    )[0]
    chains = [
        Chain("".join(result[i][0].values()), scheme="chothia")
        for i in range(len(result))
    ]
    if len(chains) == 2 and not chains[0].is_heavy_chain():
        chains.reverse()
    indices: list[int] = []
    for chain in chains:
        for cdr in (chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq):
            start = binder_sequence.find(cdr)
            assert start >= 0
            indices.extend(range(start, start + len(cdr)))
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
    assert distogram_logits.shape[0] == target_length + binder_length

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
    if current >= size:
        return tensor.narrow(dim, 0, size)
    pad_shape = list(tensor.shape)
    pad_shape[dim] = size - current
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=dim)


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
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
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
        [PROTEIN_3TO1[TOKENS[int(tkn.item())]] for tkn in token_list]
        for token_list in token_lists
    ]
    seq_list = [target_seq + "|" + "".join(seq) for seq in designed_seq]
    max_atoms = None if len(seq_list) == 1 else ((len(seq_list[0]) - 1) * 14) // 32 * 32

    inputs_list = []
    chain_info_list = []
    for seq in seq_list:
        target, binder = seq.split("|")
        input_types = model.input_types
        inputs_raw = input_types.StructurePredictionInput(
            sequences=[
                input_types.ProteinInput(id="A", sequence=target, msa=None),
                input_types.ProteinInput(id="B", sequence=binder, msa=None),
            ]
        )
        features, chain_infos = prepare_esmfold2_tensors(
            model, inputs_raw, max_atoms=max_atoms, seed=seed
        )
        inputs_list.append(features)
        chain_info_list.append(chain_infos)

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
    return F.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.size(-1)).to(
        probs.dtype
    )


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
        pass_masks = torch.zeros(
            (n_passes, binder_design.size(1)), dtype=torch.bool, device=device
        )
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
            nlls = -(log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)).sum(
                dim=-1
            )
            masked_nlls.append(nlls[pass_masks[start:stop]])
        losses.append(torch.cat(masked_nlls, dim=0).mean())
    return torch.stack(losses, dim=0)


def normalized_gradient_tensor(
    grad: torch.Tensor, gradient_mask: torch.Tensor
) -> torch.Tensor:
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
) -> tuple[list[str], dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    assert (target_name is None) ^ (
        target_sequence is None
    ), "Provide either target name or target sequence."
    assert (binder_name is None) ^ (
        binder_sequence is None
    ), "Provide either binder name or binder sequence."

    device = torch.device("cuda")
    if target_name is not None:
        assert target_name in TARGET_SEQUENCES, target_name
        target_sequence = TARGET_SEQUENCES[target_name]
    else:
        assert target_sequence is not None
    target_one_hot = sequence_to_one_hot(target_sequence, device=device)

    if binder_name is None:
        assert binder_sequence is not None
        if is_antibody is None:
            is_antibody = False
    else:
        assert binder_name in BINDER_PROMPT_FACTORIES, binder_name
        binder_prompt_factory = BINDER_PROMPT_FACTORIES[binder_name]
        if is_antibody is not None:
            assert binder_prompt_factory.is_antibody == is_antibody
        is_antibody = binder_prompt_factory.is_antibody
        binder_sequence = binder_prompt_factory.sample(seed=seed)
    assert binder_sequence is not None
    assert is_antibody is not None
    mutable_binder_indices = [
        i for i, aa in enumerate(binder_sequence) if aa == MUTABLE_TOKEN
    ]
    binder_length = len(binder_sequence)
    assert "|" not in target_sequence
    assert "|" not in binder_sequence

    with seed_context(seed), torch.device(device):
        logits = build_initial_soft_sequence_logits(
            binder_sequence, batch_size=batch_size
        )
        gradient_mask = build_gradient_mask(binder_sequence, batch_size=batch_size)
    logits = logits.to(device)
    gradient_mask = gradient_mask.to(device)

    trajectory: dict[int, dict[str, torch.Tensor]] = {}
    optimizer = optim.SGD([logits], lr=learning_rate)
    best_iptm: list[float] = [-1.0] * batch_size
    best_loss: list[float] = [float("inf")] * batch_size
    best_sequences: list[str] = [""] * batch_size
    model_names = list(inversion_models)

    progress = tqdm(range(steps), desc="design", dynamic_ncols=True)
    for step in progress:
        optimizer.zero_grad()
        t = (step + 1) / steps
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = temperature_min + (1 - temperature_min) * remaining

        random.seed(seed + step)
        replicate_choice = random.randint(0, len(model_names) - 1)
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
        losses = compute_structure_losses(
            fold_result["distogram_logits"], binder_length
        )
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

        iptm = fold_result["iptm"] if "iptm" in fold_result else None
        for batch_idx in range(batch_size):
            current_loss = float(step_losses["total_loss"][batch_idx].item())
            if iptm is not None and iptm[batch_idx] is not None:
                current_iptm = float(iptm[batch_idx].item())
                if current_iptm > best_iptm[batch_idx]:
                    best_iptm[batch_idx] = current_iptm
                    best_sequences[batch_idx] = sequences[batch_idx]
                    best_loss[batch_idx] = current_loss
            elif current_loss < best_loss[batch_idx]:
                best_sequences[batch_idx] = sequences[batch_idx]
                best_loss[batch_idx] = current_loss

        if step % log_interval == 0:
            loss_str = "  ".join(
                f"{key}={_tensor_mean_float(value):.4f}"
                for key, value in step_losses.items()
            )
            logger.info("step %3d | %s T=%.4f", step, loss_str, temperature)
        progress.set_postfix(
            loss=f"{_tensor_mean_float(step_losses['total_loss']):.3f}",
            temp=f"{temperature:.3f}",
        )

    assert all(seq != "" for seq in best_sequences)
    result_dir = Path(output_dir) if output_dir is not None else None
    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)
        _write_trajectory(result_dir / "trajectory.jsonl", trajectory)
        _write_fasta(result_dir / "best_sequences.fasta", best_sequences)

    critic_results: list[dict[str, Any]] = []
    target_length = len(target_sequence.replace("|", ""))
    for batch_idx, best_seq in enumerate(best_sequences):
        binder_seq = best_seq.split("|")[-1]
        binder_design = sequence_to_one_hot(binder_seq, device=device)[..., 2:22]
        for critic_name, critic_model in critic_models.items():
            is_scaling_critic = _is_scaling_critic_name(critic_name)
            if is_scaling_critic:
                critic_model.to(device=device)
            try:
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
            finally:
                if is_scaling_critic:
                    critic_model.to(device="cpu")
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
                torch.save(logits[batch_idx].detach().cpu(), logits_path_obj)
                logits_path = str(logits_path_obj)

            row = {
                "is_antibody": is_antibody,
                "critic_name": critic_name,
                "batch_idx": batch_idx,
                "designed_sequence": best_seq,
                "binder_sequence": binder_seq,
                "target_length": target_length,
                "binder_length": len(binder_seq),
                "final_loss": float(trajectory[steps - 1]["total_loss"][batch_idx].item()),
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
        _write_official_selection_table(result_dir / "selection.parquet", critic_results)
    return best_sequences, trajectory, critic_results


def _write_trajectory(
    path: Path, trajectory: dict[int, dict[str, torch.Tensor]]
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for step, losses in trajectory.items():
            row = {"step": step}
            for key, value in losses.items():
                row[key] = [float(x) for x in value.reshape(-1).tolist()]
            handle.write(json.dumps(row) + "\n")


def _write_fasta(path: Path, sequences: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx, sequence in enumerate(sequences):
            handle.write(f">design_{idx}\n{sequence}\n")


def _write_results_table(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_parquet(path, index=False)


def _binder_sequence_from_designed_sequence(designed_sequence: str) -> str:
    parts = designed_sequence.split("|")
    assert len(parts) == 2, designed_sequence
    return parts[1]


def _compute_isoelectric_points(sequences: list[str]) -> list[float]:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    return [float(ProteinAnalysis(sequence).isoelectric_point()) for sequence in sequences]


def annotate_official_selection_scores(result_df: Any) -> Any:
    """Add the official binder-design selection components to critic rows.

    Mirrors the paper Appendix A.3.1.2 and the official notebook selection cell:
    minibinders with pI >= 6 are filtered, hero critics contribute mean iPTM,
    and optional scaling critics contribute the distogram ipTM proxy.
    """
    import pandas as pd

    if isinstance(result_df, pd.DataFrame):
        df = result_df.copy()
    else:
        df = pd.DataFrame(result_df)
    required_columns = [
        "critic_name",
        "designed_sequence",
        "is_antibody",
        "iptm",
        "distogram_iptm_proxy",
        "cdr_distogram_iptm_proxy",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    assert not missing_columns, f"Missing selection columns: {missing_columns}"

    binder_sequences = [
        _binder_sequence_from_designed_sequence(sequence)
        for sequence in df["designed_sequence"].tolist()
    ]
    is_antibody = df["is_antibody"].astype(bool)
    is_scaling = df["critic_name"].str.contains(
        SCALING_CHECKPOINT_SUBSTRING, regex=False, na=False
    )
    iptm_proxy = df["distogram_iptm_proxy"].where(
        ~is_antibody, df["cdr_distogram_iptm_proxy"]
    )

    df["binder_sequence"] = binder_sequences
    df["isoelectric_point"] = _compute_isoelectric_points(binder_sequences)
    df["passes_official_pi_filter"] = is_antibody | df["isoelectric_point"].lt(
        MINIBINDER_PI_CUTOFF
    )
    df["official_iptm_score_component"] = df["iptm"].where(~is_scaling)
    df["official_iptm_proxy_component"] = iptm_proxy.where(is_scaling)
    return df


def select_official_designs(
    result_df: Any,
    top_k: int = DEFAULT_SELECTION_TOP_K,
    consensus_iptm_threshold: float = DEFAULT_CONSENSUS_IPTM_THRESHOLD,
    group_columns: tuple[str, ...] = ("target_name", "binder_name"),
) -> Any:
    """Rank candidates using the official ESM binder-design selection strategy."""
    df = annotate_official_selection_scores(result_df)
    available_group_columns = [
        column for column in group_columns if column in df.columns
    ]
    selection_columns = available_group_columns + [
        "designed_sequence",
        "iptm_score",
        "iptm_proxy_score",
        "hero_iptm_min",
        "hero_iptm_median",
        "hero_iptm_max",
        "scaling_proxy_mean",
        "critic_count",
        "hero_critic_count",
        "scaling_critic_count",
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

    key_columns = available_group_columns + ["designed_sequence"]
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
        iptm_score=("official_iptm_score_component", "mean"),
        iptm_proxy_score=("official_iptm_proxy_component", "mean"),
        hero_iptm_min=("official_iptm_score_component", "min"),
        hero_iptm_median=("official_iptm_score_component", "median"),
        hero_iptm_max=("official_iptm_score_component", "max"),
        scaling_proxy_mean=("official_iptm_proxy_component", "mean"),
        critic_count=("critic_name", "count"),
        hero_critic_count=(
            "official_iptm_score_component",
            lambda values: int(values.notna().sum()),
        ),
        scaling_critic_count=(
            "official_iptm_proxy_component",
            lambda values: int(values.notna().sum()),
        ),
        **summary_aggregations,
    )
    scores["selection_score"] = 0.5 * scores["iptm_score"].fillna(
        0.0
    ) + 0.5 * scores["iptm_proxy_score"].fillna(0.0)
    scores["all_hero_critics_pass"] = scores["hero_iptm_min"].gt(
        consensus_iptm_threshold
    )
    scores["consensus_iptm_threshold"] = consensus_iptm_threshold

    if available_group_columns:
        sort_columns = available_group_columns + ["selection_score"]
        ascending = [True] * len(available_group_columns) + [False]
        scores = scores.sort_values(sort_columns, ascending=ascending)
        return (
            scores.groupby(available_group_columns, group_keys=False, sort=False)
            .head(top_k)
            .reset_index(drop=True)
        )
    return scores.nlargest(min(len(scores), top_k), "selection_score").reset_index(
        drop=True
    )


def _write_official_selection_table(path: Path, rows: list[dict[str, Any]]) -> None:
    selection_df = select_official_designs(rows)
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


def _load_fold_model(
    model_name: str,
    lm_dropout: float,
    cache_esmc: bool,
    device: torch.device | str,
    kernel_backend: str | None,
    compile_model: bool,
) -> Any:
    global _ESMC_CACHE
    model = AutoModel.from_pretrained(
        _repo_name(model_name),
        trust_remote_code=True,
        load_esmc=not cache_esmc,
        dtype=torch.float32,
    )
    if cache_esmc:
        if _ESMC_CACHE is None:
            model.load_esmc(model.config.esmc_id)
            _ESMC_CACHE = model._esmc
        else:
            model._esmc = _ESMC_CACHE
    model.configure_lm_dropout(
        lm_dropout, force_lm_dropout_during_inference=True
    )
    if kernel_backend is not None:
        model.set_kernel_backend(kernel_backend)
    if compile_model:
        model.apply_torch_compile()
    return model.to(device=device).eval().requires_grad_(False)


class FastPLMsBinderDesign:
    lm_name = "Synthyra/ESMplusplus_6B"
    inversion_model_names = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
    ]
    hero_critic_model_names = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
        "ESMFold2-Experimental",
        "ESMFold2-Experimental-Cutoff2025",
    ]

    def load(
        self,
        use_scaling_critics: bool = False,
        device: str = "cuda",
        kernel_backend: str | None = None,
        compile_model: bool = False,
    ) -> None:
        scaling_critic_names: list[str] = []
        if use_scaling_critics:
            scaling_critic_names = [
                f"ESMFold2-Experimental-Fast-base{size}-step{step}k"
                for size in ("300M", "600M", "6B")
                for step in ("250", "500", "750", "1000", "1500")
            ]
        self.inversion_models = {
            model_name: _load_fold_model(
                model_name,
                lm_dropout=0.5,
                cache_esmc=True,
                device=device,
                kernel_backend=kernel_backend,
                compile_model=compile_model,
            )
            for model_name in self.inversion_model_names
        }
        self.critic_models = {
            model_name: _load_fold_model(
                model_name,
                lm_dropout=0.25,
                cache_esmc=True,
                device=device,
                kernel_backend=kernel_backend,
                compile_model=compile_model,
            )
            for model_name in self.hero_critic_model_names
        }
        for model_name in scaling_critic_names:
            self.critic_models[model_name] = _load_fold_model(
                model_name,
                lm_dropout=0.25,
                cache_esmc=False,
                device="cpu",
                kernel_backend=kernel_backend,
                compile_model=False,
            )
        self.lm_model = (
            AutoModelForMaskedLM.from_pretrained(
                self.lm_name, trust_remote_code=True, dtype=torch.float32
            )
            .to(device=device)
            .eval()
            .requires_grad_(False)
        )

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
        )


def _build_modal_image():
    assert modal is not None, "Modal is not installed."
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12"
        )
        .apt_install("git", "build-essential")
        .uv_pip_install(
            "torch==2.11.0",
            "transformers==4.57.6",
            "accelerate==1.12.0",
            "hf-xet==1.5.0",
            "huggingface_hub",
            "numpy==1.26.4",
            "einops==0.8.2",
            "tokenizers",
            "safetensors",
            "pandas",
            "pyarrow",
            "biopython",
            "tqdm",
            "abnumber",
            "biotite==1.6.0",
            "rdkit==2026.3.2",
            "msgpack-numpy==0.4.8",
            "py3dmol==2.5.5",
            index_url="https://download.pytorch.org/whl/cu128",
            extra_index_url="https://pypi.org/simple",
        )
        .add_local_python_source("fastplms")
        .env({"HF_HOME": "/models", "HF_XET_HIGH_PERFORMANCE": "1"})
    )


if modal is not None:
    app = modal.App(name="fastplms-binder-design")
    _MODAL_IMAGE = _build_modal_image()
    _MODAL_MODEL_CACHE = modal.Volume.from_name(
        "fastplms-binder-design-models", create_if_missing=True
    )

    @app.cls(
        gpu="H100",
        image=_MODAL_IMAGE,
        volumes={"/models": _MODAL_MODEL_CACHE},
        timeout=60 * 60,
        cpu=16,
        memory=10 * 1024,
    )
    class FastPLMsBinderDesignModal(FastPLMsBinderDesign):
        use_scaling_critics: bool = modal.parameter(default=False)
        kernel_backend: str | None = modal.parameter(default=None)
        compile_model: bool = modal.parameter(default=False)

        @modal.enter()
        def load(self) -> None:
            super().load(
                use_scaling_critics=self.use_scaling_critics,
                kernel_backend=self.kernel_backend,
                compile_model=self.compile_model,
            )

        @modal.method()
        def design(self, *args, **kwargs):
            return super().design(*args, **kwargs)

    @app.local_entrypoint()
    def modal_main(
        target_name: str | None = "pd-l1",
        target_sequence: str | None = None,
        binder_name: str | None = "minibinder",
        binder_sequence: str | None = None,
        use_scaling_critics: bool = False,
        is_antibody: bool | None = None,
        seed: int = 0,
        batch_size: int = 1,
        steps: int = DEFAULT_STEPS,
        output_dir: str | None = "binder_design_out",
    ) -> None:
        remote_app = FastPLMsBinderDesignModal(
            use_scaling_critics=use_scaling_critics
        )
        best_sequences, _, results = remote_app.design.remote(
            target_name=target_name,
            target_sequence=target_sequence,
            binder_name=binder_name,
            binder_sequence=binder_sequence,
            is_antibody=is_antibody,
            seed=seed,
            batch_size=batch_size,
            steps=steps,
            output_dir=output_dir,
        )
        logger.info("Designed sequences: %s", best_sequences)
        logger.info("Returned %d critic rows", len(results))
        _log_official_selection_summary(results)


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
    runner = FastPLMsBinderDesign()
    runner.load(
        use_scaling_critics=args.use_scaling_critics,
        kernel_backend=args.kernel_backend,
        compile_model=args.compile_model,
    )
    best_sequences, _, results = runner.design(**_design_kwargs_from_args(args))
    logger.info("Designed sequences: %s", best_sequences)
    logger.info("Returned %d critic rows", len(results))
    _log_official_selection_summary(results)


def run_deployed_modal(args: argparse.Namespace) -> None:
    assert modal is not None, "Modal is not installed."
    remote_cls = modal.Cls.from_name(args.modal_app_name, "FastPLMsBinderDesignModal")
    remote_runner = remote_cls(
        use_scaling_critics=args.use_scaling_critics,
        kernel_backend=args.kernel_backend,
        compile_model=args.compile_model,
    )
    best_sequences, _, results = remote_runner.design.remote(
        **_design_kwargs_from_args(args)
    )
    logger.info("Designed sequences: %s", best_sequences)
    logger.info("Returned %d critic rows", len(results))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_results_table(output_dir / "results.parquet", results)
    _write_official_selection_table(output_dir / "selection.parquet", results)
    _log_official_selection_summary(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--modal-app-name", default="fastplms-binder-design")
    parser.add_argument("--target-name", default="pd-l1")
    parser.add_argument("--target-sequence", default=None)
    parser.add_argument("--binder-name", default="minibinder")
    parser.add_argument("--binder-sequence", default=None)
    parser.add_argument("--use-scaling-critics", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output-dir", default="binder_design_out")
    parser.add_argument("--kernel-backend", default=None)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--is-antibody", dest="is_antibody", action="store_true")
    parser.add_argument("--not-antibody", dest="is_antibody", action="store_false")
    parser.set_defaults(is_antibody=None)
    args = parser.parse_args()
    if args.target_sequence is not None:
        args.target_name = None
    if args.binder_sequence is not None:
        args.binder_name = None
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.backend == "local":
        run_local(cli_args)
    else:
        run_deployed_modal(cli_args)
