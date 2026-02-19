"""Shared test utilities for the FastPLMs test suite."""

import random

import torch
from typing import Optional

from tests.model_registry import ModelSpec


CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
LOAD_DTYPE = torch.float32
RUNTIME_DTYPE = torch.bfloat16


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def generate_sequences(num_sequences: int, min_length: int, max_length: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    sequences: list[str] = []
    for _ in range(num_sequences):
        length = rng.randint(min_length, max_length)
        sequence = "M" + "".join(rng.choices(CANONICAL_AMINO_ACIDS, k=length - 1))
        sequences.append(sequence)
    return sequences


def load_our_model(spec: ModelSpec, device: torch.device, dtype: torch.dtype, attn_backend: Optional[str] = None):
    """Load our implementation of a model. Returns (model, tokenizer)."""
    if spec.family == "esm2":
        from esm2.modeling_fastesm import FastEsmConfig, FastEsmForMaskedLM
        config = FastEsmConfig.from_pretrained(spec.repo_id)
        if attn_backend is not None:
            config.attn_backend = attn_backend
        model = FastEsmForMaskedLM.from_pretrained(spec.repo_id, config=config, dtype=dtype)
    elif spec.family == "esmplusplus":
        from esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig, ESMplusplusForMaskedLM
        config = ESMplusplusConfig.from_pretrained(spec.repo_id)
        if attn_backend is not None:
            config.attn_backend = attn_backend
        model = ESMplusplusForMaskedLM.from_pretrained(spec.repo_id, config=config, dtype=dtype)
    elif spec.family == "e1":
        from e1_fastplms.modeling_e1 import E1Config, E1ForMaskedLM
        config = E1Config.from_pretrained(spec.repo_id)
        model = E1ForMaskedLM.from_pretrained(spec.repo_id, config=config, dtype=dtype)
    elif spec.family == "dplm":
        from dplm_fastplms.modeling_dplm import DPLMConfig, DPLMForMaskedLM
        config = DPLMConfig.from_pretrained(spec.repo_id)
        if attn_backend is not None:
            config.attn_backend = attn_backend
        model = DPLMForMaskedLM.from_pretrained(spec.repo_id, config=config, dtype=dtype)
    elif spec.family == "dplm2":
        from dplm2_fastplms.modeling_dplm2 import DPLM2Config, DPLM2ForMaskedLM
        config = DPLM2Config.from_pretrained(spec.repo_id)
        if attn_backend is not None:
            config.attn_backend = attn_backend
        model = DPLM2ForMaskedLM.from_pretrained(spec.repo_id, config=config, dtype=dtype)
    else:
        raise ValueError(f"Unknown model family: {spec.family}")

    model = model.to(device=device, dtype=dtype).eval()
    tokenizer = None if spec.family == "e1" else model.tokenizer
    return model, tokenizer


def load_official_model(spec: ModelSpec, device: torch.device, dtype: torch.dtype):
    """Load the official reference model. Returns (model, tokenizer_or_batch_preparer)."""
    assert spec.reference_repo_id is not None, f"No reference repo for {spec.key}"
    if spec.family == "esm2":
        from esm2.load_official import load_official_model as _load
    elif spec.family == "esmplusplus":
        from esm_plusplus.load_official import load_official_model as _load
    elif spec.family == "e1":
        from e1_fastplms.load_official import load_official_model as _load
    elif spec.family == "dplm":
        from dplm_fastplms.load_official import load_official_model as _load
    elif spec.family == "dplm2":
        from dplm2_fastplms.load_official import load_official_model as _load
    else:
        raise ValueError(f"Unknown model family: {spec.family}")
    return _load(reference_repo_id=spec.reference_repo_id, device=device, dtype=dtype)


def tokenize_batch(
    spec: ModelSpec,
    sequences: list[str],
    model,
    tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of sequences for the given model family."""
    if spec.family == "e1":
        return model.prep_tokens.get_batch_kwargs(sequences, device=device)
    assert tokenizer is not None
    batch = tokenizer(sequences, return_tensors="pt", padding="longest")
    return batch.to(device)


def tokenize_official_batch(
    spec: ModelSpec,
    sequences: list[str],
    tokenizer_or_preparer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenize for the official model."""
    if spec.family == "e1":
        raw_batch = tokenizer_or_preparer.get_batch_kwargs(sequences, device=device)
        return {
            "input_ids": raw_batch["input_ids"],
            "within_seq_position_ids": raw_batch["within_seq_position_ids"],
            "global_position_ids": raw_batch["global_position_ids"],
            "sequence_ids": raw_batch["sequence_ids"],
        }
    assert tokenizer_or_preparer is not None
    batch = tokenizer_or_preparer(sequences, return_tensors="pt", padding="longest")
    return batch.to(device)


def get_non_pad_mask(spec: ModelSpec, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Get a boolean mask for non-pad positions. Returns (batch, seq_len) bool tensor on CPU."""
    if spec.family == "e1":
        return (batch["sequence_ids"] != -1).cpu()
    return batch["attention_mask"].cpu().bool()


def extract_official_state_dict(spec: ModelSpec, official_model) -> dict[str, torch.Tensor]:
    """Extract the comparable state dict from the wrapped official model.

    Handles: wrapper nn.Module prefix stripping, family-specific key
    filtering (position_embeddings for ESM2, contact_head for DPLM2,
    net. prefix for DPLM/DPLM2).
    """
    sd = official_model.state_dict()

    # All our load_official wrappers are nn.Modules with self.model,
    # which adds a 'model.' prefix to every key.
    if sd and all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}

    if spec.family == "esm2":
        sd = {k: v for k, v in sd.items()
              if "position_embeddings" not in k and "position_ids" not in k}

    if spec.family in ("dplm", "dplm2"):
        # Official DPLM wraps the ESM model inside .net
        sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}

    if spec.family == "dplm2":
        excluded = {"esm.contact_head.regression.weight", "esm.contact_head.regression.bias"}
        sd = {k: v for k, v in sd.items() if k not in excluded}

    return sd


def compare_state_dicts(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    max_report: int = 5,
) -> dict:
    """Compare two state dicts. Returns a dict with match info."""
    ref_keys = set(reference.keys())
    cand_keys = set(candidate.keys())
    only_ref = sorted(ref_keys - cand_keys)
    only_cand = sorted(cand_keys - ref_keys)
    common = sorted(ref_keys & cand_keys)

    diffs: list[dict] = []
    max_abs_diff = 0.0
    max_diff_param = ""

    for name in common:
        ref_t = reference[name].detach().cpu().to(torch.float32)
        cand_t = candidate[name].detach().cpu().to(torch.float32)
        if ref_t.shape != cand_t.shape:
            diffs.append({"name": name, "error": "shape_mismatch", "ref": list(ref_t.shape), "cand": list(cand_t.shape)})
            continue
        if torch.equal(ref_t, cand_t):
            continue
        abs_d = torch.abs(ref_t - cand_t)
        param_max = float(abs_d.max().item())
        param_mean = float(abs_d.mean().item())
        diffs.append({"name": name, "max_abs_diff": param_max, "mean_abs_diff": param_mean})
        if param_max > max_abs_diff:
            max_abs_diff = param_max
            max_diff_param = name

    return {
        "match": len(common) > 0 and len(diffs) == 0,
        "common_params": len(common),
        "only_in_reference": only_ref[:max_report],
        "only_in_candidate": only_cand[:max_report],
        "diffs": diffs[:max_report],
        "max_abs_diff": max_abs_diff,
        "max_diff_param": max_diff_param,
    }
