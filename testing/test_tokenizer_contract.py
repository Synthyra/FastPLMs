"""Tokenizer contract tests for all FastPLMs sequence checkpoints."""

from __future__ import annotations

from typing import Dict

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer, EsmTokenizer

from fastplms.ankh.modeling_ankh import FastAnkhConfig, _load_ankh_tokenizer
from fastplms.e1.modeling_e1 import E1BatchPreparer, get_tokenizer
from fastplms.esm_plusplus.modeling_esm_plusplus import EsmSequenceTokenizer
from testing.conftest import CANONICAL_AAS, FULL_MODEL_REGISTRY, mark_by_size


TOKENIZER_MODEL_KEYS = [
    key
    for key, value in FULL_MODEL_REGISTRY.items()
    if value["uses_tokenizer"]
]
CANONICAL_SEQUENCES = [
    "M" + CANONICAL_AAS,
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    "MXXBZUOACDEFGHIKLMNPQRSTVWY",
]


def _fast_tokenizer(config: Dict):
    if config["model_type"] == "ANKH":
        fast_config = AutoConfig.from_pretrained(
            config["fast_path"],
            trust_remote_code=True,
        )
        return _load_ankh_tokenizer(fast_config)
    if config["model_type"] == "ESMC":
        return EsmSequenceTokenizer()
    if config["model_type"] in ("ESM2", "DPLM", "DPLM2"):
        return EsmTokenizer.from_pretrained(config["fast_path"])
    return AutoTokenizer.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
    )


def _reference_tokenizer(config: Dict):
    if config["model_type"] == "ESMC":
        return EsmSequenceTokenizer()
    if config["model_type"] in ("ESM2", "DPLM", "DPLM2"):
        return EsmTokenizer.from_pretrained(config["official_path"])
    return AutoTokenizer.from_pretrained(
        config["official_path"],
        trust_remote_code=True,
    )


def _token_ids(tokenizer, sequence: str) -> torch.Tensor:
    encoded = tokenizer(
        sequence,
        return_tensors="pt",
    )
    return encoded["input_ids"]


def _special_token_ids(tokenizer) -> Dict[str, int | None]:
    return {
        "pad_token_id": tokenizer.pad_token_id,
        "cls_token_id": tokenizer.cls_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }


@pytest.mark.parametrize(
    "model_key",
    mark_by_size(TOKENIZER_MODEL_KEYS, FULL_MODEL_REGISTRY),
)
def test_sequence_tokenizer_matches_reference(model_key: str) -> None:
    config = FULL_MODEL_REGISTRY[model_key]
    fast_tok = _fast_tokenizer(config)
    reference_tok = _reference_tokenizer(config)

    fast_vocab = fast_tok.get_vocab()
    reference_vocab = reference_tok.get_vocab()
    assert len(fast_vocab) == len(reference_vocab), (
        f"{model_key}: vocab size mismatch fast={len(fast_vocab)} "
        f"reference={len(reference_vocab)}"
    )

    missing_in_fast = [
        token
        for token in reference_vocab
        if token not in fast_vocab
    ]
    assert not missing_in_fast, (
        f"{model_key}: tokens missing from fast tokenizer: {missing_in_fast[:5]}"
    )

    id_mismatches = [
        (token, reference_vocab[token], fast_vocab[token])
        for token in reference_vocab
        if reference_vocab[token] != fast_vocab[token]
    ]
    assert not id_mismatches, (
        f"{model_key}: token id mismatches: {id_mismatches[:5]}"
    )

    assert _special_token_ids(fast_tok) == _special_token_ids(reference_tok), (
        f"{model_key}: special token ids differ"
    )

    for sequence in CANONICAL_SEQUENCES:
        fast_ids = _token_ids(fast_tok, sequence)
        reference_ids = _token_ids(reference_tok, sequence)
        assert torch.equal(fast_ids, reference_ids), (
            f"{model_key}: encoded ids differ for {sequence[:16]} "
            f"fast={fast_ids[0, :8].tolist()} "
            f"reference={reference_ids[0, :8].tolist()}"
        )


def test_ankh_tokenizer_loader_falls_back_for_bare_config() -> None:
    fast_tok = _load_ankh_tokenizer(FastAnkhConfig())
    reference_tok = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")

    assert len(fast_tok.get_vocab()) == len(reference_tok.get_vocab())
    assert _special_token_ids(fast_tok) == _special_token_ids(reference_tok)
    assert torch.equal(
        _token_ids(fast_tok, CANONICAL_SEQUENCES[0]),
        _token_ids(reference_tok, CANONICAL_SEQUENCES[0]),
    )


def test_e1_sequence_mode_tokenizer_contract() -> None:
    tokenizer = get_tokenizer()
    preparer = E1BatchPreparer(tokenizer=tokenizer)
    sequences = [
        "M" + CANONICAL_AAS,
        "M" + CANONICAL_AAS[::-1],
    ]

    assert tokenizer.token_to_id("<pad>") == 0
    for token in ("<bos>", "<eos>", "1", "2", "?", "X"):
        token_id = tokenizer.token_to_id(token)
        assert token_id is not None, f"E1 token missing from tokenizer: {token}"

    batch = preparer.get_batch_kwargs(
        sequences,
        device=torch.device("cpu"),
    )
    input_ids = batch["input_ids"]
    sequence_ids = batch["sequence_ids"]
    within_seq_position_ids = batch["within_seq_position_ids"]
    global_position_ids = batch["global_position_ids"]

    assert input_ids.shape == sequence_ids.shape
    assert input_ids.shape == within_seq_position_ids.shape
    assert input_ids.shape == global_position_ids.shape
    assert input_ids.shape[0] == len(sequences)
    assert bool((sequence_ids == -1).eq(input_ids == tokenizer.token_to_id("<pad>")).all())
    assert bool((within_seq_position_ids[sequence_ids != -1] >= 0).all())
    assert bool((global_position_ids[sequence_ids != -1] >= 0).all())
