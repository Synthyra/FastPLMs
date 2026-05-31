"""Tokenizer contract tests for all FastPLMs sequence checkpoints."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, EsmTokenizer

from fastplms.ankh.modeling_ankh import FAST_ANKH_ENCODER, FastAnkhConfig
from fastplms.dplm2.modeling_dplm2 import _normalize_dplm2_input_ids
from fastplms.e1.modeling_e1 import E1BatchPreparer, E1Config, E1ForMaskedLM, get_tokenizer
from fastplms.esm3.modeling_esm3 import (
    SEQUENCE_VOCAB as ESM3_SEQUENCE_VOCAB,
    EsmSequenceTokenizer as ESM3SequenceTokenizer,
)
from fastplms.esm_plusplus.modeling_esm_plusplus import EsmSequenceTokenizer
from testing.conftest import CANONICAL_AAS, FULL_MODEL_REGISTRY, mark_by_size


TOKENIZER_REFERENCE_KEYS = [
    key
    for key, value in FULL_MODEL_REGISTRY.items()
    if value["uses_tokenizer"] and value["model_type"] != "ESM3"
]
ESM3_MODEL_KEYS = [
    key
    for key, value in FULL_MODEL_REGISTRY.items()
    if value["model_type"] == "ESM3"
]
DPLM2_MODEL_KEYS = [
    key
    for key, value in FULL_MODEL_REGISTRY.items()
    if value["model_type"] == "DPLM2"
]
CANONICAL_SEQUENCES = [
    "M" + CANONICAL_AAS,
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    "MXXBZUOACDEFGHIKLMNPQRSTVWY",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _e1_tokenizer_json() -> Path:
    return _repo_root() / "fastplms" / "e1" / "tokenizer.json"


def _tiny_ankh_config(name_or_path: str = "") -> FastAnkhConfig:
    config = FastAnkhConfig(
        vocab_size=4,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_heads=2,
        num_layers=1,
    )
    config._name_or_path = name_or_path
    return config


def _fast_tokenizer(config: Dict):
    if config["model_type"] == "ANKH":
        encoder = FAST_ANKH_ENCODER(_tiny_ankh_config(config["fast_path"]))
        return encoder.tokenizer
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
    mark_by_size(TOKENIZER_REFERENCE_KEYS, FULL_MODEL_REGISTRY),
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


@pytest.mark.parametrize(
    "model_key",
    mark_by_size(ESM3_MODEL_KEYS, FULL_MODEL_REGISTRY),
)
def test_esm3_sequence_tokenizer_contract(model_key: str) -> None:
    tokenizer = ESM3SequenceTokenizer()
    expected_vocab = {
        token: token_id
        for token_id, token in enumerate(ESM3_SEQUENCE_VOCAB)
    }

    assert tokenizer.get_vocab() == expected_vocab, (
        f"{model_key}: ESM3 sequence vocabulary changed"
    )
    assert _special_token_ids(tokenizer) == {
        "pad_token_id": 1,
        "cls_token_id": 0,
        "eos_token_id": 2,
        "mask_token_id": 32,
        "unk_token_id": 3,
    }

    for sequence in CANONICAL_SEQUENCES:
        encoded = _token_ids(tokenizer, sequence)
        expected_ids = [0] + [expected_vocab[token] for token in sequence] + [2]
        assert encoded[0].tolist() == expected_ids, (
            f"{model_key}: encoded ids differ for {sequence[:16]}"
        )


def test_ankh_tokenizer_loader_falls_back_for_bare_config() -> None:
    encoder = FAST_ANKH_ENCODER(_tiny_ankh_config())
    fast_tok = encoder.tokenizer
    reference_tok = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")

    assert len(fast_tok.get_vocab()) == len(reference_tok.get_vocab())
    assert _special_token_ids(fast_tok) == _special_token_ids(reference_tok)
    assert torch.equal(
        _token_ids(fast_tok, CANONICAL_SEQUENCES[0]),
        _token_ids(reference_tok, CANONICAL_SEQUENCES[0]),
    )


@pytest.mark.parametrize(
    "model_key",
    mark_by_size(DPLM2_MODEL_KEYS, FULL_MODEL_REGISTRY),
)
def test_dplm2_tokenizer_special_ids_normalize_in_range(model_key: str) -> None:
    config = FULL_MODEL_REGISTRY[model_key]
    fast_config = AutoConfig.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
    )
    tokenizer = EsmTokenizer.from_pretrained(config["fast_path"])

    generic_special_ids = torch.tensor([[
        fast_config.vocab_size,
        fast_config.vocab_size + 1,
        fast_config.vocab_size + 2,
        fast_config.vocab_size + 3,
        -100,
    ]])
    expected = torch.tensor([[2, 3, 0, 32, -100]])
    normalized_special_ids = _normalize_dplm2_input_ids(
        generic_special_ids,
        vocab_size=fast_config.vocab_size,
    )
    assert torch.equal(normalized_special_ids, expected)

    encoded = tokenizer(
        CANONICAL_SEQUENCES,
        return_tensors="pt",
        padding=True,
    )
    normalized_input_ids = _normalize_dplm2_input_ids(
        encoded["input_ids"],
        vocab_size=fast_config.vocab_size,
    )
    valid_ids = normalized_input_ids[normalized_input_ids.ge(0)]
    assert bool(valid_ids.lt(fast_config.vocab_size).all())


def test_e1_config_uses_static_token_constants() -> None:
    with patch(
        "fastplms.e1.modeling_e1.get_tokenizer",
        side_effect=AssertionError("E1Config should not load tokenizer.json"),
    ):
        config = E1Config()

    assert config.vocab_size == 34
    assert config.pad_token_id == 0
    assert config.bos_token_id == 1
    assert config.eos_token_id == 2


def test_e1_get_tokenizer_prefers_local_model_dir(tmp_path: Path) -> None:
    shutil.copyfile(_e1_tokenizer_json(), tmp_path / "tokenizer.json")

    with patch(
        "huggingface_hub.hf_hub_download",
        side_effect=AssertionError("local tokenizer load should not call Hub download"),
    ) as hf_hub_download:
        tokenizer = get_tokenizer(tmp_path, local_files_only=True)

    assert not hf_hub_download.called
    assert tokenizer.token_to_id("<pad>") == 0
    assert tokenizer.get_vocab_size() == 34


def test_e1_get_tokenizer_local_files_only_missing_local_source_raises(
    tmp_path: Path,
) -> None:
    with patch("fastplms.e1.modeling_e1.os.path.isfile", return_value=False):
        with patch(
            "huggingface_hub.hf_hub_download",
            side_effect=AssertionError("missing local tokenizer should not call Hub download"),
        ) as hf_hub_download:
            with pytest.raises(FileNotFoundError):
                get_tokenizer(tmp_path, local_files_only=True)

    assert not hf_hub_download.called


def test_e1_automodel_local_files_only_uses_local_tokenizer(tmp_path: Path) -> None:
    config = E1Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_num_sequences=4,
        max_num_positions_within_seq=64,
        max_num_positions_global=128,
    )
    config.auto_map = {
        "AutoConfig": "modeling_e1.E1Config",
        "AutoModelForMaskedLM": "modeling_e1.E1ForMaskedLM",
    }
    model = E1ForMaskedLM(config)
    model.save_pretrained(tmp_path)
    shutil.copyfile(_e1_tokenizer_json(), tmp_path / "tokenizer.json")
    shutil.copyfile(
        _repo_root() / "fastplms" / "e1" / "modeling_e1.py",
        tmp_path / "modeling_e1.py",
    )

    with patch(
        "huggingface_hub.hf_hub_download",
        side_effect=AssertionError("local AutoModel load should not call Hub download"),
    ) as hf_hub_download:
        loaded = AutoModelForMaskedLM.from_pretrained(
            tmp_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    assert not hf_hub_download.called
    assert loaded.prep_tokens.tokenizer.token_to_id("<pad>") == 0


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
