"""Tokenizer contract tests for all FastPLMs sequence checkpoints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, EsmTokenizer

from fastplms.models.ankh.modeling_ankh import (
    FastAnkhConfig,
    _load_ankh_tokenizer,
)
from fastplms.models.dplm.modeling_dplm import DPLMPreTrainedModel
from fastplms.models.dplm2.modeling_dplm2 import (
    DPLM2Config,
    DPLM2PreTrainedModel,
    _normalize_dplm2_input_ids,
)
from fastplms.models.dplm2.tokenization_dplm2 import DPLM2Tokenizer
from fastplms.models.e1.modeling_e1 import E1BatchPreparer, E1Config, E1ForMaskedLM, get_tokenizer
from fastplms.models.esm2.modeling_fastesm import FastEsmPreTrainedModel, FastEsmTokenizer
from fastplms.models.esm3.modeling_esm3 import (
    SEQUENCE_VOCAB as ESM3_SEQUENCE_VOCAB,
)
from fastplms.models.esm3.modeling_esm3 import (
    EsmSequenceTokenizer as ESM3SequenceTokenizer,
)
from fastplms.models.esm_plusplus.modeling_esm_plusplus import EsmSequenceTokenizer
from tests.conftest import CANONICAL_AAS, FULL_MODEL_REGISTRY, mark_by_size

TOKENIZER_REFERENCE_KEYS = [
    key
    for key, value in FULL_MODEL_REGISTRY.items()
    if value["uses_tokenizer"] and value["model_type"] not in {"DPLM2", "ESM3"}
]
ESM3_MODEL_KEYS = [
    key for key, value in FULL_MODEL_REGISTRY.items() if value["model_type"] == "ESM3"
]
DPLM2_MODEL_KEYS = [
    key for key, value in FULL_MODEL_REGISTRY.items() if value["model_type"] == "DPLM2"
]


def test_dplm2_model_tokenizer_uses_checkpoint_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_tokenizer = object()
    requests: list[tuple[str, dict[str, str]]] = []

    def load_tokenizer(source: str, **kwargs: str) -> object:
        requests.append((source, kwargs))
        return expected_tokenizer

    monkeypatch.setattr(DPLM2Tokenizer, "from_pretrained", staticmethod(load_tokenizer))
    owner = SimpleNamespace(
        _fastplms_tokenizer=None,
        config=SimpleNamespace(
            _name_or_path="local-dplm2-artifact",
            _commit_hash="b" * 40,
        ),
    )

    actual = DPLM2PreTrainedModel.tokenizer.fget(owner)

    assert actual is expected_tokenizer
    assert owner.__dict__["_fastplms_tokenizer"] is expected_tokenizer
    assert requests == [("local-dplm2-artifact", {"revision": "b" * 40})]


def test_dplm2_model_tokenizer_rejects_missing_checkpoint_provenance() -> None:
    owner = SimpleNamespace(
        _fastplms_tokenizer=None,
        config=SimpleNamespace(_name_or_path="", _commit_hash=None),
    )

    with pytest.raises(RuntimeError, match="loaded with from_pretrained"):
        DPLM2PreTrainedModel.tokenizer.fget(owner)


def test_dplm2_sequence_adapter_adds_amino_acid_boundaries_without_generic_specials() -> None:
    observed: dict[str, object] = {}

    class RecordingTokenizer:
        aa_cls_token = "<cls_aa>"
        aa_eos_token = "<eos_aa>"

        def __call__(self, sequences: list[str], **kwargs: object) -> dict[str, object]:
            observed.update(sequences=sequences, kwargs=kwargs)
            return {"input_ids": sequences}

    tokenizer = RecordingTokenizer()
    owner = SimpleNamespace(tokenizer=tokenizer)

    encoded = DPLM2PreTrainedModel._tokenize_sequence_batch(
        owner,
        ["AC", "M"],
        tokenizer=tokenizer,
        return_tensors="pt",
        padding=True,
    )

    assert encoded == {"input_ids": ["<cls_aa>AC<eos_aa>", "<cls_aa>M<eos_aa>"]}
    assert observed == {
        "sequences": ["<cls_aa>AC<eos_aa>", "<cls_aa>M<eos_aa>"],
        "kwargs": {
            "add_special_tokens": False,
            "return_tensors": "pt",
            "padding": True,
        },
    }


@pytest.mark.parametrize(
    ("model_class", "model_name"),
    (
        (DPLMPreTrainedModel, "DPLM"),
        (FastEsmPreTrainedModel, "ESM2"),
    ),
)
def test_esm_tokenizer_loaders_use_checkpoint_provenance(
    monkeypatch: pytest.MonkeyPatch,
    model_class: type,
    model_name: str,
) -> None:
    expected_tokenizer = object()
    requests: list[tuple[str, dict[str, str]]] = []

    def load_tokenizer(source: str, **kwargs: str) -> object:
        requests.append((source, kwargs))
        return expected_tokenizer

    monkeypatch.setattr(EsmTokenizer, "from_pretrained", staticmethod(load_tokenizer))
    owner = SimpleNamespace(
        _fastplms_tokenizer=None,
        config=SimpleNamespace(
            _name_or_path=f"local-{model_name.lower()}-artifact",
            _commit_hash="c" * 40,
        ),
    )

    actual = model_class.tokenizer.fget(owner)

    assert actual is expected_tokenizer
    assert requests == [(f"local-{model_name.lower()}-artifact", {"revision": "c" * 40})]


def test_esm2_tokenizer_normalizes_cls_as_bos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = SimpleNamespace(bos_token_id=None, cls_token="<cls>")
    monkeypatch.setattr(
        EsmTokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: tokenizer),
    )
    owner = SimpleNamespace(
        _fastplms_tokenizer=None,
        config=SimpleNamespace(_name_or_path="local-esm2-artifact", _commit_hash=None),
    )

    actual = FastEsmPreTrainedModel.tokenizer.fget(owner)

    assert actual.bos_token == "<cls>"


def test_esm2_tokenizer_rejects_residues_outside_official_alphabet() -> None:
    tokenizer = object.__new__(FastEsmTokenizer)
    tokenizer._token_to_id = {"A": 5}

    assert tokenizer._convert_token_to_id("A") == 5
    with pytest.raises(KeyError, match="J"):
        tokenizer._convert_token_to_id("J")


@pytest.mark.parametrize(
    ("model_class", "model_name"),
    (
        (DPLMPreTrainedModel, "DPLM"),
        (FastEsmPreTrainedModel, "ESM2"),
    ),
)
def test_esm_tokenizer_loaders_reject_missing_checkpoint_provenance(
    model_class: type,
    model_name: str,
) -> None:
    owner = SimpleNamespace(
        _fastplms_tokenizer=None,
        config=SimpleNamespace(_name_or_path="", _commit_hash=None),
    )

    with pytest.raises(RuntimeError, match=rf"{model_name} tokenizer loading requires"):
        model_class.tokenizer.fget(owner)


def test_ankh_tokenizer_loader_uses_checkpoint_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        backend_tokenizer = SimpleNamespace(pre_tokenizer=None)

    expected_tokenizer = Tokenizer()
    requests: list[tuple[str, dict[str, str]]] = []

    def load_tokenizer(source: str, **kwargs: str) -> object:
        requests.append((source, kwargs))
        return expected_tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", staticmethod(load_tokenizer))
    config = _tiny_ankh_config("local-ankh-artifact", "d" * 40)

    assert _load_ankh_tokenizer(config) is expected_tokenizer
    assert requests == [("local-ankh-artifact", {"revision": "d" * 40})]
    assert expected_tokenizer.backend_tokenizer.pre_tokenizer is not None


CANONICAL_SEQUENCES = [
    "M" + CANONICAL_AAS,
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    "MXXBZUOACDEFGHIKLMNPQRSTVWY",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _e1_tokenizer_json() -> Path:
    return _repo_root() / "src" / "fastplms" / "models" / "e1" / "tokenizer.json"


def _tiny_ankh_config(
    name_or_path: str = "",
    revision: str | None = None,
) -> FastAnkhConfig:
    config = FastAnkhConfig(
        vocab_size=4,
        d_model=8,
        d_kv=4,
        d_ff=16,
        num_heads=2,
        num_layers=1,
    )
    config._name_or_path = name_or_path
    config._commit_hash = revision
    return config


def _fast_tokenizer(config: dict):
    if config["model_type"] == "ANKH":
        # ANKH artifacts are built from the manifest's pinned official source,
        # including its byte-exact tokenizer assets.
        return _load_ankh_tokenizer(
            _tiny_ankh_config(config["official_path"], config["official_revision"])
        )
    if config["model_type"] == "ESMC":
        return EsmSequenceTokenizer()
    if config["model_type"] in ("ESM2", "DPLM"):
        return EsmTokenizer.from_pretrained(
            config["fast_path"],
            revision=config["fast_revision"],
        )
    return AutoTokenizer.from_pretrained(
        config["fast_path"],
        revision=config["fast_revision"],
        trust_remote_code=True,
    )


def _reference_tokenizer(config: dict):
    if config["model_type"] == "ANKH":
        return _load_ankh_tokenizer(
            _tiny_ankh_config(config["official_path"], config["official_revision"])
        )
    if config["model_type"] == "ESMC":
        return EsmSequenceTokenizer()
    if config["model_type"] in ("ESM2", "DPLM"):
        return EsmTokenizer.from_pretrained(
            config["official_path"],
            revision=config["official_revision"],
        )
    return AutoTokenizer.from_pretrained(
        config["official_path"],
        revision=config["official_revision"],
        trust_remote_code=True,
    )


def _token_ids(tokenizer, sequence: str) -> torch.Tensor:
    encoded = tokenizer(
        sequence,
        return_tensors="pt",
    )
    return encoded["input_ids"]


def _special_token_ids(tokenizer) -> dict[str, int | None]:
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
        f"{model_key}: vocab size mismatch fast={len(fast_vocab)} reference={len(reference_vocab)}"
    )

    missing_in_fast = [token for token in reference_vocab if token not in fast_vocab]
    assert not missing_in_fast, (
        f"{model_key}: tokens missing from fast tokenizer: {missing_in_fast[:5]}"
    )

    id_mismatches = [
        (token, reference_vocab[token], fast_vocab[token])
        for token in reference_vocab
        if reference_vocab[token] != fast_vocab[token]
    ]
    assert not id_mismatches, f"{model_key}: token id mismatches: {id_mismatches[:5]}"

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
    expected_vocab = {token: token_id for token_id, token in enumerate(ESM3_SEQUENCE_VOCAB)}

    assert tokenizer.get_vocab() == expected_vocab, f"{model_key}: ESM3 sequence vocabulary changed"
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


def test_ankh_tokenizer_loader_rejects_missing_checkpoint_provenance() -> None:
    with pytest.raises(RuntimeError, match="ANKH tokenizer loading requires"):
        _load_ankh_tokenizer(_tiny_ankh_config())


@pytest.mark.parametrize(
    "model_key",
    mark_by_size(DPLM2_MODEL_KEYS, FULL_MODEL_REGISTRY),
)
def test_dplm2_tokenizer_special_ids_normalize_in_range(model_key: str) -> None:
    config = FULL_MODEL_REGISTRY[model_key]
    fast_config = DPLM2Config.from_pretrained(
        config["fast_path"],
        revision=config["fast_revision"],
    )
    tokenizer = DPLM2Tokenizer.from_pretrained(
        config["fast_path"],
        revision=config["fast_revision"],
    )

    generic_special_ids = torch.tensor(
        [
            [
                fast_config.vocab_size,
                fast_config.vocab_size + 1,
                fast_config.vocab_size + 2,
                fast_config.vocab_size + 3,
                -100,
            ]
        ]
    )
    expected = torch.tensor([[2, 3, 0, 32, -100]])
    normalized_special_ids = _normalize_dplm2_input_ids(
        generic_special_ids,
        vocab_size=fast_config.vocab_size,
    )
    assert torch.equal(normalized_special_ids, expected)

    aa_sequences = [
        f"{tokenizer.aa_cls_token}{sequence}{tokenizer.aa_eos_token}"
        for sequence in CANONICAL_SEQUENCES
    ]
    encoded = tokenizer(
        aa_sequences,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    normalized_input_ids = _normalize_dplm2_input_ids(
        encoded["input_ids"],
        vocab_size=fast_config.vocab_size,
    )
    valid_ids = normalized_input_ids[normalized_input_ids.ge(0)]
    assert bool(valid_ids.lt(fast_config.vocab_size).all())


def test_dplm2_tokenizer_preserves_multimodal_contract(tmp_path: Path) -> None:
    model_key = DPLM2_MODEL_KEYS[0]
    config = FULL_MODEL_REGISTRY[model_key]
    tokenizer = DPLM2Tokenizer.from_pretrained(
        config["official_path"],
        revision=config["official_revision"],
    )

    expected_tokens = {
        "<cls_aa>": 0,
        "<pad>": 1,
        "<eos_aa>": 2,
        "<unk_aa>": 3,
        "<mask_aa>": 32,
        "<cls_struct>": 33,
        "<eos_struct>": 34,
        "<unk_struct>": 35,
        "0000": 36,
        "8191": 8227,
        "<mask_struct>": 8229,
    }
    vocabulary = tokenizer.get_vocab()
    assert {token: vocabulary[token] for token in expected_tokens} == expected_tokens
    assert tokenizer.vocab_size == 8229
    assert len(tokenizer) == 8229
    assert _special_token_ids(tokenizer) == {
        "pad_token_id": 1,
        "cls_token_id": None,
        "eos_token_id": None,
        "mask_token_id": None,
        "unk_token_id": None,
    }
    assert tokenizer.all_special_ids == [0, 2, 3, 32, 33, 34, 35, 8229, 1]

    aa_track = tokenizer(
        "<cls_aa>AC<eos_aa>",
        add_special_tokens=False,
    )["input_ids"]
    structure_track = tokenizer(
        "<cls_struct>00000042<eos_struct>",
        add_special_tokens=False,
    )["input_ids"]
    assert aa_track == [0, 5, 23, 2]
    assert structure_track == [33, 36, 78, 34]
    with pytest.raises(ValueError):
        tokenizer("AC")

    tokenizer.save_pretrained(tmp_path)
    saved_config = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    for name in DPLM2Tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
        assert saved_config[name] == getattr(tokenizer, name)
    reloaded = DPLM2Tokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert reloaded.get_vocab() == vocabulary
    assert reloaded.all_special_ids == tokenizer.all_special_ids
    assert reloaded("<cls_aa>AC<eos_aa>", add_special_tokens=False) == tokenizer(
        "<cls_aa>AC<eos_aa>",
        add_special_tokens=False,
    )


def test_e1_config_uses_static_token_constants() -> None:
    with patch(
        "fastplms.models.e1.modeling_e1.get_tokenizer",
        side_effect=AssertionError("E1Config should not load tokenizer.json"),
    ):
        config = E1Config()

    assert config.vocab_size == 34
    assert config.pad_token_id == 0
    assert config.bos_token_id == 1
    assert config.eos_token_id == 2


def test_e1_sequence_preparer_uses_model_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class RecordingPreparer:
        def __init__(self, *, data_prep_config, **kwargs: object) -> None:
            observed["config"] = data_prep_config
            observed["kwargs"] = kwargs

    monkeypatch.setattr(
        "fastplms.models.e1.modeling_e1.E1BatchPreparer",
        RecordingPreparer,
    )
    config = E1Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_num_sequences=7,
        max_num_positions_within_seq=31,
        max_num_positions_global=64,
    )
    model = object.__new__(E1ForMaskedLM)
    torch.nn.Module.__init__(model)
    model.config = config
    model.__dict__["_fastplms_tokenizer_kwargs"] = {}
    model.__dict__["_fastplms_prep_tokens"] = None

    assert model.prep_tokens is model.prep_tokens
    prep_config = observed["config"]
    assert prep_config.max_num_sequences == 7
    assert prep_config.max_num_positions_within_seq == 31


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
    with (
        patch("fastplms.models.e1.preparation.os.path.isfile", return_value=False),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=AssertionError("missing local tokenizer should not call Hub download"),
        ) as hf_hub_download,
        pytest.raises(FileNotFoundError),
    ):
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
    e1_source = _repo_root() / "src" / "fastplms" / "models" / "e1"
    for source_name in (
        "attention.py",
        "cache.py",
        "modeling_e1.py",
        "preparation.py",
        "retrieval.py",
    ):
        shutil.copyfile(e1_source / source_name, tmp_path / source_name)

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
