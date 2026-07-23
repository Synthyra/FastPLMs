"""Pinned real-tokenizer contracts for ANKH source, decoder, and TTT paths."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from examples.ankh_embeddings import generate_ankh_task
from fastplms.models.ankh.modeling_ankh import (
    FastAnkhConfig,
    FastAnkhForConditionalGeneration,
    FastAnkhForMaskedLMExtension,
    FastAnkhModel,
    tokenize_ankh_decoder_prompts,
    tokenize_ankh_sequences,
)
from fastplms.registry import FileDigest, get_model_registry

pytestmark = [pytest.mark.compliance, pytest.mark.network, pytest.mark.reference]
_SNAPSHOT_ENVIRONMENT = "FASTPLMS_ANKH_TOKENIZER_SNAPSHOT"
_SNAPSHOT_ROOT_ENVIRONMENT = "FASTPLMS_ANKH_TOKENIZER_SNAPSHOT_ROOT"
_ANKH_SPEC_IDS = tuple(spec.id for spec in get_model_registry().by_family("ankh"))
_TOKENIZER_PATHS = (
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _file_digest(path: Path, identity: FileDigest) -> str:
    payload = path.read_bytes()
    if identity.algorithm == "sha256":
        return hashlib.sha256(payload).hexdigest()
    if identity.algorithm == "git-sha1":
        header = f"blob {len(payload)}\0".encode()
        return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()
    raise AssertionError(f"Unsupported manifest digest: {identity.algorithm}")


@pytest.fixture(scope="module")
def pinned_ankh_tokenizer_snapshot(request: pytest.FixtureRequest) -> tuple[Path, str]:
    spec_id = getattr(request, "param", "ankh_base")
    spec = get_model_registry()[spec_id]
    configured = os.environ.get(_SNAPSHOT_ENVIRONMENT)
    configured_root = os.environ.get(_SNAPSHOT_ROOT_ENVIRONMENT)
    if configured_root:
        snapshot = Path(configured_root).expanduser().resolve() / spec.id
    elif configured and spec.id == "ankh_base":
        snapshot = Path(configured).expanduser().resolve()
    else:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                spec.official.repo_id,
                revision=spec.official.revision,
                allow_patterns=list(_TOKENIZER_PATHS),
            )
        )
    assert snapshot.is_dir(), f"Pinned ANKH tokenizer snapshot is missing: {snapshot}"
    for relative_path in _TOKENIZER_PATHS:
        identity = spec.official.file_map[relative_path]
        path = snapshot / relative_path
        assert path.is_file(), f"Pinned ANKH tokenizer asset is missing: {path}"
        assert _file_digest(path, identity) == identity.digest, (
            f"Pinned ANKH tokenizer asset does not match the manifest: {relative_path}"
        )
    return snapshot, spec.official.revision


def _tiny_config(snapshot: Path, revision: str, vocab_size: int) -> FastAnkhConfig:
    config = FastAnkhConfig(
        vocab_size=vocab_size,
        d_model=16,
        d_kv=8,
        d_ff=32,
        num_heads=2,
        num_layers=1,
        num_decoder_layers=1,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        attn_backend="eager",
        use_cache=True,
    )
    config._name_or_path = str(snapshot)
    config._commit_hash = revision
    return config


@pytest.mark.parametrize(
    "pinned_ankh_tokenizer_snapshot",
    _ANKH_SPEC_IDS,
    indirect=True,
)
def test_every_real_ankh_tokenizer_preserves_raw_residues_and_tight_sentinels(
    pinned_ankh_tokenizer_snapshot: tuple[Path, str],
) -> None:
    snapshot, _ = pinned_ankh_tokenizer_snapshot
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)

    raw = tokenize_ankh_sequences(
        tokenizer,
        "MSTNPK",
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    legacy_spaced = tokenize_ankh_sequences(
        tokenizer,
        "M S T N P K",
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    tight_prompt = tokenize_ankh_decoder_prompts(
        tokenizer,
        "M<extra_id_0>",
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    spaced_prompt = tokenize_ankh_decoder_prompts(
        tokenizer,
        "M <extra_id_0>",
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    assert raw.shape == (1, 6)
    assert tokenizer.unk_token_id not in raw
    assert torch.equal(raw, legacy_spaced)
    assert torch.equal(tight_prompt, spaced_prompt)
    assert tokenizer.unk_token_id not in tight_prompt
    assert tokenizer.convert_tokens_to_ids("<extra_id_0>") in tight_prompt


def test_real_explicit_tokenizer_matches_model_scoped_tokenizer(
    pinned_ankh_tokenizer_snapshot: tuple[Path, str],
) -> None:
    snapshot, revision = pinned_ankh_tokenizer_snapshot
    explicit = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = FastAnkhModel(_tiny_config(snapshot, revision, len(explicit))).eval()
    model.__dict__["_fastplms_tokenizer_load_context"] = {"local_files_only": True}

    explicit_ids = tokenize_ankh_sequences(
        explicit,
        ["MSTNPK", "ACDE"],
        return_tensors="pt",
        padding=True,
    )["input_ids"]
    scoped_ids = tokenize_ankh_sequences(
        model.tokenizer,
        ["MSTNPK", "ACDE"],
        return_tensors="pt",
        padding=True,
    )["input_ids"]

    assert torch.equal(explicit_ids, scoped_ids)
    assert explicit.unk_token_id not in explicit_ids
    assert model.tokenizer.unk_token_id not in scoped_ids
    assert explicit.backend_tokenizer.to_str() == model.tokenizer.backend_tokenizer.to_str()


def test_real_sentinel_decoder_extraction_and_generation(
    pinned_ankh_tokenizer_snapshot: tuple[Path, str],
) -> None:
    snapshot, revision = pinned_ankh_tokenizer_snapshot
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = FastAnkhForConditionalGeneration(
        _tiny_config(snapshot, revision, len(tokenizer))
    ).eval()
    model.tokenizer = tokenizer

    batch = model._embedding_batch(
        ["MSTNPK"],
        tokenizer=tokenizer,
        hidden_state_source="decoder",
        decoder_inputs=["M <extra_id_0>"],
    )
    generated = generate_ankh_task(
        model,
        tokenizer,
        "MSTNPK",
        "M <extra_id_0>",
        max_new_tokens=1,
    )

    assert batch.residue_mask.sum().item() == 1
    assert generated.ndim == 2
    assert generated.shape[0] == 1
    assert tokenizer.unk_token_id not in generated[:, :3]


def test_real_ankh_ttt_uses_the_shared_raw_sequence_contract(
    pinned_ankh_tokenizer_snapshot: tuple[Path, str],
) -> None:
    snapshot, revision = pinned_ankh_tokenizer_snapshot
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = FastAnkhForMaskedLMExtension(_tiny_config(snapshot, revision, len(tokenizer))).eval()
    model.tokenizer = tokenizer

    ttt_ids = model._ttt_tokenize(seq="MSTNPK")
    shared_ids = tokenize_ankh_sequences(
        tokenizer,
        ["MSTNPK"],
        return_tensors="pt",
        padding=True,
    )["input_ids"]

    assert torch.equal(ttt_ids, shared_ids)
    assert tokenizer.unk_token_id not in ttt_ids
