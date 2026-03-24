"""Embedding mixin tests: NaN stability, batch-vs-single match, FASTA parsing, DPLM2 utilities."""

import os
import random
import tempfile
from typing import Dict, List

import pytest
import torch

from testing.conftest import (
    CANONICAL_AAS, FULL_MODEL_REGISTRY, MODEL_REGISTRY, SEED,
    mark_by_size,
)


BATCH_SIZE = 4
MAX_EMBED_LEN = 128


# Models that use tokenizer mode (not E1)
TOKENIZER_MODEL_KEYS = [k for k, v in MODEL_REGISTRY.items() if v["uses_tokenizer"]]
ALL_MODEL_KEYS = list(MODEL_REGISTRY.keys())
ALL_FULL_MODEL_KEYS = list(FULL_MODEL_REGISTRY.keys())
FULL_TOKENIZER_KEYS = [k for k, v in FULL_MODEL_REGISTRY.items() if v["uses_tokenizer"]]


class FixedLengthTokenizer:
    """Wraps a tokenizer so every call pads to exactly MAX_EMBED_LEN tokens.

    Both batch=1 and batch=N therefore receive tensors of the same shape,
    keeping max_seqlen_in_batch identical and eliminating floating-point
    variability from different softmax vector lengths / flash-attention tile sizes.
    """
    def __init__(self, tokenizer: object, max_length: int = MAX_EMBED_LEN) -> None:
        self._tok = tokenizer
        self.max_length = max_length

    def __call__(self, sequences: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        return self._tok(
            sequences,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )


def _random_sequences(n: int, min_len: int = 8, max_len: int = 64) -> List[str]:
    return [
        "M" + "".join(random.choices(CANONICAL_AAS, k=random.randint(min_len, max_len)))
        for _ in range(n)
    ]


def _random_sequences_fixed_len(n: int, length: int = 64) -> List[str]:
    return [
        "M" + "".join(random.choices(CANONICAL_AAS, k=length - 1))
        for _ in range(n)
    ]


def _assert_no_nan(embeddings: Dict[str, torch.Tensor], label: str) -> None:
    for seq, emb in embeddings.items():
        assert not torch.isnan(emb).any(), (
            f"[{label}] NaN found in embedding for sequence '{seq[:20]}...'"
        )


def _assert_embeddings_match(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    label: str,
    atol: float = 5e-3,
) -> None:
    assert set(a) == set(b), f"[{label}] Key sets differ between batch and single runs"
    for seq in a:
        ea, eb = a[seq].float(), b[seq].float()
        assert ea.shape == eb.shape, (
            f"[{label}] Shape mismatch for '{seq[:20]}': {ea.shape} vs {eb.shape}"
        )
        max_diff = (ea - eb).abs().max().item()
        assert max_diff <= atol, (
            f"[{label}] Max abs diff {max_diff:.5f} > {atol} for '{seq[:20]}'"
        )


# --- CPU-only utility tests ---

def test_parse_fasta() -> None:
    from fastplms.embedding_mixin import parse_fasta

    fasta_content = (
        ">seq1 a simple protein\n"
        "MKTLLLTLVVVTIVCLDLGYT\n"
        ">seq2 multi-line sequence\n"
        "ACDEFGHIKL\n"
        "MNPQRSTVWY\n"
        ">seq3 another entry\n"
        "MALWMRLLPLLALL\n"
    )
    expected = [
        "MKTLLLTLVVVTIVCLDLGYT",
        "ACDEFGHIKLMNPQRSTVWY",
        "MALWMRLLPLLALL",
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(fasta_content)
        tmp_path = f.name
    parsed = parse_fasta(tmp_path)
    os.unlink(tmp_path)
    assert parsed == expected


@pytest.mark.gpu
def test_dplm2_multimodal_layout_guard() -> None:
    from fastplms.dplm2.modeling_dplm2 import _has_packed_multimodal_layout

    plain = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 2], [1, 1, 1, 1, 1, 0, 2, 2]])
    packed = torch.tensor([[1, 1, 1, 2, 0, 0, 0, 2], [1, 1, 2, 2, 0, 0, 2, 2]])
    mismatched = torch.tensor([[1, 1, 1, 2, 0, 0, 2, 2]])

    assert not _has_packed_multimodal_layout(plain, aa_type=1, struct_type=0, pad_type=2)
    assert _has_packed_multimodal_layout(packed, aa_type=1, struct_type=0, pad_type=2)
    assert not _has_packed_multimodal_layout(mismatched, aa_type=1, struct_type=0, pad_type=2)


@pytest.mark.gpu
def test_dplm2_special_token_normalization() -> None:
    from fastplms.dplm2.modeling_dplm2 import _normalize_dplm2_input_ids

    input_ids = torch.tensor([[8231, 5, 23, 13, 8229, 1, 8232, -100]])
    normalized = _normalize_dplm2_input_ids(input_ids, vocab_size=8229)
    expected = torch.tensor([[0, 5, 23, 13, 2, 1, 32, -100]])
    assert torch.equal(normalized, expected), (
        f"DPLM2 normalization mismatch: got {normalized.tolist()}, expected {expected.tolist()}"
    )


# --- GPU model tests ---

@pytest.mark.gpu
@pytest.mark.parametrize("model_key", ALL_MODEL_KEYS)
def test_nan_stability(model_key: str) -> None:
    """Batched embed_dataset produces no NaN in real-token rows."""
    from transformers import AutoModelForMaskedLM

    random.seed(SEED)
    config = MODEL_REGISTRY[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    uses_tokenizer = config["uses_tokenizer"]
    if uses_tokenizer:
        tokenizer = FixedLengthTokenizer(model.tokenizer)
        sequences = _random_sequences(n=8)
    else:
        tokenizer = None
        sequences = _random_sequences_fixed_len(n=8)

    embs = model.embed_dataset(
        sequences=sequences,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        full_embeddings=True,
        embed_dtype=torch.bfloat16,
        save=False,
    )
    _assert_no_nan(embs, f"{model_key} NaN check batch_size={BATCH_SIZE}")

    del model
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", TOKENIZER_MODEL_KEYS)
def test_batch_single_match(model_key: str) -> None:
    """Batched and single-item embedding produce matching results (tokenizer models only).

    E1 is excluded: flash varlen is not bit-deterministic across different batch sizes.
    For SDPA models we cast to float32 to avoid bfloat16 CUBLAS algorithm selection differences.
    """
    from transformers import AutoModelForMaskedLM

    random.seed(SEED)
    config = MODEL_REGISTRY[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
        dtype=torch.float32,
        device_map=device,
    ).eval()

    tokenizer = FixedLengthTokenizer(model.tokenizer)
    sequences = _random_sequences(n=8)

    batch_embs = model.embed_dataset(
        sequences=sequences,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        full_embeddings=True,
        embed_dtype=torch.float32,
        save=False,
    )
    single_embs = model.embed_dataset(
        sequences=sequences,
        batch_size=1,
        tokenizer=tokenizer,
        full_embeddings=True,
        embed_dtype=torch.float32,
        save=False,
    )
    _assert_no_nan(batch_embs, f"{model_key} match test batch_size={BATCH_SIZE}")
    _assert_no_nan(single_embs, f"{model_key} match test batch_size=1")
    _assert_embeddings_match(batch_embs, single_embs, model_key)

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Full model registry tests: NaN stability across all checkpoints
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("model_key", mark_by_size(ALL_FULL_MODEL_KEYS, FULL_MODEL_REGISTRY))
def test_full_nan_stability(model_key: str) -> None:
    """Every checkpoint's embed_dataset produces no NaN in real-token rows."""
    from transformers import AutoModelForMaskedLM

    random.seed(SEED)
    config = FULL_MODEL_REGISTRY[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    uses_tokenizer = config["uses_tokenizer"]
    if uses_tokenizer:
        tokenizer = FixedLengthTokenizer(model.tokenizer)
        sequences = _random_sequences(n=8)
    else:
        tokenizer = None
        sequences = _random_sequences_fixed_len(n=8)

    embs = model.embed_dataset(
        sequences=sequences,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        full_embeddings=True,
        embed_dtype=torch.bfloat16,
        save=False,
    )
    _assert_no_nan(embs, f"{model_key} NaN check batch_size={BATCH_SIZE}")

    del model
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", mark_by_size(FULL_TOKENIZER_KEYS, FULL_MODEL_REGISTRY))
def test_full_batch_single_match(model_key: str) -> None:
    """Every tokenizer-mode checkpoint matches batch vs single-item embedding."""
    from transformers import AutoModelForMaskedLM

    random.seed(SEED)
    config = FULL_MODEL_REGISTRY[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
        dtype=torch.float32,
        device_map=device,
    ).eval()

    tokenizer = FixedLengthTokenizer(model.tokenizer)
    sequences = _random_sequences(n=8)

    batch_embs = model.embed_dataset(
        sequences=sequences,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        full_embeddings=True,
        embed_dtype=torch.float32,
        save=False,
    )
    single_embs = model.embed_dataset(
        sequences=sequences,
        batch_size=1,
        tokenizer=tokenizer,
        full_embeddings=True,
        embed_dtype=torch.float32,
        save=False,
    )
    _assert_no_nan(batch_embs, f"{model_key} match test batch_size={BATCH_SIZE}")
    _assert_no_nan(single_embs, f"{model_key} match test batch_size=1")
    _assert_embeddings_match(batch_embs, single_embs, model_key)

    del model
    torch.cuda.empty_cache()
