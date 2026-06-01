"""Embedding mixin tests: NaN stability, batch-vs-single match, FASTA parsing, DPLM2 utilities."""

import os
import random
import sqlite3
import tempfile
from typing import Dict, List

import pytest
import torch

from fastplms.embedding_mixin import (
    EmbeddingMixin,
    load_pooled_embeddings_from_db,
    load_pooled_embeddings_from_pth,
    pool_embeddings,
    tensor_to_embedding_blob,
)
from testing.conftest import (
    CANONICAL_AAS, FULL_MODEL_REGISTRY, MODEL_REGISTRY, SEED,
    mark_by_size, strict_fp32_matmul,
)


BATCH_SIZE = 4
MAX_EMBED_LEN = 128
EMBED_MATCH_TOL = {
    "default": {"maxabs": 6e-3, "rel_maxabs": None},
    # ESM3 exposes the pre-final-norm residual stream as embeddings. Absolute
    # fp32 batch-shape noise can slightly exceed 1e-2 while relative error stays tiny.
    "esm3": {"maxabs": 1.5e-2, "rel_maxabs": 5e-6},
}


# Models that use tokenizer mode (not E1)
TOKENIZER_MODEL_KEYS = [k for k, v in MODEL_REGISTRY.items() if v["uses_tokenizer"]]
ALL_MODEL_KEYS = list(MODEL_REGISTRY.keys())
ALL_FULL_MODEL_KEYS = list(FULL_MODEL_REGISTRY.keys())
FULL_TOKENIZER_KEYS = [k for k, v in FULL_MODEL_REGISTRY.items() if v["uses_tokenizer"]]


class DummyEmbeddingConfig:
    model_type = "E1"
    hidden_size = 2


class DummyHiddenStateModel(torch.nn.Module, EmbeddingMixin):
    def __init__(self) -> None:
        super().__init__()
        self.config = DummyEmbeddingConfig()
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def _embed(
        self,
        sequences: List[str],
        return_attention_mask: bool = False,
        hidden_state_index: int = -1,
        store_all_hidden_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        max_len = max(len(sequence) for sequence in sequences)
        attention_mask = torch.zeros(len(sequences), max_len, dtype=torch.long)
        hidden_states = []
        for layer_idx in range(3):
            layer = torch.zeros(len(sequences), max_len, 2)
            for batch_idx, sequence in enumerate(sequences):
                seq_len = len(sequence)
                attention_mask[batch_idx, :seq_len] = 1
                positions = torch.arange(seq_len, dtype=torch.float32)
                layer[batch_idx, :seq_len, 0] = layer_idx
                layer[batch_idx, :seq_len, 1] = positions
            hidden_states.append(layer)
        if store_all_hidden_states:
            embeddings = torch.stack(hidden_states, dim=1)
        elif hidden_state_index == -1:
            embeddings = hidden_states[-1]
        else:
            embeddings = hidden_states[hidden_state_index]
        if return_attention_mask:
            return embeddings, attention_mask
        return embeddings


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
) -> None:
    assert set(a) == set(b), f"[{label}] Key sets differ between batch and single runs"
    if label.startswith("esm3"):
        tol = EMBED_MATCH_TOL["esm3"]
    else:
        tol = EMBED_MATCH_TOL["default"]
    for seq in a:
        ea, eb = a[seq].float(), b[seq].float()
        assert ea.shape == eb.shape, (
            f"[{label}] Shape mismatch for '{seq[:20]}': {ea.shape} vs {eb.shape}"
        )
        max_diff = (ea - eb).abs().max().item()
        base_maxabs = max(ea.abs().max().item(), eb.abs().max().item())
        rel_maxabs = max_diff / base_maxabs if base_maxabs > 1e-12 else 0.0
        rel_tol = tol["rel_maxabs"]
        rel_ok = rel_tol is None or rel_maxabs <= rel_tol
        assert max_diff <= tol["maxabs"] and rel_ok, (
            f"[{label}] Max abs diff {max_diff:.5f} > {tol['maxabs']} "
            f"or rel maxabs {rel_maxabs:.3e} > {rel_tol} for '{seq[:20]}'"
        )


@pytest.fixture
def disable_tf32_for_batch_single_match():
    # TF32 kernels can be batch-shape-dependent on Hopper/GH200, which defeats
    # this test's batch-vs-single equality check.
    with strict_fp32_matmul():
        yield


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


def test_pool_embeddings_selects_layer_from_all_hidden_states() -> None:
    all_layers = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [1.0, 2.0]],
            [[2.0, 0.0], [2.0, 2.0]],
        ]
    )
    pooled = pool_embeddings(
        {"AA": all_layers},
        pooling_types=["mean"],
        hidden_state_index=1,
    )
    assert torch.equal(pooled["AA"], torch.tensor([1.0, 1.0]))


def test_load_pooled_embeddings_from_pth_and_db() -> None:
    all_layers = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [1.0, 2.0]],
            [[2.0, 0.0], [2.0, 2.0]],
        ]
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        pth_path = os.path.join(tmp_dir, "embeddings.pth")
        db_path = os.path.join(tmp_dir, "embeddings.db")
        torch.save({"AA": all_layers}, pth_path)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "CREATE TABLE embeddings (sequence TEXT PRIMARY KEY, embedding BLOB NOT NULL)"
            )
            cursor.execute(
                "INSERT INTO embeddings VALUES (?, ?)",
                ("AA", tensor_to_embedding_blob(all_layers)),
            )
            conn.commit()

        pth_pooled = load_pooled_embeddings_from_pth(
            pth_path,
            pooling_types=["mean"],
            hidden_state_index=1,
        )
        db_pooled = load_pooled_embeddings_from_db(
            db_path,
            pooling_types=["mean"],
            hidden_state_index=1,
        )

    assert torch.equal(pth_pooled["AA"], torch.tensor([1.0, 1.0]))
    assert torch.equal(db_pooled["AA"], torch.tensor([1.0, 1.0]))


def test_embed_dataset_hidden_state_index_sequence_mode() -> None:
    model = DummyHiddenStateModel()
    sequences = ["ACD", "M"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "embeddings.pth")
        default_embeddings = model.embed_dataset(
            sequences=sequences,
            tokenizer=None,
            batch_size=2,
            pooling_types=["mean"],
            save=False,
            save_path=save_path,
            padding="longest",
        )
        selected_embeddings = model.embed_dataset(
            sequences=sequences,
            tokenizer=None,
            batch_size=2,
            pooling_types=["mean"],
            hidden_state_index=1,
            save=False,
            save_path=save_path,
            padding="longest",
        )

    assert torch.equal(default_embeddings["ACD"], torch.tensor([2.0, 1.0]))
    assert torch.equal(selected_embeddings["ACD"], torch.tensor([1.0, 1.0]))
    assert torch.equal(selected_embeddings["M"], torch.tensor([1.0, 0.0]))


def test_embed_dataset_store_all_hidden_states_sequence_mode() -> None:
    model = DummyHiddenStateModel()
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "embeddings.pth")
        embeddings = model.embed_dataset(
            sequences=["ACD", "M"],
            tokenizer=None,
            batch_size=2,
            full_embeddings=True,
            store_all_hidden_states=True,
            save=False,
            save_path=save_path,
            padding="longest",
        )

    assert embeddings["ACD"].shape == (3, 3, 2)
    assert embeddings["M"].shape == (3, 1, 2)
    assert torch.equal(embeddings["ACD"][1].mean(dim=0), torch.tensor([1.0, 1.0]))


def test_embed_dataset_store_all_hidden_states_sql_roundtrip() -> None:
    model = DummyHiddenStateModel()
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "embeddings.db")
        model.embed_dataset(
            sequences=["ACD", "M"],
            tokenizer=None,
            batch_size=2,
            full_embeddings=True,
            store_all_hidden_states=True,
            sql=True,
            sql_db_path=db_path,
            save=False,
            padding="longest",
        )
        pooled = model.load_pooled_embeddings_from_db(
            db_path,
            pooling_types=["mean"],
            hidden_state_index=1,
        )

    assert torch.equal(pooled["ACD"], torch.tensor([1.0, 1.0]))
    assert torch.equal(pooled["M"], torch.tensor([1.0, 0.0]))


def test_embed_dataset_store_all_hidden_states_requires_full_embeddings() -> None:
    model = DummyHiddenStateModel()
    with pytest.raises(AssertionError, match="requires full_embeddings"):
        model.embed_dataset(
            sequences=["ACD"],
            tokenizer=None,
            store_all_hidden_states=True,
            save=False,
            padding="longest",
        )


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
def test_batch_single_match(model_key: str, disable_tf32_for_batch_single_match) -> None:
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

    with strict_fp32_matmul():
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


@pytest.mark.gpu
def test_tokenizer_model_embed_dataset_uses_default_tokenizer() -> None:
    from transformers import AutoModelForMaskedLM

    config = MODEL_REGISTRY["esmc"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    embeddings = model.embed_dataset(
        sequences=["MKTAYIAKQ", "GGGG"],
        batch_size=2,
        max_len=16,
        pooling_types=["mean"],
        save=False,
    )

    assert set(embeddings) == {"MKTAYIAKQ", "GGGG"}
    assert embeddings["MKTAYIAKQ"].shape == (model.config.hidden_size,)

    del model
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", ALL_MODEL_KEYS)
def test_hidden_state_index_embed_dataset_smoke(model_key: str) -> None:
    from transformers import AutoModelForMaskedLM

    config = MODEL_REGISTRY[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    if config["uses_tokenizer"]:
        tokenizer = FixedLengthTokenizer(model.tokenizer, max_length=32)
        sequences = ["MKTAYIAKQ", "GGGG"]
    else:
        tokenizer = None
        sequences = ["MKTAYIAKQ", "GGGG"]

    embeddings = model.embed_dataset(
        sequences=sequences,
        tokenizer=tokenizer,
        batch_size=2,
        max_len=32,
        pooling_types=["mean"],
        hidden_state_index=0,
        save=False,
    )

    assert set(embeddings) == set(sequences)
    for embedding in embeddings.values():
        assert embedding.shape == (model.config.hidden_size,)
        assert not torch.isnan(embedding).any()

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
def test_full_batch_single_match(model_key: str, disable_tf32_for_batch_single_match) -> None:
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

    with strict_fp32_matmul():
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
