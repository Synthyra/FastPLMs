import entrypoint_setup

import random
import tempfile
import os
import torch

from esm2.modeling_fastesm import FastEsmForMaskedLM
from esm_plusplus.modeling_esm_plusplus import ESMplusplusForMaskedLM
from e1_fastplms.modeling_e1 import E1ForMaskedLM
from dplm_fastplms.modeling_dplm import DPLMForMaskedLM
from dplm2_fastplms.modeling_dplm2 import (
    DPLM2ForMaskedLM,
    _has_packed_multimodal_layout,
    _normalize_dplm2_input_ids,
)
from embedding_mixin import parse_fasta


CANONICAL_AAS = "ACDEFGHIKLMNPQRSTVWY"
SEED = 42
DEFAULT_BATCH_SIZE = 4
MAX_EMBED_LEN = 128  # fixed pad length used to keep max_seqlen identical across runs


# (display_name, model_class, hf_path, use_model_tokenizer)
MODEL_CONFIGS = [
    ("ESM2",  FastEsmForMaskedLM,       "Synthyra/ESM2-8M",           True),
    ("ESM++", ESMplusplusForMaskedLM,   "Synthyra/ESMplusplus_small",  True),
    ("E1",    E1ForMaskedLM,            "Synthyra/Profluent-E1-150M",  False),
    ("DPLM",  DPLMForMaskedLM,          "Synthyra/DPLM-150M",          True),
    ("DPLM2", DPLM2ForMaskedLM,         "Synthyra/DPLM2-150M",         True),
]


def test_parse_fasta() -> None:
    """Test parse_fasta with single-line and multi-line sequences."""
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(fasta_content)
        tmp_path = f.name
    parsed = parse_fasta(tmp_path)
    os.unlink(tmp_path)
    assert parsed == expected, f"parse_fasta mismatch:\n  got:      {parsed}\n  expected: {expected}"
    print("test_parse_fasta: OK")


class FixedLengthTokenizer:
    """Wraps a tokenizer so every call pads to exactly MAX_EMBED_LEN tokens.

    Both batch=1 and batch=N therefore receive tensors of the same shape,
    keeping max_seqlen_in_batch identical and eliminating floating-point
    variability from different softmax vector lengths / flash-attention tile sizes.
    """
    def __init__(self, tokenizer, max_length: int = MAX_EMBED_LEN):
        self._tok = tokenizer
        self.max_length = max_length

    def __call__(self, sequences, **kwargs):
        return self._tok(
            sequences,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )


def random_sequences(n: int, min_len: int = 8, max_len: int = 64) -> list[str]:
    """Variable-length sequences; used for the NaN test."""
    return [
        "M" + "".join(random.choices(CANONICAL_AAS, k=random.randint(min_len, max_len)))
        for _ in range(n)
    ]


def random_sequences_fixed_len(n: int, length: int = 64) -> list[str]:
    """Fixed-length sequences; used for the match test with E1 (sequence mode)."""
    return [
        "M" + "".join(random.choices(CANONICAL_AAS, k=length - 1))
        for _ in range(n)
    ]


def assert_no_nan(embeddings: dict[str, torch.Tensor], label: str) -> None:
    for seq, emb in embeddings.items():
        assert not torch.isnan(emb).any(), (
            f"[{label}] NaN found in embedding for sequence '{seq[:20]}...'"
        )


def assert_embeddings_match(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    label: str,
    atol: float = 5e-3,
) -> None:
    """Compare real-token embeddings from two runs.

    full_embeddings=True already strips padding via emb[mask.bool()], so both
    dicts contain only non-pad token rows and the comparison is over those rows.
    """
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


def test_dplm2_multimodal_layout_guard() -> None:
    plain_sequence_type_ids = torch.tensor([
        [1, 1, 1, 1, 1, 1, 0, 2],
        [1, 1, 1, 1, 1, 0, 2, 2],
    ])
    packed_multimodal_type_ids = torch.tensor([
        [1, 1, 1, 2, 0, 0, 0, 2],
        [1, 1, 2, 2, 0, 0, 2, 2],
    ])
    mismatched_multimodal_type_ids = torch.tensor([
        [1, 1, 1, 2, 0, 0, 2, 2],
    ])

    assert not _has_packed_multimodal_layout(plain_sequence_type_ids, aa_type=1, struct_type=0, pad_type=2)
    assert _has_packed_multimodal_layout(packed_multimodal_type_ids, aa_type=1, struct_type=0, pad_type=2)
    assert not _has_packed_multimodal_layout(mismatched_multimodal_type_ids, aa_type=1, struct_type=0, pad_type=2)
    print("test_dplm2_multimodal_layout_guard: OK")


def test_dplm2_special_token_normalization() -> None:
    input_ids = torch.tensor([[8231, 5, 23, 13, 8229, 1, 8232, -100]])
    normalized_input_ids = _normalize_dplm2_input_ids(input_ids, vocab_size=8229)
    expected = torch.tensor([[0, 5, 23, 13, 2, 1, 32, -100]])
    assert torch.equal(normalized_input_ids, expected), (
        f"DPLM2 special-token normalization mismatch:\n"
        f"  got:      {normalized_input_ids.tolist()}\n"
        f"  expected: {expected.tolist()}"
    )
    print("test_dplm2_special_token_normalization: OK")


def test_model(name: str, model_cls, model_path: str, use_model_tokenizer: bool, batch_size: int) -> None:
    print(f"\n--- {name} ({model_path}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    ).eval()

    if use_model_tokenizer:
        # FixedLengthTokenizer pads every batch to MAX_EMBED_LEN regardless of
        # actual sequence lengths, so batch=1 and batch=N see the same tensor
        # shape and produce numerically identical real-token outputs.
        tokenizer = FixedLengthTokenizer(model.tokenizer)
        sequences = random_sequences(n=8)          # variable lengths, all padded to MAX_EMBED_LEN
    else:
        # E1 (sequence mode): control padding length via fixed-length sequences
        # so max_seqlen_in_batch is the same in every forward call.
        tokenizer = None
        sequences = random_sequences_fixed_len(n=8)  # fixed length, no padding variability

    nan_kwargs = dict(
        tokenizer=tokenizer,
        full_embeddings=True,  # extracts only real (non-pad) token rows via emb[mask.bool()]
        embed_dtype=torch.bfloat16,
        save=False,
    )

    # NaN test ----------------------------------------------------------------
    # Run in bfloat16 to match the real-world user scenario.
    # batch_size > 1 with padding present must produce no NaN in real-token rows.
    nan_embs = model.embed_dataset(sequences=sequences, batch_size=batch_size, **nan_kwargs)
    assert_no_nan(nan_embs, f"{name} NaN check batch_size={batch_size}")
    shapes = [tuple(e.shape) for e in list(nan_embs.values())[:3]]
    print(f"  NaN check batch_size={batch_size}: OK  sample shapes={shapes}")

    # Match test (tokenizer / SDPA models only) --------------------------------
    # The NaN fix only touches SDPA backends; E1 uses flash varlen which
    # inherently unpads and is unaffected.  Flash varlen is also NOT
    # bit-deterministic across different batch sizes (different numbers of
    # packed query blocks → different online-softmax accumulation order), so
    # a tight match test for E1 is not meaningful.
    #
    # For SDPA models we cast to float32: bfloat16 CUBLAS selects different
    # mat-mul algorithms for batch=1 vs batch=N (simple vs batched GEMM),
    # producing 1-ULP differences.  Float32 differences are < 1e-3.
    if not use_model_tokenizer:
        return

    model.to(torch.float32)
    batch_embs = model.embed_dataset(
        sequences=sequences, batch_size=batch_size,
        tokenizer=tokenizer, full_embeddings=True, embed_dtype=torch.float32, save=False,
    )
    single_embs = model.embed_dataset(
        sequences=sequences, batch_size=1,
        tokenizer=tokenizer, full_embeddings=True, embed_dtype=torch.float32, save=False,
    )
    assert_no_nan(batch_embs, f"{name} match test batch_size={batch_size}")
    assert_no_nan(single_embs, f"{name} match test batch_size=1")
    assert_embeddings_match(batch_embs, single_embs, name)
    print(f"  Match test batch_size={batch_size} vs 1: OK  (non-pad tokens only)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test embed_dataset produces no NaN with batch_size > 1.")
    parser.add_argument("--models", nargs="+", default=["ESM2", "ESM++", "E1", "DPLM", "DPLM2"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    random.seed(SEED)
    test_parse_fasta()
    test_dplm2_multimodal_layout_guard()
    test_dplm2_special_token_normalization()

    valid_names = {cfg[0] for cfg in MODEL_CONFIGS}
    for name in args.models:
        assert name in valid_names, f"Unknown model '{name}'. Choose from {sorted(valid_names)}"

    configs_by_name = {cfg[0]: cfg for cfg in MODEL_CONFIGS}
    for model_name in args.models:
        name, model_cls, model_path, use_model_tokenizer = configs_by_name[model_name]
        test_model(name, model_cls, model_path, use_model_tokenizer, args.batch_size)

    print("\nAll tests passed!")
