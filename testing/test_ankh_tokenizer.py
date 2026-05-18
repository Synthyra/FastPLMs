"""Ensure Fast ANKH models attach the checkpoint-matched tokenizer (not always ankh-base)."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

from fastplms.ankh.modeling_ankh import FAST_ANKH_ENCODER, FastAnkhForMaskedLM

_CANONICAL_SEQ = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

# (fast hub id, official tokenizer repo)
_ANKH_TOKENIZER_CASES = [
    ("Synthyra/ANKH_base", "ElnaggarLab/ankh-base"),
    ("Synthyra/ANKH3_large", "ElnaggarLab/ankh3-large"),
    ("Synthyra/ANKH3_xl", "ElnaggarLab/ankh3-xl"),
]


@pytest.mark.slow
@pytest.mark.parametrize("fast_path,official_path", _ANKH_TOKENIZER_CASES)
def test_fast_ankh_encoder_tokenizer_matches_official(fast_path: str, official_path: str) -> None:
    """Tokenizer on FAST_ANKH_ENCODER must match the official repo for that checkpoint."""
    config = AutoConfig.from_pretrained(fast_path, trust_remote_code=True)
    encoder = FAST_ANKH_ENCODER(config)
    official_tok = AutoTokenizer.from_pretrained(official_path)
    fast_tok = encoder.tokenizer

    assert len(fast_tok) == len(official_tok), (
        f"vocab size mismatch for {fast_path}: fast={len(fast_tok)} official={len(official_tok)}"
    )
    fast_ids = fast_tok(_CANONICAL_SEQ, return_tensors="pt")["input_ids"]
    off_ids = official_tok(_CANONICAL_SEQ, return_tensors="pt")["input_ids"]
    assert torch.equal(fast_ids, off_ids), (
        f"token ids differ for {fast_path}: fast={fast_ids[0, :8].tolist()} "
        f"official={off_ids[0, :8].tolist()}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("fast_path,official_path", _ANKH_TOKENIZER_CASES)
def test_fast_ankh_masked_lm_tokenizer_matches_official(fast_path: str, official_path: str) -> None:
    """``FastAnkhForMaskedLM`` wrapper exposes the same tokenizer as the encoder."""
    config = AutoConfig.from_pretrained(fast_path, trust_remote_code=True)
    model = FastAnkhForMaskedLM.from_pretrained(
        fast_path,
        config=config,
        dtype=torch.bfloat16,
    )
    official_tok = AutoTokenizer.from_pretrained(official_path)
    fast_ids = model.tokenizer(_CANONICAL_SEQ, return_tensors="pt")["input_ids"]
    off_ids = official_tok(_CANONICAL_SEQ, return_tensors="pt")["input_ids"]
    assert torch.equal(fast_ids, off_ids)
