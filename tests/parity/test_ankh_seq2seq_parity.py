"""Official ANKH sequence-to-sequence head and alias compliance."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM

from fastplms.models.ankh.modeling_ankh import tokenize_ankh_sequences
from fastplms.registry import ModelSpec, get_model_registry
from tests.parity.support.reference_adapters.ankh import load_official_seq2seq

pytestmark = [pytest.mark.compliance, pytest.mark.gpu, pytest.mark.slow]
ANKH_SPECS = get_model_registry().by_family("ankh")


def _parameter(spec: ModelSpec) -> Any:
    marks: list[Any] = []
    if spec.size_category == "xlarge":
        marks.append(pytest.mark.large)
    return pytest.param(spec, id=spec.id, marks=marks)


def _alias_groups(model: torch.nn.Module) -> set[frozenset[str]]:
    groups: dict[int, set[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), set()).add(name)
    return {frozenset(names) for names in groups.values() if len(names) > 1}


@pytest.mark.parametrize("spec", [_parameter(spec) for spec in ANKH_SPECS])
def test_ankh_official_seq2seq_state_aliases_and_seeded_inference(
    spec: ModelSpec,
    tmp_path: Path,
) -> None:
    device = torch.device("cuda")
    fast = AutoModelForSeq2SeqLM.from_pretrained(
        spec.fast.repo_id,
        revision=spec.fast.revision,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    ).eval()
    official, tokenizer = load_official_seq2seq(
        reference_repo_id=spec.official.repo_id,
        reference_revision=spec.official.revision,
        device=device,
        dtype=torch.bfloat16,
    )

    fast_state = fast.state_dict()
    official_state = official.state_dict()
    assert set(fast_state) == set(official_state)
    for name in sorted(fast_state):
        assert fast_state[name].shape == official_state[name].shape
        assert fast_state[name].dtype == official_state[name].dtype
        assert torch.equal(fast_state[name], official_state[name]), (
            f"{spec.id}:{name}: sequence-to-sequence weight differs"
        )
    assert _alias_groups(fast) == _alias_groups(official), (
        f"{spec.id}: sequence-to-sequence tied-weight contract differs"
    )

    encoded = tokenize_ankh_sequences(
        tokenizer,
        ["MSTNPK", "ACDE"],
        return_tensors="pt",
        padding=True,
    )
    inputs = {name: value.to(device) for name, value in encoded.items() if torch.is_tensor(value)}
    decoder_input_ids = inputs["input_ids"]
    with torch.inference_mode():
        fast_logits = fast(
            **inputs,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        ).logits.float()
        official_logits = official(
            **inputs,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        ).logits.float()
    relative_l2 = torch.linalg.vector_norm(fast_logits - official_logits) / (
        torch.linalg.vector_norm(official_logits).clamp_min(torch.finfo(torch.float32).tiny)
    )
    cosine = F.cosine_similarity(fast_logits, official_logits, dim=-1)
    assert float(relative_l2) <= 1e-2
    assert float(torch.quantile(cosine, 0.01)) >= 0.999

    generation_kwargs = {
        **inputs,
        "do_sample": True,
        "top_k": 5,
        "max_new_tokens": 4,
    }
    torch.manual_seed(42)
    fast_tokens = fast.generate(**generation_kwargs)
    torch.manual_seed(42)
    official_tokens = official.generate(**generation_kwargs)
    assert torch.equal(fast_tokens, official_tokens), (
        f"{spec.id}: seeded sequence-to-sequence generation differs"
    )

    save_path = tmp_path / "seq2seq"
    fast.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)
    del fast
    torch.cuda.empty_cache()
    reloaded = AutoModelForSeq2SeqLM.from_pretrained(
        save_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    ).eval()
    reloaded_state = reloaded.state_dict()
    assert set(reloaded_state) == set(official_state)
    for name, official_tensor in official_state.items():
        assert torch.equal(reloaded_state[name], official_tensor)
    torch.manual_seed(42)
    reloaded_tokens = reloaded.generate(**generation_kwargs)
    assert torch.equal(reloaded_tokens, official_tokens), (
        f"{spec.id}: save/reload changed seeded sequence-to-sequence generation"
    )

    del reloaded, official
    gc.collect()
    torch.cuda.empty_cache()
