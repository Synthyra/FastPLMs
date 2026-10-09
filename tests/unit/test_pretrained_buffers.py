"""Tensors no checkpoint holds must carry their constructed values after ``from_pretrained``.

Transformers 5 builds a model on the meta device and fills only the tensors its checkpoint
holds. A buffer registered with ``persistent=False`` is then materialized uninitialized and
stays that way unless the family rebuilds it: in ``_init_weights``, in ``_apply``, or on the
first forward. Uninitialized memory is usually finite, so a buffer nobody rebuilds fails
quietly. The micro round trips below make it loud by filling uninitialized memory with NaN,
or the largest integer, and every test compares each unsaved tensor of a loaded model with a
directly constructed twin.
"""

from __future__ import annotations

import importlib
import pytest
import torch

from collections.abc import Callable, Iterator
from pathlib import Path
from tests.unit.checkpoint_cache import find_verified_snapshot
from tests.unit.test_structure_output_contracts import _tiny_fast_esmfold_config
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from fastplms.models.boltz import modeling_boltz2
from fastplms.models.boltz.modeling_boltz2 import Boltz2Config, Boltz2Model
from fastplms.models.boltz.vb_modules_diffusionv2 import AtomDiffusion
from fastplms.models.dplm.modeling_dplm import DPLMConfig, DPLMModel
from fastplms.models.dplm2.modeling_dplm2 import DPLM2Config, DPLM2Model
from fastplms.models.e1.modeling_e1 import E1Config, E1ForMaskedLM
from fastplms.models.esm2.modeling_fastesm import FastEsmConfig, FastEsmModel
from fastplms.models.esm3.modeling_esm3 import FastESM3Config, FastESM3Model
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusModel,
)
from fastplms.models.esmfold.modeling_fast_esmfold import FastEsmForProteinFolding
from fastplms.registry import ModelSpec, get_model_registry


Inputs = dict[str, torch.Tensor]
SequenceCase = Callable[[], tuple[type[PreTrainedModel], PretrainedConfig, Inputs]]

TOKEN_IDS = torch.tensor([[0, 5, 6, 7, 2]])  # (b=1, l=5)
PROTEIN = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
# Families whose checkpoints carry rotary state the loader must rebuild, and whose
# AutoModel accepts token IDs from its own tokenizer or batch preparer.
ROTARY_FAMILIES = ("dplm", "dplm2", "e1", "esm2", "esm3", "esm_plusplus")


@pytest.fixture
def uninitialized_memory_is_visible() -> Iterator[None]:
    """Fill uninitialized memory with NaN, or the largest integer, for one test."""

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill


def _unsaved_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every buffer and plain tensor attribute absent from the state dict, by dotted name."""

    saved = set(model.state_dict())
    found: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for name, buffer in module._buffers.items():
            if buffer is not None and prefix + name not in saved:
                found[prefix + name] = buffer
        for name, value in vars(module).items():
            if torch.is_tensor(value) and not isinstance(value, nn.Parameter):
                found[prefix + name] = value
    return found  # (...) one tensor per dotted name, shapes as registered


def _assert_unsaved_tensors_match(loaded: nn.Module, direct: nn.Module) -> None:
    loaded_tensors = _unsaved_tensors(loaded)
    direct_tensors = _unsaved_tensors(direct)
    assert loaded_tensors.keys() == direct_tensors.keys()
    for name, expected in direct_tensors.items():
        actual = loaded_tensors[name]
        assert not actual.is_meta, name
        assert (actual.shape, actual.dtype) == (expected.shape, expected.dtype), name
        assert torch.equal(actual, expected), f"{name} differs from direct construction"


def _hidden_states(output: object) -> torch.Tensor:
    hidden_states = getattr(output, "last_hidden_state", None)  # (b, l, d)
    assert isinstance(hidden_states, torch.Tensor)
    return hidden_states  # (b, l, d)


def _esm2_case() -> tuple[type[PreTrainedModel], PretrainedConfig, Inputs]:
    config = FastEsmConfig(
        vocab_size=33,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=64,
        pad_token_id=1,
        mask_token_id=32,
        position_embedding_type="rotary",
        add_pooling_layer=False,
        attn_backend="eager",
        use_cache=False,
        token_dropout=False,
    )
    return FastEsmModel, config, {"input_ids": TOKEN_IDS}


def _dplm_case() -> tuple[type[PreTrainedModel], PretrainedConfig, Inputs]:
    config = DPLMConfig(
        vocab_size=33,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=32,
        position_embedding_type="rotary",
        add_pooling_layer=False,
        attn_backend="eager",
        use_cache=False,
    )
    return DPLMModel, config, {"input_ids": TOKEN_IDS}


def _dplm2_case() -> tuple[type[PreTrainedModel], PretrainedConfig, Inputs]:
    config = DPLM2Config(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=32,
        position_embedding_type="rotary",
        add_pooling_layer=False,
        attn_backend="sdpa",
        use_cache=False,
    )
    return DPLM2Model, config, {"input_ids": TOKEN_IDS}


def _esm_plusplus_case() -> tuple[type[PreTrainedModel], PretrainedConfig, Inputs]:
    config = ESMplusplusConfig(
        vocab_size=64,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        dropout=0.0,
        pad_token_id=1,
        mask_token_id=32,
        attn_backend="eager",
    )
    return ESMplusplusModel, config, {"input_ids": TOKEN_IDS}


def _esm3_case() -> tuple[type[PreTrainedModel], PretrainedConfig, Inputs]:
    config = FastESM3Config(
        hidden_size=16,
        num_attention_heads=2,
        num_vector_heads=4,
        num_hidden_layers=1,
        attn_backend="sdpa",
    )
    inputs = {"input_ids": TOKEN_IDS, "attention_mask": torch.ones_like(TOKEN_IDS)}
    return FastESM3Model, config, inputs


def _e1_case() -> tuple[type[PreTrainedModel], PretrainedConfig, Inputs]:
    config = E1Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_num_sequences=4,
        max_num_positions_within_seq=16,
        max_num_positions_global=16,
        attn_backend="sdpa",
    )
    input_ids = torch.tensor([[1, 4, 5, 6, 2]])  # (b=1, l=5)
    positions = torch.arange(5).unsqueeze(0)  # (b=1, l=5)
    inputs = {
        "input_ids": input_ids,
        "within_seq_position_ids": positions,
        "global_position_ids": positions,
        "sequence_ids": torch.zeros_like(input_ids),  # (b=1, l=5)
    }
    return E1ForMaskedLM, config, inputs


SEQUENCE_CASES = [
    pytest.param(_esm2_case, id="esm2"),
    pytest.param(_dplm_case, id="dplm"),
    pytest.param(_dplm2_case, id="dplm2"),
    pytest.param(_esm_plusplus_case, id="esm_plusplus"),
    pytest.param(_esm3_case, id="esm3"),
    pytest.param(_e1_case, id="e1"),
]


@pytest.mark.parametrize("case", SEQUENCE_CASES)
def test_reloaded_sequence_model_rebuilds_every_unsaved_tensor(
    case: SequenceCase,
    tmp_path: Path,
    uninitialized_memory_is_visible: None,
) -> None:
    model_class, config, inputs = case()
    direct = model_class(config).eval()
    direct.save_pretrained(tmp_path, safe_serialization=True)
    loaded = model_class.from_pretrained(tmp_path, local_files_only=True).eval()

    # A lazily built table only exists after a forward, so compare after one on each.
    with torch.inference_mode():
        direct_hidden = _hidden_states(direct(**inputs))  # (b=1, l=5, d=16)
        loaded_hidden = _hidden_states(loaded(**inputs))  # (b=1, l=5, d=16)

    assert torch.isfinite(loaded_hidden).all()
    assert torch.equal(loaded_hidden, direct_hidden)
    _assert_unsaved_tensors_match(loaded, direct)


def test_reloaded_folding_model_rebuilds_every_unsaved_tensor(
    tmp_path: Path,
    uninitialized_memory_is_visible: None,
) -> None:
    config = _tiny_fast_esmfold_config(bypass_lm=False)
    direct = FastEsmForProteinFolding(config).eval()
    direct.save_pretrained(tmp_path, safe_serialization=True)
    loaded = FastEsmForProteinFolding.from_pretrained(tmp_path, local_files_only=True).eval()

    _assert_unsaved_tensors_match(loaded, direct)


class _DiffusionCore(nn.Module):
    """The smallest Boltz2 core that holds the real diffusion module and its unsaved zero."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_s": width,
                "atom_s": width,
                "atoms_per_window_queries": 4,
                "atoms_per_window_keys": 8,
                "dim_fourier": width,
                "atom_encoder_depth": 1,
                "atom_encoder_heads": 1,
                "token_transformer_depth": 1,
                "token_transformer_heads": 1,
                "atom_decoder_depth": 1,
                "atom_decoder_heads": 1,
                "conditioning_transition_layers": 1,
            },
        )


def test_reloaded_boltz2_rebuilds_the_diffusion_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uninitialized_memory_is_visible: None,
) -> None:
    monkeypatch.setattr(modeling_boltz2, "Boltz2InferenceCore", _DiffusionCore)
    config = Boltz2Config(core_kwargs={"width": 8})
    direct = Boltz2Model(config).eval()
    direct.save_pretrained(tmp_path)
    loaded = Boltz2Model.from_pretrained(tmp_path, local_files_only=True).eval()

    assert loaded.core.structure_module.zero.item() == 0.0
    _assert_unsaved_tensors_match(loaded, direct)


def _verified_cached_snapshot(spec: ModelSpec) -> Path:
    """A cached snapshot whose files match the manifest's pinned digests, else a skip."""

    snapshot = find_verified_snapshot(spec)
    if snapshot is None:
        pytest.skip(
            f"No cached snapshot of {spec.fast.repo_id} matches the files pinned at "
            f"{spec.fast.revision}."
        )
    return snapshot


def _auto_model_class(spec: ModelSpec) -> type[PreTrainedModel]:
    module_name, _, class_name = spec.auto_map["AutoModel"].rpartition(".")
    model_class = getattr(importlib.import_module(module_name), class_name)
    assert issubclass(model_class, PreTrainedModel)
    return model_class


def _protein_inputs(model: PreTrainedModel, family: str) -> Inputs:
    if family == "e1":
        batch = model.prep_tokens.get_batch_kwargs([PROTEIN], device=torch.device("cpu"))
        names = ("input_ids", "within_seq_position_ids", "global_position_ids", "sequence_ids")
        return {name: batch[name] for name in names}  # each (b=1, l)
    # Tokenize as the embedding API does: DPLM2 names no generic boundary tokens, so its
    # `_tokenize_sequence_batch` adds the amino-acid track's own.
    sequence_tokenizer = getattr(model, "_tokenize_sequence_batch", None)
    if callable(sequence_tokenizer):
        batch = sequence_tokenizer([PROTEIN], return_tensors="pt")
    else:
        batch = model.tokenizer([PROTEIN], return_tensors="pt")
    return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}  # each (b=1, l)


CACHED_ROTARY_MODELS = [
    pytest.param(spec.id, id=spec.id)
    for spec in get_model_registry().values()
    if spec.family.id in ROTARY_FAMILIES
]


@pytest.mark.checkpoint
@pytest.mark.parametrize("model_id", CACHED_ROTARY_MODELS)
def test_cached_checkpoint_loads_with_constructed_unsaved_tensors(model_id: str) -> None:
    """A real checkpoint, loaded offline on CPU, runs finite and holds constructed buffers."""

    spec = get_model_registry()[model_id]
    snapshot = _verified_cached_snapshot(spec)
    model_class = _auto_model_class(spec)
    loaded = model_class.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.float32,
    ).eval()
    # The twin holds the checkpoint's saved tensors, so a tensor derived from one compares like
    # with like: official ESM2 checkpoints store `inv_freq` rounded through float16, and the
    # rotary tables built from it differ from ones built from the formula.
    direct = model_class(loaded.config).eval()
    direct.load_state_dict(loaded.state_dict())

    inputs = _protein_inputs(loaded, spec.family.id)
    with torch.inference_mode():
        loaded_hidden = _hidden_states(loaded(**inputs))  # (b=1, l, d)
        direct(**inputs)

    assert torch.isfinite(loaded_hidden).all()
    _assert_unsaved_tensors_match(loaded, direct)
