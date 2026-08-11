"""Focused CPU contracts for ESM++ hidden-state sparse autoencoders."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    ESMplusplusForSequenceClassification,
    ESMplusplusForTokenClassification,
    ESMplusplusModel,
)


ROOT = Path(__file__).resolve().parents[2]
PINNED_SAE_SOURCE = (
    ROOT
    / "vendor"
    / "upstream"
    / "biohub-transformers"
    / "src"
    / "transformers"
    / "models"
    / "esmc"
    / "modeling_esmc_sae.py"
)


@dataclass
class _SAEOutput:
    feature_magnitudes: torch.Tensor
    reconstruction_loss: torch.Tensor | None = None


class _SyntheticSAELayer(nn.Module):
    """Small structural equivalent of Biohub's private ``_ESMCSAELayer``."""

    def __init__(
        self,
        *,
        layer: int,
        d_model: int = 4,
        codebook_dim: int = 6,
        k: int = 2,
    ) -> None:
        super().__init__()
        self.params = SimpleNamespace(
            layer=layer,
            d_model=d_model,
            codebook_dim=codebook_dim,
            k=k,
        )
        self.W_enc = nn.Parameter(torch.empty(d_model, codebook_dim))
        self.W_dec = nn.Parameter(torch.zeros(codebook_dim, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("idf", torch.arange(2, codebook_dim + 2, dtype=torch.float32))
        self.register_buffer(
            "max",
            torch.tensor([1.0, 2.0, 4.0, 5.0, 3.0, 7.0])[:codebook_dim],
        )
        self.call_count = 0
        with torch.no_grad():
            self.W_enc.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, -1.0, 0.5, 2.0, -2.0],
                        [0.0, 1.0, 0.5, -1.0, -0.5, 2.0],
                        [-1.0, 0.5, 1.0, 2.0, 0.0, -0.5],
                        [0.5, -1.0, 2.0, 0.0, 1.0, 0.5],
                    ][:d_model]
                )[:, :codebook_dim]
            )

    @property
    def layer(self) -> int:
        return int(self.params.layer)

    def forward(self, x: torch.Tensor, **_kwargs: object) -> _SAEOutput:
        self.call_count += 1
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-5)
        preactivations = F.relu((x - self.b_dec) @ self.W_enc)
        topk = torch.topk(preactivations, self.params.k, dim=-1)
        features = torch.zeros_like(preactivations).scatter(-1, topk.indices, topk.values)
        return _SAEOutput(feature_magnitudes=features)

    def get_sae_output(
        self,
        layer_states: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> _SAEOutput:
        hidden_size = layer_states.shape[-1]
        return self(layer_states[token_mask].view(-1, hidden_size))


def _config() -> ESMplusplusConfig:
    return ESMplusplusConfig(
        vocab_size=40,
        hidden_size=4,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_labels=3,
        attn_backend="eager",
        pad_token_id=1,
        mask_token_id=32,
    )


def _model(model_class: type[nn.Module] = ESMplusplusModel) -> nn.Module:
    torch.manual_seed(7)
    return model_class(_config()).eval()


def _input_ids() -> torch.Tensor:
    return torch.tensor(((0, 4, 5, 2, 1), (0, 6, 2, 1, 1)), dtype=torch.long)


def _expected_features(
    sae: _SyntheticSAELayer,
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    normalized: bool = False,
) -> torch.Tensor:
    with torch.no_grad():
        features = sae.get_sae_output(hidden_states, token_mask).feature_magnitudes
        if normalized:
            features = (features / sae.max) * sae.idf
    return features


def test_sae_outputs_are_exact_detached_sparse_features_without_padding() -> None:
    model = _model()
    sae = _SyntheticSAELayer(layer=0)
    model.add_sae_models([sae])
    input_ids = _input_ids()
    token_mask = input_ids.ne(model.config.pad_token_id)
    expected = _expected_features(sae, model.embed(input_ids), token_mask)

    output = model(input_ids=input_ids)

    assert output.hidden_states is None
    assert output.sae_outputs is not None
    assert set(output.sae_outputs) == {"layer0"}
    actual = output.sae_outputs["layer0"]
    assert actual.layout == torch.sparse_coo
    assert not actual.requires_grad
    assert tuple(actual.shape) == (int(token_mask.sum()), sae.params.codebook_dim)
    torch.testing.assert_close(actual.to_dense(), expected, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(actual.to_dense(), dim=-1).max().item() <= sae.params.k


def test_sae_normalization_matches_biohub_idf_over_max_order() -> None:
    model = _model()
    sae = _SyntheticSAELayer(layer=0)
    model.add_sae_models([sae])
    input_ids = _input_ids()
    token_mask = input_ids.ne(model.config.pad_token_id)
    expected = _expected_features(
        sae,
        model.embed(input_ids),
        token_mask,
        normalized=True,
    )

    output = model(input_ids=input_ids, normalize_sae=True)

    torch.testing.assert_close(
        output.sae_outputs["layer0"].to_dense(),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_sae_selectively_collects_pre_block_and_final_normalized_states() -> None:
    model = _model()
    sae0 = _SyntheticSAELayer(layer=0)
    sae2 = _SyntheticSAELayer(layer=2)
    model.add_sae_models([sae2, sae0])
    input_ids = _input_ids()
    token_mask = input_ids.ne(model.config.pad_token_id)

    with torch.inference_mode():
        reference = model(input_ids=input_ids, output_hidden_states=True, compute_sae=False)
        actual = model(input_ids=input_ids)

    assert reference.sae_outputs is None
    assert reference.hidden_states is not None
    assert len(reference.hidden_states) == model.config.num_hidden_layers + 1
    assert actual.hidden_states is None
    assert set(actual.sae_outputs) == {"layer0", "layer2"}
    torch.testing.assert_close(
        actual.sae_outputs["layer0"].to_dense(),
        _expected_features(sae0, reference.hidden_states[0], token_mask),
    )
    torch.testing.assert_close(
        actual.sae_outputs["layer2"].to_dense(),
        _expected_features(sae2, reference.hidden_states[2], token_mask),
    )


@pytest.mark.parametrize(
    "model_class",
    (
        ESMplusplusModel,
        ESMplusplusForMaskedLM,
        ESMplusplusForSequenceClassification,
        ESMplusplusForTokenClassification,
    ),
)
def test_all_public_model_heads_propagate_sae_outputs(model_class: type[nn.Module]) -> None:
    model = _model(model_class)
    model.add_sae_models([_SyntheticSAELayer(layer=1)])

    with torch.inference_mode():
        output = model(input_ids=_input_ids())

    assert output.sae_outputs is not None
    assert set(output.sae_outputs) == {"layer1"}
    assert output.sae_outputs["layer1"].layout == torch.sparse_coo


@pytest.mark.parametrize(
    "model_class",
    (ESMplusplusForSequenceClassification, ESMplusplusForTokenClassification),
)
def test_classifier_tuple_outputs_keep_existing_prefix_and_append_sae(
    model_class: type[nn.Module],
) -> None:
    model = _model(model_class)
    model.add_sae_models([_SyntheticSAELayer(layer=1)])
    input_ids = _input_ids()

    with torch.inference_mode():
        baseline = model(input_ids=input_ids, compute_sae=False, return_dict=False)
        with_sae = model(input_ids=input_ids, return_dict=False)

    assert len(with_sae) == len(baseline) + 1
    for baseline_value, sae_value in zip(baseline, with_sae[:-1], strict=True):
        torch.testing.assert_close(sae_value, baseline_value, rtol=0.0, atol=0.0)
    assert isinstance(with_sae[-1], dict)
    assert set(with_sae[-1]) == {"layer1"}


def test_compute_sae_false_is_a_zero_work_bypass() -> None:
    model = _model()
    sae = _SyntheticSAELayer(layer=0)
    model.add_sae_models([sae])
    input_ids = _input_ids()
    input_ids[0, 1] = model.config.mask_token_id

    with torch.inference_mode():
        from_ids = model(input_ids=input_ids, compute_sae=False)
        from_embeds = model(inputs_embeds=model.embed(input_ids), compute_sae=False)

    assert sae.call_count == 0
    assert from_ids.sae_outputs is None
    assert from_embeds.sae_outputs is None


def test_sae_uses_sequence_id_instead_of_attention_mask_for_valid_tokens() -> None:
    model = _model()
    sae = _SyntheticSAELayer(layer=0)
    model.add_sae_models([sae])
    input_ids = torch.tensor(((0, 4, 5, 2),), dtype=torch.long)
    sequence_id = torch.tensor(((0, 0, -1, -1),), dtype=torch.long)

    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        sequence_id=sequence_id,
    )

    actual = output.sae_outputs["layer0"]
    expected = _expected_features(sae, model.embed(input_ids), sequence_id.ne(-1))
    assert tuple(actual.shape) == (2, sae.params.codebook_dim)
    torch.testing.assert_close(actual.to_dense(), expected, rtol=0.0, atol=0.0)


def test_sae_registration_validates_layer_dimension_and_uniqueness() -> None:
    model = _model()

    with pytest.raises(ValueError, match="hidden|d_model"):
        model.add_sae_models([_SyntheticSAELayer(layer=0, d_model=3)])
    with pytest.raises(ValueError, match="layer"):
        model.add_sae_models([_SyntheticSAELayer(layer=3)])

    model.add_sae_models([_SyntheticSAELayer(layer=1)])
    with pytest.raises(ValueError, match="already|duplicate|one SAE"):
        model.add_sae_models([_SyntheticSAELayer(layer=1)])


def test_sae_computation_rejects_masked_tokens_and_unmasked_embedding_inputs() -> None:
    model = _model()
    model.add_sae_models([_SyntheticSAELayer(layer=0)])
    masked_ids = _input_ids()
    masked_ids[0, 1] = model.config.mask_token_id

    with pytest.raises((ValueError, AssertionError), match="mask"):
        model(input_ids=masked_ids)
    # Embedding inputs are supported, but only with an explicit token mask, since embeddings carry
    # no token identities to derive one from. Rejecting mask tokens then becomes the caller's
    # precondition. See tests/unit/test_esmplusplus_sae_differentiable.py for the working path.
    with pytest.raises(ValueError, match="explicit attention_mask or sequence_id"):
        model(inputs_embeds=model.embed(_input_ids()))


def _load_pinned_biohub_sae_layer() -> type[nn.Module]:
    """Execute only Biohub's pinned SAE layer class, without importing vendor code."""

    source = PINNED_SAE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PINNED_SAE_SOURCE))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_ESMCSAELayer"
    )
    isolated = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias("annotations")],
                level=0,
            ),
            class_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "ESMCSAEOutput": _SAEOutput,
    }
    exec(compile(isolated, str(PINNED_SAE_SOURCE), "exec"), namespace)
    return namespace["_ESMCSAELayer"]


def test_attached_features_match_the_pinned_biohub_sae_layer_source() -> None:
    official_layer_class = _load_pinned_biohub_sae_layer()
    params = SimpleNamespace(d_model=4, codebook_dim=6, k=2, layer=0)
    official_sae = official_layer_class(params)
    synthetic = _SyntheticSAELayer(layer=0)
    official_sae.load_state_dict(synthetic.state_dict())
    model = _model()
    model.add_sae_models([official_sae])
    input_ids = _input_ids()
    token_mask = input_ids.ne(model.config.pad_token_id)

    with torch.inference_mode():
        expected = official_sae.get_sae_output(
            model.embed(input_ids),
            token_mask,
        ).feature_magnitudes
        expected_normalized = (expected / official_sae.max) * official_sae.idf
        output = model(input_ids=input_ids)
        normalized_output = model(input_ids=input_ids, normalize_sae=True)

    torch.testing.assert_close(
        output.sae_outputs["layer0"].to_dense(),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        normalized_output.sae_outputs["layer0"].to_dense(),
        expected_normalized,
        rtol=0.0,
        atol=0.0,
    )
