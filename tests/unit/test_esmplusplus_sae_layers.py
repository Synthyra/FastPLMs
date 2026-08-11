"""Contracts for the FastPLMs hidden-state SAE implementation and its checkpoint loader.

FastPLMs implements the Biohub hidden-state SAE contract itself, so ESM++ users need no Biohub
runtime code. These tests hold that implementation to exact agreement with the pinned Biohub layer
source, and hold the loader to the published repository layout, including its fail-closed edges.
"""

from __future__ import annotations

import json
import pytest
import torch
import torch.nn as nn

from pathlib import Path
from safetensors.torch import save_file
from types import SimpleNamespace

from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusModel
from fastplms.models.esm_plusplus.modeling_esm_plusplus_sae import (
    ESMplusplusSAELayer,
    ESMplusplusSAEParams,
    load_esmc_sae_layers,
)

from .test_esmplusplus_sae import _config, _input_ids, _load_pinned_biohub_sae_layer


D_MODEL = 4
CODEBOOK_DIM = 6
TOP_K = 2


def _params(layer: int = 0) -> ESMplusplusSAEParams:
    return ESMplusplusSAEParams(d_model=D_MODEL, codebook_dim=CODEBOOK_DIM, k=TOP_K, layer=layer)


def _sae_state(seed: int = 11, d_model: int = D_MODEL) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "W_enc": torch.randn((d_model, CODEBOOK_DIM), generator=generator),
        "W_dec": torch.randn((CODEBOOK_DIM, d_model), generator=generator),
        "b_dec": torch.randn(d_model, generator=generator),
        "idf": torch.rand(CODEBOOK_DIM, generator=generator) + 1.0,
        "max": torch.rand(CODEBOOK_DIM, generator=generator) + 1.0,
    }


def _trained_layer(layer: int = 0, seed: int = 11) -> ESMplusplusSAELayer:
    sae_layer = ESMplusplusSAELayer(_params(layer))
    sae_layer.load_state_dict(_sae_state(seed))
    return sae_layer.eval()


def _hidden_states(seed: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    # layer_states: (2, 5, d) for d = D_MODEL; token_mask: (2, 5)
    layer_states = torch.randn((2, 5, D_MODEL), generator=generator)
    token_mask = torch.tensor(((True, True, True, True, False), (True, True, True, False, False)))
    return layer_states, token_mask


def _write_repository(
    directory: Path,
    *,
    layers: tuple[int, ...] = (0, 1),
    state: dict[str, torch.Tensor] | None = None,
    config: dict[str, object] | None = None,
    shard_layers: tuple[int, ...] | None = None,
    d_model: int = D_MODEL,
) -> Path:
    """Write a local copy of the published SAE repository layout."""

    directory.mkdir(parents=True, exist_ok=True)
    declared = {
        "model_type": "esmc_sae",
        "d_model": d_model,
        "codebook_dim": CODEBOOK_DIM,
        "k": TOP_K,
        "available_layers": list(layers),
    }
    (directory / "config.json").write_text(
        json.dumps(declared if config is None else config),
        encoding="utf-8",
    )
    for layer in layers if shard_layers is None else shard_layers:
        save_file(
            dict(_sae_state(d_model=d_model) if state is None else state),
            str(directory / f"layer_{layer}.safetensors"),
        )
    return directory


def _official_layer(layer: int = 0, seed: int = 11) -> nn.Module:
    official_class = _load_pinned_biohub_sae_layer()
    official = official_class(
        SimpleNamespace(d_model=D_MODEL, codebook_dim=CODEBOOK_DIM, k=TOP_K, layer=layer)
    )
    official.load_state_dict(_sae_state(seed))
    return official.eval()


def test_ported_layer_features_match_the_pinned_biohub_layer_exactly() -> None:
    layer_states, token_mask = _hidden_states()

    with torch.inference_mode():
        expected = _official_layer().get_sae_output(layer_states.clone(), token_mask)
        actual = _trained_layer().get_sae_output(layer_states, token_mask)

    assert tuple(actual.feature_magnitudes.shape) == (int(token_mask.sum()), CODEBOOK_DIM)
    assert torch.count_nonzero(actual.feature_magnitudes, dim=-1).max().item() <= TOP_K
    torch.testing.assert_close(
        actual.feature_magnitudes,
        expected.feature_magnitudes,
        rtol=0.0,
        atol=0.0,
    )


def test_reconstruction_loss_is_opt_in_and_matches_the_pinned_biohub_layer() -> None:
    layer_states, token_mask = _hidden_states()
    residues = layer_states[token_mask]

    with torch.inference_mode():
        expected = _official_layer()(residues.clone())
        default = _trained_layer()(residues)
        requested = _trained_layer()(residues, compute_reconstruction_loss=True)

    assert default.reconstruction_loss is None
    assert requested.reconstruction_loss is not None
    torch.testing.assert_close(
        requested.reconstruction_loss,
        expected.reconstruction_loss,
        rtol=0.0,
        atol=0.0,
    )


def test_layer_reports_its_backbone_index_and_codebook_shapes() -> None:
    sae_layer = _trained_layer(layer=3)

    assert sae_layer.layer == 3
    assert tuple(sae_layer.W_enc.shape) == (D_MODEL, CODEBOOK_DIM)
    assert tuple(sae_layer.W_dec.shape) == (CODEBOOK_DIM, D_MODEL)
    assert tuple(sae_layer.b_dec.shape) == (D_MODEL,)


def test_loader_reads_only_the_requested_layers_and_preserves_values(tmp_path: Path) -> None:
    repository = _write_repository(tmp_path / "sae", layers=(0, 1))
    state = _sae_state()

    sae_layers = load_esmc_sae_layers(repository, [1])

    assert set(sae_layers) == {1}
    loaded = sae_layers[1]
    assert loaded.params == _params(layer=1)
    assert loaded.layer == 1
    for name, expected in state.items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0.0, atol=0.0)


def test_loader_keeps_the_shard_dtype_by_default_and_casts_on_request(tmp_path: Path) -> None:
    state = {name: value.to(torch.bfloat16) for name, value in _sae_state().items()}
    repository = _write_repository(tmp_path / "sae", layers=(0,), state=state)

    stored = load_esmc_sae_layers(repository, [0])[0]
    cast = load_esmc_sae_layers(repository, [0], dtype=torch.float32)[0]

    assert stored.W_enc.dtype == torch.bfloat16
    assert stored.idf.dtype == torch.bfloat16
    assert cast.W_enc.dtype == torch.float32
    torch.testing.assert_close(cast.W_enc, state["W_enc"].to(torch.float32), rtol=0.0, atol=0.0)


def test_loader_defaults_absent_statistics_to_identity_normalization(tmp_path: Path) -> None:
    state = _sae_state()
    untrained = {name: state[name] for name in ("W_enc", "W_dec", "b_dec")}
    repository = _write_repository(tmp_path / "sae", layers=(0,), state=untrained)

    loaded = load_esmc_sae_layers(repository, [0])[0]

    torch.testing.assert_close(loaded.idf, torch.ones(CODEBOOK_DIM), rtol=0.0, atol=0.0)
    torch.testing.assert_close(loaded.max, torch.ones(CODEBOOK_DIM), rtol=0.0, atol=0.0)


def test_loader_rejects_a_shard_that_breaks_the_state_contract(tmp_path: Path) -> None:
    state = _sae_state()
    without_decoder = {name: value for name, value in state.items() if name != "W_dec"}
    extra = {**state, "W_extra": state["b_dec"].clone()}
    missing_repository = _write_repository(tmp_path / "missing", layers=(0,), state=without_decoder)
    extra_repository = _write_repository(tmp_path / "extra", layers=(0,), state=extra)

    with pytest.raises(ValueError, match="missing \\['W_dec'\\]"):
        load_esmc_sae_layers(missing_repository, [0])
    with pytest.raises(ValueError, match="unexpected \\['W_extra'\\]"):
        load_esmc_sae_layers(extra_repository, [0])


def test_loader_rejects_a_shard_without_an_encoder(tmp_path: Path) -> None:
    state = _sae_state()
    repository = _write_repository(
        tmp_path / "sae",
        layers=(0,),
        state={"W_dec": state["W_dec"], "b_dec": state["b_dec"]},
    )

    with pytest.raises(ValueError, match="no 'W_enc' entry"):
        load_esmc_sae_layers(repository, [0])


def test_loader_rejects_a_layer_the_repository_does_not_publish(tmp_path: Path) -> None:
    repository = _write_repository(tmp_path / "sae", layers=(0, 1))

    with pytest.raises(ValueError, match="does not publish layer 7"):
        load_esmc_sae_layers(repository, [7])


def test_loader_reports_a_declared_layer_whose_shard_is_absent(tmp_path: Path) -> None:
    repository = _write_repository(tmp_path / "sae", layers=(0, 1), shard_layers=(0,))

    with pytest.raises(FileNotFoundError, match=r"layer_1\.safetensors"):
        load_esmc_sae_layers(repository, [1])


def test_loader_rejects_a_config_without_the_sae_shape(tmp_path: Path) -> None:
    repository = _write_repository(
        tmp_path / "sae",
        layers=(0,),
        config={"model_type": "esmc_sae", "d_model": D_MODEL},
    )

    with pytest.raises(ValueError, match=r"omits \['codebook_dim', 'k'\]"):
        load_esmc_sae_layers(repository, [0])


def test_loader_requires_at_least_one_layer(tmp_path: Path) -> None:
    repository = _write_repository(tmp_path / "sae", layers=(0,))

    with pytest.raises(ValueError, match="at least one backbone layer"):
        load_esmc_sae_layers(repository, [])


def test_load_sae_models_attaches_layers_in_the_model_dtype_and_device(tmp_path: Path) -> None:
    state = {name: value.to(torch.bfloat16) for name, value in _sae_state().items()}
    repository = _write_repository(tmp_path / "sae", layers=(0, 1), state=state)
    torch.manual_seed(7)
    model = ESMplusplusModel(_config()).eval()
    input_ids = _input_ids()
    token_mask = input_ids.ne(model.config.pad_token_id)

    sae_layers = model.load_sae_models(repository, [1])

    assert set(sae_layers) == {1}
    assert sae_layers[1].W_enc.dtype == model.dtype
    assert sae_layers[1].W_enc.device == model.device
    with torch.inference_mode():
        reference = model(input_ids=input_ids, output_hidden_states=True, compute_sae=False)
        output = model(input_ids=input_ids)
        expected = sae_layers[1].get_sae_output(reference.hidden_states[1], token_mask)

    torch.testing.assert_close(
        output.sae_outputs["layer1"].to_dense(),
        expected.feature_magnitudes,
        rtol=0.0,
        atol=0.0,
    )


def test_load_sae_models_rejects_a_codebook_trained_for_another_scale(tmp_path: Path) -> None:
    repository = _write_repository(tmp_path / "sae", layers=(0,), d_model=D_MODEL * 2)
    torch.manual_seed(7)
    model = ESMplusplusModel(_config()).eval()

    with pytest.raises(ValueError, match="d_model"):
        model.load_sae_models(repository, [0])
