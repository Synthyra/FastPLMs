"""Safe persistence contracts for the FastPLMs Boltz2 model."""

from __future__ import annotations

import pytest
import torch
from pathlib import Path
from typing import Any
from torch import nn

from fastplms.models.boltz import modeling_boltz2
from fastplms.models.boltz.modeling_boltz2 import Boltz2Config, Boltz2Model


class _TinyCore(nn.Module):
    """Minimal checkpoint-facing core used to exercise model persistence."""

    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))  # (d=width,)


def _install_tiny_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modeling_boltz2, "Boltz2InferenceCore", _TinyCore)


def test_lightning_checkpoint_rejects_pickle_without_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_called = False

    def _unexpected_load(*args: Any, **kwargs: Any) -> None:
        nonlocal load_called
        load_called = True

    monkeypatch.setattr(torch, "load", _unexpected_load)

    with pytest.raises(
        ValueError,
        match=r"allow_unsafe_pickle=True only for a trusted, hash-verified checkpoint",
    ):
        Boltz2Model.from_boltz_checkpoint("untrusted.ckpt")

    assert not load_called


def test_lightning_checkpoint_rejects_truthy_non_boolean_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail("unsafe deserializer was reached"),
    )

    with pytest.raises(ValueError, match=r"allow_unsafe_pickle=True"):
        Boltz2Model.from_boltz_checkpoint(
            "untrusted.ckpt",
            allow_unsafe_pickle=1,  # type: ignore[arg-type]
        )


def test_lightning_checkpoint_load_requires_and_honors_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_core(monkeypatch)
    expected = torch.tensor([1.25, -2.5])  # (d=2,)
    load_call: dict[str, Any] = {}

    def _load(
        path: str,
        *,
        map_location: str | torch.device,
        weights_only: bool,
    ) -> dict[str, Any]:
        load_call.update(
            path=path,
            map_location=map_location,
            weights_only=weights_only,
        )
        return {
            "hyper_parameters": {},
            "state_dict": {"model.weight": expected.clone()},  # (d=2,)
        }

    def _config_from_hyperparameters(
        cls: type[Boltz2Config],
        hparams: dict[str, Any],
        **kwargs: Any,
    ) -> Boltz2Config:
        assert hparams == {}
        assert kwargs["use_kernels"] is False
        return cls(core_kwargs={"width": 2})

    monkeypatch.setattr(torch, "load", _load)
    monkeypatch.setattr(
        Boltz2Config,
        "from_hyperparameters",
        classmethod(_config_from_hyperparameters),
    )

    model = Boltz2Model.from_boltz_checkpoint(
        "trusted.ckpt",
        allow_unsafe_pickle=True,
    )

    assert load_call == {
        "path": "trusted.ckpt",
        "map_location": "cpu",
        "weights_only": False,
    }
    assert torch.equal(model.core.weight, expected)
    assert not model.training


def test_lightning_checkpoint_missing_state_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_core(monkeypatch)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {
            "hyper_parameters": {},
            "state_dict": {},
        },
    )
    monkeypatch.setattr(
        Boltz2Config,
        "from_hyperparameters",
        classmethod(lambda cls, _hparams, **_kwargs: cls(core_kwargs={"width": 2})),
    )

    with pytest.raises(RuntimeError, match="missing required parameters"):
        Boltz2Model.from_boltz_checkpoint(
            "trusted-but-incomplete.ckpt",
            allow_unsafe_pickle=True,
        )


def test_save_pretrained_defaults_to_safetensors_and_round_trips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_tiny_core(monkeypatch)
    source = Boltz2Model(Boltz2Config(core_kwargs={"width": 3}))
    source.core.weight.data.copy_(torch.tensor([0.25, -1.5, 3.0]))  # (d=3,)

    source.save_pretrained(tmp_path)

    assert (tmp_path / "model.safetensors").is_file()
    assert not (tmp_path / "pytorch_model.bin").exists()
    reloaded = Boltz2Model.from_pretrained(tmp_path, local_files_only=True)
    assert torch.equal(reloaded.core.weight, source.core.weight)


def test_floating_features_default_to_fp32_parameter_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_core(monkeypatch)
    model = Boltz2Model(Boltz2Config(core_kwargs={"width": 3}))
    features = {
        "positions": torch.randn(2, 3, dtype=torch.bfloat16),  # (n=2, xyz=3)
        "indices": torch.tensor((1, 2), dtype=torch.int64),  # (n=2,)
        "mask": torch.tensor((True, False)),  # (n=2,)
    }

    moved = model._to_model_device(features)  # shapes unchanged

    assert moved["positions"].dtype == torch.float32
    assert moved["indices"].dtype == torch.int64
    assert moved["mask"].dtype == torch.bool
    assert {tensor.device for tensor in moved.values()} == {model.device}
